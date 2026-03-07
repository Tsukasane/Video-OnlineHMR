import sys
sys.path.insert(0, 'thirdparty/Tracking-Anything-with-DEVA')

from os import path
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import numpy as np

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args

from deva.inference.object_info import ObjectInfo
from deva.inference.frame_utils import FrameInfo
from deva.inference.demo_utils import get_input_frame_for_deva

# Some DEVA tracking settings
# For semi-online mode (default):
# args = ['--chunk_size', '4', '--amp', '--temporal_setting', 'semionline',
#         '--size', '480', '--model', 'data/pretrain/DEVA-propagation.pth',
#         '--suppress_small_objects', '--max_long_term_elements', '1000', '--max_num_objects', '50',
#         '--detection_every', '5']

# For fully online mode (no delay, immediate results):
args = ['--chunk_size', '4', '--amp', '--temporal_setting', 'online',
        '--size', '480', '--model', 'data/pretrain/DEVA-propagation.pth',
        '--suppress_small_objects', '--max_long_term_elements', '1000', '--max_num_objects', '50',
        '--detection_every', '5']  # Detect every frame for online mode --chunk_size, '1'  '--detection_every', '5'

parser = ArgumentParser()
add_common_eval_args(parser)
add_ext_eval_args(parser)
add_auto_default_args(parser)

args = parser.parse_args(args)
cfg = vars(args)
cfg['enable_long_term'] = not cfg['disable_long_term']

deva_model = DEVA(cfg).cuda().eval()
model_weights = torch.load(args.model)
_ = deva_model.load_weights(model_weights)


def get_deva_tracker(vid_length, out_path, online_mode=False):
    """
    Get DEVA tracker instance.
    
    Args:
        vid_length: Total video length (frames)
        out_path: Output path for results
        online_mode: If True, use fully online mode (no buffering, immediate results)
    """
    cfg['enable_long_term_count_usage'] = (  # default deva long-term handling
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
                cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    deva = DEVAInferenceCore(deva_model, config=cfg)
    
    if online_mode:
        # Online mode: process every frame immediately, no voting delay
        print(f"debug -- init online mode")
        deva.next_voting_frame = 0  # Start voting from frame 0
        cfg['detection_every'] = 1  # Detect every frame
    else:
        # Semi-online mode: use voting buffer
        deva.next_voting_frame = args.num_voting_frames - 1
    
    print(f"debug -- next_voting_frame: {deva.next_voting_frame}")
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)
    result_saver.json_style = 'burst'
    result_saver.visualize = False

    return deva, result_saver


@torch.inference_mode()
def track_with_mask(deva: DEVAInferenceCore,
                    masks: torch.Tensor,
                    scores: torch.Tensor,
                    image_np: np.ndarray,  #RGB
                    frame_path: str,
                    result_saver: ResultSaver,
                    ti: int,
                    save_vos=True,
                    online_mode=False) -> dict:
    """
    DEVA tracking step with mask input.
    
    Args:
        deva: DEVA inference core
        masks: Input masks tensor [N, H, W]
        scores: Confidence scores [N]
        image_np: RGB image array
        frame_path: Path to current frame
        result_saver: Result saver instance
        ti: Frame index
        save_vos: Whether to save VOS results
        online_mode: If True, use fully online mode (immediate results, no buffering)
    
    Returns:
        dict with keys:
            - 'prob': Probability map [H, W, num_objects+1]
            - 'object_ids': List of object IDs present in this frame
            - 'boxes': List of bounding boxes for each object (optional)
    """
    cfg = deva.config
    save_the_mask = save_vos

    h, w = image_np.shape[:2]
    min_side = cfg['size']
    need_resize = min_side > 0
    image = get_input_frame_for_deva(image_np, min_side)

    new_h, new_w = image.shape[1:]
    mask, segments_info = transform_masks(masks, scores, new_h, new_w)
    mask = mask.to('cuda')

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    prob = None
    if online_mode:
        print(f"Start tracking using online_mode")
        # ========== FULLY ONLINE MODE ==========
        # Process every frame immediately, no buffering or voting delay
        # This gives immediate results but may be less robust than semi-online
        
        # Incorporate detection directly (no voting)
        prob = deva.incorporate_detection(image, mask, segments_info)
        
        # Update next voting frame for next detection
        deva.next_voting_frame = ti + cfg['detection_every']
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np,
                               save_the_mask=save_the_mask)
    
    else:
        print(f"Starting tracking using semi_online_mode")
        # ========== SEMI-ONLINE MODE (Original) ==========
        # Buffer frames and use voting for robustness (has delay)
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            deva.add_to_temporary_buffer(frame_info)  # wait for more frames 

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image, mask, new_segments_info)
                deva.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       save_the_mask=save_the_mask)

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np,
                                           save_the_mask=save_the_mask)
                deva.clear_buffer()
        else:
            # standard propagation (no new detection, just propagate)
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np,
                                   save_the_mask=save_the_mask)


def transform_masks(masks, scores, new_h, new_w):
    """ Convert masks to index-mask format for DEVA """
    area = masks.sum([1,2])
    device = masks.device

    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
    curr_id = 1
    segments_info = []

    # sort by descending area to preserve the smallest object
    for i in np.flip(np.argsort(area).tolist()):
        mask = masks[i]
        confidence = scores[i].item()
        mask = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), 
                             (new_h, new_w), 
                             mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()

        if mask.sum() > 0:
            output_mask[mask > 0] = curr_id
            segments_info.append(ObjectInfo(id=curr_id, category_id=None, score=confidence))
            curr_id += 1

    return output_mask, segments_info


def match_detections_to_tracks(det_boxes, det_confs, track_result, iou_thresh=0.5, conf_thresh=0.5):
    """
    Match detection boxes to DEVA tracking results.
    
    Args:
        det_boxes: Detection boxes [N, 4] in format [x1, y1, x2, y2]
        det_confs: Detection confidence scores [N]
        track_result: Result dict from track_with_mask containing:
            - 'prob': Probability map [H, W, num_objects+1] or index mask [H, W]
            - 'object_ids': List of object IDs
        iou_thresh: IoU threshold for matching
        conf_thresh: Confidence threshold for detections
    
    Returns:
        dict: {
            'matched': List of tuples (det_idx, track_id, iou, conf),
            'unmatched_dets': List of detection indices without matches,
            'unmatched_tracks': List of track IDs without matches,
            'track_boxes': Dict {track_id: [x1, y1, x2, y2]} - bboxes from tracking masks
        }
    """
    
    # Filter detections by confidence
    valid_det_mask = det_confs >= conf_thresh
    valid_det_boxes = det_boxes[valid_det_mask]
    valid_det_confs = det_confs[valid_det_mask]
    valid_det_indices = np.where(valid_det_mask)[0]
    
    if len(valid_det_boxes) == 0:
        return {
            'matched': [],
            'unmatched_dets': [],
            'unmatched_tracks': track_result['object_ids'],
            'track_boxes': {}
        }
    
    # Extract track boxes from probability map
    prob = track_result['prob']
    object_ids = track_result['object_ids']
    track_boxes_dict = extract_boxes_from_prob(prob, object_ids)

    if len(track_boxes_dict) == 0:
        return {
            'matched': [],
            'unmatched_dets': list(range(len(valid_det_boxes))),
            'unmatched_tracks': [],
            'track_boxes': {}
        }
    
    # Convert to tensors for IoU computation
    det_boxes_tensor = torch.from_numpy(valid_det_boxes[:, :4]).float()  # [N, 4]
    
    # Build track boxes tensor
    track_ids_list = []
    track_boxes_list = []
    for track_id, bbox in track_boxes_dict.items():
        track_ids_list.append(track_id)
        track_boxes_list.append(bbox)
    
    if len(track_boxes_list) == 0:
        return {
            'matched': [],
            'unmatched_dets': list(range(len(valid_det_boxes))),
            'unmatched_tracks': object_ids,
            'track_boxes': track_boxes_dict
        }
    
    track_boxes_tensor = torch.tensor(track_boxes_list).float()  # [M, 4]
    
    # Compute IoU matrix
    from lib.pipeline.tools import box_iou
    iou_matrix = box_iou(det_boxes_tensor, track_boxes_tensor)  # [N, M]
    
    # Greedy matching: assign each detection to best matching track
    matched = []
    matched_track_indices = set()
    matched_det_indices = set()
    
    # Sort by IoU descending
    iou_flat = iou_matrix.flatten()
    indices_flat = torch.arange(len(iou_flat))
    sorted_indices = torch.argsort(iou_flat, descending=True)
    
    for idx_flat in sorted_indices:
        if iou_flat[idx_flat] < iou_thresh:
            break
        
        det_idx = idx_flat // iou_matrix.shape[1]
        track_idx = idx_flat % iou_matrix.shape[1]
        
        if det_idx not in matched_det_indices and track_idx not in matched_track_indices:
            track_id = track_ids_list[track_idx]
            iou_val = iou_flat[idx_flat].item()
            conf_val = valid_det_confs[det_idx].item()
            matched.append((valid_det_indices[det_idx], track_id, iou_val, conf_val))
            matched_track_indices.add(track_idx)
            matched_det_indices.add(det_idx)
    
    # Find unmatched detections and tracks
    unmatched_dets = [valid_det_indices[i] for i in range(len(valid_det_boxes)) 
                      if i not in matched_det_indices]
    unmatched_tracks = [track_ids_list[i] for i in range(len(track_ids_list))
                       if i not in matched_track_indices]
    
    return {
        'matched': matched,  # List of (det_idx, track_id, iou, conf)
        'unmatched_dets': unmatched_dets,
        'unmatched_tracks': unmatched_tracks,
        'track_boxes': track_boxes_dict
    }


def extract_boxes_from_prob(prob, object_ids, threshold=0.5):
    """
    Extract bounding boxes from DEVA probability map.
    
    Args:
        prob: Probability map [H, W, num_objects+1] or index mask [H, W]
        object_ids: List of object IDs to extract boxes for
        threshold: Probability threshold for mask binarization
    
    Returns:
        dict: {object_id: [x1, y1, x2, y2], ...}
    """
    if isinstance(prob, torch.Tensor):
        prob_np = prob.cpu().numpy()
    else:
        prob_np = prob
    
    boxes_dict = {}
    
    if prob_np.ndim == 3:
        # Probability map format [H, W, num_objects+1]
        H, W = prob_np.shape[:2]
        for obj_id in object_ids:
            if obj_id >= prob_np.shape[2]:
                continue
            obj_mask = prob_np[:, :, obj_id] > threshold
            if obj_mask.sum() > 0:
                # Find bounding box
                rows = np.any(obj_mask, axis=1)
                cols = np.any(obj_mask, axis=0)
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    boxes_dict[obj_id] = [int(x1), int(y1), int(x2), int(y2)]
    else:
        # Index mask format [H, W]
        H, W = prob_np.shape
        for obj_id in object_ids:
            obj_mask = (prob_np == obj_id)
            if obj_mask.sum() > 0:
                rows = np.any(obj_mask, axis=1)
                cols = np.any(obj_mask, axis=0)
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    boxes_dict[obj_id] = [int(x1), int(y1), int(x2), int(y2)]
    
    return boxes_dict