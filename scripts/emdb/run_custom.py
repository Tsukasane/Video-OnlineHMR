import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')
sys.path.insert(0, '/scr/yiwenzh5/Video-OnlineHMR/thirdparty/MASt3R-SLAM')

import cv2
import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
import open3d as o3d
import datetime
import pathlib
import time
import lietorch
from lib.pipeline import video2frames
import trimesh
import imageio
import signal
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
from mast3r_slam.lietorch_utils import as_SE3
import torch.multiprocessing as mp
from lib.models import get_hmr_vimo
from pytorch3d.transforms import quaternion_to_matrix

# from lib.camera import run_metric_slam, align_cam_to_world
from lib.pipeline.tools import arrange_boxes
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy

from lib.utils.eval_utils import *
from lib.vis.traj import *

from torch.amp import autocast
from detectron2.config import LazyConfig
from segment_anything import SamPredictor, sam_model_registry

"""
python ./scripts/emdb/run_custom.py --video ./tdance.mp4 --no-viz --calib false
"""

single_color1 = [
    [1.0, 0.2, 0.0],
]
single_color2 = [
    [0.0, 0.8, 1.0],
]

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model_path_or_model, states, keyframes, K):
    """
    Backend process for SLAM optimization.
    
    Args:
        cfg: Configuration
        model_path_or_model: Either a model object (for fork mode) or None (for spawn mode, will load fresh)
        states: Shared states
        keyframes: Shared keyframes
        K: Camera intrinsics
    """
    set_global_config(cfg)

    device = keyframes.device
    
    # In spawn mode, passing CUDA models causes duplication. Load model fresh in child process.
    # This is more memory efficient than serializing/deserializing CUDA tensors.
    if model_path_or_model is None or isinstance(model_path_or_model, str):
        # Load model from path or use default
        from mast3r_slam.mast3r_utils import load_mast3r
        mast3r_model = load_mast3r(device=device)
        mast3r_model.eval()
    else:
        # Fork mode: use passed model
        mast3r_model = model_path_or_model
        if next(mast3r_model.parameters()).device != device:
            mast3r_model = mast3r_model.to(device)
    
    factor_graph = FactorGraph(mast3r_model, keyframes, K, device)
    retrieval_database = load_retriever(mast3r_model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)

def init_detector(device):
    cfg_path = './data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    detector.model.eval()
    return detector


def init_sam(device):
    sam = sam_model_registry["vit_h"](checkpoint="./data/pretrain/sam_vit_h_4b8939.pth")
    sam = sam.to(device)
    sam.eval() 
    predictor = SamPredictor(sam)
    return predictor


def bbox_est(center, scale, img_focal, img_center):
    '''
    Pixel representation
    '''
    # Original image center
    img_cx, img_cy = img_center[:,0], img_center[:,1]

    # Implement CLIFF (Li et al.) bbox feature
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    bbox_info = torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)
    bbox_info[:, :2] = bbox_info[:, :2] / img_focal.unsqueeze(-1) * 2.8 
    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal)  

    return bbox_info


def camera_coord_HMR(hmr_model, imgfiles, boxes, cache, img_focal, img_center):
    results, cache = hmr_model.inference_chunk_ar(imgfiles, boxes,
                    img_focal=img_focal, img_center=img_center, cache=cache)
    
    return results, cache
    

def load_camera_poses(cam_t, cam_q, scale=0.2, color_offset=0, gt=True):
    """
    Load camera poses from text file.
    Each line: tx ty tz qx qy qz qw
    Returns a list of LineSet pyramids.
    """
    
    tx, ty, tz = cam_t
    qx, qy, qz, qw = cam_q

    # quaternion → rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    t = np.array([tx, ty, tz])

    # pyramid points (camera frustum)
    cam_points = np.array([
        [0, 0, 0],
        [scale, scale, scale * 1.5],
        [scale, -scale, scale * 1.5],
        [-scale, -scale, scale * 1.5],
        [-scale, scale, scale * 1.5],
    ])
    cam_lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    cam = o3d.geometry.LineSet()
    cam.points = o3d.utility.Vector3dVector(cam_points @ R.T + t)
    cam.lines = o3d.utility.Vector2iVector(cam_lines)
    # Use color_offset to differentiate between two trajectories
    if not gt:
        cam.colors = o3d.utility.Vector3dVector([single_color1[0]] * len(cam_lines))
    else:
        cam.colors = o3d.utility.Vector3dVector([single_color2[0]] * len(cam_lines))
        # incremental color along the time sequence
        # cam.colors = o3d.utility.Vector3dVector([rainbow_colors[(l_id + color_offset) % 20]] * len(cam_lines))

    return cam

def load_mogev2_model(device):
    from moge.model.v2 import MoGeModel
    mogev2_model = MoGeModel.from_pretrained("./data/pretrain/mogev2_model.pt").to(device)                             
    mogev2_model.eval()
    return mogev2_model



if __name__=='__main__':
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=2)
    parser.add_argument("--save-as", default="default")
    parser.add_argument('--output_dir', type=str, default='results/emdb/camera')
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--save_dir", default="./res_human_camera")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", type=bool)
    parser.add_argument("--video", type=str, required=True, help="path to the input video")

    args = parser.parse_args()

    load_config(args.config)
    # print(config)

    savefolder = args.output_dir
    os.makedirs(savefolder, exist_ok=True)

    detector = init_detector(device) # ViTDet
    sam_predictor = init_sam(device) # SAM for human mask
    hmr_model = get_hmr_vimo(checkpoint='./results/checkpoint_best.pth.tar') # NOTE(yiwen) change inference checkpoint path here.
    metric_depth_model = load_mogev2_model(device)
    
    # Load MASt3R-SLAM model once for all sequences (shared across sequences)
    # share_memory() is called once here, as the model instance is reused across sequences
    # Each sequence will have its own backend process that accesses this shared model
    mast3r_model = load_mast3r(device=device)
    mast3r_model.eval()
    mast3r_model.share_memory()  # Enable IPC for multiprocessing (spawn mode)

    # SMPL
    smpl = SMPL()
    smpls = {g:SMPL(gender=g) for g in ['neutral', 'male', 'female']}

    # Estimate camera motion on EMDB (subset: spl)
    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)
    print(f'Running on custom video...')

    print(f"Split video to frames and register camera calibration")
    # File and folders
    file = args.video
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = f'results/{seq}'
    img_folder = f'{seq_folder}/images'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)

    print('Extracting frames ...')
    nframes = video2frames(file, img_folder)
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    name_prefix = seq
    out_gif = f"human_camera_{name_prefix}.gif"

    # lightweight dataset loading (read and sort images)
    img_h, img_w = cv2.imread(imgfiles[0]).shape[:2]
    dataset = load_dataset(img_folder)
    dataset.subsample(config["dataset"]["subsample"]) # set to 1 by default
    rimg_h, rimg_w = dataset.get_img_shape()[0] # resized image and resized shape
    
    # if args.calib: #TODO(yiwen) to support provided intrinsics
    #     # load annotations if eval on emdb2
    #     annfile = args.calib
    #     ann = pkl.load(open(annfile, 'rb'))
    #     intr = ann['camera']['intrinsics']
    #     cam_int = [intr[0,0], intr[1,1], intr[0,2], intr[1,2]]
    #     img_focal = (intr[0,0] +  intr[1,1]) / 2.
    #     img_center = intr[:2, 2]

    #     # register to mast3r-slam
    #     intrinsics = intr # yaml.load(f, Loader=yaml.SafeLoader)
    #     config["use_calib"] = True
    #     dataset.use_calibration = True
    #     dataset.camera_intrinsics = Intrinsics.from_calib(
    #         dataset.img_size, # resized
    #         img_w, # original
    #         img_h,
    #         cam_int,
    #     )

    # Allow configurable buffer size for long videos
    # Default is 512, but can be increased via config
    max_keyframes = config.get("tracking", {}).get("max_keyframes", 512)
    keyframes = SharedKeyframes(manager, rimg_h, rimg_w, buffer=max_keyframes)
    states = SharedStates(manager, rimg_h, rimg_w)
    
    if not args.no_viz: # NOTE(yiwen) 这里的visualization换成viser，但是需要看一下涉及到的multi processing
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()
    
    # Use the pre-loaded model (already share_memory() called globally)
    # Each sequence has its own backend process that accesses this shared model
    # model = mast3r_model
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    img_focal = None
    img_center = None
    
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)
        # Extract focal length and center from calibration
        K_np = dataset.camera_intrinsics.K_frame
        img_focal = (K_np[0, 0] + K_np[1, 1]) / 2.0
        img_center = K_np[:2, 2]
    else:
        # Simple heuristic based on image dimensions. Does not support fov change
        # Most cameras have focal length roughly 0.7-1.0 * max(width, height)
        img_focal = max(img_w, img_h) * 0.8  # Rough estimate: 80% of max dimension
        img_center = np.array([img_w / 2., img_h / 2.])
        print(f"[Info] No calibration provided. SLAM does not estimate intrinsics.")
        print(f"[Info] Using heuristic focal length for HMR: {img_focal:.1f} pixels (estimated from image size)")
        

    if dataset.save_results: # remove previously saved results
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()
    
    tracker = FrameTracker(mast3r_model, keyframes, device)
    last_msg = WindowMsg()

    # start backend
    backend_model_arg = None  # Backend will load model fresh in its own process
    backend = mp.Process(target=run_backend, args=(config, backend_model_arg, states, keyframes, K))
    backend.start()

    # NOTE(yiwen) frontend loop, incrementally loop all frames
    i = 0
    fps_timer = time.time()
    frames = []
    img_chunk = []
    box_chunk = []

    # cam coords
    pred_cam = []
    pred_pose = []
    pred_shape = []
    pred_rotmat = []
    pred_trans = []

    frame_feat_cache = None
    naive_scaler = 1.0 # init depth-based scaler
    fix_scaler = 1.0
    imgs = [] # for rendered images
    print(f"Start per frame processing")

    # Offscreen renderer
    visualize_hcgif = True
    visualize_depth = False
    angle = 180
    w, h = 800, 600
    render_interval = 2
    render = o3d.visualization.rendering.OffscreenRenderer(w, h)
    render.scene.set_background([0, 0, 0, 1])
    set_render_camera = False # render cam (the third viewpoint)

    # materials
    human_mat = o3d.visualization.rendering.MaterialRecord()
    human_mat.shader = "defaultLit"
    cam_mat = o3d.visualization.rendering.MaterialRecord()
    cam_mat.shader = "unlitLine"
    
    # start looping the video seq
    while True:
        ###### Camera Pose SLAM --> output Cam_R, Cam_T, also camera coordinates absolute depth (then convert to world depth) ######
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break
        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue
        if not last_msg.is_paused:
            states.unpause()
        if i == len(dataset): # NOTE(yiwen) end of the dataset
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i] # the original size, 0-1 scale
        img_cv2 = dataset.read_img(i) # the original size, 0-255 scale

        # SAM mask
        boxes_np = None

        with torch.no_grad():
            with autocast('cuda'):
                det_out = detector(img_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                boxes_np = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        if boxes_np is not None and boxes_np.shape[0] > 0:
            with autocast('cuda'):
                sam_predictor.set_image(img_cv2, image_format='BGR')
                bb = torch.tensor(boxes_np[:, :4]).to(device)
                bb = sam_predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])
                masks, scores, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=bb,
                    multimask_output=False
                )
            masks = masks.detach().cpu().squeeze(1)  # (N, H, W)
            human_mask = (masks.sum(dim=0) > 0).numpy()  # (H, W) bool
            if human_mask.any():
                # img is float [0,1], HxWx3; zero-out human regions
                img[human_mask] = 0.0

        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device) # SE(3)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        # Mast3r-SLAM init
        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(mast3r_model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        # Mast3r-SLAM tracking
        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

            # Extract depth information from current frame
            if frame.X_canon is not None:

                depths = frame.X_canon[:, 2].cpu().numpy()  # Z coordinates as depth
                confidences = frame.get_average_conf().cpu().numpy()
                
                # Reshape to image dimensions
                img_shape = frame.img_shape.flatten()[:2].cpu().numpy()  # [H, W]
                depth_map = depths.reshape(img_shape[0], img_shape[1])
                conf_map = confidences.reshape(img_shape[0], img_shape[1])
                
                # Filter valid depths (remove invalid/negative depths)
                valid_mask = depths > 0
                valid_depths = depths[valid_mask]
                valid_confs = confidences[valid_mask]

                # metric scale depth
                depth_input_img = torch.tensor(img).permute(2, 0, 1) # 3, 960, 720
                metric_depth = metric_depth_model.infer(depth_input_img)["depth"]
                naive_scaler = metric_depth.min() / valid_depths.min()
                """
                slam depth * scale = pred depth

                pred_cam_t = torch.tensor(traj[:, :3]) * scale
                pred_cam_q = torch.tensor(traj[:, 3:])
                """
                
                if len(valid_depths) > 0:
                    if visualize_depth and (i % 10 == 0):  # Save every 10th frame
                        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                        cv2.imwrite(f"depth_frame_{i:06d}.png", depth_uint8)

        # Mast3r-SLAM relocation          
        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(mast3r_model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            
            # Extract depth information during relocalization
            if frame.X_canon is not None:
                depths = frame.X_canon[:, 2].cpu().numpy()
                confidences = frame.get_average_conf().cpu().numpy()
                
                img_shape = frame.img_shape.flatten()[:2].cpu().numpy()
                depth_map = depths.reshape(img_shape[0], img_shape[1])
                
                valid_mask = depths > 0
                valid_depths = depths[valid_mask]
                
                if len(valid_depths) > 0:
                    print(f"Frame {i} (RELOC): Depth range [{valid_depths.min():.3f}, {valid_depths.max():.3f}]")
                    print(f"Frame {i} (RELOC): Valid depth pixels: {len(valid_depths)}/{len(depths)}")
            
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)
        else:
            raise Exception("Invalid mode")

        # save per frame results
        if dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(args, dataset)
            traj_file = save_dir / f"{name_prefix}_{seq_name}_incremental_all.txt"
            with open(traj_file, "a") as f:  # append
                t = dataset.timestamps[frame.frame_id]
                T_WC = as_SE3(frame.T_WC)
                x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
                f.write(f"{naive_scaler} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                # f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
        
        # save key frame results
        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)

            if dataset.save_results:
                save_dir, seq_name = eval.prepare_savedir(args, dataset)
                traj_file = save_dir / f"{name_prefix}_{seq_name}_incremental_kf.txt"
                with open(traj_file, "a") as f:  # append
                    t = dataset.timestamps[frame.frame_id]
                    T_WC = as_SE3(frame.T_WC)
                    x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
                    if len(keyframes)==2:
                        f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n") #TODO(yiwen) may need to make to first one to 1
                    f.write(f"{naive_scaler} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        ###### Camera Coordinate Human Mesh Recovery --> Only support single person for now ######
        # --- Detect Bounding Boxes ---
        
        with torch.no_grad():
            with autocast('cuda'):
                det_out = detector(img_cv2)
                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                confs = det_instances.scores[valid_idx].cpu().numpy()

                boxes = np.hstack([boxes, confs[:, None]])
                boxes = arrange_boxes(boxes, mode='size', min_size=100)

        if boxes.shape[0]<1: # NOTE(yiwen) in emdb2 evaluation, boxes should come from gt annotations
            # TODO(yiwen) if no boxes detected, skip the camera coord hmr
            # record some missing hmr e.g. all zeros to npy/txt
            continue

        # TODO(yiwen) multiple persons 的时候还是需要一下tracking？否则会检测出来多个bounding boxes，confidence都足够高，这种情况下应该不能直接用bbox去筛选
        # 或者用bbox center过滤一下
        elif boxes.shape[0]>1: # when multiple person detected
            boxes = boxes[0:1]
        
        img_ck = np.array([imgfiles[i]])
        box_ck = np.array([boxes]).reshape(-1, 5)
        
        # img_ck = np.array(img_chunk)
        # try:
        #     box_ck = np.array(box_chunk).reshape(-1, 5)
        # except:
        #     breakpoint()

        frame_results, frame_feat_cache = camera_coord_HMR(
            hmr_model, img_ck, box_ck, frame_feat_cache, img_focal, img_center)
        
        # NOTE(yiwen) two frame cache, shape[1] = h*w*mem_t
        # print(f"cache length {frame_feat_cache['layers'][0]['mem_k'].shape}")


        pred_cam.append(frame_results['pred_cam'])
        pred_pose.append(frame_results['pred_pose'])
        pred_shape.append(frame_results['pred_shape'])
        pred_rotmat.append(frame_results['pred_rotmat'])
        pred_trans.append(frame_results['pred_trans'])

        """
        TODO(yiwen) 
        camcoord_hmr = mp.process(target=function,args=())
        camcoord_hmr.start()
        """
        if i==0:
            fix_scaler = naive_scaler
        # world coord camera trajectory
        current_camt = torch.tensor([fix_scaler*x, fix_scaler*y, fix_scaler*z]).unsqueeze(0)
        current_camq = torch.tensor([qw, qx, qy, qz]).unsqueeze(0)

        current_camr = quaternion_to_matrix(current_camq)

        # frame_results['pred_rotmat'] # T, 24, 3, 3
        # frame_results['pred_shape'] # T, 10
        # frame_results['pred_trans'] # T, 1, 3

        pred = smpls['neutral'](body_pose=frame_results['pred_rotmat'][:,1:], 
                                global_orient=frame_results['pred_rotmat'][:,[0]], 
                                betas=frame_results['pred_shape'], 
                                transl=frame_results['pred_trans'].squeeze(1),
                                pose2rot=False, 
                                default_smpl=True)
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]

        # world coords human mesh
        pred_vert_w = torch.einsum('bij,bnj->bni', current_camr, pred_vert) + current_camt[:,None] # 1, 6890, 3
        pred_j3d_w = torch.einsum('bij,bnj->bni', current_camr, pred_j3d) + current_camt[:,None] # 1, 24, 3 -- pose
        pred_ori_w = torch.einsum('bij,bjk->bik', current_camr, frame_results['pred_rotmat'][:,0]) # 1, 3, 3

        # visualize human-camera gif
        if visualize_hcgif and i % render_interval == 0:
            # Remove i-2 geometry to avoid accumulation (keep only current and previous frame)
            num_keep_frames = 1
            prev_frame_idx = i - render_interval * num_keep_frames  # keep five previous frames
            if prev_frame_idx >= 0:
                try:
                    render.scene.remove_geometry(f"human{prev_frame_idx}")
                except:
                    pass
                try:
                    render.scene.remove_geometry(f"cam_frame{prev_frame_idx}")
                except:
                    pass
            
            cam_frame = load_camera_poses(current_camt[0], current_camq[0]) # o3d camera
            mesh = trimesh.Trimesh(vertices=pred_vert_w[0], faces=smpls['neutral'].faces)
            human_mesh = o3d.geometry.TriangleMesh()
            human_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            human_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            human_mesh.compute_vertex_normals()
            human_mesh.paint_uniform_color([0.8, 0.6, 0.6])

            render.scene.add_geometry(f"human{i}", human_mesh, human_mat)
            render.scene.add_geometry(f"cam_frame{i}", cam_frame, cam_mat)

            if not set_render_camera:
                center = human_mesh.get_center()
                view_radius = 8
                eye0 = center + np.array([view_radius, view_radius, view_radius])
                radius = np.linalg.norm(eye0 - center)

                # Yaw around the vertical (Y) axis, keeping camera above the object
                theta = np.deg2rad(angle)
                eye = center + radius * np.array([np.sin(theta), -0.2, np.cos(theta)])  # 0.2: small height above ground
                up = np.array([0, 1, 0])  # keep world Y up
                render.setup_camera(60, center, eye, up)
                set_render_camera = True
            img_o3d = render.render_to_image()

            # flip vertically for correct image orientation
            img_np = np.asarray(img_o3d)
            img_np = np.flipud(img_np)
            img_np = np.fliplr(img_np)
            imgs.append(img_np)

        # print FPS per 30 frames
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    # Save gif
    if visualize_hcgif:
        imageio.mimsave(out_gif, imgs, fps=30)
        print(f"✅ Saved gif to {out_gif}")

    
    ###### Save Global Results ######
    # cam coord results of the whole sequence, for eval
    os.makedirs(args.save_dir, exist_ok=True)
    cam_coord_results = {'pred_cam': torch.cat(pred_cam),
            'pred_pose': torch.cat(pred_pose),
            'pred_shape': torch.cat(pred_shape),
            'pred_rotmat': torch.cat(pred_rotmat),
            'pred_trans': torch.cat(pred_trans)}
    np.savez(f'{args.save_dir}/{name_prefix}.npz', **cam_coord_results)

    # the final cam pose and scene pc after global optimization
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        cam_savedir = os.path.join(args.save_dir, "global_results")
        os.makedirs(cam_savedir, exist_ok=True)
        eval.save_traj(cam_savedir, f"{name_prefix}_{seq_name}_globalOptimized_kf.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            cam_savedir,
            f"{name_prefix}_{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )

    print("done")
    
    # Send termination signal to viz
    if not args.no_viz:
        try:
            main2viz.put(WindowMsg(is_terminated=True), timeout=1)
        except:
            pass
    
    # Wait for processes to finish with timeout
    print("Waiting for backend to finish...")
    timeout = 30
    start_time = time.time()
    
    while backend.is_alive() and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    if backend.is_alive():
        print("Warning: Backend did not finish, terminating...")
        backend.terminate()
        backend.join()
    else:
        backend.join()
        print("Backend joined")
    
    if not args.no_viz:
        print("Waiting for visualization to finish...")
        start_time = time.time()
        while viz.is_alive() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if viz.is_alive():
            print("Warning: Visualization did not finish, terminating...")
            viz.terminate()
            viz.join()
        else:
            viz.join()
            print("Visualization joined")
    
    # Clean up resources
    print("Cleaning up resources...")
    del tracker
    del keyframes
    del states
    if not args.no_viz:
        del main2viz
        del viz2main
    del manager    
    torch.cuda.empty_cache()
    del render
    
    print(f"Sequence {root} completed and cleaned up")