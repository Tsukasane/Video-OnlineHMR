import sys
import os

# osmesa headless setup for Open3D and PyOpenGL, before importing Open3D
conda_env_path = os.environ.get('CONDA_PREFIX', sys.prefix)
conda_lib_path = os.path.join(conda_env_path, 'lib')

os.environ["LD_LIBRARY_PATH"] = conda_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LIBGL_DRIVERS_PATH"] = os.path.join(conda_lib_path, "dri")

os.environ["OPEN3D_HEADLESS"] = "1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import torch
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import open3d as o3d
sys.path.insert(0, os.path.dirname(__file__) + '/..')
sys.path.insert(0, './thirdparty/MASt3R-SLAM')
import cv2
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import pickle
from glob import glob
import open3d as o3d
import datetime
import time
import lietorch
from lib.pipeline import video2frames
import trimesh
import imageio

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
python ./scripts/run_custom.py --video ./tdance.mp4 --no-viz --calib false --smooth-method none
python ./scripts/run_custom.py --video ./rock_climbing1.mp4 --no-viz --calib false --smooth-method ema
"""

single_color1 = [[1.0, 0.2, 0.0]]
single_color2 = [[0.0, 0.8, 1.0]]

def quaternion_translation_to_Sim3(t, q, device):
    """
    Convert quaternion and translation to lietorch.Sim3.
    
    Args:
        t: translation [x, y, z] or numpy array/torch tensor
        q: quaternion [qx, qy, qz, qw] or numpy array/torch tensor  
        device: torch device
    
    Returns:
        lietorch.Sim3 object
    """
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q).float()
    
    # Ensure correct shape: [3] for translation, [4] for quaternion
    if t.dim() > 1:
        t = t.flatten()[:3]
    if q.dim() > 1:
        q = q.flatten()[:4]
    
    # Ensure we have exactly 3 and 4 elements
    t = t[:3].to(device)
    q = q[:4].to(device)
    
    # Sim3 format: [t, q, s] where s is scale (set to 1.0 for SE(3))
    # lietorch.Sim3 expects data in format [tx, ty, tz, qx, qy, qz, qw, s]
    # Shape should be [1, 8] for batch dimension
    s = torch.ones(1, device=device, dtype=t.dtype)
    sim3_data = torch.cat([t, q, s], dim=0).unsqueeze(0)  # [1, 8]
    return lietorch.Sim3(sim3_data)


def relocalization(frame, keyframes, factor_graph, retrieval_database):
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
        kf_idx = list(kf_idx)
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

def init_detector():
    cfg_path = './data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg) # to cuda here
    detector.model.eval()
    return detector


def init_sam(device):
    sam = sam_model_registry["vit_h"](checkpoint="./data/pretrain/sam_vit_h_4b8939.pth")
    sam = sam.to(device)
    sam.eval() 
    predictor = SamPredictor(sam)
    return predictor


def bbox_est(center, scale, img_focal, img_center):
    # approximate image center
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
    cam_t: tx ty tz 
    cam_q: qx qy qz qw
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
        ## [Option] incremental color along the time sequence
        # cam.colors = o3d.utility.Vector3dVector([rainbow_colors[(l_id + color_offset) % 20]] * len(cam_lines))

    return cam

def load_mogev2_model(device):
    from moge.model.v2 import MoGeModel
    mogev2_model = MoGeModel.from_pretrained("./data/pretrain/mogev2_model.pt").to(device)                             
    mogev2_model.eval()
    return mogev2_model


def process_mask_soft_margin(mask, method='gaussian', kernel_size=15, sigma=5.0, dilation_iterations=3):
    """
    Process binary mask to create soft margin using Gaussian blur and/or dilation.
    
    Args:
        mask: binary mask (H, W) bool or (H, W) float [0,1]
        method: 'gaussian', 'dilation', or 'both'
        kernel_size: kernel size for Gaussian blur (should be odd)
        sigma: sigma for Gaussian blur
        dilation_iterations: number of dilation iterations
    
    Returns:
        soft_mask: float mask [0,1] with soft margins
    """
    # Convert to float if boolean
    if mask.dtype == bool:
        mask = mask.astype(np.float32)
    else:
        mask = mask.astype(np.float32)
    
    if method == 'gaussian':
        # Apply Gaussian blur to create soft edges
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd
        soft_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        return soft_mask
    
    elif method == 'dilation':
        # Apply dilation to expand mask, then blur for soft edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask, kernel, iterations=dilation_iterations)
        # Apply Gaussian blur to smooth the dilated edges
        kernel_size_blur = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        soft_mask = cv2.GaussianBlur(dilated, (kernel_size_blur, kernel_size_blur), sigma)
        return soft_mask
    
    elif method == 'both':
        # First dilate, then apply Gaussian blur
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask, kernel, iterations=dilation_iterations)
        kernel_size_blur = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        soft_mask = cv2.GaussianBlur(dilated, (kernel_size_blur, kernel_size_blur), sigma)
        return soft_mask
    
    else:  # 'none' or invalid
        return mask


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
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--save_dir", default="./res_human_camera")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", type=bool)
    parser.add_argument("--video", type=str, required=True, help="path to the input video")
    parser.add_argument("--smooth-method", type=str, default='ema', 
                       choices=['none', 'ema'],
                       help='Smoothing method for camera poses')
    parser.add_argument("--smooth-window", type=int, default=5,
                       help='Window size for moving average smoothing')
    parser.add_argument("--smooth-alpha", type=float, default=0.2,
                       help='Alpha for exponential moving average (0-1)')
    parser.add_argument("--ema-history", type=int, default=10,
                       help='Number of history frames to use for EMA weighted average')
    parser.add_argument("--ema-clamp-multiplier", type=float, default=0.2,
                       help='Multiplier for velocity-based clamp threshold')
    parser.add_argument("--ema-clamp-absolute-max", type=float, default=None,
                       help='Absolute maximum update allowed regardless of velocity (default: None, disabled)')
    parser.add_argument("--update-slam-pose", action="store_true",
                       help='Update SLAM internal pose with smoothed pose (may affect optimization)')
    parser.add_argument("--mask-method", type=str, default='both',
                       choices=['none', 'gaussian', 'dilation', 'both'],
                       help='Method for mask soft margin: none, gaussian, dilation, both')
    parser.add_argument("--mask-kernel-size", type=int, default=15,
                       help='Kernel size for mask processing')
    parser.add_argument("--mask-sigma", type=float, default=5.0,
                       help='Sigma for Gaussian blur in mask processing')
    parser.add_argument("--mask-dilation-iterations", type=int, default=3,
                       help='Number of dilation iterations for mask processing')
    parser.add_argument("--depth-mask", action="store_true",
                       help='Apply human mask to depth maps (SLAM depth and metric depth) when computing scaler')

    args = parser.parse_args()

    load_config(args.config)

    detector = init_detector() # ViTDet
    sam_predictor = init_sam(device) # SAM for human mask
    hmr_model = get_hmr_vimo(checkpoint='./results/onlinehmr/checkpoint.pth.tar')
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

    # Estimate camera motion on EMDB
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
    depth_img_folder = f'{seq_folder}/depth_images'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(depth_img_folder, exist_ok=True)

    print('Extracting frames ...')
    nframes = video2frames(file, img_folder)
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    name_prefix = seq
    out_gif = f"human_camera_{name_prefix}.gif"

    # lightweight dataset loading (read and sort images)
    img_h, img_w = cv2.imread(imgfiles[0]).shape[:2]
    dataset = load_dataset(img_folder)
    dataset.subsample(config["dataset"]["subsample"]) # set to 1 by default
    rimg_h, rimg_w = dataset.get_img_shape()[0] # resized shape

    # Allow configurable buffer size for long videos
    # Default is 512, but can be increased via config
    max_keyframes = config.get("tracking", {}).get("max_keyframes", 512)
    keyframes = SharedKeyframes(manager, rimg_h, rimg_w, buffer=max_keyframes)
    states = SharedStates(manager, rimg_h, rimg_w)
    
    if not args.no_viz: # TODO(yiwen) remove this
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()
    
    ## init intrinsics
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"] # set to false by default

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
        K_np = dataset.camera_intrinsics.K_frame
        img_focal = (K_np[0, 0] + K_np[1, 1]) / 2.0
        img_center = K_np[:2, 2]
    else:
        # Simple heuristic based on image dimensions. Does not support fov change
        img_focal = max(img_w, img_h) * 0.8  # Rough estimate: 80% of max dimension
        img_center = np.array([img_w / 2., img_h / 2.])
        print(f"[Info] No calibration provided. SLAM does not estimate intrinsics.")
        print(f"[Info] Using heuristic focal length for HMR: {img_focal:.1f} pixels (estimated from image size)")

    if dataset.save_results:
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
    refined_scaler = 1.0 # refined depth-based scaler
    fix_scaler = 1.0
    imgs = [] # for rendered images
    
    mask_thres = 1e-4
    # Online smoothing for camera poses
    from collections import deque
    
    # Get smoothing parameters from args
    smooth_method = args.smooth_method
    smooth_window = args.smooth_window
    smooth_alpha = args.smooth_alpha
    ema_history = args.ema_history
    ema_clamp_multiplier = args.ema_clamp_multiplier
    ema_clamp_absolute_max = args.ema_clamp_absolute_max
    
    # Initialize smoothing buffers
    smoothed_pose = None
    
    # EMA history buffer: store (x, y, z) for translation smoothing and velocity estimation
    ema_history_buffer = deque(maxlen=ema_history)  # Store (x, y, z) tuples
    
    print(f"Start per frame processing")
    print(f"Smoothing method: {smooth_method}, window: {smooth_window}, alpha: {smooth_alpha}")
    if smooth_method == 'ema':
        print(f"EMA history: {ema_history} frames, clamp multiplier: {ema_clamp_multiplier}")
        if ema_clamp_absolute_max is not None:
            print(f"EMA absolute max: {ema_clamp_absolute_max}")

    # Offscreen renderer
    visualize_hcgif = True
    visualize_depth = True
    angle = 180
    w, h = 800, 600
    render_interval = 2

    if visualize_hcgif:
        try:
            print("[Info] Initializing OSMesa/headless renderer...")
            render = o3d.visualization.rendering.OffscreenRenderer(w, h)
            render.scene.set_background([0, 0, 0, 1])
            set_render_camera = False # render cam (the third viewpoint)
            print("[Info] OSMesa renderer initialized successfully")
        except Exception as e:
            print(f"[Error] Failed to initialize OSMesa renderer: {e}")
            print("[Error] Please ensure OSMesa is installed: sudo apt-get install libosmesa6-dev")
            raise
        
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

        depth_img = img.copy()
        # SAM mask
        boxes_np = None
        human_mask = None  # Store mask for confidence masking in flow space

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

            hard_human_mask = human_mask.copy()
            if human_mask.any():
                # Process mask to create soft margin
                if args.mask_method != 'none':
                    soft_mask = process_mask_soft_margin(
                        human_mask,
                        method=args.mask_method,
                        kernel_size=args.mask_kernel_size,
                        sigma=args.mask_sigma,
                        dilation_iterations=args.mask_dilation_iterations
                    )
                    # Apply soft mask: multiply image by (1 - soft_mask) to gradually fade out human regions
                    # soft_mask is [0,1] where 1 is human region, so (1 - soft_mask) is background weight
                    img = img * (1.0 - soft_mask[..., np.newaxis])  # Add channel dimension for broadcasting
                    # Store soft_mask for confidence masking (use soft_mask instead of binary mask)
                    human_mask = soft_mask

                    ## visulize soft mask overlay
                    # if i==29:
                    #     mask = human_mask.detach().cpu().numpy()
                    #     img_np = img.detach().cpu().numpy() if hasattr(img, 'detach') else img

                    #     # Normalize img_np to [0,1] if needed
                    #     if img_np.max() > 1.5:
                    #         img_np = img_np / 255.0

                    #     overlay_color = np.array([0.0, 0.0, 0.0])

                    #     overlay = img_np * (1 - mask[..., None]) + overlay_color * mask[..., None]

                    #     plt.imshow(overlay)
                    #     plt.axis('off')
                    #     plt.title("Soft Mask Overlay (with Transparency)")
                    #     plt.savefig("soft_mask_overlay.png", bbox_inches='tight', pad_inches=0)
                    #     plt.close()
                    #     breakpoint()
                else:
                    # Hard mask: zero-out human regions directly
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
                # metric scale depth
                depth_input_img = torch.tensor(depth_img).permute(2, 0, 1) # 3, 960, 720
                metric_depth = metric_depth_model.infer(depth_input_img)["depth"]

                valid_depths = depths[valid_mask]
                print(f"debug -- before human mask scaler: {metric_depth.min() / valid_depths.min()}")
                
                # Also remove human region from SLAM depths (only if --depth-mask is enabled)
                if args.depth_mask and hard_human_mask is not None and hard_human_mask.any():
                    mask_h, mask_w = hard_human_mask.shape
                    depth_h, depth_w = img_shape[0], img_shape[1]
                    
                    if mask_h != depth_h or mask_w != depth_w:
                        mask_torch = torch.from_numpy(hard_human_mask).float().unsqueeze(0).unsqueeze(0)
                        mask_resized = F.interpolate(
                            mask_torch,
                            size=(depth_h, depth_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                        human_mask_resized = mask_resized.cpu().numpy() > mask_thres
                    else:
                        # Same size, just convert to bool if needed
                        human_mask_resized = hard_human_mask > mask_thres if hard_human_mask.dtype == bool else hard_human_mask > mask_thres
                    
                    # Flatten the mask to match depths shape
                    human_mask_flat = human_mask_resized.flatten()

                    # Exclude human regions from valid_mask
                    valid_mask = valid_mask & (~human_mask_flat)
                
                valid_depths = depths[valid_mask]
                valid_confs = confidences[valid_mask]

                # Prepare masked versions for est_scale_hybrid if --depth-mask is enabled
                depth_map_for_scale = depth_map.copy()
                metric_depth_for_scale = metric_depth.cpu().numpy() if isinstance(metric_depth, torch.Tensor) else metric_depth.copy()
                metric_depth_min = None
                
                # Remove human region from metric depth before computing min (only if --depth-mask is enabled)
                if args.depth_mask and hard_human_mask is not None and hard_human_mask.any():
                    # Ensure metric_depth is 2D (H, W)
                    if metric_depth.dim() > 2:
                        metric_depth_2d = metric_depth.squeeze()
                    else:
                        metric_depth_2d = metric_depth
                    
                    # Convert human_mask to same shape and type as metric_depth
                    if isinstance(hard_human_mask, np.ndarray):
                        # human mask and metric_depth are all in original image size
                        mask_h, mask_w = hard_human_mask.shape
                        metric_h, metric_w = metric_depth_2d.shape

                        if mask_h != metric_h or mask_w != metric_w:
                            # Resize mask to match metric_depth dimensions
                            mask_torch = torch.from_numpy(hard_human_mask).float().unsqueeze(0).unsqueeze(0)
                            mask_resized_metric = F.interpolate(
                                mask_torch,
                                size=(metric_h, metric_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)
                            human_mask_resized_metric = mask_resized_metric.cpu().numpy() > mask_thres
                        else:
                            human_mask_resized_metric = hard_human_mask > mask_thres if hard_human_mask.dtype == bool else hard_human_mask > mask_thres
                        
                        # Mask out human regions: set to a large value so they're ignored in min()
                        if isinstance(metric_depth_2d, torch.Tensor):
                            metric_depth_masked = metric_depth_2d.clone().cpu().numpy()
                        else:
                            metric_depth_masked = metric_depth_2d.copy()
                        metric_depth_masked[human_mask_resized_metric] = np.inf
                        valid_metric_depths = metric_depth_masked[metric_depth_masked != np.inf]
                        if len(valid_metric_depths) > 0:
                            metric_depth_min = np.min(valid_metric_depths)
                        else:
                            # Fallback: if all values are masked, use original min
                            metric_depth_min = metric_depth.min().item() if isinstance(metric_depth, torch.Tensor) else metric_depth.min()
                        
                        # Use masked metric_depth for est_scale_hybrid
                        metric_depth_for_scale = metric_depth_masked.copy()
                    else:
                        # If mask is not available or empty, use original min
                        metric_depth_min = metric_depth.min().item() if isinstance(metric_depth, torch.Tensor) else metric_depth.min()
                else:
                    # No human mask, use original min
                    metric_depth_min = metric_depth.min().item() if isinstance(metric_depth, torch.Tensor) else metric_depth.min()

                # Apply mask to depth_map for est_scale_hybrid if --depth-mask is enabled
                if args.depth_mask and hard_human_mask is not None and hard_human_mask.any():
                    # Create masked version of depth_map (set human regions to 0 to exclude from scale estimation)
                    depth_map_masked = depth_map.copy()
                    if 'human_mask_resized' in locals():
                        depth_map_masked[human_mask_resized] = 0.0
                    else:
                        # resize the mask
                        mask_h, mask_w = hard_human_mask.shape
                        depth_h, depth_w = depth_map.shape
                        if mask_h == depth_h and mask_w == depth_w:
                            depth_map_masked[hard_human_mask > mask_thres] = 0.0
                        else:
                            mask_torch = torch.from_numpy(hard_human_mask).float().unsqueeze(0).unsqueeze(0)
                            mask_resized = F.interpolate(
                                mask_torch,
                                size=(depth_h, depth_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).squeeze(0)
                            human_mask_resized_for_depth = mask_resized.cpu().numpy() > mask_thres
                            depth_map_masked[human_mask_resized_for_depth] = 0.0
                    depth_map_for_scale = depth_map_masked
                

                if len(metric_depth) > 0:
                    if visualize_depth and (i % 10 == 0):  # Save every 10th frame
                        metric_depth_normalized = (metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min())
                        metric_depth_normalized = metric_depth_normalized.detach().cpu().numpy()
                        metric_depth_uint8 = (metric_depth_normalized * 255).astype(np.uint8)
                        cv2.imwrite(f"{os.path.join(depth_img_folder, f'mdepth_frame_{i:06d}.png')}", metric_depth_uint8)

                # Use est_scale_hybrid for refined scaler estimation
                from emdb.refine_depth import est_scale_hybrid
                refined_scaler = est_scale_hybrid(
                    slam_depth_raw=depth_map_for_scale,
                    pred_depth=metric_depth_for_scale
                )
                naive_scaler = metric_depth_min / valid_depths.min()
                
                """
                slam depth * scale = pred depth

                pred_cam_t = torch.tensor(traj[:, :3]) * scale
                pred_cam_q = torch.tensor(traj[:, 3:])
                """
                
                # visualization of the SLAM depth
                if len(valid_depths) > 0:
                    if visualize_depth and (i % 10 == 0):  # Save every 10th frame
                        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                        cv2.imwrite(f"{os.path.join(depth_img_folder, f'depth_frame_{i:06d}.png')}", depth_uint8)

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
                
                # Filter valid depths (remove invalid/negative depths)
                valid_mask = depths > 0
                
                # Also remove human region from valid depths (only if --depth-mask is enabled)
                if args.depth_mask and hard_human_mask is not None and hard_human_mask.any():
                    # Resize hard_human_mask to match depth_map dimensions if needed
                    mask_h, mask_w = hard_human_mask.shape
                    depth_h, depth_w = img_shape[0], img_shape[1]
                    
                    if mask_h != depth_h or mask_w != depth_w:
                        # Resize mask to match depth_map dimensions
                        mask_torch = torch.from_numpy(hard_human_mask).float().unsqueeze(0).unsqueeze(0)
                        mask_resized = F.interpolate(
                            mask_torch,
                            size=(depth_h, depth_w),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                        human_mask_resized = mask_resized.cpu().numpy() > mask_thres
                    else:
                        # Same size, just convert to bool if needed
                        human_mask_resized = hard_human_mask > mask_thres if hard_human_mask.dtype == bool else hard_human_mask > mask_thres
                    
                    # Flatten the mask to match depths shape
                    human_mask_flat = human_mask_resized.flatten()
                    valid_mask = valid_mask & (~human_mask_flat)
                
                valid_depths = depths[valid_mask]
            
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
            
            t = dataset.timestamps[frame.frame_id]
            T_WC = as_SE3(frame.T_WC)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            
            # Apply online smoothing
            if smooth_method == 'none':
                # No smoothing, use raw tracking result
                x_smooth, y_smooth, z_smooth = x, y, z
                qx_smooth, qy_smooth, qz_smooth, qw_smooth = qx, qy, qz, qw

            elif smooth_method == 'ema':
                # Exponential moving average with history-weighted smoothing and velocity-based clamping
                ema_history_buffer.append((x, y, z))
                
                if len(ema_history_buffer) < 2:
                    # Not enough history yet, use current pose
                    x_smooth, y_smooth, z_smooth = x, y, z
                    qx_smooth, qy_smooth, qz_smooth, qw_smooth = qx, qy, qz, qw
                    smoothed_pose = (x_smooth, y_smooth, z_smooth, qx_smooth, qy_smooth, qz_smooth, qw_smooth)
                else:
                    # Estimate velocity from history (average velocity over recent frames)
                    history_list = list(ema_history_buffer)
                    velocities = []
                    for j in range(1, len(history_list)):
                        dx = history_list[j][0] - history_list[j-1][0]
                        dy = history_list[j][1] - history_list[j-1][1]
                        dz = history_list[j][2] - history_list[j-1][2]
                        velocities.append(np.array([dx, dy, dz]))
                    
                    if len(velocities) > 0:
                        velocity_magnitudes = np.array([np.linalg.norm(v) for v in velocities])
                        base_velocity = np.median(velocity_magnitudes)
                        
                        # Clamp threshold based on velocity
                        clamp_threshold = base_velocity * ema_clamp_multiplier
                        
                        if ema_clamp_absolute_max is not None:
                            clamp_threshold = min(clamp_threshold, ema_clamp_absolute_max)
                        
                        avg_velocity = np.mean(velocity_magnitudes)
                        max_velocity = np.max(velocity_magnitudes)
                        median_velocity = np.median(velocity_magnitudes)
                        
                        # Print velocity statistics periodically (every 30 frames)
                        if i % 30 == 0:
                            print(f"[EMA Stats] Frame {i}: Base={base_velocity:.6f}, "
                                  f"Mean={avg_velocity:.6f}, Median={median_velocity:.6f}, Max={max_velocity:.6f}")
                            print(f"  Clamp threshold={clamp_threshold:.6f} (multiplier={ema_clamp_multiplier})")
                            if ema_clamp_absolute_max is not None:
                                print(f"  Absolute max limit={ema_clamp_absolute_max:.6f}")
                            print(f"  Note: If too many outliers are NOT clamped, REDUCE multiplier or use more conservative mode.")
                    else:
                        clamp_threshold = float('inf')  # No clamp if no velocity estimate
                        if ema_clamp_absolute_max is not None:
                            clamp_threshold = ema_clamp_absolute_max  # Use absolute max if available
                    
                    # History-weighted EMA: configurable weight distribution
                    n_history = len(ema_history_buffer)
                    
                    weights = np.array([(1 - smooth_alpha) ** (n_history - 1 - i) for i in range(n_history)])
                    weights = weights / weights.sum()  # Normalize
                    
                    # Weighted average of translation
                    x_weighted = sum(w * p[0] for w, p in zip(weights, ema_history_buffer))
                    y_weighted = sum(w * p[1] for w, p in zip(weights, ema_history_buffer))
                    z_weighted = sum(w * p[2] for w, p in zip(weights, ema_history_buffer))
                    
                    # Compute update from current measurement
                    x_update = x - x_weighted
                    y_update = y - y_weighted
                    z_update = z - z_weighted
                    update_norm = np.linalg.norm([x_update, y_update, z_update])
                    
                    # Clamp update if it exceeds threshold
                    was_clamped = False
                    if update_norm > clamp_threshold and clamp_threshold > 0:
                        was_clamped = True
                        scale = clamp_threshold / update_norm
                        x_update_orig = x_update
                        y_update_orig = y_update
                        z_update_orig = z_update
                        x_update *= scale
                        y_update *= scale
                        z_update *= scale
                        
                        # Print clamp information
                        exceed_ratio = update_norm / clamp_threshold if clamp_threshold > 0 else float('inf')
                        # print(f"[EMA Clamp] Frame {i}: Update clamped! (exceeded by {exceed_ratio:.2f}x)")
                        # print(f"  Scale factor: {scale:.4f} (clamped to {scale*100:.1f}% of original)")
                    
                    # Apply clamped update
                    x_smooth = x_weighted + smooth_alpha * x_update
                    y_smooth = y_weighted + smooth_alpha * y_update
                    z_smooth = z_weighted + smooth_alpha * z_update
                    
                    # Rotation: use current (no smoothing for now)
                    # qx_smooth, qy_smooth, qz_smooth, qw_smooth = qx, qy, qz, qw
                    # EMA for rotation (quaternion SLERP approximation)
                    q_prev = np.array([smoothed_pose[3], smoothed_pose[4], smoothed_pose[5], smoothed_pose[6]])
                    q_curr = np.array([qx, qy, qz, qw])
                    # Normalize
                    q_prev = q_prev / np.linalg.norm(q_prev)
                    q_curr = q_curr / np.linalg.norm(q_curr)
                    # Ensure same hemisphere
                    if np.dot(q_prev, q_curr) < 0:
                        q_curr = -q_curr
                    # Linear interpolation (approximation of SLERP)
                    q_smooth = (1 - smooth_alpha) * q_prev + smooth_alpha * q_curr
                    q_smooth = q_smooth / np.linalg.norm(q_smooth)
                    qx_smooth, qy_smooth, qz_smooth, qw_smooth = q_smooth
                    
                    # smoothed_pose = (x_smooth, y_smooth, z_smooth, qx, qy, qz, qw)
                    smoothed_pose = (x_smooth, y_smooth, z_smooth, qx_smooth, qy_smooth, qz_smooth, qw_smooth)

            with open(traj_file, "a") as f:  # append
                f.write(f"{refined_scaler} {x_smooth} {y_smooth} {z_smooth} {qx_smooth} {qy_smooth} {qz_smooth} {qw_smooth}\n")
                # f.write(f"{t} {x_smooth} {y_smooth} {z_smooth} {qx_smooth} {qy_smooth} {qz_smooth} {qw_smooth}\n")
            
            # Update SLAM internal pose with smoothed pose if requested
            if args.update_slam_pose and smooth_method != 'none':
                # Convert smoothed pose back to Sim3 format
                t_smooth = np.array([x_smooth, y_smooth, z_smooth])
                q_smooth = np.array([qx_smooth, qy_smooth, qz_smooth, qw_smooth])
                frame.T_WC = quaternion_translation_to_Sim3(t_smooth, q_smooth, device)
                # Also update the frame in states
                states.set_frame(frame)
        
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
                        f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
                    f.write(f"{refined_scaler} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

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
    

        frame_results, frame_feat_cache = camera_coord_HMR(
            hmr_model, img_ck, box_ck, frame_feat_cache, img_focal, img_center)
        
        # NOTE(yiwen) two frame cache, shape[1] = h*w*mem_t
        # print(f"cache length {frame_feat_cache['layers'][0]['mem_k'].shape}")

        pred_cam.append(frame_results['pred_cam'])
        pred_pose.append(frame_results['pred_pose'])
        pred_shape.append(frame_results['pred_shape'])
        pred_rotmat.append(frame_results['pred_rotmat'])
        pred_trans.append(frame_results['pred_trans'])

        if i==0:
            fix_scaler = refined_scaler
        # world coord camera trajectory
        current_camt = torch.tensor([fix_scaler*x_smooth, fix_scaler*y_smooth, fix_scaler*z_smooth]).unsqueeze(0)
        # current_camt = torch.tensor([fix_scaler*x, fix_scaler*y, fix_scaler*z]).unsqueeze(0)
        current_camq = torch.tensor([qw_smooth, qx_smooth, qy_smooth, qz_smooth]).unsqueeze(0)

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
        imageio.mimsave(out_gif, imgs, fps=20)
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
    if visualize_hcgif:
        del render
    
    print(f"Sequence {root} completed and cleaned up")