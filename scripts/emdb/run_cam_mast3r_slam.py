import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')
sys.path.insert(0, '/ocean/projects/cis240055p/yzhao16/MASt3R-SLAM') # TODO(yiwen) modify this to thirdparty/MASt3R-SLAM

import cv2
import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm

import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
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

# from lib.camera import run_metric_slam, align_cam_to_world
from lib.pipeline.tools import arrange_boxes
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy

from torch.amp import autocast
from detectron2.config import LazyConfig

from lib.datasets.image_dataset import ImageDataset
from torch.utils.data import default_collate

"""
python scripts/emdb/run_cam_mast3r_slam.py --split 2 --output_dir "results/emdb/camera-mast3rslam"

multiprocess: one frame in
slam-frontend 
slam-backend
human cam coord hmr
viser visualization(?)
"""

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


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

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


def register_emdb(args):
    # EMDB dataset and splits
    roots = []
    for p in range(10):
        if p>1: #NOTE(yiwen) debug
            break
        folder = f'/ocean/projects/cis240055p/yzhao16/Video-OnlineHMR/datasets/emdb/EMDB/P{p}'
        root = sorted(glob(f'{folder}/*'))
        roots.extend(root)

    emdb = []
    spl = args.split
    for root in roots:
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))
        if ann[f'emdb{spl}']:
            emdb.append(root)

    return emdb


def init_detector(device):
    cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    return detector


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


def camera_coord_HMR(hmr_model, imgfiles, boxes, cache):
    results, cache = hmr_model.inference_chunk_ar(imgfiles, boxes,
                    img_focal=img_focal, img_center=img_center, cache=cache)
    
    return results, cache
    

if __name__=='__main__':
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='results/emdb/camera')
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)
    # print(config)

    # Save folder
    savefolder = args.output_dir
    os.makedirs(savefolder, exist_ok=True)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    emdb = register_emdb(args) # dataset
    detector = init_detector(device) # ViTDet
    hmr_model = get_hmr_vimo(checkpoint='/ocean/projects/cis240055p/yzhao16/Video-OnlineHMR/results/online_videohmrv2/checkpoint_best.pth.tar') # NOTE(yiwen) change inference checkpoint path here.

    # Estimate camera motion on EMDB (subset: spl)
    for root in emdb:
        print(f'Running on {root}...')

        print(f"Split video to frames and register camera calibration")
        seq = root.split('/')[-1]
        img_folder = f'{root}/images'
        imgfiles = sorted(glob(f'{root}/images/*.jpg'))

        # --lightweight dataset loading (read and sort images)
        img_h, img_w = cv2.imread(os.path.join(img_folder, "00000.jpg")).shape[:2]
        dataset = load_dataset(img_folder)
        dataset.subsample(config["dataset"]["subsample"]) # set to 1 by default
        rimg_h, rimg_w = dataset.get_img_shape()[0] # resized image and resized shape
        
        # load annotations
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))
        ext = ann['camera']['extrinsics']
        intr = ann['camera']['intrinsics']
        ann_boxes = ann['bboxes']['bboxes'] # (2009, 4) NOTE(yiwen) for emdb, boxes are given, if not, run detection?

        cam_int = [intr[0,0], intr[1,1], intr[0,2], intr[1,2]] 
        img_focal = (intr[0,0] +  intr[1,1]) / 2.
        img_center = intr[:2, 2]

        # init
        # db_hmr = ImageDataset(imgfiles, ann_boxes, img_focal=img_focal, 
        #               img_center=img_center, normalization=True)
        # items = []
        # for i in tqdm(range(len(db_hmr))):
        #     item = db_hmr[i]
        #     items.append(item)

        # batch = default_collate(items)
       
        if args.calib:
            with open(args.calib, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            dataset.use_calibration = True
            dataset.camera_intrinsics = Intrinsics.from_calib(
                dataset.img_size, # resized
                img_w, # original
                img_h,
                cam_int,
            )

        keyframes = SharedKeyframes(manager, rimg_h, rimg_w)
        states = SharedStates(manager, rimg_h, rimg_w)
        
        if not args.no_viz: # NOTE(yiwen) 这里的visualization换成viser，但是需要看一下涉及到的multi processing
            viz = mp.Process(
                target=run_visualization,
                args=(config, states, keyframes, main2viz, viz2main),
            )
            viz.start()
        
        model = load_mast3r(device=device) # TODO(yiwen) change the name of model
        model.share_memory()

        has_calib = dataset.has_calib()
        use_calib = config["use_calib"]

        if use_calib and not has_calib:
            print("[Warning] No calibration provided for this dataset!")
            sys.exit(0)
        K = None
        if use_calib:
            K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
                device, dtype=torch.float32
            )
            keyframes.set_intrinsics(K)

        if dataset.save_results: # remove previously saved results
            save_dir, seq_name = eval.prepare_savedir(args, dataset)
            traj_file = save_dir / f"{seq_name}.txt"
            recon_file = save_dir / f"{seq_name}.ply"
            if traj_file.exists():
                traj_file.unlink()
            if recon_file.exists():
                recon_file.unlink()
        
        tracker = FrameTracker(model, keyframes, device)
        last_msg = WindowMsg()

        # start backend
        backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
        backend.start()

        i = 0
        fps_timer = time.time()

        frames = []

        # NOTE(yiwen) frontend loop, incrementally loop all frames
        print(f"Start per frame processing")
        img_chunk = []
        box_chunk = []

        # cam coords
        pred_cam = []
        pred_pose = []
        pred_shape = []
        pred_rotmat = []
        pred_trans = []

        frame_feat_cache = None
        while True:
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

            ### --- Detection ---
            with torch.no_grad():
                with autocast('cuda'):
                    det_out = detector(img_cv2)
                    det_instances = det_out['instances']
                    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                    confs = det_instances.scores[valid_idx].cpu().numpy()

                    boxes = np.hstack([boxes, confs[:, None]])
                    boxes = arrange_boxes(boxes, mode='size', min_size=100)
    
            this_img_file = imgfiles[i]
            if len(img_chunk)==0: # initialize, cache=2
                img_chunk.append(imgfiles[i+2])
                img_chunk.append(imgfiles[i+1])
                box_chunk.append(boxes)
                box_chunk.append(boxes)
            elif len(img_chunk)>=3: # FIFO
                img_chunk.pop(0)
                box_chunk.pop(0)
            img_chunk.append(this_img_file)
            box_chunk.append(boxes)
            
            img_ck = np.array(img_chunk)
            box_ck = np.array(box_chunk).reshape(-1, 5)

            frame_results, frame_feat_cache = camera_coord_HMR(hmr_model, img_ck, box_ck, frame_feat_cache)

            print(f"cache length {frame_feat_cache['layers'][0]['mem_k'].shape[0]}")

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

            if save_frames:
                frames.append(img)


            # Single frame SLAM frontend
            # get frames last camera pose
            T_WC = (
                lietorch.Sim3.Identity(1, device=device) # SE(3)
                if i == 0
                else states.get_frame().T_WC
            )
            frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

            if mode == Mode.INIT:
                # Initialize via mono inference, and encoded features neeed for database
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                states.set_mode(Mode.TRACKING)
                states.set_frame(frame)
                i += 1
                continue

            if mode == Mode.TRACKING:
                add_new_kf, match_info, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                states.set_frame(frame)

            elif mode == Mode.RELOC:
                X, C = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X, C)
                states.set_frame(frame)
                states.queue_reloc()
                # In single threaded mode, make sure relocalization happen for every frame
                while config["single_thread"]:
                    with states.lock:
                        if states.reloc_sem.value == 0:
                            break
                    time.sleep(0.01)

            else:
                raise Exception("Invalid mode")

            # save pre frame results
            if dataset.save_results:
                save_dir, seq_name = eval.prepare_savedir(args, dataset)
                traj_file = save_dir / f"{seq_name}_incremental_all.txt"
                with open(traj_file, "a") as f:  # append
                    t = dataset.timestamps[frame.frame_id]
                    T_WC = as_SE3(frame.T_WC)
                    x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
                    f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

            # save key frame results
            if add_new_kf:
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)

                if dataset.save_results:
                    save_dir, seq_name = eval.prepare_savedir(args, dataset)
                    traj_file = save_dir / f"{seq_name}_incremental_kf.txt"
                    with open(traj_file, "a") as f:  # append
                        t = dataset.timestamps[frame.frame_id]
                        T_WC = as_SE3(frame.T_WC)
                        x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
                        if len(keyframes)==2:
                            f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
                        f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

                # In single threaded mode, wait for the backend to finish
                while config["single_thread"]:
                    with states.lock:
                        if len(states.global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)
            # log time
            if i % 30 == 0:
                FPS = i / (time.time() - fps_timer)
                print(f"FPS: {FPS}")
            i += 1

            # TODO(yiwen) 这里其实应该每一帧去做多进程？然后.join()
            # breakpoint()

            # TODO(yiwen) add depth and world recons



        ### --- Save Global Results ---

        cam_coord_results = {'pred_cam': torch.cat(pred_cam),
                'pred_pose': torch.cat(pred_pose),
                'pred_shape': torch.cat(pred_shape),
                'pred_rotmat': torch.cat(pred_rotmat),
                'pred_trans': torch.cat(pred_trans)}
        

        # NOTE(yiwen) save the final results after global optimization
        if dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(args, dataset)
            eval.save_traj(save_dir, f"{seq_name}_globalOptimized_kf.txt", dataset.timestamps, keyframes)
            # NOTE(yiwen) here we get the parameters from 
            eval.save_reconstruction(
                save_dir,
                f"{seq_name}.ply",
                keyframes,
                last_msg.C_conf_threshold,
            )
            eval.save_keyframes(
                save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
            )

        if save_frames:
            savedir = pathlib.Path(f"logs/frames/{datetime_now}")
            savedir.mkdir(exist_ok=True, parents=True)
            for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
                frame = (frame * 255).clip(0, 255)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{savedir}/{i}.png", frame)

        print("done")
        backend.join()
        if not args.no_viz:
            viz.join()
        """
        camcoord_hmr.join()
        """

        # TODO(yiwen) write a API, given img_folder and cam_int, probably modify from slam demo
        cam_R, cam_T = run_metric_slam(img_folder, masks=None, calib=cam_int)
        # wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

        camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(), 
                'img_focal': cam_int[0], 'img_center': cam_int[2:]}

    