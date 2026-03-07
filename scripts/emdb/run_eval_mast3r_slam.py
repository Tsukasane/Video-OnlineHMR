import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict

from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import *
from lib.camera.slam_utils import eval_slam

"""
python ./scripts/emdb/run_eval_mast3r_slam.py --split 2 --human_rootdir ./res_human_camera --camera_rootdir ./hard_mask/camera
"""

failed_seqs = []
parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--input_dir', type=str, default='results/emdb')
parser.add_argument('--human_rootdir', type=str, default='res_human_camera')
parser.add_argument('--camera_rootdir', type=str, default='logs')
args = parser.parse_args()

# EMDB dataset and splits
roots = []
for p in range(10):
    folder = f'./../../datasets/emdb/P{p}'
    root = sorted(glob(f'{folder}/*'))
    roots.extend(root)

emdb = []
spl = args.split
for root in roots:
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    if ann[f'emdb{spl}']:
        emdb.append(root)

failed_cnt = 0
for f in failed_seqs:
    emdb.pop(f-failed_cnt)
    failed_cnt += 1

# SMPL
smpl = SMPL()
smpls = {g:SMPL(gender=g) for g in ['neutral', 'male', 'female']}


# Evaluations: world-coordinate SMPL
accumulator = defaultdict(list)
m2mm = 1e3
human_traj = {}
total_invalid = 0

for root in tqdm(emdb):
    # GT
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))

    ext = ann['camera']['extrinsics']  # in the forms of R_cw, t_cw
    intr = ann['camera']['intrinsics']
    img_focal = (intr[0,0] +  intr[1,1]) / 2.
    img_center = intr[:2, 2]

    valid = ann['good_frames_mask']
    gender = ann['gender']
    poses_body = ann["smpl"]["poses_body"]
    poses_root = ann["smpl"]["poses_root"]
    betas = np.repeat(ann["smpl"]["betas"].reshape((1, -1)), repeats=ann["n_frames"], axis=0)
    trans = ann["smpl"]["trans"]
    total_invalid += (~valid).sum()

    tt = lambda x: torch.from_numpy(x).float()
    gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root), betas=tt(betas), transl=tt(trans),
                    pose2rot=True, default_smpl=True)
    gt_vert = gt.vertices
    gt_j3d = gt.joints[:,:24] 
    gt_ori = axis_angle_to_matrix(tt(poses_root))

    # Groundtruth local motion
    poses_root_cam = matrix_to_axis_angle(tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))
    gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam, betas=tt(betas),
                           pose2rot=True, default_smpl=True)
    gt_vert_cam = gt_cam.vertices
    gt_j3d_cam = gt_cam.joints[:,:24] 
    
    # PRED
    seq = "_".join(root.split('/')[-2:])
    pred_res = dict(np.load(f'{args.human_rootdir}/{seq}.npz'))
    
    pred_rotmat = torch.tensor(pred_res['pred_rotmat']) # T, 24, 3, 3
    pred_shape = torch.tensor(pred_res['pred_shape']) # T, 10
    pred_trans = torch.tensor(pred_res['pred_trans']) # T, 1, 3

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.repeat(len(pred_shape), 1)

    pred = smpls['neutral'](body_pose=pred_rotmat[:,1:], 
                            global_orient=pred_rotmat[:,[0]], 
                            betas=pred_shape, 
                            transl=pred_trans.squeeze(),
                            pose2rot=False, 
                            default_smpl=True)
    pred_vert = pred.vertices
    pred_j3d = pred.joints[:, :24]

    cam_prefix = "_".join(seq.split("_")[:2])
    cam_root = f"./{args.camera_rootdir}/{seq}_images_incremental_all.txt"

    with open(cam_root, "r") as f:
        lines = f.readlines()

    pred_camt_ls = []
    pred_camr_ls = []
    depth_frame_register = 2
    if depth_frame_register:
        naive_scaler = 0.0
        for fm in range(depth_frame_register):
            naive_scaler += list(map(float, lines[fm].strip().split()))[0]
        naive_scaler /= depth_frame_register
    else:
        naive_scaler = 1.0
    scaler_cnt = 0

    scale_ls = []
    for l_id, line in enumerate(lines):
        vals = list(map(float, line.strip().split()))
        scale_ls.append(vals[0])
        tx, ty, tz = vals[1:4]
        qx, qy, qz, qw = vals[4:]
        wxyz = [qw, qx, qy, qz] # to wxyz
    
        current_camt = torch.tensor([tx, ty, tz]).unsqueeze(0)
        current_camq = torch.tensor([qw, qx, qy, qz]).unsqueeze(0)
        current_camr = quaternion_to_matrix(current_camq)

        pred_camt_ls.append(current_camt)
        pred_camr_ls.append(current_camr)
    
    scaler_cnt /= len(lines)

    # pred_camt = scaler_cnt * torch.stack(pred_camt_ls).squeeze(1) # T, 3
    pred_camt = torch.stack(pred_camt_ls).squeeze(1) # T, 3
    pred_camr = torch.stack(pred_camr_ls).squeeze(1) # T, 3, 3

    pred_camt = pred_camt * naive_scaler

    pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:,None]
    pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:,None]
    pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:,0])
    pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)

    valid = valid[1:]
    # Valid mask

    gt_j3d = gt_j3d[1:][valid]
    gt_ori = gt_ori[1:][valid]
    pred_j3d_w  = pred_j3d_w[valid]
    pred_ori_w = pred_ori_w[valid]

    gt_j3d_cam = gt_j3d_cam[1:][valid]
    gt_vert_cam = gt_vert_cam[1:][valid]
    pred_j3d = pred_j3d[valid]
    pred_vert = pred_vert[valid]

    # <======= Evaluation on the local motion
    pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
        [pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam], pelvis_idxs=[1,2]
    )
    
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
    pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
    mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
    pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

    accel = compute_error_accel(joints_pred=pred_j3d.cpu(), joints_gt=gt_j3d_cam.cpu())[1:-1]
    accel = accel * (30 ** 2)       # per frame^s to per s^2

    accumulator['pa_mpjpe'].append(pa_mpjpe)
    accumulator['mpjpe'].append(mpjpe)
    accumulator['pve'].append(pve)
    accumulator['accel'].append(accel)
    # =======>

    # <======= Evaluation on the global motion
    chunk_length = 100
    w_mpjpe, wa_mpjpe = [], []
    for start in range(0, valid.sum() - chunk_length, chunk_length):
        end = start + chunk_length
        if start + 2 * chunk_length > valid.sum(): end = valid.sum() - 1
        
        target_j3d = gt_j3d[start:end].clone().cpu()
        pred_j3d = pred_j3d_w[start:end].clone().cpu()
        
        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)
        
        w_jpe = compute_jpe(target_j3d, w_j3d)
        wa_jpe = compute_jpe(target_j3d, wa_j3d)
        w_mpjpe.append(w_jpe)
        wa_mpjpe.append(wa_jpe)

    w_mpjpe = np.concatenate(w_mpjpe) * m2mm
    wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
    # =======>

    # <======= Evaluation on the entier global motion
    # RTE: root trajectory error
    pred_j3d_align = first_align_joints(gt_j3d, pred_j3d_w)
    rte_align_first= compute_jpe(gt_j3d[:,[0]], pred_j3d_align[:,[0]])
    rte_align_all = compute_rte(gt_j3d[:,0], pred_j3d_w[:,0]) * 1e2 

    # ERVE: Ego-centric root velocity error
    erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm
    # =======>

    # <======= Record human trajectory
    human_traj[seq] = {'gt': gt_j3d[:,0], 'pred': pred_j3d_align[:, 0]}
    # =======>

    accumulator['wa_mpjpe'].append(wa_mpjpe)
    accumulator['w_mpjpe'].append(w_mpjpe)
    accumulator['rte'].append(rte_align_all)
    accumulator['erve'].append(erve)
    
for k, v in accumulator.items():
    accumulator[k] = np.concatenate(v).mean()


# Save evaluation results
for k, v in accumulator.items():
    print(k, accumulator[k])