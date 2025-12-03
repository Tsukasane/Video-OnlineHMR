import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
import numpy as np
import pickle as pkl
from glob import glob

from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import *

from lib.camera.slam_utils import eval_slam

"""
python ./scripts/emdb/cam_only_eval.py --camera_root ./soft_mask_mast3r_slam/camera
"""

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--input_dir', type=str, default='results/emdb')
parser.add_argument('--camera_root', type=str, default='logs')
args = parser.parse_args()
input_dir = args.input_dir

slam_method = "mast3r_slam"

# EMDB dataset and splits
roots = []
for p in range(10):
    folder = f'./../datasets/emdb/P{p}'
    root = sorted(glob(f'{folder}/*'))
    roots.extend(root)

emdb = []
spl = args.split
for root in roots:
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    if ann[f'emdb{spl}']:
        emdb.append(root)

# Evaluation: Camera motion
results = {}
all_ate = 0
ate_cnt = 0

for root in emdb:
    # Annotation
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))

    print(f"ann {annfile}")

    # GT
    ext = ann['camera']['extrinsics']
    cam_r = ext[:,:3,:3].transpose(0,2,1)
    cam_t = np.einsum('bij, bj->bi', cam_r, -ext[:, :3, -1]) # T, 3 (all frames in one video)
    cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy() # T, 4

    # PRED
    if slam_method=="mask_droid_slam":
        seq = root.split('/')[-1]
        pred_cam = dict(np.load(f'{input_dir}/camera/{seq}.npz'))

        pred_camt = torch.tensor(pred_cam['pred_cam_T'])
        pred_camr = torch.tensor(pred_cam['pred_cam_R'])
        pred_camq = matrix_to_quaternion(pred_camr)
    
    elif slam_method=="mast3r_slam":
        root_dir = args.camera_root
        scene_name = "_".join(root.split('/')[-2:]) # P0_09
        
        txt_file = os.path.join(root_dir, f"{scene_name}_images_incremental_all.txt")
        
        with open(txt_file, "r") as f:
            lines = f.readlines()

        pred_camt_ls = []
        pred_camq_ls = []
        # init frame
        pred_camt_ls.append([0.0, 0.0, 0.0])
        pred_camq_ls.append([1.0, 0.0, 0.0, 0.0]) # wxyz
        for l_id, line in enumerate(lines):
            # print(f"debug -- l_id {l_id}")
            vals = list(map(float, line.strip().split()))
            timestep = vals[0]
            tx, ty, tz = vals[1:4]
            qx, qy, qz, qw = vals[4:]
            wxyz = [qw, qx, qy, qz]
            # to wxyz
            pred_camt_ls.append(vals[1:4])
            pred_camq_ls.append(wxyz)

        pred_camt = torch.tensor(pred_camt_ls)
        pred_camq = torch.tensor(pred_camq_ls)

    pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()
    try:
        stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
    except:
        breakpoint()
        print(f"The sequence is not comparible with gt")
        # breakpoint()
        print(f"pred length {pred_traj.shape};  gt length {cam_t.shape}")
    # stats_metric, traj_ref, traj_est = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)
  
    current_ate = np.mean(stats_slam['mean'])
    print(f"debug -- current ate:{current_ate}")

    all_ate += current_ate
    ate_cnt += 1


print(f"average ate: {all_ate / ate_cnt}")



# ate = np.mean([re['stats_slam']['mean'] for re in results.values()])
# ate_s = np.mean([re['stats_metric']['mean'] for re in results.values()])
