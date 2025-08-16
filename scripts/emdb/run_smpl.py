import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm

from torch.utils.data import default_collate
from lib.models import get_hmr_vimo
from lib.datasets.image_dataset import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=2)
parser.add_argument('--output_dir', type=str, default='results/emdb/smpl')
parser.add_argument('--efficient', action='store_true', help='efficient option, but increase ACC error.')
args = parser.parse_args()

valid_range = (0,2)

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


# Save folder
savefolder = args.output_dir
os.makedirs(savefolder, exist_ok=True)

# HPS model
device = 'cuda'
model = get_hmr_vimo(checkpoint='/ocean/projects/cis240055p/yzhao16/Video-OnlineHMR/results/onlinetram_debug2_LR2_save/checkpoint_best.pth.tar').to(device)

# Predict SMPL on EMDB (subset: spl)
for i, root in enumerate(emdb):
    print('Running HPS on', root)

    seq = root.split('/')[-1]
    imgfiles = sorted(glob(f'{root}/images/*.jpg'))
    annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
    ann = pkl.load(open(annfile, 'rb'))
    
    ext = ann['camera']['extrinsics']
    intr = ann['camera']['intrinsics']
    ann_boxes = ann['bboxes']['bboxes']
    img_focal = (intr[0,0] +  intr[1,1]) / 2.
    img_center = intr[:2, 2]
    
    db = ImageDataset(imgfiles, ann_boxes, img_focal=img_focal, 
                      img_center=img_center, normalization=True)
    dataloader = torch.utils.data.DataLoader(db, batch_size=64, shuffle=False, num_workers=12)

    items = []
    for i in tqdm(range(len(db))):
        item = db[i]
        items.append(item)

    batch = default_collate(items)
    
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items() if type(v)==torch.Tensor}
        # batch.keys() ['img', 'img_idx', 'scale', 'center', 'img_focal', 'img_center']
        out, _ = model.inference_forward(batch)
        # out, _ = model.forward(batch, valid_range=valid_range)
        
        # out.keys() 'pred_cam', 'pred_pose', 'pred_shape', 'pred_rotmat', 'pred_rotmat_0', 'trans_full'

    results = {'pred_cam': out['pred_cam'].cpu(),
            'pred_pose': out['pred_pose'].cpu(),
            'pred_shape': out['pred_shape'].cpu(),
            'pred_rotmat': out['pred_rotmat'].cpu(),
            'pred_trans': out['trans_full'].cpu(),
            'img_focal': img_focal,
            'img_center': img_center}

    np.savez(f'{savefolder}/{seq}.npz', **results)
    

    