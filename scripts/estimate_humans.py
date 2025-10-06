import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import argparse
import numpy as np
from glob import glob

from lib.models import get_hmr_vimo
from lib.pipeline import visualize_tram

import time
import random

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


timestep5 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='./example_video.mov', help='input video')
parser.add_argument('--max_humans', type=int, default=20, help='maximum number of humans to reconstruct')
args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
hps_folder = f'{seq_folder}/hps'
os.makedirs(hps_folder, exist_ok=True)

##### Preprocess results from estimate_camera.py #####
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
camera = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
tracks = np.load(f'{seq_folder}/tracks.npy', allow_pickle=True).item()

img_focal = camera['img_focal']
img_center = camera['img_center']

# Sort the tracks by length
tid = [k for k in tracks.keys()]
lens = [len(trk) for trk in tracks.values()]
rank = np.argsort(lens)[::-1]
tracks = [tracks[tid[r]] for r in rank]

##### Run HPS (here we use tram) #####
print('Estimate HPS ...')
model = get_hmr_vimo(checkpoint='/ocean/projects/cis240055p/yzhao16/Video-OnlineHMR/results/online_videohmrv2/checkpoint_best.pth.tar') # NOTE(yiwen) change inference checkpoint path here.

timestep6 = time.time()
all_fps = 0
for k, trk in enumerate(tracks):
    timestep7 = time.time()
    valid = np.array([t['det'] for t in trk])
    boxes = np.concatenate([t['det_box'] for t in trk])
    frame = np.array([t['frame'] for t in trk]) # (129,)
    # results = model.inference(imgfiles, boxes, valid=valid, frame=frame,
    #                           img_focal=img_focal, img_center=img_center)

    results = model.inference_ar(imgfiles, boxes, valid=valid, frame=frame, # temporally no use
                              img_focal=img_focal, img_center=img_center)
    
    if results is not None:
        np.save(f'{hps_folder}/hps_track_{k}.npy', results)

    timestep8 = time.time()

    this_fps = frame.shape[0] / (timestep8 - timestep7)
    all_fps += this_fps
    print(f"Human {k+1} FPS: {this_fps}")
    
    if k+1 >= args.max_humans:
        break

print(f"FPS: {all_fps / (k+1)}")
