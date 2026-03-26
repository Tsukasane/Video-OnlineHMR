#!/bin/bash
gdown --folder https://drive.google.com/drive/folders/1GWQKcrpJl84g9ouG5L3S1uy-eIloZJGI?usp=sharing -O tmp_download/

mkdir -p data/pretrain data/smpl

mv tmp_download/cascade_mask_rcnn_vitdet_h_75ep.py data/pretrain/

mv tmp_download/downsample_mat.pkl data/smpl/
mv tmp_download/J_regressor_extra.npy data/smpl/
mv tmp_download/J_regressor_h36m.npy data/smpl/
mv tmp_download/kintree_table.pkl data/smpl/
mv tmp_download/smpl_mean_params.npz data/smpl/

mv tmp_download/colors.txt data/

rm -rf tmp_download/