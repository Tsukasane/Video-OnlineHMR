#!/bin/bash
#SBATCH -p ROBO
#SBATCH -A cis240055p
#SBATCH -N 1 
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00


# export HF_HOME=/ocean/projects/cis240055p/yzhao16/hub_model
# export DF_CACHE_DIR=/ocean/projects/cis240055p/yzhao16/hub_model
export CUDA_HOME=/ocean/projects/cis210027p/yzhao16/miniconda3/envs/mast3r-slam

source /ocean/projects/cis210027p/yzhao16/miniconda3/bin/activate mast3r-slam
python train.py --cfg configs/config_vimo.yaml