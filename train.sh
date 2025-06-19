#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2-00:00:00

# export HF_HOME=/ocean/projects/cis240055p/yzhao16/hub_model
# export DF_CACHE_DIR=/ocean/projects/cis240055p/yzhao16/hub_model
export CUDA_HOME=/ocean/projects/cis210027p/yzhao16/miniconda3/envs/wham

source /ocean/projects/cis210027p/yzhao16/miniconda3/bin/activate wham
python train.py --cfg configs/config_vimo.yaml