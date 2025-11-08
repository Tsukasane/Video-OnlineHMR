# Video-based Online Human Mesh Recovery

## Installation
1. Clone this repo with the `--recursive` flag.
```Bash TODO(yiwen) change this
git clone --recursive https://github.com/yufu-wang/tram
```
2. Creating a new anaconda environment.
```Bash
conda create -n onlinetram python=3.11 cmake
conda activate onlinetram
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
cd thirdparty

# Mast3r-SLAM installation (modified /scr/yiwenzh5/Video-OnlineHMR/thirdparty/MASt3R-SLAM/mast3r_slam/dataloader.py #L273, TODO switch to a local fork)
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
pip install torchcodec==0.1

# Detectron2 installation
pip install 'git+https://github.com/facebookresearch/detectron2.git@a59f05630a8f205756064244bf5beb8661f96180'

# pytorch3d installation
conda install -c conda-forge libstdcxx-ng
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# MoGe installation
pip install git+https://github.com/microsoft/MoGe.git
```

3. Download Checkpoints 
* Mast3r-SLAM checkpoints as released [here](https://github.com/rmurai0610/MASt3R-SLAM/tree/c3d0d5b67bf51d558d7640ff6032407f68041f92?tab=readme-ov-file#installation).
```Bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```
* MoGe-v2 --> ``./pretrain/mogev2_model.pt``
* Segment-Anything --> ``./pretrain/sam_vit_h_4b8939.pth``

## Prepare data
Register at [SMPLify](https://smplify.is.tue.mpg.de) and [SMPL](https://smpl.is.tue.mpg.de), whose usernames and passwords will be used by our script to download the SMPL models. In addition, we will fetch trained checkpoints and an example video. Note that thirdparty models have their own licenses. 

Run the following to fetch all models and checkpoints to `data/`. It also downloads `example_video.mov` for the demo.
```Bash
bash scripts/download_models.sh
```

## Run demo on videos

```bash
python ./scripts/emdb/run_custom.py --video <YOUR/VIDEO/PATH>.mp4 --no-viz --calib false
```

## Preparation
* Data organization
    ```
    ./datasets
        |--3dpw
        |--bedlam_30fps
        |--dataset_ann
        |--emdb
        |--h36m
        |--training_data
    ```
* Reset ``ROOT`` and ``DATASET_NPZ_PATH`` in ``./data_config.py`` to your own folders.


## Train
```
python train.py --cfg configs/config_vimo.yaml
```
* Modify the ``valid_range`` in ``./configs/config_vimo.yaml`` to ablation on estimated frame number.

## Evaluation
```
# online inference (set --calib true on emdb2)
python scripts/emdb/run_cam_mast3r_slam.py --split 2 --output_dir "results/emdb/camera-mast3rslam" --no-viz --calib true
# cam traj eval
python ./scripts/emdb/cam_only_eval.py --camera_root <PRED_CAMERA_DIR>
# world coords HMR
python ./scripts/emdb/run_eval_mast3r_slam.py --split 2 --human_rootdir <PRED_HUMAN_DIR> --camera_rootdir <PRED_CAMERA_DIR>
"""
```
The output camera trajectory is saved to ``./logs``, camera coordinates hmr is saved to ``./res_human_camera``.

**Metrics**
- Pose and Shape
    * MPJPE: mean per-joint error.
    * PA-MPJPE: Procrustes-aligned per-joint error.
    * PVE: per-vertex error.
    * ACCEL: acceleration error against the ground truth acceleration.

- Camera Trajectory
    * ATE: absolute trajectory error
    * ATE-S: using our estimated scale

- Human Trajectory
    * W-MPJPE100: slice a sequence into 100-frame segments and evaluate 3D joint error after aligning the first two frames
    * WA-MPJPE100: Align the entire segment
    * ERVE: egocentric-frame root velocity error (measure the root motion accuracy)
    * RTE: root translation error normalized by the total displacement after rigid alignment without scaling