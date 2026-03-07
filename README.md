# Video-based Online Human Mesh Recovery

## Installation
1. Clone this repo with the `--recursive` flag (Please follow the corresponding licenses of thirdparty models). 
    ```Bash
    git clone --recursive https://github.com/Tsukasane/Video-OnlineHMR.git
    ```

2. Creating a new anaconda environment.
    ```Bash
    # Base environment installation
    conda create -n onlinehmr python=3.11.5 cmake
    conda activate onlinehmr
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git
    pip install -r requirements.txt
    cd thirdparty

    # MASt3R-SLAM installation
    cd MASt3R-SLAM/
    git submodule update --init --recursive
    pip install --no-build-isolation -e thirdparty/mast3r
    pip install thirdparty/in3d
    pip install --no-build-isolation -e .
    pip install torchcodec==0.1

    # Detectron2 installation
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@a59f05630a8f205756064244bf5beb8661f96180'

    # pytorch3d installation
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

    # MoGe installation
    pip install git+https://github.com/microsoft/MoGe.git

    # DEVA installation
    cd ../Tracking-Anything-with-DEVA
    pip install -e .

    cd ../..
    ```

3. Prepare data and models
    
    Register at [SMPLify](https://smplify.is.tue.mpg.de) and [SMPL](https://smpl.is.tue.mpg.de), whose usernames and passwords will be used by our script to download the SMPL models. Run the following to fetch all models and checkpoints to `data/`. Thirdparty models include [MASt3r-SLAM checkpoints](https://github.com/rmurai0610/MASt3R-SLAM/tree/c3d0d5b67bf51d558d7640ff6032407f68041f92?tab=readme-ov-file#installation), [HMR2.0b checkpoints](https://github.com/shubham-goel/4D-Humans).
    ```Bash
    bash scripts/download_models.sh
    ```

4. Check repo structure

* The pretrained models and templates are placed at
    ```
    data/
    └── pretrain/
        └── hmr2b/
            └── epoch=35-step=1000000.ckpt
        ├── camcalib_sa_biased_l2.ckpt
        ├── cascade_mask_rcnn_vitdet_h_75ep.py
        ├── DEVA-propagation.pth
        ├── mogev2_model.pt
        ├── sam_vit_h_4b8939.pth
        └── vimo_checkpoint.pth.tar
    └── smpl/
        ├── downsample_mat.pkl
        ├── J_regressor_extra.npy
        ├── J_regressor_h36m.npy
        ├── kintree_table.pkl
        ├── SMPL_FEMALE.pkl
        ├── SMPL_MALE.pkl
        ├── smpl_mean_params.npz
        └── SMPL_NEUTRAL.pkl
    └── colors.txt
    └── pascal_occluders.pkl
    ```

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

## Run demo on videos
```bash
# inference
python ./scripts/run_custom.py --video <YOUR/VIDEO/PATH>.mp4 --no-viz --calib false --depth-mask

# inference w tracking (multipersons)
python ./scripts/run_custom_mt.py --video <YOUR/VIDEO/PATH>.mp4 --no-viz --calib false --depth-mask
```
Results are saved to
* ``./res_human_camera/global_results`` (global optimized scene).
* ``./res_human_camera/{trackID}_{videoName}*.npz`` (camera coordinate human motion).
* ``./logs/*_images_incremental_all.txt`` (incremental camera extrinsics).
```bash
# visualization
python visualize_viser.py --human_npz_path <HUMAN/NPZ/PATH>.npz --camera_path <CAMERA/TXT/PATH>.txt
```
You can also use scripts under ``vis_tools/`` to check visualization in ``.gif`` format.

## Training
```Bash
# Fine-tune a online camera coordinates HMR model based on HMR2.0
python train.py --cfg configs/config_vimo.yaml
```
The training log and output will be placed at ``./results``. Set ``self.visualize_spec=True`` in ``./lib/utils/pose_utils.py`` if you want to visualize the 3D joint spectrogram.

## Evaluation
For camera coordinate evaluation, you can keep the current config, then run the above training script. Before starting the resumed training, the eval code will run on the checkpoint you downloaded before and produce camera coordinate metrics.
```Bash
# cam traj eval
python ./scripts/emdb/cam_only_eval.py --camera_root results/emdb2_results/camera

# world coords HMR eval
python ./scripts/emdb/run_eval_mast3r_slam.py --split 2 --human_rootdir results/emdb2_results/human --camera_rootdir results/emdb2_results/camera
```

## Acknowledgement
