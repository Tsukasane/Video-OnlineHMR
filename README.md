# Video-based Online Human Mesh Recovery

## Installation
1. Clone this repo with the `--recursive` flag.
```Bash
git clone --recursive https://github.com/yufu-wang/tram
```
2. Creating a new anaconda environment.
```Bash
conda create -n tram python=3.10 -y
conda activate tram
bash install.sh
```
3. Compile DROID-SLAM. If you encountered difficulty in this step, please refer to its [official release](https://github.com/princeton-vl/DROID-SLAM) for more info. In this project, DROID is modified to support masking. 
```Bash
cd thirdparty/DROID-SLAM
python setup.py install
cd ../..
```

## Prepare data
Register at [SMPLify](https://smplify.is.tue.mpg.de) and [SMPL](https://smpl.is.tue.mpg.de), whose usernames and passwords will be used by our script to download the SMPL models. In addition, we will fetch trained checkpoints and an example video. Note that thirdparty models have their own licenses. 

Run the following to fetch all models and checkpoints to `data/`. It also downloads `example_video.mov` for the demo.
```Bash
bash scripts/download_models.sh
```

## Run demo on videos
Run the following scripts **sequentially**. All results will be saved in a folder with the same name as the video.

1. Run Masked Droid SLAM (also detect+track humans in this step)
    ```bash
    python scripts/estimate_camera.py --video "./example_video1.mov"
    # You can indicate if the camera is static. The algorithm will try to catch it as well.
    python scripts/estimate_camera.py --video "./another_video.mov" --static_camera
    ```

2. Run 4D human capture with VIMO.
    ```bash
    # modify #Line45 checkpoint path
    # modify valid_range in config ./lib/models/configs/config_vimo.yaml
    python scripts/estimate_humans.py --video "./example_video1.mov"
    ```

3. Put everything together. Render the output video.
    ```bash
    python scripts/visualize_tram.py --video "./example_video1.mov"
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
# modify valid_range in ./lib/models/configs/config_vimo.yaml, also run_smpl.py
bash scripts/emdb/run.sh
# or separately
python scripts/emdb/run_cam.py --split 2 --output_dir "results/emdb/camera"
python scripts/emdb/run_smpl.py --split 2 --output_dir "results/emdb/smpl"
python scripts/emdb/run_eval.py --split 2 --input_dir "results/emdb"

# for mast3r-slam evaluation
# world coords HMR
python scripts/emdb/run_eval_mast3r_slam.py --split 2 --input_dir ./res_human_camera
# cam traj eval
python /ocean/projects/cis240055p/yzhao16/Video-OnlineHMR/scripts/emdb/cam_only_eval.py
# online inference (set --calib true on emdb2)
python scripts/emdb/run_cam_mast3r_slam.py --split 2 --output_dir "results/emdb/camera-mast3rslam" --no-viz --calib true
```

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