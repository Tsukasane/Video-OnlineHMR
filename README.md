# Video-Based Online Human Mesh Recovery


# TODOs
- sequential 之后怎么保证对比的batch_size一致
- 在没有memory时，BEDLAM 作为 trainset 每个连续 sequence 短一点没关系，因为训练时只依赖三帧窗口内部，三种对比方法 validation set 是一致的
- Black image corresponds to 'invalid' label in annotations, and will be aborted in video_dataset
- bbox is kept at the center of the video
- [ ] validation set 从一个video到下一个video的连接处 在 jittering eval的时候需要分开
- [ ] 如果一个 batch 里面的 chunks 是跨越了一个video segment的末尾和另一个video semgent的开始，在建立单个patch的T关系的时候好像会有点问题，TRAM本身没这个问题因为一个chunk都会来自同一个video，只出现在试图建立 batch 内 chunk 之间的关系时
- the step in 16 frames setting and 3 frames setting are different, 有warmup之类的config是以step为单位定义的，另外evaluation按照step来算的话太频繁了一点
- achieve the best performance in the early epoch
- [ ] 单个frame作为transformer的一支输入的话，还需不需要加positional encoding(原本是加在T维度上)
- [ ] architecture design for online parts
- [ ] single joint visualization


# Findings
* TRAM2f and TRAM3f only has small performance gap. The mean difference between TRAM and onlineTRAM is the SA/CA, also the temporal expansion.
* pred 3 (didn't change architecture, i.e. the head dim)，loss ablation on prev / prev + curr / prev + curr + future
* TRAM 的结构对temporal的信息没有WHAM那么强的依赖性，没有贯穿始终的h0，所以改成3帧对效果的影响相对没有那么大
* TRAM 原本的训练已经使用了sliding window的形式，16frames in 16 frames out
* TRAM 原本的validation sliding window 也是切好，没有重叠，因为最后estimate出来的结果直接“加”在init condition上，不像WHAM把init condition当成nn输入的一部分; "For human trajectory evaluation, we slice a sequence into 100-frame segments and evaluate 3D joint error after aligning the first two frames (W-MPJPE100) or the entire segment (WA-MPJPE100)."


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