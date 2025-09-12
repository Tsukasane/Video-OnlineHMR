# Video-Based Online Human Mesh Recovery

--> 我们能否考虑添加一个action_rate 之类的，限制下一个time step的action不能过多的偏离当前帧的估计，因为默认sequential前后帧的人不能有过大的变化（不能瞬移）
--> check visualization的脚本，应该还是一个一个人估计，然后画在一起。
smpl decoder在init pose基础上去做 /ocean/projects/cis240055p/yzhao16/tram/lib/models/modules.py
过往帧的信息太多，少一点信息的使用，可以少一点patch，当前帧可以用更多的patch，试试加权
try reshape to batch
change causal attention to sliding window attention
the order of self and cross atten, does q in cache really mean something?
one frame, all patch, info moves to D dimension (spatial --> channel)
haven't add the bbox
# Findings
# 在highly dynamics的setting下估计很差，在有occlusion的情况下还可以
    - use shorter memory in cache? --> first visualize the correspondense in temporal transformer, which prev frame has higher correspondense?
    - eliminate the pooling, since in some regions the detailed body pose seems inaccurate. Probably the spatial feature shouldn't be further compressed, as the 16*12 feature map is already a downsampled version.
# Causal Transformer Decoder Arch
# input: B, T, H, W, C  H*W=num_patch
# training: windowsize=16, tril mask to apply attention
    # learnable q tokens: a clue for output smpl --> after transformer, pass q tokens to ffn --> then to smpl head
        # self-attn: q tokens & q tokens + mask
        # cross-attn: q tokens & image features + mask
# inference: one frame each time
    # one frame q token each time --> update then get one frame SMPL paras
        # caches use FIFO, to keep the memory updated
        # self cache: previous q tokens (previous smpl clues)
        # cross cache: previous image feature after projection
    # cat all SMPL outputs together


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