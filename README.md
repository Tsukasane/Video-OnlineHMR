# Video-Based Online Human Mesh Recovery



# Metrics
- Pose and Shape
MPJPE: mean per-joint error
PA-MPJPE: Procrustes-aligned per-joint error
PVE: per-vertex error
ACCEL: acceleration error against the ground truth acceleration.

- Camera Trajectory
ATE: absolute trajectory error
ATE-S: using our estimated scale

- Human Trajectory
W-MPJPE100: slice a sequence into 100-frame segments and evaluate 3D joint error after aligning the first two frames
WA-MPJPE100: Align the entire segment
ERVE: egocentric-frame root velocity error (measure the root motion accuracy)
RTE: root translation error normalized by the total displacement after rigid alignment without scaling


# Settings
folder -- expansion
/home/yiwenzh5/onlineHMR_t/results/tram_prev+curr+future 24


# Results
Official released 16 frames tram
pa_mpjpe 34.42669
mpjpe 45.53531
pve 50.97477
accel 4.473362
wa_mpjpe 78.92915
w_mpjpe 221.90569
rte 2.0372977
erve 8.908163
ate 0.5238046122890667
ate_s 1.346546306407243


# TODOs
- [ ] TRAM 3 frames 也需要改成 16 frame chunk算valacc
- [ ] we always use the best model, but the standard is just pa-mpjpe, sometimes it is not for accer
- [ ] 单个frame作为transformer的一支输入的话，还需不需要加positional encoding(原本是加在T维度上)
- [ ] architecture design for online parts
- [ ] tune model parameters smaller than 24
- [ ] accel debug，可能一个batch的window没有取到连续的，用demo visualization
- [ ] accel measures how similar the gt accel compared to the pred accel


# Findings
* the author has mentioned about the black image cropping in their supplementary material.
* compare tram_online_v524 and retrain_online_v1, longer temporal expansion leads to smaller accer. 对前两个指标的影响基本可以忽略不计，增加了计算量（balance）,不用再加了
* pred 3 (didn't change architecture, i.e. the head dim)，loss ablation on prev / prev + curr / prev + curr + future
* prev future ablation 不是pred 不监督，而是直接不pred
* TRAM 的结构对temporal的信息没有WHAM那么强的依赖性，没有贯穿始终的h0，所以改成3帧对效果的影响相对没有那么大
* TRAM 原本的训练已经使用了sliding window的形式，16frames in 16 frames out
* TRAM 原本的validation sliding window 也是切好，没有重叠，因为最后estimate出来的结果直接“加”在init condition上，不像WHAM把init condition当成nn输入的一部分; "For human trajectory evaluation, we slice a sequence into 100-frame segments and evaluate 3D joint error after aligning the first two frames (W-MPJPE100) or the entire segment (WA-MPJPE100)."


## Run demo on videos
This project integrates the complete 4D human system, including tracking, slam, and 4D human capture in the world space. We separate the core functionalities into different scripts, which should be run **sequentially**. Each step will save its result to be used by the next step. All results will be saved in a folder with the same name as the video.

```bash
# 1. Run Masked Droid SLAM (also detect+track humans in this step)
python scripts/estimate_camera.py --video "./example_video000088_trampcf.mp4"
# # -- You can indicate if the camera is static. The algorithm will try to catch it as well.
# python scripts/estimate_camera.py --video "./another_video.mov" --static_camera

# 2. Run 4D human capture with VIMO.
# NOTE(yiwen) modify #Line45 checkpoint path
# NOTE(yiwen) modify valid_range in config 
# /home/yiwenzh5/onlineHMR_t/lib/models/configs/config_vimo.yaml
python scripts/estimate_humans.py --video "./example_video000088_trampcf.mp4"

# 3. Put everything together. Render the output video.
python scripts/visualize_tram.py --video "./example_video000088_trampcf.mp4"
```

```
# modify valid_range in /home/yiwenzh5/onlineHMR_t/lib/models/configs/config_vimo.yaml, also run_smpl.py
# evaluation
bash scripts/emdb/run.sh
```

```
# train
python train.py --cfg configs/config_vimo.yaml
```
