# Video-Based Online Human Mesh Recovery


# TODOs
[ ] TRAM 3 frames baseline
[ ] vit huge 的前一个feature，例如 128，768 --> 1, 1024 (在这里保留一些维度)
[ ] 确认原本的写法是 window T frames in， 1 frame out？
[ ] check /home/yiwenzh5/onlineHMR_t/lib/models/configs useful?
[ ] 单个frame作为transformer的一支输入的话，还需不需要加positional encoding(原本是加在T维度上)

[ ] 结果整理成一个表格
[ ] 画图
[ ] 自己的demo video

# Findings
* TRAM 的结构对temporal的信息没有WHAM那么强的依赖性，没有贯穿始终的h0，所以改成3帧对效果的影响相对没有那么大
* TRAM 原本的训练已经使用了sliding window的形式，16frames in 16 frames out
* TRAM 原本的validation sliding window 也是切好，没有重叠，因为最后estimate出来的结果直接“加”在init condition上，不像WHAM把init condition当成nn输入的一部分; "For human trajectory evaluation, we slice a sequence into 100-frame segments and evaluate 3D joint error after aligning the first two frames (W-MPJPE100) or the entire segment (WA-MPJPE100)."

```
# train
python train.py --cfg configs/config_vimo.yaml
```