# RTX 4090 Native 与 Comfy V7-D 严格 A/B

日期：2026-08-11  
任务：原始权重、同prompt、同seed、同画布、同帧数、同调度与同9/11实际/预测步

## 契约

- prompt：中世纪十字军泥泞行军与城堡围攻的同一完整英文提示词；
- seed：62002；
- 画布：1280x736；
- 帧数：124，24 fps；
- scheduler/sampler：simple / res-multistep；
- 20个调度点，实际步为`0,1,2,3,4,8,12,16,19`，其余11步预测；
- 量化与注意力：相同Pruned INT8 ConvRot、CUDA SM89与SageAttention路径。

## 结果

| 实现 | 热任务端到端 | 采样/去噪 | Video VAE | Mux | 峰值显存证据 |
|---|---:|---:|---:|---:|---:|
| Comfy V7-D等价融合 | 110.970 s | 94.205 s | 13.755 s | 2.537 s | 22,722 MiB NVML |
| 独立Native Block | 101.371 s | 78.316 s | 17.046 s | 2.737 s | 8.821 GiB CUDA allocated |

Native严格端到端加速为`1.0947x`。两种峰值显存来自不同观测口径，不能做逐字节
等价比较；它们足以证明Native Block没有依赖整DiT常驻，并保留了约13 GiB量级的
设备空间。

## 质量门控

两条视频均抽取12帧覆盖全时序检查。未见彩色色斑、散光重影、分块噪声、结构
崩坏或时间跳变。SSIM仅作为轨迹诊断：视频All为0.880156；两边编码码率不同，
解码压缩差异也包含在该数值中。音频apsnr为167.5/168.1 dB，接近逐样本一致。
SSIM不是硬门控，也不能用来宣称一个视频主观上更好。

## 证据

- Comfy请求：`optimizations/v7d_release_4090/results/strict_ab_v7dr_720p5s_seed62002.json`
- Comfy服务Profile：`profile/raw/strict_ab_v7dr_720p5s_seed62002_server.json`
- Comfy视频：`ComfyUI/output/video/strict_ab_v7dr_720p5s_seed62002_00001_.mp4`
- Native报告：`release/serve/runtime/calibration/block_offload_720p/block_720p5s_v7d911_hot_session.json`
- Native视频：`release/serve/runtime/calibration/block_offload_720p/block_720p5s_v7d911_request1_1280x736_124f_seed62002.mp4`
- 接触表：`release/serve/runtime/calibration/strict_comfy_ab_720p5s/`

