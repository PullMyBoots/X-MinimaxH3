# Native H3 与 ComfyUI 运行时间对照（RTX 4090）

测试日期：2026-08-10。

## 公平任务契约

- GPU：RTX 4090 24 GB，单卡，batch 1
- 任务：T2AV（文生视频并生成同步立体声音频）
- 画布：864×480
- 帧数 / FPS：124 / 24，成片 5.167 秒
- 采样：RES multistep，20 个实际 DiT 步，0 个预测步
- seed：4404
- 权重：相同的 Pruned INT8 ConvRot DiT、NVFP4/AWQ Qwen、FP16 视频
  VAE 与 FP32 音频 VAE
- prompt、编码格式：三组运行完全相同

这里把两种 ComfyUI 对照分开，避免把已有算法优化伪装成 stock
ComfyUI：

1. `stock ComfyUI`：不加载 V7 代码钩子；只启用正常可选的 Triton、
   SageAttention 与 async offload。
2. `ComfyUI + V7-D exact`：加载已验收的底层等价融合，但将质量计划设为
   `native20`，关闭全部预测步。它不改变 20 步采样预算。

## 实测结果

| 实现 | 冷态完整任务 | 热态完整任务 | 冷态 20 步采样 | 热态 20 步采样 | GPU 峰值 |
|---|---:|---:|---:|---:|---:|
| 当前 standalone native correctness baseline | 496.211 s | 尚未实现进程内复用 | 334.305 s | 尚未实现 | 22.664 GiB PyTorch allocated；23.797 GiB reserved |
| stock ComfyUI | 211.351 s | 86.618 s | 143.671 s | 73.054 s | 23,352 / 22,976 MiB，来自 nvidia-smi 冷/热采样 |
| ComfyUI + V7-D exact | 203.315 s | 77.559 s | 138.222 s | 66.835 s | 23,000 / 22,976 MiB，来自 nvidia-smi 冷/热采样 |

相对当前 standalone native 冷态总耗时：

- stock ComfyUI 冷态快 2.348×，热态快 5.729×；
- V7-D exact 冷态快 2.441×，热态快 6.398×。

只比较 20 步采样阶段：

- native 比 stock ComfyUI 冷/热采样分别慢 2.327× / 4.576×；
- native 比 V7-D exact 冷/热采样分别慢 2.419× / 5.002×。

## Native 阶段分解

| 阶段 | 时间 | 峰值分配显存 |
|---|---:|---:|
| 量化文本编码 | 49.796 s | 1.069 GiB |
| DiT 装配与文本 refiner | 57.910 s | 19.642 GiB |
| 20 步去噪 | 334.305 s | 22.664 GiB |
| 视频 VAE 装载 / 解码 | 27.168 / 8.822 s | 4.870 / 6.600 GiB |
| 音频 VAE 装载 / 解码 | 15.739 / 0.151 s | 0.581 / 0.693 GiB |
| H.264 + AAC mux | 1.313 s | 0.581 GiB |

第一步因 kernel warmup 为 20.824 秒；第 2–20 步稳定在约
16.45–16.62 秒。ComfyUI 热态实际步约 3.50 秒/步（stock）或
3.19 秒/步（V7-D exact）。

## 正确性门控

三组 MP4 均为 864×480、124 帧、24 FPS、H.264 + 32 kHz 双声道 AAC。
已对 native、stock ComfyUI 与 V7-D exact 分别抽取 8 个等间隔帧并人工
检查。三者均未见彩色色块、散光重影、整体结构崩坏或明显时序跳变；
视觉硬门控通过。SSIM 未作为硬门控。

## 结论与边界

独立计算图已经真正生成了有效的音视频 MP4，但它目前是 correctness
baseline，不是性能发布版。差距主要来自两点：

1. native 每次进程运行都重新流式编码文本、装配 21 GB DiT、再分别装载
   两个 VAE；尚无进程内模型/文本/layout 热缓存。
2. native DiT 只有生产 ConvRot INT8 与 SageAttention，尚未接入 V7-D
   已验证的 fused RMS/AdaLN、partial RMS/RoPE、fused SwiGLU、chunked MLP、
   resident weights 以及优化的 VAE 路径。

因此本次结果证明“脱离 ComfyUI 的数学图与媒体流水线可以跑通”，同时也
给出了下一阶段必须追平的可信 comparator。当前 runner 的 VAE 图仍从本地
Apache MiniMax/LightX 源树显式导入；将其 vendor 到 release 包并补齐进程内
residency 后，才能声称单目录独立发布。

## 证据文件

- Native 计时：`runtime/outputs/native_baseline_480p5s_20actual_seed4404.timing.json`
- stock ComfyUI：`runtime/comparisons/comfy_stock_native20_{cold,hot}_480p5s_seed4404.json`
- V7-D exact：`runtime/comparisons/comfy_v7d_native20_{cold,hot}_480p5s_seed4404.json`
- 多帧门控：`runtime/visual_gate/`
