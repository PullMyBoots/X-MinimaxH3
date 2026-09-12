# RTX 4090 原始权重 Video-VAE Block Compile（Round 13）

日期：2026-08-12  
状态：**原始权重待 Human 连续播放审核；未进入正式默认路由**

## 目的与边界

本轮只优化共享 Video-VAE 解码器，不改变原始 H3 权重、9/11 调度、
DiT 注意力、RMS/AdaLN、scheduler、音频 VAE 或 mux。正式原始权重路线
继续使用未融合的精确 RMS/AdaLN；LoRA 结果不能替代本轮原始权重证据。

候选把 36 个重复 `TransformerBlock` 分别交给 `torch.compile/Inductor`，
关闭 Triton CUDA graphs，并通过 `ContextVar` 按请求选择 eager/compiled，
避免候选策略泄漏到未授权请求。请求若声明使用候选而会话未预编译，运行时
会 fail closed。

## 微基准

| 工作量 | Eager VAE | Compiled VAE | VAE 提速 | 峰值显存 |
|---|---:|---:|---:|---:|
| 864×480×124 | 5.949 s | 5.317 s | 1.119× | 6.598 GiB（相同） |
| 1280×736×362 | 40.658 s | 36.206 s | 1.123× | 16.367 GiB（相同） |

证据：

- `runtime/calibration/workload_routing_round13/vae_blockcompile_480p5_ab.json`
- `runtime/calibration/workload_routing_round13/vae_blockcompile_720p15_ab.json`

浮点输出并非逐位一致，因此它不是数学零误差优化，不能仅凭微基准发布。

## 原始权重复杂任务，同一最终 latent 的隔离 A/B

任务：864×480、243 帧（10.125 秒）、原始权重 9 实际/11 预测、seed
82301；三句普通话、两人连续大幅动作、关门、取火把、登台阶。生成源请求
严格使用 dense SageAttention 与未融合 RMS/AdaLN。

源请求热态结果：

- 总计 80.476 s；
- DiT 64.003 s；
- eager Video-VAE 11.852 s；
- 峰值 8.561 GiB。

保存该请求的最终 video/audio latent 后，在新进程中只改变 Video-VAE：

| 解码器 | 同 latent 解码时间 |
|---|---:|
| 原始 eager | 11.749 s |
| TransformerBlock compile | 10.314 s |
| 提速 | **1.139×** |

像素诊断：98.3752% 的 RGB 字节完全一致，平均绝对差 0.01625/255，最大
绝对差 2/255。两条 MP4 解码后的 PCM SHA-256 完全相同，说明台词与音轨
未被候选改变。抽取 0.75、2.25、4.0、6.0、8.0、9.75 秒共 6 组匹配帧，
未发现色斑、闪烁、重影、形变或动作节点变化。

证据目录：

- `runtime/calibration/workload_routing_round13/original911_480p10_vae_same_latent/`
- A/B 报告：`vae_ab/round13_original911_same_latent.json`
- 视觉门控：`vae_ab/six_frame_ab_contact.jpg`

## 当前判断

这是原始权重和 LoRA 都可复用的小型机械收益，但对完整请求的收益受 VAE
占比限制：本例约省 1.44 秒，即总耗时约 1.8%。720p×15 秒根据独立 VAE
实测约省 4.45 秒。它不能解决长序列 DiT 主瓶颈，也不能被表述成 1.1× 的
整机加速。

候选在 Human 连续播放两条同-latent 视频并确认无差异前保持实验状态；正式
高保真默认仍为 eager VAE。若通过，下一步应同时在原始权重与 LoRA 路线做
端到端复核，然后只对已经覆盖的像素/帧数区间启用内部路由。
