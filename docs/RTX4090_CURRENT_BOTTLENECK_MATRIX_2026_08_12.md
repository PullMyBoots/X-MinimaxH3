# RTX 4090 Native H3当前跨分辨率/时长瓶颈报告

日期：2026-08-12  
硬件：NVIDIA RTX 4090 24 GB，单卡，batch size 1  
范围：独立Native后端；原始权重9实际/11预测、Larry LoRA 6完整步

## 结论

热态任务的第一瓶颈在所有已测档位都是DiT去噪。低分辨率短视频中Video-VAE已经
占20%到30%，但随着分辨率和时长增加，约10万packed tokens的全局Attention和
INT8 ConvRot GEMM使DiT占比升至84%到87%。720p/15秒当前最大的可优化子瓶颈是
长序列Attention，第二瓶颈是Linear/MLP GEMM，Video-VAE居第三。

首次请求必须单独看：模型主机缓存和装配约58到74秒，可能超过360p/5秒或接近
480p/10秒本身。常驻服务后该成本不随每条视频重复。

## 热态阶段拆账

| 引擎/任务 | 完整时间 | DiT | DiT占比 | Video-VAE | VAE占比 | Mux | 峰值CUDA allocated |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始9/11，360p/5s | 17.380 s | 12.165 s | 70.0% | 3.568 s | 20.5% | 0.683 s | 5.810 GiB |
| 原始9/11，480p/5s | 33.569 s | 25.454 s | 75.8% | 5.966 s | 17.8% | 1.213 s | 6.605 GiB |
| 原始9/11，480p/10s复杂台词 | 81.628 s | 64.089 s | 78.5% | 12.113 s | 14.8% | 2.225 s | 8.279 GiB |
| 原始9/11，720p/5s | 101.371 s | 78.316 s | 77.3% | 17.046 s | 16.8% | 2.737 s | 8.821 GiB |
| 原始9/11，720p/15s | 468.186 s | 406.453 s | 86.8% | 50.931 s | 10.9% | 7.685 s | 16.415 GiB |
| LoRA6，360p/5s | 14.688 s | 8.580 s | 58.4% | 4.418 s | 30.1% | 0.777 s | 5.809 GiB |
| LoRA6，480p/5s | 26.020 s | 17.798 s | 68.4% | 5.950 s | 22.9% | 1.304 s | 6.603 GiB |
| LoRA6，480p/10s复杂台词 | 61.201 s | 43.468 s | 71.0% | 12.103 s | 19.8% | 2.233 s | 8.275 GiB |
| LoRA6，720p/5s | 69.726 s | 52.752 s | 75.7% | 13.449 s | 19.3% | 2.584 s | 8.816 GiB |
| LoRA6，720p/15s | 316.089 s | 264.265 s | 83.6% | 40.856 s | 12.9% | 7.731 s | 16.404 GiB |

480p/10秒两行使用2026-08-12统一复杂台词与大动作任务；其余是此前真实性能锚点，
提示词不同，因此本表用于瓶颈定位，不用于跨行成片质量比较。新协议的完整多shape
矩阵和Whisper结果仍待重新生成。

## 为什么分辨率和时长越大，DiT增长越快

H3把文本、视频、音频和首尾帧条件打包进同一序列。视频主Token近似为：

```text
spatial_tokens = (width / 32) * (height / 32)
latent_frames  = ((frames - 5) / 17) * 5 + 2
video_tokens   = spatial_tokens * latent_frames
```

720p/5秒约34.5k packed tokens，720p/15秒约99.6k，序列长度约2.89倍。MLP/Linear
接近线性增长，而全局Attention计算接近平方增长。真实结果中：

- 原始路线720p从5秒到15秒：Video-VAE约2.99倍，DiT约5.19倍，端到端约4.62倍；
- LoRA路线720p从5秒到15秒：Video-VAE约3.04倍，DiT约5.01倍，端到端约4.53倍。

因此15秒不是5秒成本的简单三倍，超线性部分主要来自DiT联合长序列。

## DiT内部最大子瓶颈

原始权重9实际/11预测、720p/15秒的当前精确路径也已用真实复杂任务做了
单实际步Kineto Profile。一个实际步墙钟为41.845秒；其中10万Token的
SageAttention主内核累计28.545秒，约占实际步68.2%。其后才是INT8/MLP
GEMM和逐Block权重搬运。因此原始权重路线的大幅优化同样必须优先处理长序列
Attention，不能用LoRA的较短调度来代替其性能结论。

LoRA6、720p/5秒Kineto实测：

- SageAttention d128：20.078秒，300次调用；
- CUTLASS主GEMM：15.741秒，3000次调用；
- pinned H2D累计6.000秒，但已与计算重叠，不能直接加到墙钟时间。

Attention是第一内核瓶颈，Linear/MLP GEMM是第二。长到720p/15秒后Attention占比
继续上升。端到端稀疏实验也提供了机制证据：

| LoRA6任务 | 稠密 | Sparse topk=0.50 | 提速 | DiT变化 | VAE变化 |
|---|---:|---:|---:|---:|---:|
| 480p/10s复杂台词 | 61.201 s | 57.618 s | 1.062x | 43.468→38.887 s | 12.103→12.088 s |
| 720p/15s | 316.089 s | 240.189 s | 1.316x | 264.265→188.416 s | 40.856→40.848 s |

同一Sparse技术在短序列只带来6.2%，在约10万Token长序列带来31.6%，且节省几乎
全部来自DiT。这证明当前路由应按真实Token/时间规模决定是否启用长序列优化，而
不是对所有480p/720p任务统一套一个Attention策略。Sparse仍需按新Whisper协议完成
台词、时间窗和多seed质量验收，不能仅凭速度进入正式路由。

## 冷启动瓶颈

当前独立进程并行准备Qwen、DiT、Video-VAE和Audio-VAE，关键路径约58到74秒。

- 对360p/5秒，冷启动远大于14到17秒热任务，是首次体验第一瓶颈；
- 对480p/10秒，冷启动与57到82秒生成相当；
- 对720p/15秒，冷启动只占总等待约13%到19%，DiT仍占主导。

产品必须维持常驻模型服务并复用主机Pinned权重和kernel缓存。通过每次请求重启进程
来测试会严重歪曲短视频吞吐。

## 当前不是主要瓶颈的部分

- 初始DiT/Video-VAE H2D各约0.2到0.3秒；逐Block复制由双缓冲隐藏；
- 文本编码通常0到2.3秒，复杂长提示词仍不是主要时间；
- 音频VAE解码约0.15到0.32秒；
- Mux随输出时长从约0.7秒增长至约7.7秒，但远小于DiT；
- 简单增加Resident Block或将显存从约9 GiB提高到18到22 GiB，已有消融未产生速度
  收益，说明当前不是因过度offload而等待PCIe。

## 当前优化优先级

1. 原始权重与LoRA均为正式主线；任何加速证据、质量审核和路由授权均按引擎
   分开，LoRA通过不能替代原始权重通过；
2. 720p/15秒：长序列Attention，然后是INT8 ConvRot GEMM/MLP；
3. 480p/10秒及720p/5秒：Attention与GEMM并重，随后优化Video-VAE；
4. 360p/5秒：常驻服务冷启动优先；热态下Video-VAE已经是约20%到30%的第二瓶颈；
5. 所有档位：不得为了速度改变用户质量计划；Sparse、缓存、预测等近似技术必须走
   Whisper台词时间戳、4到8帧和Human终审；
6. 下一轮使用统一复杂任务重跑360p/480p/720p和5/10/15秒矩阵，至少三个seed，
   再用相同任务的Whisper指令遵循结果决定正式路由边界。

## 主要证据

- `runtime/calibration/workload_routing_round7/`
- `runtime/calibration/workload_routing_round8/`
- `runtime/calibration/block_offload_720p/`
- `runtime/calibration/block_offload_720p15s/`
- `runtime/calibration/lora_block_720p15s/`
- `runtime/profile/current_native_lora6_720p5/`
- `runtime/profile/round12_original_exact_720p15_step1/`
- `runtime/profile/video_vae_sm89_720p15_round12/`
- `runtime/calibration/workload_routing_round12/video_transport_480p3.json`
- `docs/RTX4090_WORKLOAD_ROUTING_ROUND6.md`
- `docs/COMPLEX_DIALOGUE_ACCEPTANCE_PROTOCOL.md`
