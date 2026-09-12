# W4A8 主机内存能否继续换取速度：8GB 路线分项 Profile

## 结论

额外主机内存可以换取少量速度，但当前 8GB W4A8 执行图已经接近这条路径的收益上限。
它能做的是把反复读取的 CPU Block 权重变成 pinned memory，并启用双缓冲，让下一层权重的
H→D 复制与当前层计算重叠；它不能让这些权重永久驻留 8GB 显存，也不能加速 Attention、
MLP、量化或其他 GPU Kernel。

无 profiler 的重复中位数为：

| 8GB 执行点 | 进程主机峰值 | DiT 去噪 | 相对紧凑路径 |
|---|---:|---:|---:|
| 0 pinned，单缓冲 | 10.61GiB | 77.62s | 1.000× |
| 10.97GiB Block pinned，双缓冲 | 22.02GiB | 75.35s | 1.030× |

多使用约 11.4GiB 主机内存，只减少约 2.27 秒去噪时间（2.92%）。完整端到端任务还包含
Qwen、VAE 和封装，因此最终百分比会更低。

## 内部分项证据

在相同 1280×736、362 帧、6步、加速力度95、5/6正式断点上，使用框架内 Kineto
记录 CUDA 活动。Profiler 会显著扰动绝对墙钟时间，因此本节只解释设备分项；速度结论仍
以上面的无探针中位数为准。

| CUDA分项 | 0 pinned | Block全pinned | 变化 |
|---|---:|---:|---:|
| Pageable→Device | 3.233s | 0.182s | 大部分重复权重不再来自pageable页 |
| Pinned→Device | 1.620s | 3.367s | 重复权重改走pinned异步复制 |
| H→D累计 | 4.853s | 3.549s | 设备复制累计减少约1.304s |
| `h3_long_video_attention` | 20.972s | 20.826s | 基本不变 |
| `h3_long_mlp` | 18.788s | 18.795s | 基本不变 |

H→D累计约占无探针紧凑去噪时间的6.25%，而这还包含条件、小张量和一次性传输，不能全部
由更多主机内存消除。实际完整Block缓存通过更快复制和copy/compute overlap回收2.27秒，
与这个很低的物理上限一致。继续增加主机内存没有新的高价值重复权重可缓存。

## 8/16/24GB 三条路线是否为了容量牺牲速度

如果“三条路线”指8、16、24GB显存档，答案不是完全相同：

1. **8GB明确牺牲了速度来换容量。** 50个DiT Block都需要流式搬运；16GB主机的单缓冲
   路径相对24GB显存的49常驻路径约慢9.3%。主机内存放开后只能将差距缩到约6.1%。
2. **16GB只有轻度妥协。** 安全的30 Block常驻、无pinned路径为72.25秒；利用充足RAM的
   24常驻＋5.71GiB pinned路径为70.88秒，已与24GB路线的70.99秒处于同一噪声平台。
3. **24GB W4A8在权重驻留方面基本没有妥协。** 49/50 Block常驻后，再增加主机缓存只带来
   约0.1%，低于工程噪声。这里剩余瓶颈是GPU计算而非主机容量。

三条对照保持同一W4A8权重、V24调度、Attention动作和compact full-context K/V；这里讨论
的是数值等价的驻留/搬运实现，没有通过减少用户步数或进一步降低Attention精度来省内存。

## 哪些内存节省项不能由更多主机RAM解除

- **GPU Block常驻数量**由显存决定。主机RAM再多也不能成为4090显存。
- **compact K/V、Projection/MLP chunk**的工作空间位于GPU，若要放宽需要更多显存，而不是
  更多主机内存。
- **Attention、MLP、W4解量化、KV量化/统计**属于GPU计算。主机缓存不会提高其吞吐。
- **扩展到第三个Block buffer**同样消耗显存；8GB路线已只够一到两个buffer。

因此产品上可以自动利用多余RAM，但不应把它宣传成明显的“高速档”。对单任务延迟而言，
更大的提速空间在GPU算子、调度和近似计算，不在继续扩大CPU权重缓存。

## 证据文件

- 无探针曲线：`runtime/validation/w4a8_resource_knee_20260830/`
- Kineto端点：`runtime/validation/w4a8_resource_knee_20260830/torch_transfer_profile/`
- 复现脚本：`scripts/benchmark_w4a8_resource_knee.py --torch-profile`

Nsight Systems在当前WSL/CUDA 13.3组合下会使生成任务失败，因此未将其不完整trace作为模型
证据；最终分项来自能够完整到达`checkpointed`状态的框架内Kineto采集。
