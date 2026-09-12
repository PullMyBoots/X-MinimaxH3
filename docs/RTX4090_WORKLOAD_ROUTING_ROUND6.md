# RTX 4090 H3 工作负载路由：第六轮内核与端到端实践

日期：2026-08-11  
GPU：NVIDIA RTX 4090 24 GB，SM89  
状态：真实 720p/5秒实验；稠密 Sage 保持生产默认，稀疏路径仅为人工审核候选

## 1. 本轮回答什么

本轮不依靠历史文字推导结论，而是在独立 Native 后端上完成三层证据：

1. LoRA6、1280x736、124帧的 MLP chunk 受控扫描；
2. 同一负载的 PyTorch Kineto 完整请求 Profile；
3. SpargeAttention 先做真实 H3 形状微基准，再做两条相同 prompt/seed 的完整
   稠密/稀疏视频 A/B。

所有“更快”只描述本轮测过的候选，不代表全局理论上界。

## 2. MLP chunk 扫描

固定 Larry step600 EMA、6个完整 Turbo 步、Block双缓冲、prefetch 1、VAE tile
288、同一 prompt/seed，只改变 MLP chunk：

| chunk | 端到端 | DiT去噪 | 峰值 allocated | 结论 |
|---:|---:|---:|---:|---|
| 8192 首次 | 71.920 s | 52.505 s | 8.816 GiB | 首次形状 |
| 12288 热态 | 69.643 s | 52.578 s | 8.816 GiB | 收益仅0.18%，噪声量级 |
| 16384 热态 | 72.032 s | 55.002 s | 8.816 GiB | 明确回退 |
| 8192 回切热态 | 69.765 s | 52.744 s | 8.816 GiB | 保留 |

12288和16384输出与8192不再字节等价，虽然7帧门控未见明显异常，但没有可复现
的实质速度收益，因此生产策略继续使用8192。

## 3. 完整请求 Profile

第二次720p/5秒请求的阶段耗时为：DiT去噪52.783秒、Video-VAE解码13.558秒、
mux 2.786秒。Kineto中最大的实际CUDA内核为：

- SageAttention d128：20.078秒，300次调用；
- CUTLASS主GEMM内核：15.741秒，3000次调用；
- pinned H2D累计6.000秒；它与计算存在重叠，不能直接与阶段墙钟相加。

因此720p下Attention已经是第一内核瓶颈，但并非全部请求；即使只把Attention
加速1.3倍，端到端也不可能自动获得1.3倍。

Profile证据：`runtime/profile/current_native_lora6_720p5/`。Chrome trace约960 MB，
阶段墙钟作为端到端权威时间，Profiler内核累计用于定位瓶颈。

## 4. 稀疏注意力实践

### 4.1 隔离构建与微基准

知识库 SpargeAttention 固定提交
`ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`，只在临时目录构建，没有安装进
生产环境。其上游构建在WSL需要显式提供`/usr/lib/wsl/lib`才能链接CUDA。

真实H3长序列形状 `[1,34519,56,128]` 的中位数：

| 后端 | Attention时间 | 相对稠密 | 稀疏率 |
|---|---:|---:|---:|
| dense Sage | 73.782 ms | 1.000x | 0% |
| Sparge topk 0.75 | 64.013 ms | 1.153x | 24.72% |
| Sparge topk 0.65 | 56.872 ms | 1.297x | 34.69% |
| Sparge topk 0.60 | 53.223 ms | 1.386x | 39.67% |
| Sparge topk 0.50 | 46.222 ms | 1.596x | 49.63% |

随机张量误差只能筛选候选，不能代表视频质量。选择中间的topk=0.65进入完整视频，
其余更激进参数未被宣称可用。

第一次完整集成还暴露出真实失败：H3 text refiner序列短于128，Sparge内核拒绝，
两条请求在生成前报`seq_len should be not less than 128`。最终实现为短文本序列继续
走精确Sage，只有长packed AV DiT序列走Sparge，然后从头重跑。

### 4.2 严格端到端 A/B

固定：Larry LoRA、6完整步、1280x736、124帧、Block、prefetch 1、MLP 8192、
VAE tile 288。每一行的稠密与稀疏请求拥有相同prompt和seed：

| seed/场景 | 稠密总时 | 稀疏总时 | 端到端提速 | 稠密去噪 | 稀疏去噪 | 去噪提速 |
|---|---:|---:|---:|---:|---:|---:|
| 8833/攻城行军 | 71.770 s | 68.053 s | 1.055x | 52.563 s | 48.581 s | 1.082x |
| 8834/雨夜城墙 | 70.775 s | 66.747 s | 1.060x | 52.623 s | 48.480 s | 1.085x |

两种后端峰值均约8.817 GiB。微基准1.297x最终只变成约1.06x端到端收益，符合
Profile显示的Attention占比，而不是把孤立内核数字冒充产品速度。

## 5. 质量门控与当前决策

四条视频都重新抽取7帧检查。未见彩色色斑、持续散光重影、整帧闪烁或结构崩坏；
两组都能保持攻城行军和雨夜城墙的连贯叙事，基础视觉门控PASS。

但稀疏注意力不是数学等价变换。相同seed下诊断SSIM分别为0.753和0.783，只说明
输出已经明显分叉，不作为质量高低的硬门控。仅凭两条视频也不足以把它放入默认
“高质量”路径。因此当前决策是：

- 稠密Sage继续作为发布默认；
- Sparge topk=0.65只作为明确标记的实验候选，等待Human审片和更多场景；
- 不把约1.06x收益写成“无损提速”；
- 若后续Human接受，再补四场景、首尾帧、纵横屏和15秒边界，才能进入路由器。

## 6. 可复现证据

- MLP扫描：`runtime/calibration/workload_routing_round6/lora6_720p5_chunk_scan/`
- 稠密同seed对照：
  `runtime/calibration/workload_routing_round6/dense_lora6_720p5_seed8833/`
- 稀疏同seed候选：
  `runtime/calibration/workload_routing_round6/sparge065_lora6_720p5/`
- 多帧门控图：`runtime/calibration/workload_routing_round6/visual_gate/`
- Kineto Profile：`runtime/profile/current_native_lora6_720p5/`
- 微基准：`scripts/benchmark_spargeattention_h3_shape.py`
- 完整实验入口：`scripts/benchmark_native_hot_session.py`

