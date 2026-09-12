# RTX 4090 H3 工作负载路由：第五轮留一形状验证

日期：2026-08-11  
GPU：NVIDIA RTX 4090 24 GB，SM89  
状态：真实留一形状验证；只支持本报告所比较的两个机械策略

## 1. 为什么补做这一轮

前四轮已经在真实 Native 后端上测量多个 16:9、9:16 和空间/时间组合，但这些
形状也参与了第一版路由规则的归纳。只用训练点解释训练点，不能证明规则对未见
形状有效。

本轮选择此前路由校准未使用的精确画布 `640x480`（4:3）、124 帧，在相同
prompt、seed、Larry step600 EMA、6 个完整 Turbo 步、MLP chunk 8192 和
Video-VAE tile 288 下，只改变 DiT 权重驻留策略：

1. Block 双缓冲首次运行；
2. Resident 全驻留；
3. 回切 Block 双缓冲。

这是受控的机械策略 A/B，不改变采样步、模型权重或生成语义。

## 2. 锁定运行环境

- PyTorch 2.8.0+cu126；
- CUDA capability 8.9；
- Comfy-Kitchen CUDA 扩展 SHA-256：
  `aaefcd38ba30379e5707b22bdc7e3209188e75c78ab4dd4a259f9e1d83eafa9b`；
- SageAttention SM89 扩展 SHA-256：
  `c44f6878acd51920192d0d9fdbbeebeaade1e4c8eda21ac372f7d0332d99ef5f`；
- 模型准备关键路径 65.213 秒，只计为服务冷启动，不混入请求耗时。

曾误用缺少 SageAttention 的通用 Python 环境；运行在模型加载前被
`SM89RuntimeError` 拒绝，没有产生候选数据。下面只采用锁定环境的成功运行。

## 3. 实测结果

| 顺序 | 机械策略 | 端到端 | 文本编码 | DiT 去噪 | DiT H2D | 峰值 CUDA allocated |
|---|---|---:|---:|---:|---:|---:|
| 1 | Block 首次形状 | 21.020 s | 1.756 s | 12.174 s | 0.233 s | 6.152 GiB |
| 2 | Resident | 19.430 s | 0.000 s | 11.892 s | 1.210 s | 21.811 GiB |
| 3 | Block 回切热态 | 18.470 s | 0.000 s | 11.995 s | 0.243 s | 6.152 GiB |

不能用第一行直接断言 Resident 更快，因为第一行独占首次 prompt/首次形状开销。
在提示词已缓存的可比请求中，Block 为 18.470 秒，Resident 为 19.430 秒：

- Block 端到端快 `1.052x`（节省 0.960 秒）；
- Block 峰值少 15.659 GiB，Resident 峰值约为 Block 的 3.545 倍；
- Resident 纯去噪只快 0.103 秒，但整 DiT 搬入多花约 0.967 秒，因此当前请求
  生命周期下端到端仍由 Block 胜出。

该结论针对当前“每个请求结束后为 Video-VAE 释放 DiT”的单卡生命周期；不能
外推为任意常驻服务实现的理论结论。

## 4. 输出等价性与视觉门控

三个 MP4 的 SHA-256 完全相同：

`d3348ed89fa4487233da2385729a219032d230ab9c21bbe355417013c2eccc83`

媒体结构均为 640x480 H.264、32 kHz 双声道 AAC、5.167 秒。重新抽取第
0/20/40/60/80/100/123 帧检查，未见彩色色斑、持续重影、闪烁或结构崩坏，视觉
门控为 PASS。这里使用 MP4 字节等价作为更强的数值证据，SSIM 不参与硬门控。

## 5. 对路由设计的影响

这一留一形状支持：

1. 4:3 不需要仅因比例不同而建立一套新模型或新数学路径；
2. 当前 LoRA6 候选集合中，Block 仍是这个未见形状的最低端到端时间且显存裕量
   最大的策略；
3. cold、shape-cold 和 hot 必须分别报告，不能混在一张速度表中；
4. “当前候选策略中胜出”不等于“理论极致”。Hybrid 驻留比例、更多 VAE tile、
   编译策略和调度重叠没有在此形状完成全搜索，仍然不能宣称全局最优。

## 6. 可复现证据

- 场景契约：`runtime/calibration/workload_routing_round5/unseen_4x3_lora6.json`
- 原始计时：
  `runtime/calibration/workload_routing_round5/unseen_4x3_lora6/round5_unseen_4x3_lora6_hot_session.json`
- 三个实际 MP4：同上目录；
- 7 帧门控图：
  `runtime/calibration/workload_routing_round5/visual_gate/round5_unseen_4x3_lora6_contact.jpg`
- 执行入口：`scripts/benchmark_native_hot_session.py`

