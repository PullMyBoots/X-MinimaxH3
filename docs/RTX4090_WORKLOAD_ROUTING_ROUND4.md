# RTX 4090 H3 工作负载路由：第四轮实测校准

日期：2026-08-11  
GPU：NVIDIA RTX 4090 24 GB，SM89  
状态：支持第一版路由特征设计；不等同于完整发布矩阵

后续证据：未参与本轮归纳的 4:3 精确形状已完成 Block/Resident/Block 留一验证，
见 `RTX4090_WORKLOAD_ROUTING_ROUND5.md`。该验证支持当前 LoRA6 Block 路由，
同时进一步限定：文中的“占优”只指已经比较的候选机械策略，不代表理论全局最优。

## 1. 本轮回答什么

本轮不是从历史日志外推，而是在同一份 Native 后端、同一张 RTX 4090 上重新生成
13 条带 H.264 视频和 32 kHz 双声道 AAC 的 MP4，集中检验三个问题：

1. 相同像素数和帧数的横、竖屏是否需要两个执行策略；
2. 形状切换后是否需要重新加载模型，或者污染后续任务；
3. 单一 packed-token 指标能否同时解释 DiT、显存和端到端时间。

所有请求都在持久热 Session 中执行。约 58--60 秒的权重准备是服务进程冷启动，
与每条请求耗时分开记录。

## 2. 固定条件

- LoRA 路线：Larry step600 EMA，6 个完整 Turbo 实际步，Block 双缓冲，
  prefetch depth 1，MLP chunk 8192，Video-VAE tile 288；
- 原始权重路线：20 个调度点，9 实际/11 预测，自动路由到短任务 Resident 计划；
- quant backend：锁定的 Comfy-Kitchen CUDA 实现；
- attention：锁定的 SageAttention SM89 实现；
- 同一组内提示词不变；回切复测使用相同 seed；
- 普通用户不接触内部 profile 名，实验报告保留内部 ID 以便复现。

## 3. 同像素横竖屏与回切

### 3.1 LoRA 六步

| 请求 | 端到端 | DiT | Video VAE | 峰值显存 |
|---|---:|---:|---:|---:|
| 640x352x124，首次横屏 | 17.326 s | 8.775 s | 4.674 s | 5.809 GiB |
| 352x640x124，切到竖屏 | 14.935 s | 8.566 s | 4.412 s | 5.809 GiB |
| 640x352x124，切回横屏 | 14.688 s | 8.580 s | 4.418 s | 5.809 GiB |
| 864x480x124，首次横屏 | 25.866 s | 17.667 s | 5.942 s | 6.603 GiB |
| 480x864x124，切到竖屏 | 26.412 s | 17.832 s | 5.983 s | 6.604 GiB |
| 864x480x124，切回横屏 | 26.020 s | 17.798 s | 5.950 s | 6.603 GiB |

360p 首条包含 1.896 秒首次 prompt 编码；缓存 prompt 后，横竖端到端只差
0.247 秒。480p 组不存在首次 prompt 干扰，横竖 DiT 相差 0.165 秒，峰值显存
相差约 0.001 GiB。

### 3.2 原始权重 9/11

| 请求 | 端到端 | DiT | Video VAE | 峰值显存 |
|---|---:|---:|---:|---:|
| 640x352x124，首次横屏 | 20.757 s | 11.970 s | 3.750 s | 20.726 GiB |
| 352x640x124，切到竖屏 | 18.148 s | 11.803 s | 3.578 s | 20.726 GiB |
| 640x352x124，切回横屏 | 17.896 s | 11.788 s | 3.600 s | 20.726 GiB |

原始权重路线得到相同机制结论：缓存 prompt 后，横竖 DiT 相差 0.015 秒，峰值
显存相同。首次请求的额外时间来自 prompt 编码、首次 kernel 和首次阶段搬运，不是
更换比例后重新从磁盘装载整套模型。

### 3.3 确定性和状态安全

四组“首次形状/切换后回到相同形状”的 MP4 均字节级相同：

- LoRA 360p：`ea6e021f...6386c263`；
- LoRA 480p：`98b6a0e8...5172ea05`；
- 原始权重 360p：`d0996866...20f40e4`；
- LoRA 360p15/480p8 复测见下一节。

这排除了已测路径上的跨形状随机状态污染和持久切换惩罚。

## 4. 相近主 Token、不同空间/时间组成

LoRA 六步 Block 路径实测：

| 请求 | packed token | S / Tv | 端到端 | DiT | Video VAE | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| 640x352x362，360p15 | 24,814 | 220 / 107 | 50.226 s | 33.685 s | 13.261 s | 7.625 GiB |
| 864x480x192，480p8 | 23,793 | 405 / 57 | 44.100 s | 31.859 s | 9.366 s | 7.560 GiB |

两者 packed token 相差 4.29%，DiT 相差 5.73%，峰值显存相差 0.86%。因此
packed token 是有效的 DiT 主轴，二者可以共享当前 Block 执行机制。

但端到端相差 13.89%，主要来自 3.895 秒的 Video-VAE 差异。单一标量可以用于
形成初始计算聚类，不能独自预测完整请求耗时。Router 至少需要把 DiT 和 VAE
拆成两个成本分量：

```text
DiT cost  <- packed_tokens + spatial_tokens + latent_frames + condition_tokens
VAE cost  <- output_pixel_frames + exact latent geometry + tile policy
Peak VRAM <- activation geometry + residency/offload plan + safety margin
```

360p15 与 480p8 的回切复测分别与首次输出 MP4 字节级相同：
`ead9aa49...7f9ad12`、`38f7a406...d4f20a2`。

## 5. 视觉先行门控

13 条新视频均先抽取覆盖首、中、尾的多帧拼图再采信性能数字。检查未发现：

- 彩色色斑或分块污染；
- 散光重影或持续闪烁；
- 明显结构崩坏；
- 音视频流缺失。

SSIM 没有作为硬门控。本轮相同形状相同 seed 已有更强的 MP4 字节等价证据。

## 6. 实测支持的路由结论

1. **不按横屏/竖屏拆执行策略。** 宽高交换后，模型相关工作量、DiT、显存和
   最优机械路径没有出现可分离的聚类；比例只保留为精确 shape/compile cache key。
2. **不按“360p/480p/720p”标签直接路由。** 用户档位先规范化为实际宽高和帧数，
   Planner 只看模型相关特征。
3. **packed token 是主轴，不是唯一轴。** 它可聚合 DiT 负载；端到端预测仍需
   `S`、`Tv`、pixel-frame、条件帧和 VAE tile 等修正特征。
4. **当前没有证据支持频繁卸载/重载多个整引擎。** 同路线、同权重内只需切换
   请求级 ExecutionPlan；已测比例切换没有持续惩罚。
5. **不要为了产品叙事强造多个策略。** LoRA 六步在已测 360p5 到 720p15 包络及
   后续未见 4:3 形状中，`Block+tile288` 都是已比较候选里的端到端胜者；这不是
   对 Hybrid、compile 或其他尚未测量策略的全局最优声明。原始权重目前才有短任务
   Resident、大任务 Block 两个有实测依据的执行簇。

## 7. 尚未得到实践支持的部分

- 720p 竖屏和首/尾帧条件任务尚未进入本轮横竖切换矩阵；
- LoRA 4/8 步与原始权重其他实际/预测步组合需要独立质量与性能标定；
- compile/CUDA Graph 尚未形成可靠的胜出策略，不能写进发布路由；
- 目前锚点不足以拟合一个可宣称普适的连续阈值函数；发布前仍需边界重复、OOM
  安全裕量和 leave-one-shape-out 验证。

## 8. 可复现证据

- 场景：`runtime/calibration/workload_routing_round4/*.json`
- LoRA 横竖/回切：`runtime/calibration/workload_routing_round4/lora6_aspect_switch/`
- 原始权重横竖/回切：`runtime/calibration/workload_routing_round4/original911_aspect_switch/`
- 空间/时间组成：`runtime/calibration/workload_routing_round4/lora6_spacetime/`
- 多帧门控：`runtime/calibration/workload_routing_round4/visual_gate/`
- 执行脚本：`scripts/benchmark_native_hot_session.py`
