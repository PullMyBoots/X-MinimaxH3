# INT8 在 16/24GB 显存下的主机内存下限与速度拐点

日期：2026-08-31

## 结论

- 本机现有 FL2VA INT8 权重在 16GB 与 24GB 显存启动器下，共享近似相同的主机内存曲线。
- **技术下限是约 11GiB 可供服务进程使用的内存**：10GiB 冷缓存硬上限会被 cgroup OOM 杀死；11GiB 可完成 720p×5s 成片，也可在 24GB 显存路线完成 720p×15s 成片。
- 修复 Video-VAE 峰值漏算后，16GB 显存路线也已在 11GiB 主机硬上限内完成 720p×15s，并在 16GiB 主机硬上限内完成 1080p×15s。规划器会在解码前选择字节等价的时序 Host Sink，不再等到整段 FP32 像素转换时触发 CUDA OOM。
- 11GiB 是贴着上限、持续回收文件页的极限，不是安全产品配置。按给系统保留 6GiB 计算，16GB 物理内存不能安全承诺 INT8；应继续使用 W4A8。
- 单纯放宽 pageable 文件缓存没有稳定提速；主动扩大 pinned DiT block coverage 才能把多余主机内存转换为热态速度。
- **速度拐点约为 32GiB 可供服务进程使用的内存**。16GB 显存路线从 pin 3.765GiB 的 24.20s/步降到 pin 17.693GiB 的 20.55s/步；24GB 显存路线从 25.45s/步降到 21.18s/步，约为 1.18--1.20×。
- 从 32GiB 继续增加到 36GiB 应用额度，pinned coverage 从 86.98% 增到 92.53%，DiT 没有继续变快，已进入平台区。
- 折算真实主机并保留 6GiB：24GB 物理内存可做紧凑 INT8，32GB 物理内存可接近速度上限，40GB 以上可覆盖实测速度拐点；常见标准容量中 48GB 是舒服的全速 INT8 配置。

## 测试契约

- 权重：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- 启动器：`fl2va_int8_16gb`、`fl2va_int8_24gb`
- 任务曲线：1280×736、362 帧（720p×15s），Base 5 步、加速力度 95，在第 1 步正式断点，确保执行一次完整 Actual DiT，但跳过最终解码。
- 成片下限：720p×5s 和 720p×15s，Base 5 步、加速力度 95，包含文本编码、DiT、视频/音频 VAE 与封装。
- 极限规格回归：1920×1088、362 帧（1080p×15s），Base 5 步、加速力度 95，16GB 显存启动器和 16GiB 主机 cgroup 硬上限，包含完整成片链路。
- 主机限制：cgroup v2 `memory.max`，`memory.swap.max=0`，`memory.oom.group=1`。
- 冷缓存归一化：每个 case 开始前仅对 H3 INT8、Qwen layer cache 和 VAE 权重执行 `POSIX_FADV_DONTNEED`；不删除或修改文件。
- 数值不变量：权重、输入、seed、推理计划不变；只改变精确等价的主机/GPU residency 与 pinned coverage。

## 最低可运行内存

| 显存路线 | 应用内存上限 | 任务 | 结果 | 备注 |
|---|---:|---|---|---|
| 16GB | 10GiB | 720p×15s，1 Actual | 失败 | cgroup OOM，进程 `-9` |
| 16GB | 11GiB | 720p×15s，1 Actual | 成功 | DiT 24.72s，53,131 次 `memory.max` 事件 |
| 16GB | 11GiB | 720p×5s，完整成片 | 成功 | 56.47s，DiT 27.78s，视频解码 11.12s |
| 16GB | 11GiB | 720p×15s，完整成片 | 成功 | 修复后 114.38s，DiT 60.49s，视频解码 33.40s；CUDA 峰值 12.06GiB |
| 16GB | 16GiB | 1080p×15s，完整成片 | 成功 | 修复后 276.94s，DiT 169.16s，视频解码 83.50s；CUDA 峰值 14.46GiB |
| 24GB | 11GiB | 720p×15s，完整成片 | 成功 | 114.32s，DiT 61.21s，视频解码 33.40s |

11GiB 点在整个任务中持续顶住硬上限并进行干净文件页回收，因此这里只能证明“能跑”，不能证明带参考图、参考音频和其他桌面负载时仍安全。

## Pageable 基线

不 pin transformer 时，32GiB 上限的自然冷峰值约为 29.6GiB；40/48GiB 仍为约 29.6GiB。单 Actual DiT 在约 25--27s 内波动，没有随 pageable 容量出现稳定加速：

| 显存路线 | 应用额度 | 主机峰值 | DiT |
|---|---:|---:|---:|
| 16GB | 32GiB | 29.604GiB | 25.18s |
| 16GB | 40GiB | 29.646GiB | 24.93s |
| 16GB | 48GiB | 29.644GiB | 25.63s |
| 24GB | 16GiB | 16.000GiB | 25.78s |
| 24GB | 24GiB | 24.000GiB | 26.96s |
| 24GB | 32GiB | 29.637GiB | 26.13s |
| 24GB | 40GiB | 29.644GiB | 26.45s |
| 24GB | 48GiB | 29.643GiB | 25.78s |

这说明 Linux 文件页缓存已能隐藏大部分顺序读取；仅提高 `memory.max` 不足以稳定提速。

## Pinned coverage 曲线

| 显存路线 | 应用额度 | pinned GiB | pinned coverage | DiT | 冷启动 |
|---|---:|---:|---:|---:|---:|
| 16GB | 16GiB | 3.765 | 18.51% | 24.20s | 23.76s |
| 16GB | 24GiB | 11.670 | 57.37% | 23.62s | 32.45s |
| 16GB | 28GiB | 15.811 | 77.72% | 21.76s | 39.51s |
| 16GB | 32GiB | 17.693 | 86.98% | 20.55s | 42.77s |
| 16GB | 36GiB | 18.823 | 92.53% | 20.77s | 45.99s |
| 24GB | 16GiB | 3.765 | 18.51% | 25.45s | 24.36s |
| 24GB | 24GiB | 11.670 | 57.37% | 23.11s | 33.70s |
| 24GB | 32GiB | 17.693 | 86.98% | 21.18s | 42.43s |
| 24GB | 36GiB | 18.823 | 92.53% | 20.94s | 43.68s |

冷启动随 pin coverage 增长，但它是服务生命周期内的一次性成本。对于连续生成任务，热态 DiT 收益会被反复复用。

## GPU residency 对照

- 16GB 显存常驻 7 个 INT8 block：GPU 峰值从约 12.06GiB 增到 13.77GiB，但 DiT 25.67s → 25.74s，没有收益。
- 24GB 显存常驻 28 个 block 的单次结果为 23.81s，但与对应冷基线的波动区间重叠，证据弱于 pinned host 曲线。
- 当前不应根据显存空余机械地扩大 resident prefix；优先依据主机可用内存连续调整 pinned coverage。

## 产品解释与边界

1. **16GB 物理内存**：给系统保留 6GiB 后服务只剩约 10GiB，低于实测 11GiB 技术下限。INT8 不应作为安全默认，W4A8 更合适。
2. **24GB 物理内存**：服务约可使用 18GiB；INT8 可流式运行并使用少量 pinned cache，属于紧凑可用。
3. **32GB 物理内存**：服务约可使用 26GiB；可获得大部分 pinned 加速，但未达到 32GiB 应用速度拐点。
4. **40/48GB 物理内存**：能覆盖约 32GiB 应用额度，进入 INT8 热态速度平台区。
5. **16GB 显存的长视频 VAE 路由已修复**：旧规划只计算“完整解码张量”，漏掉随后像素归一化所需的第二份完整 FP32 工作张量，并低估约 2GiB 服务常驻工作集。修复后 720p×15s 在解码前进入 `host_temporal_exact`，不改变 VAE 权重、像素公式或最终 uint8 字节；24GB 同任务仍保留整段路径。

## 后端设计建议

- 不向用户暴露离散“低配/高配”内存模式。
- 启动时检测 `MemAvailable`、cgroup limit 和系统保留量，计算连续预算：
  - `<11GiB`：拒绝 INT8；
  - `>=11GiB`：低内存顺序流式；
  - 其余预算按可用余量连续增加完整 block-group pinned coverage；
  - 在约 17.7GiB pinned / 32GiB 应用额度处停止，避免为无收益内存继续 pin。
- pin 只接受完整 block group，不能按 tensor 任意截断；保持现有数值等价性和确定性。
- 保留冷启动构建峰值保护。当前 packing 在完成前仍持有源 mmap tensor 视图，pin 预算不能简单等于 `可用内存-6GiB`。

## 证据位置

- `runtime/validation/int8_resource_knee_20260831/lower_actual/`
- `runtime/validation/int8_resource_knee_20260831/m11_probe/`
- `runtime/validation/int8_resource_knee_20260831/base_curve/`
- `runtime/validation/int8_resource_knee_20260831/pin_curve_v2/`
- `runtime/validation/int8_resource_knee_20260831/pin_saturation/`
- `runtime/validation/int8_resource_knee_20260831/full_720p5_m11/`
- `runtime/validation/int8_resource_knee_20260831/full_720p15_m11/`
- `runtime/validation/int8_resource_knee_20260831/fix3_16gb_720p15/`
- `runtime/validation/int8_resource_knee_20260831/fix3_16gb_1080p15/`

主机内存曲线实验通过 `H3_NATIVE_RESEARCH_INT8_HOST_CURVE=1` 研究门控完成；由失败证据定位出的 Video-VAE 峰值修复已进入默认规划器。
