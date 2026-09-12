# RTX 4090 单卡 H3 工作负载自动路由技术设计

状态：设计与六轮实测；已有保守可执行表，仍等待完整发布矩阵标定  
适用范围：`release/serve` 独立 Native H3 服务，单张 RTX 4090 24 GB，batch size 1  
目标引擎：原始权重高保真路线、Turbo LoRA 极速路线

最新的留一形状验证与720p内核/端到端实测见
`RTX4090_WORKLOAD_ROUTING_ROUND5.md`和`RTX4090_WORKLOAD_ROUTING_ROUND6.md`。
本文中未被六轮实验报告覆盖的内容仍是设计
提案，不视为已经完成的实验事实。

## 1. 结论

服务应提供两种请求方式，但只维护一套内部执行系统：

1. 创作者模式只暴露引擎、分辨率档位、画面比例、时长、质量预设、提示词、seed 和可选首尾帧。
2. 高级模式允许精确指定宽高、合法帧数、采样步数、实际计算步位置、预测步和受支持的采样器参数。
3. 两种模式先被标准化为相同的 `GenerationIntent`，再由隐藏的 RTX 4090 Planner 选择执行策略。
4. Planner 不得修改用户已经选定的模型行为。它只能改变数学上等价、或已经独立通过质量验收的执行参数，例如驻留方式、Block Offload、MLP chunk、VAE tile、预取和编译缓存。
5. 路由档位不能以“480p版”“720p版”人工命名。档位必须由真实任务特征、4090全策略对照、速度交叉点和 OOM 边界共同产生。

目标数据流如下：

```text
CreatorRequest / AdvancedRequest
              |
              v
       RequestNormalizer
              |
              v
       GenerationIntent       用户要求的模型行为，不可被路由器降级
              |
              v
       WorkloadAnalyzer       计算H3真实时空负载
              |
              v
       RTX4090Planner         从已标定策略中选择最快可运行项
              |
              v
       ExecutionPlan          仅含内部运行机制
              |
              v
       NativeH3Pipeline
```

## 2. 设计原则

### 2.1 路由是内部机制

普通用户不需要知道 `resident`、`block_offload`、`mlp_chunk=4096` 等实现细节。API 对外只承诺请求内容、最终解析的画布/帧数和质量语义。

内部日志必须记录路由证据，但普通任务响应不暴露内部策略名称。管理员诊断接口可以返回经过权限控制的 Planner 信息。

### 2.2 质量计划与执行计划正交

必须把两类参数拆开：

```text
QualityPlan
  engine
  scheduler / sampler
  total_steps
  actual_step_indices
  forecast policy
  LoRA strength

ExecutionPlan
  transformer residency
  block buffers / prefetch
  attention implementation
  MLP chunk tokens
  VAE tile geometry
  compile / CUDA Graph bucket
  reusable buffer and cache policy
```

当用户选择“原始权重、均衡、9实际/11预测”后，Planner 不得因为任务较大而偷偷改成 8实际/12预测。若显存不足，只能切换到更节省显存的执行策略；仍无法运行时必须明确拒绝任务。

### 2.3 档位是性能等价类

“同一档位”的定义不是分辨率相同，而是：一组任务在相同质量计划下，经过全部候选策略对照后，拥有相同的最快可行执行策略。

因此档位标签来自实验结果：

```text
任务特征 -> 全候选策略实测 -> 最快可行策略标签 -> 聚合为连续区间
```

只有满足以下条件，多个任务才允许归入同一个档位：

- 最快策略相同；
- 相对第二名有稳定收益，或至少不存在显著回退；
- 峰值显存有统一安全余量；
- 冷/热态和相邻形状下结论稳定；
- 同一模型行为下通过数值对照和强制多帧视觉门控。

## 3. 当前实现审阅

### 3.1 已有基础

当前代码已经具备以下基础：

- `contract.py` 提供 360p、480p、720p，五种比例和四个质量预设；创作者参数与具体采样计划已有初步分离。
- `GenerationInput` 已要求宽高为 32 的倍数，帧数满足 `17*n+5`，符合 H3 画布和时间网格约束。
- `NativeH3Pipeline` 已按文本编码、首尾帧编码、latent准备、DiT去噪、视频/音频VAE和mux分阶段执行。
- `ResidencyManager` 已采用 eviction-first，避免 Qwen、DiT、VAE 在24 GB显存中无意重叠。
- `DoubleBufferBlockExecutor` 已有正确的两个设备Buffer、copy/compute stream和逐slot event抽象，并避免 LightX2V 最后一层后无用地再次预取Block 0。
- `NativeT2AVHotSession` 已实现进程内主机权重复用和提示词一项缓存。
- 当前 SM89 policy 已锁定经过审计的 Comfy-Kitchen CUDA INT8/ConvRot 与 SageAttention Ada kernel。

### 3.2 关键缺口

当前实现还不是动态路由系统：

1. `GenerationSpec` 只有创作者模式，没有独立的高级请求契约。
2. `RuntimeConfig` 在服务启动时固定一个 `offload_mode`，无法按请求选择。
3. `fused_mlp.py` 通过进程环境变量读取 chunk，不能安全地逐请求改变，也无法在Profile中完整追溯。
4. `DoubleBufferBlockExecutor` 目前只在selftest中接入FakeBuffer，生产 DiT Block Stack尚未绑定真实BlockBuffer。
5. packed layout和RoPE设备表目前是请求内复用，没有跨请求的有界形状缓存。
6. `_ROW_MAP_CACHE` 是无界全局字典，需要变为带显存预算的LRU。
7. `BackendManager.desired_key()` 仍会因兼容Comfy后端的引擎/预设变化重启子进程；Native发布路径不能继承这种生命周期。
8. 当前Creator UI的时长是1到15秒、步进0.5的数值输入，不是拟定的3/5/8/10/12/15秒常用档位。
9. 现有热态Profile记录了阶段时间和逐步时间，但缺少完整的 `max_memory_reserved`、NVML峰值、H2D重叠率、cache hit和策略ID。

### 3.3 当前性能证据

当前 864x480、124帧、RTX 4090热Session的第二次请求结果为：

| 路线 | 质量计划 | 完整时间 | 去噪 | 视频解码 | 启动准备 |
|---|---|---:|---:|---:|---:|
| 原始权重 | 9实际/11预测 | 38.488 s | 25.136 s | 9.020 s | 65.298 s |
| Larry LoRA | 6完整蒸馏步 | 30.766 s | 17.526 s | 9.025 s | 59.993 s |

这证明热Session机制有效，也表明两个路线在相同画布下共享约9秒的视频VAE瓶颈。它不能用于推导720p/长视频路由边界。

### 3.4 第一轮新实践证据

2026-08-11已完成第一轮真实跨负载/跨策略实验，完整报告见`RTX4090_WORKLOAD_ROUTING_ROUND1.md`。固定原始权重9实际/11预测，只改变MLP chunk，并对每个形状连续运行两次。

| Shape-hot任务 | 主Token | chunk=8192 | chunk=4096 | 峰值显存 |
|---|---:|---:|---:|---:|
| 360p×5秒 | 8,554 | 17.947 s | 18.078 s | 两者20.729 GiB |
| 480p×3秒 | 9,154 | 20.401 s | 20.507 s | 两者20.801 GiB |
| 360p×15秒 | 24,746 | 94.967 s | 130.523 s | 两者22.661 GiB |

相同seed的4096/8192输出通过8帧视觉门控，并且最终MP4的SHA-256完全相同。当前证据说明：

- 小负载下两个chunk差距不到1%；
- 360p×15秒下8192端到端快1.374x、去噪快1.443x；
- 4096没有产生峰值显存收益，因此当前测试点上是被支配策略；
- 480p×8秒与360p×15秒主Token接近，但前者在8192路径中超过8分钟未完成并被中止，说明单一总Token不能作为完整路由指标。

这轮实践仍不足以产生发布档位：还缺真实Block Offload、VAE tile、异常形状根因Profile和LoRA独立矩阵。

## 4. H3工作负载指标

### 4.1 主指标：Packed Sequence Length

令：

- `W, H`：对齐后的输出宽高；
- `F`：满足 `17*n+5` 的输出帧数；
- `K`：首尾帧条件数量，T2AV为0，首帧或尾帧为1，首尾帧为2；
- `L`：文本编码完成后的实际文本行数；
- `fps=24`；
- H3 Video VAE空间缩放为16，DiT空间Patch为2x2。

空间Patch数：

```text
S = (H / 32) * (W / 32)
```

视频latent时间长度：

```text
Tv = ((F - 5) / 17) * 5 + 2
```

音频行数：

```text
A = 2 * round((F / fps) * 40)
```

最终packed序列长度：

```text
N = L + (Tv + K) * S + A
```

其中：

- `Tv*S` 是目标视频Token；
- `K*S` 是首尾帧条件Token；
- `A` 是双声道音频Token；
- `L` 是文本Token。

`N`是最重要的主负载指标，因为H3 DiT把文本、条件图、音频和视频合并进同一序列。

### 4.2 为什么不能只使用一个标量

仅按`N`排序适合寻找初始区间，但最终路由还必须保留以下修正特征：

```text
WorkloadFeatures = {
  packed_tokens: N,
  spatial_tokens_per_frame: S,
  latent_frames: Tv,
  condition_count: K,
  output_pixels_frames: W * H * F,
  engine,
  actual_model_evaluations,
  forecast_evaluations,
  available_device_bytes,
  shape_cache_hit
}
```

原因：

- fused Attention的存储量接近随`N`增长，但计算量更接近随`N^2`增长；
- MLP工作量近似随`N`增长，峰值激活受`min(N, chunk_tokens)`控制；
- 视频VAE更直接受`W*H*F`和tile方式影响；
- 相同`N`的宽视频、长视频可能有不同VAE和kernel行为；
- 横屏与竖屏总Token可相同，但精确张量形状和编译缓存不同；
- 首尾帧会增加条件Token和一次VAE编码阶段。

对外可以把它压缩成一个不可见的 `workload_index`，但内部Planner不能丢弃这些特征。

### 4.3 当前Creator档位的例子

以下只统计主视频与音频Token，不含可变文本`L`和首尾帧`K*S`：

| 16:9任务 | 实际画布 | 合法帧数 | 主负载Token |
|---|---:|---:|---:|
| 360p，3秒 | 640x352 | 73 | 5,084 |
| 360p，15秒 | 640x352 | 362 | 24,746 |
| 480p，5秒 | 864x480 | 124 | 15,399 |
| 480p，8秒 | 864x480 | 192 | 23,725 |
| 480p，15秒 | 864x480 | 362 | 44,541 |
| 720p，3秒 | 1280x736 | 73 | 20,484 |
| 720p，5秒 | 1280x736 | 124 | 34,454 |
| 720p，15秒 | 1280x736 | 362 | 99,646 |

因此：

- 360p×15秒比720p×3秒更重；
- 480p×8秒与360p×15秒非常接近；
- 720p×15秒约为480p×5秒主负载的6.47倍。

这直接否定了单纯按360p/480p/720p路由的方案。

## 5. 候选执行策略空间

策略必须是同一数学图的配置，不应继续维护V7/V8式分叉代码。首轮标定至少比较以下策略族。

### 5.1 DiT驻留策略

1. `phase_resident`：整个DiT在去噪阶段驻留显存，完成后再驱逐，为VAE让出空间。适合较小负载。
2. `model_offload`：阶段级整模型H2D/D2H，保留主机Pinned权重。当前HotSession接近此模式。
3. `block_double_buffer`：50个Block源权重留在主机，只保留两个设备Block Buffer，并使Block n计算与Block n+1拷贝重叠。适合全模型权重和激活无法共存的负载。

### 5.2 激活和算子策略

- `mlp_chunk_tokens`：候选如2048/4096/8192/全序列，具体集合由预Profile缩小；
- Attention backend保持在已验收的SM89 Sage路径，除非替代路径独立通过质量门；
- attention split/tile参数按`N`和可用显存标定；
- modulation row map、RoPE和packed layout使用有界LRU；
-禁止把环境变量作为逐请求策略接口。

### 5.3 VAE策略

- 整体解码；
- 时间分块；
- 空间tile；
- 时空联合tile；
- 解码后逐段搬到Host，避免完整FP32视频长期占用GPU。

VAE策略应主要根据`W*H*F`、空间宽高和实测峰值选择，而不是只复用DiT的`N`阈值。

### 5.4 编译和缓存策略

- Creator常用形状提前warmup；
- 精确形状映射到有限的compile bucket；
- 罕见高级尺寸允许shape-cold首跑；
- 编译产物尽可能持久化；
- CUDA Graph只用于形状和控制流稳定、且实测有收益的桶；
-缓存必须有版本键：模型哈希、kernel哈希、CUDA/PyTorch版本、engine、精确形状和策略修订号。

## 6. 路由档位如何由实验产生

### 6.1 不能直接枚举所有组合

Creator空间包含3个分辨率、5个比例、6个常用时长、4种条件模式、2个引擎和4个质量档。全组合为2880项；再乘候选策略和重复次数，直接穷举成本过高。

应采用分层实验：

1. **策略筛选**：在小、中、大、极大负载锚点上淘汰明显劣势策略。
2. **交叉点搜索**：沿`N`、`S`、`Tv`逐步增加负载，寻找最快策略发生变化的位置。
3. **边界复验**：在交叉点两侧和不同宽高比上重复验证。
4. **公开矩阵验证**：只使用胜出的少量策略覆盖Creator正式选项。
5. **高级模式外推验证**：随机和对抗性精确形状测试保守回退。

### 6.2 每个样本的测试状态

每个任务必须区分：

1. `process_cold`：进程、权重和kernel全部冷；
2. `model_hot_shape_cold`：模型主机缓存已就绪，但形状首次出现；
3. `model_hot_shape_hot`：模型和形状缓存都已命中；
4. `transition_hot`：从另一负载桶或另一引擎切换而来。

产品主要以`model_hot_shape_hot`衡量持续服务速度，但必须单独报告其他三类开销。

### 6.3 记录指标

每次运行至少记录：

- 端到端、文本、条件编码、每个实际步、每个预测步、视频/音频VAE、mux时间；
- CUDA event计算时间和H2D时间；
- copy/compute重叠率；
- PyTorch allocated/reserved峰值和NVML峰值；
- 当前空闲显存和安全余量；
- GPU利用率、功率和PCIe吞吐；
- compile/autotune/capture时间及cache hit；
- OOM和稳定错误码；
- `planner_revision`、完整特征和策略ID。

### 6.4 最优策略定义

对工作负载`x`和候选策略`p`：

```text
p*(x) = argmin latency_hot_p50(p, x)
```

约束条件：

```text
视觉门控通过
模型质量计划完全不变
OOM率为0
峰值显存 <= 当次可用显存 - 安全余量
重复运行无资源泄漏
```

服务混合负载时，实际决策加入切换成本：

```text
route_cost = predicted_generation_time
           + shape_cache_miss_cost
           + engine_switch_cost
```

交互服务不能为了少量吞吐无限延迟某个用户。引擎亲和调度必须同时受最大等待时间和公平队列约束。

### 6.5 聚合性检验

工作负载指标是否设计合理，不能只看相关系数，必须检验它能否形成稳定策略聚类：

- 同一叶节点内，至少大多数样本选择同一最优策略；
- 策略后悔值 `regret = chosen_latency / best_latency - 1` 足够低；
- 边界附近使用保守策略时，性能损失可接受；
- leave-one-shape-out验证中，未参与拟合的新比例/时长仍能正确路由；
- 可用显存波动后仍不发生OOM；
- 路由规则保持单调或能够解释，避免相邻尺寸无意义跳变。

首版建议使用受约束的小型决策树或人工可读的分段规则，而不是黑盒神经网络。阈值由测量数据拟合，规则本身保持可审计。

## 7. 请求切换和热态生命周期

### 7.1 480p切换到720p

在Native最终架构中，同一引擎的480p到720p切换不应重启进程，也不应重新读取全部模型文件。可能产生的额外成本包括：

- 新工作区分配；
- 新packed layout和RoPE表；
- 新shape的kernel autotune或compile；
- 新CUDA Graph捕获；
- 更大VAE tile buffer。

第二次遇到同一shape bucket时，应命中缓存。

### 7.2 原始权重与LoRA切换

24 GB显存不能同时保留两套完整DiT设备权重。可采用：

1. 主机内存充足时，原始和LoRA变体都保留Pinned Host Master，GPU只保留当前任务所需权重；
2. 主机内存不足时，保留一份Base和LoRA delta，切换时装配受影响权重；
3. 队列采用有限的engine affinity，在不破坏公平性的前提下减少来回切换。

两条路线共享Qwen、Video VAE、Audio VAE和媒体Pipeline，不能启动两套完整服务进程。

### 7.3 缓存层次

```text
L0 进程级：kernel模块、模型元数据、文件映射
L1 主机级：Pinned Qwen/DiT/VAE权重、可选两引擎Host Master
L2 引擎级：当前DiT设备Buffer、LoRA装配状态
L3 形状级：packed layout、RoPE、row map、compile/CUDA Graph
L4 请求级：latents、condition rows、decoded chunks
```

L3必须是有界LRU；L4任务完成立即释放。Planner在选择策略时必须知道L2/L3是否命中。

## 8. Planner运行算法

推荐两阶段规划：

```python
intent = normalize(public_request)
quality_plan = resolve_quality(intent)

# 文本编码前可用于排队、预估和明显超限拒绝
provisional = analyze_with_text_upper_bound(intent, quality_plan)

text_conditioning = encode_text(intent.prompt)

# 使用真实文本长度和真实可用显存最终决策
features = analyze_exact(
    intent,
    quality_plan,
    text_length=text_conditioning.length,
    free_device_bytes=current_free_vram(),
    cache_state=cache_registry.snapshot(),
)
execution_plan = router.select(features)

run_native_pipeline(intent, quality_plan, execution_plan)
```

选择步骤：

1. 枚举支持该引擎和任务条件的策略；
2. 使用显存模型淘汰预计不安全项；
3. 使用标定表预测热态时间；
4. 加入当前cache miss和引擎切换成本；
5. 选择总成本最低项；
6. 生成不可变`ExecutionPlan`，整个任务期间不得再次变化；
7. 运行完成后记录预测误差，供离线重新标定。

若运行前发现可用显存低于标定环境，直接选择下一个更保守策略。若仍发生CUDA OOM，必须清理设备状态，并最多使用同一质量计划重跑一次更低显存策略；禁止自动降低实际步数或分辨率。

## 9. 数据契约

### 9.1 创作者请求

```json
{
  "mode": "creator",
  "engine": "original",
  "quality": "balanced",
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "duration_seconds": 5,
  "prompt": "...",
  "seed": 4090
}
```

### 9.2 高级请求

```json
{
  "mode": "advanced",
  "engine": "original",
  "width": 1056,
  "height": 608,
  "num_frames": 192,
  "num_steps": 20,
  "actual_step_indices": [0, 1, 2, 3, 4, 8, 12, 16, 19],
  "sampler": "res_multistep",
  "scheduler": "simple",
  "prompt": "...",
  "seed": 4090
}
```

高级模式仍需验证：

- 宽高为32的倍数；
- 帧数满足`17*n+5`；
- 不超过发布版经过验证的像素、帧数和Token上限；
- 原始权重和LoRA只能选择各自兼容的sampler与步数契约；
- batch size固定为1。

### 9.3 内部执行计划

```json
{
  "schema_version": 1,
  "planner_revision": "rtx4090-sm89-r1",
  "profile_id": "internal-profile-03",
  "workload": {
    "packed_tokens": 34454,
    "spatial_tokens": 920,
    "latent_frames": 37,
    "condition_count": 0
  },
  "runtime": {
    "offload_mode": "block",
    "block_buffers": 2,
    "mlp_chunk_tokens": 4096,
    "vae_tile": [null, null],
    "shape_bucket": "..."
  }
}
```

上例数值只展示schema，不代表已经标定完成的720p策略。

## 10. 代码落地方案

建议新增：

```text
h3serve/native_engine/planner/
  contracts.py       WorkloadFeatures、ExecutionPlan、RouteDecision
  analyzer.py        H3精确Token、VAE工作量、显存特征
  profiles.py        读取并校验发布策略表
  router.py          可行性过滤、时间预测、切换成本、回退
  cache_registry.py  有界shape/kernel/buffer缓存状态
  telemetry.py       预测与实测证据

runtime_profiles/
  rtx4090_sm89.json  标定生成的只读发布路由表

scripts/
  calibrate_rtx4090.py
  validate_router.py
```

现有文件需要调整：

- `contract.py`：拆出Creator/Advanced请求并标准化为`GenerationIntent`；
- `engine.py`：解析`QualityPlan`，调用Planner，向Pipeline传递不可变`ExecutionPlan`；
- `pipeline/contracts.py`：显式携带quality和execution计划；
- `pipeline/stages.py`：在文本编码后、DiT装载前加入精确layout/规划阶段；
- `runtime/config.py`：保留硬件不变量，把逐请求策略移到`ExecutionPlan`；
- `model/fused_mlp.py`：移除逐请求环境变量，显式接收chunk配置；
- `model/dit.py`：生产接入真实`DoubleBufferBlockExecutor`；
- `hot_session.py`：由单固定Session升级为共享组件和策略缓存的EngineSession；
- `static/index.html`：Creator时长改为常用选项，并增加独立高级面板；
- `app.py`：普通响应不暴露profile ID，管理员诊断响应可以查看。

## 11. 发布门槛

首个自动路由版本必须同时满足：

1. Creator与Advanced请求能产生相同的规范化任务；
2. Planner绝不改变质量计划；
3. 每个正式档位都有全策略对照证据；
4. 档位边界两侧至少三次热态重复，报告中位数和尾延迟；
5. 480p/720p、5秒/15秒、T2AV/首帧/尾帧/首尾帧覆盖；
6. 每个发布策略先过多帧视觉硬门，再记录数值指标；
7. SSIM只作漂移诊断，不作为质量硬门；
8. 从480p切换720p不得重启Native进程；
9. shape-cold成本和后续shape-hot成本分别报告；
10. 显存波动和未知高级形状能够安全回退，不静默降质；
11. 路由表包含环境、权重和kernel哈希，可复现；
12. 所有内部profile名称不进入普通用户界面和公开Job契约。

## 12. 分阶段实施

### 阶段A：先完成可测的Planner骨架

- 实现精确`WorkloadAnalyzer`；
- 显式传递ExecutionPlan；
- 补齐显存、缓存、切换和阶段Telemetry；
- 不改变当前默认执行结果。

### 阶段B：打通真实策略

- 接入真实H3 Block双缓冲；
- MLP chunk配置化；
- 实现VAE tile候选；
- 建立有界shape缓存；
- 同一任务做策略等价性对照。

### 阶段C：4090标定

- 锚点策略筛选；
- 交叉点和OOM边界搜索；
- 拟合小型可解释路由树；
- 验证任务聚合性和regret；
- 生成`rtx4090_sm89.json`。

### 阶段D：产品接入

- Creator常用档位和Advanced面板；
- 进程内无重启切换；
- 管理员Planner诊断；
- 完成发布矩阵、视觉门控和Human成片终审。

## 13. 参考实现的取舍

- LightX2V：复用H3整模型/Block Offload和双Buffer预取机制，但修正最后一个Block后循环预取Block 0的问题；不采用其按JSON人工选择策略的产品方式。
- FastVideo：复用实际latent geometry、阶段化Pipeline和通用layerwise offload的工程思想；不引入其完整分布式框架。
- SGLang Diffusion：复用`ModelDeploymentConfig + auto tuner`的策略分层思想；当前源码只对Wan自动启用逐层卸载，H3仍需自行标定。
- 当前Comfy兼容后端：只作为正确性和性能Comparator，最终Native路由不能通过启动多个Comfy子服务实现。

## 14. 最终判断

“任务参数经过一个合理指标后聚合到最优策略”是正确的总体方向，但严谨实现应是：

```text
一个主指标N负责描述H3联合序列规模
+ 少量结构特征负责修正VAE、显存和shape差异
+ 4090全策略对照负责产生真实档位标签
+ 小型可解释路由器负责在线选择
```

这样产生的是可证明、可复现、可继续校准的RTX 4090性能等价类，而不是人为命名的分辨率版本。只有完成交叉点实验后，才能宣称某个档位已经把4090压榨到当前策略空间内的极致。
