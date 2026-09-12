# 1080p×15s 单卡高质量高效路径

状态：实验路径已完成低步数精确端到端门、Video-VAE/后处理物理显存门，以及第一组
20 步质量承载 1080p15 A/B；Human 长片验收和多任务/多 seed 仍未完成，未扩大正式
UI 上限。  
边界：单张 RTX 4090 24GB；模型权重、采样语义和用户的 `steps + acceleration`
接口不变。

> `1080p×15s` 是当前最困难的压力测试锚点，不是代码里的固定任务模板。
> 最终机制必须同时服务 FL2VA/Ref2VA，并适应任意文本长度、参考图片和参考
> 音频带来的 packed-prefix 变化。

## 0. 任务不敏感合同

1. 不读取提示词语义，也不按某条提示词、seed 或固定 text-token 数调参。
2. 每次请求从真实 `PackedLayout` 读取目标视频起点、latent frame 数和每帧
   token 数；不得写死 `1,723`、`220,003` 或某个 condition count。
3. 文本、参考图片、参考音频和目标音频组成可变 protected prefix。其全部
   token 始终保留为 K/V；prefix Query 也可为控制显存而分块，但不能稀疏删除。
4. Query/projection chunk size 只能由 packed 几何、物理后端和当次可用显存
   决定；它是执行参数，不是内容或质量参数。
5. 当前产品目标不主动支持超长参考视频，但如果请求含参考视频，规划器仍须把
   其真实 token 计入预算；未经验证的形状必须 fail closed，不能借用文生视频
   的固定参数继续运行。

## 1. 问题定义

当前研发主基准不是 720p×5s，而是单卡长视频。基准分层如下：

| 层级 | 任务 | 用途 | 能否支持发布主张 |
|---|---|---|---|
| 快速回归 | 720p×5s | 检查崩坏、执行 digest、机制方向 | 否 |
| 调度开发 | 720p×10/15s | 暴露 Forecast 累积、动作/口型漂移 | 仅作候选筛选 |
| 压力锚点 | 1080p×15s（及不同宽高比） | 时间、质量、峰值显存和稳定性联合前沿 | 是，须多任务 Human 门 |

因此“帕累托”至少是三目标：Human 质量更高、E2E 更短、峰值显存更低；稳定完成、
不依赖 WSL 共享显存和任务内容不敏感是硬约束。功率/SM 利用率是诊断信号，不单独
作为优化目标，避免为了追求表面 450W 而增加无效计算。

当前 720p×15s 的 V19 路径已有真实速度证据：100,163 packed tokens、峰值
17.368 GiB。把同一时长提升到 1920×1088 后，本次真实请求的 packed sequence
为 219,659（随提示词和参考媒体变化），
不是简单增加约两倍工作：whole-query Sparse 的 selector/sort/LUT 同时按
`query_blocks × key_blocks` 近似平方增长。

| Shape | Output | Latent | Tokens/frame | Packed tokens | 当前合同 |
|---|---:|---:|---:|---:|---|
| 720p×15s | 1280×736×362 | 107×46×80 | 920 | 100,163 | 支持 |
| 1080p×15s | 1920×1088×362 | 107×68×120 | 2,040 | 219,659¹ | 实验路径通过 |

¹ 仅为本次无参考媒体请求；路由使用请求派生的几何和 token 数，不匹配提示词内容。

早期合成压力探针的 `220,003` 仅是一个标定点。原执行关系为：

```text
full hidden
   └─ full QKV projection (仅 Q/K/V 输入就 8.812 GiB)
        └─ whole-query selector + sort + LUT (估算下界 5.812 GiB)
             └─ full Attention output → full out projection → residual
```

真实 block 20 已复现两种失败表现：Dense 为 263.279s/26.157GiB，whole Sparse
为 71.699s/31.110GiB；二者 NVML 显存贴顶、平均功耗仅约 111–114W。因此当前首要
矛盾不是继续调 Top-K，而是消除 full-sequence 临时张量和 selector 峰值。

## 2. 两层方案

### A. 等价内存救援层（优先实现）

这一层不得改变模型结果，目标是先让 1080p×15s 在物理显存内正常计算：

```text
full hidden
   ├─ 一次性生成/量化 prepared K + FP8 V
   └─ block-aligned Query chunks
         ├─ chunk Q projection + 原 RMS/QKNorm/RoPE
         ├─ chunk selector/sort/LUT（完整 K/V 与 MTCR 均保留）
         ├─ Dense 或 Sparse Attention chunk
         └─ chunk out projection + gated residual 原位写回
```

具体约束：

1. Query chunk 必须按 128-row Sparge block 对齐；尾块保留原 padding 语义。
2. 每个 chunk 使用完整 text/reference-image/reference-audio/target-audio/video
   K/V，不能改成局部窗口；可变 prefix Query 自身也分块，避免参考数量改变时
   重新产生峰值。
3. Query 使用原始全局 video index，使 MTCR、本地 rail 和远程 anchor 相位不变。
4. K pooling、smooth-K INT8 quant、V FP8 quant 每个 layer 只做一次，并由所有 Query
   chunks 复用；现有可行性探针的重复 K quant 只允许存在于实验脚本。
5. INT8 ConvRot QKV 按输出行切成 Q 与 K/V，但保持同一行量化、BF16 输出和 bias
   边界；先在 720p5、720p15 上证明逐元素等价。
6. Attention output 不得重新拼成完整 `[L, 7168]` 后再投影；out projection 和
   AdaLN gate 在 chunk 内执行并原位累加到 `[L, 5376]` residual。
7. Dense action 同样按 Query chunk 执行。Dense Sage 的每个 Query 行只依赖完整
   K/V，理论上可精确分块，但必须由数值测试证实。
8. MLP 继续使用现有精确 chunk 路线；路由器可在 2,048/4,096/8,192 中按剩余
   headroom 选择，不允许依赖 WSL 显存超额。

已完成的机制证据：32,768-token Query chunks 相对 whole Sparse 在 271 个跨边界
采样行上 BF16 完全一致；两遍式真实 block 的 streamed Sparse 约 1.93s、峰值约
17.34GiB，streamed Dense 约 3.55s、峰值约 17.45GiB。独立探针已证明 QKV
projection、fused QKNorm/RoPE、output projection 的 whole/chunk 边界一致。旧式
strided-QKV Dense 与 separate contiguous Q/K/V 的 Sage 执行边界仍不应被当作
等价 comparator；生产实现因此保留原物理后端语义，不借此替换 action。

整步和端到端证据已经补齐：

| Gate | Steps | Denoise / step | E2E | DiT peak allocated | 结果 |
|---|---:|---:|---:|---:|---|
| V19 v007 checkpoint | 1/20 | 131.117s | 135.477s（不解码） | 19.573GiB | 50 层完成，约 474–482W |
| Round229 accel75，旧 cache 行为 | 4/4 | 807.299s | 912.941s | 计算中发生跨步 cache/HMM 抖动 | 完成但不可接受 |
| Round229 accel75，跨步 compaction | 4/4 | 302.560s | 410.706s | 19.704GiB | 完成，E2E 2.223× |
| 上项 + 精确 VAE 时间流式后处理 | 4/4 | 303.483s | 397.144s | 19.704GiB | 相邻基线 E2E 1.034×，MP4 字节相同 |

跨步 compaction 只在 long-sequence streaming 已启用时，于 solver step 的自然边界
同步并释放已无引用的 Dense/Sparse 临时 workspace。它不读取提示词，不改变 latent、
权重、action、采样器或后端。修正前后的两个 68.54MB MP4 具有相同 SHA-256：
`bcba0bf8485a3beac6944cceac57344befd2704c27ce6c63c049ee1fb2c9004d`，因此本次
2.223× 是同一输出的执行层提速，而不是质量换速度。

Video-VAE 的物理风险已经单独处理。原后处理会把
`1×3×362×1088×1920` 的完整输出一次性转为约 `8.451 GiB` FP32，再产生 uint8；
新路径只由输出几何选择时间块（本形状为 10 帧），逐块执行完全相同的
clamp、scale、round-to-even 和 uint8 D2H。隔离后处理峰值从 `23.242 GiB` 降至
`4.754 GiB`，速度 `1.559×`，完整输出 SHA 相同。正式 Video-VAE 权重完整解码峰值
`16.952 GiB`；新 4-step 完整请求累计 peak `19.704 GiB`、Video-VAE `83.341s`，
相邻旧基线分别为 `32.566 GiB`、`92.673s`。旧/新 MP4 SHA 相同。

这说明低步数机械执行链已经处于物理显存内，但四步视频不提供质量证据；下一门是
质量承载调度的长片，而不是把低步数结果当作发布证书。

第一组质量承载 20-step 门现已完成。真实 1920×1088×362、219,890 packed-token
灯塔双人对白/动作任务，在同一热会话得到：

| 调度 | Actual/Forecast | Denoise | E2E | Peak allocated |
|---|---:|---:|---:|---:|
| v009 短片清晰度对照 | 10/10 | 1176.203s | 1268.669s | 19.777GiB |
| v012 长序列 Round188 replay | 12/8 | **635.362s** | **731.638s** | 19.781GiB |

v012 E2E `1.734×`、Denoise `1.851×`，节省 537.030s。12 个 Actual 的均值仅
48.238s，而 v009 的 10 个 Actual 均值为 111.508s；这证明长序列的真实成本由逐层
Attention action 主导，不能用 Actual 数量代理。

本次还发现 Forecast 控制器原本会为每个 Actual 保存完整 pinned tail。单份 BF16
history 为 2,359,913,472 bytes；CUDA pinned 登记尝试约 4.70GB 并在首步 OOM。
修正后，单份 history 不小于 2GiB 时使用 pageable host 和 4,096 行分块保存/回读；
720p15 约 1.08GB，仍保留原 pinned 快路径。最终运行的 Actual observation 平均
1.381s、Forecast history prediction 平均 1.224s，数学不变，相关零容差/CUDA 测试
通过。活跃功率 p10/p50/p90 为 471.0/478.33/480.15W，未回到早期 110W 抖动路径。

两条 1080p15 视频静态灾难门均通过；Human 尚未审核嘴型、接触因果、对白顺序、
闪烁和音频，因此上述结果是性能晋级证据，不是质量发布证书。

### B. 计算量缩减层（A 稳定后再做）

即使消除抖动，22 万 token 的固定比例 Top-K 仍近似二次增长。第二层才研究真正的
长序列提速，并接受 Human 质量门：

1. 将 action 的“保留比例”改为“结构 rail + 受限绝对 KV-block 预算”，避免序列
   翻倍时每个 Query 的证据数也无条件翻倍；预算上限从已通过的 720p15 绝对 block
   数推导，而不是拍阈值。
2. 使用低成本 pooled spatial/temporal router，为运动边界和接触区域追加预算；
   text/audio prefix、首尾 latent frame、MTCR local rail 始终保留。
3. 探索前期低分辨率 latent、后期 1080p correction 的多尺度 Actual 路径；音频轨
   全程保持原始 schedule，不随视觉降采样。
4. 只有前述路径失败时，才研究带重叠状态与统一音频轨的 temporal window；简单把
   15 秒切成两段会破坏身份、动作和语音连续性，不作为首选。

这层允许“可控近似”，但必须加入当前 V19 的 Pareto/Human 体系，不能仅凭单 block
误差或 VQA 分数发布。

## 3. 路由和显存合同

实验实现现已由内部 workload planner 自动选择，用户仍只看到 `steps` 和
`acceleration`：

- 以请求派生的 `packed_tokens`、video tokens 和实际执行计划为输入，不读取
  提示词语义；Attention 内部仍以真实 `PackedLayout` 处理 prefix 边界；
- `video_tokens < 200,000` 不启用长序列 Query streaming/跨步 compaction，因此
  720p×15s、1080p×5s 等已验证安全配置不会承担这部分开销；Video-VAE uint8
  后处理独立按输出 FP32 工作集路由，只在工作集超过 4GiB 时精确时间分块；
- 已测的普通 1080p×15s bucket 使用 Query 32,768 / projection 8,192；packed
  prefix 达 230,000 后使用 Query 16,384 / projection 8,192；超过当前 250,000
  token 证据包络时 fail closed，显式研究参数仍可覆盖该门；
- 目标 peak allocated 不高于 `22.5 GiB`，为 CUDA context、mux/VAE 切换和分配碎片
  保留至少约 `1.5 GiB`；
- chunk size 是执行参数，不是质量参数，不暴露到 UI；
- 每次请求记录 chunk size、prepared-KV reuse、逐层 peak、NVML power/utilization 和
  是否发生 allocator retry/显存超额；
- 未通过的 shape fail-closed，不能静默借用 720p profile，也不能依赖 WSL shared
  memory 把 OOM 变成长时间抖动。

### 3.1 统一调度接口的实验接入

v012 已从基准脚本中的手工蓝图接入原生热会话选择层
`V19ExperimentalLongRuntimeSelector`。用户仍只提交 `sampling_steps` 与
`acceleration`；选择发生在 Qwen 编码和参考媒体预处理之后，因此使用真实
`packed_tokens`，不读取提示词文字、seed 或场景类别。

当前接入边界故意小于未来产品目标：

- 只接纳已完成 E2E 的 `1280×736×243×20`、`1280×736×362×20` 和
  `1920×1088×362×20` Base 无参考媒体 envelope；其他比例、Ref2VA 和参考
  图片/音频继续 fail closed；
- 只有 `acceleration >= 75` 才进入 v012，75--100 都锁在同一个已测 12 Actual /
  8 Forecast 端点，不能在人审前继续减少计算；
- 生成的 execution digest 必须是 Batch07 实测的
  `495c2c8ff75f76aed16b1fd81a41f3e050df5bad38456dd68d0806c3e9c7cbad`；
- 若预览需要的锚点不是 v012 Actual、token/参考 profile 超出实测包络，立即委托给
  正式 V19 release selector；没有正式 bundle 时执行全 Dense，不借用近似路线；
- 只有部署侧显式设置 `H3_NATIVE_V19_EXPERIMENTAL_LONG_HORIZON=1` 才加载该层。
  默认值为 0，且执行摘要固定记录 `experimental=true`、
  `release_eligible=false`。Human 连续播放通过之前，不能写进签名 release bundle。

这一接入先解决“实验跑得出来、统一框架却不会选”的控制面断层。它没有把单一灯塔
提示词固化成策略；相反，prompt 根本不属于选择器输入。720p10 的新 held-out
无线电控制台场景实测为 67,535 packed tokens，v012 从 v009 的 190.157s 降至
152.124s，E2E `1.250×`、Denoise `1.283×`，反向执行顺序下仍通过 1.10× 速度门。
其连续播放 Human 门仍待审核。参考媒体暂未接纳是因为它会
增加真实 packed rows 和峰值显存，属于尚未测量的物理资源 envelope，而不是因为选择器
理解或区分了参考内容。后续按 `不同 prefix/参考图音频 → 不同 1080p 比例 → Ref2VA`
逐项扩大证据边界。

## 4. 验收顺序

1. Attention：whole vs chunk 输出边界/随机行/全量小 shape 等价；除 Dense
   strided-QKV comparator 外已完成第一组。
2. 单真实 block：720p5/15 全量逐元素对照；1080p15 peak ≤22.5GiB，且 block 不再
   出现分钟级抖动。已完成。
3. 单真实 DiT step：50 层 action count、Forecast anchor、音频/视频 tensor hash 和
   V19 blueprint digest 一致。v007 checkpoint 门已完成。
4. 1080p×8s：当前上限内 A/B，证明启用 streaming 不降低质量也不拖慢正常热路径。
5. 720p×10/15s：将通过短片硬门槛的近似调度放大，先排除 Forecast 长时累积、
   动作突变、嘴型模糊和语速异常；不通过者不耗费 1080p 资源。
6. 1080p×15s：低步数机械端到端门和首个完整 20-step 固定 seed A/B 已完成；继续
   补多任务/multi-seed、不同 prefix 和 Ref2VA。每次记录 E2E、Denoise、VAE decode、
   peak allocated/reserved、allocator retry、NVML 功耗/利用率和 host PSS。
7. Human 连续播放：清晰度、口型、运动因果、身份、语音自然度和长时漂移分别判断。
8. 至少多 prompt/multi-seed、不同 prefix 长度、FL2VA/Ref2VA 通过后，才扩大
   `MAX_NATIVE_PIXEL_FRAMES` 和 UI preset limit。

最终成功门是：不改权重的前提下可靠完成 1080p×15s；相对任何可运行的当前路径
E2E 至少 `1.10×`，或者在相同 E2E 下由 Human 确认质量更好。低步数机械门已通过
跨步 compaction 相对原路径达到 `2.223×`，并通过 VAE 后处理相对紧邻基线再达到
`1.034×`，均为字节相同输出；首个完整 20-step 已达到相对可运行对照 `1.734×`
E2E。多任务/多 seed、Ref2VA 和 Human 长片质量验收仍未完成，因此没有声称发布
目标已经全部达成。
