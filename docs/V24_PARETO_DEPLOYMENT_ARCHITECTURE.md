# V24 H3 帕累托部署架构

状态：V24 final v3 发布候选控制面。模型权重保持不变；用户接口只有
`sampling_steps` 与 `acceleration ∈ [0, 100]`。

## 1. 设计结论

V24 不能是“分辨率/时长命中某个版本，然后复制那张动作表”的路由器，也不能把
Forecast、Attention 和长序列执行分别交给几组互不认识的比例公式。最终系统采用三层：

```text
Human 审核与实测成本证据
          ↓ 约束可发布边界，不充当版本路由
连续工作负载模型（真实 geometry / packed tokens / resource counts）
          ↓
联合成本—风险预算求解（Actual/Forecast 与逐步逐层 Attention 共用一本账）
          ↓
请求内 Forecast 误差债务观测（复用已计算的 Actual，不增加 Dense 教师）
          ↓
请求局部物理执行表与完整审计证书
```

`acceleration=75` 是当前 Human 证据约束的质量拐点；`0→75` 决定 Dense 与该拐点
之间的预算位置，`75→100` 才进入明确允许肉眼缺陷的激进外推区。它不选择 V7、V9、
V13、V18 或 V22；这些名称只存在于 provenance 和 Human 审核记录中。

## 2. 预定义层不是机械配置表

三个共享 Human 锚点来自 720p5、720p10 和 1080p15 的连续播放裁决；720p15 的 V009
是稳定回退锚点，另保留三条只改变该连续曲面长片 knot 的最终候选供 Human 终审。
锚点保存：

- 完整 Actual 位置和 50 层物理 Attention 动作；
- 原执行摘要、成片 SHA-256 和 Human 结论；
- 该工作负载上目前允许发布的最快证据下界。

它们不保存“当条件 X 时选择版本 Y”的规则。真实生成视频 token 数在相邻证据点间用
对数尺度连续定位；步数改变时，Actual 校正点按归一化 sigma 位置缩放，并始终限制最长
连续 Forecast。历史锚点因此只承担三件事：校准、约束和可复现，不承担策略选择。

## 3. 联合求解状态

一个候选状态包含：

- 哪些 sigma 位置执行完整 Actual；
- 每个保留 Actual 的 50 个 Attention 单元处于 6.25%、10%、25%、50% 或 Dense；
- Forecast 位置只运行 0–2 层的固定浅层锚点；
- 当前估计计算量、轨迹误差风险和 Attention 误差风险。

计算成本使用同一 Dense 等效单位：

- Forecast 浅层锚点：`0.045`；
- Actual 非 Attention 部分：`0.286`；
- Attention 部分：`0.714 × 实测动作成本比`。

风险不是场景分类器。轨迹风险由连续 Forecast 长度和 sigma 相位的平滑函数决定；
Attention 风险由动作误差、sigma 相位和层位置的平滑函数决定。当前层风险使用两个平滑
峰描述既往 Human 失败集中出现的因果/交互桥接区域，而不是 `layer in 30..45` 之类的
手写档位切换。

## 4. 一个预算账本，而不是两套比例

求解以 Human 质量拐点为中心构建两段首尾相接的确定性嵌套链。`0→75` 从锚点向
Dense 方向升级，每轮让
以下动作在同一本账上竞争：

1. 把一个 Forecast 提升为完整 Dense Actual；
2. 把一个 `(step, layer)` Attention 单元提高一级保真度。

竞争指标是“单位新增计算量能降低多少建模风险”。动作顺序只由工作负载、成本模型、
风险模型和 Human 边界决定，与用户本次滑块值无关。滑块只截取这条链的一个预算前缀，
因此自动满足：

- 加速力度越大，计算量不增加；
- 更高加速档的 Actual 集合是更低档的子集；
- 两档共享的 Attention 单元不会在低加速档反而更稀疏；
- `0` 严格等于完整 Actual + Dense Attention；
- `75` 严格复现/插值 Human 质量拐点；
- `100` 是显式激进端点，不伪装成人工审核通过的高质量端点。

`75→100` 先按相同风险代理逐单元降低 Attention 保真；Actual/Forecast 轨迹在该段前
80%保持不动，只在最顶端20%允许移除最多20%的非开局、非末端校正，并继续约束最长
Forecast。这个二段曲线是技术风险层级，而不是按任务类型写的路由分支。

每次选择记录完整 chain digest、两类动作总数与已使用数、目标/实际计算单位、剩余风险和
下一项升级。这使“为什么是这张表”可以重放，而不是依赖阅读一串分支代码。

当前求解器准确的形式化主张是：在声明的平滑风险代理上执行确定性的嵌套边际分配。
它不声称真实 Human 观感全局最优。项目已有的全局有限 DP 将作为离线 oracle 检查关键
预算点的代理最优差距；在它从旧 V1–V18 策略输入中完全解耦、并证明不会破坏锚点与滑块
嵌套性之前，不直接替换发布路径。

## 5. 运行时自动反馈

旧在线 Attention verifier 需要额外 Dense 教师采样，既消耗预算，又没有可靠识别
Human 报告的门、把手、嘴型和背景异常。V24 不把它接入发布计划。

新的 `v24_request_local_forecast_debt_v1` 复用每个计划内 Actual 校正已经产生的数据：

1. 从音频与视频 target 中均匀抽取少量行，只使用 Forecast 已保存的浅层通道；
2. 用前两次 Actual 历史重建“本次若继续 Forecast”的 secant tail；
3. 与本次已经算出的真实 tail 比较 relative-L1 与 cosine loss；
4. 以开局连续 Actual 的真实 secant-tail 误差建立请求内包络，horizon 只作遥测；
5. 超出包络的连续量累积为 Forecast debt，稳定校正则连续衰减旧债务。

这条观测路径：

- 不读取 prompt、seed、人物、对白或场景标签；
- 不增加任何 DiT block 或 Dense Attention 教师；
- 不跨请求共享状态；
- checkpoint 会保存并校验债务、包络和已消费预算；
- 将来触发恢复时只允许“升级”计算，且由有限 token bucket 限制最多提升多少个 Forecast。

当前发布接线是 `observe_only`：它记录误差但不改变 Human 审核过的物理轨迹。原因是
正常任务的 debt 包络还需要 GPU 样本校准。校准完成后，恢复不是一句
`if error > threshold`，而是连续误差债务积分达到一个完整计算 token 后，才把未来一个
计划 Forecast 提升为 Actual；消费完有限预算后不能继续扩张。

## 6. 允许保留的硬分支

以下判断属于能力或安全约束，不是质量策略，因此应显式 fail closed：

- 非 H3 Base、非 SM89、非已支持 sampler/scheduler；
- 几何不满足 H3 latent 网格，或 packed tokens 超过证据上限；
- `acceleration == 0` 的 Dense 数学端点；
- 参考视频尚无 Forecast 证据时，禁止跳过完整 DiT；
- checkpoint、预览强制 Actual、缺失物理 action 或扩展不可用时回退 Dense；
- 1080p15 显存路径按输出工作集选择 exact streaming/chunking 原语。

这些分支决定“能否安全执行某种技术”，不决定“哪个历史版本更好”。参考图片、参考音频
和文本只以实际增加的 packed rows / condition rows 进入连续资源压力，不以内容语义进入。

## 7. 明确禁止的退化方向

- `if 720p15: use V009; elif 720p10: use V022`；
- 按是否有人说话、是否开门、提示词长度或 seed 选择调度表；
- Forecast 数量和 Attention 稀疏率各自按滑块独立线性缩放；
- 用瞬时功率、单帧清晰度或 raw conservative fraction 直接判定质量；
- 未经 Human 审核，把运行时观测阈值直接打开为自动恢复；
- 把 exact kernel/chunking 的工作负载能力分支伪装成质量调度版本。

## 8. 当前验证证据

截至 2026-08-26：

- 四个 `acceleration=75` Human 质量锚点全部复现原 physical execution digest；
- `acceleration=100` 在代表负载上都比75计算更少，其中720p15候选约从7.30降至
  4.83 Dense等效单位，1080p15约从5.70降至4.83；该端点明确需要单独质量审核；
- 4–30 步均能生成完整的逐步逐层执行表；
- 四个代表 geometry 的 0–100 全档单调与嵌套约束通过；
- 普通 prompt 长度变化不改变物理调度；
- 参考视频禁 Forecast、预览强制 Dense Actual、OOD Dense 回退通过；
- V22 720p10 与 V18/V13 1080p15 exact helper 只绑定各自已验证执行边界；
- Forecast debt 的 observe-only、有限 token bucket 和无教师声明通过单元测试；
- FL2VA/Ref2VA Base 共用同一曲面，参考图片/音频只通过资源压力连续加保护；两条
  LoRA 路线保持完整蒸馏步轨迹，只让同一加速控制量调度 Attention；
- 1080p×15秒已成为四条服务路线的公共接单上限，Base长序列在250k packed-token
  静态参考媒体包络内使用精确流式路径。

仍需完成的发布证据：

1. Human 连续播放终审三个新720p15 knot；Round3四短片与四长片已因门接触因果、
   背景/器具形变和交接手残影整体判负，不能进入发布曲面；
2. 用 720p5、720p10/15 和 1080p15 正常样本标定 debt 的 null envelope 与观测开销；
3. 用至少一个历史失败样本验证 debt 是否真的先于可见崩坏升高；
4. bounded recovery 与 observe-only 做同 seed A/B，确认质量收益能够抵偿额外 Actual；
5. 只有上述证据通过后，才把运行时模式从 `observe_only` 改为
`bounded_recovery`。

首条 720p5 GPU 校准曾验证过 horizon-square 归一化，并立即将其否定：正常校正的风险比
被压到 `0.064–0.144`，信号衰减 7–16 倍。取消该先验后，同一条 byte-identical 成片
的请求内 raw ratio 为 `1.020–1.296`，正常债务累计仍低于一个恢复 token。代码保留
horizon 供后续拟合，但不再假定误差必然按 horizon 平方增长。

随后完成的 720p10/V22 与 720p15/V009 校准再次保持 Human 锚点逐字节一致；denoise 分别
为 `129.810s` 与 `295.090s`。但两条正常、Human 已通过的轨迹也产生了 `0.923` 与
`0.711` 的末态债务，其中中段 raw ratio 可达 `1.649`。这证明“只用开局 Actual
建立基线，然后在 debt >= 1 时补算”仍可能误触发，不能作为发布控制律。下一版控制信号
必须是经过训练/留出验证的连续 phase×workload null envelope，并在历史 Human 拒绝样本
上证明具有区分度；在此之前 `bounded_recovery` 继续保持不可达，发布只记录 telemetry。
