# V19 最终候选版本开发与验收合同

日期：2026-08-19  
硬件边界：单张 RTX 4090 24GB  
模型边界：MiniMax-H3 Base/既有 LoRA；不训练、不修改 checkpoint  
服务边界：FL2VA、Ref2VA；不得减少 Dense 原本支持的提示词与参考素材能力

## 1. V19 的完成定义

V19 不是 V18 再增加一批阈值。它必须同时满足：

1. 校准、风险、求解、执行和证书指向同一个物理动作实现；
2. 给定 `total_steps + acceleration`，返回不可被已知候选同时在速度与质量上支配的计划；
3. 清晰度、物理因果、轨迹、身份、音频和异常是不可互相抵偿的约束；
4. 未标定/OOD 请求继续接受，但自动回退，不因稀疏框架缩减输入能力；
5. 缓存、Forecast、稀疏 Attention 都是可拒绝的一等动作，不再由隐藏 if/else 注入；
6. 只有盲 Human A/B 通过后，动作/计划才能进入发布前沿。

## 2. 冻结比较器

- Dense 同 prompt/seed/shape/steps：模型能力与归因比较器；
- Round143/144：720p×5s 接触因果正例；
- Round188：12/8 历史手工速度—质量比较器；
- Round216：预算框架下 Human 正例；
- Round63、82、105、149、154、158、V13：已知失败拒绝集；
- Round24～26：Segment Cache 音频漂移、模糊和鬼影拒绝集；
- NVIDIA H3 RTX4090 Sol/TeaCache：同模型同硬件机制比较器，不直接比较官方倍率。

V19 自动计划若比已知候选更慢且没有逐维 Human 质量收益，必须被 dominance 门拒绝。

## 3. 开发阶段

### Gate A：测量对象闭环

- 版本化 Action Registry；
- 每个动作绑定 implementation、executor、calibration、risk、Human evidence；
- registry digest 进入校准产物和计划证书；
- 缺失或身份不匹配一律 fail-closed。

完成标准：发布桶成本 p90 误差不超过 5%，实际 action count 与证书逐格一致。

### Gate B：复合动作

- Dense/AdaptiveSparse；
- `(forecast run, anchor depth, anchor action, extrapolator, correction)`；
- 受控 Cache/refresh 动作。

历史 Segment Cache 登记为 `REJECTED`，不得默认复活。新缓存动作必须是新 implementation
和新证据，不能继承旧缓存的安全结论。

### Gate C：Human 风险与轨迹债务

逐维约束：提示词遵循、接触因果、轨迹连续、时序清晰、身份绑定、音频、异常。
Dense refresh 只能降低经实验证明可恢复的债务，不能默认清零历史 forecast/cache/sparse 债务。

### Gate D：统一规划与有限 Online

- 离线 Pareto/DP 选择候选计划；
- acceleration 映射到已认证前沿，而非固定 Top-K；
- 在线只允许 densify、提前 actual、refresh、拒绝 forecast；
- 在线税上限 0.5%，无自然改善正例则删除在线控制。

### Gate E：逐级验收

1. CPU 合同/证书测试；
2. 单 Block 和单真实 DiT step；
3. 480p×5s 五类物理/音频机制、每类多 seed；
4. 720p×5s；
5. 720p×15s；
6. Base/LoRA、FL2VA/Ref2VA；
7. API/UI/ComfyUI 只映射 `total_steps + acceleration`，高级诊断参数不成为创作者开关。

## 4. 当前实现状态

- 已完成：V19 Action Registry、证据强绑定、非抵偿风险合同、轨迹债务初始状态、输入能力合同、严格 Pareto 前沿与全身份计划证书、带 comparator 归因的 Human 机制证据格式与首批历史迁移；
- 已完成：Round215、Round188 和 Round229 的实现特定物理动作校准；Round229 的 Forecast 一等复合动作、完整请求级成本、运行时蓝图无损编译；
- 已完成：第一版有限预算逐格优化器。它冻结已审 comparator 的 Dense 因果轨，使用精确物理 p90 与 Dense 相对最坏数值误差做候选搜索，但明确不把数值误差冒充 Human 风险；
- 已完成：发布候选不再按一个精确 `packed_tokens` 点死匹配；改为由真实端点、
  参考素材组合和 Human 场景证据共同封存的闭区间 workload envelope。区间之外
  接受请求但回退 Dense，不限制提示词长度或参考素材能力；
- 已完成：服务在 Qwen 真正分词、参考媒体真正预处理之后才进行 V19 选择，避免
  使用字符数估算把长提示词路由到错误计划；完整实际/Forecast 轨迹与逐格
  Attention 动作一起由证书替换，V18 不再二次改写；
- 已完成：V19 release bundle v2、启动时 runtime/registry/digest 复验、预览锚点
  必须为实际 DiT 步的约束，以及缺 bundle/OOD/未标定预览点的 Dense fail-closed；
- 已完成：Human 风险证书 v2 将候选成片与对照成片的内容 SHA256 一并封存，
  不再只信任可被替换的文件路径；API 任务记录同时返回并持久化最终
  `inference_plan`，便于核对实际候选、证书、Actual/Forecast 排布与 Dense 回退原因；
- 未完成：新 V19 候选的完整 Human 风险 UCB、多场景/多 seed 校准，以及
  Base/LoRA、FL2VA/Ref2VA 的最终 Gate E 发布覆盖；
- 发布默认保持原路径，V19 在 Gate E 完成人审前不得成为默认。

## 5. Gate A 实测闭环（2026-08-19）

固定工作负载为 Base FL2VA、1280×736、124 帧、20 个采样位置、10 个真实步与
10 个 Forecast 步，真实步为 `0,1,2,3,4,8,12,15,18,19`，packed token 为
34,871。运行时 digest 为
`b9dd3f9ec35aedbbeb24e5b7272c67885d99d29f473d3c7906f6d9f1bda819a9`。

Round229 融合 RMS 比较器重复三次：

- 端到端：68.745 / 66.181 / 66.059 秒；p50 66.181 秒，p90 68.745 秒；
- 去噪：52.321 / 52.004 / 51.905 秒；
- 峰值显存：9.148 GiB；
- 三次均逐格执行相同的 530 个动作单元（500 个真实 Attention 单元 + 30 个
  Forecast anchor 单元）。

配置继承缺陷曾产生一组未启用 fused RMS 的诊断结果（约 74.6～77.2 秒）。该组
已移动至 `runtime/calibration/v19_gateA/diagnostic_unfused/`，不得混入最终基线。

精确逐层校准包含 500 个 Dense 单元和 2,000 个 Round229 稀疏单元，每个单元
均有三次热态耗时样本。比较器 Attention p90 为 13.236 秒；同一物理动作集逐格
取最快的理论下界约 12.88 秒。因此，在固定 10/10 轨迹上继续降低 Attention
保留率最多只返还约 0.36 秒，不能支撑大幅端到端提速。下一阶段的大收益必须来自
真实步/Forecast 复合轨迹与非 Attention DiT 主干，而不是继续手调一个 Top-K。

严格证据位于：

- `runtime/calibration/v19_gateA/registry_bound_final/round229_base_first_last_1280x736x124_all_actual.json`
- `runtime/calibration/v19_gateA/registry_bound_final/dense_round229_workload_base_first_last_1280x736x124_all_actual.json`
- `runtime/calibration/v19_gateA/registry_bound_final/round229_forecast_base_first_last_1280x736x124_repeat3.json`
- `runtime/calibration/v19_gateA/registry_bound_final/round229_comparator_base_first_last_1280x736x124_e2e.json`

## 6. 第一版 V19 候选搜索

比较器 500 个真实 Attention 单元的动作分布为 310 个 Top-K 0.0625、180 个
Top-K 0.1、10 个 Top-K 0.25。预算优化器在不增加逐层 p90 的条件下找到的首个
候选为 320/169/11；其逐层 p90 为 13.176 秒，且 Dense 相对最坏误差代理总量
低于比较器。这个结论只足以支持“值得实际跑片”，不支持“Human 质量更好”。

候选蓝图由 execution digest 封存，并由 benchmark 直接编译为物理调度表；不再
经过 V18 scheduler 二次改写。完整重复实测和 Human 盲审通过前，其风险仍按最坏
值处理。

## 7. Gate B：720p×15s 完整计划实测

使用同一 V19 execution digest
`0dfe2d16c7c573c2018031c62b7d1f336efa98c5324d3ad08fa315c42d960069`，
同一 Base FL2VA 任务、1280×736、362 帧、20 个采样位置、10 Actual/10
Forecast，packed token 为 100,163。三次热态端到端实测为：

- 217.295 / 216.893 / 214.394 秒；p50 216.893 秒，p90 217.295 秒；
- 去噪 174.643 / 174.427 / 174.257 秒；p50 174.427 秒，p90 174.643 秒；
- 峰值显存三次均为 17.368 GiB；
- 执行轨迹三次均为 Actual `0,1,2,3,4,8,12,15,18,19`，其余位置为
  Directional Forecast；每次均执行相同 530 个 Attention/anchor 单元。

严格成本证据位于：

- `runtime/calibration/v19_gateB/registry_bound/v19_frontier_r1p0000_base_first_last_1280x736x362_schedule.json`
- `runtime/calibration/v19_gateB/registry_bound/v19_frontier_r1p0000_base_first_last_1280x736x362_forecast.json`

这组结果证明当前完整计划的热态速度稳定，但不是发布质量证书。720p×15s 的
Round229 逐层动作校准和逐维 Human 风险覆盖尚未完成前，它仍是 research/OOD
结果，不能仅凭一次“看起来正常”写入发布 bundle。

## 8. 发布选择与能力保持

V19 的 creator-facing 输入仍只有 `sampling_steps + acceleration`。内部顺序为：

1. 保留用户的完整文本、首尾帧或多参考输入；
2. 完成真实 Qwen token 与参考媒体 token 计数；
3. 用模型、服务族、几何、总步数、真实 packed-token 闭区间和精确参考组合选择
   最具体且唯一的认证 envelope；
4. 在该 envelope 的不可支配、逐维风险单调前沿中选择完整计划；
5. 计划一次性给出 Actual/Forecast 排布与每个实际层的 Attention 动作；
6. 任何证据、runtime digest、参考组合、提示词长度端点或预览锚点不匹配都执行
   Dense，而不是拒绝原模型本来支持的请求。

当前已重复测速的 720p×5s/15s source 都是 Base、无外部参考媒体的
`first_last` 服务族请求。现行 schedule 证据只封存 packed/condition token，尚未封存
Ref2VA 的逐类图片、音频、视频数量，因此发布构建器明确拒绝用 source-spec 手写
非零参考数量；Ref2VA、LoRA 和未认证的首尾帧组合继续走能力不缩水的 Dense 路径，
不能借用 T2VA 的风险或耗时证据。

发布包构建入口为 `scripts/build_v19_release_bundle.py`。它会重新加载并复验
Attention、Forecast、完整 E2E、Human 风险、计划证书与 evidence SHA256；任何
一项不完整都会失败，不能通过手写 JSON 绕过。默认发布目标是
`h3serve/native_engine/planner/evidence/v19_release_bundle.json`，该路径由
`pyproject.toml` 的 package-data 收进 wheel；部署方仍可用
`H3_NATIVE_V19_RELEASE_BUNDLE` 指向另一个已封存 bundle。
