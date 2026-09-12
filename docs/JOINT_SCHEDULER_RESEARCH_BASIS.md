# H3 联合调度器的方法依据与原创边界

日期：2026-08-19

## 1. 问题类型

当前问题是带资源约束的有限路径优化。每个 σ 位置是一层图；边表示预测步，或一个真实
DiT 步及其七层带 Attention 动作组合。边有 4090 实测成本与风险债务，路径必须满足开局、
末端、真实步数量和最大连续预测长度约束。v3 使用 Pareto-pruned 动态规划精确求解声明的
有限问题。这类形式化与 resource-constrained shortest path 一致；参考
[Ahmadi et al., AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17450)。

“精确”只修饰有限代理问题，不修饰人类观感。风险表、动作集合或成本模型错误时，DP 会
精确地优化错误问题。因此证书、4090 实测和 Human 连续播放是三条不同证据，不能互相替代。

## 2. 为什么整体预规划、局部 Online

整体预规划负责硬预算、真实/预测位置和初始 Attention 配额；它可审计、可缓存，并能避免
请求内无边界搜索。Online 层只读取当前请求的 Dense-vs-draft 因果探针和实际 CUDA 时间，
在显式恢复金内把局部动作升级，随后重算剩余轨迹。它不能降低既定保护，也不能假设末段
Dense 会消除早期形成的错误。

外部工作支持“稀疏模式与请求/Head有关，在线信号有价值”，但并不直接给出 H3 的调度器：

- [AdaSpa, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xia_Training-free_and_Adaptive_Sparse_Attention_for_Efficient_Long_Video_Generation_ICCV_2025_paper.html)
  使用层次化 block pattern、LSE cache 和在线精确搜索，并报告其他视频 DiT 上 1.59–2.04×；
  本项目只复用“离线结构＋在线请求适配”的原则，H3 的多模态 prefix、因果层和 4090 成本
  必须重新验证。
- [Sol-Attn](https://arxiv.org/abs/2607.24027) 使用在线阈值、proxy 复用和跳过块补偿；它是
  候选 Attention 动作，不是调度器质量证明。本项目已有激进 tau 导致动作异常的 Human
  反证，故不能照搬论文倍率。
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814) 可在独立 Human 校准集和单调损失
  条件成立后把经验风险阈值变成统计风险控制；当前 v3 标签远不足以提出该保证，只保留接口。

## 3. v3 真正新增的内容

与 v1/v2 相比，v3 的新增不是调一个 TopK：

1. 把 Round215 的 34,871/100,163-token 七层带×五动作真实成本固化为两个可审计端点；
2. 将真实/预测位置和 Attention 动作放进同一个有限 DP，而不是先手写骨架再分配 Attention；
3. 每条计划带 shape 模型、预算、动作路径和最优值哈希，可完整重放；
4. 保留 2% Dense-equivalent 恢复金，并只允许请求内因果探针做局部升级；
5. 评价器把 Dense 恒等、预算、单调性、结构轨、shape 匹配、证书、Human 正负证据、规划
   延迟和双参数接口拆成独立门。

H3 特有价值来自本地证据：30–43/45 层的接触因果风险、Round216 正标签、Round217 仅放松
因果岛即失败、长序列 per-warp 稳定性，以及音频/身份/运动不能被静态清晰度抵消。外部论文
没有提供这些结论。

## 4. 当前限制与下一轮证伪

- v3 风险仍为加性代理，尚未由足量 Human 标签校准；
- 两个 shape 端点之间是插值，端点外是显式 OOD 外推；Ref2VA 视频条件仍缺独立成本表；
- v3 会提出不同于历史 12/8 的中段真实步位置，必须独立审核，不能继承 Round143 标签；
- Online 探针的恢复开销虽有预算，但运行时尚未形成严格 SLA 超时/质量优先的双政策闭环；
- 发布前必须证明：v3 在至少一个档位不被 Round143/216 同质量同任务 Pareto 支配，并在多类
  接触、遮挡、交接、对白和参考条件任务中取得自己的 Human 正标签。

## 5. V11：为什么只做局部 Online，而不是全程在线重规划

Round221 的 14 条真实遥测显示，sampled Dense-vs-draft relative RMS 的绝对值随请求 shape
明显变化，但同一任务同一层的相位增长更稳定。因此 V11 使用请求内首相位作 scale，阈值由
校准集最大 order statistic 加 10% 边际构造。其统计主张严格限制为：若未来任务与校准任务
可交换，单一 task-max 分数的边际覆盖下界为 `14/15`；它不是视频质量覆盖保证。

不采用“每一步在线重新搜索完整动作表”的原因：

1. 当前 Dense probe 本身有成本，全局在线搜索会破坏用户预算的可预测性；
2. 早期物理错误可能不可逆，看到误差后重新分配剩余层不能保证修复历史；
3. Human 数据不足以让在线 proxy 可靠区分合理的大动作与错误运动；
4. 预先 DP 已能对声明的有限问题给出可重放最优证书，Online 应只处理离线模型没见过的局部
   偏离，并且只做 upgrade。

这与 [AdaSpa](https://arxiv.org/abs/2502.21079)、
[Sparse-vDiT](https://arxiv.org/abs/2506.03065) 和
[SPADE](https://arxiv.org/abs/2608.03335) 的输入/层适配方向一致，但 V11 没有声称复现这些
方法。特别是本地 Round28 的 map reuse 在 100k tokens 视频分支只有 1.0425×，且没有跨步
稳定性证明，所以没有为追求低个位数收益把缓存状态加入生产热路径。

准确结论是：当前合理架构仍是“整体预设＋局部 Online 纠偏”。若未来获得足量按场景划分的
Human 标签和自然触发正例，才有依据把局部守卫升级为带风险上界的 receding-horizon/MPC；
在此之前，全 Online 只是复杂度更高、证据更弱。
