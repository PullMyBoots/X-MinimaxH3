# V24 偏好贝叶斯优化与 40 视频收敛协议

状态：历史发布前学习协议；实际收敛结果已由 V24 final v3 取代。
冻结模型权重与物理近似动作集合，只校准策略曲线。由于 Human 明确要求
100 档具有可见的速度吸引力并接受质量缺陷，最终语义是 75 为人工审阅拐点、
75–100 为明确的激进外推区；以
[`V24_FINAL_RELEASE_2026_08_26.md`](V24_FINAL_RELEASE_2026_08_26.md) 为准。

## 1. 建模对象

用户仍只输入总步数 `N` 与加速力度 `a∈[0,100]`。系统通过低维曲线参数
`θ` 生成完整高维执行向量：

```text
z = F(N, a, workload; θ)
```

`z` 包含每个 sigma 位置的 Actual/Forecast、每个已执行 DiT 层的 Attention
保真度与物理 rail。20 步 H3 的核心向量超过 2000 个混合离散坐标，但这 40
个视频不直接拟合这些坐标。被学习的 `θ` 只有以下连续机制参数：

- Forecast 相对于 Attention 误差的交换尺度；
- 连续 Forecast run 的复合代价；
- opening/terminal 相位的平滑幅度与宽度；
- causal/bridge 层峰的平滑幅度、中心与宽度；
- Dense 到 Human 锚点之间的预算曲率。

这使有限 Human 样本学习机制规律，而不是记忆某些视频的离散动作表。

## 2. Human 观测

每个比较组固定 prompt、seed、尺寸、帧数和步数。Human 提供：

1. 组内从好到差的排序，允许并列；
2. 每条视频 0–100 总体评分；
3. 九个问题维度的 `present / absent / not_reported`；
4. 对具体画面或声音机制的自由说明。

问题维度固定为 prompt adherence、contact causality、trajectory continuity、
temporal clarity、identity binding、object geometry consistency、object count
consistency、audio integrity 与 anomaly。物体形状变化和物体数量错误分开记录，
避免把钥匙弯折、灯数错误等可诊断故障压缩成笼统 anomaly。跨分辨率视频不做
强制直接排序，以免把分辨率偏好错误归因到调度向量。

## 3. 后验模型

排序与评分差编译为成对比较：

```text
P(z_i ≻ z_j) = sigmoid(βᵀ(φ(z_i)-φ(z_j)))
```

`φ(z)` 是从完整向量得到的可解释投影，包括 Forecast 数量/连续长度/相位暴露、
Attention 的时间与层级暴露、Forecast×Attention 交互和计算比例。`β` 使用高斯
先验，采用 Bradley–Terry likelihood 的 Laplace 后验。模型还加入策略特征与
空间负载、时间长度的中心化交互项；纯 workload 主效应不会进入模型，prompt 或
seed 标识也不会成为特征。这样既能表达同一策略在480p/720p或5s/15s下的不同
风险，又不能靠记住某个场景取巧。九种问题各有一个独立的
贝叶斯逻辑风险头。

这属于 preference-based Bayesian optimization，而不是在 2000 维空间上直接跑
黑盒 Gaussian Process。每轮提案同时考虑后验质量、问题风险、不确定性和批内多样性。

## 4. 40 视频预算

### Round 1：计算交换率识别，10 条

- 480p×10s 与 720p×10s 两个组；
- 每组五个近似等计算量策略；
- 从 Attention-heavy 到 11/12 Actual，再分别强调 causal/bridge 与边界相位；
- 目标是识别同速度下，完整 DiT 次数与 Attention 保真度应怎样交换。

### Round 2：时长外推，10 条

- 使用 Round 1 后验选出的前三个策略族；
- 720p×5s 和 720p×15s；
- 其余名额用于后验不确定性最大的两个反事实策略；
- 目标是校准 Forecast horizon 与长时边缘稳定性。

### Round 3：滑块曲率，10 条

- 在 25/50/75/95 四个加速位置选择最有信息量的比较；
- 检查质量与计算是否随滑块平滑、嵌套和无局部反转；
- 目标是学习预算指数，而不是只让一个端点最优。

### Round 4：发布留出，10 条

- 锁定 `θ` 后不再更新；
- 覆盖 720p 长片、1080p×15s、参考图片/参考音频资源压力与一个未见 seed；
- 只做发布判定，不把结果反向用于挑选同批候选；
- 失败则回到 Round 3 后重新冻结，不能在留出集上临时写规则。

## 5. 发布条件

- `a=0` 严格 Dense；`a=75` 复现 Human 质量拐点；`a=100` 是明确的激进端点；
- 全滑块计算单调、Actual 集合嵌套、共享 Attention 单元保真度嵌套；
- 默认曲线有完整参数摘要、策略摘要、执行 digest 与 Human provenance；
- 关键发布 workload 没有新增的声音、嘴型、因果、结构或边缘稳定性回归；
- 未审核的新算法维度保持关闭，运行时反馈仍为 observe-only。
