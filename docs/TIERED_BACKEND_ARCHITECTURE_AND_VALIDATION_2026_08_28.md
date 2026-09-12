# 三档 H3 后端架构与发布验收（2026-08-28）

## 结论

发布实现不是多份复制代码，也不是在任务中用一串分辨率 `if/else` 临时切换。
它由三个正交维度组成：FL2VA/Ref2VA服务族、INT8/W4A8权重档，以及
24GB/16GB/8GB资源契约。控制台只暴露四个服务族/权重选择，加载前自动把物理显存
映射为五个资源后端之一；十个内部启动身份仅用于隔离服务族与资源契约。底层共享
同一套经过测试的调度器、DiT、
Attention、采样器、VAE和封装实现。任务运行中不跨显存档位，也不静默降分辨率、
时长、步数或加速力度。

当前唯一能力来源是`h3serve/deployment_profiles.py`：

| 资源后端 | 权重 | 首遍生成 | H3二采 | 额外边界 |
|---|---|---|---|---|
| `int8_24gb` | INT8 | 360P–1080P，1–15秒 | 720P/1080P/1440P | 无 |
| `int8_16gb` | INT8 | 360P–1080P，1–15秒 | 720P/1080P/1440P | 1440P×15秒使用全空间时间窗口 |
| `w4a8_24gb` | W4A8 | 360P–1080P，1–15秒 | 720P/1080P | 1080P已通过少步完整资源门 |
| `w4a8_16gb` | W4A8 | 360P–1080P，1–15秒（实验性） | 720P/1080P | 1080P已通过少步完整资源门 |
| `w4a8_8gb` | W4A8 | 360P–720P，1–15秒 | 720P/1080P | 720P×15秒Ref2VA限单参考；1440P拒绝 |

`contract.py`、`models.py`、`session_factory.py`、预检和Web能力响应均从该表派生；
以后修改产品边界只改这一处，并由一致性测试阻止UI、API、加载器各说各话。

## 分层职责

1. **发布能力层**：`deployment_profiles.py`定义不可变资源后端、内部启动身份、权重档、
   几何与二采边界。它不导入CUDA，因此预检和UI可以安全读取。
2. **请求契约层**：`contract.py`解析用户的步数、加速力度、几何与条件素材，并在GPU
   工作开始前拒绝越界请求。
3. **会话隔离层**：`session_factory.py`按启动器装配一个热会话。切换档位或服务族必须
   在队列为空时完整重建；Base/LoRA则是同一会话内的请求级选择。
4. **策略与容量层**：V24连续曲面把`步数 × 加速力度`映射为Actual/Forecast与逐层
   Attention预算；`memory_execution.py`只依据真实token几何、条件前缀和可用显存，
   在whole-query、exact-streaming、compact-streaming中求最低预测延迟的可行图。
5. **物理执行层**：共享的50层H3 DiT、Sage/Sparge Attention、INT8/W4A8线性、
   Block Offload、采样器和VAE执行计划。近似质量由V24计划控制；显存分块本身不删计算。
6. **交付层**：任务队列、取消、checkpoint、latent缓存、二采、AV mux与MP4下载。

该分层使“质量/计算预算”和“显存/物理执行图”彼此正交：同一V24推断能力不会因为
选择16GB或24GB而改变。2026-08-28同种INT8任务的16GB与24GB六组成片SHA-256逐对
完全一致，覆盖FL2VA/Ref2VA的Base、LoRA与720P二采。

## 本轮缺陷与修复

### 1440P二采的K/V布局冲突

请求计划允许HND专用QKNorm/FP8 V快路，但某些1440P cell的实际Attention后端选择NHD，
旧代码因此在运行时抛出`direct HND FP8 V requires HND K/V layout`。现在每个cell先求
物理布局契约：HND保留快路；NHD只关闭HND专用优化并继续走已有NHD实现。短1440P、
1个真实二采步已成功生成MP4，证明该崩溃路径关闭。

### 8GB 720P×15秒最终投影OOM

旧路径完成50个DiT block后，一次性把约十万行hidden转成FP32，末端额外申请约
1.97GiB而越过7.25GiB硬上限。W4A8后端现在将最终RMSNorm、AdaLN与输出Linear按
2048行执行。三项运算均逐行独立，不改变权重、token、采样轨迹或计算量；16/24GB
仍走原始未分块分支。CPU测试证明分块与原路径零容差一致，FL2VA和单图Ref2VA的
720P×15秒首个真实步均在硬8GB契约下通过。

修复前后还分别重跑了相同seed的360P×1秒完整W4A8任务：FL2VA与Ref2VA两组最终
MP4的SHA-256都逐字节一致。这一对GPU非回归证据说明分块没有改变短任务的既有输出。

### 1080P断点预览比例

固定360P预览在1088短边请求上的真实比例是352/1088≈0.324，旧校验却要求至少0.4，
导致Ref2VA 1080P断点在DiT前被拒绝。下限调整为0.3后，24GB Ref2VA 1080P×15秒
的固定360P预览与首个真实步均通过；主任务轨迹未改变。

## 真实GPU快速验收

环境：RTX 4090 SM89、PyTorch 2.13+cu130、CUDA Toolkit 13.3。完整烟测使用少步
高稀疏，以验证端到端能力而不冒充质量/吞吐基准：Base 5步/加速95，LoRA 4步/加速95，
二采1步/加速95。

| 启动器 | Base完整MP4 | LoRA完整MP4 | 二采完整MP4 |
|---|---:|---:|---:|
| 24GB FL2VA INT8 | 7.053s | 5.604s | 720P 5.269s |
| 24GB Ref2VA INT8 | 9.603s | 5.445s | 720P 7.763s |
| 16GB FL2VA INT8 | 5.937s | 5.378s | 720P 4.372s |
| 16GB Ref2VA INT8 | 8.964s | 5.399s | 720P 7.592s |
| 8GB FL2VA W4A8 | 4.789s | 3.845s | 1080P×15秒 216.938s |
| 8GB Ref2VA W4A8 | 7.856s | 3.862s | 1080P×15秒单图 235.000s |

24GB FL2VA另完成一次1440P、1步二采，耗时14.067秒。这里源片只有360P×1秒，
该项目只验证1440P代码路径、layout与MP4交付，不替代既有1440P×15秒分窗成片证据。

最高首遍几何使用5步/加速95并在第1个Actual后checkpoint，不等待整片：

| 启动器 | 边界 | 状态 | 墙钟时间 |
|---|---|---|---:|
| 24GB FL2VA | 1080P×15秒 | checkpointed | 41.578s |
| 24GB Ref2VA | 1080P×15秒、单图 | checkpointed | 90.837s |
| 16GB FL2VA | 720P×15秒 | checkpointed | 19.829s |
| 16GB Ref2VA | 720P×15秒、单图 | checkpointed | 58.905s |
| 8GB FL2VA | 720P×15秒 | checkpointed | 26.109s |
| 8GB Ref2VA | 720P×15秒、单图 | checkpointed | 74.996s |

Ref2VA的边界墙钟包含固定360P、4步预览分支，因此不能拿该数字当单个1080P或720P
DiT步耗时。以上checkpoint是容量和执行路径门，不是完整长片质量或完整耗时声明。

原始机器可读回执及MP4位于：

- `runtime/validation/tiered_backend_20260828/tiered_backend_gpu_matrix.json`
- `runtime/validation/tiered_backend_20260828/ref24/`
- `runtime/validation/tiered_backend_20260828/int8_16gb/`
- `runtime/validation/tiered_backend_20260828/w4a8_8gb/`
- `runtime/validation/tiered_backend_20260828/2k_layout_regression/`
- `runtime/validation/tiered_backend_20260828/boundary_checkpoints_fixed/`

可复现命令由`scripts/validate_tiered_backends.py`提供。它只通过公开API工作，不直接
调用内部模型，因此同时覆盖启动器切换、请求契约、队列、热会话、DiT、VAE、封装和下载。

同日完整自动回归在发布Python与已安装SM89 Sparge扩展环境中运行696项，全部通过，
其中4项条件门按设计跳过。严格的direct-HND FP8 writer对Sparge ABI字节一致性CUDA测试
也实际执行并通过，而不是因扩展搜索路径缺失被误报为后端失败。

真实取消恢复烟测另提交8GB FL2VA 720P×15秒、5步/加速95任务：DELETE请求立即把
运行态卡片标为`cancelling`，后端在安全边界终止为`cancelled`；同一热会话随后用
5步/加速95完成360P×1秒MP4（5.584秒），GPU回到0%利用率与552MiB物理占用。
恢复回执位于`runtime/validation/tiered_backend_20260828/cancel_recovery/`。

条件输入也通过真实端到端烟测（24GB INT8、360P×1秒、Base 5步/加速95）：首帧
9.685秒、尾帧7.936秒、首尾帧8.534秒、Ref2VA独立参考音频6.731秒、Ref2VA参考
视频14.567秒，五项均生成H.264/AAC MP4。另一个任务在第1步于3.316秒正确进入
`checkpointed`，随后从同一状态恢复并在总计7.222秒成功交付，证明正式断点不是
只停不续的展示接口。

## 主张边界

- 本轮证明六启动器可加载，Base/LoRA可完整出片，INT8与W4A8二采可完整出片，六个最大首遍
  几何可完成真实DiT步；没有用checkpoint冒充完整15秒长片。
- 少步高稀疏只用于快速功能验收，不代表发布默认质量。V24的质量结论仍来自既有Human
  A/B；本轮没有重新声称W4A8与INT8画质等价。
- 16/24GB相同输出哈希证明资源执行图没有改变这六个短任务的推理结果；W4A8是不同
  权重量化档，不能与INT8做字节等价声明。
- 8GB发布前仍建议在物理8GB SM89设备做一次安装与完整720P×15秒烟测；当前证据是在
  RTX 4090上用7.25GiB PyTorch硬分配上限获得。
