# RTX 4090 H3工作负载路由：第二轮Block Offload真实实验

日期：2026-08-11  
GPU：NVIDIA RTX 4090 24 GB，SM89  
状态：原始权重路线首个可执行路由表的证据；LoRA尚未纳入该表

## 1. 本轮不是只做分析

本轮完成了真实代码、真实21 GB Pruned INT8 ConvRot DiT、真实SM89 CUDA
kernel、真实视频/音频VAE和真实MP4封装。主要新增：

- 请求级`ExecutionPlan`，不再靠环境变量切换MLP chunk；
- `ImmutablePinnedModuleResidency`的权重分区驻留；
- 50层H3 Block的两个预分配GPU Buffer；
- 下一Block H2D与当前Block计算重叠；
- 原始权重实际步/预测步的分段Block执行；
- 经过实测约束的fail-closed RTX4090路由表。

完整CPU回归为61项通过、4项按设计跳过。

## 2. 异步错误、视觉门控和根因消融

第一次GPU双缓冲实现虽然生成了MP4，峰值只有约5 GiB，但多帧视觉检查出现
灾难性分块彩色噪声，因此立即判定失败，没有把速度和显存作为可发布成绩。

消融结果：

| 320x192、22帧、1实际步 | DiT单步 | 峰值显存 | 视觉/等价性 |
|---|---:|---:|---|
| 全模型驻留 | 0.314 s | 19.765 GiB | 正常参考 |
| Block串行复制 | 1.651 s | 4.990 GiB | 与参考MP4 SHA-256完全相同 |
| 初版独立compute stream | 1.047 s | 4.998 GiB | 彩色分块噪声，FAIL |
| 修复后caller stream计算+异步copy | 1.126 s | 4.990 GiB | 与参考MP4 SHA-256完全相同 |

根因不是权重格式或Block装配，而是第三方SM89扩展与自建compute stream之间的
执行顺序不可靠。最终实现把模型计算固定在进入请求的caller stream，仅在独立
copy stream预取下一Block；跨Block range时，copy stream显式等待上一range的
caller-stream读取结束。

固定seed的全驻留、串行Block、修复后重叠Block三条MP4 SHA-256均为：

`43d91bc76d77816dcd086c377b814011bc18fa1b9053a1e662b055278f0e918a`

另一个2步、预测控制器分段执行对照中，Block与全驻留MP4 SHA-256也完全相同：

`26c6c3197fed8ccf95002928ac3273b9a02825f889db7e2b9bc29e6fe0136783`

## 3. 720p/5秒完整结果

配置：原始权重、1280x736计算画布、124帧、20调度点、9实际/11预测、
MLP chunk 8192、两个Block Buffer、prefetch depth 1。

| 指标 | 实测 |
|---|---:|
| 端到端 | 101.371 s |
| DiT去噪 | 78.316 s |
| 视频VAE | 17.046 s |
| Mux | 2.737 s |
| 峰值CUDA allocated | 8.821 GiB |
| 实际步中位数（约） | 7.98 s |
| 预测步中位数（约） | 0.56 s |

全部9个实际步和11个预测步按计划执行。抽取12帧覆盖全时序检查，未见彩色色斑、
散光重影或结构崩坏。交付文件含H.264视频、AAC 32 kHz双声道音频。

严格同prompt、seed、画布、调度和9/11步的Comfy V7-D等价融合为110.970秒，
Native为101.371秒，严格端到端加速`1.0947x`。详见
`docs/RTX4090_STRICT_COMFY_AB.md`。

## 4. 720p/15秒边界结果

配置同上，帧数改为362（交付15.084秒）。

| 指标 | 实测 |
|---|---:|
| 端到端 | 468.186 s |
| DiT去噪 | 406.453 s |
| 视频VAE | 50.931 s |
| Mux | 7.685 s |
| 峰值CUDA allocated | 16.415 GiB |
| 实际步中位数（约） | 41.84 s |
| 预测步中位数（约） | 2.68 s |

采样中GPU持续约100%利用、约466--480 W；没有出现历史低功率卡死。12帧视觉门控
未见彩色色斑、重影或结构崩坏，H.264/AAC媒体结构正常。文件SHA-256：

`b2d45eaad1825a3de8b40f945c974f2b1f56973fd76216fdfbe907eee82e5621`

历史原V7-D为516.505秒、此前发布优化版为499.996秒；本次分别约1.103x和1.068x。
同样，这两项是历史同档工程比较，不冒充同seed重复A/B。

## 5. 路由表为何这样划分

第一版只启用两个原始权重执行聚类：

1. `resident_short`：空间Patch不超过405、latent时间不超过37、packed Token不超过
   17,000、输出pixel-frame不超过60M。对应已经完成的360p/5秒、480p/3秒和
   480p/5秒附近；采用整DiT阶段驻留和MLP chunk 8192。
2. `block_large`：上限为空间Patch 920、latent时间107、packed Token 105,000、
   输出pixel-frame 342M。覆盖到实测720p/15秒；采用两个Block Buffer、异步
   预取和MLP chunk 8192。

路由器同时使用`packed_tokens`、`spatial_tokens`、`latent_frames`和
`output_pixel_frames`。这保留了第一轮中“360p/15秒正常而近Token的480p/8秒
病态”的反例，不会退化为单标量拍脑袋路由。

超出720p/15秒验证边界的请求不会外推猜测，而是`NoFeasibleProfile`；LoRA不借用
原始权重表，其后续独立矩阵与路由见`RTX4090_LORA_ROUTING_ROUND3.md`。

## 6. 当前限制和下一步

- 720p5严格Comfy A/B已完成；720p15仍只有历史同档工程比较；
- 首帧、尾帧、首尾帧会增加条件Token并触发VAE编码，尚未加入本轮720p矩阵；
- LoRA 6步已在第三轮独立校准；4/5/8步仍会fail closed，不能套用6步延迟模型；
- 480p/8秒全驻留病态点仍需逐kernel Profile；Block模式可作为安全回退，但尚未
  对该精确形状完成同seed质量/速度对照；
- 当前路由表已写入代码，但服务入口还需把请求特征分析、显存查询和Plan选择接到
  正式`NativeH3Engine`生命周期。

## 7. 证据位置

- 路由实现：`h3serve/native_engine/planner/`
- Block实现：`h3serve/native_engine/model/block_offload.py`
- 驻留实现：`h3serve/native_engine/runtime/residency.py`
- 流修复：`h3serve/native_engine/runtime/streams.py`
- 720p/5秒：`runtime/calibration/block_offload_720p/`
- 720p/15秒：`runtime/calibration/block_offload_720p15s/`
- 等价性消融：`runtime/calibration/block_offload_smoke/`
