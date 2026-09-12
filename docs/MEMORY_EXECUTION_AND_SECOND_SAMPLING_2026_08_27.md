# 三套隔离资源后端与 H3 二次采样

## 发布接口

新任务不再暴露`performance/low_vram`。用户只设置采样步数与加速力度。启动器先固定
24GB、16GB或8GB资源后端；档位之间不做任务级自动切换。每个后端再把V24质量计划
投影到本档显存预算内实测最快的可行执行图。旧任务里的
`memory_mode=auto|performance|low_vram`仍可反序列化，但不会改变当前后端。

当前资源档位：

- `int8_24gb`：首遍360P至1080P全部比例与1–15秒；二采720P/1080P/1440P。
- `int8_16gb`：首遍360P至1080P全部比例与1–15秒；二采720P/1080P/1440P。
- `w4a8_8gb`：FL2VA/Ref2VA首遍最高720P×15秒；二采720P/1080P、1–15秒；1440P拒绝；首遍极限下Ref2VA为单参考发布边界。

## 规划目标与执行图

规划器使用输出几何、H3 latent网格、精确packed-token数和运行时可用显存，不读取
提示词语义，也不针对具体测试场景写分辨率名称分支。目标是：

```text
minimize predicted latency(graph)
subject to predicted DiT peak <= budget
           predicted VAE peak <= budget
           weights / steps / V24 schedule unchanged
```

DiT与VAE顺序执行且中间卸载，因此端到端峰值取两阶段最大值而不是相加。INT8物理
A/B中完整上下文Query streaming为39.624秒/稠密步，whole Query为39.683秒/步，
因此前者是24/16GB默认而非容量降级。候选图继续包含紧凑全上下文K/V、双/单block buffer、
8192投影块、4096 MLP块与长序列per-warp稳定性保护。8GB W4A8在相同优化器中使用
4096 Query/投影块、2048 MLP块和必要时的单block buffer，为真实8GB CUDA上下文保留
余量。紧凑K/V属于近似数值路线并在
回执标为`bit_exact=false`；它不删上下文、不改采样步或稀疏预算。

## 精确低显存 Video-VAE

长视频原实现会把所有时间段拼成完整FP32 GPU视频，再分块转uint8。新实现只在因果
overlap完成后把已定稿时间段做相同FP32像素变换并逐字节写到CPU uint8输出，因而删掉
随分辨率×时长增长的完整GPU拼接张量。空间tile仍保持实测更快的288，不再用更慢的
256伪装成“省显存”。

同一1080P×15秒latent、同一权重、同一288 tile的物理A/B：

| VAE图 | 热解码 | 峰值allocated | 最终uint8 SHA-256 |
|---|---:|---:|---|
| 原完整FP32拼接 | 82.782秒 | 16.952GiB | `a5704e...08b8` |
| 精确时间流式、硬16GiB | 82.583秒 | 8.500GiB | `a5704e...08b8` |

输出哈希完全一致；显存减少8.452GiB且没有速度税。2K×15秒在硬23GiB限制下也完成
完整解码：154.772秒、峰值11.304GiB、输出4,003,430,400字节。

原始证据：

- `runtime/calibration/v19_long_video_20260825/video_vae_long_decode_1080p15.json`
- `runtime/calibration/unified_vram_backend_20260827/vae_physical/`

## 产品矩阵与实机门

静态容量矩阵覆盖三套显存档、全部五种比例、1/5/10/15秒和各档发布条件，共680项，
结果680/680通过。INT8要求Ref2VA十五项静态条件；8GB W4A8要求单参考，十五项条件
作为明确可能拒绝的诊断边界保留。最坏要求项：

| 档位 | 任务 | 最坏预测峰值 | DiT路线 | VAE路线 |
|---|---|---:|---|---|
| 16GB | 720P首遍 | 12.936GiB | exact streaming | 按预算选择 |
| 16GB | 1080P二采 | 15.012GiB | compact/single-buffer | exact host temporal |
| 24GB | 1080P二采 | 19.409GiB | exact streaming | 按预算选择 |
| 24GB | 1440P二采 | 22.912GiB | compact | exact host temporal |
| 8GB | 720P首遍（单参考） | 约7.123GiB | compact/single-buffer | 按预算选择 |
| 8GB | 1080P×15秒二采 | ≤6.5GiB/DiT窗 | compact/single-buffer，4个全空间时间窗 | 12-block尾部流式、8-tile batch |

W4A8额外使用同一H3拓扑的group-16 packed 4-bit ConvRot权重和运行时INT8激活。
在RTX 4090上把PyTorch分配额度硬限制为7.25GiB后，FL2VA 720P×15秒、5步、95加速
完整生成成功：121.458秒，峰值allocated/reserved为6.9793/7.0879GiB，NVML观测峰值
约7460MiB。Ref2VA单参考图Base与LoRA也在同一硬上限下分别完成真实MP4。
完整8GB实现、数值边界与证据见
[`W4A8_8GB_BACKEND_2026_08_27.md`](W4A8_8GB_BACKEND_2026_08_27.md)。

快速物理烟测遵循5步、95加速，不跑20步稠密任务。硬16GiB下720P×5秒完整MP4：

- V24实际保留3个Actual与2个Forecast；
- 热态端到端33.625秒，DiT 15.422秒，Video-VAE 12.813秒；
- 峰值allocated/reserved为9.147/11.148GiB；
- 平均/峰值功耗392.99/479.50W；
- 输出位于`runtime/calibration/unified_vram_backend_20260827/physical_16gb/`。

2K×15秒DiT已有首个Actual物理门：compact K/V、q8192、峰值allocated 22.330GiB；
它只证明最重DiT图可执行，不冒充完整长片耗时。2026-08-27首个真实多步任务暴露了
持续13–16.5GB/s PCIe换页与110–148W低功耗，因此2K首遍生成已从发布矩阵撤回。
完整2K VAE门和单Actual门仅作为二采分窗执行器的容量证据。完整报告位于
`runtime/calibration/memory_execution_20260827/e2e_2k15_gate/`。

2026-08-28原生136/17时间窗执行器已完成首个端到端门：480P×15秒clean latent
二采至2560×1440，1真实步、75加速、0.20重绘，热后端287.931秒，产出362帧
H.264/AAC MP4。DiT窗口连续473–482W、SM 99–100%，没有旧整段路线的持续PCIe
换页；音频latent完全保留，三个视频latent窗口交叉淡化后只解码一次。证据位于
`runtime/calibration/memory_execution_20260827/second_sampling_480p15_to_2k15_window_r2/`。

超长参考视频会显著增加packed前缀。当前额外35项video-heavy诊断有4项被预算器
明确拒绝；它们不在本轮图片/音频/文本条件发布承诺内，也不会静默降低分辨率、步数
或质量。后续需要条件前缀流式执行再扩展该边界。

## H3 原生二次采样

旧FlashVSR超分与一次性LoRA预览已从新任务入口移除。二采流程为：

1. 低分辨率任务完成并保存干净Video+Audio latent；
2. 从成功任务卡选择720P、1080P或1440P目标；
3. 复用原提示词与参考图片/视频/音频，以H3专用学习式3D BF16网络空间放大Video latent；
4. 按重绘强度重新加噪并执行1–8个真实H3 DiT二采步；
5. Forecast关闭；加速力度只调度Attention；默认复用首遍Audio latent；
6. 统一显存预算器自动选择执行图，整幅能放下时不做有冗余的时间切片；硬8GB长
   1080P按相位对齐时间窗执行DiT，并在拼接后只解码一次。

完成首遍生成时，latent checkpoint还会保存该任务已经算出的精确Qwen条件张量和
内容指纹。二采在提示词、服务族、参考素材及其预处理契约一致时直接复用这份CPU
缓存，不重新执行50层Qwen；时间窗口共享同一份条件，不按窗口重复编码。指纹按素材
内容而不是路径计算：纯文字及Ref2VA参考图/参考音频可跨目标分辨率和时长复用，
首尾帧与参考视频则保留几何/时间约束。旧latent、缓存损坏或指纹不匹配时自动回退到
正常Qwen编码，并把新缓存写入二采结果。该优化不近似、不量化条件张量，也不改变
DiT输入；回执中的`qwen_conditioning_cache.status`可审计`encoded`、
`hot_session_hit`或`checkpoint_hit`。

H3的24通道Video latent不是可直接当图像特征插值的表示。旧实验路径对其做
Bicubic/Bilinear放大会产生周期性重影和多重曝光，现已从跨分辨率二采契约中删除；
服务只接受`learned_3d`初始化。放大器与DiT分阶段驻留，避免两套权重同时占用显存。

二次采样对INT8与W4A8启动器开放。W4A8响应声明720P/1080P，API在开始GPU工作前
拒绝1440P或跨服务族latent；FL2VA与Ref2VA源latent不能互换。硬7.25GiB物理门已覆盖
FL2VA 1080P×15秒、Ref2VA图片/音频条件、同一热会话连续任务和真实取消回收。

2026-08-29将16GB的1440P入口从内部实验提升为发布能力。最重门使用Ref2VA、1张
参考图、1段参考音频、480P×15秒源片和2560×1440×362帧目标，在15.25GiB硬预算下
完成3个全空间时间窗口、1个真实SA Solver二采步及最终AV封装：二采256.542秒，
峰值分配13.6587GiB、峰值预留14.6172GiB，输出为24fps、362帧和32kHz双声道。
证据位于`/root/h3-new-serve-runtime/runtime/validation/int8_16gb_1440p15_release_gate_r1/`。

API不再提交显存模式：

```http
POST /api/v1/jobs/{source_job_id}/second-sampling
Content-Type: application/json

{"resolution":"1080p","steps":1,"acceleration":75,"denoise":0.20}
```

本轮针对性代码回归为101项通过，三档容量矩阵680/680通过。容量矩阵见
`runtime/calibration/isolated_resource_backends_20260827/PRODUCT_ENVELOPE.md`。
