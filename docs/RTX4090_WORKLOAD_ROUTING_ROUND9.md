# RTX 4090 H3 全任务域优化：Round 9

日期：2026-08-12  
范围：360p/480p/720p、短/长视频；高分辨率长视频为本轮重点，但不是唯一目标。  
质量协议：复杂多台词 + 大动作；Whisper 时间戳、6帧视觉检查、Human 连续播放终审；SSIM不作硬门。

## 1. 本轮结论

1. 所有档位共享的Video-VAE FFN局部编译可提供约2%的VAE热态收益，峰值显存不变；服务启动预热约0.81秒，避免首个请求承担JIT。该优化已接入生产`NativeSessionFactory`。
2. 空间tile批量解码没有收益，batch=2仅在480p出现1.5%的单次波动，在720p持平或变慢；保留实验能力但生产默认仍为batch=1。
3. 当前发行CUDA ConvRot INT8 kernel在全部H3真实维度上明显快于Triton，不能按任务尺寸切换到Triton。
4. 当前稠密SageAttention的量化粒度/平滑变体没有实质收益；长序列的大幅收益只能来自已经通过质量协议的结构化稀疏，而不是换一个Sage参数。
5. LoRA6、复杂720p×15秒、完整六步下，固定Sparge topk=0.50将端到端从315.319秒降到241.194秒，`1.307x`；初步视觉门和Whisper均未比稠密参考退化，但仍等待Human连续播放终审。
6. 原始权重9/11与topk=0.50虽然在复杂480p×10秒达到`1.129x`，但第三句台词明显退化，必须隔离；topk=0.75保住台词，但该尺寸只有`1.041x`，不适合短中任务默认启用。

## 2. 长任务一致稀疏Attention

### 2.1 LoRA6复杂720p×15秒

固定Larry step600 EMA、6个真实Turbo步、Block双缓冲、prefetch 1、MLP chunk 8192、VAE tile 288、同prompt/seed：

| Attention | 端到端 | DiT | Video-VAE | 峰值allocated | 相对稠密 |
|---|---:|---:|---:|---:|---:|
| 稠密Sage | 315.319 s | 264.546 s | 40.725 s | 16.404 GiB | 1.000x |
| Sparge 0.75 | 287.122 s | 234.805 s | 40.720 s | 18.502 GiB | 1.098x |
| Sparge 0.50 | 241.194 s | 189.569 s | 40.758 s | 18.502 GiB | **1.307x** |

topk=0.50相对稠密节省74.125秒；DiT为`1.396x`。其六个真实步分别为32.47、31.37、31.40、31.43、31.44、31.46秒，没有非有限输出。

Whisper large-v3 CPU/int8结果：

| 路线 | 全局CER | 三句顺序/时间窗 | 观察 |
|---|---:|---|---|
| 稠密 | 0.0435 | PASS | “北巷”识别为“北向” |
| Sparge 0.75 | 0.0435 | PASS | “北巷”识别为“北上” |
| Sparge 0.50 | 0.0435 | PASS | “北巷”识别为“北疆” |

三者的差异都是同一近音词位置，不能据此判定稀疏音频更差。topk=0.50的6帧初检未见色斑、整帧崩坏、VAE接缝或持续重影；人物、地图、头盔、火把及最后台阶构图均存在。连续运动、闪烁、口型仍由Human播放终审。

### 2.2 为什么不采用跨步稀疏/稠密混合

同一复杂720p×15秒任务中，稀疏中间步、稠密首尾步的候选发生非有限输出：topk=0.60在第6个采样调用（zero-based step 5）出现非有限video DiT prediction。跨步改变Attention近似会破坏轨迹一致性，已隔离；固定稀疏度贯穿全部采样步反而稳定。

## 3. 原始权重路线不能照搬LoRA结论

固定原始权重、9真实/11预测、复杂480p×10秒、同prompt/seed：

| Attention | 端到端 | DiT | 相对稠密 | Whisper CER |
|---|---:|---:|---:|---:|
| 稠密Sage | 81.628 s | 64.089 s | 1.000x | **0.000** |
| Sparge 0.75 | 78.425 s | 60.917 s | 1.041x | **0.000** |
| Sparge 0.50 | 72.300 s | 54.599 s | 1.129x | **0.185** |

topk=0.50前两句正确，第三句“好，所有人守住北门”不再完整可识别。即使6帧无灾难视觉错误，也不能晋级。topk=0.75保住三句台词，但当前任务收益太小；只有更长序列可能使其具有产品价值，须在720p×10/15秒上另测，不能外推。

## 4. 全任务共享Video-VAE优化

### 4.1 失败的tile batching

真实FP16 VAE、相同随机latent、tile 288：

| 任务 | batch 1 | batch 2 | batch 3 | batch 4 |
|---|---:|---:|---:|---:|
| 864×480×124帧 | 6.090 s | 6.001 s | 6.350 s | 6.705 s |
| 1280×736×124帧 | 13.566 s | 13.616 s | 14.666 s | 14.951 s |

批量化未提高4090利用率，反而使大尺寸变慢。默认保持串行tile；该失败避免了“用更多显存就一定更快”的错误假设。

### 4.2 通过的FFN区域编译

只编译ViT decoder的`Linear -> SiLU×gate -> Linear`区域，不捕获tile循环、RoPE或整个decoder：

| 任务 | eager热态中位数 | compiled热态中位数 | VAE提速 |
|---|---:|---:|---:|
| 480p×5秒 | 6.009 s | 5.876 s | 1.023x |
| 720p×5秒 | 13.628 s | 13.352 s | 1.021x |

数值抽样：mean abs `0.000215`、max abs `0.002417`、cosine `0.99999964`；峰值allocated不变。复杂480p×10秒成片的6帧初检正常，音频PCM SHA-256与eager参考完全一致。

生产实现用7个时间latent token（5主体+2 overlap）、18×18空间latent预热固定288px tile图。真实`NativeSessionFactory.build("lora")`结果：启动60.591秒，其中编译预热0.814秒，36个FFN模块成功启用；后续480p解码5.930秒且全finite。

曾测试RoPE的`reduce-overhead`编译，但重复tile调用触发CUDA Graph输出覆盖保护错误；该路线已隔离，不能启用。

## 5. 公共内核筛选

8192 rows、真实H3 ConvRot维度的CUDA/Triton中位数：

| Linear | CUDA | Triton | CUDA领先 |
|---|---:|---:|---:|
| QKV 5376→21504 | 3.276 ms | 3.618 ms | 1.104x |
| Attention out 7168→5376 | 1.280 ms | 1.566 ms | 1.223x |
| MLP FC1 5376→28672 | 4.237 ms | 4.810 ms | 1.135x |
| MLP FC2 SwiGLU 28672→5376 | 2.655 ms | 4.589 ms | 1.728x |

因此当前CUDA后端继续锁定。Sage dense变体在34.5k token最快仅`1.016x`，在100k token最快仅`1.006x`且会改变输出；不采用。

## 6. 路由决策

当前可落地的路由原则不是“只优化720p×15秒”，而是两层：

1. **全任务公共层**：CUDA ConvRot INT8、Sage dense、Block双缓冲、MLP 8192、VAE tile 288、预热后的VAE FFN compile。360p到720p、文本和首/尾帧任务都能复用。
2. **长序列候选层**：根据真实packed token而不是营销分辨率选择Attention预算。LoRA高负载可候选topk=0.50；原始权重只允许更保守的0.75候选。该层属于近似质量策略，必须受用户质量preset授权且通过Human终审，不能悄悄用于“超高质量”。

尚不能写死最终token阈值。当前证据显示约30k token的480p×10秒收益仅6%（LoRA）或4%（原始0.75），约100k token的720p×15秒LoRA收益31%。下一轮必须补720p×10秒或480p×15秒交叉点、多seed和首尾帧，拟合收益/风险后才能发布阈值。

## 7. 外部证据边界

- MiniMax MSA官方kernel当前面向SM100/Blackwell，不可直接用于RTX 4090 SM89：<https://github.com/MiniMax-AI/MSA>
- SVG-EAR提供训练免费centroid compensation与error-aware routing，但官方实现和证据不是H3/SM89，移植前需要独立kernel原型：<https://arxiv.org/abs/2603.08982>、<https://github.com/dyxg/SVG-EAR>
- LVSA的长视频结构稀疏使用旋转全局anchor，但官方实验不是H3/4090：<https://arxiv.org/abs/2605.31057>
- LightX2V提供H3、offload、compile及多种Attention的系统参考，但没有公布H3/4090同契约数据：<https://github.com/modeltc/lightx2v>

这些资料只支撑路线选择，不替代本报告的4090实测。

## 8. 证据入口

- dense/0.75/0.50复杂720p15：`runtime/calibration/workload_routing_round9/{dense,full_sparse075,full_sparse050_complex}/`
- 原始9/11稀疏A/B：`runtime/calibration/workload_routing_round9/original911_sparse{050,075}_complex_480p10/`
- Whisper：`runtime/calibration/workload_routing_round9/quality/whisper/`
- 联系表：`runtime/calibration/workload_routing_round9/quality/*contact.jpg`
- VAE扫描：`runtime/calibration/workload_routing_round9/vae_*json`
- INT8/Sage微基准：`runtime/calibration/workload_routing_round9/{int8_convrot_cuda_triton_sm89,sage_variants_*}.json`

