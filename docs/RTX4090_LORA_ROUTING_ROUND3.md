# RTX 4090 LoRA路线：第三轮实测路由

日期：2026-08-11  
权重：Larry step600 EMA，strength 1.0  
质量契约：6个完整Turbo实际步，不使用预测步

## 发布机械策略

当前在360p/5s至720p/15s的实测包络内统一使用：

`2-Block双缓冲 + prefetch depth 1 + MLP chunk 8192 + Video-VAE tile 288`

这不是把360p、480p、720p拍脑袋拆成三套代码。四个锚点表现为同一机制聚类：
权重复制被当前Block计算隐藏，整DiT H2D被消除，显存随激活规模平滑增长。

## 四个真实锚点

下表端到端为完整MP4生成；360p/480p/720p5的“热态”去除了实测首次同prompt
编码，720p15同样列出归一化热态。服务启动约59--61秒只发生在进程建立时。

| 负载 | 完整首次任务 | 同prompt热态 | DiT | Video VAE | 峰值CUDA allocated |
|---|---:|---:|---:|---:|---:|
| 640x352x124 | 16.985 s | 15.341 s | 8.799 s | 4.561 s | 5.809 GiB |
| 864x480x124 | 28.789 s | 26.923 s | 17.860 s | 6.225 s | 6.603 GiB |
| 1280x736x124 | 71.628 s | 69.979 s | 52.577 s | 13.553 s | 8.816 GiB |
| 1280x736x362 | 316.089 s | 314.290 s | 264.265 s | 40.856 s | 16.404 GiB |

720p15交付时长为15.084秒。抽取12帧覆盖全时序通过视觉门控，H.264与32 kHz
AAC双声道正常。视频SHA-256：

`5a72c22f07498339ae3411e244eef4bcf40b33731337caf73cfa4a4e37b06f2f`

## Resident/Block消融

同seed、同tile、同模型数学路径：

| 负载 | Block | Resident | 峰值Block | 峰值Resident | 等价性 |
|---|---:|---:|---:|---:|---|
| 360p5热态归一 | 15.341 s | 15.506 s | 5.809 GiB | 21.482 GiB | MP4 SHA相同 |
| 480p5首次任务 | 28.789 s | 29.245 s | 6.603 GiB | 22.279 GiB | MP4 SHA相同 |

因此Block不仅节约约15--16 GiB，在这两个产品锚点还略快。720p Resident会进入
约24 GiB占用、低功率长时间运行的病态区，禁止作为720p发布路线。

## VAE tile 256/288消融

720p5同seed下，tile 288把Video VAE从16.762秒降到13.553秒，约1.237x；峰值
仍为8.816 GiB。12帧视觉未见接缝或污染，视频SSIM为0.977398（仅诊断），音频
逐样本一致。历史tile 384已因纹理与分块污染在视觉门控被拒绝；局部
`torch.compile`只有约0.23秒、不具稳定发布价值，也不采用。

## 路由模型与闭环

路由使用packed token、空间token、latent帧和output pixel-frame共同约束，不按
“720p”标签猜测。LoRA延迟曲线由四个热锚点拟合，显存线被上移到不低估任何锚点。
超过空间920、latent 107、packed 105000或pixel-frame 342M会fail closed。

真实`--auto-route` 360p5请求选择了
`sm89_lora6_block_360p_720p15_r1`，预测与实测峰值均为5.809 GiB，输出MP4与显式
Block完全相同。这验证了分析器、路由器、Block执行和VAE tile的完整链路。

## 不推广的实验

- Hybrid驻留32层在720p5为76.204秒、18.511 GiB，慢于纯Block且多占约9.7 GiB；
- 720p全Resident发生低功率病态运行；
- tile 384视觉失败；
- 4步和8步属于不同质量预设，必须另做质量与延迟校准，不能套用本6步表。

## 证据位置

- 360p策略A/B与自动路由：`runtime/calibration/lora_360p_route/`
- 480p策略A/B：`runtime/calibration/lora_480p_route/`
- 720p5 tile A/B：`runtime/calibration/lora_vae_tile288_720p/`
- 720p15：`runtime/calibration/lora_block_720p15s/`
- 路由代码：`h3serve/native_engine/planner/calibrated.py`

