# 64GB 发布档验证（2026-08-14）

## 结论

64GB 档不需要降低采样步数、注意力精度或模型权重。主要固定开销来自 WSL
环境中反复从 `/mnt/c`（DrvFS/9p）流式读取约 15GB 的 Qwen 权重。发行版现在会在
需要时创建经过 SHA-256 校验的用户级 Linux 原生磁盘副本，并在编码结束后主动
释放其可回收页缓存。2026-08-16又加入按实际执行顺序排列的50层磁盘缓存和有界
双层pinned流水；四条引擎路线共用同一个会话工厂，因此该机制同时覆盖
高保真、Turbo、多参考和多参考 Turbo。

## 等价 A/B

同一 Ref2VA 多图、多音频输入、相同提示词和权重：

| Qwen 存储 | 视觉准备 | 50 层 Qwen | 条件阶段总计 |
|---|---:|---:|---:|
| Linux 原生 ext4 缓存 | 4.380 s | 9.556 s | 13.936 s |
| WSL `/mnt/c` 源文件 | 7.176 s | 45.842 s | 53.018 s |

两次输出嵌入校验值完全相同：`sum=189768.49150629947`，
`abs_sum=17147404.416318007`。Qwen GPU 峰值均为 2.275 GiB；进程最大 RSS
约为 15.864 GiB。该优化仅改变字节读取位置，不改变计算图和浮点运算顺序。

## 完整任务验证

新增流水使用同一480p×5.17秒、12实际/8预测任务进行严格A/B：

| 路径 | 文本编码 | 完整任务 |
|---|---:|---:|
| 原compact原生文件流式读取 | 47.767秒 | 89.453秒 |
| 执行顺序层缓存＋双层流水 | 8.312秒 | 49.804秒 |

文本阶段提升5.75倍，完整任务提升1.80倍。两次解码后视频SHA-256均为
`862b734625cda98ed86c65601c147b6d4d13652c51f275a798810fb966049b99`，音频
SHA-256均为`8d5551db9704a643f5f960993366244423c2e761be621978b812eb5c1d2fa6d5`。
因此该优化没有改变Qwen张量、H3采样或最终音视频数据。

- 内存策略：`compact`（64GB 兼容）
- 引擎：Turbo，6 步
- 任务：复杂台词与动作，640×352，124 帧（约 5.17 秒）
- 热态完整任务：23.432 秒
- 同步进程 PSS 峰值：49.096 GiB
- Swap：0
- 输出：H.264 640×352 + AAC 32 kHz stereo
- 视觉门控：抽帧未见彩色色斑、VAE 拼接缝、结构崩塌或明显重影

随后又在 systemd cgroup `MemoryMax=58GiB`、`MemorySwapMax=0` 的硬约束下重跑
完全相同任务。服务正确自动选择 compact，模型启动为 64.818 秒，健康检查返回
`qwen_storage=native_cache`；完整生成耗时 21.597 秒。受限版与上一次未限额版的
MP4 SHA-256 均为
`4c31aad2d3edfa51d9929c971cecbc0764fa1dfe42f83cc792f70ca8f45c5d91`，
即容器字节级一致。生成完成后 cgroup 内存为约 38.02GiB，整个任务始终受 58GiB
硬上限约束且 Swap 被禁止。受限成片为
`output/b00d1714-63ec-4c01-9aba-8b8a7dbe2f16.mp4`，抽帧证据位于
`runtime/validation/memory/compact_58g_cgroup_gate/contact.png`。

内存证据保存在
`runtime/validation/memory/compact_ext4_qwen_full_job_v1.json`。此前的
720p×15 秒 compact 实测峰值为 48.795 GiB；完整生成、1080p 超分、恢复 H3
链路峰值为 51.745 GiB，均未使用 Swap。

## 发行行为与边界

- 默认缓存：`~/.cache/h3serve/checkpoints`
- 首次复制：原子写入并校验发行清单 SHA-256；中断文件不会被采用
- 磁盘要求：建议约45GB可用Linux磁盘（约15GB原生副本、约13GB层缓存和构建余量）
- 自定义：`H3_SERVE_LOCAL_MODEL_CACHE=/linux/path`
- 禁用：`H3_SERVE_LOCALIZE_QWEN=0`
- 仅禁用按层缓存：`H3_SERVE_QWEN_LAYER_CACHE=0`
- 如果磁盘不足或缓存目录仍位于 `/mnt/c`，服务会安全回退到原文件并给出警告；
  输出能力不受影响，但新提示词延迟会恢复到跨盘水平
- `/healthz` 的 `warm_state.qwen_storage` 为 `native_cache` 时表示优化已生效
- 自动内存检测同时读取当前 cgroup 及其父级限制，容器/systemd 部署不会误选大内存档

这不是常驻第二份内存权重：副本占用磁盘，compact 模式仍按任务读取并在使用后
请求内核回收干净页。因此它不会以额外 15GB RAM 换速度。
