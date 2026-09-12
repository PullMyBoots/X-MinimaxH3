# RTX 4090 Split-Modality Attention Round 15

日期：2026-08-15  
任务：原始INT8权重，1280×736，362帧，15.083秒，20步，9实际＋11预测，seed 82303。

## 性能复验

同一自然语言复杂提示词的第二次热态结果：

| 后端 | 热态端到端 | DiT | 峰值CUDA |
|---|---:|---:|---:|
| Dense Sage | 450.316s | 406.628s | 17.368GiB |
| Split-Modality top-k 0.50 | 336.326s | 292.534s | 19.799GiB |

端到端1.339×，DiT 1.390×。Split两次同seed的DiT为292.566/292.534秒，
且MP4 SHA256完全一致；Dense两次MP4不同，证明当前dense执行本身存在轨迹漂移。

## Human发现与提示词修正

Human审核自然语言版本时发现Dense repeat2中间约两秒胡言乱语，而Split视频正常；
其他画面没有明显问题，但720p整体略糊。该观察不能归因于Split稀疏。

原测试提示词随后改为控制台正式T2VA序列化：

```text
integrated_multimodal_description: [Shot 1]
...
<d>[Chinese] 精确台词 </d>
...
overall_soundscape: ...
non_diegetic_music: N/A
```

并明确禁止三个`<d>`标签之外的任何人声。保持seed、geometry与9/11不变。

| H3结构化提示词后端 | 总时间（含冷文本编码） | DiT | 峰值CUDA | WhisperX全局CER |
|---|---:|---:|---:|---:|
| Dense Sage | 496.157s | 407.018s | 17.368GiB | 0.0870 |
| Split-Modality 0.50 | 385.682s | 294.407s | 19.822GiB | 0.0435 |

结构化提示词把packed tokens从100000增加到100163，DiT成本变化不足1%；两条成片
六帧灾难门通过。WhisperX未发现额外人声，主要错误为`北巷→北汉`等近音识别。

## 当前判断

- 性能收益在新seed与正式H3提示词下复现。
- Split没有表现出比Dense更差的自动对白结果，但自动指标不替代Human连续播放。
- Human指出的720p柔化可能来自9/11 forecast比例，也可能是H3长视频本身；下一步必须
  用同一结构化提示词比较9/11与更保守的12/8，不能同时改变其它变量。
- I-004继续保持`REVIEW`，不自动进入发布planner。

证据：`runtime/calibration/workload_routing_round15/i004_*seed82303/`。
