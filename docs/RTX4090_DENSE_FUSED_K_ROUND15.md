# RTX 4090 长序列 Dense Sage fused smooth-K Round 15

日期：2026-08-15  
目标：在不修改模型权重、不改变 SageAttention 数值语义的前提下，验证消除完整 BF16 `K - mean(K)` 中间张量是否值得进入完整 H3 DiT。

## 结论

候选在两个真实 H3 长序列 shape 上均逐元素完全一致，但性能收益只有
`1.00096x` 和 `1.00462x`。它没有达到 idea I-002 预先规定的 5% 完整
step 门槛，也低于 3% 微基准停止线，因此本轮在微基准处停止，不再消耗
GPU 时间生成完整视频。

这不是正确性失败；它说明当前 SM89/Sage kernel 中，单独消除 smooth-K
中间张量不是 H3 长序列的有效主杠杆。

## 固定环境

- GPU：NVIDIA GeForce RTX 4090，SM89
- PyTorch：2.8.0+cu126
- CUDA runtime：12.6
- Comfy-Kitchen CUDA SHA256：
  `aaefcd38ba30379e5707b22bdc7e3209188e75c78ab4dd4a259f9e1d83eafa9b`
- SageAttention SM89 SHA256：
  `c44f6878acd51920192d0d9fdbbeebeaade1e4c8eda21ac372f7d0332d99ef5f`

## 命令

```bash
cd release/serve
PYTHONPATH="$PWD/backends/turbo/vendor:$PWD" \
  /root/miniconda3/envs/voxcpm/bin/python \
  scripts/benchmark_sage_fused_k_sm89.py \
  --sequence 34519 --sequence 100000 \
  --warmup 3 --repeat 9 \
  --output runtime/calibration/workload_routing_round15/sage_fused_k_exact_sm89.json
```

## 结果

| Packed tokens | 当前 dense Sage 中位数 | fused smooth-K 中位数 | 加速比 | `max_abs` | 逐元素一致 |
|---:|---:|---:|---:|---:|---|
| 34,519 | 73.7720 ms | 73.7014 ms | 1.00096x | 0 | 是 |
| 100,000 | 601.7173 ms | 598.9498 ms | 1.00462x | 0 | 是 |

证据：
`runtime/calibration/workload_routing_round15/sage_fused_k_exact_sm89.json`

## 决策

1. I-002 改为 `REJECTED`；候选代码可以保留为研究实现，但不进入生产 planner。
2. 不运行完整 50-block DiT 和成片，因为微基准已经远低于预注册门槛。
3. P0 GPU 预算转给 I-004：split-modality/shared-V 的新 seed、复杂任务质量闭环。
4. 只有上游 Sage 的 K 量化边界或 SM89 主 kernel 发生实质变化时才重开。
