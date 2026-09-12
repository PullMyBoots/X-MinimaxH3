# 独立 H3 单卡核心：SGLang 复用审阅与装配边界

## 结论

当前代码已经形成一个**可在无 GPU 环境导入和测试的完整 Pruned H3 DiT 图**，并能：

- 解析现有 `comfy_quant` INT8 ConvRot 契约；
- 从 safetensors 真实加载单个或全部 50 个 pruned H3 block；
- 执行 fused QKV、逐 head QK RMSNorm、3D partial RoPE、SDPA、SwiGLU、Indexed AdaLN 和 gated residual；
- 正确区分 pruned 8 维 AdaLN curve 与 Larry LoRA 所需的 2688 维完整 SiLU time-embedding curve；
- 将 Larry 的原生 H3 名称直接映射为 259 组 runtime LoRA，不借助 Comfy key map；
- 构造 T2VA、首帧、尾帧和首尾帧的 batch-one packed layout。
- 装配 video/audio patch projection、两层 text refiner、50-block 主干、final AdaLN/norm 和 FP32 video/audio heads；
- 从 video sigma 构造音视频/条件 unique timestep 与 Indexed AdaLN 索引；
- 完成 condition/target scatter、RoPE、patch/unpatch crop，并输出带正确符号与 audio schedule slope 的视频/音频 velocity。

但它**现在仍不能生成视频文件**。`FullH3DiT` 的边界是输入已编码的文本、视频/音频 latent 与 video sigma，输出视频/音频 velocity；它不包含 Qwen 文本编码器、scheduler、VAE 或 mux。不得把本阶段描述为“已经脱离 ComfyUI 跑通端到端生成”。

## 来源与许可证边界

主要对照：

- SGLang commit `c2fbe2f6d88692fa7756ed1be73ef9e85bd6b7cf`，Apache-2.0：
  - `runtime/models/dits/minimax_h3.py`
  - `runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py`
- 当前 ComfyUI H3 commit `9a9fdb10ed144ce760d9682cb247526ea23cc525`，GPL-3.0，只用作行为和 checkpoint 契约对照。
- Comfy Kitchen commit `5aab6c4bdb2bb73f4021277a58cc62d6185a7aed`，Apache-2.0，未来生产 INT8/ConvRot kernel 的优先复用来源。

`layers.py` 的 fused-QKV 图、Indexed AdaLN 和 partial RoPE 设计由 SGLang Apache 实现改写为单卡 PyTorch 形式，删除 TP、Ulysses、Ring、FSDP、SGLang registry 和 breakable CUDA graph 依赖。文件内已经标明来源与修改方向。

没有复制 ComfyUI 的 GPL Python 实现。`comfy_quant` JSON、safetensors 键名与张量 shape 属于互操作契约；本目录对其进行了独立实现。项目总许可证尚未由所有者决定，公开发布前仍须按根目录 `LICENSE-DECISION-REQUIRED.md` 处理许可证，并将 SGLang Apache 声明加入发行 notices。

## 已实现入口

公开入口是：

```python
from h3serve.native_engine.model import (
    audit_pruned_convrot_checkpoint,
    audit_larry_lora,
    SafeTensorSource,
    assemble_full_pruned_dit,
    assemble_pruned_block,
    assemble_pruned_block_stack,
    load_larry_updates_from_safetensors,
    build_fl2va_layout,
    FullH3DiT,
)
```

真实权重 header 审计结果：

| 文件 | 张量数 | 通过的关键契约 |
|---|---:|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 932 | 50 blocks、200 个 ConvRot INT8 linear、`adaln_t_table=[1025,8]` |
| `minimax_h3_turbo_v4_step600_ema.safetensors` | 518 | 259 对 LoRA A/B（50×5 block、2×4 refiner、1 final AdaLN） |

启动期建议先执行 header-only 审计：

```python
base_audit = audit_pruned_convrot_checkpoint(base_path)
base_audit.require_compatible()

lora_audit = audit_larry_lora(lora_path)
lora_audit.require_compatible()
```

单 block 装配是后续 LightX 风格 block offload 的接口：

```python
with SafeTensorSource(base_path) as source:
    block = assemble_pruned_block(
        0,
        source,
        device="cpu",
        int8_kernel=production_convrot_kernel,
        attention_backend=production_attention,
    )
```

全部 50 blocks 可以装配到 CPU，但不应该直接全部装配到 24 GB GPU：

```python
updates = load_larry_updates_from_safetensors(lora_path, strength=1.0)
with SafeTensorSource(base_path) as source:
    stack = assemble_pruned_block_stack(
        source,
        device="cpu",
        int8_kernel=production_convrot_kernel,
        attention_backend=production_attention,
        lora_updates=updates,
        full_silu_curve=full_silu_curve,
    )
```

`full_silu_curve` 对 Larry 路线是必要条件，shape 应为 `[1025,2688]`。基础 checkpoint 的 `[1025,8]` 表只供 pruned base AdaLN 使用。把 Larry 的 `[*,2688]` update 合并或投影到 8 维 base 权重会改变数学含义。

完整 DiT 装配入口：

```python
with SafeTensorSource(base_path) as source:
    dit = assemble_full_pruned_dit(
        source,
        device="cpu",
        int8_kernel=production_convrot_kernel,
        attention_backend=production_attention,
        lora_updates=updates,
        full_silu_curve=full_silu_curve,
    )
```

`dit.forward(...)` 的 `sigma_video` 是 `[0,1]` 的实际视频 sigma，不是 Comfy 包装层使用的 `sigma*1000` timestep。

## 已验证内容

- Python compileall 通过。
- Torch 2.8 CPU import/smoke 通过。
- packed video/audio round-trip 通过。
- group-wise normalized Hadamard 自逆测试通过。
- tiny config 的 ConvRot reference linear、fused QKV block、LoRA 和 block-stack forward 通过。
- tiny config 的完整 T2VA DiT forward 与首尾帧 scatter/gather forward 通过，输出 shape 分别恢复为 video `[B,C,T,H,W]` 和 audio `[B,C,2,T]`。
- 对本机 21 GB base 与 780 MB Larry 文件执行 header 审计，结果分别为 `compatible=True`。
- 21 GB base 已真实装配为 `FullH3DiT(50 blocks, 2 refiner blocks)`；Base+Larry+`[1025,2688]` curve 的完整 Turbo DiT 也已真实装配。
- ConvRot CPU reference 已逐元素对照 Comfy Kitchen Apache regular-H4 实现，`max_abs=0`。它不是普通 Walsh-Hadamard 排列。
- RTX 4090上以真实base checkpoint、生产Comfy-Kitchen ConvRot kernel执行了最小完整DiT前向：`context=[1,1,5376]`、video latent `[1,24,1,2,2]`、audio latent `[1,32,2,1]`；50 blocks及final head均执行完成，两个输出全为有限值。装配约56.41秒，单次前向约1.44秒，PyTorch峰值分配约19.63GiB。该结果只证明完整真实图可执行，不代表480p性能或与参考输出等价。

这些测试验证接口、shape 与基本数学一致性，不等于 H3 真实输出等价性。

## 生产路径与 reference 路径

`ConvRotInt8Linear` 有两个路径：

1. 注入 CUDA kernel：生产路径；应复用 Comfy Kitchen Apache kernel 或现有已验证的项目 kernel。
2. 无 kernel 时：执行在线 group-wise Hadamard、反量化后的 PyTorch linear；这是 CPU/正确性 reference，不能用于速度 benchmark。

ConvRot 的在线激活旋转是正确性要求。它与已经旋转的权重共同保持原 linear 的数学关系；遗漏旋转会产生灾难性彩色噪声，不能当作普通性能回退。

Attention 默认 `torch.scaled_dot_product_attention`。4090 发布路径应注入已验证的 Sage/Flash varlen backend；SGLang 的 fused QKNorm+RoPE kernel 可以作为后续对照，但不能引入完整 SGLang runtime。

## 尚缺的端到端流水线和生产运行时

`FullH3DiT` 已完成原列表中的真实 DiT forward 门槛。剩余工作是：

1. Qwen 单文件 NVFP4/AWQ 的独立 SM89 loader；presentation/token-tag
   协议和预处理已落地，但普通 Transformers/SGLang loader 不兼容该权重。
2. 将已审计匹配的 SGLang Apache fused video-VAE 与 raw audio-VAE 图抽取
   到发布包；当前 adapter、header audit 与首尾帧预处理已经落地。
3. 把已经实现并通过 synthetic AV 测试的 scheduler/noise、RES/Turbo sampler
   与原子 PyAV mux 接到真实 model/VAE pipeline。
4. LightX 风格双 buffer block offload、copy/compute stream 和 prefetch。
5. 生产 ConvRot/Attention/QKNorm-RoPE kernel binding、warmup 与固定 shape autotune。
6. block offload 下Larry tensor的驻留/预取，避免每层每步同步CPU→GPU复制。
7. 取消请求安全点、峰值显存记账和错误恢复。
8. 与当前正确实现逐阶段张量对照及真实视频视觉门控。

1–3 完成后才能生成真实 H3 MP4；4–7 完成后才能公平讨论相对历史优化路线的性能；8通过后才能宣称产品正确性。

## NativeH3Engine 建议调用方式

服务层的统一接口保持：

```python
await engine.generate(
    spec,
    output_path=output_path,
    first_frame=first_frame,
    last_frame=last_frame,
    cancel_event=cancel_event,
)
```

`job_id` 属于服务/任务管理层。manager 负责把它解析为受控的
`output_path`；NativeH3Engine 只接收路径，不读取任务数据库，也不自行
推导 job 目录。

建议内部生命周期：

1. `NativeH3Engine.start()`：执行 base/LoRA audit，选择生产 kernels，把共享 curve/小模块常驻 GPU，把 50 blocks 装配到 pinned CPU。
2. `prepare(spec, frames)`：构造并缓存固定 shape 的 `PackedLayout`、position ids、RoPE、modality/timestep indices。
3. `encode()`：按生命周期加载文本编码器与 VAE conditioning，随后释放或 offload。
4. `denoise()`：逐 block 双缓冲预取；高保真与 Turbo 共用 graph，仅切换 LoRA、scheduler 和用户质量档位。
5. `decode_and_mux()`：先卸载 DiT，再加载视频/音频 VAE，写临时文件后原子移动到最终输出。
6. 返回现有服务契约：`runtime_key`、`elapsed_seconds`、`output_path`。

在 Qwen/VAE 独立图与完整 pipeline 接线未完成，且真实权重首轮输出没有通过多帧视觉门控前，`NativeH3Engine` 必须返回明确的 `native_engine_not_ready`，不能静默回退到旧 oracle 后仍声称是独立引擎。

## 下一次真实验证

按以下次序推进：

1. 只装配 block 0，固定同一真实 packed activation、curve rows 和 RoPE，与当前正确 Comfy 执行抓取的 block 0 输入/输出比较。
2. 分别测试 eager reference、生产 ConvRot kernel、生产 attention；先比 RMS/max-abs，再检查逐阶段漂移。
3. 使用已装配的完整50 blocks与final head，在同sigma下比较video/audio velocity张量。
4. 接入block offload后运行最小真实shape DiT forward，并确认峰值不越过4090边界。
5. 完成VAE后生成480p/5s；先执行首、中、尾及全时序多帧视觉硬门槛，再记录速度、显存与诊断指标。
6. 通过基础20实际步基线后，才迁移质量滑块、预测步和Larry Turbo。
