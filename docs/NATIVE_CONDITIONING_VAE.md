# Native H3 Conditioning / VAE 迁移说明

状态：适配协议、真实 checkpoint 审计、首尾帧预处理和 CPU self-test 已完成；
完整 Qwen/VAE 模型图与权重加载尚未进入发布热路径，因此当前仍是**可执行迁移基座，
不是可生成视频的完成声明**。

本目录的实现不导入 `comfy`，不读取 workflow JSON，也不通过 HTTP 调用 ComfyUI。
算法参考只来自本地检出的 Apache-2.0 项目（SGLang Diffusion、LightX2V、
FastVideo、Comfy Kitchen）；没有复制 GPL ComfyUI/Spectrum 源码。

## 1. 真实权重审计结论

审计对象是 `release/serve/models` 指向现有模型文件的软链接。审计只解析
safetensors header/metadata，不读取大张量 payload。

| artifact | 大小/键数 | 实际格式 | 可直接复用的图 | 结论 |
|---|---:|---|---|---|
| Qwen3-VL text encoder | 15.69 GB / 2054 | 单文件、50 层；INT8 embedding + 350 个 packed NVFP4 linear；FP8 block scale、FP32 tensor scale、100 个 AWQ pre-quant scale | SGLang/LightX2V Qwen3-VL 数学图 | 标准 Transformers/SGLang loader **不能直接加载** |
| video VAE | 5.21 GB / 562 | fused `to_qkv`、`to_out`、`ff.w1/w2`，24 latent channel，FP16 | SGLang fused H3 video VAE | 与 FastVideo 703-key split Diffusers 图不兼容 |
| audio VAE | 605 MB / 917 | FP32 raw `weight`，32 latent channel，32 kHz stereo metadata | SGLang raw-weight DAC/BigVGAN 图 | 与 FastVideo 的 `weight_g/weight_v` 构造态不兼容 |

代表性硬断言包括：

- text `model.embed_tokens.weight`: I8 `[151936,5120]`；
- text `model.layers.0.self_attn.q_proj.weight`: U8 `[8192,2560]`（两个
  FP4 nibble/byte）；其 block scale 是 F8 E4M3 `[8192,320]`；
- text 只含 language layer 0--49，H3 直接使用 layer 49 后的未归一化
  5120 维输出；
- video `decoder.transformer_blocks.0.attn.to_qkv.weight`: FP16
  `[6144,2048]`；metadata 给出 24 组 mean/std、17-frame clip、token drop 3；
- audio `decoder.conv_pre.weight`: FP32 `[1024,2048,7]`，metadata 给出
  32 组 mean/std、32 kHz、2 channels。

可执行审计：

```bash
cd subprojects-main/main/release/serve
PYTHONPATH=. python -m \
  h3serve.native_engine.adapters.conditioning_vae.preflight models
```

`--require-runtime` 会在完整图/loader 未交付时返回 3，防止“权重存在”被错误解释为
“原生推理已就绪”。权重 header 不匹配返回 2。

## 2. 首尾帧迁移契约

`preprocess.prepare_keyframes(request)` 是 text/Qwen 和 video VAE 的共同输入规则：

- 只解码输入文件的第一张/第一帧，应用 EXIF orientation，转 RGB；
- `first`: LANCZOS 直接 resize 到已经解析好的目标 canvas；
- `last`: 等比 cover，LANCZOS resize，再做确定性的整数中心裁剪；
- 顺序固定为 first、last；语义索引分别为 `0`、`-1`，像素索引分别为
  `0`、`num_frames-1`；
- 行为由 role 决定。last-only 仍执行 cover/crop，不会因为它是列表第一项而误走
  first-frame stretch。

这里刻意遵从 `COMFY_MIGRATION_CONTRACT.md`。当前 Apache SGLang 实现把
“请求里的第一个语义 keyframe”一律 stretch，因此 last-only 会 stretch；接入前必须
用现有正确流程做一次 last-only oracle，确认产品契约与历史行为哪个才是最终真值，
不能静默混用。

## 3. Qwen3-VL adapter

`H3Qwen3VLConditioner` 只依赖三个注入对象：tokenizer、Qwen3-VL processor、
具有 `encode_ids()` 的 layer-50 encoder。它实现：

- T2VA：原 prompt、不加特殊 token，tags 全为 text(1)；
- first/last/first+last：按顺序构造 `<Picture i>: `、vision start、N 个
  image pad、vision end，再拼原 prompt；vision block tags 为 0，其他为 1；
- Qwen image processor 的 token count 为 `prod(image_grid_thw)/merge_size²`；
- 输出 shape 硬断言为 `[sequence,5120]`。

本地 key prefix 到 SGLang 图的映射是：

```text
model.*  -> model.language_model.*
visual.* -> model.visual.*
```

`load_local_qwen_encoder()` 必须收到明确声明支持
`comfy_nvfp4_awq_single_file_v1` 的 quantized loader，否则 fail closed。原因是 U8
payload 不是普通 U8 权重；忽略 FP8 block scale、FP32 tensor scale 或 AWQ
pre-quant scale 都会产生数值错误。

4090 是 SM89，没有 Blackwell 原生 NVFP4 Tensor Core。Apache Comfy Kitchen 的
NVFP4 数据结构/反量化原语可以复用，但其直接 NVFP4 matmul 快路标注为 SM100+；
SM89 发布实现必须选择并验证以下之一：分块反量化到 BF16/FP16 后 GEMM，或已有的
Ada 专用 quant/full-precision multiplication 路径。不能把“能解析 NVFP4”写成
“4090 有 NVFP4 硬件加速”。

## 4. Video VAE adapter

`H3VideoVAEAdapter` 支持注入两类窄接口：

- SGLang fused graph：`encode_images(PIL, use_fp16_latent=True)`、
  `decode_base()`、`processor.revert_tensor()`；
- Diffusers-like graph：`encode_keyframe()`、`decode()`，可选
  `normalize_pixels()/denormalize_pixels()`。

条件图像的 adapter 公共边界为 RGB `[0,1]`，tensor encode 边界为 `[-1,1]`；
若 checkpoint 图声明 ImageNet normalization，adapter 再在内部桥接。关键帧 posterior
使用隔离的 seed 42 sampled encode，且整个 first+last pair 只做一次 VAE FP32 dtype
切换。随后按 metadata mean/std 标准化，并以 `[1,2,2]` patch 得到 96 维 FP32
condition rows。返回值保存每个 anchor 的 role、语义/像素索引与 latent geometry，
防止 packer 只见一块拼接 tensor 后丢失条件语义。

decode 接收 normalized `[B,24,T,H,W]`，反标准化后解码，返回 batch-1 的
`[frames,H,W,3]` float32 `[0,1]`。

尚有一个必须在主 pipeline 解除的接口阻断：fused tiled decoder 可能在 bottom/right
产生比请求 canvas 更大的输出，必须按请求 width/height 裁剪；目前 pipeline 的
`video_vae.decode(latents)` 没传目标 shape。不能依赖“多数尺寸碰巧无需 crop”。可接受
的解除方式是让 decode 接收 request/shape，或把目标 shape 绑定到单请求 decode
context，并增加 480p/720p 非 tile-aligned shape 回归测试。

## 5. Audio VAE adapter

H3 denoiser 的音频 latent 契约是 `[B,32,2,T]`。`H3AudioVAEAdapter`：

1. 转成 `[B*2,32,T]`，把左右声道当作 mono batch；
2. 用 metadata mean/std 反标准化；
3. 调 raw-weight DAC/BigVGAN decoder；
4. 重排为 `[B,2,samples]`；
5. 对每个 batch 在 channel/time 上计算 `std * 5`，divisor 最低为 1，做除法；
6. clamp `[-1,1]`，返回 32 kHz。

adapter 也提供了独立的 stereo `encode()` 协议，供未来 Ref2VA 使用；当前
T2VA/FL2VA 发布路径不需要音频输入编码。

## 6. 最小依赖与源码选择

最终发布不应把三个通用框架都作为运行依赖。建议：

1. 以 SGLang 的 fused video VAE、raw-weight audio VAE、Qwen3-VL layer-50 图为
   Apache-2.0 主来源，抽取模型图与直接依赖并保留 SPDX/NOTICE；
2. 复用 Transformers 的 tokenizer/Qwen3-VL processor 数据文件和处理接口，不让
   Transformers 普通 linear loader接触本地 packed text 权重；
3. 只抽取 Comfy Kitchen 中 SM89 实际会调用的量化 primitive；不要把整个
   ComfyUI runtime、ModelPatcher 或节点系统带回来；
4. LightX2V 的 component/offload 生命周期适合作为后续 residency 优化参考，
   FastVideo 的 stage/test 结构适合作为工程参考，但二者当前 checkpoint layout
   不应直接加载本地 VAE。

## 7. 测试与上线门槛

不依赖 torch/safetensors 的最小测试：

```bash
cd subprojects-main/main/release/serve
PYTHONPATH=. python -m \
  h3serve.native_engine.adapters.conditioning_vae.selftest -v
```

在实际 H3 Python 环境运行完整 CPU tensor/header 测试：

```bash
PYTHONPATH=. /root/miniconda3/envs/voxcpm/bin/python -m \
  h3serve.native_engine.adapters.conditioning_vae.selftest -v
```

当前覆盖 canvas geometry、role/index/order、RGB、三个真实大权重 header、视频
patchify 和音频 stereo/反标准化/响度协议。正式上线前还必须全部满足：

- standalone Qwen graph + SM89 quant loader 加载真实文件，并与当前正确流程逐 token
  对照 ids/tags、逐 tensor 对照 layer-50 hidden states；
- standalone VAE 图 strict-load：video 562/562、audio 917/917，无 missing/unexpected；
- first/last/first+last 的 prepared pixels、VAE latent、patch rows 分别对 oracle；
- video decode 覆盖 480p/720p、5s/15s 和 tile crop；audio 覆盖声道顺序、长度、
  sample rate、响度；
- 四种任务真实生成先通过多帧视觉硬门，再记录媒体、速度、显存与诊断指标；
  SSIM 只能描述相对漂移，不能作为硬门或覆盖视觉失败。

在这些门槛解除前，preflight 的 `runtime_ready=false` 是正确状态。
