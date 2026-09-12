# Native H3 runtime and pipeline

This package is the lightweight execution base intended to replace ComfyUI in
the Web/API service. It is not another general graph framework. It supports one
MiniMax H3 request at a time on one RTX 4090 (24 GiB), with batch size one and
the product's fixed resolution, duration and conditioning choices.

## What is implemented

```text
GenerationInput
      |
validate -> text conditioning -> optional frame conditioning
      -> prepare AV latents -> denoise -> video decode -> audio decode -> mux
                                  |
                       ResidencyManager
                                  |
             DoubleBufferBlockExecutor
                  copy stream || compute stream
```

The package separates three contracts:

1. `pipeline/` declares deterministic H3 stages and the mutable state passed
   between them. It never imports ComfyUI, FastVideo, LightX2V or CUDA.
2. `runtime/residency.py` enforces eviction-first component transitions and a
   conservative 23 GiB device budget. Text encoder, transformer and decoded
   video cannot accidentally overlap.
3. `runtime/offload.py` traverses host-resident transformer blocks through two
   preallocated device buffers. The copy of block `n+1` overlaps compute of
   block `n`; per-slot CUDA events prevent overwrite without a device-wide
   synchronization after every block.

Importing either package is safe on CPU. PyTorch is imported lazily only when a
CUDA stream coordinator is actually constructed or cache cleanup is requested.
The checked-in CPU smoke test runs with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m h3serve.native_engine.pipeline.selftest
```

## Model-core adapter contract

The H3 mathematical/model layer registers these names with
`ResidencyManager`:

| Name | Required method | Residency estimate |
|---|---|---|
| `text_encoder` | `encode(GenerationInput)` | Qwen quantized runtime bytes |
| `video_vae` | `encode_conditioning(request)`, `decode(latents)` | video VAE runtime bytes |
| `scheduler` | `prepare(PipelineState) -> dict` | `0` (host component) |
| `transformer` | `denoise(PipelineState, cancel_check=...) -> dict` | pre/post + two block buffers |
| `audio_vae` | `decode(latents)` | audio VAE runtime bytes |
| `muxer` | `write(video=..., audio=..., ...)` | `0` (host component) |

The transformer adapter should own `DoubleBufferBlockExecutor`. Its two buffer
objects implement:

```python
def load_from(source_block, *, block_index: int, non_blocking: bool) -> None:
    ...
```

This is the seam where the fused H3 model graph and ConvRot INT8 weight loader
connect. Source block tensors must be page-locked when asynchronous H2D transfer
is enabled; otherwise `non_blocking=True` is correct but will not overlap.

Minimal construction:

```python
runtime = RuntimeConfig()
residency = ResidencyManager(runtime)
residency.register(text_encoder_adapter)
residency.register(video_vae_adapter)
residency.register(HostComponentResidency("scheduler", scheduler))
residency.register(transformer_adapter)
residency.register(audio_vae_adapter)
residency.register(HostComponentResidency("muxer", muxer))

pipeline = NativeH3Pipeline(default_h3_stages(), residency)
state = pipeline.generate(request)
```

The public service can keep one pipeline hot and serialize requests through its
existing single-GPU queue. Creative presets belong in `SamplingConfig` or the
service request mapper; CUDA/offload policy belongs only in `RuntimeConfig`.
Video and audio decode results are synchronously moved to host memory before the
next lifecycle phase so output tensors are not invisible to the residency
budget.

## Service entry point and cancellation

`NativeH3Pipeline.generate(...)` is the synchronous model-runtime entry point.
The service-facing `NativeH3Engine` should preserve its asynchronous contract by
calling it in the existing dedicated single-GPU worker (or via
`asyncio.to_thread`), rather than blocking the API event loop:

```python
request = map_generation_spec(spec, job_id, first_frame, last_frame)
state = await asyncio.to_thread(
    pipeline.generate,
    request,
    cancel_check=cancel_event.is_set,
)
return map_pipeline_result(state.result, state.metrics)
```

The exact call order for `NativeH3Engine` is:

1. At process startup, load/check model manifests, construct component
   adapters, create the CUDA stream coordinator and two transformer block
   buffers, register all components, then construct one `NativeH3Pipeline`.
2. For each serialized job, normalize the service spec to `GenerationInput`
   and invoke `pipeline.generate` with `cancel_event.is_set`.
3. The pipeline activates text, optional conditioning VAE, transformer, video
   VAE and audio VAE in that order. Eviction always precedes admission.
4. The transformer adapter checks `cancel_check()` between every denoise step.
   It must not abort or free storage in the middle of a CUDA kernel or an
   in-flight block copy.
5. On success, mux receives host-resident audio/video. On error or cancellation,
   `generate` releases every active device component before propagating the
   exception. The service maps `PipelineCancelled` to its normal cancelled job
   status rather than a generation failure.
6. At process shutdown, call `pipeline.close()`, synchronize/release block
   streams and dispose device buffers in the transformer adapter.

Cancellation is therefore cooperative, not instantaneous. Worst-case latency is
one real denoise step. This is safer than trying to interrupt CUDA work and is
compatible with the current `asyncio.Event` service contract.

## Deliberate source reuse

The code is a small, newly written implementation informed by these Apache-2.0
projects, not a copied package subtree:

- LightX2V revision `231e2307f15b9eb60fe3f877f7eed945c8d8d717` informed
  two device buffers, independent copy/compute streams, and separate
  text/transformer/VAE residency. Relevant references are
  `lightx2v/common/offload/manager.py`,
  `lightx2v/models/networks/minimax_h3/infer/offload/transformer_infer.py`, and
  `lightx2v/models/runners/minimax_h3/minimax_h3_runner.py`.
- FastVideo revision `ffc1a7a58b7b1ec70e4d1dc5925d83cc0e065b98` informed the
  typed request/state object and one-verb stage composition. Relevant references
  are `fastvideo/pipelines/stages/base.py`, `pipeline_batch_info.py`, and
  `composed_pipeline_base.py`.

No LightX2V/FastVideo imports or distribution dependency is introduced. Before
public release, the project owner must still choose the service license and
decide whether these design-source acknowledgements should also be copied into
the release-wide third-party notices.

## Correctness and performance boundaries

- Supported production hardware is exactly CUDA compute capability 8.9. CPU is
  for import, orchestration and unit tests only.
- H3 canvas dimensions must be multiples of 32 and frames must be `17*n+5`.
- Batch size is fixed at one. Multi-GPU, training, arbitrary graphs and generic
  model registration are intentionally absent.
- `max_device_bytes` is a planning guard, not a replacement for measured CUDA
  peak memory. Each adapter must report realistic runtime allocation estimates.
- `retain_block_buffers_between_requests` is policy metadata for the model
  adapter. Whether retaining two buffers is faster and still safe across the
  DiT-to-VAE boundary must be settled by 480p/720p, 5s/15s profiling.
- CUDA graph capture should not be attempted until pointer-stable buffers and
  fixed shapes are proven. Block offload and graph capture often conflict.
- A successful import/unit test proves lifecycle mechanics only. Release parity
  requires full-step output against the working external reference, followed by
  the project's mandatory multi-frame visual gate, audio inspection, hot/cold
  timing and measured peak VRAM.

## Integration sequence

1. Wrap the independent H3 scheduler, text encoder, VAEs and muxer with the six
   component methods above and prove a CPU fake-component pipeline test.
2. Connect the fused H3/ConvRot model adapter to `DoubleBufferBlockExecutor`;
   validate one block and then a full 20-step denoise traversal.
3. Run full-compute, fixed prompt/seed parity before enabling forecast steps,
   alternative attention kernels, LoRA or compile.
4. Measure H2D/compute overlap in a CUDA trace. If copies serialize, confirm
   host weights are pinned before changing stream priorities.
5. Only after correctness and the visual gate pass should the Web/API backend
   switch from the migration ComfyUI process manager to this pipeline.
