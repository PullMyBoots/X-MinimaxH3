# FlashVSR online hot runtime

## Production policy

- H3 and FlashVSR remain separate Python processes because their pinned Torch
  and CUDA extension ABIs differ.
- H3 preloads first. FlashVSR then loads its immutable v1.1 Tiny Long weights
  into CPU RAM and waits on a line-delimited local pipe.
- The GPU queue remains serial. Only an upscale request moves FlashVSR modules
  to CUDA; completion moves the complete graph back to CPU, clears temporal,
  cross-KV and attention-mask caches, and empties the allocator cache.
- Input videos up to `H3_FLASHVSR_GPU_INPUT_LIMIT_MIB` (default 1536 MiB) are
  transferred once. Larger inputs retain the vendor CPU-streaming path.
- Cancellation terminates the isolated daemon, which is recreated on the next
  request. A daemon crash cannot corrupt the H3 service process.

No model weights, inference steps, block-sparse ratios, colour correction or
output encoding settings were changed by this optimisation.

## RTX 4090 evidence (2026-08-13)

Input: the same H3-generated source video for both runs; output target
1152x640; official FlashVSR v1.1 Tiny Long one-step route.

| Input | First task | Second task | Peak allocated | Repeat result |
|---|---:|---:|---:|---|
| 22 frames | 6.470 s | 5.021 s | 8517.4 MiB | identical SHA-256 |
| 124 frames | 14.729 s | 13.419 s | 8930.9 MiB | identical SHA-256 |

The strict same-task 22-frame one-shot process took 38.06 s wall time. Its MP4
SHA-256 is identical to both persistent-worker outputs. The second hot request
therefore provides a measured 7.58x end-to-end postprocess speedup for this
short task; the gain is intentionally expected to shrink as video duration and
the irreducible inference/encode portion grow.

The 124-frame hot worker breakdown was 1.665 s decode/preprocess, 1.543 s GPU
prepare, 7.313 s FlashVSR inference and 2.271 s postprocess/encode. The model
load itself was 11.034 s, while WSL page-fault and process-start wall time was
35.418 s; that cost now occurs once after service startup, not once per job.

Evidence files:

- `runtime/validation/flashvsr-hot/report_v2.json`
- `runtime/validation/flashvsr-hot/report_124f.json`
- `runtime/validation/flashvsr-hot/input_124f_run1_flashvsr_1152x640.mp4`
- `runtime/validation/flashvsr-hot/input_124f_run2_flashvsr_1152x640.mp4`

After stopping the validation daemon, observed GPU memory returned to the
already-running H3 service baseline (~1970 MiB). In the production service,
`GET /healthz` reports `resident_state`, preload timing, completed request count
and the `on_demand_release_after_each_task` policy.

## Verification

The regular suite has 116 passing tests and four separately enabled strict
release gates. `tests/test_upscaler.py` covers daemon reuse and cancellation;
the real GPU evidence above covers repeat determinism, output AV streams, peak
VRAM and shutdown cleanup.
