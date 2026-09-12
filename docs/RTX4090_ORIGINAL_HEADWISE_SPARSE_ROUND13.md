# RTX 4090 Native original-weight head-wise sparse calibration (Round 13)

Date: 2026-08-12

Status: review-only. The dense Native original-weight route remains the release
default. No result below authorizes automatic routing.

## Question

Can the standalone Native original-weight engine reduce long-sequence H3
attention cost without changing weights or reducing the default 9 actual / 11
forecast sampling schedule?

## Bottleneck evidence

The accepted 1280x736x362 one-step trace has 100,000 packed tokens. One true
DiT evaluation took about 41.845 seconds. The 50 SM89 SageAttention2++ kernels
consumed 28.545 seconds, about 68.2% of that evaluation. Video-VAE compile is a
separate small lever and cannot address this dominant term.

## Real-H3 calibration, not random tensors

`scripts/benchmark_native_hot_session.py --attention-backend head-calibration`
runs the accepted dense attention and always returns its output. On the first
true long H3 attention call it computes sparse diagnostics in groups of eight
heads, writes metrics, then intentionally stops generation. This avoids a
diagnostic-only 24 GiB peak while ensuring the candidate cannot alter a video.

Evidence:

- `runtime/calibration/workload_routing_round13/original911_720p15_head_metrics.json`
- original weights, seed 82302, 1280x736x362, packed sequence 100,000
- protected text/condition/audio prefix: 1,560 tokens
- 56 heads, head dimension 128

At 50% versus 75% block budget, head 37 improved relative L1 by 0.06914, while
heads 1 and 11 changed by only 0.00001 and 0.00004. A uniform per-head budget
is therefore not error-optimal for this observed H3 call.

An equal-mean-budget allocation was constructed:

- 21 sensitive heads at 75%
- 21 middle heads at 65%
- 14 insensitive heads at 50%
- arithmetic mean budget: 65%

On the sampled true H3 call this reduced mean relative L1 from 0.025525 for
uniform 65% to 0.021730, a 14.9% reduction. Mean cosine increased from
0.996102 to 0.996300. These are local attention-output diagnostics, not video
quality claims.

## SM89 kernel microbenchmark

At 100,000 tokens, 1,560 protected prefix tokens, 56x128 heads:

- dense SageAttention2++: 603.473 ms
- equal-budget head-wise split sparse: 451.010 ms
- attention speedup: 1.338x

The head-wise allocation was within about 0.5% of the uniform-65% production
kernel, so allocating different budgets did not materially lose the expected
kernel speed. Evidence:
`runtime/calibration/workload_routing_round13/split_modality_headwise065_seq100000.json`.

## Strict generated-video A/B and rejection

The first complete candidate used the exact historical dense prompt, seed,
geometry, scheduler and 9/11 sampling contract at 864x480x243:

- dense source total: 80.476 s; denoise 64.003 s
- all-actual-step head-wise candidate total: 78.227 s; denoise 61.750 s
- speedup: 1.029x total
- peak allocated: 8.561 GiB for both route family measurements
- six-frame visual gate: pass
- FFmpeg full-video SSIM: 0.7023, diagnostic only

The official local WhisperX large-v3 CUDA/FP16 transcript rejected this
candidate. Dense global CER was 4.35%, while the all-step head-wise candidate
was 17.39%. The second line became materially less intelligible. This is a
counterexample to using first-layer error alone as a release gate.

Dense anchor steps 0, 1, 2 and 19 were then restored while the five remaining
actual steps used head-wise sparse attention:

- total: 78.473 s; denoise 62.264 s; speedup versus dense: 1.026x
- six-frame visual gate: pass
- WhisperX global CER: 4.35%, matching the dense reference
- recognized full transcript matched the dense transcript, including the same
  homophone `北乡` versus requested `北巷`

## 720p x 15 s original-weight result

The same dense-anchor policy completed the identical historical prompt, seed
82302, 1280x736x362 geometry and 9/11 schedule:

- dense original: 453.733 s total, 406.664 s denoise, 39.569 s Video-VAE,
  17.368 GiB peak allocated
- candidate: 411.293 s total, 364.720 s denoise, 39.678 s Video-VAE,
  19.798 GiB peak allocated
- total speedup: 1.103x; denoise speedup: 1.115x
- six-frame visual gate: pass; matched A/B frames preserve composition,
  identities and key objects

The local WhisperX large-v3 transcript reported 21.7% global CER for the dense
source and 34.8% for the candidate. Both videos contained or were transcribed
with extra ambiguous words around the second/third-line boundary; the candidate
was worse by this diagnostic. It is therefore still pending Human continuous
playback, especially 8.5-12 seconds, and is not an automatic release route.

Evidence:

- `runtime/calibration/workload_routing_round13/original911_720p15_headwise065_denseanchors/`
- `runtime/calibration/workload_routing_round12/original911_720p15_dense/`

## Step-3 layer calibration and sensitive-layer protection

The first per-head probe did not identify where approximation error is most
concentrated in the 50-block DiT.  A second diagnostic therefore measured all
50 real H3 blocks at actual denoise step 3.  It always returned dense output;
the sparse output was a side computation only.  The double-buffered offload
graph initially deep-copied the diagnostic state, causing the two CUDA slots to
overwrite alternating partial reports.  Sharing the read-only observer across
both block buffers fixed that calibration-only bug.  The 92-test CPU regression
suite remained green.

Evidence:

- `runtime/calibration/workload_routing_round13/original911_720p15_layer_metrics_step3.json`
- original weights, 1280x736x362, 100,000 packed tokens, step 3
- 50/50 layers measured; accepted dense output returned at every layer

Error was strongly structured.  Fifteen of the highest-error layers were
`30..43` plus `45`; together they contained 47.6% of the summed mean relative
L1 error.  A conservative candidate kept those layers dense at every step,
kept steps 0, 1, 2 and 19 wholly dense, and only used calibrated head-wise
sparsity elsewhere.  Checkpoint weights, scheduler and the 9/11 schedule were
unchanged.

At 864x480x243 with the exact seed-82301 dialogue task:

- dense original: 80.476 s total, 64.003 s denoise
- sensitive-layer-protected candidate: 78.974 s total, 62.711 s denoise
- total speedup: 1.019x
- six-frame visual gate: pass
- official local WhisperX global CER: 4.35%, exactly matching dense; the full
  transcript matched the dense transcript

At 1280x736x362 with the exact seed-82302 dialogue task:

- dense original: 453.733 s total, 406.664 s denoise
- all-layer sparse candidate: 411.293 s total, 364.720 s denoise (1.103x)
- sensitive-layer-protected candidate: 422.858 s total, 376.267 s denoise
- protected candidate total speedup: 1.073x
- Video-VAE: 39.689 s; six-frame visual gate: pass
- official local WhisperX global CER: 34.8%, versus 21.7% dense

The protected candidate therefore remains review-only.  It recovered neither
a better automatic dialogue score nor enough end-to-end speed at 480p to
justify default routing.  Human continuous playback is still required because
the recognizer's extra ambiguous phrase is concentrated around 9--12 seconds,
where prior review has found false positives.

Evidence:

- `runtime/calibration/workload_routing_round13/original911_480p10_headlayer_protected/`
- `runtime/calibration/workload_routing_round13/original911_720p15_headlayer_protected/`

## Mechanism and limits

The implementation changes no checkpoint tensor and does not reduce sampling
steps. It preserves the packed text/condition/audio prefix exactly, assigns
per-head block budgets only to video queries, and can retain selected sampling
steps on dense SageAttention2++. It is approximate attention, not mathematically
lossless inference.

The 480p result also confirms workload routing is necessary: at roughly 30k
tokens sparse attention saves little end-to-end time. The candidate should only
be considered near the long-sequence region where attention is dominant.

## Release gate

The candidate stays review-only until all are true:

1. exact same prompt, seed, geometry, 9/11 schedule and original checkpoint;
2. 720p x 15 s completes without OOM (passed once at 19.798 GiB allocated);
3. 4-8 frame visual gate passes before numerical diagnostics;
4. local WhisperX large-v3 transcript/timing is no worse than the dense source;
5. Human continuous playback accepts motion, acting, identities, audio and
   dialogue;
6. a second seed and a conditioned FL2AV task pass;
7. only then may a measured workload profile expose it through internal routing.

## External mechanism provenance

The experiment is informed by training-free head-adaptive sparse-attention
work, particularly HASTE's temporal-mask reuse and error-guided per-head budget
calibration, and by Sparse VideoGen / SVG-EAR's parameter-free compensation and
error-aware routing. Those projects do not publish MiniMax-H3/RTX4090 evidence;
all H3 claims in this document come from the local measurements above.
