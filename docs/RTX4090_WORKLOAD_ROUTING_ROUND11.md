# RTX 4090 workload routing — Round 11

Date: 2026-08-12

## Scope

The optimization target is the complete supported product envelope, not only
720p/15s.  High-resolution and long-duration requests are the current priority
because their long packed sequences expose the dominant DiT cost and the
24 GiB memory boundary most clearly.  An optimization is classified as:

1. **shared** when the same implementation can benefit original/LoRA and
   short/long shapes;
2. **routed** only when measured shape or memory crossovers require different
   mechanics;
3. **quality-authorized** when it changes numerical execution or attention
   sparsity.  Such a route remains experimental until Human playback review.

## Results

All full-video comparisons use the same LoRA six-step model, prompt, seed,
1280x736 compute canvas, 362 frames and one hot process.  The prompt contains
three timed Mandarin lines and large character/camera movements.

| Candidate | End-to-end | DiT | Peak allocated | E2E vs dense | DiT vs dense |
|---|---:|---:|---:|---:|---:|
| Dense reference | 315.319 s | 264.546 s | 16.404 GiB | 1.000x | 1.000x |
| Sparse top-k 0.50 | 241.194 s | 189.569 s | 18.502 GiB | 1.307x | 1.396x |
| Fused RMS/AdaLN + dense per-warp QK | 305.957 s | 254.946 s | 16.404 GiB | 1.031x | 1.038x |
| Sparse top-k 0.50 + fused RMS/AdaLN | **226.915 s** | **176.060 s** | 18.504 GiB | **1.390x** | **1.503x** |

The combined route is additive: it improves over sparse-only by 1.063x
end-to-end and 1.077x in DiT.  Video decode remains about 40.5 seconds and mux
about 7.5 seconds, so attention improvements alone cannot remove the remaining
fixed tail.

At 720p/10s, request-scoped fused RMS/AdaLN reduced the measured dense mean
from 172.876 seconds to 164.507 seconds (1.051x end-to-end, 1.065x DiT) without
changing the 12.611 GiB peak.  This is evidence that the implementation is a
shared long-sequence optimization rather than a 15-second-only trick.

The follow-up 480p/5s hot A/B strengthens that conclusion.  DiT fell from
17.735/17.854 seconds to 15.775 seconds (1.128x versus the dense mean).  Using
the cached-text dense repeat for a fair whole-request comparison, end-to-end
fell from 26.118 to 24.069 seconds (1.085x); peak allocation stayed essentially
flat at 6.604 versus 6.603 GiB.  Both expected Mandarin lines had CER 0 and
correct windows in exact and fused videos, while the six-frame preliminary
gate found no catastrophic visual defect.

### Original-weight Human verdict

The original-weight 9-real/11-forecast 480p/5s A/B was also measured manually:
the cached exact repeat took 34.107 seconds and fused RMS/AdaLN took 30.814
seconds, a 1.107x speedup.  Human continuous-playback review found the overall
video, picture and requested actions essentially the same, but preferred the
exact version's acting as slightly more natural.  In the fused candidate the
woman sounded more urgent and the object-placement performance changed subtly.

Because fused RMS/AdaLN was proposed as a lossless optimization, this
perceptible performance-style drift is sufficient to reject it from the
original-weight release route.  The original engine therefore keeps the exact
unfused computation path.  The fused implementation remains available for
research, and this verdict does not by itself decide the separate LoRA route.
The checked-in fused review profiles support only `engine=lora`; an explicit
planner regression test prevents an original-weight request from selecting
them even when experimental review is enabled.

## Stability finding at 100,000 packed tokens

The initial fused run exposed an illegal-memory interaction in SageAttention's
`per_thread_int8` Q quantizer.  The fused kernel itself passed 600 consecutive
100,000x5,376 executions, and dense SageAttention passed isolated long-shape
tests.  A synchronized full-chain trace localized the failure to the combined
allocation/layout state before the per-thread query quantizer.

The fix is request-scoped, explicit and fail-closed: long dense fused requests
use Sage's supported `per_warp` QK quantization.  Sparse requests do not consume
this setting, but retain it as a deterministic diagnostic fallback.  No global
package state is mutated.

## Quality gate

For the combined 720p/15s candidate:

- Whisper global CER is 0.043478, equal to the dense reference;
- all three expected lines remain in the expected order and time windows;
- six sampled frames show no colored patches, gross ghosting or catastrophic
  body/object collapse in the preliminary Main review;
- reference SSIM is approximately 0.7843 and is diagnostic only, never a hard
  quality gate;
- Human continuous-playback review is still required before expansion beyond
  the exact measured route.

The planner therefore registers only exact measured shapes as experimental
profiles. Normal startup excludes them. Dense fused profiles require
`H3_NATIVE_REVIEW_FUSED_RMS=1`, sparse profiles require
`H3_NATIVE_REVIEW_SPARSE=1`, and the combined profile requires both switches.
A one-token neighbor and every unmeasured condition mode fall back to the
validated dense plan.

## Rejected candidates

- A bitwise-exact segmented residual fusion was 1–2% slower on SM89 and was
  reverted.
- Coalescing the approximately 20 GiB host-pinned DiT storage did not improve
  cold assembly and was reverted.
- Modality-protected sparse attention preserved text/audio query cosine
  (0.99918) much better than regular sparse attention (0.7149), but produced
  only a 1.10x attention-kernel speedup at 67,368 tokens.  It is not worth the
  quality uncertainty or integration complexity yet.

## Cold versus hot priorities

Cold construction is about 63–73 seconds on the measured machine.  Qwen cache
construction reaches roughly 47–60 seconds, Video-VAE roughly 61–68 seconds,
and DiT roughly 62–72 seconds; the large host-pinning/assembly path dominates.
This cost is amortized by a persistent service and is not paid per hot request.
Current engineering priority remains hot DiT, then long Video-VAE decode, while
cold assembly work is accepted only when it does not compromise hot residency.

## Evidence

- Dense reference report: `runtime/calibration/workload_routing_round9/dense/round9_dense_lora6_hot_session.json`
- Sparse reference report: `runtime/calibration/workload_routing_round9/full_sparse050_complex/round9_full_sparse050_lora6_complex_hot_session.json`
- Dense fused report: `runtime/calibration/workload_routing_round11/rmsfusion_perwarp_720p15_full/round11_rmsfusion_perwarp_720p15_full_hot_session.json`
- Combined report: `runtime/calibration/workload_routing_round11/combo_sparse050_rms_720p15/round11_combo_sparse050_rms_720p15_hot_session.json`
- Combined video: `runtime/calibration/workload_routing_round11/combo_sparse050_rms_720p15/round11_combo_sparse050_rms_720p15_complex_dialogue_720p15_fused_rms_1280x736_362f_seed82302.mp4`
- Combined contact sheet: `runtime/calibration/workload_routing_round11/combo_sparse050_rms_720p15/combo_six_frame_contact.jpg`
- 480p/5s A/B report: `runtime/calibration/workload_routing_round11/rmsfusion_480p5/round11_rmsfusion_480p5_hot_session.json`
- 480p/5s fused video: `runtime/calibration/workload_routing_round11/rmsfusion_480p5/round11_rmsfusion_480p5_480p5_fused_rms_864x480_124f_seed82531.mp4`
- Protected-sparse microbenchmark: `runtime/calibration/workload_routing_round11/modality_sparse_querymetrics_67368.json`

## Next measurements

1. Keep original-weight fused RMS/AdaLN out of release after Human playback
   preferred the exact acting; assess the LoRA route independently.
2. Profile LoRA low-rank projection overhead separately; integrate only a
   measured SM89 win that preserves the existing adapter equation.
3. Attack the 40-second 720p/15s Video-VAE tail with exact tile/stream reuse;
   keep decode equivalence and seam checks mandatory.
4. Expand any approximate route only by resolution-duration-condition buckets
   that have their own complex-video and Human evidence.
5. Treat original weights and LoRA as two equal product mainlines.  Shared
   Video-VAE/mux/runtime changes require regression coverage on both; DiT,
   schedule and sparse-attention authorization remain engine-specific.

## Round 12 continuation

The shared Video-VAE host boundary now converts the already-decoded RGB tensor
to final uint8 on the GPU before D2H.  A real-weight 864x480x73 audit found
zero differing codec-input bytes across 90,823,680 RGB bytes.  Host traffic
fell from 363,294,720 to 90,823,680 bytes and measured decode-tail wall time
fell from 3.752 to 3.237 seconds.  The reference float32 boundary remains
available for audits; production original and LoRA sessions use the byte-exact
uint8 transport.

The current original-weight exact-path 1280x736x362 one-step trace measured
41.845 seconds for one full DiT evaluation.  The 100,000-token dense attention
kernel consumed 28.545 seconds across 50 blocks, approximately 68% of the
step.  This makes the next original-weight experiment explicit: conservative
and aggressive sparse candidates must be generated and reviewed on the same
complex 9-real/11-forecast task.  Neither candidate is a release route until
Human playback approval.

That full original-weight scan is now available.  The prior exact-route timing
anchor was 468.186 seconds.  A newly generated same-prompt/same-seed dense
reference completed in 453.733 seconds; its DiT was 406.664 seconds, matching
the prior 406.453-second DiT anchor.  With identical 9-real/11-forecast scheduling,
prompt, seed and geometry, top-k 0.75 completed in 402.335 seconds (1.128x
versus the new fair dense reference), while top-k 0.50 completed in 327.821
seconds (1.384x).  Against the older 468.186-second timing anchor these are
1.164x and 1.428x respectively.  Both six-frame sheets
passed the catastrophic visual gate.  Formal WhisperX large-v3/CUDA-FP16
global CER was 0.17391 for dense, 0.04348 for 0.75 and 0.30435 for 0.50.
This non-monotonic ASR result is evidence that CER is diagnostic rather than a
quality oracle.  The 0.50 candidate's second line was nevertheless
substantially degraded, so it is not eligible for a high-fidelity route before
Human playback; 0.75 remains a review candidate, not a release default.

Evidence:

- report: `runtime/calibration/workload_routing_round12/original911_720p15_sparse_scan/round12_original911_sparse_hot_session.json`
- dense reference: `runtime/calibration/workload_routing_round12/original911_720p15_dense/round12_original911_dense_complex_dialogue_720p15_1280x736_362f_seed82302.mp4`
- top-k 0.75 video: `runtime/calibration/workload_routing_round12/original911_720p15_sparse_scan/round12_original911_sparse_original911_720p15_sparse075_1280x736_362f_seed82302.mp4`
- top-k 0.50 video: `runtime/calibration/workload_routing_round12/original911_720p15_sparse_scan/round12_original911_sparse_original911_720p15_sparse050_1280x736_362f_seed82302.mp4`
- visual sheets: `runtime/calibration/workload_routing_round12/original911_720p15_sparse_scan/visual_gate/`
- WhisperX: `runtime/calibration/workload_routing_round12/original911_720p15_sparse_scan/whisperx/`

### Round 12 original-weight attention-policy follow-up

The original-weight line was investigated independently rather than treated as
a quality reference for LoRA.  A profiled Larry-LoRA one-step request took
44.400 seconds versus 41.845 seconds for original weights at the same
100,000-token shape.  The approximately 2.56-second LoRA increment is only
6.1% of the original step; both engines spend about 28.5 seconds in the same
50 long SageAttention kernels.  Shared long-attention work therefore has much
more product leverage than optimizing only LoRA's low-rank projections.

Two additional original-weight policies were generated with the same prompt,
seed, 1280x736 canvas, 362 frames and 9-real/11-forecast schedule:

| Original-weight candidate | End-to-end | DiT | Peak | vs fair dense | WhisperX global CER |
|---|---:|---:|---:|---:|---:|
| Fair dense | 453.733 s | 406.664 s | 17.37 GiB | 1.000x | 0.17391 |
| Text/audio protected, video-video top-k 0.50 | 427.530 s | 382.364 s | 17.37 GiB | 1.061x | 0.13043 |
| Full top-k 0.75 | 402.335 s | 357.171 s | 18.44 GiB | 1.128x | 0.04348 |
| Dense steps 0,1,16,19; middle actual steps top-k 0.50 | 380.586 s | 334.681 s | 18.44 GiB | 1.192x | 0.34783 |
| Full top-k 0.50 | 327.821 s | 284.594 s | 18.44 GiB | 1.384x | 0.30435 |

The modality-protected backend derives the protected prefix from the actual H3
packed layout: `[text | optional frame conditions | audio | video]`.  Every
prefix query retains all keys, and every video query retains all prefix keys;
only video-to-video blocks are sparse.  At the real 100,000-token shape this
raised protected-query cosine from about 0.7145 to 0.99918, but reduced the
attention-kernel speedup from 1.770x to 1.109x.  The complete video recovered
all three dialogue lines in order; the ASR transcript was `我们快守不住了，带伤员从北向撤离，
我留下等援军进城`.  Its six-frame catastrophe gate passed.

The step-hybrid candidate also passed the six-frame catastrophe gate but lost
the first spoken line in WhisperX.  Keeping early and late diffusion steps
dense therefore did not protect dialogue as reliably as protecting the actual
conditioning modality.  It is excluded from automatic routing.  The protected
and full-0.75 candidates remain Human-review candidates; neither is a default
until continuous playback confirms motion, acting, audio artifacts and exact
instruction adherence.

Additional evidence:

- original one-step trace: `runtime/profile/round12_original_exact_720p15_step1/`
- LoRA one-step trace: `runtime/profile/round12_lora_exact_720p15_step1/`
- modality-protected report/video/gates: `runtime/calibration/workload_routing_round12/original911_720p15_modality_sparse050/`
- step-hybrid report/video/gates: `runtime/calibration/workload_routing_round12/original911_720p15_hybrid_sparse050/`

### Round 12 split-modality candidate and dual-engine audit

The original-weight line remains a first-class optimization target.  The
packed sequence is `[text | optional frame conditions | audio | video]`, so a
new experimental backend executes all prefix queries with dense
SageAttention2++ and only video queries with SpargeAttention2++.  Sparse video
queries are also forced to retain every prefix key block.  This avoids the
large speed penalty of making the small prefix fully connected inside every
row of one global sparse LUT.

At 100,000 tokens with 1,560 protected tokens, the split kernel measured
361.739 ms versus 602.524 ms dense (1.666x).  Prefix-query cosine versus dense
was 0.9999983.  Sharing the already-quantized FP8 value tensor between the
dense and sparse calls was output-identical in the isolated kernel and removed
one full V quantization.  Sharing Sparge's block-quantized K as well was faster
but reduced prefix cosine to 0.999335; that variant is rejected because the
protected region carries prompt and audio conditioning.

The first full-video shared-V implementation mistakenly kept both full K
quantization buffers alive.  Peak allocation rose from 19.80 to 21.83 GiB and
end-to-end time regressed to 451.020 seconds.  It is a rejected result, not an
optimization.  Executing the prefix first, deleting its Q/K temporaries and
only then building the video sparse LUT restored the peak to 19.80 GiB.

The corrected same-prompt/same-seed full-video measurements are:

| Engine / route | End-to-end | DiT | Peak | Relevant comparison |
|---|---:|---:|---:|---:|
| Original 9/11 fair dense | 453.733 s | 406.664 s | 17.37 GiB | 1.000x |
| Original 9/11 split modality top-k 0.50 | **337.524 s** | **292.202 s** | 19.80 GiB | **1.344x vs original dense** |
| LoRA 6-step fair dense | 315.319 s | 264.546 s | 16.40 GiB | 1.000x |
| LoRA 6-step split modality top-k 0.50 | **240.540 s** | **194.318 s** | 19.87 GiB | **1.311x vs LoRA dense** |

Both six-frame catastrophe sheets passed.  Formal local WhisperX `large-v3`,
CUDA FP16 and forced Chinese alignment produced the same full transcript for
both corrected candidates: `城门快守不住了带伤员从北向撤离我留下等援军进城`.
The global CER was 0.04348, with `北巷` recognized as `北向`.  The three lines
remain in order and in the intended broad timeline.

This does **not** make the sparse route lossless.  The local attention call was
output-identical after sharing V, but the final MP4 differed from an earlier
run of the same sparse policy (video SSIM approximately 0.8454).  Diffusion is
sensitive to execution/allocation ordering and kernel nondeterminism, so only
Human continuous playback can authorize the acting, motion and audio quality.
The dense original-weight route remains unchanged and is still the
high-fidelity fallback.  Original and LoRA sparse routes require independent
authorization even though the SM89 implementation is shared.

Evidence:

- split/shared-V microbenchmark: `runtime/calibration/workload_routing_round12/split_modality_sequential_sharedv_seq100000.json`
- original corrected report/video/gates: `runtime/calibration/workload_routing_round12/original911_720p15_split_sequential_sharedv_sparse050/`
- LoRA corrected report/video/gates: `runtime/calibration/workload_routing_round12/lora6_720p15_split_sequential_sharedv_sparse050/`
- rejected overlapping-buffer report: `runtime/calibration/workload_routing_round12/original911_720p15_split_sharedv_sparse050/`
