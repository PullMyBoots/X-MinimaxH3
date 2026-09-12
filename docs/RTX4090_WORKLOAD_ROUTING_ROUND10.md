# RTX 4090 workload routing — Round 10

Date: 2026-08-12  
Status: measured implementation; sparse route remains quality-gated

## Outcome

The Native runtime can now switch a single loaded DiT between exact dense
SageAttention and request-scoped SpargeAttention without rebuilding or
reloading the model.  The request records its exact packed-token count and the
selected attention budget.  A two-request real-weight smoke and a four-request
cross-over run both completed in one hot Session.

This is an internal mechanism.  It does not expose routing decisions as a
creative control and it is not enabled as the release default yet.

## Same-session performance

Engine: Larry LoRA, six real evaluations, CUDA INT8 ConvRot, Block offload,
8,192-token MLP chunks, 288 Video-VAE tiles, regionally compiled Video-VAE
feed-forward modules.  Dense and sparse members of each pair use identical
prompt, seed, canvas and frame count.

| Workload | Packed tokens | Attention | Total | Total minus text encode | DiT | Video VAE | Peak allocated |
|---|---:|---|---:|---:|---:|---:|---:|
| 720p, 10.125 s | 67,368 | dense | 174.748 s | 172.963 s | 139.825 s | 26.958 s | 12.610 GiB |
| 720p, 10.125 s | 67,368 | top-k 0.50 | 140.380 s | 140.378 s | 107.294 s | 26.913 s | 13.157 GiB |
| 480p, 15.083 s | 44,895 | top-k 0.50 | 86.377 s | 84.763 s | 62.331 s | 17.906 s | 9.944 GiB |
| 480p, 15.083 s | 44,895 | dense | 98.593 s | 98.592 s | 76.009 s | 17.940 s | 9.944 GiB |

Prompt-cache-normalized end-to-end speedups are **1.232x** at 720p10 and
**1.163x** at 480p15.  DiT-only speedups are **1.303x** and **1.219x**.
The attention route changes only the DiT phase; the matching VAE times are a
useful negative control.

## Quality gate

The acceptance prompt contains three Mandarin lines in explicit time windows,
two recurring speakers and large continuous actions.  The original automated
report used Faster-Whisper large-v3 directly with CPU INT8, beam size 5, no VAD
and no WhisperX alignment.  Six requested frames were extracted from every
video before any similarity metric was considered.  Human continuous playback
subsequently overruled the original automated 480p15 dialogue verdict.

| Workload | Dense global CER | Sparse global CER | Dialogue verdict | Six-frame preliminary gate |
|---|---:|---:|---|---|
| 720p10 | 0.000 | 0.000 | PASS; all three lines in their windows | PASS |
| 480p15 | 0.043 | 0.217 | All requested dialogue is present by Human review; the sparse opening contains a reverse-tape-like non-speech artifact | PASS |

The old direct Faster-Whisper path hallucinated “谢谢大家” over non-speech audio;
WhisperX 3.7.4 large-v3, GPU FP16, VAD and Chinese forced alignment did not
produce that text.  Its VAD instead excluded the sparse opening that Human
identified as a reverse-tape-like artifact.  ASR therefore did not establish a
dialogue failure, while Human review did establish a real opening audio
artifact.  Long timeline density can still be more fragile even when its packed
sequence is shorter.  Human continuous-playback review remains the final audio
and visual authority; ASR measures dialogue evidence and cannot certify
non-speech sound quality.

## Routing consequence

For now:

1. exact dense attention remains the universal fallback;
2. top-k 0.50 is a measured candidate only for the full 720p landscape bucket
   at ten seconds or longer;
3. 480p15 stays dense because Human review found an opening audio artifact in
   the sparse candidate, not because of the obsolete “谢谢大家” ASR result;
4. other aspect ratios and first/last-frame conditioning remain unvalidated and
   fail closed to dense;
5. an automatic release route must depend on at least spatial-token count,
   latent-frame count, condition count and engine/preset—not just the user-facing
   resolution label.

## Evidence

- Performance report:
  `runtime/calibration/workload_routing_round10/crossover/round10_crossover_hot_session.json`
- Whisper reports:
  `runtime/calibration/workload_routing_round10/whisper/`
- Corrective WhisperX reports:
  `runtime/calibration/workload_routing_round10/whisperx_large_v3_fp16/`
- Contact sheets:
  `runtime/calibration/workload_routing_round10/contact_sheets/`
- Videos:
  `runtime/calibration/workload_routing_round10/crossover/`
- One-session route smoke:
  `runtime/calibration/workload_routing_round10/routed_smoke/round10_routed_smoke_hot_session.json`

The full CPU regression after implementation was 77 tests passing with four
strict release gates skipped by design.

## Exact-math VRAM utilization experiment

The pruned base stores 369.415 MiB per DiT block and Larry adds about
14.865 MiB of LoRA tensors per block.  A common 480p5 LoRA6 request was run in
one Session with 0, 8, 16, 24, 32 and 36 prefix blocks permanently resident.

| Resident blocks | DiT H2D | DiT denoise | Total | Peak allocated |
|---:|---:|---:|---:|---:|
| 0 | 0.241 s | 17.877 s | 28.301 s | 6.603 GiB |
| 8 | 0.367 s | 17.660 s | 25.949 s | 7.169 GiB |
| 16 | 0.489 s | 17.658 s | 26.092 s | 10.192 GiB |
| 24 | 0.657 s | 17.665 s | 26.367 s | 13.215 GiB |
| 32 | 0.787 s | 17.670 s | 26.453 s | 16.239 GiB |
| 36 | 0.883 s | 17.671 s | 26.580 s | 17.750 GiB |

All six decoded RGB streams have the same SHA-256 and all six decoded PCM
streams have the same SHA-256, confirming exact output equality.  The apparent
0-to-8 denoise change comes from the first request's first-kernel warm-up: the
remaining step times converge to about 2.935 seconds for every residency
count.  Double-buffered pinned-host prefetch therefore already hides block
transfer behind compute.  Extra residency does not improve steady-state DiT
time and only increases initial H2D and memory use.  The release planner should
keep `resident_block_count=0`; spare VRAM remains safety margin for activations,
conditioning and concurrent library work rather than a latent performance
opportunity.

Evidence:
`runtime/calibration/workload_routing_round10/resident_sweep_480p5/round10_resident_hot_session.json`.

## First/last-frame route

Frames 0 and 242 of the dense 720p10 reference were used as first/last inputs
for a real FL2AV pair.  The request contains 70,939 packed tokens.

| Attention | Total | Total minus first visual-text encode | DiT | Peak |
|---|---:|---:|---:|---:|
| dense | 199.902 s | 187.668 s | 152.726 s | 12.611 GiB |
| top-k 0.50 | 151.321 s | 151.300 s | 116.425 s | 13.732 GiB |

Normalized end-to-end speedup is **1.240x** and DiT speedup is **1.312x**.
Both full transcripts contain the three requested lines in order.  Whisper
rendered the sparse transcript in traditional characters, so raw code-point
CER is not a valid speech error for this pair; after script-equivalent
normalization both transcripts match the target.  Six-frame inspection passed.
Input-to-output anchor SSIM was 0.697/0.560 for dense first/last and
0.693/0.582 for sparse first/last; this is diagnostic only and shows no
sparse-specific loss of endpoint adherence.

Evidence:
`runtime/calibration/workload_routing_round10/fl2av_720p10/`.

## Original-weight conservative route

The original-weight balanced schedule (9 real / 11 forecast evaluations) was
tested on the complex 720p10 dialogue task with top-k 0.75:

| Attention | Total | Total minus text encode | DiT | Peak | Whisper CER |
|---|---:|---:|---:|---:|---:|
| dense | 250.376 s | 248.590 s | 212.745 s | 12.618 GiB | 0.000 |
| top-k 0.75 | 225.384 s | 225.382 s | 192.107 s | 13.093 GiB | 0.000 |

Normalized end-to-end speedup is **1.103x** and DiT speedup is **1.107x**.
All line timestamps agree closely and the six-frame preliminary gate passed.
The conservative route therefore remains distinct from LoRA's more aggressive
top-k 0.50 route.

Evidence:
`runtime/calibration/workload_routing_round10/original911_720p10/`.

## Fail-closed implementation

`NativeSessionFactory` can optionally construct one request-routed attention
backend and a planner containing the dated review profiles.  It was exercised
with the real LoRA checkpoint: preflight passed, the assembled DiT contained
`RequestRoutedSpargeAttentionBackend`, and the planner was experimental-aware.
The option is disabled by default and requires both a pinned optional install
and `H3_NATIVE_REVIEW_SPARSE=1`.

The review profiles permit only:

- LoRA6 T2AV, exact 920-token spatial grid, 72–107 latent frames;
- LoRA6 FL2AV, exact 920-token spatial grid and 72 latent frames;
- original 9/11 T2AV, exact 920-token spatial grid and 72 latent frames.

One-frame conditioning, other aspect ratios/presets, and the Human-rejected
480p15 sparse case select the existing dense plan.  Other original/LoRA step presets receive
an explicit exact mechanical plan instead of being sent through an uncalibrated
latency model.

## NVENC screening

System FFmpeg can use the RTX 4090 NVENC encoder, but the pinned PyAV wheel used
by the in-process atomic muxer reports `avcodec_open2(h264_nvenc)` as not
implemented.  A system-FFmpeg raw-frame prototype encoded the 720p10 sample in
1.288 seconds versus 1.56–1.72 seconds for in-process x264, with comparable
diagnostic SSIM (0.9927 versus 0.9923).  The sub-second saving is too small to
justify a new system-FFmpeg/process/pipe dependency, while frame conversion and
atomic publication still dominate the measured 4–5 second mux phase.  The
release therefore keeps libx264.
