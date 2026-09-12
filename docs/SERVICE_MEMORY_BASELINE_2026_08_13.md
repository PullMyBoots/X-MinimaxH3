# H3 service memory baseline — 2026-08-13

## Lossless residency update

The release now packs immutable tensors into exact-size CUDA-registered host
slabs, clones the two Video-VAE normalization vectors so they do not retain the
whole checkpoint mapping, and trims startup-only assembly arenas. These are
storage/lifetime changes only: no checkpoint value, model graph, sampler,
schedule, attention budget or output transport byte changed.

| Policy/run | Process-tree PSS | Result |
|---|---:|---|
| Old H3 + FlashVSR ready baseline | 65.123 GiB | historical comparator |
| Compact H3 + FlashVSR ready | **55.061 GiB** | 10.062 GiB / 15.5% lower |
| H3 full-speed ready, FlashVSR on demand | **51.118 GiB** | 96GB profile |
| Same profile, 720p × 15s full request | **58.031 GiB peak** | succeeded, no swap |
| 64GB compact (Qwen on demand), ready | **35.795 GiB** | compatibility profile |
| 96GB profile, 720p × 15s | **56.646 GiB peak** | validated, no swap |
| 64GB compact, 480p × 15s | **40.954 GiB peak** | validated, no swap |
| 64GB compact, 720p × 15s | **48.795 GiB peak** | validated, no swap |
| LoRA full-speed ready | 51.871 GiB | validated |
| LoRA dual-hot, ready | 56.533 GiB | validated ready state |
| H3 hot + FlashVSR 15s→1080p | **88.835 GiB peak** | excludes dual-hot below 128GB |
| Full-speed H3→upscale→restore | **73.141 GiB peak** | 96GB stage-exclusive profile |
| Compact H3→upscale→restore | **49.931 GiB peak** | 64GB stage-exclusive + sequential build |
| 64GB final 720p15→1920×1104→restore | **51.745 GiB peak** | 718.378s, no swap |
| Rejected minimum policy, ready | 7.656 GiB | not a supported tier |
| Rejected minimum, 360p × 0.92s request | **27.905 GiB peak** | too large for 24/32GB |

The deterministic 640×352×22-frame seed-8313 audit took 13.650 seconds in the
dual-hot policy and 13.167 seconds in the generation-hot policy. Decoded video
frame hashes and decoded audio sample hashes matched exactly. This establishes
that deferring FlashVSR has no H3 quality or latency penalty.

The final request-boundary host-cache release reduced full-speed 720p×15s
peak PSS to 56.646 GiB. Runtime was 451.261 seconds versus 449.674 seconds
before cleanup (+0.35%), and decoded video/audio hashes matched exactly. This
profile is therefore validated for the complete release workload envelope.

For the compact profile, 480p×15s completed in 133.061 seconds at 40.954 GiB
peak PSS. The 720p×15s probe also completed, remained exact and peaked at
48.795 GiB. The final continuous 1280×736×362 generation, 1920×1104 FlashVSR
upscale and H3 restore completed in 718.378 seconds at 51.745 GiB peak PSS with
no swap. Therefore this strategy belongs to the 64GB class and no 48GB tier is
exposed.

The same residency mechanics were audited on the LoRA route rather than assumed
from the original-weight engine. Generation-hot ready PSS was 51.871 GiB and
dual-hot ready PSS was 56.533 GiB. The deterministic short request took 9.050
and 8.722 seconds respectively, with identical decoded video/audio hashes.

The 64GB compact policy produced byte-identical decoded video/audio hashes for
the same deterministic request, but its first new-prompt request increased from
13.167 to 84.411 seconds because 13.5 GiB of packed Qwen weights are streamed
from the checkpoint instead of retained in RAM. Post-request H3 PSS was
36.065 GiB. This is a real lower-memory compatibility policy, not a full-speed
alias. Its completed 720p×15s generation probe establishes the full generation
envelope; sequential component assembly reduces cold-start overlap further.

An mmap/synchronous-copy experiment reduced idle PSS to 7.656 GiB, but even a
640×352×22-frame request peaked at 27.905 GiB and took 141.542 seconds. Its
decoded video/audio hashes still matched, proving mathematical correctness but
also proving that a 24GB host tier is infeasible and 32GB lacks responsible OS
headroom. The complete product cycle reaches 49.931 GiB even with stage
exclusivity, so the product exposes no tier below 64GB.

Evidence:

- `runtime/validation/memory/lossless_fullspeed_ready_state.json`
- `runtime/validation/memory/generation_hot_ready_state.json`
- `runtime/validation/memory/generation_hot_720p15_full_job.json`
- `runtime/validation/memory/generation_hot_720p15_cleanup_final.json`
- `runtime/validation/memory/compact48_ready_state.json`
- `runtime/validation/memory/compact48_480p15_final.json`
- `runtime/validation/memory/compact48_720p15_final.json`
- `runtime/validation/memory/lora_generation_hot_ready_state.json`
- `runtime/validation/memory/lora_fullspeed_ready_state.json`
- `runtime/validation/memory/minimum24_ready_state.json`
- `runtime/validation/memory/minimum24_short_task.json`
- `runtime/validation/memory/64gb_sequential_cold_start.json`
- `runtime/validation/memory/64gb_sequential_full_cycle_short.json`
- `runtime/validation/memory/64gb_720p15_to1080_full_cycle.json`

Scope: the final high-fidelity service on one RTX 4090, after both the Native
H3 engine and the persistent CPU-hot FlashVSR worker reported `ready`. Values
use binary GiB (`1024^3`) and distinguish host RAM from CUDA VRAM.

## Host RAM observed in the live service

| Component | Ready-state RSS | Ready-state PSS | Per-process VmHWM |
|---|---:|---:|---:|
| Native H3 server | 60.807 GiB | 60.661 GiB | 87.124 GiB |
| FlashVSR daemon | 3.957 GiB | 3.948 GiB | 8.455 GiB |
| Current process-tree total | **64.764 GiB** | **64.609 GiB** | n/a |

A later 1-second synchronous sample included all H3 worker children: the summed
RSS was 75.663 GiB, but it counted the same shared mappings once per worker.
The corresponding process-tree PSS was **65.123 GiB**, which is the primary
ready-state physical-RAM estimate used for capacity planning. The raw report is
`runtime/validation/memory/ready_state_sample.json`.

`VmHWM` is a lifetime high-water mark maintained separately by Linux for each
process. The two process values did not necessarily occur at the same instant
and must not be added and presented as a measured service-tree peak.

The startup policy is serial: H3 reaches `ready` before FlashVSR begins its CPU
preload. With the observed H3 ready-state RSS plus the FlashVSR preload HWM, a
conservative startup/idle simultaneous bound is **69.262 GiB**. Adding both
independent process HWMs yields **95.578 GiB**, but this is only a non-synchronous
extreme upper bound, not an observed system peak.

Thus the currently defensible host figures are: **65.123 GiB measured
synchronous ready-state PSS**, **69.262 GiB conservative startup bound**, and
**95.578 GiB non-synchronous extreme upper bound**. A full generation plus
upscale synchronous peak still requires sampling one real submitted job.

Raw live evidence:

- H3 PID 833168: `VmRSS=63761052 KiB`, `VmHWM=91355900 KiB`, no swap.
- FlashVSR PID 833921: `VmRSS=4148904 KiB`, `VmHWM=8865304 KiB`, no swap.
- Combined ready-state RSS: 67909956 KiB (64.764 GiB).

The PIDs are evidence identifiers for this run only and are expected to change
after restart.

## CUDA VRAM

- Ready and idle after both preloads: about **492 MiB** reported by NVML.
- FlashVSR 1152x640, 22 frames: **8517.4 MiB allocated** peak.
- FlashVSR 1152x640, 124 frames: **8930.9 MiB allocated** peak.
- Largest observed Native H3 release-calibration peak in the supported task
  envelope: **22.661 GiB allocated**. This is a task-specific historical peak,
  not idle use; normal validated routed profiles are typically lower.

H3 generation and FlashVSR upscaling are serialized, so their CUDA peaks must
not be added. The service's worst stage determines required VRAM. The release
target remains a 24 GiB RTX 4090.

## Release sizing conclusion

- 64 GiB RAM is below the measured 64.764 GiB ready-state process RSS and is
  not a safe supported configuration for the dual-hot policy.
- 80 GiB can run the observed ready state but leaves limited margin for WSL,
  filesystem cache, uploads, video buffers and transient allocations.
- **128 GiB host RAM is the supported dual-hot tier.** A 128GB Windows host may
  expose about 110GiB to WSL, which still exceeds the measured 88.835GiB
  15-second upscale peak plus the 8GiB operational reserve.
- The 96GB and 64GB tiers use exclusive stage residency: H3 is released before
  FlashVSR starts and restored after FlashVSR exits. It never overlaps the two
  measured peaks merely to avoid a reload.
- The 64GB tier additionally streams Qwen weights per request and assembles H3
  components sequentially. This preserves outputs but increases new-prompt and
  post-upscale recovery latency.

## Strict future measurement

For a full generation + upscale job, record the synchronous process-tree peak:

```bash
python scripts/record_service_memory.py \
  --pid "$(pgrep -f 'release/serve/server.py.*--port 8090' | head -n1)" \
  --interval 0.1 \
  --output runtime/validation/memory/full_job_memory.json
```

Start the command immediately before submitting the task, then press Ctrl-C
after completion. Unlike adding `VmHWM`, this samples H3 and every child process
at the same timestamps. Use `synchronous_peak.pss_gib` as the main system-RAM
number; summed RSS can double-count shared pages in forked workers.
