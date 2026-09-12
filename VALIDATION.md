# Release validation

This document records reproducible checks for the public source tree. Generated
videos and model weights are deliberately excluded from Git.

## Validated platform

- OS: Linux x86-64 under WSL2
- GPU: NVIDIA GeForce RTX 4090 (SM89)
- Python: 3.10.20
- PyTorch: 2.13.0+cu130 (CUDA runtime 13.0)
- Service launch toolkit: CUDA 13.3; the host's default `nvcc` symlink remains 12.8
- NVIDIA driver: 610.88
- Runtime model store: external to the repository

## Commands

```bash
./setup.sh --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/pinned/vendor \
  --sparse-build-dir /path/to/compiled/sparge
./doctor.sh
bash -n setup.sh run.sh stop.sh doctor.sh scripts/*.sh
python -m compileall -q h3serve scripts tests
./test.sh
PYTHONPATH=integrations/comfyui python -m unittest \
  integrations/comfyui/tests/test_client.py
./doctor.sh --full
./scripts/build_release.sh
```

## Result ledger — 2026-08-29

- Shell syntax, Python compilation, JSON parsing and browser JavaScript syntax:
  passed.
- Unit/contract/runtime regression suite: **709 passed, 4 skipped, 0 failed**
  in 184.253 seconds.
- ComfyUI connector suite: **24 passed, 0 failed**. Both English workflows
  passed JSON parsing, dynamic input/output schema validation and four-file
  installer discovery alongside their Chinese equivalents.
- Full preflight: all 12 declared weight files passed byte-size and SHA-256
  checks; both pinned upstream revisions, all ten internal launcher profiles and the
  SM89 kernel smoke test passed. `end_to_end_runtime_ready` was `true`.
- Web/API: version 1.0.0 served from the Linux hot mirror on port 8091; health,
  model matrix, LoRA registry and no-cache index response passed.
- Real generation smoke tests (640x352, 56 frames, 24fps):

| Runtime | Steps | Generation time | MP4 SHA-256 |
|---|---:|---:|---|
| Base FL2VA INT8 | 5 | 9.524s | `0ef45b18e329eec8b2d8935436a5f9f5bde9c8261c849e91b79d5db3e453b0dd` |
| LightX2V FL2VA 4-step v1.1 | 4 | 8.784s | `d746a7ff4370ea4d717b31b71e84b34ab6f01def7c92b7d0388e12372d62a0c6` |
| LightX2V FL2VA 8-step v1.0 | 8 | 11.784s | `3b152459d5265ad1b263422e835e44e6d599ab84096b250cac5668edd96b1d18` |
| LightX2V Ref2VA 4-step v0.1 | 4 | 11.350s | `43cf39f1d98364a7bb518d7481ac96e3a7e9210bdd33ab45abe36c4769111dce` |

Cold model loading is excluded from the table. Test videos remain in the
ignored local `output/validation/` directory and are not part of the archive.

A valid release must continue to satisfy all of the following:

- no syntax, import or unit-test failures;
- exact model sizes, hashes and pinned upstream revisions pass preflight;
- the bilingual Web console and REST catalog respond from the Linux hot mirror;
- at least one real Base generation and each newly declared LightX2V task
  family can load and produce an output with installed weights;
- the release archive contains no model weights, task outputs, caches,
  credentials or local configuration. Historical calibration records may keep
  their original absolute provenance paths so their evidence is not rewritten.

Status: **passed**.

## Automatic resource and hard-RAM regression — 2026-08-31

The public console now exposes four model choices and privately routes the
detected GPU to an 8GB, 16GB or 24GB executor. The host-memory slider is backed
by cgroup v2 `memory.max`, includes child processes, disables swap for that
group and always reserves 6GiB outside H3.

Focused API, contract, memory-policy and execution-planner regression:
**76 tests passed, 0 failed**. Browser JavaScript syntax checks also passed.
The complete public-tree regression then passed **717 tests, 5 intentionally
skipped and 0 failed** in 127.800 seconds. The additional skip covers retained
human-review MP4 evidence that is deliberately excluded from the public archive.

Real RTX 4090 checks used Base, 5 steps, acceleration 95 and a short 360p job:

| Public choice | Private route | Hard H3 RAM limit | Result | End-to-end | cgroup OOM / kill |
|---|---|---:|---|---:|---|
| W4A8 FL2VA | `fl2va_w4a8_24gb` | 16GiB | passed | 23.120s | 0 / 0 |
| INT8 FL2VA | `fl2va_int8_24gb` | 32GiB | passed | 18.509s | 0 / 0 |

Both cgroups reached the selected `memory.max`, proving that the UI value was
an enforced process-tree ceiling rather than a cache hint. Model exit also
moved inherited TorchInductor workers back to the original scope and removed
the private cgroup successfully. See
[`docs/AUTOMATIC_RESOURCE_BUDGET_2026_08_31.md`](docs/AUTOMATIC_RESOURCE_BUDGET_2026_08_31.md).

## 16GB long-video regression — 2026-08-31

The Video-VAE planner now admits the complete-GPU-output graph only after
accounting for both clip-sized FP32 tensors used by decode and exact uint8
postprocessing, plus the physically measured service working set. When that
peak crosses the launcher ceiling it selects the byte-equivalent temporal host
sink before decode begins.

Both cases below used the logical 16GB INT8 launcher with a hard 15.25GiB Torch
allocator ceiling, Base weights, 5 declared steps and acceleration 95:

| Workload | Hard host limit | Result | End-to-end | CUDA peak | VAE route |
|---|---:|---|---:|---:|---|
| 720p × 15s (1280×736×362) | 11GiB | passed | 114.382s | 12,346MiB | `host_temporal_exact`, 17-frame chunk |
| 1080p × 15s (1920×1088×362) | 16GiB | passed | 276.938s | 14,804MiB | `host_temporal_exact`, 10-frame chunk |

Both outputs contain a 24fps H.264 video stream and AAC audio with a duration
of 15.084 seconds. The generated MP4 files remain outside the public source
tree. The focused canonical and public-tree regression suites each passed 47
tests with zero failures.

## 8/16/24GB resource matrix — 2026-09-01

`scripts/validate_release_resource_matrix.py` exercises the real HTTP service,
not a planner-only simulation. Each row switches the internal launcher, applies
the selected cgroup-v2 RAM ceiling, submits a complete Base generation, waits
for both VAEs and MP4 muxing, downloads the video and records CUDA/service-RAM
telemetry. The low-cost gate used five declared steps, acceleration 95 and a
one-second clip.

Result: **23/23 rows passed** — 18 FL2VA resolution/RAM rows and five real
single-image Ref2VA boundary rows.

The sealed `h3_release_resource_matrix_v2` report keeps all 23 rows in
`expected_rows` even after a targeted `--only` recheck. A row counts as passed
only when its job succeeded and its retained MP4 size exactly matches the HTTP
download receipt. The final public-tree regression passed **746 tests**, with
**4 intentional skips and 0 failures**, in 174.916 seconds; this includes the
byte-exact SM89 direct-FP8/Sparge ABI check.

| Weight/backend | Hard H3 RAM | Complete FL2VA resolutions | Ref2VA boundary | Maximum CUDA allocated/reserved |
|---|---:|---|---|---:|
| W4A8 8GB | 12GiB | 480p, 720p | 720p + one image | 6.087/6.193GiB |
| W4A8 16GB | 12GiB | 480p, 720p, 1080p | 1080p + one image | 9.260/9.326GiB |
| W4A8 24GB | 12GiB | 480p, 720p, 1080p | 1080p + one image | 14.671/14.740GiB |
| INT8 16GB | 24GiB | 480p, 720p, 1080p | 1080p + one image | 10.108/15.139GiB |
| INT8 24GB | 24GiB | 480p, 720p, 1080p | 1080p + one image | 10.108/15.141GiB |

Additional rows cover the measured host-memory knees at W4A8 8GB/22GiB,
W4A8 16GB/17GiB and INT8 16/24GB/32GiB. RAM changes only exact residency,
pinned-block coverage and copy scheduling. The resource endpoint now separates
anonymous pages from reclaimable file cache and reports cgroup OOM/kill counts.

## X-MinimaxH3 1.1.0 release-package regression — 2026-09-12

Version 1.1.0 refreshes the public tree from the current service implementation.
The release gate covers the SelfLift two-resolution trajectory, isolated fork
preview, full-film overlapping temporal final sampling, fixed audio-token
authority and creator-window level balancing, online and JSON long-video
creation, project deletion, automatic FL2VA face repair, separate first/final
  acceleration controls, per-window Ref2VA JSON reference replacement and
  inheritance, and the bilingual Web console.

Validation used the checked-in release directory itself:

```bash
bash -n setup.sh run.sh stop.sh doctor.sh test.sh scripts/*.sh
python -m json.tool RELEASE_MANIFEST.json
python -m json.tool models/manifest.json
node --check static/*.js
python -m compileall -q h3serve scripts tests integrations/comfyui
H3_NATIVE_SPARGE_BUILD_DIR=/path/to/validated/sparge-build \
  H3_SERVE_PYTHON=/path/to/h3-python ./test.sh
PYTHONPATH=integrations/comfyui python -m unittest discover \
  -s integrations/comfyui/tests -v
./scripts/build_release.sh
```

Results:

- release regression: **959 tests run, 4 skipped, 0 failed** in 146.860 seconds;
- ComfyUI connector: **25 tests run, 0 failed** in 0.520 seconds;
- shell syntax, Python compilation, JSON parsing, browser JavaScript syntax,
  version consistency and mandatory runtime-asset checks: passed;
- bundled YuNet detector, H3 kernel calibration grid and the CPython 3.11
  Block-Sparse Attention wheel matched the SHA-256 values in
  `RELEASE_MANIFEST.json`;
- the archive excludes `.env.local`, Git metadata, Python caches, model weights,
  task data, generated output and runtime directories.
- the extracted archive repeated the complete **959-test** release suite in
  **141.459 seconds** with the same 4 skips and zero failures; its extracted
  ComfyUI connector repeated **25/25** passes in **0.510 seconds**.

The full-model and real-video rows above are retained as hardware evidence from
version 1.0.0. This packaging regression did not rerun expensive model-weight
inference while the live service was in use; it validates that the current
implementation and its release/install boundary are internally consistent.
