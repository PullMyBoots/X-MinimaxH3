# X-MinimaxH3 User Guide

## 1. Start the console

After setup, run:

```bash
./run.sh
```

Open `http://127.0.0.1:8090`. The process starts as a lightweight console and
loads H3 only after you select a model. The page reports actual loading
stages and progress. Use `./stop.sh` to stop the backend; closing a browser tab
does not stop the service.

## 2. Four model choices and automatic resources

Choose W4A8 or INT8, then FL2VA or Ref2VA. The internal VRAM executor is not a
user setting:

| Weights | Minimum VRAM | First generation | H3 second sampling |
| --- | ---: | --- | --- |
| W4A8 | 8GB | up to 720p × 15s | up to 1080p |
| INT8 | 16GB | up to 1080p × 15s | up to 1440p |

Set the maximum host RAM for the complete H3 process tree on the same page.
W4A8 can be lowered experimentally to 12GiB and INT8 to 24GiB; their validated
floors remain 16GiB and 32GiB. The slider keeps 6GiB outside H3 for
the operating system. Linux enforces the selection with cgroup v2; the backend
then derives pinned-memory and GPU block residency once while loading the
model. It does not run a resource router on every DiT step.

FL2VA supports text-only, first-frame, last-frame and first+last-frame tasks.
Ref2VA accepts up to nine images, three videos and three independent audio
references. Reference videos may total at most 15 seconds; embedded audio is
not used as a voice reference.

The queue must be empty before changing service family, weights or RAM budget.
Base and the selected LoRA share one hot session and switch per request.

## 3. Prompts

The console supports a storyboard editor and a free-form prompt. Free-form text
is sent verbatim to H3 and never invokes an external model automatically. A
recommended H3 structure is:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: N/A
```

Dialogue may use `<d>[English]dialogue</d>`. Ref2VA references are addressed as
`<Picture 1>`, `<Video 1>` and `<Audio 1>`, matching the upload port numbers.

## 4. Sampling steps and acceleration

- `sampling_steps` is the full trajectory length: 5–30 for Base and 4–10 for LoRA.
- `acceleration` is a continuous compute budget from 0 to 100. Zero is the
  dense reference endpoint; 75 is the current human-reviewed release knee;
  values above 75 explicitly trade more quality risk for speed.

For Base, the scheduler jointly allocates actual DiT evaluations, forecast
locations and per-step/per-layer attention. Distilled LoRA trajectories do not
receive unvalidated forecast steps; their acceleration only changes attention
allocation. Composition, causal interaction, audio and terminal-detail guards
remain internal non-disableable constraints.

## 5. LoRA profiles

Settings → LoRA scans `models/loras/` recursively. Built-in profiles are:

| LoRA | Family | Calibrated steps |
| --- | --- | ---: |
| Larry Turbo v4-600 EMA | FL2VA / Ref2VA | 4–8, default 6 |
| LightX2V FL2VA Turbo v1.1 768p | FL2VA | 4 |
| LightX2V FL2VA Turbo v1.0 768p | FL2VA | 8 |
| LightX2V Ref2VA Turbo v0.1 | Ref2VA | 4 |

LightX2V FL2VA and Ref2VA adapters are not interchangeable. Switching the
global LoRA requires an empty queue and rebuilds the hot model. The sampling
control automatically moves to the adapter's calibrated step count.

## 6. Checkpoint previews

A checkpoint task only asks where the formal trajectory should pause. Preview
resolution and preview steps are global settings (default: 360p and four
steps). The backend atomically saves formal latents and scheduler state, then
releases the GPU execution slot. The preview is disposable:

- Continue resumes the same formal trajectory without replaying earlier steps.
- Discard deletes the checkpoint and preview.
- Clearing latent cache keeps MP4 files but removes resume/second-sampling data.

## 7. Native H3 second sampling

Generate low-resolution source cards first, then select H3 second sampling on
a successful task. It reuses clean AV latents, the original prompt and original
references; it is not a conventional image upscaler applied to an MP4.

Controls are target resolution, one to eight actual DiT steps, four redraw
strengths, and the same acceleration budget. Stronger redraw allows larger
identity, detail and motion changes and is not monotonically better. For faces,
lip motion or exact object geometry, start with Preserve/Standard and more
steps rather than Strong with very few steps.

## 8. Task management

The task page refreshes CPU, host memory, GPU load, VRAM and power every second.
Queued tasks can be reordered; active tasks can be cancelled; completed tasks
can be played, downloaded, second-sampled or deleted. Deleting a record also
deletes uploads, latent state and output media.

## 9. ComfyUI

The optional connector calls the HTTP service and does not load a second H3
model inside ComfyUI:

```bash
runtime/venv/bin/python integrations/comfyui/install_local.py /path/to/ComfyUI
integrations/comfyui/start_comfyui.sh /path/to/ComfyUI
```

Start X-MinimaxH3 and select the service family before opening the matching
example workflow under `integrations/comfyui/example_workflows/`.

## 10. REST API

The machine-readable contract is `GET /openapi.json`. Important endpoints are
`/healthz`, `/readyz`, `/api/v1/engine`, `/api/v1/generations`,
`/api/v1/jobs`, job resume, job second sampling and latent-cache cleanup.

Binding to a non-loopback address requires `H3_SERVE_API_KEY`. The service does
not provide TLS or tenant isolation; use a secure reverse proxy for network use.

## 11. Troubleshooting

- Console keeps loading: inspect the server's model-loading stage and run
  `./doctor.sh`.
- WSL imports are slow: keep the Git checkout and task data on Windows, but let
  `./run.sh` use its Linux-native execution mirror. Keep weights on native Linux
  storage where possible.
- VRAM is full while power is low: Qwen, VAE and CPU→GPU transfer stages are
  not DiT compute. A short drop is normal; a long stall is not.
- Cancellation appears stuck: cancellation is checked at safe DiT layer
  boundaries. Run `./stop.sh` if an obsolete process owns the port.
- `./doctor.sh --full` is slow because it hashes every large weight. Use the
  default quick check for routine startup diagnostics.
