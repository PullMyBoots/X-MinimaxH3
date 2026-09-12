# H3 Serve Connector for ComfyUI

**English** · [简体中文](README.md)

This optional connector lets ComfyUI submit jobs to an independently running
X-MinimaxH3 service. ComfyUI does not load another H3 model: model residency,
the serial GPU queue, acceleration scheduling, resource profiles and native H3
second sampling remain inside the service on port 8090.

## Install

From this directory, run:

```bash
python install_local.py /path/to/ComfyUI
```

The installer links the connector into `custom_nodes/` and installs all four
example workflows into `user/default/workflows/`. Restart ComfyUI after
installation.

## Required startup order

1. Start X-MinimaxH3 with `./scripts/start.sh` (or the documented release
   launcher).
2. Open <http://127.0.0.1:8090>, select the FL2VA or Ref2VA engine and wait for
   **Service ready**.
3. Start ComfyUI and open the workflow matching the selected engine family.

ComfyUI does not switch the resident FL2VA/Ref2VA engine. The generation node
may select **Base weights** or **LoRA Turbo** inside the already loaded family.
If no engine is selected, or it is still loading, the connection node rejects
the job instead of silently selecting a different model.

If ComfyUI is used only as an H3 client, start it in CPU mode so it does not
consume the generation GPU:

```bash
./start_comfyui.sh /path/to/ComfyUI
```

## Example workflows

English:

- `example_workflows/H3_Serve_FL2VA_First_Last_EN.json` — text-only, first
  frame, last frame, or first-and-last-frame FL2VA generation.
- `example_workflows/H3_Serve_Ref2VA_Multi_Reference_EN.json` — Ref2VA with up
  to nine pictures, three reference videos and three independent audio
  references.

Simplified Chinese:

- `example_workflows/H3_Serve_FL2VA_First_Last.json`
- `example_workflows/H3_Serve_Ref2VA_Multi_Reference.json`

The English and Chinese workflows call the same HTTP contract and backend.
Only labels, choices and default prompt text differ; language selection does
not change numerical inference or performance.

## Generation controls

- `sampling_steps` is the total sampling trajectory: Base accepts 5–30 and
  LoRA accepts 4–10.
- `acceleration` is the continuous 0–100 compute budget control. `0` is the
  all-real-step, dense-Attention reference endpoint. Higher values let the
  scheduler allocate fewer real DiT evaluations and/or smaller per-layer
  Attention budgets while preserving its mandatory quality guards.
- `model_variant` selects **Base weights** or **LoRA Turbo** without loading a
  second service.
- `preview_mode=On` stops the formal trajectory at `preview_step`, saves a
  low-cost preview and releases GPU execution. Use `checkpoint_action` to
  continue or discard that same job. Preview resolution and preview steps are
  configured globally in the 8090 console.

Generation duration accepts 1–60 seconds. Requests above the physical native
window are transparently executed as protected joint audio/video latent
continuations. Ordinary H3 `[Shot N] At MM:SS` timestamps and plain time ranges
are localized by the service; there is no separate long-video workflow and no
decode between windows. Intermediate checkpoint previews are currently
unavailable above 15 seconds, so keep `preview_mode=Off` for long requests.

The connector forwards the prompt as one unmodified `STRING`; it does not rewrite
dialogue, append soundtrack instructions, or compile storyboards. Write the complete
H3 prompt, including `integrated_multimodal_description`, `overall_soundscape`, and
`non_diegetic_music`, when those fields are needed.

## FL2VA and Ref2VA inputs

FL2VA exposes optional `first_frame` and `last_frame` inputs. Leave both empty
for text-to-video generation.

Ref2VA exposes `Picture 1`–`Picture 9`, `Video 1`–`Video 3`, and `Audio 1`–
`Audio 3` directly. Their numbers are the same labels used in prompts, such as
`<Picture 1>` and `<Audio 1>`. Connect core ComfyUI `Load Image`, `Load Video`
and `Load Audio` outputs directly. Embedded audio inside reference videos is
not used; provide a separate audio reference when voice or sound identity is
required. Reference resizing follows the global 8090 service settings and
preserves aspect ratio and duration.

## Native H3 second sampling

Connect the generation node's `final_video` output to the English
`H3 Serve · H3 second sampling` node. It exposes:

- target resolution (`720p`, `1080p`, or `1440p` when admitted by the active
  resource profile);
- 1–8 real second-sampling steps;
- four redraw-strength presets;
- the same 0–100 acceleration control.

Second sampling is fixed to Base weights, Simple scheduling and SA Solver. It
requires the clean H3 AV latent retained by the source service job; an external
MP4 cannot replace that state. The source job must therefore still exist and
its cache must not have been cleared. The node downloads the resulting MP4 to
`ComfyUI/output/h3_serve/` and previews it directly without a second encode.

## Outputs and cancellation

Generation and second-sampling nodes are output nodes. They download the
service-produced MP4 and display it directly; do not attach ComfyUI `SaveVideo`
unless a separate re-encode is intentional. Interrupting a running ComfyUI
prompt sends a cancellation request to H3 Serve.

## HTTP endpoints

The connector uses the shared service endpoints for readiness, options, job
submission, polling, cancellation, checkpoint continuation, second sampling
and video download. The machine-readable contract is available at
<http://127.0.0.1:8090/openapi.json> while the service is running.

## Common failures

- **No engine selected / engine still loading:** enter the appropriate engine
  in the 8090 console and wait until it reports ready.
- **Workflow family does not match the active service:** open the FL2VA
  workflow for FL2VA or the Ref2VA workflow for Ref2VA.
- **Second sampling cannot recover the source job:** keep the direct VIDEO
  connection from the H3 generation node and do not delete the source job or
  clear its latent cache first.
- **Checkpoint is awaiting a choice:** select **Continue generation** or
  **Discard generation** in the same creator node, then queue it again.
