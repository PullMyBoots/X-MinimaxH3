# LoRA no-forecast scheduler release validation (2026-08-19)

## Frozen product boundary

The reviewed INT8/Base scheduler remains a separate product domain.  A Base
V19 release selector or Base research override must never inject forecast
steps into the distilled LoRA trajectory.

LoRA uses `h3_lora_v1_no_forecast_round229`:

- the user selects `sampling_steps` in `[4, 8]`;
- every selected sigma position is a real Turbo/LoRA evaluation;
- `forecast_step_indices` is always empty and `forecast_allowed` is false;
- `acceleration` in `[0, 100]` changes only the adaptive per-step/per-layer
  Attention allocation;
- FL2VA and Ref2VA use the same scheduler contract, with workload geometry and
  visual-condition count included in the allocation context.

Full optimality-certificate replay is a release-test operation.  Online
requests retain cheap fail-closed trajectory and step/layer coverage checks,
but do not solve the certification problem twice.

## Automated evidence

The focused release suite passed on 2026-08-19:

```text
54 scheduler/contract/native-engine/API tests: PASS
12 ComfyUI connector tests: PASS
static/app.js syntax check: PASS
```

The tests cover all 4--8 LoRA step counts, the acceleration endpoints and
midpoint, absence of forecast steps, all 50 layers at every real step, Base
V19/LoRA routing isolation, Web/API serialization, ComfyUI fields, and a
checkpoint/preview/resume request retaining one identical schedule.

## Real RTX 4090 evidence

Environment: one RTX 4090 24 GiB, `fullspeed` host-memory profile, production
`./scripts/start.sh` service, hot task time excluding the one-time family
construction.

| Input mode | Geometry / duration | LoRA request | End-to-end | Artifact |
|---|---:|---:|---:|---|
| FL2VA first+last | 736x736 / 192 frames | 8 real, 0 forecast, acceleration 50 | 83.184 s | `output/67753ff7-9097-4892-9855-c477134de43f.mp4` |
| Ref2VA 3 images + 2 audios | 864x480 / 192 frames | 8 real, 0 forecast, acceleration 50 | 69.925 s | `output/24f3b9aa-df40-4c2d-ba26-d18a96398a5f.mp4` |

Both persisted job records identify
`scheduler_family=h3_lora_v1_no_forecast_round229` and contain all eight real
step indices with an empty forecast list.  These runs prove execution and
contract integration; Human review remains authoritative for subjective video
and audio quality.

## Formal checkpoint evidence

A Ref2VA LoRA task at 640x480 / 124 frames used the same 8-step,
acceleration-50 plan and stopped after step 3:

- checkpoint phase: 29.648 s;
- disposable four-step LoRA preview:
  `output/5300f078-3c5d-4d5a-ab5f-ce0b259cb0ba.checkpoint-preview.mp4`;
- formal state:
  `data/checkpoints/5300f078-3c5d-4d5a-ab5f-ce0b259cb0ba.pt`;
- after return to the queue: 628 MiB device memory, 0% GPU utilisation;
- resumed final output:
  `output/5300f078-3c5d-4d5a-ab5f-ce0b259cb0ba.mp4`;
- cumulative formal generation time after resume: 49.467 s.

The resume request regenerated the deterministic plan and required an exact
match for the sigma schedule, all actual-step indices, the complete Attention
action table, online guard identity/state, prompt digest, model variant,
geometry and seed.  It loaded `next_step_index=3`; the formal prefix was not
replayed.  The preview branch was disposable and did not mutate the persisted
formal latent.
