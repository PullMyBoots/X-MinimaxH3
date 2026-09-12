# V24 final completion audit

Audit status: **complete; final Human candidate selected**.
This audit evaluates the original release objective item by item.  It does not
convert automated checks or low-step smoke outputs into Human quality evidence.

## Requirement-by-requirement evidence

| Original requirement | Authoritative evidence | Finding |
|---|---|---|
| One final deployment framework | `h3serve/app.py` exposes one unified console, one generation endpoint, one queue and one download endpoint. `runs/http_service_matrix/report.json` records deployment mode `unified_console`. | Proven |
| FL2VA Base and FL2VA LoRA are connected | Real HTTP jobs resolved to `original` and `lora`; both succeeded and were downloaded. Exact maximum-geometry core runs also succeeded. | Proven |
| Ref2VA Base and Ref2VA LoRA are connected | Real multipart HTTP jobs with three images and two audios resolved to `reference` and `reference_lora`; both succeeded and were downloaded. Exact maximum-geometry core runs also succeeded. | Proven |
| Public range reaches 1080p×15s | Contract resolves the boundary to 1920×1088×362. `runs/max_service_matrix/report.json` and four ffprobe-verified outputs prove all four real-weight routes complete that geometry without OOM. | Proven |
| User sees only total steps and acceleration | OpenAPI/UI contract exposes Base 5–30, LoRA 4–10 and acceleration 0–100. V24 and LoRA schedulers internally compile the high-dimensional strategy. | Proven |
| Base and LoRA use valid distinct scheduling semantics | Maximum and HTTP reports show Base Actual/Forecast scheduling; both LoRA routes retain every distilled step and report zero Forecast. | Proven |
| Historical Human feedback constrains the final strategy | `round03/human_feedback_source.json` records no-contact door/background-drift and long-video hand/tool failures as hard rejects. The final structured-prompt review passed all candidates; C02 is selected because its handoff motion is better than C03 while retaining lower latency. | Proven |
| Multiple calibration experiments and videos are delivered | `candidate_registry.json`, `HUMAN_REVIEW.md` and the C01/C02/C03 720p15 videos preserve three audited research knots with measured 1.108–1.154× speedup over V009. Only C02 is compiled into the production service. | Proven |
| Release is reproducible and checked | 610 of 614 tests pass with 4 environment skips; 27 post-boundary focused tests pass; Python compile, JavaScript syntax, wheel build, clean-install import and 20 manifest/file hashes pass. A Linux-mirror console smoke returned HTTP 200 for home, health, options and OpenAPI, reported `0.7.0`, exposed FlashVSR as available, then shut down cleanly. | Proven |
| Human chooses the final candidate | Human passed all formal-prompt holdouts, reported only small differences, and selected the C02 tradeoff after C03 showed weaker edge flicker but worse key handoff. | **C02 selected** |

## Scope of the two real service matrices

- `runs/max_service_matrix/`: exact 1080p×15s physical-capability stress through
  the current Native hot engine, all four routes, real conditioning and real
  weights. Low steps bound cost; these are not quality candidates.
- `runs/http_service_matrix/`: real server, unified-console model entry, public
  multipart request, queue, status polling, result download and model exit at a
  small geometry. This proves service integration independently of maximum-load
  stability.

Together they cover both dimensions required by “all services up to
1080p×15s”: the maximum physical execution boundary and the complete public
service path. Neither replaces the Human review of the 20-step 720p15 knots.

## Closed release gate

The Human gate is closed.  The service has one immutable production surface,
`v24_final_c02_round2_trajectory_u7p00`.  V009/C01/C03 are offline evidence,
not request or service-level alternatives.  Operational rollback replaces the
whole policy by disabling V24 before process startup.  No UI, API or
model-weight change is required.
