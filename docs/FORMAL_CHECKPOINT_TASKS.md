# Formal checkpoint jobs

## Product contract

A generation request selects its lifecycle before entering the queue:

- `execution_mode=complete` runs the formal trajectory to its final video.
- `execution_mode=checkpoint` stops after `checkpoint_step` sigma positions.

The checkpoint position is one-based. For the Base route it counts all 20
formal sigma positions, including both real DiT evaluations and forecast
positions. For a LoRA route it counts the selected 4–8 distilled steps. A
checkpoint must stop before the final position.

`checkpoint_retain` and `checkpoint_preview` are independent:

- retaining persists a resumable formal state;
- previewing runs a disposable LoRA fast-finish branch and decodes an MP4;
- preview tensors never replace the formal tensors;
- when preview is requested without retention, the service removes the formal
  checkpoint after the preview is safely published.

## State boundary

`formal_sampler_checkpoint_v1` contains the minimum state needed to continue
the same mathematical trajectory without replaying its prefix:

- next noisy video and audio latents;
- the complete original sigma schedule and next global step index;
- Base RES previous-x0 tensors and source sigma;
- directional forecast anchors, tail residuals and step records;
- seed, canvas, frame count, engine variant, actual-step schedule and prompt
  digest used to reject an incompatible resume.

The file is first written to a sibling `.tmp` path and atomically renamed. On
resume the scheduler uses the untouched suffix of the original sigma list and
preserves global step indices. It does not construct a new shortened sigma
schedule.

After the checkpoint result is returned, the request's DiT block executor,
CUDA latents, preview tensors and host staging allocator are released. The
shared model session, immutable pinned weights and bounded reusable service
caches remain hot. The stopped job owns no GPU execution slot and a resume is
a new queue action, so another job can execute between checkpoint and resume.

## Interfaces

The Web studio exposes `直接跑完` and `断点任务`. A checkpointed task card can
show its optional preview and enqueue formal continuation.

The HTTP API adds:

- checkpoint request fields documented by `GET /openapi.json`;
- status `checkpointed` and a `checkpoint` result object;
- `POST /api/v1/jobs/{id}/resume`.

The bundled ComfyUI connector adds four nodes:

- `H3 Serve · FL2VA断点任务`;
- `H3 Serve · Ref2VA断点任务`;
- `H3 Serve · 查看断点预览`;
- `H3 Serve · 恢复断点任务`.

## RTX 4090 smoke evidence (2026-08-18)

A real 360p/3-second Base 9/11 request stopped after position 6, a separate
LoRA request completed, and the Base request then resumed. The resumed output
and an uninterrupted same-prompt/same-seed run were identical at every checked
surface:

- complete MP4 SHA-256: `e6d71c2f9b3bcaac442be3bcc691a1f9a87426269752455a83f8aa17998a5e6e`;
- decoded video-frame stream digest: `188d6e609f673c626537af4c10d8cb9184883ab1c48a63654b547af09a57ade6`;
- decoded PCM digest: `d0723de8207f8da6a0f289bb2fd8ddaf8d0392e5d36aba01e9c9438db30c0aab`.

The prefix took 7.873 seconds, the complete checkpoint/resume lifecycle took
17.774 seconds, and the uninterrupted control took 13.405 seconds. This small
smoke is a correctness and lifecycle test, not a performance baseline: disk
serialization and a second request boundary intentionally add overhead. The
checkpoint file was 113,761,869 bytes. Idle service VRAM returned to roughly
550–566 MiB, and the intervening LoRA job completed in 7.078 seconds.

A second real smoke requested a two-step LoRA preview without retaining the
formal checkpoint. It published a 640×352 MP4 with video and audio, transitioned
to `checkpointed`, automatically deleted the checkpoint file, and again
returned to roughly 550 MiB idle VRAM.

The connector was also loaded by the repository's real ComfyUI 0.30.0 process;
all four node definitions appeared in `/object_info`.

## Current limits

- Segment-cache research state is deliberately rejected by formal checkpoint
  serialization. Released directional forecast and dense routes are covered.
- Checkpoint files grow with latent sequence length and forecast history; a
  720p long-video state can require gigabytes of disk. Deleting the task record
  deletes its service-owned checkpoint.
- A resume may reuse shared prompt/reference caches, but correctness does not
  depend on them. Text or reference conditioning can be recomputed; formal DiT
  prefix steps are never replayed.
