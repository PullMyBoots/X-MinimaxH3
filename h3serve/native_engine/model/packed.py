"""Compact T2VA/FL2VA packed-token layout for batch-one H3 inference.

The equations are independently implemented from the MiniMax H3 layout
contract. SGLang's Apache-2.0 H3 packed-sequence implementation was used as a
cross-check; no SGLang runtime dependency is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

SegmentKind = Literal["text", "condition", "ref_audio", "audio", "video"]
_FRAME_PATTERN = (1, 4, 4, 4, 4)
_FRAME_SCALE = 5.0 / 3.0


@dataclass(frozen=True, slots=True)
class PackedSegment:
    kind: SegmentKind
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(slots=True)
class PackedLayout:
    """CPU-resident structural tensors reusable across all denoise steps."""

    segments: tuple[PackedSegment, ...]
    position_ids: torch.Tensor
    video_positions: torch.Tensor
    video_update_mask: torch.Tensor
    audio_positions: torch.Tensor
    audio_update_mask: torch.Tensor
    text_positions: torch.Tensor
    signature: tuple
    # Request-local device tensors.  A layout is reused across all denoise
    # steps and discarded afterwards, so these avoid repeated H2D/trigonometry
    # without retaining GPU memory between jobs.
    device_video_update_mask: torch.Tensor | None = None
    device_audio_update_mask: torch.Tensor | None = None
    device_video_condition_rows: torch.Tensor | None = None
    device_audio_condition_rows: torch.Tensor | None = None
    device_video_condition_embeddings: torch.Tensor | None = None
    device_audio_condition_embeddings: torch.Tensor | None = None
    device_rope_table: torch.Tensor | None = None

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    def segment(self, kind: SegmentKind, *, last: bool = False) -> PackedSegment:
        matches = [segment for segment in self.segments if segment.kind == kind]
        if not matches:
            raise KeyError(kind)
        return matches[-1] if last else matches[0]


def _axis_grid(size: int, patch: int, square_root_area: float) -> torch.Tensor:
    count = size // patch
    ratio = size / square_root_area
    return (
        torch.arange(count, dtype=torch.float64) * (ratio / count)
        + (1.0 - ratio) / 2.0
    ) * 32.0


def _frame_grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    area = math.sqrt(height * width)
    h_axis = _axis_grid(height, 2, area)
    w_axis = _axis_grid(width, 2, area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    return torch.stack((hh.flatten(), ww.flatten()), dim=-1), w_axis


def _temporal_grid(count: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_SCALE * _FRAME_PATTERN[index % 5] for index in range(count)],
        dtype=torch.float64,
    )
    return origin + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def _temporal_span(count: int) -> float:
    return float(sum(_FRAME_SCALE * _FRAME_PATTERN[index % 5] for index in range(count)))


def build_fl2va_layout(
    *,
    text_length: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    audio_frames: int,
    keyframe_indices: Sequence[int] = (),
    output_frame_count: int | None = None,
    target_time_offset: float = 0.0,
) -> PackedLayout:
    """Build `[text | first/last conditions | audio | video]`.

    ``keyframe_indices`` accepts only ``0`` and ``-1``. An empty sequence is
    the text-to-audio-video layout. H3 patch dimensions require even latent
    height and width.
    """

    positive = (text_length, latent_frames, latent_height, latent_width, audio_frames)
    if any(value <= 0 for value in positive):
        raise ValueError("all packed-layout dimensions must be positive")
    if latent_height % 2 or latent_width % 2:
        raise ValueError("latent height and width must be divisible by two")
    target_time_offset = float(target_time_offset)
    if not math.isfinite(target_time_offset) or target_time_offset < 0.0:
        raise ValueError("target_time_offset must be finite and non-negative")
    if any(index not in (0, -1) for index in keyframe_indices):
        raise ValueError("only first-frame (0) and last-frame (-1) anchors are supported")
    if len(set(keyframe_indices)) != len(keyframe_indices):
        raise ValueError("duplicate keyframe anchors are not allowed")
    if keyframe_indices and (output_frame_count is None or output_frame_count <= 0):
        raise ValueError("output_frame_count is required for keyframe conditioning")
    if keyframe_indices:
        assert output_frame_count is not None
        resolved = [0 if index == 0 else output_frame_count - 1 for index in keyframe_indices]
        if len(set(resolved)) != len(resolved):
            raise ValueError("keyframe anchors resolve to the same output frame")

    frame, width_axis = _frame_grid(latent_height, latent_width)
    frame_rows = int(frame.shape[0])
    segments: list[PackedSegment] = []
    positions: list[torch.Tensor] = []
    video_positions: list[torch.Tensor] = []
    video_updates: list[torch.Tensor] = []
    offset = 0

    text_grid = torch.zeros(text_length, 3, dtype=torch.float64)
    text_grid[:, 0] = torch.arange(text_length, dtype=torch.float64)
    segments.append(PackedSegment("text", offset, offset + text_length))
    positions.append(text_grid)
    text_positions = torch.arange(offset, offset + text_length)
    offset += text_length

    target_t = _temporal_grid(
        latent_frames,
        float(text_length) + target_time_offset,
    )
    for anchor in keyframe_indices:
        condition = torch.empty(frame_rows, 3, dtype=torch.float64)
        condition[:, 0] = target_t[0] if anchor == 0 else target_t[-1]
        condition[:, 1:] = frame
        segments.append(PackedSegment("condition", offset, offset + frame_rows))
        positions.append(condition)
        video_positions.append(torch.arange(offset, offset + frame_rows))
        video_updates.append(torch.zeros(frame_rows, dtype=torch.bool))
        offset += frame_rows

    audio_rows = audio_frames * 2
    audio_grid = torch.zeros(audio_rows, 3, dtype=torch.float64)
    audio_grid[:, 0] = (
        float(text_length)
        + target_time_offset
        + torch.arange(audio_frames, dtype=torch.float64)
    ).repeat(2)
    audio_grid[:audio_frames, 2] = width_axis[0]
    audio_grid[audio_frames:, 2] = width_axis[-1]
    segments.append(PackedSegment("audio", offset, offset + audio_rows))
    positions.append(audio_grid)
    audio_positions = torch.arange(offset, offset + audio_rows)
    offset += audio_rows

    target_rows = latent_frames * frame_rows
    video_grid = torch.empty(latent_frames, frame_rows, 3, dtype=torch.float64)
    video_grid[:, :, 0] = target_t[:, None]
    video_grid[:, :, 1:] = frame[None]
    segments.append(PackedSegment("video", offset, offset + target_rows))
    positions.append(video_grid.reshape(target_rows, 3))
    video_positions.append(torch.arange(offset, offset + target_rows))
    video_updates.append(torch.ones(target_rows, dtype=torch.bool))

    return PackedLayout(
        segments=tuple(segments),
        position_ids=torch.cat(positions),
        video_positions=torch.cat(video_positions),
        video_update_mask=torch.cat(video_updates),
        audio_positions=audio_positions,
        audio_update_mask=torch.ones(audio_rows, dtype=torch.bool),
        text_positions=text_positions,
        signature=(
            text_length,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            (
                tuple(int(index) for index in keyframe_indices)
                if target_time_offset == 0.0
                else (
                    tuple(int(index) for index in keyframe_indices),
                    ("target_time_offset", target_time_offset),
                )
            ),
        ),
    )


def build_ref2va_layout(
    *,
    text_length: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    audio_frames: int,
    reference_shapes: Sequence[Sequence[int]],
    reference_kinds: Sequence[str] = (),
    reference_audio_frames: Sequence[int] = (),
) -> PackedLayout:
    """Build ``[presentation | reference media | target audio | target video]``.

    Each reference shape is ``(latent_frames, latent_height, latent_width)``.
    Images advance one rotary-time unit; videos consume their latent-time span.
    """

    positive = (text_length, latent_frames, latent_height, latent_width, audio_frames)
    if any(value <= 0 for value in positive):
        raise ValueError("all packed-layout dimensions must be positive")
    if latent_height % 2 or latent_width % 2:
        raise ValueError("latent height and width must be divisible by two")
    if not reference_shapes and not reference_audio_frames:
        raise ValueError("Ref2VA requires at least one reference item")
    if reference_kinds and len(reference_kinds) != len(reference_shapes):
        raise ValueError("reference_kinds must align with reference_shapes")
    kinds = tuple(reference_kinds) if reference_kinds else ("image",) * len(reference_shapes)

    normalized_shapes: list[tuple[int, int, int]] = []
    for shape, kind in zip(reference_shapes, kinds):
        if len(shape) != 3:
            raise ValueError("reference shape must be (frames, height, width)")
        ref_t, ref_h, ref_w = (int(value) for value in shape)
        if kind not in ("image", "video"):
            raise ValueError(f"unsupported reference kind: {kind}")
        if kind == "image" and ref_t != 1:
            raise ValueError("image references must contain one latent frame")
        if ref_t <= 0:
            raise ValueError("reference latent frames must be positive")
        if ref_h <= 0 or ref_w <= 0 or ref_h % 2 or ref_w % 2:
            raise ValueError("reference latent H/W must be positive and even")
        normalized_shapes.append((ref_t, ref_h, ref_w))
    normalized_audio_frames = tuple(int(value) for value in reference_audio_frames)
    if any(value <= 0 for value in normalized_audio_frames):
        raise ValueError("reference audio frames must be positive")

    target_frame, target_width_axis = _frame_grid(latent_height, latent_width)
    target_frame_rows = int(target_frame.shape[0])
    segments: list[PackedSegment] = []
    positions: list[torch.Tensor] = []
    video_positions: list[torch.Tensor] = []
    video_updates: list[torch.Tensor] = []
    audio_positions_parts: list[torch.Tensor] = []
    audio_updates: list[torch.Tensor] = []
    offset = 0

    text_grid = torch.zeros(text_length, 3, dtype=torch.float64)
    text_grid[:, 0] = torch.arange(text_length, dtype=torch.float64)
    segments.append(PackedSegment("text", offset, offset + text_length))
    positions.append(text_grid)
    text_positions = torch.arange(offset, offset + text_length)
    offset += text_length

    rotary_time = float(text_length)
    for kind, (ref_t, ref_h, ref_w) in zip(kinds, normalized_shapes):
        ref_frame, _ = _frame_grid(ref_h, ref_w)
        ref_frame_rows = int(ref_frame.shape[0])
        rows = ref_t * ref_frame_rows
        grid = torch.empty(ref_t, ref_frame_rows, 3, dtype=torch.float64)
        grid[:, :, 0] = _temporal_grid(ref_t, rotary_time)[:, None]
        grid[:, :, 1:] = ref_frame[None]
        segments.append(PackedSegment("condition", offset, offset + rows))
        positions.append(grid.reshape(rows, 3))
        video_positions.append(torch.arange(offset, offset + rows))
        video_updates.append(torch.zeros(rows, dtype=torch.bool))
        offset += rows
        rotary_time += 1.0 if kind == "image" else _temporal_span(ref_t)

    for ref_audio_frames in normalized_audio_frames:
        ref_audio_rows = ref_audio_frames * 2
        audio_grid = torch.zeros(ref_audio_rows, 3, dtype=torch.float64)
        audio_grid[:, 0] = (
            rotary_time + torch.arange(ref_audio_frames, dtype=torch.float64)
        ).repeat(2)
        audio_grid[:ref_audio_frames, 2] = target_width_axis[0]
        audio_grid[ref_audio_frames:, 2] = target_width_axis[-1]
        segments.append(PackedSegment("ref_audio", offset, offset + ref_audio_rows))
        positions.append(audio_grid)
        audio_positions_parts.append(torch.arange(offset, offset + ref_audio_rows))
        audio_updates.append(torch.zeros(ref_audio_rows, dtype=torch.bool))
        offset += ref_audio_rows
        rotary_time += float(ref_audio_frames)

    audio_rows = audio_frames * 2
    audio_grid = torch.zeros(audio_rows, 3, dtype=torch.float64)
    audio_grid[:, 0] = (
        rotary_time + torch.arange(audio_frames, dtype=torch.float64)
    ).repeat(2)
    audio_grid[:audio_frames, 2] = target_width_axis[0]
    audio_grid[audio_frames:, 2] = target_width_axis[-1]
    segments.append(PackedSegment("audio", offset, offset + audio_rows))
    positions.append(audio_grid)
    audio_positions_parts.append(torch.arange(offset, offset + audio_rows))
    audio_updates.append(torch.ones(audio_rows, dtype=torch.bool))
    offset += audio_rows

    target_rows = latent_frames * target_frame_rows
    target_grid = torch.empty(
        latent_frames, target_frame_rows, 3, dtype=torch.float64
    )
    target_grid[:, :, 0] = _temporal_grid(latent_frames, rotary_time)[:, None]
    target_grid[:, :, 1:] = target_frame[None]
    segments.append(PackedSegment("video", offset, offset + target_rows))
    positions.append(target_grid.reshape(target_rows, 3))
    video_positions.append(torch.arange(offset, offset + target_rows))
    video_updates.append(torch.ones(target_rows, dtype=torch.bool))

    shape_signature = tuple(
        value for shape in normalized_shapes for value in shape
    )
    return PackedLayout(
        segments=tuple(segments),
        position_ids=torch.cat(positions),
        video_positions=torch.cat(video_positions),
        video_update_mask=torch.cat(video_updates),
        audio_positions=torch.cat(audio_positions_parts),
        audio_update_mask=torch.cat(audio_updates),
        text_positions=text_positions,
        signature=(
            text_length,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            (shape_signature, kinds, normalized_audio_frames),
        ),
    )


def build_hybrid_condition_layout(
    *,
    text_length: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    audio_frames: int,
    reference_shapes: Sequence[Sequence[int]],
    keyframe_indices: Sequence[int],
    output_frame_count: int,
    reference_kinds: Sequence[str] = (),
    reference_audio_frames: Sequence[int] = (),
) -> PackedLayout:
    """Build a mixed reference-memory + FL2VA keyframe layout.

    Long-horizon FL2VA needs two independent kinds of immutable visual input:
    bounded reference-style memory from completed windows and a user-authored
    first/last frame anchored to the current target timeline.  Treating the
    latter as an ordinary reference image loses its endpoint semantics, while
    treating memory as extra keyframes falsely places every memory item at an
    endpoint.  This layout preserves both contracts without changing weights.

    Packed order is ``[text | references | reference audio | keyframes |
    target audio | target video]``.  Reference rotary time advances before the
    target chart.  Keyframe rows then share the exact first/last target rotary
    positions, just as they do in :func:`build_fl2va_layout`.
    """

    positive = (
        text_length,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        output_frame_count,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all hybrid-layout dimensions must be positive")
    if latent_height % 2 or latent_width % 2:
        raise ValueError("latent height and width must be divisible by two")
    if not reference_shapes and not reference_audio_frames:
        raise ValueError("hybrid layout requires reference-style memory")
    if not keyframe_indices:
        raise ValueError("hybrid layout requires at least one keyframe")
    if any(index not in (0, -1) for index in keyframe_indices):
        raise ValueError("only first-frame (0) and last-frame (-1) anchors are supported")
    if len(set(keyframe_indices)) != len(keyframe_indices):
        raise ValueError("duplicate keyframe anchors are not allowed")
    resolved = [
        0 if index == 0 else int(output_frame_count) - 1
        for index in keyframe_indices
    ]
    if len(set(resolved)) != len(resolved):
        raise ValueError("keyframe anchors resolve to the same output frame")
    if reference_kinds and len(reference_kinds) != len(reference_shapes):
        raise ValueError("reference_kinds must align with reference_shapes")
    kinds = (
        tuple(reference_kinds)
        if reference_kinds
        else ("image",) * len(reference_shapes)
    )

    normalized_shapes: list[tuple[int, int, int]] = []
    for shape, kind in zip(reference_shapes, kinds):
        if len(shape) != 3:
            raise ValueError("reference shape must be (frames, height, width)")
        ref_t, ref_h, ref_w = (int(value) for value in shape)
        if kind not in ("image", "video"):
            raise ValueError(f"unsupported reference kind: {kind}")
        if kind == "image" and ref_t != 1:
            raise ValueError("image references must contain one latent frame")
        if ref_t <= 0:
            raise ValueError("reference latent frames must be positive")
        if ref_h <= 0 or ref_w <= 0 or ref_h % 2 or ref_w % 2:
            raise ValueError("reference latent H/W must be positive and even")
        normalized_shapes.append((ref_t, ref_h, ref_w))
    normalized_audio_frames = tuple(int(value) for value in reference_audio_frames)
    if any(value <= 0 for value in normalized_audio_frames):
        raise ValueError("reference audio frames must be positive")

    target_frame, target_width_axis = _frame_grid(latent_height, latent_width)
    target_frame_rows = int(target_frame.shape[0])
    segments: list[PackedSegment] = []
    positions: list[torch.Tensor] = []
    video_positions: list[torch.Tensor] = []
    video_updates: list[torch.Tensor] = []
    audio_positions_parts: list[torch.Tensor] = []
    audio_updates: list[torch.Tensor] = []
    offset = 0

    text_grid = torch.zeros(text_length, 3, dtype=torch.float64)
    text_grid[:, 0] = torch.arange(text_length, dtype=torch.float64)
    segments.append(PackedSegment("text", offset, offset + text_length))
    positions.append(text_grid)
    text_positions = torch.arange(offset, offset + text_length)
    offset += text_length

    rotary_time = float(text_length)
    for kind, (ref_t, ref_h, ref_w) in zip(kinds, normalized_shapes):
        ref_frame, _ = _frame_grid(ref_h, ref_w)
        ref_frame_rows = int(ref_frame.shape[0])
        rows = ref_t * ref_frame_rows
        grid = torch.empty(ref_t, ref_frame_rows, 3, dtype=torch.float64)
        grid[:, :, 0] = _temporal_grid(ref_t, rotary_time)[:, None]
        grid[:, :, 1:] = ref_frame[None]
        segments.append(PackedSegment("condition", offset, offset + rows))
        positions.append(grid.reshape(rows, 3))
        video_positions.append(torch.arange(offset, offset + rows))
        video_updates.append(torch.zeros(rows, dtype=torch.bool))
        offset += rows
        rotary_time += 1.0 if kind == "image" else _temporal_span(ref_t)

    for ref_audio_frames in normalized_audio_frames:
        ref_audio_rows = ref_audio_frames * 2
        audio_grid = torch.zeros(ref_audio_rows, 3, dtype=torch.float64)
        audio_grid[:, 0] = (
            rotary_time + torch.arange(ref_audio_frames, dtype=torch.float64)
        ).repeat(2)
        audio_grid[:ref_audio_frames, 2] = target_width_axis[0]
        audio_grid[ref_audio_frames:, 2] = target_width_axis[-1]
        segments.append(PackedSegment("ref_audio", offset, offset + ref_audio_rows))
        positions.append(audio_grid)
        audio_positions_parts.append(torch.arange(offset, offset + ref_audio_rows))
        audio_updates.append(torch.zeros(ref_audio_rows, dtype=torch.bool))
        offset += ref_audio_rows
        rotary_time += float(ref_audio_frames)

    target_t = _temporal_grid(latent_frames, rotary_time)
    for anchor in keyframe_indices:
        condition = torch.empty(target_frame_rows, 3, dtype=torch.float64)
        condition[:, 0] = target_t[0] if anchor == 0 else target_t[-1]
        condition[:, 1:] = target_frame
        segments.append(PackedSegment("condition", offset, offset + target_frame_rows))
        positions.append(condition)
        video_positions.append(torch.arange(offset, offset + target_frame_rows))
        video_updates.append(torch.zeros(target_frame_rows, dtype=torch.bool))
        offset += target_frame_rows

    audio_rows = audio_frames * 2
    audio_grid = torch.zeros(audio_rows, 3, dtype=torch.float64)
    audio_grid[:, 0] = (
        rotary_time + torch.arange(audio_frames, dtype=torch.float64)
    ).repeat(2)
    audio_grid[:audio_frames, 2] = target_width_axis[0]
    audio_grid[audio_frames:, 2] = target_width_axis[-1]
    segments.append(PackedSegment("audio", offset, offset + audio_rows))
    positions.append(audio_grid)
    audio_positions_parts.append(torch.arange(offset, offset + audio_rows))
    audio_updates.append(torch.ones(audio_rows, dtype=torch.bool))
    offset += audio_rows

    target_rows = latent_frames * target_frame_rows
    target_grid = torch.empty(
        latent_frames, target_frame_rows, 3, dtype=torch.float64
    )
    target_grid[:, :, 0] = target_t[:, None]
    target_grid[:, :, 1:] = target_frame[None]
    segments.append(PackedSegment("video", offset, offset + target_rows))
    positions.append(target_grid.reshape(target_rows, 3))
    video_positions.append(torch.arange(offset, offset + target_rows))
    video_updates.append(torch.ones(target_rows, dtype=torch.bool))

    shape_signature = tuple(
        value for shape in normalized_shapes for value in shape
    )
    return PackedLayout(
        segments=tuple(segments),
        position_ids=torch.cat(positions),
        video_positions=torch.cat(video_positions),
        video_update_mask=torch.cat(video_updates),
        audio_positions=torch.cat(audio_positions_parts),
        audio_update_mask=torch.cat(audio_updates),
        text_positions=text_positions,
        signature=(
            text_length,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            (
                "hybrid_reference_keyframe_v1",
                shape_signature,
                kinds,
                normalized_audio_frames,
                tuple(int(index) for index in keyframe_indices),
                int(output_frame_count),
            ),
        ),
    )


def patchify_video(latent: torch.Tensor, patch_size: Sequence[int] = (1, 2, 2)) -> torch.Tensor:
    if latent.ndim != 5:
        raise ValueError("video latent must be [B,C,T,H,W]")
    pt, ph, pw = (int(value) for value in patch_size)
    batch, channels, full_t, full_h, full_w = latent.shape
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError("video latent dimensions must be divisible by patch_size")
    t, h, w = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(batch, channels, t, pt, h, ph, w, pw)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(batch * t * h * w, channels * pt * ph * pw).contiguous()


def unpatchify_video(
    rows: torch.Tensor,
    *,
    latent_shape: Sequence[int],
    patch_size: Sequence[int] = (1, 2, 2),
) -> torch.Tensor:
    t, h, w, channels = (int(value) for value in latent_shape)
    pt, ph, pw = (int(value) for value in patch_size)
    rows_per_sample = t * h * w
    if rows.ndim != 2 or rows.shape[0] % rows_per_sample:
        raise ValueError("video token rows do not match latent_shape")
    packed = rows.reshape(-1, t, h, w, channels, pt, ph, pw)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channels, t * pt, h * ph, w * pw).contiguous()


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4 or latent.shape[0] != 1 or latent.shape[2] != 2:
        raise ValueError("audio latent must be [1,C,2,T]")
    return latent[0].permute(1, 2, 0).reshape(-1, latent.shape[1]).contiguous()


def unpack_audio(rows: torch.Tensor, channels: int = 2) -> torch.Tensor:
    if rows.ndim != 2 or rows.shape[0] % channels:
        raise ValueError("audio rows must be [channels*time, latent_dim]")
    time = rows.shape[0] // channels
    return rows.reshape(channels, time, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0).contiguous()
