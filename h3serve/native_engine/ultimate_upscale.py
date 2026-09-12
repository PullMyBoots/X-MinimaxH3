"""Native planning primitives adapted from MMH3 UltimateUpscale.

The upstream ComfyUI node always exposes temporal/spatial pieces directly.
H3 Serve first asks whether the target lies inside the model's calibrated
native temporal horizon and can use the full-context low-VRAM execution graph.
If yes, one full-canvas second pass is both cheaper and more coherent.  Longer
sequences are temporally bounded before memory admission; spatial tiling is
introduced only when a bounded full-spatial piece still cannot fit.

This module contains no ComfyUI dependency and no prompt/scene heuristics.
Every decision is a function of target geometry, H3's latent grids and the
device budget, which makes it reusable by FL2VA and Ref2VA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .planner import H3WorkloadAnalyzer, select_memory_execution
from .resource_backends import ResourceBackendId, WeightTier


H3_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
H3_AUDIO_TOKENS_PER_FRAME = 5.0 / 3.0
H3_SPATIAL_COMPRESSION = 16
H3_NATIVE_MAX_FRAMES = 362
H3_TEMPORAL_OVERLAP_FRAMES = 17
# Keep an automatic long-horizon piece inside the model's native 15-second
# context even after both ends are snapped independently to the five-token
# H3 VAE phase grid.  362 - 2 * 17 = 328; real snapped pieces are <= 340
# frames, while a requested 362-frame moving window can grow beyond the
# calibrated native horizon.
H3_AUTO_LONG_WINDOW_FRAMES = (
    H3_NATIVE_MAX_FRAMES - 2 * H3_TEMPORAL_OVERLAP_FRAMES
)


def frames_for_video_tokens(tokens: int) -> int:
    if tokens < 0:
        raise ValueError("video token count cannot be negative")
    cycles, tail = divmod(int(tokens), len(H3_FRAME_PER_TOKEN))
    return cycles * sum(H3_FRAME_PER_TOKEN) + sum(H3_FRAME_PER_TOKEN[:tail])


def video_tokens_for_frames(frames: int) -> int:
    if frames < 0:
        raise ValueError("frame count cannot be negative")
    tokens = 0
    covered = 0
    while covered < frames:
        covered += H3_FRAME_PER_TOKEN[tokens % len(H3_FRAME_PER_TOKEN)]
        tokens += 1
    return tokens


def audio_token_range(frame_start: int, frame_stop: int) -> tuple[int, int]:
    if frame_start < 0 or frame_stop < frame_start:
        raise ValueError("invalid audio frame interval")
    return (
        round(frame_start * H3_AUDIO_TOKENS_PER_FRAME),
        round(frame_stop * H3_AUDIO_TOKENS_PER_FRAME),
    )


def _snap_frame_boundary(
    frame: int,
    *,
    maximum_video_tokens: int,
    token_phase: int = 5,
) -> tuple[int, int]:
    best_tokens = 0
    best_frames = 0
    best_distance = abs(int(frame))
    for tokens in range(0, maximum_video_tokens + 1, token_phase):
        candidate_frames = frames_for_video_tokens(tokens)
        distance = abs(candidate_frames - frame)
        if distance < best_distance:
            best_tokens = tokens
            best_frames = candidate_frames
            best_distance = distance
    return best_tokens, best_frames


@dataclass(frozen=True, slots=True)
class TemporalPiece:
    video_token_start: int
    frame_start: int
    video_token_stop: int
    frame_stop: int
    audio_token_start: int
    audio_token_stop: int

    @property
    def frames(self) -> int:
        return self.frame_stop - self.frame_start


def slice_av_temporal_piece(
    video: torch.Tensor,
    audio: torch.Tensor,
    piece: TemporalPiece,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one phase-aligned H3 AV window.

    Adapted from MMH3 UltimateUpscale's outer temporal loop.  Video and audio
    use different clocks, so slicing both tensors by the video-token interval
    would silently desynchronise the result.
    """

    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("H3 AV latent must be video[B,C,T,H,W] + audio[B,C,F,T]")
    if not 0 <= piece.video_token_start < piece.video_token_stop <= video.shape[2]:
        raise ValueError("temporal piece exceeds the video latent")
    if not 0 <= piece.audio_token_start < piece.audio_token_stop <= audio.shape[-1]:
        raise ValueError("temporal piece exceeds the audio latent")
    return (
        video[:, :, piece.video_token_start:piece.video_token_stop].contiguous(),
        audio[..., piece.audio_token_start:piece.audio_token_stop].contiguous(),
    )


def _crossfade(left: torch.Tensor, right: torch.Tensor, *, dim: int) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("crossfade tensors must have identical shapes")
    count = left.shape[dim]
    if count <= 0:
        return left
    weight = torch.linspace(
        0.0, 1.0, count, device=left.device, dtype=left.dtype
    )
    shape = [1] * left.ndim
    shape[dim] = count
    weight = weight.view(shape)
    return left + (right - left) * weight


def append_av_temporal_piece(
    accumulated_video: torch.Tensor | None,
    accumulated_audio: torch.Tensor | None,
    piece_video: torch.Tensor,
    piece_audio: torch.Tensor,
    piece: TemporalPiece,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-fade one sampled window into global H3 AV coordinates.

    This is the native equivalent of upstream ``temporal_append``.  It keeps
    overlap work useful as a seam constraint instead of discarding duplicate
    rows or hard-cutting independently sampled windows.
    """

    if accumulated_video is None or accumulated_audio is None:
        if piece.video_token_start != 0 or piece.audio_token_start != 0:
            raise ValueError("the first temporal piece must start at zero")
        return piece_video, piece_audio
    if accumulated_video.ndim != 5 or accumulated_audio.ndim != 4:
        raise ValueError("accumulated H3 AV latent has invalid rank")
    if piece_video.ndim != 5 or piece_audio.ndim != 4:
        raise ValueError("piece H3 AV latent has invalid rank")

    video_start = piece.video_token_start
    audio_start = piece.audio_token_start
    video_total = max(accumulated_video.shape[2], video_start + piece_video.shape[2])
    audio_total = max(accumulated_audio.shape[-1], audio_start + piece_audio.shape[-1])
    result_video = accumulated_video.new_zeros(
        (*accumulated_video.shape[:2], video_total, *accumulated_video.shape[3:])
    )
    result_audio = accumulated_audio.new_zeros(
        (*accumulated_audio.shape[:-1], audio_total)
    )
    result_video[:, :, : accumulated_video.shape[2]] = accumulated_video
    result_audio[..., : accumulated_audio.shape[-1]] = accumulated_audio

    video_overlap = min(
        max(0, accumulated_video.shape[2] - video_start), piece_video.shape[2]
    )
    if video_overlap:
        result_video[:, :, video_start : video_start + video_overlap] = _crossfade(
            result_video[:, :, video_start : video_start + video_overlap].clone(),
            piece_video[:, :, :video_overlap],
            dim=2,
        )
    video_tail = piece_video[:, :, video_overlap:]
    if video_tail.shape[2]:
        write = video_start + video_overlap
        result_video[:, :, write : write + video_tail.shape[2]] = video_tail

    audio_overlap = min(
        max(0, accumulated_audio.shape[-1] - audio_start), piece_audio.shape[-1]
    )
    if audio_overlap:
        result_audio[..., audio_start : audio_start + audio_overlap] = _crossfade(
            result_audio[..., audio_start : audio_start + audio_overlap].clone(),
            piece_audio[..., :audio_overlap],
            dim=3,
        )
    audio_tail = piece_audio[..., audio_overlap:]
    if audio_tail.shape[-1]:
        write = audio_start + audio_overlap
        result_audio[..., write : write + audio_tail.shape[-1]] = audio_tail
    return result_video, result_audio


def temporal_pieces(
    video_tokens: int,
    *,
    chunk_frames: int,
    overlap_frames: int,
) -> tuple[TemporalPiece, ...]:
    """Return UltimateUpscale-compatible phase-aligned time pieces."""

    total_frames = frames_for_video_tokens(video_tokens)
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if overlap_frames < 0 or overlap_frames >= chunk_frames:
        raise ValueError("temporal overlap must lie inside [0, chunk_frames)")
    hop = chunk_frames - overlap_frames
    pieces: list[TemporalPiece] = []
    previous_stop_tokens = 0
    index = 0
    while True:
        requested_start = index * hop
        requested_stop = min(requested_start + chunk_frames, total_frames)
        if index == 0:
            start_tokens, start_frames = 0, 0
        else:
            start_tokens, start_frames = _snap_frame_boundary(
                requested_start,
                maximum_video_tokens=video_tokens,
            )
            if start_tokens > previous_stop_tokens:
                start_tokens = previous_stop_tokens
                start_frames = frames_for_video_tokens(start_tokens)
        if requested_stop >= total_frames:
            stop_tokens, stop_frames = video_tokens, total_frames
        else:
            stop_tokens, stop_frames = _snap_frame_boundary(
                requested_stop,
                maximum_video_tokens=video_tokens,
            )
            if stop_tokens <= start_tokens:
                stop_tokens = min(video_tokens, start_tokens + 5)
                stop_frames = frames_for_video_tokens(stop_tokens)
        audio_start, audio_stop = audio_token_range(start_frames, stop_frames)
        pieces.append(TemporalPiece(
            video_token_start=start_tokens,
            frame_start=start_frames,
            video_token_stop=stop_tokens,
            frame_stop=stop_frames,
            audio_token_start=audio_start,
            audio_token_stop=audio_stop,
        ))
        if stop_tokens >= video_tokens:
            break
        previous_stop_tokens = stop_tokens
        index += 1
    return tuple(pieces)


def _grid_axis(
    size: int,
    tile: int,
    overlap: int,
    minimum_tile: int,
) -> tuple[tuple[int, int, int], ...]:
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError("invalid tile/overlap axis")
    if minimum_tile < 0 or minimum_tile > tile:
        raise ValueError("invalid minimum tile axis")
    if size <= tile:
        return ((0, size, 0),)
    stride = tile - overlap
    count = math.ceil((size - overlap) / stride)
    origins = [index * stride for index in range(count)]
    extents = [min(tile, size - origin) for origin in origins]
    if minimum_tile and count >= 2 and extents[-1] < minimum_tile:
        adjusted = size - minimum_tile
        if origins[-2] < adjusted < origins[-2] + extents[-2]:
            origins[-1] = adjusted
            extents[-1] = size - adjusted
    overlaps = [0]
    overlaps.extend(
        max(0, origins[index - 1] + extents[index - 1] - origins[index])
        for index in range(1, count)
    )
    return tuple(zip(origins, extents, overlaps))


@dataclass(frozen=True, slots=True)
class SpatialTile:
    row: int
    column: int
    height: int
    width: int
    top_overlap: int
    left_overlap: int


def spatial_tiles(
    height: int,
    width: int,
    *,
    tile_height: int,
    tile_width: int,
    overlap_height: int,
    overlap_width: int,
    minimum_tile: int = 256,
) -> tuple[SpatialTile, ...]:
    """Build the upstream frozen-overlap grid in pixel coordinates."""

    values = (
        height, width, tile_height, tile_width,
        overlap_height, overlap_width, minimum_tile,
    )
    if any(value % 32 for value in values):
        raise ValueError("H3 spatial pieces must be multiples of 32 pixels")
    rows = _grid_axis(height, tile_height, overlap_height, minimum_tile)
    columns = _grid_axis(width, tile_width, overlap_width, minimum_tile)
    return tuple(
        SpatialTile(
            row=row,
            column=column,
            height=row_extent,
            width=column_extent,
            top_overlap=row_overlap,
            left_overlap=column_overlap,
        )
        for row, row_extent, row_overlap in rows
        for column, column_extent, column_overlap in columns
    )


@dataclass(frozen=True, slots=True)
class UltimateUpscalePlan:
    target_width: int
    target_height: int
    frames: int
    device_budget_bytes: int
    full_canvas: bool
    memory_execution: dict[str, object]
    temporal: tuple[TemporalPiece, ...]
    spatial: tuple[SpatialTile, ...]
    sampled_pixel_frames: int
    source_pixel_frames: int
    redundancy_ratio: float
    provenance: str = "h3_learned_3d_second_sampling_v2"

    def telemetry(self) -> dict[str, object]:
        return {
            **asdict(self),
            "temporal": [asdict(piece) for piece in self.temporal],
            "spatial": [asdict(tile) for tile in self.spatial],
        }


def plan_ultimate_upscale(
    *,
    target_width: int,
    target_height: int,
    frames: int,
    device_budget_bytes: int,
    text_tokens: int,
    condition_count: int,
    engine: str = "original",
    actual_evaluations: int = 5,
    requested_mode: str = "auto",
    weight_tier: WeightTier = "int8",
    resource_profile: ResourceBackendId | None = None,
    allow_spatial_tiles: bool = False,
    temporal_window_frames: int | None = None,
    temporal_overlap_frames: int | None = None,
) -> UltimateUpscalePlan:
    """Prefer a zero-redundancy full canvas, else search bounded pieces.

    The piece search is intentionally small and deterministic.  It minimizes
    duplicated pixel-frames, then prefers fewer pieces.  It does not inspect
    prompts, reference semantics or generated pixels.
    """

    analyzer = H3WorkloadAnalyzer()
    full_features = analyzer.analyze(
        width=target_width,
        height=target_height,
        frames=frames,
        text_tokens=text_tokens,
        condition_count=condition_count,
        engine=engine,
        actual_evaluations=actual_evaluations,
        forecast_evaluations=0,
    )
    full_decision = select_memory_execution(
        full_features,
        requested_mode=requested_mode,
        device_budget_bytes=device_budget_bytes,
        weight_tier=weight_tier,
        resource_profile=resource_profile,
        include_vae=True,
    )
    source_work = target_width * target_height * frames
    requested_window = (
        None
        if temporal_window_frames is None
        else min(frames, max(1, int(temporal_window_frames)))
    )
    requested_overlap = (
        None
        if temporal_overlap_frames is None
        else max(0, int(temporal_overlap_frames))
    )
    # The outer orchestrator has not run Qwen yet and only owns a conservative
    # prompt-length estimate.  On hard 8-GiB W4, a mathematically fitting
    # 153-frame piece left ~120 MiB and was then correctly rejected by the
    # inner planner after real text encoding and allocator fragmentation were
    # visible.  Reserve 768 MiB here so window boundaries are insensitive to
    # prompt/reference tokenization details and the admitted single reference.
    piece_budget_bytes = (
        min(device_budget_bytes, int(6.5 * 1024**3))
        if resource_profile == "w4a8_8gb"
        else device_budget_bytes
    )
    # The activation estimator describes the CUDA allocation peak, but it
    # cannot see sustained host<->device page traffic caused by a nearly-full
    # 24 GiB execution graph.  A real 2560x1440x362 run reached 23.85 GiB yet
    # remained at 110--148 W with 13--16.5 GB/s PCIe RX: it "fit" while
    # continuously paging.  Treat 2K-class long sequences as a separate
    # throughput admission domain and route them through UltimateUpscale's
    # temporal executor even when the allocation estimate says yes.
    requires_throughput_windows = bool(
        target_width * target_height > 1920 * 1088 and frames > 141
    )
    # The estimator is a useful admission signal inside the native training
    # horizon, but it is not allowed to authorize a super-native Attention
    # sequence.  A measured 1280x736x1076 second pass was estimated to fit and
    # then OOMed in DiT block 4 with ~21.67 GiB already allocated.  Bound every
    # automatic >15-second pass before considering the estimate, and retain
    # enough phase-snap margin that each actual piece remains <=362 frames.
    requires_long_horizon_windows = frames > H3_NATIVE_MAX_FRAMES
    automatic_window_frames = (
        136
        if requires_throughput_windows
        else H3_AUTO_LONG_WINDOW_FRAMES
        if requires_long_horizon_windows
        else None
    )
    explicit_full_context = bool(
        requested_window is not None and requested_window >= frames
    )
    if full_decision.fits_budget and (
        explicit_full_context
        or (requested_window is None and automatic_window_frames is None)
    ):
        temporal = temporal_pieces(
            full_features.latent_frames,
            chunk_frames=frames,
            overlap_frames=0,
        )
        spatial = spatial_tiles(
            target_height,
            target_width,
            tile_height=target_height,
            tile_width=target_width,
            overlap_height=0,
            overlap_width=0,
            minimum_tile=0,
        )
        return UltimateUpscalePlan(
            target_width=target_width,
            target_height=target_height,
            frames=frames,
            device_budget_bytes=device_budget_bytes,
            full_canvas=True,
            memory_execution=full_decision.telemetry(),
            temporal=temporal,
            spatial=spatial,
            sampled_pixel_frames=source_work,
            source_pixel_frames=source_work,
            redundancy_ratio=1.0,
        )

    if requested_window is not None and requested_window < frames:
        # A user-selected shorter context is a throughput/continuity trade-off,
        # not a different sampler.  Keep full spatial context and the native
        # 17-frame overlap; temporal_pieces snaps the visible value onto H3's
        # five-token VAE phase grid.  If this request still does not fit, the
        # ordinary bounded search below may only choose an equal or shorter
        # safe window.
        time = temporal_pieces(
            full_features.latent_frames,
            chunk_frames=requested_window,
            overlap_frames=min(
                H3_TEMPORAL_OVERLAP_FRAMES
                if requested_overlap is None
                else requested_overlap,
                requested_window - 1,
            ),
        )
        maximum_piece = max(
            time,
            key=lambda piece: piece.video_token_stop - piece.video_token_start,
        )
        piece_features = analyzer.analyze(
            width=target_width,
            height=target_height,
            frames=maximum_piece.frames,
            text_tokens=text_tokens,
            condition_count=condition_count,
            engine=engine,
            actual_evaluations=actual_evaluations,
            forecast_evaluations=0,
            latent_frames_override=(
                maximum_piece.video_token_stop - maximum_piece.video_token_start
            ),
            audio_frames_override=(
                maximum_piece.audio_token_stop - maximum_piece.audio_token_start
            ),
        )
        piece_decision = select_memory_execution(
            piece_features,
            requested_mode=requested_mode,
            device_budget_bytes=piece_budget_bytes,
            weight_tier=weight_tier,
            resource_profile=resource_profile,
            include_vae=False,
        )
        if piece_decision.fits_budget:
            spatial = spatial_tiles(
                target_height,
                target_width,
                tile_height=target_height,
                tile_width=target_width,
                overlap_height=0,
                overlap_width=0,
                minimum_tile=0,
            )
            sampled = sum(piece.frames for piece in time) * target_width * target_height
            return UltimateUpscalePlan(
                target_width=target_width,
                target_height=target_height,
                frames=frames,
                device_budget_bytes=device_budget_bytes,
                full_canvas=False,
                memory_execution={
                    **piece_decision.telemetry(),
                    "admission_reason": "user_temporal_window",
                    "requested_temporal_window_frames": requested_window,
                    "requested_temporal_overlap_frames": requested_overlap,
                },
                temporal=time,
                spatial=spatial,
                sampled_pixel_frames=sampled,
                source_pixel_frames=source_work,
                redundancy_ratio=sampled / source_work,
            )

    if automatic_window_frames is not None and requested_window is None:
        # Keep the complete spatial canvas and route automatic long sequences
        # through a bounded temporal executor.  2K-class workloads retain the
        # upstream 136/17 throughput operating point; lower resolutions use a
        # larger 328-frame target whose phase-snapped pieces remain inside the
        # native 362-frame context.  This avoids spatial seams while bounding
        # both Attention complexity and VRAM independently of total duration.
        time = temporal_pieces(
            full_features.latent_frames,
            chunk_frames=automatic_window_frames,
            overlap_frames=H3_TEMPORAL_OVERLAP_FRAMES,
        )
        maximum_piece = max(
            time,
            key=lambda piece: piece.video_token_stop - piece.video_token_start,
        )
        maximum_frames = maximum_piece.frames
        piece_features = analyzer.analyze(
            width=target_width,
            height=target_height,
            frames=maximum_frames,
            text_tokens=text_tokens,
            condition_count=condition_count,
            engine=engine,
            actual_evaluations=actual_evaluations,
            forecast_evaluations=0,
            latent_frames_override=(
                maximum_piece.video_token_stop - maximum_piece.video_token_start
            ),
            audio_frames_override=(
                maximum_piece.audio_token_stop - maximum_piece.audio_token_start
            ),
        )
        piece_decision = select_memory_execution(
            piece_features,
            requested_mode=requested_mode,
            device_budget_bytes=piece_budget_bytes,
            weight_tier=weight_tier,
            resource_profile=resource_profile,
            include_vae=False,
        )
        if piece_decision.fits_budget:
            spatial = spatial_tiles(
                target_height,
                target_width,
                tile_height=target_height,
                tile_width=target_width,
                overlap_height=0,
                overlap_width=0,
                minimum_tile=0,
            )
            sampled = sum(piece.frames for piece in time) * target_width * target_height
            return UltimateUpscalePlan(
                target_width=target_width,
                target_height=target_height,
                frames=frames,
                device_budget_bytes=device_budget_bytes,
                full_canvas=False,
                memory_execution={
                    **piece_decision.telemetry(),
                    "admission_reason": (
                        "2k_long_pcie_thrash_guard"
                        if requires_throughput_windows
                        else "native_horizon_guard"
                    ),
                    "automatic_temporal_window_frames": automatic_window_frames,
                    "native_max_frames": H3_NATIVE_MAX_FRAMES,
                    "full_canvas_estimate": full_decision.telemetry(),
                },
                temporal=time,
                spatial=spatial,
                sampled_pixel_frames=sampled,
                source_pixel_frames=source_work,
                redundancy_ratio=sampled / source_work,
            )

    # Ultimate's seams operate on 32-pixel cells.  Try spatial capacity before
    # temporal splitting because a full-duration tile preserves motion without
    # a time seam.  Temporal candidates remain available when one spatial tile
    # still exceeds the budget.
    spatial_sizes = (
        (2048, 1536, 1280, 1024, 768, 640, 512, 384, 320, 256)
        if allow_spatial_tiles
        else (max(target_width, target_height),)
    )
    # Never let the fallback search undo an automatic safety guard merely
    # because the same optimistic full-context estimator reports a fit.
    maximum_chunk_frames = min(
        frames,
        requested_window
        if requested_window is not None
        else automatic_window_frames
        if automatic_window_frames is not None
        else frames,
    )
    chunk_frames = tuple(
        chunk
        for index in range((maximum_chunk_frames - 5) // 17, -1, -1)
        if (chunk := 5 + 17 * index) > 17
    )
    candidates: list[tuple[float, int, object, tuple[TemporalPiece, ...], tuple[SpatialTile, ...], int]] = []
    for tile_limit in spatial_sizes:
        tile_width = min(target_width, tile_limit)
        tile_height = min(target_height, tile_limit)
        tile_width -= tile_width % 32
        tile_height -= tile_height % 32
        if min(tile_width, tile_height) < 256:
            continue
        overlap_width = 0 if tile_width == target_width else min(128, tile_width - 32)
        overlap_height = 0 if tile_height == target_height else min(128, tile_height - 32)
        tiles = spatial_tiles(
            target_height,
            target_width,
            tile_height=tile_height,
            tile_width=tile_width,
            overlap_height=overlap_height,
            overlap_width=overlap_width,
        )
        maximum_tile_width = max(tile.width for tile in tiles)
        maximum_tile_height = max(tile.height for tile in tiles)
        for chunk in chunk_frames:
            time = temporal_pieces(
                full_features.latent_frames,
                chunk_frames=chunk,
                overlap_frames=0 if chunk == frames else 17,
            )
            maximum_piece = max(
                time,
                key=lambda piece: (
                    piece.video_token_stop - piece.video_token_start
                ),
            )
            piece_features = analyzer.analyze(
                width=maximum_tile_width,
                height=maximum_tile_height,
                frames=maximum_piece.frames,
                text_tokens=text_tokens,
                condition_count=condition_count,
                engine=engine,
                actual_evaluations=actual_evaluations,
                forecast_evaluations=0,
                latent_frames_override=(
                    maximum_piece.video_token_stop
                    - maximum_piece.video_token_start
                ),
                audio_frames_override=(
                    maximum_piece.audio_token_stop
                    - maximum_piece.audio_token_start
                ),
            )
            decision = select_memory_execution(
                piece_features,
                requested_mode=requested_mode,
                device_budget_bytes=piece_budget_bytes,
                weight_tier=weight_tier,
                resource_profile=resource_profile,
                include_vae=False,
            )
            if not decision.fits_budget:
                continue
            sampled = sum(piece.frames for piece in time) * sum(
                tile.height * tile.width for tile in tiles
            )
            redundancy = sampled / source_work
            candidates.append((
                redundancy,
                len(time) * len(tiles),
                decision,
                time,
                tiles,
                sampled,
            ))
            # For a fixed spatial grid, shorter chunks only add time seams.
            break
    if not candidates:
        raise RuntimeError(
            "UltimateUpscale cannot find a >=256px H3 piece inside the device budget"
        )
    redundancy, _piece_count, decision, time, tiles, sampled = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return UltimateUpscalePlan(
        target_width=target_width,
        target_height=target_height,
        frames=frames,
        device_budget_bytes=device_budget_bytes,
        full_canvas=False,
        memory_execution=decision.telemetry(),
        temporal=time,
        spatial=tiles,
        sampled_pixel_frames=sampled,
        source_pixel_frames=source_work,
        redundancy_ratio=redundancy,
    )


__all__ = [
    "H3_AUDIO_TOKENS_PER_FRAME",
    "H3_FRAME_PER_TOKEN",
    "SpatialTile",
    "TemporalPiece",
    "UltimateUpscalePlan",
    "audio_token_range",
    "append_av_temporal_piece",
    "frames_for_video_tokens",
    "plan_ultimate_upscale",
    "spatial_tiles",
    "slice_av_temporal_piece",
    "temporal_pieces",
    "video_tokens_for_frames",
]
