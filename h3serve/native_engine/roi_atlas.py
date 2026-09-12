"""Region-atlas helpers for budgeted H3 second sampling.

The H3 DiT still receives one ordinary rectangular latent canvas.  Difficult
regions are cropped from the accepted full-frame trajectory, magnified into a
regular atlas, refined together, and reduced back into their original target
locations.  This increases spatial-token density where the source is weak
without increasing the DiT canvas or disturbing the rest of the frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class ROIAtlasRecord:
    """One reversible mapping between a full-frame crop and an atlas cell."""

    source_y0: int
    source_y1: int
    source_x0: int
    source_x1: int
    atlas_y0: int
    atlas_y1: int
    atlas_x0: int
    atlas_x1: int

    @property
    def source_height(self) -> int:
        return self.source_y1 - self.source_y0

    @property
    def source_width(self) -> int:
        return self.source_x1 - self.source_x0


@dataclass(frozen=True, slots=True)
class ROIDifficultyCandidate:
    """One automatically selected spatial region and its routing evidence."""

    box: tuple[float, float, float, float]
    score: float
    side: int
    convergence: float
    temporal_instability: float
    representation_deficit: float


def _resize_video_spatial(
    video: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str = "bicubic",
) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("H3 video latent must have shape B,C,T,H,W")
    if height <= 0 or width <= 0:
        raise ValueError("target spatial size must be positive")
    batch, channels, frames, source_height, source_width = video.shape
    if (source_height, source_width) == (height, width):
        return video
    values = video.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, source_height, source_width
    )
    if mode == "area":
        resized = F.interpolate(values.float(), size=(height, width), mode=mode)
    else:
        resized = F.interpolate(
            values.float(),
            size=(height, width),
            mode=mode,
            align_corners=False,
            antialias=True,
        )
    return resized.reshape(batch, frames, channels, height, width).permute(
        0, 2, 1, 3, 4
    ).contiguous()


def _channel_sketch(video: torch.Tensor, maximum_channels: int = 8) -> torch.Tensor:
    """Retain an evenly spaced channel sketch for inexpensive routing scores."""

    channels = int(video.shape[1])
    if channels <= maximum_channels:
        return video.float()
    indices = torch.linspace(
        0,
        channels - 1,
        maximum_channels,
        device=video.device,
    ).round().long()
    return video.index_select(1, indices).float()


def _spatial_cutoff_detail(
    video: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
) -> torch.Tensor:
    """Return detail above the spatial band representable by the source grid.

    The result uses ``B,T,C,H,W`` so temporal operations are contiguous.  This
    is a routing feature only; selected H3 evaluations still consume every
    latent channel.
    """

    batch, _, frames, height, width = video.shape
    values = _channel_sketch(video).permute(0, 2, 1, 3, 4)
    sketch_channels = int(values.shape[2])
    flat = values.reshape(batch * frames, sketch_channels, height, width)
    low = F.interpolate(
        flat,
        size=(source_height, source_width),
        mode="area",
    )
    low = F.interpolate(
        low,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return (flat - low).reshape(
        batch, frames, sketch_channels, height, width
    )


def _source_structure_map(
    motion_video: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
) -> torch.Tensor:
    """Estimate where the low-resolution source contains compact structure."""

    batch, _, frames, height, width = motion_video.shape
    values = _channel_sketch(motion_video).permute(0, 2, 1, 3, 4)
    channels = int(values.shape[2])
    flat = values.reshape(batch * frames, channels, height, width)
    native = F.interpolate(
        flat,
        size=(source_height, source_width),
        mode="area",
    )
    dx = F.pad(native[..., 1:] - native[..., :-1], (0, 1, 0, 0))
    dy = F.pad(native[..., 1:, :] - native[..., :-1, :], (0, 0, 0, 1))
    gradient = (dx.square() + dy.square()).mean(dim=1, keepdim=True).sqrt()
    local_mean = F.avg_pool2d(
        F.pad(native, (1, 1, 1, 1), mode="replicate"),
        kernel_size=3,
        stride=1,
    )
    contrast = (native - local_mean).square().mean(dim=1, keepdim=True).sqrt()
    # A small object or a shelf of repeated products has high *density* of
    # fine structure.  Pooling the fine response rewards that case over one
    # isolated high-contrast boundary on a large foreground object.
    fine = gradient + 0.75 * contrast
    structure = F.avg_pool2d(
        F.pad(fine, (1, 1, 1, 1), mode="replicate"),
        kernel_size=3,
        stride=1,
    )
    structure = F.interpolate(
        structure,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return structure.reshape(batch, frames, height, width)


def _temporal_innovation(video: torch.Tensor) -> torch.Tensor:
    if video.shape[1] <= 2:
        return torch.zeros_like(video)
    previous = torch.cat((video[:, :1], video[:, :-1]), dim=1)
    following = torch.cat((video[:, 1:], video[:, -1:]), dim=1)
    return video - (previous + 2.0 * video + following) * 0.25


def _robust_unit_map(value: torch.Tensor) -> torch.Tensor:
    """Map each batch item to [0,1] without scene-specific fixed thresholds."""

    flat = value.flatten(start_dim=1)
    low = torch.quantile(flat, 0.20, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat, 0.90, dim=1).view(-1, 1, 1)
    return ((value - low) / (high - low).clamp_min(1.0e-6)).clamp_(0.0, 1.0)


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_y0, first_y1, first_x0, first_x1 = first
    second_y0, second_y1, second_x0, second_x1 = second
    overlap_h = max(0, min(first_y1, second_y1) - max(first_y0, second_y0))
    overlap_w = max(0, min(first_x1, second_x1) - max(first_x0, second_x0))
    intersection = overlap_h * overlap_w
    union = (
        (first_y1 - first_y0) * (first_x1 - first_x0)
        + (second_y1 - second_y0) * (second_x1 - second_x0)
        - intersection
    )
    return 0.0 if union <= 0 else float(intersection) / float(union)


def _candidate_component_mean(
    component: torch.Tensor,
    box: tuple[int, int, int, int],
) -> float:
    y0, y1, x0, x1 = box
    return float(component[..., y0:y1, x0:x1].mean().item())


def _select_difficulty_candidates(
    score: torch.Tensor,
    convergence: torch.Tensor,
    temporal_instability: torch.Tensor,
    representation_deficit: torch.Tensor,
    *,
    maximum_regions: int,
    minimum_side_fraction: float,
    maximum_side_fraction: float,
) -> tuple[ROIDifficultyCandidate, ...]:
    """Select diverse multi-scale boxes with deterministic spatial NMS."""

    if score.ndim != 3:
        raise ValueError("difficulty score must have shape B,H,W")
    spatial = score.mean(dim=0)
    height, width = (int(value) for value in spatial.shape)
    short_edge = min(height, width)
    minimum_side = max(4, int(round(short_edge * minimum_side_fraction)))
    maximum_side = max(minimum_side, int(round(short_edge * maximum_side_fraction)))
    sides = sorted(
        {
            minimum_side,
            int(round((minimum_side * maximum_side) ** 0.5)),
            maximum_side,
        }
    )
    raw_candidates: list[tuple[float, tuple[int, int, int, int], int]] = []
    for side in sides:
        side = min(side, height, width)
        stride = max(1, side // 4)
        pooled = F.avg_pool2d(
            spatial.view(1, 1, height, width),
            kernel_size=side,
            stride=stride,
        ).flatten()
        count = min(int(pooled.numel()), maximum_regions * 12)
        if count <= 0:
            continue
        values, indices = torch.topk(pooled, count, sorted=True)
        columns = (width - side) // stride + 1
        for value, index in zip(values.tolist(), indices.tolist()):
            y0 = (index // columns) * stride
            x0 = (index % columns) * stride
            box = (y0, y0 + side, x0, x0 + side)
            # A slight context preference prevents one-pixel noise from always
            # winning while allowing truly compact distant subjects to retain
            # the highest magnification.
            objective = float(value) * (0.88 + 0.12 * side / maximum_side)
            raw_candidates.append((objective, box, side))

    if not raw_candidates:
        return ()
    raw_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    # Spend the first atlas slots across the scene before allowing a second
    # crop from the same coarse area.  This prevents one large foreground
    # subject from consuming the whole budget while several small background
    # structures remain untreated.  Remaining slots still follow the global
    # score order, so clustered difficult content is not forbidden.
    diversity_rows = 2
    diversity_columns = min(3, maximum_regions)
    best_by_cell: dict[
        tuple[int, int], tuple[float, tuple[int, int, int, int], int]
    ] = {}
    for candidate in raw_candidates:
        _, box, _ = candidate
        centre_y = (box[0] + box[1]) * 0.5
        centre_x = (box[2] + box[3]) * 0.5
        cell = (
            min(diversity_rows - 1, int(centre_y * diversity_rows / height)),
            min(
                diversity_columns - 1,
                int(centre_x * diversity_columns / width),
            ),
        )
        best_by_cell.setdefault(cell, candidate)
    frontier = sorted(best_by_cell.values(), key=lambda item: (-item[0], item[1]))
    frontier_ids = {id(candidate) for candidate in frontier}
    raw_candidates = frontier + [
        candidate for candidate in raw_candidates if id(candidate) not in frontier_ids
    ]
    flat = spatial.flatten()
    median = float(flat.median().item())
    mad = float((flat - median).abs().median().item())
    minimum_score = median + 0.75 * max(1.4826 * mad, 1.0e-6)
    if raw_candidates[0][0] <= max(minimum_score, 1.0e-8):
        return ()

    selected: list[ROIDifficultyCandidate] = []
    selected_boxes: list[tuple[int, int, int, int]] = []
    for objective, box, side in raw_candidates:
        if objective < minimum_score:
            continue
        if any(_box_iou(box, accepted) > 0.20 for accepted in selected_boxes):
            continue
        centre_y = (box[0] + box[1]) * 0.5
        centre_x = (box[2] + box[3]) * 0.5
        if any(
            (centre_y - (accepted[0] + accepted[1]) * 0.5) ** 2
            + (centre_x - (accepted[2] + accepted[3]) * 0.5) ** 2
            < (0.80 * min(side, accepted[1] - accepted[0])) ** 2
            for accepted in selected_boxes
        ):
            continue
        y0, y1, x0, x1 = box
        selected_boxes.append(box)
        selected.append(
            ROIDifficultyCandidate(
                box=(x0 / width, y0 / height, x1 / width, y1 / height),
                score=float(objective),
                side=side,
                convergence=_candidate_component_mean(convergence, box),
                temporal_instability=_candidate_component_mean(
                    temporal_instability, box
                ),
                representation_deficit=_candidate_component_mean(
                    representation_deficit, box
                ),
            )
        )
        if len(selected) >= maximum_regions:
            break
    return tuple(selected)


def detect_difficult_regions(
    motion_video: torch.Tensor,
    previous_prediction: torch.Tensor | None,
    refined_video: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    maximum_regions: int = 4,
    minimum_side_fraction: float = 0.12,
    maximum_side_fraction: float = 0.28,
) -> tuple[tuple[tuple[float, float, float, float], ...], dict]:
    """Find H3 regions that deserve a magnified continuation of denoising.

    Three model-local signals are combined:

    * disagreement between the final two clean-state predictions (unfinished
      denoising),
    * high-frequency temporal innovation unsupported by the accepted motion
      trajectory (flicker), and
    * compact source structure for which the refined target delivered little
      new target-grid detail (representation deficit).

    No detector, VAE decode, text model or serial crop inference is required.
    The returned boxes are spatially diverse and can be packed into one H3
    atlas evaluation batch.
    """

    if motion_video.shape != refined_video.shape or motion_video.ndim != 5:
        raise ValueError("difficulty inputs must share B,C,T,H,W shape")
    if previous_prediction is not None and previous_prediction.shape != refined_video.shape:
        raise ValueError("previous prediction must match the refined video shape")
    height, width = int(refined_video.shape[-2]), int(refined_video.shape[-1])
    if not (0 < source_height <= height and 0 < source_width <= width):
        raise ValueError("source geometry must fit the refined latent")
    if not 1 <= maximum_regions <= 9:
        raise ValueError("automatic ROI selection supports one to nine regions")
    if not 0.0 < minimum_side_fraction <= maximum_side_fraction <= 1.0:
        raise ValueError("automatic ROI side fractions are invalid")

    final_delta = refined_video.float() - motion_video.float()
    final_detail = _spatial_cutoff_detail(
        final_delta,
        source_height=source_height,
        source_width=source_width,
    )
    delivered = final_detail.square().mean(dim=2).sqrt()
    delivered_spatial = torch.quantile(delivered, 0.65, dim=1)

    if previous_prediction is None:
        convergence_spatial = torch.zeros_like(delivered_spatial)
    else:
        disagreement = refined_video.float() - previous_prediction.float()
        disagreement_detail = _spatial_cutoff_detail(
            disagreement,
            source_height=source_height,
            source_width=source_width,
        )
        convergence = disagreement_detail.square().mean(dim=2).sqrt()
        convergence_spatial = torch.quantile(convergence, 0.75, dim=1)
        del disagreement_detail, convergence

    detail_innovation = _temporal_innovation(final_detail)
    motion_innovation = _temporal_innovation(
        _channel_sketch(motion_video).permute(0, 2, 1, 3, 4)
    )
    temporal = (
        detail_innovation.square().mean(dim=2).sqrt()
        - 1.25 * motion_innovation.square().mean(dim=2).sqrt()
    ).clamp_min_(0.0)
    temporal_spatial = torch.quantile(temporal, 0.75, dim=1)
    del detail_innovation, motion_innovation, temporal, final_detail

    structure = _source_structure_map(
        motion_video,
        source_height=source_height,
        source_width=source_width,
    )
    structure_spatial = torch.quantile(structure, 0.65, dim=1)
    del structure

    convergence_unit = _robust_unit_map(convergence_spatial)
    temporal_unit = _robust_unit_map(temporal_spatial)
    structure_unit = _robust_unit_map(structure_spatial)
    delivered_unit = _robust_unit_map(delivered_spatial)
    representation_deficit = structure_unit * (1.0 - delivered_unit)
    difficulty = (
        0.45 * convergence_unit * (0.40 + 0.60 * structure_unit)
        + 0.25 * temporal_unit
        + 0.30 * representation_deficit
    )
    difficulty = F.avg_pool2d(
        difficulty.unsqueeze(1), kernel_size=3, stride=1, padding=1
    ).squeeze(1)

    candidates = _select_difficulty_candidates(
        difficulty,
        convergence_unit,
        temporal_unit,
        representation_deficit,
        maximum_regions=maximum_regions,
        minimum_side_fraction=minimum_side_fraction,
        maximum_side_fraction=maximum_side_fraction,
    )
    regions = tuple(candidate.box for candidate in candidates)
    profile = {
        "policy": "h3_cross_step_structural_difficulty_v1",
        "detector_model": None,
        "intermediate_decode": False,
        "source_latent_shape": [source_height, source_width],
        "target_latent_shape": [height, width],
        "maximum_regions": maximum_regions,
        "selected_regions": [
            {
                "normalized_box": [round(value, 6) for value in candidate.box],
                "score": round(candidate.score, 6),
                "latent_side": candidate.side,
                "convergence": round(candidate.convergence, 6),
                "temporal_instability": round(
                    candidate.temporal_instability, 6
                ),
                "representation_deficit": round(
                    candidate.representation_deficit, 6
                ),
            }
            for candidate in candidates
        ],
    }
    return regions, profile


def _normalized_box_to_slice(
    box: Sequence[float],
    *,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("each ROI must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = (float(value) for value in box)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("normalized ROI coordinates must lie inside [0,1]")
    left = max(0, min(width - 1, int(round(x0 * width))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    return top, bottom, left, right


def build_region_atlas(
    video: torch.Tensor,
    regions: Sequence[Sequence[float]],
    *,
    rows: int = 2,
    columns: int = 3,
    atlas_height: int | None = None,
    atlas_width: int | None = None,
) -> tuple[torch.Tensor, tuple[ROIAtlasRecord, ...]]:
    """Pack normalized full-frame crops into square cells on the same canvas.

    Unused cells repeat real selected motion, matching the proven FaceRefine
    atlas behavior and avoiding large empty areas that are out of distribution
    for the video DiT.  Only records for unique requested regions are returned,
    so repeated filler cells are never merged back.
    """

    if video.ndim != 5:
        raise ValueError("H3 video latent must have shape B,C,T,H,W")
    if rows <= 0 or columns <= 0:
        raise ValueError("atlas grid dimensions must be positive")
    if not regions:
        raise ValueError("at least one ROI is required")
    capacity = rows * columns
    if len(regions) > capacity:
        raise ValueError(f"atlas grid can hold at most {capacity} regions")

    batch, channels, frames, height, width = video.shape
    target_height = height if atlas_height is None else int(atlas_height)
    target_width = width if atlas_width is None else int(atlas_width)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("atlas spatial size must be positive")
    cell = min(target_height // rows, target_width // columns)
    if cell < 2:
        raise ValueError("atlas cells are too small")
    used_height = cell * rows
    used_width = cell * columns
    margin_y = (target_height - used_height) // 2
    margin_x = (target_width - used_width) // 2
    atlas = torch.zeros(
        (batch, channels, frames, target_height, target_width),
        device=video.device,
        dtype=torch.float32,
    )
    records: list[ROIAtlasRecord] = []
    unique_tiles: list[torch.Tensor] = []

    for index, box in enumerate(regions):
        source_y0, source_y1, source_x0, source_x1 = _normalized_box_to_slice(
            box, height=height, width=width
        )
        row, column = divmod(index, columns)
        atlas_y0 = margin_y + row * cell
        atlas_y1 = atlas_y0 + cell
        atlas_x0 = margin_x + column * cell
        atlas_x1 = atlas_x0 + cell
        crop = video[
            ...,
            source_y0:source_y1,
            source_x0:source_x1,
        ]
        tile = _resize_video_spatial(crop, height=cell, width=cell)
        atlas[..., atlas_y0:atlas_y1, atlas_x0:atlas_x1] = tile
        unique_tiles.append(tile)
        records.append(
            ROIAtlasRecord(
                source_y0=source_y0,
                source_y1=source_y1,
                source_x0=source_x0,
                source_x1=source_x1,
                atlas_y0=atlas_y0,
                atlas_y1=atlas_y1,
                atlas_x0=atlas_x0,
                atlas_x1=atlas_x1,
            )
        )

    # Repeat real motion into unused cells.  Cycling all unique tiles avoids
    # over-representing the final region when the atlas is sparsely occupied.
    for index in range(len(regions), capacity):
        row, column = divmod(index, columns)
        y0 = margin_y + row * cell
        x0 = margin_x + column * cell
        atlas[..., y0 : y0 + cell, x0 : x0 + cell] = unique_tiles[
            index % len(unique_tiles)
        ]

    # Extend the outermost real cells through the small aspect-ratio margins.
    # A 16:9 target with a 3x2 square grid has only six latent columns on each
    # side, but filling them is friendlier than a featureless latent border.
    if margin_x:
        atlas[..., :margin_x] = atlas[..., margin_x : margin_x + 1]
        right = margin_x + used_width
        atlas[..., right:] = atlas[..., right - 1 : right]
    if margin_y:
        atlas[..., :margin_y, :] = atlas[..., margin_y : margin_y + 1, :]
        bottom = margin_y + used_height
        atlas[..., bottom:, :] = atlas[..., bottom - 1 : bottom, :]
    return atlas, tuple(records)


def remap_region_atlas_positions(
    layout,
    records: Sequence[ROIAtlasRecord],
    *,
    atlas_height: int,
    atlas_width: int,
    full_height: int,
    full_width: int,
    patch_size: int = 2,
) -> None:
    """Give atlas storage rows their real full-frame continuous RoPE positions.

    The rectangular atlas remains convenient storage for patch projection and
    output reconstruction.  Attention, however, should not interpret its
    cells as quadrants of a synthetic collage.  Each fine atlas token instead
    receives a fractional coordinate inside the source ROI, yielding a sparse
    high-density sampling of the original scene without changing sequence
    length or model weights.
    """

    if patch_size <= 0:
        raise ValueError("patch size must be positive")
    if any(value <= 0 for value in (atlas_height, atlas_width, full_height, full_width)):
        raise ValueError("atlas and full-frame geometry must be positive")
    if atlas_height % patch_size or atlas_width % patch_size:
        raise ValueError("atlas geometry must be divisible by patch size")
    grid_height = atlas_height // patch_size
    grid_width = atlas_width // patch_size
    video_segment = layout.segment("video", last=True)
    frame_tokens = grid_height * grid_width
    if video_segment.length % frame_tokens:
        raise ValueError("atlas layout does not contain complete video frames")
    latent_frames = video_segment.length // frame_tokens
    positions = layout.position_ids[
        video_segment.start : video_segment.stop
    ].reshape(latent_frames, grid_height, grid_width, 3)
    spatial = positions[0, ..., 1:].clone()
    full_area = math.sqrt(full_height * full_width)

    def coordinate(values: torch.Tensor, size: int) -> torch.Tensor:
        return (
            values / full_area + (1.0 - size / full_area) / 2.0
        ) * 32.0

    for record in records:
        boundaries = (
            record.atlas_y0,
            record.atlas_y1,
            record.atlas_x0,
            record.atlas_x1,
        )
        if any(value % patch_size for value in boundaries):
            raise ValueError("atlas ROI boundaries must align to patch tokens")
        y0, y1 = record.atlas_y0 // patch_size, record.atlas_y1 // patch_size
        x0, x1 = record.atlas_x0 // patch_size, record.atlas_x1 // patch_size
        rows, columns = y1 - y0, x1 - x0
        source_y = record.source_y0 + torch.arange(
            rows, dtype=torch.float64
        ) * (record.source_height / rows)
        source_x = record.source_x0 + torch.arange(
            columns, dtype=torch.float64
        ) * (record.source_width / columns)
        yy, xx = torch.meshgrid(
            coordinate(source_y, full_height),
            coordinate(source_x, full_width),
            indexing="ij",
        )
        spatial[y0:y1, x0:x1, 0] = yy
        spatial[y0:y1, x0:x1, 1] = xx

    positions[..., 1:] = spatial.unsqueeze(0)

    # H3 places the two packed audio rows at the left and right edges of the
    # visual canvas.  Keep those anchors in the original full-frame RoPE
    # domain as well; otherwise a square storage atlas silently narrows the
    # audio/video geometry even though every visual token uses source-space
    # coordinates.
    audio_segment = layout.segment("audio", last=True)
    if audio_segment.length % 2:
        raise ValueError("H3 packed audio segment must contain two equal rows")
    audio_frames = audio_segment.length // 2
    full_x = coordinate(
        torch.tensor([0.0, float(full_width - patch_size)], dtype=torch.float64),
        full_width,
    )
    layout.position_ids[
        audio_segment.start : audio_segment.start + audio_frames, 2
    ] = full_x[0]
    layout.position_ids[
        audio_segment.start + audio_frames : audio_segment.stop, 2
    ] = full_x[1]
    layout.device_rope_table = None


def _temporal_outlier_filter(
    delta: torch.Tensor,
    motion: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    if strength <= 0.0 or delta.shape[2] <= 2:
        return delta
    sequence = delta.permute(0, 2, 1, 3, 4)
    previous = torch.cat((sequence[:, :1], sequence[:, :-1]), dim=1)
    following = torch.cat((sequence[:, 1:], sequence[:, -1:]), dim=1)
    smoothed = (previous + 2.0 * sequence + following) * 0.25
    innovation = sequence - smoothed

    motion_sequence = motion.float().permute(0, 2, 1, 3, 4)
    motion_previous = torch.cat(
        (motion_sequence[:, :1], motion_sequence[:, :-1]), dim=1
    )
    motion_following = torch.cat(
        (motion_sequence[:, 1:], motion_sequence[:, -1:]), dim=1
    )
    motion_innovation = motion_sequence - (
        motion_previous + 2.0 * motion_sequence + motion_following
    ) * 0.25
    score = innovation.square().mean(dim=2).sqrt()
    allowance = 1.25 * motion_innovation.square().mean(dim=2).sqrt()
    excess = (score - allowance).clamp_min(0.0)
    median = excess.median(dim=1, keepdim=True).values
    mad = (excess - median).abs().median(dim=1, keepdim=True).values
    threshold = median + 2.5 * (1.4826 * mad).clamp_min(1e-6)
    weight = (
        (excess - threshold).clamp_min(0.0) / excess.clamp_min(1e-6)
    ).unsqueeze(2)
    filtered = sequence + float(strength) * weight * (smoothed - sequence)
    return filtered.permute(0, 2, 1, 3, 4).contiguous()


def _temporal_motion_gated_filter(
    delta: torch.Tensor,
    motion: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Suppress ROI-only temporal innovation unsupported by source motion.

    A magnified region pass can invent useful spatial frequencies while also
    redrawing them independently at each latent frame.  The accepted global
    trajectory already tells us where temporal change is real.  Preserve
    residual changes in those moving areas and low-pass only the excess in
    comparatively static areas.  This operates on the generated residual, so
    the base trajectory and constant new detail are both exact fixed points.
    """

    if strength <= 0.0 or delta.shape[2] <= 2:
        return delta
    sequence = delta.float().permute(0, 2, 1, 3, 4)
    previous = torch.cat((sequence[:, :1], sequence[:, :-1]), dim=1)
    following = torch.cat((sequence[:, 1:], sequence[:, -1:]), dim=1)
    smoothed = (previous + 2.0 * sequence + following) * 0.25
    innovation = sequence - smoothed

    motion_sequence = motion.float().permute(0, 2, 1, 3, 4)
    motion_previous = torch.cat(
        (motion_sequence[:, :1], motion_sequence[:, :-1]), dim=1
    )
    motion_following = torch.cat(
        (motion_sequence[:, 1:], motion_sequence[:, -1:]), dim=1
    )
    motion_smoothed = (
        motion_previous + 2.0 * motion_sequence + motion_following
    ) * 0.25
    motion_score = (
        motion_sequence - motion_smoothed
    ).square().mean(dim=2).sqrt()

    # Absolute latent-motion magnitudes are much larger than the ROI residual,
    # so comparing them directly leaves the gate effectively disabled.  Use a
    # request-local median as a rank reference instead: static locations tend
    # toward one, highly dynamic locations toward zero.  The inverse-square
    # curve stays smooth across a moving object's boundary.
    motion_reference = torch.quantile(
        motion_score.flatten(1),
        0.5,
        dim=1,
    ).reshape(-1, 1, 1, 1).clamp_min(1e-6)
    static_confidence = 1.0 / (
        1.0 + (motion_score / motion_reference).square()
    )
    batch, frames, height, width = static_confidence.shape
    gate = F.avg_pool2d(
        static_confidence.reshape(batch * frames, 1, height, width),
        kernel_size=3,
        stride=1,
        padding=1,
    ).reshape(batch, frames, height, width)
    filtered = sequence + float(strength) * gate.unsqueeze(2) * (
        smoothed - sequence
    )
    return filtered.permute(0, 2, 1, 3, 4).contiguous()


def _temporal_residual_lowpass(
    delta: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Low-pass only newly generated ROI detail, never the base trajectory."""

    if strength <= 0.0 or delta.shape[2] <= 2:
        return delta
    sequence = delta.float().permute(0, 2, 1, 3, 4)
    previous = torch.cat((sequence[:, :1], sequence[:, :-1]), dim=1)
    following = torch.cat((sequence[:, 1:], sequence[:, -1:]), dim=1)
    smoothed = (previous + 2.0 * sequence + following) * 0.25
    filtered = sequence + float(strength) * (smoothed - sequence)
    return filtered.permute(0, 2, 1, 3, 4).contiguous()


def _feather_mask(
    height: int,
    width: int,
    *,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    radius = max(0, min(int(radius), (min(height, width) - 1) // 2))
    if radius == 0:
        return torch.ones((1, 1, 1, height, width), device=device)
    y = torch.ones(height, device=device, dtype=torch.float32)
    x = torch.ones(width, device=device, dtype=torch.float32)
    # Do not force the boundary to zero: adjacent source pixels still need a
    # small coherent update, while the centre receives the complete residual.
    ramp = torch.linspace(0.15, 1.0, radius + 1, device=device)[:-1]
    y[:radius] = ramp
    y[-radius:] = torch.flip(ramp, dims=(0,))
    x[:radius] = ramp
    x[-radius:] = torch.flip(ramp, dims=(0,))
    return (y[:, None] * x[None, :]).view(1, 1, 1, height, width)


def merge_region_atlas(
    base_video: torch.Tensor,
    clean_atlas: torch.Tensor,
    refined_atlas: torch.Tensor,
    records: Sequence[ROIAtlasRecord],
    *,
    source_scale: float,
    low_frequency_gain: float = 0.08,
    mid_frequency_gain: float = 0.0,
    coarse_scale: float = 0.25,
    blend: float = 0.9,
    feather_radius: int = 2,
    temporal_outlier_strength: float = 0.65,
    temporal_filter: str = "outlier",
) -> torch.Tensor:
    """Reduce refined atlas cells and merge only target-grid detail.

    ``source_scale`` is the source-to-target short-edge ratio (480/768 in the
    first experiment).  Frequencies representable at the source grid form the
    protected low band; the magnified H3 pass owns the band newly available on
    the requested 720P grid.
    """

    if (
        clean_atlas.shape != refined_atlas.shape
        or base_video.ndim != 5
        or clean_atlas.ndim != 5
        or base_video.shape[:3] != clean_atlas.shape[:3]
    ):
        raise ValueError(
            "base video and atlas tensors must share B,C,T; atlas shapes must match"
        )
    if not 0.0 < float(source_scale) <= 1.0:
        raise ValueError("source scale must lie inside (0,1]")
    if not 0.0 <= float(low_frequency_gain) <= 1.0:
        raise ValueError("low-frequency gain must lie inside [0,1]")
    if not 0.0 <= float(mid_frequency_gain) <= 1.0:
        raise ValueError("mid-frequency gain must lie inside [0,1]")
    if not 0.0 < float(coarse_scale) <= float(source_scale):
        raise ValueError("coarse scale must lie inside (0, source_scale]")
    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError("blend must lie inside [0,1]")
    if not 0.0 <= float(temporal_outlier_strength) <= 1.0:
        raise ValueError("temporal outlier strength must lie inside [0,1]")
    if temporal_filter not in ("outlier", "motion_gated", "residual_lowpass"):
        raise ValueError(
            "temporal filter must be outlier, motion_gated or residual_lowpass"
        )

    base = base_video.float()
    accumulated = torch.zeros_like(base)
    weights = torch.zeros(
        (base.shape[0], 1, 1, base.shape[-2], base.shape[-1]),
        device=base.device,
        dtype=torch.float32,
    )
    for record in records:
        tile = refined_atlas[
            ...,
            record.atlas_y0 : record.atlas_y1,
            record.atlas_x0 : record.atlas_x1,
        ]
        refined = _resize_video_spatial(
            tile,
            height=record.source_height,
            width=record.source_width,
        )
        clean_tile = clean_atlas[
            ...,
            record.atlas_y0 : record.atlas_y1,
            record.atlas_x0 : record.atlas_x1,
        ]
        clean_reduced = _resize_video_spatial(
            clean_tile,
            height=record.source_height,
            width=record.source_width,
        )
        motion = base[
            ...,
            record.source_y0 : record.source_y1,
            record.source_x0 : record.source_x1,
        ]
        # Subtract the exact atlas round-trip, not the full-frame crop.  This
        # guarantees that a no-op H3 pass is also a no-op merge and prevents
        # interpolation error from masquerading as generated detail.
        delta = refined.float() - clean_reduced.float()
        source_band_height = max(
            1, int(round(record.source_height * source_scale))
        )
        source_band_width = max(
            1, int(round(record.source_width * source_scale))
        )
        source_band = _resize_video_spatial(
            delta,
            height=source_band_height,
            width=source_band_width,
            mode="area",
        )
        source_band = _resize_video_spatial(
            source_band,
            height=record.source_height,
            width=record.source_width,
        )
        coarse_height = max(1, int(round(record.source_height * coarse_scale)))
        coarse_width = max(1, int(round(record.source_width * coarse_scale)))
        coarse = _resize_video_spatial(
            delta,
            height=coarse_height,
            width=coarse_width,
            mode="area",
        )
        coarse = _resize_video_spatial(
            coarse,
            height=record.source_height,
            width=record.source_width,
        )
        high = delta - source_band
        middle = source_band - coarse
        accepted = (
            high
            + float(mid_frequency_gain) * middle
            + float(low_frequency_gain) * coarse
        )
        if temporal_filter == "residual_lowpass":
            accepted = _temporal_residual_lowpass(
                accepted,
                strength=float(temporal_outlier_strength),
            )
        elif temporal_filter == "motion_gated":
            accepted = _temporal_motion_gated_filter(
                accepted,
                motion,
                strength=float(temporal_outlier_strength),
            )
        else:
            accepted = _temporal_outlier_filter(
                accepted,
                motion,
                strength=float(temporal_outlier_strength),
            )
        mask = _feather_mask(
            record.source_height,
            record.source_width,
            radius=feather_radius,
            device=base.device,
        )
        accumulated[
            ...,
            record.source_y0 : record.source_y1,
            record.source_x0 : record.source_x1,
        ] += accepted * mask
        weights[
            ...,
            record.source_y0 : record.source_y1,
            record.source_x0 : record.source_x1,
        ] += mask

    normalized = accumulated / weights.clamp_min(1.0)
    merged = base + float(blend) * normalized
    return merged.to(dtype=base_video.dtype)


def atlas_plan_dict(
    records: Sequence[ROIAtlasRecord],
    *,
    source_scale: float,
    steps: int,
    denoise: float,
    atlas_shape: tuple[int, int] | None = None,
) -> dict:
    result = {
        "policy": "magnified_region_atlas_h3_refinement_v1",
        "steps": int(steps),
        "denoise": float(denoise),
        "source_scale": float(source_scale),
        "region_count": len(records),
        "regions": [
            {
                "source_latent": [
                    item.source_x0,
                    item.source_y0,
                    item.source_x1,
                    item.source_y1,
                ],
                "atlas_latent": [
                    item.atlas_x0,
                    item.atlas_y0,
                    item.atlas_x1,
                    item.atlas_y1,
                ],
                "linear_magnification": round(
                    min(
                        (item.atlas_x1 - item.atlas_x0)
                        / max(item.source_width, 1),
                        (item.atlas_y1 - item.atlas_y0)
                        / max(item.source_height, 1),
                    ),
                    3,
                ),
            }
            for item in records
        ],
    }
    if atlas_shape is not None:
        result["atlas_latent_shape"] = [int(atlas_shape[0]), int(atlas_shape[1])]
    return result
