"""Planning primitives from ComfyUI-H3-FaceRefine-Accelerated.

Kept as a production-local module so the service uses the same global face
tracking, ranking and progressive-window rules as the published ComfyUI node
without importing ComfyUI itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _state(box: Sequence[float]) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    return np.asarray(((x0 + x1) * 0.5, (y0 + y1) * 0.5, w, h), dtype=np.float64)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection)
    return intersection / union


def _pair_cost(a: Sequence[float], b: Sequence[float], width: int, height: int) -> float:
    pa, pb = _state(a), _state(b)
    diagonal = max(math.hypot(width, height), 1.0)
    motion = math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / diagonal
    size = abs(math.log(max(pb[3], 1.0) / max(pa[3], 1.0)))
    return motion * 5.0 + size * 0.75 + (1.0 - _iou(a, b)) * 0.35


@dataclass
class FaceTrack:
    track_id: int
    shot_index: int
    start: int
    end: int
    boxes: list[Sequence[float] | None]
    confidences: list[float]
    detected: list[bool]
    # Authored-shot bounds are distinct from the first/last successful
    # detection.  Importance filtering must measure visibility against the
    # complete shot; otherwise a one-frame false positive has a misleading
    # visibility ratio of 100% and can outrank the real subject.
    shot_start: int = 0
    shot_end: int = 0
    score: float = 0.0
    selected: bool = False
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def build_tracks(
    all_boxes: Sequence[Sequence[Sequence[float]]],
    all_confs: Sequence[Sequence[float]],
    segments: Sequence[Sequence[int]],
    width: int,
    height: int,
    *,
    max_gap: int = 5,
    max_cost: float = 1.45,
) -> list[FaceTrack]:
    """Build geometry tracks with globally optimal frame-to-frame assignments.

    Tracks reset at authored cuts.  Unmatched detections create new tracks, which
    means a person entering after the first frame is not silently ignored.
    """

    frame_count = len(all_boxes)
    tracks: list[FaceTrack] = []
    next_id = 0
    for shot_index, bounds in enumerate(segments):
        shot_start, shot_end = int(bounds[0]), int(bounds[1])
        active: dict[int, tuple[int, Sequence[float]]] = {}
        shot_tracks: dict[int, FaceTrack] = {}
        for frame in range(shot_start, shot_end):
            boxes = [tuple(float(v) for v in q) for q in all_boxes[frame]]
            confs = [float(v) for v in all_confs[frame]]
            candidates = [(tid, info) for tid, info in active.items() if frame - info[0] <= max_gap]
            assignments: list[tuple[int, int]] = []
            if candidates and boxes:
                matrix = np.asarray([
                    [_pair_cost(info[1], box, width, height) for box in boxes]
                    for _tid, info in candidates
                ], dtype=np.float64)
                rows, cols = linear_sum_assignment(matrix)
                assignments = [
                    (candidates[int(r)][0], int(c))
                    for r, c in zip(rows, cols)
                    if float(matrix[int(r), int(c)]) <= float(max_cost)
                ]

            matched_boxes: set[int] = set()
            for track_id, box_index in assignments:
                box = boxes[box_index]
                track = shot_tracks[track_id]
                track.boxes[frame] = box
                track.confidences[frame] = confs[box_index] if box_index < len(confs) else 1.0
                track.detected[frame] = True
                track.end = frame + 1
                active[track_id] = (frame, box)
                matched_boxes.add(box_index)

            for box_index, box in enumerate(boxes):
                if box_index in matched_boxes:
                    continue
                track_id = next_id
                next_id += 1
                track = FaceTrack(
                    track_id=track_id,
                    shot_index=shot_index,
                    start=frame,
                    end=frame + 1,
                    boxes=[None] * frame_count,
                    confidences=[0.0] * frame_count,
                    detected=[False] * frame_count,
                    shot_start=shot_start,
                    shot_end=shot_end,
                )
                track.boxes[frame] = box
                track.confidences[frame] = confs[box_index] if box_index < len(confs) else 1.0
                track.detected[frame] = True
                shot_tracks[track_id] = track
                active[track_id] = (frame, box)

            active = {tid: info for tid, info in active.items() if frame - info[0] <= max_gap}
        tracks.extend(shot_tracks.values())
    return tracks


def interpolate_track(track: FaceTrack, smooth_window: int = 17, size_window: int = 31):
    """Return [frames,4] centre/size states and soft detection weights."""

    valid = np.asarray(track.detected, dtype=bool)
    if not valid.any():
        raise ValueError("face track contains no detections")
    states = np.zeros((len(valid), 4), dtype=np.float64)
    for index, box in enumerate(track.boxes):
        if box is not None:
            states[index] = _state(box)
    indices = np.arange(len(valid))
    known = indices[valid]
    for column in range(4):
        states[:, column] = np.interp(indices, known, states[known, column])

    def smooth(values: np.ndarray, window: int) -> np.ndarray:
        window = max(1, int(window))
        if window <= 1 or len(values) <= 2:
            return values
        if window % 2 == 0:
            window += 1
        radius = window // 2
        sigma = max(window / 6.0, 0.5)
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        padded = np.pad(values, (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    states[:, 0] = smooth(states[:, 0], smooth_window)
    states[:, 1] = smooth(states[:, 1], smooth_window)
    states[:, 2] = smooth(states[:, 2], size_window)
    states[:, 3] = smooth(states[:, 3], size_window)
    weights = np.clip(smooth(valid.astype(np.float64), max(9, smooth_window // 2)), 0.0, 1.0)
    # Never paste an interpolated track outside the span where that subject existed.
    weights[: int(known[0])] = 0.0
    weights[int(known[-1]) + 1 :] = 0.0
    return states, valid, weights


def score_tracks(
    tracks: Iterable[FaceTrack],
    width: int,
    height: int,
    *,
    max_faces: int,
    min_face_height_px: float,
    min_visibility_ratio: float,
    clear_face_height_px: float,
    skip_clear_faces: bool,
    max_repair_face_height_px: float | None = None,
    fps: float = 24.0,
    sharpness: dict[int, float] | None = None,
) -> list[FaceTrack]:
    """Apply cheap hard filters, score eligible tracks and mark global Top-K."""

    tracks = list(tracks)
    sharpness = sharpness or {}
    diagonal = max(math.hypot(width, height), 1.0)
    eligible: list[FaceTrack] = []
    for track in tracks:
        indices = [i for i, found in enumerate(track.detected) if found]
        heights = np.asarray([
            float(track.boxes[i][3]) - float(track.boxes[i][1]) for i in indices
        ], dtype=np.float64)
        centres = np.asarray([
            _state(track.boxes[i])[:2] for i in indices
        ], dtype=np.float64)
        conf = np.asarray([track.confidences[i] for i in indices], dtype=np.float64)
        shot_start = max(0, int(track.shot_start))
        shot_end = int(track.shot_end) if int(track.shot_end) > shot_start else len(track.boxes)
        shot_frames = max(1, shot_end - shot_start)
        visibility = len(indices) / shot_frames
        p90_height = float(np.percentile(heights, 90)) if len(heights) else 0.0
        max_height = float(np.max(heights)) if len(heights) else 0.0
        median_height = float(np.median(heights)) if len(heights) else 0.0
        mean_conf = float(conf.mean()) if len(conf) else 0.0
        if len(centres):
            distances = np.hypot(centres[:, 0] - width * 0.5, centres[:, 1] - height * 0.5)
            centre = float(np.clip(1.0 - distances.mean() / (0.6 * diagonal), 0.0, 1.0))
        else:
            centre = 0.0
        blur = float(np.clip(1.0 - sharpness.get(track.track_id, 0.35), 0.0, 1.0))
        repair_need = float(np.clip((clear_face_height_px - median_height) /
                                    max(clear_face_height_px - min_face_height_px, 1.0), 0.0, 1.0))
        track.metrics = {
            "visible_frames": float(len(indices)),
            "visibility_ratio": visibility,
            "median_face_height_px": median_height,
            "p90_face_height_px": p90_height,
            "max_face_height_px": max_height,
            "mean_confidence": mean_conf,
            "centre_score": centre,
            "sharpness": float(sharpness.get(track.track_id, 0.35)),
            "repair_need": repair_need,
        }
        if p90_height < float(min_face_height_px):
            track.reason = f"skipped: p90 face height {p90_height:.1f}px is below {min_face_height_px:.1f}px"
            continue
        if visibility < float(min_visibility_ratio):
            track.reason = f"skipped: visible in only {visibility:.0%} of its shot"
            continue
        if mean_conf < 0.12:
            track.reason = f"skipped: detector confidence {mean_conf:.2f} is too low"
            continue
        if (
            max_repair_face_height_px is not None
            and max_height > float(max_repair_face_height_px)
        ):
            track.reason = (
                f"skipped: face height {max_height:.1f}px cannot obtain the "
                "minimum Atlas enlargement"
            )
            continue
        if skip_clear_faces and median_height >= float(clear_face_height_px) and blur < 0.30:
            track.reason = f"skipped: already clear at {median_height:.1f}px"
            continue
        duration = min(1.0, len(indices) / max(1.0, 2.0 * float(fps)))
        reliability = min(1.0, mean_conf) * visibility
        # Rank only by evidence measured from the source video. Dividing by
        # 0.92 normalizes the remaining five terms after removing the old
        # manual-preference term without changing their relative importance.
        track.score = (
            0.34 * repair_need + 0.17 * blur + 0.16 * duration +
            0.13 * centre + 0.12 * reliability
        ) / 0.92
        eligible.append(track)

    eligible.sort(key=lambda item: (-item.score, item.track_id))
    selected_ids = {item.track_id for item in eligible[: max(0, int(max_faces))]}
    for track in tracks:
        if track.track_id in selected_ids:
            track.selected = True
            track.reason = f"selected: importance {track.score:.3f}"
        elif not track.reason:
            track.reason = f"skipped: below global Top-{int(max_faces)} cutoff ({track.score:.3f})"
    return tracks


def h3_padded_length(valid_frames: int, maximum: int = 362) -> int:
    """Smallest legal 17k+5 H3 length that can contain ``valid_frames``."""

    valid_frames = max(1, int(valid_frames))
    if valid_frames > maximum:
        raise ValueError(f"window has {valid_frames} frames; H3 supports at most {maximum}")
    if valid_frames <= 5:
        return 5
    length = 5 + 17 * math.ceil((valid_frames - 5) / 17)
    if length > maximum:
        raise ValueError(f"window pads to {length} frames, above H3 maximum {maximum}")
    return int(length)


def plan_windows(
    segments: Sequence[Sequence[int]],
    fps: float,
    max_window_seconds: float,
    overlap_seconds: float,
) -> list[dict]:
    """Split each shot and return normalized overlap weights for source frames."""

    requested_frames = max(1, int(round(float(fps) * float(max_window_seconds))))
    # H3 must run on a 17k+5 timeline. Treat the user's seconds as a target and
    # use the nearest legal upper bucket, avoiding a tiny fourth job for a
    # 15-second / 362-frame clip split around five seconds.
    max_frames = h3_padded_length(min(requested_frames, 362))
    overlap = max(0, min(max_frames - 1, int(round(float(fps) * float(overlap_seconds)))))
    raw: list[tuple[int, int]] = []
    for a0, b0 in segments:
        a, b = int(a0), int(b0)
        if b <= a:
            continue
        length = b - a
        if length <= max_frames:
            raw.append((a, b))
            continue
        count = int(math.ceil(length / max_frames))
        edges = [a + int(round(length * index / count)) for index in range(count + 1)]
        base_max = max(edges[index + 1] - edges[index] for index in range(count))
        actual_overlap = min(overlap, max(0, max_frames - base_max))
        left = actual_overlap // 2
        right = actual_overlap - left
        for index in range(count):
            start = edges[index] - (left if index else 0)
            end = edges[index + 1] + (right if index + 1 < count else 0)
            raw.append((start, end))

    coverage: dict[int, float] = {}
    provisional: list[np.ndarray] = []
    for start, end in raw:
        n = end - start
        weights = np.ones(n, dtype=np.float64)
        if overlap and n > 1:
            ramp = min(overlap, n // 2)
            if ramp:
                if any(previous_end > start for _previous_start, previous_end in raw[: len(provisional)]):
                    weights[:ramp] = np.linspace(1.0 / (ramp + 1), ramp / (ramp + 1), ramp)
                if any(next_start < end for next_start, _next_end in raw[len(provisional) + 1 :]):
                    weights[-ramp:] = np.minimum(
                        weights[-ramp:], np.linspace(ramp / (ramp + 1), 1.0 / (ramp + 1), ramp)
                    )
        provisional.append(weights)
        for offset, weight in enumerate(weights):
            coverage[start + offset] = coverage.get(start + offset, 0.0) + float(weight)

    windows: list[dict] = []
    for index, ((start, end), weights) in enumerate(zip(raw, provisional)):
        normalized = [float(weight) / max(coverage[start + offset], 1e-8)
                      for offset, weight in enumerate(weights)]
        windows.append({
            "index": index,
            "start": start,
            "end": end,
            "valid_frames": end - start,
            "padded_frames": h3_padded_length(end - start),
            "weights": normalized,
        })
    return windows


def _legal_at_or_below(frame_limit: int) -> int:
    """Largest legal H3 frame count (17k+5) that does not exceed a limit."""

    frame_limit = max(5, int(frame_limit))
    return 5 + 17 * max(0, (frame_limit - 5) // 17)


def plan_progressive_windows(
    segments: Sequence[Sequence[int]],
    fps: float,
    max_new_content_seconds: float,
    handoff_context_seconds: float,
) -> list[dict]:
    """Plan causal H3 windows with a protected prefix from the preceding result.

    H3 pixel timelines are ``17k+5`` frames.  The opening window is therefore a
    legal timeline by itself; every continuation consists of one legal context
    prefix plus a multiple of 17 new frames.  ``max_new_content_seconds`` is a
    real upper bound rather than a nearest-grid target.
    """

    fps = max(float(fps), 1e-6)
    requested_max = max(5, int(math.floor(fps * float(max_new_content_seconds) + 1e-9)))
    physical_limit = min(362, _legal_at_or_below(requested_max))

    requested_context = max(5, int(round(fps * float(handoff_context_seconds))))
    context_frames = _legal_at_or_below(requested_context + 8)
    # _legal_at_or_below(x + 8) is nearest-grid alignment, with ties downward.
    context_frames = min(context_frames, physical_limit - 17)
    if context_frames < 5:
        raise ValueError(
            "max_new_content_seconds is too short for an H3 continuation: it must "
            "fit at least 5 context frames and 17 new frames"
        )
    new_limit = ((physical_limit - context_frames) // 17) * 17
    if new_limit < 17:
        raise ValueError("progressive H3 windows need room for at least 17 new frames")

    windows: list[dict] = []
    index = 0
    for shot_index, bounds in enumerate(segments):
        shot_start, shot_end = int(bounds[0]), int(bounds[1])
        if shot_end <= shot_start:
            continue

        opening_end = min(shot_end, shot_start + physical_limit)
        opening_source = list(range(shot_start, opening_end))
        windows.append({
            "index": index,
            "shot_index": shot_index,
            "start": shot_start,
            "end": opening_end,
            "new_start": shot_start,
            "new_end": opening_end,
            "source": opening_source,
            "context_frames": 0,
            "new_frames": len(opening_source),
            "valid_frames": len(opening_source),
            "padded_frames": h3_padded_length(len(opening_source)),
            "weights": [1.0] * len(opening_source),
        })
        index += 1
        cursor = opening_end

        while cursor < shot_end:
            new_end = min(shot_end, cursor + new_limit)
            actual_context = min(context_frames, cursor - shot_start)
            # Every normal continuation uses the same aligned context.  This
            # fallback matters only for an unusually short opening shot.
            context_start = cursor - actual_context
            source = list(range(context_start, cursor)) + list(range(cursor, new_end))
            windows.append({
                "index": index,
                "shot_index": shot_index,
                "start": context_start,
                "end": new_end,
                "new_start": cursor,
                "new_end": new_end,
                "source": source,
                "context_frames": actual_context,
                "new_frames": new_end - cursor,
                "valid_frames": len(source),
                "padded_frames": h3_padded_length(len(source)),
                "weights": [0.0] * actual_context + [1.0] * (new_end - cursor),
            })
            index += 1
            cursor = new_end

    return windows


def atlas_grid(faces_per_atlas: int) -> tuple[int, int]:
    count = max(1, min(9, int(faces_per_atlas)))
    side = 1 if count == 1 else (2 if count <= 4 else 3)
    return side, side


def chunked(values: Sequence, size: int):
    size = max(1, int(size))
    return [list(values[index : index + size]) for index in range(0, len(values), size)]
