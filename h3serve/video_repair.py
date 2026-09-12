"""Native service adapter for ComfyUI-H3-FaceRefine-Accelerated.

The pixel planner, global face ranking, per-frame 2.5x crops, multi-face Atlas,
progressive handoff and face-only stitch mirror the published ComfyUI project.
Only the graph/runtime boundary differs: Atlas jobs are sent to this service's
resident native H3 session instead of ComfyUI's sampler nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable


@dataclass(frozen=True)
class RepairWindow:
    index: int
    start: int
    end: int
    padded_frames: int
    weights: tuple[float, ...]
    source: tuple[int, ...] = ()
    context_frames: int = 0
    new_start: int = 0
    new_end: int = 0

    @property
    def valid_frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RepairRegion:
    x: int
    y: int
    size: int
    score: float
    face_score: float = 0.0
    track_id: int = 0
    quality_need: float = 0.0
    importance: float = 0.0
    visibility: float = 0.0
    face_size: float = 0.0
    positions: tuple[tuple[int, int], ...] = ()
    crop_boxes: tuple[tuple[float, float, float, float], ...] = ()
    face_rects: tuple[tuple[float, float, float, float], ...] = ()
    detected: tuple[bool, ...] = ()
    weights: tuple[float, ...] = ()
    face_heights: tuple[float, ...] = ()


@dataclass(frozen=True)
class PreparedRepairWindow:
    window: RepairWindow
    batch_index: int
    canvas_size: int
    grid_size: int
    source_path: Path
    atlas_path: Path
    manifest_path: Path
    refined_atlas_path: Path
    repaired_path: Path
    chain_id: str = ""
    refined_latents_path: Path | None = None


def h3_padded_length(valid_frames: int, maximum: int = 362) -> int:
    """Smallest legal H3 pixel timeline (17k+5) containing valid_frames."""

    valid_frames = max(1, int(valid_frames))
    if valid_frames > maximum:
        raise ValueError(f"repair window has {valid_frames} frames; maximum is {maximum}")
    if valid_frames <= 5:
        return 5
    length = 5 + 17 * math.ceil((valid_frames - 5) / 17)
    if length > maximum:
        raise ValueError(f"repair window pads to {length} frames; maximum is {maximum}")
    return int(length)


def plan_repair_windows(
    frame_count: int,
    fps: float,
    *,
    minimum_seconds: float = 4.0,
    maximum_seconds: float = 6.0,
    overlap_seconds: float = 0.2,
) -> tuple[RepairWindow, ...]:
    """Plan balanced causal windows with a protected handoff prefix.

    Boundaries are distributed over
    the complete clip instead of greedily filling the opening window, which
    avoids a long first window followed by a short padded tail.  Candidate
    layouts also prefer one repeated legal H3 shape so Triton kernels and
    execution buffers are reused.  Very short clips and mathematically awkward
    durations use the closest layout; duplicated padding is never stitched
    back into the delivered video.
    """

    if frame_count <= 0 or fps <= 0:
        raise ValueError("video repair requires a positive frame count and fps")
    if minimum_seconds < 0 or maximum_seconds <= 0:
        raise ValueError("video repair window bounds must be positive")
    if minimum_seconds > maximum_seconds:
        raise ValueError("minimum repair window cannot exceed its maximum")

    frame_count = int(frame_count)
    fps = float(fps)
    requested_context = max(1, int(round(fps * float(overlap_seconds))))
    legal_lengths = tuple(range(5, 363, 17))
    context_frames = min(
        legal_lengths,
        key=lambda value: (abs(value - requested_context), value),
    )
    minimum_frames = int(math.ceil(fps * float(minimum_seconds)))
    maximum_frames = int(math.floor(fps * float(maximum_seconds)))
    target_frames = fps * (float(minimum_seconds) + float(maximum_seconds)) * 0.5
    # H3 accepts only 17k+5 timelines. Pad short and awkward tails to the
    # first legal shape inside the requested band instead of launching tiny
    # kernels with poor throughput and another first-seen compilation shape.
    native_minimum_padded = h3_padded_length(max(1, minimum_frames))

    # Only a narrow band of counts can possibly place every physical window
    # near the requested duration range. Include adjacent counts so awkward
    # short clips choose the least wasteful side of the boundary.
    minimum_count = max(
        1,
        int(math.ceil(
            max(1, frame_count - context_frames)
            / max(1, maximum_frames - context_frames)
        )),
    )
    maximum_count = max(
        minimum_count + 2,
        int(math.ceil(
            max(1, frame_count - context_frames)
            / max(1, minimum_frames - context_frames)
        )) + 2,
    )

    candidates: list[tuple[tuple[float, ...], list[int], list[int]]] = []
    for count in range(1, maximum_count + 1):
        physical_total = frame_count + context_frames * (count - 1)
        base, remainder = divmod(physical_total, count)
        physical = [base + (1 if index < remainder else 0) for index in range(count)]
        new_frames = [
            value if index == 0 else value - context_frames
            for index, value in enumerate(physical)
        ]
        if any(value <= 0 for value in new_frames) or sum(new_frames) != frame_count:
            continue
        try:
            padded = [
                max(h3_padded_length(value), native_minimum_padded)
                for value in physical
            ]
        except ValueError:
            continue
        duration_violation = float(sum(
            max(0, minimum_frames - value) + max(0, value - maximum_frames)
            for value in padded
        ))
        score = (
            duration_violation,
            float(len(set(padded))),
            float(sum(padded)),
            float(sum(abs(value - target_frames) for value in padded)),
            float(count),
        )
        candidates.append((score, new_frames, padded))
    if not candidates:
        raise ValueError("could not construct legal H3 repair windows")
    _score, selected_new_frames, selected_padded = min(
        candidates, key=lambda item: item[0]
    )

    planned: list[dict[str, Any]] = []
    cursor = 0
    for index, new_count in enumerate(selected_new_frames):
        actual_context = 0 if index == 0 else min(context_frames, cursor)
        new_end = min(frame_count, cursor + new_count)
        start = cursor - actual_context
        source = list(range(start, cursor)) + list(range(cursor, new_end))
        planned.append({
            "index": index,
            "start": start,
            "end": new_end,
            "new_start": cursor,
            "new_end": new_end,
            "source": source,
            "context_frames": actual_context,
            "padded_frames": selected_padded[index],
            "weights": [0.0] * actual_context + [1.0] * (new_end - cursor),
        })
        cursor = new_end

    return tuple(
        RepairWindow(
            index=int(item["index"]),
            start=int(item["start"]),
            end=int(item["end"]),
            padded_frames=max(int(item["padded_frames"]), native_minimum_padded),
            weights=tuple(float(value) for value in item["weights"]),
            source=tuple(int(value) for value in item["source"]),
            context_frames=int(item["context_frames"]),
            new_start=int(item["new_start"]),
            new_end=int(item["new_end"]),
        )
        for item in planned
    )


def probe_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open repair source video: {path}")
    result = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 24.0),
    }
    capture.release()
    if min(result["width"], result["height"], result["frames"]) <= 0:
        raise ValueError("repair source video has invalid metadata")
    return result


def _encode_frames(frames: list[Any], fps: float, output: Path) -> None:
    """Encode an internal pixel handoff once, losslessly, as FFV1."""
    if not frames:
        raise ValueError("cannot encode an empty repair window")
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{float(fps):.8f}", "-i", "-", "-an", "-c:v", "ffv1",
            "-level", "3", "-pix_fmt", "bgr0", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame[..., :3].tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"could not create lossless repair handoff: {output}")


def _read_window_frames(source: Path, window: RepairWindow) -> tuple[list[Any], float]:
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, window.start)
    frames = []
    for _ in range(window.valid_frames):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != window.valid_frames:
        raise RuntimeError(
            f"repair source ended early: read {len(frames)}/{window.valid_frames} frames"
        )
    frames.extend([frames[-1].copy() for _ in range(window.padded_frames - len(frames))])
    return frames, fps


def _iou(a: RepairRegion, b: RepairRegion) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.size, a.y + a.size
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.size, b.y + b.size
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    union = a.size * a.size + b.size * b.size - intersection
    return intersection / max(union, 1)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(aw * ah + bw * bh - intersection, 1.0)


def _tracked_position(region: RepairRegion, frame_index: int) -> tuple[int, int]:
    if not region.positions:
        return region.x, region.y
    return tuple(region.positions[min(frame_index, len(region.positions) - 1)])


def _face_detections(frame: Any) -> list[tuple[int, int, int, int, float]]:
    """Run lightweight YuNet detection, with Haar as an install fallback."""

    import cv2

    height, width = frame.shape[:2]
    scale = min(2.0, max(1.0, 720.0 / max(min(height, width), 1)))
    prepared = frame
    if scale > 1.01:
        prepared = cv2.resize(
            prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    bundled_model = (
        Path(__file__).resolve().parent
        / "assets" / "face_detection_yunet_2023mar.onnx"
    )
    external_model = (
        Path(__file__).resolve().parents[1]
        / "models" / "detection" / "face_detection_yunet_2023mar.onnx"
    )
    model = bundled_model if bundled_model.is_file() else external_model
    raw: list[tuple[float, float, float, float, float]] = []
    if model.is_file() and hasattr(cv2, "FaceDetectorYN_create"):
        detector = cv2.FaceDetectorYN_create(
            str(model), "", (prepared.shape[1], prepared.shape[0]),
            0.55, 0.30, 5000,
        )
        _, faces = detector.detect(prepared)
        if faces is not None:
            raw = [
                (*tuple(float(value) for value in row[:4]), float(row[-1]))
                for row in faces
            ]
    else:
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        minimum = max(12, int(round(min(height, width) * 0.018 * scale)))
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        raw = [
            (*tuple(float(value) for value in row), 0.65)
            for row in cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(minimum, minimum),
            )
        ]
    detections: list[tuple[int, int, int, int, float]] = []
    for x, y, face_w, face_h, confidence in raw:
        item = (
            int(round(x / scale)),
            int(round(y / scale)),
            max(8, int(round(face_w / scale))),
            max(8, int(round(face_h / scale))),
            float(confidence),
        )
        if any(
            _box_iou(item[:4], existing[:4]) > 0.30
            for existing in detections
        ):
            continue
        detections.append(item)
    return detections


def _smooth(values: Any) -> Any:
    import numpy as np

    if len(values) < 3:
        return values
    padded = np.pad(values, (2, 2), mode="edge")
    return np.convolve(padded, np.asarray([1, 2, 3, 2, 1]) / 9.0, mode="valid")


def _aligned_canvas(value: float) -> int:
    return int(math.ceil(float(value) / 32.0) * 32)


def _face_atlas_layout(
    regions: tuple[RepairRegion, ...] | list[RepairRegion],
    magnification: float,
    *,
    minimum_canvas: int = 192,
    maximum_canvas: int = 1088,
) -> tuple[int, int]:
    """Return the smallest safe square canvas for one face batch."""

    if not regions:
        raise ValueError("face Atlas layout requires at least one region")
    grid = int(math.ceil(math.sqrt(len(regions))))
    required_cell = max(region.size for region in regions) * magnification
    canvas = max(minimum_canvas, _aligned_canvas(grid * required_cell))
    return grid, min(canvas, maximum_canvas)


def plan_face_atlas_batches(
    regions: tuple[RepairRegion, ...],
    magnification: float,
    *,
    minimum_canvas: int = 192,
    maximum_canvas: int = 1088,
) -> tuple[tuple[tuple[RepairRegion, ...], int, int], ...]:
    """Pack faces into the fewest bounded canvases that preserve magnification."""

    pending = sorted(regions, key=lambda item: item.size, reverse=True)
    batches: list[tuple[tuple[RepairRegion, ...], int, int]] = []
    current: list[RepairRegion] = []
    for region in pending:
        candidate = current + [region]
        grid = int(math.ceil(math.sqrt(len(candidate))))
        required = max(item.size for item in candidate) * magnification * grid
        if current and (
            len(candidate) > 9 or _aligned_canvas(required) > maximum_canvas
        ):
            current_grid, current_canvas = _face_atlas_layout(
                current,
                magnification,
                minimum_canvas=minimum_canvas,
                maximum_canvas=maximum_canvas,
            )
            batches.append((tuple(current), current_canvas, current_grid))
            current = [region]
        else:
            current = candidate
    if current:
        grid, canvas = _face_atlas_layout(
            current,
            magnification,
            minimum_canvas=minimum_canvas,
            maximum_canvas=maximum_canvas,
        )
        batches.append((tuple(current), canvas, grid))
    return tuple(batches)


def detect_regions(
    frames: list[Any],
    *,
    mode: str,
    crop_size: int,
    maximum_regions: int,
) -> tuple[RepairRegion, ...]:
    """Use the published global tracker and importance policy on every frame."""

    import cv2
    import numpy as np
    from .face_refine_scheduler import build_tracks, interpolate_track, score_tracks

    if not frames:
        raise ValueError("face repair requires decoded source frames")
    if mode not in {"face", "smart"}:
        raise ValueError("video repair supports face targets only")
    height, width = frames[0].shape[:2]
    all_boxes: list[list[tuple[float, float, float, float]]] = []
    all_confs: list[list[float]] = []
    for frame in frames:
        boxes, confs = [], []
        for detection in _face_detections(frame):
            x, y, face_w, face_h = (float(value) for value in detection[:4])
            boxes.append((x, y, x + face_w, y + face_h))
            confs.append(float(detection[4]) if len(detection) > 4 else 1.0)
        all_boxes.append(boxes)
        all_confs.append(confs)

    tracks = build_tracks(
        all_boxes,
        all_confs,
        ((0, len(frames)),),
        width,
        height,
        # YuNet may miss a moving or briefly occluded face for several frames.
        # A half-second bridge keeps one person in one global identity track,
        # while the geometric assignment cost still rejects unrelated boxes.
        max_gap=12,
        max_cost=1.45,
    )
    sharpness: dict[int, float] = {}
    for track in tracks:
        values = []
        for index, box in enumerate(track.boxes):
            if box is None:
                continue
            x0, y0, x1, y1 = box
            side = max(8.0, max(x1 - x0, y1 - y0) * 1.25)
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            x = int(max(0, min(width - 1, round(cx - side * 0.5))))
            y = int(max(0, min(height - 1, round(cy - side * 0.5))))
            x2 = int(max(x + 1, min(width, round(x + side))))
            y2 = int(max(y + 1, min(height, round(y + side))))
            patch = frames[index][y:y2, x:x2]
            if not patch.size:
                continue
            gray = cv2.cvtColor(cv2.resize(patch, (96, 96)), cv2.COLOR_BGR2GRAY)
            values.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        raw = float(np.median(values)) if values else 0.0
        sharpness[track.track_id] = float(np.clip(np.log1p(raw / 24.0) / 5.0, 0.0, 1.0))

    score_tracks(
        tracks,
        width,
        height,
        max_faces=int(maximum_regions),
        min_face_height_px=max(8.0, min(width, height) * 0.012),
        min_visibility_ratio=0.10,
        clear_face_height_px=max(48.0, min(width, height) * 0.14),
        skip_clear_faces=True,
        max_repair_face_height_px=float(crop_size) / 2.5,
        fps=24.0,
        sharpness=sharpness,
    )
    selected = sorted(
        (track for track in tracks if track.selected),
        key=lambda track: (-track.score, track.track_id),
    )
    if not selected:
        raise ValueError("no eligible face was detected")

    result: list[RepairRegion] = []
    for track in selected:
        states, detected, weights = interpolate_track(track, 17, 31)
        crop_boxes = []
        face_rects = []
        positions = []
        face_heights = []
        for cx, cy, face_w, face_h in states:
            side = min(max(8.0, float(face_h) * 2.5), float(width), float(height))
            x = min(max(float(cx) - side * 0.5, 0.0), max(0.0, width - side))
            y = min(max(float(cy) - side * 0.5, 0.0), max(0.0, height - side))
            crop_boxes.append((x, y, side, side))
            face_rects.append((
                (float(cx) - float(face_w) * 0.5 - x) / side,
                (float(cy) - float(face_h) * 0.5 - y) / side,
                float(face_w) / side,
                float(face_h) / side,
            ))
            positions.append((int(round(x)), int(round(y))))
            face_heights.append(float(face_h))
        representative_side = int(round(float(np.percentile(
            [box[2] for box in crop_boxes], 90
        ))))
        metrics = track.metrics
        result.append(RepairRegion(
            x=positions[0][0],
            y=positions[0][1],
            size=max(8, representative_side),
            score=float(track.score),
            face_score=1.0,
            track_id=int(track.track_id),
            quality_need=float(metrics.get("repair_need", 0.0)),
            importance=float(metrics.get("centre_score", 0.0)),
            visibility=float(metrics.get("visibility_ratio", 0.0)),
            face_size=float(metrics.get("median_face_height_px", 0.0)),
            positions=tuple(positions),
            crop_boxes=tuple(crop_boxes),
            face_rects=tuple(face_rects),
            detected=tuple(bool(value) for value in detected.tolist()),
            weights=tuple(float(value) for value in weights.tolist()),
            face_heights=tuple(face_heights),
        ))
    return tuple(result)
def _read_all_frames(source: Path) -> tuple[list[Any], float]:
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("video repair source contains no decodable frames")
    return frames, fps


def _slice_region(region: RepairRegion, indices: list[int]) -> RepairRegion:
    padded = indices
    return RepairRegion(
        x=region.positions[padded[0]][0],
        y=region.positions[padded[0]][1],
        size=max(8, int(round(max(region.crop_boxes[index][2] for index in padded)))),
        score=region.score,
        face_score=region.face_score,
        track_id=region.track_id,
        quality_need=region.quality_need,
        importance=region.importance,
        visibility=region.visibility,
        face_size=region.face_size,
        positions=tuple(region.positions[index] for index in padded),
        crop_boxes=tuple(region.crop_boxes[index] for index in padded),
        face_rects=tuple(region.face_rects[index] for index in padded),
        detected=tuple(region.detected[index] for index in padded),
        weights=tuple(region.weights[index] for index in padded),
        face_heights=tuple(region.face_heights[index] for index in padded),
    )


def prepare_repair_windows(
    source: Path,
    work_dir: Path,
    repair: Any,
    *,
    progress: Callable[[float, str, str], None] | None = None,
) -> tuple[dict[str, Any], tuple[PreparedRepairWindow, ...]]:
    """Detect identities once, then emit stable progressive Atlas chains."""

    meta = probe_video(source)
    work_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(5, "video_repair_detect", "逐帧检测并排序全片人脸")
    all_frames, fps = _read_all_frames(source)
    meta["frames"] = len(all_frames)
    regions = detect_regions(
        all_frames,
        mode=repair.mode,
        crop_size=repair.source_crop_size,
        maximum_regions=repair.maximum_regions,
    )
    meta["selected_face_count"] = len(regions)
    meta["face_repair_capacity"] = int(repair.maximum_regions)
    meta["face_repair_cell_size"] = int(repair.cell_size)
    windows = plan_repair_windows(
        len(all_frames), fps,
        minimum_seconds=repair.minimum_window_seconds,
        maximum_seconds=repair.window_seconds,
        overlap_seconds=repair.overlap_seconds,
    )
    # One configured square Atlas owns one configured square cell grid. The
    # detector has already capped the result at this capacity and rejected
    # faces that cannot gain at least 1.5x inside a cell.
    batches = ((regions, int(repair.canvas_size), int(repair.grid_size)),)
    prepared: list[PreparedRepairWindow] = []
    for window_index, window in enumerate(windows):
        if progress:
            progress(
                8 + 15 * window_index / max(len(windows), 1),
                "video_repair_pack",
                f"构建渐进修复窗口 {window_index + 1}/{len(windows)}",
            )
        source_indices = list(window.source or range(window.start, window.end))
        valid_count = len(source_indices)
        padded_indices = source_indices + [source_indices[-1]] * (
            window.padded_frames - valid_count
        )
        window_frames = [all_frames[index] for index in padded_indices]
        source_path = work_dir / f"window-{window_index:03d}-source.mkv"
        _encode_frames(window_frames, fps, source_path)
        for batch_index, (batch_regions, canvas_size, grid_size) in enumerate(batches):
            if not any(
                region.detected[index]
                for region in batch_regions
                for index in source_indices
            ):
                continue
            local_regions = tuple(
                _slice_region(region, padded_indices) for region in batch_regions
            )
            stem = f"window-{window_index:03d}-batch-{batch_index:02d}"
            atlas_path = work_dir / f"{stem}-atlas.mkv"
            manifest_path = work_dir / f"{stem}.json"
            refined_path = work_dir / f"{stem}-atlas-refined.mp4"
            repaired_path = work_dir / f"{stem}-repaired.mkv"
            latent_path = work_dir / f"{stem}-refined-latents.pt"
            build_atlas(
                window_frames,
                fps,
                atlas_path,
                manifest_path,
                regions=local_regions,
                canvas_size=canvas_size,
                grid_size=grid_size,
                valid_frames=valid_count,
                window_weights=window.weights,
            )
            prepared.append(PreparedRepairWindow(
                window=window,
                batch_index=batch_index,
                canvas_size=canvas_size,
                grid_size=grid_size,
                source_path=source_path,
                atlas_path=atlas_path,
                manifest_path=manifest_path,
                refined_atlas_path=refined_path,
                repaired_path=repaired_path,
                chain_id=f"shot-0-group-{batch_index}",
                refined_latents_path=latent_path,
            ))
    del all_frames
    if not prepared:
        raise ValueError("selected faces are not visible in any repair window")
    return meta, tuple(prepared)


def build_atlas(
    frames: list[Any],
    fps: float,
    output: Path,
    manifest_path: Path,
    *,
    regions: tuple[RepairRegion, ...],
    canvas_size: int,
    grid_size: int,
    valid_frames: int,
    window_weights: tuple[float, ...] | None = None,
) -> None:
    """Pack per-frame float crops exactly once into a lossless Atlas video."""

    import cv2
    import numpy as np

    cell = (canvas_size // grid_size) // 8 * 8
    margin = (canvas_size - cell * grid_size) // 2
    atlas_frames = []
    for frame_index, frame in enumerate(frames):
        atlas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        for cell_index, region in enumerate(regions):
            x, y, side, _ = region.crop_boxes[frame_index]
            scale = cell / max(float(side), 1.0e-6)
            matrix = np.asarray(
                [[scale, 0.0, -float(x) * scale],
                 [0.0, scale, -float(y) * scale]],
                dtype=np.float32,
            )
            tile = cv2.warpAffine(
                frame,
                matrix,
                (cell, cell),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            row, column = divmod(cell_index, grid_size)
            ox, oy = margin + column * cell, margin + row * cell
            atlas[oy:oy + cell, ox:ox + cell] = tile
        for boundary in range(1, grid_size):
            point = margin + boundary * cell
            atlas[:, max(0, point - 2):min(canvas_size, point + 2)] = 5
            atlas[max(0, point - 2):min(canvas_size, point + 2), :] = 5
        atlas_frames.append(atlas)
    _encode_frames(atlas_frames, fps, output)
    records = []
    for index, region in enumerate(regions):
        row, column = divmod(index, grid_size)
        origin = (margin + column * cell, margin + row * cell)
        face_rects = [
            (fx * cell, fy * cell, fw * cell, fh * cell)
            for fx, fy, fw, fh in region.face_rects
        ]
        weights = []
        for frame_index in range(len(frames)):
            temporal = (
                float(window_weights[frame_index])
                if window_weights is not None and frame_index < len(window_weights)
                else 0.0 if frame_index >= valid_frames else 1.0
            )
            weights.append(float(region.weights[frame_index]) * temporal)
        records.append({
            **asdict(region),
            "origin": origin,
            "face_rects": face_rects,
            "weights": weights,
        })
    manifest_path.write_text(json.dumps({
        "schema_version": 2,
        "canvas_size": canvas_size,
        "grid_size": grid_size,
        "cell_size": cell,
        "margin": margin,
        "valid_frames": valid_frames,
        "regions": records,
        "crop_factor": 2.5,
        "pixel_handoff": "ffv1_lossless",
    }, indent=2), encoding="utf-8")


def atlas_denoise_regions(
    prepared: PreparedRepairWindow,
) -> tuple[tuple[float, float, float, float, tuple[float, ...]], ...]:
    """Create the published per-cell, per-frame small-face denoise curves."""

    import numpy as np

    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    canvas = float(manifest["canvas_size"])
    cell = float(manifest["cell_size"])
    curves = []
    for record in manifest["regions"]:
        heights = np.asarray(record["face_heights"], dtype=np.float64)
        position = np.clip((heights - 30.0) / 90.0, 0.0, 1.0)
        strength = 1.0 + (0.35 - 1.0) * position
        visibility = np.asarray(record["detected"], dtype=np.float64)
        if len(strength) > 1:
            radius = min(4, (len(strength) - 1) // 2)
            if radius:
                x = np.arange(-radius, radius + 1, dtype=np.float64)
                kernel = np.exp(-(x * x) / (2.0 * max(radius / 2.0, 0.5) ** 2))
                kernel /= kernel.sum()
                strength = np.convolve(np.pad(strength, (radius, radius), mode="edge"), kernel, mode="valid")
                visibility = np.convolve(np.pad(visibility, (radius, radius), mode="edge"), kernel, mode="valid")
        strength = np.clip(strength * visibility, 0.0, 1.0)
        ox, oy = record["origin"]
        curves.append((
            float(ox) / canvas,
            float(oy) / canvas,
            cell / canvas,
            cell / canvas,
            tuple(float(value) for value in strength),
        ))
    return tuple(curves)
def _correlation(a: Any, b: Any) -> float:
    import numpy as np

    left = a.reshape(-1).astype(np.float64); right = b.reshape(-1).astype(np.float64)
    left -= left.mean(); right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-10 else 0.0


def _detail(gray: Any) -> Any:
    import cv2

    fine = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    mid = cv2.GaussianBlur(gray, (0, 0), 1.2) - cv2.GaussianBlur(
        gray, (0, 0), 5.0
    )
    return fine + 0.55 * mid


def _flow_warp(previous: Any, current: Any, value: Any) -> Any:
    import cv2
    import numpy as np

    flow = cv2.calcOpticalFlowFarneback(
        current, previous, None, 0.5, 3, 17, 3, 5, 1.2, 0
    )
    height, width = current.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        value,
        x + flow[..., 0],
        y + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _grid_periodicity(residual: Any) -> float:
    """Measure new phase-locked 4/8/16-pixel energy in one residual."""

    import numpy as np

    gx = np.abs(np.diff(residual, axis=1))
    gy = np.abs(np.diff(residual, axis=0))
    baseline = float((gx.mean() + gy.mean()) * 0.5) + 1.0e-8
    strongest = 1.0
    for period in (4, 8, 16):
        if gx.shape[1] >= period * 2:
            phase = np.asarray(
                [gx[:, offset::period].mean() for offset in range(period)]
            )
            strongest = max(strongest, float(phase.max() / baseline))
        if gy.shape[0] >= period * 2:
            phase = np.asarray(
                [gy[offset::period, :].mean() for offset in range(period)]
            )
            strongest = max(strongest, float(phase.max() / baseline))
    return strongest


def _colour_match(refined: Any, source: Any, mask: Any) -> Any:
    import numpy as np

    weights = np.clip(mask.astype(np.float32), 0.0, 1.0)
    denom = float(weights.sum()) + 1.0e-6
    result = refined.astype(np.float32).copy()
    source_f = source.astype(np.float32)
    for channel in range(3):
        source_values = source_f[..., channel]
        refined_values = result[..., channel]
        source_mean = float((source_values * weights).sum() / denom)
        refined_mean = float((refined_values * weights).sum() / denom)
        source_std = math.sqrt(float((((source_values - source_mean) ** 2) * weights).sum() / denom) + 1.0e-6)
        refined_std = math.sqrt(float((((refined_values - refined_mean) ** 2) * weights).sum() / denom) + 1.0e-6)
        result[..., channel] = (
            (refined_values - refined_mean) * min(3.0, source_std / refined_std)
            + source_mean
        )
    return result


def merge_refined_atlas(prepared: PreparedRepairWindow) -> dict[str, Any]:
    """Apply the published full face-only stitch, not a weakened residual."""

    import cv2
    import numpy as np

    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    records = manifest["regions"]
    cell = int(manifest["cell_size"])
    source_cap = cv2.VideoCapture(str(prepared.source_path))
    atlas_cap = cv2.VideoCapture(str(prepared.atlas_path))
    refined_cap = cv2.VideoCapture(str(prepared.refined_atlas_path))
    fps = float(source_cap.get(cv2.CAP_PROP_FPS) or 24.0)
    rendered = []
    for frame_index in range(prepared.window.valid_frames):
        ok_source, source_frame = source_cap.read()
        ok_atlas, atlas_frame = atlas_cap.read()
        ok_refined, refined_frame = refined_cap.read()
        if not ok_source or not ok_atlas or not ok_refined:
            raise RuntimeError("refined Atlas does not cover its progressive window")
        if refined_frame.shape[:2] != atlas_frame.shape[:2]:
            refined_frame = cv2.resize(
                refined_frame,
                (atlas_frame.shape[1], atlas_frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        height, width = source_frame.shape[:2]
        base = source_frame.astype(np.float32)
        delta_sum = np.zeros_like(base)
        mask_sum = np.zeros((height, width, 1), dtype=np.float32)
        for record in records:
            temporal_weight = float(record["weights"][frame_index])
            if temporal_weight <= 1.0e-5:
                continue
            ox, oy = (int(value) for value in record["origin"])
            original_tile = atlas_frame[oy:oy + cell, ox:ox + cell]
            refined_tile = refined_frame[oy:oy + cell, ox:ox + cell]
            fx, fy, fw, fh = (
                float(value) for value in record["face_rects"][frame_index]
            )
            mask = np.zeros((cell, cell), dtype=np.float32)
            x0 = max(0, min(cell - 1, int(math.floor(fx))))
            y0 = max(0, min(cell - 1, int(math.floor(fy))))
            x1 = max(x0 + 1, min(cell, int(math.ceil(fx + fw))))
            y1 = max(y0 + 1, min(cell, int(math.ceil(fy + fh))))
            mask[y0:y1, x0:x1] = 1.0
            mask = cv2.dilate(mask, np.ones((25, 25), dtype=np.uint8))
            crop_x, crop_y, crop_side, _ = (
                float(value) for value in record["crop_boxes"][frame_index]
            )
            feather_canvas = max(1.0, min(cell / 3.0, 6.0 * cell / max(crop_side, 1.0)))
            mask = cv2.GaussianBlur(mask, (0, 0), feather_canvas).clip(0.0, 1.0)
            matched = _colour_match(refined_tile, original_tile, mask)
            scale = crop_side / float(cell)
            matrix = np.asarray(
                [[scale, 0.0, crop_x], [0.0, scale, crop_y]],
                dtype=np.float32,
            )
            warped = cv2.warpAffine(
                matched,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
            warped_mask = cv2.warpAffine(
                mask,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )[..., None]
            effective = warped_mask * temporal_weight
            delta_sum += (warped - base) * effective
            mask_sum += effective
        output = base + delta_sum / np.maximum(mask_sum, 1.0)
        rendered.append(np.clip(output, 0, 255).astype(np.uint8))
    source_cap.release()
    atlas_cap.release()
    refined_cap.release()
    _encode_frames(rendered, fps, prepared.repaired_path)
    public_regions = []
    gates = []
    for record in records:
        crop_sides = [float(box[2]) for box in record["crop_boxes"]]
        public_regions.append({
            "track_id": int(record["track_id"]),
            "score": float(record["score"]),
            "face_size": float(record["face_size"]),
            "tracked_frames": len(record["crop_boxes"]),
            "actual_magnification_median": round(
                cell / max(float(sorted(crop_sides)[len(crop_sides) // 2]), 1.0), 3
            ),
        })
        gates.append({
            "accepted": True,
            "blend_weight": 1.0,
            "reason": "published_face_only_stitch",
        })
    return {
        "regions": public_regions,
        "quality_gate": gates,
        "merge": "full_refined_face_colour_matched_dilate12_feather6",
    }


def assemble_repaired_video(
    source: Path,
    output: Path,
    prepared: tuple[PreparedRepairWindow, ...],
    meta: dict[str, Any],
) -> None:
    """Add disjoint face/window contributions and encode the public MP4 once."""

    import cv2
    import numpy as np

    captures = [cv2.VideoCapture(str(item.repaired_path)) for item in prepared]
    source_capture = cv2.VideoCapture(str(source))
    width, height = int(meta["width"]), int(meta["height"])
    fps = float(meta["fps"])
    silent = output.with_suffix(".video-only.mp4")
    process = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.8f}", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(silent),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame_index in range(int(meta["frames"])):
            ok_source, source_frame = source_capture.read()
            if not ok_source:
                raise RuntimeError("repair source ended before its metadata")
            accumulated_delta = np.zeros_like(source_frame, dtype=np.float32)
            for item, capture in zip(prepared, captures):
                if not item.window.start <= frame_index < item.window.end:
                    continue
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("repaired window ended before its manifest")
                # The progressive window and track weights were already applied
                # by face-only stitch. Context prefixes therefore contribute 0.
                accumulated_delta += frame.astype(np.float32) - source_frame.astype(np.float32)
            final = np.clip(
                source_frame.astype(np.float32) + accumulated_delta, 0, 255
            ).astype(np.uint8)
            process.stdin.write(final.tobytes())
    finally:
        process.stdin.close()
        source_capture.release()
        for capture in captures:
            capture.release()
    if process.wait() != 0:
        raise RuntimeError("could not encode repaired video")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(silent), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy",
            "-c:a", "copy", "-shortest", str(output),
        ],
        check=True,
    )
    silent.unlink(missing_ok=True)
def cleanup_repair_workdir(work_dir: Path) -> None:
    if os.environ.get("H3_VIDEO_REPAIR_KEEP_WORKDIR", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return
    shutil.rmtree(work_dir, ignore_errors=True)
