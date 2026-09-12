"""Training-free bounded-window co-denoising primitives for MiniMax H3.

The important distinction from clip continuation is ownership of the solver
state.  A :class:`GlobalAVPlan` describes overlapping *views* of one global
audio/video latent; it does not describe independently completed clips.  At a
given diffusion timestep every view predicts the same global state, weighted
window sampling fuses those predictions, and the scheduler advances once.

All geometry is prompt agnostic.  Window starts lie on a 51-pixel-frame grid:
51 is the least common temporal period of H3's 17-frame video stride and the
24-fps -> 40-Hz audio clock.  Consequently both modalities share exact
absolute-time boundaries without per-window rounding drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


FPS = 24
AUDIO_LATENT_HZ = 40
H3_FRAME_ORIGIN = 5
H3_FRAME_STRIDE = 17
AV_EXACT_START_STRIDE = 51
DEFAULT_WINDOW_FRAMES = 277
DEFAULT_WINDOW_STRIDE_FRAMES = 204
# Three seconds is the smallest creator-facing setting. 73 frames is the
# nearest legal H3 ``5 + 17*k`` window and still permits one exact 51-frame AV
# stride with a positive overlap.
MIN_WINDOW_FRAMES = 73
MINIMUM_TERMINAL_FRAMES = 39
DEFAULT_MINIMUM_WINDOW_FRAMES = 124
MAX_WINDOW_FRAMES = 362


def window_geometry_for_seconds(
    seconds: float,
    overlap_seconds: float | None = None,
) -> tuple[int, int]:
    """Snap one creator-facing duration to an exact H3 audio/video view.

    Window lengths live on H3's ``5 + 17*k`` video grid. Starts live on the
    stricter 51-frame joint AV grid.  The returned stride is the largest exact
    start step that retains at least one 17-frame video overlap.
    """

    requested = float(seconds)
    if not math.isfinite(requested) or not 3.0 <= requested <= 15.0:
        raise ValueError("co-denoise window seconds must lie inside [3, 15]")
    target = requested * FPS
    candidates = tuple(
        frame
        for frame in range(MIN_WINDOW_FRAMES, MAX_WINDOW_FRAMES + 1)
        if (frame - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE == 0
    )
    window = min(candidates, key=lambda frame: (abs(frame - target), frame))
    if overlap_seconds is None:
        stride = max(
            AV_EXACT_START_STRIDE,
            ((window - H3_FRAME_STRIDE) // AV_EXACT_START_STRIDE)
            * AV_EXACT_START_STRIDE,
        )
        if stride >= window:
            stride = AV_EXACT_START_STRIDE
    else:
        requested_overlap = float(overlap_seconds)
        if not math.isfinite(requested_overlap) or not (
            0.0 <= requested_overlap <= 4.0
        ):
            raise ValueError("co-denoise overlap seconds must lie inside [0, 4]")
        # H3 view starts share the exact 51-frame audio/video clock.  Keep the
        # user's control continuous at the API while snapping execution to the
        # nearest legal positive overlap.  Ties prefer less repeated work.
        strides = tuple(
            value
            for value in range(AV_EXACT_START_STRIDE, window, AV_EXACT_START_STRIDE)
        )
        target_overlap = requested_overlap * FPS
        stride = min(
            strides,
            key=lambda value: (
                abs((window - value) - target_overlap),
                window - value,
            ),
        )
    return int(window), int(stride)


def _video_tokens(frame_count: int) -> int:
    frames = int(frame_count)
    if frames < H3_FRAME_ORIGIN or (frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
        raise ValueError("window frames must satisfy H3's 5 + 17*k grid")
    return 2 + 5 * ((frames - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE)


def _audio_tick(frame_offset: int) -> int:
    return int(round(int(frame_offset) / FPS * AUDIO_LATENT_HZ))


@dataclass(frozen=True, slots=True)
class GlobalAVWindow:
    index: int
    start_frame: int
    frames: int
    video_start: int
    video_stop: int
    audio_start: int
    audio_stop: int
    # A view may read a wider temporal halo while only contributing the
    # portion owned by one creator prompt. This prevents a neighbouring
    # prompt from rewriting frames across an authored window boundary.
    prompt_index: int = 0
    video_write_start: int | None = None
    video_write_stop: int | None = None
    audio_write_start: int | None = None
    audio_write_stop: int | None = None

    @property
    def stop_frame(self) -> int:
        return self.start_frame + self.frames

    @property
    def video_tokens(self) -> int:
        return self.video_stop - self.video_start

    @property
    def audio_tokens(self) -> int:
        return self.audio_stop - self.audio_start

    @property
    def writable_video_start(self) -> int:
        return (
            self.video_start
            if self.video_write_start is None
            else self.video_write_start
        )

    @property
    def writable_video_stop(self) -> int:
        return (
            self.video_stop
            if self.video_write_stop is None
            else self.video_write_stop
        )

    @property
    def writable_audio_start(self) -> int:
        return (
            self.audio_start
            if self.audio_write_start is None
            else self.audio_write_start
        )

    @property
    def writable_audio_stop(self) -> int:
        return (
            self.audio_stop
            if self.audio_write_stop is None
            else self.audio_write_stop
        )


@dataclass(frozen=True, slots=True)
class GlobalAVPlan:
    output_frames: int
    windows: tuple[GlobalAVWindow, ...]
    window_frames: int
    stride_frames: int
    fusion: str = "overlap_partition_of_unity_v2"
    mechanism: str = "global_joint_av_co_denoise_v1"

    @property
    def video_tokens(self) -> int:
        return _video_tokens(self.output_frames)

    @property
    def audio_tokens(self) -> int:
        return _audio_tick(self.output_frames)

    def telemetry(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "fusion": self.fusion,
            "output_frames": self.output_frames,
            "window_frames": self.window_frames,
            "stride_frames": self.stride_frames,
            "window_count": len(self.windows),
            "global_scheduler_updates": True,
            "independent_completed_clips": False,
            "audio_clock": "absolute_24fps_to_40hz_v1",
            "windows": [
                {
                    "index": item.index,
                    "start_frame": item.start_frame,
                    "stop_frame": item.stop_frame,
                    "frames": item.frames,
                    "video_start": item.video_start,
                    "video_stop": item.video_stop,
                    "audio_start": item.audio_start,
                    "audio_stop": item.audio_stop,
                    "prompt_index": item.prompt_index,
                    "video_write_start": item.writable_video_start,
                    "video_write_stop": item.writable_video_stop,
                    "audio_write_start": item.writable_audio_start,
                    "audio_write_stop": item.writable_audio_stop,
                }
                for item in self.windows
            ],
        }


def plan_global_av_windows(
    output_frames: int,
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_WINDOW_STRIDE_FRAMES,
    minimum_window_frames: int | None = None,
) -> GlobalAVPlan:
    """Cover one H3 timeline with prompt-independent overlapping AV views.

    ``stride_frames`` must be divisible by 51.  This is a stronger condition
    than H3's video-only grid, but it makes every window start an integer audio
    latent tick as well.  The final view grows or moves one 51-frame unit
    earlier when necessary; no tiny tail inference is emitted.
    """

    output = int(output_frames)
    window = int(window_frames)
    stride = int(stride_frames)
    uses_default_geometry = (
        window == DEFAULT_WINDOW_FRAMES
        and stride == DEFAULT_WINDOW_STRIDE_FRAMES
    )
    minimum = int(
        DEFAULT_MINIMUM_WINDOW_FRAMES
        if minimum_window_frames is None and uses_default_geometry
        else MINIMUM_TERMINAL_FRAMES
        if minimum_window_frames is None
        else minimum_window_frames
    )
    terminal_limit = MAX_WINDOW_FRAMES if uses_default_geometry else window
    _video_tokens(output)
    _video_tokens(window)
    if not MIN_WINDOW_FRAMES <= window <= MAX_WINDOW_FRAMES:
        raise ValueError("co-denoise window must stay inside H3's native range")
    if stride <= 0 or stride % AV_EXACT_START_STRIDE:
        raise ValueError("co-denoise stride must be a positive multiple of 51")
    if stride >= window:
        raise ValueError("co-denoise windows must overlap")
    if minimum < H3_FRAME_ORIGIN or (minimum - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
        raise ValueError("minimum window must satisfy H3's 5 + 17*k grid")
    if output <= window:
        starts = [0]
    else:
        starts = [0]
        # The terminal view may grow beyond the preferred operating point up
        # to H3's native maximum.  This is cheaper and better connected than
        # emitting a fourth tiny view merely to preserve an exact 277 frames.
        while output - starts[-1] > terminal_limit:
            candidate = starts[-1] + stride
            remainder = output - candidate
            if remainder < minimum:
                retreat_units = math.ceil(
                    (minimum - remainder) / AV_EXACT_START_STRIDE
                )
                candidate -= retreat_units * AV_EXACT_START_STRIDE
            if candidate <= starts[-1]:
                # Some legal H3 timelines cannot be partitioned into the
                # preferred view length on the stricter 51-frame joint AV
                # start grid. Keep the current terminal view in that case;
                # it can exceed the preferred size by less than one stride
                # while remaining below H3's native safety ceiling.
                if output - starts[-1] <= min(MAX_WINDOW_FRAMES, window + stride):
                    break
                raise ValueError("timeline cannot be covered by bounded AV windows")
            starts.append(candidate)

    windows: list[GlobalAVWindow] = []
    for index, start in enumerate(starts):
        frames = min(window, output - start)
        if index == len(starts) - 1:
            frames = output - start
        if frames < minimum and len(starts) > 1:
            raise RuntimeError("global AV planner emitted a tiny tail")
        video_start = 5 * (start // H3_FRAME_STRIDE)
        video_stop = video_start + _video_tokens(frames)
        audio_start = _audio_tick(start)
        audio_stop = _audio_tick(start + frames)
        # The 51-frame start grid makes relative and absolute audio rounding
        # identical.  Assert this invariant rather than repairing it later.
        if audio_stop - audio_start != _audio_tick(frames):
            raise RuntimeError("global AV window missed the exact audio clock")
        windows.append(
            GlobalAVWindow(
                index=index,
                start_frame=start,
                frames=frames,
                video_start=video_start,
                video_stop=video_stop,
                audio_start=audio_start,
                audio_stop=audio_stop,
            )
        )

    if windows[0].start_frame != 0 or windows[-1].stop_frame != output:
        raise RuntimeError("global AV windows do not cover the requested timeline")
    if any(left.stop_frame <= right.start_frame for left, right in zip(windows, windows[1:])):
        raise RuntimeError("global AV windows lost their overlap")
    if windows[-1].video_stop != _video_tokens(output):
        raise RuntimeError("global AV video views missed the output latent clock")
    if windows[-1].audio_stop != _audio_tick(output):
        raise RuntimeError("global AV audio views missed the output latent clock")
    return GlobalAVPlan(
        output_frames=output,
        windows=tuple(windows),
        window_frames=window,
        stride_frames=stride,
    )


def plan_balanced_global_av_windows(
    output_frames: int,
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_WINDOW_STRIDE_FRAMES,
    minimum_window_frames: int = MINIMUM_TERMINAL_FRAMES,
) -> GlobalAVPlan:
    """Balance a completed-audio SelfLift tail across visual views.

    The ordinary planner keeps every start on the stricter 51-frame joint AV
    clock. That is necessary while audio is still being generated, but a
    SelfLift high-resolution tail carries an already completed audio latent.
    Reusing the 51-frame rule there can produce a very short terminal view
    (for example 107/107/39 for a 243-frame clip), which under-fills the GPU
    and increases the sum of quadratic attention work.

    This planner keeps the requested overlap and H3's exact 17-frame video
    lattice, distributes the same covered-frame budget as evenly as possible,
    and admits a layout only when every audio conditioning slice has the exact
    relative tick count. Unsupported geometries fall back to the conservative
    joint AV planner.
    """

    output = int(output_frames)
    window = int(window_frames)
    stride = int(stride_frames)
    minimum = int(minimum_window_frames)
    if not MIN_WINDOW_FRAMES <= window <= MAX_WINDOW_FRAMES:
        raise ValueError("co-denoise window must stay inside H3's native range")
    if stride <= 0 or stride % H3_FRAME_STRIDE:
        raise ValueError("balanced view stride must be a positive multiple of 17")
    if stride >= window:
        raise ValueError("co-denoise windows must overlap")
    _video_tokens(output)
    _video_tokens(window)
    conservative = (
        plan_global_av_windows(
            output,
            window_frames=window,
            stride_frames=stride,
            minimum_window_frames=minimum,
        )
        if stride % AV_EXACT_START_STRIDE == 0
        else None
    )
    if output <= window:
        frames = output
        return GlobalAVPlan(
            output_frames=output,
            windows=(
                GlobalAVWindow(
                    index=0,
                    start_frame=0,
                    frames=frames,
                    video_start=0,
                    video_stop=_video_tokens(frames),
                    audio_start=0,
                    audio_stop=_audio_tick(frames),
                ),
            ),
            window_frames=window,
            stride_frames=stride,
            mechanism="global_joint_av_balanced_views_v2",
        )

    def fallback() -> GlobalAVPlan:
        if conservative is None:
            raise ValueError("completed-audio visual windows could not be balanced")
        return conservative

    overlap = window - stride
    if overlap <= 0 or (overlap - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
        return fallback()
    count = max(2, math.ceil((output - overlap) / stride))
    covered_frames = output + overlap * (count - 1)
    lattice_units, remainder = divmod(
        covered_frames - count * H3_FRAME_ORIGIN,
        H3_FRAME_STRIDE,
    )
    if remainder:
        return fallback()
    minimum_units = (minimum - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE
    maximum_units = (window - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE
    low_units, high_count = divmod(lattice_units, count)
    if low_units < minimum_units or low_units + bool(high_count) > maximum_units:
        return fallback()

    # There are at most two adjacent legal lengths. Rotate the longer views
    # through the sequence until every absolute audio slice has the same tick
    # count as its local view. The public 15-second limit keeps this tiny.
    unit_candidates = [low_units + 1] * high_count + [low_units] * (
        count - high_count
    )
    arrangements: list[list[int]] = []
    for rotation in range(count):
        arrangement = unit_candidates[rotation:] + unit_candidates[:rotation]
        if arrangement not in arrangements:
            arrangements.append(arrangement)

    for arrangement in arrangements:
        starts: list[int] = []
        lengths: list[int] = []
        start = 0
        valid = True
        for units in arrangement:
            frames = H3_FRAME_ORIGIN + H3_FRAME_STRIDE * units
            local_audio = _audio_tick(frames)
            absolute_audio = _audio_tick(start + frames) - _audio_tick(start)
            if local_audio != absolute_audio:
                valid = False
                break
            starts.append(start)
            lengths.append(frames)
            start += frames - overlap
        if not valid or starts[-1] + lengths[-1] != output:
            continue
        windows = tuple(
            GlobalAVWindow(
                index=index,
                start_frame=start_frame,
                frames=frames,
                video_start=5 * (start_frame // H3_FRAME_STRIDE),
                video_stop=(
                    5 * (start_frame // H3_FRAME_STRIDE) + _video_tokens(frames)
                ),
                audio_start=_audio_tick(start_frame),
                audio_stop=_audio_tick(start_frame + frames),
            )
            for index, (start_frame, frames) in enumerate(zip(starts, lengths))
        )
        return GlobalAVPlan(
            output_frames=output,
            windows=windows,
            window_frames=window,
            stride_frames=stride,
            mechanism="global_joint_av_balanced_views_v2",
        )
    return fallback()


def _video_boundary(frame: int) -> int:
    value = int(frame)
    if value == 0:
        return 0
    return _video_tokens(value)


def plan_prompt_owned_global_av_windows(
    output_frames: int,
    prompt_ranges: tuple[tuple[int, int], ...],
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_WINDOW_STRIDE_FRAMES,
    balanced: bool = False,
) -> GlobalAVPlan:
    """Partition short views by creator range with read-only boundary halos.

    Each view has exactly one prompt owner and its writable interval never
    crosses that owner's creator-window boundary.  A later creator range may
    read a short part of the preceding clean global state as temporal context,
    but that halo is clipped out of its write interval.  This avoids evaluating
    the same read view twice under conflicting prompts while preserving one
    overlapping global solver state.
    """

    output = int(output_frames)
    ranges = tuple((int(start), int(stop)) for start, stop in prompt_ranges)
    if not ranges:
        raise ValueError("prompt-owned co-denoise requires prompt ranges")
    if ranges[0][0] != 0 or ranges[-1][1] != output:
        raise ValueError("prompt ranges must cover the complete output timeline")
    for index, (start, stop) in enumerate(ranges):
        if not 0 <= start < stop <= output:
            raise ValueError("prompt ranges must be positive ordered intervals")
        if index and ranges[index - 1][1] != start:
            raise ValueError("prompt ranges must be contiguous")
        _video_boundary(start)
        _video_boundary(stop)

    window = int(window_frames)
    stride = int(stride_frames)
    _video_tokens(window)
    if not MIN_WINDOW_FRAMES <= window <= MAX_WINDOW_FRAMES:
        raise ValueError("co-denoise window must stay inside H3's native range")
    if stride <= 0 or stride % AV_EXACT_START_STRIDE:
        raise ValueError("co-denoise stride must be a positive multiple of 51")
    if stride >= window:
        raise ValueError("co-denoise windows must overlap")
    overlap = window - stride
    # One extra native 17-frame unit is allowed when it removes a tiny
    # creator-boundary tail.  The configured five-second view therefore stays
    # below 5.9 seconds rather than turning into a ten-second creator window.
    maximum_extended = min(MAX_WINDOW_FRAMES, window + H3_FRAME_STRIDE)
    owned: list[GlobalAVWindow] = []
    for prompt_index, (frame_start, frame_stop) in enumerate(ranges):
        if prompt_index == 0:
            read_start = 0
        else:
            read_start = max(
                0,
                ((frame_start - overlap) // AV_EXACT_START_STRIDE)
                * AV_EXACT_START_STRIDE,
            )
        if read_start >= frame_stop:
            raise RuntimeError("creator-aligned view has no readable interval")
        owner_video_start = _video_boundary(frame_start)
        owner_video_stop = _video_boundary(frame_stop)
        owner_audio_start = _audio_tick(frame_start)
        owner_audio_stop = _audio_tick(frame_stop)
        current = read_start
        balanced_views: tuple[tuple[int, int], ...] | None = None
        readable_frames = frame_stop - read_start
        if balanced and readable_frames > window:
            # The completed low-resolution audio no longer requires every
            # visual view to begin on the stricter 51-frame joint AV grid.
            # Keep one prompt owner, but distribute that owner's total read
            # interval over the minimum number of bounded H3 views.  This
            # avoids a maximum-size first view followed by an under-filled
            # tail while retaining the requested overlap exactly.
            count = max(2, math.ceil((readable_frames - overlap) / stride))
            covered_frames = readable_frames + overlap * (count - 1)
            lattice_units, remainder = divmod(
                covered_frames - count * H3_FRAME_ORIGIN,
                H3_FRAME_STRIDE,
            )
            minimum_units = (
                MINIMUM_TERMINAL_FRAMES - H3_FRAME_ORIGIN
            ) // H3_FRAME_STRIDE
            maximum_units = (window - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE
            if not remainder:
                low_units, high_count = divmod(lattice_units, count)
                if (
                    low_units >= minimum_units
                    and low_units + bool(high_count) <= maximum_units
                ):
                    candidates = [low_units + 1] * high_count + [low_units] * (
                        count - high_count
                    )
                    arrangements: list[list[int]] = []
                    for rotation in range(count):
                        arrangement = candidates[rotation:] + candidates[:rotation]
                        if arrangement not in arrangements:
                            arrangements.append(arrangement)
                    for arrangement in arrangements:
                        views: list[tuple[int, int]] = []
                        start = read_start
                        valid = True
                        for units in arrangement:
                            frames = H3_FRAME_ORIGIN + H3_FRAME_STRIDE * units
                            if (
                                _audio_tick(start + frames) - _audio_tick(start)
                                != _audio_tick(frames)
                            ):
                                valid = False
                                break
                            views.append((start, frames))
                            start += frames - overlap
                        if valid and views[-1][0] + views[-1][1] == frame_stop:
                            balanced_views = tuple(views)
                            break

        planned_views: list[tuple[int, int]] = []
        if balanced_views is not None:
            planned_views.extend(balanced_views)
        else:
            # Conservative maximum-fill layout used when balanced placement
            # cannot satisfy H3's exact video/audio clocks.
            while True:
                remaining = frame_stop - current
                if remaining < MIN_WINDOW_FRAMES:
                    raise RuntimeError("creator-aligned planner emitted a tiny tail")
                frames = remaining if remaining <= maximum_extended else window
                planned_views.append((current, frames))
                if current + frames == frame_stop:
                    break
                candidate = current + stride
                tail = frame_stop - candidate
                if tail < MIN_WINDOW_FRAMES:
                    retreat = math.ceil(
                        (MIN_WINDOW_FRAMES - tail) / AV_EXACT_START_STRIDE
                    )
                    candidate -= retreat * AV_EXACT_START_STRIDE
                if candidate <= current:
                    raise RuntimeError("creator-aligned planner could not advance")
                current = candidate

        for current, frames in planned_views:
            _video_tokens(frames)
            video_start = 5 * (current // H3_FRAME_STRIDE)
            video_stop = video_start + _video_tokens(frames)
            audio_start = _audio_tick(current)
            audio_stop = _audio_tick(current + frames)
            if audio_stop - audio_start != _audio_tick(frames):
                raise RuntimeError("creator-aligned view missed the exact audio clock")
            video_write_start = max(video_start, owner_video_start)
            video_write_stop = min(video_stop, owner_video_stop)
            audio_write_start = max(audio_start, owner_audio_start)
            audio_write_stop = min(audio_stop, owner_audio_stop)
            if (
                video_write_start < video_write_stop
                and audio_write_start < audio_write_stop
            ):
                owned.append(GlobalAVWindow(
                    index=len(owned),
                    start_frame=current,
                    frames=frames,
                    video_start=video_start,
                    video_stop=video_stop,
                    audio_start=audio_start,
                    audio_stop=audio_stop,
                    prompt_index=prompt_index,
                    video_write_start=video_write_start,
                    video_write_stop=video_write_stop,
                    audio_write_start=audio_write_start,
                    audio_write_stop=audio_write_stop,
                ))

    total_video_tokens = _video_tokens(output)
    total_audio_tokens = _audio_tick(output)
    video_coverage = [0] * total_video_tokens
    audio_coverage = [0] * total_audio_tokens
    for view in owned:
        for index in range(view.writable_video_start, view.writable_video_stop):
            video_coverage[index] += 1
        for index in range(view.writable_audio_start, view.writable_audio_stop):
            audio_coverage[index] += 1
    if not owned or not all(video_coverage) or not all(audio_coverage):
        raise RuntimeError("prompt-owned views left part of the global latent uncovered")
    return GlobalAVPlan(
        output_frames=output,
        windows=tuple(owned),
        window_frames=window,
        stride_frames=stride,
        fusion="creator_aligned_partition_of_unity_v2",
        mechanism=(
            "global_creator_aligned_balanced_co_denoise_v3"
            if balanced
            else "global_creator_aligned_co_denoise_v2"
        ),
    )


def triangular_weights(
    length: int,
    *,
    device: Any,
    dtype: Any,
    edge_floor: float = 1.0 / 32.0,
) -> Any:
    """Return strictly-positive center-weighted WWS coefficients."""

    import torch

    count = int(length)
    if count <= 0:
        raise ValueError("fusion weights require a positive length")
    if not 0.0 < float(edge_floor) <= 1.0:
        raise ValueError("edge floor must lie inside (0, 1]")
    if count == 1:
        return torch.ones(1, device=device, dtype=dtype)
    coordinate = torch.linspace(-1.0, 1.0, count, device=device, dtype=torch.float32)
    weight = 1.0 - coordinate.abs()
    weight = weight.mul(1.0 - float(edge_floor)).add(float(edge_floor))
    return weight.to(dtype=dtype)


def _overlap_partition_weights(
    plan: GlobalAVPlan,
    window_index: int,
    *,
    modality: str,
    device: Any,
    dtype: Any,
) -> Any:
    """Return a smooth partition-of-unity weight for one writable interval.

    Earlier WWS fusion used a strictly-positive triangular edge floor.  That
    made a newly entering view affect the global prediction on its very first
    token, which can remain visible after temporal VAE decoding.  Adjacent
    views now use complementary raised-cosine ramps: the entering prediction
    starts at exactly zero while the leaving prediction starts at one.  A
    creator-prompt boundary has no writable overlap, so it receives no blend
    and neighbouring instructions still cannot rewrite one another.
    """

    import torch

    if modality not in {"video", "audio"}:
        raise ValueError("fusion modality must be video or audio")
    current = plan.windows[int(window_index)]

    def interval(item: GlobalAVWindow) -> tuple[int, int]:
        if modality == "video":
            return item.writable_video_start, item.writable_video_stop
        return item.writable_audio_start, item.writable_audio_stop

    start, stop = interval(current)
    count = stop - start
    if count <= 0:
        raise ValueError("fusion interval must be positive")

    previous = None
    for candidate in reversed(plan.windows[: int(window_index)]):
        candidate_start, candidate_stop = interval(candidate)
        if candidate.prompt_index != current.prompt_index:
            continue
        if candidate_start < start < candidate_stop:
            previous = candidate
            break

    following = None
    for candidate in plan.windows[int(window_index) + 1 :]:
        candidate_start, candidate_stop = interval(candidate)
        if candidate.prompt_index != current.prompt_index:
            continue
        if candidate_start < stop and candidate_stop > start:
            following = candidate
            break

    left_overlap = 0
    if previous is not None:
        previous_start, previous_stop = interval(previous)
        left_overlap = min(stop, previous_stop) - max(start, previous_start)
    right_overlap = 0
    if following is not None:
        following_start, following_stop = interval(following)
        right_overlap = min(stop, following_stop) - max(start, following_start)
    if left_overlap < 0 or right_overlap < 0:
        raise RuntimeError("fusion planner produced a negative overlap")
    if left_overlap + right_overlap > count:
        raise RuntimeError("fusion planner produced unsupported triple overlap")

    weight = torch.ones(count, device=device, dtype=torch.float32)

    def rising(length: int) -> Any:
        if length == 1:
            return torch.full((1,), 0.5, device=device, dtype=torch.float32)
        phase = torch.linspace(
            0.0, math.pi / 2.0, length, device=device, dtype=torch.float32
        )
        return phase.sin().square()

    if left_overlap:
        weight[:left_overlap] = rising(left_overlap)
    if right_overlap:
        weight[-right_overlap:] = rising(right_overlap).flip(0)
    return weight.to(dtype=dtype)


def stabilize_global_selflift_seams(
    anchor: Any,
    prediction: Any,
    plan: GlobalAVPlan,
    *,
    strength: float = 0.85,
    padding_tokens: int = 2,
) -> Any:
    """Suppress view-specific temporal residuals while preserving base motion.

    ``anchor`` is the learned high-resolution lift of the accepted connected
    low-resolution x0 trajectory.  The H3 tail should add target-grid detail,
    but a local view can add a slightly different temporal detail residual at
    its overlap.  Around same-prompt overlaps, compare the residual's temporal
    innovation with the anchor motion and smooth only the excess.  This keeps
    real camera/subject motion carried by the anchor and removes the fixed
    texture/contrast pulse caused by a local chart entering the consensus.
    """

    import torch
    import torch.nn.functional as F

    if anchor.shape != prediction.shape or prediction.ndim != 5:
        raise ValueError("SelfLift seam tensors must share B,C,T,H,W shape")
    amount = float(strength)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("SelfLift seam strength must lie inside [0, 1]")
    padding = int(padding_tokens)
    if padding < 0:
        raise ValueError("SelfLift seam padding must be non-negative")
    if amount == 0.0 or int(prediction.shape[2]) < 3:
        return prediction

    seam_mask = torch.zeros(
        int(prediction.shape[2]),
        device=prediction.device,
        dtype=torch.float32,
    )
    previous_by_prompt: dict[int, GlobalAVWindow] = {}
    overlap_ranges: list[tuple[int, int]] = []
    for window in plan.windows:
        previous = previous_by_prompt.get(window.prompt_index)
        if previous is not None:
            start = max(
                previous.writable_video_start,
                window.writable_video_start,
            )
            stop = min(
                previous.writable_video_stop,
                window.writable_video_stop,
            )
            if start < stop:
                overlap_ranges.append((start, stop))
        previous_by_prompt[window.prompt_index] = window
    if not overlap_ranges:
        return prediction

    def smoothstep(value: Any) -> Any:
        value = value.clamp(0.0, 1.0)
        return value.square() * (3.0 - 2.0 * value)

    for start, stop in overlap_ranges:
        seam_mask[start:stop] = 1.0
        if padding:
            left = max(0, start - padding)
            if left < start:
                phase = torch.linspace(
                    0.0,
                    1.0,
                    start - left + 1,
                    device=prediction.device,
                    dtype=torch.float32,
                )[:-1]
                seam_mask[left:start] = torch.maximum(
                    seam_mask[left:start], smoothstep(phase)
                )
            right = min(int(prediction.shape[2]), stop + padding)
            if stop < right:
                phase = torch.linspace(
                    1.0,
                    0.0,
                    right - stop + 1,
                    device=prediction.device,
                    dtype=torch.float32,
                )[1:]
                seam_mask[stop:right] = torch.maximum(
                    seam_mask[stop:right], smoothstep(phase)
                )

    anchor_value = anchor.float()
    residual = prediction.float() - anchor_value

    def temporal_binomial(value: Any) -> Any:
        previous = torch.cat((value[:, :, :1], value[:, :, :-1]), dim=2)
        following = torch.cat((value[:, :, 1:], value[:, :, -1:]), dim=2)
        return (previous + 2.0 * value + following) * 0.25

    residual_smoothed = temporal_binomial(residual)
    anchor_smoothed = temporal_binomial(anchor_value)
    residual_innovation = residual - residual_smoothed
    anchor_innovation = anchor_value - anchor_smoothed
    residual_score = residual_innovation.square().mean(dim=1).sqrt()
    motion_score = anchor_innovation.square().mean(dim=1).sqrt()
    excess_fraction = (
        (residual_score - 1.15 * motion_score).clamp_min(0.0)
        / residual_score.clamp_min(1e-6)
    )
    # A coherent spatial mask avoids channel-wise speckle.  A small floor
    # also damps a smooth view-wide bias that is too subtle to be a statistical
    # outlier but remains visible as a quality pulse during playback.
    excess_fraction = F.avg_pool3d(
        excess_fraction.unsqueeze(1),
        kernel_size=(1, 3, 3),
        stride=1,
        padding=(0, 1, 1),
    )
    gate = (0.12 + 0.88 * excess_fraction).clamp_max(1.0)
    gate = gate * seam_mask.view(1, 1, -1, 1, 1) * amount
    stabilized_residual = residual + gate * (residual_smoothed - residual)
    return (anchor_value + stabilized_residual).to(prediction.dtype)


def hybrid_temporal_noise(noise: Any, *, shared_strength: float) -> Any:
    """Inject a low-rank global component into one modality's initial noise.

    This is the global-latent form of Diff-VF HNI.  A small request-wide
    component gives distant windows a common stochastic basis while the
    independent term retains motion diversity.  Per-channel normalization
    keeps the marginal scale close to the original standard Gaussian.
    """

    import torch

    strength = float(shared_strength)
    if not 0.0 <= strength < 1.0:
        raise ValueError("HNI shared strength must lie inside [0, 1)")
    if strength == 0.0:
        return noise
    if noise.ndim not in (4, 5):
        raise ValueError("HNI expects an H3 audio or video latent")
    time_axis = -1 if noise.ndim == 4 else 2
    value = noise.float()
    shared = value.mean(dim=time_axis, keepdim=True)
    reduce_axes = tuple(
        axis
        for axis in range(1, shared.ndim)
        if axis != (time_axis % shared.ndim)
    )
    shared = shared - shared.mean(dim=reduce_axes, keepdim=True)
    shared = shared / shared.square().mean(
        dim=reduce_axes, keepdim=True
    ).sqrt().clamp_min(1e-6)
    mixed = (
        math.sqrt(1.0 - strength) * value
        + math.sqrt(strength) * shared
    )
    return mixed.to(noise.dtype).contiguous()


def tes_video_indices(
    global_tokens: int,
    view_tokens: int,
    *,
    device: Any,
) -> Any:
    """Select a monotonic sparse video view spanning first through last."""

    import torch

    total = int(global_tokens)
    count = int(view_tokens)
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("TES view size must lie inside the global video clock")
    indices = torch.linspace(
        0, total - 1, count, device=device, dtype=torch.float32
    ).round().to(torch.long)
    if int(torch.unique_consecutive(indices).numel()) != count:
        raise RuntimeError("TES index construction produced duplicate positions")
    return indices


def resample_audio_summary(audio: Any, target_tokens: int) -> Any:
    """Build a continuous low-resolution audio companion for video-only TES."""

    import torch.nn.functional as F

    target = int(target_tokens)
    if audio.ndim != 4 or target <= 0:
        raise ValueError("TES audio summary requires a ranked positive clock")
    source = int(audio.shape[-1])
    if source == target:
        return audio
    shape = audio.shape
    value = F.interpolate(
        audio.reshape(-1, 1, source).float(),
        size=target,
        mode="linear",
        align_corners=True,
    )
    return value.reshape(*shape[:-1], target).to(audio.dtype).contiguous()


def expand_tes_video_prediction(prediction: Any, global_tokens: int) -> Any:
    """Interpolate an early TES structural prediction onto the global clock."""

    import torch.nn.functional as F

    if prediction.ndim != 5:
        raise ValueError("TES video prediction must be five-dimensional")
    target = int(global_tokens)
    batch, channels, source, height, width = prediction.shape
    if target <= 0:
        raise ValueError("TES global clock must be positive")
    if source == target:
        return prediction
    value = prediction.permute(0, 1, 3, 4, 2).reshape(
        batch * channels * height * width, 1, source
    )
    value = F.interpolate(
        value.float(), size=target, mode="linear", align_corners=True
    )
    return value.reshape(batch, channels, height, width, target).permute(
        0, 1, 4, 2, 3
    ).to(prediction.dtype).contiguous()


def fuse_global_av_predictions(
    video: Any,
    audio: Any,
    plan: GlobalAVPlan,
    predict_window: Callable[[GlobalAVWindow, Any, Any], tuple[Any, Any]],
) -> tuple[Any, Any]:
    """Fuse one timestep of local H3 predictions into a global AV prediction.

    The callback receives views of the current global noisy latent and returns
    model predictions in the same shapes.  Fusion is performed in float32 for
    stable overlap normalization, then converted back to the input dtypes.
    """

    import torch

    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("global co-denoise expects H3 video/audio latent ranks")
    if int(video.shape[2]) != plan.video_tokens:
        raise ValueError("global video latent does not match its window plan")
    if int(audio.shape[-1]) != plan.audio_tokens:
        raise ValueError("global audio latent does not match its window plan")

    video_sum = torch.zeros_like(video, dtype=torch.float32)
    audio_sum = torch.zeros_like(audio, dtype=torch.float32)
    video_weight = torch.zeros(
        (1, 1, video.shape[2], 1, 1), device=video.device, dtype=torch.float32
    )
    audio_weight = torch.zeros(
        (1, 1, 1, audio.shape[-1]), device=audio.device, dtype=torch.float32
    )
    for window_index, window in enumerate(plan.windows):
        local_video = video[:, :, window.video_start:window.video_stop]
        local_audio = audio[..., window.audio_start:window.audio_stop]
        predicted_video, predicted_audio = predict_window(
            window, local_video, local_audio
        )
        if tuple(predicted_video.shape) != tuple(local_video.shape):
            raise ValueError("window video prediction changed latent geometry")
        if tuple(predicted_audio.shape) != tuple(local_audio.shape):
            raise ValueError("window audio prediction changed latent geometry")
        vw = _overlap_partition_weights(
            plan,
            window_index,
            modality="video",
            device=video.device,
            dtype=torch.float32,
        ).view(1, 1, -1, 1, 1)
        aw = _overlap_partition_weights(
            plan,
            window_index,
            modality="audio",
            device=audio.device,
            dtype=torch.float32,
        ).view(1, 1, 1, -1)
        video_local_start = window.writable_video_start - window.video_start
        video_local_stop = window.writable_video_stop - window.video_start
        audio_local_start = window.writable_audio_start - window.audio_start
        audio_local_stop = window.writable_audio_stop - window.audio_start
        video_sum[
            :, :, window.writable_video_start:window.writable_video_stop
        ].add_(
            predicted_video[:, :, video_local_start:video_local_stop].float() * vw
        )
        audio_sum[
            ..., window.writable_audio_start:window.writable_audio_stop
        ].add_(
            predicted_audio[..., audio_local_start:audio_local_stop].float() * aw
        )
        video_weight[
            :, :, window.writable_video_start:window.writable_video_stop
        ].add_(vw)
        audio_weight[
            ..., window.writable_audio_start:window.writable_audio_stop
        ].add_(aw)

    if bool((video_weight <= 0).any().item()) or bool((audio_weight <= 0).any().item()):
        raise RuntimeError("global AV fusion left uncovered latent positions")
    return (
        (video_sum / video_weight).to(video.dtype),
        (audio_sum / audio_weight).to(audio.dtype),
    )


__all__ = [
    "AV_EXACT_START_STRIDE",
    "DEFAULT_WINDOW_FRAMES",
    "DEFAULT_WINDOW_STRIDE_FRAMES",
    "window_geometry_for_seconds",
    "GlobalAVPlan",
    "plan_balanced_global_av_windows",
    "GlobalAVWindow",
    "fuse_global_av_predictions",
    "expand_tes_video_prediction",
    "hybrid_temporal_noise",
    "plan_prompt_owned_global_av_windows",
    "plan_global_av_windows",
    "resample_audio_summary",
    "stabilize_global_selflift_seams",
    "tes_video_indices",
    "triangular_weights",
]
