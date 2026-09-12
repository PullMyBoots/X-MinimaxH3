"""Native MiniMax H3 long-horizon planning and latent continuity primitives.

The public request remains one prompt plus one duration.  Internally a long
request is represented as one ordinary H3 opening window followed by bounded
continuation windows.  Every continuation carries a fixed-size exact joint AV
prefix; the prefix is restored after every solver update and removed once when
the clean segment latents are assembled.

The transport follows the masked joint-AV prefix mechanism validated by H3
Continuum V3.6/V3.7.  This module is an independent native-service adaptation:
it contains no ComfyUI dependency and never decodes between windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


FPS = 24
AUDIO_LATENT_HZ = 40
H3_FRAME_ORIGIN = 5
H3_FRAME_STRIDE = 17
MAX_NATIVE_WINDOW_FRAMES = 362
DEFAULT_CONTEXT_FRAMES = 39
DEFAULT_VISIBLE_STRIDE_FRAMES = 119
# Keep the proven V1/V2 video window geometry exactly unchanged.  Audio repairs
# use causal overlap-save across the existing 39-frame context instead of
# lengthening the joint AV window (which changes video noise geometry and
# prompt timing).  The repainted overlap is burn-in for the incoming trajectory
# and is discarded; independently completed Audio-VAE latents are never mixed.
DEFAULT_AUDIO_BRIDGE_CONTEXT_FRAMES = DEFAULT_CONTEXT_FRAMES
# Repaint and discard the final 65 Audio-VAE ticks (1.625 s) of each context.
# Shorter burn-in produced large joint AV bursts at every seam in the matched
# V20 ablation; the full band therefore remains the conservative default.
DEFAULT_AUDIO_BRIDGE_TICKS = 65
CONTINUATION_CANDIDATE_MAX_ATTEMPTS = 4
CONTINUATION_OVERLAP_LOW_FREQUENCY_RELATIVE_RMS_MAX = 0.30
CONTINUATION_OVERLAP_LOW_FREQUENCY_COSINE_MIN = 0.94
CONTINUATION_TRAJECTORY_RELATIVE_RMS_MIN = 0.60
CONTINUATION_TRAJECTORY_LOCAL_OUTLIER_RATIO_MIN = 1.50
# A declared single take needs more exact history than the general multi-shot
# route to preserve camera velocity and occlusion/reappearance state.  The
# Fifty-six frames preserve enough exact camera/object state for the matched
# single-take cases without the insert-shot regression observed at 73/90
# frames.  The width remains fixed, so duration scaling is linear in bounded
# windows; audio quality is handled after the joint trajectory rather than by
# perturbing this validated visual geometry.
SINGLE_TAKE_CONTEXT_FRAMES = 56
# Keep larger research contexts loadable for controlled ablations without
# making the rejected 90-frame geometry the production single-take default.
MAX_CONTINUATION_CONTEXT_FRAMES = 90
MIN_PHYSICAL_WINDOW_FRAMES = 124

# Calibrated on the native 4090 INT8 path: a window has substantial fixed
# conditioning/session cost, while the dominant token interaction grows
# superlinearly.  Absolute units are irrelevant; their ratio selects the
# latency knee.  A smooth event penalty keeps an internal seam away from a
# user-authored H3 timestamp without creating prompt-specific branches.
_WINDOW_FIXED_COST = 16.0
_WINDOW_QUADRATIC_COST = 0.000152
_EVENT_SEAM_PENALTY = 30.0
_EVENT_SEAM_SIGMA_FRAMES = 24.0
# Structured director requests benefit from placing a compute-window boundary
# near an authored shot boundary: the next continuation can then carry the
# exact clean world/audio state while receiving only the newly active shot
# description.  Keep the attraction below the fixed cost of an extra window;
# it may move an already useful boundary, but cannot manufacture windows just
# to collect the reward.
_DIRECTOR_EVENT_SEAM_REWARD = 8.0


def align_h3_frames(frame_count: int) -> int:
    """Return the nearest positive frame count on H3's ``5 + 17*k`` grid."""

    requested = max(H3_FRAME_ORIGIN, int(frame_count))
    index = max(0, round((requested - H3_FRAME_ORIGIN) / H3_FRAME_STRIDE))
    return H3_FRAME_ORIGIN + H3_FRAME_STRIDE * index


def video_latent_frames(frame_count: int) -> int:
    frames = int(frame_count)
    if frames < H3_FRAME_ORIGIN or (frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
        raise ValueError("frame count must satisfy H3's 5 + 17*k grid")
    return 2 + 5 * ((frames - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE)


def audio_latent_frames(frame_count: int) -> int:
    return int(round(int(frame_count) / FPS * AUDIO_LATENT_HZ))


def _derived_seed(seed: int, index: int) -> int:
    """SplitMix64-derived per-window seed without a correlated ``seed + i`` run."""

    if index == 0:
        return int(seed) & ((1 << 64) - 1)
    value = (int(seed) + index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)


@dataclass(frozen=True, slots=True)
class LongHorizonSegment:
    index: int
    window_frames: int
    context_frames: int
    global_context_start_frame: int
    visible_start_frame: int
    visible_frames: int
    seed: int
    prompt: str
    # Optional compiler-owned boundary condition. Free-form legacy plans leave
    # this unset; structured v2 continuation windows can provide a condition
    # that is deliberately free of old actions and future destinations.
    continuation_bridge_prompt: str | None = None
    # Semantic camera boundary.  It is carried independently of prefix width
    # because final temporal VAE decoding must reset exactly at authored cuts.
    transition: str = "continue"
    video_prefix_frames: int | None = None
    # Optional lower bound for visual long-memory routing.  A structured
    # single-take continuation uses only anchors from its exact carried-state
    # band.  This prevents an obsolete early location from competing with the
    # current physical state while keeping the free-prompt route unchanged.
    visual_memory_floor_frame: int | None = None
    # Include the opening identity anchor only when the director has decided it
    # cannot contradict the active camera or mutable object state. Continuous
    # takes leave this false because the clean prefix and latest anchor are the
    # authoritative visible state.
    visual_memory_include_canonical: bool = False
    visual_memory_layout_anchor_start_frame: int | None = None
    visual_memory_layout_anchor_stop_frame: int | None = None
    visual_memory_progressive_layout_state: bool = False
    preserve_latest_visual: bool = False
    # ``None`` preserves the legacy/free-prompt behavior.  A non-negative
    # value is compiled only from structural ``<d>...</d>`` ownership after
    # the global timeline has been localized; dialogue text is never
    # interpreted here.
    authorized_dialogue_count: int | None = None
    # Global onset frames of structurally timed ``<d>`` events owned by this
    # writable suffix.  Audio-memory selection uses only these coordinates to
    # avoid mistaking continuous ambience for a voice exemplar; it never
    # reads, classifies, or rewrites the dialogue text.
    authorized_dialogue_frames: tuple[int, ...] = ()
    # Public reference audio and inferred long-term audio memory are both
    # voice-bearing authorities.  A structured window with an explicit
    # dialogue contract may receive them only when it owns at least one local
    # dialogue event.  The exact short-term AV prefix remains available for
    # physical seam continuity independently of these long-term conditions.
    reference_audio_active: bool = True
    audio_memory_active: bool = True
    opening_preroll_frames: int = 0
    novel_camera_cut: bool = False
    terminal_video_seed: bool = False
    novel_camera_layout_probe: bool = False

    @property
    def visible_stop_frame(self) -> int:
        return self.visible_start_frame + self.visible_frames


@dataclass(frozen=True, slots=True)
class LongHorizonPlan:
    requested_duration_seconds: float
    output_frames: int
    actual_duration_seconds: float
    segments: tuple[LongHorizonSegment, ...]
    context_frames: int = DEFAULT_CONTEXT_FRAMES
    mechanism: str = "masked_joint_av_prefix_v1"
    planning_policy: str = "event_aware_continuous_cost_v1"
    structured_director: bool = False

    @property
    def continuation_count(self) -> int:
        return max(0, len(self.segments) - 1)

    def telemetry(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "requested_duration_seconds": self.requested_duration_seconds,
            "output_frames": self.output_frames,
            "actual_duration_seconds": self.actual_duration_seconds,
            "context_frames": self.context_frames,
            "planning_policy": self.planning_policy,
            "prompt_localization_policy": (
                "structured_director_causal_state_v13_isolated_context_latch"
                if self.structured_director
                else "visible_interval_event_ownership_v2"
            ),
            "structured_director": self.structured_director,
            "audio_clock_policy": "cumulative_global_resample_v2",
            "continuation_count": self.continuation_count,
            "segments": [
                {
                    "index": segment.index,
                    "window_frames": segment.window_frames,
                    "context_frames": segment.context_frames,
                    "global_context_start_frame": segment.global_context_start_frame,
                    "visible_start_frame": segment.visible_start_frame,
                    "visible_frames": segment.visible_frames,
                    "seed": segment.seed,
                    "transition": segment.transition,
                    "continuation_boundary_condition": bool(
                        segment.continuation_bridge_prompt
                    ),
                    "video_prefix_frames": segment.video_prefix_frames,
                    "visual_memory_floor_frame": (
                        segment.visual_memory_floor_frame
                    ),
                    "visual_memory_include_canonical": (
                        segment.visual_memory_include_canonical
                    ),
                    "visual_memory_layout_anchor_start_frame": (
                        segment.visual_memory_layout_anchor_start_frame
                    ),
                    "visual_memory_layout_anchor_stop_frame": (
                        segment.visual_memory_layout_anchor_stop_frame
                    ),
                    "visual_memory_progressive_layout_state": (
                        segment.visual_memory_progressive_layout_state
                    ),
                    "preserve_latest_visual": segment.preserve_latest_visual,
                    "authorized_dialogue_count": (
                        segment.authorized_dialogue_count
                    ),
                    "authorized_dialogue_frames": list(
                        segment.authorized_dialogue_frames
                    ),
                    "reference_audio_active": segment.reference_audio_active,
                    "audio_memory_active": segment.audio_memory_active,
                    "opening_preroll_frames": segment.opening_preroll_frames,
                    "novel_camera_cut": segment.novel_camera_cut,
                    "terminal_video_seed": segment.terminal_video_seed,
                    "novel_camera_layout_probe": segment.novel_camera_layout_probe,
                }
                for segment in self.segments
            ],
        }


@dataclass(frozen=True, slots=True)
class ShotDecodeGroup:
    """One temporal-VAE domain bounded by authored hard cuts.

    A cut window has a hidden, writable visual preroll.  ``physical_frames``
    retains that preroll for VAE context while ``visible_frames`` is the part
    copied into the delivered timeline after decoding.
    """

    index: int
    segment_indices: tuple[int, ...]
    lead_context_frames: int
    physical_frames: int
    visible_frames: int


def plan_shot_decode_groups(
    segments: Iterable[LongHorizonSegment],
) -> tuple[ShotDecodeGroup, ...]:
    """Partition windows at explicit cuts without changing their clocks."""

    items = tuple(segments)
    if not items:
        raise ValueError("at least one long-horizon segment is required")
    starts = [0]
    for position, segment in enumerate(items):
        transition = str(segment.transition)
        if transition not in ("opening", "continue", "cut"):
            raise ValueError(f"unsupported segment transition: {transition}")
        if position and transition == "opening":
            raise ValueError("opening transition is valid only for the first segment")
        if position and transition == "cut":
            starts.append(position)
        opening_preroll = int(getattr(segment, "opening_preroll_frames", 0))
        if opening_preroll < 0 or (opening_preroll and position):
            raise ValueError("opening preroll is valid only on the first segment")
        if opening_preroll % H3_FRAME_STRIDE:
            raise ValueError("opening preroll must contain whole H3 temporal units")
        if segment.window_frames != (
            segment.context_frames + segment.visible_frames + opening_preroll
        ):
            raise ValueError(
                "segment physical clock does not equal context, preroll and visible frames"
            )
    starts.append(len(items))

    groups: list[ShotDecodeGroup] = []
    for group_index, (start, stop) in enumerate(zip(starts, starts[1:])):
        group_items = items[start:stop]
        lead = (
            int(getattr(group_items[0], "opening_preroll_frames", 0))
            if not start else
            int(group_items[0].context_frames)
        )
        visible = sum(int(segment.visible_frames) for segment in group_items)
        groups.append(ShotDecodeGroup(
            index=group_index,
            segment_indices=tuple(range(start, stop)),
            lead_context_frames=lead,
            physical_frames=lead + visible,
            visible_frames=visible,
        ))
    return tuple(groups)


_SHOT_RE = re.compile(r"(?=\[Shot\s+\d+\])", re.IGNORECASE)
_AT_TIME_RE = re.compile(
    r"\bAt\s+(?:about\s+)?(?:(?P<hour>\d{1,2}):)?(?P<minute>\d{2}):(?P<second>\d{2}(?:\.\d+)?)",
    re.IGNORECASE,
)
_TAIL_RE = re.compile(
    r"(?=\n\s*(?:overall_soundscape|non_diegetic_music)\s*:)",
    re.IGNORECASE,
)
_OVERALL_SOUNDSCAPE_RE = re.compile(
    r"^\s*overall_soundscape\s*:\s*(?P<body>.*?)"
    r"(?=^\s*non_diegetic_music\s*:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_NON_DIEGETIC_MUSIC_RE = re.compile(
    r"^\s*non_diegetic_music\s*:\s*(?P<body>.*)\Z",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_MAIN_DESCRIPTION_RE = re.compile(
    r"^(?:integrated_multimodal_description|detailed_description)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_SUMMARY_SECTION_RE = re.compile(
    r"^summary\s*:", re.IGNORECASE | re.MULTILINE,
)
_RETENTION_SECTION_RE = re.compile(
    r"^retention_analysis\s*:", re.IGNORECASE | re.MULTILINE,
)
_TASK_TYPES_RE = re.compile(r"\[[^\]\n]+\]")
_REFERENCE_RELATION_RE = re.compile(
    r"^\s*(?P<label><(?:Subject|Picture|Video|Audio)\s+\d+>)"
    r"[^\n]*?:\s*(?P<relation>fully_preserved|partially_preserved|"
    r"attribute_transfer|weak_reference|fully_copy|partially_copy|reference)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CLOCK_RANGE_RE = re.compile(
    r"(?:\bfrom\s+)?"
    r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}(?:\.\d+)?)\s*"
    r"(?:to|through|until|[-–—~～]|到|至)\s*"
    r"(?P<stop>(?:\d{1,2}:)?\d{2}:\d{2}(?:\.\d+)?)",
    re.IGNORECASE,
)
_DIALOGUE_SPAN_RE = re.compile(r"<d>.*?</d>", re.IGNORECASE | re.DOTALL)
_SECOND_RANGE_RE = re.compile(
    r"(?:第\s*)?(?P<start>\d+(?:\.\d+)?)\s*秒\s*"
    r"(?:到|至|[-–—~～])\s*"
    r"(?:第\s*)?(?P<stop>\d+(?:\.\d+)?)\s*秒"
)
_SENTENCE_BREAK_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=</d>)\s+",
    re.IGNORECASE,
)
# H3 prompts often put an atomic onset and its visible result on opposite
# sides of a semicolon ("presses once; the light appears").  This narrower
# splitter is intentionally confined to event latching; changing the general
# sentence splitter would also alter shot-progress and reference retention.
_EVENT_CONTINUATION_BREAK_RE = re.compile(
    r"(?<=[.!?;；])\s+|(?<=</d>)\s+",
    re.IGNORECASE,
)
def _time_seconds(match: re.Match[str]) -> float:
    return (
        float(match.group("hour") or 0) * 3600.0
        + float(match.group("minute")) * 60.0
        + float(match.group("second"))
    )


def _format_local_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def _shot_opening_clock_match(text: str) -> re.Match[str] | None:
    """Return only a timestamp that directly schedules a ``[Shot N]`` cut.

    A later ``At ...`` inside the same shot is a separately owned action or
    dialogue event.  Treating that clock as the shot start makes the same
    utterance visible to two overlapping continuation windows.
    """

    value = str(text)
    header_close = value.find("]")
    if header_close < 0:
        return None
    match = _AT_TIME_RE.search(value, header_close + 1)
    if match is None:
        return None
    if value[header_close + 1 : match.start()].strip(" ,"):
        return None
    return match


def scale_h3_prompt_timeline(
    prompt: str,
    *,
    source_frames: int,
    target_frames: int,
) -> str:
    """Compress explicit H3 shot clocks onto a TES temporal view.

    TES presents sparse positions from the complete output as one native-size
    sequence.  Its text clock must undergo the same linear coordinate change;
    otherwise an event at global 00:20 remains outside an 11-second TES view.
    This operation interprets only explicit timestamps and is independent of
    subjects, scenes, actions, or continuity wording.
    """

    source = int(source_frames)
    target = int(target_frames)
    if source <= 0 or target <= 0:
        raise ValueError("timeline scale requires positive frame counts")
    scale = target / source

    def replace(match: re.Match[str]) -> str:
        return f"At {_format_local_time(_time_seconds(match) * scale)}"

    return _AT_TIME_RE.sub(replace, str(prompt))


def _clock_text_seconds(value: str) -> float:
    parts = [float(part) for part in str(value).split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"invalid timeline clock: {value!r}")


def _timeline_ranges(text: str) -> list[tuple[int, int, float, float]]:
    found: list[tuple[int, int, float, float]] = []
    for match in _CLOCK_RANGE_RE.finditer(text):
        found.append((
            match.start(), match.end(),
            _clock_text_seconds(match.group("start")),
            _clock_text_seconds(match.group("stop")),
        ))
    for match in _SECOND_RANGE_RE.finditer(text):
        found.append((
            match.start(), match.end(),
            float(match.group("start")), float(match.group("stop")),
        ))
    found.sort(key=lambda item: item[0])
    clean: list[tuple[int, int, float, float]] = []
    previous_stop = -1
    for item in found:
        if item[0] < previous_stop:
            continue
        if item[3] <= item[2]:
            continue
        clean.append(item)
        previous_stop = item[1]
    return clean


def _prompt_event_frames(prompt: str, output_frames: int) -> tuple[int, ...]:
    """Extract authored shot boundaries for seam placement only.

    Dialogue and action clocks are useful to the local text timebase but are
    not scene boundaries.  Feeding every ``At about`` cue into the partition
    optimizer makes wording changes move the physical windows and can put a
    seam through the very utterance that the cue was meant to schedule.  In a
    shot-structured H3 prompt, accept only a clock immediately following the
    ``[Shot N]`` header.  Range-only prompts retain their explicit intervals.
    """

    original = str(prompt)
    description_match = _MAIN_DESCRIPTION_RE.search(original)
    narrative = (
        original[description_match.end() :]
        if description_match is not None
        else original
    )
    shot_starts = [match.start() for match in _SHOT_RE.finditer(narrative)]
    seconds: list[float] = []
    if shot_starts:
        for index, start in enumerate(shot_starts):
            stop = (
                shot_starts[index + 1]
                if index + 1 < len(shot_starts)
                else len(narrative)
            )
            chunk = narrative[start:stop]
            match = _shot_opening_clock_match(chunk)
            if match is None:
                continue
            seconds.append(_time_seconds(match))
    for _, _, start, stop in _timeline_ranges(narrative):
        seconds.extend((start, stop))
    return tuple(sorted({
        int(round(value * FPS))
        for value in seconds
        if 0 < int(round(value * FPS)) < int(output_frames)
    }))


def _single_take_event_frames(prompt: str, output_frames: int) -> tuple[int, ...]:
    """Return every explicit event onset in a declared single take.

    A multi-shot plan uses only shot openings because dialogue/action clocks
    must not masquerade as camera cuts.  A single-take plan has the opposite
    constraint: there are no legal cuts, and making a continuation start only
    a few frames before a substantial timed action encourages H3 to jump to
    that action's destination.  These clocks therefore supply bounded
    denoising pre-roll, without changing their meaning or adding a cut.
    """

    original = str(prompt)
    description_match = _MAIN_DESCRIPTION_RE.search(original)
    narrative = (
        original[description_match.end() :]
        if description_match is not None
        else original
    )
    return tuple(sorted({
        int(round(_time_seconds(match) * FPS))
        for match in _AT_TIME_RE.finditer(narrative)
        if 0 < int(round(_time_seconds(match) * FPS)) < int(output_frames)
    }))


def _dialogue_event_frames(prompt: str, output_frames: int) -> tuple[int, ...]:
    """Return clocks whose syntactic event payload contains H3 dialogue.

    This is a structural clock/tag join, not semantic prompt inspection.  The
    span after each ``At`` marker ends at the next marker, so a later dialogue
    cannot accidentally lend its timestamp to an earlier action.  Untimed
    dialogue remains valid generation input but supplies no trusted coordinate
    for long-term voice-excerpt selection.
    """

    original = str(prompt)
    description_match = _MAIN_DESCRIPTION_RE.search(original)
    narrative = (
        original[description_match.end() :]
        if description_match is not None
        else original
    )
    clocks = list(_AT_TIME_RE.finditer(narrative))
    frames: set[int] = set()
    for index, match in enumerate(clocks):
        stop = (
            clocks[index + 1].start()
            if index + 1 < len(clocks)
            else len(narrative)
        )
        if not _DIALOGUE_SPAN_RE.search(narrative[match.start() : stop]):
            continue
        frame = int(round(_time_seconds(match) * FPS))
        if 0 <= frame < int(output_frames):
            frames.add(frame)
    return tuple(sorted(frames))


def has_structured_timeline(prompt: str) -> bool:
    """Return whether a prompt supplies an explicit executable timeline.

    This deliberately recognizes syntax rather than story semantics.  One
    declared ``[Shot N]`` block is itself an executable full-timeline
    single-take contract.  Two or more blocks still require at least one
    authored opening clock so incidental repeated labels do not create an
    ordered edit.  Explicit clock/second ranges are accepted as the simpler
    user-facing alternative.  Free prose remains on the unchanged compatible
    path even when structured-director mode is requested globally.
    """

    original = str(prompt)
    description_match = _MAIN_DESCRIPTION_RE.search(original)
    narrative = (
        original[description_match.end() :]
        if description_match is not None
        else original
    )
    if _timeline_ranges(narrative):
        return True
    shot_starts = [match.start() for match in _SHOT_RE.finditer(narrative)]
    if len(shot_starts) == 1:
        return True
    if not shot_starts:
        return False
    for index, start in enumerate(shot_starts):
        stop = (
            shot_starts[index + 1]
            if index + 1 < len(shot_starts)
            else len(narrative)
        )
        if _shot_opening_clock_match(narrative[start:stop]) is not None:
            return True
    return False


def _seam_event_cost(
    frame: int,
    event_frames: tuple[int, ...],
    *,
    structured_director: bool = False,
    director_preroll_frames: int = 0,
) -> float:
    if not event_frames:
        return 0.0
    if structured_director:
        # A continuation must begin *before* the next authored cut, with one
        # protected-context span of writable pre-roll.  Starting only a few
        # frames before the cut gives the noisy suffix no time to plan the new
        # composition: H3 commonly performs it seconds late.  The preceding
        # exact prefix still owns the outgoing shot, while the pre-roll lets a
        # single trajectory place the discrete cut at the user's clock.
        forward = [event - int(frame) for event in event_frames if event >= int(frame)]
        if not forward:
            return 0.0
        preroll = max(0, int(director_preroll_frames))
        distance = min(abs(gap - preroll) for gap in forward)
    else:
        distance = min(abs(int(frame) - event) for event in event_frames)
    scaled = distance / _EVENT_SEAM_SIGMA_FRAMES
    strength = (
        -_DIRECTOR_EVENT_SEAM_REWARD
        if structured_director
        else _EVENT_SEAM_PENALTY
    )
    return strength * math.exp(-0.5 * scaled * scaled)


def _optimize_window_frames(
    *,
    output_frames: int,
    context_frames: int,
    event_frames: tuple[int, ...],
    maximum_frames: int,
    structured_director: bool = False,
) -> tuple[int, tuple[int, ...]]:
    """Solve the bounded H3 window partition on its exact integer grid.

    The first window is ``5 + 17*k`` frames.  Every later window contains the
    fixed clean context plus ``17*k`` new visible frames.  Dynamic programming
    minimizes a continuous latency proxy plus a smooth semantic-boundary cost;
    it therefore works for arbitrary duration and prompt timing without a
    table of duration-specific schedules.
    """

    if output_frames <= maximum_frames:
        return output_frames, ()
    if (output_frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
        raise ValueError("long output is not on the H3 temporal grid")
    minimum_opening_units = (
        MIN_PHYSICAL_WINDOW_FRAMES - H3_FRAME_ORIGIN
    ) // H3_FRAME_STRIDE
    maximum_opening_units = (
        maximum_frames - H3_FRAME_ORIGIN
    ) // H3_FRAME_STRIDE
    # A long protected context can itself exceed the minimum physical window.
    # The visible suffix must nevertheless contain at least one H3 grid unit;
    # allowing a zero/negative stride also makes ``remainder`` exceed the DP
    # table and used to crash valid 141/192-frame context plans.
    minimum_stride_units = max(
        1,
        math.ceil(
            (MIN_PHYSICAL_WINDOW_FRAMES - context_frames) / H3_FRAME_STRIDE
        ),
    )
    maximum_stride_units = (
        maximum_frames - context_frames
    ) // H3_FRAME_STRIDE
    total_units = (output_frames - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE
    # A continuation state's global cursor is fully determined by the number
    # of output-grid units still ungenerated.  The former implementation kept
    # both ``strides_left`` and ``visible_cursor`` in a separately rebuilt
    # cache for every possible window count/opening pair.  That enumerated the
    # same partitions many times and became impractical near the public 60 s
    # limit.  This single shortest-path pass visits each remaining-unit state
    # once: O(total_units * legal_stride_units), with exactly the same cost.
    # Each value is (tail cost, continuation count, stride sequence); the
    # latter fields preserve the old deterministic tie-breaking semantics.
    tail: list[tuple[float, int, tuple[int, ...]] | None] = [
        None
    ] * (total_units + 1)
    tail[0] = (0.0, 0, ())
    for units_left in range(1, total_units + 1):
        local_best: tuple[float, int, tuple[int, ...]] | None = None
        high = min(maximum_stride_units, units_left)
        for stride_units in range(minimum_stride_units, high + 1):
            remainder = units_left - stride_units
            downstream = tail[remainder]
            if downstream is None:
                continue
            downstream_cost, downstream_count, downstream_strides = downstream
            visible = H3_FRAME_STRIDE * stride_units
            physical = context_frames + visible
            next_cursor = output_frames - H3_FRAME_STRIDE * remainder
            semantic = (
                _seam_event_cost(
                    next_cursor,
                    event_frames,
                    structured_director=structured_director,
                    director_preroll_frames=(
                        context_frames if structured_director else 0
                    ),
                )
                if remainder > 0
                else 0.0
            )
            candidate = (
                _WINDOW_FIXED_COST
                + _WINDOW_QUADRATIC_COST * physical * physical
                + semantic
                + downstream_cost,
                1 + downstream_count,
                (stride_units,) + downstream_strides,
            )
            if local_best is None or candidate < local_best:
                local_best = candidate
        tail[units_left] = local_best

    best: tuple[float, int, int, tuple[int, ...]] | None = None
    opening_candidates = range(
        minimum_opening_units, maximum_opening_units + 1
    )
    for opening_units in opening_candidates:
        remaining_units = total_units - opening_units
        if remaining_units <= 0 or remaining_units >= len(tail):
            continue
        downstream = tail[remaining_units]
        if downstream is None:
            continue
        tail_cost, continuation_count, stride_units = downstream
        opening_frames = H3_FRAME_ORIGIN + H3_FRAME_STRIDE * opening_units
        objective = (
            _WINDOW_FIXED_COST
            + _WINDOW_QUADRATIC_COST * opening_frames * opening_frames
            + _seam_event_cost(
                opening_frames,
                event_frames,
                structured_director=structured_director,
                director_preroll_frames=(
                    context_frames if structured_director else 0
                ),
            )
            + tail_cost
        )
        candidate = (
            objective,
            1 + continuation_count,
            opening_units,
            stride_units,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no bounded H3 long-horizon partition fits the request")
    _, _, opening_units, stride_units = best
    return (
        H3_FRAME_ORIGIN + H3_FRAME_STRIDE * opening_units,
        tuple(H3_FRAME_STRIDE * value for value in stride_units),
    )


def _first_director_cut_bridge_frame(
    prompt: str,
    output_frames: int,
) -> int | None:
    """Choose one syntax-derived opening target across the first hard cut.

    The first version placed the opening seam before the first authored cut
    and copied untimed prose from the next shot into the opening prompt.  H3
    sometimes executed that supposedly read-only action early.  A stronger
    causal construction is to let the opening trajectory actually include the
    first cut, then stop halfway to that shot's next explicit event.  If the
    shot has no later clock, its following shot boundary is the stop marker.

    This function reads only ``[Shot N]`` and ``At HH:MM:SS`` syntax.  It does
    not inspect subjects, actions, language, or scene words.
    """

    original = str(prompt)
    description_match = _MAIN_DESCRIPTION_RE.search(original)
    narrative = (
        original[description_match.end() :]
        if description_match is not None
        else original
    )
    shot_starts = [match.start() for match in _SHOT_RE.finditer(narrative)]
    if len(shot_starts) < 2:
        return None
    chunks = [
        narrative[start : (
            shot_starts[index + 1]
            if index + 1 < len(shot_starts)
            else len(narrative)
        )]
        for index, start in enumerate(shot_starts)
    ]
    first_cut_match = _shot_opening_clock_match(chunks[1])
    if first_cut_match is None:
        return None
    first_cut = _time_seconds(first_cut_match)
    following = [
        _time_seconds(match)
        for match in _AT_TIME_RE.finditer(chunks[1])
        if _time_seconds(match) > first_cut + 1e-6
    ]
    if following:
        next_event = min(following)
    elif len(chunks) > 2:
        next_cut_match = _shot_opening_clock_match(chunks[2])
        if next_cut_match is None:
            return None
        next_event = _time_seconds(next_cut_match)
    else:
        next_event = int(output_frames) / FPS
    if next_event <= first_cut + 1e-6:
        return None
    target = align_h3_frames(round((first_cut + next_event) * 0.5 * FPS))
    if not int(round(first_cut * FPS)) < target < int(output_frames):
        return None
    return target


def _apply_director_first_cut_bridge(
    *,
    prompt: str,
    opening_frames: int,
    visible_strides: tuple[int, ...],
    output_frames: int,
    context_frames: int,
    maximum_frames: int,
) -> tuple[int, tuple[int, ...], bool]:
    """Move only the first seam so one real trajectory owns the first cut."""

    if not visible_strides:
        return opening_frames, visible_strides, False
    shot_frames = _prompt_event_frames(prompt, output_frames)
    if not shot_frames:
        return opening_frames, visible_strides, False
    first_cut = min(shot_frames)
    # No intervention is needed when the optimized opening already generates
    # the first cut.  Avoid making unrelated prompt timing move the windows.
    if int(opening_frames) > first_cut:
        return opening_frames, visible_strides, False
    target = _first_director_cut_bridge_frame(prompt, output_frames)
    if target is None or target <= int(opening_frames) or target > int(maximum_frames):
        return opening_frames, visible_strides, False
    second_boundary = int(opening_frames) + int(visible_strides[0])
    replacement_stride = second_boundary - target
    if replacement_stride <= 0 or replacement_stride % H3_FRAME_STRIDE:
        return opening_frames, visible_strides, False
    replacement_window = int(context_frames) + replacement_stride
    if not MIN_PHYSICAL_WINDOW_FRAMES <= replacement_window <= int(maximum_frames):
        return opening_frames, visible_strides, False
    return target, (replacement_stride, *visible_strides[1:]), True


def _apply_director_final_shot_lock(
    *,
    opening_frames: int,
    visible_strides: tuple[int, ...],
    output_frames: int,
    context_frames: int,
    shot_start_frames: tuple[int, ...],
    maximum_frames: int,
) -> tuple[int, tuple[int, ...], bool]:
    """Give a sufficiently long final shot one composition-pure tail window.

    A final authored shot has no following shot boundary to refresh its camera
    state. H3 can therefore obey the cut, hold the requested composition for
    a few seconds, and then fall back to an earlier coverage view at a dialogue
    or action cue. Textual prohibitions alone are weaker than that learned
    editing prior.

    When geometry permits, reserve the final minimum-size continuation as a
    *shot lock*. Its exact context lies wholly after the last authored cut, so
    both its read-only visual state and localized text belong to only the final
    shot. A minimum-size transition window immediately before it establishes
    the new composition. The construction is syntax-driven and fixed-budget.
    """

    if not shot_start_frames:
        return opening_frames, visible_strides, False
    final_shot_start = max(int(frame) for frame in shot_start_frames)
    minimum_visible = max(
        H3_FRAME_STRIDE,
        math.ceil(
            (MIN_PHYSICAL_WINDOW_FRAMES - int(context_frames))
            / H3_FRAME_STRIDE
        )
        * H3_FRAME_STRIDE,
    )
    tail_start = int(output_frames) - minimum_visible
    transition_start = tail_start - minimum_visible
    # The tail prefix must contain only the final shot. The transition window
    # begins before that shot so it owns the authored cut.
    if (
        tail_start - int(context_frames) < final_shot_start
        or transition_start >= final_shot_start
        or transition_start < MIN_PHYSICAL_WINDOW_FRAMES
    ):
        return opening_frames, visible_strides, False

    starts = [0, int(opening_frames)]
    cursor = int(opening_frames)
    for stride in visible_strides[:-1]:
        cursor += int(stride)
        starts.append(cursor)
    # Replace a late optimized boundary if it would leave a sub-minimum
    # transition before the shot-lock window.
    kept = [value for value in starts if value < transition_start]
    if not kept or kept[0] != 0:
        return opening_frames, visible_strides, False
    while len(kept) > 1 and transition_start - kept[-1] < minimum_visible:
        kept.pop()
    candidate_starts = kept + [transition_start, tail_start, int(output_frames)]
    candidate_visible = [
        right - left
        for left, right in zip(candidate_starts, candidate_starts[1:])
    ]
    if not candidate_visible:
        return opening_frames, visible_strides, False
    if not (
        MIN_PHYSICAL_WINDOW_FRAMES <= candidate_visible[0] <= int(maximum_frames)
    ):
        return opening_frames, visible_strides, False
    for visible in candidate_visible[1:]:
        physical = int(context_frames) + visible
        if not MIN_PHYSICAL_WINDOW_FRAMES <= physical <= int(maximum_frames):
            return opening_frames, visible_strides, False
    if any(value % H3_FRAME_STRIDE for value in candidate_visible[1:]):
        return opening_frames, visible_strides, False
    if (
        candidate_visible[0] < H3_FRAME_ORIGIN
        or (candidate_visible[0] - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE
    ):
        return opening_frames, visible_strides, False
    return candidate_visible[0], tuple(candidate_visible[1:]), True


def _retime_shot(
    text: str,
    context_start_seconds: float,
    *,
    shot_start_seconds: float,
    visible_start_seconds: float | None = None,
) -> str:
    already_visible_cutoff = (
        context_start_seconds
        if visible_start_seconds is None
        else float(visible_start_seconds)
    )
    if shot_start_seconds < already_visible_cutoff - 1e-6:
        # The shot began before this continuation's carried context.  Merely
        # clamping its old timestamp to zero tells H3 to perform the cut again,
        # which creates exactly the camera reset a latent continuation is
        # intended to avoid.  Mark the shot as already established and remove
        # only its first global timestamp/cut verb; the scene description and
        # all later timed events remain available to the model.
        updated = _AT_TIME_RE.sub("", text, count=1)
        updated = re.sub(
            r"\b(?:the\s+)?camera\s+cuts?(?:\s+back)?\s+to\b",
            "the established camera framing remains",
            updated,
            count=1,
            flags=re.IGNORECASE,
        )
        updated = re.sub(r"\]\s*,?\s*", "] ", updated, count=1)
        # The opening clock belongs to the already established shot and is
        # removed above.  Later clocks still describe real events inside the
        # active shot and must use this window's local timebase.  Leaving a
        # global ``At about 00:27`` in an eleven-second continuation makes the
        # event unreachable and weakens both dialogue and shot timing.
        updated = _AT_TIME_RE.sub(
            lambda match: (
                f"At {_format_local_time(_time_seconds(match) - context_start_seconds)}"
            ),
            updated,
        )
        marker = (
            "[Continuation already in progress at local 00:00.000; preserve "
            "the carried physical state and framing, and do not replay this "
            "shot's opening action.] "
        )
        shot_end = updated.find("]") + 1
        return updated[:shot_end] + " " + marker + updated[shot_end:].lstrip(" ,")

    def replace(match: re.Match[str]) -> str:
        return f"At {_format_local_time(_time_seconds(match) - context_start_seconds)}"

    return _AT_TIME_RE.sub(replace, text)


def _trim_completed_shot_actions(
    text: str,
    progress: float,
    *,
    keep_terminal_clause: bool = True,
) -> str:
    """Remove already completed clauses from a continued shot.

    A clean latent prefix tells H3 *what the world currently looks like*, but
    repeating every earlier sentence from a long shot can still make the next
    noisy suffix replay an entrance, camera cut, dialogue, or object
    interaction.  Conversely, deleting every positive description of the
    active shot leaves a future shot as the first strong visual instruction
    and can make that cut happen at the window boundary instead of its stated
    time.  H3 prompts normally put the current composition in the first clause
    and actions after it.  Preserve that first clause as a state-only anchor,
    discard the completed fraction of the chronological clauses, and leave at
    least the terminal clause.  This is deterministic and adds no
    language-model pass to the hot path.
    """

    amount = min(1.0, max(0.0, float(progress)))
    if amount <= 0.0:
        return text
    first_close = text.find("]")
    if first_close < 0:
        return text
    marker = "[Continuation already in progress"
    marker_start = text.find(marker, first_close + 1)
    if marker_start < 0:
        return text
    marker_close = text.find("]", marker_start)
    if marker_close < 0:
        return text
    prefix = text[: marker_close + 1].rstrip()
    clauses = [
        part.strip(" ,")
        for part in _SENTENCE_BREAK_RE.split(text[marker_close + 1 :].strip(" ,"))
        if part.strip(" ,")
    ]
    if not clauses:
        return text
    if os.environ.get("H3_LONG_RETAIN_SHOT_SETUP", "0") == "1":
        # Opt-in research ablation. Keep the authored setup, not the timed
        # event history. Its size does not grow with generated duration. A
        # generic first-sentence heuristic can otherwise erase the camera,
        # cast and layout whenever a style sentence precedes those details.
        # This is historical reference, NOT an assertion that opening prop
        # ownership, pose or presence is still current. Test that risk on film.
        setup = [part for part in clauses if not _DIALOGUE_SPAN_RE.search(part)]
        return " ".join((
            prefix,
            "[Authored shot setup reference: retain the described identity, "
            "wardrobe and shot constraints. This is not a new action or a "
            "request to restore the opening. Initial poses, presence, object "
            "locations and held objects are superseded by the exact carried "
            "physical state and the selected current events below.]",
            *setup,
            "[End of setup reference. Continue from the actual carried frame, "
            "not from the beginning of this description.]",
        ))
    state_anchor_guard = (
        "[Ongoing-shot state anchor: use the following first clause only to "
        "preserve the already visible composition and subjects; do not execute "
        "or repeat any action it describes.]"
    )
    if len(clauses) == 1:
        return " ".join((prefix, state_anchor_guard, clauses[0]))
    if not keep_terminal_clause:
        # In a declared single take, the opening location is not necessarily
        # the current location: the camera can walk through several connected
        # spaces without starting a new shot.  Keeping an arbitrary terminal
        # clause from the opening preamble made that old location a fresh
        # image-condition in every continuation.  The exact latent prefix and
        # the latest completed timed event are the causal state authorities;
        # retain only the style/shot clause here.
        return " ".join((prefix, state_anchor_guard, clauses[0]))
    # With no explicit timestamp inside a shot, H3 characteristically begins
    # the ordered actions early and spends the tail settling the result.  A
    # square-root clock models that front-loaded execution better than a
    # linear sentence clock: it prevents a continuation at (say) 36% of a
    # shot from asking H3 to repeat dialogue that the carried A/V prefix has
    # already completed, while explicit later ``[Shot] At ...`` events remain
    # independently scheduled and are never removed by this function.
    completed_fraction = math.sqrt(amount)
    completed = min(
        len(clauses) - 1,
        int(math.ceil(completed_fraction * len(clauses))),
    )
    if completed <= 0:
        return text
    guard = (
        "[The earlier clauses of this shot are already completed in the carried "
        "context; do not repeat them or change framing. Continue only with the "
        "remaining actions.]"
    )
    return " ".join((
        prefix,
        guard,
        state_anchor_guard,
        clauses[0],
        *clauses[completed:],
    ))


def _latest_completed_event_state_anchor(
    text: str,
    *,
    visible_start_seconds: float,
) -> str:
    """Return a structural current-state hint from the latest past event.

    No story words are interpreted.  Explicit timestamps already provide a
    causal order, so the final sentence of the latest completed non-dialogue
    event is the least stale text available to accompany the exact latent
    prefix.  The event is marked as completed and cannot authorize replay.
    Dialogue-bearing events are excluded because even a guarded copy can seed
    a second utterance in H3's joint audio branch.
    """

    value = str(text)
    opening_clock = _shot_opening_clock_match(value)
    clocks = list(_AT_TIME_RE.finditer(value))
    events = [
        match
        for match in clocks
        if opening_clock is None or match.start() != opening_clock.start()
    ]
    completed: list[str] = []
    for index, match in enumerate(events):
        if _time_seconds(match) >= float(visible_start_seconds) - 1e-6:
            continue
        stop = events[index + 1].start() if index + 1 < len(events) else len(value)
        event = value[match.start() : stop].strip(" ,")
        if _DIALOGUE_SPAN_RE.search(event):
            continue
        completed.append(event)
    if not completed:
        return ""
    event = _AT_TIME_RE.sub("", completed[-1], count=1).strip(" ,")
    clauses = [
        part.strip(" ,")
        for part in _SENTENCE_BREAK_RE.split(event)
        if part.strip(" ,")
    ]
    if not clauses:
        return ""
    return " ".join((
        "[Carried current-state evidence: the following timed event is already "
        "complete. Do not repeat its action, sound, speech, or camera motion; "
        "preserve only its resulting physical state as visible in the exact "
        "latent prefix.]",
        clauses[-1],
    ))


def _active_timed_event_continuation(
    text: str,
    *,
    context_start_seconds: float,
    visible_start_seconds: float,
    shot_stop_seconds: float,
) -> str:
    """Carry an event whose semantic interval crosses a window boundary.

    An H3 timestamp marks an event's *onset*, not proof that every following
    action has completed at that instant.  The event remains active until the
    next authored timestamp (or the end of its shot).  Previously, an event
    beginning just before a continuation seam vanished from the next local
    prompt and was replaced by one terminal state clause.  The exact prefix
    still showed walking or camera motion, but the text jumped directly to the
    next destination; H3 could satisfy that jump with an undeclared cut.

    If the authored onset predates the exact prefix, preserve the complete
    non-speech trajectory because the model can no longer see how it began.  If
    the onset is already inside the exact prefix *and* a semicolon explicitly
    separates its atomic onset from a later result, latch the onset as consumed
    exactly once and retain only the later syntactic units.  Without that
    high-confidence structural delimiter, conservatively retain the complete
    trajectory: removing an indivisible motion clause can erase its direction.
    No story word, object class, verb, or timestamp is special-cased.

    Dialogue-bearing units are always removed, so neither branch can authorize
    a repeated utterance.
    """

    value = str(text)
    opening_clock = _shot_opening_clock_match(value)
    clocks = list(_AT_TIME_RE.finditer(value))
    events = [
        match
        for match in clocks
        if opening_clock is None or match.start() != opening_clock.start()
    ]
    active: str | None = None
    active_start: float | None = None
    for index, match in enumerate(events):
        event_start = _time_seconds(match)
        event_stop = (
            _time_seconds(events[index + 1])
            if index + 1 < len(events)
            else float(shot_stop_seconds)
        )
        if (
            event_start < float(visible_start_seconds) - 1e-6
            and event_stop > float(visible_start_seconds) + 1e-6
        ):
            stop = events[index + 1].start() if index + 1 < len(events) else len(value)
            active = value[match.start() : stop].strip(" ,")
            active_start = event_start
    if not active:
        return ""
    active = _AT_TIME_RE.sub("", active, count=1).strip(" ,")
    units = [
        part.strip(" ,")
        for part in _EVENT_CONTINUATION_BREAK_RE.split(active)
        if part.strip(" ,")
    ]
    onset_is_in_exact_prefix = bool(
        active_start is not None
        and active_start >= float(context_start_seconds) - 1e-6
    )
    latched_clauses = [
        part
        for part in units[1:]
        if not _DIALOGUE_SPAN_RE.search(part)
    ]
    onset_has_explicit_result_delimiter = bool(
        units and units[0].endswith((";", "；")) and latched_clauses
    )
    state_first = os.environ.get("H3_LONG_STATE_FIRST_CONTINUATION", "0") == "1"
    if state_first and active.startswith("[Authored state interval through local "):
        onset_has_explicit_result_delimiter = False
    if (onset_is_in_exact_prefix or state_first) and onset_has_explicit_result_delimiter:
        if state_first and not onset_is_in_exact_prefix:
            return " ".join((
                "[Scheduled onset consumed by an earlier writable interval. "
                "Do not issue that onset again. Continue the actual motion "
                "at the carried boundary toward the authored result below; "
                "do not reset poses or teleport to achieve the result.]",
                *latched_clauses,
            ))
        return " ".join((
            "[Context-latched timed event: its authored onset is already "
            "visible inside the exact carried prefix and has been consumed "
            "once. Do not restart or independently restage it in the writable "
            "suffix. Continue only the motion visible at the prefix boundary "
            "and preserve its resulting physical state. Any following clauses "
            "are post-onset continuation or result constraints, not another "
            "onset.]",
            *latched_clauses,
        ))

    # Preserve the accepted pre-latch compiler byte-for-byte whenever the
    # confidence predicate does not fire.  Even semantically equivalent guard
    # wording or a broader clause splitter can perturb H3's visual trajectory.
    clauses = [
        part.strip(" ,")
        for part in _SENTENCE_BREAK_RE.split(active)
        if part.strip(" ,") and not _DIALOGUE_SPAN_RE.search(part)
    ]
    if not clauses:
        return ""
    return " ".join((
        "[Ongoing timed-event continuation: this event began in the exact "
        "carried prefix and is still underway. Continue its present trajectory "
        "and camera motion from that prefix; do not restart its beginning, cut "
        "to its destination, replay any sound or speech, or treat it as already "
        "finished.]",
        *clauses,
    ))


def _expand_shot_state_intervals(text: str, context_start_seconds: float) -> str:
    """Make standalone clock ranges first-class in-shot event boundaries.

    The old At-only parser attaches ``From A through B, ...`` to the preceding
    action forever, and leaves A/B on the global clock in a local clip. This
    opt-in syntax-only adapter separates that interval without guessing any
    action's duration or interpreting story nouns. Dialogue text is opaque.
    Mid-sentence ranges are deliberately not promoted to independent events.
    """
    dialogue_spans = [match.span() for match in _DIALOGUE_SPAN_RE.finditer(text)]

    def replace(match: re.Match[str]) -> str:
        if any(start <= match.start() < stop for start, stop in dialogue_spans):
            return match.group(0)
        before = text[:match.start()].rstrip()
        if before and before[-1] not in ".!?;；]":
            return match.group(0)
        start = _clock_text_seconds(match.group("start"))
        stop = _clock_text_seconds(match.group("stop"))
        if stop <= start:
            return match.group(0)
        return (
            f"At {_format_local_time(start)}, "
            f"[Authored state interval through local "
            f"{_format_local_time(stop - context_start_seconds)}]"
        )

    return _CLOCK_RANGE_RE.sub(replace, str(text))


def _localize_shot_events(
    text: str,
    *,
    context_start_seconds: float,
    visible_start_seconds: float,
    visible_stop_seconds: float,
    shot_start_seconds: float,
    shot_stop_seconds: float,
    single_take: bool = False,
) -> str:
    """Give every explicitly timed in-shot event one visible-window owner.

    Continuation windows overlap physically so their clean AV latents can
    carry state.  Their generated visible intervals do not overlap.  A timed
    dialogue or action therefore belongs only to the window whose visible
    interval contains its global clock.  Earlier windows must not anticipate
    it and later windows must not replay it.  This prevents one sentence near
    a boundary from being independently completed by two DiT trajectories.

    Untimed shot setup remains available as a state/action preamble.  Existing
    progress trimming is applied only to that preamble, so it cannot delete a
    future timed dialogue from its rightful window.
    """

    value = str(text)
    if os.environ.get("H3_LONG_LOCAL_STATE_INTERVALS", "0") == "1":
        value = _expand_shot_state_intervals(value, context_start_seconds)
    opening_clock = _shot_opening_clock_match(value)
    clocks = list(_AT_TIME_RE.finditer(value))
    timed_events = [
        match
        for match in clocks
        if opening_clock is None or match.start() != opening_clock.start()
    ]
    first_event_owned = bool(
        timed_events
        and visible_start_seconds - 1e-6
        <= _time_seconds(timed_events[0])
        < visible_stop_seconds - 1e-6
    )
    preamble_stop = timed_events[0].start() if timed_events else len(value)
    preamble = value[:preamble_stop].rstrip(" ,")
    if not first_event_owned:
        preamble = re.sub(
            r"(?:,\s*)?\b(?:and|then)\s*$",
            ".",
            preamble,
            count=1,
            flags=re.IGNORECASE,
        )
    localized_preamble = _retime_shot(
        preamble,
        context_start_seconds,
        shot_start_seconds=shot_start_seconds,
        visible_start_seconds=visible_start_seconds,
    )
    if shot_start_seconds < visible_start_seconds - 1e-6:
        span = max(1.0 / FPS, shot_stop_seconds - shot_start_seconds)
        progress = (visible_start_seconds - shot_start_seconds) / span
        localized_preamble = _trim_completed_shot_actions(
            localized_preamble,
            progress,
            keep_terminal_clause=not single_take,
        )

    state_first = os.environ.get("H3_LONG_STATE_FIRST_CONTINUATION", "0") == "1"
    active_event = (
        _active_timed_event_continuation(
            value,
            context_start_seconds=context_start_seconds,
            visible_start_seconds=visible_start_seconds,
            shot_stop_seconds=shot_stop_seconds,
        )
        if (single_take or state_first) and shot_start_seconds < visible_start_seconds - 1e-6
        else ""
    )
    completed_state = (
        _latest_completed_event_state_anchor(
            value,
            visible_start_seconds=visible_start_seconds,
        )
        if (
            (single_take or state_first)
            and shot_start_seconds < visible_start_seconds - 1e-6
            and not active_event
        )
        else ""
    )

    selected_events: list[str] = []
    for index, match in enumerate(timed_events):
        event_seconds = _time_seconds(match)
        if not (
            visible_start_seconds - 1e-6
            <= event_seconds
            < visible_stop_seconds - 1e-6
        ):
            continue
        event_stop = (
            timed_events[index + 1].start()
            if index + 1 < len(timed_events)
            else len(value)
        )
        event = value[match.start() : event_stop].strip(" ,")
        event = _AT_TIME_RE.sub(
            lambda item: (
                f"At {_format_local_time(_time_seconds(item) - context_start_seconds)}"
            ),
            event,
        )
        selected_events.append(event)

    if (
        single_take
        and state_first
        and "[Authored state interval through local " in active_event
    ):
        # An explicit current hold is a stronger composition authority than
        # the opening cast/setup (which can otherwise repopulate an empty
        # scene). Preserve setup for the next timed event, not this hold.
        # Syntax only: no names, objects, absence words or camera verbs tested.
        header_stop = preamble.find("]") + 1
        setup_clauses = _SENTENCE_BREAK_RE.split(preamble[header_stop:].strip())
        reference = setup_clauses[0] if setup_clauses else ""
        if selected_events and reference and not _DIALOGUE_SPAN_RE.search(reference):
            selected_events[0] += (
                " [Appearance and shot reference for this selected event only; "
                "not an instruction during the preceding hold:] " + reference
            )
        localized_preamble = (
            preamble[:header_stop] + " Continue the established shot and the "
            "current-state interval below."
        )
    elif state_first and not single_take and active_event:
        # Multi-shot continuations previously discarded the most recent timed
        # event altogether while preserving stale shot-opening prop states.
        # Within an already active shot, the exact prefix and latest event
        # now own the state; authored new shots still keep their full setup.
        header_stop = preamble.find("]") + 1
        localized_preamble = (
            preamble[:header_stop] + " Continue the established framing, people "
            "and scene from the exact carried frames, with the current event "
            "result below. Do not restore this shot's opening state."
        )

    localized = " ".join(
        part
        for part in (
            localized_preamble,
            active_event,
            completed_state,
            *selected_events,
        )
        if part.strip()
    )
    # A timestamp may legally appear after a shared subject ("She places the
    # cup and at 00:12 says ...").  Splitting on the clock used to delete that
    # grammatical subject and emit "At 00:12 says".  Keep the conjunction
    # only when this window owns that first event, then restore lower-case
    # ``at`` when joining the independently retimed pieces.
    return re.sub(
        r"\b(and|then)\s+At\s+",
        lambda match: f"{match.group(1)} at ",
        localized,
        flags=re.IGNORECASE,
    )


def _localize_reference_contract_prefix(prefix: str) -> str:
    """Remove untimed future-story leakage from a Ref2VA continuation.

    H3 Context-IR puts the whole-video synopsis and retention prose before
    ``detailed_description``.  Keeping those global narrative sentences in an
    eleven-second continuation can make an omitted future shot happen as soon
    as the current local action finishes.  Preserve the concrete subject/audio
    definitions and each declared relationship marker, but replace narrative
    prose with a content-free local contract.  The selected, retimed shots
    remain the sole temporal authority.
    """

    value = str(prefix)
    summary = _SUMMARY_SECTION_RE.search(value)
    retention = _RETENTION_SECTION_RE.search(value)
    description = _MAIN_DESCRIPTION_RE.search(value)
    if (
        summary is None
        or retention is None
        or description is None
        or not summary.start() < retention.start() < description.start()
    ):
        return value

    definitions = value[: summary.start()].rstrip()
    summary_body = value[summary.end() : retention.start()]
    retention_body = value[retention.end() : description.start()]
    description_prefix = value[description.start() :].strip()
    task_match = _TASK_TYPES_RE.search(summary_body)
    task_types = task_match.group(0) if task_match is not None else ""
    local_summary = (
        "summary: "
        + (f"{task_types} " if task_types else "")
        + "Generate only the window-local continuation described below. "
        "Keep the defined reference roles authoritative, do not anticipate "
        "omitted future shots, and do not replay completed events."
    )

    relation_lines: list[str] = []
    seen_labels: set[str] = set()
    for match in _REFERENCE_RELATION_RE.finditer(retention_body):
        label = match.group("label")
        normalized = label.casefold()
        if normalized in seen_labels:
            continue
        seen_labels.add(normalized)
        relation = match.group("relation").lower()
        if normalized.startswith("<audio"):
            detail = (
                "retain the defined audio relationship only for locally "
                "selected events and never replay completed audio"
            )
        else:
            detail = (
                "retain the defined relationship and attributes throughout "
                "this local interval"
            )
        relation_lines.append(f"{label}: {relation} - {detail}.")
    local_retention = "retention_analysis:"
    if relation_lines:
        local_retention += "\n" + "\n".join(relation_lines)
    else:
        local_retention += (
            " Preserve every reference role defined above throughout this "
            "local interval without anticipating omitted future shots."
        )
    return "\n\n".join((
        definitions,
        local_summary,
        local_retention,
        description_prefix,
    ))


def _localize_structured_tail(tail: str, *, segment_index: int) -> str:
    """Keep persistent ambience without replaying a global sound-event list.

    H3's authoring contract puts synchronized diegetic events in the shot body
    and uses ``overall_soundscape`` as a summary.  Reattaching every summary
    sentence to every local window made late effects (stage footsteps, a lamp
    click, a radio dial) available before their timeline event.  In structured
    mode the first sentence is the persistent acoustic bed; later event
    sentences remain governed by their selected shot text.  Free prompts are
    untouched.
    """

    value = str(tail).strip()
    if not value:
        return ""
    soundscape_match = _OVERALL_SOUNDSCAPE_RE.search(value)
    music_match = _NON_DIEGETIC_MUSIC_RE.search(value)
    parts: list[str] = []
    if soundscape_match is not None:
        body = soundscape_match.group("body").strip()
        sentences = [
            item.strip()
            for item in _SENTENCE_BREAK_RE.split(body)
            if item.strip()
        ]
        persistent = sentences[0] if sentences else body
        if persistent:
            suffix = (
                " Continue this established ambience without restarting it; "
                "only selected timeline events may introduce a new sound."
                if segment_index > 0
                else ""
            )
            parts.append(f"overall_soundscape: {persistent}{suffix}")
    if music_match is not None:
        music = music_match.group("body").strip()
        if music:
            if segment_index > 0 and music.casefold() not in {"n/a", "none"}:
                music = (
                    "Continue the already established non-diegetic music without "
                    f"restarting its phrase. {music}"
                )
            parts.append(f"non_diegetic_music: {music}")
    return "\n\n".join(parts)


def _structured_director_contract(
    *,
    context_start_seconds: float,
    visible_start_seconds: float,
    visible_stop_seconds: float,
    intervals: Iterable[tuple[str, float, float]],
) -> str:
    """Compile explicit shot/range syntax into one local execution contract.

    Story text remains opaque: this layer never decides what a person, action,
    object, or sound means.  It only binds each authored interval to H3's local
    clock and states the structural consequence of declaring a shot: a new
    hard cut is legal only at the next authored shot boundary.
    """

    normalized = [
        (str(label), float(start), float(stop))
        for label, start, stop in intervals
        if float(stop) > float(start)
    ]
    writable_start = max(0.0, visible_start_seconds - context_start_seconds)
    writable_stop = max(writable_start, visible_stop_seconds - context_start_seconds)
    cut_times = sorted({
        start - context_start_seconds
        for _, start, _ in normalized
        if start > 1e-6
        and visible_start_seconds - 1e-6 <= start < visible_stop_seconds - 1e-6
    })
    if cut_times:
        cut_rule = "The only new camera cut(s) permitted in the writable suffix are at local " + ", ".join(
            _format_local_time(value) for value in cut_times
        ) + "."
    else:
        cut_rule = "No new camera cut is permitted in the writable suffix."

    if os.environ.get("H3_LONG_CONCISE_LOCAL_PROMPT", "0") == "1":
        return " ".join((
            cut_rule if cut_times else "Hold the same continuous shot.",
            "Keep the carried framing and physical results; perform each new "
            "timed action once. No subtitles, captions or overlay text.",
        ))

    interval_lines: list[str] = []
    for label, start, stop in normalized:
        local_start = max(0.0, start - context_start_seconds)
        local_stop = min(writable_stop, max(local_start, stop - context_start_seconds))
        state = (
            "already established in carried context"
            if start < visible_start_seconds - 1e-6
            else "begins at its stated boundary"
        )
        interval_lines.append(
            f"- {label}: local {_format_local_time(local_start)}–"
            f"{_format_local_time(local_stop)}; {state}."
        )

    # H3 follows concrete audiovisual prose better than a second long page of
    # abstract policy.  The former contract added roughly 200 English words to
    # every local prompt and diluted the selected shot itself.  Keep the same
    # executable invariants in a compact prefix whose local clocks remain
    # auditable in receipts.
    return "\n".join((
        "structured_director_contract:",
        f"Writable suffix: local {_format_local_time(writable_start)}–"
        f"{_format_local_time(writable_stop)}.",
        cut_rule,
        "Continue the exact carried people, objects, environment, camera state, "
        "and physical results. Execute only the selected local events below, once "
        "and in order; never anticipate or replay an omitted or completed event.",
        "Do not render dialogue as subtitles or captions. Generate no credits, "
        "watermark, overlay text, or invented readable text; only explicitly "
        "described diegetic text may remain in the scene.",
        "A timestamp inside a shot changes only its stated action or sound, never "
        "the camera. Hold that shot continuously until its declared stop.",
        *interval_lines,
    ))


def _structured_local_speech_contract(parts: Iterable[str]) -> str:
    """Bind speech authority to literal H3 dialogue spans in this window."""

    count = sum(
        len(_DIALOGUE_SPAN_RE.findall(str(part)))
        for part in parts
    )
    if os.environ.get("H3_LONG_CONCISE_LOCAL_PROMPT", "0") == "1":
        return (
            "Speak only the literal dialogue below, once at its local time. "
            "Between those lines, only environmental and action sounds are audible."
            if count else "Only environmental and action sounds are audible; no speech."
        )
    if count == 0:
        authority = (
            "The writable suffix contains no authorized H3 dialogue event. "
            "Generate no human voice, words, whisper, narration, singing, or "
            "vocalization."
        )
    else:
        authority = (
            f"The writable suffix contains exactly {count} authorized H3 dialogue "
            "event(s). Speak only the literal text inside the selected dialogue "
            "tags, once at its local clock; generate no other voice."
        )
    return "\n".join((
        "local_speech_contract:",
        authority,
        "Never vocalize prompt metadata or camera directions.",
    ))


def _compact_local_prompt(value: str) -> str:
    """Opt-in rendering ablation; event selection and geometry stay intact."""
    if os.environ.get("H3_LONG_CONCISE_LOCAL_PROMPT", "0") != "1":
        return value
    fragments = re.split(r"(<d>.*?</d>)", value, flags=re.IGNORECASE | re.DOTALL)
    value = "".join(
        fragment if index % 2 else re.sub(
            r"\[(?:Continuation already in progress|Ongoing-shot state anchor|"
            r"Ongoing timed-event continuation|Carried current-state evidence|"
            r"Context-latched timed event)[^\]]*\]\s*", "", fragment,
        )
        for index, fragment in enumerate(fragments)
    )
    if value.startswith("Continue directly from the preceding generated video."):
        _, separator, body = value.partition("\n\n")
        if separator:
            value = (
                "Continue the same recording from the carried frames without "
                "restarting completed actions or dialogue. All times below use "
                "this local clip's clock.\n\n" + body
            )
    return value


def localize_h3_prompt(
    prompt: str,
    *,
    context_start_frame: int,
    visible_start_frame: int,
    visible_stop_frame: int,
    segment_index: int,
    timeline_stop_frame: int | None = None,
    structured_director: bool = False,
) -> str:
    """Build a window-local H3 prompt without requiring new user syntax.

    Ordinary prompts are passed through unchanged (with one continuation
    instruction on later windows).  Existing H3 ``[Shot N]`` + ``At MM:SS``
    prompts are deterministically windowed and their timestamps are shifted to
    the local H3 clock.  Soundscape/music sections remain attached to every
    window so audio character does not reset at a boundary.
    """

    original = str(prompt)
    director_active = bool(
        structured_director and has_structured_timeline(original)
    )
    tail_match = _TAIL_RE.search(original)
    if tail_match is None:
        body, tail = original, ""
    else:
        body, tail = original[: tail_match.start()], original[tail_match.start() :]
    localized_tail = (
        _localize_structured_tail(tail, segment_index=segment_index)
        if director_active
        else tail
    )
    # Full-reference prompts legitimately cite ``[Shot N]`` in summary and
    # retention-analysis metadata.  Those citations are not timeline entries
    # and must never be selected, retimed, or used to trim the actual story.
    # Restrict temporal parsing to H3's narrative field while retaining all
    # preceding reference-contract sections verbatim in every local prompt.
    description_match = _MAIN_DESCRIPTION_RE.search(body)
    narrative_start = description_match.end() if description_match is not None else 0
    narrative = body[narrative_start:]
    shot_starts = [
        narrative_start + match.start()
        for match in _SHOT_RE.finditer(narrative)
    ]
    directive = ""
    if segment_index > 0:
        directive = (
            "Continue directly from the preceding generated video. Preserve the same "
            "people, voices, wardrobe, objects, environment, camera direction and current "
            "physical state. Do not replay an opening action or earlier dialogue. "
            f"This local clip begins at global {_format_local_time(context_start_frame / FPS)}; "
            "all following local timestamps are measured from this clip's first context frame.\n\n"
        )
    if not shot_starts:
        ranges = [
            (narrative_start + start, narrative_start + stop, global_start, global_stop)
            for start, stop, global_start, global_stop in _timeline_ranges(narrative)
        ]
        if not ranges:
            return directive + original
        prefix = body[: ranges[0][0]].rstrip()
        if segment_index > 0 or director_active:
            prefix = _localize_reference_contract_prefix(prefix)
        context_start_seconds = context_start_frame / FPS
        visible_start_seconds = visible_start_frame / FPS
        visible_stop_seconds = visible_stop_frame / FPS
        selected_ranges: list[str] = []
        director_intervals: list[tuple[str, float, float]] = []
        for index, (start, match_stop, global_start, global_stop) in enumerate(ranges):
            chunk_stop = ranges[index + 1][0] if index + 1 < len(ranges) else len(body)
            if global_start >= visible_stop_seconds or global_stop <= visible_start_seconds:
                continue
            local_start = max(0.0, global_start - context_start_seconds)
            local_stop = max(local_start, global_stop - context_start_seconds)
            selected_ranges.append(
                f"[Local interval {_format_local_time(local_start)}–"
                f"{_format_local_time(local_stop)}]"
                + body[match_stop:chunk_stop]
            )
            director_intervals.append((
                f"Interval {index + 1}",
                global_start,
                global_stop,
            ))
        if not selected_ranges:
            selected_ranges.append(
                "[Shot 1] Continue the current action naturally through this interval "
                "without resetting the scene."
            )
        director_contract = (
            _structured_director_contract(
                context_start_seconds=context_start_seconds,
                visible_start_seconds=visible_start_seconds,
                visible_stop_seconds=visible_stop_seconds,
                intervals=director_intervals,
            )
            if director_active and director_intervals
            else ""
        )
        speech_contract = (
            _structured_local_speech_contract(selected_ranges)
            if director_active and _DIALOGUE_SPAN_RE.search(original)
            else ""
        )
        localized = "\n\n".join(
            part
            for part in (
                prefix,
                director_contract,
                speech_contract,
                *selected_ranges,
            )
            if part.strip()
        )
        tail_suffix = f"\n\n{localized_tail}" if localized_tail else ""
        return _compact_local_prompt(directive + localized + tail_suffix) if director_active else directive + localized + tail_suffix

    prefix = body[: shot_starts[0]].rstrip()
    if segment_index > 0 or director_active:
        prefix = _localize_reference_contract_prefix(prefix)
    chunks: list[str] = []
    for position, start in enumerate(shot_starts):
        stop = shot_starts[position + 1] if position + 1 < len(shot_starts) else len(body)
        chunks.append(body[start:stop].strip())
    single_take_director = bool(director_active and len(chunks) == 1)
    starts_seconds: list[float] = []
    previous = 0.0
    for index, chunk in enumerate(chunks):
        match = _shot_opening_clock_match(chunk)
        if match is not None:
            previous = _time_seconds(match)
        elif index == 0:
            previous = 0.0
        starts_seconds.append(previous)

    context_start_seconds = context_start_frame / FPS
    visible_stop_seconds = visible_stop_frame / FPS
    visible_start_seconds = visible_start_frame / FPS
    timeline_stop_seconds = (
        int(timeline_stop_frame) / FPS
        if timeline_stop_frame is not None
        else visible_stop_seconds
    )
    selected: list[str] = []
    director_intervals: list[tuple[str, float, float]] = []
    for index, chunk in enumerate(chunks):
        shot_start = starts_seconds[index]
        shot_stop = (
            starts_seconds[index + 1]
            if index + 1 < len(starts_seconds)
            else float("inf")
        )
        if shot_start < visible_stop_seconds and shot_stop > visible_start_seconds:
            effective_stop = (
                shot_stop if math.isfinite(shot_stop) else timeline_stop_seconds
            )
            label_match = re.match(r"\[Shot\s+(\d+)\]", chunk, re.IGNORECASE)
            label = (
                f"Shot {label_match.group(1)}"
                if label_match is not None
                else f"Shot {index + 1}"
            )
            director_intervals.append((label, shot_start, effective_stop))
            # In director mode, an outgoing shot may occupy a few writable
            # frames before an authored cut because H3's 17-frame grid cannot
            # land on every user timestamp.  Repeating that outgoing shot's
            # semantic description gives it equal text authority to the new
            # shot and can make the model return to the old composition at a
            # later dialogue cue.  The exact clean latent prefix already owns
            # its visual/physical state, so carry it opaquely until the cut and
            # expire it there.  A shot that spans the complete local window
            # still keeps its semantic anchor and remaining events.
            outgoing_before_cut = bool(
                director_active
                and shot_start < visible_start_seconds - 1e-6
                and effective_stop < visible_stop_seconds - 1e-6
            )
            if outgoing_before_cut:
                local_stop = max(0.0, effective_stop - context_start_seconds)
                selected.append(
                    f"[Carried interval until local {_format_local_time(local_stop)}] "
                    "Continue only the already-visible framing and physical state from "
                    "the exact latent prefix; do not replay any described event. At this "
                    "boundary the carried framing expires and must not return later in "
                    "the window."
                )
                continue
            retimed = _localize_shot_events(
                chunk,
                context_start_seconds=context_start_seconds,
                visible_start_seconds=visible_start_seconds,
                visible_stop_seconds=visible_stop_seconds,
                shot_start_seconds=shot_start,
                shot_stop_seconds=effective_stop,
                single_take=single_take_director,
            )
            selected.append(retimed)
    if not selected:
        selected.append(
            "[Shot 1] Continue the current action naturally through this interval "
            "without resetting the scene."
        )

    director_contract = (
        _structured_director_contract(
            context_start_seconds=context_start_seconds,
            visible_start_seconds=visible_start_seconds,
            visible_stop_seconds=visible_stop_seconds,
            intervals=director_intervals,
        )
        if director_active and director_intervals
        else ""
    )
    speech_contract = (
        _structured_local_speech_contract(selected)
        if director_active and _DIALOGUE_SPAN_RE.search(original)
        else ""
    )
    localized = "\n\n".join(
        part
        for part in (
            prefix,
            director_contract,
            speech_contract,
            *selected,
        )
        if part.strip()
    )
    tail_suffix = f"\n\n{localized_tail}" if localized_tail else ""
    return _compact_local_prompt(directive + localized + tail_suffix) if director_active else directive + localized + tail_suffix


def plan_long_horizon(
    *,
    requested_duration_seconds: float,
    prompt: str,
    seed: int,
    maximum_opening_frames: int = MAX_NATIVE_WINDOW_FRAMES,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    visible_stride_frames: int | None = None,
    structured_director: bool = False,
) -> LongHorizonPlan:
    """Plan a bounded-window request while keeping the final H3 temporal grid exact."""

    requested = float(requested_duration_seconds)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested duration must be positive and finite")
    maximum = int(maximum_opening_frames)
    context = int(context_frames)
    values_to_validate = [("maximum opening", maximum), ("context", context)]
    for name, value in values_to_validate:
        if value < H3_FRAME_ORIGIN or (value - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
            raise ValueError(f"{name} frames must satisfy H3's 5 + 17*k grid")

    output_frames = align_h3_frames(round(requested * FPS))
    director_active = bool(
        structured_director and has_structured_timeline(prompt)
    )
    description_match = _MAIN_DESCRIPTION_RE.search(str(prompt))
    narrative = (
        str(prompt)[description_match.end() :]
        if description_match is not None
        else str(prompt)
    )
    single_take_director = bool(
        director_active and sum(1 for _ in _SHOT_RE.finditer(narrative)) == 1
    )
    if (
        single_take_director
        and output_frames > maximum
        and context == DEFAULT_CONTEXT_FRAMES
    ):
        context = SINGLE_TAKE_CONTEXT_FRAMES
    if output_frames <= maximum:
        opening_frames = output_frames
        visible_strides: tuple[int, ...] = ()
    elif visible_stride_frames is None:
        shot_start_frames = (
            _single_take_event_frames(prompt, output_frames)
            if single_take_director
            else _prompt_event_frames(prompt, output_frames)
        )
        opening_frames, visible_strides = _optimize_window_frames(
            output_frames=output_frames,
            context_frames=context,
            event_frames=shot_start_frames,
            maximum_frames=maximum,
            structured_director=director_active,
        )
        if director_active and not single_take_director:
            opening_frames, visible_strides, _ = _apply_director_first_cut_bridge(
                prompt=prompt,
                opening_frames=opening_frames,
                visible_strides=visible_strides,
                output_frames=output_frames,
                context_frames=context,
                maximum_frames=maximum,
            )
    else:
        stride = int(visible_stride_frames)
        if stride <= 0 or stride % H3_FRAME_STRIDE:
            raise ValueError("visible stride must be a positive multiple of 17 frames")
        window_frames = context + stride
        if (window_frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
            raise ValueError("context + visible stride must be an H3-aligned window")
        if window_frames > maximum:
            raise ValueError("continuation window exceeds the native window limit")
        continuation_count = int(math.ceil((output_frames - maximum) / stride))
        opening_frames = output_frames - continuation_count * stride
        if opening_frames < MIN_PHYSICAL_WINDOW_FRAMES:
            raise ValueError(
                "native opening window is too short for the requested long-video geometry"
            )
        visible_strides = (stride,) * continuation_count
    segments: list[LongHorizonSegment] = []
    explicit_dialogue_contract = bool(_DIALOGUE_SPAN_RE.search(prompt))
    dialogue_event_frames = (
        _dialogue_event_frames(prompt, output_frames)
        if director_active and explicit_dialogue_contract
        else ()
    )
    visible_cursor = 0
    for index in range(len(visible_strides) + 1):
        if index == 0:
            frames = opening_frames
            local_context = 0
            visible = opening_frames
            context_start = 0
        else:
            visible = visible_strides[index - 1]
            frames = context + visible
            local_context = context
            context_start = visible_cursor - context
        visible_start = visible_cursor
        visible_stop = visible_start + visible
        segment_prompt = localize_h3_prompt(
            prompt,
            context_start_frame=context_start,
            visible_start_frame=visible_start,
            visible_stop_frame=visible_stop,
            segment_index=index,
            timeline_stop_frame=output_frames,
            structured_director=director_active,
        )
        authorized_dialogue_count = (
            len(_DIALOGUE_SPAN_RE.findall(segment_prompt))
            if director_active and explicit_dialogue_contract
            else None
        )
        authorized_dialogue_frames = tuple(
            frame
            for frame in dialogue_event_frames
            if visible_start <= frame < visible_stop
        )
        long_term_voice_authority = (
            authorized_dialogue_count is None
            or authorized_dialogue_count > 0
        )
        segments.append(
            LongHorizonSegment(
                index=index,
                window_frames=frames,
                context_frames=local_context,
                global_context_start_frame=context_start,
                visible_start_frame=visible_start,
                visible_frames=visible,
                seed=_derived_seed(seed, index),
                prompt=segment_prompt,
                transition="opening" if index == 0 else "continue",
                visual_memory_floor_frame=(
                    context_start
                    if single_take_director and index > 0
                    else None
                ),
                visual_memory_include_canonical=(
                    single_take_director and index > 0
                ),
                preserve_latest_visual=single_take_director,
                authorized_dialogue_count=authorized_dialogue_count,
                authorized_dialogue_frames=authorized_dialogue_frames,
                reference_audio_active=long_term_voice_authority,
                audio_memory_active=long_term_voice_authority,
            )
        )
        visible_cursor = visible_stop
    if visible_cursor != output_frames:
        raise RuntimeError("long-horizon plan does not cover its output timeline")
    return LongHorizonPlan(
        requested_duration_seconds=requested,
        output_frames=output_frames,
        actual_duration_seconds=output_frames / FPS,
        segments=tuple(segments),
        context_frames=max(segment.context_frames for segment in segments),
        mechanism="masked_joint_av_prefix_v1",
        planning_policy=(
            "structured_director_single_take_dual_timescale_isolated_latch_v9_56f"
            if single_take_director
            else "structured_director_first_cut_bridge_causal_state_v7"
            if director_active
            else "event_aware_continuous_cost_v1"
        ),
        structured_director=director_active,
    )


def prepare_masked_av_prefix(
    noise_video: Any,
    noise_audio: Any,
    source_video: Any,
    source_audio: Any,
    *,
    context_frames: int = DEFAULT_CONTEXT_FRAMES,
    video_prefix_frames: int | None = None,
    video_prefix_from_source_end: bool = False,
    audio_bridge_ticks: int = 0,
) -> tuple[Any, Any, Any, Any]:
    """Prepare aligned exact anchors plus writable hidden AV overlap bands.

    A partial video prefix is the *leading* part of the source's complete
    context interval. The trailing part of that same interval stays writable
    and replaces the matching predecessor tail during assembly. Selecting the
    newest ``video_t`` source tokens would shift history earlier in target time
    and silently corrupt the handoff. Audio independently protects the leading
    part of its context.
    """

    protected_video_frames = (
        int(context_frames)
        if video_prefix_frames is None
        else int(video_prefix_frames)
    )
    if protected_video_frames > int(context_frames):
        raise ValueError("video prefix must fit inside the continuation context")
    video_context_t = video_latent_frames(int(context_frames))
    video_t = (
        0 if protected_video_frames == 0 else
        video_latent_frames(protected_video_frames)
    )
    audio_t = audio_latent_frames(int(context_frames))
    bridge = int(audio_bridge_ticks)
    if bridge < 0 or bridge > audio_t:
        raise ValueError("audio bridge must fit inside the continuation context")
    if source_video.ndim != 5 or noise_video.ndim != 5:
        raise ValueError("continuation video latents must be five-dimensional")
    if source_audio.ndim != 4 or noise_audio.ndim != 4:
        raise ValueError("continuation audio latents must be four-dimensional")
    source_video_required = (
        video_t if video_prefix_from_source_end else video_context_t
    )
    if (
        (video_t and source_video.shape[2] < source_video_required)
        or noise_video.shape[2] <= video_context_t
    ):
        raise ValueError("continuation video context does not fit the source and target")
    if source_audio.shape[-1] < audio_t or noise_audio.shape[-1] <= audio_t:
        raise ValueError("continuation audio prefix does not fit the target")
    if source_video.shape[:2] != noise_video.shape[:2] or source_video.shape[3:] != noise_video.shape[3:]:
        raise ValueError("continuation video geometry does not match the target")
    if source_audio.shape[:3] != noise_audio.shape[:3]:
        raise ValueError("continuation audio geometry does not match the target")

    video = noise_video.clone()
    audio = noise_audio.clone()
    video_prefix = source_video[:, :, :0].to(video).contiguous()
    if video_t:
        if video_prefix_from_source_end:
            source_prefix = source_video[:, :, -video_t:]
        else:
            source_video_context = source_video[:, :, -video_context_t:]
            source_prefix = source_video_context[:, :, :video_t]
        video_prefix = source_prefix.to(video).contiguous()
    audio_context = source_audio[..., -audio_t:].to(audio).contiguous()
    audio_prefix = audio_context[..., : audio_t - bridge].contiguous()
    video[:, :, :video_t].copy_(video_prefix)
    audio[..., : audio_t - bridge].copy_(audio_prefix)
    return video, audio, video_prefix, audio_prefix


def restore_masked_av_prefix_(
    video: Any,
    audio: Any,
    video_prefix: Any,
    audio_prefix: Any,
) -> tuple[Any, Any]:
    """Restore protected prefix values after one numerical solver update."""

    video[:, :, : video_prefix.shape[2]].copy_(video_prefix)
    audio[..., : audio_prefix.shape[-1]].copy_(audio_prefix)
    return video, audio


def _resample_audio_ticks(audio: Any, target_ticks: int) -> Any:
    """Align one visible audio piece to the cumulative 24 fps output clock.

    H3 audio runs at 40 Hz.  Rounding every physical window independently can
    accumulate a full latent tick at an internal seam.  Resampling only the
    visible suffix by that rounding delta keeps its endpoints and distributes
    the sub-frame correction instead of deleting one tick at the join.
    """

    target = int(target_ticks)
    source = int(audio.shape[-1])
    if target <= 0 or source <= 0:
        raise ValueError("audio pieces must contain positive latent ticks")
    if source == target:
        return audio
    import torch.nn.functional as F

    shape = audio.shape
    resized = F.interpolate(
        audio.reshape(-1, 1, source).float(),
        size=target,
        mode="linear",
        align_corners=True,
    )
    return resized.reshape(*shape[:-1], target).to(audio.dtype).contiguous()


def _video_repaint_tokens(context_frames: int, repaint_frames: int) -> int:
    """Return the latent suffix replaced by a same-time hidden repaint."""

    context = int(context_frames)
    repaint = int(repaint_frames)
    if repaint == 0:
        return 0
    protected = context - repaint
    if (
        repaint < 0
        or protected < H3_FRAME_ORIGIN
        or (context - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE
        or (protected - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE
    ):
        raise ValueError(
            "video repaint must leave an H3-grid protected prefix inside context"
        )
    return video_latent_frames(context) - video_latent_frames(protected)


def continuation_retry_seed(seed: int, attempt: int) -> int:
    """Return a stable alternate seed for one rejected continuation window."""

    index = int(attempt)
    if index < 0:
        raise ValueError("continuation retry attempt must be non-negative")
    return int(seed) if index == 0 else _derived_seed(int(seed), index)


def evaluate_video_repaint_overlap_files(
    previous_path: Path,
    incoming_path: Path,
    *,
    context_frames: int,
    repaint_frames: int,
    relative_rms_max: float = CONTINUATION_OVERLAP_LOW_FREQUENCY_RELATIVE_RMS_MAX,
    cosine_min: float = CONTINUATION_OVERLAP_LOW_FREQUENCY_COSINE_MIN,
) -> dict[str, Any]:
    """Detect whether a continuation repaint chose the predecessor's trajectory.

    Both tensors describe the same hidden timeline positions.  Pooling only
    the spatial axes makes the score sensitive to camera/layout divergence and
    insensitive to harmless high-frequency texture variation.  The final
    decision uses a wide calibrated margin: accepted V13 joins peaked at 0.132
    low-frequency relative RMS, while its failed frame-525 join reached 0.676.
    """

    import torch
    import torch.nn.functional as F

    context_t = video_latent_frames(int(context_frames))
    repaint_t = _video_repaint_tokens(int(context_frames), int(repaint_frames))
    if repaint_t < 2:
        raise ValueError("continuation overlap gate requires at least two repaint tokens")
    previous = torch.load(previous_path, map_location="cpu", weights_only=True)
    incoming = torch.load(incoming_path, map_location="cpu", weights_only=True)
    old_video = previous.get("video")
    new_video = incoming.get("video")
    if not isinstance(old_video, torch.Tensor) or old_video.ndim != 5:
        raise ValueError("previous continuation checkpoint has invalid video latent")
    if not isinstance(new_video, torch.Tensor) or new_video.ndim != 5:
        raise ValueError("incoming continuation checkpoint has invalid video latent")
    if old_video.shape[:2] != new_video.shape[:2] or old_video.shape[3:] != new_video.shape[3:]:
        raise ValueError("continuation overlap checkpoint geometry changed")
    if old_video.shape[2] < context_t or new_video.shape[2] <= context_t:
        raise ValueError("continuation overlap does not fit checkpoint timelines")
    old_tail = old_video[:, :, -repaint_t:].float()
    new_tail = new_video[:, :, context_t - repaint_t : context_t].float()
    height, width = map(int, old_tail.shape[-2:])
    pooled_size = (max(1, height // 4), max(1, width // 4))
    old_low = F.adaptive_avg_pool2d(
        old_tail.permute(0, 2, 1, 3, 4).reshape(-1, old_tail.shape[1], height, width),
        pooled_size,
    ).reshape(old_tail.shape[0], repaint_t, old_tail.shape[1], *pooled_size)
    new_low = F.adaptive_avg_pool2d(
        new_tail.permute(0, 2, 1, 3, 4).reshape(-1, new_tail.shape[1], height, width),
        pooled_size,
    ).reshape(new_tail.shape[0], repaint_t, new_tail.shape[1], *pooled_size)
    reduce_dims = (0, 2, 3, 4)
    old_energy = old_low.square().mean(dim=reduce_dims).sqrt().clamp_min(1.0e-8)
    difference = (old_low - new_low).square().mean(dim=reduce_dims).sqrt()
    relative_rms = difference / old_energy
    dot = (old_low * new_low).sum(dim=reduce_dims)
    cosine = dot / (
        old_low.square().sum(dim=reduce_dims).sqrt()
        * new_low.square().sum(dim=reduce_dims).sqrt()
    ).clamp_min(1.0e-8)
    finite = bool(
        torch.isfinite(relative_rms).all().item()
        and torch.isfinite(cosine).all().item()
    )
    maximum = float(relative_rms.max().item()) if finite else float("inf")
    minimum_cosine = float(cosine.min().item()) if finite else -1.0
    accepted = bool(
        finite
        and maximum <= float(relative_rms_max)
        and minimum_cosine >= float(cosine_min)
    )
    return {
        "policy": "same_time_low_frequency_latent_agreement_v1",
        "accepted": accepted,
        "repaint_video_latent_tokens": repaint_t,
        "spatial_pool_size": list(pooled_size),
        "low_frequency_relative_rms": [float(value) for value in relative_rms.tolist()],
        "low_frequency_cosine": [float(value) for value in cosine.tolist()],
        "maximum_low_frequency_relative_rms": maximum,
        "minimum_low_frequency_cosine": minimum_cosine,
        "relative_rms_max": float(relative_rms_max),
        "cosine_min": float(cosine_min),
        "finite": finite,
    }


def evaluate_visible_video_trajectory_file(
    incoming_path: Path,
    *,
    context_frames: int,
    relative_rms_min: float = CONTINUATION_TRAJECTORY_RELATIVE_RMS_MIN,
    local_outlier_ratio_min: float = (
        CONTINUATION_TRAJECTORY_LOCAL_OUTLIER_RATIO_MIN
    ),
) -> dict[str, Any]:
    """Detect an isolated camera/layout jump inside a continuation candidate.

    H3's Video-VAE emits five temporal latent slices for every 17 decoded
    frames. Adjacent five-slice groups therefore describe consecutive points
    on one physical trajectory. Fast motion can have a large group delta, so
    an absolute threshold alone is not useful. A generated hard cut is both
    large and isolated relative to the immediately preceding and following
    deltas. The gate combines those two conditions and never inspects text or
    decoded RGB frames.

    The context boundary itself is excluded because the same-time repaint
    agreement gate owns that decision. Returned frame positions use the
    candidate's local video clock.
    """

    import torch
    import torch.nn.functional as F

    document = torch.load(incoming_path, map_location="cpu", weights_only=True)
    video = document.get("video")
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("continuation candidate has invalid video latent")
    temporal_tokens = int(video.shape[2])
    if temporal_tokens < 12 or (temporal_tokens - 2) % 5:
        raise ValueError("continuation candidate is off H3's temporal latent grid")

    context_t = video_latent_frames(int(context_frames))
    if context_t >= temporal_tokens:
        raise ValueError("continuation context consumes the candidate timeline")
    context_group = (context_t - 2) // 5
    group_count = (temporal_tokens - 2) // 5

    low_source = video.float()
    height, width = map(int, low_source.shape[-2:])
    pooled_size = (max(1, height // 4), max(1, width // 4))
    low = F.adaptive_avg_pool2d(
        low_source.permute(0, 2, 1, 3, 4).reshape(
            -1,
            low_source.shape[1],
            height,
            width,
        ),
        pooled_size,
    ).reshape(
        low_source.shape[0],
        temporal_tokens,
        low_source.shape[1],
        *pooled_size,
    )

    relative_rms: list[float] = []
    cosine: list[float] = []
    finite = True
    for current_group in range(1, group_count):
        previous = low[
            :, 2 + 5 * (current_group - 1) : 2 + 5 * current_group
        ]
        current = low[
            :, 2 + 5 * current_group : 2 + 5 * (current_group + 1)
        ]
        previous_energy = previous.square().mean().sqrt().clamp_min(1.0e-8)
        delta = (current - previous).square().mean().sqrt()
        relative = delta / previous_energy
        similarity = (previous * current).sum() / (
            previous.square().sum().sqrt()
            * current.square().sum().sqrt()
        ).clamp_min(1.0e-8)
        item_finite = bool(
            torch.isfinite(relative).item()
            and torch.isfinite(similarity).item()
        )
        finite = finite and item_finite
        relative_rms.append(
            float(relative.item()) if item_finite else float("inf")
        )
        cosine.append(float(similarity.item()) if item_finite else -1.0)

    transitions: list[dict[str, Any]] = []
    # relative_rms[k - 1] is the transition into group k. Start one full
    # group after the context boundary; the boundary transition is evaluated
    # by the repaint agreement gate instead.
    for current_group in range(context_group + 1, group_count):
        transition_index = current_group - 1
        relative = relative_rms[transition_index]
        neighbors = []
        if transition_index > 0:
            neighbors.append(relative_rms[transition_index - 1])
        if transition_index + 1 < len(relative_rms):
            neighbors.append(relative_rms[transition_index + 1])
        # At the last group there is no following delta. Use one additional
        # predecessor so a terminal isolated jump remains observable without
        # changing the ordinary two-sided definition.
        if len(neighbors) == 1 and transition_index > 1:
            neighbors.append(relative_rms[transition_index - 2])
        local_baseline = (
            sum(neighbors) / len(neighbors) if neighbors else float("inf")
        )
        local_ratio = relative / max(local_baseline, 1.0e-8)
        joint_risk = min(
            relative / float(relative_rms_min),
            local_ratio / float(local_outlier_ratio_min),
        )
        unauthorized_cut = bool(
            math.isfinite(relative)
            and math.isfinite(local_ratio)
            and relative >= float(relative_rms_min)
            and local_ratio >= float(local_outlier_ratio_min)
        )
        transition_frame = H3_FRAME_ORIGIN + H3_FRAME_STRIDE * current_group
        transitions.append({
            "video_frame": transition_frame,
            "visible_frame": transition_frame - int(context_frames),
            "low_frequency_relative_rms": relative,
            "low_frequency_cosine": cosine[transition_index],
            "neighbor_relative_rms": [float(value) for value in neighbors],
            "local_relative_rms_baseline": float(local_baseline),
            "local_outlier_ratio": float(local_ratio),
            "joint_risk": float(joint_risk),
            "unauthorized_cut": unauthorized_cut,
        })

    rejected_frames = [
        int(item["video_frame"])
        for item in transitions
        if item["unauthorized_cut"]
    ]
    maximum_joint_risk = max(
        (float(item["joint_risk"]) for item in transitions),
        default=0.0,
    )
    return {
        "policy": "visible_low_frequency_temporal_outlier_v1",
        "accepted": bool(finite and not rejected_frames),
        "context_frames": int(context_frames),
        "spatial_pool_size": list(pooled_size),
        "evaluated_transition_count": len(transitions),
        "relative_rms_min": float(relative_rms_min),
        "local_outlier_ratio_min": float(local_outlier_ratio_min),
        "maximum_joint_risk": maximum_joint_risk,
        "rejected_transition_video_frames": rejected_frames,
        "transitions": transitions,
        "finite": finite,
    }


def stitch_clean_av_segments(
    documents: Iterable[dict[str, Any]],
    context_frames: Iterable[int],
    audio_bridge_ticks: Iterable[int] | None = None,
    *,
    video_repaint_frames: Iterable[int] | None = None,
    leading_preroll_frames: int = 0,
) -> tuple[Any, Any, int]:
    """Assemble clean segments with video repaint and audio overlap-save.

    Each visible audio suffix is phase-aligned to the cumulative 24 fps clock.
    This prevents independent per-window rounding from moving an internal AV
    seam by one 40 Hz latent tick.  A requested audio bridge is a denoising
    burn-in interval only: it conditions the incoming visible suffix but is
    discarded at assembly. Mixing it into the already accepted predecessor
    would average two independently completed Audio-VAE trajectories and can
    decode as a long tape/rewind artifact.

    Video repaint has different semantics: its incoming latent suffix was
    generated immediately after an exact protected anchor at the same timeline
    positions. Once the agreement gate accepts it as the same trajectory, it
    replaces the corresponding predecessor tail and remains as the physical
    bridge to the incoming visible suffix.
    """

    document_items = list(documents)
    context_items = [int(value) for value in context_frames]
    bridge_items = (
        [0 for _ in document_items]
        if audio_bridge_ticks is None
        else [int(value) for value in audio_bridge_ticks]
    )
    repaint_items = (
        [0 for _ in document_items]
        if video_repaint_frames is None
        else [int(value) for value in video_repaint_frames]
    )
    if not document_items:
        raise ValueError("at least one clean AV segment is required")
    if not (
        len(document_items)
        == len(context_items)
        == len(bridge_items)
        == len(repaint_items)
    ):
        raise ValueError(
            "clean segments, contexts, audio bridges and video repaints must align"
        )
    rows = list(zip(document_items, context_items, bridge_items, repaint_items))
    import torch

    leading_preroll = int(leading_preroll_frames)
    if leading_preroll < 0 or leading_preroll % H3_FRAME_STRIDE:
        raise ValueError("leading preroll must contain whole H3 temporal units")

    video_parts: list[Any] = []
    audio_parts: list[Any] = []
    total_frames = 0
    audio_ticks = 0
    for index, (document, context, bridge, repaint) in enumerate(rows):
        video = document.get("video")
        audio = document.get("audio")
        frames = int(document.get("frames", 0))
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("clean segment has an invalid video latent")
        if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
            raise ValueError("clean segment has an invalid audio latent")
        context = int(context)
        if index == 0:
            if context or bridge or repaint:
                raise ValueError(
                    "opening segment cannot have continuation context/bridge/repaint"
                )
            visible_opening_frames = frames - leading_preroll
            if (
                visible_opening_frames < H3_FRAME_ORIGIN
                or (visible_opening_frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE
            ):
                raise ValueError("leading preroll leaves an invalid opening frame grid")
            visible_video_t = video_latent_frames(visible_opening_frames)
            visible_audio_t = audio_latent_frames(visible_opening_frames)
            video_part = video[:, :, int(video.shape[2]) - visible_video_t:]
            audio_part = audio[..., int(audio.shape[-1]) - visible_audio_t:]
            total_frames = visible_opening_frames
        else:
            video_t = video_latent_frames(context)
            repaint_t = _video_repaint_tokens(context, repaint)
            audio_t = audio_latent_frames(context)
            if bridge < 0 or bridge > audio_t:
                raise ValueError("audio bridge must fit inside continuation context")
            if repaint_t:
                remaining = repaint_t
                while remaining:
                    if not video_parts:
                        raise ValueError("video repaint exceeds assembled history")
                    prior = video_parts[-1]
                    available = int(prior.shape[2])
                    if available <= remaining:
                        video_parts.pop()
                        remaining -= available
                    else:
                        video_parts[-1] = prior[:, :, :-remaining]
                        remaining = 0
            video_part = video[:, :, video_t - repaint_t:]
            audio_part = audio[..., audio_t:]
            if bridge and audio_ticks < bridge:
                raise ValueError("audio bridge exceeds assembled history")
            total_frames += frames - context
        required_ticks = audio_latent_frames(total_frames) - audio_ticks
        video_parts.append(video_part)
        audio_part = _resample_audio_ticks(audio_part, required_ticks)
        audio_parts.append(audio_part)
        audio_ticks += int(audio_part.shape[-1])
    stitched_video = torch.cat(video_parts, dim=2)
    stitched_audio = torch.cat(audio_parts, dim=-1)
    expected_audio = audio_latent_frames(total_frames)
    if stitched_audio.shape[-1] != expected_audio:
        raise RuntimeError("stitched audio latent missed the cumulative global clock")
    expected_video = video_latent_frames(total_frames)
    if stitched_video.shape[2] != expected_video:
        raise RuntimeError("stitched video latent is off the exact H3 temporal grid")
    return stitched_video, stitched_audio, total_frames


def stitch_clean_av_segment_files(
    paths: Iterable[Path],
    context_frames: Iterable[int],
    *,
    expected_frames: int,
    audio_bridge_ticks: Iterable[int] | None = None,
    video_repaint_frames: Iterable[int] | None = None,
    leading_preroll_frames: int = 0,
) -> tuple[Any, Any, int, Any]:
    """Stream clean segment files into one preallocated joint-AV latent.

    The in-memory helper above is useful for small callers and tests.  A long
    public request, however, must not retain every window tensor and then
    allocate a second full-size ``torch.cat`` result.  This file-backed path
    holds only the final tensor plus one source window at a time.  Copying into
    the preallocated destination is element-identical to concatenation.
    """

    path_items = [Path(path) for path in paths]
    context_items = [int(context) for context in context_frames]
    bridge_items = (
        [0 for _ in path_items]
        if audio_bridge_ticks is None
        else [int(value) for value in audio_bridge_ticks]
    )
    repaint_items = (
        [0 for _ in path_items]
        if video_repaint_frames is None
        else [int(value) for value in video_repaint_frames]
    )
    if not (
        len(path_items)
        == len(context_items)
        == len(bridge_items)
        == len(repaint_items)
    ):
        raise ValueError(
            "clean segment paths, context clocks, audio bridges and video repaints "
            "must have equal length"
        )
    rows = list(zip(path_items, context_items, bridge_items, repaint_items))
    if not rows:
        raise ValueError("at least one clean AV segment is required")

    import torch

    expected_frames = int(expected_frames)
    leading_preroll = int(leading_preroll_frames)
    if leading_preroll < 0 or leading_preroll % H3_FRAME_STRIDE:
        raise ValueError("leading preroll must contain whole H3 temporal units")
    expected_video = video_latent_frames(expected_frames)
    expected_audio = audio_latent_frames(expected_frames)
    stitched_video = None
    stitched_audio = None
    video_offset = 0
    audio_offset = 0
    total_frames = 0
    engine_metadata: Any = None

    for index, (path, context, bridge, repaint) in enumerate(rows):
        document = torch.load(path, map_location="cpu", weights_only=True)
        video = document.get("video")
        audio = document.get("audio")
        frames = int(document.get("frames", 0))
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("clean segment has an invalid video latent")
        if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
            raise ValueError("clean segment has an invalid audio latent")
        if index == 0:
            if context or bridge or repaint:
                raise ValueError(
                    "opening segment cannot have continuation context/bridge/repaint"
                )
            stitched_video = torch.empty(
                (*video.shape[:2], expected_video, *video.shape[3:]),
                dtype=video.dtype,
                device=video.device,
            )
            stitched_audio = torch.empty(
                (*audio.shape[:3], expected_audio),
                dtype=audio.dtype,
                device=audio.device,
            )
            engine_metadata = document.get("engine")
            visible_opening_frames = frames - leading_preroll
            if (
                visible_opening_frames < H3_FRAME_ORIGIN
                or (visible_opening_frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE
            ):
                raise ValueError("leading preroll leaves an invalid opening frame grid")
            visible_video_t = video_latent_frames(visible_opening_frames)
            visible_audio_t = audio_latent_frames(visible_opening_frames)
            video_part = video[:, :, int(video.shape[2]) - visible_video_t:]
            audio_part = audio[..., int(audio.shape[-1]) - visible_audio_t:]
            total_frames = visible_opening_frames
        else:
            if stitched_video is None or stitched_audio is None:
                raise RuntimeError("long-horizon stitch destination was not initialized")
            if (
                video.shape[:2] != stitched_video.shape[:2]
                or video.shape[3:] != stitched_video.shape[3:]
                or video.dtype != stitched_video.dtype
            ):
                raise ValueError("clean segment video geometry changed between windows")
            if audio.shape[:3] != stitched_audio.shape[:3] or audio.dtype != stitched_audio.dtype:
                raise ValueError("clean segment audio geometry changed between windows")
            context_video_t = video_latent_frames(context)
            repaint_t = _video_repaint_tokens(context, repaint)
            if repaint_t:
                if video_offset < repaint_t:
                    raise ValueError("video repaint exceeds assembled history")
                video_offset -= repaint_t
            video_part = video[:, :, context_video_t - repaint_t:]
            audio_t = audio_latent_frames(context)
            if bridge < 0 or bridge > audio_t:
                raise ValueError("audio bridge must fit inside continuation context")
            if bridge and audio_offset < bridge:
                raise ValueError("audio bridge exceeds assembled history")
            audio_part = audio[..., audio_t:]
            total_frames += frames - context

        if stitched_video is None or stitched_audio is None:
            raise RuntimeError("long-horizon stitch destination was not initialized")
        next_video_offset = video_offset + video_part.shape[2]
        if next_video_offset > expected_video:
            raise RuntimeError("clean segment video exceeds the requested output clock")
        stitched_video[:, :, video_offset:next_video_offset].copy_(video_part)
        video_offset = next_video_offset

        if audio_part.shape[-1] <= 0:
            raise ValueError("clean segment has an empty audio latent")
        required_audio_end = audio_latent_frames(total_frames)
        required_audio_ticks = required_audio_end - audio_offset
        audio_part = _resample_audio_ticks(audio_part, required_audio_ticks)
        stitched_audio[..., audio_offset:required_audio_end].copy_(audio_part)
        audio_offset = required_audio_end

        del document, video, audio, video_part, audio_part

    if total_frames != expected_frames:
        raise RuntimeError("clean segment clocks do not match the requested output")
    if video_offset != expected_video:
        raise RuntimeError("stitched video latent is off the exact H3 temporal grid")
    if audio_offset != expected_audio:
        raise RuntimeError("stitched audio latent missed the cumulative global clock")
    return stitched_video, stitched_audio, total_frames, engine_metadata


def stitch_audio_segment_files(
    paths: Iterable[Path],
    context_frames: Iterable[int],
    *,
    expected_frames: int,
    audio_key: str = "audio",
    audio_bridge_ticks: Iterable[int] | None = None,
    leading_preroll_frames: int = 0,
) -> Any:
    """Stream one named audio latent from window files into one timeline.

    Global SelfLift needs two audio representations at its resolution fork:
    the clean x0 estimate used as RES history and the exact noisy sampler
    state that must continue through the remaining formal steps.  The AV
    stitcher above owns the former.  This audio-only companion avoids copying
    the complete video track a second time merely to assemble the latter.
    """

    path_items = [Path(path) for path in paths]
    context_items = [int(context) for context in context_frames]
    bridge_items = (
        [0 for _ in path_items]
        if audio_bridge_ticks is None
        else [int(value) for value in audio_bridge_ticks]
    )
    if not (len(path_items) == len(context_items) == len(bridge_items)):
        raise ValueError(
            "audio segment paths, context clocks and audio bridges must "
            "have equal length"
        )
    rows = list(zip(path_items, context_items, bridge_items))
    if not rows:
        raise ValueError("at least one audio segment is required")
    if not str(audio_key).strip():
        raise ValueError("audio key cannot be empty")

    import torch

    expected_frames = int(expected_frames)
    leading_preroll = int(leading_preroll_frames)
    if leading_preroll < 0 or leading_preroll % H3_FRAME_STRIDE:
        raise ValueError("leading preroll must contain whole H3 temporal units")
    expected_audio = audio_latent_frames(expected_frames)
    stitched_audio = None
    audio_offset = 0
    total_frames = 0

    for index, (path, context, bridge) in enumerate(rows):
        document = torch.load(path, map_location="cpu", weights_only=True)
        audio = document.get(audio_key)
        frames = int(document.get("frames", 0))
        if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
            raise ValueError(
                f"audio segment has an invalid {audio_key!r} latent"
            )
        if index == 0:
            if context or bridge:
                raise ValueError(
                    "opening audio segment cannot have continuation context/bridge"
                )
            stitched_audio = torch.empty(
                (*audio.shape[:3], expected_audio),
                dtype=audio.dtype,
                device=audio.device,
            )
            visible_opening_frames = frames - leading_preroll
            if (
                visible_opening_frames < H3_FRAME_ORIGIN
                or (visible_opening_frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE
            ):
                raise ValueError(
                    "leading preroll leaves an invalid opening frame grid"
                )
            visible_audio_t = audio_latent_frames(visible_opening_frames)
            audio_part = audio[..., int(audio.shape[-1]) - visible_audio_t :]
            total_frames = visible_opening_frames
        else:
            if stitched_audio is None:
                raise RuntimeError(
                    "audio stitch destination was not initialized"
                )
            if (
                audio.shape[:3] != stitched_audio.shape[:3]
                or audio.dtype != stitched_audio.dtype
            ):
                raise ValueError(
                    "audio segment geometry changed between windows"
                )
            audio_t = audio_latent_frames(context)
            if bridge < 0 or bridge > audio_t:
                raise ValueError("audio bridge must fit inside continuation context")
            if bridge and audio_offset < bridge:
                raise ValueError("audio bridge exceeds assembled history")
            audio_part = audio[..., audio_t:]
            total_frames += frames - context

        if stitched_audio is None:
            raise RuntimeError("audio stitch destination was not initialized")
        if audio_part.shape[-1] <= 0:
            raise ValueError("audio segment has an empty latent")
        required_audio_end = audio_latent_frames(total_frames)
        required_audio_ticks = required_audio_end - audio_offset
        audio_part = _resample_audio_ticks(audio_part, required_audio_ticks)
        stitched_audio[..., audio_offset:required_audio_end].copy_(audio_part)
        audio_offset = required_audio_end
        del document, audio, audio_part

    if total_frames != expected_frames:
        raise RuntimeError("audio segment clocks do not match the requested output")
    if audio_offset != expected_audio:
        raise RuntimeError("stitched audio latent missed the cumulative global clock")
    return stitched_audio


__all__ = [
    "CONTINUATION_CANDIDATE_MAX_ATTEMPTS",
    "CONTINUATION_OVERLAP_LOW_FREQUENCY_COSINE_MIN",
    "CONTINUATION_OVERLAP_LOW_FREQUENCY_RELATIVE_RMS_MAX",
    "CONTINUATION_TRAJECTORY_LOCAL_OUTLIER_RATIO_MIN",
    "CONTINUATION_TRAJECTORY_RELATIVE_RMS_MIN",
    "DEFAULT_AUDIO_BRIDGE_CONTEXT_FRAMES",
    "DEFAULT_AUDIO_BRIDGE_TICKS",
    "DEFAULT_CONTEXT_FRAMES",
    "DEFAULT_VISIBLE_STRIDE_FRAMES",
    "MAX_CONTINUATION_CONTEXT_FRAMES",
    "SINGLE_TAKE_CONTEXT_FRAMES",
    "LongHorizonPlan",
    "LongHorizonSegment",
    "ShotDecodeGroup",
    "plan_shot_decode_groups",
    "align_h3_frames",
    "audio_latent_frames",
    "continuation_retry_seed",
    "evaluate_video_repaint_overlap_files",
    "evaluate_visible_video_trajectory_file",
    "has_structured_timeline",
    "localize_h3_prompt",
    "plan_long_horizon",
    "prepare_masked_av_prefix",
    "restore_masked_av_prefix_",
    "stitch_clean_av_segment_files",
    "stitch_audio_segment_files",
    "stitch_clean_av_segments",
    "scale_h3_prompt_timeline",
    "video_latent_frames",
]
