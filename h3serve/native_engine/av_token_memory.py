"""Fixed-budget multimodal latent memory for causal H3 continuation.

The memory is not a text summary.  It retains a small deterministic coreset of
clean Video-VAE frames and one dialogue-gated, short Audio-VAE voice excerpt
from completed windows.  A later H3 window projects these latents into its
ordinary packed conditioning tokens, so active context remains bounded even
when the completed output is long.  The audio excerpt is deliberately kept in
the model's native temporal geometry.  Runtime authority decides whether an
untrained FL2VA experiment receives only a high-noise bootstrap or the trained
Ref2VA path receives a persistent voice reference.  Free and silent windows
never see inferred voice-bearing memory.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .long_horizon import (
    AUDIO_LATENT_HZ,
    FPS,
    H3_FRAME_STRIDE,
    audio_latent_frames,
    video_latent_frames,
)


SCHEMA_VERSION = 5
METHOD = "bounded_visual_coreset_dialogue_gated_short_voice_bootstrap_v6"
DEFAULT_VIDEO_SLOTS = 6
DEFAULT_AUDIO_SLOTS = 1
DEFAULT_AUDIO_BLOCK_TICKS = 20
DEFAULT_AUDIO_RECENT_GUARD_TICKS = audio_latent_frames(39)
VOICE_FOCUS_PREROLL_TICKS = AUDIO_LATENT_HZ
VOICE_FOCUS_POSTROLL_TICKS = 3 * AUDIO_LATENT_HZ
VISUAL_MEMORY_RESOLUTIONS = ("360p", "480p", "720p", "original")
VISUAL_MEMORY_SHORT_LATENTS = {"360p": 22, "480p": 30, "720p": 46}


def _entry(position: int, latent: torch.Tensor) -> dict[str, Any]:
    return {
        "position": int(position),
        "latent": latent.detach().to("cpu", dtype=torch.float32).contiguous(),
    }


def _normalized_embedding(latent: torch.Tensor, *, audio: bool) -> torch.Tensor:
    value = latent.detach().float().cpu()
    if audio:
        # Preserve channel/timbre statistics while making different block
        # lengths comparable.  Mean, RMS and mean absolute first difference
        # separate voiced, ambient and transient regions without ASR rules.
        mean = value.mean(dim=-1)
        rms = value.square().mean(dim=-1).sqrt()
        delta = (
            value[..., 1:] - value[..., :-1]
            if value.shape[-1] > 1
            else torch.zeros_like(value)
        )
        feature = torch.cat(
            (mean.flatten(), rms.flatten(), delta.abs().mean(dim=-1).flatten())
        )
    else:
        if value.ndim != 5 or value.shape[2] != 1:
            raise ValueError("visual memory entries must be single latent frames")
        pooled = F.adaptive_avg_pool2d(value[:, :, 0], (4, 4))
        feature = pooled.flatten()
    return F.normalize(feature, dim=0, eps=1e-8)


def _deduplicate(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_position: dict[int, dict[str, Any]] = {}
    for item in entries:
        position = int(item["position"])
        latent = item["latent"]
        if not isinstance(latent, torch.Tensor):
            raise ValueError("token-memory entry latent must be a tensor")
        by_position[position] = _entry(position, latent)
    return [by_position[position] for position in sorted(by_position)]


def _select_coreset(
    entries: Iterable[dict[str, Any]],
    capacity: int,
    *,
    audio: bool,
    preserve_latest: bool = False,
) -> list[dict[str, Any]]:
    candidates = _deduplicate(entries)
    limit = int(capacity)
    if limit < 0:
        raise ValueError("token-memory capacity cannot be negative")
    if limit == 0:
        return []
    if len(candidates) <= limit:
        selected_entries = candidates
    else:
        embeddings = [
            _normalized_embedding(item["latent"], audio=audio)
            for item in candidates
        ]
        if audio:
            # A silent opening block is a poor long-term voice carrier.  Seed
            # selection from the strongest clean block, then maximize acoustic
            # diversity.  Temporal structure is retained because collapsing it
            # to a mean moves the condition off H3's learned audio manifold.
            seed = max(
                range(len(candidates)),
                key=lambda index: (
                    float(candidates[index]["latent"].float().square().mean()),
                    -int(candidates[index]["position"]),
                ),
            )
        else:
            # The first visual state is an immutable canonical anchor.
            # Remaining slots cover novel appearances/scenes through
            # farthest-point sampling.
            seed = 0
        selected = [seed]
        if (
            preserve_latest
            and not audio
            and limit > 1
            and len(candidates) - 1 != seed
        ):
            # A diversity-only visual coreset can discard every frame near the
            # current continuation boundary.  Structured single-take routing
            # then falls back to old locations and rebuilds the scene a few
            # seconds after an otherwise exact seam.  Reserve one slot for the
            # newest clean state before filling the remaining fixed budget by
            # diversity.  The legacy/free policy remains byte-for-byte.
            selected.append(len(candidates) - 1)
        remaining = set(range(len(candidates))) - set(selected)
        while remaining and len(selected) < limit:
            best_index = min(remaining)
            best_distance = -1.0
            for index in sorted(remaining):
                distance = min(
                    1.0 - float(torch.dot(embeddings[index], embeddings[chosen]))
                    for chosen in selected
                )
                if distance > best_distance + 1e-12:
                    best_distance = distance
                    best_index = index
            selected.append(best_index)
            remaining.remove(best_index)
        selected_entries = [candidates[index] for index in selected]
    selected_entries = sorted(
        selected_entries,
        key=lambda item: int(item["position"]),
    )
    return selected_entries


def _nearest_even(value: float) -> int:
    return max(2, int(round(float(value) / 2.0)) * 2)


def _resize_visual_memory(
    video: torch.Tensor,
    resolution: str,
) -> torch.Tensor:
    """Reduce clean visual-memory latents without changing their time axis."""

    mode = str(resolution).strip().lower()
    if mode == "original":
        return video
    if mode not in VISUAL_MEMORY_SHORT_LATENTS:
        raise ValueError("unsupported visual-memory resolution")
    source_h, source_w = int(video.shape[-2]), int(video.shape[-1])
    source_short = min(source_h, source_w)
    target_short = min(source_short, VISUAL_MEMORY_SHORT_LATENTS[mode])
    if target_short == source_short:
        return video
    scale = target_short / source_short
    target_h = min(source_h, _nearest_even(source_h * scale))
    target_w = min(source_w, _nearest_even(source_w * scale))
    target_h -= target_h % 2
    target_w -= target_w % 2
    batch, channels, frames = video.shape[:3]
    flattened = video.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, source_h, source_w
    )
    resized = F.interpolate(
        flattened.float(),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(
        batch, frames, channels, target_h, target_w
    ).permute(0, 2, 1, 3, 4).contiguous()


def empty_av_token_memory(
    *,
    video_slots: int = DEFAULT_VIDEO_SLOTS,
    audio_slots: int = DEFAULT_AUDIO_SLOTS,
    audio_block_ticks: int = DEFAULT_AUDIO_BLOCK_TICKS,
    visual_resolution: str = "original",
) -> dict[str, Any]:
    resolution = str(visual_resolution).strip().lower()
    if min(int(video_slots), int(audio_slots)) < 0 or int(audio_block_ticks) <= 0:
        raise ValueError("AV token-memory capacities cannot be negative")
    if resolution not in VISUAL_MEMORY_RESOLUTIONS:
        raise ValueError("unsupported visual-memory resolution")
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "video_slots": int(video_slots),
        "audio_slots": int(audio_slots),
        "audio_block_ticks": int(audio_block_ticks),
        "visual_resolution": resolution,
        "video_entries": [],
        "audio_entries": [],
        "updates": 0,
    }


def validate_av_token_memory(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("AV token memory must be a dictionary")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported AV token-memory schema")
    if document.get("method") != METHOD:
        raise ValueError("unsupported AV token-memory method")
    video_slots = int(document.get("video_slots", 0))
    audio_slots = int(document.get("audio_slots", 0))
    audio_block_ticks = int(document.get("audio_block_ticks", 0))
    visual_resolution = str(
        document.get("visual_resolution", "original")
    ).strip().lower()
    if min(video_slots, audio_slots) < 0 or audio_block_ticks <= 0:
        raise ValueError("AV token-memory capacities cannot be negative")
    if visual_resolution not in VISUAL_MEMORY_RESOLUTIONS:
        raise ValueError("unsupported visual-memory resolution")
    video_entries = _deduplicate(document.get("video_entries", ()))
    audio_entries = _deduplicate(document.get("audio_entries", ()))
    if len(video_entries) > video_slots or len(audio_entries) > audio_slots:
        raise ValueError("AV token memory exceeds its fixed capacity")
    for item in video_entries:
        latent = item["latent"]
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != 1:
            raise ValueError("visual token memory has invalid latent geometry")
    for item in audio_entries:
        latent = item["latent"]
        if latent.ndim != 4 or latent.shape[0] != 1 or latent.shape[2] != 2:
            raise ValueError("audio token memory has invalid latent geometry")
        if latent.shape[-1] != audio_block_ticks:
            raise ValueError(
                "audio token memory must contain fixed-width native excerpts"
            )
    return {
        **document,
        "video_slots": video_slots,
        "audio_slots": audio_slots,
        "audio_block_ticks": audio_block_ticks,
        "visual_resolution": visual_resolution,
        "video_entries": video_entries,
        "audio_entries": audio_entries,
        "updates": int(document.get("updates", 0)),
    }


def update_av_token_memory(
    memory: dict[str, Any] | None,
    clean_segment: dict[str, Any],
    *,
    context_frames: int,
    visible_start_frame: int,
    visible_frames: int,
    preserve_latest_visual: bool = False,
    collect_audio: bool = True,
    audio_focus_frames: Iterable[int] | None = None,
    leading_preroll_frames: int = 0,
) -> dict[str, Any]:
    """Update a bounded latent coreset from one completed continuation window."""

    current = validate_av_token_memory(
        empty_av_token_memory() if memory is None else memory
    )
    video = clean_segment.get("video")
    audio = clean_segment.get("audio")
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("clean segment has invalid video latent")
    if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
        raise ValueError("clean segment has invalid audio latent")
    if int(context_frames) < 0 or int(visible_frames) <= 0:
        raise ValueError("visible token-memory interval is invalid")
    leading_preroll = int(leading_preroll_frames)
    if leading_preroll < 0 or leading_preroll % H3_FRAME_STRIDE:
        raise ValueError("leading preroll must contain whole H3 temporal units")
    if leading_preroll and int(context_frames):
        raise ValueError("opening preroll and continuation context cannot coexist")
    if leading_preroll:
        visible_video_t = video_latent_frames(int(visible_frames))
        visible_audio_t = audio_latent_frames(int(visible_frames))
        video_prefix = int(video.shape[2]) - visible_video_t
        audio_prefix = int(audio.shape[-1]) - visible_audio_t
        if min(video_prefix, audio_prefix) < 0:
            raise ValueError("leading preroll exceeds the clean segment")
    else:
        video_prefix = 0 if not context_frames else video_latent_frames(context_frames)
        audio_prefix = 0 if not context_frames else audio_latent_frames(context_frames)
    visible_video = video[:, :, video_prefix:].detach().cpu()
    visible_audio = audio[..., audio_prefix:].detach().cpu()
    if visible_video.shape[2] <= 0 or visible_audio.shape[-1] <= 0:
        raise ValueError("clean segment has no visible AV suffix")

    visible_video = _resize_visual_memory(
        visible_video, current["visual_resolution"]
    )
    video_candidates = list(current["video_entries"])
    video_count = int(visible_video.shape[2])
    for index in range(video_count if current["video_slots"] else 0):
        position = int(visible_start_frame) + round(
            index * max(0, int(visible_frames) - 1) / max(1, video_count - 1)
        )
        video_candidates.append(
            _entry(position, visible_video[:, :, index:index + 1])
        )

    audio_candidates = list(current["audio_entries"])
    block = int(current["audio_block_ticks"])
    global_audio_start = round(
        int(visible_start_frame) / FPS * AUDIO_LATENT_HZ
    )
    audio_count = int(visible_audio.shape[-1])
    # The exact recent AV prefix already supplies high-bandwidth boundary
    # context.  Re-injecting the same tail as a reference-audio row creates a
    # second temporal interpretation of it and can encourage echo/replay at a
    # seam.  Long memory therefore sees only audio older than one short-term
    # context interval.
    eligible_audio_count = max(
        0,
        audio_count - DEFAULT_AUDIO_RECENT_GUARD_TICKS,
    )
    focus_frames = tuple(
        int(frame) for frame in (() if audio_focus_frames is None else audio_focus_frames)
    )
    if any(
        frame < int(visible_start_frame)
        or frame >= int(visible_start_frame) + int(visible_frames)
        for frame in focus_frames
    ):
        raise ValueError("audio focus frame is outside the visible segment")
    audio_selection_policy = current.get(
        "audio_selection_policy",
        "not_collected",
    )
    if current["audio_slots"] and collect_audio and eligible_audio_count >= block:
        terminal = max(0, eligible_audio_count - block)
        if focus_frames:
            starts_set: set[int] = set()
            for frame in focus_frames:
                focus_tick = round(
                    (frame - int(visible_start_frame)) / FPS * AUDIO_LATENT_HZ
                )
                # A dialogue clock inside the discarded recent-prefix guard
                # cannot authorize an older ambient block as a voice sample.
                # Wait for a later eligible dialogue instead of silently
                # filling the write-once voice bank with room tone.
                if not 0 <= focus_tick < eligible_audio_count:
                    continue
                # ``start`` indexes the beginning of an excerpt, so include
                # every block that can overlap the trusted clock neighborhood
                # rather than requiring its beginning to fall inside it.
                low = max(
                    0,
                    focus_tick - VOICE_FOCUS_PREROLL_TICKS - block + 1,
                )
                high = min(
                    terminal,
                    focus_tick + VOICE_FOCUS_POSTROLL_TICKS,
                )
                if high < low:
                    continue
                local = list(range(low, high + 1, block))
                if local and local[-1] != high:
                    local.append(high)
                starts_set.update(local)
            starts = sorted(starts_set)
            audio_selection_policy = (
                "structured_dialogue_clock_local_energy_v1"
                if starts
                else "structured_dialogue_clock_not_eligible_v1"
            )
        else:
            starts = list(range(0, terminal + 1, block))
            if starts[-1] != terminal:
                starts.append(terminal)
            audio_selection_policy = "segment_global_energy_fallback_v1"
        for start in starts:
            stop = min(eligible_audio_count, start + block)
            audio_candidates.append(
                _entry(global_audio_start + start, visible_audio[..., start:stop])
            )

    updated = {
        **current,
        "video_entries": _select_coreset(
            video_candidates,
            current["video_slots"],
            audio=False,
            preserve_latest=preserve_latest_visual,
        ),
        "audio_entries": _select_coreset(
            audio_candidates, current["audio_slots"], audio=True
        ),
        "visual_selection_policy": (
            "canonical_latest_diverse_v3"
            if preserve_latest_visual
            else "canonical_diverse_v2"
        ),
        "audio_selection_policy": audio_selection_policy,
        "updates": int(current["updates"]) + 1,
    }
    return validate_av_token_memory(updated)


def memory_conditioning(
    document: dict[str, Any],
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[tuple[int, int, int], ...],
    tuple[str, ...],
    tuple[torch.Tensor, ...],
    tuple[int, ...],
]:
    memory = validate_av_token_memory(document)
    video = tuple(item["latent"] for item in memory["video_entries"])
    audio = tuple(item["latent"] for item in memory["audio_entries"])
    shapes = tuple(
        (int(item.shape[2]), int(item.shape[3]), int(item.shape[4]))
        for item in video
    )
    kinds = tuple("image" for _ in video)
    audio_frames = tuple(int(item.shape[-1]) for item in audio)
    return video, shapes, kinds, audio, audio_frames


def route_visual_memory_interval(
    document: dict[str, Any],
    *,
    minimum_position: int,
    maximum_position: int | None = None,
    include_canonical: bool = False,
    progressive_layout_state: bool = False,
    layout_minimum_position: int | None = None,
    layout_maximum_position: int | None = None,
    novel_camera_cut: bool = False,
    novel_camera_layout_probe: bool = False,
) -> dict[str, Any]:
    """Route visual memory to active state plus an optional identity anchor.

    Global visual diversity is useful for identity and scene recall, but it is
    the wrong authority after a structured shot has established a particular
    camera composition.  This read-only view retains only visual latent frames
    generated inside the active shot.  A declared continuous take can also
    retain exactly one canonical opening frame so a face or wardrobe can
    reappear after a temporary occlusion; all other obsolete scene views stay
    excluded.  Dialogue-gated audio prototypes remain global because their
    authority is routed separately from camera state.
    """

    memory = validate_av_token_memory(document)
    lower = int(minimum_position)
    upper = None if maximum_position is None else int(maximum_position)
    if upper is not None and upper <= lower:
        raise ValueError("visual memory interval must have positive extent")
    routed_video = [
        item
        for item in memory["video_entries"]
        if int(item["position"]) >= lower
        and (upper is None or int(item["position"]) < upper)
    ]
    if novel_camera_layout_probe and not novel_camera_cut:
        raise ValueError("novel-camera layout probe requires a novel camera")
    # A novel camera has no matching historical projection.  Its normal route
    # withholds every old full-frame row.  The bounded probe route exposes one
    # canonical scene observation for a single middle denoise step, then
    # returns to text-only convergence; the route metadata below keeps that
    # observation separate from target-camera state.
    if novel_camera_cut:
        routed_video = []
    explicit_layout = (
        layout_minimum_position is not None
        or layout_maximum_position is not None
    )
    if explicit_layout and (
        layout_minimum_position is None or layout_maximum_position is None
    ):
        raise ValueError("explicit layout memory requires both interval bounds")
    layout_lower = (
        int(layout_minimum_position) if layout_minimum_position is not None else None
    )
    layout_upper = (
        int(layout_maximum_position) if layout_maximum_position is not None else None
    )
    if explicit_layout and layout_upper <= layout_lower:
        raise ValueError("layout memory interval must have positive extent")
    if progressive_layout_state and not (include_canonical or explicit_layout):
        raise ValueError("progressive layout/state routing requires a canonical anchor")
    if novel_camera_cut and (
        include_canonical or progressive_layout_state or explicit_layout
    ):
        raise ValueError(
            "novel-camera cut cannot share authority with historical layout rows"
        )
    canonical_added = False
    canonical_position: int | None = None
    layout_entries: list[dict[str, Any]] = []
    layout_source_entries: list[dict[str, Any]] = []
    if novel_camera_layout_probe and memory["video_entries"]:
        canonical = min(
            memory["video_entries"],
            key=lambda item: int(item["position"]),
        )
        canonical_position = int(canonical["position"])
        routed_video = [canonical]
        layout_entries = [canonical]
        layout_source_entries = [canonical]
        canonical_added = True
    elif explicit_layout:
        layout_source_entries = [
            item
            for item in memory["video_entries"]
            if layout_lower <= int(item["position"]) < layout_upper
        ]
        if not layout_source_entries:
            raise RuntimeError(
                "declared camera-anchor interval has no retained visual-memory frame"
            )
        # A named segment describes the camera state reached by that segment,
        # while its earliest generated tokens can still be an H3 establishing
        # transient. Global diversity sampling intentionally keeps outliers,
        # which is the opposite of what an exact camera recurrence needs. Use
        # the two latest retained observations as a compact terminal-camera
        # consensus. This keeps multi-frame evidence while excluding the
        # opening close-up that V20 showed could drag actors and props back to
        # obsolete foreground positions.
        layout_entries = sorted(
            layout_source_entries,
            key=lambda item: int(item["position"]),
        )[-2:]
        for item in layout_entries:
            if all(
                int(existing["position"]) != int(item["position"])
                for existing in routed_video
            ):
                routed_video.append(item)
                canonical_added = True
        routed_video.sort(key=lambda item: int(item["position"]))
        canonical_position = int(layout_entries[0]["position"])
    elif include_canonical and memory["video_entries"]:
        canonical = min(
            memory["video_entries"],
            key=lambda item: int(item["position"]),
        )
        canonical_position = int(canonical["position"])
        if all(
            int(item["position"]) != int(canonical["position"])
            for item in routed_video
        ):
            routed_video.append(canonical)
            routed_video.sort(key=lambda item: int(item["position"]))
            canonical_added = True
    return validate_av_token_memory({
        **memory,
        "video_entries": routed_video,
        "visual_route": {
            "policy": (
                "structured_director_terminal_camera_band_state_v3"
                if explicit_layout
                else "structured_director_novel_camera_layout_probe_v1"
                if novel_camera_layout_probe and canonical_position is not None
                else "structured_director_novel_camera_no_visual_rows_v1"
                if novel_camera_cut
                else
                "structured_director_progressive_layout_state_v1"
                if progressive_layout_state
                else
                "structured_director_canonical_plus_active_state_v2"
                if include_canonical
                else "structured_director_active_shot_only_v1"
            ),
            "minimum_position": lower,
            "maximum_position": upper,
            "include_canonical": bool(include_canonical),
            "progressive_layout_state": bool(
                progressive_layout_state
                or (novel_camera_layout_probe and canonical_position is not None)
            ),
            "novel_camera_cut": bool(novel_camera_cut),
            "novel_camera_layout_probe": bool(novel_camera_layout_probe),
            "canonical_position": canonical_position,
            "layout_positions": [
                int(item["position"])
                for item in (
                    layout_entries
                    if explicit_layout else
                    ([canonical] if canonical_position is not None else [])
                )
            ],
            "layout_source_positions": [
                int(item["position"])
                for item in layout_source_entries
            ],
            "layout_selection_policy": (
                "latest_two_retained_terminal_camera_consensus_v1"
                if explicit_layout else
                "canonical_scene_single_step_probe_v1"
                if novel_camera_layout_probe and canonical_position is not None else
                None
            ),
            "layout_minimum_position": layout_lower,
            "layout_maximum_position": layout_upper,
            "canonical_added": canonical_added,
            "source_video_entries": len(memory["video_entries"]),
            "routed_video_entries": len(routed_video),
            "audio_entries_unchanged": len(memory["audio_entries"]),
        },
    })


def route_visual_memory_authority(
    document: dict[str, Any],
    *,
    active: bool,
    latest_only: bool = False,
) -> dict[str, Any]:
    """Select long visual references independently of voice and exact AV state.

    This is a read-only conditioning view. The stored coreset remains intact;
    the exact continuation prefix is carried separately by the request.
    """
    memory = validate_av_token_memory(document)
    routed_video = memory["video_entries"] if active else []
    if active and latest_only and routed_video:
        routed_video = [max(routed_video, key=lambda item: int(item["position"]))]
    return validate_av_token_memory({
        **memory,
        "video_entries": routed_video,
        "visual_route": {
            "policy": (
                "latest_generated_visual_authority_v1"
                if active and latest_only
                else "independent_long_visual_authority_v1"
            ),
            "active": bool(active),
            "source_video_entries": len(memory["video_entries"]),
            "routed_video_entries": len(routed_video),
            "audio_entries_unchanged": len(memory["audio_entries"]),
        },
    })


def route_audio_memory_authority(
    document: dict[str, Any],
    *,
    active: bool,
) -> dict[str, Any]:
    """Return a conditioning view honoring structural dialogue authority.

    Audio coreset entries are short native acoustic excerpts.  They remain
    voice-bearing, potentially replayable conditions and must not leak into an
    explicitly speech-free director interval.  This routing decision depends
    only on whether the localized interval owns a structural ``<d>`` event;
    it is independent of language, subject, scene, or dialogue content.
    Visual memory is never modified.
    """

    memory = validate_av_token_memory(document)
    enabled = bool(active)
    routed_audio = memory["audio_entries"] if enabled else []
    return validate_av_token_memory({
        **memory,
        "audio_entries": routed_audio,
        "audio_route": {
            "policy": "structured_dialogue_authority_v1",
            "active": enabled,
            "source_audio_entries": len(memory["audio_entries"]),
            "routed_audio_entries": len(routed_audio),
            "video_entries_unchanged": len(memory["video_entries"]),
            "prompt_content_inspected": False,
        },
    })


def token_memory_telemetry(document: dict[str, Any]) -> dict[str, Any]:
    memory = validate_av_token_memory(document)
    telemetry = {
        "schema_version": memory["schema_version"],
        "method": memory["method"],
        "updates": memory["updates"],
        "video_entries": len(memory["video_entries"]),
        "video_slots": memory["video_slots"],
        "visual_resolution": memory["visual_resolution"],
        "video_positions": [
            int(item["position"]) for item in memory["video_entries"]
        ],
        "visual_selection_policy": memory.get(
            "visual_selection_policy", "canonical_diverse_v2"
        ),
        "audio_entries": len(memory["audio_entries"]),
        "audio_slots": memory["audio_slots"],
        "audio_positions": [
            int(item["position"]) for item in memory["audio_entries"]
        ],
        "audio_block_ticks": memory["audio_block_ticks"],
        "audio_reference_ticks_per_slot": memory["audio_block_ticks"],
        "audio_representation": "dialogue_gated_native_short_voice_excerpt",
        "audio_selection_policy": memory.get(
            "audio_selection_policy", "not_collected"
        ),
        "audio_experimental_opt_in": True,
        "audio_temporal_order_preserved": True,
        "audio_content_replayable": True,
        "audio_denoise_exposure": "runtime_authority_routed",
        "audio_collection_requires_structured_dialogue": True,
        "audio_injection_requires_structured_dialogue": True,
        "audio_recent_guard_ticks": DEFAULT_AUDIO_RECENT_GUARD_TICKS,
        "audio_focus_preroll_ticks": VOICE_FOCUS_PREROLL_TICKS,
        "audio_focus_postroll_ticks": VOICE_FOCUS_POSTROLL_TICKS,
        "bounded_active_context": True,
        "text_summary": False,
    }
    visual_route = memory.get("visual_route")
    if isinstance(visual_route, dict):
        telemetry["visual_route"] = dict(visual_route)
    audio_route = memory.get("audio_route")
    if isinstance(audio_route, dict):
        telemetry["audio_route"] = dict(audio_route)
    return telemetry


__all__ = [
    "DEFAULT_AUDIO_BLOCK_TICKS",
    "DEFAULT_AUDIO_RECENT_GUARD_TICKS",
    "DEFAULT_AUDIO_SLOTS",
    "DEFAULT_VIDEO_SLOTS",
    "METHOD",
    "SCHEMA_VERSION",
    "empty_av_token_memory",
    "memory_conditioning",
    "route_audio_memory_authority",
    "route_visual_memory_authority",
    "route_visual_memory_interval",
    "token_memory_telemetry",
    "update_av_token_memory",
    "validate_av_token_memory",
]
