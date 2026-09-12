"""Versioned, request-local shot/window contract and deterministic H3 compiler.

No LLM, environment variables, semantic object guessing, or hidden window splits.
Author times describe new visible content; model times include the AV prefix.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

ENTITY = re.compile(r"\{\{([a-zA-Z][a-zA-Z0-9_]*)\}\}")
MEDIA = re.compile(r"<(Picture|Audio|Video)\s+(\d+)>", re.I)
FPS = 24
H3_FRAME_ORIGIN = 5
H3_FRAME_STRIDE = 17
# Protect all but one native H3 temporal unit of a 90-frame continuation
# context.  The remaining 17 frames / five latent tokens form a short writable
# lead-in that may replace the same provisional predecessor tail only after a
# deterministic same-time trajectory-agreement gate accepts it.
# V11 showed that a 34-frame lead-in fixes the join but lets the next window's
# semantics alter accepted action too early.  One unit retains a generative
# bridge while limiting that future-semantic exposure to 0.708 seconds.
CONTINUATION_EXACT_ANCHOR_MAX_FRAMES = 73


def _align_h3_frames(frame_count: int) -> int:
    """Snap to H3's native 5 + 17*k temporal grid without loading Torch."""
    requested = max(H3_FRAME_ORIGIN, int(frame_count))
    index = max(0, round((requested - H3_FRAME_ORIGIN) / H3_FRAME_STRIDE))
    return H3_FRAME_ORIGIN + H3_FRAME_STRIDE * index


def _continuation_video_prefix_frames(context_frames: int) -> int:
    """Split an H3-grid overlap into an exact anchor and hidden repaint tail."""

    context = int(context_frames)
    if context < H3_FRAME_ORIGIN or (context - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE:
        raise ValueError("continuation context must use H3's 5 + 17*k grid")
    return min(context, CONTINUATION_EXACT_ANCHOR_MAX_FRAMES)


def _derived_seed(seed: int, index: int) -> int:
    if index == 0:
        return int(seed) & ((1 << 64) - 1)
    value = (int(seed) + index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)


@dataclass(frozen=True, slots=True)
class WindowSegment:
    index: int
    window_frames: int
    context_frames: int
    global_context_start_frame: int
    visible_start_frame: int
    visible_frames: int
    seed: int
    prompt: str
    # Retained in the internal plan schema so older private callers remain
    # readable. Public compilation leaves it empty: V14 proved that mixing a
    # second text-conditioned velocity field creates cuts at its mask edges.
    continuation_bridge_prompt: str | None = None
    # ``opening`` starts the timeline, ``continue`` preserves the active
    # camera trajectory, and ``cut`` starts a new camera shot in the same
    # authored world.  Keep this explicit all the way to final VAE decode;
    # inferring it later from prefix width loses the semantic boundary.
    transition: str = 'continue'
    # The full AV overlap remains available and is discarded at assembly. A
    # continuation protects an exact leading anchor and may repaint its hidden
    # trailing overlap; a hard cut protects no preceding video token. Scene
    # identity outside the immediate anchor is routed through memory.
    video_prefix_frames: int | None = None
    visual_memory_floor_frame: int | None = None
    visual_memory_include_canonical: bool = False
    # A declared camera recurrence can point at one earlier authored segment.
    # Its complete visible frame interval is routed as coarse-layout evidence;
    # recent memory remains the sole late-denoise authority for mutable state.
    visual_memory_layout_anchor_start_frame: int | None = None
    visual_memory_layout_anchor_stop_frame: int | None = None
    # At a same-scene cut, use the canonical frame only during coarse-layout
    # denoising, then converge on recent state frames.  This prevents an old
    # prop placement from sharing authority with the current state.
    visual_memory_progressive_layout_state: bool = False
    preserve_latest_visual: bool = False
    authorized_dialogue_count: int | None = None
    authorized_dialogue_frames: tuple[int, ...] = ()
    reference_audio_active: bool = True
    audio_memory_active: bool = True
    # Multi-shot T2VA openings get one hidden native temporal unit in which
    # the model may settle from an incidental insert into the declared first
    # camera.  It is decoded with the shot and cropped only at delivery.
    opening_preroll_frames: int = 0
    # A new, text-declared camera has no matching historical layout image.
    # It receives no full-frame memory row.  One native terminal video token
    # instead seeds a continuous camera relocation inside the discarded
    # preroll, carrying physical layout without making the old projection a
    # reference-image authority for the final view.
    novel_camera_cut: bool = False
    terminal_video_seed: bool = False
    novel_camera_layout_probe: bool = False


@dataclass(frozen=True, slots=True)
class WindowPlan:
    requested_duration_seconds: float
    output_frames: int
    actual_duration_seconds: float
    segments: tuple[WindowSegment, ...]
    context_frames: int
    mechanism: str
    planning_policy: str
    structured_director: bool

    @property
    def continuation_count(self) -> int:
        return max(0, len(self.segments) - 1)

    def telemetry(self) -> dict[str, Any]:
        return {
            'mechanism': self.mechanism,
            'requested_duration_seconds': self.requested_duration_seconds,
            'output_frames': self.output_frames,
            'actual_duration_seconds': self.actual_duration_seconds,
            'context_frames': self.context_frames,
            'planning_policy': self.planning_policy,
            'prompt_localization_policy': (
                'authored_window_script_v2'
                if self.mechanism.endswith('_v2')
                else 'authored_window_script_v1'
            ),
            'structured_director': self.structured_director,
            'audio_clock_policy': 'cumulative_global_resample_v2',
            'continuation_count': self.continuation_count,
            'segments': [
                {
                    'index': segment.index,
                    'window_frames': segment.window_frames,
                    'context_frames': segment.context_frames,
                    'global_context_start_frame': segment.global_context_start_frame,
                    'visible_start_frame': segment.visible_start_frame,
                    'visible_frames': segment.visible_frames,
                    'seed': segment.seed,
                    'transition': segment.transition,
                    'continuation_boundary_condition': bool(
                        segment.continuation_bridge_prompt
                    ),
                    'video_prefix_frames': segment.video_prefix_frames,
                    'hidden_video_repaint_frames': (
                        max(
                            0,
                            segment.context_frames
                            - int(segment.video_prefix_frames or 0),
                        )
                        if segment.index else 0
                    ),
                    'visual_memory_floor_frame': segment.visual_memory_floor_frame,
                    'visual_memory_include_canonical': segment.visual_memory_include_canonical,
                    'visual_memory_layout_anchor_start_frame': (
                        segment.visual_memory_layout_anchor_start_frame
                    ),
                    'visual_memory_layout_anchor_stop_frame': (
                        segment.visual_memory_layout_anchor_stop_frame
                    ),
                    'visual_memory_progressive_layout_state': (
                        segment.visual_memory_progressive_layout_state
                    ),
                    'preserve_latest_visual': segment.preserve_latest_visual,
                    'authorized_dialogue_count': segment.authorized_dialogue_count,
                    'authorized_dialogue_frames': list(segment.authorized_dialogue_frames),
                    'reference_audio_active': segment.reference_audio_active,
                    'audio_memory_active': segment.audio_memory_active,
                    'opening_preroll_frames': segment.opening_preroll_frames,
                    'novel_camera_cut': segment.novel_camera_cut,
                    'terminal_video_seed': segment.terminal_video_seed,
                    'novel_camera_layout_probe': segment.novel_camera_layout_probe,
                }
                for segment in self.segments
            ],
        }


def _number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}")
    return value


def _object(value, label, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"{label}: unknown fields {sorted(unknown)}, missing fields {sorted(missing)}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 12000:
        raise ValueError(f"{label} must be nonempty text (maximum 12000 characters)")
    # Shot clocks/dialogue wrappers are generated by this compiler, not nested
    # freely inside descriptions where they could escape window ownership.
    if re.search(r"\[Shot\s+\d+\]|</?d>|\bAt\s+\d\d:\d\d|(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:", value, re.I):
        raise ValueError(f"{label}: use structured actions/dialogue and numeric times instead of nested H3 fields")
    return value.strip()


def normalize_long_video(raw: Any) -> str:
    """Return an immutable canonical JSON document suitable for persisted jobs."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and raw.get('version') in (2, 3):
        return _normalize_long_video_v2(raw)
    _object(raw, 'long_video', ('version', 'overlap_seconds', 'max_window_seconds', 'memory', 'story'), ('version', 'story'))
    if raw['version'] != 1 or isinstance(raw['version'], bool):
        raise ValueError('long_video.version must be 1')
    overlap = _number(raw.get('overlap_seconds', 1.625), 'overlap_seconds', 0.25, 4)
    maximum = _number(raw.get('max_window_seconds', 15), 'max_window_seconds', 5, 15)
    memory = _number(raw.get('memory', 60), 'memory', 0, 100)
    if not memory.is_integer():
        raise ValueError('memory must be an integer from 0 to 100')
    story = raw['story']
    _object(story, 'story', ('overview', 'entities', 'initial_state', 'shots'), ('overview', 'entities', 'initial_state', 'shots'))
    overview = _text(story['overview'], 'overview')
    entities = story['entities']
    if not isinstance(entities, dict) or not 1 <= len(entities) <= 64:
        raise ValueError('entities must define 1..64 named objects, people or places')
    entities = {key: _text(value, f'entities.{key}') for key, value in entities.items()}
    if any(not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', key) for key in entities):
        raise ValueError('entity identifiers must use letters, digits and underscores')

    def text(value, label):
        result = _text(value, label)
        for key in ENTITY.findall(result):
            if key not in entities:
                raise ValueError(f'{label}: undefined entity {{{{{key}}}}}')
        if re.search(r'\{\{|\}\}', ENTITY.sub('', result)):
            raise ValueError(f'{label}: malformed entity reference')
        return result

    def states(value, label):
        if not isinstance(value, dict):
            raise ValueError(f'{label} must map entity identifiers to states')
        if set(value) - set(entities):
            raise ValueError(f'{label}: undefined entity in state map')
        return {key: text(state, f'{label}.{key}') for key, state in value.items()}

    text(overview, 'overview')
    for key, value in entities.items():
        text(value, f'entities.{key}')
    initial = states(story['initial_state'], 'initial_state')
    if not initial:
        raise ValueError('initial_state must explicitly establish at least one entity')
    shots = story['shots']
    if not isinstance(shots, list) or not 1 <= len(shots) <= 24:
        raise ValueError('shots must contain 1..24 shots')
    normalized = []
    ids = set()
    total = 0.0
    for shot_index, shot in enumerate(shots):
        label = f'shots[{shot_index}]'
        _object(shot, label, ('id', 'camera', 'overall_soundscape', 'non_diegetic_music', 'segments'), ('id', 'camera', 'overall_soundscape', 'non_diegetic_music', 'segments'))
        shot_id = _text(shot['id'], label + '.id')
        if shot_id in ids:
            raise ValueError('shot identifiers must be unique')
        ids.add(shot_id)
        segments = shot['segments']
        if not isinstance(segments, list) or not segments:
            raise ValueError(label + '.segments must be a nonempty array')
        clean_segments = []
        local_ids = set()
        for segment in segments:
            _object(segment, label + '.segment', ('id', 'duration_seconds', 'actions', 'dialogue', 'end_state'), ('id', 'duration_seconds', 'actions', 'end_state'))
            ident = _text(segment['id'], 'segment.id')
            if ident in local_ids:
                raise ValueError('segment identifiers must be unique within each shot')
            local_ids.add(ident)
            duration = _number(segment['duration_seconds'], 'segment.duration_seconds', 1, maximum)
            actions, dialogue = segment['actions'], segment.get('dialogue', [])
            if not isinstance(actions, list) or not actions or not isinstance(dialogue, list):
                raise ValueError('actions must be nonempty and dialogue must be an array')
            clean_actions, clean_dialogue = [], []
            for action in actions:
                _object(action, 'action', ('at_seconds', 'text'), ('at_seconds', 'text'))
                at = _number(action['at_seconds'], 'action.at_seconds', 0, duration)
                if at >= duration:
                    raise ValueError('action onset must precede the segment end')
                clean_actions.append({'at_seconds': at, 'text': text(action['text'], 'action.text')})
            for line in dialogue:
                _object(line, 'dialogue', ('at_seconds', 'speaker', 'language', 'text'), ('at_seconds', 'speaker', 'language', 'text'))
                at = _number(line['at_seconds'], 'dialogue.at_seconds', 0, duration)
                if at >= duration or line['speaker'] not in entities:
                    raise ValueError('dialogue must have an in-segment onset and a defined speaker')
                language = _text(line['language'], 'dialogue.language')
                if not re.fullmatch(r'[A-Za-z][A-Za-z -]{0,40}', language):
                    raise ValueError('dialogue.language must be an English language name')
                literal = _text(line['text'], 'dialogue.text')
                if ENTITY.search(literal) or MEDIA.search(literal):
                    raise ValueError('dialogue.text is literal speech, not an entity/reference template')
                clean_dialogue.append({'at_seconds': at, 'speaker': line['speaker'], 'language': language, 'text': literal})
            clean_segments.append({'id': ident, 'duration_seconds': duration, 'actions': clean_actions, 'dialogue': clean_dialogue, 'end_state': states(segment['end_state'], 'end_state')})
            total += duration
        normalized.append({'id': shot_id, 'camera': text(shot['camera'], 'camera'), 'overall_soundscape': text(shot['overall_soundscape'], 'overall_soundscape'), 'non_diegetic_music': text(shot['non_diegetic_music'], 'non_diegetic_music'), 'segments': clean_segments})
    if not 1 <= total <= 60 or sum(len(s['segments']) for s in normalized) > 32:
        raise ValueError('story must be 1..60 seconds and at most 32 windows')
    result = {'version': 1, 'overlap_seconds': overlap, 'max_window_seconds': maximum, 'memory': int(memory), 'story': {'overview': overview, 'entities': entities, 'initial_state': initial, 'shots': normalized}}
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if len(encoded) > 100000:
        raise ValueError('long_video document exceeds 100000 characters')
    return encoded


def _normalize_long_video_v2(raw: dict[str, Any]) -> str:
    """Normalize the state-safe, range-timed window-script v2 contract."""
    _object(
        raw,
        'long_video',
        ('version', 'overlap_seconds', 'max_window_seconds', 'memory', 'story'),
        ('version', 'story'),
    )
    overlap = _number(raw.get('overlap_seconds', 1.625), 'overlap_seconds', 0.25, 4)
    maximum = _number(raw.get('max_window_seconds', 15), 'max_window_seconds', 5, 15)
    memory = _number(raw.get('memory', 60), 'memory', 0, 100)
    if not memory.is_integer():
        raise ValueError('memory must be an integer from 0 to 100')
    story = raw['story']
    _object(
        story,
        'story',
        ('overview', 'entities', 'initial_state', 'shots'),
        ('overview', 'entities', 'initial_state', 'shots'),
    )
    overview = _text(story['overview'], 'overview')
    entities_raw = story['entities']
    if not isinstance(entities_raw, dict) or not 1 <= len(entities_raw) <= 64:
        raise ValueError('entities must define 1..64 named objects, people or places')
    if any(not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', key) for key in entities_raw):
        raise ValueError('entity identifiers must use letters, digits and underscores')
    entities: dict[str, dict[str, str]] = {}
    for key, value in entities_raw.items():
        _object(value, f'entities.{key}', ('kind', 'identity'), ('kind', 'identity'))
        kind = value['kind']
        if kind not in ('character', 'location', 'prop'):
            raise ValueError(f'entities.{key}.kind must be character, location or prop')
        entities[key] = {
            'kind': kind,
            'identity': _text(value['identity'], f'entities.{key}.identity'),
        }

    def text(value: Any, label: str) -> str:
        result = _text(value, label)
        for key in ENTITY.findall(result):
            if key not in entities:
                raise ValueError(f'{label}: undefined entity {{{{{key}}}}}')
        if re.search(r'\{\{|\}\}', ENTITY.sub('', result)):
            raise ValueError(f'{label}: malformed entity reference')
        return result

    for key, value in entities.items():
        text(value['identity'], f'entities.{key}.identity')
    text(overview, 'overview')

    def properties(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError(f'{label} must be a nonempty object of typed state properties')
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', key):
                raise ValueError(f'{label} property names must use letters, digits and underscores')
            if isinstance(item, str):
                result[key] = text(item, f'{label}.{key}')
            elif item is None:
                result[key] = item
            elif isinstance(item, bool):
                result[key] = item
            elif isinstance(item, (int, float)):
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError(f'{label}.{key} must be finite')
                result[key] = item
            else:
                raise ValueError(f'{label}.{key} must be a string, number, boolean or null')
        return result

    def state_map(value: Any, label: str, *, complete: bool) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            raise ValueError(f'{label} must map entity identifiers to typed state objects')
        unknown = set(value) - set(entities)
        missing = set(entities) - set(value) if complete else set()
        if unknown or missing:
            raise ValueError(f'{label}: undefined entities {sorted(unknown)}, missing entities {sorted(missing)}')
        return {key: properties(item, f'{label}.{key}') for key, item in value.items()}

    initial = state_map(story['initial_state'], 'initial_state', complete=True)
    shots_raw = story['shots']
    if not isinstance(shots_raw, list) or not 1 <= len(shots_raw) <= 24:
        raise ValueError('shots must contain 1..24 shots')
    normalized_shots = []
    shot_ids: set[str] = set()
    total = 0.0
    segment_total = 0
    for shot_index, shot in enumerate(shots_raw):
        label = f'shots[{shot_index}]'
        _object(
            shot,
            label,
            ('id', 'camera_anchor', 'camera_style', 'overall_soundscape', 'non_diegetic_music', 'segments'),
            ('id', 'camera_style', 'overall_soundscape', 'non_diegetic_music', 'segments'),
        )
        shot_id = _text(shot['id'], label + '.id')
        if shot_id in shot_ids:
            raise ValueError('shot identifiers must be unique')
        camera_anchor = None
        if 'camera_anchor' in shot:
            if memory == 0:
                raise ValueError(f'{label}.camera_anchor requires long-video memory above zero')
            anchor_raw = shot['camera_anchor']
            _object(
                anchor_raw,
                label + '.camera_anchor',
                ('shot_id', 'segment_id'),
                ('shot_id', 'segment_id'),
            )
            anchor_shot_id = _text(
                anchor_raw['shot_id'], label + '.camera_anchor.shot_id'
            )
            anchor_segment_id = _text(
                anchor_raw['segment_id'], label + '.camera_anchor.segment_id'
            )
            earlier = next(
                (item for item in normalized_shots if item['id'] == anchor_shot_id),
                None,
            )
            if earlier is None:
                raise ValueError(
                    f'{label}.camera_anchor.shot_id must name an earlier shot'
                )
            if anchor_segment_id not in {
                item['id'] for item in earlier['segments']
            }:
                raise ValueError(
                    f'{label}.camera_anchor.segment_id must name a segment in '
                    f'the earlier shot {anchor_shot_id}'
                )
            camera_anchor = {
                'shot_id': anchor_shot_id,
                'segment_id': anchor_segment_id,
            }
        shot_ids.add(shot_id)
        segments_raw = shot['segments']
        if not isinstance(segments_raw, list) or not segments_raw:
            raise ValueError(label + '.segments must be a nonempty array')
        clean_segments = []
        local_ids: set[str] = set()
        for segment_index, segment in enumerate(segments_raw):
            segment_label = f'{label}.segments[{segment_index}]'
            _object(
                segment,
                segment_label,
                ('id', 'duration_seconds', 'transition', 'establish_seconds', 'camera', 'beats', 'dialogue'),
                ('id', 'duration_seconds', 'transition', 'camera', 'beats'),
            )
            ident = _text(segment['id'], segment_label + '.id')
            if ident in local_ids:
                raise ValueError('segment identifiers must be unique within each shot')
            local_ids.add(ident)
            duration = _number(segment['duration_seconds'], segment_label + '.duration_seconds', 1, maximum)
            transition = segment['transition']
            expected_transition = (
                'opening' if shot_index == 0 and segment_index == 0
                else 'cut' if segment_index == 0
                else 'continue'
            )
            if transition != expected_transition:
                raise ValueError(
                    f'{segment_label}.transition must be {expected_transition} for its shot position'
                )
            establish = _number(
                segment.get('establish_seconds', 0.75 if transition == 'cut' else 0),
                segment_label + '.establish_seconds',
                0,
                duration,
            )
            if transition == 'cut' and establish < 0.5:
                raise ValueError(f'{segment_label}.establish_seconds must be at least 0.5 for a cut')
            camera_raw = segment['camera']
            required_camera = ('composition', 'motion', 'end') if transition in ('opening', 'cut') else ('motion', 'end')
            _object(camera_raw, segment_label + '.camera', ('composition', 'motion', 'end'), required_camera)
            camera = {
                key: text(value, f'{segment_label}.camera.{key}')
                for key, value in camera_raw.items()
            }
            beats_raw = segment['beats']
            dialogue_raw = segment.get('dialogue', [])
            if not isinstance(beats_raw, list) or not beats_raw or not isinstance(dialogue_raw, list):
                raise ValueError('beats must be nonempty and dialogue must be an array')
            beats = []
            previous_end = 0.0
            for beat_index, beat in enumerate(beats_raw):
                beat_label = f'{segment_label}.beats[{beat_index}]'
                _object(
                    beat,
                    beat_label,
                    ('start_seconds', 'end_seconds', 'text', 'state_updates'),
                    ('start_seconds', 'end_seconds', 'text', 'state_updates'),
                )
                start = _number(beat['start_seconds'], beat_label + '.start_seconds', 0, duration)
                end = _number(beat['end_seconds'], beat_label + '.end_seconds', 0, duration)
                if end <= start or start < previous_end:
                    raise ValueError(f'{beat_label} must have a positive, non-overlapping time range')
                if transition == 'cut' and start < establish:
                    raise ValueError(f'{beat_label} starts before the cut establishing interval ends')
                updates = state_map(beat['state_updates'], beat_label + '.state_updates', complete=False)
                beats.append({
                    'start_seconds': start,
                    'end_seconds': end,
                    'text': text(beat['text'], beat_label + '.text'),
                    'state_updates': updates,
                })
                previous_end = end
            dialogue = []
            for line_index, line in enumerate(dialogue_raw):
                line_label = f'{segment_label}.dialogue[{line_index}]'
                _object(
                    line,
                    line_label,
                    ('start_seconds', 'end_seconds', 'speaker', 'language', 'text'),
                    ('start_seconds', 'end_seconds', 'speaker', 'language', 'text'),
                )
                start = _number(line['start_seconds'], line_label + '.start_seconds', 0, duration)
                end = _number(line['end_seconds'], line_label + '.end_seconds', 0, duration)
                if end <= start or (transition == 'cut' and start < establish):
                    raise ValueError(f'{line_label} must occupy a positive range after the cut establishing interval')
                speaker = line['speaker']
                if speaker not in entities or entities[speaker]['kind'] != 'character':
                    raise ValueError(f'{line_label}.speaker must reference a character entity')
                language = _text(line['language'], line_label + '.language')
                if not re.fullmatch(r'[A-Za-z][A-Za-z -]{0,40}', language):
                    raise ValueError('dialogue.language must be an English language name')
                literal = _text(line['text'], line_label + '.text')
                if ENTITY.search(literal) or MEDIA.search(literal):
                    raise ValueError('dialogue.text is literal speech, not an entity/reference template')
                dialogue.append({
                    'start_seconds': start,
                    'end_seconds': end,
                    'speaker': speaker,
                    'language': language,
                    'text': literal,
                })
            clean_segments.append({
                'id': ident,
                'duration_seconds': duration,
                'transition': transition,
                'establish_seconds': establish,
                'camera': camera,
                'beats': beats,
                'dialogue': dialogue,
            })
            total += duration
            segment_total += 1
        normalized_shot = {
            'id': shot_id,
            'camera_style': text(shot['camera_style'], label + '.camera_style'),
            'overall_soundscape': text(shot['overall_soundscape'], label + '.overall_soundscape'),
            'non_diegetic_music': text(shot['non_diegetic_music'], label + '.non_diegetic_music'),
            'segments': clean_segments,
        }
        if camera_anchor is not None:
            normalized_shot['camera_anchor'] = camera_anchor
        normalized_shots.append(normalized_shot)
    if not 1 <= total <= 60 or segment_total > 32:
        raise ValueError('story must be 1..60 seconds and at most 32 windows')
    encoded = json.dumps({
        'version': int(raw['version']),
        'overlap_seconds': overlap,
        'max_window_seconds': maximum,
        'memory': int(memory),
        'story': {
            'overview': overview,
            'entities': entities,
            'initial_state': initial,
            'shots': normalized_shots,
        },
    }, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if len(encoded) > 100000:
        raise ValueError('long_video document exceeds 100000 characters')
    return encoded


def story_duration(document: str) -> float:
    return sum(segment['duration_seconds'] for shot in json.loads(document)['story']['shots'] for segment in shot['segments'])


def memory_budget(document: str, *, service_family: str, width=864, height=480, user_images=0, user_audios=0):
    data = json.loads(document)
    amount = data['memory']
    speakers = {d['speaker'] for s in data['story']['shots'] for seg in s['segments'] for d in seg['dialogue']}
    # Capacity is monotone, not a learned conditioning-strength multiplier.
    max_ticks = 240 if service_family == 'reference' else 120
    min_ticks = 80 if service_family == 'reference' else 20
    ticks = min_ticks + math.floor((max_ticks - min_ticks) * (amount - 1) / 99) if amount else 0
    if user_audios or len(speakers) != 1:
        ticks = 0
    requested_video = math.ceil(9 * amount / 100) if amount else 0
    video = max(0, min(
        requested_video,
        9 - user_images,
        12 - user_images - user_audios - int(ticks > 0),
    ))
    return {'memory': amount, 'requested_video_frames': requested_video, 'video_frames': video,
            'audio_clips': int(ticks > 0), 'audio_ticks_per_clip': ticks, 'audio_seconds': ticks / 40,
            'maximum_memory_tokens': video * (height // 32) * (width // 32) + ticks * 2,
            'automatic_audio_reason': 'user_reference_priority' if user_audios else 'single_speaker_only' if len(speakers) != 1 else 'enabled' if ticks else 'memory_off',
            'meaning': 'capacity_budget_not_quality_strength'}


def validate_reference_inputs(document: str, *, first_frame=False, last_frame=False, reference_images=0, reference_audios=0, reference_videos=0, service_family='first_last'):
    if reference_videos:
        raise ValueError('window-script v1 does not support reference videos')
    if service_family == 'reference':
        pictures = reference_images
        if reference_images + reference_audios > 12:
            raise ValueError('at most 12 public reference items')
    else:
        # Endpoints are injected by the engine; a local window must not claim
        # an endpoint reference exists when it is only sent to another window.
        pictures = 0
    used = {(kind.lower(), int(number)) for kind, number in MEDIA.findall(document)}
    for kind, number in MEDIA.findall(document):
        limit = pictures if kind.lower() == 'picture' else reference_audios if kind.lower() == 'audio' else 0
        if not 1 <= int(number) <= limit:
            raise ValueError(f'<{kind} {number}> has no matching reference input for this window-script route')
    if service_family == 'reference':
        unused_pictures = [
            number for number in range(1, reference_images + 1)
            if ('picture', number) not in used
        ]
        unused_audios = [
            number for number in range(1, reference_audios + 1)
            if ('audio', number) not in used
        ]
        if unused_pictures or unused_audios:
            raise ValueError(
                'every Ref2VA upload must be cited in the story; unused '
                f'pictures={unused_pictures}, audios={unused_audios}'
            )
    # Reject unbound Subject tags; entity references use the declared glossary.
    if re.search(r'<Subject\s+\d+>', document, re.I):
        raise ValueError('use {{entity_id}} with a definition instead of unbound Subject labels')


def compile_window_story(
    document: str,
    *,
    seed: int,
    maximum_frames: int = 362,
    service_family: str = 'first_last',
    first_frame: bool = False,
    last_frame: bool = False,
):
    if service_family not in ('first_last', 'reference'):
        raise ValueError('service_family must be first_last or reference')
    data = json.loads(document)
    if data.get('version') == 3:
        return _compile_window_story_v3(
            data,
            seed=seed,
            maximum_frames=maximum_frames,
            service_family=service_family,
            first_frame=first_frame,
            last_frame=last_frame,
        )
    if data.get('version') == 2:
        return _compile_window_story_v2(
            data,
            seed=seed,
            maximum_frames=maximum_frames,
            service_family=service_family,
            first_frame=first_frame,
            last_frame=last_frame,
        )
    story = data['story']
    maximum = min(maximum_frames, 5 + 17 * math.floor((24 * data['max_window_seconds'] - 5 + 1e-8) / 17))
    context = min(90, _align_h3_frames(round(data['overlap_seconds'] * 24)))
    if context >= maximum:
        raise ValueError('effective overlap must be shorter than the maximum window')
    state = dict(story['initial_state'])
    segments, preview = [], []
    requested_cursor = 0.0
    actual_cursor = 0
    entities = story['entities']
    speaker_order = []
    for authored_shot in story['shots']:
        for authored_segment in authored_shot['segments']:
            for line in authored_segment['dialogue']:
                if line['speaker'] not in speaker_order:
                    speaker_order.append(line['speaker'])
    speaker_ids = {
        speaker: f'S{index + 1}' for index, speaker in enumerate(speaker_order)
    }
    referenced_entities = {
        key for key, value in entities.items()
        if any(kind.lower() in ('picture', 'video') for kind, _ in MEDIA.findall(value))
        or (
            key in speaker_ids
            and any(kind.lower() == 'audio' for kind, _ in MEDIA.findall(value))
        )
    }
    subject_ids = {
        key: f'<Subject {index + 1}>'
        for index, key in enumerate(sorted(referenced_entities))
    }
    segment_count = sum(len(shot['segments']) for shot in story['shots'])

    def expand(value, *, audio_active=True):
        rendered = ENTITY.sub(
            lambda match: (
                subject_ids.get(match.group(1), match.group(1))
                if service_family == 'reference'
                else match.group(1)
            ),
            value,
        )
        if not audio_active:
            rendered = re.sub(
                r'<Audio\s+\d+>', 'the established voice identity',
                rendered, flags=re.I,
            )
        return rendered

    def entity_name(key):
        name = subject_ids.get(key, key) if service_family == 'reference' else key
        return f'{name} ({speaker_ids[key]})' if key in speaker_ids else name

    def clock(seconds):
        return f'{int(seconds // 60):02d}:{seconds % 60:06.3f}'

    for shot_index, shot in enumerate(story['shots']):
        for within_shot, segment in enumerate(shot['segments']):
            index = len(segments)
            overlap = context if index else 0
            audio_active = bool(segment['dialogue'])
            render = lambda value: expand(value, audio_active=audio_active)
            requested_stop = requested_cursor + segment['duration_seconds']
            stop = _align_h3_frames(round(requested_stop * 24))
            visible = stop - actual_cursor
            window = visible + overlap
            if overlap > actual_cursor:
                raise ValueError(f"{shot['id']}/{segment['id']}: overlap exceeds available history")
            if visible <= 0 or window > maximum or window < 5 or (window - 5) % 17:
                raise ValueError(f"{shot['id']}/{segment['id']}: new content plus effective overlap requires {window / 24:.3f}s, maximum is {maximum / 24:.3f}s; shorten the segment or change the window controls")
            offset = overlap / 24
            cut = index > 0 and within_shot == 0
            relevant_text = [story['overview'], shot['camera'], shot['overall_soundscape'], shot['non_diegetic_music']]
            relevant_text += [a['text'] for a in segment['actions']] + list(segment['end_state'].values())
            needed = set(segment['end_state']) | {d['speaker'] for d in segment['dialogue']}
            for value in relevant_text:
                needed.update(ENTITY.findall(value))
            # Include transitive definitions and the states they rely on.
            while True:
                old = set(needed)
                for key in old:
                    needed.update(ENTITY.findall(entities[key]))
                    needed.update(ENTITY.findall(state.get(key, '')))
                if needed == old:
                    break
            missing_state = needed - set(state)
            if missing_state:
                raise ValueError(f"{shot['id']}/{segment['id']}: initial/current state is missing for {sorted(missing_state)}; define offscreen/absent state explicitly")
            definitions = '\n'.join(
                f'{entity_name(key)}: {render(entities[key])}' for key in sorted(needed)
            )
            start_state = '\n'.join(
                f'{entity_name(key)}: {render(state[key])}' for key in sorted(needed)
            )
            next_state = {**state, **segment['end_state']}
            ending = '\n'.join(
                f'{entity_name(key)}: {render(next_state[key])}' for key in sorted(needed)
            )
            if cut:
                camera = (f'[Shot 1] Until local {clock(offset)}, retain only the carried preceding view. '
                          f'[Shot 2] At {clock(offset)}, the camera cuts to {render(shot["camera"])} '
                          'This is a camera cut, not a walk, pan or morph from the previous viewpoint. Preserve established identities and object states across the cut. No further cut.')
            elif index:
                camera = ('[Shot 1] Continue the existing uninterrupted take and the camera motion visible in the carried frames. '
                          'Do not reset to the opening position. Camera specification for this same shot: ' + render(shot['camera']) + ' No cut.')
            else:
                camera = '[Shot 1] ' + render(shot['camera']) + ' No cut inside this window.'
            events = []
            dialogue_frames = []
            for action in segment['actions']:
                if action['at_seconds'] >= visible / 24:
                    raise ValueError('frame-grid rounding puts an action outside its window; move the onset earlier')
                events.append((action['at_seconds'], f'At {clock(offset + action["at_seconds"])}, {render(action["text"])}'))
            for line in segment['dialogue']:
                if line['at_seconds'] >= visible / 24:
                    raise ValueError('frame-grid rounding puts dialogue outside its window; move its onset earlier')
                events.append((line['at_seconds'], f'At {clock(offset + line["at_seconds"])}, {entity_name(line["speaker"])} says once: <d>[{line["language"]}] {line["text"]}</d>'))
                dialogue_frames.append(actual_cursor + round(line['at_seconds'] * 24))
            speech = f'Speak only the {len(dialogue_frames)} specified dialogue lines, once each.' if dialogue_frames else 'No speech, narration, singing or other human voice in the newly generated interval.'
            description = (render(story['overview']) +
                      '\nEntity definitions (identity references, not instructions to place every entity on screen):\n' + definitions +
                      f'\nThis local window starts at 00:00.000. Its carried context is 00:00.000–{clock(offset)}; generate new content only from {clock(offset)} to {clock(window / 24)}.\n' +
                      camera + '\nAuthored state at the start of new content; continue from the actual carried state without replay or teleportation:\n' + start_state +
                      '\nCurrent interval events, each performed once:\n' + '\n'.join(value for _, value in sorted(events, key=lambda item: item[0])) +
                      '\nRequired state at this interval\'s END, not its beginning:\n' + (ending or 'Preserve the resulting state.') + '\n' + speech +
                      '')
            soundscape = (render(shot['overall_soundscape']) +
                          ' Keep continuous ambience continuous; event sounds occur only with the specified current actions. Do not replay earlier action sounds.')
            music = (render(shot['non_diegetic_music']) +
                     (' Continue the existing musical passage without restarting it.' if within_shot and shot['non_diegetic_music'] != 'N/A' else ''))
            if service_family == 'reference':
                local_subjects = [key for key in sorted(needed) if key in subject_ids]
                referenced_text = '\n'.join(
                    [story['overview'], shot['camera'], shot['overall_soundscape'], shot['non_diegetic_music']]
                    + [entities[key] for key in sorted(needed)]
                    + [a['text'] for a in segment['actions']]
                )
                media = sorted(
                    {
                        (kind.title(), int(number))
                        for kind, number in MEDIA.findall(referenced_text)
                        if audio_active or kind.lower() != 'audio'
                    },
                    key=lambda item: (item[0], item[1]),
                )
                subject_lines = [
                    f'{subject_ids[key]} is {render(entities[key])}'
                    for key in local_subjects
                ]
                used_picture_numbers = {
                    int(number) for key in local_subjects
                    for kind, number in MEDIA.findall(entities[key])
                    if kind.lower() == 'picture'
                }
                for kind, number in media:
                    if kind == 'Picture' and number not in used_picture_numbers:
                        subject_lines.append(
                            f'<Picture {number}> is a concrete visual reference cited in this local target window.'
                        )
                    elif kind == 'Audio':
                        bound = [
                            key for key in local_subjects
                            if (kind, str(number)) in {
                                (found.title(), index) for found, index in MEDIA.findall(entities[key])
                            }
                        ]
                        if len(bound) == 1 and bound[0] in speaker_ids:
                            key = bound[0]
                            subject_lines.append(
                                f'<Audio {number}> is the voice-timbre reference for {subject_ids[key]} ({speaker_ids[key]}).'
                            )
                        else:
                            subject_lines.append(
                                f'<Audio {number}> is an authored audio reference for this local target window.'
                            )
                task_types = ['reference generation']
                if any(kind == 'Audio' for kind, _ in media):
                    task_types.append('audio reference')
                labels = [subject_ids[key] for key in local_subjects]
                summary = (
                    f'[{" + ".join(task_types)}] {render(story["overview"])} '
                    f'This local window preserves {", ".join(labels)}.'
                    if labels else
                    f'[{" + ".join(task_types)}] {render(story["overview"])}'
                )
                retention = [
                    f'{subject_ids[key]} (appears in the current local window): fully_preserved - {render(entities[key])}'
                    for key in local_subjects
                ]
                retention.extend(
                    f'<Picture {number}> (cited in the current local window): fully_preserved - retain its authored visual role.'
                    for kind, number in media
                    if kind == 'Picture' and number not in used_picture_numbers
                )
                retention.extend(
                    f'<Audio {number}>: reference - use only its authored reference role without copying its words or timing.'
                    for kind, number in media if kind == 'Audio'
                )
                prompt = (
                    'subject_definitions:\n' + ('\n'.join(subject_lines) or 'No external reference label is active in this local window.') +
                    '\n\nsummary:\n' + summary +
                    '\n\nretention_analysis:\n' + ('\n'.join(retention) or 'No external reference retention applies in this local window.') +
                    '\n\ndetailed_description:\n' + description +
                    '\n\noverall_soundscape: ' + soundscape +
                    '\n\nnon_diegetic_music: ' + music
                )
            else:
                prompt = (
                    'integrated_multimodal_description:\n' + description +
                    '\noverall_soundscape: ' + soundscape +
                    '\n\nnon_diegetic_music: ' + music
                )
                instruction = None
                if segment_count == 1 and first_frame and last_frame:
                    instruction = (
                        'How the reference pictures align with the target video — '
                        'Picture 1 (from Shot 1) aligns with the 0.00-second mark '
                        f'of the target video; Picture 2 (from Shot 1) aligns with '
                        f'the {window / FPS:.2f}-second mark of the target video.'
                    )
                elif index == 0 and first_frame:
                    instruction = (
                        'For the target video, at 0.00 seconds into the target video, '
                        '<Picture 1> (from [Shot 1]) is fully referenced.'
                    )
                elif index == segment_count - 1 and last_frame:
                    instruction = (
                        'How the reference pictures align with the target video — '
                        f'<Picture 1> (from [Shot 1]) aligns with the '
                        f'{window / FPS:.2f}-second mark of the target video.'
                    )
                if instruction is not None:
                    prompt = instruction + '\n\n' + prompt
            if len(prompt) > 20000:
                raise ValueError('compiled window prompt exceeds 20000 characters')
            assert not ENTITY.search(prompt)
            transition = 'cut' if cut else 'continue' if index else 'opening'
            segments.append(WindowSegment(index=index, window_frames=window, context_frames=overlap,
                global_context_start_frame=actual_cursor - overlap, visible_start_frame=actual_cursor,
                visible_frames=visible, seed=_derived_seed(seed, index), prompt=prompt,
                transition=transition,
                visual_memory_floor_frame=actual_cursor - overlap if index and within_shot else None,
                visual_memory_include_canonical=bool(index and within_shot),
                preserve_latest_visual=within_shot < len(shot['segments']) - 1,
                authorized_dialogue_count=len(dialogue_frames), authorized_dialogue_frames=tuple(dialogue_frames),
                reference_audio_active=bool(dialogue_frames), audio_memory_active=bool(dialogue_frames)))
            preview.append({'shot_id': shot['id'], 'segment_id': segment['id'], 'transition': transition,
                'requested_start_seconds': requested_cursor, 'requested_end_seconds': requested_stop,
                'actual_start_seconds': actual_cursor / 24, 'actual_end_seconds': stop / 24,
                'context_seconds': offset, 'window_seconds': window / 24, 'prompt': prompt, 'entities': sorted(needed)})
            state.update(segment['end_state'])
            requested_cursor, actual_cursor = requested_stop, stop
    plan = WindowPlan(requested_duration_seconds=requested_cursor, output_frames=actual_cursor,
        actual_duration_seconds=actual_cursor / 24, segments=tuple(segments), context_frames=context,
        mechanism='authored_window_script_joint_av_v1', planning_policy='one_authored_segment_per_window_v1', structured_director=True)
    return plan, {'version': 1, 'requested_controls': {key: data[key] for key in ('overlap_seconds', 'max_window_seconds', 'memory')},
                  'effective_overlap_seconds': context / 24, 'effective_max_window_seconds': maximum / 24,
                  'actual_duration_seconds': plan.actual_duration_seconds, 'windows': preview,
                  'reference_check': 'Entity identifiers are closed; arbitrary free-text pronouns and real visual-state correctness are not semantically verified.'}


def _plan_internal_cut_windows(
    *,
    output_frames: int,
    context_frames: int,
    maximum_frames: int,
    cut_frames: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    """Choose bounded physical windows whose overlap bands stay inside shots.

    The first result is the opening-window frame count.  Remaining values are
    new visible strides; a continuation's physical size is ``context +
    stride``.  Authored cuts are semantic events inside a window.  They are
    therefore kept out of every physical overlap band and at least one second
    away from the beginning of a writable suffix.
    """

    output = int(output_frames)
    context = int(context_frames)
    maximum = int(maximum_frames)
    if output <= maximum:
        return output, ()

    minimum_physical = min(124, maximum)
    minimum_opening_units = max(
        0, math.ceil((minimum_physical - H3_FRAME_ORIGIN) / H3_FRAME_STRIDE)
    )
    maximum_opening_units = (maximum - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE
    minimum_stride_units = max(
        1, math.ceil((minimum_physical - context) / H3_FRAME_STRIDE)
    )
    maximum_stride_units = (maximum - context) // H3_FRAME_STRIDE
    total_units = (output - H3_FRAME_ORIGIN) // H3_FRAME_STRIDE
    if maximum_stride_units < minimum_stride_units:
        raise ValueError('maximum window is too short for one continuation')

    def legal_seam(seam: int) -> bool:
        for cut in cut_frames:
            # The next exact latent overlap must describe only the camera that
            # is active at the seam.  A cut inside that band would give one
            # continuation prompt two incompatible camera authorities.
            if seam - context <= cut <= seam:
                return False
            # Give the outgoing shot at least one visible second before a cut
            # that belongs to the next physical window.
            if seam < cut < seam + FPS:
                return False
        return True

    def semantic_cost(seam: int) -> float:
        if not cut_frames:
            return 0.0
        distance = min(abs(seam - cut) for cut in cut_frames)
        return 30.0 * math.exp(-0.5 * (distance / 24.0) ** 2)

    # Values are (cost, continuation count, visible-stride units).  This is
    # the same bounded shortest-path shape as the accepted September-1
    # planner, with the additional hard same-shot overlap invariant above.
    tail: list[tuple[float, int, tuple[int, ...]] | None] = [
        None
    ] * (total_units + 1)
    tail[0] = (0.0, 0, ())
    for units_left in range(1, total_units + 1):
        best = None
        for stride_units in range(
            minimum_stride_units,
            min(maximum_stride_units, units_left) + 1,
        ):
            remainder = units_left - stride_units
            downstream = tail[remainder]
            if downstream is None:
                continue
            visible = H3_FRAME_STRIDE * stride_units
            seam = output - H3_FRAME_STRIDE * remainder
            if remainder and not legal_seam(seam):
                continue
            downstream_cost, downstream_count, downstream_strides = downstream
            physical = context + visible
            candidate = (
                16.0 + 0.000152 * physical * physical
                + (semantic_cost(seam) if remainder else 0.0)
                + downstream_cost,
                1 + downstream_count,
                (stride_units,) + downstream_strides,
            )
            if best is None or candidate < best:
                best = candidate
        tail[units_left] = best

    best_plan = None
    for opening_units in range(minimum_opening_units, maximum_opening_units + 1):
        remaining = total_units - opening_units
        if remaining <= 0 or remaining >= len(tail):
            continue
        downstream = tail[remaining]
        if downstream is None:
            continue
        opening = H3_FRAME_ORIGIN + H3_FRAME_STRIDE * opening_units
        if not legal_seam(opening):
            continue
        downstream_cost, continuation_count, stride_units = downstream
        candidate = (
            16.0 + 0.000152 * opening * opening
            + semantic_cost(opening) + downstream_cost,
            1 + continuation_count,
            opening_units,
            stride_units,
        )
        if best_plan is None or candidate < best_plan:
            best_plan = candidate
    if best_plan is None:
        raise ValueError(
            'no physical-window layout can keep every overlap inside one shot; '
            'shorten the overlap, increase the maximum window, or lengthen the '
            'material on one side of a cut'
        )
    _, _, opening_units, stride_units = best_plan
    return (
        H3_FRAME_ORIGIN + H3_FRAME_STRIDE * opening_units,
        tuple(H3_FRAME_STRIDE * value for value in stride_units),
    )


def _compile_window_story_v3(
    data: dict[str, Any],
    *,
    seed: int,
    maximum_frames: int,
    service_family: str,
    first_frame: bool,
    last_frame: bool,
):
    """Compile complete prompts with authored cuts inside physical windows."""

    # Keep the Human-approved single-take route byte-for-byte intact.  Version
    # 3 changes only the mapping between multiple semantic shots and physical
    # compute windows.
    if len(data['story']['shots']) == 1:
        legacy = json.loads(json.dumps(data))
        legacy['version'] = 2
        plan, preview = _compile_window_story_v2(
            legacy,
            seed=seed,
            maximum_frames=maximum_frames,
            service_family=service_family,
            first_frame=first_frame,
            last_frame=last_frame,
        )
        preview['version'] = 3
        preview['mapping_policy'] = 'single_take_v2_preserved'
        return plan, preview
    if service_family != 'first_last':
        raise ValueError(
            'version 3 internal-cut compilation is currently validated only '
            'for the first_last service family'
        )

    story = data['story']
    maximum = min(
        int(maximum_frames),
        H3_FRAME_ORIGIN + H3_FRAME_STRIDE * math.floor(
            (FPS * data['max_window_seconds'] - H3_FRAME_ORIGIN + 1e-8)
            / H3_FRAME_STRIDE
        ),
    )
    context = min(90, _align_h3_frames(round(data['overlap_seconds'] * FPS)))
    if context >= maximum:
        raise ValueError('effective overlap must be shorter than the maximum window')

    entities = story['entities']
    speaker_order: list[str] = []
    for shot in story['shots']:
        for segment in shot['segments']:
            for line in segment['dialogue']:
                if line['speaker'] not in speaker_order:
                    speaker_order.append(line['speaker'])
    speaker_ids = {
        speaker: f'S{index + 1}' for index, speaker in enumerate(speaker_order)
    }

    def clone(value: Any) -> Any:
        return json.loads(json.dumps(value))

    def expand(value: str) -> str:
        return ENTITY.sub(lambda match: match.group(1), value)

    def entity_name(key: str) -> str:
        return f'{key} ({speaker_ids[key]})' if key in speaker_ids else key

    def state_value(value: Any) -> str:
        if isinstance(value, str):
            return expand(value)
        if value is True:
            return 'true'
        if value is False:
            return 'false'
        if value is None:
            return 'null'
        return str(value)

    def state_lines(value: dict[str, dict[str, Any]]) -> str:
        rows = []
        for key in sorted(value):
            properties = '; '.join(
                f'{field}={state_value(item)}'
                for field, item in sorted(value[key].items())
            )
            rows.append(f'{entity_name(key)}: {properties}')
        return '\n'.join(rows)

    def update_text(value: dict[str, dict[str, Any]]) -> str:
        rows = []
        for key in sorted(value):
            properties = '; '.join(
                f'{field}={state_value(item)}'
                for field, item in sorted(value[key].items())
            )
            rows.append(f'{entity_name(key)}: {properties}')
        return ' | '.join(rows)

    def apply_updates(
        target: dict[str, dict[str, Any]],
        updates: dict[str, dict[str, Any]],
    ) -> None:
        for key, values in updates.items():
            target[key].update(values)

    def clock(seconds: float) -> str:
        value = max(0.0, float(seconds))
        return f'{int(value // 60):02d}:{value % 60:06.3f}'

    timeline: list[dict[str, Any]] = []
    state_updates: list[tuple[float, dict[str, dict[str, Any]]]] = []
    cut_times: list[float] = []
    requested_cursor = 0.0
    state = clone(story['initial_state'])
    previous_camera_end: str | None = None
    for shot_index, shot in enumerate(story['shots']):
        shot_start = requested_cursor
        if shot_index:
            cut_times.append(shot_start)
        shot_opening_composition: str | None = None
        for segment_index, segment in enumerate(shot['segments']):
            start = requested_cursor
            stop = start + segment['duration_seconds']
            composition = segment['camera'].get('composition')
            if composition is not None:
                shot_opening_composition = composition
            effective_composition = (
                composition or previous_camera_end or shot_opening_composition
            )
            if effective_composition is None:
                raise ValueError(
                    f"{shot['id']}/{segment['id']}: no camera composition is available"
                )
            entry = {
                'shot': shot,
                'shot_index': shot_index,
                'segment': segment,
                'segment_index': segment_index,
                'start': start,
                'stop': stop,
                'start_state': clone(state),
                'composition': effective_composition,
            }
            for beat in segment['beats']:
                event = {
                    'kind': 'beat',
                    'start': start + beat['start_seconds'],
                    'stop': start + beat['end_seconds'],
                    'beat': beat,
                    'entry': entry,
                }
                entry.setdefault('events', []).append(event)
                state_updates.append((event['stop'], beat['state_updates']))
                apply_updates(state, beat['state_updates'])
            for line in segment['dialogue']:
                entry.setdefault('events', []).append({
                    'kind': 'dialogue',
                    'start': start + line['start_seconds'],
                    'stop': start + line['end_seconds'],
                    'line': line,
                    'entry': entry,
                })
            entry['end_state'] = clone(state)
            timeline.append(entry)
            previous_camera_end = segment['camera']['end']
            requested_cursor = stop

    output_frames = _align_h3_frames(round(requested_cursor * FPS))
    cut_frames = tuple(round(value * FPS) for value in cut_times)
    opening, visible_strides = _plan_internal_cut_windows(
        output_frames=output_frames,
        context_frames=context,
        maximum_frames=maximum,
        cut_frames=cut_frames,
    )

    geometry: list[tuple[int, int, int, int]] = []
    visible_cursor = 0
    for index in range(len(visible_strides) + 1):
        if index == 0:
            visible = opening
            local_context = 0
            context_start = 0
            window = opening
        else:
            visible = visible_strides[index - 1]
            local_context = context
            context_start = visible_cursor - context
            window = context + visible
        geometry.append((context_start, visible_cursor, visible, window))
        visible_cursor += visible
    if visible_cursor != output_frames:
        raise RuntimeError('internal-cut plan does not cover the output timeline')

    def state_at(seconds: float) -> dict[str, dict[str, Any]]:
        result = clone(story['initial_state'])
        for stop, updates in sorted(state_updates, key=lambda item: item[0]):
            if stop <= seconds + 1e-6:
                apply_updates(result, updates)
        return result

    definitions = '\n'.join(
        f"{entity_name(key)} [{entities[key]['kind']}]: {expand(entities[key]['identity'])}"
        for key in sorted(entities)
    )

    segments: list[WindowSegment] = []
    preview_windows: list[dict[str, Any]] = []
    cut_owners = {value: 0 for value in cut_times}
    for index, (context_start_frame, visible_start_frame, visible, window) in enumerate(geometry):
        visible_stop_frame = visible_start_frame + visible
        context_start_seconds = context_start_frame / FPS
        visible_start_seconds = visible_start_frame / FPS
        visible_stop_seconds = min(requested_cursor, visible_stop_frame / FPS)
        offset = (visible_start_frame - context_start_frame) / FPS
        local_window_stop = window / FPS
        active_entries = [
            entry for entry in timeline
            if entry['start'] < visible_stop_seconds - 1e-6
            and entry['stop'] > visible_start_seconds + 1e-6
        ]
        owned_cuts = [
            value for value in cut_times
            if visible_start_seconds - 1e-6 <= value < visible_stop_seconds - 1e-6
        ]
        for value in owned_cuts:
            cut_owners[value] += 1

        description: list[str] = [
            expand(story['overview']),
            'Stable entity identities for this complete local inference window:',
            definitions,
            (
                f'This H3 target runs from local 00:00.000 through {clock(local_window_stop)}. '
                f'It contributes global {clock(visible_start_seconds)} through '
                f'{clock(visible_stop_seconds)} to the delivered movie.'
            ),
        ]
        if index:
            description.append(
                f'Local 00:00.000 through {clock(offset)} is the exact carried joint '
                'audio-video latent history from the preceding physical window. It is '
                'authoritative and already happened. At '
                f'{clock(offset)}, generate the physically adjacent next frame with the '
                'same active camera, people, objects, motion, lighting and room sound. '
                'Do not restart any action or sound contained in the carried history.'
            )
        else:
            description.append('This is the opening physical window and has no earlier latent history.')
        description.extend((
            'Authoritative world state at the first newly generated frame:',
            state_lines(state_at(visible_start_seconds)),
            'Camera and shot schedule for newly generated frames:',
        ))

        local_shot_number = 1
        previous_semantic_shot: str | None = None
        for entry in active_entries:
            slice_start = max(visible_start_seconds, entry['start'])
            slice_stop = min(visible_stop_seconds, entry['stop'])
            if slice_stop <= slice_start + 1e-6:
                continue
            shot = entry['shot']
            segment = entry['segment']
            entry_starts_here = abs(entry['start'] - slice_start) < 1e-6
            is_cut = bool(
                entry['shot_index'] > 0
                and entry['segment_index'] == 0
                and entry_starts_here
            )
            is_same_shot_phase = bool(
                entry['segment_index'] > 0 and entry_starts_here
            )
            local_start = slice_start - context_start_seconds
            local_stop = slice_stop - context_start_seconds
            new_local_shot = previous_semantic_shot != shot['id']
            shot_label = f'[Shot {local_shot_number}] ' if new_local_shot else ''
            if is_cut:
                description.append(
                    f'{shot_label}At exactly {clock(local_start)}, make one '
                    f'instantaneous hard camera cut to {expand(entry["composition"])} '
                    'The cut changes only the camera position, direction, lens and framing. '
                    'Story time does not jump. The room, people, object count, object '
                    'placement, poses, contacts, lighting and continuous ambience remain at '
                    'the identical physical instant. State at the cut: '
                    + state_lines(state_at(slice_start)).replace('\n', ' | ')
                    + f'. Hold the new composition without action for '
                    f'{segment["establish_seconds"]:.3f} seconds. Camera invariants: '
                    f'{expand(shot["camera_style"])}'
                )
            elif index == 0 and entry['shot_index'] == 0 and entry['segment_index'] == 0:
                description.append(
                    f'{shot_label}From local {clock(local_start)}, establish '
                    f'{expand(entry["composition"])} Camera invariants: '
                    f'{expand(shot["camera_style"])}'
                )
            elif is_same_shot_phase:
                description.append(
                    f'At local {clock(local_start)}, continue the same semantic shot '
                    f'{shot["id"]} into its next action phase with no camera cut, dissolve, '
                    'reset or re-establishing view. Preserve the camera position, direction, '
                    'lens, framing and instantaneous motion continuously from the preceding '
                    'interval. Authoritative state at this action-phase boundary: '
                    + state_lines(state_at(slice_start)).replace('\n', ' | ')
                    + f'. Camera invariants: {expand(shot["camera_style"])}'
                )
            else:
                description.append(
                    f'{shot_label}From local {clock(local_start)} through {clock(local_stop)}, strictly '
                    f'continue semantic shot {shot["id"]} from the carried physical state. '
                    'The exact carried frames and the authoritative current state above own '
                    'all present character poses and object placements. Preserve their camera '
                    'projection without reconstructing the authored shot opening. Camera '
                    f'invariants: {expand(shot["camera_style"])}'
                )
            description.append(
                f'Across local {clock(local_start)} through {clock(local_stop)}, camera '
                f'motion is: {expand(segment["camera"]["motion"])} Required camera state '
                f'when this authored interval completes: {expand(segment["camera"]["end"])}'
            )
            if new_local_shot:
                local_shot_number += 1
                previous_semantic_shot = shot['id']

        description.append('Window-local actions and dialogue:')
        local_dialogue_frames: list[int] = []
        local_dialogue_count = 0
        ordered_events = sorted(
            (
                event
                for entry in active_entries
                for event in entry.get('events', [])
                if event['start'] < visible_stop_seconds - 1e-6
                and event['stop'] > visible_start_seconds + 1e-6
            ),
            key=lambda event: (event['start'], event['kind']),
        )
        if not ordered_events:
            description.append('No new authored action or dialogue occurs; continue the visible physical motion naturally.')
        for event in ordered_events:
            local_start = max(event['start'], visible_start_seconds) - context_start_seconds
            local_stop = min(event['stop'], visible_stop_seconds) - context_start_seconds
            if event['kind'] == 'beat':
                beat = event['beat']
                if event['start'] < visible_start_seconds - 1e-6:
                    prefix = (
                        f'At local {clock(offset)}, this action is already underway in the '
                        'carried frames. Continue its instantaneous motion without restarting '
                        f'its beginning, through local {clock(local_stop)}: '
                    )
                else:
                    prefix = (
                        f'From local {clock(local_start)} through {clock(local_stop)}: '
                    )
                sentence = prefix + expand(beat['text'])
                if event['stop'] <= visible_stop_seconds + 1e-6:
                    if beat['state_updates']:
                        sentence += (
                            ' At the end, apply only these state updates: '
                            + update_text(beat['state_updates']) + '.'
                        )
                else:
                    sentence += (
                        ' This action remains in progress at the physical-window end; do '
                        'not jump to its later result or finish it early.'
                    )
                description.append(sentence)
            elif event['start'] >= visible_start_seconds - 1e-6:
                line = event['line']
                description.append(
                    f'Between local {clock(event["start"] - context_start_seconds)} and '
                    f'{clock(event["stop"] - context_start_seconds)}, '
                    f'{entity_name(line["speaker"])} says once: '
                    f'<d>[{line["language"]}] {line["text"]}</d> The complete line '
                    'begins and ends inside this interval.'
                )
                local_dialogue_count += 1
                local_dialogue_frames.append(round(event['start'] * FPS))

        end_state = state_at(visible_stop_seconds)
        description.extend((
            'Required authoritative world state at this physical window end:',
            state_lines(end_state),
            (
                f'Speak only the {local_dialogue_count} explicitly specified dialogue '
                'line(s), once each.'
                if local_dialogue_count else
                'No speech, narration, singing or other human voice occurs in the newly generated interval.'
            ),
        ))

        active_shots: list[dict[str, Any]] = []
        active_shot_intervals: list[tuple[dict[str, Any], float, float]] = []
        for entry in active_entries:
            if all(item['id'] != entry['shot']['id'] for item in active_shots):
                active_shots.append(entry['shot'])
                matching = [
                    value for value in timeline
                    if value['shot']['id'] == entry['shot']['id']
                ]
                shot_start = min(value['start'] for value in matching)
                shot_stop = max(value['stop'] for value in matching)
                active_shot_intervals.append((
                    entry['shot'],
                    max(visible_start_seconds, shot_start),
                    min(visible_stop_seconds, shot_stop),
                ))
        sound_rows = []
        music_rows = []
        for active_shot, interval_start, interval_stop in active_shot_intervals:
            sound_rows.append(
                f'From local {clock(interval_start - context_start_seconds)} through '
                f'{clock(interval_stop - context_start_seconds)}: '
                f'{expand(active_shot["overall_soundscape"])}'
            )
            music_rows.append(
                f'From local {clock(interval_start - context_start_seconds)} through '
                f'{clock(interval_stop - context_start_seconds)}: '
                f'{expand(active_shot["non_diegetic_music"])}'
            )
        soundscape = ' '.join(dict.fromkeys(sound_rows))
        if owned_cuts:
            soundscape += (
                ' The camera cut does not restart, reverse, duplicate, mute or change the '
                'level of continuous room ambience. Only an explicitly authored visible '
                'sound event may change the soundtrack.'
            )
        else:
            soundscape += (
                ' Continue ambience from the carried audio at unchanged loudness, stereo '
                'position, timbre and reverberation. Do not replay earlier event sounds.'
            )
        music = ' '.join(dict.fromkeys(music_rows)) or 'N/A'
        prompt = (
            'integrated_multimodal_description:\n'
            + '\n'.join(description)
            + '\n\noverall_soundscape: ' + soundscape
            + '\n\nnon_diegetic_music: ' + music
        )
        if first_frame and index == 0:
            prompt = (
                'For the target video, at 0.00 seconds into the target video, '
                '<Picture 1> (from [Shot 1]) is fully referenced.\n\n' + prompt
            )
        if last_frame and index == len(geometry) - 1:
            prompt = (
                'How the reference pictures align with the target video — '
                f'<Picture 1> (from [Shot 1]) aligns with the '
                f'{window / FPS:.2f}-second mark of the target video.\n\n' + prompt
            )
        if len(prompt) > 20000:
            raise ValueError(
                f'compiled physical window {index + 1} exceeds 20000 characters'
            )
        if ENTITY.search(prompt):
            raise RuntimeError('compiled physical prompt contains an unresolved entity')

        segments.append(WindowSegment(
            index=index,
            window_frames=window,
            context_frames=0 if index == 0 else context,
            global_context_start_frame=context_start_frame,
            visible_start_frame=visible_start_frame,
            visible_frames=visible,
            seed=_derived_seed(seed, index),
            prompt=prompt,
            transition='opening' if index == 0 else 'continue',
            video_prefix_frames=None if index == 0 else context,
            visual_memory_floor_frame=None,
            visual_memory_include_canonical=False,
            preserve_latest_visual=index < len(geometry) - 1,
            authorized_dialogue_count=local_dialogue_count,
            authorized_dialogue_frames=tuple(local_dialogue_frames),
            reference_audio_active=bool(local_dialogue_count),
            audio_memory_active=bool(local_dialogue_count),
        ))
        preview_windows.append({
            'index': index,
            'physical_transition': 'opening' if index == 0 else 'continue',
            'global_context_start_seconds': context_start_seconds,
            'visible_start_seconds': visible_start_seconds,
            'visible_end_seconds': visible_stop_frame / FPS,
            'context_seconds': 0.0 if index == 0 else context / FPS,
            'window_seconds': window / FPS,
            'active_shot_ids': [item['id'] for item in active_shots],
            'semantic_cuts': [
                {
                    'global_seconds': value,
                    'local_seconds': value - context_start_seconds,
                }
                for value in owned_cuts
            ],
            'prompt': prompt,
            'visual_memory_policy': 'bounded_global_coreset_plus_latest',
        })

    if any(owner != 1 for owner in cut_owners.values()):
        raise RuntimeError('every authored camera cut must belong to exactly one physical window')
    for segment in segments[1:]:
        seam = segment.visible_start_frame
        for cut in cut_frames:
            if seam - context <= cut <= seam:
                raise RuntimeError('a physical overlap crosses an authored camera cut')

    plan = WindowPlan(
        requested_duration_seconds=requested_cursor,
        output_frames=output_frames,
        actual_duration_seconds=output_frames / FPS,
        segments=tuple(segments),
        context_frames=context,
        mechanism='authored_internal_cut_joint_av_v3',
        planning_policy='strict_continuation_internal_cut_event_aware_v1',
        structured_director=True,
    )
    return plan, {
        'version': 3,
        'mapping_policy': 'semantic_shots_many_to_many_physical_windows_v1',
        'requested_controls': {
            key: data[key] for key in ('overlap_seconds', 'max_window_seconds', 'memory')
        },
        'effective_overlap_seconds': context / FPS,
        'effective_max_window_seconds': maximum / FPS,
        'actual_duration_seconds': plan.actual_duration_seconds,
        'semantic_cut_seconds': cut_times,
        'windows': preview_windows,
        'reference_check': (
            'Every physical prompt contains stable identities, current state, camera, '
            'owned actions, dialogue, sound and end state. Every physical seam is a '
            'same-shot exact-latent continuation; cuts occur only inside one owner window.'
        ),
    }


def _compile_window_story_v2(
    data: dict[str, Any],
    *,
    seed: int,
    maximum_frames: int,
    service_family: str,
    first_frame: bool,
    last_frame: bool,
):
    """Compile state-safe local prompts without replaying a shot-wide trajectory."""
    story = data['story']
    maximum = min(
        maximum_frames,
        5 + 17 * math.floor((24 * data['max_window_seconds'] - 5 + 1e-8) / 17),
    )
    context = min(90, _align_h3_frames(round(data['overlap_seconds'] * 24)))
    if context >= maximum:
        raise ValueError('effective overlap must be shorter than the maximum window')
    entities = story['entities']
    initial_state = json.loads(json.dumps(story['initial_state']))
    state = json.loads(json.dumps(initial_state))
    authored_segments = [
        (shot, segment)
        for shot in story['shots']
        for segment in shot['segments']
    ]
    multiple_shots = len(story['shots']) > 1
    # Resolve semantic camera-anchor identifiers to the exact snapped output
    # coordinates used by memory. This is a structural lookup, never prompt
    # parsing or scene inference.
    authored_frame_intervals: dict[tuple[str, str], tuple[int, int]] = {}
    interval_requested_cursor = 0.0
    interval_actual_cursor = 0
    for interval_shot in story['shots']:
        for interval_segment in interval_shot['segments']:
            interval_start = interval_actual_cursor
            interval_requested_cursor += interval_segment['duration_seconds']
            interval_actual_cursor = _align_h3_frames(
                round(interval_requested_cursor * FPS)
            )
            authored_frame_intervals[
                (interval_shot['id'], interval_segment['id'])
            ] = (interval_start, interval_actual_cursor)
    speaker_order: list[str] = []
    for _, segment in authored_segments:
        for line in segment['dialogue']:
            if line['speaker'] not in speaker_order:
                speaker_order.append(line['speaker'])
    speaker_ids = {
        speaker: f'S{index + 1}' for index, speaker in enumerate(speaker_order)
    }
    referenced_entities = {
        key
        for key, value in entities.items()
        if any(
            kind.lower() in ('picture', 'video')
            for kind, _ in MEDIA.findall(value['identity'])
        )
        or (
            key in speaker_ids
            and any(
                kind.lower() == 'audio'
                for kind, _ in MEDIA.findall(value['identity'])
            )
        )
    }
    subject_ids = {
        key: f'<Subject {index + 1}>'
        for index, key in enumerate(sorted(referenced_entities))
    }

    def expand(value: str, *, audio_active: bool = True) -> str:
        rendered = ENTITY.sub(
            lambda match: (
                subject_ids.get(match.group(1), match.group(1))
                if service_family == 'reference'
                else match.group(1)
            ),
            value,
        )
        if not audio_active:
            rendered = re.sub(
                r'<Audio\s+\d+>',
                'the established voice identity',
                rendered,
                flags=re.I,
            )
        return rendered

    def entity_name(key: str) -> str:
        name = subject_ids.get(key, key) if service_family == 'reference' else key
        return f'{name} ({speaker_ids[key]})' if key in speaker_ids else name

    def clock(seconds: float) -> str:
        return f'{int(seconds // 60):02d}:{seconds % 60:06.3f}'

    def state_value(value: Any, *, audio_active: bool) -> str:
        if isinstance(value, str):
            return expand(value, audio_active=audio_active)
        if value is True:
            return 'true'
        if value is False:
            return 'false'
        if value is None:
            return 'null'
        return str(value)

    def state_lines(
        state_map: dict[str, dict[str, Any]],
        keys: set[str],
        *,
        audio_active: bool,
        omit_character_spatial: bool = False,
    ) -> str:
        rows = []
        for key in sorted(keys):
            omitted = (
                {'location', 'pose', 'visibility'}
                if omit_character_spatial and entities[key]['kind'] == 'character'
                else set()
            )
            values = '; '.join(
                f'{field}={state_value(value, audio_active=audio_active)}'
                for field, value in sorted(state_map[key].items())
                if field not in omitted
            )
            rows.append(f'{entity_name(key)}: {values}')
        return '\n'.join(rows)

    def update_lines(
        updates: dict[str, dict[str, Any]],
        *,
        audio_active: bool,
    ) -> str:
        rows = []
        for key in sorted(updates):
            values = '; '.join(
                f'{field}={state_value(value, audio_active=audio_active)}'
                for field, value in sorted(updates[key].items())
            )
            rows.append(f'{entity_name(key)}: {values}')
        return ' | '.join(rows)

    def apply_updates(
        target: dict[str, dict[str, Any]],
        updates: dict[str, dict[str, Any]],
    ) -> None:
        for key, values in updates.items():
            target[key].update(values)

    def mentioned_entities(value: str) -> set[str]:
        """Close local prompts even when an author omitted ``{{...}}`` once.

        Braced references remain the unambiguous authoring form. Bare entity
        identifiers are also admitted as dependencies so a local window never
        mentions a globally defined prop while silently omitting its identity
        and current state.
        """

        found = set(ENTITY.findall(value))
        found.update(
            key
            for key in entities
            if re.search(
                rf'(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])',
                value,
            )
        )
        return found

    segments: list[WindowSegment] = []
    preview: list[dict[str, Any]] = []
    requested_cursor = 0.0
    actual_cursor = 0
    camera_end: str | None = None
    for shot, segment in authored_segments:
        index = len(segments)
        transition = segment['transition']
        anchor_spec = shot.get('camera_anchor') if transition == 'cut' else None
        layout_anchor_interval = (
            authored_frame_intervals[
                (anchor_spec['shot_id'], anchor_spec['segment_id'])
            ]
            if anchor_spec is not None else
            None
        )
        opening_preroll = H3_FRAME_STRIDE if not index and multiple_shots else 0
        overlap = context if index else 0
        novel_camera_cut = bool(
            index and transition == 'cut' and layout_anchor_interval is None
        )
        terminal_video_seed = False
        # A whole historical frame carries its source-camera projection along
        # with scene appearance.  V34 showed that injecting even one such row
        # into an unseen target camera preserves the old projection instead
        # of the room's view-independent geometry.  Keep novel-camera targets
        # text/state driven until a target-view-matched anchor exists.
        novel_camera_layout_probe = False
        video_prefix = (
            None if not index else
            H3_FRAME_ORIGIN if terminal_video_seed else
            0 if transition == 'cut' else
            overlap if multiple_shots else
            _continuation_video_prefix_frames(overlap)
        )
        offset = (overlap + opening_preroll) / FPS
        audio_active = bool(segment['dialogue'])
        requested_stop = requested_cursor + segment['duration_seconds']
        stop = _align_h3_frames(round(requested_stop * FPS))
        visible = stop - actual_cursor
        window = visible + overlap + opening_preroll
        if overlap > actual_cursor:
            raise ValueError(
                f"{shot['id']}/{segment['id']}: overlap exceeds available history"
            )
        if visible <= 0 or window > maximum or window < 5 or (window - 5) % 17:
            raise ValueError(
                f"{shot['id']}/{segment['id']}: new content plus effective overlap "
                f"requires {window / FPS:.3f}s, maximum is {maximum / FPS:.3f}s; "
                'shorten the segment or change the window controls'
            )
        visible_seconds = visible / FPS
        for beat in segment['beats']:
            if beat['end_seconds'] > visible_seconds + 1e-9:
                raise ValueError(
                    f"{shot['id']}/{segment['id']}: snapped new-content interval is "
                    f"{visible_seconds:.3f}s but a beat ends at {beat['end_seconds']:.3f}s"
                )
        for line in segment['dialogue']:
            if line['end_seconds'] > visible_seconds + 1e-9:
                raise ValueError(
                    f"{shot['id']}/{segment['id']}: snapped new-content interval is "
                    f"{visible_seconds:.3f}s but dialogue ends at {line['end_seconds']:.3f}s"
                )

        relevant_text = [
            story['overview'],
            shot['camera_style'],
            shot['overall_soundscape'],
            shot['non_diegetic_music'],
            *segment['camera'].values(),
            *(beat['text'] for beat in segment['beats']),
        ]
        needed = {
            key
            for key, value in entities.items()
            if value['kind'] in ('character', 'location')
        }
        needed.update(line['speaker'] for line in segment['dialogue'])
        for beat in segment['beats']:
            needed.update(beat['state_updates'])
            for values in beat['state_updates'].values():
                relevant_text.extend(
                    value for value in values.values() if isinstance(value, str)
                )
        for value in relevant_text:
            needed.update(mentioned_entities(value))
        while True:
            old = set(needed)
            for key in old:
                needed.update(mentioned_entities(entities[key]['identity']))
                for value in state[key].values():
                    if isinstance(value, str):
                        needed.update(mentioned_entities(value))
            if needed == old:
                break

        next_state = json.loads(json.dumps(state))
        for beat in segment['beats']:
            apply_updates(next_state, beat['state_updates'])
        definitions = '\n'.join(
            f"{entity_name(key)} [{entities[key]['kind']}]: "
            f"{expand(entities[key]['identity'], audio_active=audio_active)}"
            for key in sorted(needed)
        )
        starting = state_lines(
            state,
            needed,
            audio_active=audio_active,
            omit_character_spatial=(
                transition == 'continue'
                or (transition == 'cut' and not novel_camera_cut)
            ),
        )
        ending = state_lines(next_state, needed, audio_active=audio_active)

        establish = segment['establish_seconds']
        camera = segment['camera']
        style = expand(shot['camera_style'], audio_active=audio_active)
        motion = expand(camera['motion'], audio_active=audio_active)
        end_camera = expand(camera['end'], audio_active=audio_active)
        if transition == 'opening':
            camera_instruction = (
                f"[Shot 1] At local 00:00.000 establish "
                f"{expand(camera['composition'], audio_active=audio_active)} "
                + (
                    f"The first delivered frame at local {clock(offset)} must already use "
                    "this complete camera position, direction, focal length and framing; "
                    "any hidden setup frames before it are discarded, and no delivered "
                    "insert or close-up may precede the master. "
                    if multiple_shots else ''
                )
                + f"Camera invariants: {style} During this new interval only: {motion} "
                f"Required camera state at the interval end: {end_camera} Keep this one camera "
                "position and one continuous composition for every frame."
            )
        elif transition == 'cut':
            if video_prefix is None:
                raise RuntimeError('cut transition has no preceding video prefix')
            settle_end = offset + establish
            seed_end = video_prefix / FPS
            camera_transfer_end = max(
                seed_end,
                (overlap - H3_FRAME_STRIDE) / FPS,
            ) if terminal_video_seed else 0.0
            if novel_camera_cut and terminal_video_seed:
                camera_instruction = (
                    f"[Shot 1] Local 00:00.000 through {clock(seed_end)} is the exact carried "
                    "terminal image and physical tableau. From that point through "
                    f"{clock(camera_transfer_end)}, continue with one uninterrupted, physically "
                    "continuous camera relocation around the stationary tableau. Translate and "
                    "rotate smoothly along the shortest clear path from the carried camera to "
                    f"{expand(camera['composition'], audio_active=audio_active)} "
                    "Keep the room as one connected three-dimensional space while perspective "
                    "changes: every wall, doorway, window, counter, table, person and object keeps "
                    "its identity, relative position, scale and contact. At "
                    f"{clock(camera_transfer_end)}, arrive exactly at the declared camera position, "
                    "direction, lens and composition. From that instant through the final frame, "
                    "use only this settled camera projection. "
                    f"Camera invariants after arrival: {style} From arrival onward: {motion} "
                    f"Required camera state at the interval end: {end_camera} Keep the physical "
                    f"tableau unchanged until {clock(settle_end)}."
                )
            elif novel_camera_cut:
                camera_instruction = (
                    "[Shot 1] From local 00:00.000, begin directly in "
                    f"{expand(camera['composition'], audio_active=audio_active)} "
                    "This complete local target is one uninterrupted shot photographed from this "
                    "single camera setup. Use this camera position, direction, lens and composition "
                    "from the first frame through the final frame. Reconstruct the authored room, one "
                    "person and every unique object from the complete authoritative world state below. "
                    "Keep the stated location, body pose, gaze, hand pose, prop contact and object placement. "
                    f"Hold this camera and physical tableau without action until {clock(settle_end)}. "
                    f"Camera invariants: {style} During the remaining interval: {motion} "
                    f"Required camera state at the interval end: {end_camera} Preserve the same camera "
                    "projection throughout the complete local target."
                )
            else:
                camera_instruction = (
                    "[Shot 1] From local 00:00.000, begin directly in "
                    f"{expand(camera['composition'], audio_active=audio_active)} "
                    "This complete local target is one uninterrupted shot photographed from this "
                    "single camera setup. The declared camera-anchor memory band supplies only this "
                    "camera position, lens, framing and stable room geometry. The newest terminal "
                    "visual-memory frame supplies only the same-instant character pose, hand and prop "
                    "contact, and current object placement. Reproject that physical tableau from the "
                    "declared camera and keep one instance of every person and object. "
                    f"Hold this camera and tableau without action until {clock(settle_end)}. "
                    f"Camera invariants: {style} During the remaining interval: {motion} "
                    f"Required camera state at the interval end: {end_camera} Preserve the same camera "
                    "projection throughout the complete local target."
                )
        else:
            if camera_end is None:
                raise RuntimeError('continue transition has no previous camera state')
            camera_instruction = (
                '[Shot 1] The protected carried video is the sole starting camera boundary. '
                'Continue its exact position, lens, direction and velocity; do not reconstruct '
                'a camera state from any earlier textual plan. '
                f'Camera invariants: {style} During this new interval only: {motion} '
                'Represent the entire camera change as one rate-limited continuous path: every '
                'adjacent frame must differ only by physically plausible small translation and '
                'rotation. The target view may be reached only through motion visibly accumulated '
                'across this interval; it must never appear suddenly. Keep the subject and scene '
                'continuously readable. Never let a wall, curtain, doorway edge, darkness, body, '
                'foreground object or motion blur fill the frame or conceal an edit. '
                f'Required camera state at the interval end: {end_camera} '
                'Do not replay the shot opening or any earlier camera path. No cut, reset, teleport or morph.'
            )

        timed_events: list[tuple[float, str]] = []
        beats = segment['beats']
        for beat_index, beat in enumerate(beats):
            start = offset + beat['start_seconds']
            end = offset + beat['end_seconds']
            update = update_lines(beat['state_updates'], audio_active=audio_active)
            state_suffix = (
                f' At {clock(end)}, apply only these authoritative state updates: {update}.'
                if update else
                ' Preserve all authoritative state properties at the end of this beat.'
            )
            if transition == 'continue':
                boundary_prefix = (
                    f'At {clock(start)}, there is no edit, time skip, camera reset or new '
                    'composition. Continue directly from the physically adjacent preceding '
                    'frame with the same camera pose, lens, framing, screen direction, room '
                    'geometry, lighting, subject identity and instantaneous pose. '
                )
                next_phase_suffix = (
                    f' The boundary at {clock(end)} is only an action checkpoint inside the '
                    'same uninterrupted shot. The last frame before it and the first frame '
                    'after it must be physically adjacent; preserve the exact camera state, '
                    'composition, scene geometry and screen direction while motion flows '
                    f'continuously into phase {beat_index + 2}.'
                    if beat_index + 1 < len(beats) else
                    ''
                )
                event = (
                    f'{boundary_prefix}Continuous action phase {beat_index + 1}/{len(beats)} '
                    f'inside this same local shot, from {clock(start)} through {clock(end)}: '
                    f"{expand(beat['text'], audio_active=audio_active)}"
                    f'{state_suffix}{next_phase_suffix}'
                )
            elif transition == 'cut':
                held_since = (
                    camera_transfer_end if terminal_video_seed else 0.0
                )
                boundary_prefix = (
                    f'At {clock(start)}, continue the identical locked camera position, lens, '
                    'framing, screen direction and room projection held since local 00:00.000. '
                    'This action checkpoint changes only the explicitly described subject or object '
                    'state. Any camera or view wording inside the authored action below describes '
                    'the already-held composition and leaves it unchanged. '
                )
                if terminal_video_seed:
                    boundary_prefix = (
                        f'At {clock(start)}, continue the identical locked camera position, lens, '
                        'framing, screen direction and room projection held since '
                        f'{clock(held_since)}. This action checkpoint changes only the explicitly '
                        'described subject or object state. Any camera or view wording inside the '
                        'authored action below describes the already-held composition and leaves '
                        'it unchanged. '
                    )
                next_phase_suffix = (
                    f' At {clock(end)}, preserve the identical camera and room projection while '
                    'the physical action flows into the next phase.'
                    if beat_index + 1 < len(beats) else
                    ''
                )
                event = (
                    f'{boundary_prefix}Action phase {beat_index + 1}/{len(beats)} from '
                    f'{clock(start)} through {clock(end)}: '
                    f"{expand(beat['text'], audio_active=audio_active)}"
                    f'{state_suffix}{next_phase_suffix}'
                )
            else:
                event = (
                    f'From {clock(start)} through {clock(end)}, '
                    f"{expand(beat['text'], audio_active=audio_active)}{state_suffix}"
                )
            timed_events.append((beat['start_seconds'], event))
        dialogue_frames: list[int] = []
        for line in segment['dialogue']:
            start = offset + line['start_seconds']
            end = offset + line['end_seconds']
            timed_events.append((line['start_seconds'], (
                f'Between {clock(start)} and {clock(end)}, {entity_name(line["speaker"])} '
                f'says once: <d>[{line["language"]}] {line["text"]}</d> '
                'The complete line must begin and end inside this interval.'
            )))
            dialogue_frames.append(
                actual_cursor + round(line['start_seconds'] * FPS)
            )
        authored_end = max(
            [beat['end_seconds'] for beat in segment['beats']]
            + [line['end_seconds'] for line in segment['dialogue']]
        )
        if visible_seconds - authored_end >= 0.25:
            timed_events.append((authored_end, (
                f'From {clock(offset + authored_end)} through {clock(offset + visible_seconds)}, '
                + (
                    'hold the required character, object and camera states unchanged through the '
                    'final frame of this local target.'
                    if transition == 'cut' else
                    'hold the required segment-end character, object and camera states. Do not advance '
                    'into the next segment, begin its action, or move beyond this camera endpoint.'
                )
            )))
        speech = (
            f'Speak only the {len(dialogue_frames)} specified dialogue lines, once each.'
            if dialogue_frames
            else 'No speech, narration, singing or other human voice in the newly generated interval.'
        )
        changed_props = any(
            entities[key]['kind'] == 'prop' and state[key] != initial_state[key]
            for key in needed
        )
        # A canonical opening frame contradicts the immediate camera state in a
        # continuous take, so continuations remain state-local. A same-scene cut
        # needs that frame's room geometry. If mutable props have changed, route
        # it only during coarse denoising and converge on the recent state later.
        include_canonical = bool(
            index and transition == 'cut' and layout_anchor_interval is not None
        )
        progressive_layout_state = bool(
            include_canonical and (changed_props or layout_anchor_interval is not None)
        )
        visual_policy = (
            'cut_explicit_camera_anchor_then_latest_state'
            if layout_anchor_interval is not None else
            'cut_terminal_token_continuous_camera_relocation'
            if terminal_video_seed else
            'cut_target_camera_layout_probe_then_text'
            if novel_camera_layout_probe else
            'cut_text_world_state_no_visual_refs'
            if novel_camera_cut else
            'cut_progressive_layout_then_latest_state'
            if progressive_layout_state else
            'cut_identity_anchor_plus_latest_state'
            if include_canonical else
            'continuous_latest_state_only' if index and transition == 'continue' else
            'latest_state_only' if index else 'collect_opening_state'
        )
        memory_authority = (
            'Memory authority: the protected carried video is authoritative for immediate character '
            'location, pose, visibility and camera continuity. For a continuation, those predicted '
            'character fields are deliberately omitted from the starting symbolic state below. '
            if transition == 'continue' else
            'Local authority: the declared camera-anchor memory band supplies only the camera '
            'position, lens, framing and stable room geometry. The newest terminal memory view '
            'supplies only the same-instant character pose, hand and prop contact, and current '
            'object placement. The camera definition below owns the projection for every frame. '
            if transition == 'cut' and layout_anchor_interval is not None else
            'Local authority: historical full-frame views do not define this local camera, composition '
            'or framing. Canonical memory supplies only stable room geometry and identity; the newest '
            'terminal memory view alone is authoritative for the exact same-instant '
            'character pose, hand/prop contact and current object placement. '
            if transition == 'cut' and progressive_layout_state else
            'Transport authority: the exact carried terminal target-video token owns the initial '
            'camera, room geometry, character pose, hand and prop contact at local 00:00.000. The '
            'complete world state below keeps that physical tableau fixed while the camera relocates '
            f'continuously. The declared target camera owns the projection from {clock(camera_transfer_end)} '
            'through the final frame. No full-frame visual-memory row is active. '
            if transition == 'cut' and terminal_video_seed else
            'Local authority: the complete current world state below owns room topology, character '
            'identity, physical pose, hand and prop contact, and object placement. The declared camera '
            'position, direction, lens and composition own the projection for every frame. One coarse '
            'scene-memory observation may supply material appearance and connected-room identity only; '
            'it does not own camera position, framing, subject count or current object placement. '
            'Instantiate exactly one copy of each defined person and object. '
            if transition == 'cut' and novel_camera_layout_probe else
            'Local authority: the complete current world state below owns room topology, character '
            'identity, physical pose, hand and prop contact, and object placement. The declared camera '
            'position, direction, lens and composition own the projection for every frame. Instantiate '
            'exactly one copy of each defined person and object. '
            if transition == 'cut' and novel_camera_cut else
            'Local authority: historical full-frame views do not define this local camera or framing. '
            'The newest terminal memory view is authoritative for the exact same-instant '
            'character pose, hand/prop contact and current object placement; older views carry only stable '
            'room geometry and identity. '
            if transition == 'cut' else
            'Memory authority: this opening window has no earlier visual memory and begins only from '
            'the explicit world state below. '
        ) + ('' if transition == 'opening' or novel_camera_cut else (
            'Any nonterminal earlier visual-memory frame is evidence for stable character identity or room '
            'layout only; its object positions, holders, visibility and switch states may be obsolete. '
            + (
                'The logical world state below constrains subsequent action and the interval endpoint, but '
                'must not overwrite the newest terminal view at local 00:00.000. '
                if transition == 'cut' else
                'The current authoritative world state below overrides every nonterminal older memory view. '
            )
            + 'Never restore an object to an earlier state merely because it appears there in memory.'
        ))
        camera_first = bool(multiple_shots and transition in ('opening', 'cut'))
        description = (
            expand(story['overview'], audio_active=audio_active)
            + (('\n' + camera_instruction) if camera_first else '')
            + '\n' + memory_authority
            + '\nStable entity identity (appearance only, never current location or state):\n' + definitions
            + (
                (
                    f'\nThis local target runs from 00:00.000 through {clock(window / FPS)}. '
                    f'The exact carried terminal state occupies 00:00.000–{clock(seed_end)}; one '
                    f'continuous camera relocation occupies {clock(seed_end)}–{clock(camera_transfer_end)}; '
                    f'the target camera is settled from {clock(camera_transfer_end)} through the final '
                    f'frame. The physical tableau remains held until {clock(settle_end)}. '
                    if terminal_video_seed else
                    f'\nThis local target runs from 00:00.000 through {clock(window / FPS)} with '
                    'the same camera projection throughout. The camera and starting physical tableau '
                    f'are fully established at 00:00.000 and remain held until {clock(settle_end)}. '
                )
                if transition == 'cut' else
                f'\nThis local window starts at 00:00.000. Its AV transition context spans '
                f'00:00.000–{clock(offset)}; deliver new content only from {clock(offset)} '
                f'to {clock(window / FPS)}. '
            )
            + (
                '' if transition == 'cut' else
                (
                    (
                        f'The complete {video_prefix / FPS:.3f} seconds of context are an exact protected '
                        'history anchor. No accepted predecessor frame is repainted or replaced. Generate '
                        'the first new frame as its physically adjacent continuation, and do not begin the '
                        'authored camera path, action, destination or dialogue before '
                        f'{clock(offset)}.\n'
                    )
                    if video_prefix == overlap else
                    (
                        f'The first {video_prefix / FPS:.3f} seconds of context are an exact protected '
                        'history anchor. The following '
                        f'{(overlap - video_prefix) / FPS:.3f} seconds, through {clock(offset)}, are a '
                        'writable continuation lead-in at the same timeline positions. At assembly it '
                        'replaces the matching provisional predecessor tail without changing duration. '
                        'Across that hidden lead-in, generate only physically adjacent frames of the '
                        'same take and preserve the immediate carried camera, subject pose and visible '
                        'geometry. Do not begin the authored camera path, action, destination or dialogue '
                        f'before {clock(offset)}.\n'
                    )
                )
                if transition == 'continue' else
                (
                    f'This opening window has {opening_preroll / FPS:.3f} seconds of hidden '
                    f'camera-settle preroll. Deliver nothing before {clock(offset)}; the first '
                    'delivered frame begins the authored timeline at global 00:00.000.\n'
                    if opening_preroll else
                    'This opening target begins without carried AV context.\n'
                )
            )
            + ('' if camera_first else camera_instruction)
            + '\nCurrent authoritative world state at the start of new content:\n' + starting
            + (
                '\nContinuous trajectory checkpoints inside this same local shot:\n'
                'Every entry below is a timing checkpoint along one uninterrupted camera '
                'trajectory, never a new shot, montage panel or re-establishing view. A '
                'timestamp boundary may change only the described action state. It must '
                'not discontinuously change camera pose, lens, framing, screen direction, '
                'room geometry, lighting, subject identity or physical pose. Any entity '
                'needed in a later phase must enter the existing view only through visible '
                'continuous camera, subject or object motion; never cut to reveal it.\n'
                if transition == 'continue' else
                '\nTime-ranged beats for this interval only:\n'
            )
            + '\n'.join(value for _, value in sorted(timed_events, key=lambda item: item[0]))
            + '\nRequired authoritative world state at this interval end:\n' + ending
            + '\n' + speech
        )
        soundscape = (
            expand(shot['overall_soundscape'], audio_active=audio_active)
            + (
                ' Maintain this ambience at stable loudness, stereo position, timbre and '
                'reverberation throughout the complete local target. Event sounds occur only '
                'inside their matching time-ranged beats.'
                if transition == 'cut' else
                ' Keep continuous ambience continuous across the carried boundary. '
                'Event sounds occur only inside their matching time-ranged beats. '
                'Do not replay earlier action sounds.'
            )
        )
        music = expand(shot['non_diegetic_music'], audio_active=audio_active)
        if index and shot['non_diegetic_music'] != 'N/A':
            music += ' Continue the existing musical passage without restarting it.'

        if service_family == 'reference':
            local_subjects = [
                key for key in sorted(needed) if key in subject_ids
            ]
            referenced_text = '\n'.join(
                [story['overview'], shot['camera_style'], shot['overall_soundscape'], shot['non_diegetic_music']]
                + list(segment['camera'].values())
                + [entities[key]['identity'] for key in sorted(needed)]
                + [beat['text'] for beat in segment['beats']]
            )
            media = sorted(
                {
                    (kind.title(), int(number))
                    for kind, number in MEDIA.findall(referenced_text)
                    if audio_active or kind.lower() != 'audio'
                },
                key=lambda item: (item[0], item[1]),
            )
            subject_lines = [
                f"{subject_ids[key]} is {expand(entities[key]['identity'], audio_active=audio_active)}"
                for key in local_subjects
            ]
            used_picture_numbers = {
                int(number)
                for key in local_subjects
                for kind, number in MEDIA.findall(entities[key]['identity'])
                if kind.lower() == 'picture'
            }
            for kind, number in media:
                if kind == 'Picture' and number not in used_picture_numbers:
                    subject_lines.append(
                        f'<Picture {number}> is a concrete visual reference cited in this local target window.'
                    )
                elif kind == 'Audio':
                    bound = [
                        key for key in local_subjects
                        if ('Audio', str(number)) in {
                            (found.title(), found_number)
                            for found, found_number in MEDIA.findall(entities[key]['identity'])
                        }
                    ]
                    if len(bound) == 1 and bound[0] in speaker_ids:
                        key = bound[0]
                        subject_lines.append(
                            f'<Audio {number}> is the voice-timbre reference for {subject_ids[key]} ({speaker_ids[key]}).'
                        )
                    else:
                        subject_lines.append(
                            f'<Audio {number}> is an authored audio reference for this local target window.'
                        )
            labels = [subject_ids[key] for key in local_subjects]
            summary = (
                '[reference generation] '
                + expand(story['overview'], audio_active=audio_active)
                + (f' This local window preserves {", ".join(labels)}.' if labels else '')
            )
            retention = [
                f"{subject_ids[key]} (identity only): fully_preserved - "
                f"{expand(entities[key]['identity'], audio_active=audio_active)}"
                for key in local_subjects
            ]
            retention.extend(
                f'<Audio {number}>: reference - use only voice timbre without copying words or timing.'
                for kind, number in media if kind == 'Audio'
            )
            prompt = (
                'subject_definitions:\n' + ('\n'.join(subject_lines) or 'No external reference label is active in this local window.')
                + '\n\nsummary:\n' + summary
                + '\n\nretention_analysis:\n' + ('\n'.join(retention) or 'No external reference retention applies in this local window.')
                + '\n\ndetailed_description:\n' + description
                + '\n\noverall_soundscape: ' + soundscape
                + '\n\nnon_diegetic_music: ' + music
            )
        else:
            prompt = (
                'integrated_multimodal_description:\n' + description
                + '\noverall_soundscape: ' + soundscape
                + '\n\nnon_diegetic_music: ' + music
            )
            instruction = None
            if len(authored_segments) == 1 and first_frame and last_frame:
                instruction = (
                    'How the reference pictures align with the target video — '
                    'Picture 1 (from Shot 1) aligns with the 0.00-second mark '
                    f'of the target video; Picture 2 (from Shot 1) aligns with '
                    f'the {window / FPS:.2f}-second mark of the target video.'
                )
            elif index == 0 and first_frame:
                instruction = (
                    'For the target video, at 0.00 seconds into the target video, '
                    '<Picture 1> (from [Shot 1]) is fully referenced.'
                )
            elif index == len(authored_segments) - 1 and last_frame:
                instruction = (
                    'How the reference pictures align with the target video — '
                    f'<Picture 1> (from [Shot 1]) aligns with the '
                    f'{window / FPS:.2f}-second mark of the target video.'
                )
            if instruction is not None:
                prompt = instruction + '\n\n' + prompt

        # V14 falsified sampler-side mixing of a second text-conditioned
        # velocity field: the temporal mask edges themselves decoded as hard
        # camera cuts. Keep one semantic trajectory per window. Continuity is
        # resolved after denoising by consensus between the two clean estimates
        # of the same hidden overlap.
        continuation_bridge_prompt: str | None = None
        if len(prompt) > 20000:
            raise ValueError('compiled window prompt exceeds 20000 characters')
        assert not ENTITY.search(prompt)
        segments.append(WindowSegment(
            index=index,
            window_frames=window,
            context_frames=overlap,
            global_context_start_frame=actual_cursor - overlap,
            visible_start_frame=actual_cursor,
            visible_frames=visible,
            seed=_derived_seed(seed, index),
            prompt=prompt,
            continuation_bridge_prompt=continuation_bridge_prompt,
            transition=transition,
            video_prefix_frames=video_prefix,
            visual_memory_floor_frame=(actual_cursor - overlap if index else None),
            visual_memory_include_canonical=include_canonical,
            visual_memory_layout_anchor_start_frame=(
                layout_anchor_interval[0]
                if layout_anchor_interval is not None else None
            ),
            visual_memory_layout_anchor_stop_frame=(
                layout_anchor_interval[1]
                if layout_anchor_interval is not None else None
            ),
            visual_memory_progressive_layout_state=progressive_layout_state,
            preserve_latest_visual=index < len(authored_segments) - 1,
            authorized_dialogue_count=len(dialogue_frames),
            authorized_dialogue_frames=tuple(dialogue_frames),
            reference_audio_active=bool(dialogue_frames),
            audio_memory_active=bool(dialogue_frames),
            opening_preroll_frames=opening_preroll,
            novel_camera_cut=novel_camera_cut,
            terminal_video_seed=terminal_video_seed,
            novel_camera_layout_probe=novel_camera_layout_probe,
        ))
        preview.append({
            'shot_id': shot['id'],
            'segment_id': segment['id'],
            'transition': transition,
            'requested_start_seconds': requested_cursor,
            'requested_end_seconds': requested_stop,
            'actual_start_seconds': actual_cursor / FPS,
            'actual_end_seconds': stop / FPS,
            'context_seconds': offset,
            'protected_video_prefix_seconds': (video_prefix or 0) / FPS,
            'hidden_video_repaint_seconds': (
                (overlap - int(video_prefix or 0)) / FPS if index else 0.0
            ),
            'window_seconds': window / FPS,
            'prompt': prompt,
            'continuation_boundary_prompt': continuation_bridge_prompt,
            'entities': sorted(needed),
            'visual_memory_policy': visual_policy,
            'canonical_memory_active': include_canonical,
            'camera_anchor': anchor_spec,
            'camera_anchor_actual_seconds': (
                [
                    layout_anchor_interval[0] / FPS,
                    layout_anchor_interval[1] / FPS,
                ]
                if layout_anchor_interval is not None else None
            ),
            'progressive_layout_state': progressive_layout_state,
            'opening_preroll_seconds': opening_preroll / FPS,
            'novel_camera_cut': novel_camera_cut,
            'terminal_video_seed': terminal_video_seed,
            'novel_camera_layout_probe': novel_camera_layout_probe,
        })
        state = next_state
        camera_end = end_camera
        requested_cursor = requested_stop
        actual_cursor = stop
    plan = WindowPlan(
        requested_duration_seconds=requested_cursor,
        output_frames=actual_cursor,
        actual_duration_seconds=actual_cursor / FPS,
        segments=tuple(segments),
        context_frames=context,
        mechanism='authored_window_script_joint_av_v2',
        planning_policy='state_safe_ranged_segment_per_window_v3_camera_anchor',
        structured_director=True,
    )
    return plan, {
        'version': 2,
        'requested_controls': {
            key: data[key] for key in ('overlap_seconds', 'max_window_seconds', 'memory')
        },
        'effective_overlap_seconds': context / FPS,
        'effective_max_window_seconds': maximum / FPS,
        'actual_duration_seconds': plan.actual_duration_seconds,
        'windows': preview,
        'reference_check': (
            'Entity identity is structurally separated from typed mutable state; '
            'beat updates are applied transactionally. Real visual compliance remains generative.'
        ),
    }
