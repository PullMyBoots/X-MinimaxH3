"""Public OpenAPI contract for H3 Serve clients.

The Web studio, command-line clients and ComfyUI connector all use this same
asynchronous job API.  Runtime-only implementation fields intentionally stay
out of this document.
"""

from __future__ import annotations

from typing import Any

from .contract import (
    MAX_CUSTOM_DIMENSION,
    MAX_LONG_HORIZON_DURATION_SECONDS,
    MAX_NATIVE_PIXEL_FRAMES,
)


def document(version: str) -> dict[str, Any]:
    long_video_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["version", "story"],
        "properties": {
            "version": {
                "type": "integer", "enum": [2, 3], "default": 3,
                "description": (
                    "Version 3 schedules semantic camera cuts inside physical H3 "
                    "windows; version 2 is retained for persisted jobs."
                ),
            },
            "overlap_seconds": {"type": "number", "minimum": 0.25, "maximum": 4, "default": 1.625},
            "max_window_seconds": {"type": "number", "minimum": 5, "maximum": 15, "default": 15},
            "memory": {
                "type": "integer", "minimum": 0, "maximum": 100, "default": 60,
                "description": "Bounded long-term AV reference capacity. Zero disables automatic long memory.",
            },
            "story": {"$ref": "#/components/schemas/LongVideoStory"},
        },
    }
    entity_schema = {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "identity"],
        "properties": {
            "kind": {"type": "string", "enum": ["character", "location", "prop"]},
            "identity": {
                "type": "string", "minLength": 1,
                "description": "Time-invariant appearance or identity only; mutable location and state belong in initial_state/beat updates.",
            },
        },
    }
    state_properties = {
        "type": "object", "minProperties": 1,
        "additionalProperties": {
            "oneOf": [
                {"type": "string"}, {"type": "number"},
                {"type": "boolean"}, {"type": "null"},
            ]
        },
    }
    state_map = {
        "type": "object",
        "additionalProperties": state_properties,
    }
    story_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["overview", "entities", "initial_state", "shots"],
        "properties": {
            "overview": {"type": "string", "minLength": 1},
            "entities": {
                "type": "object", "minProperties": 1, "maxProperties": 64,
                "additionalProperties": {"$ref": "#/components/schemas/LongVideoEntity"},
            },
            "initial_state": {**state_map, "minProperties": 1},
            "shots": {
                "type": "array", "minItems": 1, "maxItems": 24,
                "items": {"$ref": "#/components/schemas/LongVideoShot"},
            },
        },
    }
    shot_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "camera_style", "overall_soundscape", "non_diegetic_music", "segments"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "camera_anchor": {
                "type": "object",
                "additionalProperties": False,
                "required": ["shot_id", "segment_id"],
                "description": "Reuse the camera setup and stable layout from one earlier authored segment while current state remains authoritative.",
                "properties": {
                    "shot_id": {"type": "string", "minLength": 1},
                    "segment_id": {"type": "string", "minLength": 1},
                },
            },
            "camera_style": {
                "type": "string", "minLength": 1,
                "description": "Shot-wide lens and continuity invariants. Put the local camera path in each segment.camera.",
            },
            "overall_soundscape": {"type": "string", "minLength": 1},
            "non_diegetic_music": {"type": "string", "minLength": 1},
            "segments": {
                "type": "array", "minItems": 1,
                "description": (
                    "Authored time ranges inside this continuous shot. In version 3, "
                    "physical inference windows are scheduled independently and may "
                    "cross a shot boundary so H3 generates the cut internally."
                ),
                "items": {"$ref": "#/components/schemas/LongVideoSegment"},
            },
        },
    }
    segment_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "duration_seconds", "transition", "camera", "beats"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "duration_seconds": {"type": "number", "minimum": 1, "maximum": 15},
            "transition": {"type": "string", "enum": ["opening", "continue", "cut"]},
            "establish_seconds": {
                "type": "number", "minimum": 0,
                "description": "For a cut, a minimum 0.5-second action-free establishing interval after the carried prefix.",
            },
            "camera": {
                "type": "object", "additionalProperties": False,
                "required": ["motion", "end"],
                "properties": {
                    "composition": {"type": "string", "minLength": 1},
                    "motion": {"type": "string", "minLength": 1},
                    "end": {"type": "string", "minLength": 1},
                },
            },
            "beats": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["start_seconds", "end_seconds", "text", "state_updates"],
                    "properties": {
                        "start_seconds": {"type": "number", "minimum": 0},
                        "end_seconds": {"type": "number", "minimum": 0},
                        "text": {"type": "string", "minLength": 1},
                        "state_updates": state_map,
                    },
                },
            },
            "dialogue": {
                "type": "array", "default": [],
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["start_seconds", "end_seconds", "speaker", "language", "text"],
                    "properties": {
                        "start_seconds": {"type": "number", "minimum": 0},
                        "end_seconds": {"type": "number", "minimum": 0},
                        "speaker": {"type": "string", "minLength": 1},
                        "language": {"type": "string", "minLength": 1},
                        "text": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }
    request_properties = {
        "service_family": {"type": "string", "enum": ["first_last", "reference"]},
        "model_variant": {"type": "string", "enum": ["base", "lora"], "default": "base"},
        "mode": {"type": "string", "enum": ["preset", "advanced"], "default": "preset"},
        "prompt": {
            "type": "string", "minLength": 1, "maxLength": 20000,
            "description": (
                "Direct-path final H3 model-facing text. It is forwarded unchanged when "
                "long_video is absent. Supply long_video instead to opt into deterministic window compilation."
            ),
        },
        "long_video": {"$ref": "#/components/schemas/LongVideoRequest"},
        "quality": {"type": "string", "enum": ["fast", "balanced", "quality", "ultra"]},
        "resolution": {
            "type": "string",
            "pattern": "^(3[6-9][0-9]|[4-9][0-9]{2}|1[0-3][0-9]{2}|14[0-3][0-9]|1440)p$",
            "description": "Final short edge. Values above the active first-pass limit require SelfLift progressive generation.",
        },
        "aspect_ratio": {"type": "string", "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"]},
        "duration_seconds": {
            "type": "number", "minimum": 1,
            "maximum": MAX_LONG_HORIZON_DURATION_SECONDS,
            "description": (
                "Requested output duration. Up to 15 seconds uses one native H3 window; "
                "longer requests are transparently planned as masked joint-AV latent "
                "continuations and decoded once. Every physical window still satisfies "
                f"width*height*frames <= {MAX_NATIVE_PIXEL_FRAMES}. Query /api/v1/options "
                "for native and long-horizon ceilings."
            ),
        },
        "seed": {"oneOf": [{"type": "integer", "minimum": 0}, {"type": "string", "enum": ["random"]}]},
        "width": {"type": "integer", "minimum": 192, "maximum": MAX_CUSTOM_DIMENSION, "multipleOf": 32},
        "height": {"type": "integer", "minimum": 192, "maximum": MAX_CUSTOM_DIMENSION, "multipleOf": 32},
        "frames": {"type": "integer", "minimum": 5, "maximum": 362},
        "sampling_steps": {
            "type": "integer",
            "minimum": 4,
            "maximum": 30,
            "description": "用户指定的总采样轨迹步数；INT8支持5–30，LoRA支持4–10。LoRA超过8步未经质量校准。",
        },
        "acceleration": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "连续加速强度。0为全真实步Dense端点；"
                "75为Human审阅的发布质量拐点；"
                "75到100为明确允许质量风险的激进外推区。"
                "Base自动联合分配真实/预测步和逐步逐层注意力预算；"
                "LoRA保留用户指定的全部真实Turbo步，只自适应分配逐步逐层注意力预算。"
            ),
        },
        "second_pass_acceleration": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "default": 0,
            "description": (
                "绿色分叉点之后正式二采分支的连续加速强度；"
                "省略时沿用acceleration。"
            ),
        },
        "acceleration_transition_step": {
            "type": "integer",
            "minimum": 1,
            "description": "一采加速调度结束、二采加速调度开始的one-based正式步数。",
        },
        "actual_steps": {"type": "integer", "minimum": 5, "maximum": 20, "deprecated": True},
        "lora_steps": {"type": "integer", "minimum": 4, "maximum": 8, "deprecated": True},
        "attention_keep_ratio": {"type": "number", "minimum": 0.5, "maximum": 1.0, "deprecated": True},
        "sparse_scope": {"type": "string", "enum": ["middle_only", "guarded", "full"], "deprecated": True},
        "preview_mode": {
            "type": "string",
            "enum": ["off", "auto", "pause"],
            "default": "off",
        },
        "preview_step_index": {
            "type": "integer",
            "minimum": 0,
            "description": "Zero-based formal step index used as the shared preview branch point.",
        },
        "preview_branch_steps": {
            "type": "integer", "minimum": 1, "maximum": 30, "default": 2,
        },
        "preview_fast_finish": {"type": "boolean", "default": False},
        "selflift_enabled": {"type": "boolean", "default": False},
        "selflift_initial_resolution": {
            "type": "string",
            "pattern": "^(3[6-9][0-9]|[4-9][0-9]{2}|10[0-7][0-9]|1080)p$",
            "default": "540p",
        },
        "selflift_transition_step": {
            "type": "integer",
            "minimum": 1,
            "description": "One-based count of completed low-resolution steps before the SelfLift transition.",
        },
        "selflift_temporal_window_enabled": {
            "type": "boolean",
            "default": True,
            "description": "Run a long high-resolution SelfLift tail as overlapping global temporal views.",
        },
        "selflift_temporal_window_seconds": {
            "type": "number",
            "minimum": 3,
            "maximum": 15,
            "default": 5,
            "description": "Preferred upper duration of each high-resolution temporal view; the value is aligned to H3's joint AV frame grid.",
        },
        "selflift_temporal_overlap_seconds": {
            "type": "number",
            "minimum": 0,
            "maximum": 4,
            "default": 1,
            "description": "Preferred amount of the preceding high-resolution view used as temporal context; execution snaps to H3's legal joint AV frame grid.",
        },
        "execution_mode": {"type": "string", "enum": ["complete", "checkpoint"], "default": "complete"},
        "checkpoint_step": {"type": "integer", "minimum": 1},
        "checkpoint_retain": {"type": "boolean", "default": True},
        "checkpoint_preview": {"type": "boolean", "default": False},
        "checkpoint_preview_steps": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
        "checkpoint_preview_resolution": {
            "type": "string",
            "enum": ["source", "360p", "480p", "720p"],
            "default": "360p",
        },
        "reference_image_resolution": {
            "type": "string",
            "enum": ["original", "360p", "480p", "720p"],
            "default": "720p",
            "description": (
                "参考图片短边分辨率上限；只按比例缩小，不放大、不裁切、不拉伸。"
                "original跳过额外压缩，模型所需的最小32像素对齐填充仍会执行。"
            ),
        },
        "reference_video_resolution": {
            "type": "string",
            "enum": ["original", "360p", "480p", "720p"],
            "default": "360p",
            "description": (
                "参考视频短边分辨率上限；只按比例缩小每帧，不改变画幅、构图或时长。"
                "original跳过额外压缩，模型所需的最小32像素对齐填充仍会执行。"
            ),
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "X-MinimaxH3 API",
            "version": version,
            "description": "Asynchronous MiniMax H3 generation API for the Web studio, scripts and ComfyUI.",
        },
        "servers": [{"url": "/"}],
        "components": {
            "securitySchemes": {
                "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            },
            "schemas": {
                "GenerationRequest": {
                    "type": "object",
                    "anyOf": [{"required": ["prompt"]}, {"required": ["long_video"]}],
                    "properties": request_properties,
                    "x-h3-native-spatiotemporal-budget": {
                        "inequality": f"width*height*frames <= {MAX_NATIVE_PIXEL_FRAMES}",
                        "reference_envelope": {
                            "width": 1920, "height": 1088, "frames": 362,
                        },
                        "frame_grid": "17*n+5",
                        "absolute_max_frames": 362,
                        "transparent_long_horizon_max_seconds": (
                            MAX_LONG_HORIZON_DURATION_SECONDS
                        ),
                    },
                },
                "LongVideoRequest": long_video_schema,
                "LongVideoStory": story_schema,
                "LongVideoEntity": entity_schema,
                "LongVideoShot": shot_schema,
                "LongVideoSegment": segment_schema,
                "EngineSelectionRequest": {
                    "type": "object",
                    "required": [
                        "service_family", "weight_tier", "host_memory_limit_gib",
                    ],
                    "properties": {
                        "service_family": {
                            "type": "string",
                            "enum": ["first_last", "reference"],
                            "description": "FL2VA or Ref2VA service family.",
                        },
                        "weight_tier": {
                            "type": "string",
                            "enum": ["w4a8", "int8"],
                            "description": (
                                "Model-weight format. W4A8 admits 8GB-class GPUs; "
                                "INT8 requires at least a 16GB-class GPU."
                            ),
                        },
                        "host_memory_limit_gib": {
                            "type": "integer",
                            "minimum": 12,
                            "description": (
                                "Hard cgroup-v2 ceiling for the complete H3 service and "
                                "its child processes. W4A8 accepts experimental budgets "
                                "from 12GiB and INT8 from 24GiB; the validated floors are "
                                "16GiB and 32GiB respectively. Query /api/v1/options for the "
                                "machine-specific upper bound, which reserves 6GiB for "
                                "the operating system."
                            ),
                        },
                        "model_variant": {
                            "type": "string",
                            "enum": ["base", "lora"],
                            "default": "base",
                        },
                        "launcher": {
                            "type": "string",
                            "deprecated": True,
                            "description": (
                                "Legacy explicit internal launcher. New clients must use "
                                "service_family, weight_tier and host_memory_limit_gib; "
                                "the VRAM route is detected automatically."
                            ),
                        },
                    },
                },
                "SecondSamplingRequest": {
                    "type": "object",
                    "properties": {
                        "resolution": {
                            "type": "string",
                            "pattern": "^(7[2-9][0-9]|[89][0-9]{2}|1[0-3][0-9]{2}|14[0-3][0-9]|1440)p$",
                            "default": "1080p",
                            "description": (
                                "Continuous target short edge from 720p through 1440p. "
                                "The runtime aligns the resulting canvas to 32 pixels."
                            ),
                        },
                        "steps": {
                            "type": "integer", "minimum": 1, "maximum": 8,
                            "default": 4,
                            "description": (
                                "Real H3 refinement steps. When no legacy strength or "
                                "denoise override is supplied, the service automatically "
                                "selects the balanced start sigma for this step count."
                            ),
                        },
                        "strength": {
                            "type": "string",
                            "enum": ["preserve", "standard", "enhance", "strong"],
                            "deprecated": True,
                            "description": (
                                "Legacy override. Product clients should omit this field "
                                "and let steps select the balanced denoise automatically."
                            ),
                        },
                        "acceleration": {
                            "type": "number", "minimum": 0, "maximum": 100,
                            "default": 75,
                            "description": (
                                "Exact-only low-noise trajectory: acceleration is "
                                "projected onto the coupled Attention policy; Forecast is disabled."
                            ),
                        },
                        "temporal_window_frames": {
                            "type": ["integer", "null"],
                            "minimum": 68,
                            "maximum": 362,
                            "default": None,
                            "description": (
                                "Optional user temporal-context window. The backend "
                                "snaps it to H3's phase grid, keeps overlap/crossfade "
                                "automatic, and shortens it further only when required "
                                "by the physical VRAM budget."
                            ),
                        },
                        "temporal_overlap_frames": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                            "default": None,
                            "description": (
                                "Optional temporal context shared with the preceding "
                                "window. Product clients normally use the global "
                                "0–4 second setting, aligned by the backend."
                            ),
                        },
                        "denoise": {
                            "type": "number", "minimum": 0.05, "maximum": 0.50,
                            "deprecated": True,
                            "description": (
                                "Legacy continuous override. Product clients should omit "
                                "this field and use the automatic step policy."
                            ),
                        },
                    },
                },
                "VideoRepairRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "acceleration": {
                            "type": "number", "minimum": 0, "maximum": 100,
                            "default": 50,
                            "description": "Calibrated sparse-attention acceleration strength.",
                        },
                    },
                },
                "InfiniteProjectCreate": {
                    "type": "object",
                    "required": ["title"],
                    "additionalProperties": False,
                    "properties": {
                        "title": {
                            "type": "string", "minLength": 1, "maxLength": 120,
                            "description": "Project container name. Generation context is authored per window.",
                        },
                        "workflow_version": {"type": "integer", "enum": [2, 3], "default": 3},
                        "creation_mode": {
                            "type": "string", "enum": ["online", "json"], "default": "online",
                            "description": "Choose the segment-by-segment preview studio or the persistent JSON queue. Both retain one connected source-latent track and complete one global SelfLift final branch.",
                        },
                        "overview": {"type": "string", "maxLength": 12000},
                        "overall_soundscape": {"type": "string", "maxLength": 6000, "default": "N/A"},
                        "non_diegetic_music": {"type": "string", "maxLength": 6000, "default": "N/A"},
                        "preview_resolution": {"type": "string", "pattern": "^(3[6-9][0-9]|[4-9][0-9]{2}|10[0-7][0-9]|1080)p$", "default": "540p"},
                        "final_resolution": {"type": "string", "pattern": "^(3[6-9][0-9]|[4-9][0-9]{2}|1[0-3][0-9]{2}|14[0-3][0-9]|1440)p$", "default": "1080p"},
                        "aspect_ratio": {"type": "string", "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"], "default": "16:9"},
                        "model_variant": {"type": "string", "enum": ["base", "lora"], "default": "lora"},
                        "sampling_steps": {"type": "integer", "minimum": 4, "maximum": 30, "default": 8},
                        "final_sampling_steps": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
                        "preview_branch_steps": {
                            "type": "integer", "minimum": 1, "maximum": 4, "default": 2,
                            "description": "Online mode only: extra low-resolution steps used to decode the preview branch from the formal fork.",
                        },
                        "acceleration": {"type": "number", "minimum": 0, "maximum": 100, "default": 50},
                        "second_pass_acceleration": {"type": "number", "minimum": 0, "maximum": 100, "default": 75},
                        "window_duration_seconds": {"type": "number", "minimum": 1, "maximum": 15, "default": 5},
                        "overlap_seconds": {"type": "number", "minimum": 0, "maximum": 4, "default": 1.625},
                        "visual_memory_capacity": {"type": "integer", "minimum": 0, "maximum": 24, "default": 6},
                        "audio_memory_capacity": {"type": "integer", "minimum": 0, "maximum": 3, "default": 1},
                        "visual_memory_resolution": {"type": "string", "enum": ["360p", "480p", "720p", "original"], "default": "360p"},
                    },
                },
                "InfiniteBatchRequest": {
                    "type": "object",
                    "required": ["windows"],
                    "additionalProperties": False,
                    "properties": {
                        "overview": {"type": "string", "maxLength": 12000},
                        "overall_soundscape": {"type": "string", "maxLength": 6000},
                        "non_diegetic_music": {"type": "string", "maxLength": 6000},
                        "final_acceleration": {
                            "type": "number", "minimum": 0, "maximum": 100,
                            "default": 75,
                            "description": "Workflow-v3 JSON only: acceleration for the automatic global high-resolution branch.",
                        },
                        "references": {
                            "type": "object",
                            "description": "Optional default Ref2VA reference set. A window without its own references inherits the preceding effective set. Paths are persisted privately and never returned by the public project API.",
                            "propertyNames": {"pattern": "^(Picture [1-9]|Audio [1-3])$"},
                            "additionalProperties": {
                                "oneOf": [
                                    {"type": "string", "minLength": 1},
                                    {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {"path": {"type": "string", "minLength": 1}}},
                                ]
                            },
                        },
                        "windows": {
                            "type": "array", "minItems": 1, "maxItems": 100,
                            "items": {
                                "oneOf": [
                                    {"type": "string", "minLength": 1, "maxLength": 12000},
                                    {"type": "object", "required": ["prompt"], "additionalProperties": False, "properties": {
                                        "prompt": {"type": "string", "minLength": 1, "maxLength": 12000},
                                        "seed": {"oneOf": [{"type": "integer"}, {"type": "string", "enum": ["random"]}]},
                                        "duration_seconds": {"type": "number", "minimum": 1, "maximum": 15},
                                        "overlap_seconds": {"type": "number", "minimum": 0, "maximum": 4},
                                        "acceleration": {"type": "number", "minimum": 0, "maximum": 100},
                                        "visual_memory_capacity": {"type": "integer", "minimum": 0, "maximum": 24},
                                        "audio_memory_capacity": {"type": "integer", "minimum": 0, "maximum": 3},
                                        "visual_memory_resolution": {"type": "string", "enum": ["360p", "480p", "720p", "original"]},
                                        "references": {
                                            "type": "object",
                                            "description": "Complete Ref2VA reference set for this window. It replaces the preceding set; omit it to inherit the preceding set.",
                                            "propertyNames": {"pattern": "^(Picture [1-9]|Audio [1-3])$"},
                                            "additionalProperties": {
                                                "oneOf": [
                                                    {"type": "string", "minLength": 1},
                                                    {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {"path": {"type": "string", "minLength": 1}}},
                                                ]
                                            },
                                        },
                                    }},
                                ],
                            },
                        },
                    },
                },
                "InfiniteWindowAppend": {
                    "type": "object",
                    "required": ["window_description"],
                    "description": (
                        "Workflow-v3 keeps geometry, model route and step counts fixed while "
                        "allowing each online window to override duration, overlap, first-pass "
                        "acceleration and AV-memory controls. Workflow-v2 retains its fully "
                        "locked preview trajectory."
                    ),
                    "properties": {
                        "overview": {"type": "string"},
                        "window_description": {
                            "type": "string", "minLength": 1,
                            "description": (
                                "Complete local-clock H3 shot description for this window. "
                                "Any semantic camera cut must occur inside this text; every "
                                "physical window boundary remains a strict continuation."
                            ),
                        },
                        "overall_soundscape": {"type": "string"},
                        "non_diegetic_music": {"type": "string"},
                        "duration_seconds": {"type": "number", "minimum": 1, "maximum": 15},
                        "overlap_seconds": {"type": "number", "minimum": 0, "maximum": 4},
                        "visual_memory_capacity": {"type": "integer", "minimum": 0, "maximum": 24, "default": 6},
                        "audio_memory_capacity": {"type": "integer", "minimum": 0, "maximum": 3, "default": 1},
                        "visual_memory_resolution": {"type": "string", "enum": ["360p", "480p", "720p", "original"], "default": "360p"},
                        "memory": {"type": "integer", "minimum": 0, "maximum": 100, "deprecated": True},
                        "model_variant": {"type": "string", "enum": ["base", "lora"]},
                        "sampling_steps": {"type": "integer", "minimum": 4, "maximum": 30},
                        "acceleration": {"type": "number", "minimum": 0, "maximum": 100},
                        "execution_mode": {"type": "string", "enum": ["complete", "checkpoint"], "default": "complete"},
                        "checkpoint_step": {"type": "integer", "minimum": 1},
                        "checkpoint_retain": {"type": "boolean", "default": True},
                        "checkpoint_preview": {"type": "boolean", "default": False},
                        "checkpoint_preview_steps": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
                        "checkpoint_preview_resolution": {"type": "string", "enum": ["source", "360p", "480p", "720p"], "default": "360p"},
                        "seed": {"oneOf": [{"type": "integer"}, {"type": "string", "enum": ["random"]}]},
                        "resolution": {"type": "string", "pattern": "^(3[6-9][0-9]|[4-9][0-9]{2}|10[0-7][0-9]|1080)p$"},
                        "aspect_ratio": {"type": "string", "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"]},
                        "save_shared_as_default": {"type": "boolean", "default": False},
                    },
                },
                "Job": {
                    "type": "object",
                    "required": ["id", "status", "request", "progress"],
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "status": {"type": "string", "enum": ["queued", "starting_backend", "running", "checkpointed", "awaiting_preview", "succeeded", "failed", "cancelled"]},
                        "request": {"type": "object"},
                        "progress": {"type": "object"},
                        "video_url": {"type": "string"},
                        "checkpoint": {"type": "object"},
                        "inference_plan": {
                            "type": "object",
                            "description": (
                                "只读调度回执：Dense回退原因，或V19候选、"
                                "execution/envelope/certificate digest与实际/预测步数；"
                                "memory_execution记录统一显存优化器选择的执行图、预算和块长。"
                            ),
                        },
                        "second_sampling_available": {"type": "boolean"},
                        "second_sampling": {"type": "object"},
                        "video_repair_available": {"type": "boolean"},
                        "video_repair": {"type": "object"},
                        "error": {"type": ["string", "null"]},
                    },
                },
            },
        },
        "security": [{"ApiKey": []}],
        "paths": {
            "/healthz": {"get": {"security": [], "summary": "Liveness", "responses": {"200": {"description": "Alive"}}}},
            "/readyz": {"get": {"security": [], "summary": "Active engine readiness", "responses": {"200": {"description": "Ready"}, "503": {"description": "Not ready"}}}},
            "/api/v1/options": {"get": {"summary": "Capabilities and defaults", "responses": {"200": {"description": "Options"}}}},
            "/api/v1/engine": {
                "put": {
                    "summary": "Load one of the four public model choices",
                    "description": (
                        "The service detects physical VRAM and selects its private "
                        "8GB, 16GB or 24GB execution backend automatically. The host "
                        "memory value is applied as a hard process-tree ceiling before "
                        "the model is loaded. Engine changes require an idle queue."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/EngineSelectionRequest"
                        }}},
                    },
                    "responses": {
                        "200": {"description": "Model loaded and resource budget active"},
                        "400": {"description": "Unsupported weight, VRAM, or RAM budget"},
                        "409": {"description": "Queue or another model transition is active"},
                        "500": {"description": "Hard limit or model preload failed"},
                    },
                },
                "delete": {
                    "summary": "Release the active model and host-memory cgroup",
                    "responses": {
                        "200": {"description": "Model and hard memory budget released"},
                        "409": {"description": "Jobs are active"},
                    },
                },
            },
            "/api/v1/workspace/browse": {
                "get": {
                    "summary": "Browse server-local workspace folders",
                    "parameters": [{"name": "path", "in": "query", "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Current folder and child directories"}},
                },
            },
            "/api/v1/workspace": {
                "put": {
                    "summary": "Select the idle unified console workspace",
                    "description": "Requires no loaded engine and no active or queued jobs.",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string", "description": "Absolute server-local directory"}},
                    }}}},
                    "responses": {"200": {"description": "Workspace selected"}, "409": {"description": "Engine or queue is active"}},
                },
            },
            "/api/v1/generations": {
                "post": {
                    "summary": "Submit a generation job",
                    "description": (
                        "Use JSON for text-only jobs or multipart/form-data for first/last/reference media. "
                        "Ref2VA accepts reference_image_1..9, reference_video_1..3 and "
                        "reference_audio_1..3. Each reference video is 2–15 seconds; the total "
                        "video duration is at most 15 seconds and embedded video audio is ignored. "
                        "reference_image_resolution/reference_video_resolution select proportional "
                        "downscale caps shared by Web, API and ComfyUI; they preserve aspect ratio "
                        "and never crop, stretch or pad the user media canvas."
                    ),
                    "requestBody": {"required": True, "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/GenerationRequest"}},
                        "multipart/form-data": {"schema": {"type": "object", "anyOf": [{"required": ["prompt"]}, {"required": ["long_video"]}], "properties": {
                            **request_properties,
                            **{f"reference_image_{index}": {"type": "string", "format": "binary"} for index in range(1, 10)},
                            **{f"reference_video_{index}": {"type": "string", "format": "binary"} for index in range(1, 4)},
                            **{f"reference_audio_{index}": {"type": "string", "format": "binary"} for index in range(1, 4)},
                        }}},
                    }},
                    "responses": {"202": {"description": "Accepted", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Job"}}}}, "400": {"description": "Invalid request"}},
                }
            },
            "/api/v1/long-video/preview": {
                "post": {
                    "summary": "Validate and compile an authored long-video window script",
                    "description": "Returns frame-grid-adjusted windows, complete per-window H3 prompts and the effective AV-memory budget without starting generation.",
                    "requestBody": {"required": True, "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/GenerationRequest"}},
                        "multipart/form-data": {"schema": {"type": "object", "required": ["long_video"], "properties": {
                            **request_properties,
                            "first_frame": {"type": "string", "format": "binary"},
                            "last_frame": {"type": "string", "format": "binary"},
                            **{f"reference_image_{index}": {"type": "string", "format": "binary"} for index in range(1, 10)},
                            **{f"reference_audio_{index}": {"type": "string", "format": "binary"} for index in range(1, 4)},
                        }}},
                    }},
                    "responses": {"200": {"description": "Compiled preview"}, "400": {"description": "Invalid window script"}},
                }
            },
            "/api/v1/infinite-projects": {
                "get": {
                    "summary": "List tail-editable infinite-video projects",
                    "responses": {"200": {"description": "Project list"}},
                },
                "post": {
                    "summary": "Create an infinite-video project",
                    "requestBody": {"required": True, "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/InfiniteProjectCreate"}}
                    }},
                    "responses": {"201": {"description": "Project created"}, "400": {"description": "Invalid defaults"}},
                },
            },
            "/api/v1/infinite-projects/{project_id}": {
                "get": {
                    "summary": "Read one infinite-video project and its accepted tail",
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "responses": {"200": {"description": "Project"}, "404": {"description": "Not found"}},
                },
                "delete": {
                    "summary": "Delete an infinite-video project container",
                    "description": "Removes the project and its editable timeline. Referenced generation jobs and generated media remain available in job history.",
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "responses": {"200": {"description": "Project deleted; jobs retained"}, "404": {"description": "Not found"}},
                },
            },
            "/api/v1/infinite-projects/{project_id}/windows": {
                "post": {
                    "summary": "Append one strict-continuation generation window",
                    "description": (
                        "The opening fixes geometry. Later windows inherit it, carry an exact "
                        "clean AV latent prefix, and optionally read bounded long-term AV memory."
                    ),
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "requestBody": {"required": True, "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/InfiniteWindowAppend"}},
                        "multipart/form-data": {"schema": {
                            "allOf": [{"$ref": "#/components/schemas/InfiniteWindowAppend"}],
                            "properties": {
                                "first_frame": {"type": "string", "format": "binary"},
                                **{f"reference_image_{index}": {"type": "string", "format": "binary"} for index in range(1, 10)},
                                **{f"reference_audio_{index}": {"type": "string", "format": "binary"} for index in range(1, 4)},
                            },
                        }},
                    }},
                    "responses": {"202": {"description": "Window queued"}, "400": {"description": "Invalid window"}},
                },
            },
            "/api/v1/infinite-projects/{project_id}/windows/last": {
                "delete": {
                    "summary": "Discard the current project tail window",
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "responses": {"200": {"description": "Previous accepted tail restored"}, "409": {"description": "Tail is active or absent"}},
                },
            },
            "/api/v1/infinite-projects/{project_id}/batch": {
                "post": {
                    "summary": "Persist and run a direct JSON long-video queue",
                    "description": (
                        "For workflow-v3 JSON projects, the server builds one connected "
                        "low-resolution source-latent timeline without decoding disposable "
                        "previews. After the last source window, it lifts the complete timeline "
                        "once, completes the locked high-resolution tail through overlapping "
                        "temporal views with global scheduler updates, and decodes one final film."
                    ),
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/InfiniteBatchRequest"}}}},
                    "responses": {"202": {"description": "Persistent formal-generation queue started"}, "400": {"description": "Invalid JSON plan"}},
                },
                "delete": {
                    "summary": "Stop submitting remaining JSON windows",
                    "description": "The current GPU job is left intact; only subsequent automatic submissions are stopped.",
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "responses": {"200": {"description": "Automatic queue stopped"}},
                },
            },
            "/api/v1/infinite-projects/{project_id}/second-sampling": {
                "post": {
                    "summary": "Queue whole-project H3 second sampling",
                    "description": "Runs the accepted cumulative clean AV latent through automatic temporal windows and emits one complete refined video.",
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SecondSamplingRequest"}}}},
                    "responses": {"202": {"description": "Whole-project second sampling queued"}, "409": {"description": "No completed tail"}},
                },
            },
            "/api/v1/infinite-projects/{project_id}/final-sampling": {
                "post": {
                    "summary": "Complete or retrieve the whole-project H3 final output",
                    "description": (
                        "For workflow-v3 online projects, joins the accepted low-resolution source "
                        "latents into one timeline, lifts that timeline once, completes the locked "
                        "high-resolution tail through overlapping temporal views with global "
                        "scheduler updates, and decodes one final film. For workflow-v3 JSON "
                        "projects, the batch already queued this same global final job, so this "
                        "compatibility endpoint returns that completed result without recomputing "
                        "anything. The v3 target resolution and tail step count are fixed when "
                        "the project is created; acceleration may be selected when the online "
                        "final branch is submitted. Workflow v2 retains its legacy detached "
                        "whole-project second-sampling request."
                    ),
                    "parameters": [{"name": "project_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "oneOf": [
                                        {"type": "object", "maxProperties": 0},
                                        {"$ref": "#/components/schemas/SecondSamplingRequest"},
                                    ]
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Existing JSON direct final returned"},
                        "202": {"description": "Online final sampling queued"},
                        "409": {"description": "Required trajectory is incomplete"},
                    },
                },
            },
            "/api/v1/jobs/{job_id}/resume": {
                "post": {
                    "summary": "Queue continuation from a retained formal checkpoint",
                    "responses": {"202": {"description": "Resume queued"}, "409": {"description": "Checkpoint unavailable"}},
                },
            },
            "/api/v1/jobs/{job_id}/second-sampling": {
                "post": {
                    "summary": "Queue learned H3 latent second sampling",
                    "description": (
                        "Creates a child job from the completed source card's clean AV latent. "
                        "The source prompt and reference media are reused unchanged; audio is "
                        "preserved. A learned H3 3D latent upscaler replaces invalid image-style "
                        "latent interpolation before a Base-weight SA-Solver low-noise "
                        "refinement pass."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/SecondSamplingRequest"
                        }}},
                    },
                    "responses": {
                        "202": {"description": "Second-sampling child queued"},
                        "400": {"description": "Invalid target or source latent unavailable"},
                        "409": {"description": "Wrong active service family"},
                    },
                },
            },
            "/api/v1/jobs/{job_id}/video-repair": {
                "post": {
                    "summary": "Queue tracked face repair",
                    "description": (
                        "Detects and tracks under-resolved faces in the completed video, "
                        "ranks them by measured repair need, keeps only faces that gain "
                        "at least 1.5x in the configured square-cell Atlas, runs fixed "
                        "four-step H3 Turbo repair, and blends the result back at "
                        "source resolution, and preserves the original audio."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/VideoRepairRequest"
                        }}},
                    },
                    "responses": {
                        "202": {"description": "Video-repair child queued"},
                        "400": {"description": "Invalid repair request or source video unavailable"},
                        "409": {"description": "Wrong active service family"},
                    },
                },
            },
            "/api/v1/cache/latents": {
                "delete": {
                    "summary": "Clear reproducible latent and checkpoint caches",
                    "description": (
                        "Keeps videos and job history, but removes clean AV latent "
                        "artifacts used by second sampling and formal checkpoint tensors."
                    ),
                    "responses": {
                        "200": {"description": "Cache cleared"},
                        "409": {"description": "Jobs are active"},
                    },
                },
            },
            "/api/v1/settings/lora": {
                "get": {
                    "summary": "List installed H3 LoRA weights and the selected version",
                    "description": (
                        "Scans .safetensors below models/loras and reports native-name "
                        "H3 adapter compatibility without loading tensor payloads."
                    ),
                    "responses": {"200": {"description": "LoRA catalog and active selection"}},
                },
                "put": {
                    "summary": "Select and load one installed H3 LoRA version",
                    "description": (
                        "Requires an idle job queue. The current hot engine is released and "
                        "rebuilt; a failed build restores the previous LoRA."
                    ),
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object", "required": ["checkpoint"],
                        "properties": {"checkpoint": {
                            "type": "string",
                            "description": "Path relative to models/loras from the GET catalog",
                        }},
                    }}}},
                    "responses": {
                        "200": {"description": "LoRA selected and hot engine ready"},
                        "400": {"description": "Missing, unsafe, or incompatible LoRA"},
                        "409": {"description": "Jobs or another engine transition are active"},
                        "500": {"description": "Build failed and previous LoRA was restored"},
                    },
                },
            },
            "/api/v1/settings/reference-media": {
                "get": {
                    "summary": "Read the shared reference-media preprocessing defaults",
                    "description": (
                        "These server defaults apply to Web, API and ComfyUI requests "
                        "that do not provide a per-request override."
                    ),
                    "responses": {"200": {"description": "Current proportional downscale policy"}},
                },
                "put": {
                    "summary": "Update the shared reference-media preprocessing defaults",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "image_resolution": {"type": "string", "enum": ["original", "360p", "480p", "720p"]},
                            "video_resolution": {"type": "string", "enum": ["original", "360p", "480p", "720p"]},
                        },
                    }}}},
                    "responses": {
                        "200": {"description": "Saved policy"},
                        "400": {"description": "Invalid resolution policy"},
                    },
                },
            },
            "/api/v1/settings/face-repair": {
                "get": {
                    "summary": "Read the shared face-repair Atlas defaults",
                    "responses": {"200": {"description": "Current square canvas and cell capacity"}},
                },
                "put": {
                    "summary": "Update the shared face-repair Atlas defaults",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "canvas_size": {"type": "integer", "minimum": 192, "maximum": 1088, "multipleOf": 32},
                            "capacity": {"type": "integer", "enum": [1, 4, 9, 16]},
                        },
                    }}}},
                    "responses": {
                        "200": {"description": "Saved face-repair defaults"},
                        "400": {"description": "Invalid repair settings"},
                    },
                },
            },
            "/api/v1/settings/second-sampling-window": {
                "get": {
                    "summary": "Read the global high-resolution temporal-window policy",
                    "responses": {"200": {"description": "Current switch, preferred duration, and H3-aligned effective geometry"}},
                },
                "put": {
                    "summary": "Update the global high-resolution temporal-window policy",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "required": ["enabled", "window_seconds", "overlap_seconds"],
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "window_seconds": {"type": "number", "minimum": 3, "maximum": 15},
                            "overlap_seconds": {"type": "number", "minimum": 0, "maximum": 4},
                        },
                    }}}},
                    "responses": {
                        "200": {"description": "Saved policy and effective H3-aligned geometry"},
                        "400": {"description": "Invalid temporal-window policy"},
                    },
                },
            },
            "/api/v1/settings/generation-limits": {
                "get": {
                    "summary": "Read per-resolution and per-ratio generation ceilings",
                    "description": (
                        "The returned preset matrix is authoritative "
                        "for Web, API and ComfyUI submissions."
                    ),
                    "responses": {"200": {"description": "Current limit policy"}},
                },
                "put": {
                    "summary": "Set every preset's maximum submission duration",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "required": ["preset_limits"],
                        "properties": {
                            "preset_limits": {
                                "type": "object",
                                "description": (
                                    "Complete resolution -> aspect ratio -> seconds matrix; "
                                    "each value is 1..15 in 0.5-second increments."
                                ),
                            },
                        },
                    }}}},
                    "responses": {
                        "200": {"description": "Saved policy and effective budget"},
                        "400": {"description": "Invalid limit policy"},
                    },
                },
            },
            "/api/v1/jobs/{job_id}": {
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "get": {"summary": "Get job state", "responses": {"200": {"description": "Job", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Job"}}}}}},
                "delete": {"summary": "Cancel a job", "responses": {"200": {"description": "Cancelled or cancelling"}}},
            },
            "/api/v1/jobs/records": {
                "delete": {
                    "summary": "Delete multiple job records and managed artifacts",
                    "description": (
                        "Best-effort batch form of the guarded single-record deletion. "
                        "Unknown or active jobs are returned in errors while other "
                        "selected records continue deleting."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {
                            "type": "object",
                            "required": ["job_ids"],
                            "properties": {"job_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 100,
                                "uniqueItems": True,
                                "items": {"type": "string"},
                            }},
                        }}},
                    },
                    "responses": {
                        "200": {"description": "Per-record deletion results"},
                        "400": {"description": "Invalid selection"},
                    },
                },
            },
            "/api/v1/jobs/{job_id}/video": {
                "parameters": [{"name": "job_id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "get": {"summary": "Download completed MP4", "responses": {"200": {"description": "Video", "content": {"video/mp4": {}}}, "409": {"description": "Not ready"}}},
            },
        },
    }
