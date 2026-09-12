#!/usr/bin/env python3
"""Build one persistent Native H3 session and run independent requests."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

# When invoked as ``python scripts/benchmark_native_hot_session.py``, Python
# otherwise places only ``scripts/`` at sys.path[0].  An editable installation
# can then win and load h3serve from the Windows checkout while Linux-only
# kernel packages are resolved relative to this mirror.  Pin the owning serve
# root before importing either h3serve or the lazy ``backends`` kernel package.
_BOOTSTRAP_SERVE_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_SERVE_ROOT))

import torch

from h3serve.native_engine import (
    HotSessionCheckpointResult,
    HotSessionRequest,
    NativeT2AVHotSession,
)
from h3serve.native_engine.forecast import QualityConstrainedForecastFactory
from h3serve.native_engine.adapters.conditioning_vae import (
    H3VideoVAEAdapter,
    PackedQwen3VLT2AVConditioner,
)
from h3serve.native_engine.adapters.vae_compile import (
    enable_transformer_block_compile,
    prewarm_feed_forward_compile,
    prewarm_transformer_block_compile,
)
from h3serve.native_engine.model import (
    SafeTensorSource,
    assemble_full_pruned_dit,
    comfy_kitchen_int8_kernel,
    load_full_silu_curve,
    load_larry_updates_from_safetensors,
    make_routed_sparge_attention_sm89,
    make_joint_action_scheduled_sparge_attention_sm89,
    make_joint_physical_action_backends_sm89,
    make_modality_protected_sparge_attention_sm89,
    make_split_modality_protected_sparge_attention_sm89,
    make_quality_constrained_adaptive_sparge_attention_sm89,
    make_budget_constrained_adaptive_sparge_attention_sm89,
    make_trajectory_layer_modality_routed_sparge_attention_sm89,
    make_trajectory_layer_modality_routed_sol_attention_sm89,
    make_layer_sensitivity_routed_split_sparge_attention_sm89,
    make_sparge_attention_sm89,
    ActionScheduledAttentionBackend,
    StepScheduledAttentionBackend,
    CausalCheckpointVerifierAttentionBackend,
    SplitModalityProtectedSpargeAttentionBackend,
    LayerHeadBudgetOverrideBackend,
    sage_attention_sm89,
    sage_attention_sm89_fused_k_quant,
)
from h3serve.native_engine.runtime import ImmutablePinnedModuleResidency
from h3serve.native_engine.planner import (
    ExecutionPlan,
    H3JointAccelerationScheduler,
    H3MechanisticParetoRuntimeSelector,
    H3WorkloadAnalyzer,
    JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP,
    JointWorkloadContext,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    capture_v19_runtime_fingerprint,
    RTX4090Planner,
    validated_lora_profiles_2026_08_11,
    validated_original_profiles_2026_08_11,
    load_h3_sparse_action_calibration,
    load_h3_mechanistic_deployment_config,
    solve_measured_h3_sparse_schedule,
    load_v19_candidate_blueprint,
    runtime_schedule_from_blueprint,
    v19_blueprint_execution_digest,
    V24FinalParetoRuntimeSelector,
    V24ResearchParetoRuntimeSelector,
    V19ActionUse,
)
from h3serve.native_engine.runtime import OffloadMode, RuntimeConfig
from h3serve.native_engine.sm89_policy import configure_sm89_runtime
try:
    from run_native_t2av import decode_video, load_audio_vae, load_video_vae
except ModuleNotFoundError:
    # Support import by unit tests and future candidate orchestrators while
    # preserving direct ``python scripts/...`` execution.
    from scripts.run_native_t2av import (
        decode_video,
        load_audio_vae,
        load_video_vae,
    )
from scripts.profile_h3_1080p15_attention_memory import NvidiaSmiSampler


PROMPTS = (
    "A cinematic aerial shot of medieval crusaders marching through deep mud "
    "toward a besieged stone fortress at dawn. Horses, banners and siege "
    "engines move naturally. Volumetric mist, realistic documentary style. "
    "Stereo audio: boots and hooves in mud, armor rattling, distant war drums "
    "and wind. No subtitles, no on-screen text.",
    "A dramatic historical documentary shot follows defenders on a medieval "
    "city wall during a rainstorm, torches moving naturally in the wind. "
    "Stereo audio: rain, distant bells, footsteps and restrained orchestral "
    "music. No subtitles, no on-screen text.",
)


def normalize_joint_policy_id(policy_id: str | None) -> str | None:
    """Translate the public Round229 CLI alias to its immutable policy id."""

    if policy_id == "round229":
        return JOINT_POLICY_V18_FORECAST_AWARE_FRONTIER_GLOBAL_DP
    return policy_id


def load_v19_blueprint_batch(
    path: Path,
) -> tuple[tuple[str, Path, object], ...]:
    """Load a named blueprint batch for one persistent model session."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1:
            raise ValueError("unsupported V19 blueprint batch schema")
        candidates = document["candidates"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid V19 blueprint batch: {path}") from error
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("V19 blueprint batch requires non-empty candidates")
    result = []
    names: set[str] = set()
    base = path.resolve().parent
    for index, row in enumerate(candidates):
        if not isinstance(row, dict):
            raise ValueError(f"V19 blueprint batch candidate {index} is not an object")
        name = re.sub(
            r"[^A-Za-z0-9_.-]", "_", str(row.get("name", ""))
        ).strip("._")
        if not name or name in names:
            raise ValueError("V19 blueprint batch candidate names must be unique")
        candidate_path = Path(str(row.get("blueprint", "")))
        if not candidate_path.is_absolute():
            candidate_path = (base / candidate_path).resolve()
        if not candidate_path.is_file():
            raise ValueError(f"V19 batch blueprint does not exist: {candidate_path}")
        result.append((name, candidate_path, load_v19_candidate_blueprint(candidate_path)))
        names.add(name)
    return tuple(result)


def parse_args() -> argparse.Namespace:
    serve_root = Path(__file__).resolve().parents[1]
    main_root = serve_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=("original", "lora", "reference", "reference-lora"),
        default="lora",
    )
    parser.add_argument(
        "--attention-backend",
        choices=(
            "sage",
            "sage-fused-k",
            "sparge",
            "routed",
            "modality-sparge",
            "split-modality-sparge",
            "split-layer-routed-sparge",
            "trajectory-layer-modality-sparge",
            "quality-adaptive-sparge",
            "budget-adaptive-sparge",
            "measured-budget-sparge",
            "trajectory-layer-modality-sol",
            "split-headwise-sparge",
            "head-calibration",
            "layer-calibration",
            "causal-calibration",
            "joint-scheduled",
            "long-mass-probe",
        ),
        default="sage",
        help="sparge is an experimental approximate backend, never an automatic route",
    )
    parser.add_argument("--sparge-topk", type=float, default=0.65)
    parser.add_argument("--long-mass-probe-topk", type=float, default=0.0625)
    parser.add_argument(
        "--long-mass-probe-reference-key-blocks", type=int, default=1565
    )
    parser.add_argument("--long-mass-probe-cap-multiplier", type=float, default=1.75)
    parser.add_argument("--long-mass-probe-min-retained", type=float, default=0.95)
    parser.add_argument(
        "--long-mass-probe-cap-ladder",
        default="",
        help="comma-separated absolute block caps observed without changing output",
    )
    parser.add_argument(
        "--adaptive-attention-budget",
        type=float,
        default=0.35,
        help=(
            "mean discretionary video-to-video KV-block fraction for the "
            "task-adaptive sparse controller"
        ),
    )
    parser.add_argument(
        "--adaptive-attention-safety",
        type=float,
        default=0.65,
        help=(
            "[0,1] strength for redistributing the fixed block quota toward "
            "diffuse, interaction-risk and historically sensitive rows"
        ),
    )
    parser.add_argument(
        "--measured-sparse-calibration",
        type=Path,
        help=(
            "full-56-head real-H3 action calibration for the measured budget "
            "scheduler; the request token shape must match exactly"
        ),
    )
    parser.add_argument(
        "--measured-attention-budget-ms",
        type=float,
        help="strict total Attention-kernel budget across all actual H3 steps",
    )
    parser.add_argument(
        "--measured-terminal-minimum-topk",
        type=float,
        default=0.10,
        help="minimum sparse action retained on the last three actual steps",
    )
    parser.add_argument(
        "--measured-relax-opening",
        action="store_true",
        help="research-only: permit sparse opening computation instead of the exact v1 floor",
    )
    parser.add_argument(
        "--measured-relax-causal-island",
        action="store_true",
        help="research-only: permit sparse layers 30--43/45 instead of the exact v1 floor",
    )
    parser.add_argument(
        "--piecewise-source",
        type=Path,
        default=main_root / "DIT-knowledge/sources/projects/pisa",
        help="research snapshot containing the PISA piecewise-attention kernels",
    )
    parser.add_argument(
        "--piecewise-probe-density",
        type=float,
        default=0.0,
        help=(
            "optional PISA exact-block density evaluated beside causal-calibration; "
            "the dense result still drives generation"
        ),
    )
    parser.add_argument(
        "--reference-rms-steps",
        default="",
        help=(
            "research-only comma-separated solver steps that retain the "
            "mature RMS/AdaLN operation order"
        ),
    )
    parser.add_argument(
        "--reference-rms-layers",
        default="",
        help=(
            "research-only comma-separated DiT layers that retain the "
            "mature RMS/AdaLN operation order"
        ),
    )
    parser.add_argument(
        "--sparse-selection-mode",
        choices=(
            "fixed_topk",
            "mass_budget",
            "mass_rebate",
            "route_cache",
            "temporal_motion_guard",
            "interaction_guard",
            "interaction_rail",
            "interaction_recovery",
            "interaction_rebalance",
            "interaction_hybrid",
            "interaction_dense",
            "disagreement_sentinel",
            "causal_head_guard",
            "unified_fixed_topk",
        ),
        default="fixed_topk",
        help=(
            "mass_budget redistributes the same per-head block budget across "
            "queries according to proxy attention mass"
        ),
    )
    parser.add_argument(
        "--probe-sparse-route-stability",
        action="store_true",
        help=(
            "research-only: record sampled cross-Actual fixed-TopK block-map "
            "Jaccard without changing the executed sparse map"
        ),
    )
    parser.add_argument("--layer-routed-aggressive-topk", type=float, default=0.35)
    parser.add_argument(
        "--layer-routed-safe-topk",
        type=float,
        default=0.50,
        help=(
            "conservative sparse budget for calibration-sensitive layers; "
            "must be greater than --layer-routed-aggressive-topk"
        ),
    )
    parser.add_argument(
        "--layer-routed-sensitive-layers",
        default="30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45",
    )
    parser.add_argument(
        "--teacher-layer-head-policy",
        type=Path,
        help=(
            "offline dense-teacher phase/layer/head budget map; research-only "
            "and never exposed as a product quality control"
        ),
    )
    parser.add_argument(
        "--layer-routed-aggressive-head-topks",
        help=(
            "optional comma-separated 56-head budgets for non-sensitive layers; "
            "overrides --layer-routed-aggressive-topk"
        ),
    )
    parser.add_argument(
        "--layer-routed-safe-head-topks",
        help=(
            "optional comma-separated 56-head budgets for sensitive layers; "
            "overrides --layer-routed-safe-topk"
        ),
    )
    parser.add_argument(
        "--trajectory-anchor-steps",
        default="",
        help=(
            "comma-separated solver steps that use a conservative sparse "
            "head budget instead of either dense or the normal trajectory budget"
        ),
    )
    parser.add_argument(
        "--trajectory-anchor-aggressive-head-topks",
        help="comma-separated 56-head budgets for non-sensitive anchor layers",
    )
    parser.add_argument(
        "--trajectory-anchor-safe-head-topks",
        help="comma-separated 56-head budgets for sensitive anchor layers",
    )
    parser.add_argument(
        "--trajectory-recovery-steps",
        default="",
        help="comma-separated late solver steps that use the recovery head budget",
    )
    parser.add_argument(
        "--trajectory-recovery-aggressive-head-topks",
        help="comma-separated 56-head budgets for non-sensitive recovery layers",
    )
    parser.add_argument(
        "--trajectory-recovery-safe-head-topks",
        help="comma-separated 56-head budgets for sensitive recovery layers",
    )
    parser.add_argument(
        "--experimental-minimum-sparse-topk",
        type=float,
        default=0.25,
        help=(
            "research-only lower bound for trajectory/layer-routed head budgets; "
            "production default remains 0.25"
        ),
    )
    parser.add_argument(
        "--temporal-correspondence-radius",
        type=int,
        default=-1,
        help=(
            "protect spatially aligned video-key blocks within this many latent "
            "frames; -1 disables the experimental temporal rail"
        ),
    )
    parser.add_argument(
        "--temporal-spatial-block-radius",
        type=int,
        default=0,
        help="extra 64-token key-block halo around each temporal correspondence",
    )
    parser.add_argument(
        "--temporal-global-anchor-stride",
        type=int,
        default=0,
        help=(
            "retain same-location keys from one rotating remote latent-frame "
            "lattice; 0 disables MTCR and values >=2 select the frame stride"
        ),
    )
    parser.add_argument(
        "--temporal-global-spatial-block-radius",
        type=int,
        default=0,
        help="extra 64-token spatial halo for rotating remote MTCR anchors",
    )
    parser.add_argument(
        "--sol-attn-source",
        type=Path,
        default=main_root / "DIT-knowledge/sources/projects/solattn_h3",
        help="research snapshot containing the standalone Sol-Attn Triton kernels",
    )
    parser.add_argument("--sol-tau", type=float, default=1.0)
    parser.add_argument("--sol-sensitive-tau", type=float, default=0.8)
    parser.add_argument("--sol-anchor-tau", type=float, default=0.6)
    parser.add_argument("--sol-recovery-tau", type=float, default=0.8)
    parser.add_argument("--sol-sensitive-layers", default="30,31,32,33,34,35,36,37,38,39,40,41,42,43,45")
    parser.add_argument("--sol-anchor-steps", default="0")
    parser.add_argument("--sol-recovery-steps", default="17,18,19")
    parser.add_argument(
        "--frame-interleave-stride",
        type=int,
        default=1,
        help=(
            "compute one rotating complete latent-frame subset per block; "
            "1 disables the experimental training-free approximation"
        ),
    )
    parser.add_argument("--frame-interleave-layer-start", type=int, default=0)
    parser.add_argument("--frame-interleave-layer-stop", type=int, default=50)
    parser.add_argument(
        "--frame-interleave-dense-layers",
        default="",
        help="comma-separated H3 layers that retain every video frame",
    )
    parser.add_argument(
        "--frame-interleave-dense-steps",
        default="",
        help="comma-separated denoise steps that retain every video frame",
    )
    parser.add_argument(
        "--spatial-query-lattice-stride",
        type=int,
        default=1,
        help=(
            "rotate native 128-row generated-video Query blocks across layers; "
            "1 disables SQLR"
        ),
    )
    parser.add_argument("--spatial-query-lattice-layer-start", type=int, default=0)
    parser.add_argument("--spatial-query-lattice-layer-stop", type=int, default=50)
    parser.add_argument(
        "--spatial-query-lattice-dense-layers",
        default="",
        help="comma-separated H3 layers that retain all generated-video Queries",
    )
    parser.add_argument(
        "--spatial-query-lattice-dense-steps",
        default="",
        help="comma-separated denoise steps that retain all generated-video Queries",
    )
    parser.add_argument(
        "--mlp-spatial-lattice-stride",
        type=int,
        default=1,
        help=(
            "evaluate the row-local MLP on one rotating spatial-column lattice; "
            "attention remains complete and 1 disables the experiment"
        ),
    )
    parser.add_argument("--mlp-spatial-lattice-layer-start", type=int, default=0)
    parser.add_argument("--mlp-spatial-lattice-layer-stop", type=int, default=50)
    parser.add_argument(
        "--mlp-spatial-lattice-dense-layers",
        default="",
        help="comma-separated layers that retain exact MLP updates for every video row",
    )
    parser.add_argument(
        "--mlp-spatial-lattice-dense-steps",
        default="",
        help="comma-separated denoise steps that retain exact MLP updates",
    )
    parser.add_argument(
        "--mlp-spatial-lattice-detail-fraction",
        type=float,
        default=0.0,
        help="fraction of omitted rows restored exactly by live hidden-error ranking",
    )
    parser.add_argument("--segment-cache-layer-start", type=int, default=0)
    parser.add_argument("--segment-cache-layer-stop", type=int, default=0)
    parser.add_argument(
        "--segment-cache-reuse-steps",
        default="",
        help=(
            "comma-separated actual denoise steps that reuse one coordinate-aligned "
            "block-segment residual; empty disables the experiment"
        ),
    )
    parser.add_argument(
        "--segment-cache-directional-trust",
        action="store_true",
        help=(
            "gate and scale residual extrapolation from the live pre-segment "
            "feature direction instead of fixed step-index distance"
        ),
    )
    parser.add_argument(
        "--segment-cache-directional-max-extra", type=float, default=0.35
    )
    parser.add_argument(
        "--segment-cache-directional-min-cosine", type=float, default=0.25
    )
    parser.add_argument(
        "--segment-cache-protected-refresh",
        action="store_true",
        help=(
            "refresh text/condition/audio rows through cached blocks and apply "
            "the same-coordinate residual prediction only to generated video"
        ),
    )
    parser.add_argument(
        "--segment-cache-active-video-ratio",
        type=float,
        default=0.0,
        help=(
            "fraction of highest-risk, 128-row-aligned generated-video query "
            "blocks to refresh through a reused segment"
        ),
    )
    parser.add_argument(
        "--segment-cache-dynamic-video-budget",
        action="store_true",
        help=(
            "choose the smallest active-video block budget that covers the "
            "requested same-coordinate residual-innovation mass; the active "
            "video ratio becomes a fail-closed maximum"
        ),
    )
    parser.add_argument(
        "--segment-cache-active-video-min-ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--segment-cache-innovation-risk-coverage", type=float, default=0.80
    )
    parser.add_argument(
        "--segment-cache-innovation-max-relative", type=float, default=4.0
    )
    parser.add_argument(
        "--segment-cache-active-layer-start",
        type=int,
        default=0,
        help=(
            "first cached H3 layer that refreshes selected video blocks; "
            "0/0 means the full segment-cache layer range"
        ),
    )
    parser.add_argument(
        "--segment-cache-active-layer-stop",
        type=int,
        default=0,
        help=(
            "exclusive cached H3 layer that refreshes selected video blocks; "
            "0/0 means the full segment-cache layer range"
        ),
    )
    parser.add_argument(
        "--segment-cache-sequential-layer-groups",
        action="store_true",
        help=(
            "predict cached residuals in layer order around the active-video "
            "group so selected refreshes consume the preceding predicted state"
        ),
    )
    parser.add_argument(
        "--segment-cache-sequential-conservative-hold",
        action="store_true",
        help=(
            "when one sequential group leaves its directional trust region, "
            "reuse its newest same-coordinate residual without extrapolation"
        ),
    )
    parser.add_argument(
        "--sparge-head-topks",
        help="comma-separated 56-head sparse budgets for a calibrated review route",
    )
    parser.add_argument(
        "--head-calibration-topks",
        default="0.50,0.65,0.75",
        help="comma-separated sparse budgets for one real-H3 head sensitivity probe",
    )
    parser.add_argument(
        "--head-calibration-output",
        type=Path,
        help="write the one-shot real-H3 per-head diagnostic as JSON",
    )
    parser.add_argument(
        "--head-calibration-stop-after-probe",
        action="store_true",
        help="diagnostic-only: stop after the first true H3 long-attention probe",
    )
    parser.add_argument(
        "--head-calibration-group-size",
        type=int,
        default=8,
        help="probe this many heads at once to keep the 4090 diagnostic below 24GB",
    )
    parser.add_argument(
        "--layer-calibration-output",
        type=Path,
        help="write real-H3 per-layer sparse sensitivity metrics as JSON",
    )
    parser.add_argument(
        "--layer-calibration-full-head-topks",
        default="",
        help=(
            "optional comma-separated full-56-head sparse actions measured beside "
            "Dense on the same teacher trajectory; unlike the low-memory grouped "
            "probe, these timings are valid inputs to the RTX 4090 budget planner"
        ),
    )
    parser.add_argument(
        "--layer-calibration-warm-repeats",
        type=int,
        default=0,
        help=(
            "additional warmed physical calls retained as raw V19 timing samples; "
            "use at least 3 for a planner-eligible p90 artifact"
        ),
    )
    parser.add_argument(
        "--layer-calibration-action-implementation",
        choices=("round215", "round188", "round228", "round229"),
        default="round215",
        help=(
            "exact production executor family to probe; calibration and runtime "
            "use the same action factory"
        ),
    )
    parser.add_argument(
        "--layer-calibration-step",
        type=int,
        default=3,
        help=(
            "zero-based denoise step to probe; defaults to step 3 because the "
            "validated original route keeps the earlier anchor steps dense"
        ),
    )
    parser.add_argument(
        "--layer-calibration-steps",
        help=(
            "optional comma-separated denoise steps to probe in one dense "
            "teacher trajectory; supersedes --layer-calibration-step"
        ),
    )
    parser.add_argument(
        "--layer-calibration-stop-after-complete",
        action="store_true",
        help="diagnostic-only: stop once all 50 true H3 layers are measured",
    )
    parser.add_argument(
        "--sparge-dense-steps",
        help=(
            "comma-separated zero-based steps that retain dense SageAttention; "
            "other long AV attention calls use Sparge"
        ),
    )
    parser.add_argument(
        "--sparge-dense-layers",
        help=(
            "comma-separated zero-based H3 layers that always retain dense "
            "SageAttention; intended for calibration-derived protection"
        ),
    )
    parser.add_argument(
        "--sparge-dense-step-layer-map",
        help=(
            "research-only semicolon-separated step=layer Cartesian groups, "
            "for example '1,4=34,39,40;17,18,19=39,40,42'"
        ),
    )
    parser.add_argument(
        "--causal-verifier-effort",
        type=float,
        help=(
            "research-only unified [0,1] causal verification effort; fixed model "
            "and sampler steps, with original-attention probes and fail-closed recovery"
        ),
    )
    parser.add_argument(
        "--causal-verifier-inject-queries",
        action="store_true",
        help=(
            "reuse exact verifier rows as per-frame spatial correction rails; "
            "requires --causal-verifier-effort"
        ),
    )
    parser.add_argument(
        "--causal-verifier-repair-heads",
        action="store_true",
        help=(
            "recompute a verifier-selected coherent set of high-error heads; "
            "requires --causal-verifier-effort"
        ),
    )
    parser.add_argument(
        "--causal-verifier-graded-recovery",
        action="store_true",
        help=(
            "research-only: let the original-attention checkpoint accept a "
            "coherent 0.50 sparse recovery band when it satisfies the same "
            "request-local error envelope; otherwise recover dense"
        ),
    )
    parser.add_argument(
        "--causal-verifier-early-hysteresis",
        action="store_true",
        help=(
            "research-only: keep the rejected step fully protected but limit "
            "the following-step continuity hold to state-forming layers 30-39"
        ),
    )
    parser.add_argument(
        "--causal-verifier-probe-first",
        action="store_true",
        help=(
            "research-only: evaluate aligned sparse query blocks before the "
            "full draft so rejected checkpoints skip discarded sparse work"
        ),
    )
    parser.add_argument(
        "--causal-verifier-shared-kv-probe",
        action="store_true",
        help=(
            "research-only: retain the original verifier rows and decisions "
            "while sharing the sparse draft's prepared FP8-V with exact probes"
        ),
    )
    parser.add_argument(
        "--causal-verifier-head-island",
        action="store_true",
        help=(
            "research-only: on checkpoint rejection, replace full dense-layer "
            "fallback with an automatically selected, step-persistent set of "
            "complete exact attention heads"
        ),
    )
    parser.add_argument(
        "--self-speculative-verify-steps",
        help=(
            "research-only comma-separated actual solver steps for a whole-DiT "
            "dense endpoint verification"
        ),
    )
    parser.add_argument(
        "--self-speculative-verify-threshold",
        type=float,
        default=None,
        help="maximum whole-DiT relative RMS before accepting dense rollback",
    )
    parser.add_argument(
        "--sparge-build-dir",
        type=Path,
        help="isolated SpargeAttention build to prepend to sys.path",
    )
    parser.add_argument(
        "--quant-backend",
        choices=("cuda", "triton"),
        default="cuda",
        help="explicit Comfy-Kitchen backend; never rely on package priority defaults",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--forecast-controller",
        choices=("directional", "quality-curvature"),
        default="directional",
        help=(
            "quality-curvature uses request 1 as a full trajectory calibration "
            "and automatically minimizes full DiT evaluations on later requests"
        ),
    )
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--vae-tile-size",
        type=int,
        help=(
            "internal Video-VAE decode tile; omitted uses the model default. "
            "This is an experimental execution parameter, not a quality preset."
        ),
    )
    parser.add_argument(
        "--vae-tile-batch-size",
        type=int,
        default=1,
        help="maximum spatial Video-VAE tiles decoded together on the RTX 4090",
    )
    parser.add_argument(
        "--vae-compile-feed-forward",
        action="store_true",
        help="compile only the Video-VAE ViT FFN region; excludes CUDA graphs",
    )
    parser.add_argument(
        "--vae-compile-transformer-block",
        action="store_true",
        help=(
            "prebuild request-scoped VAE TransformerBlock Inductor graphs; "
            "experimental and CUDA graphs remain disabled"
        ),
    )
    parser.add_argument(
        "--fused-rms-adaln",
        action="store_true",
        help="enable the validated fused RMS/AdaLN mechanical path",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--mlp-chunk-tokens",
        type=int,
        help="explicit request-level MLP chunk; overrides legacy environment defaults",
    )
    parser.add_argument(
        "--long-sequence-query-chunk-tokens",
        type=int,
        help=(
            "request-local exact Query streaming chunk; memory geometry only, "
            "never a prompt/content control"
        ),
    )
    parser.add_argument(
        "--long-sequence-projection-chunk-tokens",
        type=int,
        default=8192,
        help="row chunk used by the long-sequence QKV/out projection path",
    )
    parser.add_argument(
        "--long-sequence-split-qkv-outputs",
        action="store_true",
        help=(
            "use exact output-row Q/K/V projection slices only inside the "
            "request-local long-sequence path"
        ),
    )
    parser.add_argument(
        "--long-sequence-shared-qkv-quantization",
        action="store_true",
        help=(
            "research-only: reuse one exact ConvRot row-INT8 activation for "
            "the separated Query and K/V projections"
        ),
    )
    parser.add_argument(
        "--long-sequence-exact-helper-stack",
        action="store_true",
        help=(
            "reproduce the quarantined v015 three-helper bundle; the legacy "
            "name does not imply full-video exactness"
        ),
    )
    parser.add_argument(
        "--long-sequence-single-qknorm-rope",
        action="store_true",
        help="enable the exact single-sided partial Q/K Norm+RoPE kernel",
    )
    parser.add_argument(
        "--long-sequence-parallel-sparse-lut",
        action="store_true",
        help="isolate parallel sparse block-map LUT construction",
    )
    parser.add_argument(
        "--long-sequence-partial-sparse-topk",
        action="store_true",
        help="isolate partial Top-K selection after softmax",
    )
    parser.add_argument(
        "--long-sequence-fused-prefix-k-quant",
        action="store_true",
        help="isolate fused protected-prefix K quantization",
    )
    parser.add_argument(
        "--long-sequence-fused-query-projection",
        action="store_true",
        help="research-only: project each streamed Query slab directly",
    )
    parser.add_argument(
        "--long-sequence-fused-qknorm-hnd-layout",
        action="store_true",
        help=(
            "research-only: write exact single-sided Q/K Norm+RoPE directly "
            "into the HND attention layout"
        ),
    )
    parser.add_argument(
        "--long-sequence-direct-nhd-output",
        action="store_true",
        help="research-only: materialize sparse attention output in projection layout",
    )
    parser.add_argument(
        "--long-sequence-direct-nhd-kv",
        action="store_true",
        help="research-only: keep streamed sparse K/V preparation natively NHD",
    )
    parser.add_argument(
        "--long-sequence-direct-hnd-fp8-value",
        action="store_true",
        help=(
            "research-only: quantize HND V directly into Sage's final FP8 ABI"
        ),
    )
    parser.add_argument(
        "--offload-mode",
        choices=("resident", "block"),
        default="resident",
        help="internal transformer weight-residency strategy",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        choices=(0, 1),
        default=1,
        help="0 is serial correctness reference; 1 overlaps next-block H2D",
    )
    parser.add_argument(
        "--auto-route",
        action="store_true",
        help="select a fail-closed measured RTX4090 plan after prompt encoding",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("auto", "performance", "low_vram"),
        default="auto",
        help=(
            "public physical-memory route: auto keeps the performance plan "
            "when it fits and selects full-context low-VRAM streaming under pressure"
        ),
    )
    parser.add_argument(
        "--enforce-vram-gib",
        type=float,
        help=(
            "diagnostic hard allocator cap for validating a smaller-card route "
            "on the current GPU; also sets the native planning capacity"
        ),
    )
    parser.add_argument(
        "--resident-block-count",
        type=int,
        default=0,
        help="hybrid Block plan: keep this prefix of the 50 DiT blocks on GPU",
    )
    parser.add_argument(
        "--actual-steps",
        help="comma-separated zero-based full DiT steps (original engine only)",
    )
    parser.add_argument(
        "--joint-policy",
        help=(
            "research-only joint scheduler policy; requires --joint-acceleration "
            "and --attention-backend joint-scheduled"
        ),
    )
    parser.add_argument(
        "--joint-acceleration",
        type=float,
        help="research-only [0,100] budget dial for --joint-policy",
    )
    parser.add_argument(
        "--pareto-acceleration",
        "--v24-acceleration",
        dest="v24_acceleration",
        metavar="ACCELERATION",
        type=float,
        help=(
            "run the configured Pareto selector after exact Qwen tokenisation; "
            "mutually exclusive with legacy joint policies and V19 blueprints"
        ),
    )
    parser.add_argument(
        "--mechanistic-admission",
        type=Path,
        help=(
            "schedule-free mechanistic risk admission; when omitted the "
            "production V24 selector remains the comparator"
        ),
    )
    parser.add_argument(
        "--v24-research-calibration",
        help=(
            "exact historical calibration id for offline replay; when omitted "
            "the immutable production C02 surface is used"
        ),
    )
    parser.add_argument(
        "--v19-blueprint",
        type=Path,
        help=(
            "execute one digest-sealed V19 physical schedule; mutually exclusive "
            "with the legacy joint-policy scheduler"
        ),
    )
    parser.add_argument(
        "--v19-blueprint-manifest",
        type=Path,
        help=(
            "execute multiple named V19 blueprints in one persistent model "
            "session; mutually exclusive with --v19-blueprint and --joint-policy"
        ),
    )
    parser.add_argument(
        "--v19-batch-final-latents-dir",
        type=Path,
        help=(
            "save one uniquely named final AV latent artifact per V19 batch "
            "candidate; requires --v19-blueprint-manifest"
        ),
    )
    parser.add_argument("--seed", type=int, default=8833)
    parser.add_argument(
        "--prompt",
        help="override the first request prompt (and all requests unless --second-prompt is set)",
    )
    parser.add_argument("--second-prompt")
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument("--reference-image", type=Path, action="append", default=[])
    parser.add_argument("--reference-video", type=Path, action="append", default=[])
    parser.add_argument("--reference-audio", type=Path, action="append", default=[])
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        help=(
            "run exact name/seed/prompt entries from a JSON manifest's "
            "scenarios array"
        ),
    )
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        help=(
            "expand one anchor scenario into one request per registered "
            "candidate, preserving a single hot model session"
        ),
    )
    parser.add_argument(
        "--label-prefix",
        help="stable output filename prefix used with --scenario-manifest",
    )
    parser.add_argument("--debug-step-dir", type=Path)
    parser.add_argument(
        "--sampler-state",
        type=Path,
        help=(
            "resume a formally paused noisy sampler checkpoint and reschedule "
            "its remaining sigma interval across --steps without replaying the prefix"
        ),
    )
    parser.add_argument(
        "--checkpoint-after-step",
        type=int,
        help="stop after this many formal solver steps and save exact state",
    )
    parser.add_argument(
        "--checkpoint-state",
        type=Path,
        help="output path for --checkpoint-after-step",
    )
    parser.add_argument(
        "--debug-final-latents",
        type=Path,
        help="save final AV latents for an isolated same-latent decoder A/B",
    )
    parser.add_argument("--preview-step-index", type=int)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--preview-latents", type=Path)
    parser.add_argument(
        "--preview-decode-mode",
        choices=("direct_x0", "fast_finish"),
        default="direct_x0",
        help=(
            "direct_x0 decodes the clean prediction already computed by the "
            "formal step; fast_finish runs a disposable solver branch"
        ),
    )
    parser.add_argument(
        "--preview-forecast-steps",
        type=int,
        default=0,
        help=(
            "also finish a comparison branch using only the existing "
            "directional forecast tail for this many sigma transitions"
        ),
    )
    parser.add_argument("--preview-forecast-output", type=Path)
    parser.add_argument(
        "--preview-branch-steps",
        type=int,
        default=2,
        help="real DiT evaluations used only to fast-finish the preview branch",
    )
    parser.add_argument(
        "--preview-branch-actual-steps",
        help=(
            "comma-separated local branch indices that run real DiT; omitted "
            "means every branch step is real"
        ),
    )
    parser.add_argument(
        "--preview-branch-spatial-scale",
        type=float,
        default=1.0,
        help=(
            "disposable preview latent scale; values below one trade preview "
            "resolution for more affordable convergence steps"
        ),
    )
    parser.add_argument(
        "--preview-branch-warm-history",
        action="store_true",
        help="seed the preview RES solver with the formal trajectory's prior x0",
    )
    parser.add_argument(
        "--preview-branch-force-dense",
        action="store_true",
        help="disable approximate attention only inside the disposable preview branch",
    )
    parser.add_argument(
        "--preview-branch-use-lora",
        action="store_true",
        help=(
            "temporarily enable the hot Larry adapters and Turbo sampler only "
            "for the disposable preview branch"
        ),
    )
    parser.add_argument(
        "--preview-audio-branch-use-lora",
        action="store_true",
        help=(
            "retain full-resolution Base preview video but replace its audio "
            "with a separate temporary Larry/Turbo companion branch"
        ),
    )
    parser.add_argument(
        "--preview-audio-branch-steps",
        type=int,
        default=4,
        help="real Larry/Turbo evaluations used by the audio-only companion branch",
    )
    parser.add_argument(
        "--preview-audio-branch-spatial-scale",
        type=float,
        default=0.65,
        help=(
            "video-canvas scale used only while predicting companion audio; "
            "the companion video is discarded"
        ),
    )
    parser.add_argument(
        "--pause-for-preview-decision",
        action="store_true",
        help=(
            "after publishing the isolated preview, wait on stdin for "
            "'continue' or 'discard' while preserving the formal trajectory"
        ),
    )
    parser.add_argument(
        "--preview-decision-file",
        type=Path,
        help=(
            "after publishing the preview, wait until this file contains "
            "'continue' or 'discard'; suitable for detached benchmark runs"
        ),
    )
    parser.add_argument(
        "--profile-request",
        type=int,
        help=(
            "one-based request index to capture with PyTorch Kineto; use an "
            "earlier request in the same process for kernel warmup"
        ),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("H3_SERVE_MODEL_DIR", serve_root / "models")),
    )
    parser.add_argument(
        "--minimax-source",
        type=Path,
        default=Path(
            os.environ.get("H3_SERVE_MINIMAX_SOURCE", main_root / "MiniMax-H3")
        ),
    )
    parser.add_argument(
        "--lightx-source",
        type=Path,
        default=Path(
            os.environ.get(
                "H3_SERVE_LIGHTX_SOURCE",
                main_root.parent / "backend-compare/sources/LightX2V",
            )
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=serve_root / "runtime/outputs/hot_session",
    )
    parser.add_argument(
        "--memory-profile",
        choices=("fullspeed", "generation_hot", "compact"),
        default="compact",
        help="match the service host-weight and Qwen-cache residency policy",
    )
    parser.add_argument(
        "--disable-condition-row-cache",
        action="store_true",
        help=(
            "replay Ref2VA v00 by rebuilding immutable reference video/audio "
            "rows at every denoise step"
        ),
    )
    parser.add_argument(
        "--disable-reference-latent-cache",
        action="store_true",
        help="disable the exact one-entry Ref2VA Video/Audio-VAE latent cache",
    )
    parser.add_argument(
        "--cache-condition-embeddings",
        action="store_true",
        help=(
            "reuse projected Ref2VA condition embeddings across denoise steps; "
            "mathematically equivalent but not claimed bit-exact"
        ),
    )
    args = parser.parse_args()
    args.joint_policy = normalize_joint_policy_id(args.joint_policy)
    for option_name in ("sol_sensitive_layers", "sol_anchor_steps", "sol_recovery_steps"):
        raw = getattr(args, option_name)
        try:
            values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
        except ValueError:
            parser.error(f"--{option_name.replace('_', '-')} must be comma-separated integers")
        if tuple(sorted(set(values))) != values or any(value < 0 for value in values):
            parser.error(f"--{option_name.replace('_', '-')} must be sorted, unique and non-negative")
        setattr(args, option_name, values)
    if set(args.sol_anchor_steps) & set(args.sol_recovery_steps):
        parser.error("Sol-Attn anchor and recovery steps must be disjoint")
    if any(
        value <= 0.0
        for value in (
            args.sol_tau,
            args.sol_sensitive_tau,
            args.sol_anchor_tau,
            args.sol_recovery_tau,
        )
    ):
        parser.error("Sol-Attn tau values must be positive")
    if args.attention_backend == "trajectory-layer-modality-sol" and not (
        args.sol_attn_source / "_int8_fwd.py"
    ).is_file():
        parser.error("trajectory Sol-Attn requires a complete --sol-attn-source")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if (
        args.sparse_selection_mode in ("mass_budget", "mass_rebate", "route_cache")
        and args.attention_backend != "trajectory-layer-modality-sparge"
    ):
        parser.error(
            "mass-adaptive selection currently requires "
            "--attention-backend trajectory-layer-modality-sparge"
        )
    if (
        args.probe_sparse_route_stability
        and args.attention_backend != "joint-scheduled"
    ):
        parser.error(
            "--probe-sparse-route-stability requires "
            "--attention-backend joint-scheduled"
        )
    if args.forecast_controller == "quality-curvature":
        if args.engine != "original":
            parser.error("quality-curvature currently requires --engine original")
        if args.steps not in (None, 20):
            parser.error("quality-curvature currently requires --steps 20")
        if args.repeat < 2:
            parser.error("quality-curvature requires --repeat 2 or greater")
    if not 0.5 <= args.sparge_topk <= 1.0:
        parser.error("--sparge-topk must be between 0.5 and 1.0")
    if args.frame_interleave_stride <= 0:
        parser.error("--frame-interleave-stride must be positive")
    if not (
        0
        <= args.frame_interleave_layer_start
        <= args.frame_interleave_layer_stop
        <= 50
    ):
        parser.error("frame interleave layer range must lie inside [0, 50]")
    for name in ("frame_interleave_dense_layers", "frame_interleave_dense_steps"):
        raw = getattr(args, name)
        try:
            values = tuple(
                int(value.strip()) for value in raw.split(",") if value.strip()
            )
        except ValueError:
            parser.error(f"--{name.replace('_', '-')} must be comma-separated integers")
        if tuple(sorted(set(values))) != values or any(value < 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} must be sorted, unique and non-negative")
        setattr(args, name, values)
    if any(value >= 50 for value in args.frame_interleave_dense_layers):
        parser.error("frame interleave dense layers must lie inside [0, 50)")
    if args.spatial_query_lattice_stride <= 0:
        parser.error("--spatial-query-lattice-stride must be positive")
    if not (
        0
        <= args.spatial_query_lattice_layer_start
        <= args.spatial_query_lattice_layer_stop
        <= 50
    ):
        parser.error("spatial Query lattice layer range must lie inside [0, 50]")
    for name in (
        "spatial_query_lattice_dense_layers",
        "spatial_query_lattice_dense_steps",
    ):
        raw = getattr(args, name)
        try:
            values = tuple(
                int(value.strip()) for value in raw.split(",") if value.strip()
            )
        except ValueError:
            parser.error(f"--{name.replace('_', '-')} must be comma-separated integers")
        if tuple(sorted(set(values))) != values or any(value < 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} must be sorted, unique and non-negative")
        setattr(args, name, values)
    if any(value >= 50 for value in args.spatial_query_lattice_dense_layers):
        parser.error("spatial Query lattice dense layers must lie inside [0, 50)")
    if args.mlp_spatial_lattice_stride <= 0:
        parser.error("--mlp-spatial-lattice-stride must be positive")
    if not (
        0 <= args.mlp_spatial_lattice_layer_start
        <= args.mlp_spatial_lattice_layer_stop <= 50
    ):
        parser.error("MLP spatial lattice layer range must lie inside [0, 50]")
    for name in (
        "mlp_spatial_lattice_dense_layers",
        "mlp_spatial_lattice_dense_steps",
    ):
        raw = getattr(args, name)
        try:
            values = tuple(
                int(value.strip()) for value in raw.split(",") if value.strip()
            )
        except ValueError:
            parser.error(f"--{name.replace('_', '-')} must be comma-separated integers")
        if tuple(sorted(set(values))) != values or any(value < 0 for value in values):
            parser.error(f"--{name.replace('_', '-')} must be sorted, unique and non-negative")
        setattr(args, name, values)
    if any(value >= 50 for value in args.mlp_spatial_lattice_dense_layers):
        parser.error("MLP spatial lattice dense layers must lie inside [0, 50)")
    if not 0.0 <= args.mlp_spatial_lattice_detail_fraction < 1.0:
        parser.error("--mlp-spatial-lattice-detail-fraction must lie inside [0, 1)")
    try:
        args.segment_cache_reuse_steps = tuple(
            int(value.strip())
            for value in args.segment_cache_reuse_steps.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--segment-cache-reuse-steps must be comma-separated integers")
    if (
        tuple(sorted(set(args.segment_cache_reuse_steps)))
        != args.segment_cache_reuse_steps
        or any(value < 0 for value in args.segment_cache_reuse_steps)
    ):
        parser.error("segment cache reuse steps must be sorted, unique and non-negative")
    if not (
        0 <= args.segment_cache_layer_start <= args.segment_cache_layer_stop <= 50
    ):
        parser.error("segment cache layer range must lie inside [0, 50]")
    if bool(args.segment_cache_reuse_steps) != (
        args.segment_cache_layer_start < args.segment_cache_layer_stop
    ):
        parser.error("segment cache requires a non-empty layer range and reuse steps")
    if args.segment_cache_directional_trust and not args.segment_cache_reuse_steps:
        parser.error("directional trust requires segment cache reuse steps")
    if args.segment_cache_protected_refresh and not args.segment_cache_reuse_steps:
        parser.error("protected refresh requires segment cache reuse steps")
    if not 0.0 <= args.segment_cache_active_video_ratio <= 1.0:
        parser.error("active video ratio must lie inside [0, 1]")
    if args.segment_cache_active_video_ratio and not args.segment_cache_protected_refresh:
        parser.error("active video routing requires protected refresh")
    if (
        args.segment_cache_dynamic_video_budget
        and not args.segment_cache_active_video_ratio
    ):
        parser.error("dynamic video budgeting requires a non-zero maximum ratio")
    if not (
        0.0
        <= args.segment_cache_active_video_min_ratio
        <= args.segment_cache_active_video_ratio
    ):
        parser.error("active video minimum ratio must lie inside [0, maximum ratio]")
    if not 0.0 < args.segment_cache_innovation_risk_coverage <= 1.0:
        parser.error("innovation risk coverage must lie inside (0, 1]")
    if args.segment_cache_innovation_max_relative <= 0.0:
        parser.error("innovation relative-risk limit must be positive")
    if not (
        0
        <= args.segment_cache_active_layer_start
        <= args.segment_cache_active_layer_stop
        <= 50
    ):
        parser.error("active video layer range must lie inside [0, 50]")
    has_active_layer_range = (
        args.segment_cache_active_layer_start
        < args.segment_cache_active_layer_stop
    )
    if has_active_layer_range and not args.segment_cache_active_video_ratio:
        parser.error("active video layer range requires a non-zero video ratio")
    if has_active_layer_range and not (
        args.segment_cache_layer_start
        <= args.segment_cache_active_layer_start
        < args.segment_cache_active_layer_stop
        <= args.segment_cache_layer_stop
    ):
        parser.error(
            "active video layer range must lie inside the segment cache range"
        )
    if args.segment_cache_sequential_layer_groups and not (
        args.segment_cache_protected_refresh
        and args.segment_cache_active_video_ratio
        and has_active_layer_range
    ):
        parser.error(
            "sequential layer groups require protected refresh, active video, "
            "and an explicit active layer range"
        )
    if (
        args.segment_cache_sequential_conservative_hold
        and not args.segment_cache_sequential_layer_groups
    ):
        parser.error(
            "sequential conservative hold requires sequential layer groups"
        )
    if not 0.0 <= args.segment_cache_directional_max_extra <= 1.0:
        parser.error("directional max extra must lie inside [0, 1]")
    if not -1.0 <= args.segment_cache_directional_min_cosine <= 1.0:
        parser.error("directional minimum cosine must lie inside [-1, 1]")
    if args.attention_backend in (
        "sparge",
        "routed",
        "modality-sparge",
        "split-modality-sparge",
        "split-layer-routed-sparge",
        "trajectory-layer-modality-sparge",
        "quality-adaptive-sparge",
        "budget-adaptive-sparge",
        "measured-budget-sparge",
        "split-headwise-sparge",
        "head-calibration",
        "layer-calibration",
        "causal-calibration",
        "joint-scheduled",
        "long-mass-probe",
    ):
        if args.sparge_build_dir is None or not args.sparge_build_dir.is_dir():
            parser.error("sparse attention requires an existing --sparge-build-dir")
    if bool(args.joint_policy) != (args.joint_acceleration is not None):
        parser.error("--joint-policy and --joint-acceleration must be used together")
    if args.v24_acceleration is not None:
        if not 0.0 <= args.v24_acceleration <= 100.0:
            parser.error("--v24-acceleration must lie inside [0,100]")
        if args.engine not in ("original", "reference"):
            parser.error("V24 currently requires a Base engine")
        if args.attention_backend != "joint-scheduled":
            parser.error("--v24-acceleration requires joint-scheduled Attention")
        if args.joint_policy is not None:
            parser.error("use either V24 or a legacy joint policy")
    if args.mechanistic_admission is not None:
        if args.v24_acceleration is None:
            parser.error(
                "--mechanistic-admission requires --pareto-acceleration"
            )
        if not args.mechanistic_admission.is_file():
            parser.error(
                "--mechanistic-admission does not exist: "
                f"{args.mechanistic_admission}"
            )
    if args.v24_research_calibration and args.v24_acceleration is None:
        parser.error(
            "--v24-research-calibration requires --pareto-acceleration"
        )
    if not 0.0625 <= args.long_mass_probe_topk <= 1.0:
        parser.error("--long-mass-probe-topk must lie inside [0.0625, 1]")
    if args.long_mass_probe_reference_key_blocks <= 0:
        parser.error("--long-mass-probe-reference-key-blocks must be positive")
    if not 1.0 <= args.long_mass_probe_cap_multiplier <= 4.0:
        parser.error("--long-mass-probe-cap-multiplier must lie inside [1, 4]")
    if not 0.0 < args.long_mass_probe_min_retained <= 1.0:
        parser.error("--long-mass-probe-min-retained must lie inside (0, 1]")
    try:
        args.long_mass_probe_cap_ladder = tuple(
            int(value.strip())
            for value in args.long_mass_probe_cap_ladder.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--long-mass-probe-cap-ladder must be comma-separated integers")
    if (
        tuple(sorted(set(args.long_mass_probe_cap_ladder)))
        != args.long_mass_probe_cap_ladder
        or any(value <= 0 for value in args.long_mass_probe_cap_ladder)
    ):
        parser.error("--long-mass-probe-cap-ladder must be sorted, unique and positive")
    if args.v19_blueprint is not None and args.v19_blueprint_manifest is not None:
        parser.error("use either --v19-blueprint or --v19-blueprint-manifest")
    if (
        args.v19_batch_final_latents_dir is not None
        and args.v19_blueprint_manifest is None
    ):
        parser.error(
            "--v19-batch-final-latents-dir requires --v19-blueprint-manifest"
        )
    v19_input = args.v19_blueprint or args.v19_blueprint_manifest
    if v19_input is not None:
        if not v19_input.is_file():
            parser.error(f"V19 blueprint input does not exist: {v19_input}")
        if args.attention_backend != "joint-scheduled":
            parser.error("V19 blueprints require --attention-backend joint-scheduled")
        if args.joint_policy is not None:
            parser.error("use either V19 blueprints or --joint-policy")
        if args.v24_acceleration is not None:
            parser.error("use either V19 blueprints or V24")
        if args.engine not in ("original", "reference"):
            parser.error("V19 physical blueprints currently require a Base engine")
    if args.joint_policy is not None:
        if args.attention_backend != "joint-scheduled":
            parser.error("--joint-policy requires --attention-backend joint-scheduled")
        if args.engine not in ("original", "reference"):
            parser.error("V19 forecast calibration currently requires a Base engine")
        if not 0.0 <= args.joint_acceleration <= 100.0:
            parser.error("--joint-acceleration must lie inside [0,100]")
    elif (
        args.attention_backend == "joint-scheduled"
        and v19_input is None
        and args.v24_acceleration is None
    ):
        parser.error(
            "joint-scheduled requires --v19-blueprint, "
            "--v19-blueprint-manifest or --joint-policy"
        )
    if not 0.0625 <= args.adaptive_attention_budget <= 1.0:
        parser.error("--adaptive-attention-budget must lie inside [0.0625, 1]")
    if not 0.0 <= args.adaptive_attention_safety <= 1.0:
        parser.error("--adaptive-attention-safety must lie inside [0, 1]")
    measured_options_used = (
        args.measured_sparse_calibration is not None
        or args.measured_attention_budget_ms is not None
        or args.measured_relax_opening
        or args.measured_relax_causal_island
    )
    if args.attention_backend == "measured-budget-sparge":
        if (
            args.measured_sparse_calibration is None
            or not args.measured_sparse_calibration.is_file()
        ):
            parser.error(
                "measured-budget-sparge requires an existing "
                "--measured-sparse-calibration"
            )
        if (
            args.measured_attention_budget_ms is None
            or not math.isfinite(args.measured_attention_budget_ms)
            or args.measured_attention_budget_ms <= 0.0
        ):
            parser.error(
                "measured-budget-sparge requires a positive finite "
                "--measured-attention-budget-ms"
            )
    elif measured_options_used:
        parser.error(
            "measured sparse scheduling options require measured-budget-sparge"
        )
    if args.sparge_head_topks:
        try:
            args.sparge_head_topks = tuple(
                float(value.strip()) for value in args.sparge_head_topks.split(",")
            )
        except ValueError:
            parser.error("--sparge-head-topks must be comma-separated numbers")
        minimum_head_topk = (
            args.experimental_minimum_sparse_topk
            if args.attention_backend in ("layer-calibration", "causal-calibration")
            else 0.5
        )
        if len(args.sparge_head_topks) != 56 or any(
            not minimum_head_topk <= value <= 1.0
            for value in args.sparge_head_topks
        ):
            parser.error(
                "--sparge-head-topks requires 56 values between "
                f"{minimum_head_topk} and 1.0"
            )
        if args.attention_backend not in (
            "split-headwise-sparge",
            "layer-calibration",
            "causal-calibration",
        ):
            parser.error(
                "--sparge-head-topks requires split-headwise-sparge or a calibration backend"
            )
    elif args.attention_backend == "split-headwise-sparge":
        parser.error("split-headwise-sparge requires --sparge-head-topks")
    for option_name in (
        "layer_routed_aggressive_head_topks",
        "layer_routed_safe_head_topks",
        "trajectory_anchor_aggressive_head_topks",
        "trajectory_anchor_safe_head_topks",
        "trajectory_recovery_aggressive_head_topks",
        "trajectory_recovery_safe_head_topks",
    ):
        raw = getattr(args, option_name)
        if raw is None:
            continue
        try:
            parsed = tuple(float(value.strip()) for value in raw.split(","))
        except ValueError:
            parser.error(f"--{option_name.replace('_', '-')} must be comma-separated numbers")
        if len(parsed) != 56 or any(
            not args.experimental_minimum_sparse_topk <= value <= 1.0
            for value in parsed
        ):
            parser.error(
                f"--{option_name.replace('_', '-')} requires 56 values between "
                f"{args.experimental_minimum_sparse_topk} and 1.0"
            )
        setattr(args, option_name, parsed)
    if bool(args.layer_routed_aggressive_head_topks) != bool(
        args.layer_routed_safe_head_topks
    ):
        parser.error("layer-routed head-wise policies require both aggressive and safe budgets")
    if args.layer_routed_aggressive_head_topks:
        if args.attention_backend not in (
            "trajectory-layer-modality-sparge",
            "causal-calibration",
        ):
            parser.error(
                "layer-routed head-wise policies require trajectory-layer-modality-sparge"
            )
        if any(
            aggressive >= safe
            for aggressive, safe in zip(
                args.layer_routed_aggressive_head_topks,
                args.layer_routed_safe_head_topks,
            )
        ):
            parser.error("every aggressive head budget must be smaller than its safe budget")
    if bool(args.trajectory_anchor_aggressive_head_topks) != bool(
        args.trajectory_anchor_safe_head_topks
    ):
        parser.error("trajectory anchor head-wise policies require both budgets")
    if args.trajectory_anchor_aggressive_head_topks:
        if args.attention_backend not in (
            "trajectory-layer-modality-sparge",
            "causal-calibration",
        ):
            parser.error(
                "trajectory anchor head-wise policies require "
                "trajectory-layer-modality-sparge"
            )
        if any(
            aggressive >= safe
            for aggressive, safe in zip(
                args.trajectory_anchor_aggressive_head_topks,
                args.trajectory_anchor_safe_head_topks,
            )
        ):
            parser.error("every anchor aggressive head budget must be smaller than safe")
    if bool(args.trajectory_recovery_aggressive_head_topks) != bool(
        args.trajectory_recovery_safe_head_topks
    ):
        parser.error("trajectory recovery head-wise policies require both budgets")
    if args.trajectory_recovery_aggressive_head_topks:
        if args.attention_backend not in (
            "trajectory-layer-modality-sparge",
            "causal-calibration",
        ):
            parser.error(
                "trajectory recovery head-wise policies require "
                "trajectory-layer-modality-sparge"
            )
        if any(
            aggressive >= safe
            for aggressive, safe in zip(
                args.trajectory_recovery_aggressive_head_topks,
                args.trajectory_recovery_safe_head_topks,
            )
        ):
            parser.error("every recovery aggressive head budget must be smaller than safe")
    try:
        args.head_calibration_topks = tuple(
            float(value.strip())
            for value in args.head_calibration_topks.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--head-calibration-topks must be comma-separated numbers")
    if not args.head_calibration_topks or any(
        not 0.5 <= value <= 1.0 for value in args.head_calibration_topks
    ):
        parser.error("head calibration budgets must be between 0.5 and 1.0")
    if args.attention_backend == "head-calibration" and args.head_calibration_output is None:
        parser.error("--attention-backend head-calibration requires --head-calibration-output")
    if args.attention_backend == "layer-calibration" and args.layer_calibration_output is None:
        parser.error("--attention-backend layer-calibration requires --layer-calibration-output")
    if args.attention_backend == "causal-calibration" and args.layer_calibration_output is None:
        parser.error("--attention-backend causal-calibration requires --layer-calibration-output")
    try:
        args.layer_calibration_full_head_topks = tuple(
            float(value.strip())
            for value in args.layer_calibration_full_head_topks.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error(
            "--layer-calibration-full-head-topks must be comma-separated numbers"
        )
    if args.layer_calibration_full_head_topks:
        if args.attention_backend != "layer-calibration":
            parser.error(
                "--layer-calibration-full-head-topks requires layer-calibration"
            )
        if any(
            not args.experimental_minimum_sparse_topk <= value <= 1.0
            for value in args.layer_calibration_full_head_topks
        ):
            parser.error(
                "full-head calibration budgets must lie between the configured "
                "experimental minimum and 1.0"
            )
        if len(set(args.layer_calibration_full_head_topks)) != len(
            args.layer_calibration_full_head_topks
        ):
            parser.error("full-head calibration budgets must be unique")
    if not 0.0 <= args.piecewise_probe_density <= 1.0:
        parser.error("--piecewise-probe-density must be between 0 and 1")
    if args.layer_calibration_step < 0:
        parser.error("--layer-calibration-step must be non-negative")
    if args.layer_calibration_steps:
        try:
            args.layer_calibration_steps = tuple(
                int(value.strip())
                for value in args.layer_calibration_steps.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error("--layer-calibration-steps must be comma-separated integers")
        if (
            not args.layer_calibration_steps
            or tuple(sorted(set(args.layer_calibration_steps)))
            != args.layer_calibration_steps
            or any(
                value < 0 or value >= args.steps
                for value in args.layer_calibration_steps
            )
        ):
            parser.error(
                "--layer-calibration-steps must be sorted, unique and inside --steps"
            )
    else:
        args.layer_calibration_steps = (args.layer_calibration_step,)
    if args.head_calibration_group_size <= 0:
        parser.error("--head-calibration-group-size must be positive")
    if args.layer_calibration_warm_repeats < 0:
        parser.error("--layer-calibration-warm-repeats cannot be negative")
    if args.attention_backend in (
        "split-layer-routed-sparge",
        "trajectory-layer-modality-sparge",
        "causal-calibration",
    ) and not (
        args.experimental_minimum_sparse_topk
        <= args.layer_routed_aggressive_topk
        < args.layer_routed_safe_topk
        <= 1.0
    ):
        parser.error(
            "layer-routed budgets must satisfy configured minimum <= aggressive "
            "< safe <= 1.0"
        )
    if args.teacher_layer_head_policy is not None:
        if args.attention_backend not in (
            "trajectory-layer-modality-sparge",
            "causal-calibration",
        ):
            parser.error(
                "--teacher-layer-head-policy requires "
                "trajectory-layer-modality-sparge"
            )
        if not args.teacher_layer_head_policy.is_file():
            parser.error("teacher layer/head policy file does not exist")
    if not 0.0625 <= args.experimental_minimum_sparse_topk <= 0.5:
        parser.error("--experimental-minimum-sparse-topk must be inside [0.0625, 0.5]")
    if (
        args.attention_backend == "measured-budget-sparge"
        and not args.experimental_minimum_sparse_topk
        <= args.measured_terminal_minimum_topk
        <= 1.0
    ):
        parser.error(
            "--measured-terminal-minimum-topk must lie inside the configured sparse envelope"
        )
    if args.temporal_correspondence_radius < -1:
        parser.error("--temporal-correspondence-radius must be >= -1")
    if args.temporal_spatial_block_radius < 0:
        parser.error("--temporal-spatial-block-radius cannot be negative")
    if args.temporal_global_anchor_stride not in (0,) and (
        args.temporal_global_anchor_stride < 2
    ):
        parser.error("--temporal-global-anchor-stride must be 0 or >= 2")
    if args.temporal_global_spatial_block_radius < 0:
        parser.error("--temporal-global-spatial-block-radius cannot be negative")
    if args.temporal_global_anchor_stride and args.temporal_correspondence_radius < 0:
        parser.error("remote MTCR anchors require the local temporal rail")
    if args.temporal_correspondence_radius >= 0 and args.attention_backend not in (
        "trajectory-layer-modality-sparge",
        "measured-budget-sparge",
        "layer-calibration",
        "causal-calibration",
    ):
        parser.error(
            "temporal correspondence protection requires trajectory-layer-modality-sparge"
        )
    if args.mlp_chunk_tokens is not None and args.mlp_chunk_tokens <= 0:
        parser.error("--mlp-chunk-tokens must be positive")
    if args.long_sequence_query_chunk_tokens is not None and (
        args.long_sequence_query_chunk_tokens < 128
        or args.long_sequence_query_chunk_tokens % 128
    ):
        parser.error(
            "--long-sequence-query-chunk-tokens must be a positive multiple of 128"
        )
    if args.long_sequence_projection_chunk_tokens <= 0:
        parser.error("--long-sequence-projection-chunk-tokens must be positive")
    if (
        args.long_sequence_query_chunk_tokens is not None
        and args.long_sequence_projection_chunk_tokens
        > args.long_sequence_query_chunk_tokens
    ):
        parser.error("long-sequence projection chunks cannot exceed Query chunks")
    if bool(args.checkpoint_after_step is None) != bool(args.checkpoint_state is None):
        parser.error(
            "--checkpoint-after-step and --checkpoint-state must be used together"
        )
    if args.vae_tile_size is not None and (
        args.vae_tile_size < 128 or args.vae_tile_size % 16
    ):
        parser.error("--vae-tile-size must be >= 128 and divisible by 16")
    if args.vae_tile_batch_size <= 0:
        parser.error("--vae-tile-batch-size must be positive")
    if args.vae_compile_feed_forward and args.vae_compile_transformer_block:
        parser.error(
            "--vae-compile-feed-forward and --vae-compile-transformer-block "
            "are mutually exclusive"
        )
    if not 0 <= args.resident_block_count < 50:
        parser.error("--resident-block-count must be between 0 and 49")
    if args.profile_request is not None and not 1 <= args.profile_request <= args.repeat:
        parser.error("--profile-request must be between 1 and --repeat")
    if args.steps is None:
        args.steps = 6 if args.engine in ("lora", "reference-lora") else 20
    if args.attention_backend == "trajectory-layer-modality-sol":
        if any(step >= args.steps for step in args.sol_anchor_steps + args.sol_recovery_steps):
            parser.error("Sol-Attn anchor/recovery steps must lie inside --steps")
        if any(layer >= 50 for layer in args.sol_sensitive_layers):
            parser.error("Sol-Attn sensitive layers must lie inside [0, 50)")
    if args.sparge_dense_steps:
        try:
            args.sparge_dense_steps = tuple(
                int(value.strip()) for value in args.sparge_dense_steps.split(",")
            )
        except ValueError:
            parser.error("--sparge-dense-steps must be comma-separated integers")
        if args.attention_backend not in (
            "sparge",
            "split-headwise-sparge",
            "trajectory-layer-modality-sparge",
        ):
            parser.error(
                "--sparge-dense-steps requires sparge, split-headwise-sparge, "
                "or trajectory-layer-modality-sparge"
            )
        if (
            tuple(sorted(set(args.sparge_dense_steps)))
            != args.sparge_dense_steps
            or any(index < 0 or index >= args.steps for index in args.sparge_dense_steps)
        ):
            parser.error("--sparge-dense-steps must be sorted, unique and inside --steps")
    else:
        args.sparge_dense_steps = ()
    if args.trajectory_anchor_steps:
        try:
            args.trajectory_anchor_steps = tuple(
                int(value.strip())
                for value in args.trajectory_anchor_steps.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error("--trajectory-anchor-steps must be comma-separated integers")
        if args.attention_backend not in (
            "trajectory-layer-modality-sparge",
            "causal-calibration",
        ):
            parser.error(
                "--trajectory-anchor-steps requires trajectory-layer-modality-sparge"
            )
        if (
            tuple(sorted(set(args.trajectory_anchor_steps)))
            != args.trajectory_anchor_steps
            or any(index < 0 or index >= args.steps for index in args.trajectory_anchor_steps)
        ):
            parser.error("--trajectory-anchor-steps must be sorted, unique and inside --steps")
        if set(args.trajectory_anchor_steps) & set(args.sparge_dense_steps):
            parser.error("trajectory anchor and dense steps must be disjoint")
        if not args.trajectory_anchor_aggressive_head_topks:
            parser.error("trajectory anchor steps require both anchor head budgets")
    else:
        args.trajectory_anchor_steps = ()
        if args.trajectory_anchor_aggressive_head_topks:
            parser.error("trajectory anchor head budgets require anchor steps")
    if args.trajectory_recovery_steps:
        try:
            args.trajectory_recovery_steps = tuple(
                int(value.strip())
                for value in args.trajectory_recovery_steps.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error("--trajectory-recovery-steps must be comma-separated integers")
        if args.attention_backend not in (
            "trajectory-layer-modality-sparge",
            "causal-calibration",
        ):
            parser.error(
                "--trajectory-recovery-steps requires trajectory-layer-modality-sparge"
            )
        if (
            tuple(sorted(set(args.trajectory_recovery_steps)))
            != args.trajectory_recovery_steps
            or any(
                index < 0 or index >= args.steps
                for index in args.trajectory_recovery_steps
            )
        ):
            parser.error(
                "--trajectory-recovery-steps must be sorted, unique and inside --steps"
            )
        if (
            set(args.trajectory_recovery_steps) & set(args.sparge_dense_steps)
            or set(args.trajectory_recovery_steps) & set(args.trajectory_anchor_steps)
        ):
            parser.error("trajectory recovery, anchor and dense steps must be disjoint")
        if not args.trajectory_recovery_aggressive_head_topks:
            parser.error("trajectory recovery steps require both recovery head budgets")
    else:
        args.trajectory_recovery_steps = ()
        if args.trajectory_recovery_aggressive_head_topks:
            parser.error("trajectory recovery head budgets require recovery steps")
    if args.sparge_dense_layers:
        try:
            args.sparge_dense_layers = tuple(
                int(value.strip()) for value in args.sparge_dense_layers.split(",")
            )
        except ValueError:
            parser.error("--sparge-dense-layers must be comma-separated integers")
        if args.attention_backend not in (
            "sparge",
            "split-headwise-sparge",
            "trajectory-layer-modality-sparge",
        ):
            parser.error(
                "--sparge-dense-layers requires sparge, split-headwise-sparge, "
                "or trajectory-layer-modality-sparge"
            )
        if (
            tuple(sorted(set(args.sparge_dense_layers)))
            != args.sparge_dense_layers
            or any(index < 0 or index >= 50 for index in args.sparge_dense_layers)
        ):
            parser.error(
                "--sparge-dense-layers must be sorted, unique and inside [0, 50)"
            )
    else:
        args.sparge_dense_layers = ()
    dense_step_layer_pairs: set[tuple[int, int]] = set()
    if args.sparge_dense_step_layer_map:
        try:
            for group in args.sparge_dense_step_layer_map.split(";"):
                raw_steps, raw_layers = group.split("=", 1)
                steps = tuple(
                    int(value.strip())
                    for value in raw_steps.split(",")
                    if value.strip()
                )
                layers = tuple(
                    int(value.strip())
                    for value in raw_layers.split(",")
                    if value.strip()
                )
                if not steps or not layers:
                    raise ValueError
                dense_step_layer_pairs.update(
                    (step, layer) for step in steps for layer in layers
                )
        except ValueError:
            parser.error(
                "--sparge-dense-step-layer-map must use step,step=layer,layer groups"
            )
        if args.attention_backend != "trajectory-layer-modality-sparge":
            parser.error(
                "--sparge-dense-step-layer-map requires "
                "trajectory-layer-modality-sparge"
            )
        if any(
            step < 0 or step >= args.steps or layer < 0 or layer >= 50
            for step, layer in dense_step_layer_pairs
        ):
            parser.error("dense step/layer pairs must lie inside --steps and [0, 50)")
    args.sparge_dense_step_layer_pairs = tuple(sorted(dense_step_layer_pairs))
    if args.causal_verifier_effort is not None:
        if args.attention_backend != "trajectory-layer-modality-sparge":
            parser.error(
                "--causal-verifier-effort requires trajectory-layer-modality-sparge"
            )
        if not 0.0 <= args.causal_verifier_effort <= 1.0:
            parser.error("--causal-verifier-effort must be inside [0, 1]")
        if args.sparge_dense_layers or args.sparge_dense_step_layer_pairs:
            parser.error(
                "--causal-verifier-effort owns the causal protection budget and "
                "cannot be combined with manual dense layer maps"
            )
    elif args.causal_verifier_inject_queries:
        parser.error(
            "--causal-verifier-inject-queries requires --causal-verifier-effort"
        )
    elif args.causal_verifier_repair_heads:
        parser.error(
            "--causal-verifier-repair-heads requires --causal-verifier-effort"
        )
    elif args.causal_verifier_graded_recovery:
        parser.error(
            "--causal-verifier-graded-recovery requires --causal-verifier-effort"
        )
    elif args.causal_verifier_early_hysteresis:
        parser.error(
            "--causal-verifier-early-hysteresis requires --causal-verifier-effort"
        )
    elif args.causal_verifier_probe_first:
        parser.error(
            "--causal-verifier-probe-first requires --causal-verifier-effort"
        )
    elif args.causal_verifier_shared_kv_probe:
        parser.error(
            "--causal-verifier-shared-kv-probe requires --causal-verifier-effort"
        )
    elif args.causal_verifier_head_island:
        parser.error(
            "--causal-verifier-head-island requires --causal-verifier-effort"
        )
    if args.causal_verifier_inject_queries and args.causal_verifier_repair_heads:
        parser.error("causal verifier correction modes are mutually exclusive")
    if (
        args.causal_verifier_probe_first
        and args.causal_verifier_shared_kv_probe
    ):
        parser.error("causal verifier probe execution modes are mutually exclusive")
    if args.self_speculative_verify_steps:
        try:
            args.self_speculative_verify_steps = tuple(
                int(value.strip())
                for value in args.self_speculative_verify_steps.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error("--self-speculative-verify-steps must be comma-separated integers")
        if (
            not args.self_speculative_verify_steps
            or tuple(sorted(set(args.self_speculative_verify_steps)))
            != args.self_speculative_verify_steps
            or any(
                step < 0 or step >= args.steps
                for step in args.self_speculative_verify_steps
            )
        ):
            parser.error("self-speculative verify steps must be sorted and inside --steps")
    else:
        args.self_speculative_verify_steps = ()
    if (
        args.self_speculative_verify_threshold is not None
        and (
            not math.isfinite(args.self_speculative_verify_threshold)
            or args.self_speculative_verify_threshold < 0.0
        )
    ):
        parser.error("self-speculative verify threshold cannot be negative")
    if args.actual_steps:
        try:
            args.actual_steps = tuple(
                int(value.strip()) for value in args.actual_steps.split(",")
            )
        except ValueError:
            parser.error("--actual-steps must be comma-separated integers")
        if args.engine not in ("original", "reference"):
            parser.error("--actual-steps is supported by original/reference engines")
        if (
            not args.actual_steps
            or tuple(sorted(set(args.actual_steps))) != args.actual_steps
            or any(index < 0 or index >= args.steps for index in args.actual_steps)
        ):
            parser.error("--actual-steps must be sorted, unique and inside --steps")
    else:
        args.actual_steps = tuple(range(args.steps))
    if args.preview_branch_actual_steps:
        try:
            args.preview_branch_actual_steps = tuple(
                int(value.strip())
                for value in args.preview_branch_actual_steps.split(",")
                if value.strip()
            )
        except ValueError:
            parser.error(
                "--preview-branch-actual-steps must be comma-separated integers"
            )
        if (
            tuple(sorted(set(args.preview_branch_actual_steps)))
            != args.preview_branch_actual_steps
            or any(
                value < 0 or value >= args.preview_branch_steps
                for value in args.preview_branch_actual_steps
            )
        ):
            parser.error(
                "preview branch actual steps must be sorted, unique and inside the branch"
            )
    else:
        args.preview_branch_actual_steps = None
    if args.scenario_manifest is not None and not args.scenario_manifest.is_file():
        parser.error(f"scenario manifest does not exist: {args.scenario_manifest}")
    if args.candidate_registry is not None and not args.candidate_registry.is_file():
        parser.error(f"candidate registry does not exist: {args.candidate_registry}")
    if args.candidate_registry is not None and args.scenario_manifest is not None:
        parser.error("use either --candidate-registry or --scenario-manifest")
    for role in ("first_frame", "last_frame"):
        path = getattr(args, role)
        if path is not None and not path.is_file():
            parser.error(f"--{role.replace('_', '-')} does not exist: {path}")
    for role in ("reference_image", "reference_video", "reference_audio"):
        for path in getattr(args, role):
            if not path.is_file():
                parser.error(f"--{role.replace('_', '-')} does not exist: {path}")
    return args


def load_scenarios(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.candidate_registry is not None:
        registry_path = args.candidate_registry.resolve()
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        anchor_path = registry_path.parent / str(registry["anchor"])
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        source = anchor.get("scenarios")
        candidates = registry.get("candidates")
        if not isinstance(source, list) or len(source) != 1:
            raise ValueError("candidate anchor must contain exactly one scenario")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidate registry must contain candidates")
        base = source[0]
        if not isinstance(base, dict):
            raise ValueError("candidate anchor scenario must be an object")
        expanded = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("candidate entry must be an object")
            expanded.append({
                **base,
                "name": str(candidate["version"]),
                "steps": candidate.get("steps", base.get("steps", args.steps)),
                "actual_step_indices": candidate.get(
                    "actual_steps", base.get("actual_step_indices", args.actual_steps)
                ),
                "resident_block_count": candidate.get(
                    "resident_block_count", base.get("resident_block_count", 0)
                ),
                "attention_topk": candidate.get("attention_topk"),
                "fused_rms_adaln": candidate.get(
                    "fused_rms_adaln", args.fused_rms_adaln
                ),
                "dense_qk_quant_gran": candidate.get(
                    "dense_qk_quant_gran", "per_thread"
                ),
                "long_sequence_query_chunk_tokens": candidate.get(
                    "long_sequence_query_chunk_tokens",
                    base.get(
                        "long_sequence_query_chunk_tokens",
                        args.long_sequence_query_chunk_tokens,
                    ),
                ),
                "long_sequence_projection_chunk_tokens": candidate.get(
                    "long_sequence_projection_chunk_tokens",
                    base.get(
                        "long_sequence_projection_chunk_tokens",
                        args.long_sequence_projection_chunk_tokens,
                    ),
                ),
                "long_sequence_split_qkv_outputs": candidate.get(
                    "long_sequence_split_qkv_outputs",
                    base.get(
                        "long_sequence_split_qkv_outputs",
                        args.long_sequence_split_qkv_outputs,
                    ),
                ),
                "long_sequence_shared_qkv_quantization": candidate.get(
                    "long_sequence_shared_qkv_quantization",
                    base.get(
                        "long_sequence_shared_qkv_quantization",
                        args.long_sequence_shared_qkv_quantization,
                    ),
                ),
                "long_sequence_exact_helper_stack": candidate.get(
                    "long_sequence_exact_helper_stack",
                    base.get(
                        "long_sequence_exact_helper_stack",
                        args.long_sequence_exact_helper_stack,
                    ),
                ),
                "long_sequence_single_qknorm_rope": candidate.get(
                    "long_sequence_single_qknorm_rope",
                    base.get(
                        "long_sequence_single_qknorm_rope",
                        args.long_sequence_single_qknorm_rope,
                    ),
                ),
                "long_sequence_parallel_sparse_lut": candidate.get(
                    "long_sequence_parallel_sparse_lut",
                    base.get(
                        "long_sequence_parallel_sparse_lut",
                        args.long_sequence_parallel_sparse_lut,
                    ),
                ),
                "long_sequence_partial_sparse_topk": candidate.get(
                    "long_sequence_partial_sparse_topk",
                    base.get(
                        "long_sequence_partial_sparse_topk",
                        args.long_sequence_partial_sparse_topk,
                    ),
                ),
                "long_sequence_fused_prefix_k_quant": candidate.get(
                    "long_sequence_fused_prefix_k_quant",
                    base.get(
                        "long_sequence_fused_prefix_k_quant",
                        args.long_sequence_fused_prefix_k_quant,
                    ),
                ),
                "long_sequence_fused_query_projection": candidate.get(
                    "long_sequence_fused_query_projection",
                    base.get(
                        "long_sequence_fused_query_projection",
                        args.long_sequence_fused_query_projection,
                    ),
                ),
                "long_sequence_fused_qknorm_hnd_layout": candidate.get(
                    "long_sequence_fused_qknorm_hnd_layout",
                    base.get(
                        "long_sequence_fused_qknorm_hnd_layout",
                        args.long_sequence_fused_qknorm_hnd_layout,
                    ),
                ),
                "long_sequence_direct_nhd_output": candidate.get(
                    "long_sequence_direct_nhd_output",
                    base.get(
                        "long_sequence_direct_nhd_output",
                        args.long_sequence_direct_nhd_output,
                    ),
                ),
                "long_sequence_direct_nhd_kv": candidate.get(
                    "long_sequence_direct_nhd_kv",
                    base.get(
                        "long_sequence_direct_nhd_kv",
                        args.long_sequence_direct_nhd_kv,
                    ),
                ),
                "long_sequence_direct_hnd_fp8_value": candidate.get(
                    "long_sequence_direct_hnd_fp8_value",
                    base.get(
                        "long_sequence_direct_hnd_fp8_value",
                        args.long_sequence_direct_hnd_fp8_value,
                    ),
                ),
                "cache_condition_rows": candidate.get(
                    "cache_condition_rows", True
                ),
                "cache_condition_embeddings": candidate.get(
                    "cache_condition_embeddings", False
                ),
                "cache_reference_latents": candidate.get(
                    "cache_reference_latents", True
                ),
                "save_final_latents_path": candidate.get(
                    "save_final_latents_path",
                    base.get("save_final_latents_path"),
                ),
                "width": candidate.get("width", base.get("width")),
                "height": candidate.get("height", base.get("height")),
                "refinement_latents_path": candidate.get(
                    "refinement_latents_path",
                    base.get("refinement_latents_path"),
                ),
                "refinement_denoise": candidate.get(
                    "refinement_denoise", base.get("refinement_denoise")
                ),
                "refinement_spatial_mode": candidate.get(
                    "refinement_spatial_mode",
                    base.get("refinement_spatial_mode", "strict"),
                ),
                "preserve_refinement_audio": candidate.get(
                    "preserve_refinement_audio",
                    base.get("preserve_refinement_audio", True),
                ),
                "multiscale_initial_width": candidate.get(
                    "multiscale_initial_width",
                    base.get("multiscale_initial_width"),
                ),
                "multiscale_initial_height": candidate.get(
                    "multiscale_initial_height",
                    base.get("multiscale_initial_height"),
                ),
                "multiscale_resize_after_step": candidate.get(
                    "multiscale_resize_after_step",
                    base.get("multiscale_resize_after_step"),
                ),
                "multiscale_highpass_strength": candidate.get(
                    "multiscale_highpass_strength",
                    base.get("multiscale_highpass_strength", 1.0),
                ),
                "terminal_refinement_initial_width": candidate.get(
                    "terminal_refinement_initial_width",
                    base.get("terminal_refinement_initial_width"),
                ),
                "terminal_refinement_initial_height": candidate.get(
                    "terminal_refinement_initial_height",
                    base.get("terminal_refinement_initial_height"),
                ),
                "terminal_refinement_steps": candidate.get(
                    "terminal_refinement_steps",
                    base.get("terminal_refinement_steps", 0),
                ),
                "terminal_refinement_denoise": candidate.get(
                    "terminal_refinement_denoise",
                    base.get("terminal_refinement_denoise", 0.0125),
                ),
                "terminal_refinement_dense_tail_steps": candidate.get(
                    "terminal_refinement_dense_tail_steps",
                    base.get("terminal_refinement_dense_tail_steps", 1),
                ),
                "terminal_refinement_low_frequency_gain": candidate.get(
                    "terminal_refinement_low_frequency_gain",
                    base.get("terminal_refinement_low_frequency_gain", 1.0),
                ),
                "terminal_refinement_temporal_lowpass": candidate.get(
                    "terminal_refinement_temporal_lowpass",
                    base.get("terminal_refinement_temporal_lowpass", False),
                ),
                "terminal_refinement_temporal_outlier_only": candidate.get(
                    "terminal_refinement_temporal_outlier_only",
                    base.get("terminal_refinement_temporal_outlier_only", False),
                ),
            })
        # Reuse the existing manifest validator/resolver without writing a
        # generated file: temporarily point resolution at the anchor and feed
        # the normalized objects through the common loop below.
        document_override = {"scenarios": expanded}
        manifest_path_override = anchor_path
    else:
        document_override = None
        manifest_path_override = args.scenario_manifest
    manifest_base = (
        Path.cwd()
        if manifest_path_override is None
        else manifest_path_override.resolve().parents[2]
    )

    def resolve_paths(values):
        return tuple(
            path if path.is_absolute() else (manifest_base / path).resolve()
            for path in (Path(value) for value in values or ())
        )

    if args.scenario_manifest is None and args.candidate_registry is None:
        return [
            {
                "name": f"request{index + 1}",
                "seed": args.seed + index,
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "prompt": (
                    args.second_prompt
                    if index > 0 and args.second_prompt
                    else args.prompt
                    if args.prompt
                    else PROMPTS[index % len(PROMPTS)]
                ),
                "first_frame": args.first_frame,
                "last_frame": args.last_frame,
                "reference_images": resolve_paths(args.reference_image),
                "reference_videos": resolve_paths(args.reference_video),
                "reference_audios": resolve_paths(args.reference_audio),
            }
            for index in range(args.repeat)
        ]

    document = (
        document_override
        if document_override is not None
        else json.loads(args.scenario_manifest.read_text(encoding="utf-8"))
    )
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario manifest must contain a non-empty scenarios array")
    result: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario {index} must be an object")
        name = str(scenario.get("name", "")).strip()
        prompt = str(scenario.get("prompt", "")).strip()
        seed = scenario.get("seed")
        width = scenario.get("width", args.width)
        height = scenario.get("height", args.height)
        frames = scenario.get("frames", args.frames)
        steps = scenario.get("steps", args.steps)
        if not name or not prompt or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"scenario {index} requires name, integer seed and prompt")
        geometry = {"width": width, "height": height, "frames": frames, "steps": steps}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in geometry.values()
        ):
            raise ValueError(
                f"scenario {index} width, height and frames must be positive integers"
            )
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._")
        if not safe_name:
            raise ValueError(f"scenario {index} name has no safe filename characters")
        base = {
            "seed": seed,
            "prompt": prompt,
            **geometry,
            "memory_mode": scenario.get("memory_mode", args.memory_mode),
            "offload_mode": scenario.get("offload_mode", args.offload_mode),
            "mlp_chunk_tokens": scenario.get(
                "mlp_chunk_tokens", args.mlp_chunk_tokens or 8192
            ),
            "long_sequence_query_chunk_tokens": scenario.get(
                "long_sequence_query_chunk_tokens",
                args.long_sequence_query_chunk_tokens,
            ),
            "long_sequence_projection_chunk_tokens": scenario.get(
                "long_sequence_projection_chunk_tokens",
                args.long_sequence_projection_chunk_tokens,
            ),
            "long_sequence_split_qkv_outputs": bool(
                scenario.get(
                    "long_sequence_split_qkv_outputs",
                    args.long_sequence_split_qkv_outputs,
                )
            ),
            "long_sequence_shared_qkv_quantization": bool(
                scenario.get(
                    "long_sequence_shared_qkv_quantization",
                    args.long_sequence_shared_qkv_quantization,
                )
            ),
            "long_sequence_exact_helper_stack": bool(
                scenario.get(
                    "long_sequence_exact_helper_stack",
                    args.long_sequence_exact_helper_stack,
                )
            ),
            "long_sequence_single_qknorm_rope": bool(
                scenario.get(
                    "long_sequence_single_qknorm_rope",
                    args.long_sequence_single_qknorm_rope,
                )
            ),
            "long_sequence_parallel_sparse_lut": bool(
                scenario.get(
                    "long_sequence_parallel_sparse_lut",
                    args.long_sequence_parallel_sparse_lut,
                )
            ),
            "long_sequence_partial_sparse_topk": bool(
                scenario.get(
                    "long_sequence_partial_sparse_topk",
                    args.long_sequence_partial_sparse_topk,
                )
            ),
            "long_sequence_fused_prefix_k_quant": bool(
                scenario.get(
                    "long_sequence_fused_prefix_k_quant",
                    args.long_sequence_fused_prefix_k_quant,
                )
            ),
            "long_sequence_fused_query_projection": bool(
                scenario.get(
                    "long_sequence_fused_query_projection",
                    args.long_sequence_fused_query_projection,
                )
            ),
            "long_sequence_fused_qknorm_hnd_layout": bool(
                scenario.get(
                    "long_sequence_fused_qknorm_hnd_layout",
                    args.long_sequence_fused_qknorm_hnd_layout,
                )
            ),
            "long_sequence_direct_nhd_output": bool(
                scenario.get(
                    "long_sequence_direct_nhd_output",
                    args.long_sequence_direct_nhd_output,
                )
            ),
            "long_sequence_direct_nhd_kv": bool(
                scenario.get(
                    "long_sequence_direct_nhd_kv",
                    args.long_sequence_direct_nhd_kv,
                )
            ),
            "long_sequence_direct_hnd_fp8_value": bool(
                scenario.get(
                    "long_sequence_direct_hnd_fp8_value",
                    args.long_sequence_direct_hnd_fp8_value,
                )
            ),
            "prefetch_depth": scenario.get(
                "prefetch_depth", args.prefetch_depth
            ),
            "resident_block_count": scenario.get(
                "resident_block_count", args.resident_block_count
            ),
            "vae_tile_size": scenario.get(
                "vae_tile_size", args.vae_tile_size
            ),
            "vae_tile_batch_size": scenario.get(
                "vae_tile_batch_size", args.vae_tile_batch_size
            ),
            "vae_transformer_block_compile": bool(
                scenario.get(
                    "vae_transformer_block_compile",
                    args.vae_compile_transformer_block,
                )
            ),
            "attention_topk": scenario.get("attention_topk"),
            "fused_rms_adaln": bool(
                scenario.get("fused_rms_adaln", args.fused_rms_adaln)
            ),
            "dense_qk_quant_gran": scenario.get(
                "dense_qk_quant_gran", "per_thread"
            ),
            "actual_step_indices": scenario.get("actual_step_indices"),
            "cache_condition_rows": scenario.get("cache_condition_rows"),
            "cache_condition_embeddings": scenario.get(
                "cache_condition_embeddings"
            ),
            "cache_reference_latents": scenario.get("cache_reference_latents"),
            "first_frame": scenario.get("first_frame", args.first_frame),
            "last_frame": scenario.get("last_frame", args.last_frame),
            "reference_images": resolve_paths(scenario.get("reference_images", args.reference_image)),
            "reference_videos": resolve_paths(scenario.get("reference_videos", args.reference_video)),
            "reference_audios": resolve_paths(scenario.get("reference_audios", args.reference_audio)),
            "refinement_latents_path": scenario.get("refinement_latents_path"),
            "sampler_state_path": scenario.get(
                "sampler_state_path", args.sampler_state
            ),
            "checkpoint_after_step": scenario.get(
                "checkpoint_after_step", args.checkpoint_after_step
            ),
            "checkpoint_state_path": scenario.get(
                "checkpoint_state_path", args.checkpoint_state
            ),
            "refinement_denoise": scenario.get("refinement_denoise"),
            "refinement_spatial_mode": scenario.get(
                "refinement_spatial_mode", "strict"
            ),
            "preserve_refinement_audio": scenario.get(
                "preserve_refinement_audio", True
            ),
            "save_final_latents_path": scenario.get("save_final_latents_path"),
            "multiscale_initial_width": scenario.get("multiscale_initial_width"),
            "multiscale_initial_height": scenario.get("multiscale_initial_height"),
            "multiscale_resize_after_step": scenario.get(
                "multiscale_resize_after_step"
            ),
            "multiscale_highpass_strength": scenario.get(
                "multiscale_highpass_strength", 1.0
            ),
            "terminal_refinement_initial_width": scenario.get(
                "terminal_refinement_initial_width"
            ),
            "terminal_refinement_initial_height": scenario.get(
                "terminal_refinement_initial_height"
            ),
            "terminal_refinement_steps": scenario.get(
                "terminal_refinement_steps", 0
            ),
            "terminal_refinement_denoise": scenario.get(
                "terminal_refinement_denoise", 0.0125
            ),
            "terminal_refinement_dense_tail_steps": scenario.get(
                "terminal_refinement_dense_tail_steps", 1
            ),
            "terminal_refinement_low_frequency_gain": scenario.get(
                "terminal_refinement_low_frequency_gain", 1.0
            ),
            "terminal_refinement_temporal_lowpass": scenario.get(
                "terminal_refinement_temporal_lowpass", False
            ),
            "terminal_refinement_temporal_outlier_only": scenario.get(
                "terminal_refinement_temporal_outlier_only", False
            ),
        }
        for repeat_index in range(args.repeat):
            result.append(
                {
                    **base,
                    "name": (
                        safe_name
                        if args.repeat == 1
                        else f"{safe_name}_repeat{repeat_index + 1}"
                    ),
                    "repeat_index": repeat_index + 1,
                }
            )
    return result


def main() -> int:
    args = parse_args()
    if args.enforce_vram_gib is not None:
        if not math.isfinite(args.enforce_vram_gib) or args.enforce_vram_gib <= 0:
            raise ValueError("--enforce-vram-gib must be a positive finite number")
        physical_bytes = torch.cuda.get_device_properties(0).total_memory
        cap_bytes = int(args.enforce_vram_gib * 1024**3)
        if cap_bytes > physical_bytes:
            raise ValueError("--enforce-vram-gib cannot exceed physical CUDA memory")
        torch.cuda.set_per_process_memory_fraction(cap_bytes / physical_bytes, 0)
        os.environ["H3_NATIVE_MAX_VRAM_GIB"] = str(args.enforce_vram_gib)
    v19_blueprint_batch = (
        ()
        if args.v19_blueprint_manifest is None
        else load_v19_blueprint_batch(args.v19_blueprint_manifest)
    )
    v19_blueprint = (
        None
        if args.v19_blueprint is None
        else load_v19_candidate_blueprint(args.v19_blueprint)
    )
    measured_schedule_summary = None
    if args.reference_rms_steps:
        os.environ["H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_STEPS"] = (
            args.reference_rms_steps
        )
    if args.reference_rms_layers:
        os.environ["H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_LAYERS"] = (
            args.reference_rms_layers
        )
    if args.attention_backend == "sage-fused-k":
        attention_backend = sage_attention_sm89_fused_k_quant
    elif args.attention_backend == "sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        attention_backend = make_sparge_attention_sm89(
            args.sparge_topk,
            dense_step_indices=args.sparge_dense_steps,
        )
    elif args.attention_backend == "routed":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        attention_backend = make_routed_sparge_attention_sm89()
    elif args.attention_backend == "joint-scheduled":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        attention_backend = make_joint_action_scheduled_sparge_attention_sm89(
            route_probe=args.probe_sparse_route_stability
        )
    elif args.attention_backend == "long-mass-probe":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        reference_count = math.floor(
            args.long_mass_probe_topk
            * args.long_mass_probe_reference_key_blocks
        )
        absolute_cap = math.ceil(
            reference_count * args.long_mass_probe_cap_multiplier
        )
        physical = SplitModalityProtectedSpargeAttentionBackend(
            args.long_mass_probe_topk,
            experimental_minimum_topk=0.0625,
            temporal_correspondence_radius=1,
            temporal_spatial_block_radius=1,
            temporal_global_anchor_stride=8,
            temporal_global_spatial_block_radius=0,
            selection_mode="fixed_topk_mass_probe",
            maximum_selected_key_blocks=absolute_cap,
            minimum_retained_topk_mass=args.long_mass_probe_min_retained,
            mass_probe_selected_key_blocks=args.long_mass_probe_cap_ladder,
        )

        class SharedLongMassProbe:
            """Share side-effect-free probe telemetry across block buffers."""

            approximate = True

            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

            def resolve_long_sequence_backend(self, query_tokens: int):
                return physical.resolve_long_sequence_backend(query_tokens)

            def __call__(self, query, key, value):
                return physical(query, key, value)

            @staticmethod
            def telemetry():
                return physical.telemetry()

        attention_backend = SharedLongMassProbe()
    elif args.attention_backend == "modality-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        attention_backend = make_modality_protected_sparge_attention_sm89(
            args.sparge_topk
        )
    elif args.attention_backend == "split-modality-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        attention_backend = make_split_modality_protected_sparge_attention_sm89(
            args.sparge_topk
        )
    elif args.attention_backend == "split-layer-routed-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        try:
            sensitive_layers = tuple(
                int(value.strip())
                for value in args.layer_routed_sensitive_layers.split(",")
                if value.strip()
            )
        except ValueError as error:
            raise SystemExit(
                "--layer-routed-sensitive-layers must be comma-separated integers"
            ) from error
        attention_backend = make_layer_sensitivity_routed_split_sparge_attention_sm89(
            aggressive_topk=args.layer_routed_aggressive_topk,
            sensitive_layers=sensitive_layers,
            safe_topk=args.layer_routed_safe_topk,
        )
    elif args.attention_backend == "trajectory-layer-modality-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        try:
            sensitive_layers = tuple(
                int(value.strip())
                for value in args.layer_routed_sensitive_layers.split(",")
                if value.strip()
            )
        except ValueError as error:
            raise SystemExit(
                "--layer-routed-sensitive-layers must be comma-separated integers"
            ) from error
        causal_quality_endpoint = bool(
            args.causal_verifier_head_island
            and args.causal_verifier_effort is not None
            and math.isclose(
                float(args.causal_verifier_effort), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        )
        if causal_quality_endpoint:
            # The effort continuum has a real, reproducible quality endpoint:
            # effort=1 must be the Human-accepted Round143 computation, not
            # merely "more heads" inside the faster fixed-topk draft.  Round143
            # uses a complete first solver anchor, an exact causal layer island,
            # and interaction-aware sparse selection outside that island.
            # Derive the endpoint vectors from the same checked-in risk tiers
            # used by the faster draft so the endpoint stays one mechanism and
            # cannot drift into a separately hand-authored preset.
            if (
                args.layer_routed_safe_head_topks is None
                or args.trajectory_recovery_aggressive_head_topks is None
                or args.trajectory_recovery_safe_head_topks is None
            ):
                raise SystemExit(
                    "causal verifier effort=1 requires per-head safe and recovery budgets"
                )
            args.layer_routed_safe_head_topks = tuple(
                0.070 if value >= 0.10 else 0.065
                for value in args.layer_routed_safe_head_topks
            )
            args.trajectory_recovery_aggressive_head_topks = tuple(
                0.125 if value >= 0.30 else 0.100
                for value in args.trajectory_recovery_aggressive_head_topks
            )
            args.trajectory_recovery_safe_head_topks = tuple(
                0.180 if value >= 0.35 else 0.140
                for value in args.trajectory_recovery_safe_head_topks
            )
            args.trajectory_anchor_steps = ()
            args.trajectory_anchor_aggressive_head_topks = None
            args.trajectory_anchor_safe_head_topks = None
            args.sparge_dense_steps = (0,)
            args.sparge_dense_layers = sensitive_layers
            args.sparse_selection_mode = "interaction_hybrid"
        trajectory_backend = make_trajectory_layer_modality_routed_sparge_attention_sm89(
                aggressive_topk=(
                    args.layer_routed_aggressive_head_topks
                    or args.layer_routed_aggressive_topk
                ),
                sensitive_layers=sensitive_layers,
                dense_step_indices=args.sparge_dense_steps,
                safe_topk=(
                    args.layer_routed_safe_head_topks
                    or args.layer_routed_safe_topk
                ),
                anchor_step_indices=args.trajectory_anchor_steps,
                anchor_aggressive_topk=args.trajectory_anchor_aggressive_head_topks,
                anchor_safe_topk=args.trajectory_anchor_safe_head_topks,
                recovery_step_indices=args.trajectory_recovery_steps,
                recovery_aggressive_topk=(
                    args.trajectory_recovery_aggressive_head_topks
                ),
                recovery_safe_topk=args.trajectory_recovery_safe_head_topks,
                experimental_minimum_topk=args.experimental_minimum_sparse_topk,
                temporal_correspondence_radius=args.temporal_correspondence_radius,
                temporal_spatial_block_radius=args.temporal_spatial_block_radius,
                temporal_global_anchor_stride=args.temporal_global_anchor_stride,
                temporal_global_spatial_block_radius=(
                    args.temporal_global_spatial_block_radius
                ),
                selection_mode=args.sparse_selection_mode,
            )
        if args.teacher_layer_head_policy is not None:
            try:
                teacher_document = json.loads(
                    args.teacher_layer_head_policy.read_text(encoding="utf-8")
                )
                phase_document = teacher_document["phase_layer_head_topks"]

                def teacher_phase(name: str) -> dict[int, tuple[float, ...]]:
                    return {
                        int(layer): tuple(float(value) for value in budgets)
                        for layer, budgets in phase_document.get(name, {}).items()
                    }

                teacher_anchor_steps = tuple(
                    int(value) for value in teacher_document.get("anchor_steps", ())
                )
                teacher_recovery_steps = tuple(
                    int(value) for value in teacher_document.get("recovery_steps", ())
                )
                teacher_default_steps = tuple(
                    int(value) for value in teacher_document.get("default_steps", ())
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise SystemExit(
                    "invalid --teacher-layer-head-policy document"
                ) from error
            trajectory_backend = LayerHeadBudgetOverrideBackend(
                trajectory_backend,
                default_layer_topks=teacher_phase("default"),
                anchor_layer_topks=teacher_phase("anchor"),
                recovery_layer_topks=teacher_phase("recovery"),
                default_step_indices=teacher_default_steps,
                anchor_step_indices=teacher_anchor_steps,
                recovery_step_indices=teacher_recovery_steps,
                experimental_minimum_topk=args.experimental_minimum_sparse_topk,
                temporal_correspondence_radius=args.temporal_correspondence_radius,
                temporal_spatial_block_radius=args.temporal_spatial_block_radius,
                temporal_global_anchor_stride=args.temporal_global_anchor_stride,
                temporal_global_spatial_block_radius=(
                    args.temporal_global_spatial_block_radius
                ),
                selection_mode=args.sparse_selection_mode,
            )
        attention_backend = (
            StepScheduledAttentionBackend(
                sage_attention_sm89,
                trajectory_backend,
                dense_layer_indices=args.sparge_dense_layers,
                dense_step_layer_pairs=args.sparge_dense_step_layer_pairs,
                minimum_sparse_tokens=128,
            )
            if args.sparge_dense_layers or args.sparge_dense_step_layer_pairs
            else trajectory_backend
        )
        if args.causal_verifier_effort is not None and not causal_quality_endpoint:
            effort = float(args.causal_verifier_effort)
            # One public effort value controls a single nested mechanism.  More
            # effort samples more exact query blocks and rejects smaller draft
            # disagreements; model weights and the requested 12/8 schedule do
            # not change.
            attention_backend = CausalCheckpointVerifierAttentionBackend(
                sage_attention_sm89,
                trajectory_backend,
                recovery_backend=(
                    SplitModalityProtectedSpargeAttentionBackend(
                        0.50,
                        experimental_minimum_topk=(
                            args.experimental_minimum_sparse_topk
                        ),
                        temporal_correspondence_radius=(
                            args.temporal_correspondence_radius
                        ),
                        temporal_spatial_block_radius=(
                            args.temporal_spatial_block_radius
                        ),
                        temporal_global_anchor_stride=(
                            args.temporal_global_anchor_stride
                        ),
                        temporal_global_spatial_block_radius=(
                            args.temporal_global_spatial_block_radius
                        ),
                    )
                    if args.causal_verifier_graded_recovery
                    else None
                ),
                probe_layers=(30, 34, 39),
                recovery_layers=(
                    31,
                    32,
                    33,
                    35,
                    36,
                    37,
                    38,
                    40,
                    41,
                    42,
                    43,
                    45,
                ),
                hysteresis_layers=(
                    tuple(range(30, 40))
                    if args.causal_verifier_early_hysteresis
                    else None
                ),
                probe_first_short_circuit=args.causal_verifier_probe_first,
                shared_kv_exact_probe=args.causal_verifier_shared_kv_probe,
                causal_head_island=args.causal_verifier_head_island,
                inject_verified_queries=args.causal_verifier_inject_queries,
                repair_high_error_heads=args.causal_verifier_repair_heads,
                head_error_mass_coverage=0.40 + 0.40 * effort,
                verification_query_blocks=round(8 + 24 * effort),
                # The curved envelope preserves the calibrated midpoint
                # (effort=.25 -> ~.295) while continuously converging to a
                # zero-tolerance exact causal island at effort=1.  Thus the
                # same mechanism, rather than a preset switch, owns the full
                # speed/quality continuum and its quality endpoint is the
                # Human-accepted Round143 computation.
                relative_rms_threshold=0.34 * math.sqrt(1.0 - effort),
            )
    elif args.attention_backend == "quality-adaptive-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        attention_backend = make_quality_constrained_adaptive_sparge_attention_sm89()
    elif args.attention_backend == "budget-adaptive-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        attention_backend = make_budget_constrained_adaptive_sparge_attention_sm89(
            compute_budget=args.adaptive_attention_budget,
            safety_margin=args.adaptive_attention_safety,
        )
    elif args.attention_backend == "measured-budget-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        calibration = load_h3_sparse_action_calibration(
            args.measured_sparse_calibration
        )
        if calibration.engine != args.engine:
            raise SystemExit(
                "measured sparse calibration engine mismatch: "
                f"artifact={calibration.engine}, request={args.engine}"
            )
        measured_schedule, physical_schedule = solve_measured_h3_sparse_schedule(
            calibration,
            args.actual_steps,
            attention_budget_ms=args.measured_attention_budget_ms,
            exact_opening=not args.measured_relax_opening,
            exact_causal_island=not args.measured_relax_causal_island,
            terminal_minimum_topk=args.measured_terminal_minimum_topk,
        )
        action_backends = {"dense": sage_attention_sm89}
        for action in calibration.actions_by_band[
            next(iter(calibration.actions_by_band))
        ]:
            if action.topk is None:
                continue
            action_backends[action.name] = SplitModalityProtectedSpargeAttentionBackend(
                action.topk,
                experimental_minimum_topk=args.experimental_minimum_sparse_topk,
                temporal_correspondence_radius=args.temporal_correspondence_radius,
                temporal_spatial_block_radius=args.temporal_spatial_block_radius,
                temporal_global_anchor_stride=args.temporal_global_anchor_stride,
                temporal_global_spatial_block_radius=(
                    args.temporal_global_spatial_block_radius
                ),
                selection_mode=args.sparse_selection_mode,
            )
        attention_backend = ActionScheduledAttentionBackend(
            action_backends,
            physical_schedule,
            exact_action="dense",
            expected_sequence_tokens=calibration.sequence_tokens,
        )
        measured_schedule_summary = {
            "version": "measured_budget_sparse_v1",
            "calibration": str(calibration.source),
            "calibration_step": calibration.step_index,
            "calibration_sequence_tokens": calibration.sequence_tokens,
            # Kept for existing analysis readers.  It describes only the
            # tensor shape on which absolute kernel timings were measured; it
            # is not a request/input limit.
            "sequence_tokens": calibration.sequence_tokens,
            "runtime_shape_contract": "request_adaptive_fractional_actions",
            "latency_estimate_scope": "calibration_shape_only",
            "budget_limit_ms": measured_schedule.budget_limit_ms,
            "estimated_cost_ms": measured_schedule.estimated_cost_ms,
            "estimated_risk_debt_upper": (
                measured_schedule.estimated_reject_risk_ucb
            ),
            "exact_opening": not args.measured_relax_opening,
            "exact_causal_island": not args.measured_relax_causal_island,
            "terminal_minimum_topk": args.measured_terminal_minimum_topk,
            "choices": [
                {
                    "cell": choice.key.cell_id,
                    "actual_step": choice.key.actual_step,
                    "layer_start": choice.key.layer_start,
                    "layer_stop": choice.key.layer_stop,
                    "phase": choice.key.phase,
                    "action": choice.action.name,
                    "measured_cost_ms": choice.action.measured_cost_ms,
                    "risk_debt_upper": choice.action.reject_risk_ucb,
                }
                for choice in measured_schedule.choices
            ],
        }
    elif args.attention_backend == "trajectory-layer-modality-sol":
        attention_backend = make_trajectory_layer_modality_routed_sol_attention_sm89(
            source=args.sol_attn_source,
            tau=args.sol_tau,
            sensitive_tau=args.sol_sensitive_tau,
            sensitive_layers=args.sol_sensitive_layers,
            anchor_step_indices=args.sol_anchor_steps,
            anchor_tau=args.sol_anchor_tau if args.sol_anchor_steps else None,
            recovery_step_indices=args.sol_recovery_steps,
            recovery_tau=args.sol_recovery_tau if args.sol_recovery_steps else None,
        )
    elif args.attention_backend == "split-headwise-sparge":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        headwise_backend = make_split_modality_protected_sparge_attention_sm89(
            args.sparge_head_topks
        )
        attention_backend = (
            StepScheduledAttentionBackend(
                sage_attention_sm89,
                headwise_backend,
                dense_step_indices=args.sparge_dense_steps,
                dense_layer_indices=args.sparge_dense_layers,
                minimum_sparse_tokens=128,
            )
            if args.sparge_dense_steps or args.sparge_dense_layers
            else headwise_backend
        )
    elif args.attention_backend == "head-calibration":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))

        class OneShotHeadCalibration:
            """Measure sparse error on one true H3 long-attention call.

            The accepted dense output is always returned.  Sparse candidates
            are diagnostic side computations and can never affect the video.
            """

            approximate = False

            def __init__(self, topks: tuple[float, ...], output_path: Path) -> None:
                self.topks = topks
                self.output_path = output_path
                self.done = False

            @staticmethod
            def _elapsed(operation):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                result = operation()
                stop.record()
                stop.synchronize()
                return result, float(start.elapsed_time(stop))

            @classmethod
            def _elapsed_with_peak(cls, operation):
                torch.cuda.reset_peak_memory_stats()
                result, elapsed_ms = cls._elapsed(operation)
                peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                return result, elapsed_ms, float(peak_gib)

            @staticmethod
            def _head_metrics(candidate, reference, start_token: int):
                left = candidate[start_token:].float().permute(1, 0, 2).flatten(1)
                right = reference[start_token:].float().permute(1, 0, 2).flatten(1)
                cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
                delta = left - right
                relative_l1 = delta.abs().sum(1) / right.abs().sum(1).clamp_min(1e-12)
                rmse = delta.square().mean(1).sqrt()
                return {
                    "cosine": [float(value) for value in cosine.cpu()],
                    "relative_l1": [float(value) for value in relative_l1.cpu()],
                    "rmse": [float(value) for value in rmse.cpu()],
                }

            def __call__(self, query, key, value):
                dense, dense_ms = self._elapsed(
                    lambda: sage_attention_sm89(query, key, value)
                )
                if self.done or query.shape[0] < 10_000:
                    return dense
                from h3serve.native_engine.model import kernels as kernel_context

                protected = int(kernel_context._ATTENTION_PROTECTED_PREFIX.get())
                document = {
                    "contract": {
                        "engine": args.engine,
                        "sequence_tokens": int(query.shape[0]),
                        "protected_tokens": protected,
                        "heads": int(query.shape[1]),
                        "head_dim": int(query.shape[2]),
                        "dense_result_returned": True,
                        "weights_modified": False,
                    },
                    "dense_ms": dense_ms,
                    "candidates": [],
                }
                for topk in self.topks:
                    head_metrics = {
                        "cosine": [],
                        "relative_l1": [],
                        "rmse": [],
                    }
                    grouped_probe_ms = 0.0
                    for head_start in range(
                        0, query.shape[1], args.head_calibration_group_size
                    ):
                        head_stop = min(
                            query.shape[1],
                            head_start + args.head_calibration_group_size,
                        )
                        backend = make_split_modality_protected_sparge_attention_sm89(
                            topk
                        )
                        sparse, sparse_ms = self._elapsed(
                            lambda backend=backend, head_start=head_start, head_stop=head_stop: backend(
                                query[:, head_start:head_stop],
                                key[:, head_start:head_stop],
                                value[:, head_start:head_stop],
                            )
                        )
                        grouped_probe_ms += sparse_ms
                        metrics = self._head_metrics(
                            sparse,
                            dense[:, head_start:head_stop],
                            protected,
                        )
                        for name in head_metrics:
                            head_metrics[name].extend(metrics[name])
                        del sparse, backend
                    document["candidates"].append(
                        {
                            "topk": topk,
                            "grouped_probe_ms": grouped_probe_ms,
                            "timing_note": (
                                "sum of low-memory head groups; use the existing "
                                "56-head microbenchmark for production latency"
                            ),
                            "video_heads": head_metrics,
                        }
                    )
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.done = True
                if args.head_calibration_stop_after_probe:
                    raise RuntimeError(
                        "head calibration probe completed; generation intentionally stopped"
                    )
                return dense

        attention_backend = OneShotHeadCalibration(
            args.head_calibration_topks,
            args.head_calibration_output.resolve(),
        )
    elif args.attention_backend == "causal-calibration":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))
        piecewise_attention = None
        if args.piecewise_probe_density > 0.0:
            sys.path.insert(0, str(args.piecewise_source.resolve()))
            from piecewise_attn import (
                piecewise_sparse_attention_hyd as piecewise_attention,
            )
        try:
            sensitive_layers = tuple(
                int(value.strip())
                for value in args.layer_routed_sensitive_layers.split(",")
                if value.strip()
            )
        except ValueError as error:
            raise SystemExit(
                "--layer-routed-sensitive-layers must be comma-separated integers"
            ) from error

        sparse_probe_backend = make_trajectory_layer_modality_routed_sparge_attention_sm89(
            aggressive_topk=(
                args.layer_routed_aggressive_head_topks
                or args.layer_routed_aggressive_topk
            ),
            sensitive_layers=sensitive_layers,
            dense_step_indices=(),
            safe_topk=(
                args.layer_routed_safe_head_topks
                or args.layer_routed_safe_topk
            ),
            anchor_step_indices=args.trajectory_anchor_steps,
            anchor_aggressive_topk=args.trajectory_anchor_aggressive_head_topks,
            anchor_safe_topk=args.trajectory_anchor_safe_head_topks,
            recovery_step_indices=args.trajectory_recovery_steps,
            recovery_aggressive_topk=(
                args.trajectory_recovery_aggressive_head_topks
            ),
            recovery_safe_topk=args.trajectory_recovery_safe_head_topks,
            experimental_minimum_topk=args.experimental_minimum_sparse_topk,
            temporal_correspondence_radius=args.temporal_correspondence_radius,
            temporal_spatial_block_radius=args.temporal_spatial_block_radius,
            temporal_global_anchor_stride=args.temporal_global_anchor_stride,
            temporal_global_spatial_block_radius=(
                args.temporal_global_spatial_block_radius
            ),
            selection_mode=args.sparse_selection_mode,
        )

        class DenseReturningCausalCalibration:
            """Measure block-local Dense→Sparse errors on an undisturbed teacher.

            The generated trajectory always receives the dense result.  Sparse
            attention is evaluated only as a diagnostic side branch, so errors
            observed at later steps cannot be caused by an earlier approximate
            state.  Per-query-block labels are paired with prompt-independent
            Query motion features for cross-scene predictor research.
            """

            approximate = False

            def __init__(self, output_path: Path, sparse_backend) -> None:
                self.output_path = output_path
                self.sparse_backend = sparse_backend
                self.layers: dict[int, dict[int, dict]] = {
                    index: {} for index in args.layer_calibration_steps
                }

            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

            @staticmethod
            def _elapsed(operation):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                result = operation()
                stop.record()
                stop.synchronize()
                return result, float(start.elapsed_time(stop))

            @staticmethod
            def _rounded(tensor: torch.Tensor) -> list[float]:
                return [round(float(value), 7) for value in tensor.detach().cpu()]

            @classmethod
            def _block_metrics(
                cls,
                query: torch.Tensor,
                dense: torch.Tensor,
                sparse: torch.Tensor,
                *,
                protected_tokens: int,
            ) -> dict[str, list[float] | list[int]]:
                from h3serve.native_engine.model import kernels as kernel_context

                q = query[protected_tokens:].float()
                reference = dense[protected_tokens:].float()
                candidate = sparse[protected_tokens:].float()
                token_count, heads, head_dim = q.shape
                block_size = 128
                blocks = (token_count + block_size - 1) // block_size
                padded = blocks * block_size
                pad = padded - token_count
                if pad:
                    zeros = q.new_zeros((pad, heads, head_dim))
                    q = torch.cat((q, zeros), dim=0)
                    reference = torch.cat((reference, zeros), dim=0)
                    candidate = torch.cat((candidate, zeros), dim=0)
                q = q.view(blocks, block_size, heads, head_dim)
                reference = reference.view(blocks, block_size, heads, head_dim)
                candidate = candidate.view(blocks, block_size, heads, head_dim)

                difference = (candidate - reference).abs().sum(dim=(1, 3))
                magnitude = reference.abs().sum(dim=(1, 3)).clamp_min(1.0e-8)
                per_head_error = difference / magnitude
                flat_reference = reference.flatten(1)
                flat_candidate = candidate.flatten(1)
                cosine = torch.nn.functional.cosine_similarity(
                    flat_candidate,
                    flat_reference,
                    dim=1,
                    eps=1.0e-8,
                )

                valid = torch.full(
                    (blocks,), block_size, device=q.device, dtype=torch.float32
                )
                if pad:
                    valid[-1] = block_size - pad
                pooled = q.sum(dim=1) / valid[:, None, None]
                vectors = torch.nn.functional.normalize(
                    pooled, dim=-1, eps=1.0e-6
                )
                previous_indices = torch.arange(
                    blocks, device=q.device, dtype=torch.int64
                )
                next_indices = previous_indices.clone()
                layout = kernel_context._ATTENTION_VIDEO_LAYOUT.get()
                frame_indices = torch.zeros_like(previous_indices)
                if layout is not None:
                    latent_frames, frame_tokens = layout
                    centres = (
                        torch.arange(blocks, device=q.device, dtype=torch.int64)
                        * block_size
                        + block_size // 2
                    ).clamp_max(token_count - 1)
                    frame_indices = torch.div(
                        centres, frame_tokens, rounding_mode="floor"
                    )
                    spatial = torch.remainder(centres, frame_tokens)
                    previous_tokens = (
                        (frame_indices - 1).clamp_min(0) * frame_tokens + spatial
                    )
                    next_tokens = (
                        (frame_indices + 1).clamp_max(latent_frames - 1)
                        * frame_tokens
                        + spatial
                    )
                    previous_indices = torch.div(
                        previous_tokens, block_size, rounding_mode="floor"
                    ).clamp_max(blocks - 1)
                    next_indices = torch.div(
                        next_tokens, block_size, rounding_mode="floor"
                    ).clamp_max(blocks - 1)

                previous = vectors.index_select(0, previous_indices)
                following = vectors.index_select(0, next_indices)
                incoming = vectors - previous
                outgoing = following - vectors
                motion = torch.maximum(
                    incoming.square().sum(-1).sqrt(),
                    outgoing.square().sum(-1).sqrt(),
                ).mean(dim=1)
                turn = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        incoming, outgoing, dim=-1, eps=1.0e-6
                    )
                ).mul(0.5).clamp(0.0, 1.0).mean(dim=1)
                temporal_risk = motion * turn
                q_head_dispersion = pooled.norm(dim=-1).std(dim=1)

                return {
                    "frame_index": [int(value) for value in frame_indices.cpu()],
                    "relative_l1_mean": cls._rounded(per_head_error.mean(dim=1)),
                    "relative_l1_max": cls._rounded(per_head_error.max(dim=1).values),
                    "relative_l1_std": cls._rounded(per_head_error.std(dim=1)),
                    "cosine": cls._rounded(cosine),
                    "query_motion": cls._rounded(motion),
                    "query_turn": cls._rounded(turn),
                    "query_temporal_risk": cls._rounded(temporal_risk),
                    "query_head_dispersion": cls._rounded(q_head_dispersion),
                }

            @classmethod
            def _head_metrics(
                cls,
                query: torch.Tensor,
                key: torch.Tensor,
                dense: torch.Tensor,
                candidate: torch.Tensor,
                *,
                protected_tokens: int,
            ) -> dict[str, list[float]]:
                """Measure request-local head features against the dense teacher.

                The features intentionally avoid prompt labels or image-space
                detectors.  They describe temporal continuity and coarse
                attention concentration in the true H3 latent lattice, so a
                future router can protect relation-carrying heads without a
                scene-specific mask.
                """

                from h3serve.native_engine.model import kernels as kernel_context

                q = query[protected_tokens:]
                k = key[protected_tokens:]
                reference = dense[protected_tokens:].float()
                approximation = candidate[protected_tokens:].float()
                difference = (approximation - reference).abs().sum(dim=(0, 2))
                magnitude = reference.abs().sum(dim=(0, 2)).clamp_min(1.0e-8)
                relative_l1 = difference / magnitude
                cosine = torch.nn.functional.cosine_similarity(
                    approximation.permute(1, 0, 2).flatten(1),
                    reference.permute(1, 0, 2).flatten(1),
                    dim=1,
                    eps=1.0e-8,
                )

                heads = q.shape[1]
                zeros = torch.zeros(heads, device=q.device, dtype=torch.float32)
                q_motion_mean = zeros
                q_motion_p95 = zeros
                q_turn_mean = zeros
                q_turn_p95 = zeros
                k_motion_mean = zeros
                k_motion_p95 = zeros
                layout = kernel_context._ATTENTION_VIDEO_LAYOUT.get()
                if layout is not None:
                    latent_frames, frame_tokens = layout
                    video_tokens = latent_frames * frame_tokens

                    def temporal_features(tensor):
                        frames = tensor[:video_tokens].reshape(
                            latent_frames, frame_tokens, heads, tensor.shape[-1]
                        )
                        similarity = torch.nn.functional.cosine_similarity(
                            frames[1:].float(), frames[:-1].float(), dim=-1, eps=1.0e-6
                        )
                        motion = (1.0 - similarity).clamp_(0.0, 2.0)
                        motion_flat = motion.permute(2, 0, 1).flatten(1)
                        motion_mean = motion_flat.mean(dim=1)
                        motion_p95 = torch.quantile(motion_flat, 0.95, dim=1)
                        if latent_frames < 3:
                            return motion_mean, motion_p95, zeros, zeros
                        incoming = frames[1:-1].float() - frames[:-2].float()
                        outgoing = frames[2:].float() - frames[1:-1].float()
                        turn = (
                            1.0
                            - torch.nn.functional.cosine_similarity(
                                incoming, outgoing, dim=-1, eps=1.0e-6
                            )
                        ).mul_(0.5).clamp_(0.0, 1.0)
                        turn_flat = turn.permute(2, 0, 1).flatten(1)
                        return (
                            motion_mean,
                            motion_p95,
                            turn_flat.mean(dim=1),
                            torch.quantile(turn_flat, 0.95, dim=1),
                        )

                    (
                        q_motion_mean,
                        q_motion_p95,
                        q_turn_mean,
                        q_turn_p95,
                    ) = temporal_features(q)
                    k_motion_mean, k_motion_p95, _, _ = temporal_features(k)

                def pool_rows(tensor: torch.Tensor, block: int) -> torch.Tensor:
                    rows = tensor.shape[0]
                    blocks = (rows + block - 1) // block
                    padded = blocks * block
                    if padded != rows:
                        tensor = torch.cat(
                            (
                                tensor,
                                tensor.new_zeros(
                                    (padded - rows, tensor.shape[1], tensor.shape[2])
                                ),
                            ),
                            dim=0,
                        )
                    pooled = tensor.reshape(blocks, block, heads, tensor.shape[-1]).sum(1)
                    valid = torch.full(
                        (blocks,), block, device=tensor.device, dtype=torch.float32
                    )
                    if padded != rows:
                        valid[-1] = rows - (blocks - 1) * block
                    return pooled.float() / valid[:, None, None]

                pooled_q = pool_rows(q, 128)
                pooled_k = pool_rows(k, 64)
                scores = torch.einsum("qhd,khd->hqk", pooled_q, pooled_k)
                scores.mul_(query.shape[-1] ** -0.5)
                probabilities = scores.softmax(dim=-1)
                entropy = -(
                    probabilities * probabilities.clamp_min(1.0e-12).log()
                ).sum(-1)
                entropy.div_(math.log(max(2, pooled_k.shape[0])))
                top8_mass = probabilities.topk(
                    min(8, probabilities.shape[-1]), dim=-1
                ).values.sum(-1)

                return {
                    "relative_l1": cls._rounded(relative_l1),
                    "cosine": cls._rounded(cosine),
                    "q_motion_mean": cls._rounded(q_motion_mean),
                    "q_motion_p95": cls._rounded(q_motion_p95),
                    "q_turn_mean": cls._rounded(q_turn_mean),
                    "q_turn_p95": cls._rounded(q_turn_p95),
                    "k_motion_mean": cls._rounded(k_motion_mean),
                    "k_motion_p95": cls._rounded(k_motion_p95),
                    "proxy_entropy_mean": cls._rounded(entropy.mean(dim=1)),
                    "proxy_entropy_p95": cls._rounded(
                        torch.quantile(entropy, 0.95, dim=1)
                    ),
                    "proxy_top8_mass_mean": cls._rounded(top8_mass.mean(dim=1)),
                }

            def _persist(self, contract: dict) -> None:
                steps = [
                    {
                        "step_index": step_index,
                        "layers": [
                            step_layers[index]
                            for index in sorted(step_layers)
                        ],
                        "complete": len(step_layers) == 50,
                    }
                    for step_index, step_layers in sorted(self.layers.items())
                ]
                document = {
                    "contract": contract,
                    "steps": steps,
                    "complete": all(row["complete"] for row in steps),
                }
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.output_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(document, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                temporary.replace(self.output_path)

            def __call__(self, query, key, value):
                from h3serve.native_engine.model import kernels as kernel_context

                dense, dense_ms = self._elapsed(
                    lambda: sage_attention_sm89(query, key, value)
                )
                layer = kernel_context._ATTENTION_LAYER.get()
                step = kernel_context._ATTENTION_STEP.get()
                if (
                    query.shape[0] < 10_000
                    or layer is None
                    or step is None
                    or step[0] not in self.layers
                    or layer in self.layers[step[0]]
                ):
                    return dense
                sparse, sparse_ms = self._elapsed(
                    lambda: self.sparse_backend(query, key, value)
                )
                protected = int(kernel_context._ATTENTION_PROTECTED_PREFIX.get())
                metrics = self._block_metrics(
                    query,
                    dense,
                    sparse,
                    protected_tokens=protected,
                )
                head_metrics = self._head_metrics(
                    query,
                    key,
                    dense,
                    sparse,
                    protected_tokens=protected,
                )
                piecewise_ms = None
                piecewise_metrics = None
                if piecewise_attention is not None:
                    piecewise, piecewise_ms = self._elapsed(
                        lambda: piecewise_attention(
                            query.permute(1, 0, 2).unsqueeze(0).contiguous(),
                            key.permute(1, 0, 2).unsqueeze(0).contiguous(),
                            value.permute(1, 0, 2).unsqueeze(0).contiguous(),
                            density=args.piecewise_probe_density,
                            block_size=64,
                        )
                    )
                    piecewise = piecewise.squeeze(0).permute(1, 0, 2).contiguous()
                    piecewise_metrics = self._block_metrics(
                        query,
                        dense,
                        piecewise,
                        protected_tokens=protected,
                    )
                    del piecewise
                step_index = int(step[0])
                self.layers[step_index][int(layer)] = {
                    "layer": int(layer),
                    "dense_ms": dense_ms,
                    "sparse_probe_ms": sparse_ms,
                    "query_blocks": len(metrics["cosine"]),
                    "block_metrics": metrics,
                    "head_metrics": head_metrics,
                    "piecewise_probe_ms": piecewise_ms,
                    "piecewise_block_metrics": piecewise_metrics,
                }
                del sparse
                complete = all(
                    len(step_layers) == 50
                    for step_layers in self.layers.values()
                )
                if len(self.layers[step_index]) % 10 == 0 or complete:
                    self._persist(
                        {
                            "engine": args.engine,
                            "step_indices": list(args.layer_calibration_steps),
                            "sequence_tokens": int(query.shape[0]),
                            "protected_tokens": protected,
                            "heads": int(query.shape[1]),
                            "head_dim": int(query.shape[2]),
                            "dense_result_returned": True,
                            "sparse_selection_mode": args.sparse_selection_mode,
                            "piecewise_probe_density": args.piecewise_probe_density,
                            "weights_modified": False,
                        }
                    )
                if complete and args.layer_calibration_stop_after_complete:
                    raise RuntimeError(
                        "causal calibration completed; generation intentionally stopped"
                    )
                return dense

        attention_backend = DenseReturningCausalCalibration(
            args.layer_calibration_output.resolve(),
            sparse_probe_backend,
        )
    elif args.attention_backend == "layer-calibration":
        sys.path.insert(0, str(args.sparge_build_dir.resolve()))

        class DenseReturningLayerCalibration:
            """Probe true H3 layer/step sensitivity on one dense trajectory."""

            approximate = False

            def __init__(self, output_path: Path) -> None:
                self.output_path = output_path
                self.layers: dict[int, dict[int, dict]] = {
                    index: {} for index in args.layer_calibration_steps
                }
                physical_actions = make_joint_physical_action_backends_sm89()
                action_prefix = {
                    "round215": "round215",
                    "round188": "frontier",
                    "round228": "fastfrontier",
                    "round229": "forecastfrontier",
                }[args.layer_calibration_action_implementation]
                self.full_head_backends = tuple(
                    (
                        topk,
                        physical_actions[
                            f"{action_prefix}:sparse_topk_{topk:g}"
                        ],
                    )
                    for topk in args.layer_calibration_full_head_topks
                )

            def __deepcopy__(self, memo):
                # DoubleBufferBlockExecutor deep-copies the block graph into
                # two CUDA slots.  The diagnostic state must remain shared;
                # otherwise each slot observes alternating layers and the two
                # partial JSON documents overwrite each other forever.
                memo[id(self)] = self
                return self

            @staticmethod
            def _elapsed(operation):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                result = operation()
                stop.record()
                stop.synchronize()
                return result, float(start.elapsed_time(stop))

            @classmethod
            def _elapsed_with_peak(cls, operation):
                torch.cuda.reset_peak_memory_stats()
                result, elapsed_ms = cls._elapsed(operation)
                peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                return result, elapsed_ms, float(peak_gib)

            def _persist(self, contract: dict) -> None:
                steps = [
                    {
                        "step_index": step_index,
                        "layers": [
                            step_layers[index]
                            for index in sorted(step_layers)
                        ],
                        "complete": len(step_layers) == 50,
                    }
                    for step_index, step_layers in sorted(self.layers.items())
                ]
                document = {
                    "contract": contract,
                    "steps": steps,
                    "complete": all(row["complete"] for row in steps),
                }
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            @staticmethod
            def _full_head_metrics(
                candidate: torch.Tensor,
                reference: torch.Tensor,
                *,
                protected_tokens: int,
                head_group_size: int,
            ) -> dict[str, float | list[float]]:
                """Compare a production-shaped sparse action without OOMing.

                The complete 56-head sparse call is timed before this method.
                Metrics are reduced in head chunks only to avoid materialising
                two full float32 copies of a 720p15 Attention tensor.
                """

                head_cosine: list[float] = []
                head_relative_l1: list[float] = []
                head_relative_rms: list[float] = []
                total_delta_square = 0.0
                total_reference_square = 0.0
                for head_start in range(0, reference.shape[1], head_group_size):
                    head_stop = min(
                        reference.shape[1], head_start + head_group_size
                    )
                    left = candidate[
                        protected_tokens:, head_start:head_stop
                    ].float().permute(1, 0, 2).flatten(1)
                    right = reference[
                        protected_tokens:, head_start:head_stop
                    ].float().permute(1, 0, 2).flatten(1)
                    delta = left - right
                    cosine = torch.nn.functional.cosine_similarity(
                        left, right, dim=1, eps=1.0e-8
                    )
                    relative_l1 = (
                        delta.abs().sum(1)
                        / right.abs().sum(1).clamp_min(1.0e-12)
                    )
                    delta_square = delta.square().sum(1)
                    reference_square = right.square().sum(1).clamp_min(1.0e-12)
                    relative_rms = (delta_square / reference_square).sqrt()
                    head_cosine.extend(float(value) for value in cosine.cpu())
                    head_relative_l1.extend(
                        float(value) for value in relative_l1.cpu()
                    )
                    head_relative_rms.extend(
                        float(value) for value in relative_rms.cpu()
                    )
                    total_delta_square += float(delta_square.sum().cpu())
                    total_reference_square += float(reference_square.sum().cpu())
                    del left, right, delta
                return {
                    "mean_cosine": sum(head_cosine) / len(head_cosine),
                    "min_cosine": min(head_cosine),
                    "mean_relative_l1": (
                        sum(head_relative_l1) / len(head_relative_l1)
                    ),
                    "max_relative_l1": max(head_relative_l1),
                    "global_relative_rms": math.sqrt(
                        total_delta_square / max(total_reference_square, 1.0e-12)
                    ),
                    "mean_head_relative_rms": (
                        sum(head_relative_rms) / len(head_relative_rms)
                    ),
                    "head_cosine": head_cosine,
                    "head_relative_l1": head_relative_l1,
                    "head_relative_rms": head_relative_rms,
                }

            def __call__(self, query, key, value):
                from h3serve.native_engine.model import kernels as kernel_context

                dense, dense_initialization_ms, dense_initialization_peak_gib = (
                    self._elapsed_with_peak(
                    lambda: sage_attention_sm89(query, key, value)
                    )
                )
                layer = kernel_context._ATTENTION_LAYER.get()
                step = kernel_context._ATTENTION_STEP.get()
                if (
                    query.shape[0] < 10_000
                    or layer is None
                    or step is None
                    or step[0] not in self.layers
                    or layer in self.layers[step[0]]
                ):
                    return dense
                protected = int(kernel_context._ATTENTION_PROTECTED_PREFIX.get())
                if self.full_head_backends:
                    dense_warm_ms = []
                    dense_warm_peak_gib = []
                    for _ in range(args.layer_calibration_warm_repeats):
                        warmed_dense, warm_ms, warm_peak_gib = self._elapsed_with_peak(
                            lambda: sage_attention_sm89(query, key, value)
                        )
                        dense_warm_ms.append(warm_ms)
                        dense_warm_peak_gib.append(warm_peak_gib)
                        del warmed_dense
                    candidates = []
                    for topk, backend in self.full_head_backends:
                        sparse, sparse_initialization_ms, sparse_initialization_peak_gib = (
                            self._elapsed_with_peak(
                            lambda backend=backend: backend(query, key, value)
                            )
                        )
                        sparse_warm_ms = []
                        sparse_warm_peak_gib = []
                        for _ in range(args.layer_calibration_warm_repeats):
                            warmed_sparse, warm_ms, warm_peak_gib = self._elapsed_with_peak(
                                lambda backend=backend: backend(query, key, value)
                            )
                            sparse_warm_ms.append(warm_ms)
                            sparse_warm_peak_gib.append(warm_peak_gib)
                            del warmed_sparse
                        candidates.append(
                            {
                                "name": f"sparse_topk_{topk:g}",
                                "topk": topk,
                                "full_head_ms": (
                                    sorted(sparse_warm_ms)[len(sparse_warm_ms) // 2]
                                    if sparse_warm_ms
                                    else sparse_initialization_ms
                                ),
                                "full_head_initialization_ms": (
                                    sparse_initialization_ms
                                ),
                                "full_head_initialization_peak_gib": (
                                    sparse_initialization_peak_gib
                                ),
                                "full_head_warm_ms": sparse_warm_ms,
                                "full_head_warm_peak_gib": sparse_warm_peak_gib,
                                **self._full_head_metrics(
                                    sparse,
                                    dense,
                                    protected_tokens=protected,
                                    head_group_size=(
                                        args.head_calibration_group_size
                                    ),
                                ),
                            }
                        )
                        del sparse
                    step_index = int(step[0])
                    self.layers[step_index][int(layer)] = {
                        "layer": int(layer),
                        "dense_ms": (
                            sorted(dense_warm_ms)[len(dense_warm_ms) // 2]
                            if dense_warm_ms
                            else dense_initialization_ms
                        ),
                        "dense_initialization_ms": dense_initialization_ms,
                        "dense_initialization_peak_gib": (
                            dense_initialization_peak_gib
                        ),
                        "dense_warm_ms": dense_warm_ms,
                        "dense_warm_peak_gib": dense_warm_peak_gib,
                        "candidates": candidates,
                    }
                    self._persist(
                        {
                            "engine": args.engine,
                            "step_indices": list(args.layer_calibration_steps),
                            "sequence_tokens": int(query.shape[0]),
                            "protected_tokens": protected,
                            "heads": int(query.shape[1]),
                            "head_dim": int(query.shape[2]),
                            "full_head_budgets": list(
                                args.layer_calibration_full_head_topks
                            ),
                            "temporal_correspondence_radius": (
                                args.temporal_correspondence_radius
                            ),
                            "temporal_spatial_block_radius": (
                                args.temporal_spatial_block_radius
                            ),
                            "temporal_global_anchor_stride": (
                                args.temporal_global_anchor_stride
                            ),
                            "temporal_global_spatial_block_radius": (
                                args.temporal_global_spatial_block_radius
                            ),
                            "sparse_selection_mode": args.sparse_selection_mode,
                            "physical_action_implementation": {
                                "round215": ROUND215_ACTION_IMPLEMENTATION,
                                "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
                                "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
                                "round229": (
                                    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION
                                ),
                            }[args.layer_calibration_action_implementation],
                            "timing_scope": "one complete 56-head call",
                            "warm_repeats": args.layer_calibration_warm_repeats,
                            "dense_result_returned": True,
                            "weights_modified": False,
                        }
                    )
                    if (
                        all(
                            len(step_layers) == 50
                            for step_layers in self.layers.values()
                        )
                        and args.layer_calibration_stop_after_complete
                    ):
                        raise RuntimeError(
                            "layer calibration completed; generation intentionally stopped"
                        )
                    return dense
                relative_l1: list[float] = []
                cosine: list[float] = []
                grouped_ms = 0.0
                for head_start in range(
                    0, query.shape[1], args.head_calibration_group_size
                ):
                    head_stop = min(
                        query.shape[1],
                        head_start + args.head_calibration_group_size,
                    )
                    budgets = args.sparge_head_topks[
                        head_start:head_stop
                    ] if args.sparge_head_topks else tuple(
                        args.sparge_topk for _ in range(head_stop - head_start)
                    )
                    backend = make_split_modality_protected_sparge_attention_sm89(
                        budgets,
                        experimental_minimum_topk=(
                            args.experimental_minimum_sparse_topk
                        ),
                    )
                    sparse, sparse_ms = self._elapsed(
                        lambda backend=backend, head_start=head_start, head_stop=head_stop: backend(
                            query[:, head_start:head_stop],
                            key[:, head_start:head_stop],
                            value[:, head_start:head_stop],
                        )
                    )
                    grouped_ms += sparse_ms
                    left = sparse[protected:].float().permute(1, 0, 2).flatten(1)
                    right = (
                        dense[protected:, head_start:head_stop]
                        .float()
                        .permute(1, 0, 2)
                        .flatten(1)
                    )
                    cosine.extend(
                        float(item)
                        for item in torch.nn.functional.cosine_similarity(
                            left, right, dim=1
                        ).cpu()
                    )
                    relative_l1.extend(
                        float(item)
                        for item in (
                            (left - right).abs().sum(1)
                            / right.abs().sum(1).clamp_min(1e-12)
                        ).cpu()
                    )
                    del sparse, backend, left, right
                step_index = int(step[0])
                self.layers[step_index][int(layer)] = {
                    "layer": int(layer),
                    "dense_ms": dense_ms,
                    "grouped_probe_ms": grouped_ms,
                    "mean_cosine": sum(cosine) / len(cosine),
                    "min_cosine": min(cosine),
                    "mean_relative_l1": sum(relative_l1) / len(relative_l1),
                    "max_relative_l1": max(relative_l1),
                    "head_cosine": cosine,
                    "head_relative_l1": relative_l1,
                }
                self._persist(
                    {
                        "engine": args.engine,
                        "step_indices": list(args.layer_calibration_steps),
                        "sequence_tokens": int(query.shape[0]),
                        "protected_tokens": protected,
                        "heads": int(query.shape[1]),
                        "head_dim": int(query.shape[2]),
                        "head_budgets": (
                            list(args.sparge_head_topks)
                            if args.sparge_head_topks
                            else [args.sparge_topk] * int(query.shape[1])
                        ),
                        "dense_result_returned": True,
                        "weights_modified": False,
                    }
                )
                if (
                    all(len(step_layers) == 50 for step_layers in self.layers.values())
                    and args.layer_calibration_stop_after_complete
                ):
                    raise RuntimeError(
                        "layer calibration completed; generation intentionally stopped"
                    )
                return dense

        attention_backend = DenseReturningLayerCalibration(
            args.layer_calibration_output.resolve()
        )
    else:
        attention_backend = sage_attention_sm89
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this benchmark requires one SM89 GPU")
    # Pin the requested implementation explicitly; never let a third-party
    # package update silently change benchmark math or timing.
    kernel_runtime = configure_sm89_runtime(
        quant_backend=args.quant_backend, smoke_test=True
    )
    serve_root = Path(__file__).resolve().parents[1]
    calibration_runtime_fingerprint = capture_v19_runtime_fingerprint(
        serve_root=serve_root,
        sparge_build_dir=args.sparge_build_dir,
        kernel_runtime=kernel_runtime,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    from h3serve.memory_policy import HOST_MEMORY_PROFILES
    memory_profile = HOST_MEMORY_PROFILES[args.memory_profile]
    from h3serve.native_engine.local_checkpoint_cache import (
        materialize_local_checkpoint,
        materialize_qwen_layer_cache,
    )
    base = args.model_root / "diffusion_models" / (
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        if args.engine in ("reference", "reference-lora")
        else "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    )
    text_source = args.model_root / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    text_checkpoint = (
        materialize_local_checkpoint(text_source)
        if memory_profile.key == "compact" else text_source
    )
    qwen_layer_cache = (
        materialize_qwen_layer_cache(text_checkpoint)
        if memory_profile.key == "compact" else None
    )
    video_checkpoint = args.model_root / "vae/minimax_h3_video_vae_fp16.safetensors"
    audio_checkpoint = args.model_root / "vae/minimax_h3_audio_vae_fp32.safetensors"
    lora_checkpoint = args.model_root / "loras/minimax_h3_turbo_v4_step600_ema.safetensors"
    egrid_checkpoint = args.model_root.parent / "backends/turbo/custom_node/h3_silu_temb_grid.safetensors"

    conditioner = PackedQwen3VLT2AVConditioner(
        text_checkpoint,
        args.minimax_source / "tokenizer",
        cache_pinned_weights=memory_profile.cache_qwen_weights,
        layer_cache_dir=qwen_layer_cache,
    )
    startup_started = time.perf_counter()

    def prepare_dit():
        updates = None
        curve = None
        if (
            args.engine in ("lora", "reference-lora")
            or (
                args.preview_decode_mode == "fast_finish"
                and (
                    args.preview_branch_use_lora
                    or args.preview_audio_branch_use_lora
                )
            )
        ):
            updates = load_larry_updates_from_safetensors(
                str(lora_checkpoint),
                strength=1.0,
                device="cpu",
                dtype=torch.bfloat16,
            )
            curve = load_full_silu_curve(str(egrid_checkpoint))
        with SafeTensorSource(str(base)) as source:
            model = assemble_full_pruned_dit(
                source,
                device="cpu",
                compute_dtype=torch.bfloat16,
                int8_kernel=comfy_kitchen_int8_kernel,
                attention_backend=attention_backend,
                lora_updates=updates,
                full_silu_curve=curve,
            )
        model.eval().requires_grad_(False)
        residency = ImmutablePinnedModuleResidency(
            "transformer", model,
            pin_host_weights=(
                memory_profile.pin_model_weights
                or memory_profile.pin_transformer_weights
            ),
            copy_host_weights=(
                memory_profile.copy_model_weights
                or memory_profile.copy_transformer_weights
            ),
        )
        residency.prepare_host()
        return residency

    def prepare_video():
        model, mean, std = load_video_vae(
            args.minimax_source,
            video_checkpoint,
            device="cpu",
            tile_size=args.vae_tile_size,
            tile_batch_size=args.vae_tile_batch_size,
            compile_feed_forward=args.vae_compile_feed_forward,
        )
        residency = ImmutablePinnedModuleResidency(
            "video_vae", model,
            pin_host_weights=memory_profile.pin_model_weights,
            copy_host_weights=memory_profile.copy_model_weights,
        )
        residency.prepare_host()
        return residency, mean, std

    def prepare_audio():
        model = load_audio_vae(
            args.lightx_source,
            args.minimax_source,
            audio_checkpoint,
            device="cpu",
        )
        residency = ImmutablePinnedModuleResidency(
            "audio_vae", model,
            pin_host_weights=memory_profile.pin_model_weights,
            copy_host_weights=memory_profile.copy_model_weights,
        )
        residency.prepare_host()
        return residency

    task_seconds: dict[str, float] = {}

    def timed(name, function):
        started = time.perf_counter()
        result = function()
        task_seconds[name] = time.perf_counter() - started
        print(f"startup {name}: {task_seconds[name]:.3f}s", flush=True)
        return result

    if memory_profile.parallel_model_build:
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="h3-startup") as pool:
            futures = {
                pool.submit(timed, "dit_cache", prepare_dit): "dit",
                pool.submit(timed, "video_vae_cache", prepare_video): "video",
                pool.submit(timed, "audio_vae_cache", prepare_audio): "audio",
            }
            if memory_profile.cache_qwen_weights:
                futures[pool.submit(timed, "qwen_cache", conditioner.prepare_host_cache)] = "qwen"
            prepared = {}
            for future in as_completed(futures):
                prepared[futures[future]] = future.result()
    else:
        prepared = {
            "dit": timed("dit_cache", prepare_dit),
            "video": timed("video_vae_cache", prepare_video),
            "audio": timed("audio_vae_cache", prepare_audio),
        }
        if memory_profile.cache_qwen_weights:
            timed("qwen_cache", conditioner.prepare_host_cache)

    transformer = prepared["dit"]
    video_vae, video_mean, video_std = prepared["video"]
    audio_vae = prepared["audio"]
    if args.vae_compile_feed_forward:
        def prewarm_video_vae():
            video_vae.move_to("cuda:0", non_blocking=False)
            try:
                prewarm_feed_forward_compile(video_vae.value)
                torch.cuda.synchronize()
            finally:
                video_vae.move_to("cpu", non_blocking=False)
                torch.cuda.empty_cache()

        timed("video_vae_compile_warmup", prewarm_video_vae)
    if args.vae_compile_transformer_block:
        def prewarm_video_vae_block_compile():
            video_vae.move_to("cuda:0", non_blocking=False)
            try:
                enable_transformer_block_compile(video_vae.value)
                prewarm_transformer_block_compile(video_vae.value)
                torch.cuda.synchronize()
            finally:
                video_vae.move_to("cpu", non_blocking=False)
                torch.cuda.empty_cache()

        timed("video_vae_block_compile_warmup", prewarm_video_vae_block_compile)
    startup_seconds = time.perf_counter() - startup_started
    print(f"startup critical path: {startup_seconds:.3f}s", flush=True)

    def video_decoder(model, latents, frame_count):
        return decode_video(
            model,
            latents,
            video_mean,
            video_std,
            frame_count,
            output_dtype="uint8",
        )

    video_condition_adapter = H3VideoVAEAdapter(
        video_vae.value,
        latents_mean=video_mean.tolist(),
        latents_std=video_std.tolist(),
    )

    def video_condition_encoder(_model, request):
        if request.reference_images or request.reference_videos:
            return video_condition_adapter.encode_references(request)
        return video_condition_adapter.encode_conditioning(request)

    from h3serve.native_engine.adapters.conditioning_vae import H3AudioVAEAdapter, prepare_reference_audios
    audio_condition_adapter = H3AudioVAEAdapter(
        audio_vae.value,
        latents_mean=audio_vae.value.latents_mean.tolist(),
        latents_std=audio_vae.value.latents_std.tolist(),
    )

    def audio_condition_encoder(_model, request):
        return tuple(
            audio_condition_adapter.encode(item.waveform.to("cuda:0"))
            for item in prepare_reference_audios(request)
        )

    def audio_decoder(model, latents):
        flattened = latents.permute(0, 2, 1, 3).reshape(2, 32, latents.shape[-1])
        with torch.inference_mode():
            return model.decode(flattened, stereo_batch=True, return_cpu=True)

    session = NativeT2AVHotSession(
        engine=("reference_lora" if args.engine == "reference-lora" else args.engine),
        conditioner=conditioner,
        transformer=transformer,
        video_vae=video_vae,
        audio_vae=audio_vae,
        decode_video=video_decoder,
        decode_audio=audio_decoder,
        encode_video_conditioning=video_condition_encoder,
        encode_audio_conditioning=audio_condition_encoder,
        output_root=args.output_root,
        debug_step_dir=args.debug_step_dir,
        debug_final_latents_path=args.debug_final_latents,
        runtime_config=replace(
            RuntimeConfig.for_cuda_device(),
            offload_mode=OffloadMode(args.offload_mode),
        ),
        planner=(
            RTX4090Planner(
                validated_lora_profiles_2026_08_11()
                if args.engine in ("lora", "reference-lora")
                else validated_original_profiles_2026_08_11()
            )
            if args.auto_route
            else None
        ),
    )
    session.self_speculative_verify_steps = args.self_speculative_verify_steps
    session.self_speculative_verify_threshold = (
        float("inf")
        if args.self_speculative_verify_threshold is None
        else args.self_speculative_verify_threshold
    )
    loaded_mechanistic_admission = None
    if args.v24_acceleration is not None:
        if args.mechanistic_admission is None:
            session.v19_selector = (
                V24FinalParetoRuntimeSelector()
                if args.v24_research_calibration is None
                else V24ResearchParetoRuntimeSelector(
                    candidate_id=args.v24_research_calibration
                )
            )
        else:
            loaded_mechanistic_admission = (
                load_h3_mechanistic_deployment_config(
                    args.mechanistic_admission
                )
            )
            session.v19_selector = H3MechanisticParetoRuntimeSelector(
                admission=loaded_mechanistic_admission.admission,
                calibrated_video_token_interval=(
                    loaded_mechanistic_admission
                    .calibrated_video_token_interval
                ),
                maximum_runtime_promotions=(
                    loaded_mechanistic_admission.maximum_runtime_promotions
                ),
            )
    forecast_factory = None
    if args.forecast_controller == "quality-curvature":
        forecast_factory = QualityConstrainedForecastFactory(step_count=args.steps)
        session.forecast_controller_factory = forecast_factory
    scenarios = load_scenarios(args)
    if v19_blueprint_batch:
        final_latents_root = (
            None
            if args.v19_batch_final_latents_dir is None
            else args.v19_batch_final_latents_dir.resolve()
        )
        scenarios = [
            {
                **scenario,
                "name": f"{batch_name}_{scenario['name']}",
                "_v19_blueprint": batch_blueprint,
                "_v19_blueprint_path": str(batch_path),
                "save_final_latents_path": (
                    scenario.get("save_final_latents_path")
                    if final_latents_root is None
                    else str(
                        final_latents_root
                        / f"{batch_name}_{scenario['name']}.pt"
                    )
                ),
            }
            for batch_name, batch_path, batch_blueprint in v19_blueprint_batch
            for scenario in scenarios
        ]
    label_prefix = args.label_prefix or f"native_{args.engine}"
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", label_prefix).strip("._")
    if not safe_prefix:
        raise ValueError("label prefix has no safe filename characters")
    report = {
        "engine": args.engine,
        "command": shlex.join(sys.argv),
        "kernel_runtime": kernel_runtime.to_dict(),
        "v19_runtime_fingerprint": asdict(calibration_runtime_fingerprint),
        "contract": {
            "quant_backend": args.quant_backend,
            "memory_profile": args.memory_profile,
            "attention_backend": args.attention_backend,
            "forecast_controller": args.forecast_controller,
            "joint_policy": args.joint_policy,
            "joint_acceleration": args.joint_acceleration,
            "pareto_acceleration": args.v24_acceleration,
            "v24_calibration": (
                None
                if args.v24_acceleration is None
                or args.mechanistic_admission is not None
                else session.v19_selector.candidate.candidate_id
            ),
            "mechanistic_admission": (
                None
                if loaded_mechanistic_admission is None
                else {
                    "source": str(loaded_mechanistic_admission.source),
                    "sha256": loaded_mechanistic_admission.source_sha256,
                    "status": loaded_mechanistic_admission.status,
                    "calibration_id": (
                        loaded_mechanistic_admission.admission.calibration_id
                    ),
                }
            ),
            "attention_policy": (
                "quality_constrained_adaptive_tlhb_tcr"
                if args.attention_backend == "quality-adaptive-sparge"
                else (
                    "budget_constrained_task_adaptive_sparse_attention_v1"
                    if args.attention_backend == "budget-adaptive-sparge"
                    else (
                        "measured_budget_sparse_v1"
                        if args.attention_backend == "measured-budget-sparge"
                        else None
                    )
                )
            ),
            "measured_sparse_schedule": measured_schedule_summary,
            "adaptive_attention_budget": (
                args.adaptive_attention_budget
                if args.attention_backend == "budget-adaptive-sparge"
                else None
            ),
            "adaptive_attention_safety": (
                args.adaptive_attention_safety
                if args.attention_backend == "budget-adaptive-sparge"
                else None
            ),
            "sparge_topk": (
                args.sparge_topk if args.attention_backend == "sparge" else None
            ),
            "sparse_selection_mode": args.sparse_selection_mode,
            "physical_action_implementation": (
                {
                    "round215": ROUND215_ACTION_IMPLEMENTATION,
                    "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
                    "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
                    "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
                }[args.layer_calibration_action_implementation]
                if args.attention_backend == "layer-calibration"
                else None
            ),
            "sparge_head_topks": (
                list(args.sparge_head_topks)
                if args.attention_backend == "split-headwise-sparge"
                else None
            ),
            "sparge_dense_steps": (
                list(args.sparge_dense_steps)
                if args.attention_backend in (
                    "sparge",
                    "split-headwise-sparge",
                    "trajectory-layer-modality-sparge",
                )
                else None
            ),
            "sparge_dense_layers": (
                list(args.sparge_dense_layers)
                if args.attention_backend in ("sparge", "split-headwise-sparge")
                else None
            ),
            "causal_verifier_effort": args.causal_verifier_effort,
            "causal_verifier_inject_queries": args.causal_verifier_inject_queries,
            "causal_verifier_repair_heads": args.causal_verifier_repair_heads,
            "causal_verifier_graded_recovery": (
                args.causal_verifier_graded_recovery
            ),
            "causal_verifier_early_hysteresis": (
                args.causal_verifier_early_hysteresis
            ),
            "causal_verifier_probe_first": args.causal_verifier_probe_first,
            "causal_verifier_shared_kv_probe": (
                args.causal_verifier_shared_kv_probe
            ),
            "causal_verifier_head_island": args.causal_verifier_head_island,
            "self_speculative_verify_steps": args.self_speculative_verify_steps,
            "self_speculative_verify_threshold": (
                args.self_speculative_verify_threshold
            ),
            "layer_routed_aggressive_topk": (
                args.layer_routed_aggressive_topk
                if args.attention_backend in (
                    "split-layer-routed-sparge",
                    "trajectory-layer-modality-sparge",
                )
                else None
            ),
            "layer_routed_safe_topk": (
                args.layer_routed_safe_topk
                if args.attention_backend in (
                    "split-layer-routed-sparge",
                    "trajectory-layer-modality-sparge",
                )
                else None
            ),
            "layer_routed_aggressive_head_topks": (
                list(args.layer_routed_aggressive_head_topks)
                if args.layer_routed_aggressive_head_topks
                else None
            ),
            "layer_routed_safe_head_topks": (
                list(args.layer_routed_safe_head_topks)
                if args.layer_routed_safe_head_topks
                else None
            ),
            "trajectory_anchor_steps": list(args.trajectory_anchor_steps),
            "trajectory_anchor_aggressive_head_topks": (
                list(args.trajectory_anchor_aggressive_head_topks)
                if args.trajectory_anchor_aggressive_head_topks
                else None
            ),
            "trajectory_anchor_safe_head_topks": (
                list(args.trajectory_anchor_safe_head_topks)
                if args.trajectory_anchor_safe_head_topks
                else None
            ),
            "trajectory_recovery_steps": list(args.trajectory_recovery_steps),
            "trajectory_recovery_aggressive_head_topks": (
                list(args.trajectory_recovery_aggressive_head_topks)
                if args.trajectory_recovery_aggressive_head_topks
                else None
            ),
            "trajectory_recovery_safe_head_topks": (
                list(args.trajectory_recovery_safe_head_topks)
                if args.trajectory_recovery_safe_head_topks
                else None
            ),
            "experimental_minimum_sparse_topk": args.experimental_minimum_sparse_topk,
            "temporal_correspondence_radius": args.temporal_correspondence_radius,
            "temporal_spatial_block_radius": args.temporal_spatial_block_radius,
            "temporal_global_anchor_stride": args.temporal_global_anchor_stride,
            "temporal_global_spatial_block_radius": (
                args.temporal_global_spatial_block_radius
            ),
            "sol_attn": (
                {
                    "source": str(args.sol_attn_source.resolve()),
                    "tau": args.sol_tau,
                    "sensitive_tau": args.sol_sensitive_tau,
                    "anchor_tau": args.sol_anchor_tau,
                    "recovery_tau": args.sol_recovery_tau,
                    "sensitive_layers": list(args.sol_sensitive_layers),
                    "anchor_steps": list(args.sol_anchor_steps),
                    "recovery_steps": list(args.sol_recovery_steps),
                    "conditioning_policy": "exact_kv_and_query_rows",
                }
                if args.attention_backend == "trajectory-layer-modality-sol"
                else None
            ),
            "layer_routed_sensitive_layers": (
                args.layer_routed_sensitive_layers
                if args.attention_backend in (
                    "split-layer-routed-sparge",
                    "trajectory-layer-modality-sparge",
                )
                else None
            ),
            "frame_interleave_stride": args.frame_interleave_stride,
            "frame_interleave_layer_start": args.frame_interleave_layer_start,
            "frame_interleave_layer_stop": args.frame_interleave_layer_stop,
            "frame_interleave_dense_layers": list(
                args.frame_interleave_dense_layers
            ),
            "frame_interleave_dense_steps": list(
                args.frame_interleave_dense_steps
            ),
            "spatial_query_lattice_stride": args.spatial_query_lattice_stride,
            "spatial_query_lattice_layer_start": (
                args.spatial_query_lattice_layer_start
            ),
            "spatial_query_lattice_layer_stop": (
                args.spatial_query_lattice_layer_stop
            ),
            "spatial_query_lattice_dense_layers": list(
                args.spatial_query_lattice_dense_layers
            ),
            "spatial_query_lattice_dense_steps": list(
                args.spatial_query_lattice_dense_steps
            ),
            "mlp_spatial_lattice_stride": args.mlp_spatial_lattice_stride,
            "mlp_spatial_lattice_layer_start": args.mlp_spatial_lattice_layer_start,
            "mlp_spatial_lattice_layer_stop": args.mlp_spatial_lattice_layer_stop,
            "mlp_spatial_lattice_dense_layers": list(
                args.mlp_spatial_lattice_dense_layers
            ),
            "mlp_spatial_lattice_dense_steps": list(
                args.mlp_spatial_lattice_dense_steps
            ),
            "mlp_spatial_lattice_detail_fraction": (
                args.mlp_spatial_lattice_detail_fraction
            ),
            "segment_cache_layer_start": args.segment_cache_layer_start,
            "segment_cache_layer_stop": args.segment_cache_layer_stop,
            "segment_cache_reuse_steps": list(args.segment_cache_reuse_steps),
            "segment_cache_directional_trust": (
                args.segment_cache_directional_trust
            ),
            "segment_cache_directional_max_extra": (
                args.segment_cache_directional_max_extra
            ),
            "segment_cache_directional_min_cosine": (
                args.segment_cache_directional_min_cosine
            ),
            "segment_cache_protected_refresh": (
                args.segment_cache_protected_refresh
            ),
            "segment_cache_active_video_ratio": (
                args.segment_cache_active_video_ratio
            ),
            "segment_cache_dynamic_video_budget": (
                args.segment_cache_dynamic_video_budget
            ),
            "segment_cache_active_video_min_ratio": (
                args.segment_cache_active_video_min_ratio
            ),
            "segment_cache_innovation_risk_coverage": (
                args.segment_cache_innovation_risk_coverage
            ),
            "segment_cache_innovation_max_relative": (
                args.segment_cache_innovation_max_relative
            ),
            "segment_cache_active_layer_start": (
                args.segment_cache_active_layer_start
            ),
            "segment_cache_active_layer_stop": (
                args.segment_cache_active_layer_stop
            ),
            "segment_cache_sequential_layer_groups": (
                args.segment_cache_sequential_layer_groups
            ),
            "segment_cache_sequential_conservative_hold": (
                args.segment_cache_sequential_conservative_hold
            ),
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "actual_step_indices": list(args.actual_steps),
            "actual_steps": len(args.actual_steps),
            "forecast_steps": args.steps - len(args.actual_steps),
            "cache_condition_rows": not args.disable_condition_row_cache,
            "cache_condition_embeddings": args.cache_condition_embeddings,
            "cache_reference_latents": not args.disable_reference_latent_cache,
            "sampler": "turbo" if args.engine in ("lora", "reference-lora") else "res_multistep",
            "scheduler": "simple",
            "lora_strength": 1.0 if args.engine in ("lora", "reference-lora") else None,
            "offload_mode": args.offload_mode,
            "long_sequence_query_chunk_tokens": (
                args.long_sequence_query_chunk_tokens
            ),
            "long_sequence_projection_chunk_tokens": (
                args.long_sequence_projection_chunk_tokens
            ),
            "long_sequence_split_qkv_outputs": (
                args.long_sequence_split_qkv_outputs
            ),
            "long_sequence_shared_qkv_quantization": (
                args.long_sequence_shared_qkv_quantization
            ),
            "long_sequence_exact_helper_stack": (
                args.long_sequence_exact_helper_stack
            ),
            "long_sequence_single_qknorm_rope": (
                args.long_sequence_single_qknorm_rope
            ),
            "long_sequence_parallel_sparse_lut": (
                args.long_sequence_parallel_sparse_lut
            ),
            "long_sequence_partial_sparse_topk": (
                args.long_sequence_partial_sparse_topk
            ),
            "long_sequence_fused_prefix_k_quant": (
                args.long_sequence_fused_prefix_k_quant
            ),
            "long_sequence_fused_query_projection": (
                args.long_sequence_fused_query_projection
            ),
            "long_sequence_fused_qknorm_hnd_layout": (
                args.long_sequence_fused_qknorm_hnd_layout
            ),
            "long_sequence_direct_nhd_output": (
                args.long_sequence_direct_nhd_output
            ),
            "long_sequence_direct_nhd_kv": args.long_sequence_direct_nhd_kv,
            "long_sequence_direct_hnd_fp8_value": (
                args.long_sequence_direct_hnd_fp8_value
            ),
            "checkpoint_after_step": args.checkpoint_after_step,
            "checkpoint_state": (
                None
                if args.checkpoint_state is None
                else str(args.checkpoint_state.resolve())
            ),
            "vae_tile_size": args.vae_tile_size,
            "vae_tile_batch_size": args.vae_tile_batch_size,
            "vae_compile_feed_forward": args.vae_compile_feed_forward,
            "vae_compile_transformer_block": args.vae_compile_transformer_block,
            "fused_rms_adaln": args.fused_rms_adaln,
        },
        "scenario_manifest": (
            str(args.candidate_registry.resolve())
            if args.candidate_registry is not None
            else None
            if args.scenario_manifest is None
            else str(args.scenario_manifest.resolve())
        ),
        "v19_blueprint_manifest": (
            None
            if args.v19_blueprint_manifest is None
            else str(args.v19_blueprint_manifest.resolve())
        ),
        "startup_seconds": startup_seconds,
        "startup_tasks": task_seconds,
        "requests": [],
    }
    report_path = args.output_root / f"{safe_prefix}_hot_session.json"
    try:
        for index, scenario in enumerate(scenarios):
            scenario_v19_blueprint = scenario.get("_v19_blueprint", v19_blueprint)
            scenario_v19_runtime_schedule = (
                ()
                if scenario_v19_blueprint is None
                else runtime_schedule_from_blueprint(scenario_v19_blueprint)
            )
            scenario_v19_actual_steps = (
                ()
                if scenario_v19_blueprint is None
                else tuple(sorted({
                    step
                    for use in scenario_v19_blueprint.action_uses
                    if isinstance(use, V19ActionUse)
                    for step in use.step_indices
                }))
            )
            scenario_v19_total_steps = (
                0
                if not scenario_v19_runtime_schedule
                else 1 + max(
                    step
                    for step, _layer, _action in scenario_v19_runtime_schedule
                )
            )
            name = str(scenario["name"])
            seed = int(scenario["seed"])
            prompt = str(scenario["prompt"])
            width = int(scenario.get("width", args.width))
            height = int(scenario.get("height", args.height))
            frames = int(scenario.get("frames", args.frames))
            scenario_steps = int(scenario.get("steps", args.steps))
            scenario_offload_mode = str(
                scenario.get("offload_mode", args.offload_mode)
            )
            scenario_mlp_chunk = int(
                scenario.get("mlp_chunk_tokens", args.mlp_chunk_tokens or 8192)
            )
            scenario_long_query_chunk_raw = scenario.get(
                "long_sequence_query_chunk_tokens",
                args.long_sequence_query_chunk_tokens,
            )
            scenario_long_query_chunk = (
                None
                if scenario_long_query_chunk_raw is None
                else int(scenario_long_query_chunk_raw)
            )
            scenario_long_projection_chunk = int(
                scenario.get(
                    "long_sequence_projection_chunk_tokens",
                    args.long_sequence_projection_chunk_tokens,
                )
            )
            scenario_long_split_qkv = bool(
                scenario.get(
                    "long_sequence_split_qkv_outputs",
                    args.long_sequence_split_qkv_outputs,
                )
            )
            scenario_long_shared_qkv_quantization = bool(
                scenario.get(
                    "long_sequence_shared_qkv_quantization",
                    args.long_sequence_shared_qkv_quantization,
                )
            )
            scenario_long_exact_helpers = bool(
                scenario.get(
                    "long_sequence_exact_helper_stack",
                    args.long_sequence_exact_helper_stack,
                )
            )
            scenario_long_single_qknorm = bool(
                scenario.get(
                    "long_sequence_single_qknorm_rope",
                    args.long_sequence_single_qknorm_rope,
                )
            )
            scenario_long_parallel_lut = bool(
                scenario.get(
                    "long_sequence_parallel_sparse_lut",
                    args.long_sequence_parallel_sparse_lut,
                )
            )
            scenario_long_partial_topk = bool(
                scenario.get(
                    "long_sequence_partial_sparse_topk",
                    args.long_sequence_partial_sparse_topk,
                )
            )
            scenario_long_fused_prefix_k_quant = bool(
                scenario.get(
                    "long_sequence_fused_prefix_k_quant",
                    args.long_sequence_fused_prefix_k_quant,
                )
            )
            scenario_long_fused_query_projection = bool(
                scenario.get(
                    "long_sequence_fused_query_projection",
                    args.long_sequence_fused_query_projection,
                )
            )
            scenario_long_fused_qknorm_hnd_layout = bool(
                scenario.get(
                    "long_sequence_fused_qknorm_hnd_layout",
                    args.long_sequence_fused_qknorm_hnd_layout,
                )
            )
            scenario_long_direct_nhd_output = bool(
                scenario.get(
                    "long_sequence_direct_nhd_output",
                    args.long_sequence_direct_nhd_output,
                )
            )
            scenario_long_direct_nhd_kv = bool(
                scenario.get(
                    "long_sequence_direct_nhd_kv",
                    args.long_sequence_direct_nhd_kv,
                )
            )
            scenario_long_direct_hnd_fp8_value = bool(
                scenario.get(
                    "long_sequence_direct_hnd_fp8_value",
                    args.long_sequence_direct_hnd_fp8_value,
                )
            )
            scenario_prefetch_depth = int(
                scenario.get("prefetch_depth", args.prefetch_depth)
            )
            scenario_resident_blocks = int(
                scenario.get("resident_block_count", args.resident_block_count)
            )
            scenario_vae_tile = scenario.get("vae_tile_size", args.vae_tile_size)
            if scenario_vae_tile is not None:
                scenario_vae_tile = int(scenario_vae_tile)
            scenario_vae_tile_batch = int(
                scenario.get("vae_tile_batch_size", args.vae_tile_batch_size)
            )
            scenario_vae_block_compile = bool(
                scenario.get("vae_transformer_block_compile", False)
            )
            if scenario_vae_tile_batch <= 0:
                raise ValueError("scenario vae_tile_batch_size must be positive")
            scenario_attention_topk = scenario.get("attention_topk")
            scenario_fused_rms_adaln = bool(
                scenario.get("fused_rms_adaln", args.fused_rms_adaln)
            )
            scenario_dense_qk_quant_gran = str(
                scenario.get("dense_qk_quant_gran", "per_thread")
            )
            scenario_frame_stride = int(
                scenario.get(
                    "frame_interleave_stride", args.frame_interleave_stride
                )
            )
            scenario_frame_layer_start = int(
                scenario.get(
                    "frame_interleave_layer_start",
                    args.frame_interleave_layer_start,
                )
            )
            scenario_frame_layer_stop = int(
                scenario.get(
                    "frame_interleave_layer_stop",
                    args.frame_interleave_layer_stop,
                )
            )
            scenario_frame_dense_layers = tuple(
                int(value)
                for value in scenario.get(
                    "frame_interleave_dense_layers",
                    args.frame_interleave_dense_layers,
                )
            )
            scenario_frame_dense_steps = tuple(
                int(value)
                for value in scenario.get(
                    "frame_interleave_dense_steps",
                    args.frame_interleave_dense_steps,
                )
            )
            scenario_query_lattice_stride = int(
                scenario.get(
                    "spatial_query_lattice_stride",
                    args.spatial_query_lattice_stride,
                )
            )
            scenario_query_lattice_layer_start = int(
                scenario.get(
                    "spatial_query_lattice_layer_start",
                    args.spatial_query_lattice_layer_start,
                )
            )
            scenario_query_lattice_layer_stop = int(
                scenario.get(
                    "spatial_query_lattice_layer_stop",
                    args.spatial_query_lattice_layer_stop,
                )
            )
            scenario_query_lattice_dense_layers = tuple(
                int(value)
                for value in scenario.get(
                    "spatial_query_lattice_dense_layers",
                    args.spatial_query_lattice_dense_layers,
                )
            )
            scenario_query_lattice_dense_steps = tuple(
                int(value)
                for value in scenario.get(
                    "spatial_query_lattice_dense_steps",
                    args.spatial_query_lattice_dense_steps,
                )
            )
            scenario_segment_cache_start = int(
                scenario.get(
                    "segment_cache_layer_start", args.segment_cache_layer_start
                )
            )
            scenario_segment_cache_stop = int(
                scenario.get(
                    "segment_cache_layer_stop", args.segment_cache_layer_stop
                )
            )
            scenario_segment_cache_steps = tuple(
                int(value)
                for value in scenario.get(
                    "segment_cache_reuse_steps", args.segment_cache_reuse_steps
                )
            )
            scenario_actual_steps_raw = scenario.get("actual_step_indices")
            scenario_actual_steps = (
                args.actual_steps
                if scenario_actual_steps_raw is None
                else tuple(int(value) for value in scenario_actual_steps_raw)
            )
            if (
                not scenario_actual_steps
                or tuple(sorted(set(scenario_actual_steps))) != scenario_actual_steps
                or any(value < 0 or value >= scenario_steps for value in scenario_actual_steps)
            ):
                raise ValueError(
                    "scenario actual_step_indices must be sorted, unique and inside steps"
                )
            scenario_cache_condition_rows = scenario.get(
                "cache_condition_rows"
            )
            if scenario_cache_condition_rows is None:
                scenario_cache_condition_rows = not args.disable_condition_row_cache
            scenario_cache_condition_embeddings = scenario.get(
                "cache_condition_embeddings"
            )
            if scenario_cache_condition_embeddings is None:
                scenario_cache_condition_embeddings = args.cache_condition_embeddings
            scenario_cache_reference_latents = scenario.get(
                "cache_reference_latents"
            )
            if scenario_cache_reference_latents is None:
                scenario_cache_reference_latents = not args.disable_reference_latent_cache
            if scenario_dense_qk_quant_gran not in ("per_thread", "per_warp"):
                raise ValueError(
                    "scenario dense_qk_quant_gran must be per_thread or per_warp"
                )
            if scenario_frame_stride <= 0:
                raise ValueError("scenario frame_interleave_stride must be positive")
            if scenario_query_lattice_stride <= 0:
                raise ValueError("scenario spatial_query_lattice_stride must be positive")
            if not (
                0
                <= scenario_query_lattice_layer_start
                <= scenario_query_lattice_layer_stop
                <= 50
            ):
                raise ValueError(
                    "scenario spatial Query lattice range must lie inside [0, 50]"
                )
            if any(
                layer < 0 or layer >= 50
                for layer in scenario_query_lattice_dense_layers
            ):
                raise ValueError(
                    "scenario spatial Query lattice dense layer lies outside [0, 50)"
                )
            if any(
                step < 0 or step >= scenario_steps
                for step in scenario_query_lattice_dense_steps
            ):
                raise ValueError(
                    "scenario spatial Query lattice dense step lies outside request"
                )
            if not (
                0
                <= scenario_segment_cache_start
                <= scenario_segment_cache_stop
                <= 50
            ):
                raise ValueError("scenario segment cache range must lie inside [0, 50]")
            if bool(scenario_segment_cache_steps) != (
                scenario_segment_cache_start < scenario_segment_cache_stop
            ):
                raise ValueError(
                    "scenario segment cache requires a non-empty range and reuse steps"
                )
            if (
                tuple(sorted(set(scenario_segment_cache_steps)))
                != scenario_segment_cache_steps
                or not set(scenario_segment_cache_steps).issubset(
                    scenario_actual_steps
                )
            ):
                raise ValueError(
                    "scenario segment cache steps must be sorted unique actual steps"
                )
            if scenario_attention_topk is not None:
                scenario_attention_topk = float(scenario_attention_topk)
                if not 0.5 <= scenario_attention_topk <= 1.0:
                    raise ValueError(
                        "scenario attention_topk must be between 0.5 and 1.0"
                    )
            if (
                args.attention_backend != "routed"
                and scenario_attention_topk is not None
            ):
                raise ValueError(
                    "scenario attention_topk requires --attention-backend routed"
                )
            first_frame = scenario.get("first_frame", args.first_frame)
            last_frame = scenario.get("last_frame", args.last_frame)
            reference_images = tuple(Path(value).resolve() for value in scenario.get("reference_images", ()))
            reference_videos = tuple(Path(value).resolve() for value in scenario.get("reference_videos", ()))
            reference_audios = tuple(Path(value).resolve() for value in scenario.get("reference_audios", ()))
            refinement_latents_path = scenario.get("refinement_latents_path")
            if refinement_latents_path is not None:
                refinement_latents_path = Path(refinement_latents_path).resolve()
            sampler_state_path = scenario.get(
                "sampler_state_path", args.sampler_state
            )
            if sampler_state_path is not None:
                sampler_state_path = Path(sampler_state_path).resolve()
            checkpoint_after_step_raw = scenario.get(
                "checkpoint_after_step", args.checkpoint_after_step
            )
            checkpoint_after_step = (
                None
                if checkpoint_after_step_raw is None
                else int(checkpoint_after_step_raw)
            )
            checkpoint_state_path = scenario.get(
                "checkpoint_state_path", args.checkpoint_state
            )
            if checkpoint_state_path is not None:
                checkpoint_state_path = Path(checkpoint_state_path).resolve()
            save_final_latents_path = scenario.get("save_final_latents_path")
            if save_final_latents_path is not None:
                save_final_latents_path = Path(save_final_latents_path).resolve()
            if first_frame is not None:
                first_frame = Path(first_frame).resolve()
            if last_frame is not None:
                last_frame = Path(last_frame).resolve()
            output = args.output_root / (
                f"{safe_prefix}_{name}_{width}x{height}_"
                f"{frames}f_seed{seed}.mp4"
            )
            preview_ready_callback = None
            preview_decision_wait = None
            if args.pause_for_preview_decision or args.preview_decision_file is not None:
                if args.preview_step_index is None or args.preview_output is None:
                    raise ValueError(
                        "preview pause requires "
                        "--preview-step-index and --preview-output"
                    )
                if args.pause_for_preview_decision and args.preview_decision_file is not None:
                    raise ValueError(
                        "use either --pause-for-preview-decision or "
                        "--preview-decision-file"
                    )

                decision_file = (
                    None
                    if args.preview_decision_file is None
                    else args.preview_decision_file.resolve()
                )
                if decision_file is not None:
                    decision_file.parent.mkdir(parents=True, exist_ok=True)
                    decision_file.unlink(missing_ok=True)

                def preview_ready_callback(metadata):
                    print(
                        "PREVIEW_READY "
                        + json.dumps(metadata, ensure_ascii=False),
                        flush=True,
                    )

                def preview_decision_wait():
                    if decision_file is not None:
                        print(
                            f"PREVIEW_DECISION_FILE {decision_file}",
                            flush=True,
                        )
                        while True:
                            try:
                                decision = decision_file.read_text(
                                    encoding="utf-8"
                                ).strip().lower()
                            except FileNotFoundError:
                                time.sleep(0.25)
                                continue
                            if decision in ("continue", "discard"):
                                return decision
                            time.sleep(0.25)
                    while True:
                        decision = input(
                            "PREVIEW_DECISION [continue/discard]: "
                        ).strip().lower()
                        if decision in ("continue", "discard"):
                            return decision
                        print("type 'continue' or 'discard'", flush=True)

            joint_plan = None
            if scenario_v19_blueprint is not None:
                if scenario_steps != scenario_v19_total_steps:
                    raise ValueError(
                        "scenario steps do not match the sealed V19 blueprint"
                    )
                scenario_actual_steps = scenario_v19_actual_steps
            elif args.joint_policy is not None:
                visual_condition_count = (
                    int(first_frame is not None)
                    + int(last_frame is not None)
                    + len(reference_images)
                    + len(reference_videos)
                )
                latent_frames = H3WorkloadAnalyzer.video_latent_frames(frames)
                spatial_tokens = (height // 32) * (width // 32)
                estimated_text_tokens = max(
                    128, min(1024, int(math.ceil(len(prompt) * 0.55)))
                )
                audio_tokens = 2 * round((frames / args.fps) * 40.0)
                joint_plan = H3JointAccelerationScheduler(
                    policy_id=args.joint_policy
                ).plan(
                    scenario_steps,
                    float(args.joint_acceleration),
                    allow_forecast=True,
                    workload=JointWorkloadContext(
                        packed_tokens=(
                            latent_frames * spatial_tokens
                            + visual_condition_count * spatial_tokens
                            + audio_tokens
                            + estimated_text_tokens
                        ),
                        condition_count=visual_condition_count,
                        service_family=(
                            "reference"
                            if args.engine == "reference"
                            else "first_last"
                        ),
                        model_variant="base",
                    ),
                )
                scenario_actual_steps = joint_plan.actual_step_indices

            request = HotSessionRequest(
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                frames=frames,
                fps=args.fps,
                steps=scenario_steps,
                output_path=output,
                memory_mode=scenario.get("memory_mode", args.memory_mode),
                actual_step_indices=scenario_actual_steps,
                attention_action_schedule=(
                    tuple(scenario_v19_runtime_schedule)
                    if scenario_v19_blueprint is not None
                    else (() if joint_plan is None else tuple(
                        (step, layer, action)
                        for (step, layer), action in sorted(
                            joint_plan.runtime_action_schedule().items()
                        )
                    ))
                ),
                attention_online_guard_id=(
                    None if joint_plan is None else joint_plan.online_guard_id
                ),
                attention_online_budget_dense_layers=(
                    0.0
                    if joint_plan is None or joint_plan.online_guard_id is None
                    else joint_plan.online_recovery_reserve_units * 50.0
                ),
                attention_online_rebate_schedule=(
                    () if joint_plan is None else joint_plan.online_rebate_schedule
                ),
                acceleration_plan_summary=(
                    {
                        "schema_version": "h3_v19_sealed_blueprint_execution_v1",
                        "candidate_id": scenario_v19_blueprint.candidate_id,
                        "execution_digest": v19_blueprint_execution_digest(
                            scenario_v19_blueprint
                        ),
                        "source": scenario_v19_blueprint.source,
                        "blueprint_path": scenario.get("_v19_blueprint_path"),
                    }
                    if scenario_v19_blueprint is not None
                    else (None if joint_plan is None else {
                        key: value
                        for key, value in joint_plan.to_dict().items()
                        if key != "attention_decisions"
                    })
                ),
                v19_acceleration=args.v24_acceleration,
                cache_condition_rows=bool(scenario_cache_condition_rows),
                cache_condition_embeddings=bool(
                    scenario_cache_condition_embeddings
                ),
                cache_reference_latents=bool(scenario_cache_reference_latents),
                execution_plan=(
                    None
                    if args.auto_route
                    else ExecutionPlan(
                        offload_mode=OffloadMode(scenario_offload_mode),
                        mlp_chunk_tokens=scenario_mlp_chunk,
                        prefetch_depth=scenario_prefetch_depth,
                        resident_block_count=scenario_resident_blocks,
                        vae_spatial_tile=(
                            None
                            if scenario_vae_tile is None
                            else (scenario_vae_tile, scenario_vae_tile)
                        ),
                        vae_tile_batch_size=scenario_vae_tile_batch,
                        vae_transformer_block_compile=scenario_vae_block_compile,
                        attention_topk=scenario_attention_topk,
                        fused_rms_adaln=scenario_fused_rms_adaln,
                        dense_qk_quant_gran=scenario_dense_qk_quant_gran,
                        long_sequence_query_chunk_tokens=(
                            scenario_long_query_chunk
                        ),
                        long_sequence_projection_chunk_tokens=(
                            scenario_long_projection_chunk
                        ),
                        long_sequence_split_qkv_outputs=(
                            scenario_long_split_qkv
                        ),
                        long_sequence_shared_qkv_quantization=(
                            scenario_long_shared_qkv_quantization
                        ),
                        long_sequence_exact_helper_stack=(
                            scenario_long_exact_helpers
                        ),
                        long_sequence_single_qknorm_rope=(
                            scenario_long_single_qknorm
                        ),
                        long_sequence_parallel_sparse_lut=(
                            scenario_long_parallel_lut
                        ),
                        long_sequence_partial_sparse_topk=(
                            scenario_long_partial_topk
                        ),
                        long_sequence_fused_prefix_k_quant=(
                            scenario_long_fused_prefix_k_quant
                        ),
                        long_sequence_fused_query_projection=(
                            scenario_long_fused_query_projection
                        ),
                        long_sequence_fused_qknorm_hnd_layout=(
                            scenario_long_fused_qknorm_hnd_layout
                        ),
                        long_sequence_direct_nhd_output=(
                            scenario_long_direct_nhd_output
                        ),
                        long_sequence_direct_nhd_kv=(
                            scenario_long_direct_nhd_kv
                        ),
                        long_sequence_direct_hnd_fp8_value=(
                            scenario_long_direct_hnd_fp8_value
                        ),
                        frame_interleave_stride=scenario_frame_stride,
                        frame_interleave_layer_start=(
                            scenario_frame_layer_start
                        ),
                        frame_interleave_layer_stop=scenario_frame_layer_stop,
                        frame_interleave_dense_layers=(
                            scenario_frame_dense_layers
                        ),
                        frame_interleave_dense_steps=scenario_frame_dense_steps,
                        spatial_query_lattice_stride=(
                            scenario_query_lattice_stride
                        ),
                        spatial_query_lattice_layer_start=(
                            scenario_query_lattice_layer_start
                        ),
                        spatial_query_lattice_layer_stop=(
                            scenario_query_lattice_layer_stop
                        ),
                        spatial_query_lattice_dense_layers=(
                            scenario_query_lattice_dense_layers
                        ),
                        spatial_query_lattice_dense_steps=(
                            scenario_query_lattice_dense_steps
                        ),
                        mlp_spatial_lattice_stride=args.mlp_spatial_lattice_stride,
                        mlp_spatial_lattice_layer_start=(
                            args.mlp_spatial_lattice_layer_start
                        ),
                        mlp_spatial_lattice_layer_stop=(
                            args.mlp_spatial_lattice_layer_stop
                        ),
                        mlp_spatial_lattice_dense_layers=(
                            args.mlp_spatial_lattice_dense_layers
                        ),
                        mlp_spatial_lattice_dense_steps=(
                            args.mlp_spatial_lattice_dense_steps
                        ),
                        mlp_spatial_lattice_detail_fraction=(
                            args.mlp_spatial_lattice_detail_fraction
                        ),
                        segment_cache_layer_start=scenario_segment_cache_start,
                        segment_cache_layer_stop=scenario_segment_cache_stop,
                        segment_cache_reuse_steps=scenario_segment_cache_steps,
                        segment_cache_directional_trust=(
                            args.segment_cache_directional_trust
                        ),
                        segment_cache_directional_max_extra=(
                            args.segment_cache_directional_max_extra
                        ),
                        segment_cache_directional_min_cosine=(
                            args.segment_cache_directional_min_cosine
                        ),
                        segment_cache_protected_refresh=(
                            args.segment_cache_protected_refresh
                        ),
                        segment_cache_active_video_ratio=(
                            args.segment_cache_active_video_ratio
                        ),
                        segment_cache_dynamic_video_budget=(
                            args.segment_cache_dynamic_video_budget
                        ),
                        segment_cache_active_video_min_ratio=(
                            args.segment_cache_active_video_min_ratio
                        ),
                        segment_cache_innovation_risk_coverage=(
                            args.segment_cache_innovation_risk_coverage
                        ),
                        segment_cache_innovation_max_relative=(
                            args.segment_cache_innovation_max_relative
                        ),
                        segment_cache_active_layer_start=(
                            args.segment_cache_active_layer_start
                        ),
                        segment_cache_active_layer_stop=(
                            args.segment_cache_active_layer_stop
                        ),
                        segment_cache_sequential_layer_groups=(
                            args.segment_cache_sequential_layer_groups
                        ),
                        segment_cache_sequential_conservative_hold=(
                            args.segment_cache_sequential_conservative_hold
                        ),
                    )
                ),
                first_frame=first_frame,
                last_frame=last_frame,
                reference_images=reference_images,
                reference_videos=reference_videos,
                reference_audios=reference_audios,
                refinement_latents_path=refinement_latents_path,
                refinement_denoise=(
                    None
                    if scenario.get("refinement_denoise") is None
                    else float(scenario["refinement_denoise"])
                ),
                refinement_spatial_mode=str(
                    scenario.get("refinement_spatial_mode", "strict")
                ),
                preserve_refinement_audio=bool(
                    scenario.get("preserve_refinement_audio", True)
                ),
                sampler_state_path=sampler_state_path,
                checkpoint_after_step=checkpoint_after_step,
                checkpoint_state_path=checkpoint_state_path,
                save_final_latents_path=save_final_latents_path,
                preview_step_index=args.preview_step_index,
                preview_output_path=args.preview_output,
                preview_latents_path=args.preview_latents,
                preview_decode_mode=args.preview_decode_mode,
                preview_forecast_steps=args.preview_forecast_steps,
                preview_forecast_output_path=args.preview_forecast_output,
                preview_branch_steps=args.preview_branch_steps,
                preview_branch_actual_step_indices=(
                    args.preview_branch_actual_steps
                ),
                preview_branch_spatial_scale=args.preview_branch_spatial_scale,
                preview_branch_warm_history=args.preview_branch_warm_history,
                preview_branch_force_dense=args.preview_branch_force_dense,
                preview_branch_use_lora=args.preview_branch_use_lora,
                preview_audio_branch_use_lora=args.preview_audio_branch_use_lora,
                preview_audio_branch_steps=args.preview_audio_branch_steps,
                preview_audio_branch_spatial_scale=(
                    args.preview_audio_branch_spatial_scale
                ),
                preview_ready_callback=preview_ready_callback,
                preview_decision_wait=preview_decision_wait,
                multiscale_initial_width=(
                    None
                    if scenario.get("multiscale_initial_width") is None
                    else int(scenario["multiscale_initial_width"])
                ),
                multiscale_initial_height=(
                    None
                    if scenario.get("multiscale_initial_height") is None
                    else int(scenario["multiscale_initial_height"])
                ),
                multiscale_resize_after_step=(
                    None
                    if scenario.get("multiscale_resize_after_step") is None
                    else int(scenario["multiscale_resize_after_step"])
                ),
                multiscale_highpass_strength=float(
                    scenario.get("multiscale_highpass_strength", 1.0)
                ),
                terminal_refinement_initial_width=(
                    None
                    if scenario.get("terminal_refinement_initial_width") is None
                    else int(scenario["terminal_refinement_initial_width"])
                ),
                terminal_refinement_initial_height=(
                    None
                    if scenario.get("terminal_refinement_initial_height") is None
                    else int(scenario["terminal_refinement_initial_height"])
                ),
                terminal_refinement_steps=int(
                    scenario.get("terminal_refinement_steps", 0)
                ),
                terminal_refinement_denoise=float(
                    scenario.get("terminal_refinement_denoise", 0.0125)
                ),
                terminal_refinement_dense_tail_steps=int(
                    scenario.get("terminal_refinement_dense_tail_steps", 1)
                ),
                terminal_refinement_low_frequency_gain=float(
                    scenario.get("terminal_refinement_low_frequency_gain", 1.0)
                ),
                terminal_refinement_temporal_lowpass=bool(
                    scenario.get("terminal_refinement_temporal_lowpass", False)
                ),
                terminal_refinement_temporal_outlier_only=bool(
                    scenario.get(
                        "terminal_refinement_temporal_outlier_only", False
                    )
                ),
            )
            power_sampler = NvidiaSmiSampler(interval_seconds=0.20)
            power_sampler.start()
            try:
                if args.profile_request == index + 1:
                    with torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ),
                        record_shapes=True,
                        profile_memory=False,
                    ) as profiler:
                        result = session.generate(request)
                    trace_path = args.output_root / f"{safe_prefix}_request{index + 1}.json"
                    table_path = args.output_root / f"{safe_prefix}_request{index + 1}_kernels.txt"
                    profiler.export_chrome_trace(str(trace_path))
                    table_path.write_text(
                        profiler.key_averages(group_by_input_shape=True).table(
                            sort_by="cuda_time_total", row_limit=200
                        ),
                        encoding="utf-8",
                    )
                else:
                    result = session.generate(request)
            except Exception as error:
                power_sampler.stop()
                item = {
                    "index": index + 1,
                    "scene": name,
                    "seed": request.seed,
                    "prompt": request.prompt,
                    "width": request.width,
                    "height": request.height,
                    "frames": request.frames,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "gpu_telemetry": power_sampler.summary(),
                }
                report["requests"].append(item)
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
                continue
            power_sampler.stop()
            gpu_telemetry = power_sampler.summary()
            if gpu_telemetry.get("sample_count"):
                gpu_telemetry["estimated_task_energy_wh"] = (
                    float(gpu_telemetry["power_w_mean"])
                    * float(result.total_seconds)
                    / 3600.0
                )
            checkpoint_result = (
                result if isinstance(result, HotSessionCheckpointResult) else None
            )
            # V24 resolves its physical trajectory only after exact Qwen
            # tokenisation inside the hot session.  ``request`` therefore
            # still carries the fail-closed all-Actual placeholder.  Report
            # the executed trajectory from runtime telemetry when present so
            # benchmark summaries cannot claim 20/0 for a real 10/10 run.
            effective_actual_step_indices = tuple(request.actual_step_indices)
            joint_summary = result.execution_profile.get("joint_acceleration")
            if isinstance(joint_summary, dict):
                resolved_actual = joint_summary.get("actual_step_indices")
                if (
                    isinstance(resolved_actual, list)
                    and all(isinstance(step, int) for step in resolved_actual)
                ):
                    effective_actual_step_indices = tuple(resolved_actual)
            item = {
                "index": index + 1,
                "scene": name,
                "seed": request.seed,
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "frames": request.frames,
                "steps": request.steps,
                "actual_step_indices": list(effective_actual_step_indices),
                "attention_action_schedule": [
                    list(cell) for cell in request.attention_action_schedule
                ],
                "actual_steps": len(effective_actual_step_indices),
                "forecast_steps": request.steps - len(effective_actual_step_indices),
                "refinement_latents_path": (
                    None
                    if request.refinement_latents_path is None
                    else str(request.refinement_latents_path)
                ),
                "refinement_denoise": request.refinement_denoise,
                "sampler_state_path": (
                    None
                    if request.sampler_state_path is None
                    else str(request.sampler_state_path)
                ),
                "refinement_spatial_mode": request.refinement_spatial_mode,
                "preserve_refinement_audio": request.preserve_refinement_audio,
                "save_final_latents_path": (
                    None
                    if request.save_final_latents_path is None
                    else str(request.save_final_latents_path)
                ),
                "preview_step_index": request.preview_step_index,
                "preview_output_path": (
                    None
                    if request.preview_output_path is None
                    else str(request.preview_output_path)
                ),
                "preview_latents_path": (
                    None
                    if request.preview_latents_path is None
                    else str(request.preview_latents_path)
                ),
                "multiscale_initial_width": request.multiscale_initial_width,
                "multiscale_initial_height": request.multiscale_initial_height,
                "multiscale_resize_after_step": (
                    request.multiscale_resize_after_step
                ),
                "multiscale_highpass_strength": (
                    request.multiscale_highpass_strength
                ),
                "terminal_refinement_initial_width": (
                    request.terminal_refinement_initial_width
                ),
                "terminal_refinement_initial_height": (
                    request.terminal_refinement_initial_height
                ),
                "terminal_refinement_steps": request.terminal_refinement_steps,
                "terminal_refinement_denoise": (
                    request.terminal_refinement_denoise
                ),
                "terminal_refinement_dense_tail_steps": (
                    request.terminal_refinement_dense_tail_steps
                ),
                "terminal_refinement_low_frequency_gain": (
                    request.terminal_refinement_low_frequency_gain
                ),
                "terminal_refinement_temporal_lowpass": (
                    request.terminal_refinement_temporal_lowpass
                ),
                "terminal_refinement_temporal_outlier_only": (
                    request.terminal_refinement_temporal_outlier_only
                ),
                "cache_condition_rows": request.cache_condition_rows,
                "cache_condition_embeddings": request.cache_condition_embeddings,
                "cache_reference_latents": request.cache_reference_latents,
                "mlp_chunk_tokens": request.mlp_chunk_tokens,
                "status": (
                    "checkpoint" if checkpoint_result is not None else "complete"
                ),
                "output": (
                    None
                    if checkpoint_result is not None
                    else str(result.output_path)
                ),
                "checkpoint_path": (
                    None
                    if checkpoint_result is None
                    or checkpoint_result.checkpoint_path is None
                    else str(checkpoint_result.checkpoint_path)
                ),
                "completed_steps": (
                    request.steps
                    if checkpoint_result is None
                    else checkpoint_result.completed_steps
                ),
                "total_seconds": result.total_seconds,
                "phases": result.phases,
                "step_seconds": result.step_seconds,
                "peak_allocated_gib": result.peak_allocated_gib,
                "peak_reserved_gib": result.peak_reserved_gib,
                "gpu_telemetry": gpu_telemetry,
                "forecast_profile": (
                    {}
                    if checkpoint_result is not None
                    else result.forecast_profile
                ),
                "execution_profile": result.execution_profile,
            }
            report["requests"].append(item)
            if forecast_factory is not None:
                report["adaptive_forecast"] = forecast_factory.policy.export()
            if hasattr(attention_backend, "telemetry"):
                report["adaptive_attention"] = attention_backend.telemetry()
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
    finally:
        session.close()

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
