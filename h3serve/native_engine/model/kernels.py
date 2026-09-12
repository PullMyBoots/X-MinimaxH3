"""SM89 production-kernel dispatch for the standalone H3 graph.

The public model graph remains CPU importable.  CUDA/Triton code is imported
only after a real CUDA activation reaches a supported fixed-shape fast path.
Every fast path preserves the operation and BF16/FP16 store order of the
accepted reference implementation and has an explicit eager fallback.
"""

from __future__ import annotations

import os
import importlib
import math
import sys
import time
import types
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..planner.online_guard import (
    CalibratedPhaseGrowthGuard,
    ROUND221_RUNTIME_GROWTH_THRESHOLD,
    allocate_phase_sentinels,
)

if TYPE_CHECKING:
    from .layers import ModulationSegment, RMSNorm


_ATTENTION_STEP: ContextVar[tuple[int, int] | None] = ContextVar(
    "h3_native_attention_step", default=None
)
_ATTENTION_ACTUAL_STEPS: ContextVar[tuple[int, ...] | None] = ContextVar(
    "h3_native_attention_actual_steps", default=None
)
_ATTENTION_LAYER: ContextVar[int | None] = ContextVar(
    "h3_native_attention_layer", default=None
)
_ATTENTION_SPARSE_TOPK: ContextVar[float | None] = ContextVar(
    "h3_native_attention_sparse_topk", default=None
)
_ATTENTION_ACTION_SCHEDULE: ContextVar[
    Mapping[tuple[int, int], str] | None
] = ContextVar("h3_native_attention_action_schedule", default=None)
_ATTENTION_PROTECTED_PREFIX: ContextVar[int] = ContextVar(
    "h3_native_attention_protected_prefix", default=0
)
_ATTENTION_VIDEO_LAYOUT: ContextVar[tuple[int, int] | None] = ContextVar(
    "h3_native_attention_video_layout", default=None
)
_ATTENTION_VIDEO_GRID: ContextVar[tuple[int, int] | None] = ContextVar(
    "h3_native_attention_video_grid", default=None
)
_ATTENTION_FORCE_DENSE: ContextVar[bool] = ContextVar(
    "h3_native_attention_force_dense", default=False
)
_ATTENTION_ONLINE_BUDGET: ContextVar["AttentionOnlineBudget | None"] = ContextVar(
    "h3_native_attention_online_budget", default=None
)
_LONG_VIDEO_ATTENTION_ENABLED: ContextVar[bool] = ContextVar(
    "h3_native_long_video_attention_enabled", default=False
)
_LONG_SEQUENCE_QUERY_CHUNK_TOKENS: ContextVar[int | None] = ContextVar(
    "h3_native_long_sequence_query_chunk_tokens", default=None
)
_LONG_SEQUENCE_PROJECTION_CHUNK_TOKENS: ContextVar[int] = ContextVar(
    "h3_native_long_sequence_projection_chunk_tokens", default=8192
)
_LONG_SEQUENCE_SPLIT_QKV_OUTPUTS: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_split_qkv_outputs", default=False
)
_LONG_SEQUENCE_SHARED_QKV_QUANTIZATION: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_shared_qkv_quantization", default=False
)
_LONG_SEQUENCE_COMPACT_KV: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_compact_kv", default=False
)
_LONG_SEQUENCE_SINGLE_QKNORM_ROPE: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_single_qknorm_rope", default=False
)
_LONG_SEQUENCE_EXACT_HELPER_STACK: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_exact_helper_stack", default=False
)
_LONG_SEQUENCE_PARALLEL_SPARSE_LUT: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_parallel_sparse_lut", default=False
)
_LONG_SEQUENCE_PARTIAL_SPARSE_TOPK: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_partial_sparse_topk", default=False
)
_LONG_SEQUENCE_FUSED_PREFIX_K_QUANT: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_fused_prefix_k_quant", default=False
)
_LONG_SEQUENCE_FUSED_QUERY_PROJECTION: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_fused_query_projection", default=False
)
_LONG_SEQUENCE_FUSED_QKNORM_HND_LAYOUT: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_fused_qknorm_hnd_layout", default=False
)
_LONG_SEQUENCE_DIRECT_NHD_OUTPUT: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_direct_nhd_output", default=False
)
_LONG_SEQUENCE_DIRECT_NHD_KV: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_direct_nhd_kv", default=False
)
_LONG_SEQUENCE_DIRECT_HND_FP8_VALUE: ContextVar[bool] = ContextVar(
    "h3_native_long_sequence_direct_hnd_fp8_value", default=False
)
_FUSED_RMS_ADALN: ContextVar[bool] = ContextVar(
    "h3_native_fused_rms_adaln", default=False
)
_DENSE_QK_QUANT_GRAN: ContextVar[str] = ContextVar(
    "h3_native_dense_qk_quant_gran", default="per_thread"
)
ROUND219_ONLINE_GUARD_ID = "round219_noncausal_probe_upgrade_v1"
ROUND220_ONLINE_GUARD_ID = "round220_phase_sentinel_budget_v1"
ROUND221_ONLINE_GUARD_ID = "round221_calibrated_growth_budget_v1"
ROUND223_ONLINE_GUARD_ID = "round223_reserve_rebate_budget_v1"


@dataclass(slots=True)
class AttentionOnlineBudget:
    """Request-local, fail-closed ledger measured in Dense-layer equivalents.

    One exact sampled probe or one corrective layer is conservatively charged
    as a complete Dense Attention layer.  The real sampled/head-wise work is
    smaller, so this accounting may leave budget unused but cannot authorize a
    runtime overrun.  The mutable object is request-owned and shared only by
    the two block-offload slots through a ContextVar.
    """

    policy_id: str
    limit_dense_layers: float
    rebate_schedule: tuple[tuple[int, int], ...] = ()
    spent_dense_layers: float = 0.0
    denied_count: int = 0
    events: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("online budget policy id cannot be empty")
        if not math.isfinite(self.limit_dense_layers) or self.limit_dense_layers < 0.0:
            raise ValueError("online Dense-layer budget must be finite and non-negative")
        if tuple(sorted(set(self.rebate_schedule))) != self.rebate_schedule:
            raise ValueError("online rebate schedule must be sorted and unique")
        if any(step < 0 or not 0 <= layer < 50 for step, layer in self.rebate_schedule):
            raise ValueError("online rebate schedule contains an invalid H3 cell")

    @property
    def remaining_dense_layers(self) -> float:
        return max(0.0, self.limit_dense_layers - self.spent_dense_layers)

    def try_spend(
        self,
        amount: float,
        *,
        kind: str,
        step: int,
        layer: int,
    ) -> bool:
        if not math.isfinite(amount) or amount <= 0.0:
            raise ValueError("online budget charge must be finite and positive")
        accepted = self.spent_dense_layers + amount <= self.limit_dense_layers + 1e-9
        if accepted:
            self.spent_dense_layers += amount
        else:
            self.denied_count += 1
        self.events.append(
            {
                "kind": str(kind),
                "step": int(step),
                "layer": int(layer),
                "charge_dense_layers": float(amount),
                "accepted": bool(accepted),
                "spent_dense_layers": float(self.spent_dense_layers),
                "remaining_dense_layers": float(self.remaining_dense_layers),
            }
        )
        return accepted

    def telemetry(self) -> dict[str, object]:
        return {
            "schema_version": "h3_attention_online_budget_v1",
            "policy_id": self.policy_id,
            "limit_dense_layers": self.limit_dense_layers,
            "rebate_schedule": [list(cell) for cell in self.rebate_schedule],
            "spent_dense_layers": self.spent_dense_layers,
            "remaining_dense_layers": self.remaining_dense_layers,
            "denied_count": self.denied_count,
            "budget_respected": (
                self.spent_dense_layers <= self.limit_dense_layers + 1e-9
            ),
            "upgrade_only": True,
            "events": list(self.events),
        }

    def checkpoint_state(self) -> dict[str, object]:
        """Serialize the immutable contract and replayable spending ledger."""

        return {
            "schema_version": "h3_attention_online_budget_checkpoint_v1",
            "policy_id": self.policy_id,
            "limit_dense_layers": self.limit_dense_layers,
            "rebate_schedule": [list(cell) for cell in self.rebate_schedule],
            "spent_dense_layers": self.spent_dense_layers,
            "denied_count": self.denied_count,
            "events": list(self.events),
        }

    def restore_checkpoint_state(self, state: object) -> None:
        """Replay and validate a checkpoint ledger against this request."""

        if not isinstance(state, dict):
            raise ValueError("online budget checkpoint state must be an object")
        if state.get("schema_version") != "h3_attention_online_budget_checkpoint_v1":
            raise ValueError("unexpected online budget checkpoint schema")
        if state.get("policy_id") != self.policy_id:
            raise ValueError("online budget checkpoint policy mismatch")
        if not math.isclose(
            float(state.get("limit_dense_layers", math.nan)),
            self.limit_dense_layers,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("online budget checkpoint limit mismatch")
        recorded_rebate = tuple(
            (int(step), int(layer))
            for step, layer in state.get("rebate_schedule", ())
        )
        if recorded_rebate != self.rebate_schedule:
            raise ValueError("online budget checkpoint rebate schedule mismatch")
        raw_events = state.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("online budget checkpoint events are missing")

        self.spent_dense_layers = 0.0
        self.denied_count = 0
        self.events.clear()
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise ValueError("online budget checkpoint event must be an object")
            kind = raw_event.get("kind")
            accepted = raw_event.get("accepted")
            if not isinstance(kind, str) or not kind or not isinstance(accepted, bool):
                raise ValueError("online budget checkpoint event is malformed")
            try:
                amount = float(raw_event["charge_dense_layers"])
                step = int(raw_event["step"])
                layer = int(raw_event["layer"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("online budget checkpoint event is malformed") from exc
            replayed_accepted = self.try_spend(
                amount,
                kind=kind,
                step=step,
                layer=layer,
            )
            replayed = self.events[-1]
            if replayed_accepted is not accepted:
                raise ValueError("online budget checkpoint acceptance mismatch")
            for field in ("spent_dense_layers", "remaining_dense_layers"):
                try:
                    recorded_value = float(raw_event[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "online budget checkpoint running total is malformed"
                    ) from exc
                if not math.isclose(
                    float(replayed[field]),
                    recorded_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise ValueError("online budget checkpoint running total mismatch")
        if not math.isclose(
            float(state.get("spent_dense_layers", math.nan)),
            self.spent_dense_layers,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("online budget checkpoint spent total mismatch")
        if int(state.get("denied_count", -1)) != self.denied_count:
            raise ValueError("online budget checkpoint denial total mismatch")


@contextmanager
def attention_step(step_index: int, step_count: int):
    """Expose one sampling-step boundary to opt-in attention policies.

    A context variable is used instead of mutating every transformer block.
    This also reaches the two deep-copied device blocks used by Block offload,
    while remaining isolated if multiple CPU request threads are present.
    """

    if step_count <= 0 or not 0 <= step_index < step_count:
        raise ValueError("attention step must be inside the sampling schedule")
    token = _ATTENTION_STEP.set((int(step_index), int(step_count)))
    try:
        yield
    finally:
        _ATTENTION_STEP.reset(token)


@contextmanager
def attention_actual_steps(step_indices: tuple[int, ...]):
    """Expose the complete request schedule to trajectory-budget policies."""

    normalized = tuple(int(index) for index in step_indices)
    if not normalized or tuple(sorted(set(normalized))) != normalized:
        raise ValueError("actual attention steps must be sorted, unique and non-empty")
    token = _ATTENTION_ACTUAL_STEPS.set(normalized)
    try:
        yield
    finally:
        _ATTENTION_ACTUAL_STEPS.reset(token)


@contextmanager
def attention_force_dense(enabled: bool = True):
    """Fail one request-local correction step closed to dense attention."""

    token = _ATTENTION_FORCE_DENSE.set(bool(enabled))
    try:
        yield
    finally:
        _ATTENTION_FORCE_DENSE.reset(token)


@contextmanager
def attention_online_budget(budget: AttentionOnlineBudget | None):
    """Expose one request-owned online-upgrade ledger to the hot backend."""

    token = _ATTENTION_ONLINE_BUDGET.set(budget)
    try:
        yield
    finally:
        _ATTENTION_ONLINE_BUDGET.reset(token)


@contextmanager
def long_video_attention(enabled: bool):
    """Enable the measured long-video backend for one eligible request only."""

    token = _LONG_VIDEO_ATTENTION_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _LONG_VIDEO_ATTENTION_ENABLED.reset(token)


def current_long_video_attention_enabled() -> bool:
    """Return whether this request selected the measured long-video route."""

    return bool(_LONG_VIDEO_ATTENTION_ENABLED.get())


@contextmanager
def long_sequence_query_chunking(
    chunk_tokens: int | None,
    *,
    projection_chunk_tokens: int = 8192,
    split_qkv_outputs: bool = False,
    shared_qkv_quantization: bool = False,
    compact_kv: bool = False,
    single_qknorm_rope: bool = False,
    exact_helper_stack: bool = False,
    parallel_sparse_lut: bool = False,
    partial_sparse_topk: bool = False,
    fused_prefix_k_quant: bool = False,
    fused_query_projection: bool = False,
    fused_qknorm_hnd_layout: bool = False,
    direct_nhd_output: bool = False,
    direct_nhd_kv: bool = False,
    direct_hnd_fp8_value: bool = False,
):
    """Select the exact memory-bounded Attention execution for one request.

    ``None`` preserves the established whole-query path.  A positive value is
    deliberately request scoped so the 1080p/15s memory rescue cannot leak
    into validated 720p or short-1080p workloads.  Sparge groups Query rows in
    physical blocks of 128, therefore every non-terminal chunk must respect
    that boundary.
    """

    if chunk_tokens is not None:
        chunk_tokens = int(chunk_tokens)
        if chunk_tokens < 128 or chunk_tokens % 128:
            raise ValueError(
                "long-sequence Query chunks must be positive multiples of 128"
            )
    if compact_kv and (chunk_tokens is None or not split_qkv_outputs):
        raise ValueError(
            "compact K/V requires split QKV output streaming"
        )
    if shared_qkv_quantization and (
        chunk_tokens is None or not split_qkv_outputs
    ):
        raise ValueError(
            "shared QKV activation quantization requires split QKV streaming"
        )
    if compact_kv and direct_nhd_kv:
        raise ValueError("direct NHD K/V and compact K/V are distinct executors")
    if direct_hnd_fp8_value and (
        chunk_tokens is None
        or not split_qkv_outputs
        or compact_kv
        or direct_nhd_kv
    ):
        raise ValueError(
            "direct HND FP8 V requires non-compact split-QKV HND streaming"
        )
    if fused_qknorm_hnd_layout and (
        chunk_tokens is None
        or not split_qkv_outputs
        or not single_qknorm_rope
        or not fused_query_projection
        or compact_kv
        or direct_nhd_kv
    ):
        raise ValueError(
            "fused QK-Norm/HND layout requires non-compact split-QKV "
            "streaming with the single-sided kernel"
        )
    projection_chunk_tokens = int(projection_chunk_tokens)
    if projection_chunk_tokens <= 0:
        raise ValueError("long-sequence projection chunks must be positive")
    token = _LONG_SEQUENCE_QUERY_CHUNK_TOKENS.set(chunk_tokens)
    projection_token = _LONG_SEQUENCE_PROJECTION_CHUNK_TOKENS.set(
        projection_chunk_tokens
    )
    split_qkv_token = _LONG_SEQUENCE_SPLIT_QKV_OUTPUTS.set(
        bool(split_qkv_outputs)
    )
    shared_qkv_quantization_token = (
        _LONG_SEQUENCE_SHARED_QKV_QUANTIZATION.set(
            bool(shared_qkv_quantization)
        )
    )
    compact_kv_token = _LONG_SEQUENCE_COMPACT_KV.set(bool(compact_kv))
    single_qknorm_token = _LONG_SEQUENCE_SINGLE_QKNORM_ROPE.set(
        bool(single_qknorm_rope)
    )
    exact_helper_token = _LONG_SEQUENCE_EXACT_HELPER_STACK.set(
        bool(exact_helper_stack)
    )
    parallel_lut_token = _LONG_SEQUENCE_PARALLEL_SPARSE_LUT.set(
        bool(parallel_sparse_lut)
    )
    partial_topk_token = _LONG_SEQUENCE_PARTIAL_SPARSE_TOPK.set(
        bool(partial_sparse_topk)
    )
    fused_prefix_token = _LONG_SEQUENCE_FUSED_PREFIX_K_QUANT.set(
        bool(fused_prefix_k_quant)
    )
    fused_query_projection_token = _LONG_SEQUENCE_FUSED_QUERY_PROJECTION.set(
        bool(fused_query_projection)
    )
    fused_qknorm_hnd_token = _LONG_SEQUENCE_FUSED_QKNORM_HND_LAYOUT.set(
        bool(fused_qknorm_hnd_layout)
    )
    direct_nhd_output_token = _LONG_SEQUENCE_DIRECT_NHD_OUTPUT.set(
        bool(direct_nhd_output)
    )
    direct_nhd_kv_token = _LONG_SEQUENCE_DIRECT_NHD_KV.set(
        bool(direct_nhd_kv)
    )
    direct_hnd_fp8_value_token = _LONG_SEQUENCE_DIRECT_HND_FP8_VALUE.set(
        bool(direct_hnd_fp8_value)
    )
    try:
        yield
    finally:
        _LONG_SEQUENCE_DIRECT_HND_FP8_VALUE.reset(
            direct_hnd_fp8_value_token
        )
        _LONG_SEQUENCE_DIRECT_NHD_KV.reset(direct_nhd_kv_token)
        _LONG_SEQUENCE_DIRECT_NHD_OUTPUT.reset(direct_nhd_output_token)
        _LONG_SEQUENCE_FUSED_QKNORM_HND_LAYOUT.reset(
            fused_qknorm_hnd_token
        )
        _LONG_SEQUENCE_FUSED_QUERY_PROJECTION.reset(
            fused_query_projection_token
        )
        _LONG_SEQUENCE_COMPACT_KV.reset(compact_kv_token)
        _LONG_SEQUENCE_FUSED_PREFIX_K_QUANT.reset(fused_prefix_token)
        _LONG_SEQUENCE_PARTIAL_SPARSE_TOPK.reset(partial_topk_token)
        _LONG_SEQUENCE_PARALLEL_SPARSE_LUT.reset(parallel_lut_token)
        _LONG_SEQUENCE_EXACT_HELPER_STACK.reset(exact_helper_token)
        _LONG_SEQUENCE_SINGLE_QKNORM_ROPE.reset(single_qknorm_token)
        _LONG_SEQUENCE_SHARED_QKV_QUANTIZATION.reset(
            shared_qkv_quantization_token
        )
        _LONG_SEQUENCE_SPLIT_QKV_OUTPUTS.reset(split_qkv_token)
        _LONG_SEQUENCE_PROJECTION_CHUNK_TOKENS.reset(projection_token)
        _LONG_SEQUENCE_QUERY_CHUNK_TOKENS.reset(token)


def current_long_sequence_query_chunk_tokens() -> int | None:
    """Return the request-local exact Attention Query chunk size."""

    return _LONG_SEQUENCE_QUERY_CHUNK_TOKENS.get()


def current_long_sequence_projection_chunk_tokens() -> int:
    """Return the request-local fused-QKV/out projection row chunk."""

    return int(_LONG_SEQUENCE_PROJECTION_CHUNK_TOKENS.get())


def current_long_sequence_split_qkv_outputs() -> bool:
    """Return whether exact output-row Q/K/V projection is request enabled."""

    return bool(_LONG_SEQUENCE_SPLIT_QKV_OUTPUTS.get())


def current_long_sequence_shared_qkv_quantization() -> bool:
    """Whether Q and K/V reuse one exact ConvRot row-INT8 activation."""

    return bool(_LONG_SEQUENCE_SHARED_QKV_QUANTIZATION.get())


def current_long_sequence_compact_kv() -> bool:
    """Whether this request uses two-pass quantized K/V construction."""

    return bool(_LONG_SEQUENCE_COMPACT_KV.get())


def current_long_sequence_single_qknorm_rope() -> bool:
    """Return whether the measured exact single-sided Q/K kernel is enabled."""

    return bool(_LONG_SEQUENCE_SINGLE_QKNORM_ROPE.get())


def current_long_sequence_exact_helper_stack() -> bool:
    """Return whether the quarantined v015 three-helper bundle is enabled.

    The historical field name is retained only to reproduce v015 evidence.
    Full-video evaluation disproved end-to-end exactness, so new experiments
    should use the three component switches below.
    """

    return bool(_LONG_SEQUENCE_EXACT_HELPER_STACK.get())


def current_long_sequence_parallel_sparse_lut() -> bool:
    """Return whether parallel sparse LUT construction is request enabled."""

    return bool(_LONG_SEQUENCE_PARALLEL_SPARSE_LUT.get())


def current_long_sequence_partial_sparse_topk() -> bool:
    """Return whether partial sparse Top-K selection is request enabled."""

    return bool(_LONG_SEQUENCE_PARTIAL_SPARSE_TOPK.get())


def current_long_sequence_fused_prefix_k_quant() -> bool:
    """Return whether fused prefix-K quantization is request enabled."""

    return bool(_LONG_SEQUENCE_FUSED_PREFIX_K_QUANT.get())


def current_long_sequence_fused_query_projection() -> bool:
    """Whether one live Query slab uses one output-sliced INT8 projection.

    The established memory-bounded path projects 8K row fragments and copies
    them into a second Query-sized allocation.  The fused candidate projects
    the already bounded Query slab directly.  ConvRot activation quantization
    and GEMM remain row-local and therefore preserve the model operation; only
    the temporary fragmentation and copy schedule change.
    """

    return bool(_LONG_SEQUENCE_FUSED_QUERY_PROJECTION.get())


def current_long_sequence_fused_qknorm_hnd_layout() -> bool:
    """Fuse single-sided QK-Norm/RoPE with the final NHD→HND write."""

    return bool(_LONG_SEQUENCE_FUSED_QKNORM_HND_LAYOUT.get())


def current_long_sequence_direct_nhd_output() -> bool:
    """Write streamed Attention output directly in projection-ready NHD.

    The SM89 kernels accept arbitrary sequence/head output strides even when
    Q/K use HND.  Backing their HND-shaped output view with contiguous NHD
    storage removes the otherwise mandatory HND→NHD materialization before
    the row-major output projection, without changing kernel arithmetic.
    """

    return bool(_LONG_SEQUENCE_DIRECT_NHD_OUTPUT.get())


def current_long_sequence_direct_nhd_kv() -> bool:
    """Whether sparse streamed K/V remains in projection-native NHD."""

    return bool(_LONG_SEQUENCE_DIRECT_NHD_KV.get())


def current_long_sequence_direct_hnd_fp8_value() -> bool:
    """Whether HND V is quantized directly into Sage's final FP8 ABI."""

    return bool(_LONG_SEQUENCE_DIRECT_HND_FP8_VALUE.get())


@contextmanager
def attention_layer(layer_index: int):
    """Expose the true H3 block index to request-scoped attention policies."""

    if layer_index < 0:
        raise ValueError("attention layer cannot be negative")
    token = _ATTENTION_LAYER.set(int(layer_index))
    try:
        yield
    finally:
        _ATTENTION_LAYER.reset(token)


def current_attention_layer() -> int | None:
    """Return the current true H3 block index inside the sampling boundary."""

    return _ATTENTION_LAYER.get()


def current_attention_step() -> tuple[int, int] | None:
    """Return ``(step_index, step_count)`` inside one sampling call."""

    return _ATTENTION_STEP.get()


def current_attention_actual_steps() -> tuple[int, ...] | None:
    """Return the request's complete set of real DiT evaluations."""

    return _ATTENTION_ACTUAL_STEPS.get()


def current_attention_protected_prefix() -> int:
    """Return the actual packed text/reference/audio prefix length.

    The value comes from the request's :class:`PackedLayout`; it is not a
    calibrated prompt-length constant.  Consequently Ref2VA image/audio
    references and arbitrary text lengths remain part of the exact prefix.
    """

    return int(_ATTENTION_PROTECTED_PREFIX.get())


def current_attention_video_layout() -> tuple[int, int] | None:
    """Return ``(latent_frames, tokens_per_frame)`` for generated video rows."""

    return _ATTENTION_VIDEO_LAYOUT.get()


def current_attention_video_grid() -> tuple[int, int] | None:
    """Return the true ``(patch_rows, patch_columns)`` of one latent frame."""

    return _ATTENTION_VIDEO_GRID.get()


@contextmanager
def attention_sparsity(topk: float | None):
    """Select dense or request-scoped sparse attention without rebuilding DiT."""

    if topk is not None and not 0.5 <= topk <= 1.0:
        raise ValueError("attention sparse topk must be between 0.5 and 1.0")
    token = _ATTENTION_SPARSE_TOPK.set(None if topk is None else float(topk))
    try:
        yield
    finally:
        _ATTENTION_SPARSE_TOPK.reset(token)


@contextmanager
def attention_action_schedule(
    schedule: Mapping[tuple[int, int], str] | None,
):
    """Install one immutable request-local step/layer action schedule.

    The hot service owns one model graph, so rebuilding or mutating Attention
    backends per job would destroy the warm-state contract.  A context-local
    schedule lets consecutive jobs use different acceleration strengths while
    both block-offload slots share the same read-only decision table.
    """

    normalized = None
    if schedule is not None:
        normalized = dict(schedule)
        if any(
            step < 0 or not 0 <= layer < 50 or not isinstance(action, str)
            for (step, layer), action in normalized.items()
        ):
            raise ValueError("attention action schedule contains an invalid cell")
    token = _ATTENTION_ACTION_SCHEDULE.set(normalized)
    try:
        yield
    finally:
        _ATTENTION_ACTION_SCHEDULE.reset(token)


@contextmanager
def attention_protected_prefix(token_count: int):
    """Expose the packed text/condition/audio prefix to attention policies."""

    if token_count < 0:
        raise ValueError("attention protected prefix cannot be negative")
    token = _ATTENTION_PROTECTED_PREFIX.set(int(token_count))
    try:
        yield
    finally:
        _ATTENTION_PROTECTED_PREFIX.reset(token)


@contextmanager
def attention_video_layout(
    latent_frames: int,
    frame_tokens: int,
    *,
    grid_height: int | None = None,
    grid_width: int | None = None,
):
    """Expose generated-video temporal geometry to sparse attention policies.

    H3 flattens generated video in frame-major order after the protected AV
    prefix.  The context is request scoped and lets an experimental sparse
    policy preserve a narrow same-location temporal rail without teaching the
    attention kernel about the full ``PackedLayout`` object.
    """

    if latent_frames <= 0 or frame_tokens <= 0:
        raise ValueError("attention video layout dimensions must be positive")
    if (grid_height is None) != (grid_width is None):
        raise ValueError("attention video grid requires both height and width")
    if grid_height is not None:
        if grid_height <= 0 or grid_width is None or grid_width <= 0:
            raise ValueError("attention video grid dimensions must be positive")
        if int(grid_height) * int(grid_width) != int(frame_tokens):
            raise ValueError("attention video grid does not match tokens per frame")
    token = _ATTENTION_VIDEO_LAYOUT.set((int(latent_frames), int(frame_tokens)))
    grid_token = _ATTENTION_VIDEO_GRID.set(
        None
        if grid_height is None
        else (int(grid_height), int(grid_width))
    )
    try:
        yield
    finally:
        _ATTENTION_VIDEO_GRID.reset(grid_token)
        _ATTENTION_VIDEO_LAYOUT.reset(token)


@contextmanager
def rms_adaln_fusion(enabled: bool):
    """Select the request-scoped single-kernel RMSNorm+AdaLN candidate."""

    token = _FUSED_RMS_ADALN.set(bool(enabled))
    try:
        yield
    finally:
        _FUSED_RMS_ADALN.reset(token)


@contextmanager
def dense_qk_quantization(granularity: str):
    """Select one explicit Sage dense Q/K quantization implementation."""

    if granularity not in ("per_thread", "per_warp"):
        raise ValueError("dense Q/K quantization must be per_thread or per_warp")
    token = _DENSE_QK_QUANT_GRAN.set(granularity)
    try:
        yield
    finally:
        _DENSE_QK_QUANT_GRAN.reset(token)


def _resolve_long_sequence_physical_backend(backend, query_tokens: int):
    """Resolve one wrapper stack to a memory-bounded physical operator."""

    if backend is sage_attention_sm89:
        # The prepared-K/V Dense streaming kernel is numerically validated
        # only for Sage's per-warp Q/K path.  Medium V22 requests deliberately
        # retain their reviewed per-thread path, so fail closed to the normal
        # whole-query Dense operator for those cells while still allowing the
        # exact sparse cells to use split-QKV streaming.
        return (
            _DENSE_LONG_SEQUENCE_BACKEND
            if _DENSE_QK_QUANT_GRAN.get() == "per_warp"
            else None
        )
    resolver = getattr(backend, "resolve_long_sequence_backend", None)
    return None if resolver is None else resolver(int(query_tokens))


def _can_fallback_to_unstreamed_exact_attention(
    backend, query_tokens: int
) -> bool:
    """Whether one unsupported streamed cell may safely use its old operator."""

    if backend is sage_attention_sm89:
        return True
    selector = getattr(backend, "current_action_is_exact", None)
    return bool(callable(selector) and selector(int(query_tokens)))


class StepScheduledAttentionBackend:
    """Route critical diffusion steps to dense attention.

    Short text-refiner sequences always use the dense backend.  Long packed
    audio-video sequences use the sparse backend except at explicitly selected
    dense anchor steps.  With no anchor steps this exactly preserves the former
    fixed-Sparge behavior.
    """

    def __init__(
        self,
        dense_backend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        sparse_backend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        dense_step_indices: tuple[int, ...] = (),
        dense_layer_indices: tuple[int, ...] = (),
        dense_step_layer_pairs: tuple[tuple[int, int], ...] = (),
        minimum_sparse_tokens: int = 128,
    ) -> None:
        if tuple(sorted(set(dense_step_indices))) != dense_step_indices:
            raise ValueError("dense attention steps must be sorted and unique")
        if any(index < 0 for index in dense_step_indices):
            raise ValueError("dense attention steps cannot be negative")
        if tuple(sorted(set(dense_layer_indices))) != dense_layer_indices:
            raise ValueError("dense attention layers must be sorted and unique")
        if any(index < 0 for index in dense_layer_indices):
            raise ValueError("dense attention layers cannot be negative")
        if tuple(sorted(set(dense_step_layer_pairs))) != dense_step_layer_pairs:
            raise ValueError("dense attention step/layer pairs must be sorted and unique")
        if any(step < 0 or layer < 0 for step, layer in dense_step_layer_pairs):
            raise ValueError("dense attention step/layer pairs cannot be negative")
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        self.dense_backend = dense_backend
        self.sparse_backend = sparse_backend
        self.dense_step_indices = frozenset(dense_step_indices)
        self.dense_layer_indices = frozenset(dense_layer_indices)
        self.dense_step_layer_pairs = frozenset(dense_step_layer_pairs)
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)
        # The hot session uses this marker to enable a cheap per-step finite
        # guard only for approximate attention. Exact dense requests retain
        # their established synchronization and timing behavior.
        self.approximate = True

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        if query.shape[0] < self.minimum_sparse_tokens:
            return self.dense_backend(query, key, value)
        step = _ATTENTION_STEP.get()
        # A scheduled backend outside the model sampling boundary fails safe.
        # The unscheduled fixed-Sparge backend keeps its historical behavior so
        # standalone kernel microbenchmarks remain valid.
        if self.dense_step_indices and (
            step is None or step[0] in self.dense_step_indices
        ):
            return self.dense_backend(query, key, value)
        layer = _ATTENTION_LAYER.get()
        if self.dense_layer_indices and (
            layer is None or layer in self.dense_layer_indices
        ):
            return self.dense_backend(query, key, value)
        if self.dense_step_layer_pairs and (
            step is None
            or layer is None
            or (step[0], layer) in self.dense_step_layer_pairs
        ):
            return self.dense_backend(query, key, value)
        return self.sparse_backend(query, key, value)


class ActionScheduledAttentionBackend:
    """Execute an explicit measured action for each H3 step/layer cell.

    The schedule is produced by the control-plane budget optimizer.  This
    execution class intentionally contains no quality heuristic: it maps a
    validated ``(actual_step, layer) -> action name`` decision to an existing
    Dense or sparse kernel.  Missing model context and unlisted cells fail
    closed to the configured exact action.

    A calibration sequence length is telemetry, not an input restriction.
    Every sparse action is expressed as a *fraction* of the current request's
    key blocks, while the packed conditioning prefix and generated-video
    geometry are supplied request-locally.  The same schedule therefore
    remains structurally valid when prompt length or reference media changes.
    Absolute latency estimates belong only to the calibration shape and are
    reported as such; a legal H3 request must never fail merely because its
    packed token count differs from that measurement.
    """

    approximate = True

    def __init__(
        self,
        action_backends: dict[
            str,
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ],
        schedule: dict[tuple[int, int], str],
        *,
        exact_action: str = "dense",
        minimum_sparse_tokens: int = 128,
        expected_sequence_tokens: int | None = None,
    ) -> None:
        if exact_action not in action_backends:
            raise ValueError("scheduled Attention requires an exact fallback action")
        if not schedule:
            raise ValueError("scheduled Attention requires at least one decision")
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        if expected_sequence_tokens is not None and expected_sequence_tokens < minimum_sparse_tokens:
            raise ValueError("expected sequence length must be a long Attention shape")
        unknown = set(schedule.values()) - set(action_backends)
        if unknown:
            raise ValueError(
                "scheduled Attention contains unknown actions: "
                + ", ".join(sorted(unknown))
            )
        if any(step < 0 or not 0 <= layer < 50 for step, layer in schedule):
            raise ValueError("scheduled Attention cells must use non-negative steps and H3 layers")
        self.action_backends = dict(action_backends)
        self.schedule = dict(schedule)
        self.exact_action = exact_action
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)
        self.expected_sequence_tokens = expected_sequence_tokens
        self._call_counts = {name: 0 for name in self.action_backends}
        self._fallback_calls = 0
        self._shape_adapted_calls = 0
        self._observed_sequence_tokens: dict[int, int] = {}

    def __deepcopy__(self, memo):
        # Both alternating block-offload slots execute one sequential H3
        # trajectory.  They must share the immutable plan and action telemetry.
        memo[id(self)] = self
        return self

    def _selected_action(self, query_tokens: int) -> str:
        if query_tokens < self.minimum_sparse_tokens:
            return self.exact_action
        step = current_attention_step()
        layer = current_attention_layer()
        # Token-refiner and any call outside the H3 sampling boundary are
        # exact regardless of their sequence length.  Shape validation applies
        # only after the true DiT step/layer context proves this is a scheduled
        # long Attention call.
        if step is None or layer is None:
            self._fallback_calls += 1
            return self.exact_action
        self._observed_sequence_tokens[query_tokens] = (
            self._observed_sequence_tokens.get(query_tokens, 0) + 1
        )
        if (
            self.expected_sequence_tokens is not None
            and query_tokens != self.expected_sequence_tokens
        ):
            # The measured table supplies relative layer/step actions.  Its
            # Top-K values are fractions of the request-local key-block count,
            # and SplitModalityProtectedSpargeAttentionBackend reads the true
            # protected prefix/video layout from context.  A different prompt
            # or any legal Ref2VA media set therefore changes work and timing,
            # not validity.  Preserve the action and make the extrapolation
            # explicit in telemetry instead of rejecting the user request.
            self._shape_adapted_calls += 1
        action = self.schedule.get((step[0], layer))
        if action is None:
            self._fallback_calls += 1
            return self.exact_action
        return action

    def __call__(self, query, key, value):
        action = self._selected_action(int(query.shape[0]))
        self._call_counts[action] += 1
        return self.action_backends[action](query, key, value)

    def telemetry(self) -> dict[str, object]:
        expected = self.expected_sequence_tokens
        observed = dict(sorted(self._observed_sequence_tokens.items()))
        ratios = (
            []
            if expected is None
            else [tokens / expected for tokens in observed]
        )
        return {
            "policy": "measured_budget_action_schedule",
            "scheduled_cells": len(self.schedule),
            "calibration_sequence_tokens": expected,
            # Keep the old field for report readers written before the shape
            # contract became request-adaptive.
            "expected_sequence_tokens": expected,
            "shape_contract": "request_adaptive_fractional_actions",
            "observed_sequence_tokens": observed,
            "shape_adapted_calls": self._shape_adapted_calls,
            "observed_to_calibration_ratio_min": min(ratios) if ratios else None,
            "observed_to_calibration_ratio_max": max(ratios) if ratios else None,
            "action_calls": dict(self._call_counts),
            "exact_fallback_calls": self._fallback_calls,
        }


class RequestActionScheduledAttentionBackend:
    """Execute a different joint-optimizer schedule for every hot request.

    With no schedule installed, calls delegate to ``legacy_backend`` so old
    API jobs retain their request-scoped ``attention_keep_ratio`` behavior.
    Missing cells in a new schedule fail closed to exact Dense attention.
    """

    approximate = True
    request_routed = True

    def __init__(
        self,
        action_backends: Mapping[
            str,
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ],
        *,
        exact_action: str = "dense",
        legacy_backend: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        minimum_sparse_tokens: int = 128,
    ) -> None:
        if exact_action not in action_backends:
            raise ValueError("request schedule requires an exact action")
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        self.action_backends = dict(action_backends)
        self.exact_action = str(exact_action)
        self.legacy_backend = legacy_backend or self.action_backends[exact_action]
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)
        self._action_calls = {name: 0 for name in self.action_backends}
        self._legacy_calls = 0
        self._exact_fallback_calls = 0

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def current_action(self, query_tokens: int | None = None) -> str | None:
        schedule = _ATTENTION_ACTION_SCHEDULE.get()
        if schedule is None:
            return None
        if query_tokens is not None and query_tokens < self.minimum_sparse_tokens:
            return self.exact_action
        step = current_attention_step()
        layer = current_attention_layer()
        if step is None or layer is None:
            return self.exact_action
        action = schedule.get((step[0], layer), self.exact_action)
        return action if action in self.action_backends else self.exact_action

    def current_action_is_exact(self, query_tokens: int | None = None) -> bool:
        action = self.current_action(query_tokens)
        if action is None:
            # Legacy requests are exact only when no old sparse ratio is set.
            return _ATTENTION_SPARSE_TOPK.get() is None
        return action == self.exact_action

    def resolve_long_sequence_backend(self, query_tokens: int):
        """Resolve the current V19 cell without materializing whole QKV."""

        action = self.current_action(query_tokens)
        if action is None:
            physical = _resolve_long_sequence_physical_backend(
                self.legacy_backend, query_tokens
            )
            if physical is not None:
                self._legacy_calls += 1
            return physical
        physical = _resolve_long_sequence_physical_backend(
            self.action_backends[action], query_tokens
        )
        if physical is not None:
            self._action_calls[action] += 1
        return physical

    def __call__(self, query, key, value):
        action = self.current_action(int(query.shape[0]))
        if action is None:
            self._legacy_calls += 1
            return self.legacy_backend(query, key, value)
        if action == self.exact_action and (
            current_attention_step() is None or current_attention_layer() is None
        ):
            self._exact_fallback_calls += 1
        self._action_calls[action] += 1
        return self.action_backends[action](query, key, value)

    def telemetry(self) -> dict[str, object]:
        physical: dict[str, object] = {}
        for action, backend in self.action_backends.items():
            method = getattr(backend, "telemetry", None)
            if callable(method):
                report = method()
                if report.get("route_probe_enabled", False):
                    physical[action] = report
        return {
            "policy": "request_joint_action_schedule",
            "action_calls": dict(self._action_calls),
            "legacy_calls": self._legacy_calls,
            "exact_fallback_calls": self._exact_fallback_calls,
            "physical_action_telemetry": physical,
        }


class CausalCheckpointVerifierAttentionBackend:
    """Verify a sparse causal band against exact attention before recovery.

    Motion proxies and per-head entropy failed to identify the door/contact
    regressions in the H3 stress clips.  This wrapper therefore asks the
    original attention operator itself.  At a small number of true H3 layers
    it evaluates evenly distributed, complete query blocks with both the
    draft and exact backends.  If their normalized output disagreement exceeds
    the request's quality envelope, the complete probe layer and the remaining
    causal band are recomputed exactly for that solver step.

    The verifier never changes sampler steps or model weights.  Its state is
    shared by the two block-offload slots and is reset at every solver step.
    Missing step/layer context fails closed to exact attention.
    """

    approximate = True

    def __init__(
        self,
        dense_backend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        draft_backend: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        recovery_backend: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        probe_layers: tuple[int, ...] = (30, 34, 39),
        recovery_layers: tuple[int, ...] = (
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
        detail_step_indices: tuple[int, ...] = (17, 18, 19),
        detail_layers: tuple[int, ...] = (42, 45),
        recovery_horizon: int = 1,
        hysteresis_layers: tuple[int, ...] | None = None,
        online_probe_growth_prediction: bool = True,
        probe_first_short_circuit: bool = False,
        shared_kv_exact_probe: bool = False,
        causal_head_island: bool = False,
        inject_verified_queries: bool = False,
        repair_high_error_heads: bool = False,
        head_error_mass_coverage: float = 0.50,
        head_repair_activation_ratio: float = 0.75,
        verification_query_blocks: int = 16,
        relative_rms_threshold: float = 0.24,
        minimum_sparse_tokens: int = 128,
        online_guard_id: str | None = None,
        additional_online_guard_ids: tuple[str, ...] = (),
        phase_probe_guard_ids: tuple[str, ...] = (),
        phase_growth_guard_ids: tuple[str, ...] = (),
        reserve_rebate_guard_ids: tuple[str, ...] = (),
    ) -> None:
        if tuple(sorted(set(probe_layers))) != probe_layers or any(
            not 0 <= layer < 50 for layer in probe_layers
        ):
            raise ValueError("causal verifier probe layers must be sorted and unique")
        if tuple(sorted(set(recovery_layers))) != recovery_layers or any(
            not 0 <= layer < 50 for layer in recovery_layers
        ):
            raise ValueError("causal verifier recovery layers must be sorted and unique")
        if set(probe_layers) & set(recovery_layers):
            raise ValueError("causal verifier probe and recovery layers must be disjoint")
        if tuple(sorted(set(detail_step_indices))) != detail_step_indices or any(
            step < 0 for step in detail_step_indices
        ):
            raise ValueError("causal verifier detail steps must be sorted and unique")
        if tuple(sorted(set(detail_layers))) != detail_layers or any(
            not 0 <= layer < 50 for layer in detail_layers
        ):
            raise ValueError("causal verifier detail layers must be sorted and unique")
        if verification_query_blocks <= 0:
            raise ValueError("causal verifier query block count must be positive")
        if relative_rms_threshold < 0.0:
            raise ValueError("causal verifier threshold cannot be negative")
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        if recovery_horizon < 0:
            raise ValueError("causal verifier recovery horizon cannot be negative")
        if hysteresis_layers is not None and (
            tuple(sorted(set(hysteresis_layers))) != hysteresis_layers
            or any(not 0 <= layer < 50 for layer in hysteresis_layers)
        ):
            raise ValueError(
                "causal verifier hysteresis layers must be sorted and unique"
            )
        if inject_verified_queries and repair_high_error_heads:
            raise ValueError("causal verifier correction modes are mutually exclusive")
        if not 0.0 < head_error_mass_coverage <= 1.0:
            raise ValueError("head error mass coverage must be inside (0, 1]")
        if not 0.0 <= head_repair_activation_ratio <= 1.0:
            raise ValueError("head repair activation ratio must be inside [0, 1]")
        self.dense_backend = dense_backend
        self.draft_backend = draft_backend
        self.recovery_backend = recovery_backend
        self.probe_layers = frozenset(probe_layers)
        self.recovery_layers = frozenset(recovery_layers)
        self.detail_step_indices = frozenset(detail_step_indices)
        self.detail_layers = frozenset(detail_layers)
        self.recovery_horizon = int(recovery_horizon)
        self.hysteresis_layers = frozenset(
            probe_layers + recovery_layers
            if hysteresis_layers is None
            else hysteresis_layers
        )
        self.online_probe_growth_prediction = bool(online_probe_growth_prediction)
        self.probe_first_short_circuit = bool(probe_first_short_circuit)
        self.shared_kv_exact_probe = bool(shared_kv_exact_probe)
        self.causal_head_island = bool(causal_head_island)
        self.inject_verified_queries = bool(inject_verified_queries)
        self.repair_high_error_heads = bool(repair_high_error_heads)
        self.head_error_mass_coverage = float(head_error_mass_coverage)
        self.head_repair_activation_ratio = float(head_repair_activation_ratio)
        self.verification_query_blocks = int(verification_query_blocks)
        self.relative_rms_threshold = float(relative_rms_threshold)
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)
        guard_ids = (
            (() if online_guard_id is None else (str(online_guard_id),))
            + tuple(str(value) for value in additional_online_guard_ids)
        )
        if len(set(guard_ids)) != len(guard_ids) or any(not value for value in guard_ids):
            raise ValueError("online guard ids must be non-empty and unique")
        if not set(phase_probe_guard_ids).issubset(guard_ids):
            raise ValueError("phase-probe guard ids must be supported online guards")
        if not set(phase_growth_guard_ids).issubset(phase_probe_guard_ids):
            raise ValueError("phase-growth guards must also use phase probes")
        if not set(reserve_rebate_guard_ids).issubset(phase_growth_guard_ids):
            raise ValueError("reserve-rebate guards must also use phase growth")
        self.online_guard_id = None if online_guard_id is None else str(online_guard_id)
        self.online_guard_ids = frozenset(guard_ids)
        self.phase_probe_guard_ids = frozenset(phase_probe_guard_ids)
        self.phase_growth_guard_ids = frozenset(phase_growth_guard_ids)
        self.reserve_rebate_guard_ids = frozenset(reserve_rebate_guard_ids)
        self._phase_growth_guard = (
            CalibratedPhaseGrowthGuard(ROUND221_RUNTIME_GROWTH_THRESHOLD)
            if self.phase_growth_guard_ids
            else None
        )
        self._active_step: int | None = None
        self._ordered_probe_layers = tuple(probe_layers)
        self._first_probe_error: float | None = None
        self._probe_growth_upper: float | None = None
        self._probe_growth_pairs = 0
        self._recover_step = False
        self._recovery_mode = "dense"
        self._hysteresis_active = False
        self._recovery_consumed_step: int | None = None
        self._pending_recovery_steps = 0
        self._pending_recovery_mode = "dense"
        self._recovery_heads: torch.Tensor | None = None
        self._pending_recovery_heads: torch.Tensor | None = None
        self._recovery_target_relative_rms = self.relative_rms_threshold
        self._pending_recovery_target_relative_rms = self.relative_rms_threshold
        self._probe_records: list[dict[str, object]] = []
        self._dense_recovery_calls = 0
        self._dense_detail_calls = 0
        self._dense_hysteresis_calls = 0
        self._graded_recovery_calls = 0
        self._graded_hysteresis_calls = 0
        self._graded_accept_count = 0
        self._graded_reject_count = 0
        self._verified_query_injection_calls = 0
        self._verified_query_injection_tokens = 0
        self._head_repair_calls = 0
        self._head_repair_heads = 0
        self._preemptive_trigger_count = 0
        self._probe_first_calls = 0
        self._probe_first_rejected_full_drafts = 0
        self._shared_kv_exact_probe_calls = 0
        self._head_island_calls = 0
        self._head_island_heads = 0
        self._head_island_total_heads = 0
        self._head_island_trigger_count = 0
        self._head_island_reverify_calls = 0
        self._head_island_records: list[dict[str, object]] = []
        self._online_budget_denied_count = 0
        self._last_online_guard_id: str | None = None
        self._last_phase_probe_slots: tuple[tuple[int, int], ...] = ()
        self._request_had_trigger = False
        self._reserve_rebate_calls = 0

    def __deepcopy__(self, memo):
        # The block executor alternates between two deep-copied device slots.
        # A decision made by a probe layer must reach all later true H3 layers.
        memo[id(self)] = self
        return self

    def current_action_is_exact(self, query_tokens: int | None = None) -> bool:
        """Expose exact offline cells through the verifier wrapper stack."""

        if query_tokens is not None and query_tokens < self.minimum_sparse_tokens:
            return True
        if _ATTENTION_STEP.get() is None or _ATTENTION_LAYER.get() is None:
            return True
        selector = getattr(self.draft_backend, "current_action_is_exact", None)
        return bool(callable(selector) and selector(query_tokens))

    def resolve_long_sequence_backend(self, query_tokens: int):
        """Resolve frozen/offline V19 cells for exact memory-bounded execution.

        Online verifier modes can condition later layers on a sampled result,
        so they intentionally remain unsupported until that state machine is
        expressed in chunk space.  Current V19 offline plans carry no online
        ledger and therefore resolve to the same draft/exact action as
        ``__call__``.
        """

        step = _ATTENTION_STEP.get()
        layer = _ATTENTION_LAYER.get()
        if query_tokens < self.minimum_sparse_tokens or step is None or layer is None:
            return _DENSE_LONG_SEQUENCE_BACKEND
        draft_is_exact = getattr(
            self.draft_backend, "current_action_is_exact", None
        )
        if draft_is_exact is not None and draft_is_exact(query_tokens):
            return _resolve_long_sequence_physical_backend(
                self.draft_backend, query_tokens
            )
        if self.online_guard_ids:
            ledger = _ATTENTION_ONLINE_BUDGET.get()
            if ledger is None or ledger.policy_id not in self.online_guard_ids:
                return _resolve_long_sequence_physical_backend(
                    self.draft_backend, query_tokens
                )
        return None

    def _reset_step(self, step_index: int) -> None:
        if self._active_step != step_index:
            self._active_step = step_index
            self._first_probe_error = None
            self._recover_step = False
            self._recovery_mode = "dense"
            self._recovery_heads = None
            self._recovery_target_relative_rms = self.relative_rms_threshold
            self._hysteresis_active = False
            self._recovery_consumed_step = None

    def _recovery_output(self, query, key, value, *, hysteresis: bool):
        if not self._spend_online_unit("recovery"):
            self._online_budget_denied_count += 1
            return self.draft_backend(query, key, value)
        if self._recovery_mode == "head_island" and self._recovery_heads is not None:
            return self._head_island_output(query, key, value)
        if self._recovery_mode == "graded" and self.recovery_backend is not None:
            if hysteresis:
                self._graded_hysteresis_calls += 1
            else:
                self._graded_recovery_calls += 1
            return self.recovery_backend(query, key, value)
        if hysteresis:
            self._dense_hysteresis_calls += 1
        else:
            self._dense_recovery_calls += 1
        return self.dense_backend(query, key, value)

    def _spend_online_unit(self, kind: str) -> bool:
        """Charge one conservative Dense-layer equivalent when V9 is active."""

        if not self.online_guard_ids:
            return True
        ledger = _ATTENTION_ONLINE_BUDGET.get()
        step = _ATTENTION_STEP.get()
        layer = _ATTENTION_LAYER.get()
        if (
            ledger is None
            or ledger.policy_id not in self.online_guard_ids
            or step is None
            or layer is None
        ):
            return False
        return ledger.try_spend(
            1.0,
            kind=kind,
            step=int(step[0]),
            layer=int(layer),
        )

    @staticmethod
    def _phase_probe_slots(
        actual_steps: tuple[int, ...],
        limit_dense_layers: float,
    ) -> tuple[tuple[int, int], ...]:
        """Allocate sparse probes across solver phases while preserving repair.

        Round219 spent its complete reserve on repeated observations.  The
        Round220 guard instead chooses at most three trajectory sentinels and
        interleaves the most evidence-sensitive physical layers.  The slot
        count leaves at least five Dense-layer equivalents once layer 4 can be
        probed; smaller reserves probe layer 24 only, whose possible recovery
        layers all lie later in the same transformer pass.
        """

        return allocate_phase_sentinels(
            actual_steps, limit_dense_layers
        ).slots

    def _head_island_output(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        draft: torch.Tensor | None = None,
        exact_sample: torch.Tensor | None = None,
        sample_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Repair each H3 layer with its own teacher-selected complete heads.

        Head indices are local to one transformer layer.  Carrying layer 30's
        indices into layer 31 is therefore not a meaningful continuity rule.
        Every protected layer rechecks a small distributed query sample and
        derives its own minimal exact head island under the same request-level
        error envelope.  The envelope persists across the causal band and the
        following solver step; the head identities deliberately do not.
        """

        if sample_indices is None:
            sample_indices, _ = self._verification_indices(
                int(query.shape[0]), query.device
            )
        if draft is None:
            draft = self.draft_backend(query, key, value)
        if exact_sample is None:
            exact_sample = self.dense_backend(
                query.index_select(0, sample_indices), key, value
            )
            self._head_island_reverify_calls += 1
        draft_sample = draft.index_select(0, sample_indices)
        heads = self._minimal_exact_head_island(
            exact_sample,
            draft_sample,
            target_relative_rms=self._recovery_target_relative_rms,
        )
        if int(heads.numel()) == int(query.shape[1]):
            output = self.dense_backend(query, key, value)
        elif int(heads.numel()):
            exact_heads = self.dense_backend(
                query.index_select(1, heads),
                key.index_select(1, heads),
                value.index_select(1, heads),
            )
            draft.index_copy_(1, heads, exact_heads)
            output = draft
        else:
            output = draft
        self._recovery_heads = heads
        self._head_island_calls += 1
        self._head_island_heads += int(heads.numel())
        self._head_island_total_heads += int(query.shape[1])
        self._head_island_records.append(
            {
                "step": int(_ATTENTION_STEP.get()[0]),
                "layer": int(_ATTENTION_LAYER.get()),
                "selected_heads": int(heads.numel()),
                "total_heads": int(query.shape[1]),
                "target_relative_rms": self._recovery_target_relative_rms,
            }
        )
        return output

    @staticmethod
    def _minimal_exact_head_island(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        *,
        target_relative_rms: float,
    ) -> torch.Tensor:
        """Return the smallest complete-head set that meets the sampled bound.

        Attention heads are independent before the output projection, so the
        global squared disagreement is additive across complete heads.  This
        lets the verifier remove the largest error contributors without any
        hand-authored head list or query-row seam.
        """

        difference_energy = (
            (reference.float() - candidate.float()).square().sum(dim=(0, 2))
        )
        reference_energy = reference.float().square().sum().clamp_min(1e-12)
        total_difference = difference_energy.sum()
        target_energy = float(target_relative_rms) ** 2 * reference_energy
        if total_difference <= target_energy:
            return torch.empty(0, device=reference.device, dtype=torch.long)
        order = torch.argsort(difference_energy, descending=True)
        remaining = total_difference - torch.cumsum(
            difference_energy.index_select(0, order), dim=0
        )
        accepted = torch.nonzero(remaining <= target_energy, as_tuple=False)
        count = (
            int(accepted[0].item()) + 1
            if int(accepted.numel())
            else int(order.numel())
        )
        return order[:count].sort().values

    def _verification_indices(self, token_count: int, device: torch.device):
        target_count = min(self.verification_query_blocks * 128, token_count)
        layout = _ATTENTION_VIDEO_LAYOUT.get()
        protected_tokens = min(_ATTENTION_PROTECTED_PREFIX.get(), token_count)
        if layout is not None:
            latent_frames, frame_tokens = layout
            video_tokens = latent_frames * frame_tokens
            if protected_tokens + video_tokens <= token_count:
                # Spend a small fixed share on the packed AV/text prefix, then
                # place exact anchors inside *every* latent frame.  A flat
                # uniform sample leaves long temporal gaps at 15 seconds and
                # can miss a short contact event despite the same FLOP budget.
                prefix_count = min(protected_tokens, max(1, target_count // 8))
                prefix_indices = (
                    torch.linspace(
                        0,
                        protected_tokens - 1,
                        prefix_count,
                        device=device,
                    )
                    .round()
                    .to(torch.long)
                    .unique()
                    if protected_tokens
                    else torch.empty(0, device=device, dtype=torch.long)
                )
                remaining = max(1, target_count - int(prefix_indices.numel()))
                per_frame = max(1, min(frame_tokens, remaining // latent_frames))
                spatial = torch.linspace(
                    0,
                    frame_tokens - 1,
                    per_frame,
                    device=device,
                ).round().to(torch.long).unique()
                frame_offsets = (
                    protected_tokens
                    + torch.arange(latent_frames, device=device, dtype=torch.long)
                    * frame_tokens
                )
                video_indices = (frame_offsets[:, None] + spatial[None, :]).reshape(-1)
                indices = torch.cat((prefix_indices, video_indices)).unique(sorted=True)
                return indices[indices < token_count], "per_frame_spatial_anchors"
        block_count = (token_count + 127) // 128
        selected_count = min(self.verification_query_blocks, block_count)
        selected_blocks = torch.linspace(
            0,
            block_count - 1,
            selected_count,
            device=device,
        ).round().to(torch.long).unique()
        offsets = torch.arange(128, device=device, dtype=torch.long)
        indices = (selected_blocks[:, None] * 128 + offsets[None, :]).reshape(-1)
        return indices[indices < token_count], "uniform_complete_query_blocks"

    def _verification_video_block_indices(
        self,
        token_count: int,
        protected_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Select complete, original-coordinate 128-row video query blocks.

        Sparse routing is defined per 128-row query block.  Repacking isolated
        rows would change their block statistics and would no longer be a
        faithful preview of the full sparse draft.  Complete aligned blocks
        preserve the same query grouping while allowing a rejected probe to
        skip the otherwise wasted full sparse layer.
        """

        video_tokens = max(0, token_count - protected_tokens)
        block_count = (video_tokens + 127) // 128
        selected_count = min(self.verification_query_blocks, block_count)
        if selected_count <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        selected_blocks = (
            torch.linspace(
                0,
                block_count - 1,
                selected_count,
                device=device,
            )
            .round()
            .to(torch.long)
            .unique()
        )
        offsets = torch.arange(128, device=device, dtype=torch.long)
        indices = (selected_blocks[:, None] * 128 + offsets[None, :]).reshape(-1)
        return indices[indices < video_tokens]

    @staticmethod
    def _relative_rms_statistics(
        reference: torch.Tensor, candidate: torch.Tensor
    ) -> dict[str, float]:
        difference = (reference.float() - candidate.float()).square().mean().sqrt()
        scale = reference.float().square().mean().sqrt().clamp_min(1e-6)
        per_query_difference = (
            (reference.float() - candidate.float())
            .square()
            .mean(dim=(-1, -2))
            .sqrt()
        )
        per_query_scale = (
            reference.float().square().mean(dim=(-1, -2)).sqrt().clamp_min(1e-6)
        )
        per_query = per_query_difference / per_query_scale
        return {
            "global": float((difference / scale).item()),
            "query_p90": float(torch.quantile(per_query, 0.90).item()),
            "query_max": float(per_query.max().item()),
        }

    def _high_error_heads(
        self, reference: torch.Tensor, candidate: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        difference = (
            (reference.float() - candidate.float())
            .square()
            .mean(dim=(0, 2))
            .sqrt()
        )
        scale = reference.float().square().mean(dim=(0, 2)).sqrt().clamp_min(1e-6)
        relative = difference / scale
        error_energy = relative.square()
        order = torch.argsort(error_energy, descending=True)
        cumulative = torch.cumsum(error_energy.index_select(0, order), dim=0)
        target = self.head_error_mass_coverage * cumulative[-1].clamp_min(1e-12)
        selected_count = int(torch.searchsorted(cumulative, target).item()) + 1
        selected = order[:selected_count].sort().values
        return selected, {
            "head_relative_rms_p90": float(torch.quantile(relative, 0.90).item()),
            "head_relative_rms_max": float(relative.max().item()),
            "selected_head_count": int(selected.numel()),
            "total_head_count": int(relative.numel()),
        }

    def __call__(self, query, key, value):
        step = _ATTENTION_STEP.get()
        layer = _ATTENTION_LAYER.get()
        if (
            query.shape[0] < self.minimum_sparse_tokens
            or step is None
            or layer is None
        ):
            return self.dense_backend(query, key, value)
        draft_is_exact = getattr(
            self.draft_backend, "current_action_is_exact", None
        )
        if draft_is_exact is not None and draft_is_exact(int(query.shape[0])):
            # The joint plan already bought the exact action for this cell.
            # Running an additional exact probe would add work without adding
            # evidence, especially at acceleration=0.
            return self.draft_backend(query, key, value)
        active_online_ledger = None
        if self.online_guard_ids:
            ledger = _ATTENTION_ONLINE_BUDGET.get()
            if ledger is None or ledger.policy_id not in self.online_guard_ids:
                # V7/V8 and all legacy requests share the same hot backend but
                # must remain byte-for-byte on their versioned offline route.
                return self.draft_backend(query, key, value)
            active_online_ledger = ledger
            self._last_online_guard_id = ledger.policy_id
            if (
                self._phase_growth_guard is not None
                and ledger.policy_id in self.phase_growth_guard_ids
            ):
                if self._phase_growth_guard.begin_request(ledger):
                    self._request_had_trigger = False
        step_index = int(step[0])
        self._reset_step(step_index)
        causal_band = self.probe_layers | self.recovery_layers
        if (
            layer in causal_band
            and self._pending_recovery_steps
            and self._recovery_consumed_step != step_index
        ):
            # Consume the hold only when a complete DiT reaches the protected
            # band. Forecast evaluations stop at the shallow anchor blocks and
            # therefore cannot accidentally spend this continuity budget.
            self._pending_recovery_steps -= 1
            self._recovery_consumed_step = step_index
            self._recover_step = True
            self._recovery_mode = self._pending_recovery_mode
            self._recovery_heads = self._pending_recovery_heads
            self._recovery_target_relative_rms = (
                self._pending_recovery_target_relative_rms
            )
            self._hysteresis_active = True
        # Low-sigma steps write the final spatial detail.  This small exact
        # floor is invariant across effort levels so the speed controller can
        # never buy throughput by eroding the terminal detail path.
        actual_steps = _ATTENTION_ACTUAL_STEPS.get()
        dynamic_terminal_steps = (
            frozenset(actual_steps[-min(3, len(actual_steps)):])
            if actual_steps
            else frozenset()
        )
        if (
            step_index in self.detail_step_indices
            or step_index in dynamic_terminal_steps
        ) and layer in self.detail_layers:
            self._dense_detail_calls += 1
            return self.dense_backend(query, key, value)
        if self._hysteresis_active:
            if layer in self.hysteresis_layers:
                return self._recovery_output(query, key, value, hysteresis=True)
            if layer in causal_band:
                # The previous solver step was rejected, but only the early
                # state-forming span needs continuity.  Later layers return to
                # the coherent draft path instead of mixing query/head rows.
                return self.draft_backend(query, key, value)
        if self._recover_step and layer in causal_band:
            return self._recovery_output(query, key, value, hysteresis=False)
        if (
            active_online_ledger is not None
            and active_online_ledger.policy_id in self.reserve_rebate_guard_ids
            and not self._request_had_trigger
            and (step_index, int(layer)) in active_online_ledger.rebate_schedule
        ):
            # Every rebate cell lies after the final scheduled sentinel.  A
            # no-trigger request can therefore reclaim the otherwise dead
            # reserve without competing with observation or repair.  This is
            # still an upgrade-only action and must pass the same immutable
            # request ledger as a corrective Dense layer.
            rebate_probe_slots = self._phase_probe_slots(
                tuple(actual_steps or ()),
                active_online_ledger.limit_dense_layers,
            )
            after_final_probe = bool(rebate_probe_slots) and step_index > max(
                probe_step for probe_step, _ in rebate_probe_slots
            )
            if after_final_probe and self._spend_online_unit("reserve_rebate"):
                self._reserve_rebate_calls += 1
                return self.dense_backend(query, key, value)
            if after_final_probe:
                self._online_budget_denied_count += 1
        if layer not in self.probe_layers:
            return self.draft_backend(query, key, value)

        if (
            active_online_ledger is not None
            and active_online_ledger.policy_id in self.phase_probe_guard_ids
        ):
            phase_slots = self._phase_probe_slots(
                tuple(actual_steps or ()),
                active_online_ledger.limit_dense_layers,
            )
            self._last_phase_probe_slots = phase_slots
            if (step_index, int(layer)) not in phase_slots:
                return self.draft_backend(query, key, value)

        if not self._spend_online_unit("probe"):
            self._online_budget_denied_count += 1
            return self.draft_backend(query, key, value)

        draft = None
        selected_video_method = getattr(
            self.draft_backend, "selected_video_queries", None
        )
        protected_tokens = min(
            int(_ATTENTION_PROTECTED_PREFIX.get()), int(query.shape[0])
        )
        use_probe_first = (
            self.probe_first_short_circuit
            and selected_video_method is not None
            and 0 < protected_tokens < int(query.shape[0])
        )
        shared_probe_method = getattr(
            self.draft_backend, "full_with_exact_sample", None
        )
        if use_probe_first:
            video_indices = self._verification_video_block_indices(
                int(query.shape[0]), protected_tokens, query.device
            )
            if int(video_indices.numel()) == 0:
                use_probe_first = False
        if use_probe_first:
            sample_query = query[protected_tokens:].index_select(0, video_indices)
            draft_sample = selected_video_method(
                sample_query,
                key,
                value,
                protected_tokens=protected_tokens,
                video_query_indices=video_indices,
            )
            exact_sample = self.dense_backend(sample_query, key, value)
            indices = video_indices + protected_tokens
            selection_strategy = "uniform_aligned_video_query_blocks_probe_first"
            self._probe_first_calls += 1
        elif self.shared_kv_exact_probe and shared_probe_method is not None:
            indices, selection_strategy = self._verification_indices(
                int(query.shape[0]), query.device
            )
            draft, exact_sample = shared_probe_method(
                query,
                key,
                value,
                sample_indices=indices,
            )
            draft_sample = draft.index_select(0, indices)
            selection_strategy += "_shared_kv_exact"
            self._shared_kv_exact_probe_calls += 1
        else:
            draft = self.draft_backend(query, key, value)
            indices, selection_strategy = self._verification_indices(
                int(query.shape[0]), query.device
            )
            exact_sample = self.dense_backend(
                query.index_select(0, indices), key, value
            )
            draft_sample = draft.index_select(0, indices)
        disagreement_stats = self._relative_rms_statistics(
            exact_sample, draft_sample
        )
        high_error_heads, head_statistics = self._high_error_heads(
            exact_sample, draft_sample
        )
        disagreement = disagreement_stats["global"]
        phase_growth_observation = None
        if (
            active_online_ledger is not None
            and self._phase_growth_guard is not None
            and active_online_ledger.policy_id in self.phase_growth_guard_ids
        ):
            phase_growth_observation = self._phase_growth_guard.observe(
                int(layer), disagreement
            )
        first_probe = self._ordered_probe_layers[0]
        last_probe = self._ordered_probe_layers[-1]
        predicted_late_disagreement: float | None = None
        preemptive_trigger = False
        if layer == first_probe:
            self._first_probe_error = disagreement
            if (
                self.online_probe_growth_prediction
                and self._probe_growth_upper is not None
            ):
                predicted_late_disagreement = (
                    disagreement * self._probe_growth_upper
                )
                preemptive_trigger = (
                    predicted_late_disagreement > self.relative_rms_threshold
                )
        elif (
            layer == last_probe
            and self._first_probe_error is not None
            and self._first_probe_error > 1e-12
        ):
            # Learn the within-request error growth from complete early/late
            # probe pairs.  Keep the observed upper envelope rather than a
            # fitted scene threshold: once a request demonstrates that error
            # grows through the causal band, later steps fail safe early.
            growth = disagreement / self._first_probe_error
            if math.isfinite(growth):
                self._probe_growth_upper = (
                    growth
                    if self._probe_growth_upper is None
                    else max(self._probe_growth_upper, growth)
                )
                self._probe_growth_pairs += 1
        triggered = (
            disagreement > self.relative_rms_threshold
            or preemptive_trigger
            or (
                phase_growth_observation is not None
                and phase_growth_observation.triggered
            )
        )
        if triggered:
            self._request_had_trigger = True
        island_target = self.relative_rms_threshold
        if preemptive_trigger and self._probe_growth_upper is not None:
            island_target = self.relative_rms_threshold / self._probe_growth_upper
        island_heads = (
            self._minimal_exact_head_island(
                exact_sample,
                draft_sample,
                target_relative_rms=island_target,
            )
            if triggered and self.causal_head_island
            else torch.empty(0, device=query.device, dtype=torch.long)
        )
        probe_record = {
                "step": step_index,
                "layer": int(layer),
                "sampled_query_tokens": int(indices.numel()),
                "selection_strategy": selection_strategy,
                "relative_rms": disagreement,
                "query_relative_rms_p90": disagreement_stats["query_p90"],
                "query_relative_rms_max": disagreement_stats["query_max"],
                **head_statistics,
                "predicted_late_relative_rms": predicted_late_disagreement,
                "online_probe_growth_upper": self._probe_growth_upper,
                "preemptive_trigger": preemptive_trigger,
                "phase_baseline_relative_rms": (
                    phase_growth_observation.baseline_relative_rms
                    if phase_growth_observation is not None
                    else None
                ),
                "phase_growth_ratio": (
                    phase_growth_observation.growth_ratio
                    if phase_growth_observation is not None
                    else None
                ),
                "phase_growth_trigger": (
                    phase_growth_observation.triggered
                    if phase_growth_observation is not None
                    else False
                ),
                "triggered": triggered,
                "head_island_target_relative_rms": (
                    island_target if triggered and self.causal_head_island else None
                ),
                "head_island_selected_heads": int(island_heads.numel()),
            }
        self._probe_records.append(probe_record)
        if not triggered:
            if draft is None:
                # The sparse candidate was accepted.  Only now pay for the
                # complete sparse layer that becomes the actual model output.
                draft = self.draft_backend(query, key, value)
            if self.inject_verified_queries:
                # Attention is pointwise in the already-RoPE-transformed query
                # rows, so each sampled dense result is exact for that token.
                # The draft output is freshly allocated; update it in place to
                # avoid cloning a 100k-token activation on a 24GB card.
                draft.index_copy_(0, indices, exact_sample)
                self._verified_query_injection_calls += 1
                self._verified_query_injection_tokens += int(indices.numel())
            elif (
                self.repair_high_error_heads
                and disagreement
                >= self.relative_rms_threshold * self.head_repair_activation_ratio
            ):
                # A complete head is coherent across the whole packed temporal
                # sequence.  Recomputing selected heads avoids the spatial
                # seams produced by mixing exact and draft query rows while
                # spending dense work only where sampled error mass resides.
                exact_heads = self.dense_backend(
                    query.index_select(1, high_error_heads),
                    key.index_select(1, high_error_heads),
                    value.index_select(1, high_error_heads),
                )
                draft.index_copy_(1, high_error_heads, exact_heads)
                self._head_repair_calls += 1
                self._head_repair_heads += int(high_error_heads.numel())
            return draft

        # The sampled exact result is a verifier only.  On rejection, rerun
        # the complete layer so no query-boundary seam is introduced.
        if use_probe_first:
            self._probe_first_rejected_full_drafts += 1
        if not self._spend_online_unit("trigger_upgrade"):
            self._online_budget_denied_count += 1
            probe_record["upgrade_applied"] = False
            probe_record["upgrade_denied_by_budget"] = True
            if draft is None:
                draft = self.draft_backend(query, key, value)
            return draft
        probe_record["upgrade_applied"] = True
        probe_record["upgrade_denied_by_budget"] = False
        self._recover_step = True
        use_head_island = (
            self.causal_head_island
            and 0 < int(island_heads.numel()) < int(query.shape[1])
        )
        self._recovery_mode = "head_island" if use_head_island else "dense"
        self._recovery_heads = island_heads if use_head_island else None
        self._recovery_target_relative_rms = island_target
        if use_head_island:
            self._head_island_trigger_count += 1
        self._hysteresis_active = False
        self._recovery_consumed_step = step_index
        output = None
        if self.recovery_backend is not None:
            graded = self.recovery_backend(query, key, value)
            graded_sample = graded.index_select(0, indices)
            graded_stats = self._relative_rms_statistics(
                exact_sample, graded_sample
            )
            graded_disagreement = graded_stats["global"]
            graded_predicted_late = (
                graded_disagreement * self._probe_growth_upper
                if preemptive_trigger and self._probe_growth_upper is not None
                else None
            )
            graded_accepted = (
                graded_disagreement <= self.relative_rms_threshold
                and (
                    graded_predicted_late is None
                    or graded_predicted_late <= self.relative_rms_threshold
                )
            )
            probe_record.update(
                {
                    "graded_relative_rms": graded_disagreement,
                    "graded_query_relative_rms_p90": graded_stats["query_p90"],
                    "graded_query_relative_rms_max": graded_stats["query_max"],
                    "graded_predicted_late_relative_rms": graded_predicted_late,
                    "graded_accepted": graded_accepted,
                }
            )
            if graded_accepted:
                self._recovery_mode = "graded"
                self._graded_accept_count += 1
                self._graded_recovery_calls += 1
                output = graded
            else:
                self._graded_reject_count += 1
        if self.recovery_horizon:
            if self._pending_recovery_steps == 0:
                self._pending_recovery_mode = self._recovery_mode
            elif self._recovery_mode == "dense":
                # Exact recovery dominates an already queued graded hold.
                self._pending_recovery_mode = "dense"
            self._pending_recovery_heads = (
                self._recovery_heads.detach().clone()
                if self._recovery_mode == "head_island"
                and self._recovery_heads is not None
                else None
            )
            self._pending_recovery_target_relative_rms = (
                self._recovery_target_relative_rms
            )
            self._pending_recovery_steps = max(
                self._pending_recovery_steps, self.recovery_horizon
            )
        if preemptive_trigger:
            self._preemptive_trigger_count += 1
        if output is not None:
            return output
        if self._recovery_mode == "head_island":
            return self._head_island_output(
                query,
                key,
                value,
                draft=draft,
                exact_sample=exact_sample,
                sample_indices=indices,
            )
        self._dense_recovery_calls += 1
        return self.dense_backend(query, key, value)

    _CHECKPOINT_COUNTERS = (
        "_dense_recovery_calls",
        "_dense_detail_calls",
        "_dense_hysteresis_calls",
        "_graded_recovery_calls",
        "_graded_hysteresis_calls",
        "_graded_accept_count",
        "_graded_reject_count",
        "_verified_query_injection_calls",
        "_verified_query_injection_tokens",
        "_head_repair_calls",
        "_head_repair_heads",
        "_preemptive_trigger_count",
        "_probe_first_calls",
        "_probe_first_rejected_full_drafts",
        "_shared_kv_exact_probe_calls",
        "_head_island_trigger_count",
        "_head_island_calls",
        "_head_island_heads",
        "_head_island_total_heads",
        "_head_island_reverify_calls",
        "_online_budget_denied_count",
        "_reserve_rebate_calls",
    )

    def online_checkpoint_state(
        self,
        ledger: AttentionOnlineBudget,
    ) -> dict[str, object]:
        """Persist request-local verifier state without serializing model state."""

        if ledger.policy_id not in self.online_guard_ids:
            raise ValueError("verifier checkpoint ledger uses an unsupported policy")
        phase_growth_state = None
        if (
            self._phase_growth_guard is not None
            and ledger.policy_id in self.phase_growth_guard_ids
            and self._phase_growth_guard.has_request(ledger)
        ):
            phase_growth_state = self._phase_growth_guard.checkpoint_state(ledger)
        return {
            "schema_version": "h3_attention_verifier_checkpoint_v1",
            "policy_id": ledger.policy_id,
            "phase_growth_state": phase_growth_state,
            "request_had_trigger": self._request_had_trigger,
            "probe_growth_upper": self._probe_growth_upper,
            "probe_growth_pairs": self._probe_growth_pairs,
            "pending_recovery_steps": self._pending_recovery_steps,
            "pending_recovery_mode": self._pending_recovery_mode,
            "pending_recovery_heads": (
                None
                if self._pending_recovery_heads is None
                else self._pending_recovery_heads.detach().cpu()
            ),
            "pending_recovery_target_relative_rms": (
                self._pending_recovery_target_relative_rms
            ),
            "phase_probe_slots": [list(slot) for slot in self._last_phase_probe_slots],
            "probe_records": list(self._probe_records),
            "head_island_records": list(self._head_island_records),
            "counters": {
                name.removeprefix("_"): int(getattr(self, name))
                for name in self._CHECKPOINT_COUNTERS
            },
        }

    def restore_online_checkpoint_state(
        self,
        ledger: AttentionOnlineBudget,
        state: object,
    ) -> None:
        """Restore one request verifier state and bind it to ``ledger``."""

        if not isinstance(state, dict):
            raise ValueError("attention verifier checkpoint state must be an object")
        if state.get("schema_version") != "h3_attention_verifier_checkpoint_v1":
            raise ValueError("unexpected attention verifier checkpoint schema")
        if state.get("policy_id") != ledger.policy_id:
            raise ValueError("attention verifier checkpoint policy mismatch")
        if ledger.policy_id not in self.online_guard_ids:
            raise ValueError("attention verifier checkpoint policy is unsupported")
        phase_growth_state = state.get("phase_growth_state")
        if (
            self._phase_growth_guard is not None
            and ledger.policy_id in self.phase_growth_guard_ids
        ):
            if phase_growth_state is None:
                self._phase_growth_guard.begin_request(ledger)
            else:
                self._phase_growth_guard.restore_checkpoint_state(
                    ledger, phase_growth_state
                )
        elif phase_growth_state is not None:
            raise ValueError("unexpected phase-growth checkpoint state")
        request_had_trigger = state.get("request_had_trigger")
        if not isinstance(request_had_trigger, bool):
            raise ValueError("invalid attention verifier trigger state")
        probe_growth_upper_raw = state.get("probe_growth_upper")
        probe_growth_upper = (
            None
            if probe_growth_upper_raw is None
            else float(probe_growth_upper_raw)
        )
        if probe_growth_upper is not None and (
            not math.isfinite(probe_growth_upper) or probe_growth_upper < 0.0
        ):
            raise ValueError("invalid attention verifier growth envelope")
        probe_growth_pairs = int(state.get("probe_growth_pairs", -1))
        pending_steps = int(state.get("pending_recovery_steps", -1))
        pending_mode = state.get("pending_recovery_mode")
        pending_target = float(
            state.get("pending_recovery_target_relative_rms", math.nan)
        )
        if probe_growth_pairs < 0 or pending_steps < 0:
            raise ValueError("invalid attention verifier checkpoint counts")
        if pending_mode not in {"dense", "graded", "head_island"}:
            raise ValueError("invalid attention verifier recovery mode")
        if not math.isfinite(pending_target) or pending_target < 0.0:
            raise ValueError("invalid attention verifier recovery target")
        pending_heads = state.get("pending_recovery_heads")
        if pending_heads is not None and (
            not isinstance(pending_heads, torch.Tensor)
            or pending_heads.ndim != 1
            or pending_heads.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("invalid attention verifier recovery heads")
        raw_slots = state.get("phase_probe_slots")
        raw_probe_records = state.get("probe_records")
        raw_head_records = state.get("head_island_records")
        raw_counters = state.get("counters")
        if not isinstance(raw_slots, list):
            raise ValueError("attention verifier checkpoint slots are missing")
        slots = tuple((int(step), int(layer)) for step, layer in raw_slots)
        if tuple(sorted(set(slots))) != slots:
            raise ValueError("attention verifier checkpoint slots are invalid")
        if not isinstance(raw_probe_records, list) or not all(
            isinstance(record, dict) for record in raw_probe_records
        ):
            raise ValueError("attention verifier checkpoint probes are invalid")
        if not isinstance(raw_head_records, list) or not all(
            isinstance(record, dict) for record in raw_head_records
        ):
            raise ValueError("attention verifier checkpoint head records are invalid")
        if not isinstance(raw_counters, dict):
            raise ValueError("attention verifier checkpoint counters are missing")
        counter_values: dict[str, int] = {}
        for name in self._CHECKPOINT_COUNTERS:
            value = int(raw_counters.get(name.removeprefix("_"), -1))
            if value < 0:
                raise ValueError("invalid attention verifier checkpoint counter")
            counter_values[name] = value

        self._active_step = None
        self._first_probe_error = None
        self._recover_step = False
        self._recovery_mode = "dense"
        self._recovery_heads = None
        self._hysteresis_active = False
        self._recovery_consumed_step = None
        self._request_had_trigger = request_had_trigger
        self._probe_growth_upper = probe_growth_upper
        self._probe_growth_pairs = probe_growth_pairs
        self._pending_recovery_steps = pending_steps
        self._pending_recovery_mode = str(pending_mode)
        self._pending_recovery_heads = (
            None if pending_heads is None else pending_heads.detach().clone()
        )
        self._pending_recovery_target_relative_rms = pending_target
        self._last_online_guard_id = ledger.policy_id
        self._last_phase_probe_slots = slots
        self._probe_records = list(raw_probe_records)
        self._head_island_records = list(raw_head_records)
        for name, value in counter_values.items():
            setattr(self, name, value)

    def telemetry(self) -> dict[str, object]:
        draft_telemetry = getattr(self.draft_backend, "telemetry", None)
        return {
            "mode": "causal_checkpoint_original_attention_verifier",
            "online_guard_id": self._last_online_guard_id or self.online_guard_id,
            "supported_online_guard_ids": sorted(self.online_guard_ids),
            "phase_probe_guard_ids": sorted(self.phase_probe_guard_ids),
            "phase_growth_guard_ids": sorted(self.phase_growth_guard_ids),
            "reserve_rebate_guard_ids": sorted(self.reserve_rebate_guard_ids),
            "phase_growth_threshold": (
                self._phase_growth_guard.threshold
                if self._phase_growth_guard is not None
                else None
            ),
            "phase_probe_slots": [list(slot) for slot in self._last_phase_probe_slots],
            "online_budget_denied_count": self._online_budget_denied_count,
            "request_had_trigger": self._request_had_trigger,
            "reserve_rebate_calls": self._reserve_rebate_calls,
            "relative_rms_threshold": self.relative_rms_threshold,
            "verification_query_blocks": self.verification_query_blocks,
            "probe_records": list(self._probe_records),
            "dense_recovery_calls": self._dense_recovery_calls,
            "dense_detail_calls": self._dense_detail_calls,
            "dense_hysteresis_calls": self._dense_hysteresis_calls,
            "graded_recovery_enabled": self.recovery_backend is not None,
            "graded_recovery_calls": self._graded_recovery_calls,
            "graded_hysteresis_calls": self._graded_hysteresis_calls,
            "graded_accept_count": self._graded_accept_count,
            "graded_reject_count": self._graded_reject_count,
            "recovery_horizon": self.recovery_horizon,
            "hysteresis_layers": sorted(self.hysteresis_layers),
            "online_probe_growth_prediction": self.online_probe_growth_prediction,
            "probe_first_short_circuit": self.probe_first_short_circuit,
            "probe_first_calls": self._probe_first_calls,
            "probe_first_rejected_full_drafts": (
                self._probe_first_rejected_full_drafts
            ),
            "shared_kv_exact_probe": self.shared_kv_exact_probe,
            "shared_kv_exact_probe_calls": self._shared_kv_exact_probe_calls,
            "causal_head_island": self.causal_head_island,
            "head_island_trigger_count": self._head_island_trigger_count,
            "head_island_calls": self._head_island_calls,
            "head_island_heads": self._head_island_heads,
            "head_island_total_heads": self._head_island_total_heads,
            "head_island_dense_fraction": (
                self._head_island_heads / self._head_island_total_heads
                if self._head_island_total_heads
                else 0.0
            ),
            "head_island_reverify_calls": self._head_island_reverify_calls,
            "head_island_records": list(self._head_island_records),
            "online_probe_growth_upper": self._probe_growth_upper,
            "online_probe_growth_pairs": self._probe_growth_pairs,
            "preemptive_trigger_count": self._preemptive_trigger_count,
            "inject_verified_queries": self.inject_verified_queries,
            "verified_query_injection_calls": self._verified_query_injection_calls,
            "verified_query_injection_tokens": self._verified_query_injection_tokens,
            "repair_high_error_heads": self.repair_high_error_heads,
            "head_error_mass_coverage": self.head_error_mass_coverage,
            "head_repair_activation_ratio": self.head_repair_activation_ratio,
            "head_repair_calls": self._head_repair_calls,
            "head_repair_heads": self._head_repair_heads,
            "draft": draft_telemetry() if draft_telemetry is not None else None,
        }


class RequestRoutedSpargeAttentionBackend:
    """Route complete requests to dense Sage or one Sparge budget."""

    approximate = True
    request_routed = True

    def __init__(self, *, minimum_sparse_tokens: int = 128) -> None:
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        topk = _ATTENTION_SPARSE_TOPK.get()
        if topk is None or query.shape[0] < self.minimum_sparse_tokens:
            return sage_attention_sm89(query, key, value)
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

        output = spas_sage2_attn_meansim_topk_cuda(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            topk=topk,
            is_causal=False,
            tensor_layout="NHD",
            return_sparsity=False,
        )
        return output.squeeze(0)

    def resolve_long_sequence_backend(self, query_tokens: int):
        """Keep the legacy Dense request route inside bounded Query memory.

        V24 normally installs a per-cell action schedule.  A fail-closed Dense
        selection intentionally leaves that schedule empty and reaches this
        legacy request router.  Returning the prepared Dense backend here is
        essential: otherwise a low-VRAM plan silently falls back to whole-QKV
        SageAttention precisely on the quality-protection path.

        The legacy sparse branch is not claimed here because its monolithic
        operator has different protected-prefix semantics; scheduled sparse
        actions already resolve through their validated physical backends.
        """

        if query_tokens < self.minimum_sparse_tokens:
            return None
        if _ATTENTION_SPARSE_TOPK.get() is not None:
            return None
        return (
            _DENSE_LONG_SEQUENCE_BACKEND
            if _DENSE_QK_QUANT_GRAN.get() == "per_warp"
            else None
        )


class ModalityProtectedSpargeAttentionBackend:
    """Sparsify video-to-video attention while preserving conditioning rows.

    H3 packs ``[text | optional frame conditions | audio | video]``.  Every
    protected-prefix query retains all keys and every video query retains all
    prefix keys; only the dominant video-to-video quadrant is sparsified.  The
    backend is experimental because that remaining approximation still needs
    generated-video and Human review.
    """

    approximate = True

    def __init__(self, topk: float, *, minimum_sparse_tokens: int = 128) -> None:
        if not 0.5 <= topk <= 1.0:
            raise ValueError("SpargeAttention topk must be between 0.5 and 1.0")
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        self.topk = float(topk)
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        protected_tokens = _ATTENTION_PROTECTED_PREFIX.get()
        if query.shape[0] < self.minimum_sparse_tokens or protected_tokens <= 0:
            return sage_attention_sm89(query, key, value)
        if protected_tokens >= query.shape[0]:
            return sage_attention_sm89(query, key, value)

        from einops import rearrange
        from spas_sage_attn import core as sparge_core
        from spas_sage_attn.utils import (
            block_map_lut_triton,
            get_block_map_meansim_fuse_quant,
            hyperparameter_check,
        )

        q, k, v = map(
            lambda tensor: rearrange(tensor, "L H D -> H L D").unsqueeze(0),
            (query, key, value),
        )
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.float16)
        key_mean = k.mean(dim=-2, keepdim=True)
        block_map, q_int8, q_scale, k_int8, k_scale = (
            get_block_map_meansim_fuse_quant(
                q,
                k,
                key_mean,
                BLKQ=128,
                BLKK=64,
                simthreshd1=-0.1,
                cdfthreshd=None,
                topk=self.topk,
                is_causal=False,
            )
        )
        protected_q_blocks = (protected_tokens + 127) // 128
        protected_k_blocks = (protected_tokens + 63) // 64
        block_map[:, :, :protected_q_blocks, :] = True
        block_map[:, :, :, :protected_k_blocks] = True
        lut, valid_block_num = block_map_lut_triton(block_map.contiguous())
        output = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, query.shape[1], query.device)
        sparge_core.qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8,
            k_int8,
            v,
            output,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            k_scale,
            1,
            0,
            1,
            1.0 / (query.shape[-1] ** 0.5),
            0,
        )
        return rearrange(output, "1 H L D -> L H D")


_H3_HEAD_RISK_TIERS = (
    1, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 0, 1, 2,
    1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 0,
    0, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0,
    0, 2, 2, 0, 1, 2, 0, 1, 1, 0, 2, 2, 2, 0,
)


@dataclass(slots=True)
class PreparedLongSequenceKV:
    """One layer's reusable SM89 K/V state for Query-streamed Attention.

    The object intentionally owns no unquantized V tensor.  The caller drops
    that 2.94-GiB 1080p/15s allocation before the sparse K pool/quant pass is
    launched, preventing two individually safe preparation phases from
    overlapping into an allocator oversubscription event.
    """

    key: torch.Tensor | None
    key_mean: torch.Tensor
    pooled_key: torch.Tensor
    similar_key_blocks: torch.Tensor
    key_int8: torch.Tensor
    key_scale: torch.Tensor
    prefix_key_int8: torch.Tensor | None
    prefix_key_scale: torch.Tensor | None
    value_fp8: torch.Tensor
    value_scale: torch.Tensor
    heads: int
    key_tokens: int
    head_dim: int
    tensor_layout: str = "HND"


@dataclass(slots=True)
class PreparedLongSequenceDenseKV:
    """Reusable exact Sage SM89 K/V state for Query-streamed Dense cells."""

    key_int8: torch.Tensor
    key_scale: torch.Tensor
    value_fp8: torch.Tensor
    value_scale: torch.Tensor
    heads: int
    key_tokens: int
    head_dim: int


def _write_compact_value_fp8(
    target: torch.Tensor,
    value: torch.Tensor,
    value_absmax: torch.Tensor,
    *,
    start: int,
    layout: str,
) -> None:
    """Quantize one NHD V slab into Sage/Sparge's permuted FP8 storage."""

    if value.ndim != 3 or value.shape[-1] != 128:
        raise ValueError("compact V slabs must use [tokens,heads,128]")
    if start < 0 or start % 16:
        raise ValueError("compact V slab offsets must align to 16 tokens")
    heads = int(value.shape[1])
    head_dim = int(value.shape[2])
    if value_absmax.shape != (heads, head_dim):
        raise ValueError("compact V absmax must use [heads,head_dim]")
    # ``torch.split`` over the fused [K|V] projection retains the original
    # 2*inner_width row stride, so the V view is normally non-contiguous even
    # though each logical slab is compact.  Materialize only this bounded
    # projection slab (at most ``projection_chunk_tokens`` rows) before the
    # Triton writer.  Requiring the caller to make the entire K/V context
    # contiguous would defeat the long-sequence memory contract.
    if not value.is_contiguous():
        value = value.contiguous()
    from .compact_fp8 import write_sage_fp8_slab

    write_sage_fp8_slab(
        target,
        value,
        value_absmax,
        start=start,
        layout=layout,
        division_mode=2,
    )


class _CompactValueBuilder:
    """Incrementally quantize V while retaining the accepted full-K path."""

    def __init__(
        self,
        *,
        key_tokens: int,
        heads: int,
        head_dim: int,
        value_absmax: torch.Tensor,
        device: torch.device,
        layout: str,
    ) -> None:
        self.key_tokens = int(key_tokens)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.layout = layout
        self.value_absmax = value_absmax.float().contiguous()
        padding = 64 if layout == "NHD" else 128
        padded_tokens = (key_tokens + padding - 1) // padding * padding
        shape = (
            (1, head_dim, heads, padded_tokens)
            if layout == "NHD"
            else (1, heads, head_dim, padded_tokens)
        )
        self.value_fp8 = torch.zeros(
            shape, device=device, dtype=torch.float8_e4m3fn
        )
        self.value_scale = (
            value_absmax.float().reshape(1, heads, head_dim) / 2.25
        ).clamp_min_(1.0e-12)

    def add(self, start: int, value: torch.Tensor) -> None:
        if value.shape[1:] != (self.heads, self.head_dim):
            raise ValueError("compact V slab shape mismatch")
        if start % 16 or start + int(value.shape[0]) > self.key_tokens:
            raise ValueError("compact V slab boundary is invalid")
        _write_compact_value_fp8(
            self.value_fp8,
            value,
            self.value_absmax,
            start=start,
            layout=self.layout,
        )

    def finish(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        return (
            self.value_fp8,
            self.value_scale,
            self.heads,
            self.key_tokens,
            self.head_dim,
        )


class _CompactDenseKVBuilder:
    """Incrementally construct NHD Sage K/V without full BF16 K or V."""

    def __init__(
        self,
        *,
        key_tokens: int,
        heads: int,
        head_dim: int,
        key_mean: torch.Tensor,
        value_absmax: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.key_tokens = int(key_tokens)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.key_mean = key_mean.to(torch.bfloat16).reshape(
            1, 1, heads, head_dim
        )
        self.key_int8 = torch.empty(
            (1, key_tokens, heads, head_dim),
            device=device,
            dtype=torch.int8,
        )
        self.key_scale = torch.empty(
            (1, heads, (key_tokens + 63) // 64),
            device=device,
            dtype=torch.float32,
        )
        padded_tokens = (key_tokens + 63) // 64 * 64
        self.value_fp8 = torch.zeros(
            (1, head_dim, heads, padded_tokens),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self.value_scale = (
            value_absmax.float().reshape(1, heads, head_dim) / 2.25
        ).clamp_min_(1.0e-12)
        self.value_absmax = value_absmax.float().contiguous()

    def add(self, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        from sageattention import _fused

        rows = int(key.shape[0])
        if key.shape != value.shape or key.shape[1:] != (
            self.heads,
            self.head_dim,
        ):
            raise ValueError("compact Dense K/V slab shape mismatch")
        if start % 64 or start + rows > self.key_tokens:
            raise ValueError("compact Dense K/V slab boundary is invalid")
        key_nhd = key.unsqueeze(0).contiguous()
        chunk_int8 = torch.empty_like(key_nhd, dtype=torch.int8)
        chunk_scale = torch.empty(
            (1, self.heads, (rows + 63) // 64),
            device=key.device,
            dtype=torch.float32,
        )
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(
            key_nhd,
            self.key_mean.squeeze(1),
            chunk_int8,
            chunk_scale,
            64,
            0,
        )
        self.key_int8[:, start : start + rows].copy_(chunk_int8)
        block_start = start // 64
        self.key_scale[
            :, :, block_start : block_start + chunk_scale.shape[-1]
        ].copy_(chunk_scale)
        _write_compact_value_fp8(
            self.value_fp8,
            value,
            self.value_absmax,
            start=start,
            layout="NHD",
        )

    def finish(self) -> PreparedLongSequenceDenseKV:
        return PreparedLongSequenceDenseKV(
            key_int8=self.key_int8,
            key_scale=self.key_scale,
            value_fp8=self.value_fp8,
            value_scale=self.value_scale,
            heads=self.heads,
            key_tokens=self.key_tokens,
            head_dim=self.head_dim,
        )


class _CompactSparseKVBuilder:
    """Incrementally construct HND Sparge K/V and discard BF16 slabs."""

    def __init__(
        self,
        *,
        key_tokens: int,
        heads: int,
        head_dim: int,
        key_mean: torch.Tensor,
        value_absmax: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.key_tokens = int(key_tokens)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.key_mean = key_mean.to(torch.bfloat16).reshape(
            1, heads, 1, head_dim
        )
        key_blocks = (key_tokens + 63) // 64
        self.pooled_key = torch.empty(
            (1, heads, key_blocks, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        self.similar_key_blocks = torch.empty(
            (1, heads, key_blocks), device=device, dtype=torch.bool
        )
        self.key_int8 = torch.empty(
            (1, heads, key_tokens, head_dim), device=device, dtype=torch.int8
        )
        self.key_scale = torch.empty(
            (1, heads, key_blocks), device=device, dtype=torch.float32
        )
        padded_tokens = (key_tokens + 127) // 128 * 128
        self.value_fp8 = torch.zeros(
            (1, heads, head_dim, padded_tokens),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self.value_scale = (
            value_absmax.float().reshape(1, heads, head_dim) / 2.25
        ).clamp_min_(1.0e-12)
        self.value_absmax = value_absmax.float().contiguous()

    def add(self, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        from spas_sage_attn.utils import (
            get_pool_sim_triton_simmean_fuse_quant,
            hyperparameter_check,
        )

        rows = int(key.shape[0])
        if key.shape != value.shape or key.shape[1:] != (
            self.heads,
            self.head_dim,
        ):
            raise ValueError("compact Sparse K/V slab shape mismatch")
        if start % 128 or start + rows > self.key_tokens:
            raise ValueError("compact Sparse K/V slab boundary is invalid")
        key_hnd = key.permute(1, 0, 2).unsqueeze(0).contiguous()
        sim_threshold = hyperparameter_check(-0.1, self.heads, key.device)
        pooled, similar, quantized, scale = (
            get_pool_sim_triton_simmean_fuse_quant(
                key_hnd, self.key_mean, 64, sim_threshold
            )
        )
        self.key_int8[:, :, start : start + rows].copy_(quantized)
        block_start = start // 64
        block_stop = block_start + int(scale.shape[-1])
        self.key_scale[:, :, block_start:block_stop].copy_(scale)
        self.pooled_key[:, :, block_start:block_stop].copy_(pooled)
        self.similar_key_blocks[:, :, block_start:block_stop].copy_(similar)
        _write_compact_value_fp8(
            self.value_fp8,
            value,
            self.value_absmax,
            start=start,
            layout="HND",
        )

    def finish(self) -> PreparedLongSequenceKV:
        return PreparedLongSequenceKV(
            key=None,
            key_mean=self.key_mean,
            pooled_key=self.pooled_key,
            similar_key_blocks=self.similar_key_blocks,
            key_int8=self.key_int8,
            key_scale=self.key_scale,
            prefix_key_int8=None,
            prefix_key_scale=None,
            value_fp8=self.value_fp8,
            value_scale=self.value_scale,
            heads=self.heads,
            key_tokens=self.key_tokens,
            head_dim=self.head_dim,
        )


class SplitModalityProtectedSpargeAttentionBackend:
    """Run packed conditioning queries dense and video queries sparse.

    Splitting the two query regions avoids widening every sparse-query LUT for
    the small fully-connected conditioning prefix.  Prefix queries use the
    accepted dense SageAttention2++ path over every key.  Video queries use
    SpargeAttention2++ while retaining every text/condition/audio key block.
    """

    approximate = True
    long_sequence_value_dtype = torch.float16
    supports_direct_nhd_kv = True

    def __init__(
        self,
        topk: float | tuple[float, ...],
        *,
        minimum_sparse_tokens: int = 128,
        experimental_minimum_topk: float = 0.5,
        temporal_correspondence_radius: int = -1,
        temporal_spatial_block_radius: int = 0,
        temporal_global_anchor_stride: int = 0,
        temporal_global_spatial_block_radius: int = 0,
        selection_mode: str = "fixed_topk",
        maximum_selected_key_blocks: int | tuple[int, ...] | None = None,
        minimum_retained_topk_mass: float = 0.95,
        mass_probe_selected_key_blocks: tuple[int, ...] | None = None,
        adaptive_safety_margin: float = 0.65,
        route_probe: bool = False,
        parallel_long_sequence_lut: bool = False,
        partial_long_sequence_topk: bool = False,
        fused_long_sequence_prefix_k_quant: bool = False,
    ) -> None:
        budgets = (float(topk),) if isinstance(topk, (float, int)) else tuple(topk)
        if not 0.0625 <= experimental_minimum_topk <= 0.5:
            raise ValueError("experimental minimum topk must lie inside [0.0625, 0.5]")
        if not budgets or any(
            not experimental_minimum_topk <= value <= 1.0 for value in budgets
        ):
            raise ValueError(
                "SpargeAttention topk lies outside the configured experimental range"
            )
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        if temporal_correspondence_radius < -1:
            raise ValueError("temporal correspondence radius must be >= -1")
        if temporal_spatial_block_radius < 0:
            raise ValueError("temporal spatial block radius cannot be negative")
        if temporal_global_anchor_stride not in (0,) and temporal_global_anchor_stride < 2:
            raise ValueError("temporal global anchor stride must be 0 or >= 2")
        if temporal_global_spatial_block_radius < 0:
            raise ValueError("temporal global spatial block radius cannot be negative")
        if not 0.0 <= adaptive_safety_margin <= 1.0:
            raise ValueError("adaptive safety margin must lie inside [0, 1]")
        if selection_mode not in (
            "fixed_topk",
            "fixed_topk_absolute_cap",
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
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
            "budget_adaptive",
            "unified_fixed_topk",
        ):
            raise ValueError(
                "selection_mode must be fixed_topk, fixed_topk_absolute_cap, "
                "fixed_topk_mass_guarded_cap, "
                "fixed_topk_mass_probe, "
                "mass_budget, mass_rebate, "
                "route_cache, temporal_motion_guard, interaction_guard, "
                "interaction_rail, interaction_recovery, "
                "interaction_rebalance, interaction_hybrid, "
                "interaction_dense, disagreement_sentinel, "
                "causal_head_guard, budget_adaptive or "
                "unified_fixed_topk"
            )
        if maximum_selected_key_blocks is None:
            maximum_block_counts = None
        elif isinstance(maximum_selected_key_blocks, int) and not isinstance(
            maximum_selected_key_blocks, bool
        ):
            maximum_block_counts = (int(maximum_selected_key_blocks),)
        elif isinstance(maximum_selected_key_blocks, tuple):
            maximum_block_counts = tuple(maximum_selected_key_blocks)
        else:
            raise ValueError(
                "maximum selected key blocks must be an integer or tuple"
            )
        if maximum_block_counts is not None and (
            not maximum_block_counts
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in maximum_block_counts
            )
        ):
            raise ValueError(
                "maximum selected key blocks must contain positive integers"
            )
        if not 0.0 < minimum_retained_topk_mass <= 1.0:
            raise ValueError("minimum retained Top-K mass must lie inside (0, 1]")
        probe_counts = (
            ()
            if mass_probe_selected_key_blocks is None
            else tuple(mass_probe_selected_key_blocks)
        )
        if probe_counts and (
            tuple(sorted(set(probe_counts))) != probe_counts
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in probe_counts
            )
        ):
            raise ValueError(
                "mass-probe selected key blocks must be sorted unique positive integers"
            )
        if probe_counts and selection_mode != "fixed_topk_mass_probe":
            raise ValueError(
                "mass-probe selected key blocks require fixed_topk_mass_probe"
            )
        if probe_counts and len(budgets) != 1:
            raise ValueError("mass-probe cap ladder currently requires scalar Top-K")
        absolute_cap_modes = (
            "fixed_topk_absolute_cap",
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
        )
        if selection_mode in absolute_cap_modes:
            if maximum_block_counts is None:
                raise ValueError(
                    f"{selection_mode} requires maximum selected key blocks"
                )
        elif maximum_block_counts is not None:
            raise ValueError(
                "maximum selected key blocks require an absolute-cap selector"
            )
        self.topk = budgets[0] if len(budgets) == 1 else budgets
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)
        self.temporal_correspondence_radius = int(temporal_correspondence_radius)
        self.temporal_spatial_block_radius = int(temporal_spatial_block_radius)
        self.temporal_global_anchor_stride = int(temporal_global_anchor_stride)
        self.temporal_global_spatial_block_radius = int(
            temporal_global_spatial_block_radius
        )
        self.selection_mode = selection_mode
        self.maximum_selected_key_blocks = (
            maximum_block_counts[0]
            if maximum_block_counts is not None and len(maximum_block_counts) == 1
            else maximum_block_counts
        )
        self.minimum_retained_topk_mass = float(minimum_retained_topk_mass)
        self.mass_probe_selected_key_blocks = probe_counts
        self.adaptive_safety_margin = float(adaptive_safety_margin)
        self.experimental_minimum_topk = float(experimental_minimum_topk)
        self.route_probe = bool(route_probe)
        self.parallel_long_sequence_lut = bool(parallel_long_sequence_lut)
        self.partial_long_sequence_topk = bool(partial_long_sequence_topk)
        self.fused_long_sequence_prefix_k_quant = bool(
            fused_long_sequence_prefix_k_quant
        )
        if self.route_probe and self.selection_mode != "fixed_topk":
            raise ValueError("route probe requires the fixed_topk selector")
        # Research-only sparse-route metadata cache. It never reuses Attention
        # values, projections, MLP outputs or hidden residuals. Entries are
        # scoped by true H3 layer and shape; two consecutive maps must first
        # demonstrate >=90% sampled Jaccard overlap before reuse is allowed.
        self._route_cache: dict[tuple[int, int, int, int], dict[str, object]] = {}
        self._route_cache_hits = 0
        self._route_cache_misses = 0
        self._route_cache_rejected = 0
        # Research-only recorder for the final, rail-protected sparse map. It
        # never changes the LUT consumed by Attention. Reductions remain on
        # device during denoising and are materialized only for telemetry, so
        # no CPU synchronization is introduced in each H3 layer.
        self._route_probe_previous: dict[
            tuple[int, int, int, int], tuple[int, torch.Tensor]
        ] = {}
        self._route_probe_records: list[dict[str, object]] = []
        self._sentinel_calls = 0
        self._sentinel_dense_query_tokens = 0
        self._sentinel_total_query_tokens = 0
        self._causal_head_guard_calls = 0
        self._causal_head_guard_dense_heads = 0
        self._causal_head_guard_total_heads = 0
        # Research-only absolute-cap evidence.  The tensors are tiny views of
        # the most recent prepared-KV call and are materialized only when
        # telemetry is requested, keeping the timed CUDA path free of host
        # synchronization.  This mode is a distinct physical action and does
        # not alter the accepted fixed_topk implementation.
        self._absolute_cap_calls = 0
        self._absolute_cap_last: dict[str, object] | None = None
        self._mass_guard_records: list[dict[str, object]] = []

    @staticmethod
    def _project_counts_to_exact_budget(
        desired: torch.Tensor,
        target_total: torch.Tensor,
        *,
        minimum: int,
        maximum: int,
    ) -> torch.Tensor:
        """Project head/query counts onto an exact request-local block quota."""

        if desired.ndim != 3:
            raise ValueError("adaptive counts must have [batch, head, query] shape")
        if target_total.shape != (desired.shape[0],):
            raise ValueError("adaptive target total must have one value per batch")
        if minimum < 1 or maximum < minimum:
            raise ValueError("invalid adaptive count bounds")

        flattened = desired.float().flatten(1)
        entries = flattened.shape[1]
        target = target_total.to(device=desired.device, dtype=torch.int64)
        target = target.clamp(minimum * entries, maximum * entries)
        finite = torch.nan_to_num(
            flattened,
            nan=float(minimum),
            posinf=float(maximum),
            neginf=float(minimum),
        ).clamp_min(1e-6)

        # Solve one continuous water-filling scale per batch item.  All
        # operations remain on device; no per-head host synchronization is
        # introduced in the long-sequence path.
        low = torch.zeros((desired.shape[0], 1), device=desired.device)
        high = torch.full_like(low, float(maximum) / float(minimum) * 4.0)
        target_float = target[:, None].float()
        for _ in range(24):
            scale = (low + high) * 0.5
            total = (finite * scale).clamp(minimum, maximum).sum(1, keepdim=True)
            low = torch.where(total < target_float, scale, low)
            high = torch.where(total >= target_float, scale, high)

        projected = (finite * high).clamp(minimum, maximum)
        integer = projected.floor().to(torch.int64)
        deficit = target - integer.sum(1)
        fraction = projected - integer.float()
        fraction.masked_fill_(integer >= maximum, -1.0)
        order = fraction.argsort(dim=1, descending=True)
        rank = order.argsort(dim=1)
        integer += rank.lt(deficit[:, None]).to(torch.int64)
        return integer.reshape_as(desired)

    @classmethod
    def _budget_adaptive_counts(
        cls,
        *,
        base_count: torch.Tensor,
        mass_count: torch.Tensor,
        high_risk: torch.Tensor | None,
        safety_margin: float,
        minimum: int,
        maximum: int,
    ) -> torch.Tensor:
        """Spend one fixed quota on the current request's hardest Attention rows.

        Attention mass, a request-local interaction signal and weak historical
        H3 head priors only decide *where* the work goes.  Exact projection
        prevents those proxies from silently increasing total work.  Prefix
        and MTCR protection are applied later as a non-negotiable safety floor.
        """

        if not 0.0 <= safety_margin <= 1.0:
            raise ValueError("adaptive safety margin must lie inside [0, 1]")
        if mass_count.ndim != 3 or base_count.ndim != 1:
            raise ValueError("adaptive mass/base count shapes are invalid")
        if mass_count.shape[1] != base_count.numel():
            raise ValueError("adaptive mass counts do not match head count")
        base = base_count.view(1, -1, 1).float()
        desired = base + safety_margin * (mass_count.float() - base)

        if base_count.numel() == len(_H3_HEAD_RISK_TIERS):
            tiers = torch.tensor(
                _H3_HEAD_RISK_TIERS,
                device=mass_count.device,
                dtype=torch.float32,
            )
            priority = 1.0 + safety_margin * 0.10 * (tiers - tiers.mean())
            desired = desired * priority.view(1, -1, 1)
        if high_risk is not None:
            if high_risk.shape != mass_count.shape:
                raise ValueError("adaptive interaction mask shape is invalid")
            desired = desired * (1.0 + 0.75 * safety_margin * high_risk.float())

        target_total = (
            base_count.to(torch.int64).sum() * mass_count.shape[-1]
        ).expand(mass_count.shape[0])
        return cls._project_counts_to_exact_budget(
            desired,
            target_total,
            minimum=minimum,
            maximum=maximum,
        )

    @staticmethod
    def _causal_head_dense_mask(probability: torch.Tensor) -> torch.Tensor:
        """Select diffuse relation-carrying heads with a robust local rule.

        Dense teacher probes across door interaction, vehicle occlusion and
        two-person handoff scenes show that proxy-attention entropy predicts
        sparse-output error substantially better than local motion magnitude.
        The decision is intentionally computed from the current layer/request:
        heads above ``median + MAD`` return to full KV coverage, while the
        remaining heads keep the caller's accepted sparse budget.

        Non-finite proxy distributions fail closed for the affected head.  A
        degenerate finite distribution protects the single most diffuse head,
        which keeps the contract useful without introducing a scene-specific
        ratio or a user-facing threshold.
        """

        if probability.ndim != 4:
            raise ValueError("causal head guard expects [batch, head, query, key]")
        key_blocks = int(probability.shape[-1])
        if key_blocks <= 1:
            return torch.ones(
                probability.shape[1], device=probability.device, dtype=torch.bool
            )
        safe_probability = probability.float().clamp_min(1e-12)
        entropy = -(
            safe_probability * safe_probability.log()
        ).sum(dim=-1) / math.log(float(key_blocks))
        head_entropy = entropy.mean(dim=(0, 2))
        nonfinite = ~torch.isfinite(head_entropy)
        finite_entropy = torch.where(
            nonfinite, torch.zeros_like(head_entropy), head_entropy
        )
        median = finite_entropy.median()
        mad = (finite_entropy - median).abs().median()
        selected = finite_entropy > median + mad
        selected |= nonfinite
        if not bool(selected.any()):
            selected[finite_entropy.argmax()] = True
        return selected

    def _temporal_motion_guard(
        self,
        pooled_q: torch.Tensor,
        *,
        query_tokens: int,
    ) -> torch.Tensor | None:
        """Find coherent high-motion query blocks without a user threshold.

        The video token stream is frame-major, while SpargeAttention groups
        queries in physical 128-row blocks.  Compare every pooled query block
        with the blocks containing the same spatial centre in its adjacent
        latent frames.  A robust per-head median/MAD boundary selects only
        temporal outliers and a one-block halo protects object boundaries.

        Early noisy solver states have no reliable temporal correspondence.
        They fail closed by returning ``None``, which preserves the caller's
        full configured recovery budget for every query block.
        """

        import torch.nn.functional as functional

        layout = _ATTENTION_VIDEO_LAYOUT.get()
        if layout is None or pooled_q.ndim != 4:
            return None
        latent_frames, frame_tokens = layout
        if query_tokens != latent_frames * frame_tokens or latent_frames < 3:
            return None
        query_blocks = int(pooled_q.shape[-2])
        if query_blocks < 3:
            return None

        device = pooled_q.device
        centres = torch.arange(query_blocks, device=device, dtype=torch.int64)
        centres = (centres * 128 + 64).clamp_max(query_tokens - 1)
        frames = torch.div(centres, frame_tokens, rounding_mode="floor")
        spatial = torch.remainder(centres, frame_tokens)
        previous_tokens = (frames - 1).clamp_min(0) * frame_tokens + spatial
        next_tokens = (frames + 1).clamp_max(latent_frames - 1) * frame_tokens + spatial
        previous_blocks = torch.div(previous_tokens, 128, rounding_mode="floor")
        next_blocks = torch.div(next_tokens, 128, rounding_mode="floor")
        previous_blocks.clamp_max_(query_blocks - 1)
        next_blocks.clamp_max_(query_blocks - 1)

        vectors = functional.normalize(pooled_q.float(), dim=-1, eps=1e-6)
        previous = vectors.index_select(-2, previous_blocks)
        following = vectors.index_select(-2, next_blocks)
        previous_similarity = (vectors * previous).sum(-1)
        next_similarity = (vectors * following).sum(-1)
        previous_similarity.masked_fill_(frames.view(1, 1, -1) == 0, 1.0)
        next_similarity.masked_fill_(
            frames.view(1, 1, -1) == latent_frames - 1, 1.0
        )
        coherence = torch.minimum(previous_similarity, next_similarity)
        median = coherence.median(dim=-1, keepdim=True).values
        mad = (coherence - median).abs().median(dim=-1, keepdim=True).values

        # No manual scene-dependent motion percentage is used.  The current
        # request establishes its own robust temporal outlier boundary.  When
        # the median correspondence is too weak, or most rows look anomalous,
        # the signal is not trustworthy and the high-budget path is retained.
        reliable = median.mean() >= 0.15
        boundary = median - 1.5 * mad.clamp_min(1e-4)
        high_motion = coherence < boundary
        high_motion = functional.max_pool1d(
            high_motion.float().reshape(-1, 1, query_blocks),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(high_motion).bool()
        high_fraction = high_motion.float().mean()
        if not bool(reliable) or float(high_fraction) > 0.55:
            return None
        return high_motion

    def _head_topk(self, heads: int, device: torch.device):
        if isinstance(self.topk, float):
            return self.topk
        if len(self.topk) != heads:
            raise ValueError(
                f"head-wise sparse budget has {len(self.topk)} values for {heads} heads"
            )
        return torch.tensor(self.topk, device=device, dtype=torch.float32)

    def _selected_key_block_counts(
        self,
        heads: int,
        key_blocks: int,
        device: torch.device,
        *,
        apply_absolute_cap: bool = True,
    ) -> torch.Tensor:
        """Return fixed-TopK counts before mandatory structural rails.

        ``fixed_topk_absolute_cap`` preserves the accepted fractional policy
        up to its calibrated reference horizon, then stops discretionary
        global KV selection from growing with sequence length.  Dense prefix,
        MTCR and rotating global anchors are applied afterwards and therefore
        remain non-negotiable.  The helper deliberately preserves the exact
        floor conversion used by the prepared-KV production path.
        """

        if heads <= 0 or key_blocks <= 0:
            raise ValueError("head and key block counts must be positive")
        budgets = self._head_topk(heads, device)
        if isinstance(budgets, float):
            budgets = torch.full(
                (heads,), budgets, device=device, dtype=torch.float32
            )
        nominal = (budgets * key_blocks).to(torch.int64)
        if not apply_absolute_cap or self.selection_mode not in (
            "fixed_topk_absolute_cap",
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
        ):
            return nominal

        maximum = self.maximum_selected_key_blocks
        if isinstance(maximum, int):
            cap = torch.full(
                (heads,), maximum, device=device, dtype=torch.int64
            )
        elif isinstance(maximum, tuple):
            if len(maximum) != heads:
                raise ValueError(
                    "head-wise absolute cap has "
                    f"{len(maximum)} values for {heads} heads"
                )
            cap = torch.tensor(maximum, device=device, dtype=torch.int64)
        else:  # Constructor validation makes this a fail-closed invariant.
            raise RuntimeError("absolute-cap selector is missing its block cap")
        return torch.minimum(nominal, cap)

    def _interaction_risk_guard(
        self,
        pooled_q: torch.Tensor,
        *,
        query_tokens: int,
    ) -> torch.Tensor | None:
        """Select local motion discontinuities that can break contact causality.

        MTCR preserves same-location evidence, but a hand contacting a door or
        a rider passing behind a car is defined by a *change* in local motion.
        This guard measures second-order temporal disagreement in pooled Query
        space.  It is self-calibrated per request/head with median and MAD and
        protects a one-block halo.  No optical flow model, semantic detector or
        user-set motion percentage is required.
        """

        import torch.nn.functional as functional

        layout = _ATTENTION_VIDEO_LAYOUT.get()
        if layout is None or pooled_q.ndim != 4:
            return None
        latent_frames, frame_tokens = layout
        if query_tokens != latent_frames * frame_tokens or latent_frames < 3:
            return None
        query_blocks = int(pooled_q.shape[-2])
        if query_blocks < 3:
            return None

        device = pooled_q.device
        centres = torch.arange(query_blocks, device=device, dtype=torch.int64)
        centres = (centres * 128 + 64).clamp_max(query_tokens - 1)
        frames = torch.div(centres, frame_tokens, rounding_mode="floor")
        spatial = torch.remainder(centres, frame_tokens)
        previous_tokens = (frames - 1).clamp_min(0) * frame_tokens + spatial
        next_tokens = (frames + 1).clamp_max(latent_frames - 1) * frame_tokens + spatial
        previous_blocks = torch.div(previous_tokens, 128, rounding_mode="floor")
        next_blocks = torch.div(next_tokens, 128, rounding_mode="floor")
        previous_blocks.clamp_max_(query_blocks - 1)
        next_blocks.clamp_max_(query_blocks - 1)

        vectors = functional.normalize(pooled_q.float(), dim=-1, eps=1e-6)
        previous = vectors.index_select(-2, previous_blocks)
        following = vectors.index_select(-2, next_blocks)
        incoming = vectors - previous
        outgoing = following - vectors
        incoming_norm = incoming.square().sum(-1).sqrt()
        outgoing_norm = outgoing.square().sum(-1).sqrt()
        motion = torch.maximum(incoming_norm, outgoing_norm)
        direction = functional.cosine_similarity(
            incoming, outgoing, dim=-1, eps=1e-6
        )
        acceleration = (1.0 - direction).mul_(0.5).clamp_(0.0, 1.0)
        risk = motion * acceleration

        interior = (frames > 0) & (frames < latent_frames - 1)
        risk.masked_fill_(~interior.view(1, 1, -1), 0.0)
        median = risk.median(dim=-1, keepdim=True).values
        mad = (risk - median).abs().median(dim=-1, keepdim=True).values
        motion_median = motion.median(dim=-1, keepdim=True).values
        high_risk = (risk > median + 2.0 * mad.clamp_min(1e-5)) & (
            motion > motion_median
        )
        if self.selection_mode == "interaction_dense":
            # Dense recovery is much more expensive than the ordinary rail.
            # Propagate only along the true temporal correspondence here;
            # spatial expansion is supplied by the geometric rail below.
            return (
                high_risk
                | high_risk.index_select(-1, previous_blocks)
                | high_risk.index_select(-1, next_blocks)
            )
        return functional.max_pool1d(
            high_risk.float().reshape(-1, 1, query_blocks),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(high_risk).bool()

    @staticmethod
    def _prepare_kv(key: torch.Tensor, value: torch.Tensor):
        """Build the exact shared K/FP8-V layout used by both query regions."""

        from einops import rearrange
        from spas_sage_attn import core as sparge_core

        k = rearrange(key, "L H D -> 1 H L D").contiguous().to(torch.bfloat16)
        v = rearrange(value, "L H D -> 1 H L D").contiguous().to(torch.float16)
        batch, heads, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        transposed = torch.empty(
            (batch, heads, head_dim, padded_len), dtype=v.dtype, device=v.device
        )
        sparge_core.fused.transpose_pad_permute_cuda(v, transposed, 1)
        v_fp8 = torch.empty_like(transposed, dtype=torch.float8_e4m3fn)
        v_scale = torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=v.device
        )
        sparge_core.fused.scale_fuse_quant_cuda(
            transposed, v_fp8, v_scale, kv_len, 2.25, 1
        )
        return k, v_fp8, v_scale, heads, kv_len, head_dim

    def resolve_long_sequence_backend(self, query_tokens: int):
        """Return this physical backend when exact Query streaming is legal."""

        if query_tokens < self.minimum_sparse_tokens:
            return None
        # The first production route deliberately implements the frozen V19
        # fixed-TopK family.  Adaptive selectors carry whole-request state and
        # must gain their own equivalence evidence before opting in.
        if self.selection_mode not in (
            "fixed_topk",
            "fixed_topk_absolute_cap",
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
        ):
            return None
        return self

    @staticmethod
    def begin_compact_long_sequence_kv(
        *,
        key_tokens: int,
        heads: int,
        head_dim: int,
        key_mean: torch.Tensor,
        value_absmax: torch.Tensor,
        device: torch.device,
    ) -> _CompactSparseKVBuilder:
        """Start an HND K/V build that never owns sequence-long BF16 K/V."""

        return _CompactSparseKVBuilder(
            key_tokens=key_tokens,
            heads=heads,
            head_dim=head_dim,
            key_mean=key_mean,
            value_absmax=value_absmax,
            device=device,
        )

    @staticmethod
    def begin_compact_long_sequence_values(
        *,
        key_tokens: int,
        heads: int,
        head_dim: int,
        value_absmax: torch.Tensor,
        device: torch.device,
    ) -> _CompactValueBuilder:
        return _CompactValueBuilder(
            key_tokens=key_tokens,
            heads=heads,
            head_dim=head_dim,
            value_absmax=value_absmax,
            device=device,
            layout="HND",
        )

    @staticmethod
    def prepare_long_sequence_values(
        value_hnd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Quantize one already-HND V tensor without materializing another V.

        ``value_hnd`` is produced directly by the chunked fused-QKV projector.
        Keeping this seam separate from K preparation lets its owner release
        the BF16/FP16 V allocation before the 1-byte K representation is built.
        """

        from spas_sage_attn import core as sparge_core

        if current_long_sequence_direct_nhd_kv():
            if value_hnd.ndim != 4 or value_hnd.shape[0] != 1:
                raise ValueError("long-sequence NHD V must use [1,L,H,D] layout")
            if value_hnd.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("long-sequence NHD V must be FP16 or BF16")
            if not value_hnd.is_contiguous():
                raise ValueError("long-sequence NHD V must be contiguous")
            batch, kv_len, heads, head_dim = value_hnd.shape
            padded_len = (kv_len + 127) // 128 * 128
            transposed = torch.empty(
                (batch, head_dim, heads, padded_len),
                dtype=value_hnd.dtype,
                device=value_hnd.device,
            )
            sparge_core.fused.transpose_pad_permute_cuda(
                value_hnd, transposed, 0
            )
            value_fp8 = torch.empty_like(
                transposed, dtype=torch.float8_e4m3fn
            )
            value_scale = torch.empty(
                (batch, heads, head_dim),
                dtype=torch.float32,
                device=value_hnd.device,
            )
            sparge_core.fused.scale_fuse_quant_cuda(
                transposed, value_fp8, value_scale, kv_len, 2.25, 0
            )
            del transposed
            return value_fp8, value_scale, heads, kv_len, head_dim

        if value_hnd.ndim != 4 or value_hnd.shape[0] != 1:
            raise ValueError("long-sequence V must use [1,H,L,D] layout")
        if value_hnd.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("long-sequence V must be FP16 or BF16")
        if not value_hnd.is_contiguous():
            raise ValueError("long-sequence V must be contiguous")
        batch, heads, kv_len, head_dim = value_hnd.shape
        padded_len = (kv_len + 127) // 128 * 128
        transposed = torch.empty(
            (batch, heads, head_dim, padded_len),
            dtype=value_hnd.dtype,
            device=value_hnd.device,
        )
        sparge_core.fused.transpose_pad_permute_cuda(value_hnd, transposed, 1)
        value_fp8 = torch.empty_like(transposed, dtype=torch.float8_e4m3fn)
        value_scale = torch.empty(
            (batch, heads, head_dim),
            dtype=torch.float32,
            device=value_hnd.device,
        )
        sparge_core.fused.scale_fuse_quant_cuda(
            transposed, value_fp8, value_scale, kv_len, 2.25, 1
        )
        del transposed
        return value_fp8, value_scale, heads, kv_len, head_dim

    def prepare_long_sequence_keys(
        self,
        key_hnd: torch.Tensor,
        value_fp8: torch.Tensor,
        value_scale: torch.Tensor,
    ) -> PreparedLongSequenceKV:
        """Pool and quantize K once for every streamed video Query chunk."""

        from spas_sage_attn.utils import (
            get_pool_sim_triton_simmean_fuse_quant,
            hyperparameter_check,
        )

        if current_long_sequence_direct_nhd_kv():
            from .sparge_nhd import (
                hnd_compatible_key_mean_nhd,
                pool_sim_quant_nhd,
            )

            if key_hnd.ndim != 4 or key_hnd.shape[0] != 1:
                raise ValueError("long-sequence NHD K must use [1,L,H,D]")
            if key_hnd.dtype != torch.bfloat16 or not key_hnd.is_contiguous():
                raise ValueError("long-sequence NHD K must be contiguous BF16")
            batch, key_tokens, heads, head_dim = key_hnd.shape
            if value_fp8.shape != (
                batch,
                head_dim,
                heads,
                (key_tokens + 127) // 128 * 128,
            ):
                raise ValueError(
                    "prepared long-sequence NHD V shape does not match K"
                )
            if value_scale.shape != (batch, heads, head_dim):
                raise ValueError("prepared long-sequence NHD V scale mismatch")
            key_mean = hnd_compatible_key_mean_nhd(key_hnd)
            sim_threshold = hyperparameter_check(-0.1, heads, key_hnd.device)
            pooled_key, similar_key_blocks, key_int8, key_scale = (
                pool_sim_quant_nhd(
                    key_hnd, key_mean, 64, sim_threshold
                )
            )
            return PreparedLongSequenceKV(
                key=key_hnd,
                key_mean=key_mean,
                pooled_key=pooled_key,
                similar_key_blocks=similar_key_blocks,
                key_int8=key_int8,
                key_scale=key_scale,
                prefix_key_int8=None,
                prefix_key_scale=None,
                value_fp8=value_fp8,
                value_scale=value_scale,
                heads=heads,
                key_tokens=key_tokens,
                head_dim=head_dim,
                tensor_layout="NHD",
            )

        if key_hnd.ndim != 4 or key_hnd.shape[0] != 1:
            raise ValueError("long-sequence K must use [1,H,L,D] layout")
        if key_hnd.dtype != torch.bfloat16 or not key_hnd.is_contiguous():
            raise ValueError("long-sequence K must be contiguous BF16")
        batch, heads, key_tokens, head_dim = key_hnd.shape
        if value_fp8.shape != (
            batch,
            heads,
            head_dim,
            (key_tokens + 127) // 128 * 128,
        ):
            raise ValueError("prepared long-sequence V shape does not match K")
        if value_scale.shape != (batch, heads, head_dim):
            raise ValueError("prepared long-sequence V scale does not match K")
        key_mean = key_hnd.mean(dim=-2, keepdim=True)
        sim_threshold = hyperparameter_check(-0.1, heads, key_hnd.device)
        pooled_key, similar_key_blocks, key_int8, key_scale = (
            get_pool_sim_triton_simmean_fuse_quant(
                key_hnd, key_mean, 64, sim_threshold
            )
        )
        return PreparedLongSequenceKV(
            key=key_hnd,
            key_mean=key_mean,
            pooled_key=pooled_key,
            similar_key_blocks=similar_key_blocks,
            key_int8=key_int8,
            key_scale=key_scale,
            prefix_key_int8=None,
            prefix_key_scale=None,
            value_fp8=value_fp8,
            value_scale=value_scale,
            heads=heads,
            key_tokens=key_tokens,
            head_dim=head_dim,
        )

    def long_sequence_prefix_queries(
        self,
        query: torch.Tensor,
        prepared: PreparedLongSequenceKV,
    ) -> torch.Tensor:
        """Evaluate the small protected prefix with the accepted Dense math."""

        if prepared.key is None:
            return self._compact_full_context_queries(query, prepared)

        if prepared.tensor_layout == "NHD":
            return self._dense_prefix_nhd(
                query,
                prepared.key,
                prepared.value_fp8,
                prepared.value_scale,
                head_dim=prepared.head_dim,
                key_mean=prepared.key_mean,
                fused_key_quant=(
                    self.fused_long_sequence_prefix_k_quant
                    or current_long_sequence_fused_prefix_k_quant()
                    or current_long_sequence_exact_helper_stack()
                ),
            )

        return self._dense_prefix(
            query,
            prepared.key,
            prepared.value_fp8,
            prepared.value_scale,
            head_dim=prepared.head_dim,
            key_mean=prepared.key_mean,
            fused_key_quant=(
                self.fused_long_sequence_prefix_k_quant
                or current_long_sequence_fused_prefix_k_quant()
                or current_long_sequence_exact_helper_stack()
            ),
        )

    @staticmethod
    def _compact_full_context_queries(
        query: torch.Tensor,
        prepared: PreparedLongSequenceKV,
    ) -> torch.Tensor:
        """Run compact prefix queries through Dense Sage over every KV row.

        The ordinary protected prefix uses Sage's dense helper and therefore
        needs sequence-long BF16 K.  Compact execution deliberately does not
        retain that tensor.  The stable per-warp Sage kernel directly reuses
        Sparge's per-64-row K representation, retaining full prefix-to-KV
        connectivity without a second sequence-long INT8 K copy.
        """

        from einops import rearrange
        from sageattention import _fused, sm89_compile

        if (
            query.ndim != 3
            or query.shape[0] <= 0
            or query.shape[1] != prepared.heads
            or query.shape[2] != prepared.head_dim
        ):
            raise ValueError("compact prefix Query does not match prepared K/V")
        q = rearrange(query, "L H D -> 1 H L D").contiguous().to(torch.bfloat16)
        query_tokens = int(q.shape[2])
        q_int8 = torch.empty_like(q, dtype=torch.int8)
        q_scale = torch.empty(
            (1, prepared.heads, ((query_tokens + 127) // 128) * 4),
            device=q.device,
            dtype=torch.float32,
        )
        _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, 128, 32, 1)
        output = torch.empty_like(q)
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8,
            prepared.key_int8,
            prepared.value_fp8,
            output,
            q_scale,
            prepared.key_scale,
            prepared.value_scale,
            1,
            0,
            2,
            1.0 / (prepared.head_dim**0.5),
            0,
        )
        return rearrange(output, "1 H L D -> L H D")

    def long_sequence_video_queries(
        self,
        query: torch.Tensor,
        prepared: PreparedLongSequenceKV,
        *,
        protected_tokens: int,
        query_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one aligned video Query chunk with reusable K/V state."""

        from einops import rearrange
        from spas_sage_attn import core as sparge_core
        from spas_sage_attn.utils import (
            block_map_lut_triton,
            fill_block_map_triton,
            get_pool_sim_triton_simmean_fuse_quant,
            hyperparameter_check,
        )

        if self.selection_mode not in (
            "fixed_topk",
            "fixed_topk_absolute_cap",
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
        ):
            raise RuntimeError(
                "long-sequence prepared-KV requires a stateless fixed-TopK selector"
            )
        if (
            query.ndim != 3
            or query.shape[0] <= 0
            or query.shape[1] != prepared.heads
            or query.shape[2] != prepared.head_dim
        ):
            raise ValueError("long-sequence Query chunk does not match prepared K/V")
        if (
            query_token_indices.ndim != 1
            or query_token_indices.numel() != query.shape[0]
            or query_token_indices.device != query.device
        ):
            raise ValueError("long-sequence Query indices must align with the chunk")

        if prepared.tensor_layout == "NHD":
            from .sparge_nhd import pool_sim_quant_nhd

            q = query.unsqueeze(0).contiguous().to(torch.bfloat16)
        else:
            q = rearrange(
                query, "L H D -> 1 H L D"
            ).contiguous().to(torch.bfloat16)
        sim_threshold = hyperparameter_check(-0.1, prepared.heads, q.device)
        if prepared.tensor_layout == "NHD":
            pooled_query, similar_query_blocks, q_int8, q_scale = (
                pool_sim_quant_nhd(q, None, 128, sim_threshold)
            )
        else:
            pooled_query, similar_query_blocks, q_int8, q_scale = (
                get_pool_sim_triton_simmean_fuse_quant(
                    q, None, 128, sim_threshold
                )
            )
        query_blocks = int(pooled_query.shape[-2])
        key_blocks = int(prepared.pooled_key.shape[-2])
        expanded_key = prepared.similar_key_blocks.unsqueeze(-2).expand(
            -1, -1, query_blocks, -1
        )
        expanded_query = similar_query_blocks.unsqueeze(-1).expand(
            -1, -1, -1, key_blocks
        )
        scores = pooled_query @ prepared.pooled_key.transpose(-1, -2)
        scores.mul_(q.shape[-1] ** -0.5)
        scores.masked_fill_(~expanded_key, -torch.inf)
        mass_guard_mode = self.selection_mode in (
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
        )
        use_partial_topk = (
            self.partial_long_sequence_topk
            or current_long_sequence_partial_sparse_topk()
            or current_long_sequence_exact_helper_stack()
        )
        if use_partial_topk and not mass_guard_mode:
            budgets = (
                (self.topk,)
                if isinstance(self.topk, float)
                else self.topk
            )
            maximum_selected = max(
                1,
                min(key_blocks, int(max(budgets) * key_blocks)),
            )
            selected_indices = torch.topk(
                scores.softmax(-1),
                maximum_selected,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            sorted_scores = None
        else:
            sorted_scores = torch.sort(
                scores.softmax(-1), dim=-1, descending=True
            )
            selected_indices = sorted_scores.indices
        nominal_head_count = self._selected_key_block_counts(
            prepared.heads,
            key_blocks,
            q.device,
            apply_absolute_cap=False,
        )
        capped_head_count = self._selected_key_block_counts(
            prepared.heads, key_blocks, q.device
        )
        nominal_selected = nominal_head_count.view(1, prepared.heads, 1).expand(
            scores.shape[0], -1, query_blocks
        )
        capped_selected = capped_head_count.view(1, prepared.heads, 1).expand(
            scores.shape[0], -1, query_blocks
        )
        cap_activated = None
        retained_mass = None
        if mass_guard_mode:
            assert sorted_scores is not None
            # The selector has already sorted proxy Attention probability.
            # Cumsum only the accepted Top-K prefix (<=10% on the current V19
            # rail), not all KV blocks.  This adds a compact request-local
            # confidence test: use the cap only where it retains the declared
            # fraction of the original fixed-TopK mass; diffuse rows fail
            # closed to their accepted nominal count.
            maximum_nominal = max(
                1,
                math.ceil(
                    max(
                        (self.topk,)
                        if isinstance(self.topk, float)
                        else self.topk
                    )
                    * key_blocks
                ),
            )
            prefix_cdf = torch.cumsum(
                sorted_scores.values[..., :maximum_nominal],
                dim=-1,
                dtype=torch.float32,
            )
            nominal_mass = prefix_cdf.gather(
                -1,
                (nominal_selected - 1).clamp_min(0).unsqueeze(-1),
            ).squeeze(-1)
            capped_mass = prefix_cdf.gather(
                -1,
                (capped_selected - 1).clamp_min(0).unsqueeze(-1),
            ).squeeze(-1)
            retained_mass = capped_mass / nominal_mass.clamp_min(1e-8)
            cap_activated = retained_mass >= self.minimum_retained_topk_mass
            selected_count = (
                nominal_selected.contiguous()
                if self.selection_mode == "fixed_topk_mass_probe"
                else torch.where(
                    cap_activated,
                    capped_selected,
                    nominal_selected,
                ).contiguous()
            )
            step = _ATTENTION_STEP.get()
            self._mass_guard_records.append(
                {
                    "step": None if step is None else int(step[0]),
                    "layer": _ATTENTION_LAYER.get(),
                    "key_blocks": key_blocks,
                    "query_blocks": query_blocks,
                    "selected_blocks": int(
                        self.maximum_selected_key_blocks
                        if isinstance(self.maximum_selected_key_blocks, int)
                        else max(self.maximum_selected_key_blocks)
                    ),
                    "activated": cap_activated.sum().detach(),
                    "total": int(cap_activated.numel()),
                    "retained_sum": retained_mass.sum().detach(),
                    "retained_min": retained_mass.min().detach(),
                    "retained_max": retained_mass.max().detach(),
                    "threshold_counts": torch.stack(
                        tuple(
                            (retained_mass >= threshold).sum()
                            for threshold in (0.90, 0.925, 0.95, 0.975, 0.99)
                        )
                    ).detach(),
                }
            )
            if (
                self.selection_mode == "fixed_topk_mass_probe"
                and self.mass_probe_selected_key_blocks
            ):
                for probe_count in self.mass_probe_selected_key_blocks:
                    probe_count = min(probe_count, maximum_nominal)
                    if (
                        isinstance(self.maximum_selected_key_blocks, int)
                        and probe_count == self.maximum_selected_key_blocks
                    ):
                        continue
                    probe_selected = torch.full_like(
                        nominal_selected, probe_count
                    )
                    probe_mass = prefix_cdf.gather(
                        -1,
                        (probe_selected - 1).clamp_min(0).unsqueeze(-1),
                    ).squeeze(-1)
                    probe_retained = probe_mass / nominal_mass.clamp_min(1e-8)
                    probe_activated = (
                        probe_retained >= self.minimum_retained_topk_mass
                    )
                    self._mass_guard_records.append(
                        {
                            "step": None if step is None else int(step[0]),
                            "layer": _ATTENTION_LAYER.get(),
                            "key_blocks": key_blocks,
                            "query_blocks": query_blocks,
                            "selected_blocks": probe_count,
                            "activated": probe_activated.sum().detach(),
                            "total": int(probe_activated.numel()),
                            "retained_sum": probe_retained.sum().detach(),
                            "retained_min": probe_retained.min().detach(),
                            "retained_max": probe_retained.max().detach(),
                            "threshold_counts": torch.stack(
                                tuple(
                                    (probe_retained >= threshold).sum()
                                    for threshold in (
                                        0.90,
                                        0.925,
                                        0.95,
                                        0.975,
                                        0.99,
                                    )
                                )
                            ).detach(),
                        }
                    )
                    del probe_selected, probe_mass, probe_retained, probe_activated
            del prefix_cdf, nominal_mass, capped_mass
        else:
            selected_count = capped_selected.contiguous()
        block_map = torch.zeros_like(scores, dtype=torch.bool)
        block_map[~expanded_key] = True
        block_map[~expanded_query] = True
        if use_partial_topk and not mass_guard_mode:
            from .sparse_lut import fill_block_map_partial_topk

            block_map = fill_block_map_partial_topk(
                block_map, selected_count, selected_indices
            )
        else:
            block_map = fill_block_map_triton(
                block_map, selected_count, selected_indices
            )
        protected_key_blocks = (protected_tokens + 63) // 64
        block_map[:, :, :, :protected_key_blocks] = True
        self._protect_temporal_correspondence(
            block_map,
            query_tokens=int(query.shape[0]),
            key_tokens=prepared.key_tokens,
            protected_tokens=protected_tokens,
            query_token_indices=query_token_indices,
        )
        block_map = block_map.contiguous()
        if (
            self.parallel_long_sequence_lut
            or current_long_sequence_parallel_sparse_lut()
            or current_long_sequence_exact_helper_stack()
        ):
            from .sparse_lut import parallel_block_map_lut

            lut, valid_block_num = parallel_block_map_lut(block_map)
        else:
            lut, valid_block_num = block_map_lut_triton(block_map)
        if self.selection_mode in (
            "fixed_topk_absolute_cap",
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
        ):
            self._absolute_cap_calls += 1
            self._absolute_cap_last = {
                "key_blocks": key_blocks,
                "query_blocks": query_blocks,
                "nominal_selected": nominal_head_count.detach(),
                "capped_selected": capped_head_count.detach(),
                "selected_before_rails": selected_count.detach(),
                "valid_after_mandatory_rails": valid_block_num.detach(),
                "cap_activated": (
                    cap_activated.detach() if cap_activated is not None else None
                ),
                "retained_mass": (
                    retained_mass.detach() if retained_mass is not None else None
                ),
            }
        tensor_layout = 0 if prepared.tensor_layout == "NHD" else 1
        direct_nhd_output = (
            prepared.tensor_layout == "NHD"
            or current_long_sequence_direct_nhd_output()
        )
        if prepared.tensor_layout == "NHD":
            output_nhd = torch.empty_like(q)
            output = output_nhd
        elif direct_nhd_output:
            output_nhd = torch.empty(
                (q.shape[0], q.shape[2], q.shape[1], q.shape[3]),
                device=q.device,
                dtype=q.dtype,
            )
            output = output_nhd.permute(0, 2, 1, 3)
        else:
            output_nhd = None
            output = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, prepared.heads, q.device)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8,
            prepared.key_int8,
            prepared.value_fp8,
            output,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            prepared.key_scale,
            prepared.value_scale,
            tensor_layout,
            0,
            1,
            1.0 / (prepared.head_dim**0.5),
            0,
        )
        if output_nhd is not None:
            return output_nhd[0]
        return rearrange(output, "1 H L D -> L H D")

    @staticmethod
    def _dense_prefix(
        query: torch.Tensor,
        k: torch.Tensor,
        v_fp8: torch.Tensor,
        v_scale: torch.Tensor,
        *,
        head_dim: int,
        key_mean: torch.Tensor | None = None,
        fused_key_quant: bool = False,
    ) -> torch.Tensor:
        from einops import rearrange
        from sageattention import sm89_compile
        from sageattention.triton.quant_per_thread import per_thread_int8

        prefix_q = rearrange(query, "L H D -> 1 H L D").contiguous()
        prefix_key_mean = (
            k.mean(dim=2, keepdim=True) if key_mean is None else key_mean
        )
        if fused_key_quant:
            from .sage_fused_quant import (
                quantize_qk_sub_mean_per_thread_int8_hnd,
            )

            prefix_q_int8, prefix_q_scale, prefix_k_int8, prefix_k_scale = (
                quantize_qk_sub_mean_per_thread_int8_hnd(
                    prefix_q, k, prefix_key_mean
                )
            )
        else:
            prefix_q_int8, prefix_q_scale, prefix_k_int8, prefix_k_scale = per_thread_int8(
                prefix_q,
                k,
                prefix_key_mean,
                tensor_layout="HND",
                BLKQ=128,
                WARPQ=32,
                BLKK=64,
                WARPK=64,
            )
        prefix_output_hnd = torch.empty_like(prefix_q)
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            prefix_q_int8,
            prefix_k_int8,
            v_fp8,
            prefix_output_hnd,
            prefix_q_scale,
            prefix_k_scale,
            v_scale,
            1,
            0,
            3,
            1.0 / (head_dim**0.5),
            0,
        )
        return rearrange(prefix_output_hnd, "1 H L D -> L H D")

    @staticmethod
    def _dense_prefix_nhd(
        query: torch.Tensor,
        key: torch.Tensor,
        value_fp8: torch.Tensor,
        value_scale: torch.Tensor,
        *,
        head_dim: int,
        key_mean: torch.Tensor,
        fused_key_quant: bool,
    ) -> torch.Tensor:
        """Dense protected-prefix Attention without leaving native NHD."""

        from sageattention import sm89_compile

        prefix_q = query.unsqueeze(0).contiguous()
        if fused_key_quant:
            from .sage_fused_quant import (
                quantize_qk_sub_mean_per_thread_int8_nhd,
            )

            prefix_q_int8, prefix_q_scale, prefix_k_int8, prefix_k_scale = (
                quantize_qk_sub_mean_per_thread_int8_nhd(
                    prefix_q, key, key_mean
                )
            )
        else:
            from sageattention.triton.quant_per_thread import per_thread_int8

            prefix_q_int8, prefix_q_scale, prefix_k_int8, prefix_k_scale = (
                per_thread_int8(
                    prefix_q,
                    key,
                    key_mean,
                    tensor_layout="NHD",
                    BLKQ=128,
                    WARPQ=32,
                    BLKK=64,
                    WARPK=64,
                )
            )
        output = torch.empty_like(prefix_q)
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            prefix_q_int8,
            prefix_k_int8,
            value_fp8,
            output,
            prefix_q_scale,
            prefix_k_scale,
            value_scale,
            0,
            0,
            3,
            1.0 / (head_dim**0.5),
            0,
        )
        return output[0]

    @staticmethod
    def _dense_from_prepared_kv(
        query: torch.Tensor,
        k: torch.Tensor,
        v_fp8: torch.Tensor,
        v_scale: torch.Tensor,
        *,
        head_dim: int,
    ) -> torch.Tensor:
        """Run the accepted dense Sage kernel while reusing prepared FP8 V.

        This follows SageAttention's SM89 implementation exactly for the
        request's selected Q/K quantization granularity.  Only the already
        materialized FP8-V tensor and scale are reused from the sparse draft;
        Q and smooth-K remain quantized by the original Sage routines.
        """

        from einops import rearrange
        from sageattention import sm89_compile
        from sageattention.quant import per_warp_int8
        from sageattention.triton.quant_per_thread import per_thread_int8

        q = rearrange(query, "L H D -> 1 H L D").contiguous()
        key_mean = k.mean(dim=2, keepdim=True)
        granularity = _DENSE_QK_QUANT_GRAN.get()
        if granularity == "per_warp":
            q_int8, q_scale, k_int8, k_scale = per_warp_int8(
                q,
                k,
                key_mean,
                tensor_layout="HND",
                BLKQ=128,
                WARPQ=32,
                BLKK=64,
            )
            granularity_code = 2
        elif granularity == "per_thread":
            q_int8, q_scale, k_int8, k_scale = per_thread_int8(
                q,
                k,
                key_mean,
                tensor_layout="HND",
                BLKQ=128,
                WARPQ=32,
                BLKK=64,
                WARPK=64,
            )
            granularity_code = 3
        else:
            raise ValueError(f"unsupported dense Q/K granularity: {granularity}")
        output = torch.empty_like(q)
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8,
            k_int8,
            v_fp8,
            output,
            q_scale,
            k_scale,
            v_scale,
            1,
            0,
            granularity_code,
            1.0 / (head_dim**0.5),
            0,
        )
        return rearrange(output, "1 H L D -> L H D")

    def _full_from_prepared_kv(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v_fp8: torch.Tensor,
        v_scale: torch.Tensor,
        *,
        protected_tokens: int,
        heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        prefix_output = self._dense_prefix(
            query[:protected_tokens], k, v_fp8, v_scale, head_dim=head_dim
        )
        video_output = self._sparse_video_queries(
            query[protected_tokens:],
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
        )
        return torch.cat((prefix_output, video_output), dim=0)

    def full_with_exact_sample(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        sample_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ordinary full draft plus exact sampled rows with one V prep."""

        protected_tokens = int(_ATTENTION_PROTECTED_PREFIX.get())
        if (
            self.selection_mode == "unified_fixed_topk"
            or protected_tokens <= 0
            or protected_tokens >= int(query.shape[0])
        ):
            full = self(query, key, value)
            exact = sage_attention_sm89(
                query.index_select(0, sample_indices), key, value
            )
            return full, exact
        k, v_fp8, v_scale, heads, _kv_len, head_dim = self._prepare_kv(
            key, value
        )
        full = self._full_from_prepared_kv(
            query,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
        )
        exact = self._dense_from_prepared_kv(
            query.index_select(0, sample_indices),
            k,
            v_fp8,
            v_scale,
            head_dim=head_dim,
        )
        return full, exact

    def protected_queries(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate only protected queries with the production shared-V math."""

        if query.shape[0] <= 0 or key.shape[0] != value.shape[0]:
            raise ValueError("protected attention requires non-empty Q and aligned K/V")
        k, v_fp8, v_scale, _heads, _kv_len, head_dim = self._prepare_kv(
            key, value
        )
        return self._dense_prefix(
            query, k, v_fp8, v_scale, head_dim=head_dim
        )

    def _unified_prefix_and_video(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        protected_tokens: int,
    ) -> torch.Tensor:
        """Quantize K once and evaluate prefix/video in one sparse kernel.

        The accepted split implementation quantizes the complete K sequence
        once for the small dense prefix and again for the dominant sparse
        video region.  This research path creates one block map for all query
        rows, marks every prefix-query block dense, and retains the identical
        prefix-key and MTCR protections.  It changes Q/K quantization grouping
        for prefix queries, so it remains an explicit approximate candidate
        until full-video Human review.
        """

        from einops import rearrange
        from spas_sage_attn import core as sparge_core
        from spas_sage_attn.utils import (
            block_map_lut_triton,
            get_block_map_meansim_fuse_quant,
            hyperparameter_check,
        )

        q = rearrange(query, "L H D -> 1 H L D").contiguous().to(torch.bfloat16)
        k, v_fp8, v_scale, heads, kv_len, head_dim = self._prepare_kv(key, value)
        key_mean = k.mean(dim=-2, keepdim=True)
        block_map, q_int8, q_scale, k_int8, k_scale = (
            get_block_map_meansim_fuse_quant(
                q,
                k,
                key_mean,
                BLKQ=128,
                BLKK=64,
                simthreshd1=-0.1,
                cdfthreshd=None,
                topk=self._head_topk(heads, q.device),
                is_causal=False,
            )
        )
        protected_q_blocks = (protected_tokens + 127) // 128
        protected_k_blocks = (protected_tokens + 63) // 64
        block_map[:, :, :protected_q_blocks, :] = True
        block_map[:, :, :, :protected_k_blocks] = True

        # The first sparse-video block starts after the rounded-up protected
        # prefix.  Supply its true video-row offset to the established MTCR
        # mapper so the structural motion rail remains unchanged.
        video_map = block_map[:, :, protected_q_blocks:, :]
        if video_map.shape[-2]:
            represented_rows = int(video_map.shape[-2]) * 128
            video_offset = protected_q_blocks * 128 - protected_tokens
            video_indices = torch.arange(
                represented_rows, device=q.device, dtype=torch.int64
            ).add_(video_offset)
            video_indices.clamp_max_(max(0, query.shape[0] - protected_tokens - 1))
            self._protect_temporal_correspondence(
                video_map,
                query_tokens=represented_rows,
                key_tokens=kv_len,
                protected_tokens=protected_tokens,
                query_token_indices=video_indices,
            )

        lut, valid_block_num = block_map_lut_triton(block_map.contiguous())
        output = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, heads, q.device)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8,
            k_int8,
            v_fp8,
            output,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            k_scale,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )
        return rearrange(output, "1 H L D -> L H D")

    def _sparse_video_queries(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v_fp8: torch.Tensor,
        v_scale: torch.Tensor,
        *,
        protected_tokens: int,
        heads: int,
        head_dim: int,
        query_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from einops import rearrange
        from spas_sage_attn import core as sparge_core
        from spas_sage_attn.utils import (
            block_map_lut_triton,
            fill_block_map_triton,
            get_block_map_meansim_fuse_quant,
            get_pool_sim_triton_simmean_fuse_quant,
            hyperparameter_check,
        )

        if self.selection_mode in (
            "fixed_topk_mass_guarded_cap",
            "fixed_topk_mass_probe",
        ):
            raise RuntimeError(
                "mass-guarded cap/probe currently requires prepared-KV streaming"
            )
        q = rearrange(query, "L H D -> 1 H L D").contiguous().to(torch.bfloat16)
        key_mean = k.mean(dim=-2, keepdim=True)
        if self.selection_mode == "route_cache":
            sim_threshold = hyperparameter_check(-0.1, heads, q.device)
            pooled_q, sim_q, q_int8, q_scale = (
                get_pool_sim_triton_simmean_fuse_quant(
                    q, None, 128, sim_threshold
                )
            )
            pooled_k, sim_k, k_int8, k_scale = (
                get_pool_sim_triton_simmean_fuse_quant(
                    k, key_mean, 64, sim_threshold
                )
            )
            query_blocks = int(pooled_q.shape[-2])
            key_blocks = int(pooled_k.shape[-2])
            layer = _ATTENTION_LAYER.get()
            if layer is None:
                raise RuntimeError("route cache requires a true H3 layer context")
            cache_key = (layer, query_blocks, key_blocks, heads)
            q_samples = torch.linspace(
                0,
                query_blocks - 1,
                min(12, query_blocks),
                device=q.device,
            ).round().to(torch.long).unique()
            k_samples = torch.linspace(
                0,
                key_blocks - 1,
                min(12, key_blocks),
                device=q.device,
            ).round().to(torch.long).unique()
            q_fingerprint = pooled_q.index_select(-2, q_samples).float()
            k_fingerprint = pooled_k.index_select(-2, k_samples).float()
            entry = self._route_cache.get(cache_key)
            similarity = -1.0
            reuse = False
            if entry is not None and bool(entry.get("enabled", False)):
                import torch.nn.functional as functional

                q_similarity = functional.cosine_similarity(
                    q_fingerprint,
                    entry["q_fingerprint"],
                    dim=-1,
                    eps=1e-6,
                ).mean()
                k_similarity = functional.cosine_similarity(
                    k_fingerprint,
                    entry["k_fingerprint"],
                    dim=-1,
                    eps=1e-6,
                ).mean()
                similarity = float(torch.minimum(q_similarity, k_similarity).item())
                reuse = similarity >= float(entry["minimum_similarity"])
            if reuse:
                lut = entry["lut"]
                valid_block_num = entry["valid_block_num"]
                self._route_cache_hits += 1
            else:
                expanded_k = sim_k.unsqueeze(-2).expand(
                    -1, -1, query_blocks, -1
                )
                expanded_q = sim_q.unsqueeze(-1).expand(
                    -1, -1, -1, key_blocks
                )
                scores = pooled_q @ pooled_k.transpose(-1, -2)
                scores.mul_(q.shape[-1] ** -0.5)
                scores.masked_fill_(~expanded_k, -torch.inf)
                sorted_scores = torch.sort(
                    scores.softmax(-1), dim=-1, descending=True
                )
                budgets = hyperparameter_check(
                    self._head_topk(heads, q.device), heads, q.device
                )
                selected_count = (
                    budgets * key_blocks
                ).to(torch.int64).view(1, heads, 1).expand(
                    scores.shape[0], -1, query_blocks
                ).contiguous()
                block_map = torch.zeros_like(scores, dtype=torch.bool)
                block_map[~expanded_k] = True
                block_map[~expanded_q] = True
                block_map = fill_block_map_triton(
                    block_map, selected_count, sorted_scores.indices
                )
                protected_k_blocks = (protected_tokens + 63) // 64
                block_map[:, :, :, :protected_k_blocks] = True
                self._protect_temporal_correspondence(
                    block_map,
                    query_tokens=int(query.shape[0]),
                    key_tokens=int(k.shape[-2]),
                    protected_tokens=protected_tokens,
                    query_token_indices=query_token_indices,
                )
                lut, valid_block_num = block_map_lut_triton(
                    block_map.contiguous()
                )
                sampled_map = block_map.index_select(-2, q_samples).detach()
                enabled = False
                minimum_similarity = 1.0
                if entry is not None:
                    previous_map = entry["sampled_map"]
                    intersection = (sampled_map & previous_map).sum().float()
                    union = (sampled_map | previous_map).sum().clamp_min(1).float()
                    overlap = float((intersection / union).item())
                    enabled = overlap >= 0.90
                    if enabled:
                        # The observed calibration drift becomes the request's
                        # own fail-closed boundary; no user-tuned threshold.
                        if similarity < 0.0:
                            import torch.nn.functional as functional

                            q_similarity = functional.cosine_similarity(
                                q_fingerprint,
                                entry["q_fingerprint"],
                                dim=-1,
                                eps=1e-6,
                            ).mean()
                            k_similarity = functional.cosine_similarity(
                                k_fingerprint,
                                entry["k_fingerprint"],
                                dim=-1,
                                eps=1e-6,
                            ).mean()
                            similarity = float(
                                torch.minimum(q_similarity, k_similarity).item()
                            )
                        minimum_similarity = similarity
                    else:
                        self._route_cache_rejected += 1
                self._route_cache[cache_key] = {
                    "lut": lut,
                    "valid_block_num": valid_block_num,
                    "q_fingerprint": q_fingerprint.detach(),
                    "k_fingerprint": k_fingerprint.detach(),
                    "sampled_map": sampled_map,
                    "enabled": enabled,
                    "minimum_similarity": minimum_similarity,
                }
                self._route_cache_misses += 1
                del scores, sorted_scores, block_map, sampled_map

            output = torch.empty_like(q)
            pv_threshold = hyperparameter_check(50, heads, q.device)
            sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                q_int8,
                k_int8,
                v_fp8,
                output,
                lut,
                valid_block_num,
                pv_threshold,
                q_scale,
                k_scale,
                v_scale,
                1,
                0,
                1,
                1.0 / (head_dim**0.5),
                0,
            )
            return rearrange(output, "1 H L D -> L H D")
        if self.selection_mode == "fixed_topk":
            block_map, q_int8, q_scale, k_int8, k_scale = (
                get_block_map_meansim_fuse_quant(
                    q,
                    k,
                    key_mean,
                    BLKQ=128,
                    BLKK=64,
                    simthreshd1=-0.1,
                    cdfthreshd=None,
                    topk=self._head_topk(heads, q.device),
                    is_causal=False,
                )
            )
        else:
            # LoSA-style query-adaptive mass allocation under the *same mean
            # block budget* as fixed Top-K.  Concentrated/static query blocks
            # spend fewer exact KV blocks, while diffuse/complex-motion blocks
            # spend more.  This changes allocation, not sampler steps or model
            # weights, and bounds each query to [0.5x, 2x] of its head budget.
            sim_threshold = hyperparameter_check(-0.1, heads, q.device)
            pooled_q, sim_q, q_int8, q_scale = (
                get_pool_sim_triton_simmean_fuse_quant(
                    q, None, 128, sim_threshold
                )
            )
            pooled_k, sim_k, k_int8, k_scale = (
                get_pool_sim_triton_simmean_fuse_quant(
                    k, key_mean, 64, sim_threshold
                )
            )
            query_blocks = pooled_q.shape[-2]
            key_blocks = pooled_k.shape[-2]
            expanded_k = sim_k.unsqueeze(-2).expand(
                -1, -1, query_blocks, -1
            )
            expanded_q = sim_q.unsqueeze(-1).expand(
                -1, -1, -1, key_blocks
            )
            scores = pooled_q @ pooled_k.transpose(-1, -2)
            scores.mul_(q.shape[-1] ** -0.5)
            scores.masked_fill_(~expanded_k, -torch.inf)
            sorted_scores = torch.sort(scores.softmax(-1), dim=-1, descending=True)
            base_count = self._selected_key_block_counts(
                heads, int(key_blocks), q.device
            ).clamp(1, key_blocks)
            query_protection_mask = None
            if self.selection_mode == "fixed_topk_absolute_cap":
                selected_count = base_count.view(1, heads, 1).expand(
                    scores.shape[0], -1, query_blocks
                ).contiguous()
                boundary_mass = None
                cdf = None
            elif self.selection_mode == "causal_head_guard":
                dense_heads = self._causal_head_dense_mask(sorted_scores.values)
                selected_count = base_count.view(1, heads, 1).expand(
                    scores.shape[0], -1, query_blocks
                )
                selected_count = torch.where(
                    dense_heads.view(1, heads, 1),
                    torch.full_like(selected_count, key_blocks),
                    selected_count,
                ).contiguous()
                self._causal_head_guard_calls += 1
                self._causal_head_guard_dense_heads += int(dense_heads.sum().item())
                self._causal_head_guard_total_heads += heads
                boundary_mass = None
                cdf = None
            elif self.selection_mode in (
                "temporal_motion_guard",
                "interaction_guard",
                "interaction_rail",
                "interaction_recovery",
                "interaction_rebalance",
                "interaction_hybrid",
                "interaction_dense",
            ):
                high_motion = (
                    self._temporal_motion_guard(
                        pooled_q,
                        query_tokens=int(query.shape[0]),
                    )
                    if self.selection_mode == "temporal_motion_guard"
                    else self._interaction_risk_guard(
                        pooled_q,
                        query_tokens=int(query.shape[0]),
                    )
                )
                selected_count = base_count.view(1, heads, 1).expand(
                    scores.shape[0], -1, query_blocks
                )
                if high_motion is not None:
                    if self.selection_mode == "temporal_motion_guard":
                        floor_count = max(
                            1,
                            math.ceil(self.experimental_minimum_topk * key_blocks),
                        )
                        selected_count = torch.where(
                            high_motion,
                            selected_count,
                            torch.full_like(selected_count, floor_count),
                        )
                    elif self.selection_mode in (
                        "interaction_guard",
                        "interaction_recovery",
                        "interaction_hybrid",
                        "interaction_dense",
                    ):
                        # Add quality budget only at temporal acceleration
                        # outliers. Ordinary motion and static regions keep the
                        # exact accepted MTCR count; nothing is discounted.
                        multiplier = 2.0
                        if self.selection_mode == "interaction_dense":
                            multiplier = math.inf
                        if self.selection_mode in (
                            "interaction_recovery",
                            "interaction_hybrid",
                        ):
                            step = _ATTENTION_STEP.get()
                            multiplier = (
                                3.0
                                if step is not None and step[0] >= step[1] - 3
                                else 1.5
                            )
                        interaction_count = (
                            torch.full_like(base_count, key_blocks)
                            if math.isinf(multiplier)
                            else torch.ceil(base_count.float() * multiplier)
                            .to(torch.int64)
                            .clamp_max(key_blocks)
                        )
                        selected_count = torch.where(
                            high_motion,
                            interaction_count.view(1, heads, 1),
                            selected_count,
                        )
                        if self.selection_mode in (
                            "interaction_hybrid",
                            "interaction_dense",
                        ):
                            query_protection_mask = high_motion
                    elif self.selection_mode == "interaction_rebalance":
                        # Transfer a bounded amount of cruise-phase compute
                        # from robust low-risk rows to locally discontinuous
                        # interaction rows. The mean target is solved from the
                        # observed per-head risk fraction rather than from a
                        # hand-authored scene mask. Late recovery keeps the
                        # original mean budget to preserve sharpness.
                        step = _ATTENTION_STEP.get()
                        recovery = step is not None and step[0] >= step[1] - 3
                        target_ratio = 1.0 if recovery else 0.92
                        high_count = torch.ceil(
                            base_count.float() * 1.75
                        ).to(torch.int64).clamp_max(key_blocks)
                        risk_fraction = high_motion.float().mean(
                            dim=-1, keepdim=True
                        )
                        target_mean = (
                            base_count.view(1, heads, 1).float() * target_ratio
                        )
                        low_count = (
                            target_mean
                            - risk_fraction * high_count.view(1, heads, 1)
                        ) / (1.0 - risk_fraction).clamp_min(1e-3)
                        floor_count = max(
                            1,
                            math.ceil(self.experimental_minimum_topk * key_blocks),
                        )
                        low_count = torch.ceil(low_count).to(torch.int64)
                        low_count.clamp_(floor_count, key_blocks)
                        selected_count = torch.where(
                            high_motion,
                            high_count.view(1, heads, 1),
                            low_count,
                        )
                        query_protection_mask = high_motion
                    else:
                        # Preserve the accepted per-query budget.  The risk
                        # mask is consumed below to force a compact local
                        # space-time rail around contacts and occlusions.
                        query_protection_mask = high_motion
                selected_count = selected_count.contiguous()
                boundary_mass = None
                cdf = None
            elif self.selection_mode == "budget_adaptive":
                cdf = torch.cumsum(sorted_scores.values, dim=-1)
                gather_index = (base_count - 1).view(1, heads, 1, 1).expand(
                    cdf.shape[0], -1, query_blocks, 1
                )
                boundary_mass = cdf.gather(-1, gather_index).squeeze(-1)
                mass_target = boundary_mass.median(dim=-1).values
                mass_count = torch.searchsorted(
                    cdf,
                    mass_target[:, :, None, None]
                    .expand(-1, -1, query_blocks, 1)
                    .contiguous(),
                    right=True,
                ).squeeze(-1)
                nominal_budgets = (
                    (self.topk,)
                    if isinstance(self.topk, float)
                    else self.topk
                )
                # The first request-adaptive prototype used only the absolute
                # experimental floor, allowing a 0.35 nominal row to collapse
                # to 0.0625 while other rows consumed the rebate.  Real 720p5
                # FasterVQA exposed the resulting quality loss.  Bound the
                # *local* deviation relative to the user budget while keeping
                # the global block sum exact.  At safety=0 this degenerates to
                # fixed Top-K; at safety=1 a row may move within 0.5x..2x.
                lower_ratio = 1.0 - 0.5 * self.adaptive_safety_margin
                upper_ratio = 1.0 + self.adaptive_safety_margin
                minimum_budget = max(
                    self.experimental_minimum_topk,
                    min(nominal_budgets) * lower_ratio,
                )
                maximum_budget = min(
                    1.0,
                    max(nominal_budgets) * upper_ratio,
                )
                minimum = max(1, math.floor(minimum_budget * key_blocks))
                maximum = max(
                    minimum,
                    min(key_blocks, math.ceil(maximum_budget * key_blocks)),
                )
                mass_count.clamp_(minimum, maximum)
                interaction_risk = self._interaction_risk_guard(
                    pooled_q,
                    query_tokens=int(query.shape[0]),
                )
                selected_count = self._budget_adaptive_counts(
                    base_count=base_count,
                    mass_count=mass_count,
                    high_risk=interaction_risk,
                    safety_margin=self.adaptive_safety_margin,
                    minimum=minimum,
                    maximum=maximum,
                ).contiguous()
                # The quota projection above reallocates work but does not
                # itself preserve a local causal neighbourhood.  Reuse the
                # same request-local interaction signal as a compact MTCR
                # rail so contact/occlusion rows keep spatially corresponding
                # evidence even when their global KV budget is redistributed.
            elif self.selection_mode == "disagreement_sentinel":
                # Start from the configured layer/head budget. A second,
                # strictly larger sparse map is evaluated below only as an
                # approximation-error probe; it does not replace the result.
                selected_count = base_count.view(1, heads, 1).expand(
                    scores.shape[0], -1, query_blocks
                ).contiguous()
                boundary_mass = None
                cdf = None
            else:
                cdf = torch.cumsum(sorted_scores.values, dim=-1)
                gather_index = (base_count - 1).view(1, heads, 1, 1).expand(
                    cdf.shape[0], -1, query_blocks, 1
                )
                # Median mass at the fixed-budget boundary becomes the per-head
                # quality target. Searchsorted then gives each query its own count.
                boundary_mass = cdf.gather(-1, gather_index).squeeze(-1)
                mass_target = boundary_mass.median(dim=-1).values
                search_target = mass_target[:, :, None, None].expand(
                    -1, -1, query_blocks, 1
                ).contiguous()
                selected_count = torch.searchsorted(
                    cdf, search_target, right=True
                ).squeeze(-1)
                minimum = torch.ceil(base_count.float() * 0.5).to(torch.int64)
                maximum = torch.ceil(base_count.float() * 2.0).to(torch.int64).clamp_max(
                    key_blocks
                )
                selected_count = torch.maximum(
                    selected_count, minimum.view(1, heads, 1)
                )
                selected_count = torch.minimum(
                    selected_count, maximum.view(1, heads, 1)
                ).contiguous()
            if self.selection_mode == "mass_rebate":
                # Reclaim compute only from query blocks whose proxy attention
                # is more concentrated than their head's median. Diffuse
                # (typically motion/interaction-heavy) queries keep the entire
                # mass-derived count. The configured experimental floor is an
                # absolute safety bound and TCR/prefix protection is applied
                # afterwards, so this cannot erase cross-modal or local-time
                # anchors.
                probability = sorted_scores.values.float().clamp_min(1e-12)
                normalized_entropy = -(
                    probability * probability.log()
                ).sum(-1) / math.log(float(key_blocks))
                median_entropy = normalized_entropy.median(dim=-1).values
                rebate = (
                    normalized_entropy
                    / median_entropy[:, :, None].clamp_min(1e-6)
                ).clamp(0.5, 1.0)
                rebated = torch.ceil(selected_count.float() * rebate).to(torch.int64)
                absolute_floor = max(
                    1, math.ceil(self.experimental_minimum_topk * key_blocks)
                )
                selected_count = rebated.clamp_min(absolute_floor).contiguous()
            block_map = torch.zeros_like(scores, dtype=torch.bool)
            block_map[~expanded_k] = True
            block_map[~expanded_q] = True
            block_map = fill_block_map_triton(
                block_map, selected_count, sorted_scores.indices
            )
            audit_block_map = None
            if self.selection_mode == "disagreement_sentinel":
                audit_count = torch.ceil(base_count.float() * 2.0).to(
                    torch.int64
                ).clamp_max(key_blocks)
                audit_count = audit_count.view(1, heads, 1).expand(
                    scores.shape[0], -1, query_blocks
                ).contiguous()
                audit_block_map = torch.zeros_like(scores, dtype=torch.bool)
                audit_block_map[~expanded_k] = True
                audit_block_map[~expanded_q] = True
                audit_block_map = fill_block_map_triton(
                    audit_block_map, audit_count, sorted_scores.indices
                )
            del pooled_q, pooled_k, scores
            if cdf is not None:
                del cdf
            if boundary_mass is not None:
                del boundary_mass
        protected_k_blocks = (protected_tokens + 63) // 64
        block_map[:, :, :, :protected_k_blocks] = True
        self._protect_temporal_correspondence(
            block_map,
            query_tokens=int(query.shape[0]),
            key_tokens=int(k.shape[-2]),
            protected_tokens=protected_tokens,
            query_token_indices=query_token_indices,
            query_protection_mask=(
                query_protection_mask
                if "query_protection_mask" in locals()
                else None
            ),
        )
        if self.route_probe:
            self._record_route_probe(block_map)
        if "audit_block_map" in locals() and audit_block_map is not None:
            audit_block_map[:, :, :, :protected_k_blocks] = True
            self._protect_temporal_correspondence(
                audit_block_map,
                query_tokens=int(query.shape[0]),
                key_tokens=int(k.shape[-2]),
                protected_tokens=protected_tokens,
                query_token_indices=query_token_indices,
            )
        lut, valid_block_num = block_map_lut_triton(block_map.contiguous())
        output = torch.empty_like(q)
        pv_threshold = hyperparameter_check(50, heads, q.device)
        sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8,
            k_int8,
            v_fp8,
            output,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            k_scale,
            v_scale,
            1,
            0,
            1,
            1.0 / (head_dim**0.5),
            0,
        )
        if "audit_block_map" in locals() and audit_block_map is not None:
            audit_lut, audit_valid_block_num = block_map_lut_triton(
                audit_block_map.contiguous()
            )
            audit_output = torch.empty_like(q)
            sparge_core.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                q_int8,
                k_int8,
                v_fp8,
                audit_output,
                audit_lut,
                audit_valid_block_num,
                pv_threshold,
                q_scale,
                k_scale,
                v_scale,
                1,
                0,
                1,
                1.0 / (head_dim**0.5),
                0,
            )
            # Compare two nested approximations. Rows whose output changes
            # sharply when the sparse evidence set is enlarged are precisely
            # the rows for which the cheap sparse answer is not trustworthy.
            relative = (
                (audit_output.float() - output.float())
                .square()
                .mean(dim=-1)
                .sqrt()
                / audit_output.float().square().mean(dim=-1).sqrt().clamp_min(1e-4)
            )
            token_count = int(query.shape[0])
            query_blocks = int(block_map.shape[-2])
            padded_tokens = query_blocks * 128
            if padded_tokens > token_count:
                import torch.nn.functional as functional

                relative = functional.pad(
                    relative, (0, padded_tokens - token_count), value=0.0
                )
            per_head = relative.view(1, heads, query_blocks, 128).sum(-1)
            valid_counts = torch.full(
                (query_blocks,), 128, device=q.device, dtype=torch.float32
            )
            valid_counts[-1] = token_count - (query_blocks - 1) * 128
            disagreement = (
                per_head / valid_counts.view(1, 1, -1)
            ).median(dim=1).values[0]
            median = disagreement.median()
            mad = (disagreement - median).abs().median().clamp_min(1e-6)
            unstable = disagreement > median + 1.5 * mad
            # Fail closed without allowing a highly noisy request to turn the
            # entire sequence dense. The cap is an initial validation bound;
            # the observed fraction will be calibrated after Human review.
            maximum_blocks = max(1, math.ceil(query_blocks * 0.08))
            if int(unstable.sum()) > maximum_blocks:
                selected_blocks = disagreement.topk(maximum_blocks).indices
                unstable.zero_()
                unstable[selected_blocks] = True
            token_mask = unstable.repeat_interleave(128)[:token_count]
            token_indices = token_mask.nonzero(as_tuple=False).flatten()
            if token_indices.numel():
                dense_query = query.index_select(0, token_indices)
                dense_output = self._dense_prefix(
                    dense_query, k, v_fp8, v_scale, head_dim=head_dim
                )
                output[0, :, token_indices, :] = dense_output.transpose(0, 1)
            self._sentinel_calls += 1
            self._sentinel_dense_query_tokens += int(token_indices.numel())
            self._sentinel_total_query_tokens += token_count
        return rearrange(output, "1 H L D -> L H D")

    def _record_route_probe(self, block_map: torch.Tensor) -> None:
        """Record sampled cross-Actual map overlap without changing output."""

        layer = _ATTENTION_LAYER.get()
        step = _ATTENTION_STEP.get()
        actual_steps = _ATTENTION_ACTUAL_STEPS.get()
        if (
            layer is None
            or step is None
            or actual_steps is None
            or int(step[0]) not in actual_steps
            or block_map.ndim != 4
            or not int(block_map.shape[-2])
        ):
            return
        step_index = int(step[0])
        query_blocks = int(block_map.shape[-2])
        key_blocks = int(block_map.shape[-1])
        heads = int(block_map.shape[1])
        sample_indices = (
            torch.linspace(
                0,
                query_blocks - 1,
                min(16, query_blocks),
                device=block_map.device,
            )
            .round()
            .to(torch.long)
            .unique(sorted=True)
        )
        sampled = block_map.index_select(-2, sample_indices).detach().clone()
        cache_key = (int(layer), query_blocks, key_blocks, heads)
        previous = self._route_probe_previous.get(cache_key)
        if previous is not None:
            previous_step, previous_map = previous
            # Solver step indices restart for every hot request. A
            # non-increasing index is a request boundary, not a reuse chance.
            if step_index > previous_step:
                self._route_probe_records.append(
                    {
                        "previous_step": previous_step,
                        "step": step_index,
                        "step_gap": step_index - previous_step,
                        "layer": int(layer),
                        "query_blocks": query_blocks,
                        "key_blocks": key_blocks,
                        "sampled_query_blocks": int(sample_indices.numel()),
                        "intersection": (sampled & previous_map).sum(
                            dim=(0, 2, 3)
                        ),
                        "union": (sampled | previous_map).sum(dim=(0, 2, 3)),
                    }
                )
        self._route_probe_previous[cache_key] = (step_index, sampled)

    def telemetry(self) -> dict[str, object]:
        total = self._sentinel_total_query_tokens
        causal_total = self._causal_head_guard_total_heads
        route_probe_records: list[dict[str, object]] = []
        if self._route_probe_records:
            intersections = torch.stack(
                [row["intersection"] for row in self._route_probe_records]
            ).float()
            unions = torch.stack(
                [row["union"] for row in self._route_probe_records]
            ).float().clamp_min_(1.0)
            per_head = (intersections / unions).detach().cpu()
            intersections_cpu = intersections.sum(dim=1).detach().cpu()
            unions_cpu = unions.sum(dim=1).detach().cpu()
            for index, source in enumerate(self._route_probe_records):
                head_values = per_head[index]
                route_probe_records.append(
                    {
                        key: value
                        for key, value in source.items()
                        if key not in ("intersection", "union")
                    }
                    | {
                        "global_jaccard": float(
                            intersections_cpu[index] / unions_cpu[index]
                        ),
                        "head_jaccard_min": float(head_values.min()),
                        "head_jaccard_p10": float(
                            torch.quantile(head_values, 0.10)
                        ),
                        "head_jaccard_median": float(head_values.median()),
                    }
                )
        absolute_cap: dict[str, object] | None = None
        if self._absolute_cap_last is not None:
            source = self._absolute_cap_last
            nominal = source["nominal_selected"].detach().cpu()
            capped = source["capped_selected"].detach().cpu()
            selected = source["selected_before_rails"].detach().cpu().float()
            final = source["valid_after_mandatory_rails"].detach().cpu().float()
            activated_source = source.get("cap_activated")
            retained_source = source.get("retained_mass")
            absolute_cap = {
                "key_blocks": int(source["key_blocks"]),
                "query_blocks": int(source["query_blocks"]),
                "nominal_selected_per_head": nominal.tolist(),
                "capped_selected_per_head": capped.tolist(),
                "capped_head_count": int((capped < nominal).sum()),
                "selected_before_rails_min": int(selected.min()),
                "selected_before_rails_mean": float(selected.mean()),
                "selected_before_rails_max": int(selected.max()),
                "valid_after_mandatory_rails_min": int(final.min()),
                "valid_after_mandatory_rails_mean": float(final.mean()),
                "valid_after_mandatory_rails_max": int(final.max()),
            }
            if activated_source is not None and retained_source is not None:
                activated = activated_source.detach().cpu()
                retained = retained_source.detach().cpu().float()
                absolute_cap.update(
                    {
                        "minimum_retained_topk_mass": (
                            self.minimum_retained_topk_mass
                        ),
                        "mass_guard_activated_rows": int(activated.sum()),
                        "mass_guard_total_rows": int(activated.numel()),
                        "mass_guard_activated_fraction": float(
                            activated.float().mean()
                        ),
                        "retained_topk_mass_min": float(retained.min()),
                        "retained_topk_mass_mean": float(retained.mean()),
                        "retained_topk_mass_max": float(retained.max()),
                    }
                )
        mass_guard_profiles: list[dict[str, object]] = []
        if self._mass_guard_records:
            grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
            for row in self._mass_guard_records:
                key = (
                    row["step"],
                    row["layer"],
                    row["key_blocks"],
                    row["selected_blocks"],
                )
                grouped.setdefault(key, []).append(row)
            for (step, layer, key_blocks, selected_blocks), rows in sorted(
                grouped.items(),
                key=lambda item: (
                    -1 if item[0][0] is None else int(item[0][0]),
                    -1 if item[0][1] is None else int(item[0][1]),
                    int(item[0][2]),
                    int(item[0][3]),
                ),
            ):
                activated = int(
                    torch.stack([row["activated"] for row in rows]).sum().cpu()
                )
                total_rows = sum(int(row["total"]) for row in rows)
                retained_sum = float(
                    torch.stack([row["retained_sum"] for row in rows]).sum().cpu()
                )
                retained_min = float(
                    torch.stack([row["retained_min"] for row in rows]).min().cpu()
                )
                retained_max = float(
                    torch.stack([row["retained_max"] for row in rows]).max().cpu()
                )
                threshold_counts = torch.stack(
                    [row["threshold_counts"] for row in rows]
                ).sum(dim=0).detach().cpu()
                mass_guard_profiles.append(
                    {
                        "step": step,
                        "layer": layer,
                        "key_blocks": int(key_blocks),
                        "selected_blocks": int(selected_blocks),
                        "query_chunk_calls": len(rows),
                        "activated_rows": activated,
                        "total_rows": total_rows,
                        "activated_fraction": activated / total_rows,
                        "retained_topk_mass_min": retained_min,
                        "retained_topk_mass_mean": retained_sum / total_rows,
                        "retained_topk_mass_max": retained_max,
                        "activation_fraction_by_retained_mass": {
                            str(threshold): float(threshold_counts[index])
                            / total_rows
                            for index, threshold in enumerate(
                                (0.90, 0.925, 0.95, 0.975, 0.99)
                            )
                        },
                    }
                )
        return {
            "sentinel_calls": self._sentinel_calls,
            "sentinel_dense_query_tokens": self._sentinel_dense_query_tokens,
            "sentinel_total_query_tokens": total,
            "sentinel_dense_query_fraction": (
                self._sentinel_dense_query_tokens / total if total else 0.0
            ),
            "causal_head_guard_calls": self._causal_head_guard_calls,
            "causal_head_guard_dense_heads": self._causal_head_guard_dense_heads,
            "causal_head_guard_total_heads": causal_total,
            "causal_head_guard_dense_fraction": (
                self._causal_head_guard_dense_heads / causal_total
                if causal_total
                else 0.0
            ),
            "route_probe_enabled": self.route_probe,
            "route_probe_record_count": len(route_probe_records),
            "route_probe_records": route_probe_records,
            "absolute_cap_enabled": (
                self.selection_mode
                in (
                    "fixed_topk_absolute_cap",
                    "fixed_topk_mass_guarded_cap",
                    "fixed_topk_mass_probe",
                )
            ),
            "absolute_cap_calls": self._absolute_cap_calls,
            "absolute_cap_last": absolute_cap,
            "mass_guard_probe_only": self.selection_mode == "fixed_topk_mass_probe",
            "mass_guard_profile_count": len(mass_guard_profiles),
            "mass_guard_profiles": mass_guard_profiles,
        }

    def _protect_temporal_correspondence(
        self,
        block_map: torch.Tensor,
        *,
        query_tokens: int,
        key_tokens: int,
        protected_tokens: int,
        query_token_indices: torch.Tensor | None = None,
        query_protection_mask: torch.Tensor | None = None,
    ) -> None:
        """Keep a cheap exact rail between nearby latent frames.

        The rail is deliberately much narrower than an all-to-all local-frame
        window: every 128-row video query block retains the spatially aligned
        64-row key blocks in nearby latent frames, plus a configurable spatial
        block halo.  This targets action-state continuity while adding only a
        small density floor on long sequences.  Unknown or selected-row layouts
        fail closed by leaving the existing TLHB map unchanged.
        """

        radius = self.temporal_correspondence_radius
        layout = _ATTENTION_VIDEO_LAYOUT.get()
        if radius < 0 or layout is None:
            return
        latent_frames, frame_tokens = layout
        expected_video_tokens = latent_frames * frame_tokens
        if key_tokens - protected_tokens != expected_video_tokens:
            return
        selected_queries = query_token_indices is not None
        if selected_queries:
            if (
                query_token_indices.ndim != 1
                or int(query_token_indices.numel()) != query_tokens
                or query_token_indices.device != block_map.device
            ):
                return
        elif query_tokens != expected_video_tokens:
            return

        query_blocks = int(block_map.shape[-2])
        key_blocks = int(block_map.shape[-1])
        device = block_map.device
        if selected_queries:
            # Calibration concatenates complete 128-row query blocks from
            # across the original sequence.  Recover each block's true centre
            # so the same TCR rail used by full inference remains active.
            q_center = torch.arange(query_blocks, device=device, dtype=torch.int64)
            q_center = torch.clamp(q_center * 128 + 64, max=query_tokens - 1)
            q_center = query_token_indices.index_select(0, q_center)
        else:
            q_center = torch.arange(query_blocks, device=device, dtype=torch.int64)
            q_center = torch.clamp(q_center * 128 + 64, max=query_tokens - 1)
        q_frame = torch.div(q_center, frame_tokens, rounding_mode="floor")
        q_spatial = torch.remainder(q_center, frame_tokens)

        k_center = torch.arange(key_blocks, device=device, dtype=torch.int64)
        k_center = k_center * 64 + 32 - protected_tokens
        valid_video_key = (k_center >= 0) & (k_center < expected_video_tokens)
        safe_k_center = k_center.clamp(0, expected_video_tokens - 1)
        k_frame = torch.div(safe_k_center, frame_tokens, rounding_mode="floor")
        k_spatial = torch.remainder(safe_k_center, frame_tokens)

        temporal = (q_frame[:, None] - k_frame[None, :]).abs() <= radius
        # A 128-row query block overlaps two 64-row key blocks.  The base 96
        # token centre distance covers that overlap; each halo unit adds one
        # further key block on either side.
        spatial_distance = 96 + self.temporal_spatial_block_radius * 64
        grid = _ATTENTION_VIDEO_GRID.get()
        row_coherent = os.environ.get(
            "H3_NATIVE_EXPERIMENTAL_MTCR_ROW_COHERENCE", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if row_coherent and grid is not None:
            grid_height, grid_width = grid
            q_row = torch.div(q_spatial, grid_width, rounding_mode="floor")
            k_row = torch.div(k_spatial, grid_width, rounding_mode="floor")
            # A flattened 128-query block spans several complete scanlines.
            # Protecting a row band of equivalent area keeps both sides of a
            # rigid contact (hand/door, rider/vehicle) on one structural rail
            # instead of cutting the relation at a 1-D flattening boundary.
            row_radius = max(0, spatial_distance // grid_width)
            spatial = (q_row[:, None] - k_row[None, :]).abs() <= row_radius
            spatial.logical_and_(
                (q_row[:, None] < grid_height) & (k_row[None, :] < grid_height)
            )
        else:
            spatial = (
                q_spatial[:, None] - k_spatial[None, :]
            ).abs() <= spatial_distance
        rail = temporal & spatial & valid_video_key[None, :]

        interaction_rail = None
        if query_protection_mask is not None:
            if query_protection_mask.shape[-1] != query_blocks:
                raise ValueError("interaction protection mask does not match query blocks")
            # At detected motion discontinuities, explicitly retain a wider
            # local space-time neighbourhood.  This is different from merely
            # increasing Top-K: the extra keys are geometrically tied to the
            # hand/object or vehicle/occluder vicinity and therefore preserve
            # the evidence required for contact causality.
            interaction_temporal = (
                q_frame[:, None] - k_frame[None, :]
            ).abs() <= max(radius, 1)
            interaction_distance = 96 + 3 * 64
            interaction_spatial = (
                q_spatial[:, None] - k_spatial[None, :]
            ).abs() <= interaction_distance
            interaction_rail = (
                interaction_temporal
                & interaction_spatial
                & valid_video_key[None, :]
            )
        # Multi-scale temporal correspondence (MTCR).  Besides the local
        # motion rail above, retain same-location evidence from a sparse set
        # of remote latent frames.  The anchor phase rotates across real H3
        # layers and solver steps, so a block never has to pay for every
        # remote frame while the complete DiT trajectory still observes the
        # whole horizon.  Unlike per-query entropy rebates, neighbouring
        # object regions receive the same structural rule, which avoids
        # discontinuous evidence budgets across rigid bodies and contacts.
        stride = self.temporal_global_anchor_stride
        if stride:
            layer = _ATTENTION_LAYER.get()
            step = _ATTENTION_STEP.get()
            phase = ((layer or 0) + (step[0] if step is not None else 0)) % stride
            anchor_frame = (
                torch.remainder(k_frame, stride) == phase
            ) | (k_frame == 0) | (k_frame == latent_frames - 1)
            global_spatial_distance = (
                96 + self.temporal_global_spatial_block_radius * 64
            )
            if row_coherent and grid is not None:
                global_row_radius = max(0, global_spatial_distance // grid_width)
                global_spatial = (
                    q_row[:, None] - k_row[None, :]
                ).abs() <= global_row_radius
            else:
                global_spatial = (
                    q_spatial[:, None] - k_spatial[None, :]
                ).abs() <= global_spatial_distance
            rail.logical_or_(
                anchor_frame[None, :]
                & global_spatial
                & valid_video_key[None, :]
            )
        block_map.logical_or_(rail[None, None, :, :])
        if interaction_rail is not None:
            block_map.logical_or_(
                query_protection_mask[..., None]
                & interaction_rail[None, None, :, :]
            )

    def selected_queries(
        self,
        prefix_query: torch.Tensor,
        video_query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate protected prefix and selected video queries with shared K/V."""

        if prefix_query.shape[0] != protected_tokens or video_query.shape[0] <= 0:
            raise ValueError("selected attention requires prefix and video queries")
        k, v_fp8, v_scale, heads, _kv_len, head_dim = self._prepare_kv(key, value)
        prefix_output = self._dense_prefix(
            prefix_query, k, v_fp8, v_scale, head_dim=head_dim
        )
        video_output = self._sparse_video_queries(
            video_query,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
            query_token_indices=video_query_indices,
        )
        return torch.cat((prefix_output, video_output), dim=0)

    def selected_video_queries(
        self,
        video_query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate aligned selected video rows without the unused prefix Q."""

        if video_query.shape[0] <= 0:
            raise ValueError("selected video attention requires video queries")
        k, v_fp8, v_scale, heads, _kv_len, head_dim = self._prepare_kv(key, value)
        return self._sparse_video_queries(
            video_query,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
            query_token_indices=video_query_indices,
        )


    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        protected_tokens = _ATTENTION_PROTECTED_PREFIX.get()
        if query.shape[0] < self.minimum_sparse_tokens or protected_tokens <= 0:
            return sage_attention_sm89(query, key, value)
        if protected_tokens >= query.shape[0]:
            return sage_attention_sm89(query, key, value)
        if self.selection_mode == "unified_fixed_topk":
            return self._unified_prefix_and_video(
                query,
                key,
                value,
                protected_tokens=protected_tokens,
            )

        k, v_fp8, v_scale, heads, _kv_len, head_dim = self._prepare_kv(key, value)
        return self._full_from_prepared_kv(
            query,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
        )


class DenseLongSequenceAttentionBackend:
    """Exact per-warp SageAttention with K/V prepared once per H3 layer."""

    approximate = False
    long_sequence_value_dtype = torch.bfloat16
    long_sequence_kv_layout = "NHD"

    @staticmethod
    def resolve_long_sequence_backend(query_tokens: int):
        return _DENSE_LONG_SEQUENCE_BACKEND if query_tokens >= 128 else None

    @staticmethod
    def begin_compact_long_sequence_kv(
        *,
        key_tokens: int,
        heads: int,
        head_dim: int,
        key_mean: torch.Tensor,
        value_absmax: torch.Tensor,
        device: torch.device,
    ) -> _CompactDenseKVBuilder:
        """Start an NHD K/V build that never owns sequence-long BF16 K/V."""

        return _CompactDenseKVBuilder(
            key_tokens=key_tokens,
            heads=heads,
            head_dim=head_dim,
            key_mean=key_mean,
            value_absmax=value_absmax,
            device=device,
        )

    @staticmethod
    def begin_compact_long_sequence_values(
        *,
        key_tokens: int,
        heads: int,
        head_dim: int,
        value_absmax: torch.Tensor,
        device: torch.device,
    ) -> _CompactValueBuilder:
        return _CompactValueBuilder(
            key_tokens=key_tokens,
            heads=heads,
            head_dim=head_dim,
            value_absmax=value_absmax,
            device=device,
            layout="NHD",
        )

    @staticmethod
    def prepare_long_sequence_values(value_hnd: torch.Tensor):
        from sageattention.quant import per_channel_fp8

        if value_hnd.ndim != 4 or value_hnd.shape[0] != 1:
            raise ValueError("long-sequence Dense V must use [1,L,H,D]")
        if value_hnd.dtype != torch.bfloat16 or not value_hnd.is_contiguous():
            raise ValueError("long-sequence Dense V must be contiguous BF16")
        batch, key_tokens, heads, head_dim = value_hnd.shape
        value_fp8, value_scale, _ = per_channel_fp8(
            value_hnd,
            tensor_layout="NHD",
            scale_max=2.25,
            smooth_v=False,
        )
        return value_fp8, value_scale, heads, key_tokens, head_dim

    @staticmethod
    def prepare_long_sequence_keys(
        key_hnd: torch.Tensor,
        value_fp8: torch.Tensor,
        value_scale: torch.Tensor,
    ) -> PreparedLongSequenceDenseKV:
        if key_hnd.ndim != 4 or key_hnd.shape[0] != 1:
            raise ValueError("long-sequence Dense K must use [1,L,H,D]")
        if key_hnd.dtype != torch.bfloat16 or not key_hnd.is_contiguous():
            raise ValueError("long-sequence Dense K must be contiguous BF16")
        from sageattention import _fused

        batch, key_tokens, heads, head_dim = key_hnd.shape
        key_mean = key_hnd.mean(dim=1, keepdim=True)
        key_int8 = torch.empty_like(key_hnd, dtype=torch.int8)
        key_scale = torch.empty(
            (batch, heads, (key_tokens + 63) // 64),
            device=key_hnd.device,
            dtype=torch.float32,
        )
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(
            key_hnd,
            key_mean.squeeze(1),
            key_int8,
            key_scale,
            64,
            0,
        )
        return PreparedLongSequenceDenseKV(
            key_int8=key_int8,
            key_scale=key_scale,
            value_fp8=value_fp8,
            value_scale=value_scale,
            heads=heads,
            key_tokens=key_tokens,
            head_dim=head_dim,
        )

    @staticmethod
    def _queries(
        query: torch.Tensor,
        prepared: PreparedLongSequenceDenseKV,
    ) -> torch.Tensor:
        from sageattention import _fused, sm89_compile

        if _DENSE_QK_QUANT_GRAN.get() != "per_warp":
            raise RuntimeError(
                "long-sequence Dense Attention requires the validated per_warp Q/K path"
            )
        if (
            query.ndim != 3
            or query.shape[1] != prepared.heads
            or query.shape[2] != prepared.head_dim
        ):
            raise ValueError("long-sequence Dense Query does not match K/V")
        q = query.unsqueeze(0).contiguous()
        query_tokens = int(q.shape[1])
        q_int8 = torch.empty_like(q, dtype=torch.int8)
        q_scale = torch.empty(
            (1, prepared.heads, ((query_tokens + 127) // 128) * 4),
            device=q.device,
            dtype=torch.float32,
        )
        _fused.quant_per_warp_int8_cuda(
            q, q_int8, q_scale, 128, 32, 0
        )
        output = torch.empty_like(q)
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8,
            prepared.key_int8,
            prepared.value_fp8,
            output,
            q_scale,
            prepared.key_scale,
            prepared.value_scale,
            0,
            0,
            2,
            1.0 / (prepared.head_dim**0.5),
            0,
        )
        return output.squeeze(0)

    def long_sequence_prefix_queries(
        self,
        query: torch.Tensor,
        prepared: PreparedLongSequenceDenseKV,
    ) -> torch.Tensor:
        return self._queries(query, prepared)

    def long_sequence_all_queries(
        self,
        query: torch.Tensor,
        prepared: PreparedLongSequenceDenseKV,
    ) -> torch.Tensor:
        """Keep per-warp Query groups aligned to the packed sequence origin."""

        return self._queries(query, prepared)

    def long_sequence_video_queries(
        self,
        query: torch.Tensor,
        prepared: PreparedLongSequenceDenseKV,
        *,
        protected_tokens: int,
        query_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        del protected_tokens, query_token_indices
        return self._queries(query, prepared)


_DENSE_LONG_SEQUENCE_BACKEND = DenseLongSequenceAttentionBackend()


class BudgetConstrainedAdaptiveSpargeAttentionBackend:
    """Allocate one exact sparse-Attention quota across the 50 H3 layers.

    ``compute_budget`` fixes the mean discretionary video-to-video KV-block
    fraction across one complete DiT pass.  ``safety_margin`` interpolates
    between a uniform fixed-TopK pass and a quality endpoint which spends the
    same quota on the layers implicated by the Round142 Dense teacher and the
    Human-accepted Round143 short trajectory.  Round144 is only the measured
    long-shape counterpart and has not received a Human long-video verdict.

    The first actual solver step measures one Dense-vs-uniform-sparse Sentinel
    error per layer.  Starting with the next actual step, that request-local
    signal may reorder layers *inside* the historical causal/non-causal
    strata.  It cannot demote the causal stratum, alter the global quota,
    change sampler steps, or touch model weights.  Query- and Head-local
    redistribution are deliberately absent: Round192--195 showed that those
    locally plausible proxies lost perceptual quality at identical compute.
    """

    approximate = True

    _LAYER_COUNT = 50
    # The Sparge route has fixed pooling/sorting/LUT costs.  Above this point
    # it is cheaper and more faithful to execute an exact Dense call than a
    # "sparse" call that retains almost every KV block.  The allocator must
    # therefore choose between an effective sparse action and Dense; linear
    # interpolation through (0.55, 1.0) is an execution dead zone.
    _MAX_EFFECTIVE_SPARSE_FRACTION = 0.55
    _CAUSAL_LAYERS = frozenset(
        (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45)
    )
    # mean_relative_l1 + (1 - mean_cosine), averaged over the five real-H3
    # Round142 Dense-teacher probe steps.  The immutable tuple is a release
    # policy artifact; runtime/calibration is evidence, not a production
    # dependency.  Round143 Human review supplies the short-trajectory gate;
    # Round144 supplies long-shape cost evidence only.
    _HISTORICAL_LAYER_ERROR = (
        0.19735881, 0.16457795, 0.20810783, 0.18412527, 0.26709484,
        0.21820146, 0.19451363, 0.20862162, 0.19643372, 0.21397598,
        0.20390691, 0.20447088, 0.20043558, 0.23121054, 0.23520514,
        0.24395207, 0.22512252, 0.23215841, 0.23089739, 0.25343191,
        0.25895908, 0.25210435, 0.29666663, 0.29324402, 0.32281117,
        0.26390720, 0.24859637, 0.27722647, 0.29786527, 0.28681173,
        0.37878144, 0.41549087, 0.41095664, 0.41671199, 0.44921617,
        0.40792523, 0.37116305, 0.34510818, 0.39216599, 0.50580860,
        0.49588941, 0.47659179, 0.49364795, 0.37016587, 0.28380505,
        0.31804628, 0.22515019, 0.27512362, 0.20794795, 0.22286341,
    )

    def __init__(
        self,
        compute_budget: float,
        *,
        safety_margin: float = 0.65,
        temporal_correspondence_radius: int = 1,
        temporal_spatial_block_radius: int = 1,
        sentinel_query_blocks: int = 7,
    ) -> None:
        if not 0.0625 <= compute_budget <= 1.0:
            raise ValueError("adaptive compute budget must lie inside [0.0625, 1]")
        if not 0.0 <= safety_margin <= 1.0:
            raise ValueError("adaptive safety margin must lie inside [0, 1]")
        if sentinel_query_blocks < 3:
            raise ValueError("adaptive attention needs at least three sentinels")
        self.compute_budget = float(compute_budget)
        self.safety_margin = float(safety_margin)
        self.temporal_correspondence_radius = int(
            temporal_correspondence_radius
        )
        self.temporal_spatial_block_radius = int(
            temporal_spatial_block_radius
        )
        self.sentinel_query_blocks = int(sentinel_query_blocks)
        self._trajectory_counts: dict[
            tuple[int, int, int, tuple[int, ...]],
            dict[int, tuple[int, ...]],
        ] = {}
        self._task_errors: dict[
            tuple[int, int, int, tuple[int, ...]], dict[int, float]
        ] = {}
        self._task_adapted: set[
            tuple[int, int, int, tuple[int, ...]]
        ] = set()
        self._backends: dict[
            tuple[float, ...], SplitModalityProtectedSpargeAttentionBackend
        ] = {}
        self._telemetry_rows: list[dict[str, object]] = []
        self._schedule_rows: list[dict[str, object]] = []

    def __deepcopy__(self, memo):
        # Both block-offload device slots belong to one sequential request and
        # must observe the same request-local calibration profiles.
        memo[id(self)] = self
        return self

    def _backend(
        self, budgets: tuple[float, ...]
    ) -> SplitModalityProtectedSpargeAttentionBackend:
        backend = self._backends.get(budgets)
        if backend is None:
            backend = SplitModalityProtectedSpargeAttentionBackend(
                budgets,
                experimental_minimum_topk=0.0625,
                temporal_correspondence_radius=self.temporal_correspondence_radius,
                temporal_spatial_block_radius=self.temporal_spatial_block_radius,
                temporal_global_anchor_stride=(
                    8 if self.safety_margin > 0.0 else 0
                ),
                temporal_global_spatial_block_radius=0,
                selection_mode=(
                    "interaction_hybrid"
                    if self.safety_margin > 0.0
                    else "fixed_topk"
                ),
            )
            self._backends[budgets] = backend
        return backend

    def _base_budgets(self, heads: int) -> tuple[float, ...]:
        return (self.compute_budget,) * heads

    @classmethod
    def _priority_order(
        cls, task_error: tuple[float, ...] | None = None
    ) -> tuple[int, ...]:
        historical = torch.tensor(cls._HISTORICAL_LAYER_ERROR)
        score = (historical - historical.median()) / (
            (historical - historical.median()).abs().median().clamp_min(1e-6)
        )
        if task_error is not None:
            current = torch.tensor(task_error)
            current = (current - current.median()) / (
                (current - current.median()).abs().median().clamp_min(1e-6)
            )
            # Historical evidence owns the fail-safe stratum; the current
            # request only refines ordering within a stratum.
            score = 0.65 * score + 0.35 * current
        causal = sorted(cls._CAUSAL_LAYERS, key=lambda i: float(score[i]), reverse=True)
        other = sorted(
            set(range(cls._LAYER_COUNT)) - cls._CAUSAL_LAYERS,
            key=lambda i: float(score[i]),
            reverse=True,
        )
        return tuple(causal + other)

    @classmethod
    def solve_layer_counts(
        cls,
        compute_budget: float,
        safety_margin: float,
        *,
        key_blocks: int,
        task_error: tuple[float, ...] | None = None,
    ) -> tuple[int, ...]:
        """Project the exact pass-wide quota onto a quality-prioritized route."""

        if key_blocks < 1:
            raise ValueError("adaptive layer solver needs at least one KV block")
        if task_error is not None and len(task_error) != cls._LAYER_COUNT:
            raise ValueError("task layer error does not match the H3 layer count")
        base = max(1, min(key_blocks, int(compute_budget * key_blocks)))
        target = base * cls._LAYER_COUNT
        minimum = max(1, int(0.0625 * key_blocks))

        endpoint = [minimum] * cls._LAYER_COUNT
        remaining = target - minimum * cls._LAYER_COUNT
        for layer in cls._priority_order(task_error):
            if remaining <= 0:
                break
            addition = min(key_blocks - minimum, remaining)
            endpoint[layer] += addition
            remaining -= addition

        desired = torch.tensor(
            [
                (1.0 - safety_margin) * base + safety_margin * endpoint[layer]
                for layer in range(cls._LAYER_COUNT)
            ],
            dtype=torch.float32,
        ).view(1, cls._LAYER_COUNT, 1)
        projected = SplitModalityProtectedSpargeAttentionBackend._project_counts_to_exact_budget(
            desired,
            torch.tensor([target], dtype=torch.int64),
            minimum=minimum,
            maximum=key_blocks,
        ).view(cls._LAYER_COUNT)
        return tuple(int(value) for value in projected.tolist())

    @classmethod
    def _trajectory_priority_order(
        cls,
        actual_steps: tuple[int, ...],
        task_error: tuple[float, ...] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        layer_order = cls._priority_order(task_error)
        layer_rank = {layer: rank for rank, layer in enumerate(layer_order)}
        first = actual_steps[0]
        recovery = frozenset(actual_steps[-min(3, len(actual_steps)):])

        def stratum(item: tuple[int, int]) -> tuple[int, int, int]:
            step, layer = item
            if layer in cls._CAUSAL_LAYERS:
                group = 0
            elif step == first:
                group = 1
            elif step in recovery:
                group = 2
            else:
                group = 3
            return group, layer_rank[layer], step

        cells = (
            (step, layer)
            for step in actual_steps
            for layer in range(cls._LAYER_COUNT)
        )
        return tuple(sorted(cells, key=stratum))

    @classmethod
    def solve_trajectory_counts(
        cls,
        compute_budget: float,
        safety_margin: float,
        *,
        key_blocks: int,
        actual_steps: tuple[int, ...],
        task_error: tuple[float, ...] | None = None,
        target_total: int | None = None,
    ) -> dict[int, tuple[int, ...]]:
        """Allocate one exact quota across solver phase × H3 layer.

        The priority topology is learned/validated, but the amount of work is
        never hidden: the returned integer counts sum to ``target_total`` (or
        ``floor(compute_budget * key_blocks) * steps * 50``).  The quality
        endpoint first protects every causal-layer call, then the opening
        trajectory anchor, then the three terminal recovery calls.
        """

        if not actual_steps or tuple(sorted(set(actual_steps))) != actual_steps:
            raise ValueError("trajectory budget needs sorted unique actual steps")
        entries = len(actual_steps) * cls._LAYER_COUNT
        base = max(1, min(key_blocks, int(compute_budget * key_blocks)))
        requested = base * entries if target_total is None else int(target_total)
        minimum = max(1, int(0.0625 * key_blocks))
        requested = max(minimum * entries, min(key_blocks * entries, requested))

        endpoint = {
            (step, layer): minimum
            for step in actual_steps
            for layer in range(cls._LAYER_COUNT)
        }
        remaining = requested - minimum * entries
        ranked = cls._trajectory_priority_order(actual_steps, task_error)
        first = actual_steps[0]
        recovery = frozenset(actual_steps[-min(3, len(actual_steps)):])
        groups = (
            tuple(cell for cell in ranked if cell[1] in cls._CAUSAL_LAYERS),
            tuple(
                cell
                for cell in ranked
                if cell[0] == first and cell[1] not in cls._CAUSAL_LAYERS
            ),
            tuple(
                cell
                for cell in ranked
                if cell[0] in recovery
                and cell[0] != first
                and cell[1] not in cls._CAUSAL_LAYERS
            ),
            tuple(
                cell
                for cell in ranked
                if cell[0] != first
                and cell[0] not in recovery
                and cell[1] not in cls._CAUSAL_LAYERS
            ),
        )
        # A risk stratum is a distributed relation-carrying band, not a list
        # of isolated winners.  Sparse work is water-filled only inside the
        # empirically useful sparse range.  If the target crosses that range,
        # cells are promoted to exact Dense actions instead of creating the
        # pathological near-Dense sparse calls observed in Round210/212.
        capacity = key_blocks - minimum
        sparse_maximum = max(
            minimum,
            min(
                key_blocks - 1,
                int(cls._MAX_EFFECTIVE_SPARSE_FRACTION * key_blocks),
            ),
        )
        sparse_capacity = sparse_maximum - minimum
        for group in groups:
            if remaining <= 0 or not group:
                break
            group_capacity = capacity * len(group)
            allocation = min(remaining, group_capacity)
            if allocation == group_capacity:
                for cell in group:
                    endpoint[cell] = key_blocks
                remaining -= allocation
                continue
            if allocation <= sparse_capacity * len(group):
                shared, remainder = divmod(allocation, len(group))
                for cell in group:
                    endpoint[cell] += shared
                for cell in group[:remainder]:
                    endpoint[cell] += 1
                remaining -= allocation
                continue

            # Find the closest representable mix of Dense cells and uniformly
            # water-filled effective-sparse cells.  Scanning at most 600
            # trajectory cells is negligible and makes the non-linear kernel
            # cost model explicit instead of pretending selected blocks map
            # linearly to latency.
            best: tuple[int, int, int] | None = None
            for dense_cells in range(len(group) + 1):
                sparse_cells = len(group) - dense_cells
                residual = allocation - dense_cells * capacity
                sparse_allocation = max(
                    0, min(sparse_cells * sparse_capacity, residual)
                )
                represented = dense_cells * capacity + sparse_allocation
                candidate = (
                    abs(represented - allocation),
                    -dense_cells,
                    sparse_allocation,
                )
                if best is None or candidate < best:
                    best = candidate
            assert best is not None
            dense_cells = -best[1]
            sparse_allocation = best[2]
            for cell in group[:dense_cells]:
                endpoint[cell] = key_blocks
            sparse_group = group[dense_cells:]
            if sparse_group:
                shared, remainder = divmod(
                    sparse_allocation, len(sparse_group)
                )
                for cell in sparse_group:
                    endpoint[cell] += shared
                for cell in sparse_group[:remainder]:
                    endpoint[cell] += 1
            represented = dense_cells * capacity + sparse_allocation
            remaining -= represented

        uniform = requested / entries
        ordered_cells = tuple(
            (step, layer)
            for step in actual_steps
            for layer in range(cls._LAYER_COUNT)
        )
        desired = torch.tensor(
            [
                (1.0 - safety_margin) * uniform
                + safety_margin * endpoint[cell]
                for cell in ordered_cells
            ],
            dtype=torch.float32,
        ).view(1, entries, 1)
        projected = SplitModalityProtectedSpargeAttentionBackend._project_counts_to_exact_budget(
            desired,
            torch.tensor([requested], dtype=torch.int64),
            minimum=minimum,
            maximum=key_blocks,
        ).view(entries).tolist()
        result: dict[int, tuple[int, ...]] = {}
        offset = 0
        for step in actual_steps:
            result[step] = tuple(
                int(value)
                for value in projected[offset : offset + cls._LAYER_COUNT]
            )
            offset += cls._LAYER_COUNT
        return result

    @classmethod
    def solve_trajectory_head_counts(
        cls,
        compute_budget: float,
        safety_margin: float,
        *,
        key_blocks: int,
        heads: int,
        actual_steps: tuple[int, ...],
    ) -> dict[int, tuple[tuple[int, ...], ...]]:
        """Project one exact request quota over step, layer and head.

        The quality endpoint reproduces the Round143 topology (Human-accepted
        at 720p5; Round144 is only its measured 720p15 cost counterpart):
        opening step and causal layers are Dense, ordinary non-causal calls
        keep the 6.25% floor, and terminal recovery gives the 21 measured
        causal heads 12.5% versus 10% for the remaining heads.  A single
        projection after flattening the whole request avoids shape-dependent
        double rounding of the public compute budget.
        """

        if key_blocks < 1 or heads < 1:
            raise ValueError("trajectory head budget needs positive dimensions")
        if not actual_steps or tuple(sorted(set(actual_steps))) != actual_steps:
            raise ValueError("trajectory head budget needs sorted unique actual steps")
        if not 0.0625 <= compute_budget <= 1.0:
            raise ValueError("trajectory head compute budget is outside [0.0625, 1]")
        if not 0.0 <= safety_margin <= 1.0:
            raise ValueError("trajectory head safety is outside [0, 1]")

        first = actual_steps[0]
        recovery = frozenset(actual_steps[-min(3, len(actual_steps)):])
        minimum = max(1, int(0.0625 * key_blocks))
        uniform = compute_budget * key_blocks
        desired: list[float] = []
        for step in actual_steps:
            for layer in range(cls._LAYER_COUNT):
                for head in range(heads):
                    if step == first or layer in cls._CAUSAL_LAYERS:
                        endpoint = float(key_blocks)
                    elif step in recovery:
                        endpoint = (
                            0.125 * key_blocks
                            if heads == len(_H3_HEAD_RISK_TIERS)
                            and _H3_HEAD_RISK_TIERS[head] == 2
                            else 0.100 * key_blocks
                        )
                    else:
                        endpoint = float(minimum)
                    desired.append(
                        (1.0 - safety_margin) * uniform
                        + safety_margin * endpoint
                    )

        entries = len(desired)
        target = round(compute_budget * key_blocks * entries)
        projected = SplitModalityProtectedSpargeAttentionBackend._project_counts_to_exact_budget(
            torch.tensor(desired, dtype=torch.float32).view(1, entries, 1),
            torch.tensor([target], dtype=torch.int64),
            minimum=minimum,
            maximum=key_blocks,
        ).view(entries).tolist()
        result: dict[int, tuple[tuple[int, ...], ...]] = {}
        offset = 0
        for step in actual_steps:
            layers = []
            for _layer in range(cls._LAYER_COUNT):
                layers.append(
                    tuple(int(value) for value in projected[offset : offset + heads])
                )
                offset += heads
            result[step] = tuple(layers)
        return result

    def _head_budgets_for_count(
        self, count: int, *, key_blocks: int, heads: int
    ) -> tuple[float, ...]:
        if heads != len(_H3_HEAD_RISK_TIERS) or self.safety_margin == 0.0:
            budget = self._count_to_budget(count, key_blocks)
            return (budget,) * heads
        tiers = torch.tensor(_H3_HEAD_RISK_TIERS, dtype=torch.float32)
        desired = count * (
            1.0 + 0.08 * self.safety_margin * (tiers - tiers.mean())
        )
        minimum = max(1, int(0.0625 * key_blocks))
        counts = SplitModalityProtectedSpargeAttentionBackend._project_counts_to_exact_budget(
            desired.view(1, heads, 1),
            torch.tensor([count * heads], dtype=torch.int64),
            minimum=minimum,
            maximum=key_blocks,
        ).view(heads)
        return tuple(
            self._count_to_budget(int(value), key_blocks)
            for value in counts.tolist()
        )

    @staticmethod
    def _count_to_budget(count: int, key_blocks: int) -> float:
        if count >= key_blocks:
            return 1.0
        # For shapes where 6.25% is fractional (for example 1565 KV blocks),
        # the legal integer floor may round-trip to 0.0623.  Clamp the public
        # ratio to the declared floor; downstream ``int(ratio * key_blocks)``
        # still selects the same integer count.
        return max(0.0625, float((count + 0.5) / key_blocks))

    def _sentinel_indices(
        self, video_tokens: int, device: torch.device
    ) -> torch.Tensor:
        block_count = (video_tokens + 127) // 128
        chosen = min(self.sentinel_query_blocks, block_count)
        blocks = torch.linspace(
            0, block_count - 1, steps=chosen, device=device
        ).round().long().unique(sorted=True)
        return torch.cat(
            tuple(
                torch.arange(
                    int(block) * 128,
                    min((int(block) + 1) * 128, video_tokens),
                    device=device,
                    dtype=torch.long,
                )
                for block in blocks
            )
        )

    def _measure_layer_error(
        self,
        *,
        layer: int,
        video_query: torch.Tensor,
        k: torch.Tensor,
        v_fp8: torch.Tensor,
        v_scale: torch.Tensor,
        protected_tokens: int,
        heads: int,
        head_dim: int,
    ) -> float:
        started = time.perf_counter()
        base = self._backend(self._base_budgets(heads))
        indices = self._sentinel_indices(video_query.shape[0], video_query.device)
        sentinel = video_query.index_select(0, indices)
        dense = base._dense_prefix(sentinel, k, v_fp8, v_scale, head_dim=head_dim)
        sparse = base._sparse_video_queries(
            sentinel,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
            query_token_indices=indices,
        )
        left = sparse.float().flatten()
        right = dense.float().flatten()
        relative_l1 = (left - right).abs().sum() / right.abs().sum().clamp_min(1e-12)
        cosine_loss = 1.0 - torch.nn.functional.cosine_similarity(
            left.view(1, -1), right.view(1, -1), dim=1
        )[0]
        error = float((relative_l1 + cosine_loss.clamp_min(0.0)).detach().cpu())
        self._telemetry_rows.append(
            {
                "layer": layer,
                "video_tokens": int(video_query.shape[0]),
                "sentinel_query_tokens": int(indices.numel()),
                "uniform_probe_budget": self.compute_budget,
                "error": error,
                "calibration_seconds": time.perf_counter() - started,
            }
        )
        return error

    def _record_schedule(
        self,
        *,
        source: str,
        shape_key: tuple[int, int, int, tuple[int, ...]],
        counts: dict[int, tuple[int, ...]],
    ) -> None:
        key_blocks = shape_key[1]
        flattened = [
            count
            for step in shape_key[3]
            for count in counts[step]
        ]
        self._schedule_rows.append(
            {
                "source": source,
                "video_tokens": shape_key[0],
                "key_blocks": key_blocks,
                "actual_steps": list(shape_key[3]),
                "mean_budget": sum(flattened) / (len(flattened) * key_blocks),
                "min_budget": min(flattened) / key_blocks,
                "max_budget": max(flattened) / key_blocks,
                "dense_layers_by_step": {
                    str(step): [
                        layer
                        for layer, count in enumerate(counts[step])
                        if count == key_blocks
                    ]
                    for step in shape_key[3]
                },
                "layer_counts_by_step": {
                    str(step): list(counts[step])
                    for step in shape_key[3]
                },
            }
        )

    @classmethod
    def _quality_topology_saturated(
        cls,
        trajectory: dict[int, tuple[int, ...]],
        *,
        actual_steps: tuple[int, ...],
        key_blocks: int,
    ) -> bool:
        """Return whether request-local ranking can no longer change safety.

        Once the opening anchor and every causal-layer call are Dense, the
        remaining movable cells are terminal/non-causal detail rebates.  A
        50-layer Dense Sentinel sweep cannot improve the protected topology
        and costs about 40 seconds on 720p15, so the controller stops itself.
        """

        return all(
            count == key_blocks for count in trajectory[actual_steps[0]]
        ) and all(
            trajectory[step][layer] == key_blocks
            for step in actual_steps
            for layer in cls._CAUSAL_LAYERS
        )

    def __call__(self, query, key, value):
        protected_tokens = _ATTENTION_PROTECTED_PREFIX.get()
        layer = _ATTENTION_LAYER.get()
        step = _ATTENTION_STEP.get()
        actual_steps = _ATTENTION_ACTUAL_STEPS.get()
        if (
            query.shape[0] < 128
            or protected_tokens <= 0
            or protected_tokens >= query.shape[0]
            or layer is None
            or step is None
            or actual_steps is None
            or step[0] not in actual_steps
            or _ATTENTION_VIDEO_LAYOUT.get() is None
        ):
            return sage_attention_sm89(query, key, value)

        if step[0] == actual_steps[0] and layer == 0:
            self._trajectory_counts.clear()
            self._task_errors.clear()
            self._task_adapted.clear()
            self._telemetry_rows.clear()
            self._schedule_rows.clear()

        base = self._backend(self._base_budgets(query.shape[1]))
        k, v_fp8, v_scale, heads, _kv_len, head_dim = base._prepare_kv(key, value)
        video_query = query[protected_tokens:]
        # Sparge pools KV tokens in BLKK=64 blocks before applying Top-K.  The
        # quota must therefore be solved in pooled-block units, not raw token
        # units (the v2 prototype accidentally mixed these two domains).
        key_blocks = (int(k.shape[-2]) + 63) // 64
        shape_key = (int(video_query.shape[0]), key_blocks, heads, actual_steps)
        trajectory = self._trajectory_counts.get(shape_key)
        if trajectory is None:
            trajectory = self.solve_trajectory_counts(
                self.compute_budget,
                self.safety_margin,
                key_blocks=shape_key[1],
                actual_steps=actual_steps,
            )
            self._trajectory_counts[shape_key] = trajectory
            self._task_errors[shape_key] = {}
            self._record_schedule(
                source="historical_teacher_prior",
                shape_key=shape_key,
                counts=trajectory,
            )

        errors = self._task_errors[shape_key]
        topology_saturated = self._quality_topology_saturated(
            trajectory,
            actual_steps=actual_steps,
            key_blocks=key_blocks,
        )
        if (
            self.safety_margin > 0.0
            and not topology_saturated
            and step[0] == actual_steps[0]
            and layer in self._CAUSAL_LAYERS
            and layer not in errors
        ):
            errors[layer] = self._measure_layer_error(
                layer=layer,
                video_query=video_query,
                k=k,
                v_fp8=v_fp8,
                v_scale=v_scale,
                protected_tokens=protected_tokens,
                heads=heads,
                head_dim=head_dim,
            )

        if (
            step[0] != actual_steps[0]
            and layer == 0
            and shape_key not in self._task_adapted
            and len(errors) == len(self._CAUSAL_LAYERS)
        ):
            task_error_values = list(self._HISTORICAL_LAYER_ERROR)
            for index, error in errors.items():
                task_error_values[index] = error
            task_error = tuple(task_error_values)
            original = self._trajectory_counts[shape_key]
            task_order = self._priority_order(task_error)
            trajectory = {actual_steps[0]: original[actual_steps[0]]}
            for actual_step in actual_steps[1:]:
                # Never cross the causal/non-causal safety boundary.  The
                # request signal only assigns the existing layer vectors to
                # harder peers inside the same historical stratum.
                adapted: list[int | None] = [None] * self._LAYER_COUNT
                for causal in (True, False):
                    targets = [
                        layer for layer in task_order
                        if (layer in self._CAUSAL_LAYERS) == causal
                    ]
                    values = sorted(
                        (
                            original[actual_step][layer]
                            for layer in range(self._LAYER_COUNT)
                            if (layer in self._CAUSAL_LAYERS) == causal
                        ),
                        reverse=True,
                    )
                    for target_layer, value in zip(targets, values):
                        adapted[target_layer] = value
                trajectory[actual_step] = tuple(adapted)
            self._trajectory_counts[shape_key] = trajectory
            self._task_adapted.add(shape_key)
            self._record_schedule(
                source="request_adapted_after_first_actual_step",
                shape_key=shape_key,
                counts=trajectory,
            )

        count = self._trajectory_counts[shape_key][step[0]][layer]
        if count >= shape_key[1]:
            return sage_attention_sm89(query, key, value)

        prefix = base._dense_prefix(
            query[:protected_tokens], k, v_fp8, v_scale, head_dim=head_dim
        )
        budgets = self._head_budgets_for_count(
            count, key_blocks=shape_key[1], heads=heads
        )
        video = self._backend(budgets)._sparse_video_queries(
            video_query,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
        )
        return torch.cat((prefix, video), dim=0)

    def selected_queries(self, *args, **kwargs):
        return sage_attention_sm89(
            torch.cat((args[0], args[1]), dim=0), args[2], args[3]
        )

    def selected_video_queries(self, *args, **kwargs):
        return sage_attention_sm89(args[0], args[1], args[2])

    def telemetry(self) -> dict[str, object]:
        return {
            "policy": "budget_constrained_task_adaptive_sparse_attention_v4",
            "compute_budget": self.compute_budget,
            "safety_margin": self.safety_margin,
            "budget_scope": "discretionary_video_to_video_kv_blocks",
            "budget_invariant": "exact_across_actual_steps_times_50_layers",
            "mandatory_floor": "dense_conditioning_prefix_plus_mtcr_plus_interaction_rail",
            "adaptation_signal": "round142_teacher_prior_plus_first_step_layer_sentinel_error",
            "adaptive_stop_rule": "skip_sentinel_when_anchor_and_causal_topology_are_dense",
            "calibration_count": len(self._telemetry_rows),
            "calibration_seconds": sum(
                float(row["calibration_seconds"]) for row in self._telemetry_rows
            ),
            "profiles": list(self._telemetry_rows),
            "schedules": list(self._schedule_rows),
        }


class LayerSensitivityRoutedSplitSpargeAttentionBackend:
    """Use an aggressive budget only outside a measured sensitive layer set.

    This is an experimental layer-sensitivity probe. Unknown layer context
    always uses the accepted 0.50 budget and the product planner never selects
    this backend automatically.
    """

    approximate = True

    def __init__(
        self,
        *,
        aggressive_topk: float | tuple[float, ...],
        sensitive_layers: tuple[int, ...],
        safe_topk: float | tuple[float, ...] = 0.50,
        experimental_minimum_topk: float = 0.25,
        temporal_correspondence_radius: int = -1,
        temporal_spatial_block_radius: int = 0,
        temporal_global_anchor_stride: int = 0,
        temporal_global_spatial_block_radius: int = 0,
        selection_mode: str = "fixed_topk",
    ) -> None:
        aggressive_budgets = (
            (float(aggressive_topk),)
            if isinstance(aggressive_topk, (float, int))
            else tuple(float(value) for value in aggressive_topk)
        )
        safe_budgets = (
            (float(safe_topk),)
            if isinstance(safe_topk, (float, int))
            else tuple(float(value) for value in safe_topk)
        )
        if len(aggressive_budgets) != len(safe_budgets) or any(
            not experimental_minimum_topk <= aggressive < safe <= 1.0
            for aggressive, safe in zip(aggressive_budgets, safe_budgets)
        ):
            raise ValueError("layer-routed sparse budgets are not ordered")
        if tuple(sorted(set(sensitive_layers))) != sensitive_layers or any(
            not 0 <= layer < 50 for layer in sensitive_layers
        ):
            raise ValueError("sensitive layers must be sorted and unique in [0, 50)")
        # The causal-head guard is a quality repair for measured sensitive
        # layers, not another global sparsity policy.  Keep ordinary layers on
        # the accepted fixed route and invoke online head protection only in
        # ``self.safe`` below.
        aggressive_selection_mode = (
            "fixed_topk" if selection_mode == "causal_head_guard" else selection_mode
        )
        self.aggressive = SplitModalityProtectedSpargeAttentionBackend(
            aggressive_topk,
            experimental_minimum_topk=experimental_minimum_topk,
            temporal_correspondence_radius=temporal_correspondence_radius,
            temporal_spatial_block_radius=temporal_spatial_block_radius,
            temporal_global_anchor_stride=temporal_global_anchor_stride,
            temporal_global_spatial_block_radius=(
                temporal_global_spatial_block_radius
            ),
            selection_mode=aggressive_selection_mode,
        )
        self.safe = SplitModalityProtectedSpargeAttentionBackend(
            safe_topk,
            experimental_minimum_topk=experimental_minimum_topk,
            temporal_correspondence_radius=temporal_correspondence_radius,
            temporal_spatial_block_radius=temporal_spatial_block_radius,
            temporal_global_anchor_stride=temporal_global_anchor_stride,
            temporal_global_spatial_block_radius=(
                temporal_global_spatial_block_radius
            ),
            selection_mode=selection_mode,
        )
        self.sensitive_layers = frozenset(sensitive_layers)

    def _backend(self):
        layer = current_attention_layer()
        return (
            self.safe
            if layer is None or layer in self.sensitive_layers
            else self.aggressive
        )

    def protected_queries(self, query, key, value):
        """Preserve the segment-cache fast path under the routed policy."""

        return self._backend().protected_queries(query, key, value)

    def selected_queries(
        self,
        prefix_query,
        video_query,
        key,
        value,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor | None = None,
    ):
        """Preserve shared-K/V evaluation for selected active video rows."""

        backend = self._backend()
        if video_query_indices is None:
            return backend.selected_queries(
                prefix_query,
                video_query,
                key,
                value,
                protected_tokens=protected_tokens,
            )
        return backend.selected_queries(
            prefix_query,
            video_query,
            key,
            value,
            protected_tokens=protected_tokens,
            video_query_indices=video_query_indices,
        )

    def selected_video_queries(
        self,
        video_query,
        key,
        value,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor,
    ):
        return self._backend().selected_video_queries(
            video_query,
            key,
            value,
            protected_tokens=protected_tokens,
            video_query_indices=video_query_indices,
        )

    def full_with_exact_sample(self, query, key, value, *, sample_indices):
        return self._backend().full_with_exact_sample(
            query, key, value, sample_indices=sample_indices
        )

    def __call__(self, query, key, value):
        return self._backend()(query, key, value)

    def telemetry(self) -> dict[str, object]:
        reports = (self.aggressive.telemetry(), self.safe.telemetry())
        calls = sum(int(report["sentinel_calls"]) for report in reports)
        dense = sum(
            int(report["sentinel_dense_query_tokens"]) for report in reports
        )
        total = sum(
            int(report["sentinel_total_query_tokens"]) for report in reports
        )
        causal_calls = sum(
            int(report.get("causal_head_guard_calls", 0)) for report in reports
        )
        causal_dense = sum(
            int(report.get("causal_head_guard_dense_heads", 0)) for report in reports
        )
        causal_total = sum(
            int(report.get("causal_head_guard_total_heads", 0)) for report in reports
        )
        return {
            "sentinel_calls": calls,
            "sentinel_dense_query_tokens": dense,
            "sentinel_total_query_tokens": total,
            "sentinel_dense_query_fraction": dense / total if total else 0.0,
            "causal_head_guard_calls": causal_calls,
            "causal_head_guard_dense_heads": causal_dense,
            "causal_head_guard_total_heads": causal_total,
            "causal_head_guard_dense_fraction": (
                causal_dense / causal_total if causal_total else 0.0
            ),
        }


class LayerHeadBudgetOverrideBackend:
    """Protect teacher-identified causal heads without densifying whole layers.

    The dense-island experiment demonstrated that H3 motion causality is
    concentrated in a small mid/late-layer band, but routing every head in
    that band to dense attention is too expensive for a 15-second request.
    This wrapper consumes an explicit, auditable phase/layer budget map and
    overrides only those real H3 layers.  All unspecified layers and phases
    preserve the established trajectory backend exactly.

    The map is intentionally produced by an offline dense-teacher probe.  It
    is not a product-facing quality knob and it never changes the requested
    solver schedule.
    """

    approximate = True

    def __init__(
        self,
        fallback,
        *,
        default_layer_topks: dict[int, tuple[float, ...]] | None = None,
        anchor_layer_topks: dict[int, tuple[float, ...]] | None = None,
        recovery_layer_topks: dict[int, tuple[float, ...]] | None = None,
        default_step_indices: tuple[int, ...] = (),
        anchor_step_indices: tuple[int, ...] = (),
        recovery_step_indices: tuple[int, ...] = (),
        experimental_minimum_topk: float = 0.0625,
        temporal_correspondence_radius: int = -1,
        temporal_spatial_block_radius: int = 0,
        temporal_global_anchor_stride: int = 0,
        temporal_global_spatial_block_radius: int = 0,
        selection_mode: str = "fixed_topk",
    ) -> None:
        if tuple(sorted(set(anchor_step_indices))) != anchor_step_indices:
            raise ValueError("teacher anchor steps must be sorted and unique")
        if tuple(sorted(set(recovery_step_indices))) != recovery_step_indices:
            raise ValueError("teacher recovery steps must be sorted and unique")
        if set(anchor_step_indices) & set(recovery_step_indices):
            raise ValueError("teacher anchor and recovery steps must be disjoint")
        if tuple(sorted(set(default_step_indices))) != default_step_indices:
            raise ValueError("teacher default steps must be sorted and unique")
        if (
            set(default_step_indices) & set(anchor_step_indices)
            or set(default_step_indices) & set(recovery_step_indices)
        ):
            raise ValueError("teacher phase step sets must be disjoint")

        def build(
            raw: dict[int, tuple[float, ...]] | None,
        ) -> dict[int, SplitModalityProtectedSpargeAttentionBackend]:
            result = {}
            for layer, budgets in (raw or {}).items():
                layer = int(layer)
                budgets = tuple(float(value) for value in budgets)
                if not 0 <= layer < 50:
                    raise ValueError("teacher budget layer must be inside [0, 50)")
                if len(budgets) != 56 or any(
                    not experimental_minimum_topk <= value <= 1.0
                    for value in budgets
                ):
                    raise ValueError(
                        "teacher layer budget requires 56 head values inside "
                        "the configured sparse envelope"
                    )
                result[layer] = SplitModalityProtectedSpargeAttentionBackend(
                    budgets,
                    experimental_minimum_topk=experimental_minimum_topk,
                    temporal_correspondence_radius=temporal_correspondence_radius,
                    temporal_spatial_block_radius=temporal_spatial_block_radius,
                    temporal_global_anchor_stride=temporal_global_anchor_stride,
                    temporal_global_spatial_block_radius=(
                        temporal_global_spatial_block_radius
                    ),
                    selection_mode=selection_mode,
                )
            return result

        self.fallback = fallback
        self.default_layers = build(default_layer_topks)
        self.anchor_layers = build(anchor_layer_topks)
        self.recovery_layers = build(recovery_layer_topks)
        self.default_step_indices = frozenset(default_step_indices)
        self.anchor_step_indices = frozenset(anchor_step_indices)
        self.recovery_step_indices = frozenset(recovery_step_indices)

    def _backend(self):
        layer = current_attention_layer()
        step = current_attention_step()
        if layer is None or step is None:
            return self.fallback
        index = step[0]
        if index in self.anchor_step_indices:
            return self.anchor_layers.get(layer, self.fallback)
        if index in self.recovery_step_indices:
            return self.recovery_layers.get(layer, self.fallback)
        if self.default_step_indices and index not in self.default_step_indices:
            return self.fallback
        return self.default_layers.get(layer, self.fallback)

    def __call__(self, query, key, value):
        return self._backend()(query, key, value)

    def protected_queries(self, query, key, value):
        backend = self._backend()
        method = getattr(backend, "protected_queries", None)
        if method is None:
            return backend(query, key, value)
        return method(query, key, value)

    def selected_queries(
        self,
        prefix_query,
        video_query,
        key,
        value,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor | None = None,
    ):
        backend = self._backend()
        method = getattr(backend, "selected_queries", None)
        if method is None:
            return backend(torch.cat((prefix_query, video_query), dim=0), key, value)
        return method(
            prefix_query,
            video_query,
            key,
            value,
            protected_tokens=protected_tokens,
            video_query_indices=video_query_indices,
        )

    def telemetry(self) -> dict[str, object]:
        reports = []
        fallback_telemetry = getattr(self.fallback, "telemetry", None)
        if fallback_telemetry is not None:
            reports.append(fallback_telemetry())
        for backend in {
            id(value): value
            for value in (
                *self.default_layers.values(),
                *self.anchor_layers.values(),
                *self.recovery_layers.values(),
            )
        }.values():
            reports.append(backend.telemetry())
        calls = sum(int(report.get("sentinel_calls", 0)) for report in reports)
        dense = sum(
            int(report.get("sentinel_dense_query_tokens", 0))
            for report in reports
        )
        total = sum(
            int(report.get("sentinel_total_query_tokens", 0))
            for report in reports
        )
        return {
            "sentinel_calls": calls,
            "sentinel_dense_query_tokens": dense,
            "sentinel_total_query_tokens": total,
            "sentinel_dense_query_fraction": dense / total if total else 0.0,
        }


class TrajectoryLayerModalityRoutedSpargeAttentionBackend:
    """Protect critical solver steps and measured sensitive H3 layers.

    The policy is deliberately three-dimensional rather than a global top-k
    knob: packed conditioning/audio queries remain dense inside the split
    backend, calibration-sensitive DiT layers use the conservative sparse
    budget, and critical early/late solver steps return to dense SageAttention.
    Missing request context always fails closed to dense execution.
    """

    approximate = True

    def __init__(
        self,
        *,
        aggressive_topk: float | tuple[float, ...],
        sensitive_layers: tuple[int, ...],
        dense_step_indices: tuple[int, ...],
        safe_topk: float | tuple[float, ...] = 0.50,
        anchor_step_indices: tuple[int, ...] = (),
        anchor_aggressive_topk: float | tuple[float, ...] | None = None,
        anchor_safe_topk: float | tuple[float, ...] | None = None,
        recovery_step_indices: tuple[int, ...] = (),
        recovery_aggressive_topk: float | tuple[float, ...] | None = None,
        recovery_safe_topk: float | tuple[float, ...] | None = None,
        experimental_minimum_topk: float = 0.25,
        minimum_sparse_tokens: int = 128,
        temporal_correspondence_radius: int = -1,
        temporal_spatial_block_radius: int = 0,
        temporal_global_anchor_stride: int = 0,
        temporal_global_spatial_block_radius: int = 0,
        selection_mode: str = "fixed_topk",
        request_guarded: bool = False,
    ) -> None:
        if tuple(sorted(set(dense_step_indices))) != dense_step_indices or any(
            step < 0 for step in dense_step_indices
        ):
            raise ValueError("dense solver steps must be sorted, unique and non-negative")
        if minimum_sparse_tokens <= 0:
            raise ValueError("minimum sparse token count must be positive")
        if tuple(sorted(set(anchor_step_indices))) != anchor_step_indices or any(
            step < 0 for step in anchor_step_indices
        ):
            raise ValueError("anchor solver steps must be sorted, unique and non-negative")
        if set(anchor_step_indices) & set(dense_step_indices):
            raise ValueError("anchor and dense solver steps must be disjoint")
        if bool(anchor_step_indices) != bool(
            anchor_aggressive_topk is not None and anchor_safe_topk is not None
        ):
            raise ValueError(
                "anchor steps require both aggressive and safe anchor budgets"
            )
        if tuple(sorted(set(recovery_step_indices))) != recovery_step_indices or any(
            step < 0 for step in recovery_step_indices
        ):
            raise ValueError("recovery solver steps must be sorted, unique and non-negative")
        if (
            set(recovery_step_indices) & set(dense_step_indices)
            or set(recovery_step_indices) & set(anchor_step_indices)
        ):
            raise ValueError("recovery, anchor and dense solver steps must be disjoint")
        if bool(recovery_step_indices) != bool(
            recovery_aggressive_topk is not None and recovery_safe_topk is not None
        ):
            raise ValueError(
                "recovery steps require both aggressive and safe recovery budgets"
            )
        self.layer_policy = LayerSensitivityRoutedSplitSpargeAttentionBackend(
            aggressive_topk=aggressive_topk,
            sensitive_layers=sensitive_layers,
            safe_topk=safe_topk,
            experimental_minimum_topk=experimental_minimum_topk,
            temporal_correspondence_radius=temporal_correspondence_radius,
            temporal_spatial_block_radius=temporal_spatial_block_radius,
            temporal_global_anchor_stride=temporal_global_anchor_stride,
            temporal_global_spatial_block_radius=(
                temporal_global_spatial_block_radius
            ),
            selection_mode=selection_mode,
        )
        self.anchor_policy = (
            LayerSensitivityRoutedSplitSpargeAttentionBackend(
                aggressive_topk=anchor_aggressive_topk,
                sensitive_layers=sensitive_layers,
                safe_topk=anchor_safe_topk,
                experimental_minimum_topk=experimental_minimum_topk,
                temporal_correspondence_radius=temporal_correspondence_radius,
                temporal_spatial_block_radius=temporal_spatial_block_radius,
                temporal_global_anchor_stride=temporal_global_anchor_stride,
                temporal_global_spatial_block_radius=(
                    temporal_global_spatial_block_radius
                ),
                selection_mode=selection_mode,
            )
            if anchor_step_indices
            else None
        )
        self.recovery_policy = (
            LayerSensitivityRoutedSplitSpargeAttentionBackend(
                aggressive_topk=recovery_aggressive_topk,
                sensitive_layers=sensitive_layers,
                safe_topk=recovery_safe_topk,
                experimental_minimum_topk=experimental_minimum_topk,
                temporal_correspondence_radius=temporal_correspondence_radius,
                temporal_spatial_block_radius=temporal_spatial_block_radius,
                temporal_global_anchor_stride=temporal_global_anchor_stride,
                temporal_global_spatial_block_radius=(
                    temporal_global_spatial_block_radius
                ),
                selection_mode=selection_mode,
            )
            if recovery_step_indices
            else None
        )
        self.dense_step_indices = frozenset(dense_step_indices)
        self.anchor_step_indices = frozenset(anchor_step_indices)
        self.recovery_step_indices = frozenset(recovery_step_indices)
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)
        self.request_guarded = bool(request_guarded)

    def _use_dense(self, query: torch.Tensor) -> bool:
        if _ATTENTION_FORCE_DENSE.get():
            return True
        if self.request_guarded and not _LONG_VIDEO_ATTENTION_ENABLED.get():
            return True
        if query.shape[0] < self.minimum_sparse_tokens:
            return True
        step = _ATTENTION_STEP.get()
        return step is None or step[0] in self.dense_step_indices

    def __call__(self, query, key, value):
        if self._use_dense(query):
            return sage_attention_sm89(query, key, value)
        return self._sparse_policy()(query, key, value)

    def _sparse_policy(self):
        step = _ATTENTION_STEP.get()
        if (
            self.anchor_policy is not None
            and step is not None
            and step[0] in self.anchor_step_indices
        ):
            return self.anchor_policy
        if (
            self.recovery_policy is not None
            and step is not None
            and step[0] in self.recovery_step_indices
        ):
            return self.recovery_policy
        return self.layer_policy

    def protected_queries(self, query, key, value):
        if self._use_dense(query):
            return sage_attention_sm89(query, key, value)
        return self._sparse_policy().protected_queries(query, key, value)

    def selected_queries(
        self,
        prefix_query,
        video_query,
        key,
        value,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor | None = None,
    ):
        combined = torch.cat((prefix_query, video_query), dim=0)
        if self._use_dense(combined):
            return sage_attention_sm89(combined, key, value)
        return self._sparse_policy().selected_queries(
            prefix_query,
            video_query,
            key,
            value,
            protected_tokens=protected_tokens,
            video_query_indices=video_query_indices,
        )

    def selected_video_queries(
        self,
        video_query,
        key,
        value,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor,
    ):
        # The routing decision is request/step based.  Do not feed the small
        # probe query count into ``_use_dense`` because the original request
        # is still a long sparse-eligible sequence.
        if _ATTENTION_FORCE_DENSE.get() or (
            self.request_guarded and not _LONG_VIDEO_ATTENTION_ENABLED.get()
        ):
            return sage_attention_sm89(video_query, key, value)
        step = _ATTENTION_STEP.get()
        if step is None or step[0] in self.dense_step_indices:
            return sage_attention_sm89(video_query, key, value)
        return self._sparse_policy().selected_video_queries(
            video_query,
            key,
            value,
            protected_tokens=protected_tokens,
            video_query_indices=video_query_indices,
        )

    def full_with_exact_sample(self, query, key, value, *, sample_indices):
        if _ATTENTION_FORCE_DENSE.get() or (
            self.request_guarded and not _LONG_VIDEO_ATTENTION_ENABLED.get()
        ):
            full = sage_attention_sm89(query, key, value)
            exact = sage_attention_sm89(
                query.index_select(0, sample_indices), key, value
            )
            return full, exact
        step = _ATTENTION_STEP.get()
        if step is None or step[0] in self.dense_step_indices:
            full = sage_attention_sm89(query, key, value)
            exact = sage_attention_sm89(
                query.index_select(0, sample_indices), key, value
            )
            return full, exact
        return self._sparse_policy().full_with_exact_sample(
            query, key, value, sample_indices=sample_indices
        )

    def telemetry(self) -> dict[str, object]:
        policies = [self.layer_policy]
        if self.anchor_policy is not None:
            policies.append(self.anchor_policy)
        if self.recovery_policy is not None:
            policies.append(self.recovery_policy)
        reports = [policy.telemetry() for policy in policies]
        calls = sum(int(report["sentinel_calls"]) for report in reports)
        dense = sum(
            int(report["sentinel_dense_query_tokens"]) for report in reports
        )
        total = sum(
            int(report["sentinel_total_query_tokens"]) for report in reports
        )
        causal_calls = sum(
            int(report.get("causal_head_guard_calls", 0)) for report in reports
        )
        causal_dense = sum(
            int(report.get("causal_head_guard_dense_heads", 0)) for report in reports
        )
        causal_total = sum(
            int(report.get("causal_head_guard_total_heads", 0)) for report in reports
        )
        return {
            "sentinel_calls": calls,
            "sentinel_dense_query_tokens": dense,
            "sentinel_total_query_tokens": total,
            "sentinel_dense_query_fraction": dense / total if total else 0.0,
            "causal_head_guard_calls": causal_calls,
            "causal_head_guard_dense_heads": causal_dense,
            "causal_head_guard_total_heads": causal_total,
            "causal_head_guard_dense_fraction": (
                causal_dense / causal_total if causal_total else 0.0
            ),
        }


class QualityConstrainedAdaptiveSpargeAttentionBackend:
    """Auto-select the fastest per-layer/head budget inside an accepted envelope.

    The fixed TLHB research route proved that H3 heads, layers and solver phases
    have very different sparse-attention risk.  Its remaining weakness is that
    the budgets were selected by hand.  This backend turns that accepted route
    into a *constraint*, not another exposed knob:

    * a few evenly-spaced video query blocks are evaluated against dense output;
    * the accepted TLHB budget defines the maximum local error for every head;
    * the smallest measured budget that is no worse on both relative-L1 and
      cosine is selected independently for every true H3 layer and phase;
    * the full request still protects all conditioning/audio keys and the TCR
      same-location temporal rail;
    * missing step/layer/layout context fails closed to the accepted envelope.

    Calibration state is shared by the two deep-copied block-offload slots and
    is reusable for subsequent hot requests with the same packed shape.  It is
    deliberately not user configurable: callers choose a quality preset while
    this policy solves the internal compute allocation.
    """

    approximate = True

    _SENSITIVE_LAYERS = frozenset(
        (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45)
    )
    # Risk tiers are derived from the real-H3 Round13 head calibration and are
    # the compact provenance of the accepted Round57 TLHB envelope.
    _HEAD_RISK_TIERS = _H3_HEAD_RISK_TIERS
    # Discrete block fractions understood by the current Sparge kernel.  These
    # are the solver's search lattice, not product quality controls.
    _CANDIDATE_BUDGETS = (
        0.0625,
        0.075,
        0.0875,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
    )

    def __init__(
        self,
        *,
        temporal_correspondence_radius: int = 1,
        temporal_spatial_block_radius: int = 1,
        sentinel_query_blocks: int = 7,
    ) -> None:
        if sentinel_query_blocks < 3:
            raise ValueError("adaptive attention needs at least three sentinel blocks")
        self.temporal_correspondence_radius = int(
            temporal_correspondence_radius
        )
        self.temporal_spatial_block_radius = int(
            temporal_spatial_block_radius
        )
        self.sentinel_query_blocks = int(sentinel_query_blocks)
        self._profiles: dict[tuple, tuple[float, ...]] = {}
        self._backends: dict[tuple[float, ...], SplitModalityProtectedSpargeAttentionBackend] = {}
        self._reference_backends: dict[
            tuple[float, ...], SplitModalityProtectedSpargeAttentionBackend
        ] = {}
        self._telemetry: dict[tuple, dict[str, object]] = {}

    def __deepcopy__(self, memo):
        # DoubleBufferBlockExecutor owns two graph copies, but a calibration
        # learned in one physical slot must be visible when the next true H3
        # layer is executed in the other slot.
        memo[id(self)] = self
        return self

    @classmethod
    def _accepted_budget(cls, phase: str, layer: int) -> tuple[float, ...]:
        sensitive = layer in cls._SENSITIVE_LAYERS
        if phase == "anchor":
            offset = 0.45 if sensitive else 0.35
            return tuple(
                round(offset + 0.05 * tier, 6) for tier in cls._HEAD_RISK_TIERS
            )
        if phase == "recovery":
            offset = 0.30 if sensitive else 0.25
            return tuple(
                round(offset + (0.05 if tier == 2 else 0.0), 6)
                for tier in cls._HEAD_RISK_TIERS
            )
        if phase == "cruise":
            if not sensitive:
                return (0.125,) * len(cls._HEAD_RISK_TIERS)
            return tuple(
                0.175 if tier == 2 else 0.15 for tier in cls._HEAD_RISK_TIERS
            )
        raise ValueError(f"unknown adaptive attention phase: {phase}")

    @staticmethod
    def _phase() -> str | None:
        step = _ATTENTION_STEP.get()
        if step is None:
            return None
        index, count = step
        if index == 0:
            return "anchor"
        if index >= max(1, count - 3):
            return "recovery"
        return "cruise"

    def _backend(
        self, budgets: tuple[float, ...]
    ) -> SplitModalityProtectedSpargeAttentionBackend:
        backend = self._backends.get(budgets)
        if backend is None:
            backend = SplitModalityProtectedSpargeAttentionBackend(
                budgets,
                experimental_minimum_topk=0.0625,
                temporal_correspondence_radius=self.temporal_correspondence_radius,
                temporal_spatial_block_radius=self.temporal_spatial_block_radius,
            )
            self._backends[budgets] = backend
        return backend

    def _reference_backend(
        self, budgets: tuple[float, ...]
    ) -> SplitModalityProtectedSpargeAttentionBackend:
        """Accepted Round57 math, intentionally without the newer TCR rail."""

        backend = self._reference_backends.get(budgets)
        if backend is None:
            backend = SplitModalityProtectedSpargeAttentionBackend(
                budgets,
                experimental_minimum_topk=0.0625,
                temporal_correspondence_radius=-1,
            )
            self._reference_backends[budgets] = backend
        return backend

    def _sentinel_indices(self, video_tokens: int, device: torch.device) -> torch.Tensor:
        block_count = (video_tokens + 127) // 128
        chosen = min(self.sentinel_query_blocks, block_count)
        blocks = torch.linspace(
            0, block_count - 1, steps=chosen, device=device
        ).round().long().unique(sorted=True)
        return torch.cat(
            tuple(
                torch.arange(
                    int(block) * 128,
                    min((int(block) + 1) * 128, video_tokens),
                    device=device,
                    dtype=torch.long,
                )
                for block in blocks
            )
        )

    @staticmethod
    def _head_error(
        reference: torch.Tensor, candidate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left = candidate.float().permute(1, 0, 2).flatten(1)
        right = reference.float().permute(1, 0, 2).flatten(1)
        relative_l1 = (
            (left - right).abs().sum(1) / right.abs().sum(1).clamp_min(1e-12)
        )
        cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
        return relative_l1, cosine

    @classmethod
    def solve_head_budgets(
        cls,
        accepted_budgets: tuple[float, ...],
        accepted_l1: torch.Tensor,
        accepted_cosine: torch.Tensor,
        candidates: tuple[tuple[float, torch.Tensor, torch.Tensor], ...],
    ) -> tuple[float, ...]:
        """Minimize total blocks under an accepted aggregate-error envelope.

        The accepted route is feasible by construction.  Unlike the former
        per-head test, this solver may spend more compute on one difficult head
        and reclaim more from several easy heads.  No head may become worse
        than the worst head already present in the accepted layer, while total
        relative-L1 and total cosine loss must both remain no worse.
        """

        heads = len(accepted_budgets)
        if accepted_l1.numel() != heads or accepted_cosine.numel() != heads:
            raise ValueError("accepted adaptive metrics do not match head count")
        accepted_l1_values = accepted_l1.detach().float().cpu().tolist()
        accepted_cosine_values = accepted_cosine.detach().float().cpu().tolist()
        worst_l1 = max(accepted_l1_values) * 1.0005 + 1e-7
        worst_cosine = min(accepted_cosine_values) - 1e-6
        choices: list[list[tuple[float, float, float]]] = [
            [
                (
                    float(accepted_budgets[head]),
                    float(accepted_l1_values[head]),
                    float(accepted_cosine_values[head]),
                )
            ]
            for head in range(heads)
        ]
        for budget, l1, cosine in candidates:
            if l1.numel() != heads or cosine.numel() != heads:
                raise ValueError("candidate adaptive metrics do not match head count")
            l1_values = l1.detach().float().cpu().tolist()
            cosine_values = cosine.detach().float().cpu().tolist()
            for head in range(heads):
                if l1_values[head] <= worst_l1 and cosine_values[head] >= worst_cosine:
                    choices[head].append(
                        (float(budget), l1_values[head], cosine_values[head])
                    )

        # Remove duplicate budgets and start from the cheapest locally safe
        # point for every head.
        normalized: list[list[tuple[float, float, float]]] = []
        for head_choices in choices:
            by_budget: dict[float, tuple[float, float, float]] = {}
            for choice in head_choices:
                previous = by_budget.get(choice[0])
                if previous is None or (choice[1], -choice[2]) < (
                    previous[1],
                    -previous[2],
                ):
                    by_budget[choice[0]] = choice
            normalized.append([by_budget[key] for key in sorted(by_budget)])

        selected_indices = [0] * heads
        l1_limit = sum(accepted_l1_values) * 1.0005 + 1e-7
        cosine_loss_limit = sum(1.0 - value for value in accepted_cosine_values)
        cosine_loss_limit = cosine_loss_limit * 1.0005 + 1e-7

        def violation(indices: list[int]) -> float:
            l1_sum = sum(normalized[h][indices[h]][1] for h in range(heads))
            cosine_loss = sum(
                1.0 - normalized[h][indices[h]][2] for h in range(heads)
            )
            return max(0.0, l1_sum / l1_limit - 1.0) + max(
                0.0, cosine_loss / cosine_loss_limit - 1.0
            )

        current_violation = violation(selected_indices)
        while current_violation > 1e-9:
            best: tuple[float, int, int, float] | None = None
            for head, head_choices in enumerate(normalized):
                current = selected_indices[head]
                current_budget = head_choices[current][0]
                for index in range(current + 1, len(head_choices)):
                    extra = head_choices[index][0] - current_budget
                    if extra <= 0.0:
                        continue
                    trial = list(selected_indices)
                    trial[head] = index
                    trial_violation = violation(trial)
                    gain = current_violation - trial_violation
                    if gain <= 0.0:
                        continue
                    score = gain / extra
                    candidate = (score, head, index, trial_violation)
                    if best is None or candidate > best:
                        best = candidate
            if best is None:
                return accepted_budgets
            _, head, index, current_violation = best
            selected_indices[head] = index

        return tuple(
            normalized[head][selected_indices[head]][0] for head in range(heads)
        )

    def _calibrate(
        self,
        *,
        phase: str,
        layer: int,
        video_query: torch.Tensor,
        k: torch.Tensor,
        v_fp8: torch.Tensor,
        v_scale: torch.Tensor,
        protected_tokens: int,
        heads: int,
        head_dim: int,
    ) -> tuple[float, ...]:
        started = time.perf_counter()
        indices = self._sentinel_indices(video_query.shape[0], video_query.device)
        sentinel = video_query.index_select(0, indices)
        # The same accepted Split kernel supplies both the dense reference and
        # the quality envelope, so the solver cannot benefit from comparing two
        # unrelated quantization paths.
        reference_backend = self._reference_backend(
            self._accepted_budget(phase, layer)
        )
        dense = reference_backend._dense_prefix(
            sentinel, k, v_fp8, v_scale, head_dim=head_dim
        )
        accepted_budgets = self._accepted_budget(phase, layer)
        accepted = reference_backend._sparse_video_queries(
            sentinel,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
        )
        accepted_l1, accepted_cosine = self._head_error(dense, accepted)

        candidate_rows: list[tuple[float, torch.Tensor, torch.Tensor]] = []
        for budget in self._CANDIDATE_BUDGETS:
            candidate_backend = self._backend((budget,) * heads)
            output = candidate_backend._sparse_video_queries(
                sentinel,
                k,
                v_fp8,
                v_scale,
                protected_tokens=protected_tokens,
                heads=heads,
                head_dim=head_dim,
                query_token_indices=indices,
            )
            l1, cosine = self._head_error(dense, output)
            candidate_rows.append((budget, l1, cosine))
            del output

        selected = self.solve_head_budgets(
            accepted_budgets,
            accepted_l1,
            accepted_cosine,
            tuple(candidate_rows),
        )
        key = (
            phase,
            layer,
            int(video_query.shape[0]),
            int(k.shape[-2]),
            int(protected_tokens),
            heads,
        )
        self._telemetry[key] = {
            "phase": phase,
            "layer": layer,
            "video_tokens": int(video_query.shape[0]),
            "protected_tokens": int(protected_tokens),
            "sentinel_query_tokens": int(indices.numel()),
            "accepted_mean_budget": sum(accepted_budgets) / heads,
            "selected_mean_budget": sum(selected) / heads,
            "selected_min_budget": min(selected),
            "selected_max_budget": max(selected),
            "reduced_heads": sum(
                candidate < accepted
                for candidate, accepted in zip(selected, accepted_budgets)
            ),
            "calibration_seconds": time.perf_counter() - started,
        }
        return selected

    def telemetry(self) -> dict[str, object]:
        rows = [self._telemetry[key] for key in sorted(self._telemetry)]
        accepted = sum(float(row["accepted_mean_budget"]) for row in rows)
        selected = sum(float(row["selected_mean_budget"]) for row in rows)
        return {
            "policy": "quality_constrained_adaptive_tlhb_tcr",
            "quality_envelope": "accepted_round57_local_error",
            "profile_count": len(rows),
            "mean_budget_ratio_vs_accepted": (
                selected / accepted if accepted else 1.0
            ),
            "calibration_seconds": sum(
                float(row["calibration_seconds"]) for row in rows
            ),
            "profiles": rows,
        }

    def __call__(self, query, key, value):
        protected_tokens = _ATTENTION_PROTECTED_PREFIX.get()
        layer = _ATTENTION_LAYER.get()
        phase = self._phase()
        layout = _ATTENTION_VIDEO_LAYOUT.get()
        if (
            query.shape[0] < 128
            or protected_tokens <= 0
            or protected_tokens >= query.shape[0]
            or layer is None
            or phase is None
            or layout is None
        ):
            return sage_attention_sm89(query, key, value)

        accepted_budgets = self._accepted_budget(phase, layer)
        accepted_backend = self._backend(accepted_budgets)
        k, v_fp8, v_scale, heads, _kv_len, head_dim = accepted_backend._prepare_kv(
            key, value
        )
        prefix = accepted_backend._dense_prefix(
            query[:protected_tokens], k, v_fp8, v_scale, head_dim=head_dim
        )
        video_query = query[protected_tokens:]
        profile_key = (
            phase,
            layer,
            int(video_query.shape[0]),
            int(k.shape[-2]),
            int(protected_tokens),
            heads,
        )
        budgets = self._profiles.get(profile_key)
        if budgets is None:
            budgets = self._calibrate(
                phase=phase,
                layer=layer,
                video_query=video_query,
                k=k,
                v_fp8=v_fp8,
                v_scale=v_scale,
                protected_tokens=protected_tokens,
                heads=heads,
                head_dim=head_dim,
            )
            self._profiles[profile_key] = budgets
        video = self._backend(budgets)._sparse_video_queries(
            video_query,
            k,
            v_fp8,
            v_scale,
            protected_tokens=protected_tokens,
            heads=heads,
            head_dim=head_dim,
        )
        return torch.cat((prefix, video), dim=0)

    def protected_queries(self, query, key, value):
        # Segment-cache research routes are deliberately not composed with the
        # adaptive policy.  Keep their protected refresh exact if called.
        return sage_attention_sm89(query, key, value)

    def selected_queries(
        self,
        prefix_query,
        video_query,
        key,
        value,
        *,
        protected_tokens: int,
        video_query_indices: torch.Tensor | None = None,
    ):
        return sage_attention_sm89(
            torch.cat((prefix_query, video_query), dim=0), key, value
        )


@lru_cache(maxsize=2)
def _load_experimental_sol_attn(source: str):
    """Load the standalone research kernel without importing its ComfyUI node."""

    root = Path(source).resolve()
    if not (root / "_int8_fwd.py").is_file():
        raise FileNotFoundError(f"Sol-Attn source is incomplete: {root}")
    package_name = f"_h3_native_sol_attn_{abs(hash(str(root))):x}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}._int8_fwd").sol_attn_int8


class TrajectoryLayerModalityRoutedSolAttentionBackend:
    """Experimental H3 policy around Sol-Attn's corrected dynamic routing.

    Unlike fixed-ratio Sparge Top-K, ``tau`` is a query-dependent threshold:
    increasing it routes fewer exact KV blocks while skipped blocks retain a
    centroid correction.  The H3 packed conditioning/audio prefix is kept exact
    both as KV and as query rows.  Step and layer policy fails closed to dense
    Sage whenever request context is absent.
    """

    approximate = True

    def __init__(
        self,
        *,
        source: str | Path,
        tau: float,
        sensitive_tau: float,
        sensitive_layers: tuple[int, ...],
        anchor_step_indices: tuple[int, ...] = (),
        anchor_tau: float | None = None,
        recovery_step_indices: tuple[int, ...] = (),
        recovery_tau: float | None = None,
        minimum_sparse_tokens: int = 4096,
    ) -> None:
        values = (tau, sensitive_tau) + tuple(
            value for value in (anchor_tau, recovery_tau) if value is not None
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("Sol-Attn tau values must be positive")
        if tuple(sorted(set(sensitive_layers))) != sensitive_layers:
            raise ValueError("sensitive layers must be sorted and unique")
        for name, steps, value in (
            ("anchor", anchor_step_indices, anchor_tau),
            ("recovery", recovery_step_indices, recovery_tau),
        ):
            if tuple(sorted(set(steps))) != steps or any(step < 0 for step in steps):
                raise ValueError(f"{name} steps must be sorted, unique and non-negative")
            if bool(steps) != (value is not None):
                raise ValueError(f"{name} steps and tau must be configured together")
        if set(anchor_step_indices) & set(recovery_step_indices):
            raise ValueError("anchor and recovery steps must be disjoint")
        self.source = str(Path(source).resolve())
        self.tau = float(tau)
        self.sensitive_tau = float(sensitive_tau)
        self.sensitive_layers = frozenset(sensitive_layers)
        self.anchor_step_indices = frozenset(anchor_step_indices)
        self.anchor_tau = None if anchor_tau is None else float(anchor_tau)
        self.recovery_step_indices = frozenset(recovery_step_indices)
        self.recovery_tau = None if recovery_tau is None else float(recovery_tau)
        self.minimum_sparse_tokens = int(minimum_sparse_tokens)

    def _current_tau(self) -> float | None:
        step = _ATTENTION_STEP.get()
        layer = _ATTENTION_LAYER.get()
        if step is None or layer is None:
            return None
        if step[0] in self.anchor_step_indices:
            base = self.anchor_tau
        elif step[0] in self.recovery_step_indices:
            base = self.recovery_tau
        else:
            base = self.tau
        assert base is not None
        return min(base, self.sensitive_tau) if layer in self.sensitive_layers else base

    def __call__(self, query, key, value):
        tau = self._current_tau()
        protected_tokens = _ATTENTION_PROTECTED_PREFIX.get()
        if (
            tau is None
            or query.shape[0] < self.minimum_sparse_tokens
            or query.shape != key.shape
            or key.shape != value.shape
            or protected_tokens <= 0
        ):
            return sage_attention_sm89(query, key, value)
        block_count = (protected_tokens + 63) // 64
        kernel = _load_experimental_sol_attn(self.source)
        output = kernel(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            tau=tau,
            sink_blocks=(0, block_count),
            sink_q=(0, block_count),
            use_tma=False,
            int8_pv=True,
        )
        return output.squeeze(0)


def _fast_paths_enabled() -> bool:
    return os.environ.get("H3_NATIVE_DISABLE_TRITON_FUSIONS", "0") != "1"


@lru_cache(maxsize=None)
def _reference_rms_island_values(name: str) -> frozenset[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    try:
        return frozenset(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(f"{name} must contain comma-separated integers") from error


def _reference_rms_island_active() -> bool:
    """Return whether this block must retain the mature RMS/AdaLN order.

    The experimental segmented kernel is globally profitable, but diffusion
    trajectories can amplify its small numerical delta.  A bounded island in
    early solver steps and motion-sensitive layers restores the mature
    one-pass operator only where coarse action topology is established.
    """

    steps = _reference_rms_island_values(
        "H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_STEPS"
    )
    layers = _reference_rms_island_values(
        "H3_NATIVE_EXPERIMENTAL_REFERENCE_RMS_LAYERS"
    )
    if not steps and not layers:
        return False
    step = _ATTENTION_STEP.get()
    layer = _ATTENTION_LAYER.get()
    if step is None or layer is None:
        return False
    return (not steps or step[0] in steps) and (not layers or layer in layers)


def rms_adaln(
    value: torch.Tensor,
    norm: "RMSNorm",
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: tuple["ModulationSegment", ...],
) -> torch.Tensor:
    """Fused dispatch for RMSNorm followed by segmented scale/shift.

    The common T2AV layout contains exactly three runs (text, audio, video).
    Two full-sequence Triton passes preserve the reference's separate BF16
    multiply and add stores while avoiding six tiny per-segment launches.
    Other layouts deliberately retain the transparent eager operation order.
    """

    supported = (
        _fast_paths_enabled()
        and value.is_cuda
        and value.ndim == 2
        and value.dtype in (torch.float16, torch.bfloat16)
        and len(segments) == 3
    )
    if supported:
        # This is project-owned SM89 kernel code.  The module name reflects
        # its compatibility-backend origin; it does not import ComfyUI.
        from backends.original.kernels.fused_norm_rope import (
            native_rms_one_pass_adaln,
            segmented_rms_adaln,
        )

        function = (
            segmented_rms_adaln
            if _FUSED_RMS_ADALN.get() and not _reference_rms_island_active()
            else native_rms_one_pass_adaln
        )
        return function(value, norm, shift, scale, segments)

    hidden = norm(value)
    previous = 0
    for start, stop, row in segments:
        if start != previous or stop <= start:
            raise ValueError("modulation segments must be an ordered partition")
        hidden[start:stop].mul_(
            1.0 + scale[row].to(hidden.dtype)
        ).add_(shift[row].to(hidden.dtype))
        previous = stop
    if previous != hidden.shape[0]:
        raise ValueError("modulation segments do not cover the packed sequence")
    return hidden


def sage_attention_sm89(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """SageAttention in native NHD layout without three full Q/K/V copies."""

    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    output = sageattn_qk_int8_pv_fp8_cuda(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran=_DENSE_QK_QUANT_GRAN.get(),
        pv_accum_dtype="fp32+fp16",
    )
    return output.squeeze(0)


def sage_attention_sm89_fused_k_quant(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Accepted SageAttention2++ with exact fused smooth-K quantization.

    This candidate is intentionally explicit rather than monkey-patching the
    third-party package.  It removes only the full BF16 ``K - mean(K)``
    intermediate while retaining the upstream mean, per-thread INT8 rounding,
    FP8-V preparation and SM89 attention kernel.
    """

    if (
        not query.is_cuda
        or query.dtype != torch.bfloat16
        or query.shape != key.shape
        or query.shape != value.shape
        or query.ndim != 3
        or query.shape[-1] != 128
        or _DENSE_QK_QUANT_GRAN.get() != "per_thread"
    ):
        return sage_attention_sm89(query, key, value)
    from sageattention import sm89_compile
    from sageattention import quant as sage_quant
    from sageattention.triton.quant_per_thread import (
        quant_query_per_thread_int8_kernel,
    )

    from .sage_fused_quant import quantize_key_sub_mean_per_thread_int8

    q = query.unsqueeze(0)
    k = key.unsqueeze(0)
    v = value.unsqueeze(0)
    batch, tokens, heads, head_dim = (int(item) for item in q.shape)
    key_mean = k.mean(dim=1, keepdim=True)
    query_int8 = torch.empty_like(q, dtype=torch.int8)
    key_int8 = torch.empty_like(k, dtype=torch.int8)
    query_scale = torch.empty(
        (batch, heads, (tokens + 127) // 128 * 32),
        device=q.device,
        dtype=torch.float32,
    )
    key_scale = torch.empty(
        (batch, heads, (tokens + 63) // 64 * 4),
        device=q.device,
        dtype=torch.float32,
    )
    query_grid = ((tokens + 127) // 128 * 32, heads, batch)
    quant_query_per_thread_int8_kernel[query_grid](
        q,
        query_int8,
        query_scale,
        tokens,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        query_int8.stride(0),
        query_int8.stride(2),
        query_int8.stride(1),
        query_scale.stride(0),
        query_scale.stride(1),
        C=head_dim,
        BLK=32,
    )
    quantize_key_sub_mean_per_thread_int8(k, key_mean, key_int8, key_scale)
    value_fp8, value_scale, _ = sage_quant.per_channel_fp8(
        v, tensor_layout="NHD", scale_max=2.25, smooth_v=False
    )
    output = torch.empty_like(q)
    sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
        query_int8,
        key_int8,
        value_fp8,
        output,
        query_scale,
        key_scale,
        value_scale,
        0,
        0,
        3,
        1.0 / (head_dim**0.5),
        0,
    )
    return output[0]


def make_sparge_attention_sm89(
    topk: float,
    *,
    dense_step_indices: tuple[int, ...] = (),
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create an opt-in sparse SM89 attention backend for controlled trials.

    This is deliberately not selected by the production planner.  Sparsity is
    an approximation rather than a numerically equivalent kernel, so a full
    generated-video gate is required before any preset can use it.
    """

    if not 0.5 <= topk <= 1.0:
        raise ValueError("SpargeAttention topk must be between 0.5 and 1.0")

    def sparse_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

        output = spas_sage2_attn_meansim_topk_cuda(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            topk=topk,
            is_causal=False,
            tensor_layout="NHD",
            return_sparsity=False,
        )
        return output.squeeze(0)

    return StepScheduledAttentionBackend(
        sage_attention_sm89,
        sparse_attention,
        dense_step_indices=dense_step_indices,
        minimum_sparse_tokens=128,
    )


def make_routed_sparge_attention_sm89() -> RequestRoutedSpargeAttentionBackend:
    return RequestRoutedSpargeAttentionBackend(minimum_sparse_tokens=128)


def make_joint_physical_action_backends_sm89(
    *, route_probe: bool = False
) -> dict[str, Callable]:
    """Return the exact action map shared by production and V19 calibration."""

    actions: dict[str, Callable] = {"dense": sage_attention_sm89}
    frontier_head_budgets = {
        # Exact Round188 reviewed rail: ordinary heads keep the 6.25% floor;
        # causal-risk heads receive the measured 8--10% differentiated floor.
        "sparse_topk_0.0625": (0.0625,) * len(_H3_HEAD_RISK_TIERS),
        "sparse_topk_0.1": tuple(
            0.10 if tier == 2 else 0.08 for tier in _H3_HEAD_RISK_TIERS
        ),
        # Higher actions retain the same risk ordering while preserving the
        # canonical mean budget used by the conservative planner cost model.
        "sparse_topk_0.25": tuple(
            0.28 if tier == 2 else 0.232 for tier in _H3_HEAD_RISK_TIERS
        ),
        "sparse_topk_0.5": tuple(
            0.55 if tier == 2 else 0.47 for tier in _H3_HEAD_RISK_TIERS
        ),
    }
    fast_frontier_head_budgets = {
        # The ordinary rail is mathematically uniform.  Keeping it scalar is
        # essential: the per-head selector is over 2x slower on the three
        # forecast anchor blocks while producing the exact same 6.25% mask.
        "sparse_topk_0.0625": 0.0625,
        "sparse_topk_0.1": frontier_head_budgets["sparse_topk_0.1"],
        "sparse_topk_0.25": frontier_head_budgets["sparse_topk_0.25"],
        "sparse_topk_0.5": frontier_head_budgets["sparse_topk_0.5"],
    }
    for name, topk in (
        ("sparse_topk_0.0625", 0.0625),
        ("sparse_topk_0.1", 0.10),
        ("sparse_topk_0.25", 0.25),
        ("sparse_topk_0.5", 0.50),
    ):
        shared = dict(
            experimental_minimum_topk=0.0625,
            temporal_correspondence_radius=1,
            temporal_spatial_block_radius=1,
            temporal_global_anchor_stride=8,
            temporal_global_spatial_block_radius=0,
        )
        # V3/V4 are retained with their originally executed fixed-TopK action
        # identity.  V5 explicitly selects the Round215 interaction-hybrid
        # family on which its cost/error table was measured.
        actions[name] = SplitModalityProtectedSpargeAttentionBackend(
            topk,
            selection_mode="fixed_topk",
            route_probe=route_probe,
            **shared,
        )
        actions[f"round215:{name}"] = SplitModalityProtectedSpargeAttentionBackend(
            topk,
            selection_mode="interaction_hybrid",
            **shared,
        )
        actions[f"frontier:{name}"] = SplitModalityProtectedSpargeAttentionBackend(
            frontier_head_budgets[name],
            selection_mode="fixed_topk",
            route_probe=route_probe,
            **shared,
        )
        actions[f"fastfrontier:{name}"] = SplitModalityProtectedSpargeAttentionBackend(
            fast_frontier_head_budgets[name],
            selection_mode="fixed_topk",
            route_probe=route_probe,
            **shared,
        )
        actions[f"forecastfrontier:{name}"] = actions[f"fastfrontier:{name}"]
    return actions


def make_joint_action_scheduled_sparge_attention_sm89(
    *, route_probe: bool = False
) -> CausalCheckpointVerifierAttentionBackend:
    """Build the hot, request-local backend for the two-control scheduler.

    The four sparse kernels share the Human-accepted Round86 structural rails.
    V9 adds a request-local Dense teacher probe only at three non-causal
    layers selected from the Round218 high-risk/temporal-variance evidence.
    Any repair is an upgrade and must be paid by the request's explicit
    Round219 ledger.  Earlier V7/V8 requests do not install that ledger and
    therefore execute their frozen offline schedules unchanged.
    """

    actions = make_joint_physical_action_backends_sm89(route_probe=route_probe)
    scheduled = RequestActionScheduledAttentionBackend(
        actions,
        exact_action="dense",
        legacy_backend=RequestRoutedSpargeAttentionBackend(
            minimum_sparse_tokens=128
        ),
        minimum_sparse_tokens=128,
    )
    return CausalCheckpointVerifierAttentionBackend(
        sage_attention_sm89,
        scheduled,
        probe_layers=(4, 24, 44),
        recovery_layers=(10, 20, 28, 49),
        recovery_horizon=0,
        hysteresis_layers=(),
        online_probe_growth_prediction=False,
        causal_head_island=True,
        head_error_mass_coverage=0.50,
        verification_query_blocks=14,
        relative_rms_threshold=0.34 * math.sqrt(0.75),
        online_guard_id=ROUND219_ONLINE_GUARD_ID,
        additional_online_guard_ids=(
            ROUND220_ONLINE_GUARD_ID,
            ROUND221_ONLINE_GUARD_ID,
            ROUND223_ONLINE_GUARD_ID,
        ),
        phase_probe_guard_ids=(
            ROUND220_ONLINE_GUARD_ID,
            ROUND221_ONLINE_GUARD_ID,
            ROUND223_ONLINE_GUARD_ID,
        ),
        phase_growth_guard_ids=(
            ROUND221_ONLINE_GUARD_ID,
            ROUND223_ONLINE_GUARD_ID,
        ),
        reserve_rebate_guard_ids=(ROUND223_ONLINE_GUARD_ID,),
    )


def make_modality_protected_sparge_attention_sm89(
    topk: float,
) -> ModalityProtectedSpargeAttentionBackend:
    return ModalityProtectedSpargeAttentionBackend(topk, minimum_sparse_tokens=128)


def make_split_modality_protected_sparge_attention_sm89(
    topk: float | tuple[float, ...],
    *,
    experimental_minimum_topk: float = 0.5,
) -> SplitModalityProtectedSpargeAttentionBackend:
    return SplitModalityProtectedSpargeAttentionBackend(
        topk,
        minimum_sparse_tokens=128,
        experimental_minimum_topk=experimental_minimum_topk,
    )


def make_layer_sensitivity_routed_split_sparge_attention_sm89(
    *,
    aggressive_topk: float | tuple[float, ...],
    sensitive_layers: tuple[int, ...],
    safe_topk: float | tuple[float, ...] = 0.50,
    experimental_minimum_topk: float = 0.25,
) -> LayerSensitivityRoutedSplitSpargeAttentionBackend:
    return LayerSensitivityRoutedSplitSpargeAttentionBackend(
        aggressive_topk=aggressive_topk,
        sensitive_layers=sensitive_layers,
        safe_topk=safe_topk,
        experimental_minimum_topk=experimental_minimum_topk,
    )


def make_trajectory_layer_modality_routed_sparge_attention_sm89(
    *,
    aggressive_topk: float | tuple[float, ...],
    sensitive_layers: tuple[int, ...],
    dense_step_indices: tuple[int, ...],
    safe_topk: float | tuple[float, ...] = 0.50,
    anchor_step_indices: tuple[int, ...] = (),
    anchor_aggressive_topk: float | tuple[float, ...] | None = None,
    anchor_safe_topk: float | tuple[float, ...] | None = None,
    recovery_step_indices: tuple[int, ...] = (),
    recovery_aggressive_topk: float | tuple[float, ...] | None = None,
    recovery_safe_topk: float | tuple[float, ...] | None = None,
    experimental_minimum_topk: float = 0.25,
    temporal_correspondence_radius: int = -1,
    temporal_spatial_block_radius: int = 0,
    temporal_global_anchor_stride: int = 0,
    temporal_global_spatial_block_radius: int = 0,
    selection_mode: str = "fixed_topk",
    request_guarded: bool = False,
) -> TrajectoryLayerModalityRoutedSpargeAttentionBackend:
    return TrajectoryLayerModalityRoutedSpargeAttentionBackend(
        aggressive_topk=aggressive_topk,
        sensitive_layers=sensitive_layers,
        dense_step_indices=dense_step_indices,
        safe_topk=safe_topk,
        anchor_step_indices=anchor_step_indices,
        anchor_aggressive_topk=anchor_aggressive_topk,
        anchor_safe_topk=anchor_safe_topk,
        recovery_step_indices=recovery_step_indices,
        recovery_aggressive_topk=recovery_aggressive_topk,
        recovery_safe_topk=recovery_safe_topk,
        experimental_minimum_topk=experimental_minimum_topk,
        temporal_correspondence_radius=temporal_correspondence_radius,
        temporal_spatial_block_radius=temporal_spatial_block_radius,
        temporal_global_anchor_stride=temporal_global_anchor_stride,
        temporal_global_spatial_block_radius=temporal_global_spatial_block_radius,
        selection_mode=selection_mode,
        request_guarded=request_guarded,
    )


def make_quality_constrained_adaptive_sparge_attention_sm89(
) -> QualityConstrainedAdaptiveSpargeAttentionBackend:
    """Create the no-knob TLHB-TCR quality-constrained optimizer."""

    return QualityConstrainedAdaptiveSpargeAttentionBackend(
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
        sentinel_query_blocks=7,
    )


def make_budget_constrained_adaptive_sparge_attention_sm89(
    *,
    compute_budget: float,
    safety_margin: float = 0.65,
) -> BudgetConstrainedAdaptiveSpargeAttentionBackend:
    """Create the two-knob request-adaptive fixed-budget controller."""

    return BudgetConstrainedAdaptiveSpargeAttentionBackend(
        compute_budget,
        safety_margin=safety_margin,
        temporal_correspondence_radius=1,
        temporal_spatial_block_radius=1,
    )


def make_trajectory_layer_modality_routed_sol_attention_sm89(
    *,
    source: str | Path,
    tau: float,
    sensitive_tau: float,
    sensitive_layers: tuple[int, ...],
    anchor_step_indices: tuple[int, ...] = (),
    anchor_tau: float | None = None,
    recovery_step_indices: tuple[int, ...] = (),
    recovery_tau: float | None = None,
) -> TrajectoryLayerModalityRoutedSolAttentionBackend:
    return TrajectoryLayerModalityRoutedSolAttentionBackend(
        source=source,
        tau=tau,
        sensitive_tau=sensitive_tau,
        sensitive_layers=sensitive_layers,
        anchor_step_indices=anchor_step_indices,
        anchor_tau=anchor_tau,
        recovery_step_indices=recovery_step_indices,
        recovery_tau=recovery_tau,
    )


__all__ = [
    "RequestRoutedSpargeAttentionBackend",
    "RequestActionScheduledAttentionBackend",
    "ModalityProtectedSpargeAttentionBackend",
    "SplitModalityProtectedSpargeAttentionBackend",
    "LayerSensitivityRoutedSplitSpargeAttentionBackend",
    "LayerHeadBudgetOverrideBackend",
    "TrajectoryLayerModalityRoutedSpargeAttentionBackend",
    "TrajectoryLayerModalityRoutedSolAttentionBackend",
    "QualityConstrainedAdaptiveSpargeAttentionBackend",
    "StepScheduledAttentionBackend",
    "attention_protected_prefix",
    "current_attention_protected_prefix",
    "current_long_video_attention_enabled",
    "long_sequence_query_chunking",
    "current_long_sequence_projection_chunk_tokens",
    "current_long_sequence_query_chunk_tokens",
    "current_long_sequence_split_qkv_outputs",
    "current_long_sequence_shared_qkv_quantization",
    "current_long_sequence_single_qknorm_rope",
    "current_long_sequence_exact_helper_stack",
    "current_long_sequence_parallel_sparse_lut",
    "current_long_sequence_partial_sparse_topk",
    "current_long_sequence_fused_prefix_k_quant",
    "current_long_sequence_fused_query_projection",
    "current_long_sequence_fused_qknorm_hnd_layout",
    "current_long_sequence_direct_nhd_output",
    "current_long_sequence_direct_nhd_kv",
    "current_long_sequence_direct_hnd_fp8_value",
    "attention_layer",
    "current_attention_layer",
    "attention_sparsity",
    "attention_action_schedule",
    "attention_step",
    "attention_video_layout",
    "make_routed_sparge_attention_sm89",
    "make_joint_physical_action_backends_sm89",
    "make_joint_action_scheduled_sparge_attention_sm89",
    "make_modality_protected_sparge_attention_sm89",
    "make_split_modality_protected_sparge_attention_sm89",
    "make_layer_sensitivity_routed_split_sparge_attention_sm89",
    "make_trajectory_layer_modality_routed_sparge_attention_sm89",
    "make_quality_constrained_adaptive_sparge_attention_sm89",
    "make_trajectory_layer_modality_routed_sol_attention_sm89",
    "make_sparge_attention_sm89",
    "rms_adaln",
    "rms_adaln_fusion",
    "sage_attention_sm89",
    "sage_attention_sm89_fused_k_quant",
]
