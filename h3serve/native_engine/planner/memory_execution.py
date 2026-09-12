"""Latency-minimal H3 execution planning under a physical VRAM budget.

There is deliberately no user-facing ``performance``/``low_vram`` branch.
Those names described historical implementations, not useful product goals.
The unified planner enumerates the measured physical execution graphs, rejects
graphs whose predicted peak crosses the current device budget, and selects the
fastest remaining full-context graph.  Packed-token accounting makes the same
policy valid for FL2VA, Ref2VA and second sampling without prompt heuristics.

Legacy mode strings remain accepted only so persisted jobs can be resumed.
They never influence the selected graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

from ..resource_backends import ResourceBackendId, get_resource_backend
from .contracts import WorkloadFeatures


# Backward-compatible deserialization surface.  New requests are normalized to
# ``auto`` by the public contract and all three values run the same optimizer.
MemoryExecutionMode = Literal["auto", "performance", "low_vram"]
MEMORY_EXECUTION_MODES: tuple[MemoryExecutionMode, ...] = (
    "auto",
    "performance",
    "low_vram",
)

_GIB = 1024**3
_MIB = 1024**2

# The whole-query line is the upper envelope of the checked-in Base/LoRA
# 360p5--720p15 measurements.  Dense 720p15 alone can use less, but the V24
# sparse graph needs another sequence-long output workspace and physically
# OOMed under a hard 16-GiB allocator.  Keep the joint envelope conservative
# rather than admitting a sparse graph from a Dense-only peak measurement. It
# also predicts why 1080p15 cannot use the ordinary materialized QKV path on
# 24 GiB.
_WHOLE_QUERY_BASE_BYTES = int(round(4.90 * _GIB))
_WHOLE_QUERY_BYTES_PER_PACKED_TOKEN = 125_100.0
_CONDITIONED_WHOLE_QUERY_FLOOR_BYTES = int(round(10.25 * _GIB))

# Query-streaming keeps complete K/V context but bounds live Q/QKV/output
# rows.  The line is conservative relative to the 220,003-token SM89 ladder:
# q16k--q49k measured 17.45 GiB allocated / 18.66 GiB NVML, while q98k rose
# to 23.04 GiB NVML.  It is an admission model, not a speed claim.
_STREAM_BASE_BYTES = int(round(5.60 * _GIB))
_STREAM_BYTES_PER_PACKED_TOKEN = 58_000.0
_STREAM_LOW_CHUNK_OVERHEAD_BYTES = int(round(1.25 * _GIB))
_STREAM_LARGE_CHUNK_FREE_ROWS = 49_152
_STREAM_LARGE_CHUNK_BYTES_PER_ROW = 80_000.0

# Compact streaming never materializes sequence-long BF16 K or V.  Its peak
# cannot be predicted by subtracting those tensors from the ordinary route:
# Sparge's selector and row-local Query workspace become the new overlap at
# very long sequences.  The checked-in real-Block ladder at q=8192 measured
# 2.848 / 10.625 / 18.308 GiB for 45,058 / 220,003 / 386,923 packed tokens.
# This conservative line covers all three.  Above 8k, the 2K ladder measured
# another ~120k bytes for each live Query row (18.308 / 19.228 / 21.070 GiB at
# q=8k / 16k / 32k).  Keep the model tied to physical tensors, not resolution
# names, so reference prefixes are accounted for through ``packed_tokens``.
_COMPACT_STREAM_BASE_BYTES = int(round(0.82 * _GIB))
_COMPACT_STREAM_BYTES_PER_PACKED_TOKEN = 48_500.0
_COMPACT_QUERY_FREE_ROWS = 8_192
_COMPACT_QUERY_BYTES_PER_EXTRA_ROW = 120_000.0
# The complete 2K15 checkpoint gate peaked at 22.330 GiB versus 18.308 GiB
# for the isolated real Block.  This fixed envelope covers alternating block
# residency, packed trajectory state and service-owned CUDA tensors absent
# from the Block probe.  Keeping it explicit avoids confusing a workspace
# estimate with an end-to-end admission estimate.
_COMPACT_SERVICE_OVERHEAD_BYTES = int(round(4.05 * _GIB))
# Safetensors metadata for all 50 FL2VA INT8 blocks reports exactly
# 387,359,520 registered bytes per block slot (0.3607567 GiB).  A serial
# one-buffer route recovers one such slot when the ordinary two-slot overlap
# misses an 8/16 GiB admission boundary.
_H3_INT8_BLOCK_BUFFER_BYTES = 387_359_520
_H3_W4A8_BLOCK_BUFFER_BYTES = 218_768_128

# The generic compact line above is deliberately an upper envelope for the
# 24-GiB INT8 service and includes a 4.05-GiB end-to-end envelope measured at
# 2K.  Applying that fixed envelope to the smaller W4 graph would reject every
# useful long 8-GiB workload.  A hard allocator-capped hot-session gate gives
# a more relevant anchor: 44,617 packed tokens completed one formal H3 DiT
# step at 4.384 GiB allocated / 5.984 GiB reserved with the larger INT8 block.
# Use a conservative 4.55-GiB single-buffer anchor for W4, retain the measured
# 48.5-kB/token compact slope, and account for every extra W4 block slot.  The
# narrow 720p15 admission produced by this line still has to pass the physical
# 8-GiB release gate; this model merely stops an unrelated 2K fixed envelope
# from making that validation impossible.
_W4_COMPACT_HARD8_ANCHOR_PACKED_TOKENS = 44_617
_W4_COMPACT_HARD8_SINGLE_BUFFER_BYTES = int(round(4.55 * _GIB))

# Video-VAE executes only after DiT eviction, so it is modeled as an
# independent phase and the request peak is max(DiT, VAE), never their sum.
# The exact temporal host sink was physically measured on SM89 at
# 1920x1088x362: 8.500 GiB versus 16.952 GiB for the materialized FP32 concat,
# 82.58s versus 82.78s, with an identical final uint8 SHA-256.  Its live
# decoder working set scales with one spatial canvas but not total duration.
# Decoder materialization adds one 3-channel FP32 output tensor; the later
# single-tensor pixel transform adds a second clip-sized FP32 workspace.
_VAE_FIXED_MODEL_BYTES = int(round(5.15 * _GIB))
_VAE_1080P_ACTIVE_BYTES = int(round(3.65 * _GIB))
_VAE_1080P_PIXELS = 1920 * 1088
_VAE_OUTPUT_CHANNELS = 3
_VAE_HOST_PIXEL_WORKING_SET_BYTES = 256 * _MIB
_VAE_TEMPORAL_PIECE_FRAMES = 17
# ``decode_base`` first owns the complete FP32 decoded clip.  The established
# single-tensor uint8 transport then creates another complete FP32 tensor for
# the checkpoint pixel transform before quantization.  A hard-16-GiB 720p15
# product run physically measured 12.60 GiB live immediately before that
# second 3.811-GiB allocation, while the analytical materialized-decoder line
# predicted 10.607 GiB.  Keep the observed ~2-GiB service envelope explicit;
# otherwise a 16-GiB launcher can admit a graph that only a physical 24-GiB
# card would survive.  This guard affects admission only and never numerics.
_VAE_MATERIALIZED_SERVICE_GUARD_BYTES = 2 * _GIB

_QUERY_CHUNK_CANDIDATES = (49_152, 32_768, 16_384, 8_192, 4_096, 2_048)
# On the compact 2K route q=8k was both faster and lower-memory than q=16k
# and q=32k.  Smaller chunks are pressure fallbacks, not preferred defaults.
_COMPACT_QUERY_CHUNK_CANDIDATES = (8_192, 16_384, 32_768, 49_152, 4_096, 2_048)
# An 8-GiB card has materially less Torch-allocatable space after the CUDA
# context than an 8-GiB allocator allowance on a 24-GiB development card.
# The first complete W4 720p15 gate used 7.015 GiB allocated but 7.893 GiB
# reserved with q=8192/mlp=4096.  Prefer half-sized row slabs for W4 so the
# release route preserves physical-card headroom instead of relying on a
# simulator-only allocator reserve.
_W4_COMPACT_QUERY_CHUNK_CANDIDATES = (4_096, 8_192, 2_048)

# Sparse/Forecast scheduling changes how often expensive cells execute; it
# does not remove the exact Dense anchors from the product trajectory.  A
# Dense anchor may leave the compact Sparge graph and enter SageAttention's
# full-context Q/K quantization path.  The latter owns complete Q/K/V rows and
# a sequence-long centered-K temporary (``k - mean(k)`` in the pinned SM89
# build).  The compact sparse estimate is therefore not a safe residency
# envelope by itself.
#
# The 2026-08-31 W4A8 720p15 failure reached 21.31 GiB allocated with 49
# resident blocks and then requested another 1.33 GiB centered-K tensor.  The
# exact-streaming model tracks the non-resident Dense working set, while this
# guard covers allocator/cache variance and the request-local quantization
# temporary.  It is deliberately independent of sampling steps, Forecast
# count and sparse keep ratio: those are latency/quality controls, not peak
# VRAM capacity controls.
_DENSE_ACTUAL_RUNTIME_GUARD_BYTES = int(round(1.50 * _GIB))
_RESIDENCY_ADMISSION_GUARD_BYTES = 128 * _MIB
# A distilled LoRA block includes adapter tensors absent from the original
# W4/INT8 calibration, and SageAttention's FP8 Value quantizer requests one
# final ~654-MiB workspace after the resident prefix is already live.  The
# 2026-09-01 physical 720p long-window failure showed that the generic
# 128-MiB residency reserve is insufficient even when actual block bytes are
# used.  This guard changes only latency-oriented GPU residency, never model
# weights, Attention precision or the sampling schedule.
_LORA_RESIDENCY_ADMISSION_GUARD_BYTES = int(round(1.25 * _GIB))


@dataclass(frozen=True, slots=True)
class MemoryExecutionDecision:
    requested_mode: MemoryExecutionMode
    backend_profile: ResourceBackendId
    selected_scheme: Literal[
        "whole_query", "exact_streaming", "compact_streaming"
    ]
    reason: str
    device_budget_bytes: int
    estimated_performance_peak_bytes: int
    estimated_dit_peak_bytes: int
    estimated_vae_materialized_peak_bytes: int
    estimated_vae_postprocess_peak_bytes: int
    estimated_vae_host_peak_bytes: int
    estimated_vae_selected_peak_bytes: int
    estimated_selected_peak_bytes: int
    query_chunk_tokens: int | None
    compact_kv: bool
    block_buffer_count: int
    projection_chunk_tokens: int
    mlp_chunk_tokens: int
    vae_spatial_tile: int
    vae_temporal_tile: int | None
    fits_budget: bool
    weight_tier: Literal["int8", "w4a8"] = "int8"

    @property
    def resource_profile(self) -> str:
        return self.backend_profile

    def telemetry(self) -> dict[str, object]:
        return {
            "schema_version": "h3_isolated_resource_execution_v4",
            "policy": "minimum_predicted_latency_under_vram_budget",
            "resource_profile": self.resource_profile,
            "weight_tier": self.weight_tier,
            "requested_mode": self.requested_mode,
            "legacy_mode_ignored": self.requested_mode != "auto",
            "selected_scheme": self.selected_scheme,
            "reason": self.reason,
            "device_budget_bytes": self.device_budget_bytes,
            "device_budget_gib": self.device_budget_bytes / _GIB,
            "estimated_performance_peak_bytes": (
                self.estimated_performance_peak_bytes
            ),
            "estimated_performance_peak_gib": (
                self.estimated_performance_peak_bytes / _GIB
            ),
            "estimated_selected_peak_bytes": self.estimated_selected_peak_bytes,
            "estimated_selected_peak_gib": self.estimated_selected_peak_bytes / _GIB,
            "estimated_dit_peak_bytes": self.estimated_dit_peak_bytes,
            "estimated_dit_peak_gib": self.estimated_dit_peak_bytes / _GIB,
            "estimated_vae_materialized_peak_bytes": (
                self.estimated_vae_materialized_peak_bytes
            ),
            "estimated_vae_materialized_peak_gib": (
                self.estimated_vae_materialized_peak_bytes / _GIB
            ),
            "estimated_vae_postprocess_peak_bytes": (
                self.estimated_vae_postprocess_peak_bytes
            ),
            "estimated_vae_postprocess_peak_gib": (
                self.estimated_vae_postprocess_peak_bytes / _GIB
            ),
            "estimated_vae_host_peak_bytes": self.estimated_vae_host_peak_bytes,
            "estimated_vae_host_peak_gib": (
                self.estimated_vae_host_peak_bytes / _GIB
            ),
            "estimated_vae_selected_peak_bytes": (
                self.estimated_vae_selected_peak_bytes
            ),
            "estimated_vae_selected_peak_gib": (
                self.estimated_vae_selected_peak_bytes / _GIB
            ),
            "query_chunk_tokens": self.query_chunk_tokens,
            "compact_kv": self.compact_kv,
            "block_buffer_count": self.block_buffer_count,
            "copy_compute_overlap": self.block_buffer_count == 2,
            "projection_chunk_tokens": self.projection_chunk_tokens,
            "mlp_chunk_tokens": self.mlp_chunk_tokens,
            "vae_spatial_tile": self.vae_spatial_tile,
            "vae_temporal_tile": self.vae_temporal_tile,
            "vae_output_strategy": (
                "not_executed"
                if self.estimated_vae_selected_peak_bytes == 0
                else "gpu_materialized_exact"
                if self.vae_temporal_tile is None
                else "host_temporal_exact"
            ),
            "fits_budget": self.fits_budget,
            "quality_controls_unchanged": True,
            "weights_steps_schedule_unchanged": self.weight_tier == "int8",
            "sampling_steps_schedule_unchanged": True,
            "base_weight_quantization": (
                "grouped_w4a8_convrot"
                if self.weight_tier == "w4a8"
                else "tensorwise_int8_convrot"
            ),
            "numerical_contract": (
                "w4a8_compact_quantized_full_context"
                if self.weight_tier == "w4a8" and self.compact_kv
                else "w4a8_full_context_streaming"
                if self.weight_tier == "w4a8"
                else "compact_quantized_full_context"
                if self.compact_kv
                else "exact_whole_query_full_context"
                if self.selected_scheme == "whole_query"
                else "exact_full_context_streaming"
            ),
            "bit_exact": self.weight_tier == "int8" and not self.compact_kv,
        }


def estimate_performance_peak_bytes(features: WorkloadFeatures) -> int:
    """Predict the materialized whole-query peak from the measured envelope."""

    peak = _WHOLE_QUERY_BASE_BYTES + round(
        _WHOLE_QUERY_BYTES_PER_PACKED_TOKEN * features.packed_tokens
    )
    if features.condition_count:
        peak = max(peak, _CONDITIONED_WHOLE_QUERY_FLOOR_BYTES)
    return int(peak)


def estimate_streaming_peak_bytes(
    features: WorkloadFeatures,
    *,
    query_chunk_tokens: int,
) -> int:
    """Conservative peak for the current full-context Query-streaming path."""

    if query_chunk_tokens < 128 or query_chunk_tokens % 128:
        raise ValueError("Query chunk must be a positive multiple of 128")
    large_chunk_rows = max(
        0, int(query_chunk_tokens) - _STREAM_LARGE_CHUNK_FREE_ROWS
    )
    peak = (
        _STREAM_BASE_BYTES
        + round(_STREAM_BYTES_PER_PACKED_TOKEN * features.packed_tokens)
        + _STREAM_LOW_CHUNK_OVERHEAD_BYTES
        + round(_STREAM_LARGE_CHUNK_BYTES_PER_ROW * large_chunk_rows)
    )
    if features.condition_count:
        peak = max(peak, _CONDITIONED_WHOLE_QUERY_FLOOR_BYTES)
    return int(peak)


def estimate_dense_actual_peak_bytes(
    features: WorkloadFeatures,
    *,
    query_chunk_tokens: int | None,
) -> int:
    """Conservative non-weight peak of the heaviest exact Actual cell.

    Compact K/V is an implementation detail of sparse cells.  Any trajectory
    containing a Dense action must also fit this full-context envelope.  The
    estimate intentionally ignores the number of Actual/Forecast evaluations:
    one Dense cell is sufficient to establish the request peak.
    """

    chunk = 4_096 if query_chunk_tokens is None else int(query_chunk_tokens)
    # The calibrated streaming estimator accepts aligned row chunks.  Product
    # candidates already satisfy this contract, but keeping the helper total
    # makes it safe for explicit/research plans as well.
    chunk = max(128, chunk - chunk % 128)
    return int(
        estimate_streaming_peak_bytes(
            features,
            query_chunk_tokens=chunk,
        )
        + _DENSE_ACTUAL_RUNTIME_GUARD_BYTES
    )


def select_dense_safe_resident_blocks(
    features: WorkloadFeatures,
    decision: MemoryExecutionDecision,
    *,
    requested_blocks: int,
    block_bytes: int,
) -> tuple[int, int, int]:
    """Clamp a latency-only resident prefix to the Dense Actual envelope.

    Returns ``(selected_blocks, capacity_blocks, dense_peak_bytes)``.  The
    result depends only on geometry/context and the physical backend budget;
    sampling steps, Forecast evaluations and sparse ratios are intentionally
    absent.
    """

    requested = int(requested_blocks)
    block_size = int(block_bytes)
    if not 0 <= requested < 50:
        raise ValueError("requested resident blocks must lie inside [0, 49]")
    if block_size <= 0:
        raise ValueError("resident block size must be positive")
    dense_peak = estimate_dense_actual_peak_bytes(
        features,
        query_chunk_tokens=decision.query_chunk_tokens,
    )
    residency_guard = (
        _LORA_RESIDENCY_ADMISSION_GUARD_BYTES
        if features.engine == "lora"
        else _RESIDENCY_ADMISSION_GUARD_BYTES
    )
    capacity_bytes = max(
        0,
        int(decision.device_budget_bytes)
        - residency_guard
        - dense_peak,
    )
    capacity_blocks = min(49, capacity_bytes // block_size)
    return min(requested, int(capacity_blocks)), int(capacity_blocks), dense_peak


def estimate_compact_streaming_peak_bytes(
    features: WorkloadFeatures,
    *,
    query_chunk_tokens: int,
    block_buffer_count: int = 2,
    weight_tier: Literal["int8", "w4a8"] = "int8",
) -> int:
    """Estimate the two-pass compact-KV full-context streaming peak."""

    if query_chunk_tokens < 128 or query_chunk_tokens % 128:
        raise ValueError("Query chunk must be a positive multiple of 128")
    if block_buffer_count not in (1, 2):
        raise ValueError("compact streaming supports one or two block buffers")
    if weight_tier not in ("int8", "w4a8"):
        raise ValueError("weight_tier must be int8 or w4a8")
    large_chunk_rows = max(
        0, int(query_chunk_tokens) - _COMPACT_QUERY_FREE_ROWS
    )
    if weight_tier == "w4a8":
        # This service-level physical anchor already includes one live W4
        # block slot. Variable Ref2VA media is represented by packed_tokens;
        # do not apply the unrelated conditioned whole-query floor here.
        return int(
            _W4_COMPACT_HARD8_SINGLE_BUFFER_BYTES
            + round(
                _COMPACT_STREAM_BYTES_PER_PACKED_TOKEN
                * (
                    features.packed_tokens
                    - _W4_COMPACT_HARD8_ANCHOR_PACKED_TOKENS
                )
            )
            + round(_COMPACT_QUERY_BYTES_PER_EXTRA_ROW * large_chunk_rows)
            + (block_buffer_count - 1) * _H3_W4A8_BLOCK_BUFFER_BYTES
        )

    block_bytes = _H3_INT8_BLOCK_BUFFER_BYTES
    peak = (
        _COMPACT_STREAM_BASE_BYTES
        + round(_COMPACT_STREAM_BYTES_PER_PACKED_TOKEN * features.packed_tokens)
        + round(_COMPACT_QUERY_BYTES_PER_EXTRA_ROW * large_chunk_rows)
        + _COMPACT_SERVICE_OVERHEAD_BYTES
        - 2 * _H3_INT8_BLOCK_BUFFER_BYTES
        + block_buffer_count * block_bytes
    )
    if features.condition_count:
        peak = max(peak, _CONDITIONED_WHOLE_QUERY_FLOOR_BYTES)
    return int(peak)


def estimate_vae_host_streaming_peak_bytes(features: WorkloadFeatures) -> int:
    """Conservative peak for the byte-exact temporal host-output VAE graph."""

    canvas_pixels = int(features.width) * int(features.height)
    active = round(
        _VAE_1080P_ACTIVE_BYTES * canvas_pixels / _VAE_1080P_PIXELS
    )
    return int(_VAE_FIXED_MODEL_BYTES + active)


def estimate_vae_materialized_peak_bytes(features: WorkloadFeatures) -> int:
    """Peak when the complete decoded FP32 video is concatenated on GPU."""

    decoded_fp32 = (
        _VAE_OUTPUT_CHANNELS * int(features.output_pixel_frames) * 4
    )
    return int(estimate_vae_host_streaming_peak_bytes(features) + decoded_fp32)


def estimate_vae_postprocess_peak_bytes(features: WorkloadFeatures) -> int:
    """Peak of the complete-GPU-output path through exact uint8 transport.

    ``estimate_vae_materialized_peak_bytes`` stops when ``decode_base`` has
    concatenated the complete FP32 clip.  Production still has to apply the
    checkpoint pixel transform.  The non-streamed implementation needs a
    second clip-sized FP32 workspace at that boundary, plus the physically
    observed service envelope that is not present in the isolated VAE model.
    """

    decoded_fp32 = (
        _VAE_OUTPUT_CHANNELS * int(features.output_pixel_frames) * 4
    )
    return int(
        estimate_vae_materialized_peak_bytes(features)
        + decoded_fp32
        + _VAE_MATERIALIZED_SERVICE_GUARD_BYTES
    )


def select_vae_temporal_host_chunk(features: WorkloadFeatures) -> int:
    """Bound exact per-piece pixel conversion to a 256-MiB FP32 workspace."""

    one_frame = (
        _VAE_OUTPUT_CHANNELS * int(features.width) * int(features.height) * 4
    )
    return max(
        1,
        min(
            _VAE_TEMPORAL_PIECE_FRAMES,
            _VAE_HOST_PIXEL_WORKING_SET_BYTES // one_frame,
        ),
    )


def select_memory_execution(
    features: WorkloadFeatures,
    *,
    requested_mode: MemoryExecutionMode,
    device_budget_bytes: int,
    existing_query_chunk_tokens: int | None = None,
    weight_tier: Literal["int8", "w4a8"] = "int8",
    resource_profile: ResourceBackendId | None = None,
    include_vae: bool = True,
) -> MemoryExecutionDecision:
    """Run one isolated resource backend and select its fastest feasible graph.

    The launcher fixes ``resource_profile`` before weights are loaded.  The
    profile then owns its candidate graphs and capacity limit; it is never
    inferred again from a request's prompt or geometry.  Within that backend,
    the graph order comes from paired physical latency gates.  On 720p15 the
    exact streaming and whole-query graphs measured 39.624 and 39.683 seconds
    per Dense Actual step respectively, so exact streaming remains the INT8
    latency winner as well as the lower-pressure route.  Whole Query is kept
    only as a fail-closed diagnostic graph.
    """

    if requested_mode not in MEMORY_EXECUTION_MODES:
        raise ValueError(f"unknown memory execution mode: {requested_mode}")
    if device_budget_bytes <= 0:
        raise ValueError("device budget must be positive")
    if weight_tier not in ("int8", "w4a8"):
        raise ValueError("weight_tier must be int8 or w4a8")

    if resource_profile is None:
        if weight_tier == "w4a8":
            provisioned_gib = device_budget_bytes / _GIB + 0.75
            resource_profile = (
                "w4a8_24gb" if provisioned_gib >= 20.0
                else "w4a8_16gb" if provisioned_gib >= 12.0
                else "w4a8_8gb"
            )
        else:
            provisioned_gib = device_budget_bytes / _GIB + 0.75
            resource_profile = (
                "int8_24gb" if provisioned_gib >= 20.0 else "int8_16gb"
            )
    backend = get_resource_backend(
        resource_profile,
        weight_tier=weight_tier,
    )
    profile_budget_bytes = int(backend.planner_budget_gib * _GIB)
    # Isolation is enforced twice: RuntimeConfig limits the CUDA allocator at
    # model construction, and the planner independently clamps admission here.
    # A 16GB launcher therefore cannot borrow spare capacity merely because it
    # happens to be tested on a 24GB development card.
    research_w4_capacity = os.environ.get(
        "H3_NATIVE_RESEARCH_W4_VRAM_GIB", ""
    ).strip()
    if research_w4_capacity and weight_tier == "w4a8":
        # Explicit calibration-only override. RuntimeConfig still enforces the
        # allocator ceiling; this merely prevents the public 8-GiB backend
        # definition from hiding the additional 16/24-GiB capacity being
        # measured by the resource-knee benchmark.
        device_budget_bytes = int(device_budget_bytes)
    else:
        device_budget_bytes = min(int(device_budget_bytes), profile_budget_bytes)

    # RuntimeConfig already removes 768 MiB from physical capacity.  Keep a
    # second 128-MiB allocator guard here (896 MiB total) without throwing away
    # the final ~0.5 GiB needed by conditioned 1080p15 on a 16-GiB device.
    admission_budget = max(0, int(device_budget_bytes) - 128 * _MIB)
    performance_peak = estimate_performance_peak_bytes(features)

    # Each backend is separately maintainable.  INT8 starts from the physically
    # measured exact-streaming winner; W4A8 starts from compact K/V because no
    # ordinary full-K/V graph has passed the hard physical 8-GiB gate.
    exact_candidates = (
        _W4_COMPACT_QUERY_CHUNK_CANDIDATES
        if weight_tier == "w4a8" and resource_profile == "w4a8_8gb"
        else _QUERY_CHUNK_CANDIDATES
    )
    compact_candidates = (
        _W4_COMPACT_QUERY_CHUNK_CANDIDATES
        if weight_tier == "w4a8"
        else _COMPACT_QUERY_CHUNK_CANDIDATES
    )
    diagnostic_override = os.environ.get(
        "H3_NATIVE_DIAGNOSTIC_EXECUTION_GRAPH", ""
    ).strip().lower()
    if diagnostic_override not in {"", "whole_query", "exact_streaming", "compact_streaming"}:
        raise ValueError(
            "H3_NATIVE_DIAGNOSTIC_EXECUTION_GRAPH must be whole_query, "
            "exact_streaming or compact_streaming"
        )
    if (
        diagnostic_override
        and diagnostic_override not in backend.execution_preference
    ):
        raise RuntimeError(
            f"{diagnostic_override} is not admitted by {resource_profile}"
        )

    use_whole_query = (
        diagnostic_override == "whole_query"
        and performance_peak <= admission_budget
    )
    if diagnostic_override == "whole_query" and not use_whole_query:
        # Preserve a fail-closed diagnostic: forcing a graph must never bypass
        # the physical admission model.
        raise RuntimeError("diagnostic whole-query graph does not fit this backend")

    selected_chunk: int | None
    prefer_compact = (
        diagnostic_override == "compact_streaming"
        or (
            not diagnostic_override
            and backend.execution_preference[0] == "compact_streaming"
        )
    )
    if use_whole_query:
        selected_chunk = None
        selected_peak = performance_peak
        compact_kv = False
        block_buffer_count = 2
        exact_fit = True
        selected_scheme = "whole_query"
    else:
        candidates = exact_candidates
        if existing_query_chunk_tokens is not None:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate <= existing_query_chunk_tokens
            ) or (int(existing_query_chunk_tokens),)
        selected_chunk = candidates[-1]
        selected_peak = estimate_streaming_peak_bytes(
            features, query_chunk_tokens=selected_chunk
        )
        compact_kv = False
        block_buffer_count = 2
        exact_fit = False
        selected_scheme = "exact_streaming"
        if not prefer_compact:
            for candidate in candidates:
                candidate_peak = estimate_streaming_peak_bytes(
                    features, query_chunk_tokens=candidate
                )
                if candidate_peak <= admission_budget:
                    selected_chunk = candidate
                    selected_peak = candidate_peak
                    exact_fit = True
                    break

    if not exact_fit:
        compact_kv = True
        selected_scheme = "compact_streaming"
        if existing_query_chunk_tokens is not None:
            compact_candidates = tuple(
                candidate
                for candidate in compact_candidates
                if candidate <= existing_query_chunk_tokens
            ) or (int(existing_query_chunk_tokens),)
        selected_chunk = compact_candidates[-1]
        selected_peak = estimate_compact_streaming_peak_bytes(
            features,
            query_chunk_tokens=selected_chunk,
            weight_tier=weight_tier,
        )
        for candidate in compact_candidates:
            candidate_peak = estimate_compact_streaming_peak_bytes(
                features,
                query_chunk_tokens=candidate,
                weight_tier=weight_tier,
            )
            if candidate_peak <= admission_budget:
                selected_chunk = candidate
                selected_peak = candidate_peak
                break
        if selected_peak > admission_budget:
            block_buffer_count = 1
            for candidate in compact_candidates:
                candidate_peak = estimate_compact_streaming_peak_bytes(
                    features,
                    query_chunk_tokens=candidate,
                    block_buffer_count=1,
                    weight_tier=weight_tier,
                )
                selected_chunk = candidate
                selected_peak = candidate_peak
                if candidate_peak <= admission_budget:
                    break

    dit_peak = selected_peak
    vae_materialized_peak = estimate_vae_materialized_peak_bytes(features)
    vae_postprocess_peak = estimate_vae_postprocess_peak_bytes(features)
    vae_host_peak = estimate_vae_host_streaming_peak_bytes(features)
    if not include_vae:
        # UltimateUpscale pieces stop at clean latents.  The Video-VAE runs
        # only once after CPU stitching, so charging its full-clip peak to
        # every piece incorrectly rejects otherwise valid temporal windows.
        vae_selected_peak = 0
        vae_temporal_tile = None
    elif vae_postprocess_peak <= admission_budget:
        vae_selected_peak = vae_postprocess_peak
        vae_temporal_tile = None
    else:
        vae_selected_peak = vae_host_peak
        vae_temporal_tile = select_vae_temporal_host_chunk(features)
    selected_peak = max(dit_peak, vae_selected_peak)

    # 4k MLP chunks were at least as fast as 8k/16k/32k in the checked-in
    # 220k-token ladder, while reducing row-local activation headroom.
    mlp_chunk = (
        8192
        if selected_scheme == "whole_query"
        else
        min(2048, selected_chunk)
        if weight_tier == "w4a8" and compact_kv
        else 4096
        if selected_chunk is not None and selected_chunk >= 4096
        else int(selected_chunk)
    )
    reason = (
        f"{resource_profile}_whole_query_diagnostic"
        if selected_scheme == "whole_query"
        else f"{resource_profile}_exact_streaming_measured_latency_winner"
        if selected_scheme == "exact_streaming"
        else "compact_streaming_capacity_single_buffer"
        if block_buffer_count == 1
        else f"{resource_profile}_compact_streaming_capacity"
    )
    if diagnostic_override:
        reason = f"diagnostic_forced_{diagnostic_override}"
    if dit_peak > admission_budget or vae_selected_peak > admission_budget:
        reason = "no_full_context_graph_fits_budget"
    return MemoryExecutionDecision(
        requested_mode=requested_mode,
        backend_profile=resource_profile,
        selected_scheme=selected_scheme,
        reason=reason,
        device_budget_bytes=int(device_budget_bytes),
        estimated_performance_peak_bytes=performance_peak,
        estimated_dit_peak_bytes=dit_peak,
        estimated_vae_materialized_peak_bytes=vae_materialized_peak,
        estimated_vae_postprocess_peak_bytes=vae_postprocess_peak,
        estimated_vae_host_peak_bytes=vae_host_peak,
        estimated_vae_selected_peak_bytes=vae_selected_peak,
        estimated_selected_peak_bytes=selected_peak,
        query_chunk_tokens=selected_chunk,
        compact_kv=compact_kv,
        block_buffer_count=block_buffer_count,
        projection_chunk_tokens=(
            8192 if selected_chunk is None else min(8192, selected_chunk)
        ),
        mlp_chunk_tokens=mlp_chunk,
        # The exact temporal sink removes duration-dependent concat memory, so
        # every tier can retain the physically faster 288 spatial tile.
        vae_spatial_tile=288,
        vae_temporal_tile=vae_temporal_tile,
        fits_budget=(
            dit_peak <= admission_budget
            and vae_selected_peak <= admission_budget
        ),
        weight_tier=weight_tier,
    )


__all__ = [
    "MEMORY_EXECUTION_MODES",
    "MemoryExecutionDecision",
    "MemoryExecutionMode",
    "estimate_performance_peak_bytes",
    "estimate_compact_streaming_peak_bytes",
    "estimate_dense_actual_peak_bytes",
    "estimate_streaming_peak_bytes",
    "estimate_vae_host_streaming_peak_bytes",
    "estimate_vae_materialized_peak_bytes",
    "estimate_vae_postprocess_peak_bytes",
    "select_vae_temporal_host_chunk",
    "select_memory_execution",
    "select_dense_safe_resident_blocks",
]
