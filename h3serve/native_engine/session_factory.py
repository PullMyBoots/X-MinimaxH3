"""Production construction of the measured RTX 4090 Native hot session."""

from __future__ import annotations

import importlib
import ctypes
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..deployment_profiles import LAUNCHER_DEFINITIONS
from ..memory_policy import HOST_MEMORY_PROFILES, HostMemoryProfile
from ..lora_registry import resolve_lora_profile
from .hot_session import NativeT2AVHotSession
from .local_checkpoint_cache import (
    drop_file_page_cache,
    materialize_local_checkpoint,
    materialize_qwen_layer_cache,
    should_localize_checkpoint,
)
from ..contract import (
    launcher_family,
    launcher_vram_profile,
    launcher_weight_tier,
    normalize_launcher,
    resolve_engine,
)
from .long_video_motion_detail import candidate_requested


def _generic_sparse_requested() -> bool:
    return (
        os.environ.get("H3_NATIVE_ENABLE_SPARSE", "0") == "1"
        or os.environ.get("H3_NATIVE_REVIEW_SPARSE", "0") == "1"
    )


def _experimental_long_horizon_requested() -> bool:
    """Deployment-side opt-in for Human-pending 15-second V19 schedules."""

    return os.environ.get(
        "H3_NATIVE_V19_EXPERIMENTAL_LONG_HORIZON", "0"
    ) == "1"


def _mechanistic_admission_path() -> Path | None:
    configured = os.environ.get(
        "H3_NATIVE_MECHANISTIC_ADMISSION", ""
    ).strip()
    return (
        None
        if not configured
        else Path(configured).expanduser().resolve()
    )


def _mechanistic_deployment_requested() -> bool:
    """An explicit schedule-free admission artifact overrides V24."""

    return (
        _mechanistic_admission_path() is not None
        and os.environ.get("H3_NATIVE_ENABLE_SPARSE", "auto") != "0"
    )


def _v24_deployment_requested() -> bool:
    """Enable the built-in V24 policy whenever the release sparse runtime is on.

    ``H3_NATIVE_PARETO_V24=0`` is the explicit rollback switch.  A value of
    ``1`` also permits direct launchers that provide the extension path without
    first setting the generic sparse flag; normal project launchers set
    ``H3_NATIVE_ENABLE_SPARSE=1`` after validating or building the extension.
    """

    mode = os.environ.get("H3_NATIVE_PARETO_V24", "auto").strip().lower()
    if mode in ("0", "false", "off", "no"):
        return False
    if mode in ("1", "true", "on", "yes"):
        return os.environ.get("H3_NATIVE_ENABLE_SPARSE", "auto") != "0"
    return _generic_sparse_requested()


def _review_sparse_requested() -> bool:
    # Product installs may opt into the pinned SM89 extension explicitly. The
    # legacy review flag remains accepted for existing calibration scripts.
    return (
        _generic_sparse_requested()
        or _mechanistic_deployment_requested()
        or _v24_deployment_requested()
        or _experimental_long_horizon_requested()
        or candidate_requested()
        or (
            _v19_release_bundle_path().is_file()
            and os.environ.get("H3_NATIVE_ENABLE_SPARSE", "auto") != "0"
        )
    )


def _v19_release_bundle_path() -> Path:
    serve_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("H3_NATIVE_V19_RELEASE_BUNDLE", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (
            serve_root
            / "h3serve/native_engine/planner/evidence/v19_release_bundle.json"
        )
    )


def _review_fused_rms_requested() -> bool:
    return os.environ.get("H3_NATIVE_REVIEW_FUSED_RMS", "0") == "1"


def _prepare_review_sparse_import() -> None:
    build_dir = os.environ.get("H3_NATIVE_SPARGE_BUILD_DIR")
    if build_dir:
        path = Path(build_dir).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError(f"H3_NATIVE_SPARGE_BUILD_DIR is not a directory: {path}")
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        importlib.import_module("spas_sage_attn")
    except Exception as error:
        raise RuntimeError(
            "review sparse routing was requested but the pinned SpargeAttention "
            "SM89 extension is unavailable"
        ) from error


@dataclass(frozen=True, slots=True)
class NativeSessionPaths:
    model_root: Path
    minimax_source: Path
    lightx_source: Path
    turbo_curve: Path
    output_root: Path

    def resolved(self) -> "NativeSessionPaths":
        return NativeSessionPaths(
            model_root=self.model_root.resolve(),
            minimax_source=self.minimax_source.resolve(),
            lightx_source=self.lightx_source.resolve(),
            turbo_curve=self.turbo_curve.resolve(),
            output_root=self.output_root.resolve(),
        )


@dataclass(frozen=True, slots=True)
class BuiltNativeSession:
    session: NativeT2AVHotSession
    startup_seconds: float
    startup_tasks: dict[str, float]
    qwen_storage: str = "source"
    qwen_layer_cache: bool = False
    host_memory_profile: str = "fullspeed"
    dit_host_pinned: bool = False
    dit_host_cache_gib: float = 0.0
    dit_host_pinned_gib: float = 0.0
    dit_host_pinned_fraction: float = 0.0
    v19_release_bundle: str | None = None
    v19_release_digest: str | None = None
    pareto_policy_id: str | None = None
    pareto_candidate_id: str | None = None
    launcher: str = "fl2va_int8_24gb"
    weight_tier: str = "int8"
    vram_profile: str = "24gb"
    allocator_ceiling_gib: float = 23.25
    lora_checkpoint: str = "minimax_h3_turbo_v4_step600_ema.safetensors"
    lora_profile_id: str = "larry_turbo_v4_step600_ema"
    lora_display_name: str = "Larry Turbo v4-600 EMA"
    lora_recommended_steps: tuple[int, ...] = (4, 5, 6, 7, 8)
    lora_default_steps: int = 6


class NativeSessionFactory:
    """Build one real-weight session without importing a UI graph runtime."""

    def __init__(
        self,
        paths: NativeSessionPaths,
        memory_profile: HostMemoryProfile | None = None,
    ) -> None:
        self.paths = paths.resolved()
        self.memory_profile = memory_profile or HOST_MEMORY_PROFILES["fullspeed"]
        self._progress_callback = None
        self._lora_checkpoint = (
            self.paths.model_root
            / "loras/minimax_h3_turbo_v4_step600_ema.safetensors"
        )
        self._lora_profile = resolve_lora_profile(self._lora_checkpoint)

    def set_progress_callback(self, callback) -> None:
        """Install a transient, thread-safe startup progress observer."""

        self._progress_callback = callback

    def _progress(self, percent: float, stage: str, detail: str) -> None:
        callback = self._progress_callback
        if callable(callback):
            callback(float(percent), str(stage), str(detail))

    def set_memory_profile(self, profile: HostMemoryProfile) -> None:
        self.memory_profile = profile

    def set_lora_checkpoint(self, checkpoint: Path) -> None:
        """Select one administrator-installed H3 LoRA for the next build."""

        # Public releases commonly keep ``models`` as a symlink into the
        # Linux-local weight store.  Compare canonical targets so a legitimate
        # checkpoint reached through that symlink is not rejected as escaping
        # models/loras.
        candidate = Path(checkpoint).expanduser().resolve()
        lora_root = (self.paths.model_root / "loras").resolve()
        allowed_targets = {
            item.resolve()
            for item in lora_root.rglob("*.safetensors")
            if item.is_file()
        }
        if candidate not in allowed_targets:
            raise ValueError("LoRA checkpoint must be installed inside models/loras")
        if candidate.suffix.lower() != ".safetensors" or not candidate.is_file():
            raise ValueError("selected LoRA checkpoint is not a safetensors file")
        self._lora_checkpoint = candidate
        self._lora_profile = resolve_lora_profile(candidate)

    @property
    def lora_checkpoint(self) -> Path:
        """Return the selected administrator LoRA without exposing internals."""

        return Path(self._lora_checkpoint)

    def set_output_root(self, output_root: Path) -> None:
        """Retarget future sessions while the owning hot engine is cold."""

        self.paths = NativeSessionPaths(
            model_root=self.paths.model_root,
            minimax_source=self.paths.minimax_source,
            lightx_source=self.paths.lightx_source,
            turbo_curve=self.paths.turbo_curve,
            output_root=output_root.resolve(),
        )

    @property
    def v19_release_enabled(self) -> bool:
        return (
            _v19_release_bundle_path().is_file()
            and os.environ.get("H3_NATIVE_ENABLE_SPARSE", "auto") != "0"
        )

    @property
    def v24_release_enabled(self) -> bool:
        """Whether the built-in Human-calibrated deployment policy is active."""

        return (
            not _mechanistic_deployment_requested()
            and _v24_deployment_requested()
        )

    @property
    def mechanistic_deployment_enabled(self) -> bool:
        admission = _mechanistic_admission_path()
        return bool(
            _mechanistic_deployment_requested()
            and admission is not None
            and admission.is_file()
        )

    @property
    def v19_scheduler_enabled(self) -> bool:
        """Compatibility name for any exact-token Base Pareto selector."""

        return (
            self.mechanistic_deployment_enabled
            or self.v24_release_enabled
            or self.v19_release_enabled
            or _experimental_long_horizon_requested()
        )

    def preflight(self, engine: str) -> dict[str, object]:
        launcher = normalize_launcher(engine)
        family = launcher_family(launcher)
        weight_tier = launcher_weight_tier(launcher)
        vram_profile = launcher_vram_profile(launcher)
        resource_backend = LAUNCHER_DEFINITIONS[launcher].backend
        root = self.paths.model_root
        checks = {
            "base_weight": (
                root / "diffusion_models" / (
                    f"minimax_h3_{'ref2va' if family == 'reference' else 'fl2va'}_"
                    + (
                        "pruned_w4a8_mixed.safetensors"
                        if weight_tier == "w4a8"
                        else "pruned_int8_convrot.safetensors"
                    )
                )
            ).is_file(),
            "text_weight": (
                root / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
            ).is_file(),
            "video_vae_weight": (
                root / "vae/minimax_h3_video_vae_fp16.safetensors"
            ).is_file(),
            "audio_vae_weight": (
                root / "vae/minimax_h3_audio_vae_fp32.safetensors"
            ).is_file(),
            "tokenizer": (self.paths.minimax_source / "tokenizer").is_dir(),
            "video_vae_source": (
                self.paths.minimax_source / "FL2VA/video_vae/klvae.py"
            ).is_file(),
            "audio_vae_config": (
                self.paths.minimax_source / "audio_vae/config.json"
            ).is_file(),
            "lightx_audio_source": (
                self.paths.lightx_source
                / "lightx2v/models/audio_encoders/hf/minimax_h3/audio_vae.py"
            ).is_file(),
        }
        if resource_backend.second_sampling_levels:
            checks["latent_upscaler_weight"] = (
                root
                / "latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors"
            ).is_file()
        # Both request routes are embedded in one family session.  Readiness is
        # therefore honest only when the base and its hot-switchable adapter
        # assets are present.
        checks.update(
            {
                "lora_weight": (
                    self._lora_checkpoint
                ).is_file(),
                "turbo_curve": self.paths.turbo_curve.is_file(),
                "lora_task_compatibility": family
                in self._lora_profile.task_families,
            }
        )
        sparse_requested = _review_sparse_requested()
        sparse_available = False
        if sparse_requested:
            try:
                _prepare_review_sparse_import()
                checks["review_sparse_backend"] = True
                sparse_available = True
            except RuntimeError:
                checks["review_sparse_backend"] = False
        configured_v19 = os.environ.get(
            "H3_NATIVE_V19_RELEASE_BUNDLE", ""
        ).strip()
        if configured_v19:
            checks["v19_release_bundle"] = _v19_release_bundle_path().is_file()
        mechanistic_admission = _mechanistic_admission_path()
        if mechanistic_admission is not None:
            checks["mechanistic_admission"] = mechanistic_admission.is_file()
        v24_policy_id = None
        v24_candidate_id = None
        if self.v24_release_enabled:
            from .planner import (
                V24_FINAL_DEFAULT_CANDIDATE,
                V24_FINAL_POLICY_ID,
            )

            checks["v24_release_policy"] = True
            v24_policy_id = V24_FINAL_POLICY_ID
            v24_candidate_id = V24_FINAL_DEFAULT_CANDIDATE
        return {
            "ready": all(checks.values()),
            "checks": checks,
            "capabilities": {
                "launcher": launcher,
                "weight_tier": weight_tier,
                "vram_profile": vram_profile,
                "second_sampling": bool(
                    resource_backend.second_sampling_levels
                ),
                "maximum_first_generation": (
                    resource_backend.maximum_first_generation
                ),
                "sparse_attention": sparse_available,
                "v19_certified_frontier": bool(
                    sparse_available and self.v19_release_enabled
                ),
                "pareto_v24": bool(
                    sparse_available and self.v24_release_enabled
                ),
                "pareto_v24_policy_id": v24_policy_id,
                "pareto_v24_candidate_id": v24_candidate_id,
                "pareto_v24_quality_knee": 75.0,
                "mechanistic_pareto": bool(
                    sparse_available and self.mechanistic_deployment_enabled
                ),
                "v19_experimental_long_horizon": bool(
                    sparse_available and _experimental_long_horizon_requested()
                ),
            },
        }

    @property
    def sparse_attention_available(self) -> bool:
        """Whether this process was explicitly built for approximate attention."""

        return _review_sparse_requested()

    def build(self, engine: str) -> BuiltNativeSession:
        import torch

        self._progress(2, "preflight", "检查模型文件与CUDA执行环境")

        from .adapters.conditioning_vae import (
            H3AudioVAEAdapter,
            H3VideoVAEAdapter,
            PackedQwen3VLT2AVConditioner,
        )
        from .adapters.sampling_mux import TurboClockMode
        from .adapters.real_vae import (
            decode_native_video,
            load_native_audio_vae,
            load_native_video_vae,
        )
        from .adapters.vae_compile import (
            enable_transformer_block_compile,
            prewarm_feed_forward_compile,
            prewarm_transformer_block_compile,
        )
        from .model import (
            SafeTensorSource,
            assemble_full_pruned_dit,
            comfy_kitchen_int8_kernel,
            comfy_kitchen_w4a8_kernel,
            load_full_silu_curve,
            load_h3_updates_from_safetensors,
            make_joint_action_scheduled_sparge_attention_sm89,
            sage_attention_sm89,
        )
        from .planner import (
            RTX4090Planner,
            review_combined_profiles_2026_08_12,
            review_fused_rms_profiles_2026_08_12,
            review_sparse_profiles_2026_08_12,
            validated_profiles_for_engine,
        )
        from .runtime import ImmutablePinnedModuleResidency, RuntimeConfig
        from .sm89_policy import configure_sm89_runtime

        launcher = normalize_launcher(engine)
        family = launcher_family(launcher)
        weight_tier = launcher_weight_tier(launcher)
        vram_profile = launcher_vram_profile(launcher)
        resource_backend = LAUNCHER_DEFINITIONS[launcher].backend
        base_engine = resolve_engine(family, "base")
        lora_engine = resolve_engine(family, "lora")
        preflight = self.preflight(launcher)
        if not preflight["ready"]:
            missing = ", ".join(
                name for name, ready in preflight["checks"].items() if not ready
            )
            raise RuntimeError(f"native {engine} runtime is incomplete: {missing}")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
            raise RuntimeError("native H3 release requires one RTX 4090 / SM89 GPU")
        research_w4_vram = os.environ.get(
            "H3_NATIVE_RESEARCH_W4_VRAM_GIB", ""
        ).strip()
        if research_w4_vram and weight_tier == "w4a8":
            try:
                research_w4_vram_gib = float(research_w4_vram)
            except ValueError as error:
                raise ValueError(
                    "H3_NATIVE_RESEARCH_W4_VRAM_GIB must be numeric"
                ) from error
            if not 8.0 <= research_w4_vram_gib <= 24.0:
                raise ValueError(
                    "H3_NATIVE_RESEARCH_W4_VRAM_GIB must lie inside [8, 24]"
                )
            # Research-only capacity sweep.  Leaving backend_profile unset
            # retains W4 kernels and execution preferences while allowing the
            # allocator to expose more than the public 8-GiB product tier.
            runtime_config = RuntimeConfig.for_cuda_device(
                weight_tier="w4a8",
                provisioned_limit_gib=research_w4_vram_gib,
                backend_profile=None,
            )
        else:
            runtime_config = RuntimeConfig.for_cuda_device(
                weight_tier=weight_tier,
                provisioned_limit_gib=resource_backend.provisioned_gib,
                backend_profile=resource_backend.profile_id,
            )
        # A VRAM profile is a physical contract, not a planner hint. Apply the
        # allocator ceiling before CUDA smoke tests or model construction, so
        # a 16GB launcher running on a 24GB development card cannot borrow the
        # spare capacity. The reserved headroom covers the CUDA context and
        # custom-kernel workspaces that are visible to NVML but not Torch.
        gc.collect()
        torch.cuda.empty_cache()
        physical_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        allocator_fraction = min(
            1.0, float(runtime_config.max_device_bytes) / physical_bytes
        )
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, 0)
        kernel_runtime = configure_sm89_runtime(
            quant_backend="cuda",
            smoke_test=True,
            require_w4a8=weight_tier == "w4a8",
        )
        review_sparse = _review_sparse_requested()
        # A certified V19 bundle contains request-local physical sparse action
        # schedules.  Loading the selector without the matching dispatching
        # backend would silently execute Dense Attention under a sparse plan
        # certificate, invalidating both timing and quality provenance.
        experimental_long = _experimental_long_horizon_requested()
        generic_sparse = (
            _generic_sparse_requested()
            or self.mechanistic_deployment_enabled
            or self.v24_release_enabled
            or self.v19_release_enabled
            or experimental_long
        )
        long_video_review = candidate_requested()
        review_fused_rms = _review_fused_rms_requested()
        if review_sparse:
            _prepare_review_sparse_import()
        v19_selector = None
        v19_bundle_source = None
        v19_bundle_digest = None
        pareto_policy_id = None
        pareto_candidate_id = None
        if self.mechanistic_deployment_enabled:
            from .planner import (
                H3MechanisticParetoRuntimeSelector,
                MECHANISTIC_DEPLOYMENT_POLICY_ID,
                load_h3_mechanistic_deployment_config,
            )

            admission_path = _mechanistic_admission_path()
            if admission_path is None:
                raise AssertionError("mechanistic admission path disappeared")
            loaded = load_h3_mechanistic_deployment_config(admission_path)
            v19_selector = H3MechanisticParetoRuntimeSelector(
                admission=loaded.admission,
                calibrated_video_token_interval=(
                    loaded.calibrated_video_token_interval
                ),
                maximum_runtime_promotions=loaded.maximum_runtime_promotions,
            )
            v19_bundle_source = str(loaded.source)
            v19_bundle_digest = loaded.source_sha256
            pareto_policy_id = MECHANISTIC_DEPLOYMENT_POLICY_ID
        elif self.v24_release_enabled:
            from .planner import (
                V24_FINAL_POLICY_ID,
                V24FinalParetoRuntimeSelector,
            )

            # The production service has one immutable Human-selected C02
            # surface. Historical knots live only in the offline research
            # compiler and cannot be selected by a service environment flag.
            v19_selector = V24FinalParetoRuntimeSelector()
            pareto_policy_id = V24_FINAL_POLICY_ID
            pareto_candidate_id = v19_selector.candidate.candidate_id
        elif self.v19_release_enabled:
            from .planner import (
                FIXED_TOPK_ACTION_IMPLEMENTATION,
                ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
                ROUND215_ACTION_IMPLEMENTATION,
                ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
                ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
                V19RuntimeSelector,
                build_v19_bootstrap_registry,
                capture_v19_runtime_fingerprint,
                load_v19_release_bundle,
            )

            sparse_package = importlib.import_module("spas_sage_attn")
            sparse_build_dir = Path(sparse_package.__file__).resolve().parent.parent
            serve_root = Path(__file__).resolve().parents[2]
            runtime_fingerprint = capture_v19_runtime_fingerprint(
                serve_root=serve_root,
                sparge_build_dir=sparse_build_dir,
                kernel_runtime=kernel_runtime,
            )
            registry = build_v19_bootstrap_registry(implementation_ids={
                "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
                "round215": ROUND215_ACTION_IMPLEMENTATION,
                "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
                "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
                "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
            })
            loaded_v19 = load_v19_release_bundle(
                _v19_release_bundle_path(),
                registry=registry,
            )
            v19_selector = V19RuntimeSelector(
                loaded_v19.catalog,
                runtime_digest=runtime_fingerprint.digest,
            )
            v19_bundle_source = str(loaded_v19.source)
            v19_bundle_digest = loaded_v19.bundle_digest
            pareto_policy_id = "h3_v19_certified_frontier"
        if experimental_long and not self.v24_release_enabled:
            from .planner import V19ExperimentalLongRuntimeSelector

            # This wrapper is intentionally not part of the signed release
            # bundle.  It overlays only measured 15-second research envelopes
            # and delegates every other request to the certified selector.
            v19_selector = V19ExperimentalLongRuntimeSelector(v19_selector)
            pareto_policy_id = "h3_v19_long_15s_round188_experimental_v1"
        if long_video_review:
            from .long_video_motion_detail import make_attention_backend

            attention_backend = make_attention_backend()
        elif generic_sparse:
            attention_backend = make_joint_action_scheduled_sparge_attention_sm89()
        else:
            attention_backend = sage_attention_sm89

        root = self.paths.model_root
        self._progress(7, "model_paths", "解析Linux本地模型权重")
        base = root / "diffusion_models" / (
            f"minimax_h3_{'ref2va' if family == 'reference' else 'fl2va'}_"
            + (
                "pruned_w4a8_mixed.safetensors"
                if weight_tier == "w4a8"
                else "pruned_int8_convrot.safetensors"
            )
        )
        text_source = root / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        video_checkpoint = root / "vae/minimax_h3_video_vae_fp16.safetensors"
        audio_checkpoint = root / "vae/minimax_h3_audio_vae_fp32.safetensors"
        lora_checkpoint = self._lora_checkpoint
        lora_profile = self._lora_profile
        latent_upscaler_checkpoint = (
            root
            / "latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors"
        )
        task_seconds: dict[str, float] = {}

        def timed(name, operation):
            started = time.perf_counter()
            result = operation()
            task_seconds[name] = time.perf_counter() - started
            return result

        # Compact hosts cannot retain the 13.5+ GiB packed Qwen slab. Under
        # WSL the release model is commonly linked into /mnt/c (9p/DrvFS),
        # where thousands of layer-local tensor reads dominate every new
        # prompt. Keep a byte-identical, reclaimable ext4 disk copy instead.
        # This is storage locality, not a second resident weight cache.
        stream_qwen_weights = not self.memory_profile.cache_qwen_weights
        text = (
            timed(
                "qwen_native_checkpoint",
                lambda: materialize_local_checkpoint(text_source),
            )
            if stream_qwen_weights
            else text_source
        )
        self._progress(12, "text_encoder", "准备Qwen文本编码器")
        qwen_storage = "native_cache" if text != text_source.resolve() else "source"
        qwen_layer_cache = (
            timed(
                "qwen_layer_cache",
                lambda: materialize_qwen_layer_cache(text),
            )
            if stream_qwen_weights
            else None
        )
        if stream_qwen_weights:
            if qwen_storage == "native_cache":
                print(
                    f"{self.memory_profile.label} Qwen storage: Linux native cache ready",
                    flush=True,
                )
            elif should_localize_checkpoint(text_source):
                print(
                    f"WARNING: {self.memory_profile.label} Qwen storage is still on a WSL cross-drive "
                    "mount; generation remains available but new-prompt latency "
                    "will be higher. Check Linux cache disk space or "
                    "H3_SERVE_LOCAL_MODEL_CACHE.",
                    flush=True,
                )
            if qwen_layer_cache is not None:
                print(
                    f"{self.memory_profile.label} Qwen streaming: execution-ordered layer cache ready",
                    flush=True,
                )
        conditioner = PackedQwen3VLT2AVConditioner(
            text,
            self.paths.minimax_source / "tokenizer",
            cache_pinned_weights=self.memory_profile.cache_qwen_weights,
            layer_cache_dir=qwen_layer_cache,
        )
        self._progress(18, "model_graphs", "并行装配DiT、VAE与二采模型")

        def prepare_dit():
            def load_lora_assets():
                updates = load_h3_updates_from_safetensors(
                    str(lora_checkpoint),
                    strength=1.0,
                    device="cpu",
                    dtype=torch.bfloat16,
                )
                curve = (
                    load_full_silu_curve(str(self.paths.turbo_curve))
                    if any(
                        name.endswith(".adaln_proj.linear")
                        for name in updates
                    )
                    else None
                )
                return updates, curve

            updates, curve = timed("dit_lora_assets", load_lora_assets)

            def assemble_dit():
                with SafeTensorSource(str(base)) as source:
                    return assemble_full_pruned_dit(
                        source,
                        device="cpu",
                        compute_dtype=torch.bfloat16,
                        int8_kernel=comfy_kitchen_int8_kernel,
                        w4a8_kernel=comfy_kitchen_w4a8_kernel,
                        attention_backend=attention_backend,
                        lora_updates=updates,
                        full_silu_curve=curve,
                    )

            model = timed("dit_graph_assembly", assemble_dit)
            model.eval().requires_grad_(False)
            research_pin_raw = os.environ.get(
                "H3_NATIVE_RESEARCH_PIN_TRANSFORMER_GIB", ""
            ).strip()
            pin_budget_bytes = (
                None
                if self.memory_profile.pin_transformer_budget_gib is None
                else int(
                    self.memory_profile.pin_transformer_budget_gib * 1024**3
                )
            )
            if research_pin_raw:
                try:
                    research_pin_gib = float(research_pin_raw)
                except ValueError as error:
                    raise ValueError(
                        "H3_NATIVE_RESEARCH_PIN_TRANSFORMER_GIB must be numeric"
                    ) from error
                if research_pin_gib < 0.0:
                    raise ValueError(
                        "H3_NATIVE_RESEARCH_PIN_TRANSFORMER_GIB cannot be negative"
                    )
                pin_budget_bytes = int(research_pin_gib * 1024**3)
            research_resident_raw = os.environ.get(
                "H3_NATIVE_RESEARCH_RESIDENT_BLOCKS", ""
            ).strip()
            try:
                resident_blocks = (
                    int(research_resident_raw)
                    if research_resident_raw
                    else int(self.memory_profile.resident_transformer_blocks)
                )
            except ValueError as error:
                raise ValueError(
                    "H3_NATIVE_RESEARCH_RESIDENT_BLOCKS must be an integer"
                ) from error
            block_count = len(model.block_stack.blocks)
            if not 0 <= resident_blocks < block_count:
                raise ValueError(
                    "H3_NATIVE_RESEARCH_RESIDENT_BLOCKS must leave at least one block offloaded"
                )
            pin_transformer = (
                self.memory_profile.pin_model_weights
                or self.memory_profile.pin_transformer_weights
                or (pin_budget_bytes is not None and pin_budget_bytes > 0)
            )
            copy_transformer = (
                self.memory_profile.copy_model_weights
                or self.memory_profile.copy_transformer_weights
            )
            release_threshold = 256 * 1024**2
            copied_since_release = 0

            def release_copied_checkpoint_pages(copied_bytes: int) -> None:
                nonlocal copied_since_release
                copied_since_release += int(copied_bytes)
                if copied_since_release >= release_threshold:
                    # Packing touches mmap-backed source pages. Evict clean
                    # pages incrementally so cold construction never holds a
                    # complete 12.5 GiB source copy beside the pinned master.
                    drop_file_page_cache(base)
                    copied_since_release = 0

            residency = ImmutablePinnedModuleResidency(
                "transformer", model,
                pin_host_weights=pin_transformer,
                copy_host_weights=copy_transformer,
                source_copied=(
                    release_copied_checkpoint_pages
                    if (
                        self.memory_profile.pin_transformer_weights
                        or (
                            pin_budget_bytes is not None
                            and pin_budget_bytes > 0
                        )
                    )
                    else None
                ),
                pin_host_budget_bytes=pin_budget_bytes,
                pin_host_module_prefixes=(
                    tuple(
                        f"block_stack.blocks.{index}"
                        for index in range(resident_blocks, block_count)
                    )
                    if pin_budget_bytes is not None
                    else None
                ),
            )
            try:
                timed("dit_pin_host", residency.prepare_host)
            except (RuntimeError, MemoryError) as error:
                if not pin_transformer:
                    raise
                # Other applications can consume RAM between admission and
                # model construction.  Preserve service availability by
                # falling back to the already validated pageable masters.
                print(
                    f"WARNING: {self.memory_profile.label} DiT pinned cache "
                    f"unavailable ({error}); using low-RAM streaming",
                    flush=True,
                )
                gc.collect()
                residency = ImmutablePinnedModuleResidency(
                    "transformer", model,
                    pin_host_weights=False,
                    copy_host_weights=False,
                )
                timed("dit_streaming_fallback", residency.prepare_host)
            if residency.host_is_pinned:
                # The immutable pinned masters now own every byte used by DiT.
                # Evict the clean checkpoint mmap pages touched while packing
                # so they do not count twice against the 32 GiB host budget.
                drop_file_page_cache(base)
            return residency

        def prepare_video():
            model, mean, std = load_native_video_vae(
                self.paths.minimax_source,
                video_checkpoint,
                device="cpu",
                tile_size=288,
                compile_feed_forward=True,
            )
            if long_video_review or generic_sparse:
                # Install both eager and compiled dispatch paths while the
                # model is still on CPU.  The joint scheduler and retained
                # long-video route share this mature mechanical baseline;
                # request-local execution plans still choose the path.
                enable_transformer_block_compile(model)
            residency = ImmutablePinnedModuleResidency(
                "video_vae", model,
                pin_host_weights=self.memory_profile.pin_model_weights,
                copy_host_weights=self.memory_profile.copy_model_weights,
            )
            residency.prepare_host()
            return residency, mean, std

        def prepare_audio():
            model = load_native_audio_vae(
                self.paths.lightx_source,
                self.paths.minimax_source,
                audio_checkpoint,
                device="cpu",
            )
            residency = ImmutablePinnedModuleResidency(
                "audio_vae", model,
                pin_host_weights=self.memory_profile.pin_model_weights,
                copy_host_weights=self.memory_profile.copy_model_weights,
            )
            residency.prepare_host()
            return residency

        def prepare_latent_upscaler():
            from .latent_upscaler import load_h3_latent_upscaler

            model = load_h3_latent_upscaler(latent_upscaler_checkpoint)
            residency = ImmutablePinnedModuleResidency(
                "latent_upscaler",
                model,
                pin_host_weights=self.memory_profile.pin_model_weights,
                copy_host_weights=self.memory_profile.copy_model_weights,
            )
            residency.prepare_host()
            return residency

        started = time.perf_counter()
        if self.memory_profile.parallel_model_build:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="h3-native-startup") as pool:
                futures = {
                    pool.submit(timed, "dit_cache", prepare_dit): "dit",
                    pool.submit(timed, "video_vae_cache", prepare_video): "video",
                    pool.submit(timed, "audio_vae_cache", prepare_audio): "audio",
                }
                if resource_backend.second_sampling_levels:
                    futures[
                        pool.submit(
                            timed, "latent_upscaler_cache", prepare_latent_upscaler
                        )
                    ] = "latent_upscaler"
                if self.memory_profile.cache_qwen_weights:
                    futures[
                        pool.submit(timed, "qwen_cache", conditioner.prepare_host_cache)
                    ] = "qwen"
                prepared = {}
                completed_builds = 0
                for future in as_completed(futures):
                    prepared[futures[future]] = future.result()
                    completed_builds += 1
                    self._progress(
                        18 + 52 * completed_builds / len(futures),
                        "model_graphs",
                        f"模型组件已准备 {completed_builds}/{len(futures)}",
                    )
        else:
            # Low-capacity hosts trade cold-start latency for a lower assembly
            # peak. The resulting immutable models and generation math are
            # identical to the parallel path.
            prepared = {
                "dit": timed("dit_cache", prepare_dit),
                "video": timed("video_vae_cache", prepare_video),
                "audio": timed("audio_vae_cache", prepare_audio),
            }
            self._progress(58, "model_graphs", "DiT与VAE模型组件已装配")
            if resource_backend.second_sampling_levels:
                prepared["latent_upscaler"] = timed(
                    "latent_upscaler_cache", prepare_latent_upscaler
                )
            if self.memory_profile.cache_qwen_weights:
                timed("qwen_cache", conditioner.prepare_host_cache)

        transformer = prepared["dit"]
        video_vae, video_mean, video_std = prepared["video"]
        audio_vae = prepared["audio"]
        latent_upscaler = prepared.get("latent_upscaler")

        self._progress(74, "vae_warmup", "预热视频VAE编译图")

        def prewarm_video_vae_compile():
            video_vae.move_to("cuda:0", non_blocking=False)
            try:
                if long_video_review or generic_sparse:
                    prewarm_transformer_block_compile(video_vae.value)
                else:
                    prewarm_feed_forward_compile(video_vae.value)
                torch.cuda.synchronize()
            finally:
                video_vae.move_to("cpu", non_blocking=False)
                torch.cuda.empty_cache()

        timed("video_vae_compile_warmup", prewarm_video_vae_compile)

        self._progress(88, "host_memory", "整理并锁定模型主机内存")

        def release_startup_host_scratch() -> None:
            """Return one-time graph assembly scratch to the OS.

            Live immutable weights already reside in compact registered host
            slabs. CPython can release superseded source tensors while glibc
            still keeps their multi-gigabyte arenas mapped. This trim runs once
            after startup/prewarm and never enters the request path.
            """

            gc.collect()
            try:
                libc = ctypes.CDLL(None)
                malloc_trim = libc.malloc_trim
                malloc_trim.argtypes = [ctypes.c_size_t]
                malloc_trim.restype = ctypes.c_int
                malloc_trim(0)
            except (AttributeError, OSError):
                pass

        timed("release_startup_host_scratch", release_startup_host_scratch)

        def video_decoder(model, latents, frame_count):
            return decode_native_video(
                model,
                latents,
                video_mean,
                video_std,
                frame_count,
                output_dtype="uint8",
                cuda_allocator_ceiling_bytes=runtime_config.max_device_bytes,
            )

        frame_adapter = H3VideoVAEAdapter(
            video_vae.value,
            latents_mean=video_mean.tolist(),
            latents_std=video_std.tolist(),
            encode_precision=(
                "fp16_weights_fp32_posterior"
                if weight_tier == "w4a8"
                else "full_fp32"
            ),
        )

        def frame_encoder(_model, request):
            if request.reference_images or request.reference_videos:
                return frame_adapter.encode_references(request)
            return frame_adapter.encode_conditioning(request)

        audio_adapter = H3AudioVAEAdapter(
            audio_vae.value,
            latents_mean=audio_vae.value.latents_mean.tolist(),
            latents_std=audio_vae.value.latents_std.tolist(),
        )

        def audio_condition_encoder(_model, request):
            from .adapters.conditioning_vae import prepare_reference_audios

            return tuple(
                audio_adapter.encode(item.waveform.to("cuda:0"))
                for item in prepare_reference_audios(request)
            )

        def audio_decoder(model, latents):
            flattened = latents.permute(0, 2, 1, 3).reshape(
                2, 32, latents.shape[-1]
            )
            with torch.inference_mode():
                return model.decode(flattened, stereo_batch=True, return_cpu=True)

        # The request chooses base or LoRA without rebuilding this graph.  Give
        # the shared planner the measured profiles for both routes.
        profiles = (
            validated_profiles_for_engine(base_engine)
            + validated_profiles_for_engine(lora_engine)
        )
        if generic_sparse:
            profiles += tuple(
                profile
                for profile in review_sparse_profiles_2026_08_12()
                if base_engine in profile.supported_engines
                or lora_engine in profile.supported_engines
                or family == "reference"
            )
        if review_fused_rms:
            profiles += tuple(
                profile
                for profile in review_fused_rms_profiles_2026_08_12()
                if base_engine in profile.supported_engines
                or lora_engine in profile.supported_engines
                or family == "reference"
            )
        if generic_sparse and review_fused_rms:
            profiles += tuple(
                profile
                for profile in review_combined_profiles_2026_08_12()
                if base_engine in profile.supported_engines
                or lora_engine in profile.supported_engines
                or family == "reference"
            )
        session = NativeT2AVHotSession(
            engine=base_engine,
            conditioner=conditioner,
            transformer=transformer,
            video_vae=video_vae,
            audio_vae=audio_vae,
            latent_upscaler=latent_upscaler,
            decode_video=video_decoder,
            decode_audio=audio_decoder,
            encode_video_conditioning=frame_encoder,
            encode_audio_conditioning=audio_condition_encoder,
            output_root=self.paths.output_root,
            runtime_config=runtime_config,
            planner=RTX4090Planner(
                profiles,
                device_budget_bytes=runtime_config.max_device_bytes,
                allow_experimental=generic_sparse or review_fused_rms,
            ),
            attention_backend=attention_backend,
            v19_selector=v19_selector,
            turbo_clock_mode=(
                TurboClockMode.DUAL_SHIFT
                if lora_profile.clock_mode == "dual_shift"
                else TurboClockMode.SHARED_VIDEO
            ),
            lora_video_shift=lora_profile.video_shift,
            lora_audio_shift=lora_profile.audio_shift,
            lora_profile_id=lora_profile.profile_id,
            lora_recommended_steps=lora_profile.recommended_steps,
            lora_default_steps=lora_profile.default_steps,
            prefer_pinned_weight_prefetch=(
                weight_tier == "w4a8"
                and transformer.host_pinned_bytes > 0
            ),
            resident_transformer_blocks=(
                self.memory_profile.resident_transformer_blocks
            ),
        )
        self._progress(98, "finalize", "完成调度器与运行时会话初始化")
        return BuiltNativeSession(
            session=session,
            startup_seconds=time.perf_counter() - started,
            startup_tasks=task_seconds,
            qwen_storage=qwen_storage,
            qwen_layer_cache=qwen_layer_cache is not None,
            host_memory_profile=self.memory_profile.key,
            dit_host_pinned=transformer.host_is_pinned,
            dit_host_cache_gib=transformer.host_allocated_bytes / 1024**3,
            dit_host_pinned_gib=transformer.host_pinned_bytes / 1024**3,
            dit_host_pinned_fraction=transformer.host_pinned_fraction,
            v19_release_bundle=v19_bundle_source,
            v19_release_digest=v19_bundle_digest,
            pareto_policy_id=pareto_policy_id,
            pareto_candidate_id=pareto_candidate_id,
            launcher=launcher,
            weight_tier=weight_tier,
            vram_profile=vram_profile,
            allocator_ceiling_gib=runtime_config.max_device_bytes / 1024**3,
            lora_checkpoint=str(lora_checkpoint),
            lora_profile_id=lora_profile.profile_id,
            lora_display_name=lora_profile.display_name,
            lora_recommended_steps=lora_profile.recommended_steps,
            lora_default_steps=lora_profile.default_steps,
        )


__all__ = ["BuiltNativeSession", "NativeSessionFactory", "NativeSessionPaths"]
