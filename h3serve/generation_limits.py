from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import (
    ASPECT_RATIOS,
    MAX_DURATION_SECONDS,
    MAX_NATIVE_PIXEL_FRAMES,
    RESOLUTIONS,
    max_duration_for_geometry,
    resolve_geometry,
)


MIN_PRESET_DURATION_SECONDS = 1.0
MAX_PRESET_DURATION_SECONDS = float(MAX_DURATION_SECONDS)


def detect_gpu_vram_gib() -> float | None:
    """Read total VRAM for display only; it never dictates user limits."""

    command = [
        "nvidia-smi", "--query-gpu=memory.total",
        "--format=csv,noheader,nounits", "--id=0",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=2,
        )
        mib = float(result.stdout.splitlines()[0].strip())
        if mib <= 0:
            return None
        raw_gib = mib / 1024
        marketed_gib = round(raw_gib)
        return float(marketed_gib) if abs(raw_gib - marketed_gib) < 0.75 else round(raw_gib, 2)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def default_preset_limits() -> dict[str, dict[str, float]]:
    """Return the former 24 GB release envelope as an editable initial table."""

    return {
        resolution: {
            ratio: max_duration_for_geometry(
                *resolve_geometry(resolution, ratio),
                MAX_NATIVE_PIXEL_FRAMES,
            )
            for ratio in ASPECT_RATIOS
        }
        for resolution in RESOLUTIONS
    }


def _normalize_limits(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise ValueError("preset_limits must be an object")
    normalized: dict[str, dict[str, float]] = {}
    for resolution in RESOLUTIONS:
        row = value.get(resolution)
        if not isinstance(row, dict):
            raise ValueError(f"preset_limits.{resolution} must be an object")
        normalized[resolution] = {}
        for ratio in ASPECT_RATIOS:
            try:
                seconds = float(row[ratio])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"preset_limits.{resolution}.{ratio} must be numeric"
                ) from error
            if not MIN_PRESET_DURATION_SECONDS <= seconds <= MAX_PRESET_DURATION_SECONDS:
                raise ValueError(
                    f"preset_limits.{resolution}.{ratio} must be between "
                    f"{MIN_PRESET_DURATION_SECONDS:g} and {MAX_PRESET_DURATION_SECONDS:g} seconds"
                )
            normalized[resolution][ratio] = round(seconds * 2) / 2
    return normalized


@dataclass(frozen=True)
class GenerationLimitPolicy:
    preset_limits: dict[str, dict[str, float]] = field(
        default_factory=default_preset_limits
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "preset_limits", _normalize_limits(self.preset_limits))

    def public(self, detected_vram_gib: float | None) -> dict[str, Any]:
        return {
            "detected_vram_gib": detected_vram_gib,
            "preset_limits": self.preset_limits,
            "max_by_preset": self.preset_limits,
            "resolutions": list(RESOLUTIONS),
            "aspect_ratios": list(ASPECT_RATIOS),
            "duration_range": {
                "min": MIN_PRESET_DURATION_SECONDS,
                "max": MAX_PRESET_DURATION_SECONDS,
                "step": 0.5,
            },
            "meaning": (
                "Operator-defined submission ceilings for each resolution and "
                "aspect-ratio pair; detected VRAM is informational only."
            ),
        }


def settings_path(data_dir: Path) -> Path:
    return data_dir / "settings" / "generation_limits.json"


def load_generation_limit_policy(data_dir: Path) -> GenerationLimitPolicy:
    try:
        document = json.loads(settings_path(data_dir).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("settings document must be an object")
        # Migrate the short-lived mode/manual_vram_gib format to defaults.
        limits = document.get("preset_limits")
        # New preset stops may be added after an operator has tuned the matrix.
        # Preserve every existing row and seed only the newly introduced rows;
        # malformed fields in existing rows still fail closed.
        if isinstance(limits, dict) and any(key not in limits for key in RESOLUTIONS):
            limits = dict(limits)
            defaults = default_preset_limits()
            for resolution in RESOLUTIONS:
                limits.setdefault(resolution, defaults[resolution])
        return GenerationLimitPolicy(limits) if limits is not None else GenerationLimitPolicy()
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return GenerationLimitPolicy()


def persist_generation_limit_policy(
    data_dir: Path,
    policy: GenerationLimitPolicy,
) -> None:
    path = settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"preset_limits": policy.preset_limits}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "GenerationLimitPolicy",
    "MAX_PRESET_DURATION_SECONDS",
    "MIN_PRESET_DURATION_SECONDS",
    "default_preset_limits",
    "detect_gpu_vram_gib",
    "load_generation_limit_policy",
    "persist_generation_limit_policy",
]
