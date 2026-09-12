#!/usr/bin/env python3
"""Read-only release preflight for every private SM89 H3 executor."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3serve.config import ServicePaths  # noqa: E402
from h3serve.deployment_profiles import LAUNCHER_DEFINITIONS  # noqa: E402
from h3serve.memory_policy import (  # noqa: E402
    detect_host_memory,
    resolve_host_memory_profile,
)
from h3serve.models import model_status  # noqa: E402
from h3serve.native_engine.session_factory import (  # noqa: E402
    NativeSessionFactory,
    NativeSessionPaths,
)
from h3serve.native_engine.sm89_policy import configure_sm89_runtime  # noqa: E402
from h3serve.upscaler import FlashVSRUpscaler  # noqa: E402


PINNED_MINIMAX = "8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea"
PINNED_LIGHTX = "205d5c872d01557935dc87d67156f4f94069ea65"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        marker = path / ".source-revision"
        try:
            value = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value if len(value) == 40 else None


def main() -> None:
    paths = ServicePaths.defaults(ROOT)
    memory_status = detect_host_memory()
    memory_profile = None
    memory_error = None
    try:
        memory_profile = resolve_host_memory_profile("auto", memory_status)
    except RuntimeError as error:
        memory_error = str(error)
    checks: dict[str, object] = {
        "python_3_10": sys.version_info[:2] == (3, 10),
        "linux_x86_64": sys.platform.startswith("linux")
        and platform.machine() == "x86_64",
        "minimax_source_revision": git_head(paths.minimax_source_dir)
        == PINNED_MINIMAX,
        "lightx_source_revision": git_head(paths.lightx_source_dir)
        == PINNED_LIGHTX,
        "host_memory_supported": memory_profile is not None,
    }
    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        checks["sm89"] = bool(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (8, 9)
        )
    except Exception as error:
        checks["torch_error"] = str(error)
        checks["cuda_available"] = checks["sm89"] = False

    kernel: dict[str, object]
    try:
        kernel = configure_sm89_runtime(
            quant_backend="cuda", smoke_test=True, require_w4a8=True
        ).to_dict()
        kernel["ready"] = True
    except Exception as error:
        kernel = {"ready": False, "error": str(error)}

    factory = NativeSessionFactory(
        NativeSessionPaths(
            model_root=paths.model_dir,
            minimax_source=paths.minimax_source_dir,
            lightx_source=paths.lightx_source_dir,
            turbo_curve=paths.turbo_curve_path,
            output_root=paths.output_dir,
        )
    )
    runtimes = {
        launcher: factory.preflight(launcher)
        for launcher in LAUNCHER_DEFINITIONS
    }
    models = model_status(paths.model_dir)
    temporal_second_sampling = FlashVSRUpscaler(paths).status()
    manifest = json.loads(
        (ROOT / "models/manifest.json").read_text(encoding="utf-8")
    )
    verified = {}
    for artifact in manifest["artifacts"]:
        relative = artifact["install_path"]
        path = paths.model_dir / relative
        size_ok = path.is_file() and path.stat().st_size == int(artifact["bytes"])
        verified[relative] = {
            "exists": path.is_file(),
            "size_ok": size_ok,
            "sha256_ok": sha256(path) == artifact["sha256"] if size_ok else False,
        }

    result = {
        "checks": checks,
        "host_memory": {
            **memory_status.public(),
            "selected_profile": None if memory_profile is None else memory_profile.key,
            "error": memory_error,
            "wsl": "microsoft" in platform.release().lower(),
            "deployment_hint": (
                "A 64GB Windows host must expose about 58GiB to WSL2; see "
                "docs/DEPLOY_64GB_WSL.md"
                if memory_profile is None
                and "microsoft" in platform.release().lower()
                else None
            ),
        },
        "models": models,
        "verified_models": verified,
        "native_runtimes": runtimes,
        "sm89_kernel_runtime": kernel,
        "second_sampling": {
            "default_method": "temporal",
            "ready": bool(temporal_second_sampling["ready"]),
            "methods": {
                "temporal": temporal_second_sampling,
                "h3": {
                    "ready": True,
                    "implementation": "native_h3_clean_av_latent_refinement",
                },
            },
        },
        "end_to_end_runtime_ready": bool(
            models["ready"]
            and all(runtime["ready"] for runtime in runtimes.values())
            and kernel.get("ready") is True
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    scalar_checks_ok = all(
        value is True or isinstance(value, str) for value in checks.values()
    )
    verified_ok = all(item["sha256_ok"] for item in verified.values())
    runtime_ok = all(item["ready"] for item in runtimes.values())
    raise SystemExit(
        0
        if scalar_checks_ok
        and verified_ok
        and runtime_ok
        and kernel.get("ready") is True
        else 1
    )


if __name__ == "__main__":
    main()
