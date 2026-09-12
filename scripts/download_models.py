#!/usr/bin/env python3
"""Download the release model contract into the configured model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT.parent / "models" / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, artifact: dict) -> None:
    if not path.is_file():
        raise SystemExit(f"模型文件不存在：{path}")
    actual_size = path.stat().st_size
    if actual_size != int(artifact["bytes"]):
        raise SystemExit(f"模型大小不匹配：{path} ({actual_size} != {artifact['bytes']})")
    actual_hash = _sha256(path)
    if actual_hash != artifact["sha256"]:
        raise SystemExit(f"模型 SHA-256 不匹配：{path}\n{actual_hash} != {artifact['sha256']}")
    print(f"验证通过：{path.name}  sha256={actual_hash}", flush=True)


def _download_local_dir(
    model_root: Path,
    target: Path,
    artifact: dict,
) -> Path:
    """Map a repository filename onto the release's typed model folders."""

    source_path = Path(artifact["filename"])
    install_path = Path(artifact["install_path"])
    if source_path == install_path:
        return model_root
    if source_path.name == install_path.name:
        # Some repositories publish a checkpoint at their root while this
        # service keeps all DiT/LoRA files in typed directories.
        return target.parent
    raise SystemExit(
        "下载清单的源文件名无法映射到安装路径："
        f"{source_path} -> {install_path}"
    )


def _download_direct_url(target: Path, artifact: dict) -> Path:
    """Download one immutable non-Hugging-Face artifact atomically."""

    temporary = target.with_name(f".{target.name}.download")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(artifact["url"], timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_root", type=Path, nargs="?", default=ROOT.parent / "models",
        help="默认写入发布包的 models/ 目录",
    )
    parser.add_argument(
        "--accept-model-license",
        action="store_true",
        help=(
            "确认你已阅读并接受 MiniMax H3 Community License、LoRA 许可证"
            "以及 FlashVSR/FlashVSR-v1.1 Apache-2.0 许可证。"
        ),
    )
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--skip-local-qwen-cache", action="store_true",
        help="不为WSL /mnt/c中的Qwen权重创建Linux原生高速副本",
    )
    args = parser.parse_args()
    if not args.accept_model_license:
        parser.error("下载前必须传入 --accept-model-license")
    if not args.verify_only:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise SystemExit("缺少 huggingface_hub；请先安装项目 Python 依赖") from error

    model_root = args.model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    qwen_target = None
    for artifact in artifacts:
        if args.reference_only and artifact["role"] not in {
            "reference_diffusion_model",
            "reference_diffusion_model_w4a8",
            "text_encoder",
            "video_vae",
            "audio_vae",
        }:
            continue
        if args.base_only and artifact["profile"] in {"turbo", "reference"}:
            continue
        target = model_root / artifact["install_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not args.verify_only:
            source = (
                artifact["url"]
                if "url" in artifact
                else f"{artifact['repo']}@{artifact['revision']}/{artifact['filename']}"
            )
            print(f"下载 {source}", flush=True)
            if "url" in artifact:
                downloaded = _download_direct_url(target, artifact)
            else:
                local_dir = _download_local_dir(model_root, target, artifact)
                downloaded = Path(hf_hub_download(
                    repo_id=artifact["repo"],
                    revision=artifact["revision"],
                    filename=artifact["filename"],
                    local_dir=local_dir,
                ))
            if downloaded.resolve() != target.resolve():
                raise SystemExit(f"下载器返回了非预期路径：{downloaded} != {target}")
        _verify(target, artifact)
        if artifact["role"] == "text_encoder":
            qwen_target = target

    if qwen_target is not None and not args.skip_local_qwen_cache:
        sys.path.insert(0, str(ROOT.parent))
        from h3serve.native_engine.local_checkpoint_cache import (
            default_cache_root,
            materialize_local_checkpoint,
            should_localize_checkpoint,
        )

        if should_localize_checkpoint(qwen_target):
            print(
                "检测到Qwen权重位于WSL跨盘文件系统；正在准备64GB档高速副本…",
                flush=True,
            )
            selected = materialize_local_checkpoint(qwen_target)
            if selected.resolve() == qwen_target.resolve():
                print(
                    "警告：高速副本未创建（通常是Linux磁盘空间不足）；"
                    "服务仍可运行，但64GB档的新提示词编码会较慢。",
                    flush=True,
                )
            else:
                print(f"64GB高速副本已就绪：{selected}", flush=True)
        else:
            print(
                f"Qwen权重已位于原生文件系统，无需额外副本：{qwen_target}",
                flush=True,
            )


if __name__ == "__main__":
    main()
