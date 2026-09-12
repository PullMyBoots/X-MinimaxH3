#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="$(sed -n 's/__version__ = "\([^"]*\)"/\1/p' "${release_root}/h3serve/__init__.py")"
project_version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "${release_root}/pyproject.toml" | head -n 1)"
manifest_version="$(sed -n 's/^[[:space:]]*"version": "\([^"]*\)",\{0,1\}$/\1/p' "${release_root}/RELEASE_MANIFEST.json")"
if [[ -z "${version}" || "${version}" != "${project_version}" || "${version}" != "${manifest_version}" ]]; then
  echo "Release version mismatch: code=${version:-missing}, project=${project_version:-missing}, manifest=${manifest_version:-missing}" >&2
  exit 1
fi
destination="${1:-${release_root}/dist}"
archive="${destination}/x-minimaxh3-${version}-linux-x86_64-sm89.tar.gz"

mkdir -p "${destination}"
tar -C "${release_root}" -czf "${archive}" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
  --exclude='.git' --exclude='.env.local' --exclude='.pytest_cache' \
  README.md README.zh-CN.md RELEASE_NOTES.md VALIDATION.md RELEASE_MANIFEST.json THIRD_PARTY_NOTICES.md LICENSE SECURITY.md CONTRIBUTING.md \
  setup.sh run.sh stop.sh doctor.sh test.sh .env.example .gitattributes .github \
  pyproject.toml requirements.txt requirements.lock requirements-flashvsr.lock \
  server.py smoke_generation.py .gitignore \
  h3serve static ai-prompt-guides backends benchmarks integrations scripts tests docs patches assets \
  third_party third_party_licenses wheels models/manifest.json

# These source modules live below a directory named `runtime`.  A basename-wide
# tar exclusion such as --exclude='runtime' silently drops them together with
# the generated top-level runtime, producing an archive that cannot import the
# native engine.  The generated top-level directories are already absent from
# the explicit member list above, so assert the required package surface here.
for required_member in \
  h3serve/native_engine/runtime/__init__.py \
  h3serve/native_engine/runtime/config.py \
  h3serve/native_engine/runtime/offload.py \
  h3serve/native_engine/runtime/pinned_pool.py \
  h3serve/native_engine/runtime/residency.py \
  h3serve/native_engine/runtime/streams.py
do
  tar -tzf "${archive}" "${required_member}" >/dev/null || {
    echo "Release archive is missing required member: ${required_member}" >&2
    exit 1
  }
done

for required_member in \
  ai-prompt-guides/00-README.txt \
  ai-prompt-guides/01-FL2VA-Single-Video.txt \
  ai-prompt-guides/02-Ref2VA-Single-Video.txt \
  ai-prompt-guides/03-FL2VA-Long-Video-Online.txt \
  ai-prompt-guides/04-Ref2VA-Long-Video-Online.txt \
  ai-prompt-guides/05-FL2VA-Long-Video-JSON.txt \
  ai-prompt-guides/06-Ref2VA-Long-Video-JSON.txt \
  h3serve/assets/face_detection_yunet_2023mar.onnx \
  h3serve/native_engine/latent_upscaler.py \
  h3serve/native_engine/audio_spine.py \
  h3serve/face_refine_scheduler.py \
  static/infinite-video.js \
  third_party/flashvsr/LICENSE \
  wheels/block_sparse_attn-0.0.2-cp311-cp311-linux_x86_64.whl
do
  tar -tzf "${archive}" "${required_member}" >/dev/null || {
    echo "Release archive is missing required member: ${required_member}" >&2
    exit 1
  }
done

if tar -tzf "${archive}" | grep -Eq '(^|/)(\.env\.local|__pycache__|\.pytest_cache)(/|$)|\.py[co]$'; then
  echo "Release archive contains a local environment or cache artifact" >&2
  exit 1
fi

echo "Built ${archive}"
