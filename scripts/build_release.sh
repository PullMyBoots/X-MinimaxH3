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
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  README.md README.zh-CN.md VALIDATION.md RELEASE_MANIFEST.json THIRD_PARTY_NOTICES.md LICENSE SECURITY.md CONTRIBUTING.md \
  setup.sh run.sh stop.sh doctor.sh test.sh .env.example .gitattributes .github \
  pyproject.toml requirements.txt requirements.lock requirements-flashvsr.lock \
  server.py smoke_generation.py .gitignore \
  h3serve static backends benchmarks integrations scripts tests docs patches \
  third_party_licenses models/manifest.json

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

echo "Built ${archive}"
