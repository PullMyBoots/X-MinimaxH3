#!/usr/bin/env bash
set -euo pipefail

# The Windows checkout remains the source of truth. This script creates a
# disposable Linux-native execution mirror for import-heavy Python/CUDA work.
canonical_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
linux_root="${H3_LINUX_RUNTIME_ROOT:-/root/h3-new-serve-runtime}"
linux_worktree="${linux_root}/worktree"
python_env="${H3_LINUX_PYTHON_ENV:-/root/miniconda3/envs/h3serve213cu130}"
flashvsr_env="${H3_LINUX_FLASHVSR_ENV:-/root/.local/share/x-minimaxh3/runtime/flashvsr-venv}"
workspace_root="$(cd "${canonical_root}/../.." && pwd)"
minimax_source="${workspace_root}/subprojects-main/main/MiniMax-H3"

# LightX2V used to live in backend-compare.  The research workspace now keeps
# the checkout in main/knowledge, while some older workspaces still use the
# original location.  Resolve both layouts and keep an explicit override for
# external or relocated checkouts.
lightx_source="${H3_LINUX_LIGHTX_SOURCE:-}"
if [[ -z "${lightx_source}" ]]; then
  for candidate in \
    "${workspace_root}/subprojects-main/main/knowledge/projects/inference_frameworks/lightx2v" \
    "${workspace_root}/subprojects-main/backend-compare/sources/LightX2V"; do
    if [[ -f "${candidate}/pyproject.toml" ]]; then
      lightx_source="${candidate}"
      break
    fi
  done
fi

case "${linux_root}" in
  /root/h3-new-serve-runtime|/root/h3-new-serve-runtime/*) ;;
  *)
    echo "Refusing an unexpected Linux runtime root: ${linux_root}" >&2
    exit 2
    ;;
esac

[[ -x "${python_env}/bin/python" ]] || {
  echo "Missing validated Linux Python environment: ${python_env}" >&2
  exit 1
}
[[ -x "${flashvsr_env}/bin/python" ]] || {
  echo "Missing validated FlashVSR environment: ${flashvsr_env}" >&2
  exit 1
}
"${flashvsr_env}/bin/python" -c 'import av, torch; assert torch.cuda.is_available()' || {
  echo "Incomplete FlashVSR environment: ${flashvsr_env}" >&2
  exit 1
}
[[ -d "${minimax_source}" ]] || { echo "Missing MiniMax-H3 source: ${minimax_source}" >&2; exit 1; }
[[ -n "${lightx_source}" && -f "${lightx_source}/pyproject.toml" ]] || {
  echo "Missing LightX2V source. Set H3_LINUX_LIGHTX_SOURCE to a valid checkout." >&2
  exit 1
}

mkdir -p \
  "${linux_worktree}" \
  "${linux_root}/cache/cuda" \
  "${linux_root}/cache/huggingface" \
  "${linux_root}/cache/pycache" \
  "${linux_root}/cache/torch" \
  "${linux_root}/cache/torchinductor" \
  "${linux_root}/cache/triton" \
  "${linux_root}/cache/xdg" \
  "${linux_root}/tmp" \
  "${linux_root}/vendor/MiniMax-H3" \
  "${linux_root}/vendor/LightX2V"

rsync -a --delete-delay \
  --exclude '/models' \
  --exclude '/benchmarks' \
  --exclude '/docs' \
  --exclude '/experiments' \
  --exclude '/fl2va-test' \
  --exclude '/knowledge' \
  --exclude '/output' \
  --exclude '/tests' \
  --exclude '/third_party' \
  --exclude '/third_party_licenses' \
  --exclude '/wheels' \
  --exclude '/workspace' \
  --exclude '/runtime/venv' \
  --exclude '/runtime/flashvsr-venv' \
  --exclude '/runtime/benchmarks' \
  --exclude '/runtime/calibration' \
  --exclude '/runtime/logs' \
  --exclude '/runtime/pareto' \
  --exclude '/runtime/release' \
  --exclude '/runtime/tmp' \
  --exclude '/data' \
  --exclude '/.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${canonical_root}/" "${linux_worktree}/"

# The main mirror excludes the large third_party tree. Copy the narrowed
# FlashVSR runtime explicitly so a clean mirror never depends on stale files
# left by an older release.
mkdir -p "${linux_worktree}/third_party/flashvsr"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "${canonical_root}/third_party/flashvsr/" \
  "${linux_worktree}/third_party/flashvsr/"

# Bind the disposable Linux mirror to its Windows source tree.  start/stop can
# then identify a mirrored server as belonging to this release without relying
# on inode equality between two different filesystems.
printf '%s\n' "${canonical_root}" > "${linux_worktree}/.h3-release-source"

if [[ "${H3_LINUX_SYNC_VENDOR:-0}" == "1" \
      || ! -f "${linux_root}/vendor/MiniMax-H3/model_index.json" \
      || ! -f "${linux_root}/vendor/LightX2V/pyproject.toml" ]]; then
  rsync -a --delete-delay --exclude '/.git' \
    "${minimax_source}/" "${linux_root}/vendor/MiniMax-H3/"
  rsync -a --delete-delay --exclude '/.git' \
    "${lightx_source}/" "${linux_root}/vendor/LightX2V/"
  git -C "${minimax_source}" rev-parse HEAD \
    > "${linux_root}/vendor/MiniMax-H3/.source-revision"
  git -C "${lightx_source}" rev-parse HEAD \
    > "${linux_root}/vendor/LightX2V/.source-revision"
fi

for link_path in \
  "${linux_worktree}/models" \
  "${linux_worktree}/runtime/venv" \
  "${linux_worktree}/runtime/flashvsr-venv"; do
  if [[ -e "${link_path}" && ! -L "${link_path}" ]]; then
    echo "Refusing to replace a non-symlink runtime path: ${link_path}" >&2
    exit 2
  fi
done
mkdir -p "${linux_worktree}/runtime"
ln -sfn "$(readlink -f -- "${canonical_root}/models")" "${linux_worktree}/models"
ln -sfn "${python_env}" "${linux_worktree}/runtime/venv"
ln -sfn "${flashvsr_env}" "${linux_worktree}/runtime/flashvsr-venv"

printf '%s\n' \
  "Linux runtime mirror ready:" \
  "  source: ${canonical_root}" \
  "  mirror: ${linux_worktree}" \
  "  python: ${python_env}/bin/python" \
  "  FlashVSR python: ${flashvsr_env}/bin/python" \
  "  LightX2V source: ${lightx_source}" \
  "  models: $(readlink -f -- "${linux_worktree}/models")"
