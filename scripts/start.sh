#!/usr/bin/env bash
set -euo pipefail

# CUDA illegal-access faults cannot be recovered in-process. Disable giant
# WSL core dumps; every generation failure already has a per-job error log.
ulimit -c 0

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${release_root}/.env.local" ]]; then
  source "${release_root}/.env.local"
fi
mkdir -p "${release_root}/runtime/tmp"
export TMPDIR="${TMPDIR:-${release_root}/runtime/tmp}"
# Engine selection happens after Python starts. Fragmentation-safe expandable
# segments must therefore be selected before Torch imports, particularly for
# the 8-GiB W4A8 route. An explicit operator setting still takes precedence.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
source "${release_root}/scripts/_process.sh"
source "${release_root}/scripts/_runtime.sh"

serve_port="${H3_SERVE_PORT:-8090}"
pid_file="${H3_SERVE_PID_FILE:-${release_root}/runtime/h3serve-${serve_port}.pid}"
if [[ -s "${pid_file}" ]]; then
  existing_pid="$(<"${pid_file}")"
  if h3_is_release_server_pid "${existing_pid}"; then
    echo "H3 service is already running (PID ${existing_pid}, port ${serve_port})." >&2
    echo "Stop it with: ${release_root}/scripts/stop.sh" >&2
    exit 1
  fi
fi

# Recover a service whose PID file was removed by an older stop script or by a
# shell using different /mnt/c path casing.
mapfile -t existing_pids < <(h3_find_release_server_pids)
if ((${#existing_pids[@]})); then
  printf '%s\n' "${existing_pids[0]}" > "${pid_file}"
  echo "H3 service is already running (PID ${existing_pids[*]}, port ${serve_port})." >&2
  echo "Stop it with: ${release_root}/scripts/stop.sh" >&2
  exit 1
fi

rm -f -- "${pid_file}"
h3_configure_runtime
printf '%s\n' "$$" > "${pid_file}"
export H3_SERVE_PID_FILE="${pid_file}"

exec "${H3_SERVE_PYTHON}" "${release_root}/server.py" \
  --host "${H3_SERVE_HOST:-127.0.0.1}" \
  --port "${serve_port}" \
  --data-dir "${H3_SERVE_DATA_DIR:-${release_root}/data}" \
  --memory-profile "${H3_SERVE_MEMORY_PROFILE:-auto}" \
  --unified-console \
  "$@"
