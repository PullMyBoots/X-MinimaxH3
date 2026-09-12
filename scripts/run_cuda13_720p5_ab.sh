#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${release_root}/../.." && pwd)"
python="${H3_SERVE_PYTHON:-${release_root}/runtime/venv/bin/python}"
python="$(readlink -f -- "${python}")"
output_root="${1:-${H3_BENCHMARK_OUTPUT_ROOT:-${release_root}/runtime/benchmarks/cu133_torch213_720p5}}"
sparge_root="${release_root}/runtime/extensions/sparge-sm89-py310-torch213-cu133"
minimax_source="${H3_SERVE_MINIMAX_SOURCE:-${workspace_root}/subprojects-main/main/MiniMax-H3}"
lightx_source="${H3_SERVE_LIGHTX_SOURCE:-${workspace_root}/subprojects-main/backend-compare/sources/LightX2V}"

[[ -x "${python}" ]] || { echo "missing runtime: ${python}" >&2; exit 1; }
[[ -d "${sparge_root}" ]] || { echo "missing SpargeAttention build: ${sparge_root}" >&2; exit 1; }
mkdir -p "${output_root}"

run_one() {
  local acceleration="$1"
  CUDA_HOME=/usr/local/cuda-13.3 \
    PYTHONPATH="${release_root}" \
    PYTHONUNBUFFERED=1 \
    "${python}" -u "${release_root}/scripts/benchmark_native_hot_session.py" \
      --engine original \
      --attention-backend joint-scheduled \
      --quant-backend cuda \
      --scenario-manifest "${release_root}/benchmarks/720p5_seed82303.json" \
      --label-prefix "cu133_torch213_accel${acceleration}" \
      --output-root "${output_root}" \
      --memory-profile fullspeed \
      --steps 20 \
      --repeat 1 \
      --sparge-build-dir "${sparge_root}" \
      --experimental-minimum-sparse-topk 0.0625 \
      --joint-policy h3_joint_v18_forecast_aware_frontier_global_dp \
      --joint-acceleration "${acceleration}" \
      --forecast-controller directional \
      --fused-rms-adaln \
      --vae-compile-transformer-block \
      --model-root "${release_root}/models" \
      --minimax-source "${minimax_source}" \
      --lightx-source "${lightx_source}"
}

# Run from WSL ext4 to avoid adding the Windows mount working directory to the
# import search path. Model I/O remains identical because all paths are frozen.
cd /root
run_one 0
run_one 75
