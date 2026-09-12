#!/usr/bin/env bash

# Resolve either a self-contained release runtime or this repository's already
# validated development runtime.  Callers must define release_root first.

h3_sparse_importable() {
  local python_bin="$1"
  local search_root="${2:-}"
  PYTHONPATH="${search_root}${search_root:+:}${PYTHONPATH:-}" \
    "${python_bin}" -c 'import torch; import spas_sage_attn' >/dev/null 2>&1
}

h3_configure_sparse_runtime() {
  local python_bin="$1"
  local sparse_commit="ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a"
  local cache_root="${H3_SPARSE_CACHE_DIR:-${release_root}/runtime/extensions/sparge-sm89-py310-torch213-cu133}"
  local source_root="${release_root}/runtime/vendor/SpargeAttn"
  local legacy_root="/tmp/h3_sparge_sm89"

  # An explicitly disabled optional backend always wins. This is useful for
  # diagnosis and guarantees a dense-only service without deleting a cache.
  if [[ "${H3_NATIVE_ENABLE_SPARSE:-auto}" == "0" ]]; then
    return 0
  fi

  if [[ -n "${H3_NATIVE_SPARGE_BUILD_DIR:-}" ]]; then
    if h3_sparse_importable "${python_bin}" "${H3_NATIVE_SPARGE_BUILD_DIR}"; then
      export H3_NATIVE_ENABLE_SPARSE=1
      return 0
    fi
    echo "H3_NATIVE_SPARGE_BUILD_DIR is not compatible with the selected runtime: ${H3_NATIVE_SPARGE_BUILD_DIR}" >&2
    return 1
  fi

  if h3_sparse_importable "${python_bin}"; then
    export H3_NATIVE_ENABLE_SPARSE=1
    unset H3_NATIVE_SPARGE_BUILD_DIR
    return 0
  fi

  if h3_sparse_importable "${python_bin}" "${cache_root}"; then
    export H3_NATIVE_ENABLE_SPARSE=1
    export H3_NATIVE_SPARGE_BUILD_DIR="${cache_root}"
    return 0
  fi

  # Preserve the already audited development build before /tmp is cleaned or
  # WSL restarts. The cached directory is release-local and intentionally not
  # committed because this extension is tied to the locked binary runtime.
  if h3_sparse_importable "${python_bin}" "${legacy_root}"; then
    echo "Caching the validated RTX 4090 sparse-attention extension..." >&2
    mkdir -p "${cache_root}"
    cp -a "${legacy_root}/spas_sage_attn" "${cache_root}/"
    if h3_sparse_importable "${python_bin}" "${cache_root}"; then
      export H3_NATIVE_ENABLE_SPARSE=1
      export H3_NATIVE_SPARGE_BUILD_DIR="${cache_root}"
      return 0
    fi
  fi

  if [[ "${H3_AUTO_BUILD_SPARSE:-1}" != "1" ]]; then
    echo "Sparse attention is unavailable and automatic compilation is disabled." >&2
    return 1
  fi
  if ! command -v git >/dev/null || ! command -v nvcc >/dev/null; then
    echo "Sparse attention needs git and the CUDA nvcc compiler for its one-time build." >&2
    return 1
  fi

  echo "No compatible sparse-attention cache was found; compiling once for RTX 4090 (SM89)..." >&2
  mkdir -p "$(dirname "${source_root}")" "${cache_root}"
  if [[ ! -d "${source_root}/.git" ]]; then
    if ! git clone --filter=blob:none https://github.com/thu-ml/SpargeAttn.git "${source_root}"; then
      echo "Could not download the pinned SpargeAttention source." >&2
      return 1
    fi
  fi
  if ! git -C "${source_root}" fetch --depth 1 origin "${sparse_commit}" \
    || ! git -C "${source_root}" checkout --detach "${sparse_commit}"; then
    echo "Could not select the audited SpargeAttention commit ${sparse_commit}." >&2
    return 1
  fi
  if ! TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS="${H3_SPARSE_BUILD_JOBS:-4}" \
    "${python_bin}" -m pip install --upgrade --no-build-isolation --no-deps \
      --target "${cache_root}" "${source_root}"; then
    echo "The optional SM89 sparse-attention build failed." >&2
    return 1
  fi
  if ! h3_sparse_importable "${python_bin}" "${cache_root}"; then
    echo "The compiled sparse-attention extension is incompatible with this runtime." >&2
    return 1
  fi

  export H3_NATIVE_ENABLE_SPARSE=1
  export H3_NATIVE_SPARGE_BUILD_DIR="${cache_root}"
  echo "Sparse attention is ready and will be reused from ${cache_root}." >&2
}

h3_configure_runtime() {
  local packaged_python="${release_root}/runtime/venv/bin/python"
  local vendor_path="${release_root}/backends/turbo/vendor"
  local python_bin=""
  local candidate
  local -a candidates=()

  if [[ -n "${H3_SERVE_PYTHON:-}" ]]; then
    candidates+=("${H3_SERVE_PYTHON}")
  elif [[ -x "${packaged_python}" ]]; then
    candidates+=("${packaged_python}")
  else
    # Development checkout: prefer the environment used for the measured
    # Native H3 runs, then consider the caller's active Python installations.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      candidates+=("${CONDA_PREFIX}/envs/voxcpm/bin/python")
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
      candidates+=("$(dirname "$(dirname "${CONDA_EXE}")")/envs/voxcpm/bin/python")
    fi
    candidates+=(
      "/root/miniconda3/envs/voxcpm/bin/python"
      "$(command -v python3.10 2>/dev/null || true)"
      "$(command -v python3 2>/dev/null || true)"
      "$(command -v python 2>/dev/null || true)"
    )
  fi

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    # runtime/venv may itself be reached through a /mnt/c checkout symlink.
    # Resolve the interpreter before importing Torch so Python discovers the
    # Linux-native environment prefix instead of walking thousands of DrvFS
    # paths during import-time source inspection.
    candidate="$(readlink -f -- "${candidate}")"
    if PYTHONPATH="${vendor_path}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${candidate}" -c '
import sys
assert sys.version_info[:2] == (3, 10)
import aiohttp, av, diffusers, einops, numpy, safetensors, torch, transformers
import PIL, comfy_kitchen, sageattention
' >/dev/null 2>&1; then
      python_bin="${candidate}"
      break
    fi
  done

  if [[ -z "${python_bin}" ]]; then
    echo "No compatible H3 Python runtime was found." >&2
    if [[ -n "${H3_SERVE_PYTHON:-}" ]]; then
      echo "H3_SERVE_PYTHON is incomplete: ${H3_SERVE_PYTHON}" >&2
    fi
    echo "Run ${release_root}/scripts/install.sh, or set H3_SERVE_PYTHON to the validated Python 3.10 environment." >&2
    return 1
  fi

  export H3_SERVE_PYTHON="${python_bin}"
  export PYTHONPATH="${vendor_path}${PYTHONPATH:+:${PYTHONPATH}}"

  # This is optional: reuse or compile the locked SM89 extension once. A build
  # failure keeps the service available with exact 100% dense attention.
  if ! h3_configure_sparse_runtime "${python_bin}"; then
    export H3_NATIVE_ENABLE_SPARSE=0
    unset H3_NATIVE_SPARGE_BUILD_DIR
    echo "Continuing with complete dense attention; sparse controls stay locked." >&2
  fi

  local development_main="${release_root}/../.."
  local development_minimax="${development_main}/MiniMax-H3"
  local development_lightx="${development_main}/knowledge/projects/inference_frameworks/lightx2v"
  if [[ ! -f "${development_lightx}/pyproject.toml" ]]; then
    development_lightx="${release_root}/../../../backend-compare/sources/LightX2V"
  fi
  if [[ ! -d "${release_root}/runtime/vendor/MiniMax-H3" && -d "${development_minimax}" ]]; then
    export H3_SERVE_MINIMAX_SOURCE="${H3_SERVE_MINIMAX_SOURCE:-${development_minimax}}"
  fi
  if [[ ! -d "${release_root}/runtime/vendor/LightX2V" && -d "${development_lightx}" ]]; then
    export H3_SERVE_LIGHTX_SOURCE="${H3_SERVE_LIGHTX_SOURCE:-${development_lightx}}"
  fi
}
