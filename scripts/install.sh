#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${H3_SERVE_RUNTIME_DIR:-${release_root}/runtime}"
python_bin="${PYTHON_BIN:-python3.10}"
venv="${runtime_root}/venv"
vendor_root="${runtime_root}/vendor"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.3}"
flashvsr_python_bin="${FLASHVSR_PYTHON_BIN:-python3.11}"
flashvsr_venv="${runtime_root}/flashvsr-venv"

minimax_commit="8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea"
lightx_commit="231e2307f15b9eb60fe3f877f7eed945c8d8d717"
sparge_commit="ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a"
sage_commit="eb615cf6cf4d221338033340ee2de1c37fbdba4a"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v "${python_bin}" >/dev/null || {
  echo "Python 3.10 is required (override with PYTHON_BIN)" >&2
  exit 1
}
command -v "${flashvsr_python_bin}" >/dev/null || {
  echo "Python 3.11 is required for temporal second sampling (override with FLASHVSR_PYTHON_BIN)" >&2
  exit 1
}
[[ -x "${cuda_home}/bin/nvcc" ]] || {
  echo "CUDA 13.3 compiler is required at ${cuda_home} (override with CUDA_HOME)" >&2
  exit 1
}
for required_header in cublas_v2.h cusolverDn.h cusparse.h; do
  [[ -f "${cuda_home}/include/${required_header}" ]] || {
    echo "CUDA 13.3 development header is missing: ${cuda_home}/include/${required_header}" >&2
    exit 1
  }
done
cuda_stub_dir="${cuda_home}/targets/x86_64-linux/lib/stubs"

mkdir -p "${runtime_root}" "${vendor_root}" "${release_root}/models" "${release_root}/output"
"${python_bin}" -m venv "${venv}"
python="${venv}/bin/python"
"${python}" -m pip install --upgrade pip setuptools wheel
"${python}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0+cu130 torchvision==0.28.0+cu130

checkout() {
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

checkout https://github.com/MiniMax-AI/MiniMax-H3.git \
  "${vendor_root}/MiniMax-H3" "${minimax_commit}"
checkout https://github.com/ModelTC/LightX2V.git \
  "${vendor_root}/LightX2V" "${lightx_commit}"
checkout https://github.com/thu-ml/SageAttention.git \
  "${vendor_root}/SageAttention" "${sage_commit}"
checkout https://github.com/thu-ml/SpargeAttn.git \
  "${vendor_root}/SpargeAttn" "${sparge_commit}"
apply_once() {
  local checkout_root="$1" patch_path="$2"
  if git -C "${checkout_root}" apply --check "${patch_path}"; then
    git -C "${checkout_root}" apply "${patch_path}"
  elif ! git -C "${checkout_root}" apply --reverse --check "${patch_path}"; then
    echo "cannot apply or verify build patch: ${patch_path}" >&2
    exit 1
  fi
}
apply_once "${vendor_root}/MiniMax-H3" \
  "${release_root}/patches/minimax-h3-vae-temporal-host-sink.patch"
apply_once "${vendor_root}/SageAttention" \
  "${release_root}/patches/sageattention-pytorch213-cxx20.patch"
apply_once "${vendor_root}/SpargeAttn" \
  "${release_root}/patches/sparge-pytorch213-cxx20.patch"

"${python}" -m pip install -r "${release_root}/requirements.lock"
CUDA_HOME="${cuda_home}" PATH="${cuda_home}/bin:${PATH}" \
  LIBRARY_PATH="${cuda_stub_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
  TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS="${MAX_JOBS:-4}" \
  "${python}" -m pip install --no-build-isolation --no-deps \
    "${vendor_root}/SageAttention"
CUDA_HOME="${cuda_home}" PATH="${cuda_home}/bin:${PATH}" \
  LIBRARY_PATH="${cuda_stub_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
  TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS="${MAX_JOBS:-4}" \
  "${python}" -m pip install --no-build-isolation --no-deps \
    "${vendor_root}/SpargeAttn"
"${python}" -m pip freeze > "${runtime_root}/installed-packages.txt"

# Temporal second sampling runs in an isolated process because FlashVSR's
# pinned Torch 2.6/CUDA 12.4 extension is not ABI-compatible with H3's
# Torch 2.13/CUDA 13.0 runtime. The model stays hot in CPU RAM and only owns
# the GPU while a temporal second-sampling job is running.
"${flashvsr_python_bin}" -m venv "${flashvsr_venv}"
flashvsr_python="${flashvsr_venv}/bin/python"
"${flashvsr_python}" -m pip install --upgrade pip setuptools wheel packaging ninja
"${flashvsr_python}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
"${flashvsr_python}" -m pip install -r "${release_root}/requirements-flashvsr.lock"
block_sparse_wheel="${release_root}/wheels/block_sparse_attn-0.0.2-cp311-cp311-linux_x86_64.whl"
[[ -f "${block_sparse_wheel}" ]] || {
  echo "Missing pinned FlashVSR CUDA wheel: ${block_sparse_wheel}" >&2
  exit 1
}
"${flashvsr_python}" -m pip install --no-deps "${block_sparse_wheel}"
"${flashvsr_python}" -m pip freeze > "${runtime_root}/flashvsr-installed-packages.txt"

echo "Installation complete. Run:"
echo "  ${python} ${release_root}/scripts/preflight.py"
echo "  ${python} ${release_root}/server.py"
