# Block Sparse Attention provenance

- Upstream: `https://github.com/mit-han-lab/Block-Sparse-Attention`
- Revision: `49d6c39e4dc0303442cda3bb758b3925d4399c49`; this includes the
  upstream FlashVSR import fix after the broken `v0.0.2` package entry point.
- CUTLASS submodule: pinned by the upstream revision and vendored recursively.
- License: BSD-3-Clause, preserved in `LICENSE`
- Build: `wheels/block_sparse_attn-0.0.2-cp311-cp311-linux_x86_64.whl` is built
  from this tree for the isolated Python 3.11 / Torch 2.6 / CUDA 12.4 runtime.
  SHA-256: `322c1bfbd09b40a23146dacdd452debd5cb4a435741bcff83537228c494e9bf1`.
  `scripts/install.sh` uses that wheel and falls back to compiling this fixed
  source only when the wheel is absent. No external installed copy is used.
