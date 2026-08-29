#!/usr/bin/env bash
# Build lmcache wheel for GB10 (sm_121).
#
# LMCache is a KV cache management engine for LLM serving (vLLM, SGLang,
# TRT-LLM). It provides tiered KV cache storage (GPU/CPU/disk/remote) and
# an out-of-process multiprocess (MP) server mode for Dynamo vLLM
# integration.
#
# Build system: setup.py with setup_extensions BuildPolicy (strategy pattern).
# The CUDA profile auto-detects nvcc in PATH and builds:
#   - lmcache.cuda_ops: CUDA kernels (mem, blend, AC encode/decode, pos)
#   - lmcache.lmcache_native: common C++ (bitmap, fold, ttl_lock, utils)
#   - lmcache.lmcache_redis: Redis connector (raw sockets, no hiredis dep)
#   - lmcache.lmcache_fs: filesystem connector (standard C headers only)
#
# No source patches needed -- the fork is a pin only. setuptools_scm derives
# the version from git tags; SETUPTOOLS_SCM_PRETEND_VERSION pins it.
#
# The Rust raw_block component (rust/raw_block/) is NOT built -- it is an
# optional L2 storage backend, lazily imported and not part of setup.py.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg lmcache
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1
export LMCACHE_CUDA_MAJOR=13
export ENABLE_CXX11_ABI=1
export MAX_JOBS="${MAX_JOBS:-8}"

# setuptools_scm build dep -- not installed by setup_venv.
pip install setuptools-scm

SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_PUBLIC_VERSION" \
  python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
