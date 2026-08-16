#!/usr/bin/env bash
# Build flash-attn wheel for GB10 (sm_120, PTX forward-compat on sm_121).
# Runs inside the build-env:24.04 container on a self-hosted GB10 runner.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flash-attn
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS="120;121"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4   # NOT nproc -- flash_bwd_hdim256_*_sm80 backward kernels OOM above ~4-8

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
