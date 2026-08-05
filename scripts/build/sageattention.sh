#!/usr/bin/env bash
# Build sageattention (v2) wheel for GB10 (sm_121).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg sageattention
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=8

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
