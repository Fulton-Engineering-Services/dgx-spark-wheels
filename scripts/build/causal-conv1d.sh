#!/usr/bin/env bash
# Build causal-conv1d wheel for GB10 (aarch64, CUDA 13.x).
#
# Compiles one CUDA extension (causal_conv1d_cuda) via torch.utils.cpp_extension.
# setup.py's gencode list includes sm_121 for nvcc >= 13.0, so no arch patch is
# needed. CAUSAL_CONV1D_FORCE_BUILD=TRUE skips setup.py's prebuilt-wheel
# download shortcut so we always compile from source (Dao-AILab publishes no
# aarch64 wheels at all -- source-only on PyPI).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg causal-conv1d
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
