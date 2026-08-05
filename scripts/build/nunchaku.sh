#!/usr/bin/env bash
# Build nunchaku wheel for GB10 (sm_121a).
# setup.py probes the live GPU for compute capability, so a live GPU is required.
# Requires git submodules (Block-Sparse-Attention etc.).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg nunchaku
setup_venv
clone_fork
cd "$(src_dir)"

# nunchaku depends on third-party submodules (Block-Sparse-Attention, etc.)
git submodule update --init --recursive

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export NUNCHAKU_BUILD_WHEELS=1 MAX_JOBS=8

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
