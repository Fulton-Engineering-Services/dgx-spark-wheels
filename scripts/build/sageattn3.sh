#!/usr/bin/env bash
# Build sageattn3 (SageAttention v3, Blackwell) wheel for GB10 (sm_121a).
# setup.py probes the live GPU for compute capability, so a live GPU is required.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg sageattn3
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
