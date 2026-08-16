#!/usr/bin/env bash
# Build triton wheel for GB10 (aarch64, CUDA 13.3.1).
#
# Triton downloads its own pinned prebuilt LLVM during build (the LLVM commit
# lives in cmake/llvm-hash.txt in the fork); no system LLVM package is needed.
# Built against torch from the TORCH_WHEEL artifact if provided, else PyPI
# cu130 (triton's Python package imports torch at runtime but does not link
# torch's C++ ABI, so the fallback is safe for a first-pass build).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg triton
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

pip install pybind11
python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
