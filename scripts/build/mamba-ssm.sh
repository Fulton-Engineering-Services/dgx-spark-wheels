#!/usr/bin/env bash
# Build mamba-ssm wheel for GB10 (aarch64, CUDA 13.x).
#
# mamba-ssm compiles one CUDA extension (selective_scan_cuda) via
# torch.utils.cpp_extension. setup.py emits gencode flags for sm_75..sm_121;
# the sm_121 gencode is added when the container nvcc is >= 13.0, so no arch
# patch is needed. MAMBA_FORCE_BUILD=TRUE skips setup.py's prebuilt-wheel
# download shortcut so we always compile from source.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg mamba-ssm
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
