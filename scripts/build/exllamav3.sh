#!/usr/bin/env bash
# Build exllamav3 wheel for GB10 (aarch64, CUDA 13.x, sm_121a).
#
# Compiles the exllamav3_ext CUDA/C++ extension via torch.utils.cpp_extension
# (setup.py walks exllamav3/exllamav3_ext/**.{c,cpp,cu} and precompiles when
# torch is importable in the build venv). Gencode comes from
# TORCH_CUDA_ARCH_LIST=12.1a. Pinned to upstream v0.0.43 (commit c5d9c65) --
# the exact commit shapleymcg's scripts/bootstrap_sm100_exl3.py requires for
# its SM100 build; this wheel is the GB10/sm_121a counterpart.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg exllamav3
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.1a"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
