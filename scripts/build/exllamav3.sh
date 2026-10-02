#!/usr/bin/env bash
# Build exllamav3 wheel for GB10 (aarch64, CUDA 13.x, sm_121a).
#
# Compiles the exllamav3_ext CUDA/C++ extension via torch.utils.cpp_extension
# (setup.py walks exllamav3/exllamav3_ext/**.{c,cpp,cu} and precompiles when
# torch is importable in the build venv). Gencode comes from
# TORCH_CUDA_ARCH_LIST=12.1a. Tracks benthecarman/exllamav3 branch
# mimo-v2.6-flash @ 278a60ed (v1.5.1): MiMo-V2.6 architecture + aarch64/GB10
# guards (x86 AVX TUs gated in-file, CPU MoE stub, native TP CPU reduce).
# Fork carries no patches on this base; turboderp-org merge base is 6b84a21b.
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
