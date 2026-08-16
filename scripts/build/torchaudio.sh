#!/usr/bin/env bash
# Build torchaudio wheel for GB10 (aarch64, CUDA 13.3.1).
#
# Companion wheel for torch==2.13.0. Installs torch from the TORCH_WHEEL
# artifact (required — there is no published torch 2.13.0+cu133 wheel
# without this index).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg torchaudio
setup_venv
clone_fork
cd "$(src_dir)"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export BUILD_VERSION="$PKG_PUBLIC_VERSION"
export TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"