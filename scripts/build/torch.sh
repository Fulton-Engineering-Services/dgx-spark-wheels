#!/usr/bin/env bash
# Build torch wheel for GB10 from source (aarch64, CUDA 13.3.1).
#
# Optimized for Blackwell / sm_120+sm_121 with cuDNN, cuDSS (sparse direct
# solver), cuSPARSELt (structured sparse GEMM), cuFile (GPUDirect Storage),
# and system NCCL. Triton is a runtime pip dependency in torch 2.13.0 (not a
# build-time cmake flag — USE_SYSTEM_TRITON was removed upstream), so the
# triton wheel from the orchestrated build is installed into the venv by
# common.sh and torch imports it at runtime.
#
# This is the long pole of the whole index: ~40 min on the GB10 node at
# MAX_JOBS=$(nproc). Watch free -h / thermals on the first run.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg torch
SKIP_TORCH_INSTALL=1 setup_venv
pip install numpy ninja packaging wheel setuptools psutil pyyaml cmake typing_extensions

src="$(clone_fork)"
cd "$src"
echo "==> initializing submodules (large: this fetches third_party/* ...)"
git submodule update --init --recursive

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

export PYTORCH_BUILD_VERSION="$PKG_PUBLIC_VERSION"
export PYTORCH_BUILD_NUMBER=0

export USE_CUDA=1
export USE_CUDNN=1
export USE_CUDSS=1
export USE_CUSPARSELT=1
export USE_CUFILE=1
export USE_SYSTEM_NCCL=1
export BLAS=OpenBLAS
export USE_FBGEMM=0
export USE_NNPACK=1
export USE_MKLDNN=0
export BUILD_TEST=0
export USE_KINETO=0
export USE_ITT=0

export CMAKE_GENERATOR=Ninja
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# cuSPARSELt/cuFile dev files live under CUDA_HOME (see build-env Dockerfile).
export CUSPARSELT_ROOT_DIR="$CUDA_HOME"
export CUFILE_ROOT_DIR="$CUDA_HOME"

python setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
