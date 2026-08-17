#!/usr/bin/env bash
# Build flashinfer-python wheel for GB10 (aarch64, CUDA 13.x).
# JIT package — kernels compile at runtime via cutlass-dsl + tvm-ffi.
# No nvcc build; do NOT build flashinfer-cubin (upstream artifactory has
# no sm_120/sm_121 cubins). BUILD_NVEP=0 skips the moe_ep/nvcc/meson/UCX
# backends (NIXL-EP + NCCL-EP).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flashinfer-python
setup_venv
clone_fork
cd "$(src_dir)"

# flashinfer vendors its JIT kernel headers (cccl, cutlass, spdlog) from git
# submodules into the wheel as flashinfer/data/*. The shallow clone does NOT
# fetch them; without this the wheel is missing the CCCL >= 3.x headers
# (cuda::fast_mod_div), and the runtime JIT compile falls back to the
# CTK-bundled CCCL and fails with "qualified name is not allowed"
# (upstream issue #3159). Same submodule requirement as torch/nunchaku/ort.
git submodule update --init --recursive

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# build-backend dep, needed in the venv because we build with --no-build-isolation
pip install "apache-tvm-ffi>=0.1.6,!=0.1.8,<0.2"

BUILD_NVEP=0 pip wheel --no-build-isolation --no-deps -w dist .

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"