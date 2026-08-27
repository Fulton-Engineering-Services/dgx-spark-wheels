#!/usr/bin/env bash
# Build uccl (EP extension) wheel for GB10 (sm_121).
#
# UCCL EP uses proxy-based RDMA (CPU proxy thread drives ibv_reg_mr),
# not hardware IBGDA, so it works on GB10 without nvidia_peermem.
# Leave USE_DMABUF unset (0): the runtime probes GPU memory registration and
# automatically falls back to a pinned-host RDMA scratch buffer when GPU MR
# registration fails (no DMA-BUF / no peermem).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg uccl
setup_venv
clone_fork
cd "$(src_dir)"

# RDMA dev libs not in build-env; nanobind needed by ep/setup.py.
apt-get update -qq && apt-get install -y --no-install-recommends \
    libibverbs-dev libnl-3-dev libnl-route-3-dev libnuma-dev
rm -rf /var/lib/apt/lists/*
pip install nanobind

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1
export USE_DMABUF=0
export USE_INTEL_RDMA_NIC=0
# GB10 (sm_121) has only 99 KB optin shared memory (like Ada sm_89, not
# Hopper's 228 KB). The low-latency combine kernel's TMA buffer for 512
# experts exceeds this limit. DISABLE_SM90_FEATURES switches to the non-TMA
# code path (simple warp copy) which fits within the default shared memory.
export DISABLE_SM90_FEATURES=1

# 1. Build EP extension (produces ep/build/lib.*/ep.abi3.so).
cd ep
python3 setup.py build
cd ..

# 2. Copy EP .so into uccl/ for the top-level wheel's package_data.
cp ep/build/lib.*/ep.abi3.so uccl/ep.abi3.so
mkdir -p uccl/lib

# 3. Build the uccl wheel (top-level setup.py: find_packages + package_data
#    includes ep*.so; _platform_tag_stub.c forces cp312-abi3 tag).
python3 setup.py bdist_wheel

# 4. Stamp the variant-aware local version segment.
built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
