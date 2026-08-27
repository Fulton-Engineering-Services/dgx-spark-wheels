#!/usr/bin/env bash
# Build uccl (EP extension) wheel for GB10 (sm_121).
#
# UCCL EP uses proxy-based RDMA (D2H queue -> CPU proxy thread does ibv_reg_mr),
# not hardware IBGDA, so it works on GB10 without nvidia_peermem. USE_DMABUF=1
# enables the DMA-BUF code path; the host-alloc fallback is automatic at runtime
# when GPU memory can't be registered for RDMA.
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
export USE_DMABUF=1
export USE_INTEL_RDMA_NIC=0

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
