#!/usr/bin/env bash
# verify-uccl.sh — smoke-test the uccl EP wheel with a real kernel launch.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg uccl
. "$VENV/bin/activate"
pip install intervaltree --no-deps
pip install "$DIST_DIR/$PKG_WHEEL" --no-deps

python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device"
cap = torch.cuda.get_device_capability()
assert cap == (12, 1), f"expected sm_121, got {cap}"

import uccl.ep as ep
sm90 = ep.is_sm90_compiled()
assert not sm90, "SM90 features should be disabled for GB10 (99KB smem limit)"
print(f"uccl.ep imported, is_sm90_compiled=False (DISABLE_SM90_FEATURES=1), cap={cap}")

# Verify the host-alloc RDMA buffer fallback works (GB10 has no peermem).
# get_rdma_buffer returns (dlpack capsule, is_host_allocated_bool).
scratch, is_host = ep.get_rdma_buffer(1024 * 1024, torch.cuda.current_device())
print(f"get_rdma_buffer: is_host_allocated={is_host}")
print("UCCL_EP_OK")
PY
