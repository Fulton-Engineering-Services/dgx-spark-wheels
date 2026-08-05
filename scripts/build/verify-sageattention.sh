#!/usr/bin/env bash
# verify-sageattention.sh — smoke-test with a real SageAttention v2 CUDA kernel.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg sageattention
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
from sageattention import sageattn

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

q = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
out = sageattn(q, k, v)
assert out.shape == (1, 128, 1, 64), out.shape
torch.cuda.synchronize()
print("sageattention kernel launch OK:", tuple(out.shape))
PY
