#!/usr/bin/env bash
# verify-sageattn3.sh — smoke-test with a real SageAttention v3 (Blackwell) kernel.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg sageattn3
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
# sageattn3 (sageattention3_blackwell) exposes sageattn3(); confirm the exact
# signature against the pinned source if this fails — the v3 API may take
# additional optional kwargs but (q, k, v) is the documented core path.
from sageattn3 import sageattn3

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

q = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
out = sageattn3(q, k, v)
assert out.shape == (1, 128, 1, 64), out.shape
torch.cuda.synchronize()
print("sageattn3 kernel launch OK:", tuple(out.shape))
PY
