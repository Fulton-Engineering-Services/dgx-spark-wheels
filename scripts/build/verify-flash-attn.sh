#!/usr/bin/env bash
# verify-flash-attn.sh — smoke-test with a real flash-attn CUDA kernel launch.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flash-attn
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
from flash_attn import flash_attn_func

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

q = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
out = flash_attn_func(q, k, v)
assert out.shape == (1, 8, 16, 64), out.shape
torch.cuda.synchronize()
print("flash-attn kernel launch OK:", tuple(out.shape))
PY
