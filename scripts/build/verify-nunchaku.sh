#!/usr/bin/env bash
# verify-nunchaku.sh — smoke-test that the nunchaku CUDA extension loads and a
# quantized-linear kernel executes on GB10. nunchaku's C++/CUDA backend is the
# part that fails on an ABI mismatch, so exercising it (not just importing) is
# the real test.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg nunchaku
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
import nunchaku
from nunchaku.utils.nunchaku_utils import get_nunchaku_version_info

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

info = get_nunchaku_version_info()
print("nunchaku version info:", info)

# Exercise the quantized-linear kernel path: build a tiny Nunchaku linear
# layer and run a forward pass so the INT4 dequant + matmul kernel launches.
from nunchaku.linear.nunchaku_linear import NunchakuLinear
import torch.nn as nn

linear = nn.Linear(64, 64, bias=False).cuda().half()
nlinear = NunchakuLinear.from_float(linear)
x = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
out = nlinear(x)
assert out.shape == (2, 8, 64), out.shape
torch.cuda.synchronize()
print("nunchaku kernel launch OK:", tuple(out.shape))
PY
