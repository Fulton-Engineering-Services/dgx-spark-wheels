#!/usr/bin/env bash
# verify-mamba-ssm.sh — smoke-test with a real Mamba selective-scan CUDA kernel
# launch. `import mamba_ssm` loads the compiled selective_scan_cuda extension
# (it is imported unconditionally in ops/selective_scan_interface.py), so a
# successful import plus a forward pass exercises the ABI-sensitive .so, not
# just the Python layer.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg mamba-ssm
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
from mamba_ssm import Mamba

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

batch, seqlen, d_model = 2, 64, 64
model = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).cuda().eval()
x = torch.randn(batch, seqlen, d_model, device="cuda")
with torch.no_grad():
    out = model(x)
torch.cuda.synchronize()
assert out.shape == (batch, seqlen, d_model), out.shape
print("mamba-ssm selective_scan_cuda kernel launch OK:", tuple(out.shape))
PY
