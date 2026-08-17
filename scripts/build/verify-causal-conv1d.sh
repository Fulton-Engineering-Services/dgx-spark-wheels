#!/usr/bin/env bash
# verify-causal-conv1d.sh — smoke-test with a real causal-conv1d CUDA kernel
# launch. `import causal_conv1d` loads the compiled causal_conv1d_cuda
# extension, and causal_conv1d_fn() dispatches to it directly (no torch
# fallback), so a successful call exercises the ABI-sensitive .so.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg causal-conv1d
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

batch, dim, seqlen, width = 2, 64, 64, 4
x = torch.randn(batch, dim, seqlen, device="cuda", dtype=torch.float16)
weight = torch.randn(dim, width, device="cuda", dtype=torch.float16)
bias = torch.randn(dim, device="cuda", dtype=torch.float16)
out = causal_conv1d_fn(x, weight, bias, activation="silu")
torch.cuda.synchronize()
assert out.shape == (batch, dim, seqlen), out.shape
print("causal-conv1d kernel launch OK:", tuple(out.shape))
PY
