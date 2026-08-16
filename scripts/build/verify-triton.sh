#!/usr/bin/env bash
# verify-triton.sh — smoke-test with a real Triton kernel launch on CUDA.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg triton
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
import triton
import triton.language as tl

assert torch.cuda.is_available(), "no CUDA device"

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * 16
    offsets = block_start + tl.arange(0, 16)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

n = 1024
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty_like(x)
grid = (triton.cdiv(n, 16),)
add_kernel[grid](x, y, out, n_elements=n)
torch.cuda.synchronize()
assert torch.allclose(out, x + y, atol=1e-6), "kernel output mismatch"
print("triton kernel launch OK")
PY