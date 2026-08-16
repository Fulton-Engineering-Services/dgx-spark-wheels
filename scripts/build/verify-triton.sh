#!/usr/bin/env bash
# verify-triton.sh — smoke-test: import check always; CUDA kernel launch when a
# GPU torch is available. On the first build of a chain (triton builds before
# torch), the venv carries CPU-only PyPI torch (no aarch64 cu130 wheels exist),
# so the kernel test degrades gracefully.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg triton
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import triton
import triton.language as tl

try:
    import torch
    have_cuda = torch.cuda.is_available()
except Exception:
    have_cuda = False

print(f"triton import OK (triton {triton.__version__})")

if have_cuda:
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
else:
    print("triton verify: skipping kernel test (no CUDA device)")
PY