#!/usr/bin/env bash
# verify-flashinfer-python.sh — smoke-test with a real flashinfer CUDA kernel launch.
# First call JIT-compiles the cutlass-dsl kernel; allow generous time.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flashinfer-python
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL" "nvidia-cutlass-dsl[cu13]"

python3 - <<'PY'
import torch
from flashinfer import single_prefill_with_kv_cache

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

q = torch.randn(4, 8, 128, dtype=torch.float16, device="cuda")
k = torch.randn(4, 8, 128, dtype=torch.float16, device="cuda")
v = torch.randn(4, 8, 128, dtype=torch.float16, device="cuda")
out = single_prefill_with_kv_cache(q, k, v, kv_layout="NHD", pos_encoding_mode="NONE")
torch.cuda.synchronize()
assert out.shape == (4, 8, 128), out.shape
print("flashinfer prefill kernel launch OK:", tuple(out.shape))
PY