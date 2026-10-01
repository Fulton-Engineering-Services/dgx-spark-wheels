#!/usr/bin/env bash
# verify-exllamav3.sh — smoke-test the compiled exllamav3_ext extension on the
# live GB10 GPU. Loading the .so resolves all libtorch/CUDA symbols, so a
# successful module load plus one on-device kernel call exercises the
# ABI-sensitive artifact. Mirrors the load pattern of shapleymcg's
# scripts/smoke_sm100_stack.py (importlib load of the built extension).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg exllamav3
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import importlib
import torch

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), f"expected sm_121, got {torch.cuda.get_device_capability()}"

ext = importlib.import_module("exllamav3_ext")

a = torch.randn(4, 64, device="cuda", dtype=torch.float16)
b = torch.randn(64, 32, device="cuda", dtype=torch.float16)
c = torch.zeros(4, 32, device="cuda", dtype=torch.float16)
ext.hgemm(a, b, c)
torch.cuda.synchronize()
assert c.shape == (4, 32), c.shape
expected = a.float() @ b.float()
assert torch.allclose(c.float(), expected, atol=2e-2, rtol=2e-2), (c - expected.half()).abs().max()
print("exllamav3_ext kernel launch OK:", tuple(c.shape))
PY
