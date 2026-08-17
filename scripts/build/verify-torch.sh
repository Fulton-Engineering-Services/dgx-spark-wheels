#!/usr/bin/env bash
# verify-torch.sh — smoke-test the from-source torch wheel: CUDA availability,
# compute capability, CUDA version, cuSPARSELt presence, and a real matmul.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg torch
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

# Variant-aware CUDA assertion: cu13.3 -> 13.3, cu13.0 -> 13.0.
EXPECTED_CUDA="${CUDA_VARIANT#cu}" python3 - <<'PY'
import os
import torch

assert torch.cuda.is_available(), "no CUDA device"
cc = torch.cuda.get_device_capability()
assert cc == (12, 1), f"expected sm_121, got {cc}"

expected_cuda = os.environ["EXPECTED_CUDA"]
cuda = torch.version.cuda
assert cuda is not None and cuda.startswith(expected_cuda), f"expected CUDA {expected_cuda}, got {cuda}"

# cuSPARSELt backend present when built with USE_CUSPARSELT=1.
assert hasattr(torch.backends, "cusparselt"), "cuSPARSELt backend missing"

# Real Tensor Core GEMM.
a = torch.randn(256, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 256, device="cuda", dtype=torch.float16)
c = torch.mm(a, b)
torch.cuda.synchronize()
assert c.shape == (256, 256), c.shape
print("torch CUDA matmul OK:", tuple(c.shape), "cuda", cuda, "cc", cc)
PY