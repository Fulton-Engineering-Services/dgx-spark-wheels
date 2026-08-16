#!/usr/bin/env bash
# verify-torchvision.sh — smoke-test torchvision: load a model and run a real
# CUDA forward pass.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg torchvision
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
import torchvision

assert torch.cuda.is_available(), "no CUDA device"

model = torchvision.models.resnet18(weights=None).cuda().eval()
x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.no_grad():
    out = model(x)
torch.cuda.synchronize()
assert out.shape == (1, 1000), out.shape
print("torchvision ResNet18 CUDA forward OK:", tuple(out.shape))
PY