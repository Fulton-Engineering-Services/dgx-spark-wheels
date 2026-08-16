#!/usr/bin/env bash
# verify-torchaudio.sh — smoke-test torchaudio: import, run a Spectrogram
# transform on a generated waveform.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg torchaudio
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL"

python3 - <<'PY'
import torch
import torchaudio

assert torch.cuda.is_available(), "no CUDA device"

waveform = torch.randn(2, 16000)
spec = torchaudio.transforms.Spectrogram()(waveform)
c, t, f = spec.shape
assert t == 79, (t, f)

# Move to GPU
spec_gpu = torchaudio.transforms.Spectrogram().cuda()(waveform.cuda())
torch.cuda.synchronize()
print("torchaudio Spectrogram OK: gpu shape", tuple(spec_gpu.shape))
PY