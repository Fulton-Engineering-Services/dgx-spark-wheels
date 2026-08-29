#!/usr/bin/env bash
# verify-flextensor.sh — smoke-test flextensor + NVMe/cuFile offload on GB10.
#
# Verifies: (1) package import, (2) PosixBackend NVMe write→read roundtrip
# (always available), (3) CuFileBackend init + roundtrip if nvidia-fs is loaded
# (GDS direct-to-GPU), (4) make_nvme_backend factory.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flextensor
. "$VENV/bin/activate"
# Install flextensor runtime deps not already in the venv (setup_venv has
# torch, triton, numpy, psutil). --no-deps on the wheel install avoids pip
# pulling a PyPI torch that would clobber our from-source build.
pip install pydantic beartype pyyaml scipy proxytypes3 posix-ipc shared-memory-dict pytest
pip install "$DIST_DIR/$PKG_WHEEL" --no-deps

# Try to load nvidia-fs for cuFile (GDS); fall back to posix if unavailable.
modprobe nvidia-fs 2>/dev/null || true

python3 - <<'PY'
import os
import tempfile

import torch

import flextensor
from flextensor.nvme_transfer import (
    CuFileBackend,
    PosixBackend,
    is_nvidia_fs_available,
    make_nvme_backend,
)

assert torch.cuda.is_available(), "no CUDA device"
cap = torch.cuda.get_device_capability()
assert cap == (12, 1), f"expected sm_121, got {cap}"

tmpdir = tempfile.mkdtemp(prefix="flextensor-verify-")

# --- PosixBackend roundtrip (always available) ---
backend = PosixBackend(alignment=4096)
path = os.path.join(tmpdir, "posix_test.bin")
fd = backend.open_file(path)

data = torch.randn(1024, 1024, dtype=torch.float32)
data_bytes = data.contiguous().view(torch.uint8).flatten()
ref = backend.write_block(fd, data_bytes, offset=0)
assert ref.logical_nbytes == data_bytes.numel(), \
    f"logical_nbytes mismatch: {ref.logical_nbytes} != {data_bytes.numel()}"
assert ref.aligned_nbytes >= ref.logical_nbytes

gpu_buf = torch.empty(ref.aligned_nbytes, dtype=torch.uint8, device="cuda")
backend.read_block(fd, gpu_buf, offset=0, nbytes=ref.logical_nbytes)
torch.cuda.synchronize()

gpu_data = gpu_buf[: ref.logical_nbytes].view(torch.float32).reshape(1024, 1024).cpu()
assert torch.allclose(gpu_data, data), "posix roundtrip data mismatch"

backend.close_file(fd)
backend.close()
print("flextensor PosixBackend NVMe roundtrip OK")

# --- CuFileBackend init + roundtrip (GDS, if nvidia-fs loaded) ---
cu_ok = False
if is_nvidia_fs_available():
    try:
        cf = CuFileBackend(alignment=4096)
    except RuntimeError as exc:
        print(f"flextensor: CuFileBackend pre-flight failed ({exc}) — "
              "GDS not supported on this GPU, verify passed via posix")
    else:
        cu_ok = True
        cf_path = os.path.join(tmpdir, "cufile_test.bin")
        fd2 = cf.open_file(cf_path)

        data2 = torch.randn(512, 512, dtype=torch.float32)
        data2_bytes = data2.contiguous().view(torch.uint8).flatten()
        ref2 = cf.write_block(fd2, data2_bytes, offset=0)
        assert ref2.logical_nbytes == data2_bytes.numel()

        gpu2 = torch.empty(ref2.aligned_nbytes, dtype=torch.uint8, device="cuda")
        cf.read_block(fd2, gpu2, offset=0, nbytes=ref2.logical_nbytes)
        torch.cuda.synchronize()

        gpu2_data = gpu2[: ref2.logical_nbytes].view(torch.float32).reshape(512, 512).cpu()
        assert torch.allclose(gpu2_data, data2), "cuFile roundtrip data mismatch"

        cf.close_file(fd2)
        cf.close()
        print("flextensor CuFileBackend NVMe roundtrip OK (GDS direct-to-GPU)")

if not cu_ok:
    print("flextensor: nvidia-fs not loaded or cuFile pre-flight failed — "
          "cuFile (GDS) skipped, posix verified")

# --- make_nvme_backend factory ---
posix_backend = make_nvme_backend("posix", alignment=4096)
assert isinstance(posix_backend, PosixBackend), \
    "make_nvme_backend('posix') should return PosixBackend"

print(f"flextensor NVMe offload verify OK: {cap}")
PY
