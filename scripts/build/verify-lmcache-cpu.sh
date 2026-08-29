#!/usr/bin/env bash
# verify-lmcache-cpu.sh — Verify the CPU-only lmcache wheel.
# Checks:
#   1. Import lmcache + common C++ extensions
#   2. lmcache.torch_device_type == cpu
#   3. No CUDA ops module present
#   4. MP server module imports (needed for coordinator)
set -euo pipefail

wheel="$1"
[ -f "$wheel" ] || { echo "wheel not found: $wheel"; exit 1; }

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

# Install the wheel (no-deps to avoid pulling GPU torch from PyPI)
python3.12 -m venv "$BUILD_DIR/venv"
# shellcheck disable=SC1091
. "$BUILD_DIR/venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "$wheel" --no-deps

# Install runtime deps (from requirements/common.txt or baked-in fallback)
SRC_DIR="$BUILD_DIR/src"
git clone --depth 1 --branch cuda13.3-aarch64-gb10 \
  https://github.com/Fulton-Engineering-Services/lmcache "$SRC_DIR" 2>/dev/null || true
if [ -f "$SRC_DIR/requirements/common.txt" ]; then
  grep -vE '^#|^$|torch|cufile' "$SRC_DIR/requirements/common.txt" > "$BUILD_DIR/req.txt"
  pip install -r "$BUILD_DIR/req.txt"
else
  pip install pyzmq aiohttp httpx py-cpuinfo psutil numpy requests prometheus-client \
    cachetools sortedcontainers PyYAML msgspec cryptography multidict
fi

echo "=== lmcache CPU wheel verification ==="

# 1. Import test
python3 -c "
import lmcache
print(f'lmcache version: {lmcache.__version__}')
print(f'lmcache.torch_device_type: {lmcache.torch_device_type}')
"

# 2. Assert CPU mode
python3 -c "
import lmcache
assert lmcache.torch_device_type == 'cpu', \
    f'expected cpu, got {lmcache.torch_device_type}'
print('PASS: torch_device_type is cpu')
"

# 3. No CUDA ops module
python3 -c "
import lmcache
try:
    import lmcache.cuda_ops
    print('FAIL: cuda_ops should not be present in CPU wheel')
    exit(1)
except (ModuleNotFoundError, ImportError):
    print('PASS: cuda_ops absent (expected)')
"

# 4. Common C++ extensions present
python3 -c "
import lmcache.lmcache_native
import lmcache.lmcache_redis
import lmcache.lmcache_fs
print('PASS: common C++ extensions (lmcache_native, lmcache_redis, lmcache_fs)')
"

# 5. MP server module imports (coordinator doesn't run server, but the
#    coordinator CLI 'lmcache coordinator' is a separate entry point.
#    Verify the top-level module structure is intact.)
python3 -c "
from lmcache.v1.api_server import app
print('PASS: coordinator API server module imports')
"

echo "=== all checks passed ==="