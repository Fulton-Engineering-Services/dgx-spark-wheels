#!/usr/bin/env bash
# verify-lmcache.sh — smoke-test with real LMCache CUDA kernel launches.
#
# Verifies: (1) lmcache.cuda_ops extension loads (CUDA kernels compiled for
# sm_121), (2) lmcache.lmcache_native common C++ extension loads, (3) a real
# KV transfer kernel executes (load_and_reshape_flash extracts from paged KV
# cache to a CPU memory object, reshape_and_cache_back_flash writes it back),
# (4) the MP server CLI entry point is importable.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg lmcache
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL" --no-deps

# Install LMCache's runtime deps from its own requirements/common.txt
# (the build step cloned the source into $SRC_DIR). This is the single
# canonical source of truth -- no manual enumeration that risks missing
# transitive deps every time upstream adds a new import.
#
# Skips:
#   torch          -- already in venv from setup_venv (from-source, cu13.3)
#   cufile-python  -- cuFile unsupported on GB10 (CU_FILE_IO_NOT_SUPPORTED)
#   setuptools     -- already in venv (build dep, not runtime)
#   setuptools_scm -- build dep only
#   pytest         -- test dep only
# All other deps installed with full pip resolution (no --no-deps).
COMMON_REQ="$SRC_DIR/lmcache/requirements/common.txt"
if [ -f "$COMMON_REQ" ]; then
    TMP_REQ=$(mktemp)
    grep -v '^\s*#' "$COMMON_REQ" | grep -v '^\s*$' \
        | grep -v '^torch$' \
        | grep -v '^cufile-python$' \
        | grep -v '^setuptools' \
        | grep -v '^setuptools_scm' \
        | grep -v '^pytest$' \
        | grep -v '^\s*$' \
        > "$TMP_REQ"
    echo "==> installing LMCache deps from $COMMON_REQ" >&2
    cat "$TMP_REQ" >&2
    pip install -r "$TMP_REQ"
    rm -f "$TMP_REQ"
else
    echo "WARNING: $COMMON_REQ not found -- is the build step missing?" >&2
    # Fallback: run discover_import_deps.py locally to generate this list,
    # then paste it here.
    pip install \
        PyYAML msgspec numpy prometheus-client psutil py-cpuinfo \
        requests aiohttp httpx cryptography cachetools sortedcontainers \
        numba pyzmq opentelemetry-api opentelemetry-sdk \
        opentelemetry-exporter-otlp opentelemetry-exporter-prometheus \
        fastapi uvicorn httptools blake3 awscrt huggingface_hub \
        safetensors \
        google-api-core google-cloud-bigtable
fi

python3 - <<'PY'
import torch

assert torch.cuda.is_available(), "no CUDA device"
assert torch.cuda.get_device_capability() == (12, 1), \
    f"expected sm_121, got {torch.cuda.get_device_capability()}"

# Common C++ extension (bitmap, fold, ttl_lock, etc.)
import lmcache.lmcache_native as lmcache_native

# CUDA extension (mem/blend/AC/pos kernels)
import lmcache.cuda_ops as cuda_ops

# Minimal paged KV cache: [num_blocks, block_size, num_heads, head_size]
num_blocks = 10
block_size = 16
num_heads = 8
head_size = 128
num_layers = 1
num_tokens = 4
device = "cuda"
dtype = torch.bfloat16

shape = [num_blocks, block_size, num_heads, head_size]
key_cache = torch.rand(shape, dtype=dtype, device=device)
value_cache = torch.rand(shape, dtype=dtype, device=device)
key_cache_ref = key_cache.clone()
value_cache_ref = value_cache.clone()

# Slot mapping: map each token to a slot in the paged cache
slot_mapping = torch.tensor([0, 1, 2, 3], device=device)

# Memory object: [2 (K+V), num_layers, num_tokens, num_heads * head_size] on CPU.
# MUST be pinned -- get_kernel_ptr() calls cudaHostGetDevicePointer() on CPU
# tensors, which requires registered/mapped memory. On GB10 unified memory
# non-pinned pointers may "succeed" but produce cache-coherency-corrupted
# roundtrips (GPU writes appear to land but subsequent GPU reads return
# stale data). This matches LMCache's own test convention (test_torch_ops.py
# line 394: mem_obj_tensor = mem_obj_tensor.pin_memory()).
mem_obj = torch.zeros(
    2, num_layers, num_tokens, num_heads * head_size, dtype=dtype
).pin_memory()

# Extract: paged GPU KV -> CPU memory object (launches a real CUDA kernel)
cuda_ops.load_and_reshape_flash(
    mem_obj, key_cache, value_cache, slot_mapping, 0
)
torch.cuda.synchronize()

# Verify extraction populated the memory object
assert mem_obj.abs().sum() > 0, "load_and_reshape_flash produced empty output"

# Write back: CPU memory object -> fresh paged GPU KV (launches a kernel)
key_cache_new = torch.zeros_like(key_cache)
value_cache_new = torch.zeros_like(value_cache)
cuda_ops.reshape_and_cache_back_flash(
    mem_obj, key_cache_new, value_cache_new, slot_mapping, 0
)
torch.cuda.synchronize()

# Verify roundtrip: the written-back slots should match the original.
# slot_mapping maps tokens to linear slot indices in the paged cache; each
# slot = block_idx * block_size + block_offset. Index [block_idx, block_offset]
# to compare only the specific positions that were written, not the entire
# block (other offsets remain zero). Matches LMCache's own test convention
# (test_torch_ops.py lines 425-430).
for slot in slot_mapping.tolist():
    block_idx = slot // block_size
    block_offset = slot % block_size
    if not torch.allclose(
        key_cache_ref[block_idx, block_offset],
        key_cache_new[block_idx, block_offset],
        atol=1e-2,
    ):
        max_diff = (
            key_cache_ref[block_idx, block_offset].float()
            - key_cache_new[block_idx, block_offset].float()
        ).abs().max().item()
        raise AssertionError(
            f"key_cache slot {slot} (block={block_idx}, off={block_offset}) "
            f"mismatch after roundtrip (max_diff={max_diff:.4f})"
        )
    if not torch.allclose(
        value_cache_ref[block_idx, block_offset],
        value_cache_new[block_idx, block_offset],
        atol=1e-2,
    ):
        max_diff = (
            value_cache_ref[block_idx, block_offset].float()
            - value_cache_new[block_idx, block_offset].float()
        ).abs().max().item()
        raise AssertionError(
            f"value_cache slot {slot} (block={block_idx}, off={block_offset}) "
            f"mismatch after roundtrip (max_diff={max_diff:.4f})"
        )

# Verify the MP server CLI entry point is importable (out-of-process daemon)
from lmcache.v1.multiprocess.server import MPCacheServer
print(f"lmcache CUDA kernel launch + KV roundtrip OK: {torch.cuda.get_device_capability()}")
print("lmcache MP server entry point importable: MPCacheServer")
PY