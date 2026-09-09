#!/usr/bin/env bash
# verify-flashinfer-jit-cache.sh — prove AOT coverage on sm_121a:
# 1. version lock passes, FLASHINFER_AOT_DIR resolves into flashinfer_jit_cache
# 2. .so count is sane and the fa2 batch-MLA module carries a sm_121a ELF
# 3. with FLASHINFER_DISABLE_JIT=1, a real prefill launch succeeds (would raise
#    MissingJITCacheError if AOT were missing)
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flashinfer-jit-cache
setup_venv
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL" "$(index_wheel_url flashinfer-python)" "nvidia-cutlass-dsl[cu13]"

python3 - <<'PY'
import pathlib, subprocess
import flashinfer, flashinfer_jit_cache
from flashinfer.jit import env as jit_env

assert flashinfer_jit_cache.__version__.startswith(flashinfer.__version__), \
    (flashinfer_jit_cache.__version__, flashinfer.__version__)
aot = pathlib.Path(flashinfer_jit_cache.get_jit_cache_dir())
assert jit_env.FLASHINFER_AOT_DIR == aot
sos = list(aot.rglob("*.so"))
assert len(sos) > 100, f"only {len(sos)} AOT modules"
# AOT MLA modules are named batch_mla_attention_dtype_q_<dtype>_... (the
# backend is not in the name; fp8 KV is not in the default AOT MLA set).
mla = [p for p in sos if p.parent.name.startswith("batch_mla_attention_dtype_q_")
       and "bf16" in p.parent.name]
assert mla, "no bf16 batch-MLA AOT module (names: batch_mla_attention_dtype_q_*)"
elfs = subprocess.run(["cuobjdump", "--list-elf", str(mla[0])],
                      capture_output=True, text=True).stdout
assert "sm_121a" in elfs, f"no sm_121a ELF in {mla[0]}:\n{elfs}"
print(f"AOT dir OK: {aot} ({len(sos)} modules); batch_mla ELF sm_121a OK")
PY

FLASHINFER_DISABLE_JIT=1 python3 - <<'PY'
import torch
from flashinfer import single_prefill_with_kv_cache
assert torch.cuda.get_device_capability() == (12, 1)
q = torch.randn(4, 8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(4, 8, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(4, 8, 128, dtype=torch.bfloat16, device="cuda")
out = single_prefill_with_kv_cache(q, k, v, kv_layout="NHD", pos_encoding_mode="NONE")
torch.cuda.synchronize()
assert out.shape == (4, 8, 128)
print("AOT prefill launch OK with JIT disabled:", tuple(out.shape))
PY
