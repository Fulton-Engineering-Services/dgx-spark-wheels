#!/usr/bin/env bash
# verify-deep-ep.sh — verify the deep_ep wrapper wheel.
#
# Pure-Python verify: find_spec confirms the package is installed without
# triggering the import chain (deep_ep.__init__ imports uccl.ep which needs a
# CUDA context). Full import is verified in the vllm-gb10 Dockerfile's
# build-time assertion where both uccl and deep_ep are installed.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg deep-ep
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL" --no-deps

python3 - <<'PY'
import importlib.util
assert importlib.util.find_spec("deep_ep") is not None, "deep_ep package not installed"
print("DEEP_EP_OK")
PY
