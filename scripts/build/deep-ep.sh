#!/usr/bin/env bash
# Build deep-ep (UCCL's deep_ep wrapper) wheel — pure Python.
#
# The deep_ep wrapper is a drop-in replacement for DeepSeek's deep_ep package.
# It imports uccl.ep (the compiled EP extension) and provides the same Buffer
# API for vLLM's --all2all-backend deepep_{low_latency,high_throughput}.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg deep-ep
setup_venv
clone_fork
cd "$(src_dir)"

# Pure Python — --universal produces py3-none-any (matches packages.json wheel).
python3 setup.py bdist_wheel

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
