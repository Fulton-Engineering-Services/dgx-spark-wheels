#!/usr/bin/env bash
# Build flextensor wheel for GB10 — pure Python, torch-linked.
#
# flextensor is a tensor offloading and management library. This fork adds
# NVMe disk offload via cuFile (GDS): weight blocks are evicted to NVMe files
# during construction and read back to GPU via cuFileRead (direct NVMe→GPU DMA)
# or POSIX pread (fallback) during inference. On GB10's unified 128 GiB LPDDR,
# this frees DRAM for KV cache/activations by pushing cold weights to NVMe.
#
# Pure Python — no compiled extensions. setuptools_scm derives the version from
# git tags; SETUPTOOLS_SCM_PRETEND_VERSION pins it to the public version so the
# wheel filename is deterministic regardless of commit count since the last tag.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flextensor
setup_venv
clone_fork
cd "$(src_dir)"

# setuptools_scm build dep — install into the venv (we build with --no-deps,
# which skips runtime deps like beartype/pydantic but NOT build-system deps).
pip install setuptools-scm

SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_PUBLIC_VERSION" pip wheel --no-deps -w dist .

built="$(ls -t dist/*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
