#!/usr/bin/env bash
# verify-lmcache-mooncake.sh — verify the lmcache+mooncake wheel extracted from
# the lmcache-mooncake-wheel-builder image.
#
# The optional native lmcache.lmcache_mooncake extension links libmooncake_store
# and calls cudaHostGetDevicePointer, so it is not importable in the build-env
# container (no GPU runtime libs, no libmooncake_store). Verification therefore:
#   (1) HARD: the wheel carries the lmcache_mooncake extension (static zipfile
#       assert) — the entire point of this variant; a regression must fail.
#   (2) HARD: the wheel-builder image ships libmooncake_store.so, so the ext's
#       runtime linkage is resolvable in the lmcache-server container.
#   (3) BEST-EFFORT: import the ext inside the builder image. The builder only
#       carries a stubs libcuda.so.1 symlink (for linking), so the import may
#       fail to resolve driver symbols during a build-time check; that is a
#       WARN, not a failure. The lmcache-server image does the authoritative
#       runtime import assert under --gpus.
set -euo pipefail

. "$(dirname "$0")/common.sh"
load_pkg lmcache-mooncake

TAG="${LMCACHE_MOONCAKE_IMAGE_TAG:-24.04-cu13.3-sm121}"
: "${LMCACHE_MOONCAKE_WHEEL_IMAGE:=ghcr.io/fulton-engineering-services/dgx-spark-wheels/lmcache-mooncake-wheel-builder:${TAG}}"

echo "==> verifying $PKG_WHEEL in ${LMCACHE_MOONCAKE_WHEEL_IMAGE}" >&2
docker pull "$LMCACHE_MOONCAKE_WHEEL_IMAGE" >&2

# (1) The wheel MUST carry the native extension.
docker run --rm \
  -v "$DIST_DIR:/dist:ro" \
  "$LMCACHE_MOONCAKE_WHEEL_IMAGE" \
  python3 -c 'import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); names=z.namelist(); ok=any("lmcache_mooncake" in n for n in names); print("lmcache_mooncake entries:", [n for n in names if "lmcache_mooncake" in n][:4]); sys.exit(0 if ok else 1)' \
  "/dist/${PKG_WHEEL}"

# (2) The builder image MUST ship libmooncake_store.so (the ext's runtime dep).
docker run --rm \
  "$LMCACHE_MOONCAKE_WHEEL_IMAGE" \
  bash -c 'test -f /opt/mooncake/lib/libmooncake_store.so && echo "libmooncake_store.so present"'

# (3) Best-effort ext import (WARN on failure -- build-time libs are incomplete).
if docker run --rm --gpus all \
    -v "$DIST_DIR:/dist:ro" \
    "$LMCACHE_MOONCAKE_WHEEL_IMAGE" \
    bash -c 'export LD_LIBRARY_PATH=/opt/mooncake/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}; python3 -c "from lmcache.lmcache_mooncake import LMCacheMooncakeClient, L1RegistrationConfig; print(\"LMCACHE_MOONCAKE_EXT_IMPORT_OK\")"' >&2; then
  echo "==> lmcache_mooncake extension imports cleanly" >&2
else
  echo "WARN: lmcache_mooncake import failed in the builder image (expected: build-time" >&2
  echo "      libcuda stubs); the lmcache-server image asserts it under --gpus at runtime." >&2
fi

echo "==> verify-lmcache-mooncake OK: $PKG_WHEEL"
