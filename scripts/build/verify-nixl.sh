#!/usr/bin/env bash
# verify-nixl.sh — verify the nixl wheel inside the nixl-runtime image.
#
# The nixl runtime depends on UCX, Mooncake, and libfabric libraries that only
# exist inside the nixl-runtime Docker image. So we install the freshly built
# wheel into the runtime image and run the smoke test there. This runs on the
# bare self-hosted GB10 runner (no build-env container).
set -euo pipefail

. "$(dirname "$0")/common.sh"
load_pkg nixl

TAG="${NIXL_IMAGE_TAG:-24.04-cu13.3-sm121}"
: "${NIXL_RUNTIME_IMAGE:=ghcr.io/fulton-engineering-services/dgx-spark-wheels/nixl-runtime:${TAG}}"

echo "==> verifying $PKG_WHEEL in ${NIXL_RUNTIME_IMAGE}" >&2

# Pull the latest runtime image so we always use the most recently pushed one.
docker pull "$NIXL_RUNTIME_IMAGE" >&2

# Verify the wheel contains nixl_ep_cu13 before installing.
docker run --rm \
  -v "$DIST_DIR:/dist:ro" \
  "$NIXL_RUNTIME_IMAGE" \
  python3 -c 'import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); ep=[n for n in z.namelist() if "nixl_ep_cu13/" in n]; print("nixl_ep_cu13 files in wheel:", len(ep)); assert ep, "nixl_ep_cu13 not found in wheel"' \
  "/dist/${PKG_WHEEL}"

# Install the wheel and run the smoke test.
docker run --rm --gpus all \
  -v "$DIST_DIR:/dist:ro" \
  -v "$ROOT/docker/nixl/smoke-test.py:/opt/nixl/smoke-test.py:ro" \
  "$NIXL_RUNTIME_IMAGE" \
  bash -c 'set -euo pipefail; pip install --break-system-packages --force-reinstall --no-deps /dist/'"${PKG_WHEEL}"'; python3 /opt/nixl/smoke-test.py'

echo "==> verify-nixl OK: $PKG_WHEEL"