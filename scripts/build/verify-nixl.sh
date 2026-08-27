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

docker run --rm --gpus all \
  -v "$DIST_DIR:/dist:ro" \
  -v "$ROOT/docker/nixl/smoke-test.py:/opt/nixl/smoke-test.py:ro" \
  "$NIXL_RUNTIME_IMAGE" \
  bash -c "set -euo pipefail; \
    pip install --break-system-packages --force-reinstall --no-deps /dist/${PKG_WHEEL}; \
    python3 /opt/nixl/smoke-test.py"

echo "==> verify-nixl OK: $PKG_WHEEL"