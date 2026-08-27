#!/usr/bin/env bash
# Build the nixl wheel by extracting the pre-built wheel from the
# gb10-proofs nixl-wheel-builder image and stamping it with the variant-aware
# local version segment used by dgx-spark-wheels.
#
# This script runs with docker (the wheel-builder image already contains the
# meson/UCX/Mooncake toolchain and the built wheel at /dist). It is intended
# for the self-hosted GB10 runner or any host that can pull the GHCR image.
set -euo pipefail

. "$(dirname "$0")/common.sh"
load_pkg nixl

TAG="${NIXL_IMAGE_TAG:-24.04-cu13.3-sm121}"
: "${NIXL_WHEEL_IMAGE:=ghcr.io/fulton-engineering-services/dgx-spark-wheels/nixl-wheel-builder:${TAG}}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI is required to extract the wheel from ${NIXL_WHEEL_IMAGE}" >&2
  exit 1
fi

mkdir -p "$DIST_DIR"

echo "==> extracting pre-built nixl wheel from ${NIXL_WHEEL_IMAGE}" >&2

tmp_extract="$(mktemp -d)"
trap 'rm -rf "$tmp_extract"' EXIT

docker run --rm --gpus all \
  -v "$tmp_extract:/out" \
  "$NIXL_WHEEL_IMAGE" \
  bash -c 'set -euo pipefail; cp /dist/*.whl /out/'

built="$(ls -t "$tmp_extract"/nixl_cu13-*.whl | head -1)"
if [ ! -f "$built" ]; then
  echo "ERROR: no nixl wheel found in ${NIXL_WHEEL_IMAGE}" >&2
  exit 1
fi

# Stamp the variant-aware local segment onto the wheel filename + METADATA.
cd "$ROOT"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"

# inject-local-version.sh writes the final wheel to ./dist/ relative to cwd.
cp -f "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
