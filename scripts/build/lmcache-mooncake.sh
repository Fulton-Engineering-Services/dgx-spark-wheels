#!/usr/bin/env bash
# Build the LMCache + Mooncake-native wheel for GB10 (sm_121).
#
# Extracts the wheel from lmcache-mooncake-wheel-builder (which compiles the
# optional lmcache.lmcache_mooncake extension against a from-source Mooncake
# Store SDK) and stamps it with the variant-aware local version segment.
#
# Mirrors scripts/build/nixl.sh (extract-from-builder-image pattern); the
# builder image is built by build-nixl-images.yml-style CI, or locally:
#   docker build -t lmcache-mooncake-wheel-builder:local docker/lmcache-mooncake
set -euo pipefail

. "$(dirname "$0")/common.sh"
load_pkg lmcache-mooncake

TAG="${LMCACHE_MOONCAKE_IMAGE_TAG:-24.04-cu13.3-sm121}"
: "${LMCACHE_MOONCAKE_WHEEL_IMAGE:=ghcr.io/fulton-engineering-services/dgx-spark-wheels/lmcache-mooncake-wheel-builder:${TAG}}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI is required to extract the wheel from ${LMCACHE_MOONCAKE_WHEEL_IMAGE}" >&2
  exit 1
fi

mkdir -p "$DIST_DIR"

echo "==> extracting pre-built lmcache+mooncake wheel from ${LMCACHE_MOONCAKE_WHEEL_IMAGE}" >&2
docker pull "$LMCACHE_MOONCAKE_WHEEL_IMAGE" >&2

tmp_extract="$(mktemp -d)"
trap 'rm -rf "$tmp_extract"' EXIT

docker run --rm --gpus all \
  -v "$tmp_extract:/out" \
  "$LMCACHE_MOONCAKE_WHEEL_IMAGE" \
  bash -c 'set -euo pipefail; cp /dist/*.whl /out/'

built="$(ls -t "$tmp_extract"/lmcache-*.whl | head -1)"
if [ ! -f "$built" ]; then
  echo "ERROR: no lmcache wheel found in ${LMCACHE_MOONCAKE_WHEEL_IMAGE}" >&2
  exit 1
fi

# Verify the native mooncake extension is present.
if ! python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); ok=any('lmcache_mooncake' in n for n in z.namelist()); sys.exit(0 if ok else 1)" "$built"; then
  echo "ERROR: wheel does not contain lmcache_mooncake — was BUILD_MOONCAKE enabled?" >&2
  exit 1
fi
echo "==> lmcache_mooncake extension present in wheel" >&2

# Stamp the variant-aware local segment onto the wheel filename + METADATA.
cd "$ROOT"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"