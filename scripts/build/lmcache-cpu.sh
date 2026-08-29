#!/usr/bin/env bash
# Build lmcache CPU-only wheel for x86_64 (coordinator / Dell host).
# Standalone — no CUDA, no container, runs on any Linux x86_64 with Python 3.12.
# Builds the exact same lmcache fork (ca2f474) as the GB10 GPU wheel, but with
# NO_GPU_EXT=1 (no CUDA kernels, no GPU deps). Common C++ extensions
# (lmcache_native, lmcache_redis, lmcache_fs) ARE built.
set -euo pipefail

PKG_NAME=lmcache-cpu
FORK_URL=https://github.com/Fulton-Engineering-Services/lmcache
FORK_BRANCH=cuda13.3-aarch64-gb10
FORK_REF=ca2f474412e3b881c2f39e62561414c9676653dc
PUBLIC_VERSION=0.5.5rc1
LOCAL_SEG=cpu.torch2.13.glibc239
DIST_DIR="${DIST_DIR:-dist}"

# Typical glibc on Ubuntu 24.04 is 2.39; the manylinux tag reflects the build
# platform's glibc floor. On the Dell (24.04) this will be manylinux_2_31_x86_64
# or manylinux_2_39_x86_64 depending on the pip version.
PLAT_TAG="${PLAT_TAG:-manylinux_2_39_x86_64}"

mkdir -p "$DIST_DIR"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "==> package: $PKG_NAME  variant: cpu  ref: $FORK_REF"

# 1. Clone the fork at the pinned ref
SRC="$BUILD_DIR/src"
git clone --depth 1 --branch "$FORK_BRANCH" "$FORK_URL" "$SRC"
HEAD="$(git -C "$SRC" rev-parse HEAD)"
if [ "$HEAD" != "$FORK_REF" ]; then
  echo "HEAD ($HEAD) != pinned ref ($FORK_REF); fetching exact ref..." >&2
  git -C "$SRC" fetch --unshallow
  git -C "$SRC" checkout "$FORK_REF"
fi

# 2. Create venv with CPU-only torch
python3.12 -m venv "$BUILD_DIR/venv"
# shellcheck disable=SC1091
. "$BUILD_DIR/venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ninja packaging setuptools-scm build psutil numpy

# 3. Build CPU-only wheel
cd "$SRC"
NO_GPU_EXT=1 ENABLE_CXX11_ABI=1 \
  SETUPTOOLS_SCM_PRETEND_VERSION="$PUBLIC_VERSION" \
  python -m build --wheel --no-isolation --skip-dependency-check

# 4. Inject local version segment
built="$(ls -t dist/*.whl | head -1)"
echo "==> built $built"

inject_script="$(cd "$(dirname "$0")" && pwd)/inject-local-version.sh"
final="$("$inject_script" "$built" "$LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> $PKG_NAME -> $DIST_DIR/$(basename "$final")"