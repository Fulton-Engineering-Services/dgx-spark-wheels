#!/usr/bin/env bash
# common.sh — shared helpers for scripts/build/<pkg>.sh.
# Sourced, not executed directly. Provides:
#   load_pkg <name>   — read packages.json, export fork/ref/subdir/version/local_seg
#   setup_venv        — create a python3.12 venv with torch (cu130) + build deps
#   clone_fork        — clone the pinned fork into $SRC_DIR
#
# Runs inside the ghcr.io build-env:22.04 container (CUDA 13.0.1 devel,
# glibc 2.35, python3.12) on a self-hosted GB10 runner. CUDA is on PATH via
# the image; a live GPU is present (--gpus all in build-wheel.yml).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/packages.json"
BUILD_DIR="${BUILD_DIR:-/tmp/wheel-build}"
VENV="$BUILD_DIR/venv"
SRC_DIR="$BUILD_DIR/src"
DIST_DIR="$ROOT/dist"
mkdir -p "$DIST_DIR" "$SRC_DIR"

load_pkg() {
  local name="$1"
  # Parse packages.json with python3 (no jq guaranteed in the container).
  eval "$(python3 - "$MANIFEST" "$name" <<'PY'
import json, shlex, sys
manifest, name = sys.argv[1], sys.argv[2]
data = json.load(open(manifest))
for p in data["packages"]:
    if p["name"] == name:
        for k in ("name","version","public_version","wheel","fork","fork_branch",
                  "fork_ref","fork_subdir","upstream","upstream_license"):
            if k in p:
                print(f'export PKG_{k.upper()}={shlex.quote(str(p[k]))}')
        # local segment = everything after the first '+' in version.
        ver = p["version"]
        print(f'export PKG_LOCAL_SEG={shlex.quote(ver.split("+",1)[1] if "+" in ver else "")}')
        break
else:
    print(f"package {name!r} not found in {manifest}", file=sys.stderr)
    sys.exit(1)
PY
)"
  echo "==> package: $PKG_NAME  version: $PKG_VERSION  ref: $PKG_FORK_REF" >&2
}

setup_venv() {
  python3.12 -m venv "$VENV"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  pip install --upgrade pip setuptools wheel
  pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
  pip install ninja packaging wheel setuptools psutil numpy
  echo "==> venv ready at $VENV ($(python --version))" >&2
}

clone_fork() {
  local dest="$SRC_DIR/$(basename "$PKG_FORK")"
  rm -rf "$dest"
  git clone --depth 1 --branch "$PKG_FORK_BRANCH" "$PKG_FORK" "$dest"
  # Verify the pinned ref matches (depth-1 clone HEAD should == fork_ref).
  local head; head="$(git -C "$dest" rev-parse HEAD)"
  if [ "$head" != "$PKG_FORK_REF" ]; then
    echo "WARNING: branch HEAD ($head) != pinned fork_ref ($PKG_FORK_REF)" >&2
    echo "         fetching full history to resolve the exact ref..." >&2
    git -C "$dest" fetch --unshallow
    git -C "$dest" checkout "$PKG_FORK_REF"
  fi
  echo "==> cloned $PKG_NAME at $PKG_FORK_REF -> $dest" >&2
  echo "$dest"
}

# Resolve the src dir for the current package (with fork_subdir if present).
src_dir() {
  local base="$SRC_DIR/$(basename "$PKG_FORK")"
  if [ -n "${PKG_FORK_SUBDIR:-}" ]; then
    echo "$base/$PKG_FORK_SUBDIR"
  else
    echo "$base"
  fi
}
