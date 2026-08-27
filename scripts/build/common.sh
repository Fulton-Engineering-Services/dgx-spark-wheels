#!/usr/bin/env bash
# common.sh — shared helpers for scripts/build/<pkg>.sh.
# Sourced, not executed directly. Provides:
#   load_pkg <name>   — read packages.json, export fork/ref/subdir/version/local_seg
#   setup_venv        — create a python3.12 venv with torch (local wheel or cu130) + build deps
#   clone_fork        — clone the pinned fork into $SRC_DIR
#
# Runs inside the ghcr.io build-env:24.04-cu<variant> container (CUDA devel,
# glibc 2.39, python3.12) on a self-hosted GB10 runner. CUDA is on PATH via
# the image; a live GPU is present (--gpus all in build-wheel.yml).
#
# CUDA_VARIANT (e.g. cu13.3 | cu13.0) selects which train this build belongs
# to. It defaults to cu13.3 (back-compat for manual dispatch); CI sets it from
# build-wheel.yml's `cuda` input. It is substituted into the manifest's
# canonical cu13.3 literals (local segment + wheel filename) by load_pkg.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/packages.json"
BUILD_DIR="${BUILD_DIR:-/tmp/wheel-build}"
VENV="$BUILD_DIR/venv"
SRC_DIR="$BUILD_DIR/src"
DIST_DIR="$ROOT/dist"
mkdir -p "$DIST_DIR" "$SRC_DIR"

# Active CUDA train; default to the canonical cu13.3 for manual dispatch.
CUDA_VARIANT="${CUDA_VARIANT:-cu13.3}"

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
  # Variant plumbing: the manifest's cu13.3 literals are the canonical
  # defaults; substitute the active CUDA_VARIANT into the local segment and
  # the wheel filename (the token appears exactly once in each). A cu13.3
  # build is a no-op substitution.
  PKG_LOCAL_SEG="${PKG_LOCAL_SEG/cu13.3/$CUDA_VARIANT}"
  PKG_WHEEL="${PKG_WHEEL/cu13.3/$CUDA_VARIANT}"
  echo "==> package: $PKG_NAME  variant: $CUDA_VARIANT  ref: $PKG_FORK_REF" >&2
}

# index_wheel_url <name> — print the GitHub Release download URL for a package's
# variant-substituted wheel from OUR index, or nothing on failure. The URL
# mirrors generate-index.py and build-wheel.yml's release-tag scheme
# (<name>-v<public>-cu<variant>). Used by setup_venv so single-wheel dispatches
# consume our own published torch/triton instead of PyPI.
index_wheel_url() {
  python3 - "$1" "$CUDA_VARIANT" "$MANIFEST" <<'PY'
import json, sys
name, variant, manifest = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    pkg = next(p for p in json.load(open(manifest))["packages"] if p["name"] == name)
except (StopIteration, FileNotFoundError, json.JSONDecodeError):
    sys.exit(1)
wheel = pkg["wheel"].replace("cu13.3", variant)
public = pkg.get("public_version") or pkg["version"].split("+", 1)[0]
repo = "Fulton-Engineering-Services/dgx-spark-wheels"
print(f"https://github.com/{repo}/releases/download/{name}-v{public}-{variant}/{wheel}")
PY
}

setup_venv() {
  python3.12 -m venv "$VENV"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  pip install --upgrade pip setuptools wheel
  # torch: prefer a locally-built wheel (TORCH_WHEEL artifact), then our own
  # published index wheel (variant-matching), then PyPI cu130 as a last resort
  # (only reachable on the very first bootstrap before torch is published).
  # SKIP_TORCH_INSTALL=1 skips torch entirely (used by the torch build itself,
  # which is building torch, not consuming it).
  if [ "${SKIP_TORCH_INSTALL:-0}" = "1" ]; then
    echo "==> SKIP_TORCH_INSTALL=1: not installing torch (building it)" >&2
  elif [ -n "${TORCH_WHEEL:-}" ] && [ -f "$TORCH_WHEEL" ]; then
    echo "==> installing torch from local wheel: $TORCH_WHEEL" >&2
    pip install "$TORCH_WHEEL"
  else
    local torch_url; torch_url="$(index_wheel_url torch)"
    if [ -n "$torch_url" ] && pip install "$torch_url" 2>/dev/null; then
      echo "==> installed torch from our index: $(basename "$torch_url")" >&2
    else
      echo "==> our torch wheel not published yet; falling back to PyPI cu130" >&2
      pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
    fi
  fi
  # triton: optional locally-built wheel (TRITON_WHEEL artifact), else our own
  # published index wheel (variant-matching), else PyPI. PyTorch declares triton
  # as a normal dependency when built with USE_SYSTEM_TRITON=1, so downstream
  # wheels install it here when provided.
  if [ -n "${TRITON_WHEEL:-}" ] && [ -f "$TRITON_WHEEL" ]; then
    echo "==> installing triton from local wheel: $TRITON_WHEEL" >&2
    pip install "$TRITON_WHEEL"
  else
    local triton_url; triton_url="$(index_wheel_url triton)"
    if [ -n "$triton_url" ] && pip install "$triton_url" 2>/dev/null; then
      echo "==> installed triton from our index: $(basename "$triton_url")" >&2
    else
      echo "==> our triton wheel not published yet; falling back to PyPI" >&2
      pip install triton
    fi
  fi
  pip install ninja packaging wheel setuptools psutil numpy
  echo "==> venv ready at $VENV ($(python --version))" >&2
}

clone_fork() {
  local dest="$SRC_DIR/$(basename "$PKG_FORK")"
  rm -rf "$dest"
  local clone_url="$PKG_FORK"
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    clone_url="${PKG_FORK/#https:\/\/github.com\//https://x-access-token:${GITHUB_TOKEN}@github.com/}"
  fi
  git clone --depth 1 --branch "$PKG_FORK_BRANCH" "$clone_url" "$dest"
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
