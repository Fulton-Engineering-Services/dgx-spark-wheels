#!/usr/bin/env bash
# Build flashinfer-jit-cache wheel for GB10 (aarch64, CUDA 13.3, sm_121a).
#
# AOT sibling of flashinfer-python: nvcc-compiles upstream's default AOT
# module set (fa2/fa3 attention grids, fa2 batch-MLA ckv=512/kpe=64,
# sparse_mla_sm120, nvfp4-attn / fp4-quant / cutlass sm120 family, xqa, comm,
# misc) for compute_121a/sm_121a only. Upstream's release jit-cache wheel
# ships '8.0 8.9 9.0a 10.0a 10.3a 11.0a 12.0f' (no 12.1a) and, because AOT
# module lookup is name-keyed/arch-agnostic, must never co-exist with ours.
#
# The fork ref (shared with flashinfer-python) carries the SM121 mla.cuh
# tile-cap + _core.py capability-gate patches, so the AOT build source and
# the runtime-JIT headers agree.
#
# Hours-long on the GB10 runner (20 cores / 128 GiB unified). No GPU probe
# needed at build time (arch list comes from FLASHINFER_CUDA_ARCH_LIST).
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg flashinfer-jit-cache
setup_venv
clone_fork

# Submodule update at the REPO ROOT: the jit-cache build_backend compiles
# against 3rdparty/{cutlass,cccl,spdlog} and imports flashinfer.aot from the
# checkout (it re-runs submodule update itself if 3rdparty is missing, but
# the clone is shallow so do it explicitly, same as flashinfer-python).
SRC_ROOT="$SRC_DIR/$(basename "$PKG_FORK")"
git -C "$SRC_ROOT" submodule update --init --recursive

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# --no-build-isolation build deps: jit-cache pyproject [build-system].requires
# plus flashinfer's import-time deps (the backend imports flashinfer.aot).
pip install "setuptools>=77" "packaging>=24" wheel tqdm ninja requests numpy \
    nvidia-ml-py "apache-tvm-ffi>=0.1.6,!=0.1.8,<0.2" \
    jinja2 filelock click einops tabulate

# Single-arch AOT set for GB10. compilation_context.py respects the alpha
# suffix as-is -> -gencode=arch=compute_121a,code=sm_121a. REQUIRED by
# aot.compile_and_package_modules (raises otherwise).
export FLASHINFER_CUDA_ARCH_LIST="12.1a"
# Stamps _build_meta.py __version__ AND the wheel version to
# 0.6.18+<PKG_LOCAL_SEG>; PKG_LOCAL_SEG carries the leading gb10aot segment
# (see packages.json notes) so the wheel sorts AFTER upstream's +cu129/+cu130.
export FLASHINFER_LOCAL_VERSION="$PKG_LOCAL_SEG"
export MAX_JOBS="${MAX_JOBS:-8}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"

# ccache for the nvcc AOT compile: the default CCACHE_DIR (~/.ccache) is
# ephemeral in the job container, so a fork_ref-unchanged rebuild (e.g.
# iterating on the verify script or AOT config) would recompile all ~561
# modules. FlashInfer's ninja generator (flashinfer/jit/cpp_ext.py) wires a
# compiler launcher via FLASHINFER_NVCC_LAUNCHER — point it at ccache and put
# CCACHE_DIR under $BUILD_DIR (persisted by actions/cache) so object results
# survive. No-op when ccache is absent from the build-env image.
export CCACHE_DIR="$BUILD_DIR/ccache"
mkdir -p "$CCACHE_DIR"
if command -v ccache >/dev/null 2>&1; then
  export FLASHINFER_NVCC_LAUNCHER="${FLASHINFER_NVCC_LAUNCHER:-ccache}"
  export FLASHINFER_CXX_LAUNCHER="${FLASHINFER_CXX_LAUNCHER:-ccache}"
  echo "==> ccache enabled (dir: $CCACHE_DIR)" >&2
fi

cd "$(src_dir)"   # = $SRC_ROOT/flashinfer-jit-cache (fork_subdir)
pip wheel --no-build-isolation --no-deps -w "$ROOT/dist" .

built="$(ls -t "$ROOT/dist"/*.whl | head -1)"
# Idempotent no-op here (FLASHINFER_LOCAL_VERSION already stamped it); kept
# for pipeline uniformity with the other wheels.
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
