#!/usr/bin/env bash
# Build onnxruntime-gpu wheel for GB10 (CMAKE_CUDA_ARCHITECTURES=120).
# ~90 min on 20 cores. cuDNN comes from the venv's nvidia-cudnn-cu13 package,
# not a system path (see docs/build-environment.md). Requires submodules.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg onnxruntime-gpu
setup_venv
# cuDNN 9 ships inside the build venv's nvidia-cudnn-cu13 package.
pip install nvidia-cudnn-cu13

src="$(clone_fork)"
cd "$src"
echo "==> initializing submodules (this is large)..."
git submodule update --init --recursive

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
CUDNN_HOME="$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cudnn'))")"
export CUDNN_HOME
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"

./build.sh --config Release --build_wheel --use_cuda --skip_tests \
  --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" --cuda_version 13.0 \
  --parallel "$(nproc)" \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=120 onnxruntime_BUILD_UNIT_TESTS=OFF

# Wheel lands in build/Linux/Release/dist/onnxruntime_gpu-*.whl
built="$(ls -t build/Linux/Release/dist/onnxruntime_gpu-*.whl | head -1)"
final="$("$ROOT/scripts/build/inject-local-version.sh" "$built" "$PKG_LOCAL_SEG")"
cp "$final" "$DIST_DIR/"
echo "==> built $PKG_NAME -> $DIST_DIR/$(basename "$final")"
