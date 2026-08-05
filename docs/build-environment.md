# Build environment

All wheels in this index are built (or rebuilt) against this exact ABI.
Reproduce it before rebuilding anything.

| Fact | Value |
|---|---|
| Host arch | `aarch64` |
| GPU | GB10-class, compute capability `(12, 1)` = `sm_121` |
| CUDA (`nvcc`) | `13.0.88` |
| Python | `3.12` (`cp312` wheels only) |
| torch | `2.13.0+cu130` (`--index-url https://download.pytorch.org/whl/cu130`) |

```bash
mkdir -p ~/gb10-wheel-build && cd ~/gb10-wheel-build
uv venv --python 3.12 .venv
export VIRTUAL_ENV=~/gb10-wheel-build/.venv
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install ninja packaging wheel setuptools psutil einops numpy
```

## The glibc gotcha

A wheel's compiled `.so` links against the glibc of the **build host**, not
just its CUDA/torch/Python ABI — and none of `cp312`, `linux_aarch64`, or a
CUDA version tag in the filename capture that. Building on a bare
Ubuntu 24.04 host (glibc 2.39) produces a wheel that fails to import in an
Ubuntu 22.04 runtime (glibc 2.35) with `GLIBC_2.38' not found`, even when
CUDA/torch/Python all match exactly.

**Policy for this index: build every wheel inside a container matching the
oldest glibc we support** (`ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:22.04`,
glibc 2.35), not on the bare host — even for packages that happen not to
need any glibc symbol newer than 2.35 today. A future dependency bump could
introduce one silently. (That image is `FROM` the
`nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04` base plus deadsnakes Python
3.12 and build deps — see `docker/build-env/Dockerfile`.)

## Per-package build notes

### `flash-attn`

- Fork pinned to tag `v2.8.3.post1`, branch `cuda13-aarch64-gb10`.
- Narrowed to `sm_120` (PTX-forward-compat on `sm_121`) — not a multi-arch
  build.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export FLASH_ATTN_CUDA_ARCHS="120"
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  export MAX_JOBS=4   # NOT nproc -- hdim256 backward kernels OOM above ~4-8
  python setup.py bdist_wheel
  ```
- **Do not raise `MAX_JOBS` above ~4–8** without watching `free -h`; the
  `flash_bwd_hdim256_*_sm80` translation units are the OOM failure mode, not
  general `nvcc` load. ~30 min at `MAX_JOBS=4`.

### `sageattention` (v2)

- Fork pinned past upstream PR #297, branch `cuda13-aarch64-gb10`.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="12.1"
  export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=8
  python setup.py bdist_wheel
  ```
- Small extension set, builds in well under a minute.

### `sageattn3`

- Same fork/ref as `sageattention`; lives in `sageattention3_blackwell/`.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export MAX_JOBS=8
  python setup.py bdist_wheel
  ```
- No arch env var needed — `setup.py` probes the live GPU's compute
  capability and emits `sm_121a` directly (this means the build step needs
  a live GPU; a hosted CI runner without one won't be able to build this
  without patching `setup.py` to accept an explicit arch override).

### `nunchaku`

- Fork pinned to tag `v1.2.1`, branch `cuda13-aarch64-gb10`.
- **Must be built inside a container matching the runtime image's glibc.**
  Rebuild:
  ```bash
  docker run --rm --gpus all \
    -v $(pwd)/nunchaku:/src/nunchaku:ro \
    -v /tmp/nunchaku-rebuild-out:/out \
    ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:22.04 \
    bash -c '
      set -eux
      python3.12 -m venv /root/build-venv && source /root/build-venv/bin/activate
      pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
      pip install ninja packaging wheel setuptools psutil
      cp -r /src/nunchaku /tmp/nunchaku-src && cd /tmp/nunchaku-src
      export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
      export NUNCHAKU_BUILD_WHEELS=1 MAX_JOBS=8
      python setup.py bdist_wheel
      cp dist/*.whl /out/
    '
  ```
- The build-env image is `FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04`,
  so the build container's glibc (2.35) matches the runtime image exactly.
  Use the `-devel` (not `-runtime`) base family your runtime image's `FROM`
  uses.
- `--gpus all` is required — `setup.py` probes the live GPU to pick `121a`.

### `onnxruntime-gpu`

- Fork based on the `rel-1.22.0` cherry-pick line, plus this fork's own
  CUDA-13/aarch64 build-compatibility commits (cub/thrust compat, CUTLASS
  version-check patch). No manual patching needed at build time.
- Build:
  ```bash
  git clone --depth 1 --branch cuda13-aarch64-gb10 \
    https://github.com/Fulton-Engineering-Services/onnxruntime.git ~/ort-build/onnxruntime
  cd ~/ort-build/onnxruntime && git submodule update --init --recursive
  export CUDA_HOME=/usr/local/cuda
  export CUDNN_HOME="$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cudnn'))")"
  export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$LD_LIBRARY_PATH"
  ./build.sh --config Release --build_wheel --use_cuda --skip_tests \
    --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" --cuda_version 13.0 \
    --parallel "$(nproc)" \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=120 onnxruntime_BUILD_UNIT_TESTS=OFF
  # wheel lands in build/Linux/Release/dist/onnxruntime_gpu-*.whl
  # ~90 min on 20 cores.
  ```
- cuDNN 9 ships inside the build venv's `nvidia-cudnn-cu13` package, not a
  system path.

## Verification

Every wheel is smoke-tested with a real CUDA op before being published, not
just a bare `import`. A wheel that imports but never launches a kernel can
hide an ABI mismatch that only surfaces mid-inference. See
`scripts/build/verify-*.sh` for the exact per-package smoke test.
