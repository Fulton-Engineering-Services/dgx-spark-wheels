# Build environment

All wheels in this index are built (or rebuilt) against this exact ABI.
Reproduce it before rebuilding anything.

| Fact | Value |
|---|---|
| Host arch | `aarch64` |
| GPU | GB10-class, compute capability `(12, 1)` = `sm_121` |
| CUDA (`nvcc`) | `13.3.1` (cu13.3 train) / `13.0.3` (cu13.0 train) |
| Python | `3.12` (`cp312` wheels only) |
| torch | `2.13.0+cu133` (built from source against the train's CUDA) |
| triton | built from source (see packages.json for pinned fork/ref) |
| glibc floor | `2.39` (Ubuntu 24.04) |

The build-env container images are
`ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:24.04-cu13.3`
and `build-env:24.04-cu13.0`, built from `docker/build-env/Dockerfile`
(parametrized via `CUDA_IMAGE_TAG` / `CUDA_PKG_SUFFIX` build-args — one image
per CUDA train). The Dockerfile is `FROM nvcr.io/nvidia/cuda:13.3.1-devel-ubuntu24.04`
by default; the `cu13.0` image substitutes `13.0.3`.

## Reproducing with the build-env container

```bash
# Build scripts run inside this container (CI: self-hosted GB10 runner).
# For local iteration, pull the image and exec into it:
docker run --rm --gpus all --ulimit memlock=-1 -it \
    -v $(pwd):/src -w /src \
    ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:24.04-cu13.3 \
    bash

# Inside the container, create a venv with the from-source torch (or PyPI fallback):
python3.12 -m venv /tmp/build-venv
source /tmp/build-venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install ninja packaging wheel setuptools psutil numpy
```

## The glibc gotcha

A wheel's compiled `.so` links against the glibc of the **build host**, not
just its CUDA/torch/Python ABI — and none of `cp312`, `linux_aarch64`, or a
CUDA version tag in the filename capture that. Building on a bare
Ubuntu 26.04 host (glibc 2.41) produces a wheel that fails to import in an
Ubuntu 24.04 runtime (glibc 2.39) with `GLIBC_2.41' not found`, even when
CUDA/torch/Python all match exactly.

**Policy for this index: build every wheel inside a container matching the
oldest glibc we support** (`ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:24.04-cu13.3`
or `-cu13.0`, glibc 2.39), not on the bare host. (That image is `FROM` the
`nvcr.io/nvidia/cuda:13.x-devel-ubuntu24.04` base with Python 3.12 and
build deps — see `docker/build-env/Dockerfile`.)

## ciSPARSELt and cuFile (GDS) paths

The build-env image installs cuSPARSELt 0.8.1 and cuFile (GPUDirect Storage)
under `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda
export CUSPARSELT_HOME="$CUDA_HOME"
export CUFILE_HOME="$CUDA_HOME"
```

These are also set in the Dockerfile. PyTorch's `setup.py` discovers them
automatically when the `USE_CUSPARSELT=1` / `USE_CUFILE=1` flags are set.

## Per-package build notes

### `triton`

- Fork: `Fulton-Engineering-Services/triton`, branch `cuda13.3-aarch64-gb10`, pinned to the commit PyTorch 2.13.0 expects.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
  export MAX_JOBS=$(nproc)
  cd python && python setup.py bdist_wheel
  ```
- Triton downloads its own pinned prebuilt LLVM during build (see `cmake/llvm-hash.txt`). No system LLVM package needed.
- Depends on `torch` (PyPI cu130 or the from-source torch wheel).

### `torch`

- Fork: `Fulton-Engineering-Services/pytorch`, branch `cuda13.3-aarch64-gb10`, built from source with cuDNN, cuSPARSELt, cuFile, system NCCL, and system Triton.
- Apply `dgx_spark_config/pytorch/pytorch.patch` on the fork to add `sm_121` support.
- Build:
  ```bash
  export PYTORCH_BUILD_VERSION=2.13.0 PYTORCH_BUILD_NUMBER=1
  export USE_CUDA=1 USE_CUDNN=1 USE_CUSPARSELT=1 USE_CUFILE=1
  export USE_SYSTEM_NCCL=1 USE_SYSTEM_TRITON=1
  export BLAS=OpenBLAS USE_FBGEMM=0 USE_NNPACK=1 USE_MKLDNN=0
  export BUILD_TEST=0 USE_KINETO=0 USE_ITT=0
  export CMAKE_GENERATOR=Ninja TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"
  export CUSPARSELT_ROOT_DIR="$CUDA_HOME" CUFILE_ROOT_DIR="$CUDA_HOME"
  export MAX_JOBS=$(nproc)
  python setup.py bdist_wheel
  ```
- **Hours on 20 cores.** Monitor `free -h` and thermals. The `build-all.yml` job has a 12-hour timeout.
- Requires `triton` wheel installed first (`USE_SYSTEM_TRITON=1`).

### `torchaudio`

- Fork: `Fulton-Engineering-Services/audio`, branch `cuda13.3-aarch64-gb10`, tag `v2.13.0`.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
  export BUILD_VERSION=2.13.0 TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"
  export MAX_JOBS=$(nproc)
  python setup.py bdist_wheel
  ```
- Requires torch (from-source or PyPI cu130) installed in the venv.

### `torchvision`

- Fork: `Fulton-Engineering-Services/vision`, branch `cuda13.3-aarch64-gb10`, tag `v0.28.0`.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
  export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0;12.1+PTX"
  export MAX_JOBS=$(nproc)
  python setup.py bdist_wheel
  ```
- Requires torch and the image codec libraries (`libjpeg-dev`, `libpng-dev`) from the build-env.

### `flash-attn`

- Fork pinned to tag `v2.8.3.post1`, branch `cuda13-aarch64-gb10`.
- Narrowed to `sm_120` (PTX-forward-compat on `sm_121`) — not a multi-arch build.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
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
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="12.1"
  export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=8
  python setup.py bdist_wheel
  ```
- Small extension set, builds in well under a minute.

### `sageattn3`

- Same fork/ref as `sageattention`; lives in `sageattention3_blackwell/`.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
  export MAX_JOBS=8
  python setup.py bdist_wheel
  ```
- No arch env var needed — `setup.py` probes the live GPU's compute capability and emits `sm_121a`.

### `nunchaku`

- Fork pinned to tag `v1.2.1`, branch `cuda13-aarch64-gb10`.
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
  export NUNCHAKU_BUILD_WHEELS=1 MAX_JOBS=8
  python setup.py bdist_wheel
  ```
- `--gpus all` is required — `setup.py` probes the live GPU to pick `121a`.

### `onnxruntime-gpu`

- Fork based on `rel-1.22.0` with CUDA-13/aarch64 build-compatibility patches.
- Build:
  ```bash
  git clone --depth 1 --branch cuda13-aarch64-gb10 \
    https://github.com/Fulton-Engineering-Services/onnxruntime.git ~/ort-build/onnxruntime
  cd ~/ort-build/onnxruntime && git submodule update --init --recursive
  export CUDA_HOME=/usr/local/cuda
  export CUDNN_HOME="$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cudnn'))")"
  export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$LD_LIBRARY_PATH"
  ./build.sh --config Release --build_wheel --use_cuda --skip_tests \
    --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" --cuda_version "${CUDA_VARIANT#cu}" \
    --parallel "$(nproc)" \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=120 onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_POLICY_VERSION_MINIMUM=3.5
  ```
- cuDNN 9 ships inside the build venv's `nvidia-cudnn-cu13` package, not a system path.
- ~90 min on 20 cores.

### `flashinfer-python`

- Fork pinned to tag `v0.6.17`, branch `cuda13-aarch64-gb10`.
- **`flashinfer-python` only** — a JIT package (custom PEP 517 `build_backend.py`);
  kernels compile at runtime via cutlass-dsl + tvm-ffi, not a monolithic nvcc
  build. Do NOT build `flashinfer-cubin`: upstream artifactory has no
  `sm_120`/`sm_121` cubins (only `sm_100a`/`103a`/`110a`).
- Build:
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH="$CUDA_HOME/bin:$PATH"
  pip install "apache-tvm-ffi>=0.1.6,!=0.1.8,<0.2"
  BUILD_NVEP=0 pip wheel --no-build-isolation --no-deps -w dist .
  ```
- `BUILD_NVEP=0` skips the nvcc/meson/UCX `moe_ep` backends (NIXL-EP + NCCL-EP).
- Verify installs the `nvidia-cutlass-dsl[cu13]` extra: the base requirements
  pull cu12 libs by default; GB10 needs the cu13 libs.

## Verification

Every wheel is smoke-tested with a real CUDA op before being published, not
just a bare `import`. A wheel that imports but never launches a kernel can
hide an ABI mismatch that only surfaces mid-inference. See
`scripts/build/verify-*.sh` for the exact per-package smoke test.