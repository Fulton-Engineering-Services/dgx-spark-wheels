# syntax=docker/dockerfile:1
#
# Multi-stage Dockerfile for GB10 nixl runtime + wheel-builder images.
#
# Self-contained: clones nixl, UCX, and Mooncake source directly via git at
# pinned refs. The build context is docker/nixl/ (this Dockerfile + the
# smoke-test.py that gets baked into the runtime image).
#
# Targets:
#   runtime       - minimal image with nixl C++/Python runtime, UCX/Mooncake libs.
#   wheel-builder - extends nixl-builder with Python build tools; emits a wheel.
#
# Published tags (GHCR, under the dgx-spark-wheels namespace):
#   nixl-runtime:24.04-cu13.3-sm121
#   nixl-wheel-builder:24.04-cu13.3-sm121

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.3.1-devel-ubuntu24.04
ARG PYTHON_VERSION=3.12

# Source refs — pinned for reproducibility. Override with --build-arg to test
# other commits/tags.
ARG NIXL_REPO=https://github.com/ai-dynamo/nixl.git
ARG NIXL_REF=b0cbb237354d72b83500d5214d2b4484f9866fa3
ARG UCX_REPO=https://github.com/openucx/ucx.git
ARG UCX_REF=7a5e80f389ea345c0480ca49a73d0cbf2b1d79d0
ARG MOONCAKE_REPO=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_REF=v0.3.12.post1

# ------------------------------------------------------------------------------
# base-env: CUDA devel + build tools + RDMA / libfabric / ASIO dependencies.
# ------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base-env
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        autoconf automake libtool pkg-config \
        cmake ninja-build \
        git curl ca-certificates \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip \
        libibverbs-dev librdmacm-dev \
        libfabric-dev \
        rdma-core \
        libnuma-dev \
        libhwloc-dev \
        libasio-dev \
        libgflags-dev \
        libprotobuf-dev protobuf-compiler \
        libgoogle-glog-dev \
        libssl-dev \
        libuv1-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
WORKDIR /src

# ------------------------------------------------------------------------------
# ucx-builder: clone UCX at pinned ref, build to /opt/ucx.
# ------------------------------------------------------------------------------
FROM base-env AS ucx-builder
ARG UCX_REPO
ARG UCX_REF

RUN git init /src/ucx && \
    cd /src/ucx && \
    git remote add origin "${UCX_REPO}" && \
    git fetch --depth 1 origin "${UCX_REF}" && \
    git checkout FETCH_HEAD
WORKDIR /src/ucx
RUN ./autogen.sh && \
    ./contrib/configure-release-mt \
        --enable-shared --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations --enable-cma --enable-devel-headers \
        --with-cuda=${CUDA_HOME} --with-verbs --with-dm --without-gdrcopy \
        --prefix=/opt/ucx && \
    make -j$(nproc) && \
    make install

# ------------------------------------------------------------------------------
# mooncake-builder: clone Mooncake at pinned tag, build to /opt/mooncake.
# ------------------------------------------------------------------------------
FROM base-env AS mooncake-builder
ARG MOONCAKE_REPO
ARG MOONCAKE_REF

# mooncake links against UCX at build time, so make our UCX install visible.
COPY --from=ucx-builder /opt/ucx /opt/ucx
ENV PATH=/opt/ucx/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/ucx/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/opt/ucx/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH}
ENV PKG_CONFIG_PATH=""
ENV PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig:${PKG_CONFIG_PATH}

RUN git clone --depth 1 --branch "${MOONCAKE_REF}" "${MOONCAKE_REPO}" /src/mooncake
WORKDIR /src/mooncake
# dependencies.sh installs system packages and fetches/builds mooncake deps.
# Flags may need adjustment if Ubuntu 24.04 / CUDA 13.3 incompatibilities surface.
# Create libcuda.so.1 symlink from stubs so the linker resolves CUDA driver symbols.
RUN ln -sf ${CUDA_HOME}/lib64/stubs/libcuda.so ${CUDA_HOME}/lib64/libcuda.so.1 && \
    bash dependencies.sh && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/opt/mooncake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_RPATH=${CUDA_HOME}/lib64 \
        -DUSE_CUDA=ON && \
    make -j$(nproc) && \
    make install

# ------------------------------------------------------------------------------
# nixl-builder: clone nixl at pinned ref, build C++ libs/plugins to
# /opt/nvidia/nvda_nixl.
# ------------------------------------------------------------------------------
FROM base-env AS nixl-builder
ARG NIXL_REPO
ARG NIXL_REF

COPY --from=ucx-builder /opt/ucx /opt/ucx
COPY --from=mooncake-builder /opt/mooncake /opt/mooncake

ENV PATH=/opt/ucx/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/ucx/lib:/opt/mooncake/lib:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/opt/ucx/lib:/opt/mooncake/lib:${LIBRARY_PATH}
ENV CPLUS_INCLUDE_PATH=/opt/mooncake/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/opt/mooncake/include:${C_INCLUDE_PATH}
ENV PKG_CONFIG_PATH=""
ENV PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig:${PKG_CONFIG_PATH}

RUN python3.12 -m pip install --no-cache-dir \
    meson ninja pybind11 patchelf pyyaml

RUN git init /src/nixl && \
    cd /src/nixl && \
    git remote add origin "${NIXL_REPO}" && \
    git fetch --depth 1 origin "${NIXL_REF}" && \
    git checkout FETCH_HEAD
WORKDIR /src/nixl
RUN meson setup build \
        -Denable_plugins=UCX,LIBFABRIC,POSIX,OBJ,MOONCAKE \
        -Dnixl_cuda_arch_list=121 \
        -Dbuild_tests=false \
        -Dbuild_examples=false \
        -Ducx_path=/opt/ucx \
        -Dinstall_headers=true \
        -Dwheel_variant=cu13 && \
    ninja -C build && \
    ninja -C build install

# ------------------------------------------------------------------------------
# runtime: install artifact/runtime deps; configure GB10 default environment.
# ------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python3-pip \
        libibverbs-dev librdmacm-dev \
        libfabric-dev \
        rdma-core \
        libnuma-dev \
        libhwloc-dev \
        libasio-dev \
        libgflags-dev \
        libprotobuf-dev protobuf-compiler \
        libgoogle-glog-dev \
        libssl-dev \
        libuv1-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir numpy

COPY --from=ucx-builder /opt/ucx /opt/ucx
COPY --from=mooncake-builder /opt/mooncake /opt/mooncake
COPY --from=nixl-builder /opt/nvidia/nvda_nixl /opt/nvidia/nvda_nixl
COPY smoke-test.py /opt/nixl/smoke-test.py

# Prefer nixl/UCX/Mooncake libraries; expose nixl python package.
ENV PATH=/opt/ucx/bin:/opt/nvidia/nvda_nixl/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib:/opt/ucx/lib:/opt/mooncake/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=""
ENV PYTHONPATH=/opt/nvidia/nvda_nixl/lib/python3/dist-packages:${PYTHONPATH}

# GB10 runtime defaults: register GPU memory as DRAM_SEG so ConnectX-7 verbs
# treats it as host RDMA memory; cuda-copy is the safety fallback if anything
# is registered as VRAM_SEG. Disable the libfabric CUDA address workaround.
ENV UCX_TLS=cuda-copy,rc,sm,self,tcp
ENV NIXL_DISABLE_CUDA_ADDR_WA=1

WORKDIR /workspace

# ------------------------------------------------------------------------------
# wheel-builder: from nixl-builder, add Python build tools + torch 2.13.
# ------------------------------------------------------------------------------
FROM nixl-builder AS wheel-builder

ARG PYTHON_VERSION
RUN python3.12 -m pip install --no-cache-dir \
    build meson-python patchelf pybind11 pyyaml types-PyYAML pytest setuptools wheel && \
    python3.12 -m pip install --no-cache-dir torch==2.13.*

WORKDIR /src/nixl

# Patch pyproject.toml only inside this image; the source tree stays clean.
RUN cp pyproject.toml pyproject.toml.orig && \
    python3.12 - <<'PY'
import re

with open('pyproject.toml') as f:
    text = f.read()

text = re.sub(r'^name = "nixl-cu12"$', 'name = "nixl-cu13"', text, flags=re.M)
text = re.sub(r'torch==2\.11\.\*', 'torch==2.13.*', text)

# Replace the meson-python args section so the wheel build uses the same
# plugin/arch options as the C++ runtime build.
text = re.sub(
    r'\[tool\.meson-python\.args\]\nsetup = \["-Dinstall_headers=false"\]',
    '[tool.meson-python.args]\nsetup = [\n'
    '    "-Denable_plugins=UCX,LIBFABRIC,POSIX,OBJ,MOONCAKE",\n'
    '    "-Dnixl_cuda_arch_list=121",\n'
    '    "-Dbuild_tests=false",\n'
    '    "-Dbuild_examples=false",\n'
    '    "-Ducx_path=/opt/ucx",\n'
    '    "-Dinstall_headers=false",\n'
    '    "-Dwheel_variant=cu13"\n'
    ']',
    text,
)

with open('pyproject.toml', 'w') as f:
    f.write(text)
PY
RUN diff -u pyproject.toml.orig pyproject.toml || true && \
    rm pyproject.toml.orig

RUN python3.12 -m build --wheel --no-isolation -o /dist