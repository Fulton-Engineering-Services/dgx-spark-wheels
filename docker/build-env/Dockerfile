# Build environment for dgx-spark-wheels.
#
# Pinned to Ubuntu 24.04 / glibc 2.39 — the oldest glibc this index supports.
# Every wheel built here imports cleanly in 24.04+ containers and on bare hosts.
#
# Toolchain: CUDA devel + Python 3.12 + cuSPARSELt + cuFile (GDS) +
# PyTorch/Triton build deps. torch and per-package runtime deps are installed
# into a per-build venv by scripts/build/<pkg>.sh (torch is version-coupled
# to the wheels, so it is NOT baked in here).
#
# Parametrized for the two CUDA trains:
#   CUDA_IMAGE_TAG  — the CUDA devel image's version suffix (13.3.1 | 13.0.3)
#   CUDA_PKG_SUFFIX — the version-pinned apt package suffix (13-3 | 13-0)
# The minor-unversioned NVIDIA packages (libcudnn9-dev-cuda-13, cusparselt-cuda-13,
# libnccl2/libnccl-dev) resolve against whichever base-image CUDA is selected.
#
# Tags published to GHCR (one per variant):
#   build-env:24.04-cu13.3  (CUDA 13.3.1)  — the default build args
#   build-env:24.04-cu13.0  (CUDA 13.0.3)

ARG CUDA_IMAGE_TAG=13.3.1
FROM nvcr.io/nvidia/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu24.04

ARG CUDA_PKG_SUFFIX=13-3

# Ubuntu 24.04 ships cmake 3.28.3 (onnxruntime requires >= 3.28) and
# Python 3.12 natively — no Kitware PPA, no deadsnakes PPA needed.
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        ccache \
        git \
        ca-certificates \
        curl \
        wget \
        zlib1g-dev \
        pkg-config \
        software-properties-common \
        gnupg \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        libopenblas-dev \
        libomp-dev \
        libopenmpi-dev mpi-default-bin \
        libuv1-dev \
        libssl-dev \
        libgflags-dev \
        libprotobuf-dev protobuf-compiler \
        libgoogle-glog-dev \
        libjpeg-dev libpng-dev \
        libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libavutil-dev libswresample-dev libswscale-dev && \
    rm -rf /var/lib/apt/lists/*

# Make python3 resolve to 3.12 so build scripts get cp312 by default.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Ubuntu 24.04 strips ensurepip from the stdlib and enforces PEP 668
# (externally-managed-environment) on system-wide pip installs. python3-pip
# (apt, above) provides a working `python3.12 -m pip`; PIP_BREAK_SYSTEM_PACKAGES
# opts this build container out of the PEP 668 guard. Per-build venvs are
# unaffected either way.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3.12 -m pip install --no-cache-dir ninja packaging wheel setuptools psutil pyyaml

# --- cuSPARSELt 0.8.1 (structured sparse GEMM) ---
# Installs NVIDIA's local repo for Ubuntu 24.04 arm64, then the CUDA 13 dev package.
ARG CUSPARSELT_VERSION=0.8.1
ARG CUSPARSELT_DEB=cusparselt-local-repo-ubuntu2404-${CUSPARSELT_VERSION}_${CUSPARSELT_VERSION}-1_arm64.deb
ARG CUSPARSELT_URL=https://developer.download.nvidia.com/compute/cusparselt/${CUSPARSELT_VERSION}/local_installers/${CUSPARSELT_DEB}
RUN wget -q "${CUSPARSELT_URL}" -O "/tmp/${CUSPARSELT_DEB}" && \
    dpkg -i "/tmp/${CUSPARSELT_DEB}" && \
    rm "/tmp/${CUSPARSELT_DEB}" && \
    cp /var/cusparselt-local-repo-ubuntu2404-${CUSPARSELT_VERSION}/cusparselt-*-keyring.gpg /usr/share/keyrings/ || true && \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cusparselt-cuda-13 && \
    rm -rf /var/lib/apt/lists/*
ENV CUSPARSELT_HOME=/usr/local/cuda

# --- cuFile / GPUDirect Storage ---
# The NGC base image already configures the NVIDIA CUDA network repo
# (cuda-keyring), so the version-pinned cuFile dev package
# (libcufile-dev-${CUDA_PKG_SUFFIX}: 13-3 for CUDA 13.3, 13-0 for CUDA 13.0)
# resolves directly. It installs the userspace cuFile header + shared/static
# libs under /usr/local/cuda-<ver>/targets/sbsa-linux. Do NOT install
# `nvidia-gds`: that metapackage drags in nvidia-fs-dkms (the nvidia_fs.ko
# kernel module), which cannot build inside a container without kernel
# headers — the kernel module lives on the DGX host (DGX-OS). A build
# container only needs the header + .so to compile/link PyTorch with
# USE_CUFILE=1. Hard-fail rather than silently ship a torch without GDS.
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libcufile-dev-${CUDA_PKG_SUFFIX} && \
    rm -rf /var/lib/apt/lists/*
ENV CUFILE_HOME=/usr/local/cuda

# --- cuDNN 9 (deep learning primitives) ---
# Resolves from the NVIDIA CUDA network repo (cuda-keyring, already configured
# by the NGC devel base): provides cudnn.h + libcudnn.so. PyTorch auto-enables
# USE_CUDNN when the header/lib are found. cuDSS is deliberately NOT installed:
# cuDSS 0.7.0.20's API (cudssDataType_t etc.) is incompatible with PyTorch
# 2.13.0's cuDSS integration (see scripts/build/torch.sh).
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libcudnn9-dev-cuda-13 && \
    rm -rf /var/lib/apt/lists/*

# --- NCCL (system, from NGC base or apt) ---
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libnccl2 libnccl-dev || \
    echo "[WARN] libnccl2/libnccl-dev not found — NCCL will fall back to PyTorch-bundled" && \
    rm -rf /var/lib/apt/lists/*

# CUDA toolchain is already in the base image; expose it on PATH.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# ccache: PyTorch's cmake auto-enables it when the binary is on PATH. Point the
# cache at /tmp/ccache — /github/home is not writable in the GH container job.
# For cross-run persistence (iterative rebuilds), the self-hosted runner should
# bind-mount a host dir over this path in build-wheel.yml's container options.
ENV CCACHE_DIR=/tmp/ccache

WORKDIR /src