# Build environment for dgx-spark-wheels.
#
# Pinned to Ubuntu 24.04 / glibc 2.39 — the oldest glibc this index supports.
# Every wheel built here imports cleanly in 24.04+ containers and on bare hosts.
#
# Toolchain: CUDA 13.3.1 devel + Python 3.12 + cuSPARSELt + cuFile (GDS) +
# PyTorch/Triton build deps. torch and per-package runtime deps are installed
# into a per-build venv by scripts/build/<pkg>.sh (torch is version-coupled
# to the wheels, so it is NOT baked in here).
#
# Tag published to GHCR: build-env:24.04

FROM nvcr.io/nvidia/cuda:13.3.1-devel-ubuntu24.04

# Ubuntu 24.04 ships cmake 3.28.3 (onnxruntime requires >= 3.28) and
# Python 3.12 natively — no Kitware PPA, no deadsnakes PPA needed.
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        ca-certificates \
        curl \
        wget \
        zlib1g-dev \
        pkg-config \
        software-properties-common \
        gnupg \
        python3.12 python3.12-venv python3.12-dev \
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

RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir ninja packaging wheel setuptools psutil pyyaml

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
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nvidia-gds || \
    echo "[WARN] nvidia-gds not found in repos — cuFile will be disabled" && \
    rm -rf /var/lib/apt/lists/*
ENV CUFILE_HOME=/usr/local/cuda

# --- NCCL (system, from NGC base or apt) ---
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libnccl2 libnccl-dev || \
    echo "[WARN] libnccl2/libnccl-dev not found — NCCL will fall back to PyTorch-bundled" && \
    rm -rf /var/lib/apt/lists/*

# CUDA toolchain is already in the base image; expose it on PATH.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

WORKDIR /src