# Build environment for dgx-spark-wheels.
#
# Pinned to the oldest glibc we support (Ubuntu 22.04 / glibc 2.35) so every
# wheel built here imports cleanly in 22.04 containers, 24.04 containers, and
# on bare hosts. See README "glibc: the tag that isn't in the wheel filename".
#
# This image only provides the toolchain (CUDA devel + Python 3.12 + build
# deps). torch and per-package runtime deps are installed into a per-build
# venv by scripts/build/<pkg>.sh (torch must come from the cu130 index and is
# version-coupled to the wheels, so it is NOT baked in here).

FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04

# Kitware apt repo for a modern CMake (Ubuntu 22.04 ships cmake 3.22, but
# onnxruntime requires >= 3.28).
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common gnupg git ninja-build ca-certificates wget && \
    wget -qO- https://apt.kitware.com/kitware-archive.sh | bash -e /dev/stdin && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*

# Python 3.12 via deadsnakes (cp312 wheels only).
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Make python3 resolve to 3.12 so build scripts get cp312 by default.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir ninja packaging wheel setuptools psutil

# CUDA toolchain is already in the base image; expose it on PATH.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

WORKDIR /src
