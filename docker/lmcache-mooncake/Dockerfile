# syntax=docker/dockerfile:1
#
# Dockerfile for the LMCache + Mooncake-native wheel (GB10/sm_121).
#
# Produces an `lmcache` wheel that additionally carries the optional native
# extension `lmcache.lmcache_mooncake`, required by LMCache MP-mode's
# `mooncake_store` L2 adapter:
#   lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py ->
#     from lmcache.lmcache_mooncake import L1RegistrationConfig, LMCacheMooncakeClient
#
# Why base on the nixl-wheel-builder image: it ALREADY builds the Mooncake
# Store SDK from source to /opt/mooncake (libmooncake_store.so) and carries the
# full apt toolchain + python build deps + torch 2.13. The published
# mooncake-transfer-engine *wheel* ships no headers and no linkable
# libmooncake_store, so the extension must compile against a from-source SDK --
# reusing the one the nixl builder already produces avoids a redundant ~40 min
# Mooncake rebuild. Only the store HEADERS are taken fresh (cloned at the same
# pinned tag), since the builder image does not retain the Mooncake source tree.
#
# Compile-verified: `-fsyntax-only` of connector.cpp against exactly these
# include roots returns 0 errors in this image.
#
# Published tag (GHCR, dgx-spark-wheels namespace):
#   lmcache-mooncake-wheel-builder:24.04-cu13.3-sm121

ARG BASE_IMAGE=ghcr.io/fulton-engineering-services/dgx-spark-wheels/nixl-wheel-builder:24.04-cu13.3-sm121
FROM ${BASE_IMAGE}

# Source refs -- pinned for reproducibility. Override with --build-arg.
ARG MOONCAKE_REPO=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_REF=v0.3.13.post1
ARG LMCACHE_REPO=https://github.com/Fulton-Engineering-Services/lmcache.git
ARG LMCACHE_REF=ca2f474412e3b881c2f39e62561414c9676653dc
ARG LMCACHE_VERSION=0.5.5rc1

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
WORKDIR /src

# The native ext's connector.cpp includes Mooncake's store headers, which pull
# boost (lockfree/queue), jsoncpp, unwinding, jemalloc, msgpack, zmq, grpc,
# yaml-cpp, curl, hiredis, uring, zstd, xxhash. The nixl-wheel-builder base
# image does not carry these *-dev packages (it builds nixl, not the store
# client), so install them here.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libboost-all-dev libjsoncpp-dev libunwind-dev libjemalloc-dev \
        libmsgpack-dev libzmq3-dev libzstd-dev libxxhash-dev \
        libgrpc++-dev libgrpc-dev libprotobuf-dev protobuf-compiler-grpc \
        libyaml-cpp-dev libcurl4-openssl-dev libhiredis-dev liburing-dev && \
    rm -rf /var/lib/apt/lists/*

# Store headers (with submodules: extern/yalantinglibs carries iguana + frozen).
RUN git clone --depth 1 --branch "${MOONCAKE_REF}" "${MOONCAKE_REPO}" /src/mooncake && \
    cd /src/mooncake && \
    git submodule update --init --recursive --depth 1

# Include roots the ext's connector.cpp needs (verified: 0 errors).
ENV MOONCAKE_INCLUDE_DIR="/src/mooncake/mooncake-store/include;/src/mooncake/mooncake-store/include/cachelib_memory_allocator;/src/mooncake/mooncake-store/include/cachelib_memory_allocator/include;/src/mooncake/mooncake-store/include/cachelib_memory_allocator/fake_include;/src/mooncake/mooncake-common/include;/src/mooncake/mooncake-transfer-engine/include;/src/mooncake/extern/yalantinglibs/include;/src/mooncake/extern/yalantinglibs/include/ylt/standalone;/src/mooncake/extern/yalantinglibs/include/ylt/thirdparty"
ENV MOONCAKE_LIB_DIR=/opt/mooncake/lib
ENV BUILD_MOONCAKE=1
ENV BUILD_WITH_MOONCAKE=1
ENV TORCH_CUDA_ARCH_LIST=12.1
ENV LMCACHE_CUDA_MAJOR=13
ENV ENABLE_CXX11_ABI=1
ENV LD_LIBRARY_PATH=/opt/mooncake/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=/opt/mooncake/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH}

RUN python3.12 -m pip install --no-cache-dir \
    build wheel setuptools setuptools-scm ninja packaging psutil numpy pyyaml patchelf

RUN git clone --depth 1 "${LMCACHE_REPO}" /src/lmcache && \
    cd /src/lmcache && \
    (git fetch --depth 1 origin "${LMCACHE_REF}" && git checkout "${LMCACHE_REF}") || true
WORKDIR /src/lmcache

# SETUPTOOLS_SCM_PRETEND_VERSION pins the version (setuptools_scm needs git tags).
RUN rm -rf /dist && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${LMCACHE_VERSION}" python3.12 setup.py bdist_wheel -d /dist && \
    python3.12 -c "import zipfile,glob,sys; w=sorted(glob.glob('/dist/lmcache-*.whl'))[-1]; z=zipfile.ZipFile(w); ok=any('lmcache_mooncake' in n for n in z.namelist()); print('WHEEL:', w); print('HAS_MOONCAKE_EXT:', ok); sys.exit(0 if ok else 1)"