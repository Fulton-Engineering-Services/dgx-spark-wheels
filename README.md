# dgx-spark-wheels

A `pip`-installable package index of prebuilt Python wheels for
**NVIDIA GB10** (the chip in the DGX Spark and its OEM variants — ASUS
Ascent GX10, and others) — specifically for packages that publish **no**
upstream wheel matching this platform's ABI: `linux_aarch64`, CUDA 13.3,
`torch==2.13.0+cu133` (built from source), `cp312` (Python 3.12), compute
capability `(12, 1)` / `sm_121`.

```bash
pip install triton torch torchaudio torchvision flash-attn sageattention nunchaku onnxruntime-gpu \
    --extra-index-url https://Fulton-Engineering-Services.github.io/dgx-spark-wheels/simple/
```

or with `uv`:

```bash
uv pip install torch flash-attn --extra-index-url https://Fulton-Engineering-Services.github.io/dgx-spark-wheels/simple/
```

## Why this exists

GB10 reports `sm_121`, newer than what most ML packages' published wheels
target. As of this writing, the community consensus on packages like
`flash-attn` is "no wheel exists, build from source, or fall back to SDPA."
That's true for the *default* build target, but incomplete: a source build
targeting `sm_120` (binary-compatible via CUDA forward-compat) works, is
kernel-launch-verified on real GB10 hardware, and can simply be published as
a wheel so nobody else has to spend an afternoon rediscovering it.

Every wheel in this index is:

1. **Fork-and-pinned.** The upstream repo is forked, pinned to a known-good
   commit/tag on a dedicated branch (protects against upstream force-pushes
   or tag deletions — not because source patches were needed), then built
   from source. See [`packages.json`](packages.json) for the exact fork +
   ref behind every published wheel.
2. **Built against a known, reproducible ABI** — see
   [`docs/build-environment.md`](docs/build-environment.md).
3. **Verified with a real CUDA kernel launch**, not just a bare `import`,
   before being published.
4. **Signed with build provenance** via
   [`actions/attest-build-provenance`](https://github.com/actions/attest-build-provenance) —
   verify with `gh attestation verify <wheel> -R Fulton-Engineering-Services/dgx-spark-wheels`.

## Current packages

| Package | Version | Arch target | Upstream wheel? | Fork |
|---|---|---|---|---|
| `triton` | 3.4.0+cu13.3torch2.13.glibc239 | aarch64 (CUDA PTX JIT) | No aarch64 wheel | [fork](https://github.com/Fulton-Engineering-Services/triton) `cuda13.3-aarch64-gb10` |
| `torch` | 2.13.0+cu13.3.glibc239 | `sm_120`+`sm_121` PTX (from-source) | No | [fork](https://github.com/Fulton-Engineering-Services/pytorch) `cuda13.3-aarch64-gb10` |
| `torchaudio` | 2.13.0+cu13.3torch2.13.glibc239 | `sm_120`+`sm_121` PTX | No | [fork](https://github.com/Fulton-Engineering-Services/audio) `cuda13.3-aarch64-gb10` |
| `torchvision` | 0.28.0+cu13.3torch2.13.glibc239 | `sm_120`+`sm_121` PTX | No | [fork](https://github.com/Fulton-Engineering-Services/vision) `cuda13.3-aarch64-gb10` |
| `flash-attn` | 2.8.3.post1+cu13.3torch2.13.glibc239 | `sm_120` (PTX-forward-compat on `sm_121`) | No aarch64 wheel published | [fork](https://github.com/Fulton-Engineering-Services/flash-attention) `cuda13-aarch64-gb10` |
| `sageattention` | 2.2.0+cu13.3torch2.13.glibc239 | `sm_121` | No | [fork](https://github.com/Fulton-Engineering-Services/SageAttention) `cuda13-aarch64-gb10` |
| `sageattn3` | 1.0.0+cu13.3torch2.13.glibc239 | `sm_121a` (auto-detected) | No | same fork, `sageattention3_blackwell/` subdir |
| `nunchaku` | 1.2.1+cu13.3torch2.13.glibc239 | `sm_121a` (auto-detected) | No | [fork](https://github.com/Fulton-Engineering-Services/nunchaku) `cuda13-aarch64-gb10` |
| `onnxruntime-gpu` | 1.22.0+cu13.3.glibc239 | generic (no arch pin) | No aarch64+CUDA13 wheel | [fork](https://github.com/Fulton-Engineering-Services/onnxruntime) `cuda13-aarch64-gb10` |

See [`packages.json`](packages.json) for the machine-readable manifest (used
to generate the index) and [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md)
for the license each package is redistributed under.

> **Why the `+cu13.3torch2.13.glibc239` version suffix?** The wheel filename
> tag `cp312-linux_aarch64` says nothing about CUDA version, torch build, or
> glibc floor — so a naive `pip install` can silently pull a wheel that
> imports but launches the wrong kernels. The local-version segment encodes
> the real ABI the wheel was built against, making the filename
> self-diagnosing. Release tags use the public version only (no `+local`),
> so `pip install flash-attn` still resolves to the latest; the suffix lives
> in the wheel metadata where `pip show flash-attn` reports it.

## Before you install: run the doctor

```bash
curl -fsSL https://raw.githubusercontent.com/Fulton-Engineering-Services/dgx-spark-wheels/main/scripts/spark-doctor | bash
```

Prints your architecture, compute capability, CUDA/`nvcc` version, glibc
version, Python version, and installed `torch` build, then tells you which
index entries actually match your environment. Wheel filenames encode
`cp312-linux_aarch64` but **that tag does not capture CUDA version or
glibc version** — see the gotcha below — so a naive `pip install` can
silently pull a wheel that imports but crashes, or doesn't crash and is
silently running the wrong kernels. Run the doctor first.

## Known gotchas (read before rebuilding anything yourself)

### glibc: the tag that isn't in the wheel filename

A wheel's compiled `.so` links against the **glibc of the build host**, not
just its CUDA/torch/Python ABI — and none of `cp312`, `linux_aarch64`, or a
CUDA version in the filename capture that. If you build on a bare
Ubuntu 26.04 host (glibc 2.41) and try to `pip install` the result inside an
Ubuntu 24.04 container (glibc 2.39), you'll hit
`GLIBC_2.41' not found` even though CUDA, torch, and Python versions all
match perfectly.

**Every wheel in this index is built inside a container matching the
oldest glibc we support** (currently Ubuntu 24.04 / glibc 2.39), specifically
to avoid this. If you're building your own variant, do the same — build in
(or targeting) the *oldest* glibc your deployment targets, never the bare
host, and check the real manylinux/glibc floor with `auditwheel show`
rather than trusting the filename.

### `MAX_JOBS`: the OOM you'll hit building `flash-attn`

Building `flash-attn` from source with `MAX_JOBS` set to `nproc` will very
likely OOM on any machine with less than ~64GB free — not from general
`nvcc` load, but specifically from the `flash_bwd_hdim256_*_sm80` backward
kernel translation units, which are enormous individually. Cap `MAX_JOBS` at
4-8 and watch `free -h` during the build. See
[`docs/build-environment.md`](docs/build-environment.md) for the full
per-package build commands and known resource ceilings.

### Unified memory and `nvidia-smi`

`nvidia-smi --query-gpu=memory.total/used/free` returns the literal string
`[N/A]` on GB10's unified-memory architecture. Don't build tooling around
those fields; read `free -h` / `/proc/meminfo` instead, since CPU and GPU
share one physical pool here.

## Building your own wheels / contributing a new package

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Short version: fork upstream, pin
a branch, build inside the pinned container image
(`ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:24.04`), verify with a real
kernel launch, open a PR adding an entry to `packages.json` plus the CI
workflow reference.

## Related reading

- [NVIDIA/Dao-AILab flash-attention#1969](https://github.com/Dao-AILab/flash-attention/issues/1969) — the upstream tracking issue this project's `flash-attn` wheel directly answers.
- [`dgx-spark-field-notes`](https://github.com/Fulton-Engineering-Services/dgx-spark-field-notes) — broader GB10 operational findings (RDMA, CDI/Docker GPU access, node monitoring) that aren't wheel-shaped.
- [`gb10-torch-base`](https://github.com/Fulton-Engineering-Services/gb10-torch-base) — a Docker base image that consumes this index.

## License

Code in this repository (index generator, CI, `spark-doctor`) is
Apache-2.0 — see [LICENSE](LICENSE). The wheels themselves redistribute
third-party compiled code under their **original upstream licenses** — see
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).
