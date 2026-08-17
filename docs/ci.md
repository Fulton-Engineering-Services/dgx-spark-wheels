# CI

All wheel builds run on a **self-hosted GB10 runner** (`[self-hosted, gb10]`)
inside the build-env container, because every package here either needs a live
GPU at build time (sageattn3, nunchaku probe compute capability in `setup.py`)
or exceeds the free `ubuntu-24.04-arm` runner's 4 vCPU / 16 GB RAM / 14 GB
disk envelope (flash-attn, onnxruntime-gpu, and the from-source torch build).
The index generator and lint job are the only things that run on the free
runner.

## Runner prerequisites

The GB10 box must have:

- **≥200 GB free disk** (PyTorch's source tree + build tree + submodules +
  wheel output is ~100-150 GB; don't starve it).
- **≥32 GB RAM** (GB10's unified memory makes this the same pool the GPU
  uses — see the README "Unified memory" note).
- **Docker** with `nvidia-container-toolkit` configured for GPU containers.
- A **12-hour job timeout** is set on `build-wheel.yml`'s `build` job
  (`timeout-minutes: 720`) because the from-source torch build can run many
  hours at `MAX_JOBS=$(nproc)`.

## Registering the self-hosted GB10 runner

This is a **one-time, user-run prerequisite** — it must be done on the GB10
box itself and cannot be performed from CI or from another host. Do this after
the repo exists on GitHub (the runner is registered to a specific repo URL).

1. **Fetch a registration token** (token is short-lived; generate it fresh).
   Note the `-X POST` — the registration-token endpoint requires POST;
   `gh api` defaults to GET, which 404s:
   ```bash
   gh api -X POST repos/Fulton-Engineering-Services/dgx-spark-wheels/actions/runners/registration-token --jq .token
   ```

2. **On the GB10 box**, download the arm64 GitHub Actions runner and configure
   it with the `gb10` label (use `--ephemeral` so each job gets a clean
   workspace; install as a systemd service for auto-restart):
   ```bash
   mkdir -p ~/actions-runner && cd ~/actions-runner
   # Download the latest linux/arm64 runner tarball from
   # https://github.com/actions/runner/releases (pick the linux-arm64 asset).
   tar xzf actions-runner-linux-arm64-*.tar.gz
   ./config.sh --url https://github.com/Fulton-Engineering-Services/dgx-spark-wheels \
                --token <TOKEN_FROM_STEP_1> \
                --labels gb10 \
                --ephemeral
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

3. **Verify** the runner is online and labelled:
   ```bash
   gh api repos/Fulton-Engineering-Services/dgx-spark-wheels/actions/runners --jq '.runners[] | "\(.name) \(.status) labels=\(.labels[].name)"'
   ```
   Expect a runner with `self-hosted` + `gb10` labels, status `online`.

### Notes on the runner

- **GPU access is granted by the job, not the runner.** `build-wheel.yml`
  sets `container.options: --gpus all`, which Docker applies when it launches
  the build-env container. The host runner process itself does not need to
  hold the GPU; Docker device access handles it.
- **`--ulimit memlock=-1`** is also set on the container so CUDA/NCCL can pin
  memory (relevant for nunchaku's RDMA-capable kernels). No `CAP_IPC_LOCK`
  capability is required.
- The runner needs **Docker** installed and able to run GPU containers
  (`nvidia-container-toolkit` configured). The build-env image is pulled from
  GHCR on first use; subsequent builds use the cached layer.

## Build order for the first release

Strictly, in this order (each gated on the previous), for **both CUDA trains**
(`cu13.3` and `cu13.0`) in parallel (`max-parallel: 2` — two GB10 runners now
exist, so both trains run simultaneously):

1. **Create the four forks** (`pytorch`, `audio`, `vision`, `triton`) under
   `Fulton-Engineering-Services` with the `cuda13.3-aarch64-gb10` branch,
   apply the `dgx_spark_config/pytorch/pytorch.patch` sm_121 patch to the
   pytorch fork, then fill in each `fork_ref` (and the triton version) in
   `packages.json` with the resulting HEAD SHAs.

2. **`build-env-image.yml`** — builds + pushes both variant images
   (`build-env:24.04-cu13.3` and `build-env:24.04-cu13.0`) to GHCR. Trigger:
   `gh workflow run build-env-image.yml`. No GPU needed for the image build,
   but it runs on the GB10 runner (the CUDA-devel base is multi-GB and blows
   the free runner's 14 GB disk). The two images build in parallel.

3. **`build-all.yml`** — the ordered full-index rebuild across both CUDA
   trains. Trigger: `gh workflow run build-all.yml`. This orchestrates the
   whole dependency chain in one run, with a two-dimensional matrix
   (`cuda × package`) at each phase:
   `triton` → `torch` → `torchaudio` + `torchvision` → `flash-attn` +
   `sageattention` + `sageattn3` + `nunchaku` + `onnxruntime-gpu` +
   `flashinfer-python` + `mamba-ssm` + `causal-conv1d`.
   Each stage reuses `build-wheel.yml` via `workflow_call`, passing
   `cuda`, `torch_wheel`-cu`<variant>`, and `triton_wheel`-cu`<variant>`
   artifact names so torch-linked wheels compile against the from-source
   torch of the same CUDA train.

4. **`publish-index.yml`** — regenerates the PEP 503 index from `packages.json`
   (one entry per package × variant) and deploys to GitHub Pages. Trigger:
   `gh workflow run publish-index.yml`. Runs on the free `ubuntu-24.04-arm`
   runner (just generates HTML).

### Manual single-package builds

`build-wheel.yml` can also be triggered directly, for one package in one CUDA
train:

```bash
gh workflow run build-wheel.yml -f package=flash-attn -f cuda=13.3
gh workflow run build-wheel.yml -f package=flash-attn -f cuda=13.0
```

`cuda` defaults to `13.3` when omitted. Omit `torch_wheel` / `triton_wheel`
and the build falls back to PyPI torch (`common.sh`). This is fine for a
one-off rebuild before the from-source torch wheel is published, but the
resulting wheel will NOT carry the from-source-torch ABI — prefer
`build-all.yml` for anything you intend to publish.

## Which packages need the GB10 runner (and why)

| Package | GB10 runner? | Reason |
|---|---|---|
| `triton` | yes | downloads/compiles pinned LLVM; exceeds free runner disk |
| `torch` | yes | from-source build, hours on 20 cores, ~100 GB build tree |
| `torchaudio` | yes | needs the from-source torch wheel; CUDA extensions |
| `torchvision` | yes | needs the from-source torch wheel; CUDA extensions |
| `flash-attn` | yes | `MAX_JOBS=4` cap; hdim256 backward kernels OOM above ~8 |
| `sageattention` | yes | small/fast, but kept on GB10 for simplicity |
| `sageattn3` | yes | `setup.py` probes live GPU compute capability → `sm_121a` |
| `nunchaku` | yes | `setup.py` probes live GPU → `sm_121a`; needs `--gpus all` |
| `onnxruntime-gpu` | yes | ~90 min on 20 cores; large build tree exceeds 14 GB disk |
| `flashinfer-python` | yes | JIT package (no nvcc); kept on GB10 for simplicity; needs `--gpus all` at verify time |
| `mamba-ssm` | yes | compiled `selective_scan_cuda` ext; needs `--gpus all` at verify time |
| `causal-conv1d` | yes | compiled `causal_conv1d_cuda` ext; needs `--gpus all` at verify time |

A future hardening (per the project strategy) is to patch `setup.py` in
sageattn3/nunchaku to accept an explicit `TORCH_CUDA_ARCH_LIST` override, which
would remove the live-GPU requirement and let those build on a hosted runner —
but the disk constraint for flash-attn/onnxruntime/torch remains.