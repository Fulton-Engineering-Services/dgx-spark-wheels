# CI

All wheel builds run on a **self-hosted GB10 runner** (`[self-hosted, gb10]`)
inside the build-env container, because every package here either needs a live
GPU at build time (sageattn3, nunchaku probe compute capability in `setup.py`)
or exceeds the free `ubuntu-24.04-arm` runner's 4 vCPU / 16 GB RAM / 14 GB
disk envelope (flash-attn, onnxruntime-gpu). The index generator and lint job
are the only things that run on the free runner.

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

Strictly, in this order (each gated on the previous):

1. **`build-env-image.yml`** — builds + pushes
   `ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:22.04` to
   GHCR. Trigger: `gh workflow run build-env-image.yml`. No GPU needed for the
   image build, but it runs on the GB10 runner (the CUDA-devel base is
   multi-GB and blows the free runner's 14 GB disk).

2. **`build-wheel.yml`** — once per package (5 runs), via
   `gh workflow run build-wheel.yml -f package=<pkg>`. Each builds inside the
   build-env container, injects the local-version ABI segment, runs a real
   kernel-launch smoke test, attests provenance, and uploads the wheel to a
   Release tagged `<pkg>-v<public-version>`.

3. **`publish-index.yml`** — regenerates the PEP 503 index from `packages.json`
   and deploys to GitHub Pages. Trigger: `gh workflow run publish-index.yml`.
   Runs on the free `ubuntu-24.04-arm` runner (just generates HTML).

## Which packages need the GB10 runner (and why)

| Package | GB10 runner? | Reason |
|---|---|---|
| `flash-attn` | yes | `MAX_JOBS=4` cap; hdim256 backward kernels OOM above ~8 |
| `sageattention` | yes | small/fast, but kept on GB10 for v1 simplicity |
| `sageattn3` | yes | `setup.py` probes live GPU compute capability → `sm_121a` |
| `nunchaku` | yes | `setup.py` probes live GPU → `sm_121a`; needs `--gpus all` |
| `onnxruntime-gpu` | yes | ~90 min on 20 cores; large build tree exceeds 14 GB disk |

A future hardening (per the project strategy) is to patch `setup.py` in
sageattn3/nunchaku to accept an explicit `TORCH_CUDA_ARCH_LIST` override, which
would remove the live-GPU requirement and let those build on a hosted runner —
but the disk constraint for flash-attn/onnxruntime remains.
