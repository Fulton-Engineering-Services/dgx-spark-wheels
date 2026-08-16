# Contributing

## Adding a new package to the index

1. Fork the upstream repository to your own account.
2. Pin a known-good commit/tag on a dedicated `cuda13-aarch64-gb10` branch
   (or your platform's equivalent). This protects against upstream
   force-pushes and tag deletions — it does not imply source patches were
   needed; if none were, say so.
3. Build inside the pinned build container
   (`ghcr.io/Fulton-Engineering-Services/dgx-spark-wheels/build-env:24.04`, or a newer tag if the
   project's floor has moved — see [`docs/build-environment.md`](docs/build-environment.md)
   for why the container, not the bare host, matters).
4. **Verify with a real CUDA kernel launch**, not just `import package`. A
   wheel that imports but never executes a kernel can hide an ABI mismatch
   that only surfaces mid-inference.
5. Add an entry to [`packages.json`](packages.json) with the fork URL,
   pinned ref, upstream license, and any build notes a future rebuilder
   needs (resource ceilings, required env vars, GPU-probing behavior at
   build time, etc).
6. Add the upstream license to
   [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md).
7. Open a PR. CI will regenerate the index from `packages.json` and publish
   it via GitHub Pages once merged.

## Reporting a broken wheel

Open an issue with the output of `scripts/spark-doctor` plus the exact
`pip install` command and error. Wheel filenames encode `cp312-linux_aarch64`
but not CUDA version or glibc version, so "it doesn't import" issues are
almost always an ABI mismatch the doctor script will surface immediately.

## Design principles

- **Every wheel is fork-and-pinned.** No `pip install git+https://...` at
  an unpinned ref — reproducibility for anyone rebuilding six months from
  now matters more than tracking upstream HEAD.
- **Build in the oldest glibc you support, never the bare host.** See the
  glibc gotcha in the README.
- **Never guess an ABI tag.** If a wheel's true floor is narrower than what
  its filename implies (a real risk with `cp312-linux_aarch64`, which says
  nothing about CUDA or glibc), say so explicitly in `packages.json` rather
  than letting `pip` assume compatibility it can't verify.
- **Sign everything.** All wheels are built in CI (or a self-hosted GB10
  runner registered to this repo) and attested with
  `actions/attest-build-provenance`, so anyone can verify a wheel was built
  from the exact pinned source this repo claims, not tampered with in
  transit.
