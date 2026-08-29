# Third-party licenses

This repository redistributes compiled binary wheels of the following
third-party projects. Each wheel is built from the pinned fork listed in
[`packages.json`](packages.json); the compiled artifact is redistributed
under the license of the **upstream project it was built from**, not this
repository's own license.

| Package | Upstream project | License | License text |
|---|---|---|---|
| `triton` | [triton-lang/triton](https://github.com/triton-lang/triton) | MIT | [upstream LICENSE](https://github.com/triton-lang/triton/blob/main/LICENSE) |
| `torch` | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | [upstream LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| `torchaudio` | [pytorch/audio](https://github.com/pytorch/audio) | BSD-2-Clause | [upstream LICENSE](https://github.com/pytorch/audio/blob/main/LICENSE) |
| `torchvision` | [pytorch/vision](https://github.com/pytorch/vision) | BSD-3-Clause | [upstream LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE) |
| `flash-attn` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause | [upstream LICENSE](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE) |
| `sageattention` | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | BSD-3-Clause | [upstream LICENSE](https://github.com/thu-ml/SageAttention/blob/main/LICENSE) |
| `sageattn3` | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) (`sageattention3_blackwell/`) | BSD-3-Clause | [upstream LICENSE](https://github.com/thu-ml/SageAttention/blob/main/LICENSE) |
| `nunchaku` | [nunchaku-ai/nunchaku](https://github.com/nunchaku-ai/nunchaku) | Apache-2.0 | [upstream LICENSE](https://github.com/nunchaku-ai/nunchaku/blob/main/LICENSE) |
| `onnxruntime-gpu` | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | [upstream LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| `flashinfer-python` | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Apache-2.0 | [upstream LICENSE](https://github.com/flashinfer-ai/flashinfer/blob/main/LICENSE) |
| `mamba-ssm` | [state-spaces/mamba](https://github.com/state-spaces/mamba) | Apache-2.0 | [upstream LICENSE](https://github.com/state-spaces/mamba/blob/main/LICENSE) |
| `causal-conv1d` | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | BSD-3-Clause | [upstream LICENSE](https://github.com/Dao-AILab/causal-conv1d/blob/main/LICENSE) |
| `flextensor` | [ai-dynamo/flextensor](https://github.com/ai-dynamo/flextensor) | Apache-2.0 | [upstream LICENSE](https://github.com/ai-dynamo/flextensor/blob/main/LICENSE) |
| `lmcache` | [LMCache/LMCache](https://github.com/LMCache/LMCache) | Apache-2.0 | [upstream LICENSE](https://github.com/LMCache/LMCache/blob/dev/LICENSE) |

**Before publishing any wheel, verify the upstream license file at the exact
pinned ref** (not just the current default branch) — licenses can change
between the ref this table was written against and what upstream ships
today. Do not trust this table blindly; re-check it against
`packages.json`'s pinned refs each time a wheel is rebuilt at a newer pin.

If you contribute a new package, add its upstream license here as part of
the PR (see [CONTRIBUTING.md](CONTRIBUTING.md)).
