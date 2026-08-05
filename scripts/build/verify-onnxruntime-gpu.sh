#!/usr/bin/env bash
# verify-onnxruntime-gpu.sh — smoke-test onnxruntime with the CUDA execution
# provider: build a tiny ONNX model in-memory and run a real CUDA kernel through
# it, not just import ort / list providers.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_pkg onnxruntime-gpu
. "$VENV/bin/activate"
pip install "$DIST_DIR/$PKG_WHEEL" onnx numpy

python3 - <<'PY'
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

assert ort.get_device() == "GPU", f"ort device is {ort.get_device()}"
providers = ort.get_available_providers()
assert "CUDAExecutionProvider" in providers, f"CUDAExecutionProvider missing: {providers}"

# Build a minimal model: y = x + c (GPU-eligible Add kernel).
c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])
c_tensor = helper.make_tensor("c", TensorProto.FLOAT, [3], c.tolist())
node = helper.make_node("Add", ["x", "c"], ["y"])
graph = helper.make_graph([node], "add", [x], [y], [c_tensor])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)

sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CUDAExecutionProvider"],
)
used = sess.get_providers()
assert "CUDAExecutionProvider" in used, f"session did not pick CUDA EP: {used}"

out = sess.run(["y"], {"x": np.array([10.0, 20.0, 30.0], dtype=np.float32)})[0]
np.testing.assert_allclose(out, np.array([11.0, 22.0, 33.0], dtype=np.float32))
print("onnxruntime-gpu CUDA kernel launch OK:", out.tolist())
PY
