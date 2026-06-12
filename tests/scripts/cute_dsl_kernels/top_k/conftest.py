# Patch onnx.helper.float32_to_bfloat16 stub (onnx 1.21+ removed it,
# but onnx_graphsurgeon 0.5.8 still references it at import time when
# tensorrt_llm gets imported transitively by the kernel tests).
import onnx.helper as _oh

if not hasattr(_oh, "float32_to_bfloat16"):
    _oh.float32_to_bfloat16 = lambda *a, **kw: None
