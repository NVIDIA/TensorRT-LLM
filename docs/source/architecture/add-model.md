(add-model)=

# Adding a Model

This document describes how to add a model in TensorRT-LLM.

TensorRT-LLM provides:

- Low-level functions, for example, `concat`, `add`, and `sum`.
- Basic layers, such as, `Linear` and `LayerNorm`.
- High-level layers, such as, `MLP` and `Attention`.

**Steps**

1. Create a model directory in `tensorrt_llm/tensorrt_llm/models`, for example `bloom`.
2. Write a `model.py` with TensorRT-LLM low level functions and basic layers. It's optional to use high level layers.
