# How to add a new model

This document describes how to add a new model in TensorRT-LLM.


## Steps

What TensorRT-LLM provides:

- low level functions: concat, add, sum, etc
- basic layers: Linear, LayerNorm, etc
- high level layers: MLP, Attention

What the model developers need to implement:

1. Create a new model directory in `tensorrt_llm/tensorrt_llm/models`, e.g. `bloom`.
2. Write a `model.py` with TensorRT-LLM low level functions and basic layers. It's optional to use high level layers.
