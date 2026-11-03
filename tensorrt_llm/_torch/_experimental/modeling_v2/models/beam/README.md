# Beam StairCase implementation

This directory implements Beam using the experimental ModelingV2/StairCase architecture:
catalog operators, dense and MoE blocks, a six-layer BF16 component, and a 52-layer
NVFP4-expert target with attention TP4 and MoE TP4/EP1 on four GB300 GPUs.

The full target executes sliding-window and full attention with a BF16 KV cache.
Its checkpoint loader streams caller-provided weights into rank-local storage; no
checkpoint is included. Router projection and selection use FP32; packed NVFP4 expert
weights execute with the native GEMM path. Diagnostic host checks are opt-in through
the shared ModelingV2 validation setting and must agree across ranks.

This integration includes the StairCase implementation and tests, standard serving
registration, and the public GSM8K evaluator. Speculative decoding, FP8 KV cache, disaggregated serving, CUDA
graphs and overlap scheduling are not supported by this extracted target.
