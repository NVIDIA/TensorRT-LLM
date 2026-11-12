<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Beam StairCase implementation

This directory implements Beam using the experimental ModelingV2/StairCase architecture:
family-scoped operators, dense and MoE blocks, a six-layer BF16 component, and a 52-layer
NVFP4-expert target with attention TP4 and MoE TP4/EP1 on four GB300 GPUs.

The full target executes sliding-window and full attention with a BF16 KV cache.
Its checkpoint loader streams caller-provided weights into rank-local storage; no
checkpoint is included. Router projection and selection use FP32; packed NVFP4 expert
weights execute with the native GEMM path. Diagnostic host checks are opt-in through
the shared ModelingV2 validation setting and must agree across ranks.

This integration includes the StairCase implementation and tests, standard serving
registration, and the public GSM8K evaluator. Speculative decoding, FP8 KV cache, disaggregated serving, CUDA
graphs and overlap scheduling are not supported by this extracted target.

The named `TestModelingV2BeamFullModelNvfp4Sm103Tp4` accuracy gate exercises the
standard serving path with `modeling_v2: require`. It uses the public pinned first-100
GSM8K protocol; see the deployment guide for staged inputs and manual invocation.

Beam-only operators live under `ops/`, with tests in the matching Beam test tree.
Component-specific cases for shared catalog operators reuse the catalog test drivers,
including TP1 shapes and the per-rank TP4 projection, attention and expert shapes.

## Checkout-only bring-up helpers

The six TP1 bring-up components and their generation controller live with their tests in
`tests/unittest/_torch/modeling_v2/models/beam/_components/`. They are not installed in
the TensorRT-LLM wheel. The runtime does not expose intermediate-tensor capture hooks.
Normal runtime validation and rank-consistency diagnostics remain available.

The TP4 checkpoint layout, expert provider and full-model execution remain in this
library. Public serving integration uses `LLM()` and `trtllm-serve`.

Shared BF16 TP4 layout, loading and component base modules live in `_tp4/`; the
`full_model_nvfp4__sm_103__tp4/` package adds NVFP4 expert storage and execution.
