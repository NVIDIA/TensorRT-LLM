<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Beam StairCase components

This directory implements Beam components using the experimental ModelingV2/StairCase
architecture: family-scoped operators, dense and MoE blocks, a six-layer BF16 component,
and a 52-layer NVFP4-expert component for attention TP4 and MoE TP4/EP1 on four GB300 GPUs.

The full component implements sliding-window and full attention with a BF16 KV cache.
Its loader streams caller-provided weights into rank-local storage; no checkpoint is
included. Router projection and selection use FP32, and packed NVFP4 experts use the
native GEMM path.

These are directly constructed components, not a registered `LLM()` or `trtllm-serve`
target. Engine routing, the engine adapter, checkpoint-loader registration and a named
whole-model accuracy gate belong to the serving integration. Operator certification and
CPU component-contract tests accompany these components; broader component tests are
reviewed separately. Speculative decoding, FP8 KV cache, disaggregated serving, CUDA
graphs and overlap scheduling are outside this component baseline.

Beam-only operators live under `ops/`, with tests in the matching Beam test tree.
Component-specific cases for shared catalog operators reuse the catalog test drivers,
including TP1 shapes and the per-rank TP4 projection, attention and expert shapes.

## Direct component use

The component classes are experimental Python building blocks for isolating a
checkpoint-loading, routing, attention or cache-lifecycle problem without constructing
the serving engine. A caller supplies the configuration, weights and prepared metadata;
they do not download checkpoints or provide a server. Their precision/architecture/TP
suffixes describe the component contract, not registration in the model router.

| Component | Direct-call purpose |
| --- | --- |
| Router and expert | Inspect selection or one expert independently of the full MoE block. |
| Dense, MoE and sparse blocks | Compare one block and its captured intermediates against its declared operator contract. |
| Six-layer BF16 model | Exercise mixed attention, cached prefill/decode and prefix reuse with the component generator. The `small_model_prefill` package includes decoding. |
| Full-depth NVFP4 component | Load rank-local checkpoint slices and run the 52-layer computation through a separately supplied engine adapter. BF16 base modules define the shared non-expert computation and loading transaction. |

Capture dictionaries and `layer_capture` callbacks are opt-in debugging interfaces;
they retain detached copies of intermediate tensors and are not enabled by default.
The component generator is a debugging controller, not an alternative public serving
entry point. End users should use a registered `LLM()` or `trtllm-serve` integration.

The eager dispatch helper preserves ascending-expert accumulation and BF16 rounding
of every weighted route. Grouped/fused expert execution and changes to projection or
position-transform scheduling require their own numerical and performance validation;
component certification alone does not establish equivalence for those optimizations.

## Component comparison limits

The TP1 block components are directly constructed bring-up/debugging building blocks,
not the serving implementation or an op-for-op reference for the TP4 model. TP1 uses
FlashInfer RMSNorm and fused SiLU/multiply; the TP4 input norm and dense MLP use explicit
FP32 Beam operators. Those paths can differ by BF16 rounding units. TP1 also accepts a
narrower RoPE configuration schema. Compare each component against its declared operator
contract rather than expecting bitwise TP1/TP4 identity.
