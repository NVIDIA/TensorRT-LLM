<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing MiniMax-H3 with Modeling V2

**Develop new H3 features, kernels and deployment targets in `minimax_h3_v2/`.** The [V1 family](../minimax_h3/) remains the default during rollout and provides the reference for validating V2. Keep V1 changes scoped to fixes that must reach that implementation. Both versions use the existing VisualGen API, CLI and checkpoint identifiers.

## Run a target

Set `TLLM_VISUALGEN_MODELING2_H3=1` before starting the Python process or server. Use the same setting on every worker. Unset or `0` selects V1. Set `H3_CHECKPOINT` to your base H3 checkpoint directory and run this example from the repository root.

```bash
TLLM_VISUALGEN_MODELING2_H3=1 python examples/visual_gen/models/minimax_h3.py \
    --model "$H3_CHECKPOINT" \
    --visual_gen_args examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml
```

The [family registry](pipeline_registry.py) selects a V2 target from the resolved checkpoint before transformer construction. Check the fully qualified class in the loader log when comparing versions.

| Checkpoint contract | V2 target | Denoising schedule |
| --- | --- | --- |
| Base H3 | [default_pipeline/](default_pipeline/) | V1 and V2 default to 50 sigma points including zero, giving 49 joint audio/video updates. |
| Recognized `fasth3-` manifest | [fasth3/](fasth3/) | The manifest supplies the trained schedule. FastH3 8-Step V2 uses nine points and eight updates in both implementations. |

## When to split a target

Start in the target that supports the checkpoint. Keep a scenario there when existing operations and configuration express it clearly. For example, base H3 uses the same functional target for BF16, FP8, block-scaled FP8 and NVFP4. Changing the projection implementation for one precision can be confined to that projection.

Split out a sibling target when the differences need to evolve and be tested together. FastH3 illustrates this boundary. Its learned gate changes how trunk attention is built and its trained DMD schedule changes denoising. For a deployment such as NVFP4 on a fixed four-way Ulysses layout, ask whether changing its weight packing would also require changes to projection order and collectives. If those pieces must move together and would spread deployment-specific branches across the functional pipeline, give that path its own target.

Choose the boundary case by case. A target can specialize only the transformer and denoising loop while reusing encoding and decoding. Include an encoder or decoder when its inputs, layout or execution order also diverge. For a deployment specialization, use `<checkpoint-and-quantization>__<sm>__<parallel>/`, add the match under the corresponding checkpoint contract in the registry, and retain its functional target as the fallback. The selected pipeline handles choices that depend on an individual request or step.

## Where to put code

| Who consumes the change? | Place it here | H3 example |
| --- | --- | --- |
| One target | The target directory, including its local operations or kernels | Base Ref2VA conditioning and its TP text encoder, or FastH3's trained schedule |
| Sibling H3 targets with the same contract | The H3 family | Packed rows, transformer blocks, attention modules and H3 parallel layouts. Targets import from the family and evolve independently of sibling targets. |
| Different model families with the same operation contract | The [Modeling V2 catalog](../../../modeling_v2/catalog/) | A communication op selects a custom kernel or a general implementation given its inputs, layout and device capabilities. Each family retains its own attention and parallel composition. |
| Runtime services | The existing VisualGen runtime | Checkpoint discovery, process groups, executor and lifecycle interfaces |

For example, H3 orders its asynchronous Ulysses transfers V, Q, K and joins them before attention. The [communication ops](../../../modeling_v2/catalog/comm/ulysses.py) select a packed exchange with a fused layout kernel or the general collective path. A second family can reuse that operation while scheduling its projections differently. For attention, share an operation when tensor layout, masking and precision agree. Keep the model's projections, positional encoding and attention policy in the family. See the [LLM Modeling V2 guide](../../../modeling_v2/README.md).

## File map

Paths below are relative to this directory. The root [\_\_init\_\_.py](__init__.py) exports the base V2 pipeline class. The other `__init__.py` files mark their directories as Python packages.

| Family and target files | Responsibility |
| --- | --- |
| [pipeline_registry.py](pipeline_registry.py) | Select the checkpoint target and apply FastH3 attention settings before construction. |
| [checkpoint.py](checkpoint.py) | Read and recognize the FastH3 inference manifest. |
| [pipeline.py](pipeline.py) | Implement the shared pipeline lifecycle, request preparation, joint denoising loop and decode hooks. |
| [packing.py](packing.py) | Build packed text, video, audio and reference rows, positions and timestep indices. |
| [weights.py](weights.py) | Remap checkpoint keys and assemble transformer weights for loading. |
| [default_pipeline/pipeline.py](default_pipeline/pipeline.py) | Compose the base target, choose workflow weights and configure conditioning, text encoder and scheduler shifts. |
| [default_pipeline/ref2va.py](default_pipeline/ref2va.py) | Validate, normalize and order reference media for Ref2VA. |
| [default_pipeline/text_encoder_tp.py](default_pipeline/text_encoder_tp.py) | Shard the Qwen3-VL text encoder and load its rank-local weights. |
| [default_pipeline/ops/text_attention.py](default_pipeline/ops/text_attention.py) | Implement causal grouped-query attention for the TP text encoder. |
| [fasth3/pipeline.py](fasth3/pipeline.py) | Configure gated video sparse attention (VSA) and select the trained generation and warmup schedules. |
| [fasth3/schedule.py](fasth3/schedule.py) | Convert trained DMD steps to video and audio scheduler timesteps and sigmas. |

| Components and parallelism | Responsibility |
| --- | --- |
| [components/transformer.py](components/transformer.py) | Assemble the token refiner, DiT blocks and joint audio/video outputs. |
| [components/attention.py](components/attention.py) | Build H3 projections, per-head normalization, three-axis RoPE and dense or gated attention. |
| [components/linear.py](components/linear.py) | Own rank-local projection parameters and dispatch their GEMM. |
| [components/projection_weights.py](components/projection_weights.py) | Partition, pack and quantize checkpoint tensors and scales for those projections. |
| [components/mlp.py](components/mlp.py) | Compose gate/up projections, SwiGLU and down projection. |
| [components/norm.py](components/norm.py) | Own H3 RMSNorm and per-head RMSNorm weights. |
| [components/text_encoder.py](components/text_encoder.py) | Extract H3 conditioning from the required Qwen3-VL hidden layer. |
| [components/vae.py](components/vae.py) | Prepare the video decoder and decode video and audio latents. |
| [parallel/tp.py](parallel/tp.py) | Compute TP shard bounds and head partitions compatible with Ulysses. |
| [parallel/sequence.py](parallel/sequence.py) | Pad, shard and gather packed rows, masks, RoPE and timestep metadata. |
| [parallel/ulysses.py](parallel/ulysses.py) | Compose head/sequence exchanges, projection ordering and CUDA stream synchronization. |
| [parallel/vae.py](parallel/vae.py) | Partition video decode tiles across the participating VAE ranks and combine their results. |

| Operations and kernels | Responsibility |
| --- | --- |
| [ops/activation.py](ops/activation.py) | Define the H3 SwiGLU operation. |
| [ops/adaln.py](ops/adaln.py) | Define adaptive normalization/modulation and its gated residual variant. |
| [ops/attention.py](ops/attention.py) | Select the H3 attention implementation and adapt dense attention inputs to its kernel runner. |
| [ops/quantized_gemm.py](ops/quantized_gemm.py) | Define FP8 scaled and block GEMMs and FP4 activation quantization. |
| [ops/rope.py](ops/rope.py) | Apply partial rotary embedding and expose fused Q/K norm and RoPE. |
| [ops/trtllm_attention.py](ops/trtllm_attention.py) | Pack Q/K/V and issue TRTLLM attention with H3 masks and sparse routes. |
| [ops/trtllm_attention_prims.py](ops/trtllm_attention_prims.py) | Maintain contiguous attention plans and quantization state for an H3 instance. |
| [ops/vsa.py](ops/vsa.py) | Build tile metadata and run mixed-modality video sparse attention. |
| [kernels/fused_adaln.py](kernels/fused_adaln.py) | Implement and register H3's fused AdaLN Triton kernels. |
| [kernels/fused_rope.py](kernels/fused_rope.py) | Implement and register fused BF16 Q/K normalization and RoPE. |

Use the `trtllm::minimax_h3_v2_` prefix for H3 V2 Python-registered Triton ops so V1 and V2 can coexist in the process-wide dispatcher. Follow the [native code guide](../../../../../cpp/tensorrt_llm/modeling_v2/README.md) for a native family or target namespace.

## Test a change

Mirror family and target code in the [unit tests](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/) and give each routed target a [whole-pipeline quality case](../../../../../tests/integration/defs/visual_gen/minimax_h3_v2/). A target change exercises that target. A shared family change exercises its consumers. Registry changes exercise both checkpoint targets and both switch settings. A catalog change exercises its category and consuming families.

For V1/V2 comparisons, use the same checkpoint revision, seed, inputs, precision, backend, topology and explicit step count. Use 50 points on both sides for the base-H3 quality reference and test omitted defaults separately. Before retiring V1 and the opt-in switch, qualify generated video and audio, supported parallel paths, warmed latency and peak memory against V1 on representative deployments.
