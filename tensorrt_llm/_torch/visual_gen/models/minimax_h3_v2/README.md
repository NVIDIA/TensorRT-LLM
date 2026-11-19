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

### Choose the implementation layer

Place a change according to the decision it makes. Then choose its [sharing scope](#choose-the-sharing-scope).

| Layer | Decision it owns | H3 example |
| --- | --- | --- |
| Pipeline | How a request moves through conditioning, denoising and decoding, including which components run at each step. | The [family pipeline](pipeline.py) advances the joint audio/video denoising loop. The [FastH3 target](fasth3/pipeline.py) supplies its trained schedule. |
| Component | How a model block computes its output. It composes operations and holds the parameters or other instance state that block needs. | [H3 attention](components/attention.py) owns its projections and normalization weights and chooses when to apply fused Q/K norm and RoPE. |
| Parallel | How tensors and weights are partitioned across ranks, which process group participates, and when collectives run relative to model computation. | [H3 Ulysses](parallel/ulysses.py) schedules V, Q, K on the asynchronous path and waits for the transfers before attention. |
| Operation | What one tensor computation or collective returns for explicit inputs, layouts and parameters. The call is stateless. It can select a compatible kernel or compose several calls that satisfy the same contract. | The [Q/K norm and RoPE op](ops/rope.py) takes packed QKV, norm weights and rotation tables. Its PyTorch reference defines the expected Q and K. |
| Kernel | How the device executes the operation, including fusion, tiling, memory access and launch configuration. | The [fused Triton kernel](kernels/fused_rope.py) loads the Q/K head data, computes RMS normalization and rotates the leading channels in one launch. |

Changing the fused RoPE tile size belongs in `kernels/`. Changing the rotation from the leading channels to the trailing channels changes the contract in `ops/` and its reference tests. In asynchronous Ulysses, [parallel/ulysses.py](parallel/ulysses.py) determines when the V/Q/K transfers run. [ops/ulysses.py](ops/ulysses.py) determines how the three received buffers map to the attention layout.

### Choose the sharing scope

**Only fully model-independent operations belong in a catalog.** Independent callers must be able to use the same implementation unchanged. Express the computation in terms of tensors, layouts and explicit operation parameters. Model wrappers belong in their family or target, even when several models call the same native kernel. They own checkpoint interpretation, model-specific packing, masks, gates and collective scheduling.

| Scope of the change | Place it here | H3 example |
| --- | --- | --- |
| One target | The target directory, including its local operations or kernels | Base Ref2VA conditioning and its TP text encoder, or FastH3's trained schedule |
| Sibling H3 targets | The H3 family | Packed rows, transformer blocks, attention modules and H3 Ulysses Q/K/V exchange and receive layouts |
| A model-independent operation used unchanged by independent VisualGen callers | The [VisualGen catalog](../../catalog/) | H3 V2 and the VisualGen VSA backend share fine block-sparse attention over explicit KV tile indices. |
| An existing model-independent operation used by VisualGen and LLM callers | Its entry in the [Modeling V2 catalog](../../../modeling_v2/catalog/) | Import `thop_attention` and `cublas_mm` directly. |
| Runtime services | The existing VisualGen runtime | Checkpoint discovery, process groups, executor and lifecycle interfaces |

**Attention.** The [block-sparse attention op](../../catalog/attention/block_sparse_attention.py) takes head-major Q/K/V, selected KV tile indices, route counts, valid KV tile lengths and an optional scale. It returns attention output and logsumexp. [H3](ops/vsa.py) builds mixed-modality routes and applies its learned gate. The [VisualGen VSA backend](../../attention/backends/sparse/vsa/backend.py) supplies its predictor's routes and chooses its fallback. Both call the same operation unchanged. Each family owns its attention module and projections. Call a library operation such as PyTorch SDPA directly when it already provides the required interface.

**Communication.** The base and FastH3 targets share the [H3 Ulysses wrappers](ops/ulysses.py). They encode how H3 exchanges sequence shards for head shards and converts asynchronous receive buffers to the layout its attention expects. The [H3 parallel module](parallel/ulysses.py) owns the V, Q, K projection order, streams and barriers. Other families can call the same native kernels from their own wrappers.

Both the H3 Ulysses wrappers and the block-sparse attention op use the [Modeling V2 `OpWrapper` contract](../../../modeling_v2/README.md#writing-operations) to define a reference and test cases. Mirror each operation's tests at the scope that owns it.

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
| [ops/ulysses.py](ops/ulysses.py) | Exchange Q/K/V for synchronous H3 attention and convert asynchronous receive buffers into its attention layout. |
| [ops/activation.py](ops/activation.py) | Define the H3 SwiGLU operation. |
| [ops/adaln.py](ops/adaln.py) | Define adaptive normalization/modulation and its gated residual variant. |
| [ops/attention.py](ops/attention.py) | Select the H3 attention implementation and adapt dense attention inputs to its kernel runner. |
| [ops/quantized_gemm.py](ops/quantized_gemm.py) | Define FP8 scaled and block GEMMs and FP4 activation quantization. |
| [ops/rope.py](ops/rope.py) | Apply partial rotary embedding and expose fused Q/K norm and RoPE. |
| [ops/trtllm_attention.py](ops/trtllm_attention.py) | Pack Q/K/V and issue TRTLLM attention with H3 masks and sparse routes. |
| [ops/trtllm_attention_prims.py](ops/trtllm_attention_prims.py) | Maintain contiguous attention plans and quantization state for an H3 instance. |
| [ops/vsa.py](ops/vsa.py) | Build H3 tile routes, run fine attention and combine it with coarse attention through the learned gate. |
| [kernels/fused_adaln.py](kernels/fused_adaln.py) | Implement and register H3's fused AdaLN Triton kernels. |
| [kernels/fused_rope.py](kernels/fused_rope.py) | Implement and register fused BF16 Q/K normalization and RoPE. |

Use the `trtllm::minimax_h3_v2_` prefix for H3 V2 Python-registered Triton ops so V1 and V2 can coexist in the process-wide dispatcher. Follow the [native code guide](../../../../../cpp/tensorrt_llm/modeling_v2/README.md) for a native family or target namespace.

## Test a change

Mirror family and target code in the [unit tests](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/) and give each routed target a [whole-pipeline quality case](../../../../../tests/integration/defs/visual_gen/minimax_h3_v2/). Mirror VisualGen catalog entries under [visual_gen/catalog/ in the unit tests](../../../../../tests/unittest/_torch/visual_gen/catalog/), and compare each kernel with its reference on the declared cells. A target change exercises that target. A shared family change exercises its consumers. Registry changes exercise both checkpoint targets and both switch settings. A VisualGen catalog change also exercises its consuming model and backend paths.

For V1/V2 comparisons, use the same checkpoint revision, seed, inputs, precision, backend, topology and explicit step count. Use 50 points on both sides for the base-H3 quality reference and test omitted defaults separately. Before retiring V1 and the opt-in switch, qualify generated video and audio, supported parallel paths, warmed latency and peak memory against V1 on representative deployments.
