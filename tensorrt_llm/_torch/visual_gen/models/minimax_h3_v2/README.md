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

The [family registry](pipeline_registry.py) selects a V2 target from the resolved checkpoint before transformer construction. Check the fully qualified class in the loader's `AutoPipeline: Creating` message before timing a target.

| Checkpoint contract | V2 target | Denoising schedule |
| --- | --- | --- |
| Base H3 | [default_pipeline/](default_pipeline/) | V1 and V2 default to 50 sigma points including zero, giving 49 joint audio/video updates. |
| Recognized `fasth3-` manifest | [fasth3/](fasth3/) | The manifest supplies the trained schedule. FastH3 8-Step V2 uses nine points and eight updates in both implementations. |

## When to split a target

Start in the target that supports the checkpoint. Keep a scenario there when existing operations and configuration express it clearly. For example, base H3 uses the same functional target for BF16, FP8, block-scaled FP8 and NVFP4. Changing the projection implementation for one precision can be confined to that projection.

Split out a sibling target when the differences need to evolve and be tested together. FastH3 illustrates this boundary. Its learned gate changes how trunk attention is built and its trained DMD schedule changes denoising. For a deployment such as NVFP4 on a fixed four-way Ulysses layout, ask whether changing its weight packing would also require changes to projection order and collectives. If those pieces must move together and would spread deployment-specific branches across the functional pipeline, give that path its own target.

Choose the boundary case by case. A target can specialize only the transformer and denoising loop while reusing encoding and decoding. Include an encoder or decoder when its inputs, layout or execution order also diverge. For a deployment specialization, use `<checkpoint-and-quantization>__<sm>__<parallel>/`. In `route(config, checkpoint_dir)`, check the specialization's effective precision, device capabilities, topology and required features and return its imported pipeline class by name so CI can identify the target. On a mismatch, return the functional target for the same checkpoint. A check deferred to the constructor can fail after selection. Test both the match and fallback through [`AutoPipeline.resolve_class`](../../pipeline_registry.py). The selected pipeline handles per-request and per-step decisions.

## Where to put code

### Choose the implementation layer

Place a change according to the decision it makes. Then choose its [sharing scope](#choose-the-sharing-scope).

| Layer | Decision it owns | H3 example |
| --- | --- | --- |
| Pipeline | How a request moves through conditioning, denoising and decoding, including which components run at each step. | The [family pipeline](pipeline.py) advances the joint audio/video denoising loop. The [FastH3 target](fasth3/pipeline.py) supplies its trained schedule. |
| Component | How a model block computes its output. It composes operations and holds the parameters or other instance state that block needs. | [H3 attention](components/attention.py) owns its projections and normalization weights and chooses when to apply fused Q/K norm and RoPE. |
| Parallel | Define rank-local shard ranges and schedule collectives on the participating process group. Components use those ranges to construct paired projections and load weights. | [H3 Ulysses](parallel/ulysses.py) schedules V, Q, K on the asynchronous path and waits for the transfers before attention. |
| Operation | What one tensor computation or collective returns for explicit inputs, layouts and parameters. The call is stateless. It can select a compatible kernel or compose several calls that satisfy the same contract. | The [Q/K norm and RoPE op](ops/rope.py) takes packed QKV, norm weights and rotation tables. Its PyTorch reference defines the expected Q and K. |
| Kernel | How the device executes the operation, including fusion, tiling, memory access and launch configuration. | The [fused Triton kernel](kernels/fused_rope.py) loads the Q/K head data, computes RMS normalization and rotates the leading channels in one launch. |

Changing the fused RoPE tile size belongs in `kernels/`. Changing the rotation from the leading channels to the trailing channels changes the contract in `ops/` and its reference tests. In asynchronous Ulysses, [parallel/ulysses.py](parallel/ulysses.py) determines when the V/Q/K transfers run. [ops/ulysses.py](ops/ulysses.py) determines how the three received buffers map to the attention layout.

### Choose the sharing scope

The VisualGen catalog is for **Modeling V2**. Add a model-independent operation when its data contract, reference and tested configurations are precise enough for another V2 family to use unchanged. The first migrated family can introduce it. V1 pipelines keep their existing paths. Each family starts using a catalog entry when its V2 implementation calls it.

| Scope of the change | Place it here | Example |
| --- | --- | --- |
| One checkpoint target | The target directory | Base H3's text-only TP encoder, or FastH3's trained schedule |
| Model-specific computation shared by sibling targets | The family directory | Packed rows, H3 attention components, learned VSA gates and Ulysses scheduling |
| A reusable model-independent computation | The [VisualGen V2 catalog](../../catalog/attention/) | Attention over explicit valid KV lengths or preselected KV tiles |
| An existing kernel, library primitive or runtime interface | Its existing owner | PyTorch SDPA, process groups and compatible GEMM primitives |

### How large is a catalog operation?

Use **one testable tensor computation, collective or explicit state update** as the unit of sharing. An entry may launch one kernel or several. The [LLM catalog](../../../modeling_v2/catalog/) illustrates the range.

| LLM example | Contract | VisualGen implication |
| --- | --- | --- |
| [RMSNorm](../../../modeling_v2/catalog/norm/flashinfer_rmsnorm.py) | Given activations, weights and epsilon, return the normalized activations. | A single kernel can be an entry when V2 families use it unchanged. |
| [Reduce-scatter](../../../modeling_v2/catalog/comm/reducescatter.py) | Reduce contributions and return the local slice for an explicit group. The implementation can dispatch by communicator. | Share the collective when layouts agree. Each family decides where it runs and what work overlaps it. |
| [MLA RoPE and KV append](../../../modeling_v2/catalog/attention/mla_rope_append_paged_kv_assign_q.py) | Rotate new tokens and write specified rows to an explicit KV pool. The reference describes the resulting buffer contents. | Several steps can be one operation when independent callers need the same outputs and mutations. |

For a proposed entry, write its reference using only the arguments at the call site. Expose the numerical inputs and any observable buffer mutations. If reuse requires the caller's checkpoint, layer identity, token packing or denoising step, keep that decision in the model.

| VisualGen V2 entry | Catalog computes | H3 supplies or owns |
| --- | --- | --- |
| [FA4](../../catalog/attention/flash_attention_4.py) | Attention output and logsumexp from Q/K/V, scale, causal mode and optional valid KV lengths | Convert trailing-padding masks to prefix lengths and restore the model's output dtype |
| [FlashInfer prefill](../../catalog/attention/flashinfer_prefill.py) | Select a compatible provider kernel and compute prefill with explicit quantization formats, mask and runner | Create the runner for the loaded instance and pass its plan cache and the selected precision |
| [CuTe FMHA](../../catalog/attention/cute_fmha.py) | Full attention on equal-head BSHD tensors, including quantization and a numerical skip-softmax threshold | Choose the recipe, resolve the threshold for the step and handle padding or dense fallback |
| [Fine block-sparse attention](../../catalog/attention/block_sparse_attention.py) | Evaluate attention on explicit KV tile indices, counts and valid tile lengths | Construct text/video/audio routes and combine the result with coarse attention using the learned gate |

The [H3 attention component](components/dense_attention.py) adapts model inputs to these contracts. The [gated VSA computation](ops/vsa.py) serves both base checkpoints with gate weights and explicit VSA configuration and the FastH3 target.

**H3 communication.** [H3 parallel code](parallel/ulysses.py) owns the V, Q, K projection order, streams and barriers. The [Ulysses ops](ops/ulysses.py) exchange Q/K/V or convert receive buffers to H3 attention layouts. Models can call the same native communication kernel with different schedules and layouts.

Use the [Modeling V2 operation contract](../../../modeling_v2/README.md#writing-operations) at target, family and catalog scope. Pass execution state explicitly, put model-scoped runner lifetimes with the component, and test each supported layout and dtype against a reference. Use lower-level kernel or library interfaces in forward, fallback and loading paths, in line with the [V1 dependency rule](../../../modeling_v2/README.md#modeling-v1-code).

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
| [default_pipeline/ops/text_qk_norm_rope.py](default_pipeline/ops/text_qk_norm_rope.py) | Define per-head Q/K normalization and rotation for the TP encoder's one-dimensional text positions. |
| [default_pipeline/kernels/text_qk_norm_rope.py](default_pipeline/kernels/text_qk_norm_rope.py) | Apply text RoPE in place to the rank-local Q and K heads of a packed projection. |
| [fasth3/pipeline.py](fasth3/pipeline.py) | Configure gated video sparse attention (VSA) and select the trained generation and warmup schedules. |
| [fasth3/schedule.py](fasth3/schedule.py) | Convert trained DMD steps to video and audio scheduler timesteps and sigmas. |

| Components and parallelism | Responsibility |
| --- | --- |
| [components/dense_attention.py](components/dense_attention.py) | Select the H3 dense attention path, own its runner and adapt H3 masks and per-step choices to the operation. |
| [components/trtllm_attention.py](components/trtllm_attention.py) | Pack H3 Q/K/V and run TRTLLM attention with the model's masks and sparse routes. |
| [components/trtllm_attention_prims.py](components/trtllm_attention_prims.py) | Maintain contiguous attention plans and quantization state for an H3 instance. |
| [components/transformer.py](components/transformer.py) | Assemble the token refiner, DiT blocks and joint audio/video outputs. |
| [components/cudnn_runner.py](components/cudnn_runner.py) | Build dense cuDNN plans for H3. |
| [components/skip_softmax_threshold.py](components/skip_softmax_threshold.py) | Resolve the current threshold for H3 CuTe attention. |
| [components/attention.py](components/attention.py) | Build H3 projections, per-head normalization, three-axis RoPE and dense or gated attention. |
| [components/linear.py](components/linear.py) | Own rank-local projection parameters and dispatch their GEMM. |
| [components/projection_weights.py](components/projection_weights.py) | Partition, pack and quantize checkpoint tensors and scales for those projections. |
| [components/mlp.py](components/mlp.py) | Compose gate/up projections, SwiGLU and down projection. |
| [components/norm.py](components/norm.py) | Own H3 RMSNorm and per-head RMSNorm weights. |
| [components/text_encoder.py](components/text_encoder.py) | Extract H3 conditioning from the required Qwen3-VL hidden layer. |
| [components/vae.py](components/vae.py) | Prepare the video decoder and decode video and audio latents. |
| [parallel/tp.py](parallel/tp.py) | Compute TP shard bounds and head partitions compatible with Ulysses. |
| [parallel/sequence_sharder.py](parallel/sequence_sharder.py) | Pad and shard refiner and DiT inputs, prepare padding masks, and gather and trim output rows. |
| [parallel/ulysses.py](parallel/ulysses.py) | Compose head/sequence exchanges, projection ordering and CUDA stream synchronization. |
| [parallel/vae.py](parallel/vae.py) | Partition video decode tiles across the participating VAE ranks and combine their results. |

| Operations and kernels | Responsibility |
| --- | --- |
| [ops/uncached_attention.py](ops/uncached_attention.py) | Issue uncached TRTLLM attention for packed H3 rows. |
| [kernels/sol_attention.py](kernels/sol_attention.py) | Launch SOL attention or invoke the supplied dense fallback. |
| [ops/ulysses.py](ops/ulysses.py) | Exchange Q/K/V for synchronous H3 attention and convert asynchronous receive buffers into its attention layout. |
| [ops/activation.py](ops/activation.py) | Define the H3 SwiGLU operation. |
| [ops/adaln.py](ops/adaln.py) | Define adaptive normalization/modulation and its gated residual variant. |
| [ops/quantized_gemm.py](ops/quantized_gemm.py) | Define FP8 scaled and block GEMMs and FP4 activation quantization. |
| [ops/rope.py](ops/rope.py) | Apply partial rotary embedding and expose fused Q/K norm and RoPE. |
| [ops/vsa.py](ops/vsa.py) | Build H3 tile routes, run fine attention and combine it with coarse attention through the learned gate. |
| [kernels/fused_adaln.py](kernels/fused_adaln.py) | Implement and register H3's fused AdaLN Triton kernels. |
| [kernels/fused_rope.py](kernels/fused_rope.py) | Implement and register fused BF16 Q/K normalization and RoPE. |

Use the `trtllm::minimax_h3_v2_` prefix for H3 V2 Python-registered Triton ops so V1 and V2 can coexist in the process-wide dispatcher. For a new C++/CUDA kernel, follow the [native code guide](../../../../../cpp/tensorrt_llm/modeling_v2/README.md) for namespaces and registration. [CMake discovery](../../../../../cpp/tensorrt_llm/modeling_v2/CMakeLists.txt) scans `_torch/modeling_v2/models/`. A family library under `_torch/visual_gen/models/` also needs build discovery and loading from the VisualGen path.

## Test a change

Mirror family and target code in the [unit tests](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/) and give each routed target a [whole-pipeline quality case](../../../../../tests/integration/defs/visual_gen/minimax_h3_v2/). Mirror [family operations](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/ops/) and [catalog entries](../../../../../tests/unittest/_torch/visual_gen/catalog/attention/) at their respective scopes. Compare outputs and buffer mutations against each operation’s reference on its declared cells. Register each new test by file or directory in the [CI test lists](../../../../../tests/integration/test_lists/test-db/) for the required hardware and rank count. A target's quality case can also run on an [owned QA schedule](../../../../../tests/integration/test_lists/qa/). Change-based test selection operates on registered entries.

A target change exercises that target. A shared family change exercises its consumers. Registry changes exercise both checkpoint targets and both switch settings. A catalog change runs its mirrored contract tests and the V2 targets that import the operation. V1 paths retain their existing tests.

For V1/V2 comparisons, use the same checkpoint revision, seed, inputs, precision, backend, topology and explicit step count. Use 50 points on both sides for the base-H3 quality reference and test omitted defaults separately. Before retiring V1 and the opt-in switch, qualify generated video and audio, supported parallel paths, warmed latency and peak memory against V1 on representative deployments.
