<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing MiniMax-H3 with Modeling V2

**Develop new H3 features, kernels and deployment targets in `minimax_h3_v2/`.** The [V1 family](../minimax_h3/) remains the default during rollout and provides the reference for validating V2. Keep V1 changes scoped to fixes that must reach that implementation. Both versions use the existing VisualGen API, CLI and checkpoint identifiers.

## Run a target

Set `TLLM_VISUALGEN_MODELING2_H3=1` before starting the Python process or server. Use the same setting on every worker. An unset switch or any value other than `1` selects V1. Set `H3_CHECKPOINT` to your base H3 checkpoint directory and run this example from the repository root.

```bash
TLLM_VISUALGEN_MODELING2_H3=1 python examples/visual_gen/models/minimax_h3.py \
    --model "$H3_CHECKPOINT" \
    --visual_gen_args examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml
```

The [family registry](pipeline_registry.py) selects a V2 target before transformer construction. It recognizes FastH3 by the `fasth3-` prefix of `schema_version` in `fastvideo_inference.json`. The worker's `AutoPipeline: Creating` message prints the fully qualified class for V2 and the class name for V1.

| Checkpoint contract | V2 target | Denoising schedule |
| --- | --- | --- |
| Base H3 | [default_pipeline/](default_pipeline/) | V1 and V2 default to 50 sigma points including zero, giving 49 joint audio/video updates. |
| Recognized `fasth3-` manifest | [fasth3/](fasth3/) | The manifest supplies the trained schedule. FastH3 8-Step V2 uses nine points and eight updates in both implementations. |

## When to split a target

Start in the target that supports the checkpoint. Keep a scenario there when existing operations and configuration express it clearly. For example, base H3 uses the same functional target for BF16, FP8, block-scaled FP8 and NVFP4. Changing the projection implementation for one precision can be confined to that projection.

Split out a sibling target when the differences need to evolve and be tested together. FastH3 illustrates this boundary. Its learned gate changes how trunk attention is built and its trained DMD schedule changes denoising. For a deployment such as NVFP4 on a fixed four-way Ulysses layout, ask whether changing its weight packing would also require changes to projection order and collectives. If those pieces must move together and would spread deployment-specific branches across the functional pipeline, give that path its own target.

Choose the boundary case by case. A target can specialize only the transformer and denoising loop while reusing encoding and decoding. Include an encoder or decoder when its inputs, layout or execution order also diverge. For a deployment specialization, use `<checkpoint-and-quantization>__<sm>__<parallel>/`. In `route(config, checkpoint_dir)`, check the specialization's effective precision, device capabilities, topology and required features and return its imported pipeline class by name so CI can identify the target. On a mismatch, return the functional target for the same checkpoint. A check deferred to the constructor can fail after selection. Test both the match and fallback through [`AutoPipeline.resolve_class`](../../pipeline_registry.py). The selected pipeline handles per-request and per-step decisions.

Wire a specialization during construction. For example, [FastH3](fasth3/pipeline.py) passes `attention_cls=MiniMaxH3VSAAttention` to the [family transformer](components/transformer.py), which instantiates that class in its trunk blocks. A target with a different attention implementation can pass its own `target/components/attention.py` class at that same point. If its communication schedule also changes, that class imports `target/components/ulysses_attention.py`. Add `target/ops/` or `target/kernels/` only for computations or kernels the target owns. A target reuses family implementations through ordinary imports.

## Where to put code

Choose the **role** from the behavior you are changing and the **owner** from the callers that need it. A target and its model family can each own components, ops and kernels. The VisualGen V2 catalog exposes reusable ops. A catalog op can own a private implementation or call an existing provider kernel.

### Choose the implementation layer

| Role | What belongs here |
| --- | --- |
| Pipeline | How a request moves through conditioning, denoising steps and decoding. The [family pipeline](pipeline.py) runs the joint audio/video loop. |
| Component | A model block that composes ops and owns weights or execution state when needed. [H3 attention](components/attention.py) owns the projections and decides when to fuse Q/K normalization and RoPE. |
| Operation | One tensor computation or collective with explicit inputs, outputs and observable mutations. Its reference specifies the expected result. The caller supplies any runner or workspace. |
| Kernel | The device implementation of an op, including launch geometry, tiling, memory access and fusion. An op can dispatch to one or more kernels or an existing library primitive. |

```text
Owner           Component             Op                       Kernel
--------------  --------------------  -----------------------  ------------------
Base H3 target  TP text encoder  -->  text Q/K norm + RoPE --> Gluon text kernel
H3 family       DiT attention    -->  DiT Q/K norm + RoPE  --> Triton DiT kernel
```

Changing which channels the DiT rotates changes the [family op](ops/rope.py) and its reference tests. Changing the tile size for the same result belongs in the [kernel](kernels/fused_rope.py). The TP text encoder has its [own op](default_pipeline/ops/text_qk_norm_rope.py) and [kernel](default_pipeline/kernels/text_qk_norm_rope.py) for one-dimensional text positions.

### Choose the sharing scope

| Owner | When to use it | Example |
| --- | --- | --- |
| Target (`models/<family>/<target>/`) | The [router](pipeline_registry.py) selects this implementation for a checkpoint or deployment. Keep code specific to that target in its directory. | Base H3's TP text encoder or FastH3's trained schedule. |
| Model family (`models/<family>/`) | The family owns H3 routing and common model behavior. Put code shared by sibling targets here so each can import it independently. | Joint token packing, DiT blocks and gated VSA attention. |
| [VisualGen V2 catalog](../../catalog/) (`catalog/<category>/`) | Another V2 family could call the same op with the same tensor semantics, reference and supported configurations. The first V2 consumer may introduce it. V1 callers keep their existing backend paths. | Fine attention over explicit KV tile indices, counts and valid lengths. |
| Existing runtime or library | Call its existing interface directly when the contract fits. | H3 [projections](components/linear.py) use the LLM V2 GEMM ops and [normalization](components/norm.py) uses `flashinfer_rmsnorm`. |

```text
FastH3 with CUTEDSL VSA
  FastH3Pipeline                             [target]
    | selects the trunk attention class
    v
  MiniMaxH3VSAAttention                      [family component]
    | supplies Q/K/V, the learned gate and H3 metadata
    v
  h3_vsa_attention                           [family op]
    | passes Q/K/V, tile indices, counts and valid lengths to the fine step
    v
  block_sparse_attention                     [VisualGen V2 catalog op]
    | calls the CuTe kernel entry to compute fine attention and logsumexp
    v
  block_sparse_attn_from_indices_cute
```

Change H3's tile selection or coarse/fine merge in the [family VSA op](ops/vsa.py). Change the computation over an already selected tile list in the [catalog op](../../catalog/attention/block_sparse_attention.py). A new model supplies its own selection policy.

### Where parallel behavior belongs

| Change | Where to start |
| --- | --- |
| Change which projection channels a TP rank owns or when it sums partial results | [Linear](components/linear.py) owns projection shard bounds, quantization alignment and row-parallel output reduction. [MLP](components/mlp.py) uses the same bounds for gate/up and down. [Attention](components/attention.py) assigns complete Ulysses head groups to each TP rank. |
| Change when transfers run relative to Q/K/V projections and local attention | The [Ulysses component](components/ulysses_attention.py) holds the process group and coordinates both execution paths. |
| Change which rows or heads a rank receives, or how its receive buffers are laid out | The [H3 QKV ops](ops/ulysses_qkv.py) specify the exchange and local rearrangement. Each has an independent tensor reference. |

```text
Base H3 / H3UlyssesAttention
  sync   exchange projected Q/K/V -------------------+
                                                     +-> local attention -> output all-to-all
  async  project/push V, Q, K -> wait -> unscatter --+
```

The [sequence sharder](components/sequence_sharder.py) pads, shards and gathers DiT and refiner rows. The [tiled VAE](components/tiled_vae.py) distributes video decode tiles. Base H3 uses these parallel paths. FastH3 runs on one GPU. The V1 [token-sharded TP layout](../../parallel/token_sharded_tp.py) is a separate feature used by Wan.

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
| [CuTe FMHA](../../catalog/attention/cute_fmha.py) | Full attention on unpadded, equal-head BSHD tensors with an optional quantization recipe and skip-softmax threshold | Choose the recipe and per-step threshold. H3's SOL path uses masked SDPA when a padding mask is present. |
| [Fine block-sparse attention](../../catalog/attention/block_sparse_attention.py) | Evaluate attention on explicit KV tile indices, counts and valid tile lengths | Construct text/video/audio routes and combine the result with coarse attention using the learned gate |

FlashInfer uses *prefill* for attention over a sequence of query rows. H3 calls it for noncausal self-attention during denoising. Existing [LTX-2.3 FlashInfer deployments](../../../../../examples/visual_gen/configs/ltx23-t2v-fp8-1gpu.yaml) use the V1 backend.

The [H3 attention component](components/dense_attention.py) adapts model inputs to these contracts. The [gated VSA computation](ops/vsa.py) serves both base checkpoints with gate weights and explicit VSA configuration and the FastH3 target.

Use the public operation or runner exported by the catalog entry. For example, H3 creates `FlashInferPrefillRunner` from [flashinfer_prefill.py](../../catalog/attention/flashinfer_prefill.py) and supplies the per-model plan cache. The [FA4 entry](../../catalog/attention/flash_attention_4.py) owns compatibility with the provider's `_flash_attn_fwd` binding. Keep mask conversion, layer selection and denoising policy with the model. Use the [Modeling V2 operation contract](../../../modeling_v2/README.md#writing-operations) for new entries.

H3 keeps backend-specific adapters in the model. [Uncached attention](ops/uncached_attention.py) calls `FallbackFmha.attention`. [H3 PrimTS](components/trtllm_attention_prims.py) calls `_validate_block_sparse_static_profile`. The [TRTLLM SOL adapter](components/trtllm_attention.py) calls the [VisualGen SOL predictor](../../attention/backends/sparse/sol/predictor.py), and the [CuTe SOL path](components/dense_attention.py) checks availability through `_cute_fmha_kernel`. Treat these as implementation dependencies. New consumers should use an exported kernel or runner interface, or introduce a model-independent operation with a reference. See the [LLM V2 dependency rule](../../../modeling_v2/README.md#modeling-v1-code).

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
| [default_pipeline/kernels/text_qk_norm_rope.py](default_pipeline/kernels/text_qk_norm_rope.py) | Normalize and rotate the rank-local Q and K heads of a packed projection in place. |
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
| [components/attention.py](components/attention.py) | Partition heads across TP ranks and construct H3 projections, per-head normalization, three-axis RoPE and dense or gated attention. |
| [components/linear.py](components/linear.py) | Compute projection shard bounds, own rank-local weights, dispatch GEMM and reduce row-parallel outputs. |
| [components/projection_weights.py](components/projection_weights.py) | Partition, pack and quantize checkpoint tensors and scales for those projections. |
| [components/mlp.py](components/mlp.py) | Compose gate/up projections, SwiGLU and down projection. |
| [components/norm.py](components/norm.py) | Own H3 RMSNorm and per-head RMSNorm weights. |
| [components/text_encoder.py](components/text_encoder.py) | Extract H3 conditioning from the required Qwen3-VL hidden layer. |
| [components/vae.py](components/vae.py) | Prepare the video decoder and decode video and audio latents. |
| [components/sequence_sharder.py](components/sequence_sharder.py) | Pad and shard refiner and DiT inputs, prepare padding masks, and gather and trim output rows. |
| [components/ulysses_attention.py](components/ulysses_attention.py) | Compose head/sequence exchanges, projection ordering and CUDA stream synchronization. |
| [components/tiled_vae.py](components/tiled_vae.py) | Partition video decode tiles across the participating VAE ranks and combine their results. |

| Operations and kernels | Responsibility |
| --- | --- |
| [ops/uncached_attention.py](ops/uncached_attention.py) | Issue uncached TRTLLM attention for packed H3 rows. |
| [kernels/sol_attention.py](kernels/sol_attention.py) | Launch SOL attention or invoke the supplied dense fallback. |
| [ops/ulysses_qkv.py](ops/ulysses_qkv.py) | Exchange Q/K/V for synchronous H3 attention and convert asynchronous receive buffers into its attention layout. |
| [ops/activation.py](ops/activation.py) | Define the H3 SwiGLU operation. |
| [ops/adaln.py](ops/adaln.py) | Define adaptive normalization/modulation and its gated residual variant. |
| [ops/quantized_gemm.py](ops/quantized_gemm.py) | Define FP8 scaled and block GEMMs and FP4 activation quantization. |
| [ops/rope.py](ops/rope.py) | Apply partial rotary embedding and expose fused Q/K norm and RoPE. |
| [ops/vsa.py](ops/vsa.py) | Build H3 tile routes, run fine attention and combine it with coarse attention through the learned gate. |
| [kernels/fused_adaln.py](kernels/fused_adaln.py) | Implement and register H3's fused AdaLN Triton kernels. |
| [kernels/fused_rope.py](kernels/fused_rope.py) | Implement and register fused BF16 Q/K normalization and RoPE. |

Use the `trtllm::minimax_h3_v2_` prefix for H3 V2 Python-registered Triton ops so V1 and V2 can coexist in the process-wide dispatcher. For a new C++/CUDA kernel, follow the [native code guide](../../../../../cpp/tensorrt_llm/modeling_v2/README.md) for namespaces and registration. [CMake discovery](../../../../../cpp/tensorrt_llm/modeling_v2/CMakeLists.txt) scans `_torch/modeling_v2/models/`. A family library under `_torch/visual_gen/models/` also needs build discovery and loading from the VisualGen path.

## Test a change

Mirror family and target code in the [unit tests](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/) and give each routed target a [whole-pipeline quality case](../../../../../tests/integration/defs/visual_gen/minimax_h3_v2/). Mirror [family operations](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/ops/) and [catalog entries](../../../../../tests/unittest/_torch/visual_gen/catalog/attention/) at their respective scopes. Compare outputs and buffer mutations against each operation’s reference on its declared cells. Register pre-merge tests by file or directory in the [CI test lists](../../../../../tests/integration/test_lists/test-db/) for the required hardware and rank count. FastH3's real-checkpoint quality cases run on the [functional QA schedule](../../../../../tests/integration/test_lists/qa/functional/llm_function_core.txt). Change-based test selection operates on registered entries.

A target change exercises that target. A shared family change exercises its consumers. Registry changes exercise both checkpoint targets and both switch settings. A catalog change runs its mirrored contract tests and the V2 targets that import the operation. V1 paths retain their existing tests.

For each deployment tested, run the same request in separate V1 and V2 processes on the same GPUs. Fix the checkpoint revision, conditioning inputs, seed, weight and attention precision, backend, parallel layout, compilation settings and sigma schedule. Compare the two decoded videos and audio waveforms directly. A multi-GPU run against one GPU measures topology sensitivity. A Diffusers comparison measures reference quality. Report these separately from refactor parity. If the V1 reference has a reproducible defect, record the original comparison and show that V2 matches the checkpoint or an independent numerical reference for the affected computation. Use that reference to qualify the correction. Use 50 points for the base-H3 quality reference and test omitted defaults separately. Before retiring V1 and the opt-in switch, qualify generated video and audio, supported parallel paths, warmed latency and peak memory against V1 on representative deployments.
