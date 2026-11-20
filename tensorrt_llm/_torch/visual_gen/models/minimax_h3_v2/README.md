<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing MiniMax-H3 with Modeling V2

Modeling V2 lets contributors change and test one deployment independently. A **target** owns the pipeline for a checkpoint or deployment configuration. Its **model family** owns routing and the H3 behavior its targets share. See the [LLM Modeling V2 guide](../../../modeling_v2/README.md).

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

Start in the checkpoint’s **functional target** (the V2 pipeline used when no deployment specialization matches). Base H3 uses `default_pipeline/` for BF16, FP8, block-scaled FP8 and NVFP4. FastH3 uses `fasth3/`. Keep a change in that target when it fits an existing component, such as a projection for one precision.

Split out a sibling target when the differences need to evolve and be tested together. FastH3 illustrates this boundary. Its learned gate changes how trunk attention is built and its trained DMD schedule changes denoising. For a deployment such as NVFP4 on a fixed four-way Ulysses layout, ask whether changing its weight packing would also require changes to projection order and collectives. If those pieces must move together and would spread deployment-specific branches across the functional pipeline, give that path its own target.

A target can specialize the transformer and denoising loop while reusing family-owned encoding and decoding. Include an encoder or decoder in the target when its inputs, layout or execution order also diverge.

Keep sibling targets independent. A target can import family code, catalog ops and runtime interfaces. To reuse the base target’s TP text encoder in a second target, move the shared encoder code into the family and import it from both. A direct import from `default_pipeline/` would leave the second target untested when the encoder changes because change-based CI selects only the modified target’s tests.

Name a deployment specialization `<checkpoint-and-quantization>__<sm>__<parallel>/`. Check its effective precision, device capabilities, topology and required features in [`route(config, checkpoint_dir)`](pipeline_registry.py). Return the imported pipeline class directly so CI can identify the target. On a mismatch, return the functional target for the same checkpoint. Test both paths through [`AutoPipeline.resolve_class`](../../pipeline_registry.py). The selected pipeline handles per-request and per-step decisions.

Wire a specialization during construction. For example, [FastH3](fasth3/pipeline.py) passes `attention_cls=MiniMaxH3VSAAttention` to the [family transformer](components/transformer.py), which instantiates that class in its trunk blocks. A target with a different attention implementation can pass its own `target/components/attention.py` class at that same point. If its communication schedule also changes, that class imports `target/components/ulysses_attention.py`. Add `target/ops/` or `target/kernels/` only for computations or kernels the target owns.

## Where to put code

Choose the **role** from the behavior you are changing and the **owner** from the callers that need it. A target and its model family can each own components, ops and kernels. The VisualGen V2 catalog exposes reusable ops. A catalog op can own a private implementation or call an existing provider kernel.

### Choose the implementation layer

| Role | What belongs here |
| --- | --- |
| Pipeline | How a request moves through conditioning, denoising steps and decoding. The [family pipeline](pipeline.py) runs the joint audio/video loop. |
| Component | Defines how part of the model runs. It owns the weights or execution state it needs and chooses when to call its ops. For example, the [TP text encoder block](default_pipeline/text_encoder_tp.py) owns the QKV and output projections and calls causal attention between them. |
| Operation | Specifies what a call computes given explicit inputs, including any weights or execution state supplied by the caller. Its reference defines the outputs and observable buffer updates, even when the implementation combines several computations or collectives. For example, [text causal attention](default_pipeline/ops/text_attention.py) takes projected Q/K/V and prompt offsets and returns the attention output. |
| Kernel | The device implementation of an op, including launch geometry, tiling, memory access and fusion. An op can dispatch to one or more kernels or an existing library primitive. |

```text
Owner           Component             Op                       Kernel
--------------  --------------------  -----------------------  ------------------
Base H3 target  TP text encoder  -->  text Q/K norm + RoPE --> Gluon text kernel
H3 family       DiT attention    -->  DiT Q/K norm + RoPE  --> Triton DiT kernel
```

Changing which channels the DiT rotates changes the [family op](ops/rope.py) and its reference tests. Changing the tile size for the same result belongs in the [kernel](kernels/fused_rope.py). The TP text encoder has its [own op](default_pipeline/ops/text_qk_norm_rope.py) and [kernel](default_pipeline/kernels/text_qk_norm_rope.py) for one-dimensional text positions.

Use the `trtllm::minimax_h3_v2_` prefix for H3 V2 Python-registered Triton ops so V1 and V2 can coexist in the process-wide dispatcher. For a new C++/CUDA kernel, follow the [native code guide](../../../../../cpp/tensorrt_llm/modeling_v2/README.md) for namespaces and registration. [CMake discovery](../../../../../cpp/tensorrt_llm/modeling_v2/CMakeLists.txt) scans `_torch/modeling_v2/models/`. A family library under `_torch/visual_gen/models/` also needs build discovery and loading from the VisualGen path.

#### When to extract an op

Extract a component method when both conditions hold.

1. **Can you write a reference from the call arguments alone?** Specify the expected outputs and any mutations using the inputs, their layouts and any state the caller supplies. The reference should be usable independently of the enclosing model block.
2. **Does the behavior need an independent implementation or test?** Typical reasons include introducing a custom kernel, checking a fast path and fallback against the same reference, or reusing the exact contract across callers. A custom kernel can justify extraction with a single caller.

[`ulysses_qkv_all_to_all`](ops/ulysses_qkv.py) is a family op. Packing Q/K/V for one all-to-all and exchanging the three tensors separately must produce the same heads in the same layout on each rank. The [op tests](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/ops/test_ulysses_qkv.py) check both paths against that reference.

[`H3SequenceSharder.gather`](components/sequence_sharder.py) stays a component method. The refiner and DiT call it to gather rank-local rows and trim padding. A fused gather-and-trim kernel would justify extracting an op and using the current implementation as its reference.

Choose the [sharing scope](#choose-the-sharing-scope) separately. H3-specific ops belong with the target or family that uses them.

### Choose the sharing scope

| Owner | When to use it | Example |
| --- | --- | --- |
| Target (`models/<family>/<target>/`) | The [router](pipeline_registry.py) selects this implementation for a checkpoint or deployment. Keep code specific to that target in its directory. | Base H3's TP text encoder or FastH3's trained schedule. |
| Model family (`models/<family>/`) | The family owns H3 routing and common model behavior. Put code shared by sibling targets here so each can import it independently. | Joint token packing, DiT blocks and gated VSA attention. |
| [VisualGen V2 catalog](../../catalog/) (`catalog/<category>/`) | Own a model-independent op with a reference and tests for its supported configurations. | Fine attention over explicit KV tile indices, counts and valid lengths. |
| Existing runtime or library | Call its existing interface directly when the contract fits. | H3 [projections](components/linear.py) use the LLM V2 GEMM ops and [normalization](components/norm.py) uses `flashinfer_rmsnorm`. |

The [LLM V2 catalog](../../../modeling_v2/README.md#code-layout) holds ops used by multiple families. VisualGen allows the first V2 consumer to introduce an op if another family could use its contract unchanged. Write its reference using only the explicit call arguments, including any buffers the op updates. Keep checkpoint-, layer- and timestep-specific decisions in the model that supplies those arguments.

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
| Change which rows or heads a rank receives, or how its receive buffers are laid out | The [H3 QKV ops](ops/ulysses_qkv.py) define and test the exchange and receive-layout contracts. |

```text
Base H3 / H3UlyssesAttention (family component)
  sync   pack Q/K/V by rank -> all-to-all -> arrange received Q/K/V
  async  project/pack/push V, Q, K -> wait -> arrange received Q/K/V

  both   local attention (full sequence, local heads)
         -> output all-to-all -> local sequence shard
```

The synchronous exchange returns packed QKV when the backend accepts it. Otherwise it arranges separate Q/K/V tensors in the backend's sequence-major (NHD) or head-major (HND) order. The asynchronous path calls `ulysses_unscatter_qkv` on the receive buffers after all three pushes complete.

The [sequence sharder](components/sequence_sharder.py) pads, shards and gathers DiT and refiner rows. The [tiled VAE](components/tiled_vae.py) distributes video decode tiles. Base H3 uses these parallel paths. FastH3 runs on one GPU. The V1 [token-sharded TP layout](../../parallel/token_sharded_tp.py) is a separate feature used by Wan.

### How large is a catalog operation?

Group steps into a catalog op when another model family could use their combined result and buffer updates unchanged. Define that behavior in a reference that takes only the explicit call arguments. The implementation may fuse the steps into one kernel or use several kernels and collectives. The [LLM catalog](../../../modeling_v2/catalog/) illustrates the range.

| LLM example | Contract | VisualGen implication |
| --- | --- | --- |
| [RMSNorm](../../../modeling_v2/catalog/norm/flashinfer_rmsnorm.py) | Given activations, weights and epsilon, return the normalized activations. | A single kernel can be an entry when V2 families use it unchanged. |
| [Reduce-scatter](../../../modeling_v2/catalog/comm/reducescatter.py) | Reduce contributions and return the local slice for an explicit group. The implementation can dispatch by communicator. | Share the collective when layouts agree. Each family decides where it runs and what work overlaps it. |
| [MLA RoPE and KV append](../../../modeling_v2/catalog/attention/mla_rope_append_paged_kv_assign_q.py) | Rotate new tokens and write specified rows to an explicit KV pool. The reference describes the resulting buffer contents. | Several steps can be one operation when independent callers need the same outputs and mutations. |

| VisualGen V2 entry | Catalog computes | H3 supplies or owns |
| --- | --- | --- |
| [FA4](../../catalog/attention/flash_attention_4.py) | Attention output and logsumexp from Q/K/V, scale, causal mode and optional valid KV lengths | Convert trailing-padding masks to prefix lengths and restore the model's output dtype |
| [FlashInfer prefill](../../catalog/attention/flashinfer_prefill.py) | Select a compatible provider kernel and compute prefill with explicit quantization formats, mask and runner | Create the runner for the loaded instance and pass its plan cache and the selected precision |
| [CuTe FMHA](../../catalog/attention/cute_fmha.py) | Full attention on unpadded, equal-head BSHD tensors with an optional quantization recipe and skip-softmax threshold | Choose the recipe and per-step threshold. H3's SOL path uses masked SDPA when a padding mask is present. |
| [Fine block-sparse attention](../../catalog/attention/block_sparse_attention.py) | Evaluate attention on explicit KV tile indices, counts and valid tile lengths | Construct text/video/audio routes and combine the result with coarse attention using the learned gate |
| [TRTLLM context attention](../../catalog/attention/trtllm_context_attention.py) | Full noncausal attention for equal-length sequences through the native context kernel | Prepare packed or separate Q/K/V, share workspaces across layers and keep skip counters per attention instance |
| [PrimTS dense](../../catalog/attention/prims_dense_attention.py) and [routed attention](../../catalog/attention/prims_routed_attention.py) | Quantized attention or attention over explicit sparse routes and optional proxy summaries | Select the recipe, build routes, own the plans and decide when to refresh VC grouping |
| [SOL block prediction](../../catalog/attention/sol_predictor.py) | Predict exact-block bits and pooled K/V summaries from Q/K/V, scale and threshold | Choose the active layers and denoising steps, then pass the resulting routes to attention |
| [Fused gate/up GEMM and SwiGLU](../../catalog/gemm/gate_up_swiglu_quack.py) | Compute `silu(x @ W_gate.T) * (x @ W_up.T)` for BF16 inputs and packed gate/up weights | Own the projections, select when to fuse and run the down projection and its TP reduction |

FlashInfer uses *prefill* for attention over a sequence of query rows. H3 calls it for noncausal self-attention during denoising. Existing [LTX-2.3 FlashInfer deployments](../../../../../examples/visual_gen/configs/ltx23-t2v-fp8-1gpu.yaml) use the V1 backend.

The [H3 attention component](components/dense_attention.py) adapts model inputs to these contracts. The [gated VSA computation](ops/vsa.py) serves both base checkpoints with gate weights and explicit VSA configuration and the FastH3 target.

The same boundary applies at weight load time. H3 chooses which projection rows to fuse or shard and which weights keep their checkpoint precision. The [weight quantization ops](../../catalog/quantization/) turn an explicit weight tensor into FP8 or NVFP4 data and scales.

### Backend dependencies

For new H3 code, use an exported catalog op or runner, or a supported kernel-library interface. For example, H3 creates `FlashInferPrefillRunner` from the [FlashInfer catalog entry](../../catalog/attention/flashinfer_prefill.py) and supplies the per-model plan cache. The [FA4 entry](../../catalog/attention/flash_attention_4.py) handles compatibility with the provider’s private `_flash_attn_fwd` binding.

When an interface is missing, define the computation and its reference at the narrowest scope that needs it. Model-specific mask conversion, layer selection and denoising policy stay in H3. See the [Modeling V2 operation contract](../../../modeling_v2/README.md#writing-operations).

The [TRTLLM context op](../../catalog/attention/trtllm_context_attention.py) calls the native FMHA context entry and uses its runtime metadata schema. The PrimTS ops use FlashInfer's public plan and execution interfaces. Most paths use the [vendored provider](../../../../../3rdparty/vendor_sources.lock.yaml). Dense value-compression repair on a partial KV tile uses the [pinned FlashInfer package](../../../../../requirements.txt) so padding is excluded from softmax. H3 owns layer selection, denoising thresholds and the lifetime of each plan cache. See the [Modeling V2 dependency rule](../../../modeling_v2/README.md#modeling-v1-code).

## Test a change

Mirror family and target code in the [unit tests](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/) and give each routed target a [whole-pipeline quality case](../../../../../tests/integration/defs/visual_gen/minimax_h3_v2/). Mirror [family operations](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/ops/) and [catalog entries](../../../../../tests/unittest/_torch/visual_gen/catalog/) at their respective scopes. Compare outputs and buffer mutations against each operation’s reference on its declared cells. Register pre-merge tests by file or directory in the [CI test lists](../../../../../tests/integration/test_lists/test-db/) for the required hardware and rank count. FastH3's real-checkpoint quality cases run on the [functional QA schedule](../../../../../tests/integration/test_lists/qa/functional/llm_function_core.txt). Change-based test selection operates on registered entries.

A target change exercises that target. A shared family change exercises its consumers. Registry changes exercise both checkpoint targets and both switch settings. A catalog change runs its mirrored contract tests and the V2 targets that import the operation. V1 paths retain their existing tests.

For each deployment tested, run the same request in separate V1 and V2 processes on the same GPUs. Fix the checkpoint revision, conditioning inputs, seed, weight and attention precision, backend, parallel layout, compilation settings and sigma schedule. Compare the two decoded videos and audio waveforms directly. A multi-GPU run against one GPU measures topology sensitivity. A Diffusers comparison measures reference quality. Report these separately from refactor parity. If the V1 reference has a reproducible defect, record the original comparison and show that V2 matches the checkpoint or an independent numerical reference for the affected computation. Use that reference to qualify the correction. Use 50 points for the base-H3 quality reference and test omitted defaults separately. Before retiring V1 and the opt-in switch, qualify generated video and audio, supported parallel paths, warmed latency and peak memory against V1 on representative deployments.
