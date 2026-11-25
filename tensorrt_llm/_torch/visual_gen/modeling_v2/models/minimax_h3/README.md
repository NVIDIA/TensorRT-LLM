<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing MiniMax-H3 with Modeling V2

**Write new H3 features and kernels only in V2.** H3 is retiring [V1](../../../models/minimax_h3/), which remains the default during rollout and is available for fixes and comparisons. A *target* is the pipeline for a particular checkpoint or deployment. Base H3 and FastH3 are targets in the same family under `visual_gen/modeling_v2/models/minimax_h3/`. V1 and V2 use the same VisualGen API, CLI and checkpoint names. [VisualGen migration policy](../../README.md#where-to-add-a-feature) · [LLM Modeling V2 guide](../../../../modeling_v2/README.md)

## Code layout

| Location | What belongs here |
| --- | --- |
| `visual_gen/modeling_v2/models/<family>/<target>/` | Code **specific to one checkpoint or deployment**. |
| `visual_gen/modeling_v2/models/<family>/` | The router and **H3 code used by multiple targets**. Each target imports shared components from the family. |
| `visual_gen/modeling_v2/catalog/` | **Model-agnostic operations only.** Another family must be able to pass its own tensors to the same op without adding model-specific branches. H3 can be the first caller. The catalog defines the inputs and expected results, provides a reference, and tests each implementation against it. |

### When to split a target

**Split a target when a checkpoint or deployment needs a substantially different implementation.** It might need a different loading or execution mechanism, or enough conditional branches that two separate paths are easier to maintain. This can happen within a single stage.

For example, an existing loader can unpack FP8 weights and convert their scales. If a checkpoint needs different modules, different allocations and a different order for loading shards, its own target can handle loading and still reuse the H3 forward pass. [FastH3](fasth3/pipeline.py) has its own target because it selects gated attention and uses the checkpoint’s trained denoising schedule.

Use a checkpoint name for a target that supports several deployments, such as `fasth3/`. Name a deployment-specific target `<checkpoint-and-quantization>__<sm>__<parallel>/`. In [`route(config, checkpoint_dir)`](pipeline_registry.py), check precision, hardware, parallel layout and required features. Return the target class directly so CI can find it. If the deployment does not match, **return the default target for that checkpoint**. The selected pipeline makes decisions that depend on the request or denoising step.

### Implementation layers

A family or target can own all four layers. H3 keeps its model blocks in `components/`, numerical operations in `ops/` and GPU kernels in `kernels/`.

| Layer | What it does | H3 example |
| --- | --- | --- |
| **Pipeline** | Runs the stages of a request in order. | The [H3 pipeline](pipeline.py) prepares inputs, runs the denoising loop and decodes video and audio. |
| **Component** | Holds a model block’s weights or state and decides which ops to call. | [DiT attention](components/attention.py) projects Q/K/V, owns the normalization weights and selects the fused implementation. |
| **Op** | Defines what to compute from the supplied arguments. It can call several kernels or collectives. | The [Q/K norm + RoPE op](ops/rope.py) returns normalized, rotated Q and K. Attention takes V from the projection. |
| **Kernel** | Implements the calculation on the GPU. | The [Triton kernel](kernels/fused_rope.py) combines per-head RMSNorm and rotation in one launch. |

Write or port a kernel when existing V2 ops and library APIs do not support the required layout or hardware, compute a different result, or miss a measured performance target. **Put it with the op that owns the computation.** Build new C++/CUDA ops with their catalog, family or target and register them under its V2 namespace. Reuse the existing registration when calling a shared kernel. [Native build guide](../../../../../../cpp/tensorrt_llm/visual_gen/README.md)

### How a component differs from an op

**A component owns part of the model.** The [H3 DiT attention component](components/attention.py) takes hidden states and returns the layer output. It owns the projection and normalization weights, forms Q/K/V, applies Q/K normalization and RoPE, and applies the output projection after attention. When Ulysses is enabled, it uses the [Ulysses component](components/ulysses_attention.py) to coordinate the exchanges around attention.

**An op owns a specific calculation.** The [FA4 attention op](../../catalog/attention/flash_attention_4.py) takes prepared Q, K and V, a softmax scale and masking options. It computes attention and returns the weighted values and log-sum-exp. You can test that calculation directly with tensors. Changing how H3 projects its hidden states or distributes heads across ranks belongs in the component.

### When to extract an op

Extract an op when you can **describe its result from its arguments and have a reason to implement or test it separately**. For H3’s Q/K transform, the caller supplies the projected tensors, normalization weights and rotation tables. The [op test](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/models/minimax_h3/ops/test_rope.py) runs the fused kernel and a plain PyTorch calculation on the same inputs and compares their Q and K outputs.

Changing which channels rotate changes the expected result, so update the reference and the implementation together. When tuning GPU tile size, keep testing against the existing reference.

### Where parallel behavior belongs

**Keep each parallel rule in the component that uses it.** With tensor parallelism, [attention](components/attention.py) assigns heads to ranks and [Linear](components/linear.py) owns the weight shards and output reduction.

**Use a separate component when communication and computation must run in a particular order.** [H3UlyssesAttention](components/ulysses_attention.py) coordinates both exchanges around attention. On the async path it starts each transfer as its projection becomes ready.

```text
TP       project local heads -> attention -> output projection and sum across ranks
Ulysses  local tokens -> exchange Q/K/V -> compute attention for local heads over the full sequence -> return exchange
```

## Allowed dependencies

**Keep model implementations self-contained in V2, from op to kernel.** First look for an op in the LLM or VisualGen V2 catalog. If none fits, implement the op and any kernel it needs in V2. Put model-specific code in the H3 family or target, and model-agnostic ops that other families can use unchanged in the VisualGen V2 catalog.

H3 can also use public third-party APIs such as PyTorch and FlashInfer, and VisualGen runtime services such as process groups and autotuning. The runtime reads weight files and returns tensors by name. H3 decides how to load those tensors into its layers.

**Treat imports from a V1 backend as an exception.** Consider one only for a **complex, mature, model-agnostic kernel or runner**, typically attention or GEMM, when maintaining a second implementation would duplicate substantial work. H3 owns input preparation, checkpoint handling and the choice of which layers or steps use the kernel.

**List every V1 backend dependency in the PR description.** For a migration, check model construction, weight loading and execution. Follow catalog calls, runtime adapters, fallbacks and registered `torch.ops` or `torch.classes` names to their implementations. For each exception, name the V2 caller, the repository path and symbol it reaches, and the specific capability that justifies reuse. Reviewers check that each entry point and its fallbacks keep model decisions in V2. If the audit finds no exceptions, say so.

For example, the [V2 SOL predictor](../../catalog/attention/sol_predictor.py) calls [`visual_gen_sol_predictor`](../../../attention/backends/sparse/sol/kernels.py) to compute block predictions from Q/K/V. H3 decides which layers and denoising steps use those predictions. Other shared implementations include [CuTe sparse kernels](../../../cute_dsl_kernels/blackwell/video_sparse_attention/), [PrimTS runners](../../../../attention/backends/prims_ts/) and the [block-scaled FP8 GEMM](ops/quantized_gemm.py) that uses the shared Blackwell low-M kernel.

## Run a target

**Set `TLLM_VISUALGEN_MODELING2_H3=1` on every worker before startup.** Set `H3_CHECKPOINT` to your checkpoint directory and run from the repository root.

```bash
TLLM_VISUALGEN_MODELING2_H3=1 python examples/visual_gen/models/minimax_h3.py \
    --model "$H3_CHECKPOINT" \
    --visual_gen_args examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml
```

The [router](pipeline_registry.py) selects a target before building the transformer. Each worker logs its choice in `AutoPipeline: Creating`.

| Checkpoint | Target | Denoising steps |
| --- | --- | --- |
| Base H3 | [default_pipeline/](default_pipeline/) handles BF16, FP8, block-scaled FP8 and NVFP4 | The default is 50 sigma points including zero, giving 49 updates. |
| FastH3 (`schema_version` starts with `fasth3-` in `fastvideo_inference.json`) | [fasth3/](fasth3/) | The checkpoint supplies the schedule. FastH3 8-Step V2 uses nine points and eight updates. |

## Test a change

During the H3 rollout, include a change to an existing shared implementation only when the same supported configuration would otherwise behave differently in V1 and V2. Track defects inherited by both versions separately.

- **Check each op and kernel.** Test the optimized path and any fallback against the same reference on supported shapes, dtypes and layouts. Check returned values and any tensors modified in place. Mirror the source layout in the [H3 unit tests](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/models/minimax_h3/) or the [catalog unit tests](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/catalog/).
- **Test the affected targets.** Run the tests for the target you changed. For shared components and ops, **test every target that uses them**. Involve shared-code owners and include existing V1 or LLM consumers when changing their kernel. Give each target a [pipeline quality test](../../../../../../tests/integration/defs/visual_gen/modeling_v2/minimax_h3/).
- **Exercise the router.** Check that [`AutoPipeline.resolve_class`](../../../pipeline_registry.py) selects a matching specialization, falls back to the right checkpoint target, and selects V1 when the opt-in switch is unset.
- **Check new imports.** When a V2 op or kernel launcher imports a shared backend function, add the V2 file and the specific function or class it imports to the [dependency test](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/models/minimax_h3/test_dependencies.py).
- **Register tests with CI.** Add tests to the [CI lists](../../../../../../tests/integration/test_lists/test-db/) or [functional QA schedule](../../../../../../tests/integration/test_lists/qa/functional/llm_function_core.txt). For a `torch.ops` call, find the code that registers the op name. If CI cannot infer the dependency, map that source file to its consumers’ tests in the [VisualGen CI rule](../../../../../../jenkins/scripts/cbts/cbts/rules/visual_gen.py).

**For an end-to-end comparison, run V1 and V2 in separate processes on the same GPUs.** Match the checkpoint, inputs, seed, precision, attention backend, parallel layout, compile settings and actual denoising schedule. For bit-exact comparisons with `torch.compile`, warm the compiler and kernel caches with both versions, then restore the same snapshot at the same path before each run. Compare decoded video, audio and warmed request latency. For a loading fix, check the weights against the checkpoint and repeat the original request. Before retiring V1, also compare peak memory on representative deployments.
