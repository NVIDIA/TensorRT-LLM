<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# VisualGen Modeling V2

Keep changes for one deployment local to its pipeline so developers can merge work on different deployments independently. A **target** is a pipeline for a checkpoint or deployment. A **family** owns the code its targets share. The public VisualGen API and CLI work across implementations. [LLM Modeling V2 guide](../../modeling_v2/README.md)

## Code layout

| Location | What belongs here |
| --- | --- |
| `visual_gen/modeling_v2/models/<family>/<target>/` | The pipeline and the parts of loading or execution specific to that target. |
| `visual_gen/modeling_v2/models/<family>/` | The checkpoint router and components, ops or kernels that sibling targets import from the family. |
| `visual_gen/modeling_v2/catalog/` | **Model-agnostic operations only.** Another family can pass its own tensors to the same op unchanged. Each entry defines its inputs and results, provides a reference and tests its implementations. |

### When to split or merge a target

**Split when a checkpoint or deployment needs a substantially different implementation.** Look for deployment checks spread across components or a new mechanism within a single stage. For example, a checkpoint that needs different modules and a new shard allocation order can own its loader and still reuse the family’s forward pass. If two targets later converge on the same implementation, merge them and keep tests for the deployments they support.

**Name the target for what makes it distinct.** Checkpoint, quantization, SM and parallel layout are optional dimensions. For example, `fasth3/` identifies a checkpoint. A hardware-specific target can include the SM in its name. If Blackwell and Rubin differ only in which kernel a projection calls, keep one target and select the kernel in that projection.

### Implementation layers

| Layer | Responsibility |
| --- | --- |
| **Pipeline** | Runs request stages in order, such as encoding, denoising and decoding. |
| **Component** | Owns part of the model and coordinates its work. An attention component forms Q/K/V from its weights, applies RoPE, handles communication and projects the output. |
| **Op** | Computes a result from explicit arguments. An attention op takes prepared Q/K/V and masking options. It can call several kernels or collectives. |
| **Kernel** | Implements a calculation on the GPU for particular shapes, layouts or hardware. |

**A component owns a model decision. An op owns a calculation.** Changing how a block assigns heads to ranks belongs in the component. A kernel that computes the same attention result from prepared tensors can be tested against the op’s reference. Extract an op when you can specify its result from its arguments and have a reason to implement or test it separately.

Keep parallel rules with their caller. Create a component when several operations must coordinate computation and communication. [H3 attention and Ulysses](models/minimax_h3/README.md#attention-and-parallelism)

## Where to add a feature

Check the family’s V2 targets and its migration plan before choosing where to implement a feature. The [V2 models directory](models/) lists the migrated families.

| Model family | Where to develop |
| --- | --- |
| New family | **Prefer V2** for the initial implementation. |
| V1 only | The model owner decides whether to migrate. Development can continue in V1. |
| V2 target available, V1 retiring soon | **Add new features only in V2.** MiniMax-H3 follows this policy. |
| V1 and V2 will both remain in active use | When the model owner considers a new feature important for the family, **support and test it on every supported V1 and V2 target**. A deployment-specific optimization can stay in its target. |

## Routing

**The family router selects one pipeline before model construction.** Use the checkpoint and the deployment settings that distinguish its targets. Return the selected target class directly so CI can identify its tests. The pipeline constructs its components, and each component or op selects its implementation for the hardware and inputs.

- **V2-only family.** Register the pipeline through the usual VisualGen entry point. Users need no version switch.
- **Full migration with a short V1 rollout.** A temporary family switch can select V2 while V1 remains the default. Within V2, use the functional target for the same checkpoint when a deployment specialization does not match. [H3 rollout](models/minimax_h3/README.md#run-a-target)
- **Partial migration with ongoing V1 support.** A family can route deployments covered by a V2 target to V2 and keep the existing V1 pipeline for the others. Make the choice at load time and test both routes.

## Allowed dependencies

**Prefer a self-contained V2 implementation from op to kernel.** Check the LLM and VisualGen V2 catalogs first. Use the VisualGen runtime to read checkpoint tensors by name, manage process groups and autotune kernels. Public third-party APIs such as PyTorch and FlashInfer are also available. If no op fits, start with the target or family that needs it. Promote an op to the VisualGen catalog when its calculation and tests are independent of the model. A single family can be the first caller.

**Treat imports from a V1 backend as an exception.** Reuse a complex, mature, model-agnostic kernel or runner, typically attention or GEMM, when a second implementation would duplicate substantial work. Keep checkpoint interpretation, input preparation and per-layer policy in V2. Write or port a kernel when the available implementations miss a required result, layout, hardware capability or measured performance goal. Build new native code under its owning catalog, family or target. Reuse a shared kernel’s registration and give new V2 registrations distinct names. [Native build guide](../../../../cpp/tensorrt_llm/visual_gen/README.md)

List every V1 backend exception in the migration PR, or state that there are none. Trace construction, loading, execution and fallbacks through runtime adapters, catalog calls and registered Torch ops or classes. Identify the V2 caller, the provider path and symbol, and the capability that justifies reuse.

## Test a change

- **Test the affected code.** Check optimized ops and fallbacks against the same reference, including output layouts and in-place writes. Test each target that consumes changed shared code. Mirror source ownership in unit and quality tests and register them with CI. Test each router match and fallback through the public loading path.
- **Verify opt-in features during migration.** For each migrating target, list the opt-in features V1 supports for that checkpoint and deployment. Map each feature to a test that enables it, confirms the requested path ran and checks the behavior users rely on. Include combinations that take a distinct implementation path. Record the result or the remaining gap in the PR. An existing test or end-to-end run can cover several features.
- **Compare generation separately.** Run V1 and V2 in separate processes on the same GPUs. Match the checkpoint, inputs, seed, precision, attention backend, parallel layout, compile settings and actual denoising schedule. Compare decoded media and warmed latency. For bit-exact compiled comparisons, prewarm the caches with both versions and restore the same snapshot at the same path before each run.
- **Before retiring V1**, close gaps for features the family will continue to support, account for features added during coexistence, and compare peak memory on representative deployments.

See the [VisualGen engineering criteria](../ENGINEERING_CRITERIA.md) for feature and CI requirements. The [H3 migration](models/minimax_h3/README.md) shows how these rules apply to two checkpoints.
