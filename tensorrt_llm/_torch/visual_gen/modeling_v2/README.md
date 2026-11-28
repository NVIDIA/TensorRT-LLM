<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# VisualGen Modeling V2

Keep changes for one deployment local to its pipeline so developers can merge work on different deployments independently. A **target** is a pipeline for a checkpoint or deployment. A **family** owns the code its targets share. The public VisualGen API and CLI work across implementations. [LLM Modeling V2 guide](../../modeling_v2/README.md)

## Architecture

### Code layout

| Location | What belongs here |
| --- | --- |
| `visual_gen/modeling_v2/models/<family>/<target>/` | The pipeline and the parts of loading or execution specific to that target. |
| `visual_gen/modeling_v2/models/<family>/` | The checkpoint router and components, ops or kernels that sibling targets import from the family. |
| `visual_gen/modeling_v2/catalog/` | **Model-agnostic operations only.** Another family can pass its own tensors to the same op unchanged. Each entry defines its inputs and results, provides a reference and tests its implementations. |

#### When to split or merge a target

**Split when a checkpoint or deployment needs a substantially different implementation.** Look for deployment checks spread across components or a new mechanism within a single stage. For example, a checkpoint that needs different modules and a new shard allocation order can own its loader and still reuse the family’s forward pass. If two targets later converge on the same implementation, merge them and keep tests for the deployments they support.

**Name the target for what makes it distinct.** All fields in `<checkpoint>_<quantization>__<sm>__<parallel>/` are optional. For example, `fasth3/` identifies a checkpoint, and `minimax_h3_nvfp4__sm_107__u4/` could identify an NVFP4 implementation for SM107 with four-way Ulysses. If two deployments differ only in which kernel a projection calls, keep one target and select the kernel in that projection.

### Implementation layers

| Layer | Responsibility |
| --- | --- |
| **Pipeline** | The inference entry point for a checkpoint or deployment. It runs the stages needed to serve a request. |
| **Component** | A major building block of the pipeline, such as the text encoder, denoiser or video decoder. |
| **Op** | A fine-grained unit of work with explicit inputs and observable results. |
| **Kernel** | Code that runs on hardware to perform computation, move data or communicate between devices. [Native build guide](../../../../cpp/tensorrt_llm/visual_gen/README.md) |

**The boundary between a component and an op is flexible. In general, a component implements part of the model’s behavior, and an op defines a smaller unit of work.** Extract an op when you can describe its behavior from its inputs and have a reason to test it independently or replace its implementation.

Keep parallel rules with their caller. Create a component when several operations must coordinate computation and communication. [H3 attention and Ulysses](models/minimax_h3/README.md#attention-and-parallelism)

### Routing

**The family router selects one pipeline before model construction.** Use the checkpoint and the deployment settings that distinguish its targets. Return the selected target class directly so CI can identify its tests. The pipeline constructs its components, and each component or op selects its implementation for the hardware and inputs.

- **V2-only family.** Register the pipeline through the usual VisualGen entry point. Users need no version switch.
- **Full migration with a short V1 rollout.** A temporary family switch can select V2 while V1 remains the default. Implement the switch in the family router passed to `register_pipeline(..., router=...)`. Within V2, use the functional target for the same checkpoint when a deployment specialization does not match.
- **Partial migration with ongoing V1 support.** A family can route deployments covered by a V2 target to V2 and keep the existing V1 pipeline for the others. Make the choice at load time and test both routes.

### Allowed dependencies

- **Prefer a self-contained V2 implementation from op to kernel.** Check the LLM and VisualGen V2 catalogs first. Use VisualGen runtime services and public third-party APIs directly.
- **Keep V1 model and backend implementations unchanged during migration.** Keep model decisions in the V2 family or target and put each adapter with its V2 caller.
- **Treat imports from a V1 backend as an exception.** Reuse a mature, model-agnostic kernel or runner through its existing interface when reimplementing it would duplicate substantial work. Attention and GEMM are common examples. Write or port a V2 kernel if existing implementations fall short on results, layout, hardware support or measured performance.
- **List every V1 backend dependency in the migration PR for review.** Give the V2 caller, V1 file and symbol, and reason for reuse. Include indirect calls and fallbacks. If there are no exceptions, say so.

## Feature development during migration

- **New model families use V2.** Discuss exceptions with the model owner case by case.
- **For an existing model, develop in the implementation its family uses.** A V1-only family can keep developing in V1, and its owner decides when to migrate. Check the [V2 families](models/) and the family’s migration plan.

When V1 is retiring soon, **add new features only in V2**. When V1 and V2 will both remain in active use, **support important family-wide features on every supported V1 and V2 target**. A deployment-specific optimization can stay in its own target.

## Test a change

- **Test the affected code.** Check optimized ops and fallbacks against the same reference, including output layouts and in-place writes. Test each target that consumes changed shared code. Mirror source ownership in unit and quality tests and register them with CI. Test each router match and fallback through the public loading path.
- **Keep matching tests in V1 and V2 during migration.** Give each V2 target its own coverage of supported behavior so its tests can run after V1 retires.
- **Verify opt-in features during migration.** For each migrating target, list the opt-in features V1 supports for that checkpoint and deployment. Map each feature to a test that enables it, confirms the requested path ran and checks the behavior users rely on. Include combinations that take a distinct implementation path. Record the result or the remaining gap in the PR. An existing test or end-to-end run can cover several features.
- **Compare generation separately.** Run V1 and V2 in separate processes on the same GPUs. Match the checkpoint, inputs, seed, precision, attention backend, parallel layout, compile settings and actual denoising schedule. Compare decoded media and warmed latency. For bit-exact compiled comparisons, prewarm the caches with both versions and restore the same snapshot at the same path before each run.
- **Before retiring V1**, close gaps for features the family will continue to support, account for features added during coexistence, and compare peak memory on representative deployments.

See the [VisualGen engineering criteria](../ENGINEERING_CRITERIA.md) for feature and CI requirements. The [H3 migration](models/minimax_h3/README.md) shows how these rules apply to two checkpoints.
