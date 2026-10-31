<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Modeling V2

Modeling V2 is designed to accelerate model development and performance optimization
in the agent era. Coding agents make specialized code cheaper to produce, while
understanding, verifying, reviewing, and maintaining changes across a highly shared
framework increasingly become bottlenecks. Modeling V2 organizes model logic and
specialized kernels around isolated model families and deployment targets, while
reusing TensorRT-LLM’s serving runtime and general-purpose operators. This gives
developers and agents focused goals, makes the computation easier to follow, and
reduces the impact of model-specific changes on unrelated models, accepting some
code duplication in exchange for faster iteration. For example, the DeepSeek R1
target owns its weight loading and explicitly composes its MLA projections,
normalization, RoPE, and attention operations, allowing its execution path to be
optimized within the target.

## Using it

Set the `modeling_v2` LLM API flag to choose the model implementation:

| `modeling_v2` | Behaviour |
|---|---|
| `off` (default) | Use the existing Modeling V1 implementation; disable Modeling V2. |
| `auto` | Use a matching Modeling V2 implementation when available; otherwise use Modeling V1. |
| `require` | Require a matching Modeling V2 implementation; raise an error if none is available. |

For the Python API:

```python
LLM(model, modeling_v2="require")
```

For `trtllm-serve`, set the same flag in the serving configuration:

```yaml
# trtllm-serve --config serve.yaml
modeling_v2: require
```

Use `require` when validating or benchmarking a Modeling V2 implementation
so the run cannot fall back to Modeling V1.

## Families and targets

A **model family** groups related model implementations and owns their routing
and family-shared code. A useful starting point is a model architecture in
Hugging Face Transformers. Developers can choose a broader scope by grouping
related architectures, or a narrower scope by specializing around a subset of
models or checkpoints, according to their implementation and ownership needs.

A **target** is a self-contained model implementation selected by routing or
dispatch for a defined set of supported configurations or execution conditions.
Self-contained describes its ownership and isolation: a target can reuse
family-shared code, catalog operations and runtime services.

The family/target hierarchy is part of the design: a family lives under
`modeling_v2/models/<family>/`, and its targets live under
`modeling_v2/models/<family>/<target>/`. The package name identifies the
implementation; its routing and dispatch logic define the conditions it supports.

In this guide, "target" refers to the whole implementation package. A target may
contain runtime specializations for different execution paths; classes named
`PrefillTarget` and `DecodeTarget` are examples of specializations within a target.

## Designing a target

Choose a target's scope around the checkpoint, GPU architecture,
parallel topology and features that affect the computation. A target can cover
multiple configurations when they share an implementation; introduce a specialized
target when a deployment needs a different execution path.

For example, the
[DeepSeek R1 NVFP4 target](models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/)
belongs to the `deepseek_v3` family and supports SM100 and SM103 across the
parallel layouts admitted by its router.
Its `default` suffix identifies an implementation that handles multiple layouts.

Keep model construction, forward execution, weight loading and specialized kernels
within the owning family or target. Reuse runtime services for scheduling, cache
management, distributed execution, sampling and serving.

### Model lifecycle

Targets integrate with TensorRT-LLM through the existing `DecoderModel` and
`DecoderModelForCausalLM` lifecycle interfaces. Targets can inherit, override or
delegate their implementations as needed:

* **Construction:** establishes the model structure and parameters.
* **`load_weights`:** loads checkpoint weights into the model. Weight mapping,
  transformation and sharding may be handled by loaders or helpers.
* **`post_load_weights`:** provides post-load preparation, such as weight transforms
  and derived execution state. A default implementation is available; override or
  extend it when the target needs additional preparation.
* **`forward`:** executes the model using the inputs and metadata supplied by the
  engine.

The conventional target files are `modeling.py` for construction and execution,
and `weights.py` for checkpoint mapping, loading and sharding. Keep the operator
sequence and its arguments easy to follow. Families may use helpers, modules and
separate prefill/decode implementations when those improve clarity or performance.
See the [GPT-OSS implementation](models/gpt_oss/gpt_oss_120b__sm_10x__default/modeling.py)
and its [weight loader](models/gpt_oss/gpt_oss_120b__sm_10x__default/weights.py)
for an example.

### Routing

Each model family defines a `route` function in its `routing.py`; see
[DeepSeek R1 routing](models/deepseek_v3/routing.py) for an example. It selects
a target from the model configuration, GPU architecture, parallel topology
and relevant engine options, or raises `NoTarget` if none matches. Reject
unsupported configurations with a reason that explains which requirement failed.

The loader applies model defaults before resolving the target and constructs the
selected class through the model lifecycle. Targets are selected through their
family's router, alongside the existing Modeling V1 model registry.

```
LLM(model=..., modeling_v2=...)  ->  ModelLoader
                                        |  model defaults applied to llm_args
                                        |
                                        +-- modeling_v2_resolve(config, llm_args)
                                              |
                    _router_index.py          +-- architectures[0] -> routing module
                    models/<x>/routing.py     +-- route(model_config, llm_args,
                                              |         compute_capability)
                                              |         -> target class, or NoTarget
                                        |
                                     config.modeling_v2_target = class
                                        |
                    AutoModelForCausalLM._resolve_class(config)
                                        |
                                     returns that class; the registry is not consulted
```

Use engine-lifetime configuration to select a deployment target. Handle changing
batch composition and prefill/decode decisions inside the target.

Routing uses checkpoint configuration fields to recognize a supported model shape.
A fine-tuned checkpoint with the same shape may select the same target, but its
accuracy still needs to be evaluated on that checkpoint.

## Code organization and ownership

Use three ownership scopes:

| Scope | Use for | Location |
|---|---|---|
| Target | Code specific to a deployment target | `models/<family>/<target>/` |
| Family | Code shared by targets in one model family | `models/<family>/` |
| Catalog | Operations shared across model families | `catalog/<category>/` |

Preserve isolation between unrelated families and between sibling-target
implementations. Share family code at family scope instead of importing it from a
sibling target. Code duplication is acceptable when it keeps targets independent;
consolidation is a family-level decision.

### Shared structure

The top-level structure defines the shared architecture, ownership boundaries and
registration points. Keep cross-family operations in `catalog/`, model families
in `models/<family>/`, and targets within their owning family.

```text
modeling_v2/
├── _router_index.py                 # Family registration and target resolution
├── _native.py                       # Native family-library loading
├── catalog/
│   ├── _op.py                       # Operation contract and test-cell definitions
│   └── <category>/                  # Operations shared across families
└── models/
    └── <family>/
        ├── routing.py               # Target selection
        └── <target>/                # Target-owned implementation
```

Target directory names contain the checkpoint, GPU architecture and parallel
configuration, for example `gpt_oss_120b__sm_10x__default`. The target class name
carries the same identity. Register a new family's route with `@routes(...)` and
import that routing module in `models/__init__.py`.

### Family and target internals

Within a family or target, choose the organization that suits its implementation.
Use the following layout as a starting convention; wrappers and kernels may live
together, and families may introduce their own helpers and submodules.

```text
models/<family>/
├── ops/, kernels/                   # Optional family-shared implementation
└── <target>/
    ├── modeling.py                  # Model construction and execution
    ├── weights.py                   # Weight mapping, loading and sharding
    └── ops/, kernels/               # Optional target-owned implementation
```

Preserve the entry points used by routing and the model lifecycle. Native build
declarations, sources and fake registrations follow the locations required by the
build and loader; see the [native code guide](../../../../cpp/tensorrt_llm/modeling_v2/README.md).

Place tests under `tests/` at the ownership level corresponding to the code they
validate: shared catalog, family or target. Follow the code's internal layout
where useful; see [Testing and validation](#testing-and-validation) for the test
directory conventions.

## Writing operations

Use thin operation wrappers to make kernel calls and their contracts explicit.
The [OpWrapper interface](catalog/_op.py) defines:

| Member | Purpose |
|---|---|
| `__call__` | Invoke the operation with explicit arguments. |
| `reference` | Express the expected computation in PyTorch. |
| `is_valid` | Reject inputs known to be accepted by the underlying operation but computed incorrectly. |
| `CELLS` | Describe the input configurations used to validate the operation. |
| `ARCHS` | Declare the GPU architectures covered by that validation. |

Keep wrappers stateless: weights, workspace, batch state and configuration are
explicit call arguments. The target owns persistent execution state. Implement
`reference` and `is_valid` with compatible call signatures; a collective reference
may take the whole group's inputs when its result depends on other ranks.

`is_valid` is a test-time contract, not an automatic guard on the production call
path. Use the helpers in
[tests/_validating.py](../../../../tests/unittest/_torch/modeling_v2/_validating.py)
to exercise it. Derive numerical tolerances in the test, accounting for
accumulation order, quantization and output dtype. The reference should be at
least as precise as the computation it checks.

Extend the cells and tests when using an operation with new target-specific
shapes, layouts or dtypes. A configuration outside the validated cells is
unverified; acceptance by the kernel alone does not establish correctness.

## Native code

For model-owned C++/CUDA code, see the
[native code guide](../../../../cpp/tensorrt_llm/modeling_v2/README.md).
It covers source placement, family libraries, operation and fake registration,
build configuration and loading.

## Testing and validation

Keep tests and test-only helpers under `tests/`, organized by the ownership scope
of the code they validate. Within each family or target, the test layout can
follow the implementation's organization. The suggested locations are:

| Tests | Suggested location |
|---|---|
| Routing and package contracts | `tests/unittest/_torch/modeling_v2/test_modeling_v2_*.py` |
| Shared operation tests | `tests/unittest/_torch/modeling_v2/<category>/` |
| Family-shared tests | `tests/unittest/_torch/modeling_v2/models/<family>/` |
| Target-specific tests | `tests/unittest/_torch/modeling_v2/models/<family>/<target>/` |
| Whole-model accuracy tests | `tests/integration/defs/accuracy/modeling_v2/test_<family>.py` |
| Accuracy references | `tests/integration/defs/accuracy/references/` |

### Whole-model validation

Validate each target and feature path with `modeling_v2="require"`:

* **Boot:** construct the engine, load the weights and generate short deterministic
  continuations. Check weight coverage and basic execution.
* **Accuracy:** use the checkpoint's evaluation protocol and reference tolerance.
  Re-evaluate when changes affect model numerics, weight loading or execution.

Keep workload settings and measured results with the tests or evaluation report.

## FAQ

### Will duplication between models make the codebase unmanageable?

Some duplication is expected. Its cost should be evaluated across the full
software lifecycle: development, review, testing, optimization and maintenance.
Several factors can make that trade-off worthwhile:

* **Simpler implementations:** each family or target can focus on the configurations
  it supports, with fewer branches and abstractions to accommodate unrelated
  models. This can make its code easier to understand, review, test, debug and
  optimize, even when similar code exists elsewhere.
* **Agent-assisted maintenance:** coding agents can generate specialized code and
  help port fixes and features between implementations. This can reduce the cost of
  maintaining similar code; ported changes still need validation in each affected
  implementation.
* **Selective sharing:** reusable code can move to family scope, and standard
  operations used across families can move to the shared catalog. Sharing remains
  available when the behavior and interfaces are sufficiently common.

Model-specific optimizations can stay local without first being generalized.
Fixes that clearly benefit existing users of common code can still be made in
the shared implementation.

Duplication still adds maintenance work. The aim is to balance that work against
the clarity, isolation and faster iteration gained from independent implementations.
The benefit should be judged by the effort needed to develop and maintain the
implementations over time.

### How does Modeling V2 make code review easier?

Coding agents can produce large specialized implementations. For example, the
[Kimi K3 integration in PR #20893](https://github.com/NVIDIA/TensorRT-LLM/pull/20893)
contains over 70,000 added lines, including model-specific kernels, target
implementations and tests. Reviewing every generated kernel line in depth scales
poorly. Modeling V2 gives reviewers explicit ownership boundaries so they can
focus their effort according to the impact of each change.

* **Runtime interfaces and shared code:** review these carefully because changes
  can affect multiple consumers. This includes model/runtime contracts, common
  runtime behavior, and operations shared across targets or families.
* **Target structure:** review the implementation's scope, routing, weight loading
  and high-level execution flow. Confirm that model-specific changes remain
  isolated and that dependencies on shared components are explicit.
* **Isolated model-specific kernels:** use a more focused review centered on the
  operation contracts, test results, accuracy and performance evidence. Detailed
  inspection can concentrate on unclear assumptions, failures or suspicious
  results, without requiring the same line-by-line depth for every generated
  kernel as for shared infrastructure.

The reduced review scope depends on verified isolation and clear ownership.
Shared changes within a large model contribution still deserve careful review;
being part of a model-specific PR does not make those changes isolated.
Keep shared changes easy to identify and, when practical, review them separately
from the target integration.

### Do I have to use a particular agent or workflow?

No. The reference agent workflow is one way to develop Modeling V2 implementations.
Developers can use their own agents, skills or direct implementation. The same
flexibility applies to later fixes and optimizations.

What matters is that the implementation follows the shared interfaces, has clear
ownership and isolation, and has evidence supporting its correctness and performance.
Choose the workflow that helps you achieve these goals.

### Who owns and reviews a family or target?

The developers contributing a family or target should also take responsibility
for its maintenance, including its specialized kernels and tests. Record that
ownership in [CODEOWNERS](../../../../.github/CODEOWNERS).

Model owners review the implementation's content and quality. Modeling V2
maintainers focus on the shared design, interfaces and organization. Changes to
common runtime code or shared operations also need review from the owners of
those components, following the repository's normal review requirements.

### Does Modeling V2 replace V1 or require different user APIs?

Modeling V1 and V2 coexist. Adding a V2 implementation does not require migrating
existing V1 models. Developers can choose the implementation approach that fits
their model and optimization needs.

Users continue to use the LLM API and `trtllm-serve`. The `modeling_v2` flag
selects the implementation, as described in [Using it](#using-it). Modeling V2 is
currently opt-in, with `off` as the default.

### What stays shared? Is Modeling V2 a separate inference engine?

Modeling V2 runs within TensorRT-LLM. It reuses the serving runtime and common
services such as scheduling, cache management, distributed execution and sampling,
along with general-purpose operations. Families and targets own their model logic
and specialized implementations.

The boundary can evolve as models need more customization. Model-specific runtime
policy and adapters can be separated from shared services, while changes to common
runtime behavior still need to account for other models. Further runtime
specialization is active design work.
