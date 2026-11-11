<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Modeling V2

Modeling V2 is an experimental way to write models in TensorRT-LLM, where each
model family and each deployment target owns its own code. With coding agents,
writing specialized model code and kernels is cheap. Reviewing and maintaining
changes to code that every model shares is the expensive part. Modeling V2
keeps model-specific code inside its family or target, so a change to one model
cannot break another, and it reuses TensorRT-LLM's serving runtime and shared
operators for everything else. The trade-off is some duplicated code between
models. For example, the DeepSeek R1 target loads its own weights and calls its
MLA projections, normalization, RoPE and attention operations directly in its
forward pass, so it can be optimized without affecting any other model.

## Using it

The `modeling_v2` LLM API option selects the implementation:

| `modeling_v2` | Behavior |
|---|---|
| `off` (default) | Use Modeling V1. No Modeling V2 routing or target code is loaded. |
| `auto` | Use a Modeling V2 target if one matches this deployment; otherwise use Modeling V1. |
| `require` | Use a Modeling V2 target, or raise an error that says why none matches. |

Python API:

```python
LLM(model, modeling_v2="require")
```

`trtllm-serve` configuration:

```yaml
# trtllm-serve --config serve.yaml
modeling_v2: require
```

Use `require` when you validate or benchmark a target, so the run cannot fall
back to Modeling V1 without you noticing. A target can also read the mode with
`resolved_mode()` when it is constructed. Under `auto`, its constructor can
raise `NoTarget` to decline, and the engine then uses Modeling V1 for the
whole model. Under `require`, the same `NoTarget` is an error.

## Families and targets

A **family** is a directory `models/<family>/`. It usually covers one Hugging
Face architecture, the `architectures[0]` value in the checkpoint's
`config.json`. It owns the routing for that architecture and any code its
targets share. You can scope a family more broadly or more narrowly if that
fits the models better.

A **target** is a directory `models/<family>/<target>/` that holds one complete
model implementation. Routing selects a target for a specific set of
checkpoint, GPU and parallel configurations. A target does not import code
from its sibling targets. It can use its family's shared code, the shared
operation catalog and the TensorRT-LLM runtime, but not Modeling V1 model code.
See [Modeling V1 code](#modeling-v1-code).

Target directories are usually named `<checkpoint>__<sm>__<parallel>`, with
double underscores between the three parts:

* `<checkpoint>` identifies the checkpoint, for example `r1_0528_nvfp4`.
* `<sm>` is the GPU architecture, for example `sm_103`, or `sm_10x` for code
  that runs on SM10x.
* `<parallel>` is either one parallel layout, for example `tp4`, or `default`.
  A `default` target derives its sharding from the engine's `Mapping` and runs
  every layout its `route` admits.

The target class name contains the same three parts in CamelCase, for example
`ModelingV2DeepseekR10528Nvfp4Sm10xDefault` in
[`r1_0528_nvfp4__sm_10x__default`](models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/).
These names are conventions, not requirements.

A target can contain several execution paths. The DeepSeek R1 target, for
example, has separate `PrefillTarget` and `DecodeTarget` classes and picks one
on each forward step. In this guide, "target" always means the whole directory.

## Code layout

```text
modeling_v2/
├── _router_index.py           # Route registration and target resolution
├── _native.py                 # Loads family native libraries
├── catalog/
│   ├── _op.py                 # OpWrapper: the operation contract
│   └── <category>/            # Operations used by more than one family
└── models/
    ├── __init__.py            # Imports every family's routing.py
    └── <family>/
        ├── routing.py         # route(): picks a target or raises NoTarget
        ├── ops/, kernels/     # Optional: code shared by the family's targets
        ├── csrc/, fake_ops.py # Optional: family native code
        └── <target>/
            ├── modeling.py    # Model construction and forward
            ├── weights.py     # Checkpoint mapping, loading and sharding
            ├── ops/, kernels/ # Optional: target-only operations and kernels
            └── csrc/, fake_ops.py  # Optional: target native code
```

The layout inside a family is a convention that the existing families follow,
not a requirement. Organize a family's files in whatever way suits its
implementation. Native code is the exception: the build looks for it in fixed
locations, described in the
[native code guide](../../../../cpp/tensorrt_llm/modeling_v2/README.md).

Put code at the narrowest scope that uses it:

| Scope | Put here | Location |
|---|---|---|
| Target | Code only this target uses | `models/<family>/<target>/` |
| Family | Code shared by targets of one family | `models/<family>/` |
| Catalog | Operations used by more than one family | `catalog/<category>/` |

If two targets need the same code, move it to the family instead of importing
it from a sibling target. Duplicating code between targets is fine. Whether to
merge it is the family owners' decision.

## Modeling V1 code

A target builds its model from operations: catalog operations and its
family's and its own operations and kernels. From the rest of TensorRT-LLM,
it uses only runtime interfaces, for example:

* `DecoderModel`, `DecoderModelForCausalLM`, `ModelConfig` and `Mapping`.
* Attention and speculative metadata, and the KV cache managers.
* `MultimodalRuntimeMixin` from `_torch/multimodal/model.py`, which provides
  shared encoder execution and caching independently of model implementations.

A target does not import, subclass or call Modeling V1 model code: the model
classes in `_torch/models/`, the layers in `_torch/modules/`, MLA and the
attention backends in `_torch/attention/`, and fused MoE in `_torch/moe/`.
This applies to every path in the target: prefill, decode, the shapes a fast
path does not cover, and weight loading. The
[DeepSeek R1 target](models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/) is an
example that uses no Modeling V1 model code.

### Multimodal targets

A target for a checkpoint whose input processor accepts media declares how it
serves media. It inherits `MultimodalRuntimeMixin` from
`_torch/multimodal/model.py` to serve media through the shared multimodal
runtime, or `TextOnlyTarget` from [`_multimodal.py`](_multimodal.py) to serve
the language model only. The marker refuses requests that carry media at
admission, so each fails on its own rather than failing its batch in `forward`. `modeling_v2_resolve`
checks this for every target a `route` returns, in `auto` as well as
`require`, and `test_modeling_v2_multimodal_contract.py` checks it for every
routed target in CI.

A runtime target inherits `DecoderModelForCausalLM` and
`MultimodalRuntimeMixin`. The mixin has no notion of a separate language model
and defines nothing the decoder defines, so the decoder's speculative and
runtime hooks are unaffected. Implement `encode_multimodal_inputs`,
`embedding_dim` and `embedding_dtype`. The model loader reads the last two from
the constructed target, so they must work with the checkpoint's real
configuration. Expose `dtype` for encoder memory profiling.

Before the core's forward, call `prepare_multimodal_embeddings` with the
engine's context count, the request Params and the executor's
`mm_token_indices`. It returns the active encoder rows after request-local
reuse, persistent cache lookup, stream synchronization, chunk selection, and
model hooks, paired with their placeholder positions. Feed both to the
target's own token splice. Keep original token IDs available when other model
operations need them.

Filter position-only Params in `select_multimodal_params`: the processor emits
them for text requests, and the executor adds them for generation requests.
Enable `supports_encoder_cache` only when the processor supplies stable item
keys and exact output row counts. Declare packed input partitions or implement
item selection so partial cache hits encode only missing items. Encoder outputs
and cached tensors are read-only.

Declare the encoder's limits on the target:

* `max_encoder_tokens_per_item` bounds the dummy items that encoder memory
  profiling builds. Return the largest item the encoder runs in one piece when
  it splits or refuses larger ones.
* `supports_concurrent_encoder` permits side-stream prefetch. Set it only when
  mutable encoder state, such as kernel workspaces, is separate per CUDA
  stream.
* `allows_breakable_prefill_cuda_graph` admits breakable prefill CUDA graphs.
  Return True only when encoding happens outside the captured forward.

Item scheduling is a separate capability. Set
`supports_mm_encoder_item_scheduling` only after the complete cost, selection,
and memory-profile contracts work for the target's certified encoder shapes.
The Qwen3.8 Flash Next target uses inline encoding, chunk reuse, persistent
encoder caching, prefetch and breakable prefill CUDA graphs; it leaves item
scheduling disabled because the processor's maximum atomic image exceeds its
certified per-frame patch bound. See the
[multimodal developer guide](../../../../docs/source/developer-guide/multimodal.md)
for the shared contracts.

## Adding a family or a target

1. **New family only:** create `models/<family>/routing.py` with a `route`
   function decorated with `@routes("<Architecture>")`, and import that module
   in [`models/__init__.py`](models/__init__.py). See [Routing](#routing).
2. Create `models/<family>/<target>/` with `modeling.py` and `weights.py`,
   following the naming in [Families and targets](#families-and-targets). See
   [Model lifecycle](#model-lifecycle).
3. In `route`, return the target class for the configurations it supports.
4. Use catalog operations where they fit. Add new operations as `OpWrapper`s
   with tests. See [Writing operations](#writing-operations).
5. For C++/CUDA code, follow the [native code guide](../../../../cpp/tensorrt_llm/modeling_v2/README.md).
6. Add unit tests and a whole-model accuracy test, and add them to the CI test
   lists. See [Testing and validation](#testing-and-validation).
7. Add the new paths to [CODEOWNERS](../../../../.github/CODEOWNERS). Families
   with their own owners list three paths: the package directory, the unit
   test directory and the accuracy test directory.

## Model lifecycle

A target plugs into TensorRT-LLM through the existing `DecoderModel` and
`DecoderModelForCausalLM` interfaces. It inherits these classes directly, not
a Modeling V1 model class, and can override any of these steps:

* **Constructor:** builds the model structure and parameters.
* **`load_weights`:** loads checkpoint weights into the model. Put weight
  mapping, transformation and sharding in `weights.py`.
* **`post_load_weights`:** prepares the loaded weights, for example by
  transforming them or precomputing derived state. The default implementation
  in `DecoderModelForCausalLM` runs alias setup, weight transforms and derived
  state caching; override or extend it if the target needs more.
* **`forward`:** runs the model on the inputs and attention metadata from the
  engine.

Keep the sequence of operator calls in `forward` easy to read, with the
arguments passed explicitly. Helpers, modules and separate prefill and decode
classes are fine when they make the code clearer or faster. The
[GPT-OSS target](models/gpt_oss/gpt_oss_120b__sm_10x__default/modeling.py) and
its [weight loader](models/gpt_oss/gpt_oss_120b__sm_10x__default/weights.py)
are a small example.

## Routing

Each family's `routing.py` defines one `route` function, registered with
`@routes(...)` for the architecture names it handles. `route` receives the
`ModelConfig`, the LLM arguments and the GPU's compute capability. It returns
a target class, or raises `NoTarget` with a message that names the requirement
that failed. [DeepSeek R1 routing](models/deepseek_v3/routing.py) is an example.

Each family decides what to route on, for example the checkpoint shape, GPU
architecture, parallel layout, speculative decoding mode or KV cache dtype.
Routing runs once, when the model is loaded, so per-step differences such as
batch composition or prefill versus decode are handled inside the target.

Check in `route` every requirement the target has, not only in the target's
constructor. A configuration that `route` declines falls back to Modeling V1
under `auto`; one that the constructor rejects with an assertion fails.

How the model loader resolves a target:

```text
LLM(model=..., modeling_v2=...)  ->  ModelLoader
                                        |  model defaults applied to llm_args
                                        |
                                        +-- modeling_v2_resolve_on_all_ranks(config, llm_args)
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

## Writing operations

Wrap each kernel call in a thin, stateless operation class that makes its
contract explicit. Put the operation in the narrowest
[scope](#code-layout) that uses it: the target's `ops/`, the family's `ops/`,
or `catalog/` when more than one family uses it. The
[`OpWrapper` base class](catalog/_op.py) defines:

| Member | Purpose |
|---|---|
| `__call__` | Calls the kernel. Every argument is passed explicitly. |
| `reference` | The expected result, computed in plain PyTorch. |
| `is_valid` | Rejects inputs that the kernel accepts but computes incorrectly. |
| `CELLS` | Input configurations the operation is tested on. |
| `ARCHS` | GPU architectures those configurations were tested on. |

Pass weights, state and configuration as call arguments, and keep persistent
state in the target. An operation's tests compare it with its `reference` on
its `CELLS`. If your target uses an operation with new shapes, layouts or
dtypes, add those cases to the operation's tests. The existing
[catalog entries](catalog/) and their tests under
`tests/unittest/_torch/modeling_v2/` are examples.

## Native code

C++/CUDA code for a family or target lives in its `csrc/` directory and builds
into one library per family. Do not put it under `cpp/tensorrt_llm/kernels/`
or `cpp/tensorrt_llm/thop/`. See the
[native code guide](../../../../cpp/tensorrt_llm/modeling_v2/README.md) for
build declarations, namespaces, op and fake registration, loading and errors.

## Testing and validation

Put tests under `tests/`, in a directory that mirrors the code they test. CI
picks tests by that mirror, so a test outside it may not run. Inside a target's
test directory, the layout is up to the target.

| Tests | Location |
|---|---|
| Routing, loader and CI selection | `tests/unittest/_torch/modeling_v2/test_modeling_v2_*.py` |
| Catalog operations | `tests/unittest/_torch/modeling_v2/<category>/` |
| Family routing | `tests/unittest/_torch/modeling_v2/models/<family>/` |
| Family-shared code | `tests/unittest/_torch/modeling_v2/models/<family>/<path of the code>/` |
| Target code | `tests/unittest/_torch/modeling_v2/models/<family>/<target>/` |
| Whole-model accuracy | `tests/integration/defs/accuracy/modeling_v2/<family>/<target>/` |
| Accuracy references | `tests/integration/defs/accuracy/references/` |

CI selects tests by the scope of the change:

* A change inside a target runs that target's unit and accuracy tests.
* A change to family-shared code runs its mirrored tests and the tests of
  every target that imports it. A family's `routing.py`, and its native code
  outside a target, run the whole family.
* A change to a catalog operation runs its category's tests and the tests of
  every target that imports it. A change to code every model shares runs all
  Modeling V2 tests.

A target is a directory under `models/<family>/` whose class the family's
`route` returns. Give each target a mirrored unit test directory and an
accuracy test directory, `tests/integration/defs/accuracy/modeling_v2/<family>/<target>/`.
Make sure a test list under `tests/integration/test_lists/test-db/` reaches
every new test, either by file or by directory. CI does not run unlisted tests.
A target's accuracy tests may instead be listed under
`tests/integration/test_lists/qa/`, which runs on the QA schedule.
`tests/unittest/_torch/modeling_v2/test_modeling_v2_ci_selection.py` checks
the layout and the test lists.

Typical whole-model checks for a target:

* **Boot (local check):** build the engine, load the weights and generate a
  few deterministic tokens. This catches missing weights and basic execution
  errors before you run longer evaluations.
* **Accuracy:** run the checkpoint's evaluation and compare against its
  reference. Rerun it whenever a change affects numerics, weight loading or
  execution.
* **Acceptance:** with speculative decoding, also check the acceptance length.
  A wrong draft only slows decoding, so accuracy alone does not catch it.

Set `TRTLLM_MODELING_V2_VALIDATE=1` to make targets that support it check, on
the first forward step, that the attention metadata fields they read still
exist. The check is off by default. Tests pass the variable to all ranks
through `env_overrides`.

Record workload settings and results in the test or in the evaluation report.

## FAQ

### Will duplicated code between models become unmanageable?

Some duplication is expected, and Modeling V2 accepts it on purpose. Judge its
cost over the whole life of the code: development, review, testing,
optimization and maintenance.

A target only has to handle the configurations it supports, so it has fewer
branches and abstractions than a model class shared by many checkpoints. That
makes it easier to read, test and optimize, even if similar code exists in
another target. Coding agents also make it cheaper to port a fix or feature
between similar implementations, though each ported change still needs testing
in every implementation it touches. When code is common enough, move it to the
family, or to the catalog if other families use it too.

A model-specific optimization can stay local without first being generalized.
A fix that clearly helps every user of shared code still goes into the shared
code.

### How does Modeling V2 make code review easier?

It lets reviewers spend their effort where a change has the most impact.
Agent-written model code is large: the
[Kimi K3 integration (PR #20893)](https://github.com/NVIDIA/TensorRT-LLM/pull/20893)
added more than 80,000 lines, including model-specific kernels, targets and
tests. Reading every generated kernel line by line does not scale.

* **Runtime interfaces and shared code:** review closely, because changes affect
  other models. This includes model and runtime contracts, common runtime
  behavior, and catalog or family operations.
* **Target structure:** review the target's scope, routing, weight loading and
  forward flow. Check that model-specific code stays inside the target and that
  its dependencies on shared code are explicit.
* **Isolated kernels in one target:** review the operation contracts, tests,
  accuracy and performance results. Read the kernel code in depth where
  assumptions are unclear or results look wrong.

This lighter review applies only to code that is actually isolated. A shared
change inside a large model PR still needs a full review. Keep shared changes
easy to spot, and send them as a separate PR when you can. Do not add
model-specific branches to shared code.

### Do I have to use a particular agent or workflow?

No. There is a reference agent workflow for developing Modeling V2 targets, but
it is only one option. Use any agent, skill or manual workflow, for the first
implementation and for later fixes. What matters is the result: it uses the shared interfaces,
keeps model-specific code isolated, has clear owners, and has tests and
measurements that show it is correct and fast.

### Who owns and reviews a family or target?

The people who contribute a family or target maintain it, including its
kernels and tests. Record them in [CODEOWNERS](../../../../.github/CODEOWNERS).

Model owners review the model code. Modeling V2 maintainers review the shared
design, interfaces and layout. Changes to shared runtime code or catalog
operations also need approval from the owners of that code, as usual.

### Does Modeling V2 replace Modeling V1 or change the user API?

Not now. Modeling V1 and V2 exist side by side, and adding a V2 target does
not require migrating any V1 model. We recommend Modeling V2 for new models
whose implementation differs significantly from existing Modeling V1 models,
and for optimizations specific to one model. Modeling V1 will eventually be
deprecated, as the models it supports are deprecated, but not now.

Users keep using the LLM API and `trtllm-serve`; the `modeling_v2` option,
`off` by default, selects the implementation. See [Using it](#using-it).

### Is Modeling V2 a separate inference engine?

No. Modeling V2 targets run inside TensorRT-LLM and use its scheduler, KV cache
manager, distributed execution, sampling and serving, along with its
general-purpose operations. Families and targets own only their model code and
specialized kernels.

That boundary may move as models need more customization. Model-specific
runtime policy can be split from shared services, but a change to common
runtime behavior still has to work for every other model. How to specialize
the runtime further is still being designed.
