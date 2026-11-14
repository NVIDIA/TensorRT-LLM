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
| `off` | Use Modeling V1. No Modeling V2 routing or target code is loaded. |
| `auto` (default) | Use a Modeling V2 target if one matches this deployment; otherwise use Modeling V1. |
| `require` | Use a Modeling V2 target, or raise an error that says why none matches. |

Because `auto` is the default, a deployment that a target matches runs on
Modeling V2 without any option set. Every other deployment runs on Modeling V1
as before. Targets decorated with `@disable_v2_by_default` are the exception:
under the default they are skipped, and a deployment uses them only when it
sets `modeling_v2` to `auto` or `require` itself. See
[Adding a family or a target](#adding-a-family-or-a-target).

Python API:

```python
LLM(model, modeling_v2="require")
```

`trtllm-serve` configuration:

```yaml
# trtllm-serve --config serve.yaml
modeling_v2: require
```

To opt out and always use Modeling V1, set `modeling_v2="off"` in the Python
API or `modeling_v2: off` in the configuration file.

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

A target does not import, subclass or call Modeling V1 model code: the model
classes in `_torch/models/`, the layers in `_torch/modules/`, MLA and the
attention backends in `_torch/attention/`, and fused MoE in `_torch/moe/`.
This applies to every path in the target: prefill, decode, the shapes a fast
path does not cover, and weight loading. The
[DeepSeek R1 target](models/deepseek_v3/r1_0528_nvfp4__sm_10x__default/) is an
example that uses no Modeling V1 model code.

## Adding a family or a target

1. **New family only:** create `models/<family>/routing.py` with a `route`
   function decorated with `@routes("<Architecture>")`, and import that module
   in [`models/__init__.py`](models/__init__.py). See [Routing](#routing).
2. Create `models/<family>/<target>/` with `modeling.py` and `weights.py`,
   following the naming in [Families and targets](#families-and-targets). See
   [Model lifecycle](#model-lifecycle).
3. In `route`, return the target class for the configurations it supports.
   Under the default `auto` mode, every deployment `route` returns the target
   for switches to it as soon as the target lands. To keep a target opt-in
   until it is validated, decorate its class with `@disable_v2_by_default`
   (from `tensorrt_llm._torch._experimental.modeling_v2`). Deployments then use
   it only when they set `modeling_v2` to `auto` or `require` explicitly. A
   target that implements expert load balancing lists its modes in
   `EPLB_MODES`; see [Routing](#routing).
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

Expert load balancing (EPLB) is checked for every target. When a deployment
runs EPLB (`moe_config.load_balancer` is set, the architecture is in
`moe_model_arch_list` and the experts are split over more than one EP rank),
the loader creates the load balancer whichever class it builds, but only a
target that registers its MoE layers with it balances anything. After `route`
returns a target, the resolver compares the deployment's EPLB mode (`static`,
`online` or `per_layer_dynamic`) with the target class's `EPLB_MODES` and
treats a target that does not list it as `NoTarget`: `auto` keeps Modeling V1
for that deployment and `require` raises with the reason. `EPLB_MODES` is empty
unless the target sets it.

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

## Speculative decoding

A target supports speculative decoding in one of two modes:

* **Split: target modeling plus a speculative worker.** The target
  implements the main model, and one of the engine's speculative workers
  verifies, accepts and drafts after the target's forward. This supports the
  engine's existing speculative methods quickly. The target inherits
  `SpecDecOneEngineForCausalLM`, or builds the worker with `get_spec_worker`. It
  can also subclass the worker to speed up some of its steps, as
  `GlmMtpWorker` in the
  [GLM-5.2 sm_103 tp8 target](models/glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/spec_worker.py)
  does. For one-model MTP, the drafter normally builds the built-in MTP layer
  for the checkpoint's model type; a target that has its own MTP layer names it
  in its decoder model's `mtp_layer_cls`, and the drafter builds that class
  instead. Example: `glm_5_3_flash_nvfp4__sm_100__default` in the
  [GLM-5.3-Flash family](models/glm5_next/), with the engine's MTP worker.
* **Fused: speculation in the target's forward.** The target runs the whole
  speculative step (verification, acceptance and drafting) as part of its own
  `forward`, and builds no worker. Example:
  `glm_5_3_flash_nvfp4__sm_100__default_mtp_fused`, which runs one-model
  MTP-Eagle.

### Routing

When a family has a fused target for a deployment, `route` chooses it over the
split target: a fused target can optimize the step for that one deployment, so
it is usually faster. A fused target usually supports a narrower set of
speculative configurations than the split target, so `route` checks the fused
target first and returns the split target for the deployments the fused target
declines.

More generally, when several targets of a family support a deployment, `route`
checks the most specialized one first, the one written for the narrowest set of
deployments, and falls through to the more general ones. This holds with or
without speculative decoding: a target written for one parallel layout can come
ahead of the split target for that layout's deployments without speculation
too. A fused target ahead of its split sibling is one case of this rule.

### Writing a fused target

#### Required

A fused target takes the same inputs and returns the same outputs as the split
path, so the runtime schedules both modes the same way: it packs the inputs,
runs the speculative sampler and rewinds the KV cache without knowing which
mode the target uses.

* **Inputs.** Keep the causal LM `forward` signature, with `attn_metadata`,
  `input_ids`, `position_ids` and `spec_metadata`, packed by the engine as for
  the split path; add no inputs of your own. Read the step from
  `spec_metadata`: the previous step's drafts (`draft_tokens`), this step's
  draft length (`runtime_draft_len`), the rows that need logits
  (`gather_ids`) and each request's sampling parameters. A batch can mix
  context and generation requests, with context requests first.
* **Outputs.** Return the dict the speculative sampler reads. With `B`
  requests and `K = runtime_draft_len`:

  | Key | Shape | Contents |
  |---|---|---|
  | `logits` | verify rows x vocabulary, or empty | The verify logits; empty when the step samples without materializing them |
  | `new_tokens` | `[B, K+1]` | The tokens accepted this step; a context request defines column 0 only |
  | `new_tokens_lens` | `[B]` | Tokens accepted per request: 1 + accepted drafts |
  | `next_draft_tokens` | `[B, K]` | The drafts to verify next step |
  | `next_new_tokens` | `[B, K+1]` | Each request's last accepted token, then its drafts: the next step's input under the overlap scheduler |

  With `K = 0` the dict has the same keys, and `next_draft_tokens` is
  `[B, 0]`. A target that returns empty `logits` refuses, in
  `validate_request`, the requests that need them, such as requests with
  logits post-processors; the speculative sampler already refuses logprobs
  and context or generation logits.
* **Naming.** When a fused target is a variant of a split target, written
  for the same checkpoint, GPU and parallel layout as that split sibling, name
  it after the sibling with a suffix on the parallel segment that names the
  mode, for example `<checkpoint>__<sm>__<parallel>_mtp_fused`, and carry the
  suffix into the class name, for example
  `ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused`. A target written on its own,
  with no split sibling, follows the plain target naming in
  [Families and targets](#families-and-targets), even where its deployments
  overlap another target's, and states the modes it fuses in its module
  docstring.

#### Recommendations

* **Engine attributes.** The engine reads `spec_worker` with a `None`
  default; set `spec_worker = None` to state that no worker runs. It reads
  `draft_config` only when it loads a separate draft checkpoint into
  `draft_model`, so set it only for such a drafter.
* **Routing.** Decline what the fused target does not support in `route`,
  not only in its constructor, so the deployment falls to the split target
  instead of failing. Write each target's unsupported conditions as one
  function that returns the reasons, which `route` selects on and `NoTarget`
  reports, as the [GLM-5.3-Flash routing](models/glm5_next/routing.py) does.
* **Draft KV.** Write the draft KV into the layers the engine allocates for
  the drafter (`get_num_spec_layers`; one layer at index `num_hidden_layers`
  for one-model MTP-Eagle), within the `max_draft_len - 1` extra tokens it
  reserves per sequence, and leave the rewind of rejected tokens to the
  engine. Update `kv_lens` per draft step: step 0 drops the rejected drafts,
  and each later step adds one.
* **Attention metadata.** Restore the attention metadata fields the stock
  MTP-Eagle worker restores after its draft loop: the sequence lengths, the
  speculative packed mask, position offsets and generation lengths, and
  `all_rank_num_tokens`. CUDA graph warmup replays the forward on the same
  metadata. Save the lengths with `prepare_for_spec_dec` and restore them with
  `restore_from_spec_dec` in a `finally` block, so a failed step, such as an
  out-of-memory error during warmup, does not leave them changed. Like the
  worker, the draft chain leaves `num_contexts`, `host_request_types` and
  `kv_lens_cuda` changed.
* **Per-token state.** Commit recurrent, conv or indexer state written per
  token for the accepted tokens only, for example through the cache manager's
  `update_mamba_states`.
* **Unsupported requests.** Refuse in `validate_request` the requests that
  need an output the target does not return, such as full-vocabulary logits
  for logprobs. Return `False` from `set_guided_decoder` when the step runs no
  guided decoder.
* **Weights.** Share the target's LM head and embedding with a drafter that
  uses them rather than copying them. Append a draft layer that lives in the
  target's checkpoint to `model.layers`, so its weights load with the decoder
  layers.
* **CUDA graphs.** Make every shape depend only on the step's host integers
  (batch size, number of context requests, draft length), so the whole step is
  captured in one CUDA graph with the target.

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

Users keep using the LLM API and `trtllm-serve`. The `modeling_v2` option
selects the implementation. It defaults to `auto`, which uses a Modeling V2
target where one matches the deployment and Modeling V1 everywhere else. See
[Using it](#using-it).

### Is Modeling V2 a separate inference engine?

No. Modeling V2 targets run inside TensorRT-LLM and use its scheduler, KV cache
manager, distributed execution, sampling and serving, along with its
general-purpose operations. Families and targets own only their model code and
specialized kernels.

That boundary may move as models need more customization. Model-specific
runtime policy can be split from shared services, but a change to common
runtime behavior still has to work for every other model. How to specialize
the runtime further is still being designed.

### Beam diagnostic validation

For Beam, `TRTLLM_MODELING_V2_VALIDATE=1` additionally enables repeated per-layer
metadata checks and per-step, all-rank routing agreement using device-to-host copies
and SHA-256 hashes. The serving adapter also enables checkpoint digests in this mode.
This is expensive diagnostic behavior, not only a first-forward contract check.
Use the same setting on every rank and leave it disabled for performance measurements.
