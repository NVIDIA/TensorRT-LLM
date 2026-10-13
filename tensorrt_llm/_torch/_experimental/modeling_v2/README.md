# ModelingV2

One self-contained modeling codebase per deployment target, beside the
built-in model zoo rather than inside it.

Where `_torch/models/modeling_deepseekv3.py` is one class serving V3, V3-Lite,
R1 and V3.2 across every GPU generation and parallel topology,
`_torch/_experimental/modeling_v2/models/deepseek_v3/` is one flat forward per (checkpoint,
GPU architecture, parallel topology) triple — assembled only from `catalog/`
entries, sharing nothing with its siblings, and trusted through accuracy gates
instead of shared abstractions. The one-to-one correspondence between
`models/<x>/` here and `modeling_<x>.py` there is the point of the exercise.

## Using it

```python
LLM(model, modeling_v2="require")        # off | auto | require
```

```yaml
# trtllm-serve --config serve.yaml
modeling_v2: require
```

| `modeling_v2` | Behaviour |
|---|---|
| `off` (default) | The resolver returns immediately. Nothing beyond it is imported and behaviour is byte-for-byte what it is today. |
| `auto` | Uses a target when one claims this exact configuration; falls back to the built-in implementation otherwise, logging why. |
| `require` | Raises instead of falling back, saying which criterion did not hold. |

It is an LLM API argument rather than an environment variable so that it
reaches every rank the way the rest of the arguments do; a variable set after
MPI initialized reached the driver and not the workers, and a driver building
a target while its workers build the built-in is exactly the split `require`
exists to catch.

The decision is one function, `route`, in the family's `routing.py`. It reads
the facts that fix which code runs -- the checkpoint's config shape, the
parallel mapping, the GPU's compute capability and the speculative-decoding
config -- and either returns a target class or raises `NoTarget` with the
reason in words. Tuning knobs such as `max_batch_size`, `max_num_tokens` and CUDA graph
settings are not routing inputs: they do not change which code runs, so a
target claims a configuration regardless of them. `require` therefore promises
that the target ran, not that a particular knob setting was measured. The model
loader decides when it is about to build the model, after the built-in model's
defaults have been applied to the arguments, and carries the answer on
`ModelConfig.modeling_v2_target`.

**Use `require` for anything you will attribute to modeling_v2.** Under `auto`,
a configuration that misses a target's criteria silently gets the built-in
implementation — and a performance curve measured that way reads as
modeling_v2's. That is the single most expensive mistake available here.

The predecessor `MODELING_V2_TARGET` is gone. It named a *target*; this names
only a mode, and routing picks the target from the configuration.

The checkpoint is read exactly as published, with no target-owned
`config.json`.

## How a config finds its target

```
LLM(model=..., modeling_v2=...)  ->  ModelLoader
                                        |  model defaults applied to llm_args
                                        |
                                        +-- modeling_v2_resolve(config, llm_args)
                                              |
                    _router_index.py          +-- architectures[0] -> routing module
                    models/<x>/routing.py     +-- route(config, mapping, compute_capability,
                                              |         spec_config) -> target class, or NoTarget
                                        |
                                     config.modeling_v2_target = class
                                        |
                    AutoModelForCausalLM._resolve_class(config)
                                        |
                                     returns that class; the registry is not consulted
```

`route` imports the target class on the branch that returns it, so a routing
module that declines never loads modeling code, and the class a reader sees
returned is the class that runs -- no name to look up elsewhere.

## What may decide a target

A quantity may be a routing criterion only if it is known when
`_resolve_class` runs **and** constant for the engine's life — and only if it
changes the *structure of the forward*. Config shape, SM version, parallel
topology qualify. `max_num_tokens` and `cuda_graph_config.max_batch_size` pass
the first test and fail the second: they are tuning knobs — LLM API arguments,
not identity. Batch composition and `num_contexts` fail the
first outright — those move every step, and a target chosen from them would
be chosen once and then be wrong. Per-step specialization is a separate
mechanism (dispatch inside a target's `forward`), not a routing dimension.

### Checkpoint identity is a shape fingerprint

Routing recognizes a checkpoint by `(num_hidden_layers, hidden_size, ...)`,
the same sniffing idiom as `is_mla` / `is_nemotron_hybrid` upstream. It cannot
tell a fine-tune from the original. That is a deliberate trade, and it changes
what a gate record means: not "this target passed" but "this modeling code
passed **on the checkpoint the accuracy gate names**". Running it on any other
checkpoint of the same shape is ungated.

## Layout

```
_router_index.py    architectures[0] -> routing module. Small, stable, test-guarded.
models/<family>/
  routing.py        one forward-reading decision tree per architecture family
  <checkpoint>__<gpu arch>__<parallel>/
                      modeling.py  weights.py
                      ops/<category>/   entries only this target calls
catalog/<category>/ entries more than one target calls
```

An entry, shared or owned, is one `OpWrapper` subclass: `__call__` is the op
call a target makes, `reference` is what it is supposed to compute in plain
torch, `is_valid` names the inputs it accepts and answers wrongly, and `CELLS`
/ `ARCHS` are the configurations and architectures it is proven at. The
wrapper is stateless and `__call__` reads nothing off `self`, so a call site
shows exactly what the kernel receives; `catalog/_op.py` states the rest.

Accuracy anchors are not kept here. They live where every other model's do,
`tests/integration/defs/accuracy/references/`, read by the gates CI runs.

An entry's GPU test lives in the tests tree, mirroring the entry's path:

```
tests/unittest/_torch/modeling_v2/
  test_modeling_v2_claims.py     source-level gates over the whole package (no GPU)
  test_modeling_v2_routing.py    what modeling_v2_resolve does (no GPU)
  _validating.py                 the fixtures that interpose is_valid and arm reference
  <category>/test_modeling_v2_<entry>.py                      shared entries
  models/<family>/<target>/<category>/test_modeling_v2_<entry>.py   owned entries
  comm/_<entry>_op_matrix.py   each collective's rank bodies
  comm/_lockstep.py            the stateful matrices' launcher and shared reference
  comm/_rank_job.py            starts one at a world size and asserts on its exit code
```

That split is not a preference; it is where this repo's CI collects from, and
an in-package test is on no list. Each test is parametrized off its entry's
`CELLS` and gates `__call__` against `reference` within a band the test states
and derives, so the certification is re-proved on every commit rather than
recorded once.

The collectives' rank bodies sit in the tests tree with everything else
that only tests run. Both halves are started by file path -- the launcher must
not import `tensorrt_llm`, because that calls `MPI_Init` and an
MPI-initialized process cannot start `mpirun`, and the ranks reach the catalog
by absolute import -- so neither needs a package to live in. Neither those
file names nor their `check_*` bodies match pytest's collection patterns:
each is one fixed sequence over one job's ranks that cannot run as
independent cases. CI runs them at 4 ranks on one node. A stateful entry's
launcher also takes `--world-size N --launcher srun`, which runs the same
body as one of N ranks started across nodes: that is how its multi-node
receipt is recorded.

Identity is the directory name, and it carries all three segments:
`gpt_oss_120b__sm_103__tp1`. They were three nested directories once, which
read as a hierarchy that was never one -- every level had exactly one child,
and each needed an `__init__.py` whose only job was to exist. The class name
carries the same triple; `test_modeling_v2_claims.py` asserts they agree.

### Why beside `_torch/models/`, not inside it

Targets are not registered in the model zoo: `route` returns the class and the
loader builds it. Inside `_torch/models/` they would still be swept up by the
zoo's own guards -- `test_lazy_model_zoo.py` scans that directory and expects
every modeling module there to appear in the built-in static index.

Conceptually `models/` means "one architecture, one class, shared across
checkpoints", which is the opposite of what this package is for.

## Gates

1. **Boot** — minutes, binary. `examples/llm-api/quickstart_advanced.py` with
   the target's topology flags and `modeling_v2=require` passed. Engine
   cold start, weight-manifest coverage, a handful of greedy continuations.
   Catches catastrophes, not accuracy. A variant that changes the forward gets
   its own minutes-scale gate on the same footing.
2. **Accuracy** — the release criterion. `trtllm-eval` with the protocol from
   `tests/integration/defs/accuracy/references/` and `modeling_v2=require`
   passed. One-sided: measured >= reference − tol. This is the gate CI runs,
   in the accuracy suite's modeling_v2 files.
3. **Acceptance** — required whenever a variant is distribution-preserving by
   construction, speculative decoding above all. Rejection sampling holds the
   emitted distribution to the target model's, so a *miscomputed* draft path
   produces correct text more slowly: boot passes, accuracy passes, only
   speed moves. The detector is `acceptance_length` against the same
   checkpoint under stock in-tree modeling at the same workload and the same
   `speculative_config` — i.e. `modeling_v2=off` versus `=require`,
   which is now one variable rather than two harnesses.

Compare a variant against an identity run **in the same session**: one target
measured 94.7688 and 95.0720 on a bit-identical forward a day apart, so a
cross-session delta carries session variance into the judgement.

Perf is measured, never gated.

## Status of every record in this tree

**The 19 upstream entries are certified on sm_103. Kimi K3's sm_100
operation receipts remain in its adjacent records; its generic-operation
regressions now exercise the relocated entries. Certification is per
architecture, not implied by a wrapper's location.**

The 19 entries -- 8 shared, 8 owned by deepseek, 3 by gpt-oss -- each drive
their cells through `__call__` and `reference` in pre-merge CI, plus the two
4-rank collective matrices. Certification is per architecture: `ARCHS` says
where the cells were driven, and a missing architecture reads as unknown,
which is the honest state.

Getting the catalog there surfaced four real differences. None was resolved by
widening a tolerance, and each is written up in its own entry:

* **Op schema drift** (`thop_attention`, `mla_rope_generation`,
  `mla_rope_append_paged_kv_assign_q`). Parameters were renamed and added. The
  wrappers now mirror their schemas argument for argument, so the next drift
  fails loudly rather than shifting a positional list silently.
* **The MoE FC1 epilogue's block-scale recipe is a property of the cubin**,
  bit-exactly: `ceil(log2(amax/448))` or `floor(log2(amax))-8`. sm_100 and
  sm_103 both use the first. The reference is architecture-keyed, and the test
  refutes the recipe not in force.
* **torch 2.12 made fp32 matmul default to TF32**, so `cublas_mm`'s *reference*
  was the imprecise side; the op is bit-identical to a TF32-disabled product.
* **The MLA append op now accepts NVFP4 latent pools** as well as fp8 (accepted
  by the op, not certified here).

Both targets are gated by the accuracy suite's modeling_v2 files,
`tests/integration/defs/accuracy/test_modeling_v2_{gpt_oss,deepseek_v3}.py`,
with `modeling_v2="require"` in force, which is what rules out the built-in
implementation having been measured instead. gpt-oss-120b / sm_103 / tp1 runs
gsm8k; deepseek-r1-0528-nvfp4 / sm_103 / dep4 runs gsm8k for the identity
forward and the `mtp3` variant **paired in one session**, and the acceptance
gate against stock modeling under the same `speculative_config`. The numbers
live with those runs, not here: a number written into this file is one that
drifts with nothing to notice.

The acceptance row is the one that matters for a speculative variant:
rejection sampling makes a miscomputed draft layer *slower*, not wrong, so
boot and accuracy are blind to it and only the acceptance rate against a
reference can see it. Draft lengths 1 and 2 remain ungated -- they are the
dominated end of the measured draft-length axis.

### What replaced the version pin

Out of tree each target hard-asserted a pinned `tensorrt_llm` version at
import, because a target reads private engine surface and a drifting engine
silently voids its records. In tree that assert is meaningless — the target
moves with the trunk — so it is gone, replaced by an SM assert, which is the
part of the identity that does *not* move.

The same reasoning retires the version from the records themselves. An
entry's `ARCHS` is keyed by GPU architecture and nothing else: there is no
external pin left to drift against, and a version written into an entry is a
number that is wrong the next day with nothing to notice. What the claim is
fresh against is the entry's own test, and that CI re-proves on every commit.

The pin was earning its keep out of tree, though, and the migration paid the
bill immediately: the attention backends moved from
`_torch/attention_backend/{interface,trtllm}.py` to
`_torch/attention/backends/`. The compatibility shim left behind re-exports
the names but not the submodule paths, so all five files importing them
failed at import. They now use the canonical path.

The lesson generalizes: with no pin, a target's contact with private engine
surface is checked only by running it. The step contract check does that by
calling the target's own `step_args` on the metadata it was handed, which reads
every field the target consumes -- so a rename upstream becomes a loud failure rather
than a wrong answer. Declaring the same surface a second time, as a list of
names beside the code that reads it, only adds something to forget to update.

## Two facts this migration surfaced about upstream

* **`pip` is a runtime dependency of TensorRT-LLM itself.** `libtensorrt_llm.so`
  locates its bundled kernel headers on the NVRTC JIT path by running
  `pip show tensorrt_llm`. ModelingV2 was merely the first thing to write that
  down (a `uv`-managed venv does not install `pip` by default), and it is not
  a modeling_v2 dependency.
* **The catalog entries are now documentation of upstream ops.** For
  example `_torch/modules/linear.py` calls `cublas_mm(input, module.weight.t(), ...)`
  correctly, but nothing there says why the `.t()` is mandatory or that a
  non-contiguous weight computes a silently wrong answer — `cublas_mm`'s
  `is_valid` does. The reverse duty comes with it: if one of these ops changes
  and its entry does not follow, the misleading is no longer confined to
  modeling_v2.

## Op ownership

An entry only one target calls lives under that target's `ops/`; one that two
or more call lives in `catalog/`. A later target that needs an owned entry
promotes it by moving the file and its test to the shared tree -- the test is
the certification record, so nothing else has to travel with it. The five
other out-of-tree targets and the agent definitions are not migrated in this
batch.
