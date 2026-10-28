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
parallel mapping, the GPU's compute capability, the speculative-decoding
config, the KV cache algorithm the engine will run with, and the engine choices
it derives from the `ModelConfig` and the LLM arguments it is also given
(qwen4_exp: MoE backend, recurrent-state cache dtype, prefill CUDA graph
backend) -- and either returns a target class or raises `NoTarget` with the
reason in words. A target declines what its
constructor would refuse, so that `auto` falls back instead of failing mid-build.
Tuning knobs such as `max_batch_size`, `max_num_tokens` and the CUDA graph
bucket lists are not routing inputs: they do not change which code runs, so a
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
  comm/_<entry>_op_matrix.py   the collectives' rank bodies: allgather, reducescatter
                               and allreduce at 2/4/8 ranks in one node, mnnvl_allreduce
                               at 8 ranks over two trays (srun only, see its docstring)
  comm/_rank_job.py            starts one of those and asserts on its exit code
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
each is one fixed 4-rank sequence that cannot run as independent cases.

Identity is the directory name, and it carries all three segments:
`gpt_oss_120b__sm_10x__default`. They were three nested directories once, which
read as a hierarchy that was never one -- every level had exactly one child,
and each needed an `__init__.py` whose only job was to exist. The class name
carries the same triple; `test_modeling_v2_claims.py` asserts they agree.

### Default and specialized targets

The parallel segment of a target's name is either one topology (for
example `dep4`) or `default`. A `default` target computes its weight shapes
from the engine's `Mapping` and covers every layout its routing module
admits. A named topology is a specialization, written only where one layout
earns its own forward, and routing tries it before the default. This keeps
the number of targets proportional to the specializations someone chose to
write, not to the number of layouts a model is deployed at. Neither model
has one today: the default forward computes the same thing at every layout
a specialization once served.

The architecture segment `sm_10x` covers sm_100 (B200, GB200) and sm_103
(B300, GB300): both run the same kernels, and every entry the two targets
call is certified on both. A node holds four GPUs on GB200/GB300 and eight
on B200/B300, so the same layout can sit inside one node on one machine and
span two on the other; the target picks its allreduce from the mapping, not
from the GPU name.

`gpt_oss_120b__sm_10x__default` is the first, and gpt-oss's only target. It
admits 12 layouts, the ones it has been gated on: tp1, tp2 (MoE split by
intermediate or by expert), tp4 (by intermediate, by expert, or tp2 x ep2),
tp8 (by intermediate or by expert), dep2, dep4 (by expert or tp2 x ep2) and
dep8. No layout has a specialization: at one GPU the default forward is the
single-GPU forward, with no collective. The layout is settled once, when a
target is constructed, not by branching in the forward: at one GPU both
residual-norm slots are the local fused-add norm; under attention TP they are
an allreduce with the norm fused in (`comm/allreduce` inside one node,
`comm/mnnvl_allreduce` across the trays of an NVLink domain); under attention DP they are the local
fused-add norm and the MoE runs between `comm/allgather` and
`comm/reducescatter`. Routing also refuses three mapping facts the table does
not show: an attention TP size other than the TP size, lm_head TP under
attention DP, and attention TP across nodes without multi-node NVLink (the
built-in model falls back to NCCL there; the target has no such path).

Where it differs from the built-in `GptOssForCausalLM` at the same layout:

| | built-in | default target |
|---|---|---|
| allreduce inside one node | `AllReduce` module: symmetric memory, then the raw op with a strategy autotuner | the raw op with AUTO: one-shot / two-shot over the IPC workspace |
| MoE under attention DP, pure EP | all-to-all (NVLink one-sided first) when available | always allgather + reduce-scatter |
| embedding under attention TP | vocab-split, then gathered | replicated on every rank |

Across trays the two agree: both run the MNNVL kernel with the norm fused
in, the built-in model through `MNNVLAllReduce` and this target through
`comm/mnnvl_allreduce`.

`r1_0528_nvfp4__sm_10x__default` is the second, and the only deepseek
target: at dep4 it computes what a dep4 specialization would, so there is
none. It admits tp2, tp4 and tp8 (experts split by intermediate or by
expert, tp4 also by tp2 x ep2, and tp8 also by tp4 x ep2 and tp2 x ep4),
dep2, dep4 with experts split by expert, by intermediate or tp2 x ep2, dep8
with experts split by expert, by intermediate or tp2 x ep4, and dep16. dep16
runs the same allgather and reduce-scatter round trip as dep8. The bitwise
collective matrices stop at 8 ranks: their exact value sets and the ring
summation order they pin hold only up to there. So dep16 rests on the gsm8k
run that admitted it. Under attention TP a rank holds its share of the 128
MLA heads (`q_b_proj` and `kv_b_proj` rows, `o_proj` columns) and of every
dense MLP and shared expert; `q_a_proj`, `kv_a_proj_with_mqa` and the latent
KV cache are the same on every rank. The residual-norm slots and the
attention-DP round trip are bound the same way as gpt-oss's.

The checkpoint's MTP layer (one-model MTP-Eagle) is one more decoder layer of
this target, cut per rank like the others and run through the same
residual-norm slots and expert round trip. Its experts are bf16 -- modelopt
excludes layer 61 from NVFP4 -- so they go to `fused_moe`, an entry this
target owns, rather than the NVFP4-only trtllm-gen runner. The forward
computes the MTP layer at every layout, but routing admits a
`speculative_config` only at the six its draft acceptance was measured at:
dep4, tp4, tp4+ep4, dep4 with experts split by intermediate, tp8 with tp2 x
ep4, and dep8 -- each residual-norm path and each expert split.

Where it differs from the built-in `DeepseekV3ForCausalLM` at the same layout,
beyond the two allreduce and all-to-all rows above:

| | built-in | default target |
|---|---|---|
| dense MLP (layers 0-2) at tp8 over two trays | split 4 ways inside a tray (`_compute_mlp_tp_size`) | split 8 ways, completed by the same allreduce |
| fusions around the norms | NVFP4 quantization fused into the norm, MoE allreduce fused with the residual add | separate quantize and norm calls |

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

**Every entry the gpt-oss and deepseek default targets call is certified on
sm_100 and sm_103, and both targets are gated on both.** The qwen4_exp target
and the entries it owns are certified on sm_103.

The default targets' 21 entries -- 20 shared, 1 (`fused_moe`) owned by deepseek's default target -- each drive
their cells through `__call__` and `reference`. Where that runs depends on the
entry:

* the 17 single-GPU entries run in pre-merge CI;
* `allgather` and `reducescatter` run their 4-rank matrices, and `allreduce`
  its 2- and 4-rank matrices, in the post-merge 4-GPU job;
* `mnnvl_allreduce` needs 8 ranks over two trays. Its matrix runs only by
  hand under srun (see its docstring), and no CI job re-runs it.

Certification is per architecture: `ARCHS` says
where the cells were driven, and a missing architecture reads as unknown,
which is the honest state.

Getting the catalog there surfaced four real differences. None was resolved by
widening a tolerance, and each is written up in its own entry:

* **Op schema drift** (`thop_attention`, `mla_rope_generation`,
  `mla_rope_append_paged_kv_assign_q`). Parameters were renamed and added. The
  wrappers now mirror their schemas argument for argument, so the next drift
  fails loudly rather than shifting a positional list silently.
* **The MoE FC1 epilogue's block-scale recipe**, bit-exactly:
  `ceil(log2(amax/448))` on both sm_100 and sm_103, not the OCP
  `floor(log2(amax))-8`. The reference is architecture-keyed, and the probe
  refutes the OCP form, so a cubin that switches to it fails.
* **torch 2.12 made fp32 matmul default to TF32**, so `cublas_mm`'s *reference*
  was the imprecise side; the op is bit-identical to a TF32-disabled product.
* **The MLA append op now accepts NVFP4 latent pools** as well as fp8 (accepted
  by the op, not certified here).

Both targets are gated by the accuracy suite's modeling_v2 files,
`tests/integration/defs/accuracy/modeling_v2/test_{gpt_oss,deepseek_v3}.py`,
with `modeling_v2="require"` in force, which is what rules out the built-in
implementation having been measured instead. The two `default` targets ran
gsm8k at every layout before their routing admitted it, gpt-oss's tp1
included; deepseek-r1-0528-nvfp4 / sm_10x / default also runs gsm8k for the
identity forward and the `mtp3` variant **paired in one session**, and the
acceptance gate against stock modeling under the same `speculative_config`,
at each of the six layouts MTP is admitted at. CI runs a representative
subset -- one layout per residual-norm path and per expert split, gpt-oss
tp1, the deepseek `mtp3` pair and acceptance gate at dep4, and deepseek
chunked prefill at tp4+ep4 and dep4 -- and the default shards tests pin
every layout's weight cut, the MTP layer's included, on the CPU. The numbers
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
