<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ModelingV2

One self-contained modeling codebase per deployment target, beside the
built-in model zoo rather than inside it.

Where `_torch/models/modeling_deepseekv3.py` is one class serving V3, V3-Lite,
R1 and V3.2 across every GPU generation and parallel topology,
`_torch/_experimental/modeling_v2/models/deepseek_v3/` is one flat forward per (checkpoint,
GPU architecture, parallel topology) triple — assembled from certified `OpWrapper`
entries and trusted through accuracy gates. Model-family ops and kernels may be
shared by that family's targets; target-only code remains inside its target.
The one-to-one correspondence between `models/<x>/` here and `modeling_<x>.py`
there is the point of the exercise.

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
(such as the MoE backend, recurrent-state cache dtype and prefill CUDA graph
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

The tree below is a suggested convention. Each model family can organize its
internal ops, kernels and helpers to suit its implementation, while keeping
family-owned code isolated from unrelated families and sibling-target
implementations. Shared code within a family can live at family scope.

```
_router_index.py    architectures[0] -> routing module. Small, stable, test-guarded.
_native.py          loads a target's native code (see "Native code" below)
models/<family>/
  routing.py        one forward-reading decision tree per architecture family
  ops/<category>/   entries shared only by this family
  kernels/          family-owned CuTe DSL packages
  fake_ops.py       fakes of the family's shared native ops, if it has any
  csrc/             the family's CMakeLists.txt and the native code its targets share
  <checkpoint>__<gpu arch>__<parallel>/
                      modeling.py  weights.py
                      ops/<category>/   entries only this target calls
                      kernels/<category>/  kernel sources its entries launch (optional)
                      fake_ops.py       fakes of the target's native ops, if it has any
                      csrc/             the target's own kernels and torch bindings
catalog/<category>/ entries shared across model families
```

By convention, `ops/` holds operation wrappers and `kernels/` holds their
kernel implementations. Matching category directories and a `common/`
directory for shared helpers can make the relationship easy to follow;
families may also co-locate wrappers and kernels.

An entry, shared or owned, is one `OpWrapper` subclass: `__call__` is the op
call a target makes, `reference` is what it is supposed to compute in plain
torch, `is_valid` names the inputs it accepts and answers wrongly, and `CELLS`
/ `ARCHS` are the configurations and architectures it is proven at. The
wrapper is stateless and `__call__` reads nothing off `self`, so a call site
shows exactly what the kernel receives; `catalog/_op.py` states the rest.

Accuracy anchors are not kept here. They live where every other model's do,
`tests/integration/defs/accuracy/references/`, read by the accuracy gates.

An entry's GPU test lives in the tests tree. A suggested layout mirrors the
entry's ownership scope:

```
tests/unittest/_torch/modeling_v2/
  test_modeling_v2_claims.py     source-level gates over the whole package (no GPU)
  test_modeling_v2_routing.py    what modeling_v2_resolve does (no GPU)
  _validating.py                 the fixtures that interpose is_valid and arm reference
  <category>/test_modeling_v2_<entry>.py                      shared entries
  models/<family>/test_<entry>.py                            family-owned entries
  models/<family>/<target>/<category>/test_modeling_v2_<entry>.py   owned entries
  comm/_<entry>_op_matrix.py   the collectives' rank bodies
  comm/_lockstep.py            the stateful matrices' launcher and shared reference
  comm/_rank_job.py            starts one at a world size and asserts on its exit code
  models/<family>/<target>/comm/_<entry>_op_matrix.py   an owned collective's rank bodies
```

Tests and test-only helpers belong under `tests/`. A family can group its
tests at family level or mirror its target directories. Operation tests are
parametrized off the entry's `CELLS` and compare `__call__` against `reference`
within a band the test states and derives.

The collectives' rank bodies sit in the tests tree with everything else
that only tests run. Both halves are started by file path -- the launcher must
not import `tensorrt_llm`, because that calls `MPI_Init` and an
MPI-initialized process cannot start `mpirun`, and the ranks reach the catalog
by absolute import -- so neither needs a package to live in. Neither those
file names nor their `check_*` bodies match pytest's collection patterns:
each is one fixed sequence over one job's ranks that cannot run as
independent cases. A stateful entry's launcher also takes
`--world-size N --launcher srun`, which runs the same
body as one of N ranks started across nodes: that is how its multi-node
receipt is recorded.

Stateful matrices compare ordered per-tensor SHA-256 fingerprints of raw
bytes together with shape and dtype across ranks.

Identity is the directory name, and it carries all three segments:
`gpt_oss_120b__sm_10x__default`. They were three nested directories once, which
read as a hierarchy that was never one -- every level had exactly one child,
and each needed an `__init__.py` whose only job was to exist. The class name
carries the same triple; `test_modeling_v2_claims.py` asserts they agree.

### Native code

A target's model-owned C++/CUDA belongs with its Python code: `<target>/csrc/`,
and the family's `csrc/` for code several targets share. The family-library
build discovers `models/<family>/csrc/CMakeLists.txt` and builds one torch-op library,
`libth_modeling_v2_<family>.so`, installed next to `libth_common.so`; nothing of it is
linked into the stock libraries, and it loads only when routing picks one of the
family's targets. Adding a model with kernels edits nothing under `cpp/`.
`models/gpt_oss/` is the example: one kernel, used by its weight loader.

The shared discovery, namespace macros and linker script live under
`cpp/tensorrt_llm/modeling_v2/`; the build helpers are in
`cpp/cmake/modules/modeling_v2.cmake`. See the native infrastructure's
[README](../../../../cpp/tensorrt_llm/modeling_v2/README.md) for that layout.

Adding native code to target `foo_nvfp4__sm_103__tp4` of family `foo`:

1. **Declare it** in `models/foo/csrc/CMakeLists.txt`:

   ```cmake
   add_modeling_v2_family(foo)
   add_modeling_v2_target(foo foo_nvfp4__sm_103__tp4 ARCHS 103a SOURCES *.cu *.cpp)
   ```

2. **Write the kernels and their torch bindings** in `foo_nvfp4__sm_103__tp4/csrc/`, in
   the target's namespaces:

   ```cpp
   TRTLLM_MV2_NAMESPACE_BEGIN   // tensorrt_llm::modeling_v2::foo::foo_nvfp4_sm_103_tp4
   ...
   TRTLLM_MV2_NAMESPACE_END
   TRTLLM_MV2_LIBRARY_FRAGMENT(m) { m.def("my_op(Tensor x) -> Tensor"); }
   TRTLLM_MV2_LIBRARY_IMPL(CUDA, m) { m.impl("my_op", &TRTLLM_MV2_NS::myOp); }
   ```

   Include the family's own headers from `models/` (`"foo/foo_nvfp4__sm_103__tp4/csrc/x.h"`);
   any stock header is fine, another target's or family's code is not.

3. **Load it** from the target's `__init__.py`, `native_ops = load_target_ops(__name__)`
   (from `_native`); the target's modules use `from . import native_ops`.

4. **Fake every op** in the target's `fake_ops.py`, which the loader imports right after
   the library opens: `@torch.library.register_fake(f"{namespace_of(__name__)}::my_op")`.

5. **Test it** under `tests/unittest/_torch/modeling_v2/models/foo/`, including
   the native operation's reference comparison and fake implementation.

Code several targets call goes once in the family's `csrc/`, declared with
`add_modeling_v2_shared(foo ARCHS ... SOURCES ...)`; its ops are in `torch.ops.mv2_foo`,
reached with `load_family_ops(__name__)` and faked in `models/foo/fake_ops.py`. A target
with no code of its own declares no `SOURCES`.

- **`ARCHS`** (`NN`, `NNa`, `NNf`) is intersected with the build's architectures; a target
  with none is not built, and is a `NoTarget` at run time. `REQUIRES` (`MULTI_DEVICE`,
  `NVSHMEM`) and `DEPS` (a closed list) are the other keywords; a per-file compile option
  goes on the file with `set_source_files_properties`.
- **Loading**: a library this installation does not carry, or a target it did not build,
  is a `NoTarget` (`auto` keeps the built-in model); one that is installed but fails to
  load, or lacks an op a `fake_ops.py` declares, is a `NativeLoadError`.
- **Building only your family**: `build_wheel.py -D "TRTLLM_MODELING_V2_TARGETS=foo/*"`, and
  after an edit `cmake --build cpp/build --target th_modeling_v2_foo`.
- `test_modeling_v2_native_claims.py` checks these rules on CPU; run it before sending the PR.

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
   passed. One-sided: measured >= reference − tol. Whole-model gates live in
   `tests/integration/defs/accuracy/modeling_v2/test_<family>.py`.
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

## Validation contracts

Certification is per architecture: `ARCHS` says where an entry's cells were
driven; a missing architecture is unknown. The entry's tests describe the
inputs and numerical tolerances behind its claim. Whole-model accuracy
references live under `tests/integration/defs/accuracy/references/`.

Matrix subprocesses use the launcher's interpreter and inherited Python search
path to import the installed TensorRT-LLM package. Install a matching wheel or
complete editable build before running them; an unbuilt checkout must not be
prepended to `PYTHONPATH` in place of the installed package.

### Numerical references

A reference should be at least as precise as the operation it checks. State
and derive the comparison band in the test, accounting for accumulation order,
quantization and output dtype. For example, `cublas_mm` uses
`true_fp32_matmul` to disable TF32 while computing its reference. An operation
accepting a dtype or layout does not by itself certify that configuration;
its cells and tests describe the coverage.

### Engine metadata

Targets use private engine metadata, so their step contracts must be checked
against the engine that supplies it. `TRTLLM_MODELING_V2_VALIDATE=1` enables the
opt-in check: a target calls its own `step_args` on the metadata it receives,
covering the fields its forward consumes. Pass the setting through
`LLM(env_overrides=...)` so it reaches the worker ranks.

## Op ownership

Use these three ownership scopes as the convention for organizing operations
and their implementations:

* **Target:** an entry specific to one deployment lives under
  `models/<family>/<target>/ops/`, with target-owned kernels alongside it.
* **Family:** entries and kernels shared within a model family can live under
  `models/<family>/ops/` and `models/<family>/kernels/`.
* **Catalog:** entries shared across model families live under
  `catalog/<category>/`.

When promoting an entry to a shared scope, move its test with it so the
reference comparisons and certified cells stay attached to the operation.
Each family chooses its internal structure and whether to consolidate
similar implementations; it must preserve isolation from unrelated families
and sibling-target implementations.
