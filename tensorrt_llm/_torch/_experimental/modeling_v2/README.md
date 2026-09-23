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

```bash
export TRTLLM_MODELING_V2=require        # off | auto | require
```

| `TRTLLM_MODELING_V2` | Behaviour |
|---|---|
| unset or `off` (default) | The resolver returns immediately. Nothing in this package is imported and behaviour is byte-for-byte what it is today. |
| `auto` | Uses a target when one matches this exact configuration; falls back to the built-in implementation when none does. |
| `require` | Raises instead of falling back, naming the criterion that did not match. |

**Use `require` for anything you will attribute to modeling_v2.** Under `auto`,
a configuration that misses a target's criteria silently gets the built-in
implementation — and a performance curve measured that way reads as
modeling_v2's. That is the single most expensive mistake available here.

**Export it before the ranks start**, not merely before `LLM(...)`. Worker
ranks receive the environment as it stood when MPI initialized, and long-lived
ranks under `trtllm-llmapi-launch` receive it once at launch, so a value set
later reaches the driver and not them -- and a driver resolving a target while
its workers resolve the built-in is the silent split this package exists to
prevent.

An environment variable rather than an LLM-API field is a deliberate trade: it
keeps the entire concept inside this package, so the only change modeling_v2
needs anywhere else is the `_resolve_class` hook. The cost is that the switch
does not appear in a run's recorded `llm.args` and cannot be set through
`--extra_llm_api_options`.

The predecessor `MODELING_V2_TARGET` is gone. It named a *target*; this names
only a mode, and routing picks the target from the configuration.

The checkpoint is read exactly as published, with no target-owned
`config.json`.

## How a config finds its target

```
LLM(model=...)  ->  ModelLoader  ->  AutoModelForCausalLM._resolve_class
                                        |
                                        +-- modeling_v2_resolve(config)
                                              |
                    _router_index.py          +-- architectures[0] -> routing module
                    models/<x>/routing.py     +-- one forward-reading decision tree
                                              +-- returns a synthetic class name
                                        |
                                     get_registered_model_class(name)
```

The synthetic name (`ModelingV2GptOss120bSm103Tp1`) is a registry key that no
checkpoint declares. Upstream already does exactly this for `EAGLE3<Arch>`,
which also exists only as a `_resolve_class` rewrite.

To ask why a configuration landed where it did:

```
python -m tensorrt_llm._torch._experimental.modeling_v2.explain \
    --model /path/to/DeepSeek-R1-0528-NVFP4 --tp 4 --ep 4 --attention-dp
```

### What may decide a target

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
explain.py          why a configuration routed where it did
models/<family>/
  routing.py        one forward-reading decision tree per architecture family
  <checkpoint>__<gpu arch>__<parallel>/
                      modeling.py  weights.py
catalog/            the kernel vocabulary: contract .md + wrapper .py
```

Accuracy anchors are not kept here. They live where every other model's do,
`tests/integration/defs/accuracy/references/`, read by the gates CI runs.

The third piece of a catalog entry, its GPU test, lives in the tests tree:

```
tests/unittest/_torch/modeling_v2/
  test_modeling_v2_claims.py     routing tables vs the targets they name (no GPU)
  test_modeling_v2_routing.py    what modeling_v2_resolve does (no GPU)
  <category>/test_modeling_v2_<entry>.py
  comm/_<entry>_op_matrix.py   the two collectives' 4-rank rank bodies
  comm/_rank_job.py            starts one of those and asserts on its exit code
```

That split is not a preference; it is where this repo's CI collects from, and
an in-package test is on no list. It costs one thing worth stating: a receipt
is valid only if it post-dates the last write to *every* file of its entry, so
that check now has to look in both trees.

The two collectives' rank bodies sit in the tests tree with everything else
that only tests run. Both halves are started by file path -- the launcher must
not import `tensorrt_llm`, because that calls `MPI_Init` and an
MPI-initialized process cannot start `mpirun`, and the ranks reach the catalog
by absolute import -- so neither needs a package to live in. Neither those
file names nor their `check_*` bodies match pytest's collection patterns:
each is one fixed 4-rank sequence that cannot run as independent cases.

Identity is the directory name, and it carries all three segments:
`gpt_oss_120b__sm_103__tp1`. They were three nested directories once, which
read as a hierarchy that was never one -- every level had exactly one child,
and each needed an `__init__.py` whose only job was to exist. The class name
carries the same triple; `test_modeling_v2_claims.py` asserts they agree.

### Why beside `_torch/models/`, not inside it

Two upstream mechanisms, and together they are the worst combination —
inheriting the built-in constraints without inheriting the built-in guards:

* `is_builtin_zoo_module` matches on the zoo's package prefix. Inside it,
  modeling_v2 registrations would count as built-in and only fill *empty*
  registry slots. Outside it they are external and always win their slot.
* `test_lazy_model_zoo.py`'s scan of the zoo directory is **non-recursive**,
  so decorators in a subpackage are invisible to it — putting synthetic names
  in the built-in static index would fail its staleness assertion.

Conceptually `models/` means "one architecture, one class, shared across
checkpoints", which is the opposite of what this package is for.

## Gates

1. **Boot** — minutes, binary. `examples/llm-api/quickstart_advanced.py` with
   the target's topology flags and `TRTLLM_MODELING_V2=require` exported. Engine
   cold start, weight-manifest coverage, a handful of greedy continuations.
   Catches catastrophes, not accuracy. A variant that changes the forward gets
   its own minutes-scale gate on the same footing.
2. **Accuracy** — the release criterion. `trtllm-eval` with the protocol from
   `tests/integration/defs/accuracy/references/` and `TRTLLM_MODELING_V2=require`
   exported. One-sided: measured >= reference − tol. This is the gate CI runs,
   in the accuracy suite's modeling_v2 files.
3. **Acceptance** — required whenever a variant is distribution-preserving by
   construction, speculative decoding above all. Rejection sampling holds the
   emitted distribution to the target model's, so a *miscomputed* draft path
   produces correct text more slowly: boot passes, accuracy passes, only
   speed moves. The detector is `acceptance_length` against the same
   checkpoint under stock in-tree modeling at the same workload and the same
   `speculative_config` — i.e. `TRTLLM_MODELING_V2=off` versus `=require`,
   which is now one variable rather than two harnesses.

Compare a variant against an identity run **in the same session**: one target
measured 94.7688 and 95.0720 on a bit-identical forward a day apart, so a
cross-session delta carries session variance into the judgement.

Perf is measured, never gated.

## Status of every record in this tree

**The catalog is fully certified on sm_103. The targets construct but have
never executed.**

Two things voided every receipt in the move: each catalog test file was
rewritten, and the targets moved from sm_100 (B200) to sm_103 (GB300), where
certification is per architecture. The whole catalog was therefore re-run on
GB300 -- **19 of 19 entries pass, 312 certified cells**, plus both 4-rank
collective matrices. The sm_100 receipts were dropped rather than carried:
they predate every file of the entries they sat in, and no sm_100 machine is
in CI to re-take them on. A missing arch key reads as unknown, which is the
honest state.

Getting there surfaced four real differences. None was resolved by widening a
tolerance, and each is written up in its own contract:

* **Op schema drift** (`thop_attention`, `mla_rope_generation`,
  `mla_rope_append_paged_kv_assign_q`). Parameters were renamed and added. The
  wrappers now mirror their schemas argument for argument, so the next drift
  fails loudly rather than shifting a positional list silently.
* **The MoE FC1 epilogue changed block-scale recipe**, bit-exactly:
  `floor(log2(amax))-8` on sm_100, `ceil(log2(amax/448))` on sm_103. The
  reference is architecture-keyed and each arch refutes the other's recipe.
* **torch 2.12 made fp32 matmul default to TF32**, so `cublas_mm`'s *reference*
  was the imprecise side; the op is bit-identical to a TF32-disabled product.
* **The MLA append op now accepts NVFP4 latent pools** as well as fp8 (accepted
  by the op, not certified here).

**gpt-oss-120b / sm_103 / tp1 is gated on GB300.** Both gates pass with
`TRTLLM_MODELING_V2=require` in force, which is what rules out the built-in
implementation having been measured instead:

| Gate | Result |
|---|---|
| boot, 10 greedy continuations | **10/10** keyword asserts |
| gsm8k, full 1319 | **90.6748** (`exact_match,flexible-extract`) vs threshold 85.5989 -- pass by 5.08 |

**deepseek-r1-0528-nvfp4 / sm_103 / dep4 is gated too**, identity and the
`mtp3` variant, on 4 GB300s through `trtllm-llmapi-launch`:

| Gate | Result |
|---|---|
| boot, 10 greedy continuations | **10/10** |
| gsm8k, full 1319 | **95.0720** vs threshold 89.9962 -- pass by 5.08 |
| boot, mtp3 | **10/10**, with the layer-61 MTP module loaded |
| gsm8k, identity vs mtp3 **paired in one session** | delta **-0.3791** against a `\|delta\| < 1.2` criterion -- 0.63 sigma |
| acceptance vs stock | `acceptance_length` **3.3514** vs **3.2752**, ratio **1.023** |

That last row is the one that matters for a speculative variant: rejection
sampling makes a miscomputed draft layer *slower*, not wrong, so boot and
accuracy are blind to it and only the acceptance rate against a reference can
see it. Draft lengths 1 and 2 remain ungated -- they are the dominated end of
the measured draft-length axis.

### What replaced the version pin

Out of tree each target hard-asserted a pinned `tensorrt_llm` version at
import, because a target reads private engine surface and a drifting engine
silently voids its records. In tree that assert is meaningless — the target
moves with the trunk — so it is gone, replaced by an SM assert, which is the
part of the identity that does *not* move.

The same reasoning retires the version from the records themselves. A receipt
is keyed by GPU architecture and nothing else: there is no external pin left to
drift against, and a version written into a contract is a number that is wrong
the next day with nothing to notice. What a receipt is still fresh against is
its own entry's files, and that CI re-proves on every commit.

The pin was earning its keep out of tree, though, and the migration paid the
bill immediately: the attention backends moved from
`_torch/attention_backend/{interface,trtllm}.py` to
`_torch/attention/backends/`. The compatibility shim left behind re-exports
the names but not the submodule paths, so all five files importing them
failed at import. They now use the canonical path.

The lesson generalizes: with no pin, a target's contact with private engine
surface is checked only by running it. Each target declares that surface as
`REQUIRED_TRTLLM_OPS`, `test_modeling_v2_target_contract.py` asserts every name
in it exists, and the first-forward metadata field check catches the rest --
which turns a drifting engine from a wrong answer into a loud failure.

## Two facts this migration surfaced about upstream

* **`pip` is a runtime dependency of TensorRT-LLM itself.** `libtensorrt_llm.so`
  locates its bundled kernel headers on the NVRTC JIT path by running
  `pip show tensorrt_llm`. ModelingV2 was merely the first thing to write that
  down (a `uv`-managed venv does not install `pip` by default), and it is not
  a modeling_v2 dependency.
* **The catalog contracts are now documentation of upstream ops.** For
  example `_torch/modules/linear.py` calls `cublas_mm(input, module.weight.t(), ...)`
  correctly, but nothing there says why the `.t()` is mandatory or that a
  non-contiguous weight computes a silently wrong answer — `catalog/gemm/cublas_mm.md`
  does. The reverse duty comes with it: if one of these ops changes and its
  contract does not follow, the misleading is no longer confined to modeling_v2.

## Not migrated in this batch

The 13 catalog entries no migrated target calls (listed in `catalog/index.yaml`),
the five other targets, and the agent definitions. A later target that needs
one of those entries must bring its contract **and** its receipt, not just the
wrapper — otherwise that target consumes an op with no certification record at
all.
