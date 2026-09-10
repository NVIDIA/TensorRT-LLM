# Staircase

One self-contained modeling codebase per deployment target, beside the
built-in model zoo rather than inside it.

Where `_torch/models/modeling_deepseekv3.py` is one class serving V3, V3-Lite,
R1 and V3.2 across every GPU generation and parallel topology,
`_torch/staircase/models/deepseek_v3/` is one flat forward per (checkpoint,
GPU architecture, parallel topology) triple — assembled only from `catalog/`
entries, sharing nothing with its siblings, and trusted through accuracy gates
instead of shared abstractions. The one-to-one correspondence between
`models/<x>/` here and `modeling_<x>.py` there is the point of the exercise.

## Using it

```bash
export TRTLLM_STAIRCASE=require        # off | auto | require
```

| `TRTLLM_STAIRCASE` | Behaviour |
|---|---|
| unset or `off` (default) | The resolver returns immediately. Nothing in this package is imported and behaviour is byte-for-byte what it is today. |
| `auto` | Uses a target when one matches this exact configuration; falls back to the built-in implementation when none does. |
| `require` | Raises instead of falling back, naming the criterion that did not match. |

**Use `require` for anything you will attribute to staircase.** Under `auto`,
a configuration that misses a target's criteria silently gets the built-in
implementation — and a performance curve measured that way reads as
staircase's. That is the single most expensive mistake available here.

**Export it before the ranks start**, not merely before `LLM(...)`. Worker
ranks receive the environment as it stood when MPI initialized, and long-lived
ranks under `trtllm-llmapi-launch` receive it once at launch, so a value set
later reaches the driver and not them -- and a driver resolving a target while
its workers resolve the built-in is the silent split this package exists to
prevent.

An environment variable rather than an LLM-API field is a deliberate trade: it
keeps the entire concept inside this package, so the only change staircase
needs anywhere else is the `_resolve_class` hook. The cost is that the switch
does not appear in a run's recorded `llm.args` and cannot be set through
`--extra_llm_api_options`.

The predecessor `STAIRCASE_TARGET` is gone. It named a *target*; this names
only a mode, and routing picks the target from the configuration.

The checkpoint is read exactly as published, with no target-owned
`config.json`.

## How a config finds its target

```
LLM(model=...)  ->  ModelLoader  ->  AutoModelForCausalLM._resolve_class
                                        |
                                        +-- staircase_resolve(config)
                                              |
                    _router_index.py          +-- architectures[0] -> routing module
                    models/<x>/routing.py     +-- one forward-reading decision tree
                                              +-- returns a synthetic class name
                                        |
                                     get_registered_model_class(name)
```

The synthetic name (`StaircaseGptOss120bSm103Tp1`) is a registry key that no
checkpoint declares. Upstream already does exactly this for
`MTPDraftModelForCausalLM`, which also exists only as a `_resolve_class`
rewrite.

To ask why a configuration landed where it did:

```
python -m tensorrt_llm._torch.staircase.explain \
    --model /path/to/DeepSeek-R1-0528-NVFP4 --tp 4 --ep 4 --attention-dp
```

### What may decide a target

A quantity may be a routing criterion only if it is known when
`_resolve_class` runs **and** constant for the engine's life — and only if it
changes the *structure of the forward*. Config shape, SM version, parallel
topology qualify. `max_num_tokens` and `cuda_graph_config.max_batch_size` pass
the first test and fail the second: they are tuning knobs and belong in a
target's `configs/` variant. Batch composition and `num_contexts` fail the
first outright — those move every step, and a target chosen from them would
be chosen once and then be wrong. Per-step specialization is a separate
mechanism (dispatch inside a target's `forward`), not a routing dimension.

### Checkpoint identity is a shape fingerprint

Routing recognizes a checkpoint by `(num_hidden_layers, hidden_size, ...)`,
the same sniffing idiom as `is_mla` / `is_nemotron_hybrid` upstream. It cannot
tell a fine-tune from the original. That is a deliberate trade, and it changes
what a gate record means: not "this target passed" but "this modeling code
passed **on the checkpoint whose digests TARGET.md records**". Running it on
any other checkpoint of the same shape is ungated. Each `TARGET.md` says so.

## Layout

```
_router_index.py    architectures[0] -> routing module. Small, stable, test-guarded.
explain.py          why a configuration routed where it did
models/<family>/
  routing.py        one forward-reading decision tree per architecture family
  targets/<checkpoint>/<gpu arch>/<parallel>/
                    modeling.py  weights.py  TARGET.md  [configs/]
catalog/            the kernel vocabulary: contract .md + wrapper .py
references/         accuracy anchors, keyed by HF repo name
docs/               cross-target mechanism and runtime notes
```

The third piece of a catalog entry, its GPU test, lives in the tests tree:

```
tests/unittest/_torch/staircase/
  test_staircase_claims.py     routing tables vs the targets they name (no GPU)
  test_staircase_routing.py    what staircase_resolve does (no GPU)
  <category>/test_staircase_<entry>.py
```

That split is not a preference; it is where this repo's CI collects from, and
an in-package test is on no list. It costs one thing worth stating: a receipt
is valid only if it post-dates the last write to *every* file of its entry, so
that check now has to look in both trees.

The two collective entries keep their rank bodies in `catalog/comm/`
(`allgather_test.py`, `reducescatter_test.py`) because the launcher re-execs
them as `python -m` and the ranks need the package context; only the collected
shells moved.

Identity is the path. `targets/` keeps all three segments rather than
flattening them, and the class name carries the same triple;
`test_staircase_claims.py` asserts they agree.

### Why beside `_torch/models/`, not inside it

Two upstream mechanisms, and together they are the worst combination —
inheriting the built-in constraints without inheriting the built-in guards:

* `is_builtin_zoo_module` matches on the zoo's package prefix. Inside it,
  staircase registrations would count as built-in and only fill *empty*
  registry slots. Outside it they are external and always win their slot.
* `test_lazy_model_zoo.py`'s scan of the zoo directory is **non-recursive**,
  so decorators in a subpackage are invisible to it — putting synthetic names
  in the built-in static index would fail its staleness assertion.

Conceptually `models/` means "one architecture, one class, shared across
checkpoints", which is the opposite of what this package is for.

## Gates

1. **Boot** — minutes, binary. `examples/llm-api/quickstart_advanced.py` with
   the target's topology flags and `TRTLLM_STAIRCASE=require` exported. Engine
   cold start, weight-manifest coverage, a handful of greedy continuations.
   Catches catastrophes, not accuracy. A target whose `configs/` holds a
   variant that changes the forward has its own file there, so that path gets
   a minutes-scale gate too.
2. **Accuracy** — the release criterion. `trtllm-eval` with the protocol from
   `references/accuracy.yaml` and `TRTLLM_STAIRCASE=require` exported. One-sided: measured >= reference − tol.
   This is the gate CI runs, as `accuracy/test_staircase.py`.
3. **Acceptance** — required whenever a variant is distribution-preserving by
   construction, speculative decoding above all. Rejection sampling holds the
   emitted distribution to the target model's, so a *miscomputed* draft path
   produces correct text more slowly: boot passes, accuracy passes, only
   speed moves. The detector is `acceptance_length` against the same
   checkpoint under stock in-tree modeling at the same workload and the same
   `speculative_config` — i.e. `TRTLLM_STAIRCASE=off` versus `=require`,
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
GB300 under 1.3.0rc26 -- **19 of 19 entries pass, 312 certified cells**, plus
both 4-rank collective matrices.

Getting there surfaced four real differences. None was resolved by widening a
tolerance, and each is written up in its own contract:

* **Op schema drift rc21 -> rc26** (`thop_attention`, `mla_rope_generation`,
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

**gpt-oss-120b / sm_103 / tp1 is gated on GB300.** Its checkpoint was
downloaded and every digest re-verified against `TARGET.md`, so the new records
and the old sm_100 ones were measured on byte-identical weights. Both gates
pass with `TRTLLM_STAIRCASE=require` in force, which is what rules out the built-in
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
| boot, `configs/mtp3.yaml` | **10/10**, with the layer-61 MTP module loaded |
| gsm8k, identity vs mtp3 **paired in one session** | delta **-0.3791** against a `\|delta\| < 1.2` criterion -- 0.63 sigma |
| acceptance vs stock | `acceptance_length` **3.3514** vs **3.2752**, ratio **1.023** |

That last row is the one that matters for a speculative variant: rejection
sampling makes a miscomputed draft layer *slower*, not wrong, so boot and
accuracy are blind to it and only the acceptance rate against a reference can
see it. `configs/mtp{1,2}.yaml` remain ungated -- they are the dominated end of
the measured draft-length axis.

Measured statements throughout the contracts are left exactly as written.
They are true records of what was observed on sm_100, and rewriting them would
manufacture GB300 evidence that does not exist.

### What replaced the version pin

Out of tree each target hard-asserted `tensorrt_llm == 1.3.0rc21` at import,
because a target reads private engine surface and a drifting engine silently
voids its records. In tree that assert is meaningless — the target moves with
the trunk — so it is gone, replaced by an SM assert, which is the part of the
identity that does *not* move.

The pin was earning its keep, though, and the migration paid the bill
immediately: between rc21 and rc26 the attention backends moved from
`_torch/attention_backend/{interface,trtllm}.py` to
`_torch/attention/backends/`. The compatibility shim left behind re-exports
the names but not the submodule paths, so all five files importing them
failed at import. They now use the canonical path.

The lesson generalizes: with no pin, a target's contact with private engine
surface is checked only by running it. The import-time
`_check_static_contract` op-existence loop and the first-forward metadata
field check are what turn that from a wrong answer into a loud failure, which
is why both survived the move.

## Two facts this migration surfaced about upstream

* **`pip` is a runtime dependency of TensorRT-LLM itself.** `libtensorrt_llm.so`
  locates its bundled kernel headers on the NVRTC JIT path by running
  `pip show tensorrt_llm`. Staircase was merely the first thing to write that
  down (a `uv`-managed venv does not install `pip` by default), and it is not
  a staircase dependency.
* **The catalog contracts are now documentation of upstream ops.** For
  example `_torch/modules/linear.py` calls `cublas_mm(input, module.weight.t(), ...)`
  correctly, but nothing there says why the `.t()` is mandatory or that a
  non-contiguous weight computes a silently wrong answer — `catalog/gemm/cublas_mm.md`
  does. The reverse duty comes with it: if one of these ops changes and its
  contract does not follow, the misleading is no longer confined to staircase.

## Not migrated in this batch

The 13 catalog entries no migrated target calls (listed in `catalog/index.yaml`),
the five other targets, and the agent definitions. A later target that needs
one of those entries must bring its contract **and** its receipt, not just the
wrapper — otherwise that target consumes an op with no certification record at
all.
