---
name: staircase-modeling-core
description: Load whenever a target's modeling.py or weights.py is being written or edited — the invariants any forward must hold, the products a target consists of, and the from-scratch assembly flow. assembler loads it for core assembly and for every feature increment; tuner loads it before editing a forward.
---

# staircase-modeling-core

Two halves, and most readers need only the first.

| you are | read |
|---|---|
| assembler, core assembly | everything |
| assembler, a feature increment | **Invariants** — you are editing a file that already holds them |
| tuner, editing a forward | **Invariants**, especially `references/graph-and-sync.md` |

## Which reference, when

`references/` is read by need, not front to back:

| file | read it when |
|---|---|
| `runtime-interface.md` | always, before writing — registration, shell/model split, weight lifecycle, manifest convention, forward duties, fail-fast ladder, serving-time costs |
| `graph-and-sync.md` | **always, before touching a forward** — what a replayed forward may contain, and which patterns stall the host |
| `routing-identity.md` | placing a new target, or asking whether some quantity may select one |
| `step-args.md` | building the attention call's arguments |
| `adapter-rules.md` | this target has a feature that talks to a runtime-owned service |
| `latent-kv-cache.md` | the checkpoint's `config.json` says MLA |
| `hybrid-recurrent-state.md` | `layer_types` mixes `linear_attention` with `full_attention` — the recurrent layers need a state pool the runtime allocates only when the published config matches its predicate |
| `expert-weight-packing.md` | sparse MoE |
| `attention-sinks.md` · `sliding-window-attention.md` | the config says so |

The mechanism files are not optional reading dressed up as reference: each one
records what an earlier target had to establish by probing. Reading the right
one is the difference between an afternoon and a week.

## Invariants

These hold for every edit to a target's modeling code, by anyone, forever.

1. **Closed vocabulary.** Inside `forward` and the private methods it reaches,
   every call that creates or transforms a tensor is a catalog entry. Only
   tensor-metadata reads and Python control flow live outside. Scope is the
   whole trick — a whole-file scan false-positives on the rope table's
   `arange`/`cos`/`sin` and on load-time `.t()`/`.to()`, which run at init.
2. **The static op declaration lists exactly what the forward calls.**
   `REQUIRED_TRTLLM_OPS` names every `torch.ops.trtllm.*` symbol the forward
   reaches, and
   `tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py`
   asserts that this build registers each one (plus the pybind entry point the
   attention op is reached through). The list must *equal* the actual call
   set — a stale entry is as wrong as a missing one. With no version pin, that
   declaration is the only thing standing between a drifted engine and a wrong
   answer: it turns the drift into a loud failure instead.
3. **Every axis the assembly depends on carries an assert**, read from
   `pretrained_config`. An assert is how a reused pattern refuses a
   checkpoint it does not fit.
4. **Parameters are declared flat**, `torch.empty` only, HF `[out, in]`
   storage. Derivations happen in the post-load path, never at declaration.
5. **The manifest asserts coverage in both directions**: every declared
   parameter is filled, and every checkpoint key is consumed — with one
   explicit escape, the **known-not-loaded key set**, for weights this target
   deliberately does not implement (an unimplemented feature's module, an
   off-window expert range). Anything in that set is named in `weights.py`
   beside the predicted-leftover assert, with the reason, and reported in the
   run journal. Without the escape the assert fires on a correct target; with
   it used loosely, a real miss hides.
6. **Graph safety and sync-freedom**, per `references/graph-and-sync.md`.
   These are writing rules, not tuning rules: a forward that violates them is
   wrong the first time it is captured.
7. **The forward is flat and serves any batch composition.** One code path,
   branching only on metadata and per-capture constants.
8. **Fail loudly on what the target does not implement.** The fail-fast ladder
   is the difference between an unsupported input and a wrong answer.
9. **Self-containment.** No shared helpers across targets. Duplication between
   targets is the architecture, not a smell.
10. **Every fixed size answers to an engine limit.** A module- or class-level
    integer that bounds something (rows, tokens, batch, slots, pages) or sizes
    a buffer, and every buffer sized by a literal of 1024 or more, is either
    derived from an engine argument at construction
    (`model_config.max_num_tokens`, `max_batch_size`, `max_seq_len`) or
    declared on its own line:

    ```python
    _TOPK_MAX_ROWS = 16384  # capacity: max_num_tokens
    _GATHER_ROWS = 65536  # capacity: max_num_tokens * world_size
    _WINDOW = 128  # capacity: fixed: the checkpoint's sliding window
    ```

    (or in a module-level `CAPACITY_BOUNDS = {"NAME": "<binding>"}`). The
    orchestrator audits every one statically against the serving envelope at
    review and at delivery: unbound, or bound and below the envelope, fails.
    The case that made this an invariant: a `_TOPK_MAX_ROWS = 8192` offsets
    buffer killed every engine above `max_num_tokens` 8192 in warm-up while
    every kernel it fed was certified to 16384, and no gate launched above
    8192 to see it.

## Products

A target directory holds **two files**:

| product | |
|---|---|
| `modeling.py` | the `REQUIRED_TRTLLM_OPS` declaration · geometry with asserts · the flat `ParameterDict` · the step-args builder with its per-entry CUDA-graph classification · the forward · the runtime-value dump (below) · loud asserts for the unimplemented · the `@register_auto_model(<synthetic name>)` shell |
| `weights.py` | the MANIFEST data table and load loop with bidirectional coverage asserts, and the predicted non-load sets with their reasons; post-load derivations live in modeling's `post_load_weights` |

Four products the standalone layout had are gone. Each one's *responsibility*
moved rather than disappearing, and knowing where is how you avoid recreating
it:

| gone | where the responsibility went |
|---|---|
| `model_dir/` (stub `config.json` + symlinks) | nowhere — **the checkpoint is read exactly as published**, with no target-owned `config.json`. Nothing about the model is the target's to restate, so nothing can disagree |
| `smoke.py` | the **boot gate**: `examples/llm-api/quickstart_advanced.py` with the target's topology flags and `TRTLLM_MODELING_V2=require`, run by the orchestrator. You author its keyword cases; you do not author its runner |
| `llm_args.yaml` | the topology comes from the **routing decision** and the gate's CLI flags. There is no file for it, which is also why a tuning knob can no longer be smuggled in beside it |
| `TARGET.md` | **gate records go in the run journal** (workspace-scoped, uncommitted). Identity lives in the target path plus `models/<family>/routing.py` |

And the thing that used to decide which class name a target answered to —
`@register_auto_model("StaircaseForCausalLM")`, and the whole discussion of
when a target must register under some other name instead — is **replaced by
routing**. The checkpoint's own `architectures[0]` stays what it always was;
`modeling_v2_resolve` rewrites it into a synthetic registry key
(`ModelingV2GptOss120bSm103Tp1`) that the target's module registers, and the
rewrite is local to `_resolve_class` — every other runtime branch keyed on the
architecture string still sees the published name. So a target neither picks
nor needs to pick that name: `models/<family>/routing.py` does, the class name
carries the same triple as the path, and `test_modeling_v2_claims.py` asserts
they agree. Read `references/routing-identity.md` before adding a target or
proposing a new criterion.

**Registration goes at module scope** in `modeling.py`, which is what importing
the module does. The routing module's `TARGET_MODULES` entry names that module
by dotted path, and `modeling_v2_resolve` imports it once the tree has claimed
the config — nothing else would put the class behind the synthetic key. Above
world size 1 the engine is built in spawned ranks that import the module
themselves, and the class travels to them by pickle, which resolves it by
module name; an ordinary dotted module satisfies that without any help.

## The runtime-value dump

`forward` records the **metadata** of every catalog call it makes — shapes,
dtypes, the constants passed — into a set, and the boot gate's output carries
it out of the run.

This exists because of a class of value that nothing else surfaces: the
shapes the engine **varies at runtime**. Decode batch sizes, page sizes,
feature flags. They are written nowhere in the target, so reading the forward
does not reveal them, and a target once merged with 34 of the 35 batch sizes
its engine captured sitting outside the contract — every *constant* it passed
was inside.

Four rules make it safe and sufficient:

- **metadata only.** Never a tensor value: `.item()`/`.cpu()` are forbidden in
  a forward, and metadata is exactly what a certified column is written in
- **no allocation in the hot path.** A set of tuples, appended to; nothing
  that grows per token
- **capture is enough.** CUDA graph capture *executes* Python — that is why
  host-derived scalars get baked in — so the dump naturally covers every
  captured batch size. Replay executes none of it, so the steady state pays
  nothing
- **off unless asked for.** Record and print only when
  `TRTLLM_MODELING_V2_VALUE_DUMP=1` (read once, at import). Every gate and
  witness run sets it; a served engine must not print the dump at each exit

Comparing the dump against the contracts is a *reading* job, not an automated
one: a certified column is prose. The target's duty ends at exposing the
values; assembler judges them, and smith is dispatched only for what is
genuinely uncovered.

## From-scratch flow

1. **Derive before writing.** Take the computation-step list from the
   workspace's `vocabulary.yaml` — it already carries this checkpoint's value
   on every axis, established in Phase B. Read the checkpoint's safetensors key
   list and the HF reference for the compute graph, and decide each step's
   mapping onto catalog entries. An unmappable step is a vocabulary gap: report
   it before creating files.
2. **Place the target and its routing.** Create
   `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/`,
   and add the target to that family's `routing.py`: the config-shape
   fingerprint, the parallel segment, the synthetic class name and its
   `TARGET_MODULES` entry. A family nothing routes yet also needs one row in
   `_router_index.py`. `references/routing-identity.md` is what may decide any
   of this.
3. **Write `modeling.py` + `weights.py`**, and draft the boot gate's keyword
   cases. Keywords stay provisional.
4. **Inner loop — edit → import dry-run → boot gate, until the gate passes.**
   The dry-run imports the target module and asserts the class registers under
   the synthetic name, and that `explain.py` routes this configuration to it.
   **Do not lint between edits.** The run lints once, at delivery, with the
   repo's `pre-commit`; during iteration spend the turns on the model, not on
   whitespace. **The first engine run doubles as keyword verification** —
   freeze the cases only after observing the continuations.
5. **Check the dump** against the contracts. Anything outside a certified
   column: dispatch `Agent(smith)` with `kernel-verify`, resume on the receipt.
6. **Accuracy gate.** A shortfall is a debug signal, not a verdict: the
   reference entry pins both the anchor and the protocol, so a miss usually
   means a real defect and you hold the derivation table to find it. Two
   rules — every re-gate is backed by one explicit causal hypothesis (the full
   set costs about an hour and same-protocol rerun noise is 0.1–0.3 points, so
   a hypothesis-free rerun is expensive and information-free), and after three
   consecutive hypothesis-backed fixes without improvement, stop and report
   the shortfall with the audit trail. The anchor and tolerance are read-only:
   you fix your assembly, never the bar.
7. **Fill the run journal** from observed runs only.
8. **Lint once with the repo's `pre-commit`, then commit** your own files, by
   explicit path.

## Feature increments

Same file, same invariants, plus:

- the feature's own skill says where it attaches and what it owes
- **the base path stays exactly as it was.** If the feature is off, the forward
  must be what the recorded gate measured — bit for bit
- the weights the feature turns on come *out* of the known-not-loaded key set,
  and `weights.py`'s predicted non-load set says so
- the run journal records what the feature needs to run at all: its activation
  config and any boot requirement you discovered while gating it, as the gate
  ran with them. Neither is a tuning knob, and tuner may not change them
- gate it: boot with the config → accuracy, plus a third gate **only if**
  the feature's skill defines one. A third gate is earned by exactly one
  property — a miscomputed implementation still producing a correct answer —
  and today only speculative decoding has it. A feature that is merely inert
  produces correct output, which makes that a tuning question

## Report

- target path; checkpoint path and hash summary; the GPU used. Routing
  recognizes a checkpoint by its shape and cannot tell a fine-tune from the
  original, so the record of which bytes you measured is what pins the claim
- **derivation table**: each axis and what this model required. Flag every axis
  where this target differs from the exemplar you read
- **vocabulary audit**: the forward's call list, each mapped to its
  `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml` path.
  Scope it to `forward` plus the private methods it reaches
- **dump review**: which values the engine produced at runtime, and where each
  sits against its contract
- **gate outcomes**: boot-gate cases and verdicts; accuracy measured versus
  reference and threshold — verbatim numbers, or the precise blocker
- **symptoms**: knowledge gaps (the exact question your sources could not
  answer), vocabulary gaps, upstream oddities. State separately anything you
  had to establish by observation, naming the mechanism it belongs to — that
  is what gets written back so the next target does not repeat the probe
- **signal versus noise**: same-protocol reruns move 0.1–0.3 points, so a
  sub-0.5-point change is not a result. Never call a construct load-bearing on
  that evidence; downstream work takes the claim at face value and pays to
  disprove it
- files created; lint status
