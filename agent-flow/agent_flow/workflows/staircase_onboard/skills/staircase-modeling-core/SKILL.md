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
| `runtime-interface.md` | always, before writing — registration, the shell / core / two-target split, binding the entries, weight lifecycle, the weight table, forward duties, fail-fast ladder, serving-time costs |
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
`runtime-interface.md` §1, §3, §3a and §5 show each one in code; the two
in-tree targets (`gpt_oss/gpt_oss_120b__sm_103__tp1`,
`deepseek_v3/r1_0528_nvfp4__sm_103__dep4`) are the exemplars.

1. **Closed vocabulary.** Inside each target's `forward` and the private
   methods it reaches, every trtllm op — anything that is not native torch —
   is called through an entry: a shared `catalog/` entry, or a target-owned
   op under `<target>/ops/<category>/` when only this target calls it.
   Native torch calls (`torch.empty`, `torch.reshape`, `torch.add`,
   `nn.functional.embedding`, ...) are made directly; there are no torch
   mirrors. Composing an op's math out of torch calls where an engine op
   serves it is still a vocabulary gap. Scope is the whole trick — a
   whole-file scan false-positives on the rope table's `arange`/`cos`/`sin`
   and on load-time `.t()`/`.to()`, which run at init.
2. **The engine surface is checked by running it, never declared twice.**
   Each target's `step_args(md)` is the one projection of the attention
   metadata it consumes, and `Target._check_step_contract` calls it on the
   first forward under `TRTLLM_MODELING_V2_VALIDATE=1`. A renamed custom op
   fails at its call. Do not add `REQUIRED_TRTLLM_OPS`, a `_STEP_FIELDS`
   tuple, or any hand-kept list of names beside the code that reads them —
   it is only something to forget to update.
3. **Routing decides identity; the target carries no configuration
   asserts.** Every axis the assembly depends on is either a routing
   criterion in `models/<family>/routing.py` (it changes the structure of the
   forward: config-shape fingerprint, SM, parallel topology) or read from
   `pretrained_config` where it is bound. A configuration no target
   implements is refused by routing — `require` raises naming the criterion.
   When a reused pattern would not fit some checkpoint of the same
   fingerprint, the fix is a sharper criterion, not an assert in the target.
   A guard is still owed only where routing cannot see the fact — a runtime
   combination a feature cannot serve (its skill names them), or a field the
   runtime reads from somewhere other than the config the target read. Those
   fail loudly at construction and say why; re-validating what the
   fingerprint already pins does not.
4. **Weights are declared once, in the weight table.** `weights.py`
   subclasses `ModelWeights` with one `W` row per weight role (shape, source,
   dtype, transform, layers) and a `dims(core)` that is the only place a
   derived width or padded size is computed; `MODEL_WEIGHTS` is its instance.
   The core declares `self.w = MODEL_WEIGHTS.declare(dims)` — a flat
   `nn.ParameterDict`, `torch.empty` only, HF `[out, in]` storage — and the
   shell's `load_weights` calls `MODEL_WEIGHTS.load`. Never write a manifest
   by hand: `manifest()` unrolls the table. Load-time layout changes are a
   row's `transform`; state derived from loaded weights is built in
   `post_load_weights`, never at declaration.
5. **Load coverage holds in both directions**: every declared parameter is
   filled, and every checkpoint key is consumed — with one explicit escape,
   `expected_unconsumed(core, weights)`, the **known-not-loaded key set**,
   for weights this target deliberately does not implement (an unimplemented
   feature's module, an off-window expert range). Each member is named there
   with its reason and reported in the run journal. Without the escape the
   assert fires on a correct target; with it used loosely, a real miss hides.
6. **Graph safety and sync-freedom**, per `references/graph-and-sync.md`.
   These are writing rules, not tuning rules: a forward that violates them is
   wrong the first time it is captured.
7. **The core dispatches; a prefill and a decode target run the step.** The
   core's `forward` calls `advance_step_generation()` once and hands the step
   to `self._select_target(md)`: `DecodeTarget` when `phase_of(md)` is
   `Phase.DECODE` (`num_contexts == 0`), `PrefillTarget` otherwise — a mixed
   batch is prefill's. Both are built in the core's `post_load_weights`,
   hold the core by reference, and derive from `Target` directly with their
   own `__init__`, `step_args` and `forward` — no shared body, base or mixin
   between them. `DecodeTarget` states `num_contexts=0, num_ctx_tokens=0`
   and never reads either back. Each forward is flat: one code path,
   branching only on metadata and per-capture constants. A draft step reads
   the phase per draft step, never cached.
8. **Bind once; call with data.** Each target instantiates its own entries
   and binds them in its `__init__` against the real weights —
   `bind_const(...)` for values every layer shares, `bind_layered(i, ...)`
   per layer, only on the layers that have the operand. What is known only
   per step (the `step_args` projection, windows from `md.max_seq_len`) is
   rebound once at the top of `forward`, and per-step scratch is allocated
   there too — never inside the layer loop. Call sites pass the data tensors
   and `layer=i`, and every call states its outputs:
   `out = op(in, out=buf, layer=i)`.
9. **Self-containment.** A target directory imports the catalog, the
   framework (`_target`, `_weights`, `catalog._op`), engine surface, and its
   own siblings (`.weights`, `.ops.<category>.<name>`) — nothing of another
   target's. An op only this target calls lives in its `ops/` with its GPU
   test at the mirrored path under
   `tests/unittest/_torch/modeling_v2/models/<family>/<target>/<category>/`.
   Duplication between targets is the architecture, not a smell.
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

A target directory holds **two files**, plus target-owned ops when it has any:

| product | |
|---|---|
| `modeling.py` | the core `<Family><Par>ModelingV2(DecoderModel)`: derived geometry, `self.w = MODEL_WEIGHTS.declare(...)`, `post_load_weights` building both targets, any unconditional first-step derivation, the dispatching `forward` · `PrefillTarget` and `DecodeTarget`, each with its bindings in `__init__`, its `step_args`, its `forward` · the runtime-value dump (below) · the `@register_auto_model(<synthetic name>)` shell `ModelingV2<Ckpt><Sm><Par>(DecoderModelForCausalLM[core, PretrainedConfig])`, whose `load_weights` calls `MODEL_WEIGHTS.load` and whose `post_load_weights` calls `super()` then `self.model.post_load_weights()` |
| `weights.py` | the weight table: a `ModelWeights` subclass with its `W` rows, `dims`, load-time transforms and `expected_unconsumed` with each reason; `MODEL_WEIGHTS = <Subclass>()` |
| `ops/<category>/<name>.py` | optional: `OpWrapper` subclasses only this target calls — the same executable contract as a catalog entry (`raw_call`, `reference`, `is_valid`, `compare`, `CELLS`, `ARCHS`) — each with its GPU test in the tests tree |

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

Each target's `forward` records the **metadata** of every entry call it
makes — shapes, dtypes, the constants passed — into a set, and the boot
gate's output carries it out of the run. With bound entries the call site
holds only the data tensors, so record two things: the call's own arguments
at the call, and the per-step bindings (the `step_args` projection, any
per-step `bind_layered`) when they are bound. Construction-time bindings are
read from the source.

This exists because of a class of value that nothing else surfaces: the
shapes the engine **varies at runtime**. Decode batch sizes, page sizes,
feature flags. They are written nowhere in the target, so reading the forward
does not reveal them, and a target once merged with 34 of the 35 batch sizes
its engine captured sitting outside the contract — every *constant* it passed
was inside.

Four rules make it safe and sufficient:

- **metadata only.** Never a tensor value: `.item()`/`.cpu()` are forbidden in
  a forward, and metadata is exactly what an entry's `CELLS` are written in
- **no allocation in the hot path.** A set of tuples, appended to; nothing
  that grows per token
- **capture is enough.** CUDA graph capture *executes* Python — that is why
  host-derived scalars get baked in — so the dump naturally covers every
  captured batch size. Replay executes none of it, so the steady state pays
  nothing
- **off unless asked for.** Record and print only when
  `TRTLLM_MODELING_V2_VALUE_DUMP=1` (read once, at import). Every gate and
  witness run sets it; a served engine must not print the dump at each exit

Comparing the dump against each entry's `CELLS` is a *reading* job, not an
automated one: a cell's `spec` is entry-specific and its `why` is prose. The
target's duty ends at exposing the values; assembler judges them, and smith
is dispatched only for what is genuinely uncovered.

## From-scratch flow

1. **Derive before writing.** Take the computation-step list from the
   workspace's `vocabulary.yaml` — it already carries this checkpoint's value
   on every axis, established in Phase B. Read the checkpoint's safetensors key
   list and the HF reference for the compute graph, and decide each step's
   mapping onto catalog entries (or a target-owned op, when only this target
   will call it), and which steps differ between prefill and decode. An
   unmappable step is a vocabulary gap: report it before creating files.
2. **Place the target and its routing.** Create
   `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/`,
   and add the target to that family's `routing.py`: the config-shape
   fingerprint, the parallel segment, the synthetic class name and its
   `TARGET_MODULES` entry. A family nothing routes yet also needs one row in
   `_router_index.py`. `references/routing-identity.md` is what may decide any
   of this — including every axis the derivation table says the exemplar's
   fingerprint does not pin.
3. **Write `weights.py` (the table) and `modeling.py` (core, both targets,
   shell)**, plus any target-owned op with its test, and draft the boot
   gate's keyword cases. Keywords stay provisional.
4. **Inner loop — edit → import dry-run → boot gate, until the gate passes.**
   The dry-run imports the target module and asserts the class registers under
   the synthetic name, and that `explain.py` routes this configuration to it.
   **Do not lint between edits.** The run lints once, at delivery, with the
   repo's `pre-commit`; during iteration spend the turns on the model, not on
   whitespace. **The first engine run doubles as keyword verification** —
   freeze the cases only after observing the continuations.
5. **Check the dump** against each entry's `CELLS`. Anything outside them:
   dispatch `Agent(smith)` with `kernel-verify`, resume on the receipt.
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
- the weights the feature turns on come *out* of the known-not-loaded key set
  (`expected_unconsumed`) and *into* the table as `W` rows
- a feature that changes one phase edits that target only; one that touches
  both edits both bodies — they share none
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
- **vocabulary audit**: each target's call list, each trtllm op mapped to its
  `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml` path or
  its `<target>/ops/<category>/<name>.py`. Scope it to each target's `forward`
  plus the private methods it reaches
- **dump review**: which values the engine produced at runtime, and where each
  sits against its entry's `CELLS`
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
