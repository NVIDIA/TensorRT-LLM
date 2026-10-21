---
name: trtllm-modeling-v2-core
description: Load whenever a target's modeling.py or weights.py is being written or edited — the invariants any forward must hold, the products a target consists of, and the from-scratch assembly flow. assembler loads it for core assembly and for every feature increment; tuner loads it before editing a forward.
---

# trtllm-modeling-v2-core

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
| `runtime-interface.md` | always, before writing — resolution, the shell / core / two-target split, calling the entries, weight lifecycle, the weight table, forward duties, fail-fast ladder, serving-time costs |
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
   metadata it consumes, and the target's own `_check_step_contract` calls it
   on the first forward when `_router_index.step_contract_enabled()` says so
   (`TRTLLM_MODELING_V2_VALIDATE=1`). A renamed custom op
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
4. **Weights are declared once, in the weight table.** `weights.py` defines
   the `W` role class and a table class whose `WEIGHTS: tuple[W, ...]` holds
   one `W` per weight role (shape, source, dtype, transform, layers), with a
   `dims(core)` that is the only place a derived width or padded size is
   computed; `MODEL_WEIGHTS = <Table>()` is its module-level instance.
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
   core's `forward` hands the step to `self._select_target(md)`:
   `DecodeTarget` when `md.num_contexts == 0`, `PrefillTarget` otherwise — a
   mixed batch is prefill's. Both are built in the core's
   `post_load_weights`, hold the core by reference, and are plain classes
   with their own `__init__`, `step_args`, `_check_step_contract` and
   `forward` — no shared body, base or mixin between them. `DecodeTarget`
   states `num_contexts=0, num_ctx_tokens=0` and never reads either back. Each forward is flat: one code path,
   branching only on metadata and per-capture constants. A draft step reads
   the phase per draft step, never cached.
8. **Every argument at the call site, every call.** Entries are stateless:
   a target imports each entry's module-level instance and passes the
   weights, the per-step batch state and the config scalars explicitly at
   every call, so the call shows exactly what the kernel receives. What is
   fixed once weights exist (per-layer weight views, `.t()` GEMM operands,
   derived constants) the target takes in its `__init__`; what is known only
   per step (the `step_args` projection, windows from `md.max_seq_len`) is
   computed once at the top of `forward`, and per-step scratch is allocated
   there too — never inside the layer loop. Every call states its outputs:
   `attn_out = thop_attention(q=qkv, output=attn_out, ..., **step)`.
9. **Self-containment.** A target directory imports the catalog,
   `_router_index` (for `step_contract_enabled`), engine surface, and its
   own siblings (`.weights`, `.ops.<category>.<name>`) — nothing of another
   target's. An op only this target calls lives in its `ops/` with its GPU
   test at the mirrored path under
   `tests/unittest/_torch/modeling_v2/models/<family>/<target>/<category>/`.
   Duplication between targets is the architecture, not a smell.

**Size fixed buffers from engine limits.** A constant that bounds rows,
tokens, batch, slots or pages, or sizes a buffer, is best derived from an
engine argument at construction (`model_config.max_num_tokens`,
`max_batch_size`, `max_seq_len`). Where it cannot be, say on its line what it
answers to (`# capacity: max_num_tokens`, `# capacity: fixed: <why>`), so the
reviewer can read it against the serving envelope. A `_TOPK_MAX_ROWS = 8192`
offsets buffer once killed every engine above `max_num_tokens` 8192 in
warm-up while every kernel it fed was certified to 16384; the envelope boot
at delivery now runs at the declared limits and would see it.

## Products

A target directory holds **two files**, plus target-owned ops when it has any:

| product | |
|---|---|
| `modeling.py` | the core `<Family><Par>ModelingV2(DecoderModel)`: derived geometry, `self.w = MODEL_WEIGHTS.declare(...)`, `post_load_weights` building both targets, any unconditional first-step derivation, the dispatching `forward` · `PrefillTarget` and `DecodeTarget`, each taking the weight views and constants it reads in `__init__`, with its `step_args`, `_check_step_contract` and `forward` · the causal-LM shell `ModelingV2<Ckpt><Sm><Par>(DecoderModelForCausalLM[core, PretrainedConfig])` that `route` returns, whose `load_weights` calls `MODEL_WEIGHTS.load` and whose `post_load_weights` calls `super()` then `self.model.post_load_weights()` |
| `weights.py` | the weight table: the `W` role class, a table class with its `WEIGHTS` rows, `dims`, `declare` / `manifest` / `load`, load-time transforms and `expected_unconsumed` with each reason; `MODEL_WEIGHTS = <Table>()` |
| `ops/<category>/<name>.py` | optional: one `OpWrapper` subclass per file, only this target calls it, with its module-level instance — the same executable contract as a catalog entry (`__call__`, `reference`, `is_valid`, `CELLS`, `ARCHS`, `note`) — each with its GPU test in the tests tree |

Nothing else belongs to a target. The checkpoint is read exactly as published
(no target-owned `config.json`); the boot gate's runner is the orchestrator's
(you write its keyword cases); the topology comes from the routing decision
and the gate's flags; gate records go in the run journal.

**Nothing registers.** The checkpoint's `architectures[0]` selects the
family's routing module, whose `route` imports and returns the target's class
on the branch that matches (`from .<target>.modeling import ModelingV2<...>`).
The loader carries that class on `ModelConfig.modeling_v2_target`, so every
other runtime branch keyed on the architecture string still sees the
published name. The class name carries the same triple as the path, and
`test_modeling_v2_claims.py` asserts they agree. Read
`references/routing-identity.md` before adding a target or proposing a new
criterion.

## Placing a new target

1. **Place the target and its routing.** Create
   `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/`,
   and add the target to that family's `routing.py`: a `route` branch on the
   config-shape fingerprint, SM, parallel topology and speculative config that
   imports and returns the class, and a `NoTarget` message that states those
   facts. A family nothing routes yet also needs one row in
   `_router_index.py`. `references/routing-identity.md` is what may decide any
   of this — including every axis the derivation table says the exemplar's
   fingerprint does not pin.
2. **Write `weights.py` (the table) and `modeling.py` (core, both targets,
   shell)**, plus any target-owned op with its test, and draft the boot
   gate's keyword cases. Keywords stay provisional.
3. **Before each boot gate, an import dry-run**: import the target module and
   assert that the family's `route` returns this class for this
   configuration. **The first engine run doubles as keyword verification** —
   freeze the cases only after observing the continuations.

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

## Tests after a change to shared code

Run the tests you added or changed and those of the files you touch, and
nothing more. A change outside the target's own directories (the catalog,
the executor, another model's code) is tested against the other models' unit
tests once, in the run's cross-model regression review (`review` with
`regression: true`), which batches every such change that landed; do not run
`tests affected` or `tests compare` in your own turn. Your report names the shared paths you
changed, so the review knows what it covers.

## Report

- target path; checkpoint path and hash summary; the GPU used. Routing
  recognizes a checkpoint by its shape and cannot tell a fine-tune from the
  original, so the record of which bytes you measured is what pins the claim
- **derivation table**: each axis and what this model required. Flag every axis
  where this target differs from the exemplar you read
- **vocabulary audit**: each target's call list, each trtllm op mapped to its
  `catalog/<category>/<name>.py` entry or its
  `<target>/ops/<category>/<name>.py`. Scope it to each target's `forward`
  plus the private methods it reaches
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
- files created
