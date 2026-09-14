"""Staircase-bringup guidance appended to the PlanDrafter prompt."""

from __future__ import annotations

from ._common import (
    ACCURACY_DEBUG_METHODOLOGY,
    CATALOG_ENTRY_SPEC,
    CLOSED_VOCABULARY_RULE,
    DOMAIN_PRIMING,
    GATE_LADDER,
    PROJECT_REQUIRED_MECHANISMS,
    REFERENCE_LADDER,
    SELF_CONTAINMENT_POLICY,
    TARGET_PRODUCT_SPEC,
)

_PLANNER_GUIDANCE = """\
## PlanDrafter guidance for staircase bring-up

`plan.md` must give the Coder enough specificity to execute and the Reviewer
enough specificity to detect drift. Two tables do most of that work here, and
neither can be deferred to implementation time.

### 1. The derivation table — one row per variation axis

Read the checkpoint's `config.json`, its safetensors key list, and the
architecture's HF reference modeling code, then decide **every** axis for
*this* checkpoint:

norm kind and placement; residual structure; rope variant, theta, and whether
it is NeoX-style; qk-norm presence; activation; attention structure (GQA head
counts, or MLA's latent geometry); attention bias; sliding window; attention
sinks; `tie_word_embeddings`; dtype and quantization format; vocabulary and
`lm_head` handling.

Two rules about exemplars — the existing targets under `models/*/targets/`:

- **Patterns, never values.** An exemplar shows the shape: file structure,
  the attention argument split, the manifest idiom, the fail-fast ladder.
  Every model-specific value is re-derived from *this* checkpoint. An
  exemplar's construction asserts mark exactly the decisions to re-derive —
  when your config disagrees with one, re-derive that part of the assembly;
  never delete the assert, never copy the exemplar's answer.
- **Know when there is no exemplar, and check before assuming there is
  none.** A `tp1` exemplar answers nothing about a `tepN` / `depN` topology,
  and it asserts `tp_size == 1` precisely so that shows up. Where no exemplar
  covers an axis, say so in the plan and derive it from the checkpoint, the
  contracts, and the `mapping` object the engine hands the target.

**The parallel segment is an axis the checkpoint cannot tell you about.**
Collectives are an artifact of the topology, not of the model, so they appear
nowhere in the HF reference code. A pre-audit driven only by that reading
comes back all-green for a multi-rank target and the gap surfaces mid-build
instead. Walk the segment separately: what the topology partitions, what must
be reduced or gathered to put it back together, whether `lm_head` is
vocab-parallel, and whether entries exist for those collectives.

### 2. The capability map — one row per computation, NOT per op

List the computations the forward performs and, for each, what a call would
have to do. **Do not name the op that will serve it.** Record candidates you
happened to notice as *non-binding leads*, explicitly marked as such.

```
capability   index-gathered sparse MQA with sinks
must do      gather the KV rows named by an int32 index list, score the
             query against them, apply one sink logit per query head
leads        two candidates noticed, NEITHER verified -- the build decides
```

**This table is mine-clearing, not authority.** Its job is to stop the Coder
starting from zero, not to decide anything. Which call serves a capability is
settled by driving candidates on GPU and comparing their domain against this
checkpoint's actual geometry — `topk` divisibility, head counts, head dims,
dtypes, what an empty row returns. None of that is readable; all of it is
measurable. A planner that has run nothing cannot know it, and a plan that
pretends otherwise converts a guess into a locked Goal.

**The one judgement the plan should make is shape, not identity.** When a
capability could be served by one fused call or by several composed ones,
say that the fused form is preferred and why: a call that takes its state as
explicit arguments is a better catalog entry than one needing a paged pool, a
block table, a scheduler counter and a raw pool pointer threaded in from
elsewhere — even when both compute the same thing. Leave *which* call to the
build.

**Triage of what the map turns up** — all of it advisory:

- A capability with several plausible leads: note them, let the build choose.
- A capability with none: say so. It may still be served by something the
  planner could not see, so this is a flag for the build to hunt, **not** a
  vocabulary ceiling. Only the build, after an exhaustive search with the
  evidence written down, can declare that.
- A glue step (tensor layout, movement, allocation, lookup): note that a
  `catalog/torch/` mirror covers it; no Goal needed.
- A computation that would otherwise be composed from torch mirrors: flag it.
  That composition is forbidden and its appearance in a plan is a defect.

### 3. Where the accuracy bar comes from

The accuracy anchor and its eval protocol come from a **written source** —
the staircase accuracy references, the upstream accuracy suite's entry for
this checkpoint at this quantization, the checkpoint's own model card, or
stock trtllm measured on this machine. Record which one, and record the
protocol it was measured under: a checkpoint gated with a chat template and
thousands of output tokens reasons before it answers, and scoring it as
few-shot completion measures nothing.

**An anchor is never invented.** If no source yields one, say so in the plan
and flag it — that is a blocker for the human, not a number to estimate.

### 4. Risks worth a register entry

Contracts that can fail silently while tests pass: a reference that shares a
helper with the implementation; a hard path bypassed by configuration; a
tolerance loose enough to swallow a real error; a capability whose output is
distribution-preserving by construction (speculative decoding) and therefore
invisible to both gates.

Do not defer required work with *future work*, *follow-up*, *out of scope*,
or *after support lands*. Required catalog, modeling, weights, and test work
appears as concrete Goals in the current plan.
"""

_STAGE_GOAL_PLAN_SCHEMA = """\
## Stage/Goal plan schema (mandatory for `plan.md`)

`plan.md`'s `## Implementation Steps` section uses a two-level **Stage/Goal**
hierarchy. A **Stage** is a milestone version of the target where a defined
accuracy bar is met; a **Goal** is a concrete implementation or focused
debugging task whose closure contributes to that bar.

**A staircase bring-up is normally one Stage.** Unlike a general model
bring-up there is no simpler backend to converge on first and swap away from
later — the catalog entries *are* the production kernels, so there is no
Stage 2 "swap to performance backends". Vocabulary work and assembly work are
**Goals inside Stage 1**, not Stages of their own: they do not meet an
accuracy bar, they are what makes meeting it possible.

Plan a second Stage only for a **capability variant that changes the forward
path** — speculative decoding being the standing example, which also changes
the weight-loading path and needs a third signal (`acceptance_length` against
a reference) because rejection sampling makes a miscomputed draft path
produce correct text. Its exit is the same accuracy bar plus that signal.

### `plan.md` — `## Implementation Steps` format

**A Goal is a module, not an op.** Name the capability the module owns and
the evidence that closes it; do not name the ops it will use. Which call
serves a capability is settled by measurement during the build, not by
reading during planning — see *Capability mapping* above.

```markdown
## Implementation Steps

### Stage 1: accuracy convergence
Exit criterion: <one-line pointer to `acceptance-criteria.md`'s
                "Stage 1" subsection — the measured-vs-anchor bar>

- Goal 1.1: reference ladder — native HF baseline on fixed prompts, then a
            pure-PyTorch implementation per module aligned to it, then the
            accuracy anchor and its protocol frozen
- Goal 1.2: attention — <the capability in one line: geometry, positional
            scheme, sparsity, cache shape>
- Goal 1.3: MoE — <routing rule, expert count/topk, quantization>
- Goal 1.4: <further modules the derivation table found>
- Goal 1.5: assembly — modeling.py, weights.py, MANIFEST, registration
- Goal 1.6: smoke green on keywords verified against the real model
- Goal 1.7: accuracy debug until the bar
```

Goal IDs use `<Stage>.<Goal>`. Stages appear in execution order; Goal order
within a Stage is a suggested sequence the Reviewer may reorder at runtime.

**Goal 1.1 is always the reference ladder, and it always comes first.**
Every later Goal's exit criterion cites it, so a plan that puts it anywhere
else has ordered its own dependencies wrong.

### How a module Goal closes

Two conditions, and the second only becomes checkable once Goal 1.1 has
landed:

1. **Vocabulary closed for that module** — every catalog entry the module
   calls carries a fresh receipt on this arch, and every value the module
   passes (including the shapes the engine varies at runtime) sits inside the
   entry contract's certified column.
2. **Module parity** — the module's output matches its verified pure-PyTorch
   implementation from Goal 1.1, on the pinned prompts, at a stated tolerance.

(1) proves the parts work. (2) proves they were assembled correctly, and it
is the condition that catches the failure (1) cannot see: **every entry
certified, every receipt fresh, and the module still wrong because the pieces
were wired in the wrong order or fed the wrong argument.** A catalog
architecture is unusually exposed to that, which is why (2) is required and
not a nice-to-have.

State both in the Goal's acceptance items. Where a module genuinely cannot be
driven in isolation — a cross-layer mechanism with no standalone boundary —
say so in the plan and name what replaces (2) for it, rather than dropping
the condition silently.

**Do not tag Goals `[catalog]` or `[target]`.** A module Goal contains both
kinds of work by design: it onboards entries *and* wires them. The Reviewer
switches checklists on what the Coder reports doing **this iteration**, which
the Coder states in its summary.

### `acceptance-criteria.md` — Stage-partitioned checklist

Partition into one `## Stage N — <label>` subsection per Stage, each a flat
`- [ ] ...` checklist, with labels matching `plan.md`. QA parses these headers
to scope its verification, so the exact format matters: do not nest them under
other headers and do not change the `## Stage N — ` prefix.

```markdown
## Stage 1 — accuracy convergence
- [ ] every new/extended catalog entry carries a passing receipt on this arch
      that post-dates every file in its entry
- [ ] the forward's tensor-creating calls all map to catalog/index.yaml
- [ ] every argument value and every engine-captured shape falls inside the
      contract's certified column
- [ ] smoke passes on frozen, pre-verified keywords
- [ ] <benchmark> measured >= <anchor> - <tol> under TRTLLM_STAIRCASE=require
```

The Coder, Reviewer, and QA never edit these two files. You own both.
"""

_STAGE_GOAL_REPLAN_LOCK_MATRIX = """\
## Replan lock matrix (Stage/Goal mode)

The replan phase lets you revise `plan.md` and `acceptance-criteria.md` based
on QA findings — but execution history constrains what you may change. Before
any replan-turn edit, read `status.md` (generic `Read` tool), identify each
Stage's status from the `## Stages & Goals` table, then enforce this:

- **Stage `— CLOSED` — locked.** No edit to title, exit criterion, or Goal
  list; no edit to its acceptance subsection.
- **Stage `— IN_PROGRESS` — partially locked.** `[Done]` / `[Failed]` Goals
  locked; new Goals may be appended; title and exit criterion locked.
  Existing acceptance items locked; new ones may be appended.
- **Stage `— PENDING` — fully editable.**
- **New forward-looking Stage** — append at the tail, preserving temporal
  order.
- **New QA-REJECT gap-fix Stage** — insert immediately after the failing
  CLOSED Stage. All subsequent Stages, all `— PENDING` at that point, shift
  down and are renumbered in `plan.md`, `acceptance-criteria.md`, and
  `status.md`. CLOSED Stages keep their numbers.

After QA **APPROVE** on Stage N: update `status.md` (overwrite with the
generic `Write` tool, preserving everything below the table) — flip Stage N to
`— CLOSED`, mark the next `PENDING` Stage `— IN_PROGRESS`, promote its first
`[Undo]` Goal to `[Doing] (iterations=0)`. This is the only bookkeeping path
that mutates the table during replan.

After QA **REJECT** on Stage N: **do not demote, reopen, or mutate Stage N or
any other CLOSED Stage.** The failing items stay `- [ ]` in their locked
subsection — the failure is part of the immutable record. Instead insert a
gap-fix Stage immediately after Stage N with a **single Goal** scoped to the
QA-flagged gap, carrying its own acceptance subsection with the new pass/fail
line, marked `— IN_PROGRESS` with its Goal `[Doing] (iterations=0)`.

**The accuracy anchor and tolerance are not yours to move.** A gap-fix Stage
for a missed accuracy gate re-scopes the *work*, not the bar: it names what
will be investigated and fixed. Relaxing the anchor because the target could
not reach it inverts the release criterion. If you believe the anchor itself
is wrong — the protocol does not match, or the source was misread — say that
explicitly and cite the conflict; do not quietly write a lower number.

Renumbering downstream PENDING Stages is **mandatory and mechanical**;
revising their content is optional and judgement-driven, and every content
edit needs a justification in your `summary` citing the QA finding that made
the prior plan stale. A silent relaxation or a silently deleted Goal is a
hard no.
"""

_STAGE_GOAL_REPLAN_DECISION_MAPPING = """\
## Replan decision mapping in Stage/Goal mode

Override the generic decision table with this when invoked in `replan` mode.
Read `status.md` first to identify current Stage states.

- `DONE` — every Stage is `— CLOSED` in `status.md`, and either every Stage
  subsection has been QA-APPROVE'd, or a QA-REJECT'd Stage was followed by a
  gap-fix Stage that was itself APPROVE'd. The latest `weighted_score` must be
  at or above ``min_score``. **Hard rule:** below ``min_score`` you may not
  return `DONE` — choose `POLISHING` directly.
- `POLISHING` — QA APPROVE'd the just-closed Stage and at least one Stage
  remains `PENDING`. Bump `status.md` and apply low-risk touch-ups. The Coder
  runs again immediately; no PlanReviewer.
- `DRAFT_READY` — QA REJECT'd the just-closed Stage, or you need a substantive
  rewrite. After a QA REJECT the **only** allowed remediation is inserting a
  gap-fix Stage immediately after the failing CLOSED Stage and renumbering
  downstream PENDING Stages; never demote a CLOSED Stage or edit its locked
  subsection.

Never return `HUMAN_APPROVED` from a top-level replan turn.

This whole protocol is active only when the workflow runs with
`--replan-on-qa`.
"""

SYSTEM_PROMPT_EXTENSION = "\n".join(
    [
        DOMAIN_PRIMING,
        CLOSED_VOCABULARY_RULE,
        PROJECT_REQUIRED_MECHANISMS,
        SELF_CONTAINMENT_POLICY,
        CATALOG_ENTRY_SPEC,
        TARGET_PRODUCT_SPEC,
        GATE_LADDER,
        REFERENCE_LADDER,
        ACCURACY_DEBUG_METHODOLOGY,
        _PLANNER_GUIDANCE,
    ]
)

# Stage/Goal control flow is only wired when the workflow runs with
# --replan-on-qa; ``build_staircase_prompts`` appends this on top of
# SYSTEM_PROMPT_EXTENSION in that mode only.
STAGE_GOAL_EXTENSION = "\n".join(
    [
        _STAGE_GOAL_PLAN_SCHEMA,
        _STAGE_GOAL_REPLAN_LOCK_MATRIX,
        _STAGE_GOAL_REPLAN_DECISION_MAPPING,
    ]
)

__all__ = ["STAGE_GOAL_EXTENSION", "SYSTEM_PROMPT_EXTENSION"]
