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

### 2. The vocabulary mapping table — one row per forward step

Map every computation step in the forward onto `catalog/index.yaml`. Triage
each miss:

- **No trtllm-backed entry, but a candidate op exists upstream** — plan a
  Goal that onboards it. One op per Goal.
- **Entry exists, the needed surface is not certified** — plan a Goal that
  certifies that surface. One surface per Goal.
- **Missing `catalog/torch/` mirror, and it is pure glue** — no Goal needed;
  the Coder adds the thin wrapper plus its index entry inline.
- **A computation kernel with no upstream candidate at all** — the vocabulary
  ceiling. Say so explicitly; this is the one condition the plan cannot route
  around.

Never plan a computation as "compose it from torch mirrors". A missing
computation kernel is a vocabulary gap, not an implementation detail.

A new-capability Goal must name its implementation approach. "Add the missing
kernel" is layer drift: the Coder must not be picking the architecture during
implementation.

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

```markdown
## Implementation Steps

### Stage 1: accuracy convergence
Exit criterion: <one-line pointer to `acceptance-criteria.md`'s
                "Stage 1" subsection — the measured-vs-anchor bar>

- Goal 1.1: [catalog] onboard <op> — one op, closing on a receipt
- Goal 1.2: [catalog] certify <entry>'s <surface>
- Goal 1.3: [target] modeling.py + weights.py + MANIFEST + registration
- Goal 1.4: [target] smoke green on verified keywords
- Goal 1.5: [target] module parity against the verified pure-PyTorch rungs
- Goal 1.6: [target] accuracy debug until the bar
```

Goal IDs use `<Stage>.<Goal>`. Stages appear in execution order; Goal order
within a Stage is a suggested sequence the Reviewer may reorder at runtime.

**Tag every Goal `[catalog]` or `[target]`.** The two kinds close on
different evidence and are reviewed against different specs — a catalog Goal
closes on a fresh receipt from its own GPU test, a target Goal on the gate
tier it targets — and the Reviewer switches checklists on that tag. An
untagged Goal is a malformed plan.

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
