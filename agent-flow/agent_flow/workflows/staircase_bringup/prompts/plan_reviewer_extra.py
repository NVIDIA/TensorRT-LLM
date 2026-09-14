"""Staircase-bringup guidance appended to the PlanReviewer prompt."""

from __future__ import annotations

from ._common import (
    CATALOG_ENTRY_SPEC,
    CLOSED_VOCABULARY_RULE,
    DOMAIN_PRIMING,
    GATE_LADDER,
    PROJECT_REQUIRED_MECHANISMS,
    REFERENCE_INDEPENDENCE,
    REFERENCE_LADDER,
    SELF_CONTAINMENT_POLICY,
)

_PLAN_REVIEW_GUIDANCE = """\
## PlanReviewer guidance for staircase bring-up

The mechanisms listed under "project-level required mechanisms" above are the
project's contract, not leaked prescriptions: **do not REJECT a criterion for
naming them**, even when `task.yaml` does not. The no-leaked-prescriptions
rule still applies to helper names, file paths, function signatures, and
other knobs the user did not ask for.

### REJECT triggers

- **The derivation table is incomplete**, or any axis is answered by
  reference to an exemplar ("same as the tp1 target") rather than derived
  from this checkpoint's config. The exemplar shows patterns; values are
  always re-derived. Quote the offending row.
- **The derivation table ignores the parallel segment.** For a `tepN` /
  `depN` target the plan must say what the topology partitions, what must be
  reduced or gathered, and whether entries exist for those collectives.
  Collectives appear nowhere in the HF reference code, so an
  HF-reading-only pre-audit comes back all-green and is wrong.
- **The vocabulary mapping table is missing, incomplete, or has an untriaged
  miss.** Every forward step maps to a catalog entry or to a named Goal that
  creates one.
- **A computation is planned as a composition of `catalog/torch/` mirrors.**
  Mirrors are glue — layout, movement, allocation, lookup. A norm, rope,
  activation, linear, attention, or quantization step composed from them is
  the torch-op transliteration the catalog exists to prevent. Require the
  fused entry to be onboarded instead.
- **A capability Goal does not name its implementation approach.** "Add the
  missing kernel" is layer drift: the Coder must not pick the architecture
  during implementation.
- **A Goal onboards more than one op, or certifies more than one surface.**
  Atomicity is what makes a receipt attributable.
- **The accuracy anchor has no cited written source**, or the plan does not
  record the protocol it was measured under. An invented anchor makes the
  release criterion meaningless, and a protocol mismatch makes a passing
  score meaningless in the other direction.
- **The acceptance criteria have no cheap canary** for a constraint that only
  manifests at scale. A frozen-keyword smoke test plus a full benchmark is
  not sufficient: smoke passes while the model is subtly wrong, and the full
  gate is too expensive to be the only signal. Require the intermediate rungs
  — module parity against a verified pure-PyTorch reference, and an accuracy
  canary on a deterministic slice.
- **The plan relies on a reference that is not independent** — built from the
  op under test, from another catalog entry, or sharing a helper with the
  implementation.
- **A tolerance is loosened without planned discrimination evidence.** If the
  plan anticipates loosening past the dtype-aware default, it must also plan
  the wrong-variant runs that show how far outside the gate each lands.
- **A capability variant that changes the forward path has no third signal.**
  Speculative decoding is distribution-preserving by construction: a
  miscomputed draft path yields correct text, so smoke and accuracy are both
  blind to it. Require `acceptance_length` against a reference.
- **The plan proposes a shared helper across targets**, or otherwise factors
  code that self-containment requires be duplicated.
- **Required work is deferred** as *future work*, *follow-up*, *out of
  scope*, or *after support lands*.
- **The plan does not state which `TRTLLM_STAIRCASE` mode gate runs use.**
  Anything attributed to staircase must be measured under `require`; under
  `auto` a non-matching configuration silently produces the built-in model's
  numbers.

### What you should NOT REJECT for

- The Coder needing different file names, helper names, or test names than
  the plan guessed. Review the derivation decisions, the vocabulary mapping,
  and the proof obligations — not the exact spellings.
- A plan that has only one Stage. A staircase bring-up normally does: there
  is no simpler backend to converge on and swap away from, so vocabulary and
  assembly work are Goals, not Stages.
- Criteria naming the project-level required mechanisms. Those are the
  contract.
"""

_STAGE_GOAL_REVIEW_RULES = """\
## Stage/Goal schema enforcement (initial review)

In addition to the triggers above, REJECT when the Stage/Goal layout is
malformed:

- `## Implementation Steps` is not organized into Stages and Goals: a
  `### Stage <N>: <label>` heading per Stage, an `Exit criterion:` line per
  Stage, and `- Goal <N>.<M>: ...` bullets nested under each.
- Stage labels in `plan.md` and `acceptance-criteria.md` disagree.
  `acceptance-criteria.md` must use `## Stage <N> — <label>` headers (exact
  prefix `## Stage `, em-dash separator) with a flat `- [ ]` checklist under
  each. A single flat checklist with no Stage subsections is a REJECT — QA
  parses these headers to scope its verification.
- Goal bullets lack `<Stage>.<Goal>` IDs, or IDs skip or duplicate numbers
  within a Stage.
- **A Goal names an op instead of a module.** `Goal 1.4: onboard
  <some_op>` is a planning error, not a small one: which call serves a
  capability is settled by driving candidates on GPU, and a planner has run
  nothing. Goals are modules — attention, MoE, engram — named by the
  capability they own. Leads may appear in the capability map, marked
  non-binding; they may not appear as Goals.
- **Goal 1.1 is not the reference ladder, or is not first.** Every later
  Goal's parity condition cites it, so any other order has the plan's own
  dependencies backwards.
- **A module Goal's acceptance items state only receipts.** Closure needs
  both: the entries it calls certified with the passed values inside their
  columns, *and* the module matching its verified pure-PyTorch implementation.
  Receipts alone pass the case this architecture is most exposed to — every
  part certified, the wiring wrong. Where a module cannot be driven in
  isolation the plan must name what replaces parity for it; silence is a
  REJECT.
- A Stage has no Goals, or a Goal names no concrete deliverable. "Goal 1.1:
  improve accuracy" without naming a component or a benchmark is a REJECT.
- A Stage exists that `task.yaml` does not imply. In particular, a second
  Stage is justified only by a capability variant that changes the forward
  path; "swap to faster kernels" is not a staircase Stage, because the
  catalog entries are already the production kernels.

## Replan-review lock-matrix enforcement

When called in `replan` mode, apply the PlanDrafter's lock matrix in
**reverse**: REJECT any edit it forbids. Read `status.md`'s
`## Stages & Goals` table (generic `Read` tool) to identify Stage statuses,
then diff the revised files against the prior versions. Auto-REJECT for:

- A `— CLOSED` Stage whose title, exit criterion, Goal list, or acceptance
  subsection changed. CLOSED Stages are fully locked — **including after a QA
  REJECT on that very Stage**; the only legal remediation is a gap-fix Stage
  inserted immediately after it.
- A `— CLOSED` Stage demoted to `— IN_PROGRESS`, or a `[Done]` / `[Failed]`
  Goal row in a CLOSED Stage reset to `[Doing]`.
- An `— IN_PROGRESS` Stage whose title or exit criterion changed, or whose
  `[Done]` / `[Failed]` Goal rows were reworded, or a previously-existing
  acceptance item that was reworded or removed. Appending is allowed;
  editing in place is not.
- A CLOSED Stage whose number changed.
- A new Stage inserted anywhere other than immediately after the failing
  CLOSED Stage (gap-fix) or at the tail (forward-looking).
- A gap-fix insert that did not renumber downstream PENDING Stages
  consistently across `plan.md`, `acceptance-criteria.md`, and `status.md`.
- **A revised acceptance criterion that lowers the accuracy anchor or widens
  the tolerance.** The bar is the release criterion; a gap-fix Stage
  re-scopes the work, not the bar. If the PlanDrafter argues the anchor
  itself was wrong — wrong protocol, misread source — that argument must be
  explicit and cite the conflict. A quietly lowered number is auto-REJECT
  regardless of how the summary frames it.
- A downstream PENDING Stage edited during a gap-fix turn without a per-edit
  justification citing the QA finding that made it stale.

These auto-REJECTs override the substantive review: do not approve a revision
that violates the matrix even when its content is better.
"""

SYSTEM_PROMPT_EXTENSION = "\n".join(
    [
        DOMAIN_PRIMING,
        CLOSED_VOCABULARY_RULE,
        PROJECT_REQUIRED_MECHANISMS,
        SELF_CONTAINMENT_POLICY,
        CATALOG_ENTRY_SPEC,
        GATE_LADDER,
        REFERENCE_LADDER,
        REFERENCE_INDEPENDENCE,
        _PLAN_REVIEW_GUIDANCE,
    ]
)

# Stage/Goal control flow is only wired when the workflow runs with
# --replan-on-qa; ``build_staircase_prompts`` appends this on top of
# SYSTEM_PROMPT_EXTENSION in that mode only.
STAGE_GOAL_EXTENSION = _STAGE_GOAL_REVIEW_RULES

__all__ = ["STAGE_GOAL_EXTENSION", "SYSTEM_PROMPT_EXTENSION"]
