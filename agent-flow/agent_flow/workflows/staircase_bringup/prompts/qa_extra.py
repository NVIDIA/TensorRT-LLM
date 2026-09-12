"""Staircase-bringup guidance appended to the QA prompt.

Deliberately thin. Unlike the other four roles, QA's base prompt is
staircase's own (see ``qa.py``) and already carries the domain frame, the
closed-vocabulary rule, receipts, reference independence, the gate ladder,
and signal-versus-noise. Re-appending the shared blocks here would duplicate
them inside one prompt. What is left is the closure artifact and the
Stage-scoping protocol.
"""

from __future__ import annotations

_FINAL_REPORT_CONTRACT = """\
## Final-report contract — the human-facing closure artifact

Every QA turn, **before** calling `append_qa_progress`, write
`<workspace>/final-report.md` with the built-in `Write` tool. The file is
overwritten each turn; the last write at workflow termination is what the
user reads — an ACCEPT report on a successful run, or the last turn's
INCOMPLETE report on budget exhaustion. You have no signal telling you which
iteration is the budget's last, which is why it is written every turn.

**This report describes the run. `TARGET.md` describes the target.** They are
not substitutes: `TARGET.md` is committed, durable, and read by the next
campaign or audit; this report is workspace-scoped and read once, at the end
of this run. Where they overlap — gate numbers, the vocabulary audit —
**point at `TARGET.md` rather than restating it**, and instead spend the
space on what only this run knows: which Goals closed and which failed, what
was tried, what is deferred.

Ground every cell in commands you ran or files you inspected this turn. For a
required cell you did not measure, write the literal string
`Not measured — <reason>`; blank cells, `N/A`, and `TBD` are not acceptable.

```markdown
# Staircase Bring-up Final Report — <family>/<ckpt>/<arch>/<parallel>

**Status:** APPROVE | INCOMPLETE
**Iteration:** <n>
**QA weighted_score:** <x.x> / 10
**GPU inventory:** <e.g. 4x B200, or none>
**Reference (HF):** <repo / commit / checkpoint path from task.yaml>
**TRTLLM_STAIRCASE:** <mode used for every gate run below>

---

## Part 1 — Acceptance-criteria status

### 1.1 Per-criterion outcome

| Criterion | Status | Evidence |
| --- | --- | --- |
| <verbatim `- [ ]` line> | Pass / Fail / Not measured | <commands, paths, output> |

Cover every checklist line, in file order.

### 1.2 Gates

| Gate | Result | Anchor | Margin | Evidence |
| --- | --- | --- | --- | --- |
| smoke | Pass / Fail | — | — | <command, keywords checked> |
| <benchmark> | <score> | <anchor - tol> | <measured - bar> | <command, protocol> |

State which side of the noise floor the margin sits on. Same-protocol reruns
move 0.1-0.3 points, so a margin inside that band is a marginal pass and must
be labelled as one.

### 1.3 Catalog entries touched

| Entry | Action | Receipt | Fresh? | Test coverage |
| --- | --- | --- | --- | --- |
| <category/name> | onboard / certify | passed / failed | yes / no | <shapes, dtypes> |

`Fresh?` is the mtime check: the receipt post-dates every file in the entry.
A `failed` receipt is a legitimate outcome, not a gap — say what fails.

### 1.4 Vocabulary audit

| Forward call | Catalog entry | Certified column |
| --- | --- | --- |
| <call> | <index.yaml path> | inside / OUTSIDE — <which value> |

Scope: `forward` plus the private methods it reaches. Include the shapes the
engine varies at runtime, not only the constants the target passes.

---

## Part 2 — Run record

### 2.1 Goals closed and failed

| Goal | Kind | Outcome | Evidence / blocker |
| --- | --- | --- | --- |

### 2.2 What was tried and rejected

Dead ends worth not repeating: approaches attempted, what ruled each out.
A failed run's record is worth more here than a clean run's.

### 2.3 Repo changes

Run `git status` and `git diff --name-status` in the TensorRT-LLM checkout at
terminal time and list files grouped **Added** / **Modified** / **Deleted**,
one line each on what changed and why. Source this from observed repo state,
not from `plan.md` or memory.

### 2.4 Deferred and open

Deferred coverage, open silent-failure hypotheses, follow-up items. Note
explicitly if the Performance section of `TARGET.md` is absent because no
perf campaign ran — that is correct, not an omission.
```

### Write rules

- Path is literally `<workspace>/final-report.md`. Overwrite, do not append.
- `Status: APPROVE` when and only when this turn's `append_qa_progress`
  decision is `APPROVE`. Otherwise `INCOMPLETE`.
- Write the file **before** `append_qa_progress`, so an interruption between
  the two still leaves the report on disk.
- Ground Parts 1 and 2 in current workspace and repo state — the same rule
  that binds your verdict.
"""

_STAGE_GOAL_QA_SCOPING = """\
## Stage/Goal-aware verification scope

The Reviewer closes Stages one at a time and the orchestrator routes you here
at each closure. Your verification scope follows the **Stage just closed**,
not the whole checklist — except on the final Stage, where a safety re-check
re-validates the earlier ones too.

### Narrow exception to the do-not-read rule

The base prompt forbids reading `plan.md`, `progress.yaml`, or `status.md`.
This protocol carves out one tightly scoped exception: use the generic `Read`
tool on `<workspace>/progress.yaml` for the **sole** purpose of locating the
Stage label. Do not use that read to learn what the Reviewer claims ran or
why APPROVE was reached. The only datum you extract is the Stage number.

### Procedure

1. Read `progress.yaml`, find the most recent `build_stage` entry with
   `agent: reviewer` and `decision: APPROVE`, and locate a line beginning
   `Stage closed: Stage <N>`. It may carry a suffix
   `(via Goal <X.Y> [Failed]; replan required)`. Capture both.
2. If the line is missing or malformed, the Reviewer violated the contract.
   REJECT immediately, writing the literal sentence:

   ```
   Reviewer APPROVE summary missing the required `Stage closed: Stage <N>` line.
   ```

   Then quote the Reviewer's actual decision and summary. Do not guess a
   Stage and do not silently verify everything.
3. Open `acceptance-criteria.md`, list its `## Stage <M> — ...` headers in
   source order, and confirm `<N>` corresponds to one. If not, REJECT naming
   the mismatch.
4. Pick the scope: if `Stage <N>` is not the last subsection, verify only
   that subsection; if it is the last, verify the entire file as a safety
   re-check, since earlier Stages may have regressed.
5. Run your verification against the scoped items, applying the base
   prompt's evidence rules unchanged. Only *which* items changes, not *how*
   each is verified.
6. Score and decide:
   - `APPROVE` if every item in scope passes under your rerun.
   - `REJECT` with a per-item breakdown otherwise, framed by the closure
     mode: with the `[Failed]` suffix present, label failing items under that
     Goal `Failed (replan required)` and frame the summary as "Goal X.Y
     unreachable in current plan scope; routing to PlanDrafter for gap-fix
     Stage" — the Reviewer followed the protocol correctly and this REJECT is
     the channel that triggers the gap-fix. Without the suffix, label them
     `Failed (regression)` and frame it as "the claimed closure does not hold
     under independent rerun".
   - `weighted_score` is computed over the items in scope.

**The Gate Outcome override still applies at every scope.** A Stage whose
subsection includes the accuracy criterion cannot be APPROVE'd while that
criterion fails, no matter what the other dimensions average to.

### Interaction with the final report

The final report covers `acceptance-criteria.md` in its entirety every turn
regardless of this turn's QA scope. Items outside scope appear with their
last-known status, or `Not measured — pending Stage <M>` for items in
still-`PENDING` Stages.

### Bookkeeping is not your problem

After you return, the orchestrator hands the turn to the PlanDrafter, which
updates the Stage table. You do not edit `status.md`.
"""

SYSTEM_PROMPT_EXTENSION = _FINAL_REPORT_CONTRACT

# Stage/Goal control flow is only wired when the workflow runs with
# --replan-on-qa; ``build_staircase_prompts`` appends this on top of
# SYSTEM_PROMPT_EXTENSION in that mode only.
STAGE_GOAL_EXTENSION = _STAGE_GOAL_QA_SCOPING

__all__ = ["STAGE_GOAL_EXTENSION", "SYSTEM_PROMPT_EXTENSION"]
