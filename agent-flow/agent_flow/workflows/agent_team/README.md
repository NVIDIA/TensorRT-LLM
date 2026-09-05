# Agent-team workflow

A runnable multi-agent harness built on top of `agent_flow.AgentLayer`. A
plan is drafted, reviewed by a second model (and, optionally, signed off
by the human), and then implemented in a coder ↔ reviewer ↔ qa loop
until QA accepts.

```
PlanDrafter ⇄ PlanReviewer [⇄ Human]  →  Coder ⇄ Reviewer  →  QA  ✔
```

![Agent-team workflow](docs/agent-team-workflow.svg)

## Run it

```bash
agent-team \
    --task path/to/task.yaml \
    --workspace workspace/agent-team
```

`--task` is required and must be a path to a YAML file. The file is
copied verbatim into `<workspace>/task.yaml` and read by every agent.
The base `agent_team` workflow imposes no schema on the YAML — any keys
and values you put in are passed through to the agents. Wrappers
(e.g. `modeling_bringup`) may layer their own schema on top. The
workspace directory is created on demand and holds all shared state for
the run.

If a run is interrupted (Ctrl-C, crash, reboot), just rerun against the
same workspace — the workflow auto-detects the checkpoint and continues
from the last checkpointed stage. Pass `--clean` to wipe the checkpoint
(and the other workflow-managed files) and start over instead:

```bash
# Resume from the checkpoint (default when one exists in the workspace).
agent-team \
    --workspace workspace/agent-team

# Wipe the checkpoint and start fresh.
agent-team --clean \
    --task path/to/task.yaml \
    --workspace workspace/agent-team
```

### Useful flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--num-iterations` | `100` | Cap on coder/reviewer/qa iterations. |
| `--coder-context-reset-interval` | `2` | Recycle the coder's persistent session every N iterations (`0` disables). |
| `--reviewer-context-reset-interval` | `2` | Same for the build-phase reviewer. |
| `--min-score` | `8.0` | Floor on QA `weighted_score` (0–10). APPROVE below this floor is downgraded to a loop-back; `0` disables the gate. |
| `--plan-human-review` | off | Enable the human-review stage of the plan phase — after PlanReviewer APPROVE, the PlanDrafter asks the human for sign-off via `ask_human` before the build phase begins. Off by default; PlanReviewer APPROVE flows straight into the build phase. |
| `--build-human-review` | off | Enable the Coder's `ask_human` escape hatch during the build phase. The Coder may pause mid-iteration to ask the human a question via stdin and integrate the reply. Off by default; only enable when the task involves environment facts or user-only information the Coder cannot deduce. |
| `--replan-on-qa` | off | After each QA turn, re-invoke the PlanDrafter in **replan mode** to revise `plan.md` / `acceptance-criteria.md` based on the latest coder/reviewer/qa findings. The PlanDrafter (not QA) decides task completion via a `DONE` decision; `POLISHING` loops straight back to the Coder, `DRAFT_READY` runs the revision through the PlanReviewer first (and the human if `--plan-human-review` is set). Off by default — the build phase terminates on QA APPROVE as it always has. |
| `--clean` | off | Wipe the workspace checkpoint and workflow-managed files (`.agent_team_state.json`, `plan.md`, `acceptance-criteria.md`, `progress.yaml`, `status.md`) and start fresh. Without this flag the workflow auto-resumes from the checkpoint when one is present. |
| `--plan` | unset | Pre-supply `plan.md` (text or file path). Combined with `--acceptance-criteria` it skips the plan phase entirely; alone it still runs the plan phase so the PlanDrafter generates the missing `acceptance-criteria.md`. Ignored when resuming from a checkpoint — pass `--clean` to start fresh. |
| `--acceptance-criteria` | unset | Pre-supply `acceptance-criteria.md` (text or file path). Mirrors `--plan`: combined with `--plan` skips the plan phase; alone runs the plan phase to generate `plan.md`. |
| `--feedback` | unset | Append a human-feedback entry to `progress.yaml`'s `human_feedback` list before the next iteration runs. Value is either the literal feedback text or a path to a file containing it. Intended for resume scenarios: stop the workflow (Ctrl-C), re-run with `--feedback "..."`, and the build-phase agents (coder, reviewer, qa) pick it up via `read_human_feedback` on their next turn. Each invocation appends — prior entries are preserved. |
| `--concurrent-review` | off | Overlap the Reviewer with the next Coder iteration instead of running them strictly in sequence. See [Concurrent review](#concurrent-review-opt-in-via---concurrent-review). |
| `--review-snapshot-repo` | unset | The git checkout the Coder edits, frozen once per iteration for `--concurrent-review`. Overrides the `review_snapshot_repo` / `trtllm_repo_path` task keys. Ignored without `--concurrent-review`. |

## Workflow

### Plan phase

- **PlanDrafter** writes both `plan.md` — the implementation
  *starting hypothesis* plus a *risk register* (approach, design
  choices, considered-and-rejected directions, pitfalls, targeted
  guidance) — and `acceptance-criteria.md` — a flat markdown
  checklist of **outcome-bound** pass/fail bars derived from
  `task.yaml` (no leaked plan prescriptions).
- **PlanReviewer** (a different backend/model) reviews both files as a
  unit and emits a single `APPROVE` or `REJECT`. On `REJECT` the
  PlanDrafter addresses the feedback and re-drafts.
- Once the PlanReviewer `APPROVE`s, the workflow enters the build
  phase. From that point on `acceptance-criteria.md` is treated as
  immutable: no agent mutates it during the build phase.
- The human-review stage is **opt-in**: pass `--plan-human-review` (or
  `plan_human_review_enabled=True` in code) to enable it. With the flag
  set, once the PlanReviewer `APPROVE`s, the PlanDrafter switches to
  the **human-review phase**: it calls the `ask_human` MCP tool to ask
  the human for approval of the plan and criteria together. If the
  human requests changes, the PlanDrafter polishes either file and
  re-asks. The AI PlanReviewer is **not** re-invoked here — the human
  is the final arbiter once they take over. Without the flag,
  PlanReviewer `APPROVE` goes straight to the build phase.

### Build phase

- Each iteration runs the **Coder**, then the **Reviewer**. The reviewer
  emits a bool `decision` (`APPROVE` / `REJECT`). On `REJECT` the
  workflow loops back to the coder without running QA.
- When the reviewer `APPROVE`s, **QA** runs full build/test/evaluation
  against `task.yaml` and emits both a `decision` (`APPROVE` / `REJECT`) and
  a `weighted_score`. `APPROVE` ends the run, provided `weighted_score`
  clears `--min-score`; an APPROVE below the floor or a plain REJECT
  loops back to the coder.
- QA is **stateless** by design: each iteration starts with a fresh
  session so its verdict is not biased by prior history. The coder and
  reviewer reuse persistent sessions, recycled every N iterations to
  bound context. The coder's session is additionally reset right after
  iteration 1 so the refinement phase always starts from a clean slate.

### Replan-on-QA (opt-in via `--replan-on-qa`)

When the flag is set the build phase changes shape: after every QA
turn the PlanDrafter is re-invoked in **replan mode** to revise
`plan.md` and `acceptance-criteria.md` based on the coder/reviewer/qa
findings. The PlanDrafter — not QA — decides when the workflow
terminates.

```
Coder → Reviewer → QA → PlanDrafter (replan)
                          ├─ DONE          → exit loop
                          ├─ POLISHING     → Coder              (fast path)
                          ├─ DRAFT_READY   → PlanReviewer → ... (slow path)
                          └─ HUMAN_APPROVED → Coder             (after human, if enabled)
```

Reviewer `REJECT` still loops straight back to the Coder without
triggering replan — replan only runs after a QA turn. `--min-score`
is no longer a termination gate when the flag is set; it is passed to
the PlanDrafter as a prompt hint and enforced by the orchestrator as a
hard floor on `DONE` (a `DONE` returned while `weighted_score` is
below `--min-score` is downgraded to `POLISHING` with a warning).

The PlanDrafter's persistent session is reused for replan turns, so
the plan-phase context (the original rationale, considered-and-rejected
directions, risk register) remains visible when it is rewriting after
the build phase. In replan mode the PlanDrafter also gets a
`read_latest_build_progress` MCP tool (a `build_stage`-scoped twin of
`read_latest_progress`) and the same `read_human_feedback` the
build-phase agents use, so it can ground its revision on the latest
coder/reviewer/qa entries and any `--feedback` notes.

With `--plan-human-review` on, a `DRAFT_READY` revision that the
PlanReviewer APPROVEs runs through the same human-review sub-stage as
the initial plan, before the next coder iteration starts.

## Concurrent review (opt-in via `--concurrent-review`)

Sequentially the build phase is `coder(i) -> reviewer(i) -> verdict ->
coder(i+1)`. A reviewer that builds and runs the code can take hours, and
the Coder — plus any machines it has reserved — is idle for all of it.
With `--concurrent-review` the two overlap, at a pipeline depth of 1:

1. At the end of `coder(i)` the orchestrator **snapshots** the Coder's
   checkout: it records `HEAD` and creates the ref
   `refs/agent-flow/review/iter-<i>` at that commit. The repo comes from
   `--review-snapshot-repo`, else the task key `review_snapshot_repo`,
   else `trtllm_repo_path`.
2. `reviewer(i)` runs **on a worker thread** against that frozen commit.
   Its prompt says the checkout is frozen at `<hash>`, that the Coder is
   working elsewhere, and that it must not expect or request changes.
3. `coder(i+1)` starts **immediately**, with an addendum: a review is in
   progress against `<hash>`, so it must not touch the snapshotted
   checkout. It works in a git worktree branched from the snapshot (under
   the task's `worktrees_dir`, claiming a slot in `worktree_reservations`
   if configured, otherwise `<repo>/../worktrees`), runs the code from
   there, and folds its worktree commits back into the main checkout once
   the verdict arrives.
4. The verdict is delivered **into the running Coder turn** as a
   `progress.yaml` notice with `source: reviewer`
   (`REVIEW VERDICT for iteration i: APPROVE|REJECT, see <path>`). The
   Coder polls it with the `read_review_notices` MCP tool, which is only
   registered in this mode. `read_human_feedback` filters these out — they
   are a downstream agent's voice, not the user's.

### Verdict handling

- **REJECT** — `coder(i+1)` keeps going; it already has the rejection as
  input. When it ends, snapshot again, launch `reviewer(i+1)`, start
  `coder(i+2)`.
- **APPROVE** — `coder(i+1)` is **never killed**; it finishes its turn.
  Then the pipeline drains: iteration `i+1` gets a *fresh* sequential
  review on its own snapshot, and only if that pass also approves does
  the work reach QA. So commits made while a review was running are never
  carried into QA on the strength of a verdict that never saw them. The
  cost is one extra review pass whenever an approval lands mid-pipeline;
  the alternative — shipping unreviewed commits to QA — trades a real
  correctness gap for that saving, which is not worth it.

`--num-iterations` still caps total coder turns, and the last allowed
iteration is always reviewed sequentially (there is no next coder turn to
overlap with).

### When it falls back to sequential

For a given iteration the workflow logs a warning and reviews
sequentially — correct, just not overlapped — when:

- the working tree is **dirty**. The reviewer can only read committed
  state, so freezing a dirty tree would have it review something the
  Coder never produced. The workflow refuses to snapshot rather than
  review a lie; commit (or stash) and the next iteration overlaps again.
- no snapshot repo is configured, the path does not exist, or it is not a
  git checkout.

### Shared-file safety

Two agent turns run at once, so every file both roles write is split:

| File | Change in concurrent mode |
| --- | --- |
| `status.md` | Both roles' `update_status` *overwrites* this file. The Reviewer is repointed at a per-iteration `status-review-<i>.md`; the Coder keeps `status.md`. The verdict notice carries the path to the review report. |
| `progress.yaml` | Append-only but read-modify-written whole, so two appends could lose one another. All appends now serialize on a process-wide lock. The Reviewer also gets its own `ProgressContext`, so the two roles stamp different iteration numbers. |
| `plan.md`, `acceptance-criteria.md`, `task.yaml` | Read-only for both build-phase roles; unchanged. |

### Resume, and switching a live run over

The checkpoint records which review was in flight
(`review_in_flight_iteration`) and the exact commit it was reading
(`review_snapshot_repo` / `_commit` / `_ref`). On restart that review is
**re-run from the snapshot ref**, so it sees the same tree the
interrupted run did rather than whatever the checkout has drifted to. If
the ref is gone, the workflow logs a warning and lets the pipeline review
the next snapshot instead. These fields are purely additive, so an older
checkpoint loads unchanged and the schema version stays at 5.

To switch a **running sequential run** to concurrent mode:

1. Wait for the end of a coder step (the checkpoint is written with
   `stage: reviewer`), then stop the process with Ctrl-C. Stopping
   mid-reviewer also works — that reviewer turn is simply re-run.
2. Commit everything in the coder's checkout. A dirty tree is the one
   thing that silently keeps you on the sequential path.
3. Re-run the exact same command **without `--clean`**, adding
   `--concurrent-review` (and `--review-snapshot-repo <path>` if the task
   file does not name the repo).

Resume-without-`--clean` is the default: `AgentTeamWorkflow.__init__`
sets `self.resume = self.state_path.is_file()` and only wipes the managed
files when `clean` is passed, so the restart picks up at the checkpointed
stage and iteration. Nothing about the workspace files changes when the
flag is turned on, so the switch is safe at any iteration boundary.

## Shared workspace files

All agents communicate through files in the workspace — user prompts
only reference paths, never embed content.

| File | Writers | Readers | Notes |
| --- | --- | --- | --- |
| `task.yaml` | orchestrator (once) | all agents | The original user intent (a YAML file copied verbatim from `--task`). The ultimate ground truth — QA defers to it on conflict with `acceptance-criteria.md`. |
| `plan.md` | PlanDrafter | PlanReviewer, Coder, Reviewer | The PlanDrafter's *starting hypothesis* + *risk register* (architecture choices, considered-and-rejected directions, pitfalls). Strong default for the Coder, **not** a contract — Coder may deviate when implementation evidence forces it, as long as criteria still hold. Overwritten on each re-draft. Not read by QA. |
| `acceptance-criteria.md` | PlanDrafter | PlanReviewer, Coder, Reviewer, QA | Flat markdown checklist (`- [ ] ...`) of **outcome-bound** pass/fail bars distilled from `task.yaml` only — no leaked plan prescriptions. The build's APPROVE gate (Reviewer + QA). Co-approved with `plan.md` (PlanReviewer + human) and immutable post-approval — QA re-verifies every box at runtime. |
| `progress.yaml` | every agent (append-only); orchestrator on `--feedback` | every agent (via `read_latest_progress`); build-phase agents also via `read_human_feedback` | Structured audit log split into top-level `plan_stage`, `build_stage`, and `human_feedback` lists. Stage entries carry `decision` / `weighted_score` so the orchestrator never has to regex-scrape agent prose. `human_feedback` entries hold user-authored notes injected via `--feedback`. |
| `status.md` | Coder, Reviewer (overwrite each turn) | Coder, Reviewer | Short rolling scratchpad: current state, execution path, what was tried, what worked, what didn't, pointers for the next step. PlanDrafter / PlanReviewer / QA do not see it. |
| `status-review-<i>.md` | Reviewer (`--concurrent-review` only) | Coder (via the verdict notice) | The Reviewer's per-iteration rolling snapshot. Exists only in concurrent-review mode, where the Reviewer must not share `status.md` with a Coder that is writing it at the same time. |
| `.agent_team_state.json` | orchestrator | orchestrator (on resume) | Checkpoint of `stage` + iteration indices. Re-running the workflow against the workspace auto-resumes from it; pass `--clean` to wipe it. |

## MCP tools

Each agent ends its turn by calling a per-agent progress tool exactly
once. The tool writes a structured YAML entry, which the orchestrator
reads directly to drive control flow:

- `append_plan_drafter_progress` (decisions: `DRAFT_READY` / `POLISHING` / `HUMAN_APPROVED` / `DONE`; `DONE` is only emitted in `--replan-on-qa` mode)
- `append_plan_reviewer_progress` (decisions: `APPROVE` / `REJECT`)
- `append_coder_progress`
- `append_reviewer_progress` (decisions: `APPROVE` / `REJECT`)
- `append_qa_progress` (decisions: `APPROVE` / `REJECT`, plus `weighted_score`)

The Coder and Reviewer additionally have `update_status` and
`read_status` tools. `update_status` is a required-tool call so the stop
hook enforces that `status.md` is refreshed alongside `progress.yaml`.

The Coder, Reviewer, and QA each have a `read_human_feedback` tool that
returns the entries in `progress.yaml`'s `human_feedback` list. QA is
intentionally isolated from the agent log, but this tool is the
exception: human feedback is the user's own voice and is treated on par
with `task.yaml`.

The PlanDrafter has `human_input_enabled=True` unconditionally so it
can call the built-in `ask_human` tool during the human-review phase.
The same agent (and persistent session) is used in the draft phase,
but the prompt forbids `ask_human` there. The plan-stage human
checkpoint itself is gated by `--plan-human-review`; without the flag
the tool is registered but never reached.

The **Coder** also gets `ask_human` — but only when
`--build-human-review` is passed. With the flag set the Coder may
pause mid-iteration to ask the human a question (via stdin) and
integrate the reply before finishing its turn. Without the flag the
tool is not registered and the Coder must drive the iteration to a
build/run/test result, deviate from the plan with documented
evidence, or surface a hard blocker in its `summary` instead.

### Injecting mid-run feedback

To correct course while the workflow is running:

1. Stop the workflow (Ctrl-C). The current iteration's checkpoint is
   already on disk, so nothing is lost.
2. Re-run the workflow against the same workspace with
   `--feedback "your note to the agents"` (or `--feedback path/to/file`).
   The orchestrator appends one entry to `human_feedback` stamped with
   the next iteration and the active stage, then resumes.
3. On the next coder/reviewer/qa turn, the agents call
   `read_human_feedback` and address the new entry along with any
   prior, still-unresolved entries.

Each `--feedback` invocation **appends** — old feedback is preserved.
There is no auto-clear: entries remain visible to every subsequent
build iteration, so the agents can ground every iteration in the full
history of human guidance.

If the workflow has already completed (QA ACCEPTed or the budget was
exhausted) and you re-run against the same workspace, the orchestrator
exits with a hint instead of looping silently. Pass `--clean` to start
over, or pass `--feedback "..."` to re-engage the build phase to
address the new guidance.

## Source location

The workflow now ships as a library package — this directory only holds
documentation and the workflow diagram. The source lives in
`agent_flow/workflows/agent_team/`:

- `agent_flow/workflows/agent_team/workflow.py` — `AgentTeamWorkflow` orchestrator and
  workflow orchestration.
- `agent_flow/workflows/agent_team/cli.py` — the `agent-team` CLI entry point.
- `agent_flow/workflows/agent_team/prompts/` — per-agent system prompts (`plan_drafter`,
  `plan_reviewer`, `coder`, `reviewer`, `qa`) plus the `PromptBundle`
  extension contract.
- `agent_flow/workflows/agent_team/progress.py` — `progress.yaml` schema and the
  per-agent `append_*_progress` / `read_latest_progress` MCP tools.
- `agent_flow/workflows/agent_team/status.py` — `status.md` rolling scratchpad and the
  `update_status` / `read_status` MCP tools.
- `agent_flow/workflows/agent_team/concurrent_review.py` — git snapshotting and the
  prompt addenda for `--concurrent-review`.
- `agent_flow/workflows/agent_team/state.py` — checkpoint schema for resume (stage +
  iteration indices, persisted to `.agent_team_state.json`).

Import the workflow programmatically with
``from agent_flow.workflows.agent_team import AgentTeamWorkflow``.
