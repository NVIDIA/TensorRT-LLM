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
| `--clean` | off | Wipe the workspace checkpoint and workflow-managed files (`.agent_team_state.json`, `plan.md`, `acceptance-criteria.md`, `progress.yaml`, `status.md`) and start fresh. Without this flag the workflow auto-resumes from the checkpoint when one is present. |
| `--plan` | unset | Pre-supply `plan.md` (text or file path). Combined with `--acceptance-criteria` it skips the plan phase entirely; alone it still runs the plan phase so the PlanDrafter generates the missing `acceptance-criteria.md`. Ignored when resuming from a checkpoint — pass `--clean` to start fresh. |
| `--acceptance-criteria` | unset | Pre-supply `acceptance-criteria.md` (text or file path). Mirrors `--plan`: combined with `--plan` skips the plan phase; alone runs the plan phase to generate `plan.md`. |
| `--feedback` | unset | Append a human-feedback entry to `progress.yaml`'s `human_feedback` list before the next iteration runs. Value is either the literal feedback text or a path to a file containing it. Intended for resume scenarios: stop the workflow (Ctrl-C), re-run with `--feedback "..."`, and the build-phase agents (coder, reviewer, qa) pick it up via `read_human_feedback` on their next turn. Each invocation appends — prior entries are preserved. |

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
| `.agent_team_state.json` | orchestrator | orchestrator (on resume) | Checkpoint of `stage` + iteration indices. Re-running the workflow against the workspace auto-resumes from it; pass `--clean` to wipe it. |

## MCP tools

Each agent ends its turn by calling a per-agent progress tool exactly
once. The tool writes a structured YAML entry, which the orchestrator
reads directly to drive control flow:

- `append_plan_drafter_progress` (decisions: `DRAFT_READY` / `POLISHING` / `HUMAN_APPROVED`)
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
- `agent_flow/workflows/agent_team/state.py` — checkpoint schema for resume (stage +
  iteration indices, persisted to `.agent_team_state.json`).

Import the workflow programmatically with
``from agent_flow.workflows.agent_team import AgentTeamWorkflow``.
