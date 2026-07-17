# PR/MR review workflow

A runnable harness built on `agent_flow.AgentLayer` that reviews an existing
GitHub PR / GitLab MR with **two stages, swapping the reviewer/coder roles
between the two models**. Each stage loops until both sides agree, then the
next stage begins.

```
Stage 1:  Claude Code (reviewer) ⇄ Codex (coder)        ✔ both agree
Stage 2:  Codex (reviewer)       ⇄ Claude Code (coder)  ✔ both agree
```

Two hard rules hold in both stages:

1. **Nothing is posted to the PR/MR.** The whole conversation stays local in
   `progress.yaml` + `discussion.md`. No `gh pr comment`/`review`, no
   `glab mr note`/`approve`.
2. **The coder may push back.** It addresses what it agrees with and declines
   the rest with a recorded rationale — it never makes a change it disagrees
   with. A genuine deadlock is resolved **in the coder's favor** (the coder
   stands firm, the disagreement is logged, and the stage advances).

The coder's edits are left **in the working tree** — the workflow never
commits or pushes. Inspect and commit them yourself when you're happy.

## Prerequisites

- An authenticated `gh` (GitHub) or `glab` (GitLab) CLI on `PATH`. The
  **sourcing agent** (not the orchestrator) runs it to check the PR/MR branch
  out and read its metadata — never to post. The orchestrator itself only
  shells out to local `git`.
- The Claude Code and Codex CLIs set up as for the other workflows.

## Run it

```bash
# Pass the PR/MR as a single positional — a number or URL, GitHub or GitLab.
# The number-or-URL form reviewing the checkout in the current dir:
pr-review 1234 --repo /path/to/checkout

# A full URL works too (GitHub PR or GitLab MR — the workflow doesn't care):
pr-review https://gitlab.com/org/repo/-/merge_requests/567 --repo /path/to/checkout
```

On a fresh run a **sourcing agent** works out whether the target is a GitHub PR
or a GitLab MR (from the URL, or the repo's remotes), checks its branch out into
`--repo` (running `gh pr checkout` / `glab mr checkout` itself), and reports the
base branch and metadata back to the orchestrator. The "diff under review" is
then the merge-base diff (computed by local `git`) from the branch's fork point
to the **working tree** — so it always reflects the coder's latest (uncommitted)
edits, not just the original PR/MR.

If a run is interrupted (Ctrl-C, crash), rerun against the same workspace — the
workflow auto-detects the checkpoint and continues from the last stage/round.
Pass `--clean` to wipe the checkpoint and managed files and start over.

### Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `target` (positional) | — | The GitHub PR or GitLab MR to review — a number or URL. Required. The sourcing agent detects the platform itself. |
| `--repo` | — (required) | Local git checkout to operate in; the coder's edits land in its working tree. |
| `--base` | from PR/MR | Override the base branch/ref the diff is taken against. |
| `--workspace` | `workspace/pr-review/<id>` | Shared review files + checkpoint. |
| `--num-rounds` | `20` | Max reviewer/coder rounds per stage before advancing. |
| `--reviewer-context-reset-interval` | `2` | Recycle the reviewer's session every N rounds (`0` disables). |
| `--coder-context-reset-interval` | `2` | Same for the coder. |
| `--clean` | off | Wipe the checkpoint and managed files and start fresh. |

## How a stage ends

Each round runs the **reviewer** (records `APPROVE` / `REQUEST_CHANGES`) then
the **coder** (records `REVISE` / `AGREE` / `STAND_FIRM`). After the coder's
turn:

- **reviewer `APPROVE` + coder `AGREE`** → converged → advance.
- **coder `STAND_FIRM`** → push-back is final (coder wins) → log the declined
  items → advance.
- otherwise → another round.
- `--num-rounds` reached → advance with the current state (the coder always has
  the final say).

## Shared workspace files

| File | Writers | Readers | Notes |
| --- | --- | --- | --- |
| `pr_context.md` | orchestrator (once, from the sourcing agent's report) | both agents | The PR/MR metadata + the exact diff command. Regenerated on a fresh run. |
| `progress.yaml` | both agents (append-only) | both agents (via `read_latest_progress`) | Audit log split into `stage1` / `stage2`. Reviewer entries carry `decision` (`APPROVE`/`REQUEST_CHANGES`); coder entries carry `decision` (`AGREE`/`REVISE`/`STAND_FIRM`). The orchestrator reads these to drive the loop. |
| `discussion.md` | both agents (overwrite each turn) | both agents | Rolling local conversation: open threads + status, agreed points, declined push-backs with rationale, next-turn pointers. |
| `.pr_review_state.json` | orchestrator | orchestrator (on resume) | Checkpoint of `stage` + round index. |

## MCP tools

The sourcing agent ends its turn by calling `report_pr_context` once (a
required-tool call) to hand back the checked-out PR/MR's base branch and
metadata. Each review agent ends its turn by calling its progress tool exactly
once, plus `update_discussion` (both are required-tool calls enforced by a stop
hook):

- `report_pr_context` — the sourcing agent reports the base/head branches,
  title, author, URL, and description after checking the PR/MR out
- `append_reviewer_progress` (decisions: `APPROVE` / `REQUEST_CHANGES`)
- `append_coder_progress` (decisions: `AGREE` / `REVISE` / `STAND_FIRM`)
- `read_latest_progress` — latest entries for the current stage (optional
  `agent` / `rounds` filters)
- `read_discussion` / `update_discussion` — the rolling local conversation

## Source location

- `workflow.py` — `PrReviewWorkflow` orchestrator (two-stage state machine).
- `cli.py` — the `pr-review` CLI entry point.
- `sourcing.py` — the `report_pr_context` MCP tool + `SourcingContext` the
  sourcing agent uses to hand the checked-out PR/MR's metadata back.
- `vcs.py` — local `git` helpers (diff-base resolution + the merge-base diff
  command). It no longer runs `gh` / `glab` — the sourcing agent does.
- `prompts/` — the `reviewer` / `coder` / `sourcing` system prompts (the first
  two reused across both stages) plus the `PromptBundle` extension contract.
- `progress.py` — `progress.yaml` schema and the `append_*_progress` /
  `read_latest_progress` MCP tools.
- `discussion.py` — `discussion.md` rolling scratchpad and the
  `update_discussion` / `read_discussion` MCP tools.
- `state.py` — checkpoint schema for resume (persisted to
  `.pr_review_state.json`).

Import it programmatically with
`from agent_flow.workflows.pr_review import PrReviewWorkflow`.
