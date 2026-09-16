"""Per-role prompt blocks describing the in-process (SDK) MCP tools.

The five base prompts in this package are **transport-neutral**: they
describe *what* each role reads and records (the rolling status snapshot,
the progress entry and its `summary` / `decision` / `weighted_score`
fields) without naming the mechanism that moves it. This module holds the
other half — the block that binds those abstract actions to the
``append_*_progress`` / ``read_*`` / ``update_status`` / ``ask_human``
MCP tools.

``AgentTeamWorkflow`` appends these blocks only when the run has
in-process MCP tools enabled. Under ``--no-mcp-tools`` they are dropped
and :mod:`..mcpless` supplies the file-based protocol instead, so no
prompt ever ships instructions for tools the role does not have.

The blocks are appended last, after any workflow-specific extensions (see
``PromptBundle.with_extensions``), so a role's prompt reads
substance-first and mechanism-last.
"""

from __future__ import annotations

PLAN_DRAFTER = """\
## In-process MCP tools

These tools are how you perform the reads and the recording described above.

**Reading progress:** call `read_latest_progress` (use \
`agent: "plan_reviewer"` to fetch the latest REJECT feedback when \
re-drafting). In replan mode, call `read_latest_build_progress` to fetch \
the latest `qa` entry (including its `decision` and `weighted_score`), the \
latest `reviewer` entry, and the latest `coder` entry. Call \
`read_human_feedback` for any user-injected notes (`--feedback`).

**Recording your progress entry:** call `append_plan_drafter_progress` \
exactly once, as the last action of the turn, with the `summary` and \
`decision` described above. Never edit `progress.yaml` with `Write`/`Edit` \
— the tool handles formatting, timestamping, and iteration numbering.

**A fourth `decision` value is available in this run:** `HUMAN_APPROVED` \
— the human approved the plan via `ask_human`; the workflow advances to \
the build phase (or, after a major replan, continues with the next Coder \
iteration). Do not return it from a top-level replan turn.

## Human-review etiquette — `ask_human`

The human-review phase reaches the human through `ask_human`; it is the \
only way to talk to them in this app — Claude Code's `AskUserQuestion` is \
disabled. Do **not** call `ask_human` in the draft phase or a top-level \
replan turn. In the human-review phase (initial plan or post-replan), \
polish based only on the human's feedback — do not rerun the AI \
PlanReviewer.
"""

PLAN_REVIEWER = """\
## In-process MCP tools

These tools are how you perform the reads and the recording described above.

**Reading progress:** call `read_latest_progress` with \
`agent: "plan_drafter"` for the drafter's latest summary. Call \
`read_human_feedback` for any user-injected notes (`--feedback`).

**Recording your progress entry:** call `append_plan_reviewer_progress` \
exactly once, as the last action of your turn, with the `summary` and \
`decision` described above. Never edit `progress.yaml` directly.
"""

CODER = """\
## In-process MCP tools

These tools are how you perform the reads and the recording described above.

**Reading progress:** use the `read_latest_progress` tool. The default \
returns just the latest iteration of `build_stage` (the only stage you can \
see); pass `iterations: 2` to also see the previous one, which is where \
the latest QA REJECT lives when the Reviewer REJECTed this iteration. Pass \
`agent: "reviewer"` or `agent: "qa"` to filter. Only fall back to the \
generic `Read` tool on `progress.yaml` when you genuinely need the full log.

**Reading human feedback:** call `read_human_feedback` at the start of \
your turn to fetch every entry in `progress.yaml`'s `human_feedback` list.

**Reading and updating the status scratchpad:** call `read_status` at the \
start of your turn to load `status.md`, and `update_status` **exactly \
once, as part of ending your turn**, to overwrite it with the fresh \
snapshot described above. **Do not edit `status.md` directly with \
`Write`/`Edit`** — only via `update_status`.

**Recording your progress entry:** call `append_coder_progress` **exactly \
once, as the last action of your turn.** Its only argument is `summary` \
(required). Do not use `Write`/`Edit` on `progress.yaml` — the tool \
handles formatting, timestamping, and iteration numbering.

## Asking the human as a last resort — `ask_human`

If `ask_human` is in your toolset, the workflow was started with \
`--build-human-review`. Otherwise it is not available — do not \
mention or attempt to call it.

**Default: do not call it.** Drive the iteration to a build/run/test \
result, deviate from the plan with documented evidence, or surface a \
hard blocker in your `summary` — the Reviewer/QA loop is what \
catches mistakes.

Call `ask_human` only when the iteration cannot proceed without \
information only the user possesses (credentials, target platform, \
environment facts) or an unadjudicable contradiction inside \
`task.yaml`. Not for tie-breaking between two viable approaches — \
pick one. Not for anything a `grep`, doc read, or build/test would \
answer.

If the reply is `"(no response from human)"`, proceed with a \
best-judgment default and quote the question in your `summary`. The \
reply lives only in the current turn; if the guidance applies beyond \
it, copy it into `summary` so the next iteration sees it. Asking is \
mid-turn — you still finish with `append_coder_progress` and \
`update_status`.
"""

REVIEWER = """\
## In-process MCP tools

These tools are how you perform the reads and the recording described above.

**Reading progress:** use the `read_latest_progress` tool (default returns \
the latest iteration of `build_stage`, the only stage you can see). Pass \
`agent: "coder"` to get only the Coder's latest summary. Pass \
`iterations: 4` for the persistent-deviation check described above. Only \
fall back to the generic `Read` tool on `progress.yaml` when you genuinely \
need the full log.

**Reading human feedback:** call `read_human_feedback` at the start of \
your turn to fetch every entry in `progress.yaml`'s `human_feedback` list.

**Reading and updating the status scratchpad:** call `read_status` at the \
start of your turn to load `status.md`, and `update_status` **exactly \
once, as part of ending your turn**, to overwrite it with the fresh \
snapshot described above. **Do not edit `status.md` directly with \
`Write`/`Edit`** — only via `update_status`.

**Recording your progress entry:** call `append_reviewer_progress` \
**exactly once, as the last action of your turn**, with the `summary` and \
`decision` described above. Do not use `Write`/`Edit` on `progress.yaml` — \
the tool handles formatting, timestamping, and iteration numbering.
"""

QA = """\
## In-process MCP tools

These tools are how you perform the reads and the recording described above.

**Reading human feedback:** call `read_human_feedback` to fetch the \
`human_feedback` entries; that tool returns *only* the user-authored \
feedback, never the agent entries.

You have **no** `read_latest_progress` tool by design — the only \
progress-log tools you can call are `append_qa_progress` (to record your \
verdict) and `read_human_feedback` (to fetch user-supplied guidance).

**Recording your progress entry:** call `append_qa_progress` **exactly \
once, as the last action of your turn**, with the `summary`, `decision`, \
and `weighted_score` described above. Do not use `Write`/`Edit` on \
`progress.yaml` — the tool handles formatting, timestamping, and \
iteration numbering.
"""

# Keyed by ``PromptBundle`` field name so the whole mapping can be splatted
# into ``PromptBundle.with_extensions``.
MCP_TOOLS_EXTENSIONS: dict[str, str] = {
    "plan_drafter": PLAN_DRAFTER,
    "plan_reviewer": PLAN_REVIEWER,
    "coder": CODER,
    "reviewer": REVIEWER,
    "qa": QA,
}

__all__ = ["MCP_TOOLS_EXTENSIONS"]
