# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker's system prompt and turn prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._common import (
    CITATION_DISCIPLINE,
    DETACHED_JOBS,
    EVIDENCE_DISCIPLINE,
    GPU_POOL,
    MODELING_V2_LAYOUT,
    MODELING_V2_SWITCH,
    SLURM_ENVIRONMENT,
    SOURCE_BOUNDARY,
    tidy,
)
from .primer import primed_preamble
from .tags import TAG_PROMPTS

__all__ = ["WorkerTask", "build_worker_system_prompt", "worker_retry_prompt", "worker_turn_prompt"]

_WORKER_BASE = """\
You are a **worker** in a ModelingV2 onboarding run: one checkpoint is being
brought up as a gated ModelingV2 target in TensorRT-LLM. A planner split the
work into tasks in `plan.md`; many workers run tasks in parallel, each in its
own git worktree. **You own exactly one task.** Its id, its worktree and the
files you read and write are in your turn prompt.

## How you work

- **Do only your task.** Read its `### <task id>` section in `plan.md`, the
  whole plan for context, and `task.yaml` for the user's goal. Work that
  belongs to another task is out of bounds, even when you can see it is
  needed — say so in your result instead.
- **Edit code only in your worktree.** Never touch another task's worktree or
  the TensorRT-LLM checkout, unless your task is the final one and your
  worktree *is* that checkout.
- **Read your upstream tasks' results** before you start: they are the
  claims you build on, including what they could not finish.
- **Commit as you go**, by explicit file path (`git add <paths>`, never
  `git add -A` or a bare directory), with a short message and no AI
  attribution. Your session can be stopped at any time; uncommitted edits are
  lost when it is, committed ones are not. Never push, rebase or switch
  branches.
- **Keep notes** in your task directory (`notes.md`): what you tried, what
  worked, the commands you ran. If this task was restarted, read
  them and `git log` in your worktree first and continue from there.
- New files carry the NVIDIA SPDX copyright header; follow
  `CODING_GUIDELINES.md`.

## Ending your task

Your last action is to write your **result file** at the path in your turn
prompt. It starts with YAML front matter, then a markdown report:

```markdown
---
status: DONE        # or FAILED
summary: one line — what this task established
---
## What I did
## Evidence
<commands run, test files, numbers verbatim from tool output>
## What is not done / not certified
## For the planner
<anything that should change the plan: a missing entry, a wrong assumption,
a follow-up task>
```

- `DONE` — the task's definition of done holds, with evidence.
- `FAILED` — it does not, and you cannot make it hold within this task's
  scope. Say precisely why and what would unblock it. A `FAILED` result
  holds back every task that depends on yours until the planner revises
  the plan, so use it when the plan needs to change — not to report a caveat
  on finished work.

The orchestrator reads `status` and nothing else; a task that ends without a
valid result file is treated as `FAILED`.
"""


def build_worker_system_prompt(tags: Sequence[str], *, slurm: bool, gpu_pool: bool = False) -> str:
    """Compose the worker system prompt for a task with ``tags``.

    Args:
        tags: The task's tags, primary first. Each adds its block in order.
        slurm: Whether ``task.yaml`` carries a ``slurm-environment``.
        gpu_pool: Whether the run holds a GPU pool, which GPU work then goes
            through instead of the worker's own Slurm jobs.
    """
    blocks = [_WORKER_BASE, *(TAG_PROMPTS[tag] for tag in tags)]
    if gpu_pool:
        blocks.append(GPU_POOL)
    elif slurm:
        blocks.append(SLURM_ENVIRONMENT)
    blocks += [
        DETACHED_JOBS,
        MODELING_V2_LAYOUT,
        MODELING_V2_SWITCH,
        SOURCE_BOUNDARY,
        EVIDENCE_DISCIPLINE,
        CITATION_DISCIPLINE,
    ]
    return tidy("\n\n".join(blocks))


@dataclass(frozen=True)
class WorkerTask:
    """Everything a worker's turn prompt names.

    Attributes:
        task_id: The task's id.
        tags: Its tags, primary first.
        worktree: Where the worker edits code.
        is_final: Whether this is the run's final task (its worktree is the
            TensorRT-LLM checkout).
        side: Whether the task is ``side`` work, which the run does not merge.
        task_yaml: The user's task spec.
        plan: ``plan.md``.
        result: The result file to write.
        upstream_results: ``(task id, result file)`` for each dependency.
        notes_dir: The task's private directory.
        jobs: The task's ``jobs.json``.
        previous_attempts: Result files of earlier attempts at this task.
        test_commands: The Slurm command cache, when the run uses Slurm.
        gpu_run: The command prefix that runs a command in the run's GPU pool,
            when it has one.
        primed_repo: The checkout the session this worker was forked from
            read, when it was forked from the op primer.
        primed_commit: The commit that session read there.
        knowledge: The op primer's summary, for a worker not forked from it.
    """

    task_id: str
    tags: tuple[str, ...]
    worktree: Path
    is_final: bool
    task_yaml: Path
    plan: Path
    result: Path
    upstream_results: tuple[tuple[str, Path], ...]
    notes_dir: Path
    jobs: Path
    previous_attempts: tuple[Path, ...] = ()
    test_commands: Path | None = None
    side: bool = False
    gpu_run: str | None = None
    primed_repo: Path | None = None
    primed_commit: str | None = None
    knowledge: Path | None = None


def worker_turn_prompt(task: WorkerTask) -> str:
    """The single prompt a worker's task starts with."""
    lines = []
    if task.primed_commit is not None:
        lines += [primed_preamble(repo=task.primed_repo, commit=task.primed_commit).rstrip(), ""]
    lines += [
        f"Your task: `{task.task_id}` (tags: {', '.join(task.tags)}).",
        "",
        f"- Worktree — edit code here: `{task.worktree}`",
        f"- The task: the `### {task.task_id}` section of `{task.plan}`",
        f"- The user's goal: `{task.task_yaml}`",
        f"- Your notes directory: `{task.notes_dir}` (keep `notes.md` there)",
        f"- Your detached-job registry: `{task.jobs}`",
    ]
    if task.knowledge is not None:
        lines.append(f"- What every entry worker needs to know — read it first: `{task.knowledge}`")
    if task.test_commands is not None:
        lines.append(f"- Verified Slurm commands: `{task.test_commands}`")
    if task.gpu_run is not None:
        lines.append(
            f"- Run GPU work with: `{task.gpu_run} --task {task.task_id} --gpus <N> "
            f"--minutes <M> {task.worktree} <command...>`"
        )
    if task.is_final:
        lines += [
            "",
            "This is the run's **final task**: your worktree is the TensorRT-LLM "
            "checkout itself, and what you commit is what the run delivers.",
        ]
    if task.side:
        lines += [
            "",
            "This is a **side task**: the run does not merge your branch unless a "
            "human asks for it. The branch is kept and listed, with your result's "
            "`summary`, for a human to decide on, so make the `summary` say what you "
            "changed, the speedup you measured and at which shapes, and its effect on "
            "accuracy.",
        ]
    if task.upstream_results:
        lines += ["", "Upstream tasks (their branches were merged into your worktree):"]
        lines += [
            f"- `{tid}` (branch `node/{tid}`): `{path}`" for tid, path in task.upstream_results
        ]
    if task.previous_attempts:
        lines += [
            "",
            "Earlier attempts at this task failed; read their results first and "
            "do not repeat what they ruled out. Your worktree carries on from the last "
            "attempt's branch: check what it left before you build on it.",
        ]
        lines += [f"- `{path}`" for path in task.previous_attempts]
    lines += [
        "",
        f"When you are finished, write your result file to `{task.result}`.",
    ]
    return "\n".join(lines) + "\n"


def worker_retry_prompt(result: Path, problem: str) -> str:
    """The follow-up sent when a worker ended without a valid result file."""
    return (
        f"Your result file `{result}` is not valid: {problem}\n\n"
        "Write it now, starting with the `---` front matter block whose `status` "
        "is exactly `DONE` or `FAILED`, followed by your report. Do not start new "
        "work.\n"
    )
