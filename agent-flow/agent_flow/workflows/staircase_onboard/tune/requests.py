# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render one tuner request: everything a candidate turn needs, in one file.

The request is the turn's user prompt; ``prompts/tuner.py`` is its system
prompt. Under ``--agent-transport external`` both travel in ``dispatch/``; a
driver that delivers the request without the system prompt runs a different
agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class CampaignContext:
    repo: Path
    package: Path
    campaign_dir: Path  # e.g. <perf workdir>/05-iterations
    perf_root: Path  # the perf workdir (spool/, scripts/, 08-kernel-sources/)
    target_dir: str  # repo-relative
    target_tests: str  # repo-relative
    model: str
    dev_spools: Sequence[Path]
    env_script: Path
    embargo: Sequence[str]
    workload: str
    standings_table: str  # markdown
    notes: str = ""
    sister_engines: Sequence[str] = field(default_factory=list)


def render_request(
    ctx: CampaignContext,
    *,
    slug: str,
    worktree: Path,
    base: str,
    hypothesis: str,
    deliverables: str,
    job_prefix: str,
    modeling_edits: bool = True,
) -> Path:
    out_dir = ctx.campaign_dir / "candidates" / slug
    (out_dir / "microbench").mkdir(parents=True, exist_ok=True)
    R = ctx.repo
    spools = ", ".join(f"`{s}`" for s in ctx.dev_spools)
    embargo = "\n".join(f"- `{e}`" for e in ctx.embargo)
    sisters = ""
    if ctx.sister_engines:
        sisters = (
            "\nOther engines that implement this model (strategies may inform you; write your own kernels): "
            + ", ".join(f"`{s}`" for s in ctx.sister_engines)
            + "\n"
        )
    scope = (
        "You may edit the target's forward (`modeling.py`), `weights.py` and `ops/**` in your worktree."
        if modeling_edits
        else "**Do not edit the target's forward in this turn**: deliver the entry (or entries), their GPU tests "
        "and the micro-bench; integration into the forward is a later candidate."
    )
    text = f"""# Tuner request: {slug}

Issued {time.strftime('%Y-%m-%d %H:%M %Z')} by the tune stage.

## Paths

| name | value |
|---|---|
| repo (main checkout, read-only for you) | `{R}` |
| **your worktree** (edit here only) | `{worktree}`, branch `tune/{slug}` off `{base[:10]}` |
| target dir (inside the worktree) | `{ctx.target_dir}` |
| target tests (inside the worktree) | `{ctx.target_tests}` |
| package (skills, microbench, spoolctl) | `PKG = {ctx.package}` |
| skills to read first | `PKG/skills/staircase-perf-tuning/SKILL.md` + `references/`, `PKG/skills/staircase-modeling-core/SKILL.md` (+ `references/graph-and-sync.md`), `PKG/skills/staircase-kernel-onboard/SKILL.md` |
| dev spools (1-GPU jobs) | {spools} |
| env script every job sources | `{ctx.env_script}` |
| campaign workdir | `{ctx.campaign_dir}` |
| your output dir | `{out_dir}` (micro-bench cases + results under `microbench/`) |
| **decision file to write** | `{out_dir / 'decision.yaml'}` |
| kernel choices to append | `{ctx.perf_root / '08-kernel-sources' / 'kernel_choices.json'}` (a JSON list; create if absent) |
| model | `{ctx.model}` |

Job pattern (never block; launch, then poll with short calls; job names start with `{job_prefix}-`):

```bash
python3 PKG/substrate/spoolctl.py launch --spool <spool> --gpus 1 --wall 45m --tree {worktree} {job_prefix}-<what> \\
  'source {ctx.env_script}; cd {worktree}/tests/unittest && python -m pytest -q -p no:cacheprovider <path>'
python3 PKG/substrate/spoolctl.py status --spool <spool>; cat <spool>/done/{job_prefix}-<what>.rc; tail -n 60 <spool>/logs/{job_prefix}-<what>.log
```

Micro-bench: `python PKG/tune/microbench.py <case file> --out <json>` inside a job with `--tree {worktree}`.

## Scope

{scope}

## Embargo (do not read, grep or call)

{embargo}
{sisters}
## Workload and where the campaign stands

{ctx.workload}

{ctx.standings_table}

{ctx.notes}

## The hypothesis to pursue

{hypothesis}

## Deliverables

{deliverables}
"""
    path = out_dir / "request.md"
    path.write_text(text)
    return path


def turn_prompt(package: Path, request: Path, *, budget_hours: float = 3.0) -> str:
    """The driver's message for one tuner turn run by a general-purpose agent."""
    return f"""You are executing ONE turn of the `tuner` role of the staircase-onboard tune stage.

STEP 0 -- your role definition: read {package / 'prompts' / 'tuner.py'}; its SYSTEM_PROMPT string is your
system prompt for this turn. Follow it exactly.

STEP 1 -- your request: read {request}. It names your worktree, spools, skills (plain files: read them),
embargo, hypothesis, deliverables and the decision file you must write at the end.

Operating rules for this environment:
- You are on a login node with no GPU. All GPU work goes through the dev spools (`spoolctl.py launch ...
  --tree <worktree>`) and polling. A single tool call must finish within ~8 minutes: never block on a job.
- Never emit more than ~150 lines in a single Write/Edit call; build big files with several calls.
- Edit only inside your worktree and your candidate output dir (and append to kernel_choices.json).
- Commit in your worktree with `git -C <worktree> commit -s -m "..."` (no co-author lines). Do not push.
- Formatting polish is not required; correctness and measured numbers are.
- Get a minimal version working and measured early, then iterate. Budget ~{budget_hours:g} h of wall time;
  record and drop sub-ideas that stall.

Final message: <= 25 lines: decision, files changed, test results, micro-bench table, projection, risks."""


def render_rebase_request(ctx: CampaignContext, *, slug: str, worktree: Path, best_branch: str, reason: str) -> Path:
    """A candidate that won on its own but does not stack on the best tree: rebase it.

    Parallel candidates written from one base collide in the forward file once
    one of them is kept; the textual conflict is not a verdict on either."""
    out_dir = ctx.campaign_dir / "candidates" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# Tuner request: rebase {slug}

Kind: REBASE. Your candidate does not stack on the campaign's best tree `{best_branch}`:

```
{reason[:1500]}
```

1. `git -C {worktree} rebase {best_branch}`; resolve every conflict so BOTH sides keep their intent (the
   best tree's kept changes untouched, your candidate's change applied on top -- in both PrefillTarget and
   DecodeTarget where it applies). Read your own `decision.yaml` in `{out_dir}` for what the candidate does.
2. Run the target tests that cover both sides plus the claims / target-contract / weight-table tests on the
   dev spools ({', '.join(f'`{s}`' for s in ctx.dev_spools)}; launch + poll, `--tree {worktree}`, env script
   `{ctx.env_script}`).
3. Commit with `git -C {worktree} commit -s` if you changed code beyond the conflict markers. Do not push.
4. Update `{out_dir / 'decision.yaml'}` in place: `commits:` = `git -C {worktree} rev-list --reverse
   {best_branch}..HEAD`, `rebased_onto: <sha>`, a `notes` line on the conflict and its resolution;
   `decision: CANDIDATE` if the tests pass, else `BLOCKED` with the failure.

Embargo: {', '.join(f'`{e}`' for e in ctx.embargo)}.
"""
    path = out_dir / "rebase-request.md"
    path.write_text(text)
    return path
