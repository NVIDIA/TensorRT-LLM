# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render one tuner request: everything a candidate turn needs, in one file.

The request is the turn's user prompt; ``agents/prompts/tuner.py`` is its system
prompt. Under ``--agent-transport external`` both travel in ``dispatch/``; a
driver that delivers the request without the system prompt runs a different
agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


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
    branch: str = "",
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
    skills = (
        "`PKG/agents/skills/trtllm-modeling-v2-perf-tuning/SKILL.md` + `references/`, "
        "`PKG/agents/skills/trtllm-modeling-v2-core/SKILL.md` (+ `references/graph-and-sync.md`), "
        "`PKG/agents/skills/trtllm-modeling-v2-kernel-onboard/SKILL.md`"
    )
    kernel_choices = ctx.perf_root / "08-kernel-sources" / "kernel_choices.json"
    job = f"{job_prefix}-<what>"
    text = f"""# Tuner request: {slug}

Issued {time.strftime("%Y-%m-%d %H:%M %Z")} by the tune stage.

## Paths

| name | value |
|---|---|
| repo (main checkout, read-only for you) | `{R}` |
| **your worktree** (edit here only) | `{worktree}`, branch `{branch or "tune/" + slug}` off `{base[:10]}` |
| target dir (inside the worktree) | `{ctx.target_dir}` |
| target tests (inside the worktree) | `{ctx.target_tests}` |
| package (skills, microbench, spoolctl) | `PKG = {ctx.package}` |
| skills to read first | {skills} |
| dev spools (1-GPU jobs) | {spools} |
| env script every job sources | `{ctx.env_script}` |
| campaign workdir | `{ctx.campaign_dir}` |
| your output dir | `{out_dir}` (micro-bench cases + results under `microbench/`) |
| **decision file to write** | `{out_dir / "decision.yaml"}` |
| kernel choices to append | `{kernel_choices}` (a JSON list; create if absent) |
| model | `{ctx.model}` |

Job pattern (never block; launch, then poll with short calls; job names start with `{job_prefix}-`):

```bash
python3 PKG/infra/substrate/spoolctl.py --spool <spool> launch --gpus 1 --wall 45m --tree {worktree} {job} \\
  'source {ctx.env_script}; cd {worktree}/tests/unittest && python -m pytest -q -p no:cacheprovider <path>'
python3 PKG/infra/substrate/spoolctl.py --spool <spool> status
cat <spool>/done/{job}.rc; tail -n 60 <spool>/logs/{job}.log
```

A spool that will not start the job soon prints a `warning:` naming the dev spools that can
take it now; `launch --pick` sends the job to whichever can start it soonest and prints
which, so poll that spool.

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

## What the measurement reads from your decision

Python measures every perf config on an idle node beside controls of the best tree measured on
that node, and judges the candidate against their mean under the best tree's own spread. Name in
`decision.yaml`:

- `changed_members: [<perf config>, ...]`: the configs whose engine runs your change;
- `path_marker: "<text>"`: a line your change prints wherever its path is taken (once per rank,
  when the engine is built or the path first runs), and nowhere else.

A config outside `changed_members` whose server log lacks the marker, and whose engine pools read
as the control's, is not judged: a loss there is measured again and, if it repeats, reported as
the host's noise. Without a marker that prints, every config is judged unless `changed_members`
names none of them.
"""
    path = out_dir / "request.md"
    path.write_text(text)
    return path


def turn_prompt(package: Path, request: Path, *, budget_hours: float = 3.0) -> str:
    """The driver's message for one tuner turn run by a general-purpose agent."""
    return f"""You are executing ONE turn of the `tuner` role of the trtllm-modeling-v2 tune stage.
Your system prompt is the tuner's role prompt; follow it exactly.

Your request: read {request}. It names your worktree, spools, skills (plain files: read them),
embargo, hypothesis, deliverables and the decision file you must write at the end.

Operating rules for this environment:
- You are on a login node with no GPU. All GPU work goes through the dev spools (`spoolctl.py launch ...
  --tree <worktree>`) and polling. A single tool call must finish within ~8 minutes: never block on a job.
- Never emit more than ~150 lines in a single Write/Edit call; build big files with several calls.
- Always give grep/rg/find an explicit search root (your worktree's target dir, a shared module path, the
  package); never search from the repository root or the current directory -- an embargoed file's lines in
  a search result are a read of it. If one appears anyway, do not use it and say so in your final message.
- Edit only inside your worktree and your candidate output dir (and append to kernel_choices.json).
- Commit in your worktree with `git -C <worktree> commit -s -m "..."` (no co-author lines). Do not push.
- Formatting polish is not required; correctness and measured numbers are.
- Get a minimal version working and measured early, then iterate. Budget ~{budget_hours:g} h of wall time;
  record and drop sub-ideas that stall.

Final message: <= 25 lines: decision, files changed, test results, micro-bench table, projection, risks."""


def render_rebase_request(
    ctx: CampaignContext, *, slug: str, worktree: Path, best_branch: str, reason: str
) -> Path:
    """A candidate that won on its own but does not stack on the best tree: rebase it.

    Parallel candidates written from one base collide in the forward file once
    one of them is kept; the textual conflict is not a verdict on either.
    """
    out_dir = ctx.campaign_dir / "candidates" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# Tuner request: rebase {slug}

Kind: REBASE. Your candidate does not stack on `{best_branch}` (the campaign's best tree, or a batch of
winners being measured on top of it):

```
{reason[:1500]}
```

1. `git -C {worktree} rebase {best_branch}`; resolve every conflict so BOTH sides keep their intent (the
   best tree's kept changes untouched, your candidate's change applied on top -- in both PrefillTarget and
   DecodeTarget where it applies). Read your own `decision.yaml` in `{out_dir}` for what the candidate does.
2. Run the target tests that cover both sides plus the claims / target-contract / weight-table tests on the
   dev spools ({", ".join(f"`{s}`" for s in ctx.dev_spools)}; launch + poll, `--tree {worktree}`, env script
   `{ctx.env_script}`).
3. Commit with `git -C {worktree} commit -s` if you changed code beyond the conflict markers. Do not push.
4. Update `{out_dir / "decision.yaml"}` in place: `commits:` = `git -C {worktree} rev-list --reverse
   {best_branch}..HEAD`, `rebased_onto: <sha>`, a `notes` line on the conflict and its resolution;
   `decision: CANDIDATE` if the tests pass, else `BLOCKED` with the failure.

Embargo: {", ".join(f"`{e}`" for e in ctx.embargo)}.
"""
    path = out_dir / "rebase-request.md"
    path.write_text(text)
    return path


def render_split_request(
    ctx: CampaignContext,
    *,
    slug: str,
    variant: str,
    worktree: Path,
    base: str,
    branch: str,
    best_branch: str,
    split: Mapping[str, Any],
    deltas: Mapping[str, float],
    original: Path,
) -> Path:
    """A candidate that lost only in one parallel mode and won elsewhere: gate it on that mode.

    The request is a candidate request (:func:`render_request`) for ``variant``: the
    same change behind a condition the target knows, so the mode that lost keeps the
    old path. It is measured, judged and gated like any candidate.
    """

    def listed(keys: Sequence[str]) -> str:
        return ", ".join(f"`{k}` {deltas[k]:+.1%}" if k in deltas else f"`{k}`" for k in keys)

    broken = "".join(f"; breaks `{b}`" for b in split.get("broken") or ())
    members = ", ".join(f"`{m}`" for m in split.get("members") or ()) or "the single config"
    region = ", ".join(f"`{r}`" for r in split.get("region") or ())
    hypothesis = f"""Kind: SPLIT, a member-gated variant of `{slug}` (its decision: `{original}`).

`{slug}` lost only where **{split.get("mode")}**: {listed(split.get("losses") or ()) or "none"}{broken}.
It won elsewhere: {listed(split.get("wins") or ())}. Members that lost: {members}; every site
(member.concurrency) the mode covers: {region}.

Gate the same change on that mode. Where `{split.get("condition")}`, the target runs exactly the path it
ran before `{slug}`; everywhere else, `{slug}`'s path, unchanged. Read the parallel traits once, when the
target is built, from its `mapping` (attention DP, EP, TP and world sizes). A condition on the requests a
rank runs is read per step: a captured CUDA graph has one batch size, so the branch is fixed per graph.
Change nothing of the candidate beyond the gate."""
    deliverables = f"""Your worktree is `{slug}`'s measured tree (`{base[:10]}`: the best tree with its
commits). Add the gate on top of it.

- Tests: both sides of the condition, each on the path it selects (the old one where the condition holds),
  the candidate's own entry tests, and the claims / target-contract tests.
- `decision.yaml`: `decision: CANDIDATE`, `variant_of: {slug}`, `condition: <the expression the target
  reads>`, and `commits:` every commit the best tree does not carry, the candidate's included
  (`git -C {worktree} rev-list --reverse {best_branch}..HEAD`). `BLOCKED`, with why, when the mode cannot be
  read where the change runs.

The orchestrator measures the variant at every perf config, like any candidate. It is a WIN only if it
loses nowhere (the members in this mode are now neutral) and every point `{slug}` won still wins, or holds
its gain within the loss floor; then the keep gate runs, as for any keep. There is one variant per
candidate."""
    return render_request(
        ctx,
        slug=variant,
        worktree=worktree,
        base=base,
        hypothesis=hypothesis,
        deliverables=deliverables,
        job_prefix=f"{slug[:9]}-gv",
        branch=branch,
    )
