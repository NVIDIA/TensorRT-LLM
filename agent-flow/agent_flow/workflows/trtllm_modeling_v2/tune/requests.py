# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render one tuner request: everything a candidate turn needs, in one file.

The request is read by the turn's user prompt (:func:`turn_prompt`), which
carries the `tune` tag's section; ``agents/prompts/tuner.py`` is its system
prompt. A driver that delivers the request without the system prompt runs a
different agent.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..agents.prompts.tags import tag_section
from . import sol as sol_mod
from .records import Measurement
from .spec import TuneSpec

PACKAGE = Path(__file__).resolve().parent.parent


@dataclass
class CampaignContext:
    repo: Path
    package: Path
    campaign_dir: Path
    perf_root: Path
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
    branch: str,
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
    kernel_choices = ctx.perf_root / "08-kernel-sources" / "kernel_choices.json"
    job = f"{job_prefix}-<what>"
    text = f"""# Tuner request: {slug}

Issued {time.strftime("%Y-%m-%d %H:%M %Z")} by the tune stage.

## Paths

| name | value |
|---|---|
| repo (main checkout, read-only for you) | `{R}` |
| **your worktree** (edit here only) | `{worktree}`, branch `{branch}` off `{base[:10]}` |
| target dir (inside the worktree) | `{ctx.target_dir}` |
| target tests (inside the worktree) | `{ctx.target_tests}` |
| package (spoolctl) | `PKG = {ctx.package}` |
| dev spools (1-GPU jobs) | {spools} |
| env script every job sources | `{ctx.env_script}` |
| campaign workdir | `{ctx.campaign_dir}` |
| your output dir | `{out_dir}` (micro-bench script + results under `microbench/`) |
| **decision file to write** | `{out_dir / "decision.yaml"}` |
| kernel choices to append | `{kernel_choices}` (a JSON list; create if absent) |
| model | `{ctx.model}` |

Job pattern (never block; launch, then poll with short calls; job names start with `{job_prefix}-`):

```bash
python3 PKG/infra/compute/spoolctl.py --spool <spool> launch --gpus 1 --wall 45m --tree {worktree} {job} \\
  'source {ctx.env_script}; cd {worktree}/tests/unittest && python -m pytest -q -p no:cacheprovider <path>'
python3 PKG/infra/compute/spoolctl.py --spool <spool> status
cat <spool>/done/{job}.rc; tail -n 60 <spool>/logs/{job}.log
```

Each dev spool's `status` shows its free devices and what waits; launch on one that can start
the job now, and poll that spool.

Micro-bench: write your own script under `microbench/` and run it inside a job with `--tree {worktree}`.

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

Python verifies isolated changes with your identical `microbench.script` on the current best
and the stacked candidate, one run per revision, no repeats. Export `benchmark()` returning
`{{case: callable}}` for the exact `microbench.cases` list. Each callable runs the real op/module
on the current CUDA stream, importing the active checkout's code (eager or a captured graph).
The wrapper performs 10 warmups and 100 synchronized CUDA-event timings per case; do not return
claimed timings. Projections and development results are not evidence. Correctness gates still run before every keep.

Declare `perf_scope: e2e` for serving/scheduling, host, overlap, communication or system-wide
effects a local harness cannot represent, and explain why. Python checks known kinds/paths;
it cannot infer every semantic interaction inside a forward. Only those candidates are served,
once per revision/config against a same-host control, with no post-control or automatic rerun.
Cumulative serving validation happens once at campaign wrap-up, not after each local keep.

For e2e candidates also name in `decision.yaml`:

- `changed_members: [<perf config>, ...]`: the configs whose engine runs your change;
- `path_marker: "<text>"`: a line your change prints wherever its path is taken (once per rank,
  when the engine is built or the path first runs), and nowhere else.

A config outside `changed_members` whose server log lacks the marker, and whose engine pools
match the control, is not judged. This never triggers another measurement.
"""
    path = out_dir / "request.md"
    path.write_text(text)
    return path


def turn_prompt(request: Path) -> str:
    """The message for one tuner turn: the request to read, the `tune` section, this host's rules."""
    return f"""You are executing ONE turn of the trtllm-modeling-v2 tune campaign.

Your request: read {request}. It names your worktree, spools, embargo, hypothesis, deliverables and
the decision file you must write at the end.

{tag_section(("tune",)).strip()}

## Operating rules for this environment

- You are on a login node with no GPU. All GPU work goes through the dev spools (`spoolctl.py launch ...
  --tree <worktree>`) and polling.
- Never emit more than ~150 lines in a single Write/Edit call; build big files with several calls.
- Always give grep/rg/find an explicit search root (your worktree's target dir, a shared module path, the
  package); never search from the repository root or the current directory -- an embargoed file's lines in
  a search result are a read of it. If one appears anyway, do not use it and say so in your final message.
- Edit only inside your worktree and your candidate output dir (and append to kernel_choices.json).
- Commit in your worktree with `git -C <worktree> commit -s -m "..."` (no co-author lines). Do not push.
- Formatting polish is not required; correctness and measured numbers are.
- Get a minimal version working and measured early, then iterate. Budget ~3 h of wall time;
  record and drop sub-ideas that stall.

Final message: <= 25 lines: decision, files changed, test results, micro-bench table, projection, risks."""


def render_rebase_request(
    ctx: CampaignContext, *, slug: str, worktree: Path, best_branch: str, reason: str
) -> Path:
    """A candidate that won on its own but does not stack on the best tree: rebase it."""
    out_dir = ctx.campaign_dir / "candidates" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# Tuner request: rebase {slug}

Kind: REBASE. Your candidate does not stack on `{best_branch}` (the campaign's best tree):

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
   `decision: CANDIDATE` if the tests pass, else `BLOCKED` with the failure. Keep the same
   micro/module harness and case list runnable on both revisions; refresh them if the rebase
   changes the benchmarked interface. Python verifies the rebased stack once before a keep.

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

    The request is a candidate request (:func:`render_request`) for ``variant``.
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
- `decision.yaml`: `decision: CANDIDATE`, `perf_scope: e2e`, `variant_of: {slug}`,
  `condition: <the expression the target reads>`, and `commits:` every commit the best tree does not carry,
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


class TuneRequests:
    """Render analysis requests from the resolved spec and current standings."""

    def __init__(self, spec: TuneSpec, standings: Callable[[], str]) -> None:
        self.spec = spec
        self._standings = standings

    def sources_note(self) -> str:
        """Where a tuner looks before writing a kernel."""
        s = self.spec
        lines = [
            "## Sources to check before writing a kernel",
            "",
            "- FlashInfer **as installed** in the run's interpreter: a public callable the engine does not "
            "call yet that answers a demand is a tier-3 entry (`from flashinfer import ...` in the entry), "
            "not a new kernel.",
            f"- Earlier choices for this target: `{s.workdir / '08-kernel-sources' / 'kernel_choices.json'}` (chosen + "
            "alternatives with timings); extend it, do not re-race what is recorded.",
            f"- Prior art in this repository: `git -C {s.repo} log --oneline -i --grep='perf' -- "
            "tensorrt_llm/_torch/modules/<similar module> tensorrt_llm/_torch/models/<similar model>` (strategies "
            "for the same op shapes in other models -- never the embargoed in-tree implementation of this model).",
            "- Profiles of the current best tree: the latest `roadmap*.yaml` evidence in the campaign workdir.",
        ]
        return "\n".join(lines)

    def _roadmap_paths(self) -> str:
        """What a roadmap turn needs to profile the best tree without asking, and the embargo."""
        s = self.spec
        configs = list(s.identity_configs)
        rows = [
            ("best tree to profile", self.spec.workdir / "worktrees" / "best"),
            ("configs, applied in order", ", ".join(f"`{c}`" for c in configs)),
            ("model", s.model),
            ("dev spools (1-GPU jobs: nsys captures)", ", ".join(f"`{p}`" for p in s.dev_spools)),
            ("env script every job sources", s.env_script),
            ("package (bench.py --nsys, spoolctl)", PACKAGE),
        ]
        lines = ["## Paths", "", "| name | value |", "|---|---|"]
        lines += [f"| {k} | {v if str(v).startswith('`') else f'`{v}`'} |" for k, v in rows]
        lines += [
            "",
            "## Embargo (do not read, grep or call)",
            "",
            *(f"- `{e}`" for e in s.embargo),
        ]
        return "\n".join(lines)

    def _sisters_line(self) -> str:
        if not self.spec.sister_engines:
            return ""
        return (
            "\n- Other engines that implement this model (strategies, not code): "
            + ", ".join(f"`{e}`" for e in self.spec.sister_engines)
            + "."
        )

    def sol_request(self, baseline: Measurement, problems: Sequence[str] = ()) -> Path:
        s = self.spec
        sol_dir = s.workdir / "sol"
        sol_dir.mkdir(parents=True, exist_ok=True)
        what = "sol" if not problems else "sol-fix"
        request = s.workdir / f"{what}-request.md"
        keys = ", ".join(baseline.headline())
        fix = ""
        if problems:
            fix = (
                "\n\n## Your last `sol.yaml` was refused\n\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\n\nFix the inputs and rerun the calculator, not the output numbers.\n"
            )
        request.write_text(
            f"# Tuner request: {what}\n\n"
            "Build a public, reproducible roofline model. You change no code. No internal skills, "
            "private hardware tables, private benchmark data or remembered device specifications. "
            "Use only user-provided, explicitly sourced public hardware capacities; if unavailable, "
            "leave the bound unknown rather than guessing. Source labels are not proof of publicness: "
            "check the underlying material before copying it.\n\n"
            f"Read the input contract in `{PACKAGE / 'tune' / 'sol.py'}`. Write "
            f"`{sol_dir / 'work.yaml'}` with `hardware`, `regions`, and optional `headline`. "
            "Hardware capacities use FLOP/s and byte/s, each `{value, source}`, plus `assumptions` "
            "naming precision, dense/sparse mode, memory boundary and device/rank scope. "
            f"Hardware label (not a spec lookup): {s.sol.get('gpu') or 'unspecified'}.\n"
            f"Read `{s.model}` config and `{s.workdir / 'worktrees' / 'best'}` under `{s.target}` "
            "for necessary work. Every region has a stable `id` (module path + phase + shape + rank "
            "/ invocation scope), `flops`, `bytes`, `assumptions`, and optionally `measured_ms` with "
            "`measurement_source`. Work and timings must cover exactly the same invocations. "
            "Keep nested/overlapping regions separate. Unknown quantities are null, not zero.\n"
            "Optional target-local `overhead: {ms, source, assumption}` is added only when the "
            "measurement proves it is exposed, nonoverlapping with the modeled work, and assumed "
            "unchanged by optimization. It is a conditional estimate, never a universal hardware floor.\n"
            f"Headline keys: {keys}. Only supply `headline.<key>` if necessary work covers the "
            "whole metric interval, with `flops`, `bytes`, `assumptions`, `output_tokens` and "
            "`gpu_count` for per-GPU throughput. For tok_s_user, output_tokens and work must cover "
            "the same user's decode interval (not TTFT or other users); for output_tok_s_gpu, "
            "use all output tokens and all required work over the same benchmark interval. "
            "State shared-resource scope and concurrency explicitly. Never derive these from sums "
            "of module bounds or label a decode-only estimate as serving throughput. Omit headline "
            "models you cannot justify: unknown ceilings cannot satisfy a pct_of_sol goal.\n\n"
            "Run the CPU-only deterministic calculator (no GPU job or toolkit dependency):\n\n"
            f"```sh\npython {shlex.quote(str(PACKAGE / 'tune' / 'sol.py'))} "
            f"--input {shlex.quote(str(sol_dir / 'work.yaml'))} "
            f"--output {shlex.quote(str(s.workdir / sol_mod.SOL_FILENAME))}\n```\n\n"
            "The stage recomputes the output and rejects edited headline numbers. "
            "Write the decision file `"
            + str(s.workdir / f"{what}-decision.yaml")
            + "` with `decision: CANDIDATE` (including a partial/unknown model) or `BLOCKED` "
            "(no defensible inputs; explain in `notes`).\n\n## Standings\n\n"
            + self._standings()
            + fix
            + "\n"
        )
        return request

    def sol_ask(self, what: str) -> str:
        """Ask for independent region bounds even when headline bounds are unknown."""
        if not self.spec.sol.get("enabled", True):
            return ""
        work = self.spec.workdir / "sol" / f"{what}-work.yaml"
        output = self.spec.workdir / f"{what}-sol.json"
        return (
            "\n\nRank by headroom where known, not just by time. Join measured module times to "
            "explicit FLOPs/bytes by stable region id (module path + phase + shape + rank / "
            f"invocation scope), using the public SOL input format in `{PACKAGE / 'tune' / 'sol.py'}`. "
            "Reuse only applicable sourced public capacities, never private tables or skill scripts. "
            f"Write `{work}` then run `python {shlex.quote(str(PACKAGE / 'tune' / 'sol.py'))} "
            f"--input {shlex.quote(str(work))} --output {shlex.quote(str(output))}`. "
            "The output joins measured_ms, lower_bound_ms, headroom_ms and bound by id. "
            "Keep missing inputs unknown. Do not sum nested/overlapping regions or promise that "
            "module headroom is end-to-end gain. A near-bound region can still be eliminated or "
            "fused; a conditional overhead estimate is not a physical speed limit."
        )

    def ledger_mode(self) -> str:
        return str(self.spec.campaign.get("kernel_ledger", "require"))

    def ledger_path(self, what: str) -> Path:
        return self.spec.workdir / f"{what}-kernel_ledger.yaml"

    def _ledger_ask(self, what: str) -> str:
        """perf_optimize's kernel ledger, asked of every roadmap: the proof it is exhaustive."""
        if self.ledger_mode() == "off":
            return ""
        from agent_flow.workflows.perf_optimize import kernel_ledger

        target = float(self.spec.campaign.get("kernel_coverage_pct", 80.0))
        return (
            f"\n\nAlso write `{self.ledger_path(what)}`: the agent-flow perf_optimize **kernel ledger** (schema and "
            f"example in the docstring of `{kernel_ledger.__file__}`). One row per kernel (or kernel family) of the "
            f"best tree's decode profile, down the ranking until the rows cover >= {target:g}% of GPU time "
            "(`nsys stats --report cuda_gpu_kern_sum`); each row answers elimination / faster / fusion / "
            "overlap with a roadmap slug (`disposition: item`) or an evidence-backed dismissal. Without an "
            "ncu pass, write "
            '`ncu: "unavailable: nsys-only roadmap turn"` and the estimated `bound` on the row. The stage '
            "validates it with perf_optimize's own `load_ledger` + `cross_validate`."
        )

    def roadmap(self, n: int, tried: Sequence[str] = ()) -> Path:
        """Write the roadmap request for the current best tree."""
        what = "roadmap" if n == 1 else f"roadmap-{n}"
        path = self.spec.workdir / f"{what}.yaml"
        request = self.spec.workdir / f"{what}-request.md"
        again = ""
        if tried:
            again = (
                "\n\nThis is roadmap "
                + str(n)
                + ": re-profile the best tree (`"
                + str(self.spec.workdir / "worktrees" / "best")
                + "`) -- the kept candidates moved the profile. Already attempted (do not repeat; "
                "their decision files under `"
                + str(self.spec.workdir / "candidates")
                + "/<slug>/decision.yaml` list measured follow-ups worth ranking): "
                + ", ".join(f"`{t}`" for t in tried)
                + "."
            )
        request.write_text(
            f"# Tuner request: {what}\n\nProfile the target at the campaign config (nsys through the substrate: "
            f"`{PACKAGE}/tune/harness/bench.py --nsys` in a spool job, read with "
            "`nsys stats --report cuda_gpu_kern_sum` on its `.nsys-rep`), read the forward, and "
            f"write `{path}`: a YAML list of "
            "`{slug, kind, hypothesis, evidence, expected, modeling_edits, sources}` ranked by the "
            "exclusive time each can remove; `sources` names where the idea and its kernel options "
            "come from (`profile`, `code-reading`, `flashinfer`, `repo-prior-art`, `sister-engine`, "
            "`kernel-choices`, each with a reference). Then write the "
            "decision file `"
            + str(self.spec.workdir / f"{what}-decision.yaml")
            + "` with `decision: CANDIDATE` (roadmap ready) or `DROPPED` (nothing left worth a candidate)."
            + self.sol_ask(what)
            + self._ledger_ask(what)
            + again
            + "\n\n"
            + self._roadmap_paths()
            + "\n\n## Standings\n\n"
            + self._standings()
            + "\n\n"
            + self.sources_note()
            + self._sisters_line()
            + "\n"
        )
        return request
