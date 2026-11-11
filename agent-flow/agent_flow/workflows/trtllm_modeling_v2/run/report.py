# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What code checks, and what the run comes to: the onboard check, the final pass, the report.

**The onboard check** (:func:`onboard_check`) runs when `core` ends ready, on its clean
committed worktree, before the commit lands: the first deployment's boot and each of its
benchmarks under ``modeling_v2=require``, against the frozen anchors. A pass lands the
commit and writes the onboard record (:func:`record_onboarded`): the commit, the content
hash of its native libraries, the time, and ``refs/mv2/<slug>/onboarded`` pointing at it.
Nothing later takes the record back. A benchmark with no anchor is a score with no
verdict; with none judged the record says ``accuracy unjudged (boot only)``.

**The final pass** (:func:`final_pass`) runs once, after the last task: every other check
of every done deployment, the target's delivered tests and the commit audit, on the
checkout's head. Each is a report line naming the commit it was measured on. A deployment
whose boot failed there is not scored. When a check failed or the conventions scan has
findings, one `fix` task names them all; then the deployments that had a failure and core's
own checks run once more. A pass on record for the same content is
reused (``run/contract.py``); deployments not done or not attempted are skipped.

**The report** (:func:`write_report`), ``delivery_report.md``, is what the run delivers.
Its first line says ``onboarded`` or ``not onboarded``; the process exits 0 on the first,
whatever else failed.
"""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Sequence

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ..agents.decisions import DECISIONS_DIRNAME
from ..checks import anchors, conventions, gate_cli, pytest_runner
from ..checks import verdict as check_result
from ..infra.compute.commands import CommandSpec
from ..infra.files import write_text_atomic
from ..infra.git import (
    DEFAULT_BASE_REFS,
    CommitPolicy,
    CommitPolicyError,
    audit_commits,
    git_output,
    resolve_base,
)
from ..operators.coverage import CATALOG_TESTS_DIR, MODELING_V2_PACKAGE, model_ops_tests_dir
from ..spec import deployments as deployments_mod
from . import contract as contract_mod
from .integration import LABEL_TARGET, WORKTREES_DIRNAME, load_labels
from .journal import utc_now

REPORT_FILENAME = "delivery_report.md"
#: Under the workspace: where the run's own checks write, one directory per commit.
CHECKS_DIRNAME = "gates"
#: Under the workspace: the native libraries the onboard check accepted, when the checkout's
#: are not those.
LIBRARIES_DIRNAME = "onboarded_libraries"
#: How many times a check the infrastructure ended (no verdict) runs again.
INFRASTRUCTURE_RERUNS = 1
#: The most test modules of its own a target delivers that a reviewer accepts: reported past
#: it, never a failure.
TARGET_TEST_MODULES_MAX = 20
#: The gates of core the onboard check runs; its others are the final pass's.
ONBOARD_GATES = ("boot", "accuracy")
#: What no check of this flow looks at, said in every report.
NOT_CHECKED: tuple[str, ...] = (
    "whether the target's tests are registered with the repository's CI",
    "whether a disaggregated deployment's generation server reused what its context server "
    "computed: the proxy does not show it (unknown)",
    "performance: the flow has no tuning stage",
)


def onboard_ref(slug: str) -> str:
    return f"refs/mv2/{slug or 'run'}/onboarded"


def measure(
    workflow, check: contract_mod.Check, *, tree: Path, commit: str, bases: Collection[str] = ()
) -> contract_mod.Verdict:
    """``check`` on ``tree`` at ``commit``, recorded; a pass on record for this content is reused.

    A relative anchor's score is the other deployment's on the same content: measured here
    first when that deployment is one of ``bases``, else absent, and the check reports a
    score with no verdict. A check that cannot be run at all is a failed check with why.
    """
    identity = workflow._identity(commit, tree)
    status = workflow._contract_status()
    have = status.on(identity)
    if check.key in have and have[check.key].passed:
        workflow.journal.event("check", f"{check.key}: on record for this content, reused")
        return have[check.key]
    relative = None
    held = None
    if check.gate.startswith("accuracy["):
        held = anchors.frozen(workflow.workspace, check.deployment, check.gate[9:-1])
    if held is not None and held.kind == anchors.RELATIVE:
        base = contract_mod.Check(held.relative_to, check.gate)
        scored = have.get(base.key)
        if scored is None and held.relative_to in bases:
            scored = measure(workflow, base, tree=tree, commit=commit)
        relative = scored.measured if scored is not None else None
    result: Any = None
    for _ in range(1 + INFRASTRUCTURE_RERUNS):
        try:
            result = gate_cli.run_check(
                workflow.task_data,
                check.gate,
                check.deployment,
                workdir=workflow.workspace / CHECKS_DIRNAME / (commit[:12] or "head"),
                tree=tree,
                workspace=workflow.workspace,
                runner=workflow._shell_runner_override,
                relative_score=relative,
                measure_base=False,
            )
        except Exception as exc:  # a failed check, never an error of the run
            result = check_result.GateResult(
                name=check.gate,
                passed=False,
                deployment=check.deployment,
                detail=f"the check could not be run: {type(exc).__name__}: {exc}",
            )
        if not result.infrastructure:
            break
        workflow.journal.event("check without a verdict", f"{check.key}: {result.infrastructure}")
    verdict = status.record(check, result, tree=identity, commit=commit)
    workflow.journal.event("check", f"at {commit[:12]}: {result.summary()}")
    colour = "green" if verdict.passed else "yellow"
    print_message(f"[bold {colour}]{escape_markup(result.summary())}[/bold {colour}]")
    return verdict


def _onboard_checks(workflow) -> list[contract_mod.Check]:
    core = deployments_mod.resolve(workflow.task_data)[0].name
    return [c for c in workflow._contract.of(core) if c.gate.partition("[")[0] in ONBOARD_GATES]


def onboard_check(workflow, tree: Path, commit: str) -> str:
    """Core's boot and benchmarks on ``tree`` at ``commit``; the first failure's line, or ``""``."""
    print_rule("[bold magenta]onboard check[/bold magenta]")
    for check in _onboard_checks(workflow):
        verdict = measure(workflow, check, tree=tree, commit=commit)
        if not verdict.passed:
            return verdict.summary or f"{check.key}: {verdict.detail}"
    return ""


def record_onboarded(workflow, tree: Path, commit: str) -> None:
    """Write the onboard record for ``commit``, once; see the module docstring."""
    state = workflow._state
    if state.onboarded:
        return
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    status = workflow._contract_status()
    scores = [
        status.last(c.key, passed=True) for c in _onboard_checks(workflow) if c.gate != "boot"
    ]
    judged = any(v is not None and not v.unjudged for v in scores)
    libraries = contract_mod.library_digest(tree)
    ref = onboard_ref(state.slug)
    if workflow._git(str(repo), "update-ref", ref, commit) is None:
        ref += " (git did not write it)"
    state.onboarded = dict(
        commit=commit,
        ref=ref,
        libraries=libraries,
        time=utc_now(),
        accuracy="judged" if judged else "unjudged (boot only)",
    )
    workflow._checkpoint(state)
    workflow.journal.event("onboarded", f"{commit} ({ref}); accuracy {state.onboarded['accuracy']}")
    print_message(f"[bold green]✔ onboarded at {commit[:12]}[/bold green]")
    if Path(tree) == repo or libraries == contract_mod.library_digest(repo):
        return
    # This run rebuilt them: the checkout does not hold what was accepted. A copy that
    # fails (a full disk) is said and takes nothing back.
    try:
        for library in contract_mod.native_libraries(Path(tree)):
            kept = workflow.workspace / LIBRARIES_DIRNAME / library.relative_to(tree)
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, kept)
    except OSError as exc:
        workflow.journal.event("onboarded", f"the accepted libraries were not kept: {exc}")


# ------------------------------------------------------------------------ the final pass


@dataclass(frozen=True)
class Line:
    """One thing the final pass looked at that is no deployment's check."""

    name: str
    passed: bool
    detail: str
    commit: str = ""

    def render(self) -> str:
        at = f" at `{self.commit[:12]}`" if self.commit else ""
        return f"[{'ok' if self.passed else 'FAIL'}] {self.name}{at}: {self.detail}"


def _owed(pipeline) -> list[contract_mod.Check]:
    """The final pass's checks: every done deployment's, but core's onboard ones and what preflight left out."""
    w = pipeline.w
    left = {(r["deployment"], r["check"]) for r in getattr(w._preflight, "not_attempted", ())}
    first = _onboard_checks(w)
    return [
        check
        for name in pipeline.done_deployments()
        for check in w._contract.of(name)
        if (name, check.gate) not in left and check not in first
    ]


def _measure_owed(pipeline, commit: str, only: Collection[str] | None = None) -> list:
    """Each owed check (of the deployments ``only`` names, when given) on the head; those that failed.

    A deployment whose boot failed is not scored or tested there: every later check of it
    would start the same engine to the same end.
    """
    w, repo = pipeline.w, Path(str(pipeline.w.task_data["trtllm_repo_path"]))
    done, failed, down = pipeline.done_deployments(), [], set()
    for check in _owed(pipeline):
        if check.deployment in down or (only is not None and check.deployment not in only):
            continue
        verdict = measure(w, check, tree=repo, commit=commit, bases=done)
        if not verdict.passed:
            failed.append(verdict)
            down |= {check.deployment} if check.gate == "boot" else set()
    return failed


def git_lines(
    repo: Path, policy: CommitPolicy, trees: Sequence[str], commit: str = ""
) -> list[Line]:
    """The delivered ``trees`` are in git, and the run's commits meet the commit rules.

    Nothing outside a git work tree: a fixture in a temporary directory is no delivery. The
    audit runs over every commit since the merge base that touches the target's own paths,
    by the rules each task's commits were held to when they landed.
    """
    if not (repo / ".git").exists():
        return []
    dirt = git_output(repo, "status", "--porcelain", "--", *trees)
    dirty = [line for line in (dirt or "").splitlines() if line.strip()]
    said = "the delivered trees are clean at HEAD"
    if dirt is None:
        said = "git status could not be run"
    elif dirty:
        said = "uncommitted, so a clone would not have them:\n  " + "\n  ".join(dirty[:12])
    lines = [Line("git:target-committed", dirt is not None and not dirty, said, commit)]
    base = resolve_base(repo, [policy.base_ref] if policy.base_ref else DEFAULT_BASE_REFS)
    if not base:
        return lines
    try:
        found = audit_commits(repo, f"{base}..HEAD", policy, paths=policy.own_paths)
        said = "\n  ".join(f.render() for f in found[:12])
    except CommitPolicyError as exc:
        found, said = [exc], str(exc)
    said = (
        f"{len(found)} finding(s) since {base[:10]}:\n  {said}"
        if found
        else (f"every commit since {base[:10]} is signed off by its author, with no co-author")
    )
    return [*lines, Line("git:commit-rules", not found, said, commit)]


def test_modules(repo_root: str | Path, target_rel: str, commit: str = "") -> Line | None:
    """How many test modules the target delivers, and their lines; ``None`` with none.

    Never a failure: past :data:`TARGET_TEST_MODULES_MAX` it says so as a warning.
    """
    tests_dir = model_ops_tests_dir(target_rel)
    root = Path(repo_root) / tests_dir
    found = root.rglob("*.py") if root.is_dir() else ()
    files = sorted(p for p in found if pytest_runner.TEST_FILE.search(p.as_posix()))
    if not files:
        return None
    count = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in files)
    detail = f"{len(files)} test module(s), {count} lines, under {tests_dir}"
    if len(files) > TARGET_TEST_MODULES_MAX:
        detail += (
            f"; WARNING: more than the {TARGET_TEST_MODULES_MAX} a reviewer accepts: consolidate "
            "them by domain to 10-15 modules, with one copy of each helper"
        )
    return Line("tests:modules", True, detail, commit)


def test_job_line(command: str, tree: Path) -> str:
    """``command`` run from inside ``tree`` with the tree first on ``PYTHONPATH``, as the gates run.

    A test that starts rank processes runs them as scripts, which resolve ``tensorrt_llm``
    from ``PYTHONPATH`` and the installed package: without the tree there they import
    another checkout.
    """
    root = shlex.quote(str(Path(tree).resolve()))
    return f'cd {root} && export PYTHONPATH={root}"${{PYTHONPATH:+:$PYTHONPATH}}" && {command}'


def delivered_tests(workflow, commit: str) -> Line | None:
    """The target's own tests at ``commit``, run now in the checkout; ``None`` with none.

    It holds only on none failing, none left unrun and one at least passed.
    """
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    paths = [model_ops_tests_dir(workflow._preflight.target_rel)]
    files = pytest_runner.list_test_files(repo, commit, paths) if commit else []
    if not files:
        return None
    python = str((workflow.task_data.get("environment") or {}).get("python") or "python3")
    report = workflow.workspace / CHECKS_DIRNAME / commit[:12] / "delivered-tests.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    command = pytest_runner.pytest_command(python, repo, files, report, ("--timeout=180",))
    line = test_job_line(command, repo)
    if workflow._shell_runner_override is not None:
        done = workflow._shell_runner(line, None, pytest_runner.TIMEOUT_S)
    else:
        spec = CommandSpec(argv=line, cwd=str(repo), timeout_s=pytest_runner.TIMEOUT_S)
        done = workflow._command_executor.execute(spec)
    outcome = pytest_runner.read_report(report, files)
    if outcome.unread or (done.returncode and outcome.verdict() == "CLEAN"):
        tail = (done.stdout + done.stderr)[-2000:]
        return Line("tests", False, f"INCOMPLETE (pytest exited {done.returncode}): {tail}", commit)
    said = f"{outcome.describe()}; {len(files)} test file(s) asked for"
    return Line("tests", outcome.verdict() == "CLEAN", said, commit)


def delivery_lines(workflow, commit: str) -> list[Line]:
    """What the final pass looks at besides the deployments' checks, at ``commit``."""
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    target_rel = workflow._preflight.target_rel
    trees = (target_rel, MODELING_V2_PACKAGE, CATALOG_TESTS_DIR)
    lines: list[Line | None] = [*git_lines(repo, workflow._commit_policy(), trees, commit)]
    try:
        lines += [test_modules(repo, target_rel, commit), delivered_tests(workflow, commit)]
    except Exception as exc:  # a failed line, never an error of the run
        lines.append(Line("tests", False, f"could not be run: {type(exc).__name__}: {exc}", commit))
    return [line for line in lines if line is not None]


def final_pass(pipeline) -> None:
    """Run the final pass on the checkout's head; its lines are left on ``pipeline.final``."""
    w = pipeline.w
    print_rule("[bold magenta]final pass[/bold magenta]")
    repo, commit = Path(str(w.task_data["trtllm_repo_path"])), w._head()
    done = pipeline.done_deployments()
    w.journal.event("final pass", f"on {commit[:12]}: {', '.join(done) or 'no deployment done'}")
    failed = _measure_owed(pipeline, commit)
    lines = delivery_lines(w, commit)
    cli = gate_cli.invocation(w.task_path, w.workspace / CHECKS_DIRNAME / "fix", "<name>")
    problems = [f"check `{v.key}` failed at {commit[:12]}: {v.summary or v.detail}" for v in failed]
    problems += [
        f"{line.name} at {commit[:12]}: {line.detail}" for line in lines if not line.passed
    ]
    findings = findings_left(w)
    if problems or findings:
        said = [*problems, *(f"convention: {finding}" for finding in findings)]
        fix = pipeline.fix(said, f"A check runs again with `{cli}`.")
        if fix.status == pipeline.DONE:
            # Once more on what the fix left: the deployments that had a failure, whole, and
            # core's own checks, which nothing of this takes the onboard record from.
            commit = w._head()
            _measure_owed(pipeline, commit, only={v.deployment for v in failed})
            onboard_check(w, repo, commit)
            lines = delivery_lines(w, commit)
    pipeline.final = lines


# ------------------------------------------------------------------------------ the report


def _stopped(workflow, pipeline) -> list[str]:
    """Where a run that is not onboarded stopped: the result line, how to see it again, the last pass."""
    task = pipeline.stopped
    if task is None:
        return ["# not onboarded: the run ended before `core` was done"]
    status = workflow._contract_status()
    # A check ended core's last attempt when the last one recorded for it is the task's reason.
    keys = {check.key for check in _onboard_checks(workflow)}
    recorded = [v for v in status.verdicts if v.key in keys]
    v = recorded[-1] if recorded else None
    lines = [f"# not onboarded: stopped at `{task.id}`, not done after {task.attempts} attempt(s)"]
    lines += ["", f"- result: {task.note or '(no reason recorded)'}"]
    if task.kind == "core" and v is not None and not v.passed and v.summary in task.note:
        # The task's worktree holds what was checked; without one, the checkout does.
        tree = workflow.workspace / WORKTREES_DIRNAME / task.id
        tree = tree if tree.is_dir() else workflow.task_data["trtllm_repo_path"]
        gate = f"gate {shlex.quote(v.gate)} --task {workflow.task_path} --deployment {v.deployment}"
        lines.append(f"- reproduce: `{pipeline.cli()} {gate} --tree {tree}`")
        passed = status.last(v.key, passed=True)
        lines.append(
            f"- last commit where `{v.key}` passed: `{passed.commit if passed else 'none'}`"
        )
    else:
        files = workflow.workspace / DECISIONS_DIRNAME
        lines.append(
            f"- its last attempt ended at no check of the run: see `{files}/{task.id}-*.yaml`"
        )
    return lines


def _uncertified(workspace: Path) -> list[str]:
    """Each task's `uncertified` list from its last decision file, one line per item."""

    def order(path: Path) -> tuple[str, list[int]]:
        task, _, token = path.stem.rpartition("-")
        return task, [int(x) for x in token.split(".") if x.isdigit()]

    # By task, attempt and turn: the last of a task is the latest word of its worker.
    files = sorted((workspace / DECISIONS_DIRNAME).glob("*.yaml"), key=order)
    last = {order(path)[0]: path for path in files if order(path)[0]}
    lines: list[str] = []
    for task, path in sorted(last.items()):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        items = data.get("uncertified") if isinstance(data, dict) else None
        items = [x for x in items.splitlines() if x.strip()] if isinstance(items, str) else items
        lines += [f"- `{task}` left uncertified: {str(item).strip()}" for item in items or ()]
    return lines


def findings_left(workflow) -> list[str]:
    """What the conventions scan finds in the checkout now, each with the task that first showed it."""
    task, first = workflow.task_data, conventions.load(workflow.workspace)
    target_rel = str(getattr(workflow._preflight, "target_rel", "") or "")
    try:
        found = conventions.scan(task["trtllm_repo_path"], target_rel, str(task.get("arch") or ""))
    except Exception as exc:  # a tree the scan cannot read is said, and fails nothing
        return [f"the scan could not run: {type(exc).__name__}: {exc}"]
    return [f"{f} (first at `{first[f]}`)" if f in first else f for f in found]


def _done_rows(workflow, pipeline) -> list[str]:
    """One table row per check of each done deployment: what it measured, against what, where."""
    status, frozen = workflow._contract_status(), anchors.load(workflow.workspace)
    left = {
        (r["deployment"], r["check"]) for r in getattr(workflow._preflight, "not_attempted", ())
    }
    rows = []
    for name in pipeline.done_deployments():
        for check in (c for c in workflow._contract.of(name) if (name, c.gate) not in left):
            v = status.last(check.key)
            held = frozen.get(f"{name}:{check.gate[9:-1]}") if "[" in check.gate else None
            anchor = "-"
            if held is not None:
                value = "" if held.value is None else f" {held.value:g}"
                anchor = f"{held.kind}{value} ({held.source})".replace("|", "/")
            score = "-" if v is None or v.measured is None else f"{v.measured:g}"
            cells = (score, anchor, v.said if v else "not run", (v.path if v else "") or "-")
            commit = (v.commit if v else "")[:12] or "-"
            rows.append(f"| `{name}` | `{check.gate}` | {' | '.join(cells)} | `{commit}` |")
    head = ["| deployment | check | score | anchor | verdict | judged | commit |", "|---|" * 7]
    return [head[0], head[1].replace("||", "|"), *rows] if rows else []


def write_report(workflow, pipeline) -> Path:
    """``delivery_report.md``: written whenever the pipeline ended, whatever it came to."""
    w, task, record = workflow, workflow.task_data, workflow._state.onboarded
    repo = Path(str(task["trtllm_repo_path"]))
    if record:
        unjudged = "" if record["accuracy"] == "judged" else f"; accuracy {record['accuracy']}"
        where = f"commit `{record['commit']}`, ref `{record['ref']}`, {record['time']}"
        lines = [f"# onboarded: {where}{unjudged}"]
    else:
        lines = _stopped(w, pipeline)
    lines += [
        "",
        f"- checkpoint: `{task.get('checkpoint_path')}`",
        f"- target: `{getattr(w._preflight, 'target_rel', '')}`",
        f"- repository: `{repo}` at `{w._head() or '(no commit)'}`",
    ]
    if record:
        now, kept = contract_mod.library_digest(repo), w.workspace / LIBRARIES_DIRNAME
        same = now == record["libraries"]
        said = "are the onboarded ones" if same else f"({now}) are not the onboarded ones"
        if not same and kept.is_dir():
            said += f": the accepted ones are in `{kept}`"
        lines.append(f"- native libraries: the head's {said} ({record['libraries']})")
    for t in pipeline.tasks:
        if t.lib_end and t.lib_end != t.lib_start and t.status != pipeline.DONE:
            lines.append(f"- native libraries were changed by `{t.id}`, which did not land")

    def section(title: str, rows: list[str]) -> None:
        lines.extend(["", f"## {title}", "", *(rows or ["(none)"])])

    spent = [t for t in pipeline.tasks if t.status == pipeline.NOT_DONE]
    decisions = w.workspace / DECISIONS_DIRNAME
    section("Done", _done_rows(w, pipeline))
    section(
        "Not done",
        [
            f"- `{t.id}` ({t.kind}): {t.note or 'no reason recorded'}; evidence: "
            f"`{decisions / f'{t.id}-{t.attempts}.yaml'}`"
            for t in spent
            if t.kind != "kernel"
        ],
    )
    left = getattr(w._preflight, "not_attempted", ())
    section(
        "Not attempted",
        [f"- `{r['deployment']}` {r['check']}".rstrip() + f": {r['reason']}" for r in left],
    )
    section(
        "Changed outside the target",
        [
            f"- `{row['sha'][:12]}` {row['label']} (`{row['task']}`): {row['subject']}: "
            + ", ".join(row.get("outside") or ())
            for row in load_labels(w.workspace)
            if row.get("label") != LABEL_TARGET
        ],
    )
    section(
        "Skipped kernel rows",
        [f"- `{t.id}` (queue row {t.ref}): {t.note}" for t in spent if t.kind == "kernel"],
    )
    section("Convention findings left", [f"- {f}" for f in findings_left(w)])
    section("Final pass", [f"- {line.render()}" for line in pipeline.final])
    unchecked = [*NOT_CHECKED, *w._contract.notes]
    section("What the run did not check", [f"- {x}" for x in unchecked] + _uncertified(w.workspace))
    section(
        "Models per role",
        [
            f"- {role}: {a.backend} {a.model} ({a.reasoning_effort or 'default effort'})"
            for role, a in w._role_agents().items()
        ],
    )
    section("Tasks", [pipeline.table(), ""])
    path = w.workspace / REPORT_FILENAME
    write_text_atomic(path, "\n".join(lines))
    return path
