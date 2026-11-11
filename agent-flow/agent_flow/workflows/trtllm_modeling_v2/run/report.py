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
of every done deployment, the target's delivered tests and whether the delivered trees
are committed, on the checkout's head. Each is a report line naming the commit it was measured on. A deployment
whose boot failed there is not scored. When a check failed or the conventions scan has
findings, one `fix` task names them all; then the deployments that had a failure and core's
own checks run once more. A pass on record for the same content is
reused (``run/contract.py``); deployments not done or not attempted are skipped.

**The report** (:func:`write_report`), ``delivery_report.md``, is what the run delivers.
Its first line says ``onboarded`` or ``not onboarded``; the process exits 0 on the first,
whatever else failed. Each check's line names the commit it was measured on, which path
judged it and what it attested of the engine that ran.
"""

from __future__ import annotations

import json
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
from ..infra.git import git_output, native_artifacts, uncommitted
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
#: Written there after the copy: each library's content hash. No manifest, no finished copy.
LIBRARIES_MANIFEST = "MANIFEST.json"
#: How many times a check the infrastructure ended (no verdict) runs again.
INFRASTRUCTURE_RERUNS = 1
#: The most test modules of its own a target delivers that a reviewer accepts: reported past
#: it, never a failure.
TARGET_TEST_MODULES_MAX = 20
#: The gates of core the onboard check runs; its others are the final pass's.
ONBOARD_GATES = ("boot", "accuracy")
#: The target's own tests, on the ledger beside the deployments' checks.
DELIVERED_TESTS = contract_mod.Check("target", "delivered-tests")
#: What no check of this flow looks at, said in every report.
NOT_CHECKED: tuple[str, ...] = (
    "whether the target's tests are registered with the repository's CI",
    "whether a disaggregated deployment's generation server reused what its context server "
    "computed: the proxy does not show it (unknown)",
    "performance: the flow has no tuning stage",
    "commits on the branch that no task of this run landed: a task's are audited as it lands",
)


def onboard_ref(slug: str) -> str:
    return f"refs/mv2/{slug or 'run'}/onboarded"


def unmeasurable(workflow, tree: Path, commit: str) -> str:
    """Why a check on ``tree`` would not be a check of ``commit``; ``""`` when it would.

    Another head, or files the commit does not hold (an untracked source too): what the
    check measured would not be what lands.
    """
    head = (workflow._git(str(tree), "rev-parse", "HEAD") or "").split()
    if commit and head and head[0] != commit:
        return f"the tree is at {head[0][:12]}, not at the commit the check is for"
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    ignore = (*native_artifacts(repo), *workflow._commit_policy().denied_paths)
    stray = uncommitted(tree, ignore, workflow._git)
    if stray:
        return f"{len(stray)} file(s) in the tree are in no commit: {', '.join(stray[:6])}"
    return ""


def _no_verdict(workflow, check: contract_mod.Check, commit: str, why: str) -> contract_mod.Verdict:
    """A check that measured nothing of ``commit``: said, kept for this start's report, not recorded."""
    said = f"[NO VERDICT] {check.key}: {why}"
    workflow.journal.event("check without a verdict", f"at {commit[:12]}: {said}")
    print_message(f"[bold yellow]{escape_markup(said)}[/bold yellow]")
    verdict = contract_mod.Verdict(
        check.deployment, check.gate, False, why, commit=commit, summary=said, unjudged=True
    )
    workflow._selected[check.key] = verdict
    return verdict


def measure(
    workflow, check: contract_mod.Check, *, tree: Path, commit: str, bases: Collection[str] = ()
) -> contract_mod.Verdict:
    """``check`` on ``tree`` at ``commit``, recorded; a pass on record for this content is reused.

    The tree must be the commit, clean, before the check and after it: otherwise nothing
    was measured of the commit, and there is no verdict, on record or not. A relative
    anchor's score is the other deployment's on the same content: measured here first
    when that deployment is one of ``bases``, else absent, and the check reports a score
    with no verdict; a score that stood so is judged from the record once the other
    deployment has one. A check that cannot be run at all is a failed check with why.
    """
    identity = workflow._identity(commit, tree)
    why = unmeasurable(workflow, tree, commit) or (
        "" if identity else "the tree's content cannot be identified: git names no commit"
    )
    if why:
        return _no_verdict(workflow, check, commit, why)
    status = workflow._contract_status()
    have = status.on(identity)
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
    kept = have.get(check.key)
    # A score that waited for its base is judged now, from the record: no load.
    waited = kept is not None and kept.unjudged and None not in (kept.measured, relative)
    if kept is not None and kept.passed and not waited:
        workflow.journal.event("check", f"{check.key}: on record for this content, reused")
        workflow._selected[check.key] = kept
        return kept
    result: Any = None
    for _ in range(1 + INFRASTRUCTURE_RERUNS):
        try:
            if waited:
                result = gate_cli.rejudge(
                    workflow.task_data,
                    check.gate,
                    check.deployment,
                    kept.measured,
                    relative,
                    workspace=workflow.workspace,
                )
                break
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
    # What was measured is the commit only when the tree is still it, as it was.
    moved = unmeasurable(workflow, tree, commit)
    if not moved and workflow._identity(commit, tree) != identity:
        moved = "its native libraries are not the ones the check started on"
    if moved:
        return _no_verdict(workflow, check, commit, f"the tree changed under the check: {moved}")
    verdict = status.record(check, result, tree=identity, commit=commit)
    workflow._selected[check.key] = verdict
    workflow.journal.event("check", f"at {commit[:12]}: {result.summary()}")
    colour = "green" if verdict.passed else "yellow"
    print_message(f"[bold {colour}]{escape_markup(result.summary())}[/bold {colour}]")
    return verdict


def _onboard_checks(workflow) -> list[contract_mod.Check]:
    core = deployments_mod.resolve(workflow.task_data)[0].name
    return [c for c in workflow._contract.of(core) if c.gate.partition("[")[0] in ONBOARD_GATES]


def _unanchored(workflow, check: contract_mod.Check) -> bool:
    """Whether ``check`` is a benchmark nothing judges: no declared anchor and no reference row."""
    if not check.gate.startswith("accuracy["):
        return False
    held = anchors.frozen(workflow.workspace, check.deployment, check.gate[9:-1])
    return held is not None and held.kind == anchors.NONE


def onboard_check(workflow, tree: Path, commit: str) -> str:
    """Core's boot and benchmarks on ``tree`` at ``commit``; the first failure's line, or ``""``.

    Only what has something to judge it can refuse: a benchmark with no anchor that
    fails to run leaves the deployment onboarded on its boot, and is a line of the report.
    """
    print_rule("[bold magenta]onboard check[/bold magenta]")
    for check in _onboard_checks(workflow):
        verdict = measure(workflow, check, tree=tree, commit=commit)
        if not verdict.passed and not _unanchored(workflow, check):
            return verdict.summary or f"{check.key}: {verdict.detail}"
    return ""


def _pinned(workflow, repo: Path, ref: str) -> str:
    """The commit ``ref`` points at in ``repo``; ``""`` when there is no such ref."""
    return (workflow._git(str(repo), "rev-parse", "--verify", "--quiet", ref) or "").strip()


def _pin(workflow, repo: Path, ref: str, commit: str) -> bool:
    """Create ``ref`` at ``commit``, never move it; whether it now says ``commit``."""
    # The empty old value: git writes the ref only when there is none.
    workflow._git(str(repo), "update-ref", ref, commit, "")
    return _pinned(workflow, repo, ref) == commit


def reconcile_onboarded(workflow) -> None:
    """On a new start, make the state and the ref say the same; neither is moved.

    A ref an earlier start wrote and could not record is the record: the state takes its
    commit. A ref the record names and git did not write is written now.
    """
    state = workflow._state
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    record, ref = state.onboarded, onboard_ref(state.slug)
    if not record:
        pinned = _pinned(workflow, repo, ref)
        if pinned:
            workflow.journal.event("onboarded", f"taken from {ref}: an earlier start wrote it")
            record_onboarded(workflow, repo, pinned)
    elif record["ref"] != ref and _pin(workflow, repo, ref, record["commit"]):
        state.onboarded = {**record, "ref": ref}
        workflow._checkpoint(state)


def record_onboarded(workflow, tree: Path, commit: str) -> str:
    """Write the onboard record for ``commit``, once; why it is not written, or ``""``.

    The ref is created and never moved: where one stands already, its commit is the
    record. Nothing is onboarded that the state file does not hold: a write that fails
    is said, and the next start takes the record from the ref.
    """
    state = workflow._state
    if state.onboarded:
        return ""
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    status = workflow._contract_status()
    scores = [
        status.last(c.key, passed=True) for c in _onboard_checks(workflow) if c.gate != "boot"
    ]
    judged = any(v is not None and not v.unjudged for v in scores)
    ref = onboard_ref(state.slug)
    commit = _pinned(workflow, repo, ref) or commit
    written = _pin(workflow, repo, ref, commit)
    # The build the passing boot measured, from its verdict on record: not the tree's as it
    # is now (a later start that finds the ref without the state), nor what an evaluation
    # nothing judges left there after the boot.
    boot = next((c.key for c in _onboard_checks(workflow) if c.gate == "boot"), "")
    measured = status.at(boot, commit, passed=True)
    libraries = contract_mod.libraries_of(measured.tree if measured else "")
    libraries = libraries or contract_mod.library_digest(tree)
    state.onboarded = dict(
        commit=commit,
        ref=ref if written else f"{ref} (git did not write it)",
        libraries=libraries,
        time=utc_now(),
        accuracy="judged" if judged else "unjudged (boot only)",
    )
    try:
        workflow._checkpoint(state)
    except OSError as exc:
        state.onboarded = {}
        why = f"the onboard record could not be written ({type(exc).__name__}: {exc})"
        workflow.journal.event("onboarded", f"{why}; {ref} holds it for the next start")
        return why
    workflow.journal.event("onboarded", f"{commit} ({ref}); accuracy {state.onboarded['accuracy']}")
    print_message(f"[bold green]✔ onboarded at {commit[:12]}[/bold green]")
    _keep_libraries(workflow, Path(tree), libraries)
    return ""


def _keep_libraries(workflow, tree: Path, digest: str) -> None:
    """Copy the native libraries the onboard check accepted, then say what was copied.

    Always: a worktree links the checkout's build, so a later rebuild writes over the only
    copy of what was measured. Only the bytes of ``digest`` are kept: where the tree holds
    another build by now, nothing is copied and the report says incomplete. The manifest
    is written last, so a copy that was cut or failed has none. Nothing here takes the
    record back.
    """
    kept = workflow.workspace / LIBRARIES_DIRNAME
    try:
        if contract_mod.library_digest(tree) != digest:
            raise OSError(f"{tree} no longer holds the build the boot measured ({digest})")
        files = {}
        for library in contract_mod.native_libraries(tree):
            target = kept / library.relative_to(tree)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)
            files[library.relative_to(tree).as_posix()] = contract_mod.content_hash(target)
        manifest = json.dumps({"digest": digest, "files": files}, indent=1)
        write_text_atomic(kept / LIBRARIES_MANIFEST, manifest)
    except OSError as exc:
        workflow.journal.event("onboarded", f"the accepted libraries were not kept: {exc}")


def kept_libraries(workflow) -> str:
    """What the report says of the accepted libraries: kept and verified, or incomplete."""
    record, kept = workflow._state.onboarded, workflow.workspace / LIBRARIES_DIRNAME
    try:
        manifest = json.loads((kept / LIBRARIES_MANIFEST).read_text(encoding="utf-8"))
        wrong = [
            name
            for name, digest in manifest["files"].items()
            if contract_mod.content_hash(kept / name) != digest
        ]
        if manifest["digest"] != record["libraries"] or wrong:
            return f"incomplete in `{kept}`: {len(wrong)} file(s) are not what was copied"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"incomplete in `{kept}`: no manifest of a finished copy ({type(exc).__name__})"
    return f"kept and verified in `{kept}` ({len(manifest['files'])} file(s))"


def stopped_early(workflow, log=None) -> bool:
    """A start that ended before its first task: whether the run is onboarded all the same.

    The earlier report must not stand as this start's: it gets a line saying that this
    start stopped, and why.
    """
    record = workflow._state.onboarded
    if not record:
        return False
    why = (workflow._stopped_at or "it did not say why").splitlines()[0]
    note = f"- this start ({utc_now()}) stopped before its first task: {why}"
    path = workflow.workspace / REPORT_FILENAME
    old = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    rest = [line for line in old[1:] if not line.startswith("- this start (")]
    rest = rest[1:] if rest[:1] == [""] else rest
    write_text_atomic(path, "\n".join([_first_line(record), "", note, *rest]) + "\n")
    said = f"onboarded at {record['commit'][:12]} by an earlier start; this one stopped: {why}"
    workflow.journal.section("run closed", f"- {said}\n- report: `{path}`")
    print_message(f"[bold green]{escape_markup(said)}[/bold green]\nreport: {path}", log)
    return True


def _first_line(record: dict) -> str:
    unjudged = "" if record["accuracy"] == "judged" else f"; accuracy {record['accuracy']}"
    where = f"commit `{record['commit']}`, ref `{record['ref']}`, {record['time']}"
    return f"# onboarded: {where}{unjudged}"


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


def git_lines(repo: Path, trees: Sequence[str], commit: str = "") -> list[Line]:
    """Whether the delivered ``trees`` are committed: what is not, no check measured.

    Nothing outside a git work tree: a fixture in a temporary directory is no delivery. The
    commits themselves were audited as each task landed (``run/integration.py``).
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
    return [Line("git:target-committed", dirt is not None and not dirty, said, commit)]


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
    # On record like any check: a start cut after them does not run them again on this content.
    check, identity = DELIVERED_TESTS, workflow._identity(commit, repo)
    status = workflow._contract_status()
    kept = status.on(identity).get(check.key)
    if kept is not None and kept.passed:
        return Line("tests", True, f"{kept.detail} (on record for this content)", kept.commit)
    python = str((workflow.task_data.get("environment") or {}).get("python") or "python3")
    report = workflow.workspace / CHECKS_DIRNAME / commit[:12] / "delivered-tests.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    command = pytest_runner.pytest_command(
        python, repo, files, report, ("--timeout=180",), workflow.environment
    )
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
    line = Line("tests", outcome.verdict() == "CLEAN", said, commit)
    result = check_result.GateResult(check.gate, line.passed, said, deployment=check.deployment)
    status.record(check, result, tree=identity, commit=commit)
    return line


def delivery_lines(workflow, commit: str) -> list[Line]:
    """What the final pass looks at besides the deployments' checks, at ``commit``."""
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    target_rel = workflow._preflight.target_rel
    trees = (target_rel, MODELING_V2_PACKAGE, CATALOG_TESTS_DIR)
    lines: list[Line | None] = [*git_lines(repo, trees, commit)]
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
    problems = [f"check `{v.key}` failed at {commit[:12]}: {v.summary or v.detail}" for v in failed]
    problems += [
        f"{line.name} at {commit[:12]}: {line.detail}" for line in lines if not line.passed
    ]
    findings = findings_left(w)
    if problems or findings:
        said = [*problems, *(f"convention: {finding}" for finding in findings)]
        fix = pipeline.fix(said)
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


def _not_done(workspace: Path) -> list[str]:
    """What each task's last receipt left not done, by task; a line several tasks wrote, once."""

    def order(path: Path) -> tuple[str, list[int]]:
        task, _, token = path.stem.rpartition("-")
        return task, [int(x) for x in token.split(".") if x.isdigit()]

    # By task, attempt and turn: the last of a task is the latest word of its worker.
    files = sorted((workspace / DECISIONS_DIRNAME).glob("*.yaml"), key=order)
    last = {order(path)[0]: path for path in files if order(path)[0]}
    left: list[tuple[str, str]] = []
    for task, path in sorted(last.items()):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        # Whatever a worker wrote there: a list, a block of lines, a mapping, one word, a number.
        items = data.get("not_done") if isinstance(data, dict) else None
        if isinstance(items, dict):
            items = [{what: why} for what, why in items.items()]
        elif not isinstance(items, list):
            items = [x for x in str(items or "").splitlines() if x.strip()]
        for item in items:
            # `- what: why` is a mapping to YAML, and one item to its writer.
            pairs = item.items() if isinstance(item, dict) else ()
            said = "; ".join(f"{what}: {why}" for what, why in pairs) or str(item)
            left.append((task, " ".join(said.split())))
    # The same sentence from every kernel task is one line with its tasks, before the rest.
    writers: dict[str, list[str]] = {}
    for task, said in left:
        writers.setdefault(said, []).append(task)
    shared = [(tasks, said) for said, tasks in writers.items() if len(tasks) > 1]
    alone = [([task], said) for task, said in left if len(writers[said]) == 1]
    return [f"- {', '.join(f'`{t}`' for t in tasks)}: {said}" for tasks, said in shared + alone]


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
    """One table row per check of each done deployment: what it measured, against what, where.

    The verdict is the one this start used or measured, never the ledger's latest for the
    key: a check it did not run reads "not run". Core's own, which the final pass leaves
    alone, are the ones of the onboarded commit.
    """
    status, frozen = workflow._contract_status(), anchors.load(workflow.workspace)
    onboarded = str((workflow._state.onboarded or {}).get("commit") or "")
    first = _onboard_checks(workflow)
    left = {
        (r["deployment"], r["check"]) for r in getattr(workflow._preflight, "not_attempted", ())
    }
    rows, earlier = [], {}
    for name in pipeline.done_deployments():
        for check in (c for c in workflow._contract.of(name) if (name, c.gate) not in left):
            v = workflow._selected.get(check.key)
            if v is None and check in first:
                v = status.at(check.key, onboarded)
            if v is not None and v.commit:
                earlier.setdefault(v.commit[:12], []).append(check.key)
            held = frozen.get(f"{name}:{check.gate[9:-1]}") if "[" in check.gate else None
            anchor = "-"
            if held is not None:
                value = "" if held.value is None else f" {held.value:g}"
                floor = "" if v is None or v.floor is None else f", floor {v.floor:.2f}"
                anchor = f"{held.kind}{value}{floor} ({held.source})".replace("|", "/")
            score = "-" if v is None or v.measured is None else f"{v.measured:g}"
            cells = (score, anchor, v.said if v else "not run", (v.path if v else "") or "-")
            commit = (v.commit if v else "")[:12] or "-"
            attested = (v.attested if v else "").replace("|", "/") or "-"
            row = f"| `{name}` | `{check.gate}` | {' | '.join(cells)} | `{commit}` | {attested} |"
            rows.append(row)
    head = "| deployment | check | score | anchor | verdict | judged | commit | engine attested |"
    return [head, "|---" * 8 + "|", *rows, *_earlier(workflow, earlier)] if rows else []


def _earlier(workflow, measured: dict[str, list[str]]) -> list[str]:
    """One line under the table when rows were measured on another commit than the head.

    Each row names its commit, and a reader still takes the table for the head. Every
    such row is named, a score with no verdict too.
    """
    head = (workflow._head() or "")[:12]
    there = [
        f"{', '.join(f'`{key}`' for key in keys)} (on `{commit}`)"
        for commit, keys in measured.items()
        if head and commit != head
    ]
    if not there:
        return []
    return [
        "",
        f"Not measured on the head `{head}`: {'; '.join(there)}. Each row is of its own "
        "commit; a check is measured again only after a `fix` task, and then only for the "
        "first deployment and for a deployment that had a failure.",
    ]


def write_report(workflow, pipeline) -> Path:
    """``delivery_report.md``: written whenever the pipeline ended, whatever it came to."""
    w, task, record = workflow, workflow.task_data, workflow._state.onboarded
    repo = Path(str(task["trtllm_repo_path"]))
    lines = [_first_line(record)] if record else _stopped(w, pipeline)
    lines += [
        "",
        f"- checkpoint: `{task.get('checkpoint_path')}`",
        f"- target: `{getattr(w._preflight, 'target_rel', '')}`",
        f"- repository: `{repo}` at `{w._head() or '(no commit)'}`",
    ]
    for check in _onboard_checks(w) if record else ():
        v = w._selected.get(check.key) or w._contract_status().at(check.key, record["commit"])
        if v is not None and not v.passed and _unanchored(w, check):
            said = v.summary or v.detail
            lines.append(f"- `{check.key}` has no anchor, and its evaluation failed: {said}")
    if record:
        now = contract_mod.library_digest(repo)
        same = now == record["libraries"]
        said = "are the onboarded ones" if same else f"({now}) are not the onboarded ones"
        lines.append(
            f"- native libraries: the head's {said} ({record['libraries']}); the accepted "
            f"ones: {kept_libraries(w)}"
        )
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
    # With them, what a landed commit's message or author gets wrong: listed, never refused.
    rewrite = "rewrite it before the pull request (`git rebase -i`, then `git commit --amend`)"
    commits = [
        f"- commit `{row['sha'][:12]}` (`{row['task']}`): {rule}; {rewrite}"
        for row in load_labels(w.workspace)
        for rule in row.get("rules") or ()
    ]
    section("Convention findings left", [f"- {f}" for f in findings_left(w)] + commits)
    section("Final pass", [f"- {line.render()}" for line in pipeline.final])
    section("What the run did not check", [f"- {x}" for x in (*NOT_CHECKED, *w._contract.notes)])
    section("What the workers left", _not_done(w.workspace))
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
