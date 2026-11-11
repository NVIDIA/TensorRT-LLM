# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The final certification: every required check and the delivery checks, on one commit.

While the workers build, nothing here runs: their conclusions are trusted.
Once the pipeline has no task left, :func:`certify` runs the whole contract
on the commit the checkout is at, then the delivery checks (the files on
disk, the commit audit, the delivered tests, the CI test lists). The run is
delivered only when all of it passed on that one commit, with the checkout
clean and unmoved while it ran.

One rule decides what counts: a verdict counts for the tree it measured
(``contract.tree_id``: the commit and the native libraries built into the
checkout) and for no other. A round whose tree changed under it certifies
nothing. Nothing is shared between checks and
nothing is carried from an earlier commit. A round interrupted by a restart
goes on from the verdicts it already recorded for the same tree.

A round after a repair runs the checks that failed before first, and stops as
soon as one of them fails again: the commit cannot be certified, and the rest
would be paid for nothing.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ..checks import accuracy_policy, gate_cli, test_results
from ..checks import verdict as check_result
from ..checks.accuracy import resolve_model_key
from ..checks.delivery import Check as DeliveryCheck
from ..checks.delivery import DeliveryReport, accuracy_test_path, verify_delivery
from ..infra.compute.commands import CommandSpec
from ..infra.files import write_text_atomic
from ..operators.coverage import model_ops_tests_dir
from ..spec import variants as variants_mod
from ..spec.task_schema import (
    PROFILE_BRINGUP,
    PROFILE_SMOKE,
    assembly_units,
    profile_of,
    resolve_per_unit,
)
from . import contract as contract_mod
from . import requirements as requirements_mod

#: Under the workspace: one directory per certification round, with its checks' files.
CERTIFY_DIRNAME = "certify"
#: The round's result, rewritten by each round; the delivered run's is the certificate.
CERTIFICATION_FILENAME = "certification.yaml"
#: The delivery report: the verdict, the commit, every check, what is covered.
REPORT_FILENAME = "delivery_report.md"
#: The delivery checks that set the target up for CI: its accuracy test, the test list that
#: runs it and its anchor's row. A run whose every unit is ``smoke`` or ``bringup`` owes none.
CI_DELIVERY_CHECKS = frozenset({"accuracy:reference", "tests:accuracy-file", "tests:test-db"})
#: The most test modules of its own a target delivers that a reviewer accepts: delivery
#: warns past it and never fails on it.
TARGET_TEST_MODULES_MAX = 20
#: How many times a check the infrastructure ended (no verdict) runs again in one round.
INFRASTRUCTURE_RERUNS = 1


@dataclass(frozen=True)
class Certification:
    """One round's result: what was measured, on which tree, and whether it certifies it."""

    round: int
    commit: str
    tree: str
    evaluation: contract_mod.Evaluation
    #: ``None`` when the round did not reach the delivery checks.
    delivery: DeliveryReport | None = None
    #: What fails the round besides a check: a dirty or moved checkout, a shift finding.
    problems: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            bool(self.tree)
            and self.evaluation.holds
            and not self.problems
            and self.delivery is not None
            and self.delivery.passed
        )

    @property
    def failed_keys(self) -> list[str]:
        """The checks that failed this round: the next round runs them first."""
        return [check.key for check, _ in self.evaluation.failed]

    def failures(self) -> list[str]:
        """Everything that keeps the tree from being certified, one line each."""
        lines = [f"the checkout: {p}" for p in self.problems]
        lines += [
            f"check {c.key} FAILED: {v.summary or v.detail}" for c, v in self.evaluation.failed
        ]
        lines += [f"check {c.key} has no verdict on this commit" for c in self.evaluation.missing]
        if self.delivery is not None:
            lines += [f"delivery {c.name}: {c.detail}" for c in self.delivery.failures]
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "certified": self.passed,
            "round": self.round,
            "commit": self.commit,
            "tree": self.tree,
            "checks": {
                "passed": [c.key for c in self.evaluation.passed],
                "failed": {c.key: v.detail for c, v in self.evaluation.failed},
                "missing": [c.key for c in self.evaluation.missing],
            },
            "delivery": (
                [c.render() for c in self.delivery.checks] if self.delivery is not None else None
            ),
            "problems": list(self.problems),
        }


def round_dir(workspace: Path, round: int) -> Path:
    return Path(workspace) / CERTIFY_DIRNAME / f"round-{round}"


def _dirty(workflow, repo: str) -> str:
    """What the checkout holds uncommitted in tracked files, as a problem line; "" when clean."""
    dirt = workflow._git(repo, "status", "--porcelain", "--untracked-files=no")
    if dirt is None:
        return "git could not say whether it is clean"
    names = [line[3:] for line in dirt.splitlines() if len(line) > 3]
    if not names:
        return ""
    shown = ", ".join(names[:8]) + (f" and {len(names) - 8} more" if len(names) > 8 else "")
    return f"uncommitted changes to tracked files ({shown}): no check measures them"


def _run_check(workflow, check: contract_mod.Check, round: int, anchor: float | None) -> Any:
    """One contract check on the checkout as it is, through the shared :func:`gate_cli.run_check`.

    A check that cannot be run at all (a reference it cannot read, a file it
    cannot write) is a failed check with why, never an error of the run: a run
    that raised here would raise again at the same check on every restart.
    """
    try:
        return _measure(workflow, check, round, anchor)
    except Exception as exc:
        return check_result.GateResult(
            name=check.gate,
            passed=False,
            member=check.member,
            detail=f"the check could not be run: {type(exc).__name__}: {exc}",
        )


def _measure(workflow, check: contract_mod.Check, round: int, anchor: float | None) -> Any:
    directory = round_dir(workflow.workspace, round)
    return gate_cli.run_check(
        workflow.task_data,
        check.gate,
        check.unit,
        check.member,
        workdir=directory / "gates",
        tree=workflow.task_data.get("trtllm_repo_path"),
        workspace=workflow.workspace,
        runner=workflow._shell_runner_override,
        devices=tuple(workflow._state.devices) or None,
        anchor_score=anchor,
    )


def _relative_anchor(
    workflow, check: contract_mod.Check, have: dict[str, contract_mod.Verdict]
) -> float | None:
    """The score a relative variant's benchmark is held to: its unit's, this round, same member."""
    task = workflow.task_data
    if not check.gate.startswith("accuracy[") or not variants_mod.is_variant(task, check.unit):
        return None
    if variants_mod.anchor_state(task, check.unit) != variants_mod.ANCHOR_RELATIVE:
        return None
    base = have.get(f"{variants_mod.base_unit(task, check.unit)}:{check.gate}@{check.member}")
    return base.measured if base is not None else None


def shift_findings(task_data, verdicts: dict[str, contract_mod.Verdict]) -> list[str]:
    """Units whose every benchmarked member sits more than one SE under its reference.

    Each member can clear its own floor while all of them are low: a systematic
    shift. Computed from this round's verdicts alone; one ``task.yaml``
    acknowledges (``accuracy.floor.acknowledge``) does not fail the round.
    """
    policy = accuracy_policy.resolve_floor_policy(task_data.get("accuracy") or {})
    by_unit: dict[str, dict[str, tuple[float, float, float]]] = {}
    for verdict in verdicts.values():
        if (
            verdict.gate.startswith("accuracy[")
            and verdict.measured is not None
            and verdict.reference is not None
            and verdict.se is not None
        ):
            by_unit.setdefault(verdict.unit, {})[verdict.member] = (
                verdict.measured,
                verdict.reference,
                verdict.se,
            )
    found = []
    for unit, readings in by_unit.items():
        finding = accuracy_policy.systematic_shift(
            unit,
            readings,
            min_members=policy.shift_min_members,
            acknowledged=policy.acknowledged,
        )
        if finding is not None and not finding.acknowledged:
            found.append(f"accuracy finding: {finding.describe()}")
    return found


def certify(workflow, round: int, first: Sequence[str] = (), log=None) -> Certification:
    """Run the contract and the delivery checks on the checkout's commit; see the module docstring.

    ``first`` are the check keys that failed the round before: they run first,
    and the round stops when one of them fails again.
    """
    print_rule(f"[bold magenta]certification, round {round}[/bold magenta]", log)
    contract = workflow._contract
    repo = str(workflow.task_data.get("trtllm_repo_path") or "")
    commit = workflow._head()
    tree = contract_mod.tree_id(repo, commit)
    status = workflow._contract_status()

    def result(delivery=None, problems: Sequence[str] = ()) -> Certification:
        evaluation = contract_mod.evaluate(contract, status.on(tree, round))
        certification = Certification(round, commit, tree, evaluation, delivery, tuple(problems))
        _record(workflow, certification)
        return certification

    if not tree:
        return result(problems=[f"git could not name the commit {repo} is at"])
    dirty = _dirty(workflow, repo)
    if dirty:
        return result(problems=[dirty])
    workflow.journal.event("certification", f"round {round} on {commit[:12]} ({tree})")
    again = set(first)
    checks = sorted(contract.required(), key=lambda c: c.key not in again)
    for check in checks:
        have = status.on(tree, round)
        if check.key in have:
            continue  # recorded before a restart, for this same tree and round
        outcome = None
        for _ in range(1 + INFRASTRUCTURE_RERUNS):
            outcome = _run_check(workflow, check, round, _relative_anchor(workflow, check, have))
            if not getattr(outcome, "infrastructure", ""):
                break
            workflow.journal.event(
                "check without a verdict", f"{check.key}: {outcome.infrastructure}"
            )
        verdict = status.record(check, outcome, tree=tree, round=round)
        workflow.gate_log.record(outcome)
        workflow.journal.event("check", f"{check.unit}: {outcome.summary()}")
        colour = "green" if verdict.passed else "yellow"
        print_message(
            f"[bold {colour}]{check.unit}: {escape_markup(outcome.summary())}[/bold {colour}]",
            log,
        )
        if contract_mod.tree_id(repo, workflow._head()) != tree:
            return result(
                problems=["its commit or its native libraries changed while the checks ran"]
            )
        if not verdict.passed and check.key in again:
            workflow.journal.event(
                "certification",
                f"round {round} stops: {check.key} failed again; the rest is not run",
            )
            return result()
    verdicts = status.on(tree, round)
    evaluation = contract_mod.evaluate(contract, verdicts)
    problems = shift_findings(workflow.task_data, verdicts)
    if not evaluation.holds:
        return result(problems=problems)
    try:
        delivery = delivery_report(workflow, commit, round)
    except Exception as exc:  # as for a check: a failed delivery check, not an error of the run
        delivery = DeliveryReport(
            checks=(
                DeliveryCheck("delivery", False, f"could not be run: {type(exc).__name__}: {exc}"),
            )
        )
    if contract_mod.tree_id(repo, workflow._head()) != tree or _dirty(workflow, repo):
        problems.append("it changed while the delivery checks ran")
    return result(delivery, problems)


def _record(workflow, certification: Certification) -> None:
    """Write the round's result where the planner, the report and the operator read it."""
    write_text_atomic(
        workflow.workspace / CERTIFICATION_FILENAME,
        yaml.safe_dump(certification.as_dict(), sort_keys=False, width=100),
    )
    body = [certification.evaluation.render()]
    if certification.delivery is not None:
        body += ["", "Delivery checks:", "", "```", certification.delivery.render(), "```"]
    body += [f"- {problem}" for problem in certification.problems]
    verdict = "CERTIFIED" if certification.passed else "NOT CERTIFIED"
    workflow.journal.section(
        f"certification round {certification.round}: {verdict} "
        f"{certification.commit[:12] or '(no commit)'}",
        "\n".join(body),
    )


# ------------------------------------------------------------------------ delivery checks


def ci_waiver(task_data) -> str:
    """Why the run owes no :data:`CI_DELIVERY_CHECKS`; ``""`` when it owes them."""
    profiles = {profile_of(task_data, unit) for unit in assembly_units(task_data) or [None]}
    if not profiles or not profiles <= {PROFILE_SMOKE, PROFILE_BRINGUP}:
        return ""
    return f"not owed by the {'/'.join(sorted(profiles))} profile"


def delivery_report(workflow, commit: str, round: int = 0) -> DeliveryReport:
    """The delivery checks on the checkout at ``commit``: files, commits, test lists, tests.

    The delivered tests run here (:func:`tests_check`), with no agent turn.
    """
    task = workflow.task_data
    repo = task["trtllm_repo_path"]
    target_rel = workflow._preflight.target_rel
    report = verify_delivery(
        repo_root=repo,
        target_rel=target_rel,
        family=workflow._preflight.family,
        reference_yaml=resolve_per_unit(
            task["accuracy"], "reference_yaml", (assembly_units(task) or [None])[-1]
        ),
        model_key=resolve_model_key(task),
        accuracy_test=accuracy_test_path(repo, workflow._preflight.family, Path(target_rel).name),
        arch=str(workflow.task_data.get("arch") or ""),
        commit_policy=workflow._commit_policy(),
    )
    checks = list(report.checks)
    waiver = ci_waiver(task)
    if waiver:
        checks = [
            replace(c, passed=True, waived=waiver)
            if c.name in CI_DELIVERY_CHECKS and not c.passed
            else c
            for c in checks
        ]
    modules = target_test_modules_check(repo, target_rel)
    if modules is not None:
        checks.append(modules)
    tested = tests_check(workflow, checks, commit, round)
    if tested is not None:
        checks.append(tested)
    return DeliveryReport(checks=tuple(checks))


def static_delivery_failures(workflow, tree: str | Path) -> list[str]:
    """The delivery checks that read files, run on ``tree`` (a task's worktree): what fails.

    The same checks the certification runs, less the commit audit (integration audits a
    task's commits) and the delivered tests (they run at the certification); a CI waiver
    applies as it does there.
    """
    task = workflow.task_data
    target_rel = workflow._preflight.target_rel
    family = workflow._preflight.family
    report = verify_delivery(
        repo_root=tree,
        target_rel=target_rel,
        family=family,
        reference_yaml=resolve_per_unit(
            task["accuracy"], "reference_yaml", (assembly_units(task) or [None])[-1]
        ),
        model_key=resolve_model_key(task),
        accuracy_test=accuracy_test_path(tree, family, Path(target_rel).name),
        arch=str(task.get("arch") or ""),
    )
    waived = CI_DELIVERY_CHECKS if ci_waiver(task) else ()
    return [f"{c.name}: {c.detail}" for c in report.checks if not c.passed and c.name not in waived]


def target_test_modules_check(repo_root: str | Path, target_rel: str) -> DeliveryCheck | None:
    """How many test modules the target delivers under its tests' directory, and their lines.

    Never fails delivery: past :data:`TARGET_TEST_MODULES_MAX` it says so as a
    warning. ``None`` when the target has no test there.
    """
    tests_dir = model_ops_tests_dir(target_rel)
    root = Path(repo_root) / tests_dir
    files = (
        sorted(p for p in root.rglob("*.py") if test_results.TEST_FILE.search(p.as_posix()))
        if root.is_dir()
        else []
    )
    if not files:
        return None
    lines = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in files)
    detail = f"{len(files)} test module(s), {lines} lines, under {tests_dir}"
    if len(files) > TARGET_TEST_MODULES_MAX:
        detail += (
            f"; WARNING: more than the {TARGET_TEST_MODULES_MAX} a reviewer accepts: consolidate "
            "them by domain to 10-15 modules, with one copy of each helper"
        )
    return DeliveryCheck("tests:modules", True, detail)


def tests_check(
    workflow, checks: Sequence[DeliveryCheck], commit: str, round: int = 0
) -> DeliveryCheck | None:
    """The delivered tests pass at ``commit``, run now in the checkout.

    The target's own tests, and the catalog entries' tests it is delivered
    with (the report's ``test:`` checks). The check holds only on none failing,
    none left unrun and one at least passed. ``None`` when there is no such test.
    """
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    paths = [model_ops_tests_dir(workflow._preflight.target_rel)]
    for check in checks:
        if check.name.startswith("test:") and check.passed:
            try:
                paths.append(Path(check.detail).resolve().relative_to(repo.resolve()).as_posix())
            except (OSError, ValueError):
                continue
    files = test_results.list_test_files(repo, commit, paths) if commit else []
    if not files:
        return None
    python = str((workflow.task_data.get("environment") or {}).get("python") or "python3")
    results, jobs, code, tail = test_results.run_at(
        commit,
        files,
        cache_dir=workflow.workspace / test_results.CACHE_DIRNAME,
        out=round_dir(workflow.workspace, round) / "tests",
        tree=repo,
        run_job=lambda command, tree, timeout_s: _test_job(workflow, command, tree, timeout_s),
        python=python,
    )
    if not results.env_digest:
        return DeliveryCheck("tests", False, f"INCOMPLETE (exit {code}): {tail}")
    ran = f"; {jobs} test job(s) ran at {commit[:10]}" if jobs else ""
    found = test_results.head_results(results, files)
    return DeliveryCheck("tests", found.verdict == "CLEAN", found.describe() + ran)


def test_job_line(command: str, tree: Path) -> str:
    """``command`` run from inside ``tree`` with the tree first on ``PYTHONPATH``, as the gates run.

    pytest imports the tree's ``tensorrt_llm`` from its working directory, but a test that
    starts rank processes runs them as scripts, which resolve ``tensorrt_llm`` from
    ``PYTHONPATH`` and the installed package: without the tree there, they import another
    checkout (the editable install's) and miss the target's modules.
    """
    root = shlex.quote(str(Path(tree).resolve()))
    return f'cd {root} && export PYTHONPATH={root}"${{PYTHONPATH:+:$PYTHONPATH}}" && {command}'


def _test_job(workflow, command: str, tree: Path, timeout_s: int) -> tuple[int, str]:
    """One test job in ``tree`` on this run's executor; its exit code and output tail."""
    line = test_job_line(command, tree)
    if workflow._shell_runner_override is not None:
        done = workflow._shell_runner(line, None, timeout_s)
    else:
        done = workflow._command_executor.execute(
            CommandSpec(argv=line, cwd=str(tree), timeout_s=timeout_s)
        )
    return done.returncode, (done.stdout + done.stderr)[-2000:]


# ------------------------------------------------------------------------------ the report


def uncertified_lines(workspace: Path) -> list[str]:
    """Each task's `uncertified` list from its last decision file, one line per item.

    A worker states there what it measured and left out of an entry's `CELLS` or what a
    unit calls outside them; the report is where the reader finds it.
    """
    last: dict[str, tuple[tuple[int, int], Path]] = {}
    for path in sorted((workspace / "decisions").glob("*.yaml")):
        task, _, token = path.stem.rpartition("-")
        attempt, _, turn = token.partition(".")
        if not task or not attempt.isdigit() or task == "planner":
            continue
        key = (int(attempt), int(turn) if turn.isdigit() else 1)
        if task not in last or key > last[task][0]:
            last[task] = (key, path)
    lines: list[str] = []
    for task, (_, path) in sorted(last.items()):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        items = data.get("uncertified") if isinstance(data, dict) else None
        if isinstance(items, str):
            items = [line for line in items.splitlines() if line.strip()]
        for item in items or ():
            lines.append(f"- `{task}`: {str(item).strip()}")
    return lines


def write_report(workflow, verdict: str, certification: Certification | None, tasks: str) -> Path:
    """``delivery_report.md``: the run's verdict, what was certified on which commit, and how.

    Written when the run closes, whatever its verdict: a run that did not
    deliver says here what kept it from it.
    """
    lines = [f"# Delivery report: {verdict}", ""]
    task = workflow.task_data
    lines += [
        f"- checkpoint: `{task.get('checkpoint_path')}`",
        f"- target: `{getattr(workflow._preflight, 'target_rel', '')}`",
        f"- repository: `{task.get('trtllm_repo_path')}`",
    ]
    if certification is None:
        lines += ["", "No certification ran: the pipeline did not reach it."]
    else:
        lines += [
            f"- commit: `{certification.commit or '(none)'}`",
            f"- certification round: {certification.round}",
            "",
            "## Contract",
            "",
            certification.evaluation.render(),
            "",
            "| check | verdict | summary |",
            "|---|---|---|",
        ]
        verdicts = workflow._contract_status().on(certification.tree, certification.round)
        for check in workflow._contract.required():
            v = verdicts.get(check.key)
            state = "missing" if v is None else "PASS" if v.passed else "FAIL"
            summary = (v.summary or v.detail).replace("|", "\\|") if v is not None else ""
            lines.append(f"| `{check.key}` | {state} | {summary} |")
        if certification.delivery is not None:
            lines += ["", "## Delivery checks", "", "```", certification.delivery.render(), "```"]
        if not certification.passed:
            lines += ["", "## What keeps it from delivery", ""]
            lines += [f"- {line}" for line in certification.failures()]
    if workflow._contract is not None:
        matrix = requirements_mod.requirements_matrix(
            task, workflow._contract, workflow._gate_env().world_size
        )
        lines += [
            "",
            "## Coverage: what the contract proves, member by member",
            "",
            matrix.render(),
        ]
    left = uncertified_lines(workflow.workspace)
    if left:
        lines += ["", "## What the workers left uncertified", "", *left]
    lines += ["", "## Tasks", "", tasks or "(none ran)", ""]
    path = workflow.workspace / REPORT_FILENAME
    write_text_atomic(path, "\n".join(lines))
    return path


__all__ = [
    "CERTIFICATION_FILENAME",
    "CERTIFY_DIRNAME",
    "CI_DELIVERY_CHECKS",
    "REPORT_FILENAME",
    "Certification",
    "certify",
    "ci_waiver",
    "delivery_report",
    "shift_findings",
    "target_test_modules_check",
    "tests_check",
    "write_report",
]
