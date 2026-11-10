# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivery: the gate findings, the checks on disk, the delivered tests, and the contract.

The delivered tests run here, at the checkout's ``HEAD``, as one job with no
agent turn. The contract's journal section also says where correctness stands
delivered (``contract.Milestone``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ...checks import accuracy as accuracy_checks
from ...checks import gate_records, test_results
from ...checks.accuracy import resolve_model_key
from ...checks.delivery import Check as DeliveryCheck
from ...checks.delivery import DeliveryReport, accuracy_test_path, verify_delivery
from ...infra.git import CommitPolicy
from ...operators.coverage import model_ops_tests_dir
from ...spec.task_schema import (
    PROFILE_BRINGUP,
    PROFILE_SMOKE,
    assembly_units,
    profile_of,
    resolve_per_unit,
    widest_spanning_width,
)
from ..records.state import WorkflowState
from . import contract as contract_mod
from . import regression as regression_mod
from . import requirements as requirements_mod


@dataclass(frozen=True)
class Completion:
    """One assessment for milestones, campaign wrap-up, and final delivery."""

    evaluation: contract_mod.Evaluation
    files: tuple[str, ...] = ()
    unreviewed: tuple[str, ...] = ()
    required_open: bool = False
    unread_reports: bool = False

    @property
    def gaps(self) -> list[str]:
        if not self.evaluation.holds:
            return ["the contract does not hold"]
        gaps = []
        if self.unreviewed:
            gaps.append(f"no regression check covers {', '.join(self.unreviewed)}")
        if self.files:
            gaps.append("delivery files: " + "; ".join(self.files))
        return gaps

    @property
    def supported(self) -> bool:
        return not self.gaps

    @property
    def ready(self) -> bool:
        return self.supported and not self.required_open and not self.unread_reports

    @property
    def files_only(self) -> bool:
        return (
            self.evaluation.holds
            and bool(self.files)
            and not (self.unreviewed or self.required_open or self.unread_reports)
        )


def assess(
    evaluation: contract_mod.Evaluation,
    *,
    files: Sequence[str] = (),
    unreviewed: Sequence[str] = (),
    required_open: bool = False,
    unread_reports: bool = False,
) -> Completion:
    return Completion(evaluation, tuple(files), tuple(unreviewed), required_open, unread_reports)


def run_deliver_phase(
    workflow, state: WorkflowState, log, assessment: Callable[[Sequence[str]], Completion]
) -> bool:
    """Run delivered tests, then re-read the same completion policy on the final tree."""
    print_rule("[bold magenta]deliver[/bold magenta]", log)
    report = delivery_report(workflow, run_tests=True)
    workflow.journal.section("delivery checks", f"```\n{report.render()}\n```")
    matrix = requirements_mod.requirements_matrix(
        workflow.task_data, workflow._contract, workflow._gate_env().world_size
    )
    workflow.journal.section("coverage: what is delivered, member by member", matrix.render())
    print_message(report.render(), log)
    result = assessment([f"{c.name}: {c.detail}" for c in report.failures])
    marker = workflow._target_marker()
    workflow.journal.section(
        "contract", contract_mod.standing(result.evaluation, workflow.workspace, marker)
    )
    if result.ready:
        print_message(f"[bold green]✔ contract: {result.evaluation.render()}[/bold green]", log)
        return True
    detail = "\n".join(result.gaps) or "required work or unread worker reports remain"
    workflow.journal.section("delivery blocked", detail)
    print_message(f"[bold red]✘ delivery blocked:\n{escape_markup(detail)}[/bold red]", log)
    verdict = (
        "delivery-failed:findings"
        if any(c.name.startswith("finding:") for c in report.failures)
        else "delivery-failed:contract"
        if not result.evaluation.holds
        else "delivery-failed"
    )
    workflow._close_run(state, log, verdict=verdict)
    return False


def delivery_failures(workflow) -> list[str]:
    """What fails of the delivery checks on disk and the gate findings, one line each.

    As cheap as reading files: the execution stage reads it on every state of the
    target the contract holds on, so full support includes it.
    """
    try:
        report = delivery_report(workflow)
    except (OSError, ValueError) as exc:
        return [f"the delivery checks could not be read: {exc}"]
    return [f"{c.name}: {c.detail}" for c in report.failures]


#: The delivery checks that set the target up for CI: its accuracy test, the test list that
#: runs it and its anchor's row. A run whose every unit is ``smoke`` or ``bringup`` owes none.
CI_DELIVERY_CHECKS = frozenset(
    {"accuracy:reference", "tests:accuracy-file", "tests:test-db", "tests:test-db-width"}
)
_NO_CI_PROFILES = (PROFILE_SMOKE, PROFILE_BRINGUP)
#: The most test modules of its own a target delivers that a reviewer accepts (10-15,
#: consolidated by domain, is the aim): delivery warns past it and never fails on it.
TARGET_TEST_MODULES_MAX = 20


def ci_waiver(task_data) -> str:
    """Why the run owes no :data:`CI_DELIVERY_CHECKS`; ``""`` when it owes them."""
    profiles = {profile_of(task_data, unit) for unit in assembly_units(task_data) or [None]}
    if not profiles or not profiles <= set(_NO_CI_PROFILES):
        return ""
    return f"not owed by the {'/'.join(sorted(profiles))} profile"


def delivery_report(workflow, *, run_tests: bool = False) -> DeliveryReport:
    """The checks on disk and the delivered tests.

    Cheap, and safe to repeat, unless ``run_tests`` has the delivered tests
    run first (:func:`tests_check`).
    """
    findings = tuple(
        DeliveryCheck(
            f"finding:{finding.key}", False, str(finding.values.get("detail", finding.key))
        )
        for finding in accuracy_checks.blocking_findings(
            gate_records.records_for(workflow.workspace), workflow.task_data.get("accuracy") or {}
        )
    )
    if findings:
        return DeliveryReport(checks=findings)
    report = verify_delivery(
        repo_root=workflow.task_data["trtllm_repo_path"],
        target_rel=workflow._preflight.target_rel,
        family=workflow._preflight.family,
        reference_yaml=resolve_per_unit(
            workflow.task_data["accuracy"],
            "reference_yaml",
            (assembly_units(workflow.task_data) or [None])[-1],
        ),
        model_key=resolve_model_key(workflow.task_data),
        accuracy_test=accuracy_test_path(
            workflow.task_data["trtllm_repo_path"],
            workflow._preflight.family,
            Path(workflow._preflight.target_rel).name,
        ),
        commit_policy=CommitPolicy.from_task(workflow.task_data, workspace=workflow.workspace),
        engine_gpus=widest_spanning_width(workflow.task_data),
    )
    waiver = ci_waiver(workflow.task_data)
    if waiver:
        report = DeliveryReport(
            checks=tuple(
                replace(c, passed=True, waived=waiver)
                if c.name in CI_DELIVERY_CHECKS and not c.passed
                else c
                for c in report.checks
            )
        )
    modules = target_test_modules_check(
        workflow.task_data["trtllm_repo_path"], workflow._preflight.target_rel
    )
    if modules is not None:
        report = DeliveryReport(checks=(*report.checks, modules))
    tested = tests_check(workflow, report, run=run_tests)
    if run_tests:
        report = delivery_report(workflow)
        report = DeliveryReport(checks=tuple(c for c in report.checks if c.name != "tests"))
    return DeliveryReport(checks=(*report.checks, *((tested,) if tested else ())))


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


def tests_check(workflow, report: DeliveryReport, *, run: bool = False) -> DeliveryCheck | None:
    """The delivered tests pass at the checkout's HEAD.

    The target's own tests, and the catalog entries' tests it is delivered
    with (the report's ``test:`` checks), each by its result at ``HEAD``
    (``checks/test_results.py``). With ``run``, those with no result there run
    first (``regression.record_at``), and the check holds only on none
    failing, none left unrun and one at least passed. Without, it reads
    what is recorded and fails only on a test that fails there: a test
    that has not run at ``HEAD`` yet runs at delivery. ``None`` when there
    is no such test, or nothing to say without running it.
    """
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    paths = [model_ops_tests_dir(workflow._preflight.target_rel)]
    for check in report.checks:
        if check.name.startswith("test:") and check.passed:
            try:
                paths.append(Path(check.detail).resolve().relative_to(repo.resolve()).as_posix())
            except (OSError, ValueError):
                continue
    head = (workflow._git(str(repo), "rev-parse", "HEAD") or "").strip()
    files = test_results.list_test_files(repo, head, paths) if head else []
    if not files:
        return None
    cache_dir = workflow.workspace / test_results.CACHE_DIRNAME
    ran = ""
    if run:
        tree_at, run_job, python = regression_mod.job_runner(workflow, "delivery-tests")
        results, jobs, code, tail = regression_mod.record_at(
            head,
            files,
            cache_dir=cache_dir,
            out=workflow.workspace / regression_mod.SELECTIONS_DIRNAME / "delivery",
            tree_at=tree_at,
            run_job=run_job,
            python=python,
        )
        if (workflow._git(str(repo), "rev-parse", "HEAD") or "").strip() != head:
            return DeliveryCheck("tests", False, "HEAD changed while delivered tests ran")
        if not results.env_digest:
            return DeliveryCheck("tests", False, f"INCOMPLETE (exit {code}): {tail}")
        ran = f"; delivery ran {jobs} test job(s) at {head[:10]}" if jobs else ""
    else:
        # Historical failures can block progress; historical passes do not verify this environment.
        results = test_results.recorded(cache_dir, head)
    found = test_results.head_results(results, files)
    if not run and found.verdict != "FAILING":
        return None
    if found.verdict == "CLEAN":
        return DeliveryCheck("tests", True, found.describe() + ran)
    return DeliveryCheck("tests", False, found.describe() + ran)
