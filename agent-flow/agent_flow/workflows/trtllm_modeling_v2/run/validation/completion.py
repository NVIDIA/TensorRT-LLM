# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivery: the gate findings, the checks on disk, the delivered tests, and the contract.

The delivered tests do not run here: they are part of the final tests' selection,
which ran once on the frozen candidate. Delivery reads their results at the
checkout's ``HEAD``, and holds only on none failing and none left unrun.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ...checks import accuracy as accuracy_checks
from ...checks import anchors as anchors_mod
from ...checks import delivery as delivery_rules
from ...checks import gate_records, test_results
from ...checks.accuracy import resolve_model_key
from ...checks.delivery import Check as DeliveryCheck
from ...checks.delivery import DeliveryReport, accuracy_test_path, verify_delivery
from ...infra.git import CommitPolicy
from ...operators.coverage import model_ops_tests_dir
from ...spec import deployments as deployments_mod
from ..records.state import WorkflowState
from . import contract as contract_mod


@dataclass(frozen=True)
class Completion:
    """One assessment for delivery."""

    evaluation: contract_mod.Evaluation
    files: tuple[str, ...] = ()
    required_open: bool = False
    unread_reports: bool = False

    @property
    def gaps(self) -> list[str]:
        if not self.evaluation.holds:
            return ["the contract does not hold"]
        gaps = []
        if self.files:
            gaps.append("delivery files: " + "; ".join(self.files))
        return gaps

    @property
    def supported(self) -> bool:
        return not self.gaps

    @property
    def ready(self) -> bool:
        return self.supported and not self.required_open and not self.unread_reports


def assess(
    evaluation: contract_mod.Evaluation,
    *,
    files: Sequence[str] = (),
    required_open: bool = False,
    unread_reports: bool = False,
) -> Completion:
    return Completion(evaluation, tuple(files), required_open, unread_reports)


def run_deliver_phase(
    workflow, state: WorkflowState, log, assessment: Callable[[Sequence[str]], Completion]
) -> bool:
    """Read the delivered tests' results, then the same completion policy on the final tree."""
    print_rule("[bold magenta]deliver[/bold magenta]", log)
    report = delivery_report(workflow, final=True)
    workflow.journal.section("delivery checks", f"```\n{report.render()}\n```")
    print_message(report.render(), log)
    result = assessment([f"{c.name}: {c.detail}" for c in report.failures])
    workflow.journal.section("contract", result.evaluation.render())
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
#: runs it and its anchor's row. A run none of whose deployments lists a benchmark owes none.
CI_DELIVERY_CHECKS = frozenset({"accuracy:reference", "tests:accuracy-file", "tests:test-db"})
#: The most test modules of its own a target delivers that a reviewer accepts (10-15,
#: consolidated by domain, is the aim): delivery warns past it and never fails on it.
TARGET_TEST_MODULES_MAX = 20


def ci_waiver(task_data) -> str:
    """Why the run owes no :data:`CI_DELIVERY_CHECKS`; ``""`` when it owes them."""
    if any(d.benchmarks for d in deployments_mod.resolve(task_data)):
        return ""
    return "not owed: no deployment of the spec lists a benchmark"


def reference_yaml(task_data) -> str:
    """The reference file the run's anchor row is delivered in: the spec's, else the tree's for its benchmark."""
    accuracy = task_data.get("accuracy") or {}
    stated = accuracy.get("reference_yaml")
    if stated:
        return str(stated)
    benchmark = accuracy.get("benchmark") or deployments_mod.DEFAULT_BENCHMARK
    return f"{anchors_mod.REFERENCES_DIR}/{benchmark}.yaml"


def declared_anchor(task_data, workspace=None) -> str:
    """What the spec's declared anchors amount to as entries of the reference file, in words.

    One entry per deployment that declares an anchor on the run's benchmark: the tags
    its options imply, the score as the run froze it, and the count and the source to
    keep as a comment. ``""`` when the spec declares none: nobody then knows the row.
    """
    name = str(
        (task_data.get("accuracy") or {}).get("benchmark") or deployments_mod.DEFAULT_BENCHMARK
    )
    try:
        frozen = anchors_mod.load(workspace) if workspace is not None else {}
    except anchors_mod.AnchorError:
        frozen = {}
    entries = []
    for deployment in deployments_mod.resolve(task_data):
        for benchmark in deployment.benchmarks:
            if benchmark.name != name:
                continue
            anchor = frozen.get(f"{deployment.name}:{name}")
            if anchor is None and benchmark.anchor is not None:
                anchor = anchors_mod.resolve(task_data, deployment, benchmark)
            if anchor is None or anchor.kind != anchors_mod.DECLARED or anchor.value is None:
                continue
            tags = anchors_mod.tags(deployment, benchmark)
            row = ", ".join(
                [*(f"{k}: {v}" for k, v in tags.items()), f"accuracy: {anchor.value:g}"]
            )
            count = f"{anchor.samples} samples" if anchor.samples else "the count its source states"
            source = " ".join(str(anchor.source or "").split()) or "the spec states none"
            entries.append(
                f"`{row}` for `{deployment.name}`, with a comment above it that keeps the "
                f"count ({count}) and the source ({source})"
            )
    if not entries:
        return ""
    return (
        f"add under the key `{resolve_model_key(task_data)}` the "
        + ("entry " if len(entries) == 1 else "entries ")
        + "; ".join(entries)
    )


def ci_arguments(workflow) -> dict:
    """What the delivery's rules for CI are asked with (``delivery.check_ci``).

    One place: the delivery's report and the task that writes those files ask the
    same rules with the same words.
    """
    task_data = workflow.task_data
    return dict(
        reference_yaml=reference_yaml(task_data),
        model_key=resolve_model_key(task_data),
        accuracy_test=accuracy_test_path(
            task_data["trtllm_repo_path"],
            workflow._preflight.family,
            Path(workflow._preflight.target_rel).name,
        ),
        declared_anchor=declared_anchor(task_data, workflow.workspace),
    )


def ci_rules(workflow, root: str | Path) -> list[DeliveryCheck]:
    """The delivery's rules for CI asked of the tree at ``root``; none for a run that owes none."""
    if ci_waiver(workflow.task_data):
        return []
    return delivery_rules.check_ci(Path(root), **ci_arguments(workflow))


def delivery_report(workflow, *, final: bool = False) -> DeliveryReport:
    """The checks on disk and the delivered tests.

    Cheap, and safe to repeat: nothing runs. ``final`` holds the delivered tests to
    their results at ``HEAD`` (:func:`tests_check`).
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
        **ci_arguments(workflow),
        commit_policy=CommitPolicy.from_task(workflow.task_data, workspace=workflow.workspace),
        v1_imports=delivery_rules.v1_imports_refused(workflow.task_data),
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
    tested = tests_check(workflow, report, final=final)
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


def delivered_test_paths(workflow, report: DeliveryReport) -> list[str]:
    """Where the delivered tests are: the target's own, and the catalog entries' it is delivered with.

    The report's ``test:`` checks name the second. The final tests' selection takes
    these, so that delivery finds their results at ``HEAD``.
    """
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    paths = [model_ops_tests_dir(workflow._preflight.target_rel)]
    for check in report.checks:
        if check.name.startswith("test:") and check.passed:
            try:
                paths.append(Path(check.detail).resolve().relative_to(repo.resolve()).as_posix())
            except (OSError, ValueError):
                continue
    return paths


def tests_check(workflow, report: DeliveryReport, *, final: bool = False) -> DeliveryCheck | None:
    """The delivered tests pass at the checkout's HEAD.

    The target's own tests, and the catalog entries' tests it is delivered with
    (:func:`delivered_test_paths`), each by its result at ``HEAD``
    (``checks/test_results.py``). Nothing runs here: the final tests ran them, once,
    on the frozen candidate. With ``final``, the check holds only on none failing,
    none left unrun and one at least passed. Without, it fails only on a test that
    fails there. ``None`` when there is no such test, or nothing to say before the
    final tests ran.

    A delivered test of the engine's accuracy harness has no result at ``HEAD`` on
    purpose: the final tests collected it and left it to CI
    (``test_results.accuracy_tests``). It is not waited for here, unless the spec had
    them run.
    """
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    head = (workflow._git(str(repo), "rev-parse", "HEAD") or "").strip()
    paths = delivered_test_paths(workflow, report)
    files = test_results.list_test_files(repo, head, paths) if head else []
    left = test_results.accuracy_tests(repo, head, files) if files else {}
    files = [file for file in files if file not in left]
    if not files:
        return None
    # Results of another commit say nothing of this one: only those recorded at HEAD count.
    results = test_results.recorded(workflow.workspace / test_results.CACHE_DIRNAME, head)
    found = test_results.head_results(results, files)
    if not final and found.verdict != "FAILING":
        return None
    said = found.describe() + (f"; as the final tests left them at {head[:10]}" if final else "")
    if left:
        said += f"; {len(left)} accuracy test file(s) of the engine's harness left to CI"
    return DeliveryCheck("tests", found.verdict == "CLEAN", said)
