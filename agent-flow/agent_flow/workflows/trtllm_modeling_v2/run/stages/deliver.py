# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delivery: the gate findings, the checks on disk, and the contract on the target as it is.

The contract's journal section also says where correctness stands delivered
(``contract.Milestone``).
"""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ...checks import gates as gates_mod
from ...checks import testcache
from ...infra.committer import CommitPolicy
from ...onboard import promotion as promotion_mod
from ...onboard.delivery import (
    ACCURACY_TESTS_DIR,
    TEST_DB_DIR,
    DeliveryReport,
    accuracy_test_path,
    check_accuracy_reference,
    check_test_registration,
    model_ops_tests_dir,
    verify_delivery,
)
from ...onboard.delivery import Check as DeliveryCheck
from ...spec.task_schema import anchor_pending, assembly_units
from .. import contract as contract_mod
from .. import requirements as requirements_mod
from ..state import WorkflowState


class DeliverStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    def _run_deliver_phase(self, state: WorkflowState, log) -> bool:
        print_rule("[bold magenta]deliver[/bold magenta]", log)

        findings = gates_mod.blocking_findings(
            gates_mod.records_for(self.workspace), self.task_data.get("accuracy") or {}
        )
        if findings:
            detail = "\n".join(f"- {f.values.get('detail', f.key)}" for f in findings)
            self.journal.section("delivery blocked by gate findings", detail)
            print_message(f"[bold red]✘ delivery blocked:\n{escape_markup(detail)}[/bold red]", log)
            self._close_run(state, log, verdict="delivery-failed:findings")
            return False

        report = self._delivery_report(state)
        self.journal.section("delivery checks", f"```\n{report.render()}\n```")
        contract = self._contract or contract_mod.compile_contract(
            self.task_data, self._gate_env().world_size
        )
        matrix = requirements_mod.requirements_matrix(
            self.task_data, contract, self._gate_env().world_size
        )
        self.journal.section("coverage: what is delivered, member by member", matrix.render())
        print_message(report.render(), log)
        if not report.passed:
            names = ", ".join(check.name for check in report.failures)
            print_message(f"[bold red]✘ delivery verification failed: {names}[/bold red]", log)
            self._close_run(state, log, verdict="delivery-failed")
            return False

        return self._contract_holds(state, log)

    def _contract_holds(self, state: WorkflowState, log) -> bool:
        """Every required check passed on the target as it is now, or delivery is refused.

        The journal's section says where correctness was delivered too.
        """
        contract = self._contract or contract_mod.compile_contract(
            self.task_data, self._gate_env().world_size
        )
        marker = self._target_marker()
        evaluation = contract_mod.evaluate(
            contract,
            self._contract_status(),
            marker=marker,
            holds=getattr(self, "_verdict_holds", None),
        )
        self.journal.section("contract", contract_mod.standing(evaluation, self.workspace, marker))
        if evaluation.holds:
            print_message(f"[bold green]✔ contract: {evaluation.render()}[/bold green]", log)
            return True
        print_message(
            f"[bold red]✘ contract does not hold:\n{escape_markup(evaluation.render())}[/bold red]",
            log,
        )
        self._close_run(state, log, verdict="delivery-failed:contract")
        return False

    #: The id of the task the orchestrator adds to write the delivery files.
    DELIVERY_TASK_ID = "delivery-files"

    def _delivery_task(self, graph) -> dict | None:
        """The task that writes the delivery files, when they are missing and no task writes them.

        It writes the CI accuracy test, its entry on a test list and, when
        the accuracy reference lives in the repository, the reference entry,
        once the first unit's assembly is done. A graph already has one when a
        task names the accuracy test among its files; none is needed when
        those delivery checks already pass.
        """
        repo = self.task_data.get("trtllm_repo_path")
        preflight = getattr(self, "_preflight", None)
        units = assembly_units(self.task_data)
        if not repo or preflight is None or not units:
            return None
        accuracy = accuracy_test_path(repo, preflight.family)
        test = f"{ACCURACY_TESTS_DIR}/{accuracy}"
        if any(t.id == self.DELIVERY_TASK_ID or test in t.touches for t in graph.tasks):
            return None
        reference_yaml = str(
            gates_mod.resolve_per_unit(
                self.task_data.get("accuracy") or {}, "reference_yaml", units[-1]
            )
            or ""
        )
        pending = self._anchor_pending()
        present = [
            *check_test_registration(Path(repo), accuracy),
            *(
                check_accuracy_reference(Path(repo), reference_yaml, self._model_key())
                if reference_yaml and not pending
                else []
            ),
        ]
        if all(check.passed for check in present):
            return None
        core = [t.id for t in graph.tasks if t.kind == "assemble" and (t.ref or t.id) == units[0]]
        if not core:
            return None
        touches = [test, f"{TEST_DB_DIR}/"]
        reference = "" if pending else reference_yaml
        if reference and not reference.startswith("/"):
            touches.append(reference)
        if pending:
            goal = (
                f"write the delivery files: the CI accuracy test `{test}` (the gates' accuracy "
                f"tasks at the members the contract samples) and its entry on a list under "
                f"`{TEST_DB_DIR}/`; the accuracy anchor is pending, so write no reference "
                "entry and no score of this run's own as one; finished when the delivery "
                "checks `tests:accuracy-file` and `tests:test-db` pass"
            )
        else:
            goal = (
                f"write the delivery files: the CI accuracy test `{test}` (the gates' accuracy "
                f"tasks at the members the contract samples), its entry on a list under "
                f"`{TEST_DB_DIR}/`, and the accuracy reference entry for `{self._model_key()}`"
                + (f" in `{reference}`" if reference in touches else "")
                + "; finished when the delivery checks `tests:accuracy-file`, `tests:test-db` "
                "and `accuracy:reference` pass"
            )
        return {
            "id": self.DELIVERY_TASK_ID,
            "kind": "feature",
            "goal": goal,
            "touches": touches,
            "depends_on": core,
            "resources": {"gpus": 0, "est_minutes": 45},
            "priority": "required",
            "ref": units[0],
        }

    def _anchor_pending(self) -> bool:
        """Whether a unit has no written accuracy anchor yet (``accuracy.anchor``)."""
        accuracy = self.task_data.get("accuracy") or {}
        return any(anchor_pending(accuracy, unit) for unit in assembly_units(self.task_data))

    def _delivery_failures(self, state: WorkflowState) -> list[str]:
        """What fails of the delivery checks on disk and the gate findings, one line each.

        As cheap as reading files: the planned run reads it on every state of the
        target the contract holds on, so full support includes it.
        """
        findings = gates_mod.blocking_findings(
            gates_mod.records_for(self.workspace), self.task_data.get("accuracy") or {}
        )
        lines = [f"gate finding: {f.values.get('detail', f.key)}" for f in findings]
        try:
            report = self._delivery_report(state)
        except (OSError, ValueError) as exc:
            return [*lines, f"the delivery checks could not be read: {exc}"]
        return [*lines, *(f"{c.name}: {c.detail}" for c in report.failures)]

    def _delivery_report(self, state: WorkflowState):
        """The checks on disk, a local variant left unpromoted among them. Cheap, and safe to repeat."""
        report = verify_delivery(
            repo_root=self.task_data["trtllm_repo_path"],
            target_rel=self._preflight.target_rel,
            family=self._preflight.family,
            reference_yaml=gates_mod.resolve_per_unit(
                self.task_data["accuracy"],
                "reference_yaml",
                (assembly_units(self.task_data) or [None])[-1],
            ),
            model_key=self._model_key(),
            accuracy_test=accuracy_test_path(
                self.task_data["trtllm_repo_path"], self._preflight.family
            ),
            commit_policy=CommitPolicy.from_task(self.task_data, workspace=self.workspace),
            anchor_pending=self._anchor_pending(),
        )
        promoted = promotion_mod.check_promoted(
            self.task_data["trtllm_repo_path"],
            self._preflight.target_rel,
            keep=promotion_mod.keep_local(self.task_data),
        )
        tested = self._tests_check(report)
        return DeliveryReport(checks=(*report.checks, *promoted, *((tested,) if tested else ())))

    def _tests_check(self, report: DeliveryReport) -> DeliveryCheck | None:
        """The delivered tests pass at the checkout's HEAD, by the run's test cache.

        The target's own tests, and the catalog entries' tests it is delivered
        with (the report's ``test:`` checks), each by the newest recorded
        result that holds at ``HEAD`` (``testcache.head_results``): none
        failing, none still to run, one at least passed. ``None`` when there
        is no such test to read.
        """
        repo = Path(str(self.task_data["trtllm_repo_path"]))
        paths = [model_ops_tests_dir(self._preflight.target_rel)]
        for check in report.checks:
            if check.name.startswith("test:") and check.passed:
                try:
                    paths.append(
                        Path(check.detail).resolve().relative_to(repo.resolve()).as_posix()
                    )
                except (OSError, ValueError):
                    continue
        head = (self._git(str(repo), "rev-parse", "HEAD") or "").strip()
        try:
            cache = testcache.open_cache(self.workspace / testcache.CACHE_DIRNAME, repo, head)
            if not head or not cache.test_files(head, paths):
                return None
            results = testcache.head_results(cache, head, paths)
        except (testcache.ResultCacheError, OSError, ValueError) as exc:
            return DeliveryCheck("tests", False, f"the run's test cache cannot be read: {exc}")
        if results.verdict == "CLEAN":
            return DeliveryCheck("tests", True, results.describe())
        return DeliveryCheck(
            "tests",
            False,
            f"{results.describe()}. Run them at {head[:10]} through the run's test cache: "
            f"`trtllm-modeling-v2 tests affected --workspace {self.workspace} --head {head} "
            f"--paths {' '.join(paths)}` prints the command, run on a GPU",
        )
