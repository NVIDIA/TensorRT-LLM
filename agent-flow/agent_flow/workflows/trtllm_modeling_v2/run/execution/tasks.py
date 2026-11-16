# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution stage: the workflow's services handed to :class:`~..orchestrator.GraphOrchestrator`.

Entered after preflight. Each task's worker turn is this module's, and so is what code
does for the task list: which deployments the spec states (:func:`deployments_of`), the
package skeleton committed before the fan-out, the card a turn is asked with and the
landing of what it wrote. CheckRunner owns contract checks; deciding what runs and what
counts is the orchestrator's. Once the contract holds, delivery verifies the trees on disk.

Every contract check measures a snapshot of the commit the checkout is at
when it starts (the run's trees: ``pin`` and ``at``), never the checkout, which landings
move while checks run. Its verdict names that commit, is recorded under its
marker, and is void when a command imported any other tree.

Each attempt of a task leaves its receipt (``run/execution/handoff.py``), and a task
that starts carries the receipt of each task it depends on.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ...agents.decisions import BLOCKED, DECISION_REMINDER, READY
from ...agents.prompts import card as card_mod
from ...agents.prompts.survey import capture_lists_block
from ...agents.prompts.tags import tag_section, task_tags
from ...checks import delivery as delivery_rules
from ...checks import gate_cli, test_results
from ...checks import session as session_mod
from ...infra.compute import executor as executor_mod
from ...infra.compute import pintree
from ...infra.compute.controller import LIMIT_STOP
from ...infra.git import COMMIT_POLICY_FIELD, CommitPolicy, git_output
from ...operators import rows as rows_mod
from ...operators.coverage import QueueItem, VocabularyError, load_vocabulary, queue_item_from_ref
from ...spec import deployments as deployments_mod
from ...spec.task_schema import assembly_units, gpus_per_node
from .. import escalation as escalation_mod
from .. import evidence as evidence_mod
from .. import graph as graph_mod
from .. import keys as keys_mod
from ..orchestrator import (
    AGGREGATED,
    CHECKED,
    DEFAULT_MAX_PARALLEL_CHECKS,
    DISAGGREGATED,
    FINISHED,
    Deployment,
    GraphOrchestrator,
    Services,
    seed,
)
from ..records.state import WorkflowState
from ..validation import completion as deliver
from ..validation import contract as contract_mod
from ..validation import regression as regression_mod
from ..validation.check_runner import CheckRunner
from ..workspace import integration as trees_mod
from ..workspace.integration import (
    INTEGRATED_KINDS,
    REBASE_RETRIES,
    TaskNotStarted,
    rebase_block,
    repair_block,
    worktree_block,
)
from . import handoff as handoff_mod
from . import kernel_task as kernel_writer
from .handoff import (
    DECISIONS_DIRNAME,
    TaskResult,
    current_decision,
    decision_dispatch_matches,
    decision_file_instruction,
    decision_value,
    mark_decision_dispatch,
)

SURVEY_TASK_ID = "survey"

TASKS_DIRNAME = "tasks"
#: A task whose commit reaches outside its write scope is sent back this many times, with
#: the paths, before it is blocked for the outer session.
SCOPE_RETRIES = 1
#: How many turns a task gets to put right a finding of form it is told of when it hands
#: in: one, given beside its budget. What stands after it lands with the work.
FORM_TURNS = 1
#: The initializer a new package directory of the skeleton gets.
SKELETON_INITIALIZER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
)
#: The task the skeleton's own tree and landing are named for.
SKELETON_TASK = "skeleton"
#: What an attempt ends with when the check its card names did not pass on the pin it
#: handed in: no receipt's word changes that, and nothing lands.
CHECK_NOT_PASSED = "its check did not pass on the pin it handed in"

#: The role each task kind's turn runs as.
ROLE_BY_KIND = {
    "survey": "surveyor",
    "kernel": "kernel_writer",
    "weights": "assembler",
    "routing": "assembler",
    "state": "assembler",
    "feature": "assembler",
    "assemble": "assembler",
}
#: Checks in flight at once on a run's sessions: each is a spool job, and the
#: session's scheduler queues what its devices cannot start yet, by class.
MAX_PARALLEL_CHECKS = 32


class ExecutionStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    def __init__(self, workflow) -> None:
        self.workflow = workflow
        self._turn_lock = threading.Lock()
        self._turns_by_key = {}
        self._tasks_dispatched = set()
        self._merged = {}
        self._tls = workflow._tls

    # ------------------------------------------------------------------ entry
    def run_execution(self, state: WorkflowState, log) -> bool:
        print_rule("[bold magenta]execution stage[/bold magenta]", log)
        contract = self.workflow._contract
        self.state = state
        self._integration = self._make_integration(state)
        # Checks measure pins of the head, never the moving checkout.
        self.workflow._snapshots = self._integration
        self.checks = CheckRunner(self.workflow)
        self._own_paths = self._target_paths()
        services = Services(
            work=self._work_task,
            check=lambda check: self.checks._prioritized_check(check, log),
            checkpoint=lambda: self.workflow._checkpoint(state),
            log=lambda message: self.workflow.journal.event("execution", message),
            key=self.workflow._check_key,
            finished=self._task_finished,
            freeze=self._freeze,
            final_tests=self._final_tests,
            delivery_files=lambda: deliver.delivery_failures(self.workflow),
            deployments=lambda: deployments_of(self.workflow.task_data, contract),
            ci_files=self._ci_files,
            skeleton=self._commit_skeleton if self._integration is not None else None,
            onboarded=getattr(self.workflow, "_mark_onboarded", None),
            receipt=self._receipt_of,
            hooks=self._hooks,
            late=lambda: getattr(getattr(self.workflow, "_pool", None), "late", ""),
            end_turn=lambda task: self.workflow._turns.end(task.id),
        )
        try:
            # An onboarded baseline the spec names is a ref that resolves, or the run is
            # refused here, before anything is compiled from it.
            baseline = onboarded_baseline(
                self.workflow.task_data, git=getattr(self.workflow, "_git", git_output)
            )
        except ValueError as exc:
            self.workflow.journal.event("execution", str(exc))
            print_message(f"[bold red]✘ {escape_markup(str(exc))}[/bold red]", log)
            return False
        orchestrator = GraphOrchestrator(
            services,
            contract,
            self.workflow._contract_status(),
            self.workflow.workspace,
            allowed_paths=CommitPolicy.from_task(
                self.workflow.task_data, workspace=self.workflow.workspace
            ).allowed_paths,
            max_parallel=self.workflow.max_parallel_tasks,
            max_parallel_checks=(
                MAX_PARALLEL_CHECKS
                if self.workflow._gate_env().spool
                else DEFAULT_MAX_PARALLEL_CHECKS
            ),
            max_attempts=getattr(self.workflow, "max_attempts", graph_mod.MAX_ATTEMPTS),
            own_paths=self._own_paths,
            repo_root=self.workflow.task_data.get("trtllm_repo_path") or "",
            baseline=baseline,
        )
        # What the last process left in the spool is not queued again: a check or a test
        # job that was queued, running or finished unread is taken by the request that
        # made it, and a check that already has its row is not asked for at all.
        spool = getattr(self.workflow._gate_env(), "spool", None)
        left = executor_mod.left_in_spool(spool) if spool else []
        if left:
            self.workflow.journal.event(
                "resume",
                f"{len(left)} job(s) an earlier process queued are still in the spool or "
                f"finished unread ({', '.join(left[:6])}{' ...' if len(left) > 6 else ''}): "
                "each is adopted by the check that asked for it, none is queued twice",
            )
        self._mark_integrated = orchestrator.mark_integrated
        self._checking_turn = orchestrator.checking
        self._allow_paths = orchestrator.allow_paths
        self._recontract = orchestrator.recontract
        self._holds_onboarded = orchestrator.holds_onboarded
        # From here the pool is asked for what the task list needs, pass by pass.
        self.workflow._task_demand = orchestrator.demand
        verdict = orchestrator.run()
        self.workflow._final_record = orchestrator.final_record
        # Where the requested checks stand: on the candidate of the final pass when one
        # was judged, else on the target as it is.
        final = orchestrator.final
        evaluation = final.evaluation() if final is not None else orchestrator.standing()
        self.workflow.journal.section(f"execution stage: {verdict}", evaluation.render())
        if verdict.startswith("stopped"):
            # Preflight's probe, running beside the first turns, found the run cannot go on
            # here: nothing is closed, and the same command resumes once fixed. Or the
            # controller reached its own job's end, and said what continues the run.
            how = (
                "it continues as the controller said"
                if verdict.endswith(LIMIT_STOP)
                else "re-run the same command once the cause is fixed"
            )
            print_message(
                f"[bold yellow]⚠ execution stage {escape_markup(verdict)}; the run stays open: "
                f"{how}[/bold yellow]",
                log,
            )
            return False
        if verdict == "escalated":
            # Everything left is blocked: the run stays open for the outer session, which
            # answers each block and runs the same command again.
            waiting = escalation_mod.waiting(orchestrator.graph)
            for line in waiting:
                print_message(f"[bold yellow]⚠ {escape_markup(line)}[/bold yellow]", log)
            print_message(
                f"[bold yellow]⚠ the run is escalated: {len(waiting)} block(s) wait for an "
                f"answer. Answer each with `trtllm-modeling-v2 resolve --workspace "
                f"{self.workflow.workspace} <verb> <task|run>`, then run the same command "
                f"again[/bold yellow]",
                log,
            )
            # The run stays open, and says where it stands: what is not done, and why.
            self.workflow._write_report(verdict)
            self.workflow._escalated = True
            return False
        if verdict != "delivered":
            self.workflow._close_run(state, log, verdict=verdict)
            return False
        return deliver.run_deliver_phase(self.workflow, state, log, orchestrator.completion)

    # --------------------------------------------------------------- dispatch
    def _next_turn(self, role: str, unit: str) -> int:
        with self._turn_lock:
            turns = self._turns_by_key
            key = f"{role}:{unit}"
            turns[key] = turns.get(key, 0) + 1
            return turns[key]

    def _task_finished(self, task: graph_mod.Task) -> None:
        """Mark ``task``'s answered turns consumed once its attempt has an outcome.

        A relaunch adopts only the answered turn of an attempt with no outcome yet; a
        consumed turn is never adopted, so the next attempt is always asked afresh. A
        ``check`` task that passed, which no worker ran, leaves its receipt here.
        """
        self.workflow._turns.consume([task.id])
        if task.kind == graph_mod.CHECK_KIND and task.status == "done":
            note = (
                f"its checks passed, with no agent turn: {', '.join(task.satisfies)}"
                if task.satisfies
                else "the tasks it joins are done"
            )
            try:
                handoff_mod.write(
                    self.workflow.workspace / TASKS_DIRNAME,
                    handoff_mod.build(
                        task.id,
                        task.kind,
                        result=TaskResult(CHECKED, note),
                        attempt=task.attempts,
                    ),
                )
            except OSError as exc:
                self.workflow.journal.event("receipt not kept", f"{task.id}: {exc}")

    def _dispatch_task(
        self,
        role: str,
        unit: str,
        what: str,
        prompt: str,
        *,
        iteration: int | None = None,
        label: str = "",
        fresh: bool = False,
    ) -> TaskResult:
        """Dispatch ``what`` and read only its reserved decision file.

        ``iteration`` overrides the per-role, per-unit turn counter; ``label``
        overrides the report name. The first dispatch may adopt an unconsumed
        answer from the same task attempt. ``fresh`` forbids adoption for an
        explicit retry or rebase. Subsequent dispatches always ask afresh, and
        turns belonging to completed tasks are consumed before adoption.
        """
        turn = self._next_turn(role, unit) if iteration is None else iteration
        with self._turn_lock:
            dispatched = self._tasks_dispatched
            first = (role, what) not in dispatched
            dispatched.add((role, what))
        transport = self.workflow._turns
        tasks = self._tasks_on_disk()
        transport.consume(
            t.id for t in tasks.values() if t.status in ("done", "blocked", "skipped")
        )
        task = tasks.get(what)
        answered = None
        if first and not fresh and task is not None and task.status in ("pending", "running"):
            answered = transport.adopt_answered(
                role,
                what,
                lambda path: (
                    decision_dispatch_matches(
                        path, {"task_id": task.id, "task_attempt": task.attempts}
                    )
                    and decision_value(path, role) is not None
                ),
            )
        if answered is not None:
            print_message(
                f"[bold cyan]→ {role} {escape_markup(what)}: adopting {answered.request_id}, "
                f"answered for a previous process, instead of dispatching again[/bold cyan]",
                None,
            )
            self.workflow.journal.event(
                "decision adopted on resume",
                f"{role} {what}: {answered.request_id} {answered.decision_file}",
            )
            return TaskResult.answered(role, answered.decision_file, answered.result_file)
        context = self.workflow._context_block()
        if context:
            prompt = f"{prompt}\n\n{context}\n"
        target = mark_decision_dispatch(
            self.workflow.workspace,
            role,
            turn,
            unit=unit,
            identity={"task_id": task.id, "task_attempt": task.attempts} if task else None,
        )
        prompt = f"{prompt}\n\n{decision_file_instruction(target)}"
        report = self.workflow.journal.record_dispatch(role, label or what, prompt)
        result = self.workflow._run_turn(
            role, label or what, prompt, iteration=turn, unit=unit, work=what
        )
        self.workflow.journal.record_result(report, result)
        return TaskResult.answered(role, Path(target), report)

    # ------------------------------------------------------ worktrees, devices
    def _freeze(self) -> str:
        """The candidate of the final pass: the target's head, pinned under the landing lock.

        The head is pinned while the lock every landing takes is held, so
        the candidate is one state of the target and the build that goes with it. On a
        checkout nothing pins, the commit it is at.
        """
        integration = getattr(self, "_integration", None)
        if integration is not None:
            return str(integration.pin().commit)
        return self.workflow._head()

    def _final_tests(self, candidate: str) -> regression_mod.TestsResult | None:
        """The final tests on the frozen candidate: one selection of test files, run once.

        Selected from the change and the tasks, with no analysis of imports
        (``regression.select``): the target's own tests, the tests each done task's write
        scope names, the delivered tests, and the test files the change itself changed.
        Shared code a task with tests of its own was given is covered by those tests; a
        change to any other shared file has an unknown reach, and the package's collected
        suite runs with them. Each job states the most devices a done task's row states,
        one at least, or the most nodes when a row states whole nodes, as that row's own
        check did: never a node's count by default. Each failing case is compared with the
        commit the run started from.

        A selected file that is one of the engine's accuracy tests is collected and not
        run (``regression.accuracy_tests``): its session measured that benchmark through
        the endpoint. It is listed as left to CI, with what a session measured.

        What the tests cannot show is said, not passed over (``TestsResult.unverified``): a
        test a done task declared that the candidate does not hold, a selection that only
        skipped, and a repository with nothing to run.

        The result is a row of the ledger under the tests' key on the candidate, this
        selection, the native build and the runtime, and is kept whole beside it: the same
        candidate and selection are not run a second time. The jobs carry the build and
        the runtime as their identity, so results recorded with other libraries are not
        read as these. ``None`` when the checkout is no repository.
        """
        workflow = self.workflow
        workspace = workflow.workspace
        repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
        head = candidate or workflow._head()
        if not head or not (repo / ".git").exists():
            return None
        graph = graph_mod.Graph.load(workspace / graph_mod.GRAPH_FILENAME)
        rows = {row.id: row for row in rows_mod.load_work(workspace)}
        own = tuple(self._own_paths)
        obligations: list[str] = []
        covered: list[str] = []
        gpus, nodes = 1, 0
        for task in graph.tasks:
            if task.status != "done" or task.kind not in INTEGRATED_KINDS:
                continue
            tests = test_files(task.touches)
            obligations += tests
            row = rows.get(task.ref)
            if row is not None:
                # As the row's own check states them: devices on one node, or whole nodes.
                gpus = max(gpus, int(row.gpus or 0))
                nodes = max(nodes, int(row.nodes or 0))
            if tests:
                # Its own tests are the reach of the shared code it was given.
                covered += graph_mod.shared_touches(task, own)
        # A test a task declared that the candidate does not hold was not run: named, not dropped.
        missing = regression_mod.missing_declared(repo, head, obligations)
        try:
            obligations += deliver.delivered_test_paths(workflow, deliver.delivery_report(workflow))
        except (AttributeError, KeyError, OSError, ValueError):
            pass  # a run with no target named yet delivers no test
        base = test_results.default_base(workspace, repo, head)
        files, scope = regression_mod.select(
            repo, base, head, own=own, obligations=obligations, covered=covered
        )
        if not files:
            # A repository with nothing to run is not a pass: it says so.
            said = [f"{path} is declared by a task and is not in the candidate" for path in missing]
            return regression_mod.TestsResult(
                head, base, regression_mod.NOT_VERIFIED, scope=scope, gpus=gpus, nodes=nodes,
                unverified=said or ["no test file is selected: the target has no test to run"],
            )  # fmt: skip
        # What a test's result depends on beside the tree: the native build by content and
        # the runtime. The jobs carry it, so results recorded with other libraries or
        # another interpreter are not read as this candidate's; the row's key takes it too.
        runtime = contract_mod.runtime_inputs(workflow.task_data)
        try:
            build = pintree.record_digest(keys_mod.build_of(repo))
        except (OSError, ValueError):
            build = "unknown"
        identity = hashlib.sha256(
            json.dumps([build, runtime], sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        # The engine's accuracy tests among the selection: each case is a load and a whole
        # benchmark, which the deployment's session measured. Collected and left to CI.
        harness = regression_mod.accuracy_tests(repo, head, files)
        ci: dict[str, str] = {}
        if harness:
            measured = self._measured(head)
            ci = {
                path: regression_mod.measured_by_sessions(text, measured)
                for path, text in harness.items()
            }
        key = workflow._key(
            regression_mod.FINAL_TESTS,
            head,
            extra={"files": files, "gpus": gpus, "nodes": nodes, "base": base, "missing": missing,
                   "ci": ci},
        )  # fmt: skip
        # The tests' row is the one record of their result: a pass on the same candidate
        # and selection reads it there, with what it left to CI and could not verify.
        held = evidence_mod.lookup(workspace, regression_mod.FINAL_TESTS, key) if key else None
        if held is not None and held.passed and held.data:
            try:
                result = regression_mod.TestsResult(**held.data)
            except TypeError:
                result = None
            if result is not None and (result.passed or result.partial):
                workflow.journal.event(
                    "final tests",
                    f"not run again: they ran on this candidate and selection ({key}) and "
                    f"{'passed' if result.passed else 'failed nothing'}",
                )
                return result
        tree_at, run_job, python = regression_mod.job_runner(
            workflow, regression_mod.FINAL_TESTS, gpus=gpus, nodes=nodes, identity=identity
        )
        size = f"{nodes} node(s)" if nodes > 1 else f"{gpus} device(s)"
        workflow.journal.event(
            "final tests",
            f"one selection on the candidate {head[:10]}: {scope}; each job states {size}",
        )
        result = regression_mod.run_final_tests(
            workspace=workspace,
            repo=repo,
            head=head,
            base=base,
            files=files,
            scope=scope,
            tree_at=tree_at,
            run_job=run_job,
            python=python,
            missing=missing,
            ci=ci,
        )
        result.gpus, result.nodes = result.gpus or gpus, result.nodes or nodes
        workflow.journal.event("final tests", result.summary())
        if key:
            line = {
                "name": regression_mod.FINAL_TESTS,
                "member": "",
                # Failed nothing and does not certify: the row's third outcome, never a pass.
                "passed": result.passed or result.partial,
                "unverified": "; ".join(result.unverified) if result.partial else "",
                "detail": result.summary(),
                "data": dataclasses.asdict(result),
            }
            evidence_mod.record(
                workspace, workflow.task_data, regression_mod.FINAL_TESTS, [line], key
            )
        return result

    def _measured(self, candidate: str) -> dict[str, list[str]]:
        """The benchmarks the run's sessions measured on ``candidate``: each with its deployments.

        Read from the ledger: a benchmark's passing line recorded under the key its
        deployment's session has on the candidate. A line under another key measured
        other content, and stands for nothing here.
        """
        workflow = self.workflow
        contract = getattr(workflow, "_contract", None)
        if contract is None:
            return {}
        verdicts = workflow._contract_status().verdicts
        found: dict[str, list[str]] = {}
        for check in contract.required():
            if check.gate != session_mod.GATE_NAME:
                continue
            key = workflow._check_key(check, candidate)
            for verdict in verdicts if key else ():
                scored = verdict.gate.startswith("accuracy[") and verdict.gate.endswith("]")
                if (
                    scored
                    and verdict.passed
                    and (verdict.unit, verdict.marker) == (check.unit, key)
                ):
                    units = found.setdefault(verdict.gate[len("accuracy[") : -1], [])
                    if check.unit not in units:
                        units.append(check.unit)
        return found

    def _make_integration(self, state: WorkflowState) -> trees_mod.Integration | None:
        """The run's trees, on a git checkout; on anything else tasks share the tree.

        A whole tree per task, and the pins the checks measure. What an earlier process left
        that nobody holds any more goes now.
        """
        repo = Path(str(self.workflow.task_data.get("trtllm_repo_path") or ""))
        if not (repo / ".git").exists():
            self.workflow.journal.event(
                "integration",
                f"tasks share the checkout {repo}, and checks measure it as it is: not a git "
                "checkout",
            )
            return None
        policy = CommitPolicy.from_task(self.workflow.task_data, workspace=self.workflow.workspace)
        env = {}
        if policy.identity is not None:
            # The rebase's committer; each commit keeps its author.
            env = {k: v for k, v in policy.identity.env().items() if k.startswith("GIT_COMMITTER")}
        trees = trees_mod.make(
            self.workflow.task_data,
            repo=repo,
            workspace=self.workflow.workspace,
            prefix=f"mv2/{state.slug or 'run'}",
            env=env,
            log=lambda message: self.workflow.journal.event("integration", message),
            policy=policy,
        )
        prune = getattr(trees, "prune", None)
        if prune is not None:
            prune()
        return trees

    # ----------------------------------------------------------------- prompts
    def _target_paths(self) -> tuple[str, ...]:
        """The target's own paths (:func:`~..graph.own_paths`); none before preflight names it."""
        preflight = getattr(self.workflow, "_preflight", None)
        return graph_mod.own_paths(str(getattr(preflight, "target_rel", "") or ""))

    def _assembler_prompt(self, unit: str) -> str:
        return (
            f"Workspace: {self.workflow.workspace}\n"
            f"Repository: {self.workflow.task_data['trtllm_repo_path']}\n"
            f"Target: {self.workflow._preflight.target_rel}\n"
            f"Deployment: **{unit}**\n\n"
            f"Read `{self.workflow.task_path}` (the spec), `{self.workflow.plan_path}` (the plan) and "
            f"`{self.workflow.vocabulary_path}` (the computation-step list you derive against). "
            f"Earlier turns on this deployment, if any, left their decision files in "
            f"`{self.workflow.workspace / DECISIONS_DIRNAME}` and their reports in "
            f"`{self.workflow.journal.reports_dir}`: build on them rather than starting over.\n\n"
            f"{self._gate_history_block()}"
            f"Assemble the target for this deployment. Its check is one session of the "
            f"deployment `{unit}`: its "
            f"servers are loaded once and every check asks them, one line each. The command "
            f"on your card runs it on your tree with the settings its verdict uses, and no "
            f"other form of it is counted.\n\n" + DECISION_REMINDER
        )

    def _cli(self) -> str:
        """The workflow's CLI as an agent calls it: by path, when preflight installed it."""
        script = self.workflow.workspace / "bin" / gate_cli.CLI_NAME
        return str(script) if script.is_file() else gate_cli.CLI_NAME

    def _gate_history_block(self) -> str:
        rendered = self.workflow.gate_log.render()
        if not rendered:
            return ""
        return f"Gates so far:\n\n```\n{rendered}\n```\n\n"

    # ---------------------------------------------------------------- workers
    def _envelope(self, task: graph_mod.Task, receipts: list[dict] | None = None) -> Path:
        """``tasks/<id>.json``: what the worker is asked, in one file.

        With the receipt of each task it depends on (``receipts``), when any.
        """
        path = self.workflow.workspace / TASKS_DIRNAME / f"{task.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = task.as_dict()
        if receipts:
            data["receipts"] = receipts
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return path

    def _work_task(self, task: graph_mod.Task) -> str:
        """Run ``task``'s worker and return what its receipt says.

        The decision is the first line; anything after it is a note kept with the task.
        Each attempt's receipt is kept: a ready one for the tasks that depend on it, a
        blocked one as its question to the outer session.
        """
        decision = self._work_task_turn(task)
        self._keep_receipt(task, decision)
        return decision.text

    def _integrated_tip(self, task_id: str) -> str:
        """The branch tip the integration last fast-forwarded in for ``task_id``; "" for none."""
        integration = getattr(self, "_integration", None)
        return integration.integrated_tip(task_id) if integration is not None else ""

    def _work_task_turn(self, task: graph_mod.Task) -> TaskResult:
        """The worker, in its own tree when its kind changes the target; then its landing.

        A finished task whose rebase onto the checkout conflicts keeps its
        tree: its role is asked again, up to ``REBASE_RETRIES`` times, to
        rebase onto the head it met, resolve, re-test and hand back, and its
        branch is integrated again. Only then does it end ``INTEGRATION_FAILED``.
        """
        integration = getattr(self, "_integration", None)
        if task.ref == rows_mod.DELIVERY_ROW:
            repo = Path(str(self.workflow.task_data.get("trtllm_repo_path") or ""))
            if not escalation_mod.sent_back(task) and not self._delivery_owed(task, repo):
                # Nothing the delivery requires of the files is open: no turn is spent on it.
                # (One the outer session sent back with a note takes its turn: the note
                # is what is open, and the card carries it.)
                said = "nothing the delivery requires of the target's files is open: no turn"
                self.workflow.journal.event("integration", f"{task.id}: {said}")
                return TaskResult(READY, note=said)
        if integration is None or task.kind not in INTEGRATED_KINDS:
            return self._work_task_body(task)
        try:
            tree = integration.open(task.id)
        # Whatever stopped the checkout (a full filesystem, git), no worker ran.
        except Exception as exc:
            raise TaskNotStarted(f"{type(exc).__name__}: {exc}") from exc
        landed = False
        try:
            decision, landed = self._work_and_land(task, integration, tree)
            return decision
        finally:
            if not landed:
                # The attempt is over and nothing landed: its tree and branch stay for the next.
                integration.close(task.id, landed=False)

    def _work_and_land(
        self, task: graph_mod.Task, integration: trees_mod.Integration, tree: Path
    ) -> tuple[TaskResult, bool]:
        """``task``'s turn in ``tree`` and its landing; the decision, and whether it landed.

        Before it lands, what it committed is held to its write scope: a commit that
        reaches outside goes back to its worker once, with the paths; a second time the
        task is blocked for the outer session, which may widen the scope
        (``resolve allow-shared``). Then the files it changed are read by the delivery's
        rules for one file (:meth:`_file_rules`), which take no GPU. A finding of form
        is told, never held: the worker gets one turn to put it right while it has the
        context, beside its budget; what stands after that turn is recorded and lands
        with the work, and the delivery's own task repairs it after ``onboarded``
        (``rows.DELIVERY_ROW``). Then it is held to its evidence (:meth:`_evidence`): a
        receipt alone lands nothing. It lands when every check of its card ran and none
        failed; a check that could not be verified is recorded on the landing as such.
        """
        block = worktree_block(tree, integration.repo, scratch=False)
        # What there is to say of this tree: that it was lost and made again, say.
        notice = integration.notice(task.id)
        block = f"{block}\n\n{notice}" if notice else block
        self._tls.tree = str(tree)
        decision = self._work_in_tree(task, block, self._delivery_note(task, tree))
        rebases = sent_back = told = 0
        while decision.decision in FINISHED:
            outside = self._outside_scope(task, integration)
            if outside:
                listed = ", ".join(outside[:8]) + (" ..." if len(outside) > 8 else "")
                again = None
                if sent_back < SCOPE_RETRIES:
                    again = self._repair_turn(
                        task, block, card_mod.scope_note(outside, task.touches),
                        f"back to its worker, its commits reach outside its write scope: "
                        f"{', '.join(outside)}",
                    )  # fmt: skip
                if again is None:
                    how = "after being sent back once" if sent_back else "with no attempt left"
                    decision.decision = BLOCKED
                    decision.data.update(
                        blocker_kind="write_scope",
                        observed=f"its commits change {listed}, outside its write scope "
                        f"({', '.join(task.touches)}), {how}",
                        next_question=f"may `{task.id}` change {listed}? `resolve allow-shared "
                        f"{task.id} --path <prefix>` widens its scope; `resolve retry {task.id} "
                        "--note` sends it back again",
                    )
                    self.workflow.journal.event(
                        "integration", f"{task.id}: {decision.data['observed']}"
                    )
                    return decision.annotate(str(decision.data["observed"])), False
                sent_back += 1
                decision = again
                continue
            broken = self._file_rules(task, integration, tree)
            # The first deployment's assembly, before `onboarded`: its landing is all the
            # milestone waits for, so it is not held for a turn on form either. What stands
            # goes to the delivery's task, which starts once `onboarded` is pinned.
            waited_for = getattr(self, "_holds_onboarded", None)
            milestone = bool(broken) and waited_for is not None and waited_for(task.id)
            if broken and told < FORM_TURNS and not milestone:
                # The delivery will ask this of these files at the end of the run: the
                # task that wrote them is told now, with its context, and has one turn.
                told += 1
                self._form_scope(task, integration)
                again = self._form_turn(task, block, broken)
                if again is not None and again.decision in FINISHED:
                    decision = again
                    continue
            # Told, and it stands: a finding of form holds no landing, spends no attempt
            # and blocks nothing. It lands with the work, and is said once the work
            # has landed: a check that does not pass lands nothing, and its failure is
            # then all there is to say.
            form = (
                f"landed with {len(broken)} finding(s) of form"
                + (", not told first (`onboarded` waits for this landing)" if milestone else "")
                + f", which `{rows_mod.DELIVERY_ROW}` repairs after `onboarded` and the "
                "delivery requires at the end: " + "; ".join(broken)
                if broken
                else ""
            )
            proof = self._evidence(task, integration, tree)
            if proof is not None and not proof.passed and not proof.unverified:
                # It failed, or it did not run: nothing lands.
                decision.decision = CHECK_NOT_PASSED
                return decision.annotate(proof.said), False
            outcome = integration.land(task.id)
            self.workflow.journal.event("integration", f"{task.id}: {outcome.detail}")
            if outcome.merged:
                integration.close(task.id)
                # How many commits went in: the receipt lists them.
                self._merged[task.id] = outcome
                if form:
                    self.workflow.journal.event("integration", f"{task.id}: {form}")
                    decision.annotate(form)
                # Every check of the card ran and none failed. What one could not verify is
                # recorded on the landing, never as passed: the run then closes ``partial``
                # with it listed, and nothing is escalated for it.
                not_verified = (
                    f"{task.check}: {proof.unverified}"
                    if proof is not None and proof.unverified
                    else ""
                )
                if not_verified:
                    self.workflow.journal.event(
                        "integration", f"{task.id}: landed NOT VERIFIED ({not_verified})"
                    )
                mark = getattr(self, "_mark_integrated", None)
                if mark is not None:
                    mark(task.id, task.attempts, not_verified)
                return decision, True
            role = ROLE_BY_KIND.get(task.kind, task.kind)
            again = None
            if (outcome.base or outcome.repairable) and rebases < REBASE_RETRIES:
                if outcome.base:
                    said = (
                        f"back to its {role} to rebase onto {outcome.base[:10]} and resolve "
                        f"({rebases + 1} of {REBASE_RETRIES})"
                    )
                    note = rebase_block(outcome, tree)
                else:
                    said = (
                        f"back to its {role} to put its worktree right "
                        f"({rebases + 1} of {REBASE_RETRIES}): {outcome.detail.splitlines()[0]}"
                    )
                    note = repair_block(outcome)
                again = self._repair_turn(task, block, note, said)
            if again is None:
                decision.decision = "INTEGRATION_FAILED"
                return decision.annotate(outcome.detail), False
            rebases += 1
            decision = again
        return decision, False

    def _repair_turn(
        self, task: graph_mod.Task, block: str, note: str, said: str
    ) -> TaskResult | None:
        """One more turn of ``task`` in its tree, as its next attempt; ``None`` with none left.

        A turn that repairs what the landing refused (a commit outside the write scope, a
        rebase that conflicts, a worktree left wrong) is an attempt like any other: the
        task has one counter and one budget, and a repair spends from it.
        """
        if task.attempts >= task.budget:
            self.workflow.journal.event(
                "integration", f"{task.id}: not sent back, its {task.budget} attempt(s) are spent"
            )
            return None
        task.attempts += 1
        self.workflow.journal.event(
            "integration", f"{task.id}: {said} (attempt {task.attempts} of {task.budget})"
        )
        return self._work_in_tree(task, block, note)

    def _evidence(
        self, task: graph_mod.Task, integration: trees_mod.Integration, tree: Path
    ) -> "Proof | None":
        """What the check ``task``'s card names says of the pin of its tree; ``None`` with no check.

        Every typed row has one check and each is the gate's own (a kernel's tests, a
        transfer, a session, a row's own tests). The pin is what the tree holds now. The
        worker's own run of the card's command recorded its result under that pin's marker
        (``run/evidence.py``); when the ledger has none for it, the coordinator runs the
        same command once itself, holding no agent turn meanwhile, and reads the ledger
        again. A session's result counts only if it was judged by the requirements as they
        are now: the inputs it was recorded with are the contract's inputs for that check.
        No result is not a pass. One that failed nothing and could not verify what it is
        for is not a pass either, and says so (:attr:`Proof.unverified`).
        """
        pin_of = getattr(integration, "pin", None)
        if not evidence_mod.gated(task.check) or pin_of is None:
            return None
        workflow = self.workflow
        try:
            commit = str(pin_of(task.id).commit)
        except Exception as exc:  # nothing to have a result for
            return Proof(
                False,
                f"`{task.check}` has no pin to have a result for: its tree could not be "
                f"pinned ({type(exc).__name__}: {exc})",
            )
        marker = evidence_mod.key_of(workflow.task_data, workflow.workspace, commit, task.check)
        if not marker:
            # No result can count under no key: said, and no job is spent finding that out.
            return Proof(
                False,
                f"`{task.check}` has no key on the pin {commit[:10]}: something it consumes "
                "has no identity there (git cannot list the commit, or the checkpoint's "
                "path is not there), so no result counts for it",
            )
        inputs = self._check_inputs(task.check)
        found = evidence_mod.lookup(workflow.workspace, task.check, marker, inputs=inputs)
        if found is None:
            workflow.journal.event(
                "evidence",
                f"{task.id}: `{task.check}` has no result for the pin {commit[:10]}"
                + (" under the requirements as they are now" if inputs else "")
                + ": the coordinator runs it once",
            )
            argv = check_argv(task.check, **self._check_places(task, tree=str(tree)))
            run = getattr(workflow, "_gate_process", None) or evidence_mod.run_gate_process
            with getattr(self, "_checking_turn", contextlib.nullcontext)(task.id):
                status, tail = run(argv or [], self._cli_argv())
            found = evidence_mod.lookup(workflow.workspace, task.check, marker, inputs=inputs)
            if found is None:
                said = tail.strip().splitlines()[-1] if tail.strip() else "it said nothing"
                return Proof(
                    False,
                    f"`{task.check}` left no result for the pin {commit[:10]} (the gate "
                    f"exited {status}): {said}",
                )
        how = (
            "did not pass"
            if not found.passed
            else f"failed nothing and could not verify ({found.unverified})"
            if found.unverified
            else "passed"
        )
        workflow.journal.event("evidence", f"{task.id}: `{task.check}` {how} on {commit[:10]}")
        return Proof(
            bool(found.passed) and not found.unverified,
            f"`{task.check}` {how} on the pin {commit[:10]}: {found.detail}",
            str(found.unverified or "") if found.passed else "",
        )

    def _check_inputs(self, check_id: str) -> str | None:
        """The inputs a session is judged by now; ``""`` for a row's check; ``None`` when unknown.

        The contract's digest for the deployment's session check, from the contract the
        run holds, else compiled from the spec as it is. A session recorded with other
        inputs was judged by other requirements: it is no evidence for these.
        """
        name, _, what = check_id.partition(":")
        if name != evidence_mod.SESSION:
            return ""
        contract = getattr(self.workflow, "_contract", None)
        if contract is None:
            try:
                contract = contract_mod.compile_contract(self.workflow.task_data)
            except Exception:  # a spec no contract compiles from says nothing of its inputs
                return None
        return next(
            (
                c.inputs
                for c in contract.checks
                if c.unit == what and c.gate == evidence_mod.SESSION
            ),
            None,
        )

    def _check_places(self, task: graph_mod.Task, *, tree: str) -> dict[str, Any]:
        """Where and how the check of ``task``'s card runs.

        The spec, its tree, its own work directory; and what its row states of the job:
        its devices, its nodes and its wall, so that the pool's demand, the card and the
        job agree. A row's own tests are the test files in its write scope.
        """
        workspace = self.workflow.workspace
        row = rows_mod.row_of(workspace, task.ref)
        return dict(
            task_path=self.workflow.task_path,
            tree=tree,
            workdir=workspace / "gates" / task.id,
            vocabulary=str(row.config_axes.get("vocabulary") or "") if row is not None else "",
            gpus=int(row.gpus) if row is not None else 0,
            nodes=int(row.nodes) if row is not None else 0,
            wall=int(row.wall) if row is not None else 0,
            tests=tuple(test_files(task.touches)),
        )

    def _cli_argv(self) -> tuple[str, ...]:
        """The workflow's CLI as a process starts it: the installed script, else this interpreter's."""
        script = self.workflow.workspace / "bin" / gate_cli.CLI_NAME
        return (str(script),) if script.is_file() else ()

    def _form_turn(
        self, task: graph_mod.Task, block: str, broken: Sequence[str]
    ) -> TaskResult | None:
        """One turn of ``task`` to put right the findings of form it is told of; beside its budget.

        The turn is given, not taken from the task's attempts: a finding of form never
        exhausts a budget. So the budget grows by the one attempt the turn is numbered as.
        """
        task.budget += 1
        return self._repair_turn(
            task,
            block,
            form_note(broken),
            f"told of {len(broken)} finding(s) of form in the files it changed, with one "
            "turn to put them right",
        )

    def _form_scope(self, task: graph_mod.Task, integration: trees_mod.Integration) -> None:
        """``task``'s write scope gets where the delivery looks for the test of each entry it changed.

        Before it is told of a finding of form: the note asks for the entry's test at
        that place, and a task that obeys it is not then refused by its own scope. (A
        row that names its entries has them from the start: ``compile_tasks``.)
        """
        preflight = getattr(self.workflow, "_preflight", None)
        target_rel = str(getattr(preflight, "target_rel", "") or "")
        if not target_rel or not task.touches:
            return
        try:
            changed = [str(path) for path in integration.changed_paths(task.id) or ()]
        except Exception:  # a branch git cannot read is the landing's to refuse
            return
        places = (
            place
            for path in changed
            for place in delivery_rules.entry_test_places("", target_rel, path)
        )
        added = [p for p in dict.fromkeys(places) if not graph_mod.inside(p, task.touches)]
        if added:
            task.touches = [*task.touches, *added]
            self.workflow.journal.event(
                "integration",
                f"{task.id}: its write scope now holds where the delivery looks for the "
                f"tests of the entries it changed: {', '.join(added)}",
            )

    def _rules_of(self, root: Path, changed: Sequence[str]) -> list[Any]:
        """The delivery's rules for one file, asked of ``changed`` as they are under ``root``."""
        preflight = getattr(self.workflow, "_preflight", None)
        target_rel = str(getattr(preflight, "target_rel", "") or "")
        if not target_rel:
            return []
        checks = delivery_rules.check_changed_entries(Path(root), target_rel, changed)
        if delivery_rules.v1_imports_refused(self.workflow.task_data):
            # The guide's import rule, of the target's own files among them.
            own = [p for p in changed if graph_mod.inside(p, [target_rel])]
            checks += delivery_rules.check_v1_imports(Path(root), own)
        return checks

    def _registration(self, root: Path, task: graph_mod.Task) -> list[Any]:
        """The delivery's rules for CI (``delivery.check_ci``), asked of the delivery's task.

        The accuracy test, the list that names it and the anchor's row are that task's
        alone (``compile_tasks``). The same list, asked with the same words, as the
        delivery's own check (``completion.ci_rules``).
        """
        ci = self._ci_files()
        if not ci or not task.touches or task.ref != rows_mod.DELIVERY_ROW:
            return []
        # The file each rule is about: the reference file for the anchor's row, the
        # accuracy test for the others.
        about = {"accuracy:reference": ci[2] if len(ci) > 2 else ""}
        return [
            check
            for check in deliver.ci_rules(self.workflow, root)
            if (where := about.get(check.name, ci[0])) and graph_mod.inside(where, task.touches)
        ]

    def _file_rules(
        self, task: graph_mod.Task, integration: trees_mod.Integration, tree: Path
    ) -> list[str]:
        """What the delivery's rules for one file say of the files ``task`` changed, in its tree.

        The same check the delivery runs on the finished target, earlier and narrower
        (``delivery.check_changed_entries``): an entry is an ``OpWrapper`` with its
        contract, and has its test where CI matches it. And of the task that owns the
        target's accuracy test for CI (``rows.DELIVERY_ROW``): the file is there and a
        test list names it (``delivery.check_test_registration``). The rules of the whole target (its
        imports, its routing, what is committed) stay at the end. Each failing rule as one
        line; none when the run knows no target yet or git cannot say what changed.
        """
        try:
            changed = [str(path) for path in integration.changed_paths(task.id) or ()]
        except Exception:  # a branch git cannot read is the landing's to refuse
            return []
        try:
            checks = self._rules_of(tree, changed) + self._registration(tree, task)
        except (OSError, SyntaxError, ValueError):
            # A file the rules cannot place or read is the delivery's to judge at the end.
            return []
        return [f"{check.name}: {check.detail}" for check in checks if not check.passed]

    def _delivery_owed(self, task: graph_mod.Task, root: Path) -> list[str]:
        """What the delivery's task has to put right in the tree at ``root``, one line each.

        Derived, not kept: the delivery's rules for one file asked again of every file
        the run changed, as they are now, and of the accuracy test for CI where this
        task owns it. A finding a later task already repaired is not in it.
        """
        workflow = self.workflow
        repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
        changed: list[str] = []
        try:
            head = workflow._head() if hasattr(workflow, "_head") else ""
            base = test_results.default_base(workflow.workspace, repo, head) if head else None
            if base:
                diff = git_output(repo, "diff", "--no-renames", "--name-only", base, head)
                changed = [line for line in (diff or "").splitlines() if line]
            checks = self._rules_of(root, changed) + self._registration(root, task)
        except (OSError, SyntaxError, ValueError):
            return []
        return [f"{check.name}: {check.detail}" for check in checks if not check.passed]

    def _delivery_note(self, task: graph_mod.Task, tree: Path) -> str:
        """What the delivery's task is told beside its card: the findings that stand; else nothing."""
        if task.ref != rows_mod.DELIVERY_ROW:
            return ""
        owed = self._delivery_owed(task, tree)
        return delivery_note(owed) if owed else ""

    def _ci_files(self) -> tuple[str, ...]:
        """The files the delivery requires for CI beside the target; none when it owes none.

        The target's accuracy test where this tree keeps such tests
        (``delivery.accuracy_test_path``), then the directory of the test lists, then
        the reference file its anchor's row is read from (``completion.reference_yaml``;
        left out when the spec names one outside the checkout). A run none of whose
        deployments scores a benchmark owes none (``completion.ci_waiver``), and before
        preflight the target is not known.
        """
        workflow = self.workflow
        preflight = getattr(workflow, "_preflight", None)
        target_rel = str(getattr(preflight, "target_rel", "") or "")
        try:
            if not target_rel or deliver.ci_waiver(workflow.task_data):
                return ()
            test = delivery_rules.accuracy_test_path(
                workflow.task_data["trtllm_repo_path"], preflight.family, Path(target_rel).name
            )
        except (AttributeError, KeyError, OSError, ValueError):
            return ()
        reference = Path(deliver.reference_yaml(workflow.task_data))
        if reference.is_absolute():
            try:
                reference = reference.relative_to(Path(str(workflow.task_data["trtllm_repo_path"])))
            except ValueError:
                reference = None
        return (
            f"{delivery_rules.ACCURACY_TESTS_DIR}/{test}",
            delivery_rules.TEST_DB_DIR,
            *((reference.as_posix(),) if reference is not None else ()),
        )

    def _outside_scope(self, task: graph_mod.Task, integration: trees_mod.Integration) -> list[str]:
        """The paths ``task``'s branch changes that its write scope does not hold."""
        if not task.touches:
            return []
        try:
            changed = integration.changed_paths(task.id)
        except Exception:  # a branch git cannot read is the landing's to refuse
            return []
        return sorted(p for p in changed or () if not graph_mod.inside(p, task.touches))

    def _work_in_tree(self, task: graph_mod.Task, block: str, note: str = "") -> TaskResult:
        """``task``'s worker, told its tree (``block``) and, when sent back, what to do (``note``)."""
        self._tls.worktree = block
        try:
            item = self._queue_item(task) if note and task.kind == "kernel" else None
            if item is not None:
                # A kernel writer given a note is dispatched afresh rather than handed the
                # decision already on disk: that decision is the work to put right.
                return kernel_writer._run_one_kernel_writer(
                    self,
                    item,
                    note,
                    task,
                    card=self._card(task),
                    context=self._task_context(self._dependency_receipts(task)),
                )
            if note:
                self._tls.worktree = f"{block}\n\n{note}"
            return self._work_task_body(task)
        finally:
            self._tls.worktree = ""

    def _work_task_body(self, task: graph_mod.Task) -> TaskResult:
        receipts = self._dependency_receipts(task)
        envelope = self._envelope(task, receipts)
        card = self._card(task)
        if task.kind == "kernel":
            item = self._queue_item(task)
            if item is None:
                return TaskResult(
                    f"kernel task {task.id} names no row with an entry ({task.ref!r})"
                )
            # A new attempt needs a fresh dispatch. A relaunch can instead resume the
            # same attempt, whose completed answer still awaits integration.
            resuming = decision_dispatch_matches(
                current_decision(
                    self.workflow.workspace,
                    "kernel_writer",
                    item.n,
                    unit=kernel_writer.decision_unit_of(item),
                ),
                {"task_id": task.id, "task_attempt": task.attempts},
            )
            retry = (
                f"Attempt {task.attempts} of task {task.id}; the previous attempt did not end done."
                if task.attempts > 1 and not resuming
                else ""
            )
            return kernel_writer._run_one_kernel_writer(
                self, item, retry, task, card=card, context=self._task_context(receipts)
            )
        role = ROLE_BY_KIND[task.kind]
        row = rows_mod.row_of(self.workflow.workspace, task.ref)
        unit = (row.deployment if row is not None else "") or task.ref or task.id
        if task.kind == "assemble":
            prompt = self._assembler_prompt(unit)
        elif task.kind == "survey":
            prompt = _surveyor_prompt(self.workflow)
        else:
            prompt = ""
        tags = task_tags(task.kind)
        header = (
            f"{card}\n\nThe task envelope is `{envelope}`: the same facts as data, with the "
            f"receipts of the tasks you depend on."
        )
        # One order for a dispatch of any kind, a kernel row's included: the card, the
        # kind's own text, what the task depends on; its tree last.
        parts = (header, tag_section(tags), prompt, self._task_context(receipts))
        dispatch = "\n\n".join(p.strip() for p in parts if p)
        result = self._dispatch_task(role, unit, task.id, dispatch)
        if task.kind == "survey":
            result = self._checked_survey(result)
        return result

    def _card(self, task: graph_mod.Task) -> str:
        """The card of ``task``, rendered from its row (``agents/prompts/card.py``)."""
        workspace = self.workflow.workspace
        row = rows_mod.row_of(workspace, task.ref)
        tasks = self._tasks_on_disk()
        skipped = [
            f"`{dep}` ({tasks[dep].last or 'skipped'})"
            for dep in task.depends_on
            if dep in tasks and tasks[dep].status == "skipped"
        ]
        return card_mod.render_card(
            task,
            row,
            command=check_command(
                task.check,
                cli=self._cli(),
                **self._check_places(task, tree=getattr(self._tls, "tree", "") or "<your tree>"),
            ),
            notes=workspace / escalation_mod.NOTES_FILENAME,
            spec=self.workflow.task_path,
            skipped=skipped,
        )

    def _checked_survey(self, result: TaskResult) -> TaskResult:
        """A drafted survey, held to what code needs to make tasks from it and to nothing else."""
        if result.decision != READY:
            return result
        try:
            # The queue, and the typed rows beside it: what the task list is made from.
            rows_mod.load_rows(
                self.workflow.vocabulary_path, units=assembly_units(self.workflow.task_data)
            )
        except VocabularyError as exc:
            result.decision = f"vocabulary.yaml does not validate: {exc}"
        return result

    def _queue_item(self, task: graph_mod.Task) -> QueueItem | None:
        """The kernel writer dispatch a kernel task names.

        Its ``ref`` is its row: a queue row of the vocabulary by its number, or a row a
        split wrote, whose entry is ``<mode>:<entry>[@<site>]``
        (:func:`~...operators.coverage.queue_item_from_ref`).
        """
        if task.ref.isdigit():
            queue = load_vocabulary(
                self.workflow.vocabulary_path, units=assembly_units(self.workflow.task_data)
            )
            return next((q for q in queue if str(q.n) == task.ref), None)
        row = rows_mod.row_of(self.workflow.workspace, task.ref)
        entry = str(row.config_axes.get("entry") or "") if row is not None else task.ref
        return queue_item_from_ref(entry, task.id, task.goal)

    # ------------------------------------------------------------- receipts
    def _task_context(self, receipts: list[dict]) -> str:
        """What a task's turn carries beside its card: its dependencies, one line each.

        The receipts themselves are in the task's envelope as data, and each line names
        the file to read: a survey's whole summary is not pasted into every row's turn.
        What an engine ran with is in its session's own evidence, for a worker that asks.
        """
        return handoff_mod.brief(receipts)

    def _tasks_on_disk(self) -> dict[str, graph_mod.Task]:
        """Read the current graph, or no tasks if it is absent or unreadable mid-write."""
        try:
            return graph_mod.Graph.load(self.workflow.workspace / graph_mod.GRAPH_FILENAME).by_id()
        except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
            return {}

    def _dependency_receipts(self, task: graph_mod.Task) -> list[dict]:
        """The receipt of each done task ``task`` depends on, in its order."""
        if not task.depends_on:
            return []
        tasks = self._tasks_on_disk()
        tasks_dir = self.workflow.workspace / TASKS_DIRNAME
        found = []
        for dep in task.depends_on:
            done = tasks.get(dep)
            if done is not None and done.status != "done":
                continue
            receipt = handoff_mod.read(tasks_dir, dep)
            if receipt is not None and receipt.get("decision") in FINISHED:
                found.append(receipt)
        return found

    def _keep_receipt(self, task: graph_mod.Task, result: TaskResult) -> None:
        """Keep the attempt's evidence, ready or blocked, as its dispatch wrote it."""
        if result.decision not in FINISHED and result.decision != BLOCKED:
            return
        if result.decision == BLOCKED:
            result.annotate(handoff_mod.blocked_problem(result.data))
        try:
            handoff_mod.write(
                self.workflow.workspace / TASKS_DIRNAME,
                handoff_mod.build(
                    task.id,
                    task.kind,
                    result=result,
                    attempt=task.attempts,
                    commits=self._landed_commits(task.id) if result.decision in FINISHED else (),
                ),
            )
        except OSError as exc:
            self.workflow.journal.event("receipt not kept", f"{task.id}: {exc}")

    def _receipt_of(self, task: graph_mod.Task) -> Mapping[str, Any]:
        """The receipt ``task``'s latest attempt left; empty when it left none."""
        return (
            handoff_mod.read(self.workflow.workspace / TASKS_DIRNAME, task.id, task.attempts) or {}
        )

    def _landed_commits(self, task_id: str) -> list[str]:
        """``<sha> <subject>`` of each commit ``task_id``'s branch integrated, newest first.

        As many as its integration counted. One integrated before this
        process started has no count here: its branch tip stands for it.
        """
        integration = getattr(self, "_integration", None)
        tip = self._integrated_tip(task_id)
        if integration is None or not tip:
            return []
        outcome = self._merged.get(task_id)
        count = outcome.commits if outcome is not None else 0
        if outcome is not None and count == 0:
            return []
        log = self.workflow._git(
            str(integration.repo), "log", f"-{count or 1}", "--format=%h %s", str(tip)
        )
        lines = [line for line in (log or "").splitlines() if line.strip()]
        return lines if count else [f"{line} (the branch tip)" for line in lines[:1]]

    # ------------------------------------------------------------ correctness
    # ------------------------------------------------------------ the skeleton
    def _commit_skeleton(self, files: Sequence[str]) -> str:
        """Commit the package skeleton on the way to ``files``; what it committed.

        Every directory between the first one the head already has and a row's entry
        gets its initializer, and the target's own test directory gets one too, in one
        commit that lands as a task's does (audit, rebase, fast-forward under the landing
        lock). The tasks that write those files then start side by side and none of them
        creates, or may change, a shared initializer.
        """
        integration = self._integration
        # What is owed is read off the head as the checkout has it: nothing is remembered,
        # and where every directory on the way exists no tree is opened.
        nothing = "nothing to commit: every directory on the way exists"
        head = Path(str(self.workflow.task_data.get("trtllm_repo_path") or ""))
        if not skeleton_files(head, files, self._own_paths):
            return nothing
        tree = Path(integration.open(SKELETON_TASK))
        landed = False
        try:
            made = skeleton_files(tree, files, self._own_paths)
            if not made:
                landed = True  # nothing of it is left to land
                return nothing
            for rel in made:
                path = tree / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(SKELETON_INITIALIZER, encoding="utf-8")
            policy = CommitPolicy.from_task(
                self.workflow.task_data, workspace=self.workflow.workspace
            )
            # Authored and signed off as the run's commits are: the landing audits it too.
            identity = dict(policy.identity.env()) if policy.identity is not None else {}
            message = (
                "[None][feat] modeling_v2: the package skeleton of a new target\n\n"
                "The directories and initializers its entries and tests are written into."
            )
            for args in (("add", "--", *made), ("commit", "-s", "-q", "-m", message)):
                done = subprocess.run(
                    ["git", "-C", str(tree), *args],
                    capture_output=True,
                    text=True,
                    env={**os.environ, **identity},
                    check=False,
                )
                if done.returncode != 0:
                    said = (done.stderr or done.stdout).strip().splitlines()
                    raise RuntimeError(f"git {args[0]}: {said[-1] if said else 'failed'}")
            outcome = integration.land(SKELETON_TASK)
            if not outcome.merged:
                raise RuntimeError(outcome.detail)
            landed = True
            return f"{len(made)} initializer(s) committed and landed: {', '.join(made)}"
        finally:
            integration.close(SKELETON_TASK, landed=landed)

    # ----------------------------------------------- the pool, resolutions
    def _respec(self, keys: Sequence[str]) -> None:
        """The spec changed under ``keys`` (``resolve``): what this run holds of it follows.

        The workflow rewrites the agents' copy and its environment. A commit policy that was
        widened also reaches the two places a running controller keeps it: the landing's
        audit and the paths a task may be given. Without that a row let in by
        ``allow-shared`` was sent back at its landing until the run was started again.
        """
        respec = getattr(self.workflow, "_respec", None)
        if respec is not None:
            respec(keys)
        contract = getattr(self.workflow, "_contract", None)
        follow = getattr(self, "_recontract", None)
        if contract is not None and follow is not None:
            follow(contract)
        if not any(str(key).split(".")[0] == COMMIT_POLICY_FIELD for key in keys):
            return
        policy = CommitPolicy.from_task(self.workflow.task_data, workspace=self.workflow.workspace)
        integration = getattr(self, "_integration", None)
        if getattr(integration, "policy", None) is not None:
            integration.policy = policy
        follow = getattr(self, "_allow_paths", None)
        if follow is not None:
            follow(policy.allowed_paths)

    def _hooks(self) -> escalation_mod.Hooks:
        """What an answer of the outer session reaches outside the graph, as this run has it."""
        workflow = self.workflow
        return escalation_mod.Hooks(
            task_data=workflow.task_data,
            respec=self._respec,
            pool_retry=getattr(workflow, "_pool_retry", None),
            pool_adjust=getattr(workflow, "_pool_adjust", None),
        )


def deployments_of(
    task_data: Mapping[str, Any], contract: contract_mod.Contract
) -> list[Deployment]:
    """The deployments of this spec, the first first: each with its kind and what it depends on.

    The one place the task list reads what the spec deploys: the spec's own resolution
    (``spec.deployments.resolve``). For each: its name; whether it is one engine or a
    context side and a generation side; the contract checks its one session owes (the
    session itself, and the native tests it names); the deployment its score is judged
    against; and its session's job line, as the session itself would place it
    (``checks.session.place``: its devices on one node, or whole nodes past one, and
    its wall).
    """
    found = []
    for deployment in deployments_mod.resolve(task_data):
        try:
            placement = session_mod.place(task_data, deployment)
            nodes = placement.nodes if placement.nodes > 1 else 0
            gpus, wall = (0 if nodes else placement.gpus), placement.wall
        except session_mod.WiderThanOneNode:
            # The session says why before any load; the card still states what it holds.
            gpus, nodes = deployment.ranks, 0
            wall = session_mod.session_timeout(task_data, deployment) + session_mod.TEARDOWN_S
        found.append(
            Deployment(
                name=deployment.name,
                kind=(
                    AGGREGATED if deployment.kind == deployments_mod.AGGREGATED else DISAGGREGATED
                ),
                checks=tuple(c.key for c in contract.checks if c.unit == deployment.name),
                relative_to=str(deployment.relative_to or ""),
                gpus=int(gpus),
                nodes=int(nodes),
                wall=int(wall),
                row_gpus=min(deployment.ranks, gpus_per_node(task_data) or deployment.ranks),
            )
        )
    return found


def onboarded_baseline(task_data: Mapping[str, Any], git=git_output) -> str:
    """The commit of the onboarded ref the spec names (``onboarded``); ``""`` when it names none.

    A spec that names one adds to a model that is onboarded: its survey writes the delta
    rows only, and no deployment waits for a first one. That is taken only from a ref that
    resolves to a commit of the spec's checkout: a name that resolves to nothing
    establishes no baseline, and the run is refused with what it named.
    """
    ref = str(task_data.get("onboarded") or "").strip()
    if not ref:
        return ""
    repo = str(task_data.get("trtllm_repo_path") or "")
    commit = (git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}") or "").strip()
    if not commit:
        raise ValueError(
            f"the spec's `onboarded: {ref}` names no commit of {repo or 'the checkout'}: an "
            "onboarded baseline is a ref that resolves (`refs/mv2/<run>/onboarded`, a tag, a "
            "commit); remove the key to onboard from the start"
        )
    return commit


def skeleton_files(tree: Path, files: Sequence[str], own: Sequence[str] = ()) -> list[str]:
    """The initializers the skeleton owes for ``files`` in ``tree``, repository-relative.

    The run's own test directory (``own[1]``) is a package, as the tree's targets have
    theirs. Then, for each file, every directory down to the file's own gets an
    initializer while the directory above it is a package (it has one, or gets one
    here). Nothing that exists is rewritten, and asking again owes nothing.
    """
    owed: list[str] = []
    seen: set[str] = set()

    def package(directory: str) -> bool:
        return (tree / directory / "__init__.py").is_file() or directory in seen

    tests = (own[1] if len(own) > 1 else "").rstrip("/")
    if tests and any(f.startswith(tests + "/") for f in files) and not package(tests):
        seen.add(tests)
        owed.append(f"{tests}/__init__.py")
    for file in files:
        parts = file.split("/")[:-1]
        for depth in range(2, len(parts) + 1):
            directory = "/".join(parts[:depth])
            if not package(directory) and package("/".join(parts[: depth - 1])):
                seen.add(directory)
                owed.append(f"{directory}/__init__.py")
    return owed


test_files = rows_mod.test_files


@dataclass(frozen=True)
class Proof:
    """What a card's check says of a pin: whether it is evidence, what it said, what it could not verify."""

    passed: bool
    said: str
    unverified: str = ""


def form_note(broken: Sequence[str]) -> str:
    """What a worker is told when the files it hands in break a rule of the delivery's form."""
    lines = "\n".join(f"- {line}" for line in broken)
    return (
        "## Your work is accepted as it computes; its form has findings\n\n"
        "The delivery asks this of every such file at the end of the run. You wrote these "
        "files and have their context now:\n\n"
        f"{lines}\n\n"
        "You have this one turn to put them right: change the file to hold to the rule "
        "(or, where it is a helper and no entry, name it as one), keep what it computes, "
        "run your check again if you changed code it measures, and write your decision as "
        "you did when you finished. Do not start the task again. What still stands when "
        f"you hand in lands with your work and is repaired by the task `{rows_mod.DELIVERY_ROW}`"
        " later: it costs you no attempt."
    )


def delivery_note(owed: Sequence[str]) -> str:
    """What the delivery's task is told beside its card: what stands in the tree now."""
    lines = "\n".join(f"- {line}" for line in owed)
    return (
        "## What the delivery requires of the target's files and does not find\n\n"
        "Each line is one rule of the delivery and the file that does not hold to it, as "
        f"the tree is now:\n\n{lines}\n\n"
        "Put each right in the file it names. Change no rule and no result: an entry "
        "computes what its task's check measured."
    )


def check_argv(
    check: str,
    *,
    task_path: str | Path,
    tree: str,
    workdir: str | Path = "",
    vocabulary: str = "",
    gpus: int = 0,
    nodes: int = 0,
    wall: int = 0,
    tests: Sequence[str] = (),
) -> list[str] | None:
    """What follows ``gate`` for a check, by its registry id; ``None`` for no check of the gate's.

    The forms of the gate's own front door (``checks/gate_cli.py``): ``kernel --row``,
    ``deployment --deployment``, ``cpu --row --test``;
    each with the spec, the tree its pin is taken from, and a work directory of its own.
    A row a split wrote is found in the vocabulary written for it. A row's job states what
    its row states: its devices (``gpus``), its whole nodes (``nodes``) and its wall
    (``wall``), which is what the pool is asked to keep a holder for and what a holder
    near its end can still start. A deployment's devices and wall are its session's own.
    """
    name, _, what = check.partition(":")
    if name not in evidence_mod.GATED or not what:
        return None
    argv = [name, "--task", str(task_path)]
    row = name in (rows_mod.CHECK_NAME[rows_mod.OPERATOR], rows_mod.CPU_CHECK)
    if name == rows_mod.CHECK_NAME[rows_mod.OPERATOR]:
        argv += ["--row", what]
        if vocabulary:
            argv += ["--vocabulary", str(vocabulary)]
        if gpus > 1:
            argv += ["--gpus", str(gpus)]
    elif name == rows_mod.CPU_CHECK:
        argv += ["--row", what]
        for test in tests:
            argv += ["--test", str(test)]
        if gpus > 0:
            argv += ["--gpus", str(gpus)]
    else:
        argv += ["--deployment", what]
    if row and nodes > 1:
        argv += ["--nodes", str(nodes)]
    if row and wall > 0:
        argv += ["--timeout", str(wall)]
    argv += ["--tree", str(tree)]
    if workdir:
        argv += ["--workdir", str(workdir)]
    return argv


def check_command(check: str, *, cli: str = gate_cli.CLI_NAME, **places: Any) -> str:
    """The command of a check as its card gives it: ``places`` are :func:`check_argv`'s.

    Every one is the gate's own: it runs as a job of the run's pool, under the engine's
    interpreter, on a pin of the tree, and running it records its result in the run's ledger.
    """
    argv = check_argv(check, **places)
    if argv is not None:
        return f"{cli} gate {' '.join(argv)}"
    return f"`{check}` names no check this run knows: say so in a `blocked` receipt"


__all__ = [
    "ExecutionStage",
    "CHECK_NOT_PASSED",
    "Proof",
    "ROLE_BY_KIND",
    "check_argv",
    "check_command",
    "delivery_note",
    "deployments_of",
    "form_note",
    "onboarded_baseline",
    "skeleton_files",
    "test_files",
]


def _start_with_the_survey(self) -> bool:
    """Put the survey in a new run's graph; whether it did.

    A run with a graph already, or with a vocabulary, has had its survey.
    The task's ``ref`` is the target's directory, where the task list places the
    target's own entries when it makes a task for each row.
    """
    if not self.survey_first or self.vocabulary_path.exists():
        return False
    survey = graph_mod.Task(
        id=SURVEY_TASK_ID,
        kind="survey",
        goal=(
            "map the checkpoint's computation steps onto the catalog and name every other "
            "piece of work the target needs: write plan.md and vocabulary.yaml"
        ),
        budget=getattr(self, "max_attempts", graph_mod.MAX_ATTEMPTS),
        ref=str(getattr(self._preflight, "target_rel", "") or ""),
    )
    seeded = seed(self.workspace, [survey])
    if seeded:
        self.journal.event("phase 0", "the run starts with the survey")
    return seeded


#: What a survey is told of a row whose check cannot run before the model is assembled.
NEEDS_BLOCK = (
    "**A row whose check needs the assembled model says so.** If a row's check builds, "
    "loads or serves the whole model (a decode path, a graph capture, anything that "
    "constructs the model class), add `needs: model` to that row, beside its `kind`. Code "
    "then starts it after the assembly of its deployment (the first one, when the row "
    "names none) and before that deployment's session. Without it the row is a "
    "prerequisite of the assembly, and a check that needs the model cannot pass there. A "
    "row whose check needs only its own files says nothing.\n\n"
)


def _baseline_block(task_data: Mapping[str, Any]) -> str:
    """What a survey is told when the spec names an onboarded baseline; ``""`` when it names none."""
    ref = str(task_data.get("onboarded") or "").strip()
    if not ref:
        return ""
    return (
        f"**The model is onboarded already** (`onboarded: {ref}`): the target at that ref "
        "serves what it was onboarded with. Write the delta only: the rows this spec's "
        "deployments need beyond what that ref has, each named for the deployment that "
        "needs it. List nothing the baseline already holds, and no deployment of this run "
        "waits for a first one.\n\n"
    )


def _surveyor_prompt(self) -> str:
    """The survey task: map the checkpoint onto the catalog."""
    return (
        f"Workspace: {self.workspace}\n\n"
        f"Read `{self.task_path}` for the run's spec: the checkpoint, the "
        f"architecture, the deployments with their native options, and what each is "
        f"held to.\n\n"
        f"{_baseline_block(self.task_data)}"
        f"{self._surface_block}\n"
        f"{getattr(self, '_prior_art_block', '')}\n"
        f"{capture_lists_block(self.task_data)}\n"
        f"Write `{self.plan_path}` and `{self.vocabulary_path}` as the `survey` "
        f"section specifies: it gives the whole form, and a file that does not hold "
        f"to it comes back to you with what is wrong.\n\n"
        f"{NEEDS_BLOCK}" + DECISION_REMINDER
    )
