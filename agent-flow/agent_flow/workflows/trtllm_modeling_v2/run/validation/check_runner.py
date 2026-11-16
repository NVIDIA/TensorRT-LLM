# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework check execution: a deployment's session as one job, its lines in the ledger.

A contract check is a deployment's session (``checks/session.py``) or the native tests
it names (``checks/native_tests.py``). The runner submits each as one job that states
the devices it holds and its wall time, on a snapshot of the commit the checkout is at,
and writes what came back through the run's ledger (``contract_status.jsonl``): the
check's own verdict by the orchestrator, and one line for each check of a session here.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import replace
from typing import Any

from rich.markup import escape as escape_markup

from agent_flow.console import print_message

from ...checks import accuracy as accuracy_checks
from ...checks import gate_records, native_tests
from ...checks import session as session_mod
from ...checks import verdict as check_result
from ...infra.compute import pintree
from ...infra.compute.executor import (
    ACCURACY,
    CANCELLED,
    FEATURE,
    current_job_priority,
    job_priority,
    select_command_executor,
    shell_runner_for,
)
from ...infra.git import resolve_commit
from ...spec import deployments as deployments_mod
from ...spec.task_schema import assembly_units, gpus_per_node
from ..startup import datasets as datasets_mod
from ..workspace import snapshots as snapshots_mod
from . import contract as contract_mod


class _SessionLost(RuntimeError):
    """A gate measured nothing (lost to its session, none back; or cancelled): no verdict."""


class CheckRunner:
    """Execute framework-owned checks, with isolated snapshot state and shared evidence."""

    def __init__(self, workflow) -> None:
        self.workflow = workflow
        self.task_data = workflow.task_data
        self.workspace = workflow.workspace
        self.journal = workflow.journal
        self.gate_log = workflow.gate_log
        self._member_runners = {}
        self._lines_lock = threading.Lock()
        #: Benchmark -> deployment -> its passed accuracy line: what a shift is read across.
        self._scored: dict[str, dict[str, Any]] = {}
        self._session = threading.local()
        self.pin = None

    def _job_runner(self, gpus: int, *, nodes: int = 1, ranks: str = "") -> Any:
        """The runner of one check's job: under a scheduler, a job that states ``gpus`` devices.

        A session states every device its role instances hold, and the ranks of each
        (``ranks``: the pool's origin judge says "no result" for one that reported
        nothing); the wall time is the timeout each command is submitted with.
        """
        env = self.workflow._gate_env()
        if self.workflow._shell_runner_override is not None:
            return (
                self.pin.runner(self.workflow._shell_runner, None)
                if self.pin
                else self.workflow._shell_runner
            )
        want = str(gpus)
        # A session's spool starts a check of a higher class first (job_priority).
        priority = current_job_priority()
        cache = self._member_runners
        key = (want, priority, nodes, ranks)
        if key not in cache:
            executor = select_command_executor(
                env,
                tree=self.task_data.get("trtllm_repo_path"),
                # The job names the pin this check measures, when it measures one: the tree
                # named here is the checkout, which says nothing of which pin a job runs.
                pin=getattr(self, "_pin_file", None),
                gpus=want,
                priority=priority,
                nodes=nodes,
                # A job of several nodes is a session's own script: the lead runs it once.
                once=nodes > 1,
                ranks=ranks,
                note=lambda said: self.journal.event("waiting", said),
            )
            runner = shell_runner_for(executor)
            # A job on the spool names the pin and is entered into its tree on its node.
            runner = self.pin.runner(runner, executor) if self.pin else runner
            cache[key] = self._noting_session_loss(runner)
        return cache[key]

    def _gate_until_verdict(self, run, log) -> "check_result.GateResult":
        """Run one gate, and return what it says about the target; raise when it says nothing.

        A gate whose job was lost with its allocation measured nothing. The spool's
        scheduler already ran that job again once (the only re-run of a lost job in the
        flow), so what comes back here lost is "no result": it raises `_SessionLost`, the
        check has no verdict (it is missing, not failed), costs no assembler turn, and
        runs again later as a re-check. The same holds for a gate whose job was cancelled
        through the spool, and for one whose stated ranks reported no origin.

        A loss is what the gate says (`GateResult.infrastructure`) or what the runner saw
        while it ran (`_noting_session_loss`). A gate that passed is a verdict whatever
        happened along the way.
        """
        self._session.loss = ""
        result = self.pin.settle(run) if self.pin else run()
        reason = result.infrastructure or self._session.loss
        if result.passed or not reason:
            return result
        label = f"{result.name}@{result.member}" if result.member else result.name
        if CANCELLED in reason:
            self.journal.event("gate cancelled", f"{label}: {reason}")
            raise _SessionLost(f"{label} measured nothing: {reason}")
        self.journal.event("gate has no result", f"{label}: {reason}")
        print_message(
            f"[bold yellow]⚠ {escape_markup(label)} did not run to a verdict: "
            f"{escape_markup(reason)}[/bold yellow]",
            log,
        )
        raise _SessionLost(f"{label} has no result: {reason}")

    def _noting_session_loss(self, runner: Any) -> Any:
        """``runner``, remembering why each command it ran was ended by its session.

        A gate may report such a command as an ordinary failure; the note lets
        `_gate_until_verdict` tell the two apart.
        """

        def run(*args: Any, **kwargs: Any) -> Any:
            result = runner(*args, **kwargs)
            if result.infrastructure:
                self._session.loss = result.infrastructure
            return result

        return run

    def _prioritized_check(self, check: contract_mod.Check, log) -> Any:
        """:meth:`_run_contract_check`, its jobs queued at the check's class (:func:`job_priority`).

        The first deployment's session is on the way to ``onboarded``, and goes
        ahead of every other check in a session's spool.
        """
        units = assembly_units(self.task_data)
        first = bool(units) and check.unit == units[0] and check.gate == session_mod.GATE_NAME
        with job_priority(ACCURACY if first else FEATURE):
            return self._run_contract_check(check, log)

    def _run_contract_check(self, check: contract_mod.Check, log) -> Any:
        """Run one contract check and return its gate result.

        On a git checkout the check measures a pin of the commit the checkout
        is at as it starts (``_snapshots``, the run's trees). Each check of a session is
        written to the ledger as its own line, under the same state of the
        target. A check that passes adds the engines it attested to the runtime
        facts.
        """
        self._session.lines = ()
        snapshots = getattr(self.workflow, "_snapshots", None)
        if snapshots is None:
            marker = self.workflow._check_key(check)
            result = self._gate(check, log)
        else:
            result, marker = self._gate_on_snapshot(snapshots, check, log)
        self._record_lines(check, result, marker)
        return result

    def _record_lines(self, check: contract_mod.Check, result: Any, marker: str) -> None:
        """One ledger line for each check of the session ``check`` ran, in the order they ran.

        Each is recorded as measured on what the session measured: its tree, its
        marker, what it imported. A session that says nothing about one tree
        (void) leaves no line that passes.
        """
        lines = tuple(getattr(self._session, "lines", ()) or ())
        self._session.lines = ()
        status = getattr(self.workflow, "_contract_status", None)
        if not lines or status is None:
            return
        void = str(getattr(result, "void", "") or "")
        with self._lines_lock:
            for line in lines:
                stamped = replace(
                    line,
                    tree_state=str(getattr(result, "tree_state", "") or line.tree_state),
                    marker=str(getattr(result, "marker", "") or ""),
                    imported=tuple(getattr(result, "imported", ()) or ()),
                    void=void,
                    passed=bool(line.passed) and not void,
                )
                status().record(check.unit, stamped, marker=marker, inputs=check.inputs)
                self.gate_log.record(stamped)
                self.journal.event("check", stamped.summary())
                if stamped.name.startswith("accuracy[") and stamped.passed:
                    self._note_shift(check.unit, stamped)

    def _note_shift(self, unit: str, line: Any) -> None:
        """Once every deployment scored on a benchmark has a passing line, read them together.

        Each can clear its own floor while all of them sit more than one SE under
        their anchors: what a defect in shared code looks like. That is recorded as
        a finding and refused at delivery (``accuracy.floor.acknowledge`` accepts it
        by the benchmark's name). A deployment judged against another's own score
        has no anchor of its own to sit under, and is left out.
        """
        benchmark = line.name[len("accuracy[") : -1]
        owed = [
            d.name
            for d in deployments_mod.resolve(self.task_data)
            if d.benchmark(benchmark) is not None and not d.relative_to
        ]
        if unit not in owed:
            return
        scored = self._scored.setdefault(benchmark, {})
        scored[unit] = replace(line, member=unit)
        if set(owed) <= set(scored):
            finding = accuracy_checks.note_systematic_shift(
                gate_records.records_for(self.workspace),
                [scored[name] for name in owed],
                self.task_data.get("accuracy") or {},
                benchmark,
            )
            if finding is not None:
                self.journal.event("finding", finding.describe())

    def _gate_on_snapshot(self, snapshots: Any, check: contract_mod.Check, log) -> tuple[Any, str]:
        """``check``'s result on a pin of the checkout's head, and the marker it measured.

        ``snapshots`` is the run's trees: they pin the head (the commit and the build
        record of its libraries) and give that pin as a read-only tree for the check.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        head = resolve_commit(repo)
        if head is None:
            why = f"git could not name the commit {repo} is at, so there was nothing to pin"
            return snapshots_mod.void_result(check.gate, check.member, why), ""
        marker = self.workflow._check_key(check, head)
        try:
            held = snapshots.pinned(head)
            with snapshots.at(held) as tree:
                pin = snapshots_mod.CheckTree(
                    commit=head,
                    tree=tree,
                    marker=marker,
                    uncommitted=tuple(self.workflow._uncommitted_gate_inputs(repo) or ()),
                    snapshots=snapshots,
                )
                return self._on_snapshot(pin, held.file)._gate(check, log), marker
        except pintree.PinError as exc:
            result = snapshots_mod.void_result(check.gate, check.member, str(exc))
            self.journal.event("gate", result.summary())
            return result, marker

    def _on_snapshot(self, pin: snapshots_mod.CheckTree, pin_file: Any = None) -> "CheckRunner":
        """The runner as one check sees it: the target is ``pin``'s snapshot.

        A shallow copy, so the journal, the gate log, the lost-session record
        and the rest of the run stay shared. What differs is where the target
        is and what a result says about it:

        - the spec names the snapshot (``CheckTree.spec``), so the quickstart,
          the prompts' images and the anchor tables are the commit's;
        - every runner the check builds (`_member_runner`) names the snapshot
          as its jobs' tree and runs each command from inside it with it first
          on ``PYTHONPATH``, so every rank of every leg imports the commit's
          package, and notes what each command imported;
        - every result comes back stamped with the commit and what was
          imported (``CheckTree.settle``), a void one measured once more.
        """
        view = copy.copy(self)
        view.task_data = pin.spec(self.task_data, pin.snapshots.tracked(pin.commit))
        view._member_runners = {}
        view.pin = pin
        #: The pin's file, which every job of this check names.
        view._pin_file = str(pin_file) if pin_file else None
        return view

    # ----------------------------------------------------------------- checks run
    def _relative_scores(
        self, deployment: deployments_mod.Deployment, commit: str
    ) -> dict[str, float]:
        """The scores ``deployment``'s relative benchmarks are held to: its ``relative_to``'s, on ``commit``.

        Read from the ledger, where that deployment's session left one line per
        benchmark under its own key (its task ran first: :attr:`~..contract.Check.after`).
        A benchmark with none under that session's key on this commit is blocked by
        its session before the load; the other deployment is never loaded here.
        """
        base = next(
            (
                d
                for d in deployments_mod.resolve(self.task_data)
                if d.name == deployment.relative_to
            ),
            None,
        )
        status = getattr(self.workflow, "_contract_status", None)
        if base is None or status is None:
            return {}
        marker = self.workflow._check_key(
            contract_mod.Check(base.name, session_mod.GATE_NAME, base.shape), commit
        )
        scores: dict[str, float] = {}
        for benchmark in deployment.benchmarks:
            line = contract_mod.Check(base.name, f"accuracy[{benchmark.name}]", base.shape)
            verdict = contract_mod.verdict_on(status(), line, marker)
            if verdict is not None and verdict.passed and verdict.measured is not None:
                scores[benchmark.name] = float(verdict.measured)
        return scores

    def _dataset_paths(self, deployment: deployments_mod.Deployment) -> dict[str, str]:
        """Where each of ``deployment``'s benchmarks' datasets was staged (``startup/datasets.py``).

        A dataset still being staged is waited for; one that could not be is
        left out, and the benchmark looks for it as it would without.
        """
        staged = datasets_mod.status(self.workspace)
        paths: dict[str, str] = {}
        for benchmark in deployment.benchmarks:
            if benchmark.name in staged and not benchmark.dataset_path:
                path = datasets_mod.wait(self.workspace, benchmark.name)
                if path:
                    paths[benchmark.name] = path
        return paths

    def _await_allocation(self) -> None:
        """Wait for the start's allocation probe: no session loads a model before it answered."""
        from ..startup import bootstrap

        bootstrap._await_substrate(self.workflow)

    def _gate(self, check: contract_mod.Check, log) -> Any:
        """Run ``check`` on the target this workflow sees, journal it, and return its result."""
        deployment = deployments_mod.named(self.task_data, check.unit)
        env = self.workflow._gate_env()
        workdir = self.workspace / "gates"
        commit = self.pin.commit if self.pin else ""
        if check.gate == session_mod.GATE_NAME:
            # The allocation probe of the start may still run beside the first turns.
            self._await_allocation()
            try:
                placement = session_mod.place(self.task_data, deployment)
                gpus, nodes = placement.gpus, placement.nodes
            except session_mod.WiderThanOneNode:
                # The session itself says why, before any load.
                gpus, nodes = deployment.ranks, 1
            runner = self._job_runner(gpus, nodes=nodes, ranks=session_mod.stated_ranks(deployment))

            def run() -> Any:
                found = session_mod.run_deployment(
                    deployment,
                    env,
                    task_data=self.task_data,
                    workdir=workdir,
                    runner=runner,
                    workspace=self.workspace,
                    relative_scores=self._relative_scores(deployment, commit),
                    dataset_paths=self._dataset_paths(deployment),
                )
                self._session.lines = found.lines
                return found.result

        elif check.gate == native_tests.GATE_NAME:
            # Its tests are processes of one node: its devices at most.
            one_node = gpus_per_node(self.task_data) or deployment.ranks
            runner = self._job_runner(min(deployment.ranks, one_node))

            def run() -> Any:
                return native_tests.run_native_tests(
                    env,
                    task_data=self.task_data,
                    tests=deployment.tests,
                    workdir=workdir / deployment.name,
                    runner=runner,
                    member=check.member,
                    env=deployment.env,
                    timeout=deployment.timeout,
                )

        else:
            raise ValueError(f"contract check {check.key} names no check this workflow runs")
        result = self._gate_until_verdict(run, log)
        self.gate_log.record(result)
        self.journal.event("gate", result.summary())
        if not result.passed:
            print_message(f"[bold yellow]↻ {escape_markup(result.summary())}[/bold yellow]", log)
            if result.stdout_tail.strip():
                self.journal.section(
                    f"`{check.key}` did not pass: its last lines",
                    f"```\n{result.stdout_tail.rstrip()}\n```",
                )
        return result
