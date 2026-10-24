# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The planned run: the workflow's services handed to :class:`~..orchestrator.GraphOrchestrator`.

Entered after preflight. The planner turn, each task's worker turn and each
contract check are this module's; deciding what runs and what counts is the
orchestrator's. Once the contract holds, delivery verifies the trees on disk.

Every contract check measures a snapshot of the commit the checkout is at
when it starts (``run/snapshots.py``), never the checkout, which integrations
move while checks run. Its verdict names that commit, is recorded under its
marker, and is void when a command imported any other tree.

A task that finishes leaves its receipt (``run/receipts.py``), and a task
that starts carries the receipt of each task it depends on. A gate that
passes adds the engines it attested to ``runtime_facts.yaml``, which the
planner's and every worker's prompt carries.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ...agents.prompts.tags import tag_section, task_tags
from ...checks import envelope_boot, logit_agreement, serve_gate
from ...checks import gates as gates_mod
from ...infra.committer import CommitPolicy
from ...infra.driver import reload as reload_mod
from ...infra.execution import (
    CheckAdmission,
    CommandSpec,
    select_command_executor,
    startable_checks,
)
from ...infra.execution import priority as priority_mod
from ...infra.execution.capacity import held_gpus, offsite_kinds
from ...infra.execution.priority import job_priority
from ...infra.vcs import resolve_commit
from ...onboard import promotion as promotion_mod
from ...onboard.vocabulary import QueueItem, VocabularyError, load_vocabulary, queue_item_from_ref
from ...spec.task_schema import BUDGET_FIELD, assembly_units, members_for
from ...tune import stage as tune_stage
from .. import contract as contract_mod
from .. import drain as drain_mod
from .. import graph as graph_mod
from .. import inbox as inbox_mod
from .. import metrics as metrics_mod
from .. import receipts as receipts_mod
from .. import regression as regression_mod
from .. import runtime_facts
from .. import snapshots as snapshots_mod
from ..integration import (
    INTEGRATED_FILENAME,
    INTEGRATED_KINDS,
    REBASE_RETRIES,
    SCRATCH_KINDS,
    Integration,
    rebase_block,
    repair_block,
    soft_dependency_block,
    worktree_block,
)
from ..orchestrator import (
    CHECKED,
    FINISHED,
    GraphOrchestrator,
    PlannerAnswer,
    PlannerTurn,
    Services,
)
from ..progress import (
    collect_decision_file,
    current_decision,
    decision_file_instruction,
    decision_value,
    latest_entry,
    mark_decision_dispatch,
    parse_engine_defect,
    uncertified_report,
)
from ..state import WorkflowState
from .common import remote_turns
from .kernel_writer import FAILED_RECEIPT

PLANNER_ROLE = "planner"
PROPOSED_GRAPH = "graph.proposed.yaml"
TASKS_DIRNAME = "tasks"
#: How many of the latest measured changes the planner's request lists.
PLANNER_FACT_CHANGES = 8

#: The role each task kind's turn runs as.
ROLE_BY_KIND = {
    "survey": "surveyor",
    "kernel": "kernel_writer",
    "assemble": "assembler",
    "feature": "assembler",
    "integrate": "assembler",
    "investigate": "assembler",
    "tune": "tuner",
}
#: How long a regression check's test job may run.
REGRESSION_TIMEOUT_S = 2 * 3600


class PlannedStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    _turn_lock = threading.Lock()
    #: Guards the fix tasks for engine defects kernel writers reported (``_defect_specs``),
    #: queued until the orchestrator takes them.
    _defect_lock = threading.Lock()

    # ------------------------------------------------------------------ entry
    def _run_planned(self, state: WorkflowState, log) -> bool:
        print_rule("[bold magenta]planned run[/bold magenta]", log)
        contract = self._contract or contract_mod.compile_contract(
            self.task_data, self._gate_env().world_size
        )
        self._accuracy_by_unit: dict[str, dict[str, Any]] = {}
        self._accuracy_owed: dict[str, list[str]] = {}
        for check in contract.checks:
            if check.gate.startswith("accuracy["):
                self._accuracy_owed.setdefault(check.unit, []).append(check.member)
        self._planned_state = state
        tune = self._tune_requested()
        units = assembly_units(self.task_data)
        core_checks = [
            c.key for c in contract.required() if units and c.unit == units[0] and c.tier == "inner"
        ]
        total = max(len(state.devices), self._gate_env().world_size, 1)
        self._integration = self._make_integration(state)
        self._snapshots = self._make_snapshots()
        # How many checks run at once follows the sessions' free devices.
        admission = CheckAdmission(startable_checks)
        # Set once full support holds: a running campaign wraps up.
        self._campaign_stop = threading.Event()
        services = Services(
            plan=lambda turn: self._planner_turn(turn, contract),
            work=self._work_task,
            check=lambda check: self._prioritized_check(check, log),
            free_gpus=lambda: total,
            checkpoint=lambda: self._checkpoint(state),
            log=lambda message: self.journal.event("planned", message),
            marker=self._target_marker,
            finished=self._task_finished,
            landed=self._landed_unfinished,
            admit=lambda check: self._admit_check(admission, check),
            reload=(
                (lambda record: self._reload_process(record, log))
                if getattr(self, "reexec", None) is not None
                else None
            ),
            correct=self._mark_correct,
            bringup=getattr(self, "_mark_bringup", None),
            cells=getattr(self, "_mark_cells", None),
            budget=lambda: self._budget_spent(total),
            promotion=lambda graph: promotion_mod.plan(
                graph.tasks,
                self.task_data["trtllm_repo_path"],
                self._preflight.target_rel,
                keep=promotion_mod.keep_local(self.task_data),
            ),
            delivery_files=lambda: self._delivery_failures(state),
            delivery_task=self._delivery_task,
            stop_campaign=lambda reason: self._campaign_stop.set(),
            defect_tasks=self._take_defect_tasks,
            stage_notes=self._take_stage_notes,
            narrow=getattr(self, "_narrowed", None),
            holds=getattr(self, "_verdict_holds", None),
            regression=self._run_regression_check,
        )
        # Shared code is followed by regression checks in a run that starts on the rule:
        # one with landed work began without them.
        self._own_paths = self._regression_paths()
        orchestrator = GraphOrchestrator(
            services,
            contract,
            self._contract_status(),
            self.workspace,
            allowed_paths=CommitPolicy.from_task(
                self.task_data, workspace=self.workspace
            ).allowed_paths,
            max_gpus=total,
            max_parallel=self.max_parallel_tasks,
            max_parallel_checks=admission.cap,
            revision_cap=self.revision_cap,
            event_window_s=self.event_window_s,
            required_kinds=("tune",) if tune else (),
            start_after={"tune": core_checks} if core_checks else {},
            own_paths=self._own_paths,
            repo_root=self.task_data.get("trtllm_repo_path") or "",
        )

        # A tune task whose jobs all run on the perf sessions holds none of the run's.
        offsite = offsite_kinds(self.task_data.get("perf"), self._gate_env())

        def held(task: graph_mod.Task) -> int:
            # A running task waiting on its soft dependencies or running its checks
            # holds none: its checks are admitted by the sessions' devices of their own.
            return 0 if orchestrator.agentless(task.id) else held_gpus(task, offsite)

        def free_gpus() -> int:
            return self._task_capacity(total, orchestrator.graph, held)

        services.free_gpus = free_gpus
        services.held_gpus = held
        self._inflight = orchestrator.inflight
        self._soft_dependencies = orchestrator.soft_dependencies
        self._mark_integrated = orchestrator.mark_integrated
        # A campaign's turns, and the side turns, take only the agent slots the
        # foreground work leaves idle.
        gate = getattr(self, "_agent_gate", None)
        if gate is not None:
            gate.background_limit = orchestrator.idle_agent_slots
            gate.side_limit = orchestrator.idle_agent_slots
        self._record_restart(orchestrator.restart)
        # Beside the planner's first turn, as the run's other early work is.
        self._start_primers()
        with self._managing_sessions():
            try:
                verdict = orchestrator.run()
            finally:
                # No foreground work is left to yield to: the closing turns run at once.
                if gate is not None:
                    gate.background_limit = gate.side_limit = None
        marker = self._target_marker()
        evaluation = contract_mod.evaluate(
            contract,
            self._contract_status(),
            marker=marker,
            holds=getattr(self, "_verdict_holds", None),
        )
        self.journal.section(
            f"planned run: {verdict}", contract_mod.standing(evaluation, self.workspace, marker)
        )
        if verdict.startswith("stopped"):
            # Phase 0 found the run cannot go on here: nothing is closed and
            # nothing is learned, and the same command resumes once it is fixed.
            print_message(
                f"[bold yellow]⚠ planned run {escape_markup(verdict)}; the run stays open: "
                f"re-run the same command once the cause is fixed[/bold yellow]",
                log,
            )
            return False
        if verdict != "delivered":
            self._close_run(state, log, verdict=verdict)
            return False
        return self._run_deliver_phase(state, log)

    # ----------------------------------------------------------------- reload
    def _hold(self, kind: str, what: str) -> contextlib.AbstractContextManager:
        """``<kind> <what>`` in flight for a reload: waited for, or held back while it drains.

        Outside the planned loop there is no reload, and nothing is held.
        """
        inflight = getattr(self, "_inflight", None)
        return inflight.hold(kind, what) if inflight is not None else contextlib.nullcontext()

    def _record_restart(self, record: dict) -> None:
        """Journal what the stop before this start cost, and add it to ``restarts.jsonl``.

        Named too: the check jobs the previous process left in the spools, and
        on the external transport the turns it left: still with the driver,
        answered and never read, or answered for a task that had no outcome yet.
        """
        previous = record.get("previous") or {}
        if previous.get("started_at"):
            since = float(previous["started_at"])
            env = self.environment
            spools = (
                {env.spool, *(spool for _, spool, _ in env.sessions)} - {None, ""}
                if env is not None
                else set()
            )
            record["jobs"] = drain_mod.jobs_left(
                sorted(spools), pid=previous.get("pid"), since=since,
                before=drain_mod.PROCESS_STARTED_AT,
            )  # fmt: skip
            if remote_turns(self) is not None:
                record["turns"] = drain_mod.turns_left(
                    self.workspace,
                    since=since,
                    before=drain_mod.PROCESS_STARTED_AT,
                    tasks=record.get("tasks_reset") or (),
                )
        if record["after"] != drain_mod.AFTER_FIRST:
            self.journal.event(f"restart ({record['after']})", drain_mod.describe_restart(record))
        drain_mod.append_restart(self.workspace, record)

    def _reload_process(self, record: dict, log) -> None:
        """Replace this process with the code now at the checkout; returns only if it could not."""
        line = reload_mod.describe(record)
        self.journal.event("reload", line)
        self._checkpoint(self._planned_state)
        print_message(f"[bold cyan]→ reloading: {escape_markup(line)}[/bold cyan]", log)
        stream = getattr(log, "file", None)
        if stream is not None:
            stream.flush()
        self.reexec()

    # --------------------------------------------------------------- dispatch
    def _next_turn(self, role: str, unit: str) -> int:
        with self._turn_lock:
            turns = self.__dict__.setdefault("_turns_by_key", {})
            key = f"{role}:{unit}"
            turns[key] = turns.get(key, 0) + 1
            return turns[key]

    def _task_finished(self, task: graph_mod.Task) -> None:
        """Mark ``task``'s answered turns consumed once it has an outcome or is retried.

        A relaunch adopts only the answered turn of a task with no outcome yet; a
        consumed turn is never adopted, so a retry is always asked afresh. A
        ``check`` task that passed, which no worker ran, leaves its receipt here.
        """
        remote = remote_turns(self)
        if remote is not None:
            remote.consume([task.id])
        if task.kind == graph_mod.CHECK_KIND and task.status == "done":
            cleared = self.__dict__.get("_regression_cleared", {}).get(task.id, "")
            if task.regression:
                note = "the package tests its changes to shared code can affect passed"
            else:
                note = f"its checks passed, with no agent turn: {', '.join(task.satisfies)}"
            try:
                receipts_mod.write(
                    self.workspace / TASKS_DIRNAME,
                    receipts_mod.build(
                        task.id,
                        task.kind,
                        decision=CHECKED,
                        note=note,
                        attempt=task.attempts,
                        cleared=cleared,
                    ),
                )
            except OSError as exc:
                self.journal.event("receipt not kept", f"{task.id}: {exc}")

    def _dispatch_task(self, role: str, unit: str, what: str, prompt: str) -> str | None:
        """One turn with its unit and turn number named; returns its decision.

        **A resume adopts a task's turn the previous process never finished
        with.** On the external transport, a task's first dispatch in this
        process adopts the answer the driver gave a previous process and
        collects the decision where that turn wrote it, instead of asking
        again; a later dispatch of the same task (a rebase, say) is always
        asked. Every answered turn of a task that has its outcome is consumed
        first, so a task the planner reopens is asked afresh.
        """
        turn = self._next_turn(role, unit)
        with self._turn_lock:
            dispatched = self.__dict__.setdefault("_tasks_dispatched", set())
            first = (role, what) not in dispatched
            dispatched.add((role, what))
        transport = remote_turns(self)
        if transport is not None:
            try:
                tasks = graph_mod.Graph.load(self.workspace / graph_mod.GRAPH_FILENAME).by_id()
            except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
                # Absent, or read mid-write: nothing is consumed or adopted this time.
                tasks = {}
            transport.consume(
                t.id for t in tasks.values() if t.status in ("done", "failed", "abandoned")
            )
            task = tasks.get(what)
            answered = None
            if first and task is not None and task.status in ("pending", "running"):
                answered = transport.adopt_answered(
                    role, what, lambda path: decision_value(path, role) is not None
                )
            if answered is not None:
                print_message(
                    f"[bold cyan]→ {role} {escape_markup(what)}: adopting {answered.request_id}, "
                    f"answered for a previous process, instead of dispatching again[/bold cyan]",
                    None,
                )
                self.journal.event(
                    "decision adopted on resume",
                    f"{role} {what}: {answered.request_id} {answered.decision_file}",
                )
                with self._turn_lock:
                    files = self.__dict__.setdefault("_decision_files", {})
                    files[what] = Path(answered.decision_file)
                return collect_decision_file(
                    self._progress_ctx,
                    self.workspace,
                    role,
                    turn,
                    unit=unit,
                    path=answered.decision_file,
                )
            if first and task is None:
                unread = transport.unread_answer(role, what)
                if unread is not None:
                    kept = previous_proposal(self.workspace / PROPOSED_GRAPH)
                    if role != PLANNER_ROLE or not kept.is_file():
                        kept = None
                    prompt = f"{prompt}\n\n{_unread_answer_note(unread, kept)}\n"
                    self.journal.event(
                        "answer carried over",
                        f"{role} {what}: {unread.request_id} {unread.decision_file}",
                    )
        context = self._context_block()
        if context:
            prompt = f"{prompt}\n\n{context}\n"
        target = mark_decision_dispatch(self.workspace, role, turn, unit=unit)
        with self._turn_lock:
            self.__dict__.setdefault("_decision_files", {})[what] = Path(target)
        prompt = f"{prompt}\n\n{decision_file_instruction(target)}"
        report = self.journal.record_dispatch(role, what, prompt)
        result = self._run_turn(role, what, prompt, iteration=turn, unit=unit, work=what)
        self.journal.record_result(report, result)
        return collect_decision_file(self._progress_ctx, self.workspace, role, turn, unit=unit)

    # ------------------------------------------------------- regression checks
    def _run_regression_check(self, task: graph_mod.Task) -> list[str]:
        """Run one regression check, with no agent turn; what failed, one line each.

        On a read-only snapshot of the commit the checkout is at, as one job on
        the run's devices (``run/regression.py``). The head it finds clean is the
        receipt's ``cleared``, the next check's base.
        """
        repo = Path(str(self.task_data.get("trtllm_repo_path") or ""))
        head = (self._git(str(repo), "rev-parse", "HEAD") or "").strip()
        if not head:
            return [f"{repo} has no commit to check"]
        snapshots = getattr(self, "_snapshots", None)
        env = self._gate_env()

        def tree_at(commit: str) -> contextlib.AbstractContextManager[Path]:
            return (
                snapshots.lease(commit) if snapshots is not None else contextlib.nullcontext(repo)
            )

        def run_job(command: str, tree: Path) -> tuple[int, str]:
            executor = select_command_executor(
                env,
                name_prefix=f"regression-{task.id}",
                tree=tree,
                gpus="all",
                priority=priority_mod.FEATURE,
            )
            done = executor.execute(
                CommandSpec(argv=command, cwd=str(tree), timeout_s=REGRESSION_TIMEOUT_S)
            )
            return done.returncode, (done.stdout + done.stderr)[-2000:]

        environment = self.task_data.get("environment")
        python = str((environment or {}).get("python") or "python3")
        result = regression_mod.run_regression_check(
            workspace=self.workspace,
            repo=repo,
            head=head,
            tree_at=tree_at,
            run_job=run_job,
            python=python,
            name=task.id,
        )
        self.journal.event("regression", f"{task.id}: {result.summary()}")
        if result.passed:
            with self._turn_lock:
                self.__dict__.setdefault("_regression_cleared", {})[task.id] = head
        return result.failures

    # ------------------------------------------------------ worktrees, devices
    def _make_integration(self, state: WorkflowState) -> Integration | None:
        """Per-task worktrees on a git checkout; on anything else tasks share the tree."""
        repo = Path(str(self.task_data.get("trtllm_repo_path") or ""))
        if not self.task_worktrees or not (repo / ".git").exists():
            self.journal.event("integration", f"tasks share the checkout {repo}")
            return None
        identity = getattr(
            CommitPolicy.from_task(self.task_data, workspace=self.workspace), "identity", None
        )
        env = {}
        if identity is not None:
            env = {"GIT_COMMITTER_NAME": identity.name, "GIT_COMMITTER_EMAIL": identity.email}
        return Integration(
            repo,
            self.workspace,
            f"mv2/{state.slug or 'run'}",
            env,
            log=lambda message: self.journal.event("integration", message),
        )

    def _make_snapshots(self) -> snapshots_mod.Snapshots | None:
        """Snapshots of the checkout's commits for the checks to measure; on a git checkout only.

        Each is sparse with a task worktree's cone and the directories of the
        files the gates read. What an earlier process left that no check
        needs any more goes now.
        """
        from ..workflow import anchor_tables, gate_inputs

        repo = Path(str(self.task_data.get("trtllm_repo_path") or ""))
        if not (repo / ".git").exists():
            self.journal.event("snapshots", f"checks measure {repo} as it is: not a git checkout")
            return None
        snapshots = snapshots_mod.Snapshots(
            repo,
            self.workspace / snapshots_mod.SNAPSHOTS_DIRNAME,
            cone_paths=gate_inputs(self.task_data, repo) | set(anchor_tables(self.task_data, repo)),
            log=lambda message: self.journal.event("snapshots", message),
        )
        snapshots.prune()
        return snapshots

    def _task_capacity(
        self,
        total: int,
        graph: graph_mod.Graph,
        held: Callable[[graph_mod.Task], int] = lambda task: task.gpus,
    ) -> int:
        """Devices left for new tasks: the live sessions' devices less what running tasks hold.

        A task holds its share of the sessions, ``held(task)``
        (``capacity.held_gpus``), whatever its gate jobs hold at the moment.
        ``total`` stands in for the sessions when none is live.
        """
        capacity = self._spool_capacity()
        if capacity is None:
            capacity = total
        return capacity - sum(held(t) for t in graph.tasks if t.status == "running")

    def _budget_spent(self, total: int) -> str:
        """Why the run is past its ``budget``; ``""`` within it, or without one.

        Its device-hours are the live sessions' devices, else ``total``, for
        the hours since the run started (``run/metrics.py``).
        """
        budget = self.task_data.get(BUDGET_FIELD)
        if not budget:
            return ""
        devices = self._spool_capacity()
        return metrics_mod.budget_spent(
            budget,
            opened=self.journal.opened_at(),
            devices=total if devices is None else devices,
            now=time.time(),
        )

    def _spool_capacity(self) -> int | None:
        """Devices in service on the live spool schedulers, or None without a live one."""
        env = self._gate_env()
        spools = {env.spool, *(spool for _, spool, _ in env.sessions)} - {None, ""}
        capacity, live = 0, False
        for spool in spools:
            try:
                status = json.loads((Path(spool) / "status.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if time.time() - float(status.get("alive_epoch") or 0) > 300:
                continue
            live = True
            devices = set(status.get("devices") or []) - set(status.get("quarantined") or [])
            capacity += len(devices)
        return capacity if live else None

    # ---------------------------------------------------------------- planner
    def _regression_paths(self) -> tuple[str, ...]:
        """The target's own paths when regression checks follow shared code in this run; none otherwise.

        It binds a run whose graph has no finished task yet: a run already under
        way keeps the rules it was planned under.
        """
        current = self.workspace / graph_mod.GRAPH_FILENAME
        if current.exists():
            try:
                graph = graph_mod.Graph.load(current)
            except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
                return ()
            if any(t.status == "done" for t in graph.tasks):
                return ()
        preflight = getattr(self, "_preflight", None)
        return graph_mod.own_paths(str(getattr(preflight, "target_rel", "") or ""))

    def _planner_turn(self, turn: PlannerTurn, contract: contract_mod.Contract) -> PlannerAnswer:
        proposed = self.workspace / PROPOSED_GRAPH
        current = self.workspace / graph_mod.GRAPH_FILENAME
        _set_aside(proposed)
        lines = [
            f"Task: `{self.task_path}`",
            f"Contract: `{self.workspace / contract_mod.CONTRACT_FILENAME}`",
            f"Verdicts: `{self.workspace / contract_mod.STATUS_FILENAME}`",
            f"Current graph: `{current}`" if current.exists() else "There is no graph yet.",
            f"Write the next graph to `{proposed}`.",
            "",
            "## Contract standing",
            "",
            contract_mod.standing(turn.evaluation, self.workspace, turn.marker),
        ]
        facts = runtime_facts.render(self.workspace, changes=PLANNER_FACT_CHANGES)
        if facts:
            lines += ["", facts]
        own = getattr(self, "_own_paths", ())
        if own:
            lines += [
                "",
                "## Shared code",
                "",
                "The target's own paths: "
                + ", ".join(f"`{p}`" for p in own)
                + ". Any other path a task touches is shared code; the orchestrator follows "
                "every change to it with a regression check of its own, which you leave as it is.",
            ]
            if turn.regression:
                lines += ["", turn.regression]
        messages = [e for e in turn.events if inbox_mod.is_message(e)]
        if messages:
            lines[:0] = [
                "## Messages from the operator: act on these first",
                "",
                *[f"- {e}" for e in messages],
                "",
                "Each is an instruction about the work: answer it in this turn's graph and "
                "say in your `summary` what you did about it. None relaxes a check.",
                "",
            ]
        events = [e for e in turn.events if not inbox_mod.is_message(e)]
        if events:
            lines += ["", "## Since your last turn", "", *[f"- {e}" for e in events]]
        if turn.refusals:
            lines += ["", "## Your last graph was refused", "", *[f"- {r}" for r in turn.refusals]]
        decision = self._dispatch_task(PLANNER_ROLE, "plan", "planner", "\n".join(lines))
        entry = latest_entry(self.progress_path, PLANNER_ROLE) or {}
        summary = str(entry.get("summary", ""))
        if not proposed.exists():
            return PlannerAnswer(decision or "", summary, None)
        try:
            graph = graph_mod.Graph.load(proposed)
        except (yaml.YAMLError, TypeError, ValueError, KeyError, AttributeError) as exc:
            # The planner's own mistake: refused and sent back, like an inadmissible graph.
            problem = f"{proposed.name} could not be read: {type(exc).__name__}: {exc}"
            return PlannerAnswer(decision or "", summary, None, problem=problem)
        return PlannerAnswer(decision or "", summary, graph)

    # ---------------------------------------------------------------- workers
    def _envelope(self, task: graph_mod.Task, receipts: list[dict] | None = None) -> Path:
        """``tasks/<id>.json``: what the worker is asked, in one file.

        With the receipt of each task it depends on (``receipts``), when any.
        """
        path = self.workspace / TASKS_DIRNAME / f"{task.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = task.as_dict()
        if receipts:
            data["receipts"] = receipts
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return path

    def _work_task(self, task: graph_mod.Task) -> str:
        """Run ``task``'s worker and return its decision.

        The decision is the first line; anything after it is a note the
        planner sees with the task's event. A finished task's receipt is kept
        for the tasks that depend on it.
        """
        before = self._integrated_tip(task.id)
        decision = self._work_task_turn(task)
        if task.kind in INTEGRATED_KINDS and decision.partition("\n")[0] not in FINISHED:
            tip = self._integrated_tip(task.id)
            if tip and tip != before:
                with self._turn_lock:
                    self.__dict__.setdefault("_landed_attempts", {})[task.id] = task.attempts
        self._keep_receipt(task, decision)
        return decision

    def _integrated_tip(self, task_id: str) -> str:
        """The branch tip the integration last fast-forwarded in for ``task_id``; "" for none."""
        integration = getattr(self, "_integration", None)
        if integration is None:
            return ""
        try:
            tips = json.loads((integration.root / INTEGRATED_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return str(tips.get(task_id) or "") if isinstance(tips, dict) else ""

    def _landed_unfinished(self, task: graph_mod.Task) -> bool:
        """Whether ``task``'s latest attempt landed commits though its turn did not finish.

        A failed receipt's guard tests do (``_land_failed_receipt``): they are a
        change to the code they guard like any other, and a regression check
        covers them.
        """
        return self.__dict__.get("_landed_attempts", {}).get(task.id) == task.attempts

    def _work_task_turn(self, task: graph_mod.Task) -> str:
        """The worker, in its own worktree when its kind changes the tree; then integration.

        A finished task whose rebase onto the checkout conflicts keeps its
        worktree: its role is asked again, up to ``REBASE_RETRIES`` times, to
        rebase onto the head it met, resolve, re-test and hand back, and its
        branch is integrated again. Only then does it end ``INTEGRATION_FAILED``.
        """
        integration = getattr(self, "_integration", None)
        kinds = INTEGRATED_KINDS | SCRATCH_KINDS
        if integration is None or task.kind not in kinds:
            return self._after_soft_dependencies(task, self._work_task_body(task))
        with self._hold("worktree", task.id):
            tree = integration.open(
                task.id, task.touches, full=task.exclusive, attempt=task.attempts
            )
        block = worktree_block(
            tree,
            integration.repo,
            str(self.workspace / "bin" / "commit.sh"),
            scratch=task.kind in SCRATCH_KINDS,
        )
        decision = self._work_in_tree(task, block)
        if task.kind not in INTEGRATED_KINDS:
            return decision
        decision = self._after_soft_dependencies(task, decision)
        attempts = 0
        while decision.partition("\n")[0] in FINISHED:
            with self._hold("integration", task.id):
                outcome = integration.integrate(task.id)
                self.journal.event("integration", f"{task.id}: {outcome.detail}")
                if outcome.merged:
                    integration.close(task.id)
                    # How many commits went in: the receipt lists them.
                    self.__dict__.setdefault("_merged", {})[task.id] = outcome
                    mark = getattr(self, "_mark_integrated", None)
                    if mark is not None:
                        mark(task.id, task.attempts)
            if outcome.merged:
                return decision
            if not (outcome.base or outcome.repairable) or attempts == REBASE_RETRIES:
                return f"INTEGRATION_FAILED\n{outcome.detail}"
            attempts += 1
            role = ROLE_BY_KIND.get(task.kind, task.kind)
            if outcome.base:
                self.journal.event(
                    "integration",
                    f"{task.id}: back to its {role} to rebase onto {outcome.base[:10]} and "
                    f"resolve ({attempts} of {REBASE_RETRIES})",
                )
                note = rebase_block(outcome, tree)
            else:
                self.journal.event(
                    "integration",
                    f"{task.id}: back to its {role} to put its worktree right "
                    f"({attempts} of {REBASE_RETRIES}): {outcome.detail.splitlines()[0]}",
                )
                note = repair_block(outcome)
            decision = self._work_in_tree(task, block, note)
        return decision

    def _after_soft_dependencies(self, task: graph_mod.Task, decision: str) -> str:
        """``decision``, once each soft dependency of a finished ``task`` has integrated or ended.

        Its integration, and the checks after it, then read a head that holds
        what landed. A dependency that ended without landing is named in a
        line of its own after the decision: the planner's event and the
        task's receipt carry it. Outside the planned loop nothing is waited
        for; a loop that ends first leaves the task unfinished.
        """
        wait = getattr(self, "_soft_dependencies", None)
        if not task.depends_on_soft or wait is None or decision.partition("\n")[0] not in FINISHED:
            return decision
        ended = wait(task.id)
        if ended is None:
            return "the run ended before the task's soft dependencies landed"
        missed = sorted(
            dep for dep, status in ended.items() if status not in ("done", "integrated")
        )
        self.journal.event(
            "soft dependencies",
            f"{task.id}: integrates once "
            + ", ".join(f"{dep} {status}" for dep, status in ended.items()),
        )
        if not missed:
            return decision
        return (
            f"{decision}\nsoft dependencies {', '.join(f'{dep} ({ended[dep]})' for dep in missed)} "
            f"ended without landing: this task integrated without their change"
        )

    def _work_in_tree(self, task: graph_mod.Task, block: str, note: str = "") -> str:
        """``task``'s worker, told its worktree (``block``) and, on a rebase, what to do (``note``)."""
        self._tls.worktree = block
        try:
            item = self._queue_item(task) if note and task.kind == "kernel" else None
            if item is not None:
                # A kernel writer given a note is dispatched afresh rather than handed the
                # decision already on disk: that decision is the work to rebase.
                self._run_one_kernel_writer(item, note, task=task)
                return self._kernel_writer_decision_for(item) or ""
            if note:
                self._tls.worktree = f"{block}\n\n{note}"
            return self._work_task_body(task)
        finally:
            self._tls.worktree = ""

    def _work_task_body(self, task: graph_mod.Task) -> str:
        receipts = self._dependency_receipts(task)
        envelope = self._envelope(task, receipts)
        if task.kind == "tune":
            repo = str(self.task_data.get("trtllm_repo_path") or "")
            base = (self._git(repo, "rev-parse", "HEAD") or "").strip() if repo else ""
            stop = getattr(self, "_campaign_stop", None) or threading.Event()
            decision, note = tune_stage.run_tune_task(
                self, self._planned_state, base, attempt=task.attempts, should_stop=stop.is_set
            )
            self._checkpoint(self._planned_state)
            return f"{decision}\n{note}" if note else decision
        if task.kind == "kernel":
            item = self._queue_item(task)
            if item is None:
                return (
                    f"kernel task {task.id} names no queue row or `<mode>:<entry>` ({task.ref!r})"
                )
            # A retry is dispatched afresh: without a note the kernel writer would adopt
            # the failed attempt's decision file.
            retry = (
                f"Attempt {task.attempts} of task {task.id}; the previous attempt failed."
                if task.attempts > 1
                else ""
            )
            with self._carrying(self._task_context(receipts, item.unit, task)):
                self._run_one_kernel_writer(item, retry, task=task)
            decision = self._kernel_writer_decision_for(item) or ""
            reported = self._engine_defect_reported(item, task, decision) or self._macro_rows(
                item, decision
            )
            if not reported:
                return decision
            return f"{decision}; {reported}" if "\n" in decision else f"{decision}\n{reported}"
        role = ROLE_BY_KIND[task.kind]
        unit = task.ref or task.id
        if task.kind in ("assemble", "feature"):
            prompt = self._assembler_prompt(unit)
        elif task.kind == "survey":
            prompt = self._surveyor_prompt()
        else:
            prompt = ""
        tags = task_tags(task.kind, task.tags)
        header = (
            f"## Your task: `{task.id}` (tags: {', '.join(tags)})\n\n{task.goal}\n\n"
            f"The task envelope is `{envelope}`: the paths you may change (`touches`), the "
            f"checks your work must pass (`satisfies`, run by the orchestrator, not you), "
            f"and its priority."
        )
        context = self._task_context(receipts, unit, task)
        parts = (header, context, tag_section(tags), prompt)
        decision = (
            self._dispatch_task(role, unit, task.id, "\n\n".join(p.strip() for p in parts if p))
            or ""
        )
        if role == "assembler" and decision == "BLOCKED":
            said = self._blocked_report(task.id)
            return f"{decision}\n{said}" if said else decision
        if role == "assembler" and decision == "UNIT_READY":
            # A finished run's note is an urgent report the planner reads (`_finish`).
            said = self._uncertified_report(task.id)
            return f"{decision}\n{said}" if said else decision
        if task.kind == "survey" and decision == "DRAFT_READY":
            try:
                load_vocabulary(self.vocabulary_path, units=assembly_units(self.task_data))
            except VocabularyError as exc:
                return f"vocabulary.yaml does not validate: {exc}"
        return decision

    def _engine_defect_reported(self, item: QueueItem, task: graph_mod.Task, decision: str) -> str:
        """Queue the task that fixes the engine defect a failed receipt reports.

        A kernel writer whose decision is ``FAILED_RECEIPT`` with an ``engine_defect``
        report (``progress.parse_engine_defect``): a ``feature`` task that fixes
        it is queued for the orchestrator to add (:meth:`_take_defect_tasks`).
        What the planner hears; ``""`` for any other decision.
        """
        if not decision.startswith(FAILED_RECEIPT):
            return ""
        row = self._kernel_writer_entry_for(item) or {}
        report = parse_engine_defect(row.get("engine_defect"))
        if not report:
            return ""
        spec = self._defect_fix_task(report, item, task)
        said = f"it reports an engine defect in `{report['affects']}` ({report['trigger']})"
        if spec is None:
            return said + "; give its fix a task, the proposal is in its receipt"
        with self._defect_lock:
            self.__dict__.setdefault("_defect_specs", []).append(spec)
        return said + f"; the orchestrator adds {spec['id']} to fix it"

    def _blocked_report(self, what: str) -> str:
        """What a ``BLOCKED`` assembler wrote the planner: why, the missing entry, its summary."""
        path = self.__dict__.get("_decision_files", {}).get(what)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path else None
        except (OSError, yaml.YAMLError):
            raw = None
        if not isinstance(raw, dict):
            return ""
        said = [f"{k} {raw[k]}" for k in ("blocked_reason", "missing_entry") if raw.get(k)]
        if raw.get("summary"):
            said.append(str(raw["summary"]).strip()[:600])
        return "; ".join(said)

    def _uncertified_report(self, what: str) -> str:
        """The calls a ``UNIT_READY`` assembler listed as correct but outside their entries' CELLS.

        The certification they need is the planner's to add: the list rides on
        the decision's second line, which the planner reads with the task's event.
        """
        path = self.__dict__.get("_decision_files", {}).get(what)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path else None
        except (OSError, yaml.YAMLError):
            raw = None
        return uncertified_report(raw.get("uncertified") if isinstance(raw, dict) else None)

    def _macro_rows(self, item: QueueItem, decision: str) -> str:
        """What the planner hears of a ``MACRO``: its constituents, a kernel task each."""
        if decision != "MACRO":
            return ""
        row = self._kernel_writer_entry_for(item) or {}
        children = [c for c in row.get("constituents") or () if isinstance(c, dict)]
        if not children:
            return "it names no constituents"
        rows = "; ".join(
            f"`{c['target']}` (tier {c['tier']}"
            + (f", unit {c['unit']}" if c.get("unit") else "")
            + (f", {c['surface']}" if c.get("surface") else "")
            + ")"
            for c in children
        )
        return f"its constituents, a kernel task each (`onboard:<entry>`): {rows}"

    def _defect_fix_task(
        self, report: Mapping[str, Any], item: QueueItem, task: graph_mod.Task
    ) -> dict[str, Any] | None:
        """The task that fixes ``report``'s defect, as ``Task.from_dict`` reads one; ``None`` when unplaceable.

        It touches the file ``affects`` names (``<file>[:<symbol>]``), and the
        detecting test when ``detect`` names a file of the checkout; an
        ``affects`` that names no file of the checkout is left to the planner.
        The goal states the receipt's own words whole. The same defect,
        reported again, names the same task.
        """
        affects = str(report["affects"])
        path = affects.split(":", 1)[0].strip()
        repo = Path(str(self.task_data.get("trtllm_repo_path") or ""))
        if not path or not (repo / path).is_file():
            return None
        slug = re.sub(r"[^a-z0-9]+", "-", Path(path).stem.lower()).strip("-")[:30] or "engine"
        trigger = " ".join(str(report["trigger"]).split())
        proposal = " ".join(str(report["proposal"]).split())
        raw_detect = report.get("detect")
        detect = (
            yaml.safe_dump(dict(raw_detect), sort_keys=True).strip()
            if isinstance(raw_detect, Mapping)
            else str(raw_detect or "the failing test in the receipt")
        )
        touches = [path]
        named = raw_detect.get("test") if isinstance(raw_detect, Mapping) else raw_detect
        test = str(named or "").split("::", 1)[0].strip()
        if test and test != path and (repo / test).is_file():
            touches.append(test)
        key = zlib.crc32(f"{affects}: {trigger}".lower().encode())
        return {
            "id": f"fix-{slug}-{key:08x}",
            "kind": "feature",
            "ref": item.unit,
            "goal": (
                f"Fix the engine defect task {task.id} found in `{affects}`: "
                f"{trigger}. Proposed fix: {proposal} The detection is:\n{detect}\n"
                f"It passes on your branch, and {task.id} is retried once this lands."
            ),
            "touches": touches,
            "priority": task.priority,
            "resources": {"gpus": 1, "est_minutes": 90},
        }

    def _take_defect_tasks(self, graph: graph_mod.Graph) -> list[dict[str, Any]]:
        """The fix tasks queued since the orchestrator last asked, less those it already has."""
        with self._defect_lock:
            specs = self.__dict__.pop("_defect_specs", [])
        have = {t.id for t in graph.tasks}
        return [s for s in specs if s["id"] not in have]

    def _queue_item(self, task: graph_mod.Task) -> QueueItem | None:
        """The kernel writer dispatch a kernel task names.

        ``ref`` is a vocabulary queue row number, ``<mode>:<entry>[@<site>]``
        for an entry the planner adds itself (a MACRO's constituents, say), or
        ``promote:<entry>[,<entry>...]`` (:func:`~...onboard.vocabulary.queue_item_from_ref`).
        """
        if task.ref.isdigit():
            queue = load_vocabulary(self.vocabulary_path, units=assembly_units(self.task_data))
            return next((q for q in queue if str(q.n) == task.ref), None)
        return queue_item_from_ref(task.ref, task.id, task.goal)

    # ---------------------------------------------- receipts, runtime facts
    def _task_context(
        self, receipts: list[dict], unit: str | None, task: graph_mod.Task | None = None
    ) -> str:
        """What a task's turn carries beside its goal.

        Its dependencies' receipts, the runtime facts, and, for a task that
        starts beside its soft dependencies, where each of them stands.
        """
        blocks = (
            receipts_mod.render(receipts),
            runtime_facts.render(self.workspace, unit=unit),
            self._soft_block(task) if task is not None else "",
        )
        return "\n\n".join(block for block in blocks if block)

    def _soft_block(self, task: graph_mod.Task) -> str:
        """Where ``task``'s soft dependencies stand as it starts, read off ``graph.yaml``."""
        if not task.depends_on_soft:
            return ""
        try:
            tasks = graph_mod.Graph.load(self.workspace / graph_mod.GRAPH_FILENAME).by_id()
        except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
            tasks = {}
        states = {
            dep: tasks[dep].status if dep in tasks else "running" for dep in task.depends_on_soft
        }
        return soft_dependency_block(
            states,
            self.workspace / TASKS_DIRNAME,
            str(self.task_data.get("trtllm_repo_path") or ""),
        )

    @contextlib.contextmanager
    def _carrying(self, block: str):
        """Turns this thread dispatches meanwhile carry ``block`` after their worktree's."""
        before = getattr(self._tls, "worktree", "")
        if block:
            self._tls.worktree = f"{before}\n\n{block}" if before else block
        try:
            yield
        finally:
            self._tls.worktree = before

    def _dependency_receipts(self, task: graph_mod.Task) -> list[dict]:
        """The receipt of each finished task ``task`` depends on, in its order, soft ones after.

        Kept when that task finished, or, for one that finished before
        receipts were kept, read now off what its turn left on disk.
        """
        if not task.depends_on and not task.depends_on_soft:
            return []
        try:
            tasks = graph_mod.Graph.load(self.workspace / graph_mod.GRAPH_FILENAME).by_id()
        except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
            # Absent, or read mid-write: the receipts kept on disk are what there is.
            tasks = {}
        tasks_dir = self.workspace / TASKS_DIRNAME
        found = []
        for dep in [*task.depends_on, *task.depends_on_soft]:
            done = tasks.get(dep)
            if done is not None and done.status != "done":
                continue
            receipt = receipts_mod.read(tasks_dir, dep)
            if receipt is None and done is not None:
                try:
                    receipt = self._receipt_for(done, "")
                except OSError:
                    continue
                if not receipt["decision"]:
                    # Its turn left no decision to read: there is nothing to hand on.
                    continue
                with contextlib.suppress(OSError):
                    receipts_mod.write(tasks_dir, receipt)
            if receipt is not None:
                found.append(receipt)
        return found

    def _keep_receipt(self, task: graph_mod.Task, decision: str) -> None:
        """Keep a finished task's receipt beside its envelope; a failed one leaves none."""
        head, _, note = decision.partition("\n")
        if head not in FINISHED:
            return
        try:
            receipts_mod.write(self.workspace / TASKS_DIRNAME, self._receipt_for(task, head, note))
        except OSError as exc:
            self.journal.event("receipt not kept", f"{task.id}: {exc}")

    def _receipt_for(self, task: graph_mod.Task, decision: str, note: str = "") -> dict:
        decision_file, report = self._turn_files(task)
        return receipts_mod.build(
            task.id,
            task.kind,
            decision=decision,
            note=note,
            attempt=task.attempts,
            decision_file=decision_file,
            result_file=report,
            commits=self._landed_commits(task.id),
            workspace=self.workspace,
        )

    def _turn_files(self, task: graph_mod.Task) -> tuple[Path | None, Path | None]:
        """The decision file and the dispatch report of ``task``'s latest turn.

        A turn's report holds its prompt, which names the decision file, and
        its final message. A kernel task's kernel writer turns are filed under its
        queue row, which another task can verify again, so its report counts
        only when it names that row's latest decision. A tune task's campaign
        has neither: its note is its receipt.
        """
        if task.kind == "tune":
            return None, None
        decision: Path | None = None
        if task.kind == "kernel":
            try:
                item = self._queue_item(task)
            except (OSError, VocabularyError):
                item = None
            if item is None:
                return None, None
            role, what = "kernel_writer", f"{item.mode} {item.target}"
            decision = current_decision(self.workspace, "kernel_writer", item.n, unit=item.unit)
        else:
            role, what = ROLE_BY_KIND.get(task.kind, task.kind), task.id
        report = self.journal.latest_report(role, what)
        named = receipts_mod.decision_named_in(report)
        if decision is None:
            decision = named
        elif named != decision:
            report = None
        return decision, report

    def _landed_commits(self, task_id: str) -> list[str]:
        """``<sha> <subject>`` of each commit ``task_id``'s branch integrated, newest first.

        As many as its integration counted. One integrated before this
        process started has no count here: its branch tip stands for it.
        """
        integration = getattr(self, "_integration", None)
        tip = self._integrated_tip(task_id)
        if integration is None or not tip:
            return []
        outcome = self.__dict__.get("_merged", {}).get(task_id)
        count = outcome.commits if outcome is not None else 0
        if outcome is not None and count == 0:
            return []
        log = self._git(str(integration.repo), "log", f"-{count or 1}", "--format=%h %s", str(tip))
        lines = [line for line in (log or "").splitlines() if line.strip()]
        return lines if count else [f"{line} (the branch tip)" for line in lines[:1]]

    def _marker_now(self) -> str:
        """The target's marker as it is now; ``""`` (not established) where there is no target."""
        marker = getattr(self, "_target_marker", None)
        return marker() if marker is not None else ""

    # ------------------------------------------------------------ correctness
    def _mark_correct(self, marker: str, evaluation: contract_mod.Evaluation) -> dict | None:
        """Every required check holds on the target (``marker``): name its commit and journal it.

        The commit is the one the checks measured (``_milestone_commit``),
        whether or not the checkout has moved on since. The branch
        ``<target branch>-correct`` is set to it, and the journal says
        ``CORRECTNESS DELIVERED <sha>`` with the time since the run opened.
        Nothing is named, and ``None`` returned, when no commit is known to
        hold the state the checks measured.
        """
        sha = self._milestone_commit(marker, evaluation, self._contract.required())
        if sha is None:
            return None
        ref = self._point_branch(
            "correct",
            sha,
            "trtllm-modeling-v2: every required check of the contract holds",
            unwritten="correctness ref not written",
        )
        opened = self.journal.opened_at()
        elapsed = time.time() - opened.timestamp() if opened is not None else None
        previous = contract_mod.latest_milestone(self.workspace)
        parts = [
            f"{elapsed / 3600:.1f} h after the run started ({opened.isoformat()})"
            if opened is not None and elapsed is not None
            else "the journal does not say when the run started",
            f"all {len(evaluation.passed)} required checks of the contract hold on it, "
            "the delivery tier's included",
        ]
        if ref:
            parts.append(f"branch `{ref}`")
        if previous is not None:
            parts.append(f"moved forward from {previous.sha[:10]}")
        detail = "; ".join(parts)
        self.journal.event(f"CORRECTNESS DELIVERED {sha}", detail)
        print_message(
            f"[bold green]✔ CORRECTNESS DELIVERED {sha[:10]}: {escape_markup(detail)}[/bold green]",
            None,
        )
        return {"sha": sha, "ref": ref, "elapsed_s": elapsed}

    def _uncommitted_gate_inputs(self, repo: str) -> list[str] | None:
        """The files a gate reads that the checkout holds uncommitted; ``None`` when git cannot say.

        No check measures them: a check measures a snapshot of a commit.
        """
        from ..workflow import anchor_tables, gate_inputs, gate_relevant, porcelain_paths

        policy = CommitPolicy.from_task(self.task_data, workspace=self.workspace)
        pathspec = [*policy.allowed_paths] or ["."]
        pathspec += [f":(exclude){p}" for p in policy.denied_paths]
        dirt = self._git(
            repo, "status", "--porcelain", "-z", "--untracked-files=all", "--", *pathspec
        )
        if dirt is None:
            return None
        keep = gate_inputs(self.task_data, repo) | set(anchor_tables(self.task_data, repo))
        return sorted(
            {
                path
                for path, _ in porcelain_paths(dirt)
                if gate_relevant(path, keep=keep, denied=policy.denied_paths)
            }
        )

    def _note_runtime_facts(self, check: contract_mod.Check, result: Any, marker: str) -> None:
        """A passed gate's engines into ``runtime_facts.yaml``; a measured change is journaled.

        ``marker`` is the target the gate measured, read as it started.
        """
        engines = tuple(getattr(result, "engine", ()) or ())
        if not result.passed or not engines:
            return
        try:
            changes = runtime_facts.record(
                self.workspace,
                unit=check.unit,
                member=check.member,
                gate=check.gate,
                engines=engines,
                tree=str(getattr(result, "tree_state", "") or ""),
                marker=marker,
                current=self._marker_now(),
            )
        except OSError as exc:
            self.journal.event("runtime facts not written", f"{check.key}: {exc}")
            return
        if changes:
            self.journal.event("runtime facts changed", "; ".join(changes))

    # ----------------------------------------------------------------- checks
    def _admission_terms(self, check: contract_mod.Check) -> tuple[Any, str | None, float | None]:
        """What ``check``'s job asks of the sessions: its member's width, any other devices, its wall."""
        return (
            self._member_env(check.member),
            # The equivalence witness's leg B is two engines of the member's width.
            "all" if check.gate == "equivalence" else None,
            gates_mod.ACCURACY_TIMEOUT_S if check.gate.startswith("accuracy[") else None,
        )

    def _admit_check(
        self, admission: CheckAdmission, check: contract_mod.Check
    ) -> contextlib.AbstractContextManager:
        """Hold ``check`` until the sessions' free devices let it start (``Services.admit``)."""
        return admission.slot(*self._admission_terms(check))

    def _prioritized_check(self, check: contract_mod.Check, log) -> Any:
        """:meth:`_run_contract_check`, its jobs queued at the check's class (``priority.py``).

        The first unit's inner checks are on the way to its accuracy, and go
        ahead of every other check in a session's spool.
        """
        units = assembly_units(self.task_data)
        first = bool(units) and check.unit == units[0] and check.tier == contract_mod.TIER_INNER
        with job_priority(priority_mod.ACCURACY if first else priority_mod.FEATURE):
            return self._run_contract_check(check, log)

    def _run_contract_check(self, check: contract_mod.Check, log) -> Any:
        """Run one contract check and return its gate result.

        On a git checkout the check measures a snapshot of the commit the
        checkout is at as it starts (``_snapshots``). A check that passes adds
        the engines it attested to the runtime facts.
        """
        snapshots = getattr(self, "_snapshots", None)
        if snapshots is None:
            marker = self._marker_now()
            result = self._gate(check, log)
        else:
            result, marker = self._gate_on_snapshot(snapshots, check, log)
        if check.gate.startswith("accuracy[") and result.passed:
            self._note_shift(check.unit, check.member, result, self._spec_for(check.unit)[0])
        self._note_runtime_facts(check, result, marker)
        return result

    def _take_stage_notes(self) -> list[str]:
        """The operator's messages since the orchestrator last asked (``run/inbox.py``)."""
        messages = inbox_mod.take(self.workspace)
        if messages:
            self.journal.event("operator", "\n".join(messages))
        return messages

    def _gate_on_snapshot(
        self, snapshots: snapshots_mod.Snapshots, check: contract_mod.Check, log
    ) -> tuple[Any, str]:
        """``check``'s result on a snapshot of the checkout's head, and the marker it measured."""
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        head = resolve_commit(repo)
        if head is None:
            why = f"git could not name the commit {repo} is at, so there was nothing to snapshot"
            return snapshots_mod.void_result(check.gate, check.member, why), ""
        pin = snapshots_mod.CheckTree(
            commit=head,
            tree=snapshots.tree(head),
            marker=self._commit_marker(head),
            uncommitted=tuple(self._uncommitted_gate_inputs(repo) or ()),
            snapshots=snapshots,
        )
        try:
            with snapshots.lease(head):
                return self._on_snapshot(pin)._gate(check, log), pin.marker
        except snapshots_mod.SnapshotError as exc:
            result = snapshots_mod.void_result(check.gate, check.member, str(exc))
            self.journal.event("gate", result.summary())
            return result, pin.marker

    def _on_snapshot(self, pin: snapshots_mod.CheckTree) -> "PlannedStage":
        """This workflow as one check sees it: the target is ``pin``'s snapshot.

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
        view.__dict__["_member_runners"] = {}
        noting = self._noting_session_loss
        view._noting_session_loss = lambda runner: noting(pin.runner(runner))
        view._shell_runner = pin.runner(self._shell_runner)
        measure = view._gate_until_verdict
        view._gate_until_verdict = lambda run, log: pin.settle(
            lambda: measure(run, log), lambda why: self.journal.event("void verdict", why)
        )
        return view

    # ------------------------------------------------------------------ gates
    def _gate_call(self, check: contract_mod.Check) -> Callable[[Any], Any] | None:
        """``check``'s gate, as a function of the runner it launches with; ``None`` for a witness."""
        unit, member, gate = check.unit, check.member, check.gate
        spec, features = self._spec_for(unit)
        env = self._member_env(member)
        workdir = self.workspace / "gates" / unit

        def common(runner: Any) -> dict[str, Any]:
            return dict(task_data=spec, runner=runner, unit=unit, member=member)

        if gate == "boot":
            return lambda runner: gates_mod.run_boot_gate(
                env, features=features, workdir=workdir,
                spec=gates_mod.resolve_boot_spec(spec, unit, features, self.boot_spec_path),
                **common(runner),
            )  # fmt: skip
        if gate.startswith("accuracy["):
            return lambda runner: gates_mod.run_accuracy_gate(
                env, reference=self._reference_score(unit, spec), workdir=workdir, **common(runner)
            )
        if gate == "acceptance":
            return lambda runner: gates_mod.run_acceptance_gate(
                env, features=features, workdir=workdir,
                records=gates_mod.records_for(self.workspace), **common(runner),
            )  # fmt: skip
        if gate == envelope_boot.GATE_NAME:
            return lambda runner: envelope_boot.run_envelope_boot(
                env, features=features, workdir=self.workspace / "gates" / "envelope",
                **common(runner),
            )  # fmt: skip
        if gate == logit_agreement.GATE_NAME:
            return lambda runner: logit_agreement.run_logit_gate(
                env, features=features, workdir=workdir, **common(runner)
            )
        if gate == serve_gate.GATE_NAME:
            return lambda runner: serve_gate.run_serve_gate(
                env, features=features, workdir=workdir,
                boot=gates_mod.resolve_boot_spec(spec, unit, features, self.boot_spec_path),
                **common(runner),
            )  # fmt: skip
        return None

    def _gate(self, check: contract_mod.Check, log) -> Any:
        """Run ``check``'s gate on the target this workflow sees, journal it, and return its result."""
        call = self._gate_call(check)
        if call is None:
            return self._run_witness(check.unit, check.member, log)
        runner = self._member_runner(check.member)
        result = self._gate_until_verdict(lambda: call(runner), log)
        self.gate_log.record(result)
        self.journal.event("gate", result.summary())
        if not result.passed:
            print_message(f"[bold yellow]↻ {escape_markup(result.summary())}[/bold yellow]", log)
        return result

    def _note_shift(self, unit: str, member: str, result: Any, spec: Any = None) -> None:
        """Once every member `unit` owes an accuracy check has a verdict, check them together.

        Each member can clear its own floor while all of them sit under the
        anchor; that is recorded as a finding and refused at delivery. The
        members are those the contract owes the benchmark at
        (:func:`~..contract.benchmark_members`), else every member of the unit.
        """
        measured = self._accuracy_by_unit.setdefault(unit, {})
        measured[member] = result
        spec = spec or self.task_data
        members = getattr(self, "_accuracy_owed", {}).get(unit) or members_for(
            spec, unit, self._gate_env().world_size
        )
        if set(members) <= set(measured):
            finding = gates_mod.note_systematic_shift(
                gates_mod.records_for(self.workspace),
                [measured[m] for m in members],
                spec.get("accuracy") or {},
                unit,
            )
            if finding is not None:
                self.journal.event("finding", finding.describe())


def _set_aside(proposed: Path) -> None:
    """Clear ``proposed`` for a new turn, keeping the last one as ``<stem>.previous``.

    A planner turn answered after its process stopped leaves a proposed graph
    no process applied; deleting it before the next turn would lose that
    turn's work, which :func:`_unread_answer_note` points the next turn to.
    """
    if proposed.is_file():
        os.replace(proposed, previous_proposal(proposed))


def previous_proposal(proposed: Path) -> Path:
    """Where :func:`_set_aside` keeps the last proposed graph."""
    return proposed.with_name(f"{proposed.stem}.previous{proposed.suffix}")


def _unread_answer_note(unread, kept: Path | None = None) -> str:
    """What a turn is told of the previous process's answer to the same work, never read.

    ``kept`` is a file that answer wrote that the run set aside rather than read.
    """
    return (
        "## An answer the previous process never read\n"
        f"Turn `{unread.request_id}` answered this same work after the process that asked it "
        "had stopped, so nothing acted on it. Its decision is at "
        f"`{unread.decision_file}`"
        + (f", what it proposed at `{kept}`" if kept is not None else "")
        + ", and what it wrote is below. It answered an earlier "
        "standing: check it against the graph and events you have now, keep what still "
        "holds instead of redoing it, and write your own decision at the path this turn "
        "names.\n\n"
        f"{unread.text.strip()[:4000]}"
    )


__all__ = ["PLANNER_ROLE", "PROPOSED_GRAPH", "PlannedStage", "ROLE_BY_KIND"]
