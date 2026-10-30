# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The planned run: the workflow's services handed to :class:`~..orchestrator.GraphOrchestrator`.

Entered after preflight. The planner turn, each task's worker turn and each
contract check are this module's; deciding what runs and what counts is the
orchestrator's. Once the contract holds, delivery verifies the trees on disk.

Worker turns run concurrently, so a dispatch here names its unit and turn
explicitly instead of reading the shared progress context.

A ``tune`` task is the tune campaign, pinned to the commit the target is at
when the core's checks first hold. While the run is under way the improver
is woken beside it (``run/stages/improve.py``), and its lessons reach every
later dispatch.

Every contract check measures a snapshot of the commit the checkout is at
when it starts (``run/snapshots.py``), never the checkout, which integrations
move while checks run. Its verdict names that commit, is recorded under its
marker, and is void when a command imported any other tree.

When every required check holds on the target, the orchestrator asks this
stage to name that commit (``_mark_correct``): the branch
``<target branch>-correct`` and a ``CORRECTNESS DELIVERED <sha>`` journal line
with the time since the run opened. The planner's request says where
correctness stands and whether the target is still there.

A task that finishes leaves its receipt (``run/receipts.py``): its decision,
summary, integrated commits, decision and result files, and the notes its
worker wrote for the orchestrator or the planner. A task that starts carries
the receipt of each task it depends on, in its envelope and its prompt. A
gate that passes adds the engines it attested to ``runtime_facts.yaml``
(``run/runtime_facts.py``), and the planner's, the smiths' and every worker's
prompt carries those measured numbers.

A task's worktree being made and its branch being integrated are in flight
for a reload (``run/drain.py``): one that would start while a reload drains
waits, and one under way is waited for. When the drain is over this stage
checkpoints and re-executes the process (``_reload_process``). On start it
journals what the stop before cost.

A task with soft dependencies starts beside them and is told so; when its
worker finishes, its integration waits for them to land or end, so its
branch is rebased onto a head that holds what landed. One that ended
without landing is named in the task's decision, and so in its receipt and
the planner's event. A ``check`` task has no worker: the orchestrator runs
its checks, and once they pass its receipt names them.

A run that a failed substrate gate stopped in phase 0 is left open, not
closed: re-running the same command proves the substrate again and resumes.
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

from ...agents import lessons as lessons_mod
from ...checks import bundle as bundle_mod
from ...checks import envelope_boot, logit_agreement, serve_gate
from ...checks import gates as gates_mod
from ...infra.committer import CommitPolicy
from ...infra.driver import reload as reload_mod
from ...infra.execution import CheckAdmission, startable_checks
from ...infra.execution.capacity import held_gpus, offsite_kinds
from ...infra.transport import ExternalAgentTransport
from ...infra.vcs import resolve_commit
from ...onboard import promotion as promotion_mod
from ...onboard.vocabulary import QueueItem, VocabularyError, load_vocabulary, queue_item_from_ref
from ...spec.task_schema import assembly_units, members_for
from ...tune import stage as tune_stage
from .. import contract as contract_mod
from .. import drain as drain_mod
from .. import geometry, runtime_facts
from .. import graph as graph_mod
from .. import receipts as receipts_mod
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
    DECISIONS_DIRNAME,
    collect_decision_file,
    current_decision,
    decision_file_instruction,
    decision_value,
    latest_entry,
    mark_decision_dispatch,
    uncertified_report,
)
from ..state import WorkflowState
from .smith import FAILED_RECEIPT

PLANNER_ROLE = "planner"
PROPOSED_GRAPH = "graph.proposed.yaml"
TASKS_DIRNAME = "tasks"
#: How many of the latest measured changes the planner's request lists.
PLANNER_FACT_CHANGES = 8

#: The role each task kind's turn runs as.
ROLE_BY_KIND = {
    "survey": "plan_drafter",
    "kernel": "smith",
    "assemble": "assembler",
    "feature": "assembler",
    "integrate": "assembler",
    "investigate": "assembler",
    "review": "reviewer",
    "tune": "tuner",
}


class PlannedStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    _turn_lock = threading.Lock()
    #: Guards the fix tasks for engine defects smiths reported (``_defect_specs``),
    #: queued until the orchestrator takes them.
    _defect_lock = threading.Lock()

    # ------------------------------------------------------------------ entry
    def _run_planned(self, state: WorkflowState, log) -> bool:
        print_rule("[bold magenta]planned run[/bold magenta]", log)
        contract = self._contract or contract_mod.compile_contract(
            self.task_data, self._gate_env().world_size
        )
        self._turns_by_key: dict[str, int] = {}
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
        self._failed_lineages: set[str] = set()
        # How many checks run at once follows the sessions' free devices.
        admission = CheckAdmission(startable_checks)
        # Set once full support holds: a running campaign wraps up.
        self._campaign_stop = threading.Event()
        services = Services(
            plan=lambda turn: self._planner_turn(turn, contract),
            work=self._work_task,
            check=lambda check: self._run_contract_check(check, log),
            free_gpus=lambda: total,
            checkpoint=lambda: self._checkpoint(state),
            log=lambda message: self.journal.event("planned", message),
            marker=self._target_marker,
            critique=self._critique if self.graph_critic else None,
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
        )
        # The cross-model regression rule binds a run that starts on it: a run
        # with landed work began under rules without it, and its graph would be
        # refused for work it can no longer re-plan.
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
        # A campaign's turns, and the critic's and the improver's, take only the agent
        # slots the foreground work leaves idle.
        gate = getattr(self, "_agent_gate", None)
        if gate is not None:
            gate.background_limit = orchestrator.idle_agent_slots
            gate.side_limit = orchestrator.idle_agent_slots
        self._record_restart(orchestrator.restart)
        with self._improver_watching():
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
            if isinstance(self._turns, ExternalAgentTransport):
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
        if isinstance(self._turns, ExternalAgentTransport):
            self._turns.consume([task.id])
        if task.kind == graph_mod.CHECK_KIND and task.status == "done":
            note = f"its checks passed, with no agent turn: {', '.join(task.satisfies)}"
            try:
                receipts_mod.write(
                    self.workspace / TASKS_DIRNAME,
                    receipts_mod.build(
                        task.id, task.kind, decision=CHECKED, note=note, attempt=task.attempts
                    ),
                )
            except OSError as exc:
                self.journal.event("receipt not kept", f"{task.id}: {exc}")

    def _dispatch_task(self, role: str, unit: str, what: str, prompt: str) -> str | None:
        """One turn with its unit and turn number named; returns its decision.

        **A resume adopts a task's turn the previous process never finished
        with.** On the external transport a process can die after the driver
        answered a task's turn and before the task had its outcome: before the
        answer was read, or while the checks that follow the turn ran. The
        task starts again here, and its first dispatch in this process adopts
        that answer and collects the decision where the turn wrote it, instead
        of asking again. A later dispatch of the same task in this process (a
        rebase, say) is always asked: the earlier answer is what it revises.
        Every answered turn of a task that has its outcome is consumed first,
        so a task the planner reopens is asked afresh.
        """
        turn = self._next_turn(role, unit)
        with self._turn_lock:
            dispatched = self.__dict__.setdefault("_tasks_dispatched", set())
            first = (role, what) not in dispatched
            dispatched.add((role, what))
        transport = self._turns
        if isinstance(transport, ExternalAgentTransport):
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
                # A process that read the answer counted its lessons already.
                if answered.read_by is None:
                    self._lesson_feedback(answered.decision_file)
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
        self._lesson_feedback(Path(target))
        return collect_decision_file(self._progress_ctx, self.workspace, role, turn, unit=unit)

    def _lesson_feedback(self, decision_file: Path) -> None:
        """Count the ``lessons: {id: helped | wrong}`` a worker reported."""
        store = self._lesson_store()
        if store is None or not decision_file.is_file():
            return
        try:
            raw = yaml.safe_load(decision_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return
        marks = raw.get("lessons") if isinstance(raw, dict) else None
        if isinstance(marks, dict):
            changes = store.feedback(marks)
            if changes:
                self.journal.event("lesson feedback", "\n".join(changes))

    # ------------------------------------------------------------ the critic
    def _critique(self, graph: graph_mod.Graph, reason: str) -> str:
        """One read-only review of a newly adopted graph; empty when it approves."""
        path = self.workspace / "graph.critique.yaml"
        graph.save(path)
        prompt = (
            f"Read the task graph `{path}` the planner just wrote (its reason: {reason}), "
            f"against the spec `{self.task_path}` and the contract "
            f"`{self.workspace / contract_mod.CONTRACT_FILENAME}`. This is a paper review: do not "
            "build, run or test anything.\n\n"
            "Look for what will cost the run most if nobody says it now: required work that is "
            "missing or ordered after something that needs it; a task too large to finish within "
            "its estimate; parallel tasks that will collide in the same file or the same forward "
            "even though their `touches` differ; a task claiming a check it cannot make pass.\n\n"
            "Then for what holds the run back for nothing: a `depends_on` edge on a fix whose "
            "change the dependent needs only when its checks run (`depends_on_soft`); an agent "
            "task whose work is only re-running contract checks (a `check` task); a review, "
            "verify or check task reserving devices its jobs take for themselves; more than one "
            "cross-model regression review where one batch would cover the same changes.\n\n"
            "`APPROVE` when the graph is sound. `REJECT` with each issue and its fix in `summary`; "
            "your review is advisory, and reaches the planner's next turn."
        )
        decision = self._dispatch_task("plan_reviewer", "graph", "critic", prompt)
        if decision == "APPROVE":
            return ""
        entry = latest_entry(self.progress_path, "plan_reviewer") or {}
        return f"{decision or 'no decision'}: {entry.get('summary', '')}"

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

        A worker's turn holds no device; its jobs queue in the spools. A task's
        ``gpus`` is its share of the sessions, so the running tasks' shares
        bound how many more start, and what gate jobs hold at the moment does
        not. ``held(task)`` is that share (``capacity.held_gpus``: none for a
        tune task whose jobs all run on the perf sessions). ``total`` stands in
        for the sessions when none is live.
        """
        capacity = self._spool_capacity()
        if capacity is None:
            capacity = total
        return capacity - sum(held(t) for t in graph.tasks if t.status == "running")

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
        """The target's own paths when the cross-model rule binds this run; none otherwise.

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
                "## Cross-model regression",
                "",
                "This run enforces it. The target's own paths: "
                + ", ".join(f"`{p}`" for p in own)
                + ". Any other path a task touches is shared code.",
            ]
            if turn.regression:
                lines += ["", turn.regression]
        if turn.events:
            lines += ["", "## Since your last turn", "", *[f"- {e}" for e in turn.events]]
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
        """Run ``task``'s worker and return its decision; a failure may wake the improver.

        The decision is the first line; anything after it is a note the
        planner sees with the task's event. A finished task's receipt is kept
        for the tasks that depend on it. The improver reads a failure beside
        the run (:meth:`~.improve.ImproveStage._improver_task_ended`): the
        task's outcome does not wait for it.
        """
        before = self._integrated_tip(task.id)
        decision = self._work_task_turn(task)
        if task.kind in INTEGRATED_KINDS and decision.partition("\n")[0] not in FINISHED:
            tip = self._integrated_tip(task.id)
            if tip and tip != before:
                with self._turn_lock:
                    self.__dict__.setdefault("_landed_attempts", {})[task.id] = task.attempts
        self._keep_receipt(task, decision)
        self._improver_task_ended(task, decision)
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
        change to the code they guard like any other, and a cross-model
        regression review covers them.
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
                # A smith given a note is dispatched afresh rather than handed the
                # decision already on disk: that decision is the work to rebase.
                self._run_one_smith(item, note, task=task)
                return self._smith_decision_for(item) or ""
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
            # A retry is dispatched afresh: without a note the smith would adopt
            # the failed attempt's decision file.
            retry = (
                f"Attempt {task.attempts} of task {task.id}; the previous attempt failed."
                if task.attempts > 1
                else ""
            )
            with self._carrying(self._task_context(receipts, item.unit, task)):
                self._run_one_smith(item, retry, task=task)
            decision = self._smith_decision_for(item) or ""
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
            prompt = self._plan_drafter_prompt()
        elif task.kind == "review" and task.ref == "plan":
            role, prompt = "plan_reviewer", self._plan_reviewer_prompt()
        elif task.kind == "review" and graph_mod.is_regression_review(task):
            own = tuple(getattr(self, "_own_paths", ()) or ())
            prompt = self._regression_review_prompt(list(task.covers or ()), own)
        elif task.kind == "review":
            prompt = self._reviewer_prompt(unit)
        else:
            prompt = ""
        header = (
            f"## Your task: `{task.id}` ({task.kind})\n\n{task.goal}\n\n"
            f"The task envelope is `{envelope}`: the paths you may change (`touches`), the "
            f"checks your work must pass (`satisfies`, run by the orchestrator, not you), "
            f"and its priority."
        )
        context = self._task_context(receipts, unit, task)
        if context:
            header = f"{header}\n\n{context}"
        decision = self._dispatch_task(role, unit, task.id, f"{header}\n\n{prompt}".strip()) or ""
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
            known = self._mark_known_defects()
            return f"{decision}\n{known}" if known else decision
        return decision

    # ------------------------------------------------------- known engine defects
    def _mark_known_defects(self) -> str:
        """Mark the vocabulary rows in a known engine defect's regime; what the planner hears.

        Each such row gets the defect under ``known_defects`` and, where the
        defect is detected by a cell, that cell among its required ``cells``,
        before the orchestrator adds the rows' kernel tasks. ``""`` when no
        row is in one, in a measurement run, or without the lessons store
        (``agents/lessons.match_defects``).
        """
        raw = self._raw_vocabulary()
        rows = [row for row in raw.get("queue") or () if isinstance(row, dict)]
        hits = self._known_defect_hits(rows)
        if not hits:
            return ""
        by_n = {str(row.get("n")): row for row in rows}
        said: list[str] = []
        for hit in hits:
            defect, row = hit.defect, by_n.get(str(hit.row))
            if row is None:
                continue
            fix = (
                f"fixed upstream in `{defect.upstream_fix}`"
                if defect.upstream_fix
                else f"open: {defect.disposition}"
            )
            marks = row.setdefault("known_defects", [])
            if not any(isinstance(m, dict) and m.get("id") == defect.id for m in marks):
                marks.append(
                    {
                        "id": defect.id,
                        "affects": defect.affects,
                        "trigger": defect.trigger,
                        "detect": defect.detect,
                        "fix": fix,
                    }
                )
            if isinstance(defect.detect, dict) and defect.detect:
                cells = row.setdefault("cells", [])
                if dict(defect.detect) not in cells:
                    cells.append(dict(defect.detect))
            said.append(
                f"row {hit.row} (`{hit.target}`) is in the regime of known engine defect "
                f"`{defect.id}` ({defect.trigger}; {fix}; detected by {defect.detect or 'its test'})"
            )
        self.vocabulary_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        text = (
            "known engine defects the vocabulary falls in: "
            + "; ".join(said)
            + ". Schedule each fix or workaround now, ahead of the rows it affects: the "
            "detection is among each row's required cells (`known_defects` on the row)"
        )
        self.journal.event("known engine defects", text)
        return text

    def _engine_defect_reported(self, item: QueueItem, task: graph_mod.Task, decision: str) -> str:
        """Register the engine defect a failed receipt reports, and queue the task that fixes it.

        A smith whose decision is ``FAILED_RECEIPT`` with an ``engine_defect``
        report (``progress.parse_engine_defect``): the defect is added to the
        registry, open (``agents/lessons.defect_from_receipt``), and a
        ``feature`` task that fixes it is queued for the orchestrator to add
        (:meth:`_take_defect_tasks`). What the planner hears; ``""`` for any
        other decision.
        """
        if not decision.startswith(FAILED_RECEIPT):
            return ""
        row = self._smith_entry_for(item) or {}
        report = row.get("engine_defect")
        if not isinstance(report, dict) or not report:
            return ""
        evidence = f"{DECISIONS_DIRNAME}/{row.get('decision_file', '')}"
        defect = lessons_mod.defect_from_receipt(
            report, evidence=evidence, where=f"task {task.id}'s receipt ({evidence})"
        )
        if defect is None:
            return ""
        store = self._lesson_store()
        if store is not None:
            changes = store.merge([defect], run=self.workspace.name, evidence_root=self.workspace)
            if changes:
                self.journal.event("lessons", "\n".join(changes))
        spec = self._defect_fix_task(defect, item, task, report)
        said = (
            f"it reports an engine defect in `{defect.affects}` ({report.get('trigger')}), "
            f"registered as open engine defect `{defect.id}`"
        )
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

        The assembler carries on past such a call; the certification it needs
        is the planner's to add, and the task's summary reaches only the
        receipts of its dependents. So the list rides on the decision's second
        line, which the planner reads with the task's event.
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
        row = self._smith_entry_for(item) or {}
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
        self,
        defect: lessons_mod.Lesson,
        item: QueueItem,
        task: graph_mod.Task,
        report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """The task that fixes ``defect``, as ``Task.from_dict`` reads one; ``None`` when unplaceable.

        It touches the file ``affects`` names (``<file>[:<symbol>]``), and the
        detecting test when ``detect`` names a file of the checkout; an
        ``affects`` that names no file of the checkout is left to the planner.
        The goal states the receipt's own words (``report``), whole: the
        registry keeps them cut to its limit, and a fix task given half a
        proposal needs a planner turn to finish it.
        """
        path = defect.affects.split(":", 1)[0].strip()
        repo = Path(str(self.task_data.get("trtllm_repo_path") or ""))
        if not path or not (repo / path).is_file():
            return None
        slug = re.sub(r"[^a-z0-9]+", "-", Path(path).stem.lower()).strip("-")[:30] or "engine"
        report = report or {}
        trigger = " ".join(str(report.get("trigger") or defect.trigger).split())
        proposal = " ".join(str(report.get("proposal") or defect.lesson).split())
        raw_detect = report.get("detect") or defect.detect
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
        return {
            "id": f"fix-{slug}-{defect.id[:6]}",
            "kind": "feature",
            "ref": item.unit,
            "goal": (
                f"Fix the engine defect task {task.id} found in `{defect.affects}`: "
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
        """The smith dispatch a kernel task names.

        ``ref`` is a vocabulary queue row number, ``<mode>:<entry>[@<site>]``
        for an entry the planner adds itself (a MACRO's constituents, say), or
        ``promote:<entry>[,<entry>...]`` (:func:`~...onboard.vocabulary.queue_item_from_ref`).
        """
        if task.ref.isdigit():
            vocabulary = load_vocabulary(self.vocabulary_path, units=assembly_units(self.task_data))
            return next((q for q in vocabulary.queue if str(q.n) == task.ref), None)
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
        its final message. A kernel task's smith turns are filed under its
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
            role, what = "smith", f"{item.mode} {item.target}"
            decision = current_decision(self.workspace, "smith", item.n, unit=item.unit)
        elif task.kind == "review" and task.ref == "plan":
            role, what = "plan_reviewer", task.id
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
        whether or not the checkout has moved on since. The branch ``<target branch>-correct`` (the run's own branch
        name with ``HEAD`` detached) is set to it, and the journal says
        ``CORRECTNESS DELIVERED <sha>`` with the time since the run opened.
        Nothing is named, and ``None`` returned, when no commit is known to
        hold the state the checks measured.
        """
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        contract = getattr(self, "_contract", None)
        locate = getattr(self, "_milestone_commit", None)
        if locate is not None and contract is not None:
            sha = locate(marker, evaluation, contract.required())
        else:
            words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
            sha = words[0] if words and self._marker_now() == marker else None
        if sha is None:
            return None
        branch = (self._git(repo, "symbolic-ref", "--quiet", "--short", "HEAD") or "").strip()
        state = getattr(self, "_planned_state", None)
        ref = f"{branch or getattr(state, 'branch', '') or 'trtllm-modeling-v2'}-correct"
        reason = "trtllm-modeling-v2: every required check of the contract holds"
        if self._git(repo, "update-ref", "-m", reason, f"refs/heads/{ref}", sha) is None:
            self.journal.event("correctness ref not written", f"refs/heads/{ref} at {sha[:10]}")
            ref = ""
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
        """Hold ``check`` until the sessions' free devices let it start (``Services.admit``).

        A boot, logit, acceptance or accuracy check measured on a snapshot
        takes a seat in the bundle of its unit and member instead
        (``checks/bundle.py``). The first to arrive leads it and is admitted
        as any check is: that slot is the bundle's. The others hold none; they
        wait in the bundle for what the leader assigns them, and one assigned
        to run singly is admitted then (:meth:`_check_alone`).
        """
        self._check_admission = admission
        if getattr(self, "_snapshots", None) is None or not bundle_mod.bundled(check.gate):
            return admission.slot(*self._admission_terms(check))
        board = self.__dict__.setdefault("_check_board", bundle_mod.Board())
        return self._seated(admission, board.join((check.unit, check.member), check))

    @contextlib.contextmanager
    def _seated(self, admission: CheckAdmission, seat: bundle_mod.Seat):
        """``seat``'s admission, and the seat this thread's check runs in until it ends."""
        local = self.__dict__.setdefault("_bundle_seats", threading.local())
        local.seat = seat
        try:
            if seat.leader:
                with admission.slot(*self._admission_terms(seat.check)):
                    yield
            else:
                yield
        finally:
            local.seat = None
            seat.leave()

    def _seat_of(self, check: contract_mod.Check) -> bundle_mod.Seat | None:
        """The bundle seat this thread admitted ``check`` into, if any."""
        local = self.__dict__.get("_bundle_seats")
        seat = getattr(local, "seat", None) if local is not None else None
        return seat if seat is not None and seat.check.key == check.key else None

    def _run_contract_check(self, check: contract_mod.Check, log) -> Any:
        """Run one contract check and return its gate result.

        On a git checkout the check measures a snapshot of the commit the
        checkout is at as it starts (``_snapshots``), never the checkout. Its
        result names that commit and its marker, and what each command
        imported; it is void, and does not pass, when any command imported
        another tree, and a void one is measured once more. A check admitted
        into a bundle measures the commit its bundle measures, and is judged
        from its section of the bundle's job when it has one
        (:meth:`_check_in_bundle`). A check that passes adds the engines it
        attested to the runtime facts.
        """
        snapshots = getattr(self, "_snapshots", None)
        seat = self._seat_of(check) if snapshots is not None else None
        if seat is not None:
            result, marker = self._check_in_bundle(seat, check, log)
        elif snapshots is None:
            marker = self._marker_now()
            result = self._gate(check, log)
        else:
            result, marker = self._gate_on_snapshot(snapshots, check, log)
        if check.gate.startswith("accuracy[") and result.passed:
            self._note_shift(check.unit, check.member, result, self._spec_for(check.unit)[0])
        self._note_runtime_facts(check, result, marker)
        if result.passed and check.gate == "boot":
            self._probe_target_geometry(check)
        return result

    def _probe_target_geometry(self, check: contract_mod.Check) -> None:
        """Probe the target's geometry at a width the first time its core boots there.

        :func:`~..geometry.reprobe_target`. The probes are one mechanism: a run
        that skips the substrate gate runs neither.
        """
        units = assembly_units(getattr(self, "task_data", None) or {})
        if not units or check.unit != units[0] or getattr(self, "skip_substrate_gate", False):
            return
        geometry.reprobe_target(self, check.member, self._stage_note)

    def _stage_note(self, text: str) -> None:
        """Queue ``text`` for the planner's next turn (:meth:`_take_stage_notes`)."""
        with self._defect_lock:
            self.__dict__.setdefault("_stage_notes", []).append(text)

    def _take_stage_notes(self) -> list[str]:
        """What the stage queued for the planner since the orchestrator last asked."""
        with self._defect_lock:
            return self.__dict__.pop("_stage_notes", [])

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
            marker=self._commit_markers(head)[0],
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

    # ---------------------------------------------------------------- bundles
    def _check_in_bundle(
        self, seat: bundle_mod.Seat, check: contract_mod.Check, log
    ) -> tuple[Any, str]:
        """``check``'s result as a member of its bundle, and the marker it measured.

        The leader runs the bundle (:meth:`_lead_bundle`). Any other member
        is judged from its section of the leader's job, or runs singly when
        the leader assigns it no section or its section cannot be judged.
        """
        if seat.leader:
            return self._lead_bundle(seat, log)
        assignment = seat.wait()
        if assignment.role == bundle_mod.AWAIT and assignment.job is not None:
            return self._judge_section(assignment.job, check, log, admitted=False)
        return self._check_alone(check, log, assignment.why, admitted=False)

    def _check_alone(
        self, check: contract_mod.Check, log, why: str = "", *, admitted: bool
    ) -> tuple[Any, str]:
        """``check`` on a snapshot of the checkout's head, as a check outside any bundle runs.

        Admitted first unless it already holds a slot (``admitted``): a
        bundle's member waited in the bundle without one.
        """
        if why:
            self.journal.event("bundle", f"{check.key} runs singly: {why}")
        admission = getattr(self, "_check_admission", None)
        slot = (
            admission.slot(*self._admission_terms(check))
            if admission is not None and not admitted
            else contextlib.nullcontext()
        )
        with slot:
            return self._gate_on_snapshot(self._snapshots, check, log)

    def _lead_bundle(self, seat: bundle_mod.Seat, log) -> tuple[Any, str]:
        """Seal ``seat``'s bundle, run its largest group as one job, and return the leader's own result.

        Every member is captured on one snapshot of the checkout's head
        (``bundle.capture``) and the commands grouped by the engine they build
        (``bundle.form_groups``). The group of two or more with the leader in
        it runs, or the largest such group: its other members are told to
        judge their sections of that job, and every member outside it to run
        singly. Whatever stops the leader on its way, every member it had not
        yet told runs singly.
        """
        members = seat.seal()
        check = seat.check
        snapshots = self._snapshots
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        job: bundle_mod.Job | None = None
        alone: dict[str, str] = {}
        stopped = "its bundle's leader stopped before the job"
        try:
            if len(members) == 1:
                return self._check_alone(check, log, admitted=True)
            head = resolve_commit(repo)
            if head is None:
                stopped = f"git could not name the commit {repo} is at"
                return self._check_alone(check, log, admitted=True)
            pin = snapshots_mod.CheckTree(
                commit=head,
                tree=snapshots.tree(head),
                marker=self._commit_markers(head)[0],
                uncommitted=tuple(self._uncommitted_gate_inputs(repo) or ()),
                snapshots=snapshots,
            )
            with snapshots.lease(head):
                view = self._on_snapshot(pin)
                launches: list[bundle_mod.Launch] = []
                for member in members:
                    key = member.check.key
                    if key in alone or any(launch.key == key for launch in launches):
                        continue
                    captured = self._capture(view, member.check)
                    if isinstance(captured, bundle_mod.Launch):
                        launches.append(captured)
                    else:
                        alone[key] = "its gate launches nothing a bundle runs"
                groups, apart = bundle_mod.form_groups(launches)
                alone.update(apart)
                shared = [group for group in groups if len(group.launches) > 1]
                chosen = next((g for g in shared if check.key in g.keys), None) or next(
                    iter(shared), None
                )
                if chosen is not None:
                    job = bundle_mod.Job(chosen, context=pin)
                for member in members:
                    key = member.check.key
                    if member is seat:
                        continue
                    if job is not None and key in job.group.keys:
                        member.assign(bundle_mod.Assignment(bundle_mod.AWAIT, job=job))
                    else:
                        why = alone.get(key) or "no other check of its bundle builds its engine"
                        member.assign(bundle_mod.Assignment(bundle_mod.SINGLE, why=why))
                if job is None:
                    return self._check_alone(check, log, alone.get(check.key, ""), admitted=True)
                job.publish(self._run_bundle_job(view, job, check.unit, check.member))
                if check.key in job.group.keys:
                    return self._judge_section(job, check, log, admitted=True)
            why = alone.get(check.key) or "no other check of its bundle builds its engine"
            return self._check_alone(check, log, why, admitted=True)
        except snapshots_mod.SnapshotError as exc:
            result = snapshots_mod.void_result(check.gate, check.member, str(exc))
            self.journal.event("gate", result.summary())
            return result, ""
        finally:
            for member in members:
                if member.assignment is None:
                    member.assign(bundle_mod.Assignment(bundle_mod.SINGLE, why=stopped))
            if job is not None:
                job.fail("its bundle's job did not run")

    def _capture(self, view: "PlannedStage", check: contract_mod.Check) -> Any:
        """What ``check``'s gate on ``view`` would launch first (``bundle.capture``); ``None`` if nothing.

        A gate that raises is left to raise in its own run, singly.
        """
        call = view._gate_call(check)
        if call is None:
            return None
        try:
            return bundle_mod.capture(check.key, call)
        # Whatever the gate raises is its own single run's to raise, in its own thread.
        except Exception as exc:  # noqa: BLE001
            self.journal.event("bundle", f"{check.key} not captured: {type(exc).__name__}: {exc}")
            return None

    def _run_bundle_job(
        self, view: "PlannedStage", job: bundle_mod.Job, unit: str, member: str
    ) -> dict[str, bundle_mod.Section]:
        """Run ``job`` once on ``view``'s snapshot and cut its output into each member's section."""
        group = job.group
        directory = self.workspace / "gates" / unit / "bundles"
        job.outs = {
            launch.key: str(directory / f"{job.name}.{zlib.crc32(launch.key.encode()):08x}.json")
            for launch in group.launches
            if launch.kind == bundle_mod.KIND_LOGITS
        }
        spec = bundle_mod.write_spec(directory / f"{job.name}.bundle.json", group, job.outs)
        command = bundle_mod.bundle_command(group, spec)
        self.journal.event("bundle", f"{job.label}: one engine at {job.context.commit[:10]}")
        started = time.monotonic()
        output = view._member_runner(member)(command, None, group.timeout)
        sections = bundle_mod.split(output, group)
        single = [f"{key} ({s.why})" for key, s in sections.items() if s.result is None]
        self.journal.event(
            "bundle",
            f"{job.name} ran in {time.monotonic() - started:.0f}s: "
            f"{len(sections) - len(single)} judged from it"
            + (f"; singly: {'; '.join(single)}" if single else ""),
        )
        return sections

    def _judge_section(
        self, job: bundle_mod.Job, check: contract_mod.Check, log, *, admitted: bool
    ) -> tuple[Any, str]:
        """``check``'s verdict from its section of ``job``: its gate, answered with that section.

        On a snapshot of the job's commit leased for the judging, the gate
        runs again with a runner that answers its recorded command with the
        section (``bundle.replaying``) and runs any other command, the
        acceptance gate's ``=off`` leg, as before. A member with no section
        to judge runs singly.
        """
        pin = job.context
        snapshots = pin.snapshots
        try:
            with snapshots.lease(pin.commit):
                section = job.section(check.key)
                launch = job.group.launch_of(check.key)
                if section.result is None or launch is None:
                    return self._check_alone(check, log, section.why, admitted=admitted)
                own = snapshots_mod.CheckTree(
                    commit=pin.commit,
                    tree=pin.tree,
                    marker=pin.marker,
                    uncommitted=pin.uncommitted,
                    snapshots=snapshots,
                )
                view = self._on_snapshot(own)
                out = job.outs.get(check.key, "")
                runner = bundle_mod.replaying(
                    launch, section.result, view._member_runner(check.member), own.observe, out
                )
                try:
                    return view._gate(check, log, runner=runner, bundle=job.label), own.marker
                finally:
                    if out:
                        Path(out).unlink(missing_ok=True)
        except snapshots_mod.SnapshotError as exc:
            result = snapshots_mod.void_result(check.gate, check.member, str(exc))
            self.journal.event("gate", result.summary())
            return result, pin.marker

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

    def _gate(self, check: contract_mod.Check, log, *, runner: Any = None, bundle: str = "") -> Any:
        """Run ``check``'s gate on the target this workflow sees, journal it, and return its result.

        ``runner`` replaces the member's runner; ``bundle`` names the bundle
        the result was measured in (``bundle.mark``).
        """
        call = self._gate_call(check)
        if call is None:
            return self._run_witness(check.unit, check.member, log)
        chosen = runner if runner is not None else self._member_runner(check.member)
        result = self._gate_until_verdict(lambda: call(chosen), log)
        if bundle:
            result = bundle_mod.mark(result, bundle)
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
