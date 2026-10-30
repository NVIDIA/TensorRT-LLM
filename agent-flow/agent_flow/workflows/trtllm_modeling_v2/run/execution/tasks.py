# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution stage: the workflow's services handed to :class:`~..orchestrator.GraphOrchestrator`.

Entered after preflight. The planner turn and each task's worker turn are this
module's; CheckRunner owns contract checks; deciding what runs and what counts is the
orchestrator's. Once the contract holds, delivery verifies the trees on disk.

Every contract check measures a snapshot of the commit the checkout is at
when it starts (``run/workspace/snapshots.py``), never the checkout, which integrations
move while checks run. Its verdict names that commit, is recorded under its
marker, and is void when a command imported any other tree.

A task that finishes leaves its receipt (``run/execution/handoff.py``), and a task
that starts carries the receipt of each task it depends on. A gate that
passes adds the engines it attested to ``runtime_facts.yaml``, which the
planner's and every worker's prompt carries.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ...agents.decisions import DECISION_REMINDER
from ...agents.prompts.survey import capture_lists_block
from ...agents.prompts.tags import tag_section, task_tags
from ...checks import check_config, gate_cli
from ...infra.compute.scheduler import WATCHER_STALE_S, watcher_records
from ...infra.git import CommitPolicy
from ...operators import promotion as promotion_mod
from ...operators import survey_review
from ...operators.coverage import QueueItem, VocabularyError, load_vocabulary, queue_item_from_ref
from ...spec import lint as lint_mod
from ...spec import variants as variants_mod
from ...spec.task_schema import assembly_units, members_for, parse_parallel_segment
from ...tune import stage as tune_stage
from .. import graph as graph_mod
from .. import inbox as inbox_mod
from ..orchestrator import (
    CHECKED,
    DEFAULT_MAX_PARALLEL_CHECKS,
    FINISHED,
    GraphOrchestrator,
    PlannerAnswer,
    PlannerTurn,
    Services,
    seed,
)
from ..records.state import WorkflowState
from ..validation import completion as deliver
from ..validation import contract as contract_mod
from ..validation import regression as regression_mod
from ..validation import runtime_facts
from ..validation.check_runner import CheckRunner
from ..workspace import snapshots as snapshots_mod
from ..workspace.integration import (
    INTEGRATED_KINDS,
    REBASE_RETRIES,
    SCRATCH_KINDS,
    Integration,
    TaskNotStarted,
    rebase_block,
    repair_block,
    soft_dependency_block,
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
    engine_defect_report,
    mark_decision_dispatch,
    parse_constituents,
    uncertified_report,
)
from .kernel_task import FAILED_RECEIPT

SURVEY_TASK_ID = "survey"

PLANNER_ROLE = "planner"
PROPOSED_GRAPH = "graph.proposed.yaml"
TASKS_DIRNAME = "tasks"
#: How many of the latest measured changes the planner's request lists.
PLANNER_FACT_CHANGES = 8
#: How many events, and operator messages, one planner turn lists (the newest), and
#: the most of each one: a burst while a long turn ran must not swamp the next.
PLANNER_EVENTS = 40
PLANNER_EVENT_CHARS = 1_500

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
#: Checks in flight at once on a run's sessions: each is a spool job, and the
#: session's scheduler queues what its devices cannot start yet, by class.
MAX_PARALLEL_CHECKS = 32
#: The task kind that runs the tune campaign.
TUNE_KIND = "tune"


class ExecutionStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    def __init__(self, workflow) -> None:
        self.workflow = workflow
        self._turn_lock = threading.Lock()
        self._turns_by_key = {}
        self._tasks_dispatched = set()
        self._regression_cleared = {}
        self._landed_attempts = {}
        self._merged = {}
        self._tls = workflow._tls

    # ------------------------------------------------------------------ entry
    def run_execution(self, state: WorkflowState, log) -> bool:
        print_rule("[bold magenta]execution stage[/bold magenta]", log)
        contract = self.workflow._contract
        self.state = state
        tune = self.workflow._tune_requested()
        units = assembly_units(self.workflow.task_data)
        # The campaign starts once onboarding is complete: every required check has passed.
        onboarded = [c.key for c in contract.required()]
        core_boots = [
            c.key for c in contract.required() if units and c.unit == units[0] and c.gate == "boot"
        ]
        feature_refs = [*units[1:], *variants_mod.variants(self.workflow.task_data)]
        total = max(len(state.devices), self.workflow._gate_env().world_size, 1)
        self._integration = self._make_integration(state)
        self.workflow._snapshots = self._make_snapshots()
        self.checks = CheckRunner(self.workflow)
        services = Services(
            plan=lambda turn: self._planner_turn(turn, contract),
            work=self._work_task,
            check=lambda check: self.checks._prioritized_check(check, log),
            free_gpus=lambda: total,
            checkpoint=lambda: self.workflow._checkpoint(state),
            log=lambda message: self.workflow.journal.event("execution", message),
            marker=self.workflow._target_marker,
            finished=self._task_finished,
            landed=self._landed_unfinished,
            correct=self._mark_correct,
            bringup=getattr(self.workflow, "_mark_bringup", None),
            cells=getattr(self.workflow, "_mark_cells", None),
            promotion=lambda graph: promotion_mod.plan(
                graph.tasks,
                self.workflow.task_data["trtllm_repo_path"],
                self.workflow._preflight.target_rel,
                keep=promotion_mod.keep_local(self.workflow.task_data),
            ),
            delivery_files=lambda: deliver.delivery_failures(self.workflow),
            stage_notes=self._take_stage_notes,
            narrow=getattr(self.workflow, "_narrowed", None),
            holds=getattr(self.workflow, "_verdict_holds", None),
            regression=self._run_regression_check,
        )
        self._own_paths = self._regression_paths()
        orchestrator = GraphOrchestrator(
            services,
            contract,
            self.workflow._contract_status(),
            self.workflow.workspace,
            allowed_paths=CommitPolicy.from_task(
                self.workflow.task_data, workspace=self.workflow.workspace
            ).allowed_paths,
            max_gpus=total,
            max_parallel=self.workflow.max_parallel_tasks,
            max_parallel_checks=(
                MAX_PARALLEL_CHECKS
                if self.workflow._gate_env().kind == "session_spool"
                else DEFAULT_MAX_PARALLEL_CHECKS
            ),
            revision_cap=self.workflow.revision_cap,
            event_window_s=self.workflow.event_window_s,
            required_kinds=("tune",) if tune else (),
            start_after={"tune": onboarded} if onboarded else {},
            after_core=(feature_refs, core_boots),
            own_paths=self._own_paths,
            repo_root=self.workflow.task_data.get("trtllm_repo_path") or "",
        )

        # A tune task whose jobs all run on the perf sessions holds none of the run's.
        offsite = offsite_kinds(self.workflow.task_data.get("perf"), self.workflow._gate_env())

        def held(task: graph_mod.Task) -> int:
            # A running task waiting on its soft dependencies or running its checks
            # holds none: its checks queue for the sessions' devices of their own.
            return 0 if orchestrator.agentless(task.id) else held_gpus(task, offsite)

        def free_gpus() -> int:
            return self._task_capacity(total, orchestrator.graph, held)

        services.free_gpus = free_gpus
        services.held_gpus = held
        self._soft_dependencies = orchestrator.soft_dependencies
        self._mark_integrated = orchestrator.mark_integrated
        # A campaign's turns, and the side turns, take only the agent slots the
        # foreground work leaves idle.
        gate = getattr(self.workflow, "_agent_gate", None)
        if gate is not None:
            gate.background_limit = orchestrator.idle_agent_slots
            gate.side_limit = orchestrator.idle_agent_slots
        # Beside the planner's first turn, as the run's other early work is.
        self.workflow._start_primers()
        try:
            verdict = orchestrator.run()
        finally:
            # No foreground work is left to yield to: the closing turns run at once.
            if gate is not None:
                gate.background_limit = gate.side_limit = None
        marker = self.workflow._target_marker()
        evaluation = contract_mod.evaluate(
            contract,
            self.workflow._contract_status(),
            marker=marker,
            holds=getattr(self.workflow, "_verdict_holds", None),
        )
        self.workflow.journal.section(
            f"execution stage: {verdict}",
            contract_mod.standing(evaluation, self.workflow.workspace, marker),
        )
        if verdict.startswith("stopped"):
            # Preflight's substrate gate, running beside the first turns, found the run
            # cannot go on here: nothing is closed, and the same command resumes once fixed.
            print_message(
                f"[bold yellow]⚠ execution stage {escape_markup(verdict)}; the run stays open: "
                f"re-run the same command once the cause is fixed[/bold yellow]",
                log,
            )
            return False
        if verdict not in ("delivered", "delivery-blocked"):
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
        """Mark ``task``'s answered turns consumed once it has an outcome or is retried.

        A relaunch adopts only the answered turn of a task with no outcome yet; a
        consumed turn is never adopted, so a retry is always asked afresh. A
        ``check`` task that passed, which no worker ran, leaves its receipt here.
        """
        self.workflow._turns.consume([task.id])
        if task.kind == graph_mod.CHECK_KIND and task.status == "done":
            cleared = self._regression_cleared.get(task.id, "")
            if task.regression:
                note = "the package tests passed after the changes to shared code it covers"
            else:
                note = f"its checks passed, with no agent turn: {', '.join(task.satisfies)}"
            try:
                handoff_mod.write(
                    self.workflow.workspace / TASKS_DIRNAME,
                    handoff_mod.build(
                        task.id,
                        task.kind,
                        result=TaskResult(CHECKED, note),
                        attempt=task.attempts,
                        cleared=cleared,
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
            t.id for t in tasks.values() if t.status in ("done", "failed", "abandoned", "split")
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

    # ------------------------------------------------------- regression checks
    def _run_regression_check(self, task: graph_mod.Task) -> list[str]:
        """Run one regression check, with no agent turn; what failed, one line each.

        In a read-only snapshot of the commit the checkout is at, as test jobs
        on the run's devices (``run/validation/regression.py``). The head it finds clean
        is the receipt's ``cleared``, the next check's base.
        """
        repo = Path(str(self.workflow.task_data.get("trtllm_repo_path") or ""))
        head = self.workflow._head()
        if not head:
            return [f"{repo} has no commit to check"]
        tree_at, run_job, python = regression_mod.job_runner(self.workflow, f"regression-{task.id}")
        result = regression_mod.run_regression_check(
            workspace=self.workflow.workspace,
            repo=repo,
            head=head,
            tree_at=tree_at,
            run_job=run_job,
            python=python,
            name=task.id,
        )
        self.workflow.journal.event("regression", f"{task.id}: {result.summary()}")
        if result.passed:
            with self._turn_lock:
                self._regression_cleared[task.id] = head
        return result.failures

    # ------------------------------------------------------ worktrees, devices
    def _make_integration(self, state: WorkflowState) -> Integration | None:
        """Per-task worktrees on a git checkout; on anything else tasks share the tree."""
        repo = Path(str(self.workflow.task_data.get("trtllm_repo_path") or ""))
        if not (repo / ".git").exists():
            self.workflow.journal.event("integration", f"tasks share the checkout {repo}")
            return None
        policy = CommitPolicy.from_task(self.workflow.task_data, workspace=self.workflow.workspace)
        env = {}
        if policy.identity is not None:
            # The rebase's committer; each commit keeps its author.
            env = {k: v for k, v in policy.identity.env().items() if k.startswith("GIT_COMMITTER")}
        return Integration(
            repo,
            self.workflow.workspace,
            f"mv2/{state.slug or 'run'}",
            env,
            log=lambda message: self.workflow.journal.event("integration", message),
            policy=policy,
        )

    def _make_snapshots(self) -> snapshots_mod.Snapshots | None:
        """Snapshots of the checkout's commits for the checks to measure; on a git checkout only.

        Each is sparse with a task worktree's cone and the directories of the
        files the gates read. What an earlier process left that no check
        needs any more goes now.
        """
        from ..workflow import gate_inputs

        repo = Path(str(self.workflow.task_data.get("trtllm_repo_path") or ""))
        if not (repo / ".git").exists():
            self.workflow.journal.event(
                "snapshots", f"checks measure {repo} as it is: not a git checkout"
            )
            return None
        snapshots = snapshots_mod.Snapshots(
            repo,
            self.workflow.workspace / snapshots_mod.SNAPSHOTS_DIRNAME,
            cone_paths=gate_inputs(self.workflow.task_data, repo),
            log=lambda message: self.workflow.journal.event("snapshots", message),
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
        (:func:`held_gpus`), whatever its gate jobs hold at the moment.
        ``total`` stands in for the sessions when none is live.
        """
        capacity = self._spool_capacity()
        if capacity is None:
            capacity = total
        return capacity - sum(held(t) for t in graph.tasks if t.status == "running")

    def _spool_capacity(self) -> int | None:
        """Devices in service on the live spool schedulers, or None without a live one."""
        env = self.workflow._gate_env()
        spools = {env.spool, *(session.spool for session in env.sessions)} - {None, ""}
        capacity, live, now = 0, False, time.time()
        for spool in spools:
            records = [
                r
                for r in watcher_records(spool)
                if now - float(r["alive_epoch"]) <= WATCHER_STALE_S
            ]
            if not records:
                continue
            live = True
            # Count every node of the latest allocation; its retiring predecessor
            # contributes no additional capacity during handover.
            newest = max(
                records, key=lambda r: (float(r.get("ends_at_epoch") or 0), float(r["alive_epoch"]))
            )
            hosts = {}
            for record in sorted(records, key=lambda r: float(r["alive_epoch"])):
                if record.get("slurm_job") == newest.get("slurm_job"):
                    hosts[record.get("host") or record.get("watcher")] = record
            capacity += sum(
                len(set(r.get("devices") or []) - set(r.get("quarantined") or []))
                for r in hosts.values()
            )
        return capacity if live else None

    # ---------------------------------------------------------------- planner
    def _regression_paths(self) -> tuple[str, ...]:
        """The target's own paths (:func:`~..graph.own_paths`); none before preflight names it."""
        preflight = getattr(self.workflow, "_preflight", None)
        return graph_mod.own_paths(str(getattr(preflight, "target_rel", "") or ""))

    def _assembler_prompt(self, unit: str) -> str:
        if variants_mod.is_variant(self.workflow.task_data, unit):
            return self._variant_prompt(unit)
        command = gate_cli.invocation(
            self.workflow.task_path, self.workflow.workspace / "gates", unit, self._cli()
        )
        return (
            f"Workspace: {self.workflow.workspace}\n"
            f"Repository: {self.workflow.task_data['trtllm_repo_path']}\n"
            f"Target: {self.workflow._preflight.target_rel}\n"
            f"Unit: **{unit}**\n\n"
            f"Read `{self.workflow.task_path}` (the spec), `{self.workflow.plan_path}` (the plan) and "
            f"`{self.workflow.vocabulary_path}` (the computation-step list you derive against). "
            f"Earlier turns on this unit, if any, left their decision files in "
            f"`{self.workflow.workspace / DECISIONS_DIRNAME}` and their reports in "
            f"`{self.workflow.journal.reports_dir}`: build on them rather than starting over.\n\n"
            f"{self._gate_history_block()}"
            f"Write the unit. To reproduce one gate while debugging, run "
            f"the workflow's own entry point, never a reconstruction of its command: "
            f"`{command}`.\n\n" + DECISION_REMINDER
        )

    def _cli(self) -> str:
        """The workflow's CLI as an agent calls it: by path, when preflight installed it."""
        script = self.workflow.workspace / "bin" / gate_cli.CLI_NAME
        return str(script) if script.is_file() else gate_cli.CLI_NAME

    def _variant_prompt(self, name: str) -> str:
        """An assembler turn on a configuration variant: the unit's target, served another way."""
        base = variants_mod.base_unit(self.workflow.task_data, name)
        command = gate_cli.invocation(
            self.workflow.task_path, self.workflow.workspace / "gates", name, self._cli()
        )
        return (
            f"Workspace: {self.workflow.workspace}\n"
            f"Repository: {self.workflow.task_data['trtllm_repo_path']}\n"
            f"Target: {self.workflow._preflight.target_rel}\n"
            f"Unit: **{base}**, configuration variant **{name}**\n\n"
            f"{variants_mod.describe(self.workflow.task_data, name)}\n\n"
            f"The `{base}` target exists. Make it serve this configuration as well, without "
            f"changing what it does in its own: every `{base}` check must keep passing. The "
            f"variant's checks run with `{base}`'s settings plus the overlay above. Read "
            f"`{self.workflow.task_path}` (the spec), and for earlier turns the decision files in "
            f"`{self.workflow.workspace / DECISIONS_DIRNAME}` and the reports in "
            f"`{self.workflow.journal.reports_dir}`.\n\n"
            f"{self._gate_history_block()}"
            f"Reproduce one of its checks with "
            f"`{command}`.\n\n" + DECISION_REMINDER
        )

    def _gate_history_block(self) -> str:
        rendered = self.workflow.gate_log.render()
        if not rendered:
            return ""
        return f"Gates so far:\n\n```\n{rendered}\n```\n\n"

    def _start_member(self, contract: contract_mod.Contract) -> str:
        """Where the core and each feature's first task work: the cheapest member one engine fits.

        Sized as the intake lint sizes members (``spec.lint.engine_width``): the
        checkpoint's weight bytes against ``environment.gpu_memory_gib``.
        """
        cached = getattr(self, "_start_line", None)
        if cached:
            return cached
        task = self.workflow.task_data
        first = assembly_units(task)[0]
        members = check_config.order_members_by_cost(
            list(dict.fromkeys(c.member for c in contract.checks if c.unit == first))
            or members_for(task, first, self.workflow._gate_env().world_size)
        )
        weights = lint_mod.checkpoint_weight_bytes(task.get("checkpoint_path"))
        memory = (task.get("environment") or {}).get("gpu_memory_gib")
        width = lint_mod.engine_width(weights, memory)
        if width is None:
            line = (
                f"`{members[0]}`: the cheapest member the contract checks the core at (the run "
                "states no `environment.gpu_memory_gib`, so one engine is not sized)."
            )
        else:
            fits = [m for m in members if (parse_parallel_segment(m)[1] or 1) >= width]
            start = fits[0] if fits else members[-1]
            line = (
                f"`{start}`: one engine of the checkpoint takes {width} device(s) (its weights "
                f"at most {lint_mod.ENGINE_WEIGHT_FRACTION:.0%} of each {memory:g}-GiB device), "
                f"and `{start}` is the cheapest member the contract checks the core at that "
                "holds it. A narrower one cannot load the model."
            )
        self._start_line = line
        return line

    def _planner_turn(self, turn: PlannerTurn, contract: contract_mod.Contract) -> PlannerAnswer:
        proposed = self.workflow.workspace / PROPOSED_GRAPH
        current = self.workflow.workspace / graph_mod.GRAPH_FILENAME
        # Each turn writes its own proposal: a stale one would read as this turn's.
        proposed.unlink(missing_ok=True)
        lines = [
            f"Task: `{self.workflow.task_path}`",
            f"Contract: `{self.workflow.workspace / contract_mod.CONTRACT_FILENAME}`",
            f"Verdicts: `{self.workflow.workspace / contract_mod.STATUS_FILENAME}`",
            f"Current graph: `{current}`" if current.exists() else "There is no graph yet.",
            f"Write the next graph to `{proposed}`.",
            "",
            "## Contract standing",
            "",
            contract_mod.standing(turn.evaluation, self.workflow.workspace, turn.marker),
            "",
            "## Start member",
            "",
            self._start_member(contract),
        ]
        facts = runtime_facts.render(self.workflow.workspace, changes=PLANNER_FACT_CHANGES)
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
        messages = _newest([e for e in turn.events if inbox_mod.is_message(e)], "messages")
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
        events = _newest([e for e in turn.events if not inbox_mod.is_message(e)], "events")
        if events:
            lines += ["", "## Since your last turn", "", *[f"- {e}" for e in events]]
        if turn.refusals:
            lines += ["", "## Your last graph was refused", "", *[f"- {r}" for r in turn.refusals]]
        result = self._dispatch_task(PLANNER_ROLE, "plan", "planner", "\n".join(lines))
        decision = result.decision
        summary = str(result.data.get("summary") or "").strip()
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
        path = self.workflow.workspace / TASKS_DIRNAME / f"{task.id}.json"
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
        if task.kind in INTEGRATED_KINDS and decision.decision not in FINISHED:
            tip = self._integrated_tip(task.id)
            if tip and tip != before:
                with self._turn_lock:
                    self._landed_attempts[task.id] = task.attempts
        self._keep_receipt(task, decision)
        return decision.text

    def _integrated_tip(self, task_id: str) -> str:
        """The branch tip the integration last fast-forwarded in for ``task_id``; "" for none."""
        integration = getattr(self, "_integration", None)
        return integration.integrated_tip(task_id) if integration is not None else ""

    def _landed_unfinished(self, task: graph_mod.Task) -> bool:
        """Whether ``task``'s latest attempt landed commits though its turn did not finish.

        A failed receipt's guard tests do (``_land_failed_receipt``): they are a
        change to the code they guard like any other, and a regression check
        covers them.
        """
        return self._landed_attempts.get(task.id) == task.attempts

    def _work_task_turn(self, task: graph_mod.Task) -> TaskResult:
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
        try:
            tree = integration.open(task.id, task.touches, full=task.exclusive)
        # Whatever stopped the checkout (a full filesystem, git), no worker ran.
        except Exception as exc:
            raise TaskNotStarted(f"{type(exc).__name__}: {exc}") from exc
        block = worktree_block(tree, integration.repo, scratch=task.kind in SCRATCH_KINDS)
        decision = self._work_in_tree(task, block)
        if task.kind not in INTEGRATED_KINDS:
            return decision
        decision = self._after_soft_dependencies(task, decision)
        attempts = 0
        while decision.decision in FINISHED:
            outcome = integration.integrate(task.id)
            self.workflow.journal.event("integration", f"{task.id}: {outcome.detail}")
            if outcome.merged:
                integration.close(task.id)
                # How many commits went in: the receipt lists them.
                self._merged[task.id] = outcome
                mark = getattr(self, "_mark_integrated", None)
                if mark is not None:
                    mark(task.id, task.attempts)
            if outcome.merged:
                return decision
            if not (outcome.base or outcome.repairable) or attempts == REBASE_RETRIES:
                decision.decision = "INTEGRATION_FAILED"
                return decision.annotate(outcome.detail)
            attempts += 1
            role = ROLE_BY_KIND.get(task.kind, task.kind)
            if outcome.base:
                self.workflow.journal.event(
                    "integration",
                    f"{task.id}: back to its {role} to rebase onto {outcome.base[:10]} and "
                    f"resolve ({attempts} of {REBASE_RETRIES})",
                )
                note = rebase_block(outcome, tree)
            else:
                self.workflow.journal.event(
                    "integration",
                    f"{task.id}: back to its {role} to put its worktree right "
                    f"({attempts} of {REBASE_RETRIES}): {outcome.detail.splitlines()[0]}",
                )
                note = repair_block(outcome)
            decision = self._work_in_tree(task, block, note)
        return decision

    def _after_soft_dependencies(self, task: graph_mod.Task, decision: TaskResult) -> TaskResult:
        """``decision``, once each soft dependency of a finished ``task`` has integrated or ended.

        Its integration, and the checks after it, then read a head that holds
        what landed. A dependency that ended without landing is named in a
        line of its own after the decision: the planner's event and the
        task's receipt carry it. Outside the execution loop nothing is waited
        for; a loop that ends first leaves the task unfinished.
        """
        wait = getattr(self, "_soft_dependencies", None)
        if not task.depends_on_soft or wait is None or decision.decision not in FINISHED:
            return decision
        ended = wait(task.id)
        if ended is None:
            decision.decision = "the run ended before the task's soft dependencies landed"
            return decision
        missed = sorted(
            dep for dep, status in ended.items() if status not in ("done", "integrated")
        )
        self.workflow.journal.event(
            "soft dependencies",
            f"{task.id}: integrates once "
            + ", ".join(f"{dep} {status}" for dep, status in ended.items()),
        )
        if not missed:
            return decision
        return decision.annotate(
            f"soft dependencies {', '.join(f'{dep} ({ended[dep]})' for dep in missed)} "
            f"ended without landing: this task integrated without their change"
        )

    def _work_in_tree(self, task: graph_mod.Task, block: str, note: str = "") -> TaskResult:
        """``task``'s worker, told its worktree (``block``) and, on a rebase, what to do (``note``)."""
        self._tls.worktree = block
        try:
            item = self._queue_item(task) if note and task.kind == "kernel" else None
            if item is not None:
                # A kernel writer given a note is dispatched afresh rather than handed the
                # decision already on disk: that decision is the work to rebase.
                return kernel_writer._run_one_kernel_writer(self, item, note, task)
            if note:
                self._tls.worktree = f"{block}\n\n{note}"
            return self._work_task_body(task)
        finally:
            self._tls.worktree = ""

    def _work_task_body(self, task: graph_mod.Task) -> TaskResult:
        receipts = self._dependency_receipts(task)
        envelope = self._envelope(task, receipts)
        if task.kind == "tune":
            decision, note = tune_stage.run_tune_task(
                self.workflow, self.state, self.workflow._head(), attempt=task.attempts
            )
            self.workflow._checkpoint(self.state)
            return TaskResult(decision, note)
        if task.kind == "kernel":
            item = self._queue_item(task)
            if item is None:
                return TaskResult(
                    f"kernel task {task.id} names no queue row or `<mode>:<entry>` ({task.ref!r})"
                )
            # A new retry needs a fresh dispatch. A relaunch can instead resume the
            # same attempt, whose completed answer still awaits integration.
            resuming = decision_dispatch_matches(
                current_decision(self.workflow.workspace, "kernel_writer", item.n, unit=item.unit),
                {"task_id": task.id, "task_attempt": task.attempts},
            )
            retry = (
                f"Attempt {task.attempts} of task {task.id}; the previous attempt failed."
                if task.attempts > 1 and not resuming
                else ""
            )
            with self._carrying(self._task_context(receipts, item.unit, task)):
                result = kernel_writer._run_one_kernel_writer(self, item, retry, task)
            return result.annotate(self._engine_defect_said(result) or self._macro_rows(result))
        role = ROLE_BY_KIND[task.kind]
        unit = task.ref or task.id
        if task.kind in ("assemble", "feature"):
            prompt = self._assembler_prompt(unit)
        elif task.kind == "survey":
            prompt = _surveyor_prompt(self.workflow)
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
        dispatch = "\n\n".join(p.strip() for p in parts if p)
        result = self._dispatch_task(role, unit, task.id, dispatch)
        if role == "assembler" and result.decision == "BLOCKED":
            result.annotate(self._blocked_report(result.data))
        if role == "assembler" and result.decision == "UNIT_READY":
            result.annotate(uncertified_report(result.data.get("uncertified")))
        if task.kind == "survey":
            result = self._checked_survey(result, role, unit, task.id, dispatch)
        return result

    def _checked_survey(
        self, result: TaskResult, role: str, unit: str, task_id: str, dispatch: str
    ) -> TaskResult:
        """A drafted survey, validated, then reviewed; sent back once with what the review found.

        The review (``operators/survey_review.py``) holds the vocabulary to the
        prior art and engine limits the dispatch carried. Its findings go back
        to the surveyor once per run (``survey_review.md`` records it); after
        that they stand in the journal and the draft is taken as it is.
        """
        if result.decision != "DRAFT_READY":
            return result
        try:
            load_vocabulary(
                self.workflow.vocabulary_path, units=assembly_units(self.workflow.task_data)
            )
        except VocabularyError as exc:
            result.decision = f"vocabulary.yaml does not validate: {exc}"
            return result
        try:
            plan = self.workflow.plan_path.read_text(encoding="utf-8")
        except OSError:
            plan = ""
        findings = survey_review.review_survey(
            yaml.safe_load(self.workflow.vocabulary_path.read_text(encoding="utf-8")) or {},
            plan,
            getattr(self.workflow, "_prior_art", None),
            getattr(self.workflow, "_engine_limits", None) or (),
        )
        record = self.workflow.workspace / survey_review.REVIEW_FILENAME
        if not findings:
            return result
        if record.exists():
            self.workflow.journal.section(
                f"survey review: {len(findings)} finding(s) stand after the re-draft",
                "\n".join(f"- {f}" for f in findings),
            )
            return result
        review = survey_review.render_review(findings)
        record.write_text(review + "\n", encoding="utf-8")
        self.workflow.journal.section(
            f"survey review: sent back with {len(findings)} finding(s)", review
        )
        again = self._dispatch_task(role, unit, task_id, f"{dispatch}\n\n{review}", fresh=True)
        return self._checked_survey(again, role, unit, task_id, dispatch)

    def _engine_defect_said(self, result: TaskResult) -> str:
        """What the planner hears of the engine defect a failed receipt reports; ``""`` without one.

        Where the defect is, what triggers it and the fix proposed
        (``handoff.engine_defect_report``): the planner plans the fix, and
        retries the task once it has landed.
        """
        if result.decision != FAILED_RECEIPT:
            return ""
        report = engine_defect_report(result.data.get("engine_defect"))
        return (
            f"it reports an {report}: plan its fix, and retry this task once it lands"
            if report
            else ""
        )

    def _blocked_report(self, raw: Mapping[str, Any]) -> str:
        """What a ``BLOCKED`` assembler wrote the planner: why, the missing entry, its summary."""
        said = [f"{k} {raw[k]}" for k in ("blocked_reason", "missing_entry") if raw.get(k)]
        if raw.get("summary"):
            said.append(str(raw["summary"]).strip()[:600])
        return "; ".join(said)

    def _macro_rows(self, result: TaskResult) -> str:
        """What the planner hears of a ``MACRO``: its constituents, a kernel task each."""
        if result.decision != "MACRO":
            return ""
        children = parse_constituents(result.data.get("constituents"))
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

    def _queue_item(self, task: graph_mod.Task) -> QueueItem | None:
        """The kernel writer dispatch a kernel task names.

        ``ref`` is a vocabulary queue row number, ``<mode>:<entry>[@<site>]``
        for an entry the planner adds itself (a MACRO's constituents, say), or
        ``promote:<entry>[,<entry>...]`` (:func:`~...operators.coverage.queue_item_from_ref`).
        """
        if task.ref.isdigit():
            queue = load_vocabulary(
                self.workflow.vocabulary_path, units=assembly_units(self.workflow.task_data)
            )
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
            handoff_mod.render(receipts),
            runtime_facts.render(self.workflow.workspace, unit=unit),
            self._soft_block(task) if task is not None else "",
        )
        return "\n\n".join(block for block in blocks if block)

    def _tasks_on_disk(self) -> dict[str, graph_mod.Task]:
        """Read the current graph, or no tasks if it is absent or unreadable mid-write."""
        try:
            return graph_mod.Graph.load(self.workflow.workspace / graph_mod.GRAPH_FILENAME).by_id()
        except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
            return {}

    def _soft_block(self, task: graph_mod.Task) -> str:
        """Where ``task``'s soft dependencies stand as it starts, read off ``graph.yaml``."""
        if not task.depends_on_soft:
            return ""
        tasks = self._tasks_on_disk()
        states = {
            dep: tasks[dep].status if dep in tasks else "running" for dep in task.depends_on_soft
        }
        return soft_dependency_block(
            states,
            self.workflow.workspace / TASKS_DIRNAME,
            str(self.workflow.task_data.get("trtllm_repo_path") or ""),
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
        tasks = self._tasks_on_disk()
        tasks_dir = self.workflow.workspace / TASKS_DIRNAME
        found = []
        for dep in [*task.depends_on, *task.depends_on_soft]:
            done = tasks.get(dep)
            if done is not None and done.status != "done":
                continue
            receipt = handoff_mod.read(tasks_dir, dep)
            if receipt is not None:
                found.append(receipt)
        return found

    def _keep_receipt(self, task: graph_mod.Task, result: TaskResult) -> None:
        """Keep the successful dispatch's evidence, without reconstructing it from journal prose."""
        if result.decision not in FINISHED:
            return
        try:
            handoff_mod.write(
                self.workflow.workspace / TASKS_DIRNAME,
                handoff_mod.build(
                    task.id,
                    task.kind,
                    result=result,
                    attempt=task.attempts,
                    commits=self._landed_commits(task.id),
                ),
            )
        except OSError as exc:
            self.workflow.journal.event("receipt not kept", f"{task.id}: {exc}")

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
    def _mark_correct(
        self, marker: str, evaluation: contract_mod.Evaluation
    ) -> contract_mod.Milestone | None:
        """Every required check holds on the target (``marker``): name its commit, record it.

        The branch is ``<target branch>-correct`` (:meth:`_deliver_milestone`).
        """
        return self.workflow._deliver_milestone(
            contract_mod.MILESTONE_CORRECTNESS,
            marker,
            evaluation,
            self.workflow._contract.required(),
            f"all {len(evaluation.passed)} required checks of the contract hold on it, "
            "the delivery tier's included",
        )

    def _take_stage_notes(self) -> list[str]:
        """The operator's messages since the orchestrator last asked (``run/inbox.py``)."""
        messages = inbox_mod.take(self.workflow.workspace)
        if messages:
            self.workflow.journal.event("operator", "\n".join(messages))
        return messages


def offsite_kinds(perf: Mapping[str, Any] | None, environment: Any) -> frozenset[str]:
    """The task kinds none of whose jobs runs on the run's sessions: ``{"tune"}`` or none.

    A ``tune`` task measures on ``perf.spools``, and gates its keeps and runs its tuners'
    jobs on ``perf.dev_spools`` (the measuring spools without them). Once ``perf.spools``
    is set and none of those is one of the run's sessions (``environment.spool`` and
    ``environment.sessions``), its ``gpus`` are never the sessions' to hold. Without
    ``perf.spools`` the campaign measures on the run's own sessions, and its ``gpus``
    count as any task's.
    """
    if not isinstance(perf, Mapping) or not perf.get("spools"):
        return frozenset()
    measuring = [str(spool) for spool in perf["spools"]]
    its_own = {*measuring, *(str(spool) for spool in perf.get("dev_spools") or measuring)}
    sessions = {environment.spool, *(session.spool for session in environment.sessions)} - {
        None,
        "",
    }
    resolved = lambda spool: str(Path(spool).expanduser().resolve())  # noqa: E731
    if {resolved(s) for s in its_own} & {resolved(str(s)) for s in sessions}:
        return frozenset()
    return frozenset({TUNE_KIND})


def held_gpus(task: Any, offsite: frozenset[str] = frozenset()) -> int:
    """The devices of the run's sessions ``task`` holds while it runs.

    Its ``gpus``; none for a task of a kind in ``offsite`` (:func:`offsite_kinds`).
    """
    return 0 if task.kind in offsite else max(0, int(task.gpus or 0))


def _newest(items: list[str], what: str) -> list[str]:
    """The newest :data:`PLANNER_EVENTS` of ``items``, each clipped, and a line for the rest."""
    shown = [handoff_mod.clip(str(item), PLANNER_EVENT_CHARS) for item in items[-PLANNER_EVENTS:]]
    left = len(items) - len(shown)
    if left:
        shown.insert(0, f"({left} earlier {what} left out; the journal has every one)")
    return shown


__all__ = ["PLANNER_ROLE", "PROPOSED_GRAPH", "ExecutionStage", "ROLE_BY_KIND"]


def _start_with_the_survey(self) -> bool:
    """Put the survey in a new run's graph, beside the planner's first turn; whether it did.

    A run with a graph already, or with a vocabulary, has had its survey.
    The task's ``ref`` is the target's directory, where the orchestrator
    places the target's own entries when it adds the kernel tasks.
    """
    if not self.survey_first or self.vocabulary_path.exists():
        return False
    survey = graph_mod.Task(
        id=SURVEY_TASK_ID,
        kind="survey",
        goal=(
            "map the checkpoint's computation steps onto the catalog: write plan.md and "
            "vocabulary.yaml"
        ),
        gpus=1,
        est_minutes=120,
        ref=str(getattr(self._preflight, "target_rel", "") or ""),
    )
    seeded = seed(
        self.workspace,
        [survey],
        "the run starts with the survey",
        f"the run started with the survey (task `{SURVEY_TASK_ID}`), which runs beside "
        "this turn; keep it in your graph as it is. When it lands, the orchestrator adds "
        "one kernel task per vocabulary queue row (`k<row>-<entry>`) and they start at "
        "once. Write the first graph: the work that needs no queue row, such as the "
        "engine's configuration and cache work, starts now beside the survey",
    )
    if seeded:
        self.journal.event("phase 0", "the survey starts beside the planner's first turn")
    return seeded


def _surveyor_prompt(self) -> str:
    """The survey task: map the checkpoint onto the catalog."""
    return (
        f"Workspace: {self.workspace}\n\n"
        f"Read `{self.task_path}` for the run's spec: the checkpoint, the "
        f"architecture, the parallel segment, the features and the accuracy anchor.\n\n"
        f"{self._surface_block}\n"
        f"{getattr(self, '_prior_art_block', '')}\n"
        f"{capture_lists_block(self.task_data)}\n"
        f"Write `{self.plan_path}` and `{self.vocabulary_path}` as the `survey` "
        f"section specifies. The orchestrator parses `vocabulary.yaml` and refuses "
        f"a malformed one, so follow the schema exactly.\n\n" + DECISION_REMINDER
    )
