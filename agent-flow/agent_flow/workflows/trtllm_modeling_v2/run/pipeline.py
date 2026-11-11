# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline: the run's tasks in one fixed order, then the final certification.

The order is the code's, not an agent's::

    survey
    for each deployment of the spec, in order (the first is `core`):
        one kernel task per vocabulary queue row it needs
        the deployment itself (assemble for core, feature for the rest)
    certification of the final commit (run/certify.py)

Tasks run one at a time, each from the commits of the ones before it (queue rows in
dependency order). Each task is one agent turn in its own worktree; when the worker says
it finished, its branch is audited and integrated (run/integration.py). The
worker's conclusion is trusted: no check runs between tasks. Every check of
the contract runs once, on the final commit.

The planner is asked only when the fixed order cannot go on by itself: a
worker ended its task without finishing it, a task spent its attempts, the
certification failed, or the operator left a message. It answers with one of
four decisions (``agents/decisions.py``): ``CONTINUE``, ``REPAIR`` (a few
tasks to run first), ``SKIP`` (a failed kernel task) or ``ESCALATE``. A
planner answer that cannot be read is asked for once more, then replaced by
the default: go on while attempts are left, else stop.

A task has exactly three states an agent ever sees: pending, done, failed.
``tasks.yaml`` in the workspace is the list and where each task stands; the
run resumes from it.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ..agents import decisions as decision_table
from ..agents.decisions import (
    CONTINUE,
    DECISION_REMINDER,
    ESCALATE,
    REPAIR,
    SKIP,
    clip,
    decision_file_instruction,
    decision_path,
    decision_problem,
    decision_repair_block,
    first_paragraph,
    read_decision,
)
from ..agents.prompts import planner as planner_prompt
from ..agents.prompts.kernel import KernelContext, kernel_prompt
from ..agents.prompts.survey import capture_lists_block
from ..agents.prompts.tags import tag_section, task_tags
from ..agents.turns import TurnTimeout
from ..checks import conventions, gate_cli
from ..infra.files import write_text_atomic
from ..operators.coverage import (
    QueueItem,
    VocabularyError,
    entry_paths,
    load_vocabulary,
    queue_item_from_ref,
    shared_entry_paths,
)
from ..spec import deployments as deployments_mod
from . import certify as certify_mod
from . import contract as contract_mod
from . import inbox as inbox_mod
from .integration import (
    REPAIR_TURNS,
    Integration,
    TaskNotStarted,
    rebase_block,
    repair_block,
    worktree_block,
)

TASKS_FILENAME = "tasks.yaml"
SURVEY_ID = "survey"
#: The role each task kind's turn runs as.
ROLE_BY_KIND = {
    "survey": "surveyor",
    "kernel": "kernel_writer",
    "assemble": "assembler",
    "feature": "assembler",
    "fix": "assembler",
}
#: The kinds the planner may add with ``REPAIR``.
REPAIR_KINDS = ("kernel", "fix")
#: The tag sections a kind's dispatch carries (``agents/prompts/tags.py``).
TAGS_BY_KIND = {
    "survey": ("survey",),
    "assemble": ("assemble",),
    "feature": ("feature",),
    "fix": ("assemble", "fix"),
}
PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"
#: Decisions that mean the worker did what the task asked.
FINISHED = decision_table.finishing()
#: How many worker runs one task gets, its first included: the CLI's --max-attempts default.
DEFAULT_MAX_ATTEMPTS = 3
#: How many certification rounds a run gets, the first included: --max-rounds.
DEFAULT_MAX_ROUNDS = 4
#: How many times in a run the planner may add tasks, and how many each time.
MAX_REPAIRS = 12
MAX_REPAIR_TASKS = planner_prompt.MAX_REPAIR_TASKS
#: How many times the planner is asked for one answer before the default stands.
PLANNER_ASKS = 2
#: How many finished tasks a dispatch lists, the newest; the file has every one.
HISTORY_TASKS = 30


@dataclass
class Task:
    """One task of the pipeline: what it asks, and where it stands."""

    id: str
    kind: str
    #: The deployment it works for, by name; the first is ``core``.
    unit: str = ""
    #: A kernel task's vocabulary queue row, or ``<mode>:<entry>[@<site>]``.
    ref: str = ""
    goal: str = ""
    status: str = PENDING
    attempts: int = 0
    #: The decision its last turn wrote; empty when the turn wrote none.
    decision: str = ""
    #: Why it failed; its next attempt's worker is told this.
    note: str = ""
    #: The first paragraph of the worker's own summary.
    summary: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def open(self) -> bool:
        return self.status in (PENDING, RUNNING)


@dataclass
class TurnResult:
    """What one task's turn came to: a decision the role may give, or why there is none."""

    decision: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    #: Why the task did not finish though no worker said so: no decision file, a turn that
    #: raised, a vocabulary that does not validate, a branch that did not integrate.
    error: str = ""

    @property
    def finished(self) -> bool:
        return self.decision in FINISHED and not self.error


@dataclass
class Plan:
    """The planner's answer, validated."""

    decision: str
    summary: str = ""
    note: str = ""
    tasks: list[Task] = field(default_factory=list)


class Pipeline:
    """Drive the tasks in order, ask the planner on a failure, certify the final commit."""

    def __init__(self, workflow, state, log=None) -> None:
        self.w = workflow
        self.state = state
        self.log = log
        self.workspace: Path = workflow.workspace
        self.path = self.workspace / TASKS_FILENAME
        #: The run's units of work: its deployments by name, the first as ``core``.
        self.units = unit_names(workflow.task_data)
        self.max_attempts = max(1, int(workflow.max_attempts))
        self.max_rounds = max(1, int(workflow.max_rounds))
        self._lock = threading.Lock()
        #: Whether the units' tasks were added (after the survey), how many times the
        #: planner added tasks, and how many turns it had: all kept in ``tasks.yaml``.
        self.expanded = False
        self.repairs = 0
        self.planner_turns = 0
        self.tasks: list[Task] = []
        self.certification: certify_mod.Certification | None = None
        #: Why the run stopped ``escalated``.
        self.escalation = ""
        self._load()
        self.integration = self._make_integration()

    # ----------------------------------------------------------------- the task list
    def _load(self) -> None:
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        self.tasks = [Task.from_dict(t) for t in data.get("tasks") or () if isinstance(t, dict)]
        self.expanded = bool(data.get("expanded"))
        self.repairs = int(data.get("repairs") or 0)
        self.planner_turns = int(data.get("planner_turns") or 0)
        if self.tasks:
            return
        survey = Task(
            SURVEY_ID,
            "survey",
            goal="map the checkpoint's computation steps onto the catalog: write plan.md and "
            "vocabulary.yaml",
        )
        if not self.w.survey_first or self.w.vocabulary_path.exists():
            # A workspace that holds a vocabulary has had its survey.
            survey.status, survey.summary = DONE, "the workspace already held vocabulary.yaml"
        self.tasks = [survey]
        self._save()

    def _save(self) -> None:
        payload = {
            "expanded": self.expanded,
            "repairs": self.repairs,
            "planner_turns": self.planner_turns,
            "tasks": [asdict(t) for t in self.tasks],
        }
        write_text_atomic(
            self.path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
        )

    def by_id(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def _new_id(self, stem: str, taken: set[str]) -> str:
        """``<stem>-<n>``, the first no task and none of ``taken`` has; added to ``taken``."""
        used = {t.id for t in self.tasks} | taken
        n = 1
        while f"{stem}-{n}" in used:
            n += 1
        taken.add(f"{stem}-{n}")
        return f"{stem}-{n}"

    def _planned(self) -> list[Task]:
        """The tasks the spec and the vocabulary ask for, in the fixed order of the module docstring."""
        try:
            queue = load_vocabulary(self.w.vocabulary_path, units=self.units)
        except VocabularyError:
            queue = ()  # no vocabulary: a run told to start without a survey
        first = self.units[0] if self.units else ""
        rows = _rows_by_unit(queue, self.units)
        row_task = {item.n: f"k{item.n}-{_entry_name(item.target)}" for item in queue}
        planned: list[Task] = []
        for unit in self.units:
            for item in rows[unit]:
                planned.append(
                    Task(
                        row_task[item.n],
                        "kernel",
                        unit=unit,
                        ref=str(item.n),
                        goal=f"{item.mode} `{item.target}` as vocabulary queue row {item.n} says",
                    )
                )
            kind = "assemble" if unit == first else "feature"
            planned.append(Task(unit, kind, unit=unit, goal=_unit_goal(kind, unit)))
        return planned

    def _expand(self) -> None:
        """Once the survey is done: add the planned tasks the list lacks, and drop what the spec dropped.

        The first time, that is every deployment's tasks. On a resume it is what an
        edited spec added; a deployment the spec no longer has, and whose task has not
        run, is skipped.
        """
        survey = self.by_id(SURVEY_ID)
        if survey is None or survey.status != DONE:
            return
        planned = self._planned()
        have = {t.id for t in self.tasks}
        added = [t for t in planned if t.id not in have]
        wanted = {t.id for t in planned}
        dropped = [
            t
            for t in self.tasks
            if t.open and t.kind in ("assemble", "feature") and t.id not in wanted
        ]
        if not added and not dropped and self.expanded:
            return
        with self._lock:
            self.tasks += added
            for task in dropped:
                task.status, task.note = SKIPPED, "the spec no longer asks for it"
            self.expanded = True
            self._save()
        if added:
            self.w.journal.event(
                "pipeline", f"{len(added)} task(s) added: " + ", ".join(t.id for t in added)
            )
        if dropped:
            self.w.journal.event(
                "pipeline", "no longer in the spec, skipped: " + ", ".join(t.id for t in dropped)
            )

    def _next_task(self) -> Task | None:
        """The first open task: tasks run one at a time, in the pipeline's order."""
        return next((t for t in self.tasks if t.open), None)

    # ----------------------------------------------------------------------- the loop
    def run(self) -> str:
        """Run to a verdict: ``delivered``, ``escalated`` or ``certification-failed``."""
        print_rule("[bold magenta]execution[/bold magenta]", self.log)
        while True:
            verdict = self._step()
            if verdict:
                return verdict

    def _step(self) -> str:
        messages = inbox_mod.take(self.workspace)
        if messages:
            self.w.journal.event("operator", "\n".join(messages))
            verdict = self._apply(self._ask_planner(self._message_situation(messages)), None)
            if verdict:
                return verdict
        self._expand()
        task = self._next_task()
        if task is None:
            return self._certify()
        self._run_task(task)
        if task.status == FAILED:
            return self._after_failure(task) or ""
        return ""

    def _certify(self) -> str:
        """One certification round on the final commit; the planner is asked when it fails."""
        if not self.state.round_open:
            if self.state.round >= self.max_rounds:
                return self._rounds_spent()
            self.state.round += 1
            self.state.round_open = True
            self.w._checkpoint(self.state)
        certification = certify_mod.certify(
            self.w, self.state.round, first=self.state.failed_checks, log=self.log
        )
        self.certification = certification
        self.state.round_open = False
        self.state.failed_checks = certification.failed_keys
        self.w._checkpoint(self.state)
        if certification.passed:
            return "delivered"
        failures = certification.failures()
        print_message(
            f"[bold yellow]✘ certification round {self.state.round}: "
            f"{escape_markup('; '.join(failures)[:1500])}[/bold yellow]",
            self.log,
        )
        if self.state.round >= self.max_rounds:
            return self._rounds_spent()
        return self._apply(self._ask_planner(self._certification_situation(certification)), None)

    def _rounds_spent(self) -> str:
        self.w.journal.event(
            "certification",
            f"{self.state.round} round(s) ran and the final commit is not certified",
        )
        return "certification-failed"

    # ------------------------------------------------------------------------ a task
    def _run_task(self, task: Task) -> None:
        """One attempt of ``task``: its turn, then its integration; the task ends done or failed."""
        resumed = task.status == RUNNING and task.attempts > 0
        with self._lock:
            if not resumed:
                task.attempts += 1
            task.status = RUNNING
            self._save()
        try:
            result = self._work(task, resumed)
        except TurnTimeout as exc:
            result = TurnResult(error=f"its turn ran out of time: {exc}")
        except Exception as exc:  # the turn's own failure, not the pipeline's
            result = TurnResult(error=f"its turn raised {type(exc).__name__}: {exc}")
        with self._lock:
            task.decision = result.decision
            task.summary = first_paragraph(result.data.get("summary"))
            if result.finished:
                task.status, task.note = DONE, ""
            else:
                task.status, task.note = FAILED, _failure_note(result)
            self._save()
        said = f"{task.id} (attempt {task.attempts}): {task.status}"
        said += f" {task.decision}" if task.decision else ""
        self.w.journal.event("task", said + (f": {task.note}" if task.note else ""))
        colour = "green" if task.status == DONE else "yellow"
        print_message(f"[bold {colour}]{escape_markup(said)}[/bold {colour}]", self.log)

    def _work(self, task: Task, resumed: bool) -> TurnResult:
        role = ROLE_BY_KIND[task.kind]
        tree, block = None, ""
        if self.integration is not None and task.kind != "survey":
            try:
                tree = self.integration.open(task.id)
            except Exception as exc:  # a full filesystem, git: no worker ran
                raise TaskNotStarted(f"{type(exc).__name__}: {exc}") from exc
            block = worktree_block(tree, self.integration.repo)
        path = decision_path(self.workspace, task.id, task.attempts)
        data = read_decision(path, role) if resumed else None
        if data is not None:
            # The process stopped after the worker answered: its answer stands.
            self.w.journal.event("task", f"{task.id}: resuming with the decision at {path}")
        else:
            path.unlink(missing_ok=True)
            self._dispatch(task, role, self._prompt(task, block), path)
            data = read_decision(path, role)
            problem = decision_problem(path, role) if data is None else ""
            if problem:
                # The work may be done though the file is wrong: one short turn to say so.
                self.w.journal.event("task", f"{task.id}: decision file not valid: {problem}")
                note = decision_repair_block(problem)
                self._dispatch(task, role, self._prompt(task, f"{block}\n\n{note}"), path)
                data = read_decision(path, role)
        if data is None:
            return TurnResult(
                error=f"its turn wrote no valid decision file at {path} "
                f"(one of {', '.join(decision_table.names(role))})"
            )
        result = TurnResult(str(data["decision"]), data)
        if result.decision not in FINISHED:
            return result
        if task.kind == "survey":
            try:
                load_vocabulary(self.w.vocabulary_path, units=self.units)
            except VocabularyError as exc:
                result.error = f"vocabulary.yaml does not validate: {exc}"
            return result
        if tree is not None:
            result.error = self._integrate(task, role, tree, block)
            if not result.error:
                self._scan(task, data)
        return result

    def _scan(self, task: Task, data: Mapping[str, Any]) -> None:
        """Read the checkout a task just landed in for the conventions: reported, never a gate.

        What it finds is kept for the report and the final `fix` task (`conventions.json`);
        no finding fails the task or sends it back to its worker.
        """
        target_rel = str(getattr(self.w._preflight, "target_rel", "") or "")
        repo = self.w.task_data.get("trtllm_repo_path")
        if not (target_rel and repo):
            return
        item = self._queue_item(task) if task.kind == "kernel" else None
        paths = entry_paths(item, target_rel, repo) if item is not None else None
        try:
            found = conventions.scan(
                repo,
                target_rel,
                str(self.w.task_data.get("arch") or ""),
                landing=paths[0] if paths else "",
                named=str(data.get("entry") or "").strip(),
            )
            new = conventions.record(self.workspace, task.id, found)
        except Exception as exc:  # a scan that cannot read the tree says so, and holds nothing
            new = [f"the scan could not run: {type(exc).__name__}: {exc}"]
        if new:
            said = "; ".join(new)
            self.w.journal.event("conventions", f"{task.id}: {len(new)} finding(s): {said}")

    def _integrate(self, task: Task, role: str, tree: Path, block: str) -> str:
        """Audit and integrate ``task``'s branch; why it did not go in, or ``""``.

        A branch that does not integrate as it stands (commits that break the
        rules, uncommitted work, a rebase conflict) goes back to its worker for
        one short turn, then is tried once more.
        """
        for turn in range(2, REPAIR_TURNS + 3):
            outcome = self.integration.integrate(task.id)
            self.w.journal.event("integration", f"{task.id}: {outcome.detail}")
            if outcome.merged:
                self.integration.close(task.id)
                return ""
            if not (outcome.base or outcome.repairable) or turn > REPAIR_TURNS + 1:
                return f"its branch did not integrate: {outcome.detail}"
            note = rebase_block(outcome, tree) if outcome.base else repair_block(outcome)
            path = decision_path(self.workspace, task.id, task.attempts, turn)
            path.unlink(missing_ok=True)
            self._dispatch(task, role, self._prompt(task, f"{block}\n\n{note}"), path)
            data = read_decision(path, role)
            if data is None or data["decision"] not in FINISHED:
                return (
                    f"its branch did not integrate ({outcome.detail}), and the turn asked to "
                    "put it right did not finish"
                )
        return "its branch did not integrate"

    def _dispatch(self, task: Task, role: str, prompt: str, path: Path) -> None:
        """One turn of ``role`` for ``task``, told to write its decision at ``path``."""
        context = self.w._context_block()
        if context:
            prompt = f"{prompt}\n\n{context}"
        prompt = f"{prompt}\n\n{decision_file_instruction(path)}"
        report = self.w.journal.record_dispatch(role, task.id, prompt)
        text = self.w._run_turn(role, task.id, prompt, decision_file=path)
        self.w.journal.record_result(report, text)

    def _after_failure(self, task: Task) -> str:
        """What happens to a failed task: another attempt, or the planner's decision."""
        judged = bool(task.decision) and task.decision not in FINISHED
        if not judged and task.attempts < self.max_attempts:
            # No worker judgment to weigh: the turn left no decision, or what it left
            # did not validate or integrate. Its next attempt is told why.
            task.status = PENDING
            self._save()
            self.w.journal.event(
                "task", f"{task.id}: runs again ({task.attempts} of {self.max_attempts} used)"
            )
            return ""
        return self._apply(self._ask_planner(self._failure_situation(task), task), task)

    # --------------------------------------------------------------------- the planner
    def _allowed(self, task: Task | None) -> list[str]:
        """The decisions the planner may give now."""
        allowed = []
        if task is None or task.attempts < self.max_attempts:
            allowed.append(CONTINUE)
            if self.repairs < MAX_REPAIRS:
                allowed.append(REPAIR)
        if task is not None and task.kind == "kernel":
            allowed.append(SKIP)
        return [*allowed, ESCALATE]

    def _ask_planner(self, situation: str, task: Task | None = None) -> Plan:
        """One planner decision about ``situation``; the default when its answer cannot be read."""
        allowed = self._allowed(task)
        problem = ""
        for _ in range(PLANNER_ASKS):
            with self._lock:
                self.planner_turns += 1
                turn = self.planner_turns
                self._save()
            path = decision_path(self.workspace, "planner", turn)
            path.unlink(missing_ok=True)
            prompt = self._planner_prompt(situation, allowed, problem)
            prompt = f"{prompt}\n\n{decision_file_instruction(path)}"
            report = self.w.journal.record_dispatch("planner", f"plan-{turn}", prompt)
            try:
                text = self.w._run_turn("planner", f"plan-{turn}", prompt, decision_file=path)
            except Exception as exc:  # a planner that cannot answer never stops the run by itself
                problem = f"the turn raised {type(exc).__name__}: {exc}"
                self.w.journal.event("planner", problem)
                continue
            self.w.journal.record_result(report, text)
            plan, problem = self._read_plan(path, allowed)
            if plan is not None:
                self.w.journal.event("planner", f"{plan.decision}: {plan.summary}")
                return plan
            self.w.journal.event("planner", f"answer refused: {problem}")
        decision = CONTINUE if CONTINUE in allowed else ESCALATE
        why = f"the planner gave no usable answer ({problem}); the default is {decision}"
        self.w.journal.event("planner", why)
        return Plan(decision, why)

    def _read_plan(self, path: Path, allowed: list[str]) -> tuple[Plan | None, str]:
        """The planner's answer at ``path``, validated; else ``None`` and what is wrong with it."""
        data = read_decision(path, "planner")
        if data is None:
            return None, f"no decision file at {path} with one of {', '.join(allowed)}"
        decision = data["decision"]
        if decision not in allowed:
            return None, f"{decision} is not possible here; choose one of {', '.join(allowed)}"
        plan = Plan(decision, first_paragraph(data.get("summary")), str(data.get("note") or ""))
        if decision != REPAIR:
            return plan, ""
        raw = data.get("tasks")
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_REPAIR_TASKS:
            return None, f"REPAIR needs `tasks`: a list of 1 to {MAX_REPAIR_TASKS} tasks"
        targets = self.units
        taken: set[str] = set()
        for index, row in enumerate(raw):
            if not isinstance(row, dict) or row.get("kind") not in REPAIR_KINDS:
                return None, f"tasks[{index}].kind must be one of {', '.join(REPAIR_KINDS)}"
            unit = str(row.get("unit") or (self.units[0] if self.units else ""))
            if unit not in targets:
                return None, f"tasks[{index}].unit {unit!r} is not one of {', '.join(targets)}"
            goal = clip(str(row.get("goal") or ""), 2000)
            ref = str(row.get("ref") or "").strip()
            if row["kind"] == "kernel":
                task_id = self._new_id("kx", taken)
                if queue_item_from_ref(ref, task_id, goal) is None:
                    return None, (
                        f"tasks[{index}].ref {ref!r} must be `onboard:<category>/<name>` or "
                        "`verify:<category>/<name>`"
                    )
            else:
                task_id = self._new_id("fix", taken)
                if not goal:
                    return None, f"tasks[{index}].goal must say what to change and the evidence"
            plan.tasks.append(Task(task_id, row["kind"], unit=unit, ref=ref, goal=goal))
        return plan, ""

    def _apply(self, plan: Plan, task: Task | None) -> str:
        """Carry out the planner's decision; the run's verdict when it ends it, else ``""``."""
        if plan.decision == ESCALATE:
            self.escalation = plan.summary
            return "escalated"
        with self._lock:
            if plan.decision == SKIP and task is not None:
                task.status = SKIPPED
                task.note = f"skipped by the planner: {plan.summary}"
            elif task is not None:
                task.status = PENDING
                if plan.note:
                    task.note = f"{task.note}\n\nThe planner says: {plan.note}".strip()
            if plan.decision == REPAIR:
                self.repairs += 1
                # Before the task it repairs; with none, before the next task to run.
                before = task or next((t for t in self.tasks if t.open), None)
                at = self.tasks.index(before) if before is not None else len(self.tasks)
                self.tasks[at:at] = plan.tasks
            self._save()
        return ""

    def _planner_prompt(self, situation: str, allowed: list[str], problem: str) -> str:
        lines = [
            f"Workspace: {self.workspace}",
            f"Spec: `{self.w.task_path}`",
            f"Tasks and where each stands: `{self.path}`",
            f"Contract (every check the final commit must pass): "
            f"`{self.workspace / contract_mod.CONTRACT_FILENAME}`",
            "",
            "## What happened",
            "",
            situation,
            "",
            "## Tasks",
            "",
            self.table(),
            "",
            "## Your answer",
            "",
            f"Answer with exactly one of: {', '.join(f'`{d}`' for d in allowed)}.",
        ]
        if REPAIR in allowed:
            lines += [
                f"A `REPAIR` lists at most {MAX_REPAIR_TASKS} tasks under `tasks`; "
                f"a task's `unit` is one of: {', '.join(self.units)}.",
            ]
        if problem:
            lines += ["", "## Your last answer was refused", "", problem]
        lines += ["", DECISION_REMINDER]
        return "\n".join(lines)

    def _failure_situation(self, task: Task) -> str:
        role = ROLE_BY_KIND[task.kind]
        path = decision_path(self.workspace, task.id, task.attempts)
        left = self.max_attempts - task.attempts
        return (
            f"Task `{task.id}` ({task.kind}, unit `{task.unit or '-'}`, run by the {role}) "
            f"failed on attempt {task.attempts} of {self.max_attempts}"
            + (f" with the decision `{task.decision}`" if task.decision else " with no decision")
            + f".\n\n{task.note or 'It left no reason.'}\n\n"
            f"Its decision file: `{path}`. Its reports: `{self.w.journal.reports_dir}`.\n\n"
            + (
                f"It has {left} attempt(s) left: `CONTINUE` runs it again, with your `note` "
                "telling its worker what to do differently."
                if left > 0
                else "It has no attempt left: it cannot run again."
            )
        )

    def _certification_situation(self, certification: certify_mod.Certification) -> str:
        failures = "\n".join(f"- {clip(line, 1500)}" for line in certification.failures())
        return (
            f"Every task ended, and certification round {certification.round} of "
            f"{self.max_rounds} did not certify the final commit "
            f"`{certification.commit[:12]}`:\n\n{failures}\n\n"
            f"The round's result: `{self.workspace / certify_mod.CERTIFICATION_FILENAME}`; the "
            f"checks' files: `{certify_mod.round_dir(self.workspace, certification.round)}`.\n\n"
            "`REPAIR` with one `fix` task per cause, each naming the unit it belongs to, the "
            "failing check and its evidence; the certification then runs again on the new "
            "commit, the failed checks first. `CONTINUE` certifies again with no change: "
            "right only when the failure was the host's, not the target's."
        )

    def _message_situation(self, messages: list[str]) -> str:
        return (
            "The operator left a message for the run. Act on it: `REPAIR` adds the work it "
            "asks for, `CONTINUE` when the pipeline already does what it asks, `ESCALATE` "
            "when it cannot be done. Say in your `summary` what you did about it. A message "
            "does not change a contract check: where it asks for one to change, name the "
            "`task.yaml` key that would change it.\n\n" + "\n".join(f"- {m}" for m in messages)
        )

    def table(self) -> str:
        """Every task and where it stands, as a markdown table."""
        lines = [
            "| task | kind | unit | status | attempts | decision | summary |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in self.tasks:
            said = clip((t.summary or t.note).replace("\n", " ").replace("|", "/"), 160)
            lines.append(
                f"| `{t.id}` | {t.kind} | {t.unit or '-'} | {t.status} | {t.attempts} | "
                f"{t.decision or '-'} | {said} |"
            )
        return "\n".join(lines)

    # ---------------------------------------------------------------------- the prompts
    def _prompt(self, task: Task, block: str) -> str:
        """``task``'s dispatch: its goal, why it runs again, its section, then its worktree."""
        note = ""
        if task.note:
            note = (
                f"Attempt {task.attempts} of {self.max_attempts} of task `{task.id}`. The "
                f"previous attempt did not finish:\n\n{task.note}"
            )
        if task.kind == "kernel":
            body = self._kernel_prompt(task, note)
            parts = (body, self._history_block(task), block)
        else:
            tags = task_tags(TAGS_BY_KIND[task.kind][0], TAGS_BY_KIND[task.kind][1:])
            header = f"## Your task: `{task.id}` ({task.kind})\n\n{task.goal}"
            if note:
                header += f"\n\n## Why this task runs again\n\n{note}"
            parts = (
                header,
                self._history_block(task),
                tag_section(tags),
                self._survey_prompt() if task.kind == "survey" else self._unit_prompt(task),
                block,
            )
        return "\n\n".join(p.strip() for p in parts if p and p.strip())

    def _history_block(self, task: Task) -> str:
        """What the tasks before ``task`` came to, by reference: the newest, one line each."""
        ended = [t for t in self.tasks if t is not task and t.status in (DONE, FAILED, SKIPPED)]
        if not ended:
            return ""
        lines = [
            "## What the run did before this task",
            "",
            "Each line is a task that ended, with its worker's own summary; its decision file "
            f"is `{self.workspace / decision_table.DECISIONS_DIRNAME}/<task>-<attempt>.yaml`. "
            "A done task's commits are in the checkout. Build on them rather than starting over.",
            "",
        ]
        left = len(ended) - HISTORY_TASKS
        if left > 0:
            lines.append(f"- ({left} earlier task(s): `{self.path}`)")
        for t in ended[-HISTORY_TASKS:]:
            said = clip((t.summary or t.note).replace("\n", " "), 300)
            lines.append(f"- `{t.id}` ({t.kind}): {t.status} {t.decision}. {said}".rstrip())
        return "\n".join(lines)

    def _cli(self) -> str:
        """The workflow's CLI as an agent calls it: by path, when preflight installed it."""
        script = self.workspace / "bin" / gate_cli.CLI_NAME
        return str(script) if script.is_file() else gate_cli.CLI_NAME

    def _checks_block(self, unit: str) -> str:
        """The contract's checks for ``unit``'s deployment and the command that runs one."""
        name = deployments_mod.named(self.w.task_data, unit).name
        owed = self.w._contract.of(name) if self.w._contract else ()
        if not owed:
            return ""
        command = gate_cli.invocation(
            self.w.task_path, self.workspace / "gates" / unit, name, self._cli()
        )
        listed = ", ".join(f"`{c.key}`" for c in owed)
        return (
            f"## The checks `{unit}` owes\n\n{listed}\n\n"
            "**Run them yourself, on your worktree, before you finish**, and list each with "
            "its result under `checks` in your decision file:\n\n"
            f"    {command}\n\n"
            "Nothing else checks your work while the run builds: the run trusts your decision. "
            "Every check of the contract then runs once on the final commit, and one that "
            "fails there comes back as a `fix` task, after every later task was built on your "
            "work. Start with the cheapest (the boot); a benchmark you cannot afford in this "
            "turn you say so under `checks`, with what you ran instead."
        )

    def _unit_prompt(self, task: Task) -> str:
        """An assembler turn: on core, on a further deployment, or on a repair."""
        w = self.w
        unit = task.unit or (self.units[0] if self.units else "core")
        head = (
            f"Workspace: {self.workspace}\n"
            f"Repository: {w.task_data['trtllm_repo_path']}\n"
            f"Target: {w._preflight.target_rel}\n"
        )
        read = (
            f"Read `{w.task_path}` (the spec), `{w.plan_path}` (the plan) and "
            f"`{w.vocabulary_path}` (the computation-step list you derive against)."
        )
        deployment = deployments_mod.named(w.task_data, unit)
        body = f"{head}Unit: **{unit}**\n\n{deployments_mod.describe((deployment,))}\n\n{read}"
        if task.kind == "fix":
            body += (
                " Repair what your task names, and nothing else; every check that passed "
                "must keep passing."
            )
        elif deployment.core:
            body += " Write the unit."
        else:
            body += (
                " The target exists. Make it serve this deployment as well, without changing "
                "what it does in the ones before it: every check that passed must keep passing."
            )
        return f"{body}\n\n{self._checks_block(unit)}\n\n{DECISION_REMINDER}"

    def _survey_prompt(self) -> str:
        w = self.w
        return (
            f"Workspace: {self.workspace}\n\n"
            f"Read `{w.task_path}` for the run's spec: the checkpoint, the "
            f"architecture, the deployments it brings up and the accuracy anchor.\n\n"
            f"{w._surface_block}\n"
            f"{capture_lists_block(w.task_data)}\n"
            f"Write `{w.plan_path}` and `{w.vocabulary_path}` as the `survey` "
            f"section specifies. The orchestrator parses `vocabulary.yaml` and refuses "
            f"a malformed one, so follow the schema exactly.\n\n" + DECISION_REMINDER
        )

    def _queue_item(self, task: Task) -> QueueItem | None:
        """The kernel writer dispatch a kernel task names: a queue row, or an entry by ``ref``."""
        if task.ref.isdigit():
            try:
                queue = load_vocabulary(self.w.vocabulary_path, units=self.units)
            except VocabularyError:
                return None
            return next((q for q in queue if str(q.n) == task.ref), None)
        return queue_item_from_ref(task.ref, task.id, task.goal)

    def _kernel_prompt(self, task: Task, note: str) -> str:
        w = self.w
        item = self._queue_item(task)
        if item is None:
            raise TaskNotStarted(
                f"kernel task {task.id} names no queue row or `<mode>:<entry>` ({task.ref!r})"
            )
        target_rel = str(getattr(w._preflight, "target_rel", "") or "")
        repo = w.task_data.get("trtllm_repo_path") or None
        # A shared entry lands as its local variant in the target's own ops/.
        landing = str(item.variant) if item.variant is not None and target_rel else item.target
        try:
            rows = (yaml.safe_load(w.vocabulary_path.read_text(encoding="utf-8")) or {}).get(
                "queue"
            ) or ()
        except (OSError, yaml.YAMLError, AttributeError):
            rows = ()
        row = next((r for r in rows if isinstance(r, dict) and str(r.get("n")) == str(item.n)), {})
        return kernel_prompt(
            KernelContext(
                item=item,
                workspace=str(self.workspace),
                repo=str(w.task_data["trtllm_repo_path"]),
                arch=str(w.task_data["arch"]),
                target_rel=target_rel,
                vocabulary_path=str(w.vocabulary_path),
                note=note,
                goal=(task.goal or "").strip(),
                task_id=task.id,
                task_kind="kernel",
                tags=(),
                tier=row.get("tier"),
                landing=landing,
                paths=entry_paths(item, target_rel, repo),
                shared=tuple(shared_entry_paths(item.target)) if landing != item.target else (),
            )
        )

    # ------------------------------------------------------------------------ worktrees
    def _make_integration(self) -> Integration | None:
        """Per-task worktrees on a git checkout; on anything else tasks share the tree."""
        repo = Path(str(self.w.task_data.get("trtllm_repo_path") or ""))
        if not (repo / ".git").exists():
            self.w.journal.event("integration", f"tasks share the checkout {repo}")
            return None
        policy = self.w._commit_policy()
        env = {}
        if policy.identity is not None:
            # The rebase's committer; each commit keeps its author.
            env = {k: v for k, v in policy.identity.env().items() if k.startswith("GIT_COMMITTER")}
        return Integration(
            repo,
            self.workspace,
            f"mv2/{self.state.slug or 'run'}",
            env,
            log=lambda message: self.w.journal.event("integration", message),
            policy=policy,
        )


def _rows_by_unit(queue, units: list[str]) -> dict[str, list[QueueItem]]:
    """The queue rows each unit's kernel tasks come from, each after the rows it depends on.

    A row belongs to the first unit that needs it: its own, or an earlier unit's when a
    row of that unit depends on it. Within a unit, rows keep their numbers' order but for
    a row that depends on a later one, which follows it; rows that depend on each other
    in a circle keep their numbers' order.
    """
    index = {unit: i for i, unit in enumerate(units)}
    by_n = {item.n: item for item in queue}
    place = {n: index.get(item.unit, 0) for n, item in by_n.items()}
    changed = True
    while changed:
        changed = False
        for n, item in by_n.items():
            for dep in item.depends_on:
                if dep in place and place[dep] > place[n]:
                    place[dep], changed = place[n], True
    grouped: dict[str, list[QueueItem]] = {unit: [] for unit in units}
    for i, unit in enumerate(units):
        left = sorted(n for n in by_n if place[n] == i)
        placed: list[int] = []
        while left:
            ready = [n for n in left if not (set(by_n[n].depends_on) & set(left) - {n})]
            take = ready[0] if ready else left[0]
            placed.append(take)
            left.remove(take)
        grouped[unit] = [by_n[n] for n in placed]
    return grouped


def _entry_name(target: str) -> str:
    return target.strip("/").split("/")[-1]


def _unit_goal(kind: str, unit: str) -> str:
    if kind == "assemble":
        return f"assemble the target's `{unit}`: modeling, weights and routing"
    return f"make the target serve the deployment `{unit}` as well"


def unit_names(task_data) -> list[str]:
    """The run's units of work: its deployments by name, in order, the first as ``core``."""
    return ["core", *(d.name for d in deployments_mod.resolve(task_data)[1:])]


def _failure_note(result: TurnResult) -> str:
    """Why a task failed, for the planner and for its next attempt's worker."""
    if result.error:
        return result.error
    raw = result.data
    said = [f"{key}: {raw[key]}" for key in ("blocked_reason", "missing_entry") if raw.get(key)]
    defect = raw.get("engine_defect")
    if isinstance(defect, Mapping):
        said.append(
            "engine_defect: "
            + "; ".join(f"{k} {' '.join(str(v).split())}" for k, v in defect.items() if v)
        )
    if raw.get("summary"):
        said.append(clip(str(raw["summary"]), 800))
    return f"it ended {result.decision}" + (": " + "; ".join(said) if said else "")


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_ROUNDS",
    "FINISHED",
    "Pipeline",
    "ROLE_BY_KIND",
    "unit_names",
    "SURVEY_ID",
    "TASKS_FILENAME",
    "Task",
]
