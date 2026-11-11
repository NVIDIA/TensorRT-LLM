# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline: the run's tasks in one order, fixed here, with no agent deciding it.

    the config task        only when preflight saw that the tree cannot load the config
    survey
    core                   after the kernel rows it needs; then the onboard check, by code
    each later deployment  in the spec's order, after the kernel rows it needs

Tasks run one at a time, each one agent turn in its own worktree. A worker that says it
is ready has its branch audited and landed (``run/integration.py``); `core` lands only
when the onboard check passes on its commit, the config task only when preflight's probe
loads the config there. Each landing is read for the conventions: findings are kept for
the report and hold nothing.

The bounds are numbers. A task gets :data:`DEFAULT_MAX_ATTEMPTS` attempts; a failed
attempt runs again with its result line as the next one's note. An attempt a stopped
process cut restarts free :data:`FREE_RESTARTS` times; the next cut spends it. A task
that spent its attempts is **not done**: nothing of it lands and the run goes on, but for
the config task, the survey and `core`, where the run ends ``not onboarded``. Every new
start reopens each task not done with fresh attempts, at most :data:`GRANTS` times.

``tasks.yaml`` in the workspace is the list and where each task stands; the run resumes
from it. A deployment preflight recorded "not attempted" gets no task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ..agents import prompts
from ..agents.decisions import (
    DECISIONS_DIRNAME,
    READY,
    clip,
    decision_path,
    decision_problem,
    first_paragraph,
    read_decision,
)
from ..checks import anchors as anchors_mod
from ..checks import conventions, gate_cli
from ..infra.files import write_text_atomic
from ..operators.coverage import (
    MODE_ONBOARD,
    MODELING_V2_PACKAGE,
    QueueItem,
    VocabularyError,
    entry_paths,
    load_vocabulary,
    model_ops_tests_dir,
    rows_by_unit,
    shared_entry_paths,
)
from ..spec import deployments as deployments_mod
from . import report as report_mod
from .contract import library_digest
from .integration import REPAIR_TURNS, Integration, behind
from .startup import preflight as preflight_mod
from .state import StateError

TASKS_FILENAME = "tasks.yaml"
#: What the operator's `feedback` command appends to; every dispatch says to read it.
NOTES_FILENAME = "operator_notes.md"
CONFIG_ID, SURVEY_ID = "config", "survey"
#: Each task kind's role: the backend, model and effort the spec states for its turns.
ROLE_BY_KIND: dict[str, str] = {
    "config": "assembler",
    "survey": "surveyor",
    "kernel": "kernel_writer",
    "core": "assembler",
    "deployment": "assembler",
    "fix": "assembler",
}
PENDING, RUNNING, DONE, NOT_DONE, SKIPPED = "pending", "running", "done", "not done", "skipped"
#: How many worker runs one task gets, its first included: the CLI's --max-attempts default.
DEFAULT_MAX_ATTEMPTS = 3
#: How many times an attempt a stopped process cut restarts without being spent.
FREE_RESTARTS = 2
#: The kinds whose task, not done, ends the run ``not onboarded``; and how many times a new
#: start gives a task not done fresh attempts: theirs, then any other task's.
GATING = ("config", "survey", "core")
#: Why a task the spec or preflight dropped is skipped; it is open again once they ask for it.
_DROPPED = "the spec or preflight no longer asks for it"
GRANTS = (4, 2)
#: How many ended tasks a dispatch lists, the newest; the file has every one.
HISTORY_TASKS = 30


@dataclass
class Task:
    """One task of the pipeline: what it asks, and where it stands."""

    id: str
    kind: str
    #: The deployment it works for, by unit name (``core`` for the first), and what it
    #: refers to: a kernel task's vocabulary queue row, a deployment task's deployment.
    unit: str = ""
    ref: str = ""
    goal: str = ""
    status: str = PENDING
    #: Worker runs spent, restarts of the current one after a cut, fresh sets granted.
    attempts: int = 0
    cuts: int = 0
    grants: int = 0
    #: The decision its last turn wrote, why it is not done (its next attempt's note), and
    #: the first paragraph of the worker's own summary.
    decision: str = ""
    note: str = ""
    summary: str = ""
    #: The checkout's head once it landed, and the content hash of the native libraries its
    #: tree held when it started and when it ended.
    commit: str = ""
    lib_start: str = ""
    lib_end: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TurnResult:
    """What one attempt came to: the receipt's decision, or why there is none."""

    decision: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    #: Why the task did not finish though no worker said so: no decision file, a turn that
    #: raised, a vocabulary that does not validate, a branch or a check that did not hold.
    error: str = ""

    @property
    def finished(self) -> bool:
        return self.decision == READY and not self.error


class Pipeline:
    """Drive the tasks in order; see the module docstring."""

    DONE, NOT_DONE = DONE, NOT_DONE

    def __init__(self, workflow, state, log=None) -> None:
        self.w = workflow
        self.state = state
        self.log = log
        self.workspace: Path = workflow.workspace
        self.path = self.workspace / TASKS_FILENAME
        #: The run's units of work: its deployments by name, the first as ``core``.
        self.units = unit_names(workflow.task_data)
        self.max_attempts = max(1, int(workflow.max_attempts))
        #: The gating task the run stopped at, and the final pass's lines (``run/report.py``).
        self.stopped: Task | None = None
        self.final: list[Any] = []
        data: Any = {}
        if self.path.exists():
            # Only an absent list is begun anew: one that cannot be read holds the attempts
            # and grants every task has spent.
            try:
                data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
                held = isinstance(data, dict) and isinstance(data.get("tasks"), list)
                why = "" if held else "it is not a mapping with a `tasks` list"
            except (OSError, yaml.YAMLError) as exc:
                why = f"{type(exc).__name__}: {exc}".splitlines()[0]
            if why:
                raise StateError(
                    f"{self.path} does not hold the task list ({why}): the run does not start "
                    f"over a list it cannot read; restore it, or start over with --clean"
                )
        self.tasks = [Task.from_dict(t) for t in data.get("tasks") or () if isinstance(t, dict)]
        #: Counts the starts that followed a finished run: each may have one `fix` task.
        self.entry = int(data.get("entry") or 0) + (1 if data.get("closed", True) else 0)
        for task in self.tasks:
            # A new start: what was not done gets fresh attempts, within its grants. A `fix`
            # task belongs to its own final pass: the next one names what fails then.
            reopen = task.status == NOT_DONE and task.kind != "fix"
            if reopen and task.grants < GRANTS[task.kind not in GATING]:
                task.status, task.attempts, task.cuts = PENDING, 0, 0
                task.grants += 1
                self.w.journal.event("task", f"{task.id}: reopened, grant {task.grants}")
        self._save()
        self.integration = self._make_integration()

    # ----------------------------------------------------------------- the task list
    def _save(self, closed: bool = False) -> None:
        payload = {"entry": self.entry, "closed": closed, "tasks": [asdict(t) for t in self.tasks]}
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
        write_text_atomic(self.path, text)

    def close(self) -> None:
        """The run reached its report: the next start is a new entry."""
        self._save(closed=True)

    def by_id(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def done_deployments(self) -> list[str]:
        """The deployments whose task is done, by name, in the spec's order."""
        found = deployments_mod.resolve(self.w.task_data)
        done = {t.unit for t in self.tasks if t.kind in ("core", "deployment") and t.status == DONE}
        return [d.name for unit, d in zip(self.units, found) if unit in done]

    def _wanted(self) -> list[Task]:
        """The tasks preflight, the spec and the vocabulary ask for, in the module's order."""
        w, pre = self.w, self.w._preflight
        wanted: list[Task] = []
        error = str(getattr(pre, "config_error", "") or "")
        if error or self.by_id(CONFIG_ID):
            goal = (
                "make the tree load the checkpoint's config: `ModelConfig.from_pretrained` "
                f"fails on it with: {error}\nThe run loads it the same way on your committed "
                "worktree before your commits land."
            )
            wanted.append(Task(CONFIG_ID, "config", unit=self.units[0], goal=goal))
        survey = self.by_id(SURVEY_ID)
        if survey is None:
            goal = "map the checkpoint's computation steps onto the catalog: write plan.md and vocabulary.yaml"
            survey = Task(SURVEY_ID, "survey", goal=goal)
            if not w.survey_first or w.vocabulary_path.exists():
                # A workspace that holds a vocabulary has had its survey.
                survey.status, survey.summary = DONE, "the workspace already held vocabulary.yaml"
        wanted.append(survey)
        if survey.status != DONE:
            return wanted
        try:
            queue = load_vocabulary(w.vocabulary_path, units=self.units)
        except VocabularyError:
            queue = ()  # no vocabulary: a run told to start without a survey
        left = {r["deployment"] for r in getattr(pre, "not_attempted", ()) if not r["check"]}
        rows = rows_by_unit(queue, self.units)
        for unit, deployment in zip(self.units, deployments_mod.resolve(w.task_data)):
            if deployment.name in left:
                continue
            for item in rows[unit]:
                name = f"k{item.n}-{item.target.strip('/').split('/')[-1]}"
                goal = f"{item.mode} `{item.target}` as vocabulary queue row {item.n} says"
                wanted.append(Task(name, "kernel", unit=unit, ref=str(item.n), goal=goal))
            goal = (
                f"make the deployment `{unit}` boot and score, without changing the ones before it"
            )
            if deployment.core:
                goal = f"assemble the target for `{deployment.name}`: modeling, weights and routing"
            kind = "core" if deployment.core else "deployment"
            wanted.append(Task(unit, kind, unit=unit, ref=deployment.name, goal=goal))
        return wanted

    def _plan(self) -> None:
        """Add the wanted tasks the list lacks, in order; skip what is no longer wanted.

        A task skipped because the spec dropped it is open again when the spec asks for
        it again, with the attempts it had spent.
        """
        wanted = self._wanted()
        have = {t.id: t for t in self.tasks}
        built, now = have.get("core"), next((t for t in wanted if t.id == "core"), None)
        if built and now and built.ref and built.ref != now.ref:
            # `core` is whatever the spec lists first: its task and the onboard record are
            # about the deployment that was first then.
            raise StateError(
                f"the first deployment is `{now.ref}` now, and this workspace's `core` was built "
                f"for `{built.ref}`: put `{built.ref}` first again, or start over with --clean"
            )
        ids = {t.id for t in wanted}
        added = [t.id for t in wanted if t.id not in have]
        for task in self.tasks:
            if task.id in ids and task.status == SKIPPED and task.note == _DROPPED:
                task.status, task.note = PENDING, ""
                added.append(task.id)
        open_ = [t for t in self.tasks if t.status in (PENDING, RUNNING) and t.kind != "fix"]
        dropped = [t for t in open_ if t.id not in ids]
        for task in dropped:
            task.status, task.note = SKIPPED, _DROPPED
        order = [have.get(t.id, t) for t in wanted] + [t for t in self.tasks if t.id not in ids]
        if added or dropped or order != self.tasks:
            self.tasks = order
            self._save()
        for what, names in (("added", added), ("skipped", [t.id for t in dropped])):
            if names:
                self.w.journal.event("pipeline", f"task(s) {what}: " + ", ".join(names))

    def _next(self) -> Task | None:
        """The first open task; none once a gating task is not done, where the run stops."""
        self._plan()
        gate = (t for t in self.tasks if t.status == NOT_DONE and t.kind in GATING)
        self.stopped = next(gate, None)
        if self.stopped is not None:
            return None
        return next((t for t in self.tasks if t.status in (PENDING, RUNNING)), None)

    # ----------------------------------------------------------------------- the loop
    def run(self) -> None:
        """Run every task to done or not done, in order; ``stopped`` says where it ended early."""
        print_rule("[bold magenta]tasks[/bold magenta]", self.log)
        while (task := self._next()) is not None:
            self._attempt(task)

    def fix(self, problems: list[str]) -> Task:
        """The final pass's one `fix` task of this entry, run to its end the first time."""
        task = self.by_id(f"fix-{self.entry}")
        if task is None:
            goal = "repair what the final pass found, every item:\n" + "\n".join(
                f"- {problem}" for problem in problems
            )
            task = Task(f"fix-{self.entry}", "fix", unit=self.units[0], goal=goal)
            self.tasks.append(task)
            self._save()
        while task.status in (PENDING, RUNNING):
            self._attempt(task)
        return task

    def _attempt(self, task: Task) -> None:
        """One attempt of ``task``: its turn, then its landing; it ends done, pending or not done."""
        resumed = task.status == RUNNING
        # A turn that had answered when the process stopped was not cut: only what follows
        # the answer (the landing, its check) starts again, as often as it takes.
        path = decision_path(self.workspace, task.id, max(task.attempts, 1))
        cut = resumed and read_decision(path) is None
        if cut and task.cuts >= FREE_RESTARTS:
            result = TurnResult(error=f"its turn was cut {task.cuts + 1} times before it ended")
        else:
            if cut:
                task.cuts += 1
            elif not resumed:
                task.attempts, task.cuts = task.attempts + 1, 0
            task.status = RUNNING
            self._save()
            try:
                result = self._work(task, resumed)
            except Exception as exc:  # the turn's own failure (a timeout too), not the pipeline's
                result = TurnResult(error=f"its turn raised {type(exc).__name__}: {exc}")
        task.decision = result.decision
        task.summary = first_paragraph(result.data.get("summary"))
        if result.finished:
            task.status, task.note = DONE, ""
        else:
            spent = task.attempts >= self.max_attempts
            task.status, task.note = (NOT_DONE if spent else PENDING), _failure_note(result)
        self._save()
        said = f"{task.id} (attempt {task.attempts} of {self.max_attempts}): {task.status}"
        said += f" {task.decision}" if task.decision else ""
        self.w.journal.event("task", said + (f": {task.note}" if task.note else ""))
        colour = "green" if task.status == DONE else "yellow"
        print_message(f"[bold {colour}]{escape_markup(said)}[/bold {colour}]", self.log)

    def _work(self, task: Task, resumed: bool) -> TurnResult:
        role = ROLE_BY_KIND[task.kind]
        repo = Path(str(self.w.task_data.get("trtllm_repo_path") or ""))
        tree = None
        if self.integration is not None and task.kind != "survey":
            tree = self.integration.open(task.id)
        if not task.lib_start:
            # The task's start, not this attempt's: what an earlier attempt rebuilt counts.
            # Saved before the dispatch: a process cut while the worker rebuilds must not
            # take the rebuilt libraries for the ones the task started from.
            task.lib_start = library_digest(tree or repo)
            self._save()
        path = decision_path(self.workspace, task.id, task.attempts)
        data = read_decision(path) if resumed else None
        facts: dict[str, Any] = {}
        if data is not None:
            # The process stopped after the worker answered: its answer stands.
            self.w.journal.event("task", f"{task.id}: resuming with the receipt at {path}")
        else:
            path.unlink(missing_ok=True)
            facts = self._facts(task, tree)
            self._dispatch(task, role, facts, path)
            data = read_decision(path)
            problem = decision_problem(path) if data is None else ""
            if problem:
                # The work may be done though the file is wrong: one short turn to say so.
                self.w.journal.event("task", f"{task.id}: receipt not valid: {problem}")
                self._dispatch(
                    task, role, {**facts, "repair": prompts.repair(problem=problem)}, path
                )
                data = read_decision(path)
        task.lib_end = library_digest(tree or repo)
        if data is None:
            said = f"its turn wrote no valid receipt at {path} (`decision: ready` or `blocked`)"
            return TurnResult(error=said)
        result = TurnResult(str(data["decision"]), data)
        if result.decision != READY:
            return result
        if task.kind == "survey":
            try:
                load_vocabulary(self.w.vocabulary_path, units=self.units)
            except VocabularyError as exc:
                result.error = f"vocabulary.yaml does not validate: {exc}"
            return result
        result.error = self._land(task, role, tree, facts or self._facts(task, tree))
        if not result.error:
            self._scan(task, data)
        return result

    def _landing_check(self, task: Task) -> Callable[[Path, str], str] | None:
        """What code holds a ready task to before it lands: ``(tree, commit) -> why not``."""
        w = self.w
        if task.kind == "core":
            return lambda tree, commit: report_mod.onboard_check(w, tree, commit)
        if task.kind != CONFIG_ID:
            return None

        def loads(tree: Path, commit: str) -> str:
            spec = {**w.task_data, "trtllm_repo_path": str(tree)}
            core = deployments_mod.resolve(spec)[0]
            seen = preflight_mod.probe(spec, core, w._preflight_runner)
            error = seen.get("error") or seen.get("tree_error") or seen.get("config_error")
            return f"the tree at {commit[:12]} does not load the config: {error}" if error else ""

        return loads

    def _land(self, task: Task, role: str, tree: Path | None, facts: Mapping[str, Any]) -> str:
        """Audit, check and land ``task``'s branch; why it did not go in, or ``""``.

        A branch that does not integrate as it stands (commits that break the rules,
        uncommitted work, a rebase conflict) goes back to its worker for one short turn.
        A check that fails sends nothing back: its line is the attempt's result.
        """
        check = self._landing_check(task)
        repo = Path(str(self.w.task_data.get("trtllm_repo_path") or ""))
        if self.integration is None or tree is None:
            # Tasks share the checkout: what is checked is what it holds.
            commit = self.w._head()
            failed = check(repo, commit) if check else ""
            if not failed and task.kind == "core":
                failed = report_mod.record_onboarded(self.w, repo, commit)
            task.commit = "" if failed else commit
            return failed
        for turn in range(2, REPAIR_TURNS + 3):
            outcome = self.integration.integrate(task.id, check)
            self.w.journal.event("integration", f"{task.id}: {outcome.detail}")
            if outcome.merged:
                task.commit = outcome.commit
                # A record that could not be written is not a task done: the next start
                # finds the ref and takes the record from it.
                failed = (
                    report_mod.record_onboarded(self.w, tree, outcome.commit)
                    if task.kind == "core"
                    else ""
                )
                self.integration.close(task.id)
                return failed
            if outcome.checked:
                return outcome.detail
            if not (outcome.base or outcome.repairable) or turn > REPAIR_TURNS + 1:
                return f"its branch did not integrate: {outcome.detail}"
            files = ", ".join(f"`{f}`" for f in outcome.conflicts) or "files git did not name"
            note = prompts.repair(base=outcome.base, files=files, detail=outcome.detail, tree=tree)
            path = decision_path(self.workspace, task.id, task.attempts, turn)
            path.unlink(missing_ok=True)
            self._dispatch(task, role, {**facts, "repair": note}, path)
            data = read_decision(path)
            if data is None or data["decision"] != READY:
                return (
                    f"its branch did not integrate ({outcome.detail}), and the turn asked to "
                    "put it right did not finish"
                )
        return "its branch did not integrate"

    def _scan(self, task: Task, data: Mapping[str, Any]) -> None:
        """Read the checkout a task just landed in for the conventions: reported, never a gate.

        What it finds is kept for the report and the final `fix` task (``conventions.json``);
        no finding fails the task or sends it back to its worker.
        """
        target_rel = str(getattr(self.w._preflight, "target_rel", "") or "")
        repo, arch = self.w.task_data.get("trtllm_repo_path"), self.w.task_data.get("arch")
        if not (target_rel and repo):
            return
        item = self._queue_item(task) if task.kind == "kernel" else None
        paths = entry_paths(item, target_rel, repo) if item is not None else None
        landing, named = (paths[0] if paths else ""), str(data.get("entry") or "").strip()
        try:
            found = conventions.scan(
                repo, target_rel, str(arch or ""), landing=landing, named=named
            )
            new = conventions.record(self.workspace, task.id, found)
        except Exception as exc:  # a scan that cannot read the tree says so, and holds nothing
            new = [f"the scan could not run: {type(exc).__name__}: {exc}"]
        if new:
            said = f"{task.id}: {len(new)} finding(s): {'; '.join(new)}"
            self.w.journal.event("conventions", said)

    def _dispatch(self, task: Task, role: str, facts: Mapping[str, Any], path: Path) -> None:
        """One turn of ``role`` for ``task``, told to write its receipt at ``path``."""
        prompt = prompts.dispatch(task.kind, {**facts, "path": path})
        self.w.journal.event(f"dispatch: {role}", f"{task.id}, to write {path}")
        self.w._run_turn(role, task.id, prompt, decision_file=path)

    def table(self) -> str:
        """Every task and where it stands, as a markdown table."""
        lines = [
            "| task | kind | unit | status | attempts | grants | decision | summary |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t in self.tasks:
            said = clip((t.summary or t.note).replace("\n", " ").replace("|", "/"), 160)
            lines.append(
                f"| `{t.id}` | {t.kind} | {t.unit or '-'} | {t.status} | {t.attempts} | "
                f"{t.grants} | {t.decision or '-'} | {said} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------- the facts a task's message is built from
    def cli(self) -> str:
        """The workflow's CLI as an agent calls it: by path, when preflight installed it."""
        script = self.workspace / "bin" / gate_cli.CLI_NAME
        return str(script) if script.is_file() else gate_cli.CLI_NAME

    def _facts(self, task: Task, tree: Path | None) -> dict[str, Any]:
        """What ``task``'s message is built from (``agents/prompts``): no prose, only what the run knows."""
        w = self.w
        repo = str(w.task_data.get("trtllm_repo_path") or "")
        found = deployments_mod.resolve(w.task_data)
        deployment = found[self.units.index(task.unit)] if task.unit in self.units else found[0]
        target = str(getattr(w._preflight, "target_rel", "") or "")
        ended = [t for t in self.tasks if t is not task and t.status in (DONE, NOT_DONE, SKIPPED)]
        head, lacks = behind(tree, self.integration.repo) if tree is not None else ("", "")
        work = str(tree or repo) if task.kind != "survey" else ""
        gates = self.workspace / "gates" / deployment.name
        facts: dict[str, Any] = dict(
            task_id=task.id,
            goal=task.goal,
            attempt=task.attempts,
            attempts=self.max_attempts,
            note=task.note,
            worktree=work,
            repo=repo,
            head=head,
            behind=lacks,
            target=target,
            arch=str(w.task_data.get("arch") or ""),
            judge=", ".join(f"`{path}`" for path in w._commit_policy().judge_paths),
            package=Path(__file__).resolve().parents[1],
            notes=self.workspace / NOTES_FILENAME,
            spec=w.task_path,
            plan=w.plan_path,
            vocabulary=w.vocabulary_path,
            gate=work
            and gate_cli.invocation(w.task_path, gates, deployment.name, self.cli(), work),
            history=[
                dict(
                    id=t.id,
                    kind=t.kind,
                    state="skipped" if t.kind == "kernel" and t.status == NOT_DONE else t.status,
                    decision=t.decision,
                    said=clip((t.summary or t.note).replace("\n", " "), 300),
                )
                for t in ended[-HISTORY_TASKS:]
            ],
            earlier=max(0, len(ended) - HISTORY_TASKS),
            tasks=self.path,
            decisions=self.workspace / DECISIONS_DIRNAME,
            host_facts=w.host_facts_path if w.host_facts_path.is_file() else "",
            rebuilt=[
                t.id
                for t in self.tasks
                if t is not task and t.lib_end not in ("", t.lib_start, task.lib_start)
            ],
            # What the report says of them: kept and verified, incomplete, or not needed.
            kept=report_mod.kept_libraries(w) if self.state.onboarded else "",
            python=w.environment.python if w.environment is not None else "",
            prefix=w.environment.prefix if w.environment is not None else "",
            devices=", ".join(map(str, self.state.devices)) or "none",
            text_only=w.task_data.get("scope") == "text_only",
            spec_notes=str(w.task_data.get("notes") or "").strip(),
        )
        if task.kind == "survey":
            facts.update(
                checkpoint=w.task_data.get("checkpoint_path"),
                described=deployments_mod.describe(found),
                inventory=w._surface_block,
            )
        elif task.kind == "kernel":
            facts.update(self._row(task, target, repo, work))
        else:
            facts.update(self._unit(task, deployment, found[0].name, target))
        return facts

    def _unit(self, task: Task, deployment, first: str, target: str) -> dict[str, Any]:
        """What a task on a deployment is told of it: its options, the checks it owes, what judges them."""
        w, name = self.w, deployment.name
        skipped = getattr(w._preflight, "not_attempted", ())
        left = {r["check"] for r in skipped if r["deployment"] == name}
        owed = [c for c in w._contract.of(name) if c.gate not in left]
        frozen = getattr(w, "_anchors", None) or {}
        judged = []
        for check in owed:
            anchor = frozen.get(f"{name}:{check.gate.partition('[')[2].rstrip(']')}")
            if anchor is not None:
                value = f" {anchor.value:g}" if anchor.value is not None else ""
                judged.append(f"`{check.key}`: {anchor.kind}{value} ({anchor.source})")
            elif check.gate != "boot":
                judged.append(f"`{check.key}`")
        files = [f"`{s.options_path}`" for s in deployment.servers.values() if s.options_path]
        return dict(
            deployment=name,
            first=first,
            described=deployments_mod.describe((deployment,)),
            options=", ".join(files) or f"`deployments.{name}` in the spec",
            checks=", ".join(f"`{check.key}`" for check in owed),
            judged=judged,
            anchors=self.workspace / anchors_mod.ANCHORS_FILENAME,
            relative=deployment.relative_to or "",
            onboarded=str((self.state.onboarded or {}).get("commit") or "")[:12],
            cards=prompts.cards(deployment) if task.kind in ("core", "deployment") else [],
            models=f"{MODELING_V2_PACKAGE}/models",
            tests=model_ops_tests_dir(target) if target else "",
        )

    def _queue_item(self, task: Task) -> QueueItem | None:
        """The vocabulary queue row a kernel task names."""
        try:
            queue = load_vocabulary(self.w.vocabulary_path, units=self.units)
        except VocabularyError:
            return None
        return next((q for q in queue if str(q.n) == task.ref), None)

    def _row(self, task: Task, target: str, repo: str, tree: str) -> dict[str, Any]:
        """What a kernel task is told of its queue row: the entry, where it lands, how it is tested."""
        w = self.w
        item = self._queue_item(task)
        if item is None:
            raise RuntimeError(f"kernel task {task.id} names no queue row ({task.ref!r})")
        try:
            rows = (yaml.safe_load(w.vocabulary_path.read_text(encoding="utf-8")) or {}).get(
                "queue"
            )
        except (OSError, yaml.YAMLError, AttributeError):
            rows = ()
        row = next(
            (r for r in rows or () if isinstance(r, dict) and str(r.get("n")) == str(item.n)), {}
        )
        # A shared entry lands as its local variant in the target's own ops/.
        local = item.variant is not None and bool(target)
        landing = str(item.variant) if local else item.target
        paths = entry_paths(item, target, repo or None) or ()
        env = w.environment
        python = f"{env.prefix} {env.python}".strip() if env is not None else "python"
        return dict(
            n=item.n,
            mode=item.mode,
            entry=item.target,
            onboard=item.mode == MODE_ONBOARD,
            row_note=(item.note or "").strip(),
            surface=item.surface or "",
            cells=row.get("cells") or "",
            tier=int(row["tier"]) if str(row.get("tier")).strip().isdigit() else None,
            paths=paths,
            shared=" and ".join(f"`{p}`" for p in shared_entry_paths(item.target)) if local else "",
            name=item.target.strip("/").split("/")[-1],
            module=landing.replace("/", "."),
            test_command=f"cd {tree} && PYTHONPATH={tree} {python} -m pytest {paths[1]} -q"
            if len(paths) > 1
            else "",
        )

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
        log = lambda message: self.w.journal.event("integration", message)  # noqa: E731
        return Integration(
            repo, self.workspace, f"mv2/{self.state.slug or 'run'}", env, log, policy
        )


def unit_names(task_data) -> list[str]:
    """The run's units of work: its deployments by name, in order, the first as ``core``."""
    return ["core", *(d.name for d in deployments_mod.resolve(task_data)[1:])]


def _failure_note(result: TurnResult) -> str:
    """Why an attempt did not finish: its result line, for the report and the next attempt."""
    if result.error:
        return result.error
    raw = result.data
    said = clip(str(raw.get("summary") or ""), 800)
    return f"it ended {result.decision}" + (f": {said}" if said else "")
