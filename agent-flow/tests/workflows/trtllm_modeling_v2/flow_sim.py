# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run on the CPU: the real orchestrator, with fake workers, checks, target and planner.

:class:`Simulation` drives :class:`~...run.orchestrator.GraphOrchestrator`
end to end. The target is a counter that moves each time a task lands; a
check passes unless the scenario says it fails on that state; a worker sleeps
its task's duration and returns the decision the scenario scripts, or its
kind's success; a ``tune`` task runs until the run tells the campaign to stop.
Everything is recorded with the time it happened, so a scenario asserts the
order of events, and :meth:`Simulation.run` fails a run that does not end.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run import graph as g
from agent_flow.workflows.trtllm_modeling_v2.run.integration import INTEGRATED_KINDS
from agent_flow.workflows.trtllm_modeling_v2.run.orchestrator import (
    FINISHED,
    GraphOrchestrator,
    PlannerAnswer,
    PlannerTurn,
    Services,
)

#: What a worker of each kind ends with when the scenario says nothing.
SUCCESS = {
    "survey": "DRAFT_READY",
    "kernel": "LANDED",
    "assemble": "UNIT_READY",
    "feature": "UNIT_READY",
    "integrate": "UNIT_READY",
    "investigate": "UNIT_READY",
    "review": "APPROVE",
}


def task(
    tid: str,
    kind: str,
    satisfies: Iterable[str] = (),
    deps: Iterable[str] = (),
    *,
    soft: Iterable[str] = (),
    priority: str = "required",
    touches: Sequence[str] | None = None,
    gpus: int = 1,
    exclusive: bool = False,
    ref: str = "",
) -> g.Task:
    """A task as the planner writes one."""
    return g.Task(
        id=tid,
        kind=kind,
        goal=f"do {tid}",
        depends_on=list(deps),
        depends_on_soft=list(soft),
        touches=list(touches if touches is not None else [f"models/{tid}.py"]),
        gpus=gpus,
        est_minutes=10,
        satisfies=list(satisfies),
        priority=priority,
        exclusive=exclusive,
        ref=ref,
    )


class Target:
    """The tree under test: one more landing, one more state; its marker names the state."""

    def __init__(self) -> None:
        self.version = 1
        self.landed: list[str] = []
        self._lock = threading.Lock()

    @property
    def marker(self) -> str:
        return f"tree:{self.version}"

    def land(self, what: str) -> None:
        with self._lock:
            self.version += 1
            self.landed.append(what)


class Planner:
    """A planner that writes ``first`` on its first turn, then answers each event by ``rules``.

    A rule is ``(match, answer)``: ``match`` is a substring of an event of the
    turn, ``answer`` takes the orchestrator and returns the turn's answer. Each
    rule answers once. With no rule matching, the turn says ``DONE``.
    """

    def __init__(
        self,
        first: Sequence[g.Task],
        rules: Sequence[tuple[str, Callable[[GraphOrchestrator], PlannerAnswer]]] = (),
    ) -> None:
        self.first = list(first)
        self.rules = list(rules)
        self.turns: list[PlannerTurn] = []
        self.orchestrator: GraphOrchestrator | None = None
        self.raising: dict[int, Exception] = {}

    def __call__(self, turn: PlannerTurn) -> PlannerAnswer:
        self.turns.append(turn)
        error = self.raising.get(len(self.turns))
        if error is not None:
            raise error
        if len(self.turns) == 1:
            return PlannerAnswer("GRAPH_UPDATED", "first plan", g.Graph(tasks=list(self.first)))
        for index, (match, answer) in enumerate(self.rules):
            if any(match in event for event in turn.events):
                del self.rules[index]
                return answer(self.orchestrator)
        return PlannerAnswer("DONE", "every check holds")

    def heard(self, words: str) -> bool:
        return any(words in event for turn in self.turns for event in turn.events)


def revise(
    orchestrator: GraphOrchestrator, *changes: g.Task, reason: str = "revise"
) -> PlannerAnswer:
    """The graph as it stands, with ``changes`` written over the tasks of the same id."""
    by_id = {t.id: t for t in orchestrator.graph.tasks}
    for change in changes:
        by_id[change.id] = change
    return PlannerAnswer("GRAPH_UPDATED", reason, g.Graph(tasks=list(by_id.values())))


class Simulation:
    """One run's fakes, the orchestrator over them, and what happened when."""

    def __init__(
        self,
        tmp_path: Path,
        contract: c.Contract,
        planner: Planner,
        *,
        decisions: Mapping[str, Sequence[str]] | None = None,
        durations: Mapping[str, float] | None = None,
        failing: Callable[[c.Check, int], bool] = lambda check, version: False,
        raising: Callable[[c.Check, int], bool] = lambda check, version: False,
        gpus: int = 8,
        **orchestrator_kw: Any,
    ) -> None:
        self.tmp_path = Path(tmp_path)
        self.contract = contract
        self.planner = planner
        self.target = Target()
        self.decisions = {k: list(v) for k, v in (decisions or {}).items()}
        self.durations = dict(durations or {})
        self.failing = failing
        self.raising = raising
        self.timeline: list[tuple[float, str]] = []
        self.ran: list[tuple[str, str, bool]] = []
        self.milestones: list[str] = []
        self.stop_reasons: list[str] = []
        self.stopped = threading.Event()
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self.services = Services(
            plan=planner,
            work=self.work,
            check=self.check,
            free_gpus=lambda: gpus,
            marker=lambda: self.target.marker,
            correct=self.correct,
            stop_campaign=self.stop,
            log=lambda message: self.note(f"log {message}"),
        )
        self.orchestrator = GraphOrchestrator(
            self.services,
            contract,
            c.ContractStatus.load(self.tmp_path),
            self.tmp_path,
            poll_s=0,
            **orchestrator_kw,
        )
        planner.orchestrator = self.orchestrator

    # -------------------------------------------------------------- the fakes
    def note(self, what: str) -> None:
        with self._lock:
            self.timeline.append((time.monotonic() - self._t0, what))

    def when(self, prefix: str) -> list[float]:
        """When each event starting with ``prefix`` happened, in order."""
        return [t for t, what in self.timeline if what.startswith(prefix)]

    def at(self, kind: str, tid: str) -> list[float]:
        """When task ``tid``'s events of ``kind`` (``start``, ``end``, ``land``) happened, in order."""
        return [t for t, what in self.timeline if what.split(" ", 2)[:2] == [kind, tid]]

    def work(self, t: g.Task) -> str:
        self.note(f"start {t.id}")
        if t.kind == "tune":
            # A campaign with hours of budget left runs until it is told to stop.
            stopped = self.stopped.wait(timeout=20)
            self.note(f"end {t.id}")
            return "TUNED\ncampaign ended: best branch tune/best" if stopped else "TUNED"
        time.sleep(self.durations.get(t.id, self.durations.get(t.kind, 0.01)))
        scripted = self.decisions.get(t.id)
        decision = scripted.pop(0) if scripted else SUCCESS.get(t.kind, "UNIT_READY")
        if decision.split("\n", 1)[0] in FINISHED and t.kind in INTEGRATED_KINDS:
            self.target.land(t.id)
            self.note(f"land {t.id} -> {self.target.marker}")
        self.note(f"end {t.id} {decision.split(chr(10), 1)[0]}")
        return decision

    def check(self, chk: c.Check) -> Any:
        version = self.target.version
        if self.raising(chk, version):
            self.note(f"raise {chk.key}")
            raise RuntimeError("the session running the check was lost")
        time.sleep(0.002)
        passed = not self.failing(chk, version)
        with self._lock:
            self.ran.append((chk.key, f"tree:{version}", passed))
        self.note(f"check {chk.key} {'pass' if passed else 'FAIL'} tree:{version}")
        return SimpleNamespace(name=chk.gate, member=chk.member, passed=passed, detail="")

    def correct(self, marker: str, evaluation: c.Evaluation) -> Mapping[str, Any]:
        self.milestones.append(marker)
        self.note(f"correct {marker}")
        return {"sha": marker.replace("tree:", "c0ffee"), "ref": "x-correct", "elapsed_s": 1.0}

    def stop(self, reason: str) -> None:
        self.stop_reasons.append(reason)
        self.note(f"stop campaign: {reason}")
        self.stopped.set()

    # -------------------------------------------------------------- the run
    def run(self, timeout: float = 30.0) -> str:
        """The run's verdict; fails when it has not ended within ``timeout`` seconds."""
        result: dict[str, Any] = {}

        def body() -> None:
            try:
                result["verdict"] = self.orchestrator.run()
            except BaseException as exc:  # noqa: BLE001 -- reported to the test below
                result["error"] = exc

        thread = threading.Thread(target=body, name="sim-run", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.stopped.set()
            raise AssertionError(f"the run did not end within {timeout}s: {self.timeline[-10:]}")
        if "error" in result:
            raise result["error"]
        return result["verdict"]

    def statuses(self) -> dict[str, str]:
        return {t.id: t.status for t in self.orchestrator.graph.tasks}


__all__ = ["Planner", "SUCCESS", "Simulation", "Target", "revise", "task"]
