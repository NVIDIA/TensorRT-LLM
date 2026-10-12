# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The orchestrator's side of the evidence ledger.

A verdict needs context no gate has: the repository and target it measured,
the unit being gated, the environment every gate command runs under, and the
spec. The orchestrator has all of it. :class:`RunEvidence` binds the two:
``gates.GateLog`` calls :meth:`RunEvidence.record` with every result, and the
result goes into the workspace ledger with that context attached.

At delivery, :meth:`RunEvidence.settle_delivery` makes the evidence describe
the tree being delivered. It re-runs every gate whose latest verdict is
stale, failed, or was measured on another tree or native build, records the
full regression once every required verdict passes on this tree, and refuses
delivery when either is not achieved.

The workflow holds one instance and calls into it from a few hook points, so
``workflow.py`` itself carries no ledger logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message

from . import envelope as envelope_mod
from . import gates as gates_mod
from .freshness import (
    RERUN_QUEUE_FILENAME,
    Assessment,
    RerunEntry,
    assess,
    complete_regression,
    drain,
    last_regression,
    record_regression,
    regression_binding,
    rerun_queue,
    write_rerun_queue,
)
from .ledger import (
    LEDGER_FILENAME,
    Ledger,
    LedgerPolicy,
    NativeBuild,
    TreeIdentity,
    Verdict,
    VerdictContext,
    hash_paths,
    measure,
    native_build,
    record_gate_result,
    tree_identity,
)
from .report import (
    FINAL_REPORT,
    PROGRESS_HTML,
    PROGRESS_JSON,
    check_consistency,
    collect_facts,
    write_reports,
)
from .roles import ROLES, resolve_role_agent
from .task_schema import assembly_units, members_for

#: The verdict a run closes with when its evidence does not describe the tree
#: it would deliver. Remediable: the repair (a fix, then a re-run of the
#: failing gate) happens outside the run, and re-entering delivery re-checks.
VERDICT_EVIDENCE = "delivery-failed:evidence"

_QUEUE_LINES_SHOWN = 20

#: Ledger keys for verdicts that are not engine gates. They carry no unit, so
#: they are reported and kept as history but never re-run or required by the
#: final regression: an early check re-runs after every unit by itself, and
#: the delivery checks run again on every delivery.
EARLY_CHECK_PREFIX = "early:"
DELIVERY_CHECK_PREFIX = "delivery:"

#: ``at B=8 T=2048 ...: <boot detail>`` -- how an envelope probe names its point.
_PROBE_POINT = re.compile(r"^at (?P<point>.+?): ")


@dataclass(frozen=True)
class _CheckResult:
    """A check shaped like a gate result, so it records through the one entry point."""

    name: str
    passed: bool
    detail: str
    finding: bool = False


def role_agents(task_data: Any) -> dict[str, dict[str, Any]]:
    """The backend and model every role is routed to, as the ``agents`` block resolves them."""
    agents: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        agent = resolve_role_agent(task_data or {}, role)
        agents[role] = {"backend": agent.backend, "model": agent.model}
        if agent.reasoning_effort:
            agents[role]["reasoning_effort"] = agent.reasoning_effort
    return agents


def gate_key(result: Any) -> str:
    """The ledger key for ``result``: its name, qualified by the point for an envelope probe.

    Every probe of a unit at one member is named ``envelope``; keyed by name
    alone, the last probe would supersede the others and a failing point
    could hide behind a passing one.
    """
    name = str(getattr(result, "name", "") or "gate")
    if name == envelope_mod.PROBE_GATE:
        match = _PROBE_POINT.match(str(getattr(result, "detail", "") or ""))
        if match:
            return f"{name}[{match.group('point')}]"
    return name


#: Where ``--clean`` moves the previous run's evidence. Moved, not deleted:
#: the ledger is append-only, and a fresh run does not change what an old
#: one measured.
ARCHIVE_DIRNAME = "archive"
_ARCHIVED = (LEDGER_FILENAME, RERUN_QUEUE_FILENAME, FINAL_REPORT, PROGRESS_JSON, PROGRESS_HTML)


class RunEvidence:
    """One run's ledger, and the hooks the orchestrator calls."""

    def __init__(
        self, workflow: Any, *, policy: LedgerPolicy | None = None, clean: bool = False
    ) -> None:
        self._workflow = workflow
        self.ledger = Ledger.in_workspace(workflow.workspace)
        self.policy = policy or LedgerPolicy()
        #: Verdict ids already reported as stale in the journal.
        self._warned: set[str] = set()
        if clean:
            self.archive()

    def archive(self) -> list[Path]:
        """Move the previous run's ledger and reports under ``archive/``, stamped."""
        workspace = Path(self._workflow.workspace)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        moved = []
        for name in _ARCHIVED:
            path = workspace / name
            if path.is_file():
                target = workspace / ARCHIVE_DIRNAME / f"{stamp}-{name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
                moved.append(target)
        return moved

    # -------------------------------------------------------------- reports

    def refresh(self, state: Any = None) -> None:
        """Regenerate the reports from the ledger, and journal verdicts that just went stale.

        Called after every checkpoint, so the page never trails the run by
        more than one step. The git-history switch check is left to delivery:
        it costs a ``git diff`` per verdict commit, and this runs often.
        A report that cannot be written is journaled rather than raised; it
        is regenerated at the next checkpoint and must not stop a run.
        """
        wf = self._workflow
        try:
            facts = collect_facts(
                wf.workspace,
                repo_root=self.repo_root() or None,
                task_data=getattr(wf, "task_data", None) or None,
                scope=self.reportable(),
                policy=self.policy,
            )
            write_reports(wf.workspace, facts=facts, final_report=bool(getattr(state, "done", 0)))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            wf.journal.event("reports not regenerated", f"{type(exc).__name__}: {exc}")
            return
        newly = [
            item
            for item in facts.assessments.values()
            if item.stale and item.verdict.id not in self._warned
        ]
        if newly:
            self._warned.update(item.verdict.id for item in newly)
            wf.journal.section(
                f"evidence: {len(newly)} verdict(s) went stale",
                "\n".join(f"- {item.verdict.label()}: {'; '.join(item.reasons)}" for item in newly)
                + "\n\nDelivery re-runs them on the delivered tree.",
            )

    # ------------------------------------------------------------ recording

    def unit(self) -> str:
        """The unit whose gates are running now, as the orchestrator last set it."""
        ctx = getattr(self._workflow, "_progress_ctx", None)
        return str(getattr(ctx, "current_unit", "") or "")

    def repo_root(self) -> str:
        return str((getattr(self._workflow, "task_data", None) or {}).get("trtllm_repo_path") or "")

    def context(self, unit: str | None = None) -> VerdictContext:
        """Everything a verdict records that the gate result does not carry."""
        wf = self._workflow
        task = dict(getattr(wf, "task_data", None) or {})
        unit = self.unit() if unit is None else unit
        preflight = getattr(wf, "_preflight", None)
        env = dict(((task.get("environment") or {}).get("env")) or {})
        if getattr(wf, "environment", None) is not None:
            env = dict(wf._gate_env().env)
        features: list[str] = []
        if unit and task:
            features = list(wf._features_through(unit))
        return VerdictContext(
            repo_root=self.repo_root(),
            target_rel=str(getattr(preflight, "target_rel", "") or ""),
            unit=unit,
            env=env,
            features=features,
            task_data=task or None,
            policy=self.policy,
            agents=role_agents(task),
        )

    def record(self, result: Any) -> Verdict:
        """The ``GateLog`` sink: put one gate result into the ledger."""
        return record_gate_result(self.ledger, result, self.context(), gate=gate_key(result))

    def record_checks(self, prefix: str, checks: Iterable[_CheckResult]) -> list[Verdict]:
        """Record checks that are not engine gates, under ``prefix``, against one reading.

        They carry no unit and no target dependencies: they are about the
        repository as a whole, and their freshness is that they run again.
        """
        context = replace(self.context(unit=""), target_rel="")
        measurement = measure(context)
        return [
            record_gate_result(
                self.ledger, check, context, gate=f"{prefix}{check.name}", measurement=measurement
            )
            for check in checks
        ]

    def record_early_checks(self, report: Any) -> Any:
        """Record an early-checks report: a ``warn`` is a FINDING, ``ok``/``info`` a PASS."""
        self.record_checks(
            EARLY_CHECK_PREFIX,
            (
                _CheckResult(
                    name=str(f.name),
                    passed=True,
                    detail=str(f.detail),
                    finding=str(f.level) == "warn",
                )
                for f in getattr(report, "findings", ())
            ),
        )
        return report

    def record_delivery_checks(self, report: Any) -> Any:
        """Record every delivery check (the capacity audit and matrix evidence among them)."""
        self.record_checks(
            DELIVERY_CHECK_PREFIX,
            (
                _CheckResult(name=str(c.name), passed=bool(c.passed), detail=str(c.detail))
                for c in getattr(report, "checks", ())
            ),
        )
        return report

    # ------------------------------------------------------------ freshness

    def in_scope(self) -> Callable[[Verdict], bool]:
        """Whether the current spec still requires a verdict: its unit, and its member."""
        wf = self._workflow
        task = getattr(wf, "task_data", None) or {}
        units = assembly_units(task)
        world = wf._gate_env().world_size if getattr(wf, "environment", None) is not None else 1

        plan = getattr(getattr(wf, "_preflight", None), "certification", None)

        def keep(verdict: Verdict) -> bool:
            if verdict.unit not in units:
                return False
            if verdict.gate.startswith(f"{envelope_mod.PROBE_GATE}[") and plan is not None:
                return self.probe_for(verdict.gate, verdict.unit, verdict.member) is not None
            return not verdict.member or verdict.member in members_for(task, verdict.unit, world)

        return keep

    def reportable(self) -> Callable[[Verdict], bool]:
        """What the reports show: every required verdict, and every unit-less check."""
        keep = self.in_scope()
        return lambda verdict: not verdict.unit or keep(verdict)

    def probe_for(self, gate: str, unit: str, member: str) -> Any:
        """The certification plan's probe a ledger key names, or None."""
        plan = getattr(getattr(self._workflow, "_preflight", None), "certification", None)
        for probe in plan.probes_for(unit) if plan is not None else ():
            if probe.member == member and gate == f"{envelope_mod.PROBE_GATE}[{probe.point()}]":
                return probe
        return None

    def current(self) -> tuple[TreeIdentity, NativeBuild]:
        """The tree and native build the ledger's verdicts would be delivered on."""
        repo = self.repo_root()
        deps: set[str] = set()
        for verdict in self.ledger.latest().values():
            deps.update(verdict.dependencies)
        tree = tree_identity(repo, dependencies=hash_paths(repo, sorted(deps)))
        return tree, native_build(repo, globs=self.policy.native_globs)

    def assess(self, *, check_history: bool = True) -> list[Assessment]:
        """The latest in-scope verdicts, judged against the tree as it is now."""
        keep = self.in_scope()
        return assess(
            [v for v in self.ledger.latest().values() if keep(v)],
            self.repo_root(),
            task_data=getattr(self._workflow, "task_data", None),
            policy=self.policy,
            check_history=check_history,
        )

    def queue(self) -> tuple[list[RerunEntry], TreeIdentity, NativeBuild]:
        """What delivery must run again before the evidence describes this tree."""
        tree, native = self.current()
        return rerun_queue(self.assess(), tree=tree, native=native), tree, native

    # ------------------------------------------------------------- re-runs

    def rerunner(self, entry: RerunEntry, log: Any = None) -> Callable[[], Any] | None:
        """A callable that re-measures ``entry`` and records it, or None if unknown.

        Only gates the orchestrator itself fires are known here, invoked the
        way the assembly loop invokes them. A gate that cannot be re-run from
        here stays queued, and delivery refuses on it rather than guessing.
        """
        from .workflow import WITNESS_GATED_UNITS

        wf = self._workflow
        unit, member = entry.unit, entry.member
        if entry.gate.startswith(f"{envelope_mod.PROBE_GATE}["):
            probe = self.probe_for(entry.gate, unit, member)
            if probe is None:
                return None
            return lambda: wf.gate_log.record(
                envelope_mod.run_envelope_probe(
                    wf._gate_env(),
                    task_data=wf.task_data,
                    probe=probe,
                    spec=gates_mod.BootSpec.load(wf.boot_spec_path),
                    runner=wf._shell_runner,
                    features=wf._features_through(unit),
                )
            )
        if entry.gate.startswith("boot"):
            return lambda: wf.gate_log.record(
                gates_mod.run_boot_gate(
                    wf._gate_env(),
                    task_data=wf.task_data,
                    spec=gates_mod.resolve_boot_spec(
                        wf.task_data, unit, wf._features_through(unit), wf.boot_spec_path
                    ),
                    runner=wf._shell_runner,
                    features=wf._features_through(unit),
                    unit=unit,
                    member=member,
                    workdir=wf.workspace / "gates" / unit,
                )
            )
        if entry.gate.startswith("accuracy"):
            return lambda: wf.gate_log.record(
                gates_mod.run_accuracy_gate(
                    wf._gate_env(),
                    task_data=wf.task_data,
                    reference=wf._reference_score(unit),
                    runner=wf._shell_runner,
                    unit=unit,
                    member=member,
                )
            )
        if entry.gate == "acceptance":
            return lambda: wf.gate_log.record(
                gates_mod.run_acceptance_gate(
                    wf._gate_env(),
                    task_data=wf.task_data,
                    runner=wf._shell_runner,
                    unit=unit,
                    features=wf._features_through(unit),
                    member=member,
                    workdir=wf.workspace / "gates" / unit,
                    records=gates_mod.records_for(wf.workspace),
                )
            )
        if entry.gate == "scheduler_equivalence":
            from . import step_check

            return lambda: wf.gate_log.record(
                step_check.run_scheduler_equivalence(
                    wf._gate_env(),
                    task_data=wf.task_data,
                    runner=wf._shell_runner,
                    unit=unit,
                    member=member,
                    features=wf._features_through(unit),
                    workdir=wf.workspace / "gates" / unit / "scheduler_equivalence",
                )
            )
        if entry.gate == "launch":
            from . import launch_gate

            return lambda: wf.gate_log.record(
                launch_gate.run_launch_gate(
                    wf._gate_env(),
                    task_data=wf.task_data,
                    reference=launch_gate.smoke_reference(wf.task_data, unit, wf._model_key),
                    runner=wf._shell_runner,
                    unit=unit,
                    member=member,
                    features=wf._features_through(unit),
                    workdir=wf.workspace / "gates" / unit / "launch",
                )
            )
        if unit in WITNESS_GATED_UNITS:
            # Records itself; returns None on a misconfigured block.
            return lambda: wf._run_witness(unit, member, log)
        return None

    def rerun(self, entry: RerunEntry, log: Any = None) -> bool | None:
        """Re-measure one queued verdict. See :func:`freshness.drain` for the answer."""
        run = self.rerunner(entry, log)
        if run is None:
            return None
        wf = self._workflow
        wf._progress_ctx.current_unit = entry.unit
        wf.journal.event("gate re-run for delivery", f"{entry.gate}@{entry.member} [{entry.unit}]")
        result = run()
        if result is None:
            return False
        wf.journal.event("gate", result.summary())
        print_message(f"  re-run: {escape_markup(result.summary())}", log)
        return True

    # ------------------------------------------------------------- delivery

    def settle_delivery(self, state: Any, log: Any = None) -> bool:
        """Bring the evidence onto the delivery tree, or close the run saying why not.

        1. Queue every in-scope latest verdict that is stale, did not pass,
           or was measured on another tree or native build.
        2. Re-run the queue.
        3. Record a full regression if every required verdict now passed on
           this tree and build.
        4. Refuse unless nothing is still queued and the tree under merge is
           the tree of the last full regression.

        Outside a git work tree there is no tree to bind, so only freshness is
        checked; that is the same "absence of git is absence of a claim" rule
        the delivery checks follow.
        """
        wf = self._workflow
        if not self.repo_root():
            return True
        queue, tree, native = self.queue()
        write_rerun_queue(wf.workspace, queue)
        if queue:
            wf.journal.section(
                f"evidence: {len(queue)} gate(s) to re-run before delivery", _render_queue(queue)
            )
            print_message(
                f"[bold cyan]→ evidence: re-running {len(queue)} gate(s) so every verdict "
                f"describes tree {tree.short()}[/bold cyan]",
                log,
            )
            drain(queue, lambda entry: self.rerun(entry, log))
            queue, tree, native = self.queue()
            write_rerun_queue(wf.workspace, queue)

        problems: list[str] = []
        if queue:
            problems.append(
                f"{len(queue)} verdict(s) still do not hold on {tree.short()}:\n"
                + _render_queue(queue)
            )
        else:
            self._record_regression(tree, native)
        binding = regression_binding(self.ledger, self.repo_root(), policy=self.policy)
        if binding.claim and not binding.ok:
            problems.append(f"final regression binding: {binding.detail}")
        self.refresh(state)
        contradictions = check_consistency(wf.workspace)
        if contradictions:
            problems.append(
                f"{len(contradictions)} report statement(s) contradict the ledger:\n"
                + "\n".join(f"- {c.render()}" for c in contradictions)
            )

        body = "\n\n".join(problems) if problems else f"ok: {binding.detail}"
        wf.journal.section("evidence at delivery", body)
        if problems:
            print_message(
                "[bold red]✘ delivery refused: the evidence does not describe the tree being "
                "delivered.[/bold red]\n" + escape_markup(body),
                log,
            )
            wf._close_run(state, log, verdict=VERDICT_EVIDENCE)
            return False
        print_message(f"[bold green]✔ evidence: {escape_markup(binding.detail)}[/bold green]", log)
        return True

    def annotate_premerge(self, pack: str | Path) -> str:
        """Append the final-regression binding to the pre-merge pack; return a one-line summary.

        The pack's remaining steps include a lint pass and a split into PR
        branches, and both can produce a tree no regression ran on. So the
        pack says which tree is bound, and gives the command that re-checks
        whatever is finally merged.
        """
        repo = self.repo_root()
        binding = regression_binding(self.ledger, repo, policy=self.policy)
        workspace = Path(self._workflow.workspace)
        command = (
            f"python -m agent_flow.workflows.staircase_onboard.freshness check-merge "
            f"{workspace} --repo {repo} --rev <the branch or commit being merged>"
        )
        state = "bound" if binding.ok else "NOT bound"
        section = [
            "",
            f"## Evidence binding ({state})",
            "",
            binding.detail + ".",
            "",
            "A lint fix, a split or any later commit changes the tree; re-check the exact rev "
            "that merges, and re-enter delivery (which re-runs the gates on it) if it is refused:",
            "",
            f"```\n{command}\n```",
            "",
        ]
        path = Path(pack)
        if path.is_file():
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(section))
        return f"evidence {state}: {binding.detail}"

    def _record_regression(self, tree: TreeIdentity, native: NativeBuild) -> None:
        latest = self.ledger.latest()
        regression = complete_regression(latest, tree, native, scope=self.in_scope())
        if regression is None:
            return
        last = last_regression(self.ledger)
        if (
            last is not None
            and last.tree.key == regression.tree.key
            and last.native.fingerprint == regression.native.fingerprint
            and set(last.verdict_ids) == set(regression.verdict_ids)
        ):
            return
        record_regression(self.ledger, regression)
        self._workflow.journal.event(
            "full regression recorded",
            f"{len(regression.keys)} verdicts on {tree.short()} (native {native.short()})",
        )


def _render_queue(queue: list[RerunEntry]) -> str:
    lines = [
        f"- {e.gate}@{e.member or '-'} [{e.unit}]: {'; '.join(e.reasons)}"
        for e in queue[:_QUEUE_LINES_SHOWN]
    ]
    if len(queue) > _QUEUE_LINES_SHOWN:
        lines.append(f"- (+{len(queue) - _QUEUE_LINES_SHOWN} more in rerun_queue.json)")
    return "\n".join(lines)


__all__ = [
    "DELIVERY_CHECK_PREFIX",
    "EARLY_CHECK_PREFIX",
    "RunEvidence",
    "VERDICT_EVIDENCE",
    "gate_key",
    "role_agents",
]
