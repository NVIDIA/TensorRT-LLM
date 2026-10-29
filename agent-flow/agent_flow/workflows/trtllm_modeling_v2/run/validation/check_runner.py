# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework check execution: member runners, snapshot provenance and gates."""

from __future__ import annotations

import copy
import threading
from dataclasses import replace
from typing import Any, Callable

from rich.markup import escape as escape_markup

from agent_flow.console import print_message

from ...checks import acceptance as acceptance_checks
from ...checks import accuracy as accuracy_checks
from ...checks import boot as boot_checks
from ...checks import check_config, disagg, gate_records, serve
from ...checks import verdict as check_result
from ...checks.accuracy import resolve_model_key
from ...infra.compute.environment import RunEnvironment
from ...infra.compute.executor import (
    ACCURACY,
    FEATURE,
    current_job_priority,
    job_priority,
    select_command_executor,
    shell_runner_for,
    wait_for_session,
)
from ...infra.compute.sessions import session_wait_from_task
from ...infra.git import resolve_commit
from ...spec import variants as variants_mod
from ...spec.task_schema import (
    assembly_units,
    features_through,
    members_for,
    parse_parallel_segment,
)
from ..workspace import snapshots as snapshots_mod
from . import contract as contract_mod
from . import runtime_facts

#: How many times in a row one gate may be lost to its session and re-run.
#: Each re-run waits for a heartbeat first, so this only binds on a session
#: that keeps coming back and dying again.
SESSION_LOSS_RERUNS = 3


class _SessionLost(RuntimeError):
    """A gate was lost to its session and no session came back: the check has no verdict."""


class CheckRunner:
    """Execute framework-owned checks, with isolated snapshot state and shared evidence."""

    def __init__(self, workflow) -> None:
        self.workflow = workflow
        self.task_data = workflow.task_data
        self.workspace = workflow.workspace
        self.journal = workflow.journal
        self.gate_log = workflow.gate_log
        self._member_runners = {}
        self._accuracy_by_unit = {}
        self._accuracy_lock = threading.Lock()
        self._session = threading.local()
        self._accuracy_owed = {}
        self.pin = None
        for check in workflow._contract.checks if workflow._contract else ():
            if check.gate.startswith("accuracy["):
                self._accuracy_owed.setdefault(check.unit, []).append(check.member)

    def _features_through(self, unit: str) -> list[str]:
        """Features implemented once ``unit`` is assembled, in order.

        A feature unit declares the prefix ending at that feature; ``core``
        implements none. Unknown units retain the full feature list.
        """
        units = assembly_units(self.task_data)
        if unit not in units:
            return [str(f) for f in self.task_data.get("features", [])]
        return list(features_through(self.task_data, unit))

    def _member_env(self, member: str | None) -> RunEnvironment:
        """The run's environment at ``member``'s own world size.

        Members of one unified target may differ in width (tp1 beside tep4)
        when a scheduler hands each gate job its own devices.
        """
        env = self.workflow._gate_env()
        size = parse_parallel_segment(member)[1] if member else None
        return replace(env, world_size=size) if size else env

    def _member_runner(self, member: str | None, gpus: str | None = None) -> Any:
        """The gate runner for ``member``: under a scheduler, a job asking for its world size.

        ``gpus`` overrides the request, for a check that starts more engines
        than one member's worth (the disaggregated-serving test's two).
        """
        env = self._member_env(member)
        if self.workflow._shell_runner_override is not None:
            return (
                self.pin.runner(self.workflow._shell_runner)
                if self.pin
                else self.workflow._shell_runner
            )
        want = gpus or str(env.world_size)
        # A session's spool starts a check of a higher class first (job_priority).
        priority = current_job_priority()
        cache = self._member_runners
        if (want, priority) not in cache:
            executor = select_command_executor(
                env, tree=self.task_data.get("trtllm_repo_path"), gpus=want, priority=priority
            )
            cache[(want, priority)] = self._noting_session_loss(shell_runner_for(executor))
        return cache[(want, priority)]

    def _spec_for(self, unit: str) -> tuple[dict, list[str]]:
        """The spec ``unit``'s checks run under, and the features its target implements.

        A configuration variant runs under its unit's spec with its own
        overlay (``spec/variants.py``), on its unit's target.
        """
        if not variants_mod.is_variant(self.task_data, unit):
            return self.task_data, self._features_through(unit)
        spec = check_config.variant_spec(
            self.task_data,
            unit,
            world_size=self.workflow._gate_env().world_size,
            options_dir=self.workspace / variants_mod.OPTIONS_DIRNAME,
        )
        return spec, self._features_through(variants_mod.base_unit(self.task_data, unit))

    def _gate_until_verdict(self, run, log) -> "check_result.GateResult":
        """Run one gate until it says something about the target, and return that.

        A gate whose session was lost under it -- an allocation requeued, or
        ended before its successor started -- measured nothing, so what it
        returns is not a verdict: it is not recorded and costs no assembler
        turn. The gate runs again once a session heart-beats on the spool,
        after waiting up to `environment.session_wait`, at most
        `SESSION_LOSS_RERUNS` times in a row. Past either bound this raises
        `_SessionLost`: the check has no verdict, and the loop runs it again.

        A loss is what the gate says (`GateResult.infrastructure`) or what the
        runner saw while it ran (`_noting_session_loss`). A gate that passed is
        a verdict whatever happened along the way.
        """
        lost_in_a_row = 0
        while True:
            self._session.loss = ""
            result = (
                self.pin.settle(run, lambda why: self.journal.event("void verdict", why))
                if self.pin
                else run()
            )
            reason = result.infrastructure or self._session.loss
            if result.passed or not reason:
                return result
            lost_in_a_row += 1
            label = f"{result.name}@{result.member}" if result.member else result.name
            self.journal.event("gate lost to its session", f"{label}: {reason}")
            print_message(
                f"[bold yellow]⚠ {escape_markup(label)} did not run to a verdict: "
                f"{escape_markup(reason)}[/bold yellow]",
                log,
            )
            if lost_in_a_row > SESSION_LOSS_RERUNS:
                raise _SessionLost(
                    f"{label} was lost to its session {lost_in_a_row} times in a row, "
                    f"most recently: {reason}"
                )
            if not self._await_session(log):
                raise _SessionLost(
                    f"{label} was lost to its session ({reason}) and no session "
                    f"heart-beat again within {session_wait_from_task(self.task_data):.0f}s"
                )

    def _await_session(self, log) -> bool:
        """Wait up to `environment.session_wait` for the gate session to heart-beat again."""
        limit = session_wait_from_task(self.task_data)
        print_message(
            f"[bold cyan]→ waiting up to {limit:.0f}s for the session to heart-beat again, "
            f"then re-running the gate[/bold cyan]",
            log,
        )
        back = wait_for_session(self.workflow._command_executor, limit)
        self.journal.event(
            "session wait",
            f"{'a session is back' if back else 'no session came back'} "
            f"(waited up to {limit:.0f}s)",
        )
        return back

    def _noting_session_loss(self, runner: Any) -> Any:
        """``runner``, remembering why each command it ran was ended by its session.

        A gate may report such a command as an ordinary failure; the note lets
        `_gate_until_verdict` tell the two apart.
        """
        runner = self.pin.runner(runner) if self.pin else runner

        def run(*args: Any, **kwargs: Any) -> Any:
            result = runner(*args, **kwargs)
            if result.infrastructure:
                self._session.loss = result.infrastructure
            return result

        return run

    def _reference_score(self, unit: str | None = None, spec: Any = None) -> float:
        # `unit` is load-bearing: the accuracy references discriminate rows by
        # `quant_algo` / `spec_dec_algo`, so core and the `mtp` increment
        # anchor to DIFFERENT numbers.
        spec = spec or self.task_data
        return accuracy_checks.resolve_reference(
            spec, model_key=resolve_model_key(spec, unit), unit=unit
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
                current=self.workflow._target_marker(),
            )
        except OSError as exc:
            self.journal.event("runtime facts not written", f"{check.key}: {exc}")
            return
        if changes:
            self.journal.event("runtime facts changed", "; ".join(changes))

    # ----------------------------------------------------------------- checks
    def _prioritized_check(self, check: contract_mod.Check, log) -> Any:
        """:meth:`_run_contract_check`, its jobs queued at the check's class (:func:`job_priority`).

        The first unit's inner checks are on the way to its accuracy, and go
        ahead of every other check in a session's spool.
        """
        units = assembly_units(self.task_data)
        first = bool(units) and check.unit == units[0] and check.tier == contract_mod.TIER_INNER
        with job_priority(ACCURACY if first else FEATURE):
            return self._run_contract_check(check, log)

    def _run_contract_check(self, check: contract_mod.Check, log) -> Any:
        """Run one contract check and return its gate result.

        On a git checkout the check measures a snapshot of the commit the
        checkout is at as it starts (``_snapshots``). A check that passes adds
        the engines it attested to the runtime facts.
        """
        snapshots = getattr(self.workflow, "_snapshots", None)
        if snapshots is None:
            marker = self.workflow._target_marker()
            result = self._gate(check, log)
        else:
            result, marker = self._gate_on_snapshot(snapshots, check, log)
        if check.gate.startswith("accuracy[") and result.passed:
            with self._accuracy_lock:
                self._note_shift(check.unit, check.member, result, self._spec_for(check.unit)[0])
        self._note_runtime_facts(check, result, marker)
        return result

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
            marker=self.workflow._commit_marker(head),
            uncommitted=tuple(self.workflow._uncommitted_gate_inputs(repo) or ()),
            snapshots=snapshots,
        )
        try:
            with snapshots.lease(head):
                return self._on_snapshot(pin)._gate(check, log), pin.marker
        except snapshots_mod.SnapshotError as exc:
            result = snapshots_mod.void_result(check.gate, check.member, str(exc))
            self.journal.event("gate", result.summary())
            return result, pin.marker

    def _on_snapshot(self, pin: snapshots_mod.CheckTree) -> "CheckRunner":
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
        return view

    # ------------------------------------------------------------------ gates
    def _gate_call(self, check: contract_mod.Check) -> Callable[[Any], Any]:
        """``check``'s gate, as a function of the runner it launches with."""
        unit, member, gate = check.unit, check.member, check.gate
        spec, features = self._spec_for(unit)
        env = self._member_env(member)
        workdir = self.workspace / "gates" / unit

        def common(runner: Any) -> dict[str, Any]:
            return dict(task_data=spec, runner=runner, unit=unit, member=member)

        if gate == "boot":
            return lambda runner: boot_checks.run_boot_gate(
                env, features=features, workdir=workdir,
                spec=boot_checks.resolve_boot_spec(spec, unit, features),
                **common(runner),
            )  # fmt: skip
        if gate.startswith("accuracy["):
            if variants_mod.anchor_state(self.task_data, unit) == variants_mod.ANCHOR_RELATIVE:
                return lambda runner: self._relative_accuracy(check, env, spec, workdir, runner)
            return lambda runner: accuracy_checks.run_accuracy_gate(
                env, reference=self._reference_score(unit, spec), workdir=workdir, **common(runner)
            )
        if gate == "acceptance":
            return lambda runner: acceptance_checks.run_acceptance_gate(
                env, features=features, workdir=workdir,
                records=gate_records.records_for(self.workspace), **common(runner),
            )  # fmt: skip
        if gate == disagg.GATE_NAME:
            return lambda runner: disagg.run_disagg_gate(
                env, workdir=workdir, task_data=spec, runner=runner, unit=unit, member=member
            )
        if gate == serve.GATE_NAME:
            return lambda runner: serve.run_serve_gate(
                env, features=features, workdir=workdir,
                boot=boot_checks.resolve_boot_spec(spec, unit, features),
                **common(runner),
            )  # fmt: skip
        raise ValueError(f"contract check {check.key} names no gate this workflow runs")

    def _relative_accuracy(
        self, check: contract_mod.Check, env: RunEnvironment, spec: Any, workdir: Any, runner: Any
    ) -> "check_result.GateResult":
        """A variant's benchmark against its unit's own score at the same member, on this target.

        The unit's verdict on this target gives the score (the orchestrator runs
        that check first: :attr:`~..contract.Check.after`). Without one -- a
        change landed between the two -- the unit's benchmark runs here first,
        on the same snapshot and devices.
        """
        base = variants_mod.base_unit(self.task_data, check.unit)
        key = f"{base}:{check.gate}@{check.member}"
        marker = self.pin.marker if self.pin else self.workflow._target_marker()
        status = getattr(self.workflow, "_contract_status", None)
        holds = getattr(self.workflow, "_verdict_holds", None)
        verdict = contract_mod.verdict_on(status(), key, marker, holds) if status else None
        if verdict is not None:
            score, why = verdict.measured, verdict.detail
        else:
            first = self._gate_call(
                contract_mod.Check(base, check.gate, check.member, contract_mod.TIER_INNER)
            )(runner)
            self.journal.event("gate", f"{first.summary()} (measured as {check.unit}'s anchor)")
            if first.infrastructure:
                return first
            score, why = first.measured, first.detail
        if score is None:
            return check_result.GateResult(
                name=check.gate,
                member=check.member,
                passed=False,
                detail=f"{key} measured no score on this target ({why}), so {check.unit}, whose "
                "anchor is that score, has nothing to be judged against",
            )
        return accuracy_checks.run_accuracy_gate(
            env,
            reference=score,
            workdir=workdir,
            task_data=spec,
            runner=runner,
            unit=check.unit,
            member=check.member,
            relative=True,
        )

    def _gate_gpus(self, check: contract_mod.Check) -> str | None:
        """The devices ``check``'s job asks for when it is not one member's worth."""
        if check.gate != disagg.GATE_NAME:
            return None
        spec = disagg.DisaggSpec.from_task(self._spec_for(check.unit)[0])
        count = spec.device_count(check.member) if spec else None
        return str(count) if count else None

    def _gate(self, check: contract_mod.Check, log) -> Any:
        """Run ``check``'s gate on the target this workflow sees, journal it, and return its result."""
        call = self._gate_call(check)
        runner = self._member_runner(check.member, self._gate_gpus(check))
        result = self._gate_until_verdict(lambda: call(runner), log)
        self.gate_log.record(result)
        self.journal.event("gate", result.summary())
        if not result.passed:
            print_message(f"[bold yellow]↻ {escape_markup(result.summary())}[/bold yellow]", log)
            if check.gate == disagg.GATE_NAME and result.stdout_tail.strip():
                self.journal.section(
                    f"`{check.key}` failed: the test's last lines",
                    f"```\n{result.stdout_tail.rstrip()}\n```",
                )
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
            spec, unit, self.workflow._gate_env().world_size
        )
        if set(members) <= set(measured):
            finding = accuracy_checks.note_systematic_shift(
                gate_records.records_for(self.workspace),
                [measured[m] for m in members],
                spec.get("accuracy") or {},
                unit,
            )
            if finding is not None:
                self.journal.event("finding", finding.describe())
