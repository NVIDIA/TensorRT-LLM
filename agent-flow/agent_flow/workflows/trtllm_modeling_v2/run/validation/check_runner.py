# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework check execution: member runners, snapshot provenance and gates."""

from __future__ import annotations

import copy
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from rich.markup import escape as escape_markup

from agent_flow.console import print_message

from ...checks import accuracy as accuracy_checks
from ...checks import boot as boot_checks
from ...checks import check_config, disagg, e2e, serve, user_tests
from ...checks import verdict as check_result
from ...infra.compute.environment import RunEnvironment
from ...infra.compute.executor import (
    ACCURACY,
    CANCELLED,
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
from ...spec import verification as verification_mod
from ...spec.task_schema import (
    assembly_units,
    features_through,
    parse_parallel_segment,
    resolve_per_unit,
)
from ...spec.verification import DATASET_ACCEPTANCE
from ..startup import datasets as datasets_mod
from ..workspace import snapshots as snapshots_mod
from . import anchors as anchors_mod
from . import contract as contract_mod

#: How many times in a row one gate may be lost to its session and re-run.
#: Each re-run waits for a heartbeat first, so this only binds on a session
#: that keeps coming back and dying again.
SESSION_LOSS_RERUNS = 3


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
        self._session = threading.local()
        self.pin = None
        self._tests: dict[tuple[str, str], verification_mod.Test] | None = None

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
        """The run's environment at ``member``'s own world size, on the nodes it spans.

        Members of one unified target may differ in width (tp1 beside tep4)
        when a scheduler hands each gate job its own devices.
        """
        env = self.workflow._gate_env()
        size = parse_parallel_segment(member)[1] if member else None
        return env.at_width(size) if size else env

    def _member_runner(
        self, member: str | None, gpus: str | None = None, *, nodes: int = 1, lead: bool = False
    ) -> Any:
        """The gate runner for ``member``: under a scheduler, a job asking for its world size.

        ``gpus`` overrides the request, for a check that starts more engines
        than one member's worth (the disaggregated-serving test's two). ``lead``:
        the session's lead runs the job.
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
        key = (want, priority, nodes, *(("lead",) if lead else ()))
        if key not in cache:
            executor = select_command_executor(
                env,
                tree=self.task_data.get("trtllm_repo_path"),
                gpus=want,
                priority=priority,
                nodes=nodes,
                lead=lead,
                note=lambda said: self.journal.event("waiting", said),
            )
            cache[key] = self._noting_session_loss(shell_runner_for(executor))
        return cache[key]

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

        A gate whose job was cancelled through the spool measured nothing either,
        but its session is fine: it raises `_SessionLost` at once, so the check
        has no verdict (it is missing, not failed) and runs again later, as a
        re-check, not straight back into the queue it was taken out of.
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
            label = f"{result.name}@{result.member}" if result.member else result.name
            if CANCELLED in reason:
                self.journal.event("gate cancelled", f"{label}: {reason}")
                raise _SessionLost(f"{label} measured nothing: {reason}")
            lost_in_a_row += 1
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

    # ----------------------------------------------------------------- checks
    def _prioritized_check(self, check: contract_mod.Check, log) -> Any:
        """:meth:`_run_contract_check`, its jobs queued at the check's class (:func:`job_priority`).

        The first unit's own landing checks of the flow's tests are on the way to
        its accuracy, and go ahead of every other check in a session's spool. The
        spec's own tests do not.
        """
        units = assembly_units(self.task_data)
        first = (
            bool(units)
            and check.unit == units[0]
            and check.stage == contract_mod.STAGE_LANDING
            and check.origin != verification_mod.ORIGIN_SPEC
        )
        with job_priority(ACCURACY if first else FEATURE):
            return self._run_contract_check(check, log)

    def _run_contract_check(self, check: contract_mod.Check, log) -> Any:
        """Run one contract check and return its gate result.

        On a git checkout the check measures a snapshot of the commit the
        checkout is at as it starts (``_snapshots``).
        """
        snapshots = getattr(self.workflow, "_snapshots", None)
        if snapshots is None:
            return self._gate(check, log)
        return self._gate_on_snapshot(snapshots, check, log)

    def _gate_on_snapshot(
        self, snapshots: snapshots_mod.Snapshots, check: contract_mod.Check, log
    ) -> Any:
        """``check``'s result on a snapshot of the checkout's head."""
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        head = resolve_commit(repo)
        if head is None:
            why = f"git could not name the commit {repo} is at, so there was nothing to snapshot"
            return snapshots_mod.void_result(check.gate, check.member, why)
        pin = snapshots_mod.CheckTree(
            commit=head,
            tree=snapshots.tree(head),
            marker=self.workflow._commit_marker(head),
            uncommitted=tuple(self.workflow._uncommitted_gate_inputs(repo) or ()),
            snapshots=snapshots,
        )
        try:
            with snapshots.lease(head):
                return self._on_snapshot(pin)._gate(check, log)
        except snapshots_mod.SnapshotError as exc:
            result = snapshots_mod.void_result(check.gate, check.member, str(exc))
            self.journal.event("gate", result.summary())
            return result

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
    def _run_tests(self) -> dict[tuple[str, str], verification_mod.Test]:
        """The run's tests by ``(feature, name)`` (``contract.run_tests``), read once."""
        if self._tests is None:
            world = self.workflow._gate_env().world_size
            self._tests = {
                (t.feature, t.name): t for t in contract_mod.run_tests(self.task_data, world)
            }
        return self._tests

    def _test_of(self, check: contract_mod.Check) -> verification_mod.Test:
        test = self._run_tests().get((check.unit, check.gate))
        if test is None:
            raise ValueError(f"contract check {check.key} names no test of this run")
        return test

    def _gate_call(self, check: contract_mod.Check) -> Callable[[Any], Any]:
        """``check``'s test, as a function of the runner it launches with.

        Its result is named by the test, whatever the runner calls it.
        """
        call = self._test_call(check)
        return lambda runner: _named(call(runner), check.gate)

    def _test_call(self, check: contract_mod.Check) -> Callable[[Any], Any]:
        unit, member = check.unit, check.member
        test = self._test_of(check)
        spec, features = self._spec_for(unit)
        env = self._member_env(member)
        workdir = self.workspace / "gates" / unit
        if test.runner == verification_mod.RUN_DATASETS:
            return lambda runner: self._datasets(check, test, env, spec, workdir, runner)
        if test.runner == verification_mod.RUN_DISAGG:
            return lambda runner: disagg.run_disagg_gate(
                env, workdir=workdir, task_data=self._with_dataset(spec, unit),
                runner=runner, unit=unit, member=member,
            )  # fmt: skip
        if test.runner == verification_mod.RUN_SERVE:
            return lambda runner: serve.run_serve_gate(
                env, features=features, workdir=workdir,
                boot=boot_checks.resolve_boot_spec(spec, unit, features),
                task_data=spec, runner=runner, unit=unit, member=member,
            )  # fmt: skip
        return lambda runner: user_tests.run_user_test(
            env, task_data=spec, test=test, workdir=workdir, runner=runner,
            member=member, features=features,
        )  # fmt: skip

    def _with_dataset(self, spec: Any, unit: str | None = None) -> Any:
        """``spec`` with the staged dataset the disaggregated gate reads (``startup/datasets.py``), once ready.

        A stated deployment's benchmark gets it as ``accuracy.dataset_path``
        (``trtllm-eval --dataset_path``); the disaggregated test, which reads it under
        ``$LLM_MODELS_ROOT``, a models root that holds it. A dataset still being staged is
        waited for; one that could not be is left out, and the gate runs as it would
        without it.
        """
        accuracy = dict(spec.get("accuracy") or {})
        block = dict(accuracy.get("disagg") or {})
        benchmark = "gsm8k"
        if not block.get("test"):
            benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or benchmark)
        if benchmark not in datasets_mod.status(self.workspace):
            return spec
        path = datasets_mod.wait(self.workspace, benchmark)
        if not path:
            return spec
        if block.get("test"):
            env = dict(block.get("env") or {})
            outer = (spec.get("environment") or {}).get("env") or {}
            root = env.get("LLM_MODELS_ROOT") or outer.get("LLM_MODELS_ROOT") or ""
            staged = datasets_mod.models_root(self.workspace, str(root), benchmark, path)
            if not staged or staged == root:
                return spec
            accuracy["disagg"] = {**block, "env": {**env, "LLM_MODELS_ROOT": staged}}
        elif not accuracy.get("dataset_path"):
            accuracy["dataset_path"] = path
        else:
            return spec
        return {**spec, "accuracy": accuracy}

    def _models_root_env(self, spec: Any, datasets: Sequence[str]) -> dict[str, str]:
        """``LLM_MODELS_ROOT`` for a test that reads ``datasets``: a root that holds them.

        Where the spec's models root lacks a staged dataset (``startup/datasets.py``),
        a root of links to its entries with the staged copy beside them stands in for
        it. Empty when the spec's root serves as it is.
        """
        staged = datasets_mod.status(self.workspace)
        outer = (spec.get("environment") or {}).get("env") or {}
        root = str(outer.get(e2e.MODELS_ROOT_ENV) or "")
        chosen = root
        for name in datasets:
            if name not in staged or name not in datasets_mod.KNOWN:
                continue
            path = datasets_mod.wait(self.workspace, name)
            if path:
                chosen = datasets_mod.models_root(self.workspace, chosen, name, path) or chosen
        return {e2e.MODELS_ROOT_ENV: chosen} if chosen and chosen != root else {}

    def _reference_row(self, unit: str, spec: Any, dataset: str) -> float | None:
        """The score ``dataset``'s references row gives ``unit``, as the flow reads it; ``None`` if none.

        The row ``accuracy_core`` matches: by the spec's discriminators and the
        engine's speculative decoding (``checks/accuracy.reference_discriminators``).
        """
        path = Path(str(spec["trtllm_repo_path"])) / e2e.REFERENCES_DIR / f"{dataset}.yaml"
        try:
            return accuracy_checks.load_reference_score(
                path,
                accuracy_checks.resolve_model_key(spec, unit),
                discriminators=accuracy_checks.reference_discriminators(spec, unit),
            )
        except (KeyError, OSError, TypeError, ValueError):
            return None

    def _datasets(
        self,
        check: contract_mod.Check,
        test: verification_mod.Test,
        env: RunEnvironment,
        spec: Any,
        workdir: Any,
        runner: Any,
    ) -> "check_result.GateResult":
        """Run ``check``'s case of the target's generated e2e test, judged against its anchors.

        A benchmark is judged against its references row, by the tree's runner,
        but for two anchors the flow keeps (``validation/anchors.py``):

        - a row that is a low placeholder (:data:`anchors.PLAUSIBLE_ANCHOR`): the
          first score the feature measured on it at this member, once it has
          one; the first pass keeps its score as that anchor;
        - a configuration variant whose anchor is relative: its unit's score at
          the same member on this target, for the benchmark its unit's ``e2e``
          scores first, under the same protocol (``contract.relative_benchmark``).
          The unit's verdict on this target gives it (the orchestrator runs that
          check first: :attr:`~..contract.Check.after`); without one the unit's
          check runs here first, on the same snapshot and devices.
        """
        unit, member = check.unit, check.member
        references: dict[str, float] = {}
        said: list[str] = []
        placeholders: list[str] = []
        relative = contract_mod.relative_benchmark(self.task_data, test, self._run_tests().values())
        for dataset in test.benchmarks:
            if relative and dataset == relative:
                score, why = self._unit_score(check, runner)
                if isinstance(score, check_result.GateResult):
                    return score
                if score is None:
                    return check_result.GateResult(
                        name=check.gate,
                        member=member,
                        passed=False,
                        detail=f"{why}, so {unit}, whose {dataset} anchor is that score, has "
                        "nothing to be judged against",
                    )
                references[dataset] = score
                said.append(f"{dataset} judged against {why}")
                continue
            row = self._reference_row(unit, spec, dataset)
            plausible = anchors_mod.PLAUSIBLE_ANCHOR.get(dataset)
            if row is None or plausible is None or row >= plausible:
                continue
            own = anchors_mod.measured(self.workspace, unit, f"{check.gate}[{dataset}]", member)
            if own is None:
                placeholders.append(dataset)
                continue
            references[dataset] = own.score
            said.append(
                f"{dataset} anchored to {unit}'s own first measured score {own.score:.2f} "
                f"(tree {own.tree}), which replaced the placeholder {row:g}"
            )
        preflight = self.workflow._preflight
        node = e2e.node_id(preflight.family, preflight.target_rel, unit, check.gate, member)
        reads = [
            *test.benchmarks,
            *([DATASET_ACCEPTANCE] if DATASET_ACCEPTANCE in test.datasets else []),
        ]
        result, lines = e2e.run_e2e(
            env,
            task_data=spec,
            test=test,
            member=member,
            node=node,
            workdir=workdir,
            runner=runner,
            references=references,
            env=self._models_root_env(
                spec, ["gsm8k" if d == DATASET_ACCEPTANCE else d for d in reads]
            ),
        )
        for dataset in placeholders:
            line = next((x for x in lines if x.get("dataset") == dataset), None)
            score = line.get("score") if line else None
            if (
                line is not None
                and line.get("passed")
                and isinstance(score, (int, float))
                and not result.infrastructure
                and anchors_mod.record(
                    self.workspace,
                    unit,
                    f"{check.gate}[{dataset}]",
                    member,
                    float(score),
                    str(result.tree_state or ""),
                )
            ):
                self.journal.event(
                    "anchor",
                    f"{check.key}: measured {dataset} {score:.2f} on a placeholder row; that "
                    f"score anchors {unit}'s later {dataset} at {member}",
                )
        if said:
            result = replace(result, detail=f"{result.detail}; " + "; ".join(said), explained=True)
        return result

    def _unit_score(self, check: contract_mod.Check, runner: Any) -> tuple[Any, str]:
        """A relative variant's anchor: its unit's benchmark score at the same member, on this target.

        The score and where it came from; a lost run's result in place of the score.
        """
        base = variants_mod.base_unit(self.task_data, check.unit)
        key = check.after or f"{base}:{verification_mod.E2E}@{check.member}"
        marker = self.pin.marker if self.pin else self.workflow._target_marker()
        status = getattr(self.workflow, "_contract_status", None)
        holds = getattr(self.workflow, "_verdict_holds", None)
        verdict = contract_mod.verdict_on(status(), key, marker, holds) if status else None
        if verdict is not None:
            score, why = verdict.measured, verdict.detail
        else:
            unit, _, rest = key.partition(":")
            gate, _, member = rest.partition("@")
            if (unit, gate) not in self._run_tests():
                return None, f"{base} owes no benchmark at {check.member}"
            first = self._gate_call(
                contract_mod.Check(unit, gate, member, contract_mod.STAGE_LANDING, scores=True)
            )(runner)
            self.journal.event("gate", f"{first.summary()} (measured as {check.unit}'s anchor)")
            if first.infrastructure:
                return first, ""
            score, why = first.measured, first.detail
        if score is None:
            return None, f"{key} measured no score on this target ({why})"
        return score, f"{key}'s score {score:.2f} on this target"

    def _gate_gpus(self, check: contract_mod.Check) -> str | None:
        """The devices ``check``'s job asks for when it is not one member's worth."""
        if check.runner != verification_mod.RUN_DISAGG:
            return None
        spec = disagg.DisaggSpec.from_task(self._spec_for(check.unit)[0])
        count = spec.device_count(check.member) if spec else None
        return str(count) if count else None

    def _gate(self, check: contract_mod.Check, log) -> Any:
        """Run ``check``'s test on the target this workflow sees, journal it, and return its result."""
        call = self._gate_call(check)
        nodes, lead = disagg.job_span(
            self._spec_for(check.unit)[0],
            disagg.GATE_NAME if check.runner == verification_mod.RUN_DISAGG else check.gate,
            self._member_env(check.member).nodes,
        )
        runner = self._member_runner(check.member, self._gate_gpus(check), nodes=nodes, lead=lead)
        result = self._gate_until_verdict(lambda: call(runner), log)
        self.gate_log.record(result)
        self.journal.event("gate", result.summary())
        if not result.passed:
            print_message(f"[bold yellow]↻ {escape_markup(result.summary())}[/bold yellow]", log)
            if check.runner != verification_mod.RUN_SERVE and result.stdout_tail.strip():
                self.journal.section(
                    f"`{check.key}` failed: the test's last lines",
                    f"```\n{result.stdout_tail.rstrip()}\n```",
                )
        return result


def _named(result: Any, name: str) -> Any:
    """``result`` named ``name``: a check's verdict carries its test's name."""
    return result if getattr(result, "name", name) == name else replace(result, name=name)
