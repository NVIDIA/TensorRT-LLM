# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A unit's checks and prompts: each member's runner, its gates and witnesses, and what its workers are asked."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rich.markup import escape as escape_markup

from agent_flow.console import print_message

from ...checks import gate_cli
from ...checks import gates as gates_mod
from ...checks import witness as witness_mod
from ...infra.environment import RunEnvironment
from ...infra.execution import select_command_executor, shell_runner_for, wait_for_session
from ...infra.execution.priority import current_job_priority
from ...infra.substrate.sessions import session_wait_from_task
from ...spec import variants as variants_mod
from ...spec.task_schema import (
    EQUIVALENCE_WITNESS_REQUIRED,
    REPLAY_WITNESS_REQUIRED,
    assembly_units,
    parse_parallel_segment,
)
from ..preflight import resolve_model_key
from .common import SESSION_LOSS_RERUNS, WITNESS_GATED_UNITS, _SessionLost


class AssembleStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    def _features_through(self, unit: str) -> list[str]:
        """Features implemented once ``unit`` is assembled, in order.

        Units are built one at a time and each is gated before the next
        begins, so the target under test after unit *k* implements
        ``features[:k]`` — never the whole list. ``core`` implements none.
        """
        units = assembly_units(self.task_data)
        if unit not in units:
            return [str(f) for f in self.task_data.get("features", [])]
        return [str(u) for u in units[1 : units.index(unit) + 1]]

    def _member_env(self, member: str | None) -> RunEnvironment:
        """The run's environment at ``member``'s own world size.

        Members of one unified target may differ in width (tp1 beside tep4)
        when a scheduler hands each gate job its own devices.
        """
        env = self._gate_env()
        size = parse_parallel_segment(member)[1] if member else None
        return replace(env, world_size=size) if size else env

    def _member_runner(self, member: str | None, gpus: str | None = None) -> Any:
        """The gate runner for ``member``: under a scheduler, a job asking for its world size.

        ``gpus`` overrides the request, for a check that starts more engines
        than one member's worth (the equivalence witness's leg B).
        """
        env = self._member_env(member)
        if self._shell_runner_override is not None:
            return self._shell_runner
        want = gpus or str(env.world_size)
        # A session's spool starts a check of a higher class first (priority.py).
        priority = current_job_priority()
        cache = self.__dict__.setdefault("_member_runners", {})
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
        spec = variants_mod.variant_spec(
            self.task_data,
            unit,
            world_size=self._gate_env().world_size,
            options_dir=self.workspace / variants_mod.OPTIONS_DIRNAME,
        )
        return spec, self._features_through(variants_mod.base_unit(self.task_data, unit))

    def _gate_until_verdict(self, run, log) -> "gates_mod.GateResult":
        """Run one gate until it says something about the target, and return that.

        A gate whose session was lost under it -- an allocation requeued, or
        ended before its successor started -- measured nothing, so what it
        returns is not a verdict: it is not recorded and costs no assembler
        turn. The gate runs again once a session heart-beats on the spool,
        after waiting up to `environment.session_wait`, at most
        `SESSION_LOSS_RERUNS` times in a row. Past either bound this raises
        `_SessionLost`, and the run closes `session-lost:<unit>`, which
        re-running resumes.

        A loss is what the gate says (`GateResult.infrastructure`) or what the
        runner saw while it ran (`_noting_session_loss`). A gate that passed is
        a verdict whatever happened along the way.
        """
        lost_in_a_row = 0
        while True:
            seen = len(self._lost_commands)
            result = run()
            during = self._lost_commands[seen:]
            reason = result.infrastructure or (during[-1] if during else "")
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
        back = wait_for_session(self._command_executor, limit)
        self.journal.event(
            "session wait",
            f"{'a session is back' if back else 'no session came back'} "
            f"(waited up to {limit:.0f}s)",
        )
        return back

    def _noting_session_loss(self, runner: Any) -> Any:
        """``runner``, remembering why each command it ran was ended by its session.

        A witness reports such a command as an ordinary failure; the note lets
        `_gate_until_verdict` tell the two apart.
        """

        def run(*args: Any, **kwargs: Any) -> Any:
            result = runner(*args, **kwargs)
            if result.infrastructure:
                self._lost_commands.append(result.infrastructure)
            return result

        return run

    @staticmethod
    def _witness_block_gap(unit: str, kind: str, block: Any) -> str | None:
        """What `accuracy.witness.<unit>` lacks for its witness, or None.

        The schema rejects every one of these; this names the key for a spec
        that changed underneath a resumed run.
        """
        if not isinstance(block, dict):
            return "a block at all"
        if kind == "replay":
            absent = [key for key in REPLAY_WITNESS_REQUIRED[unit] if not block.get(key)]
            return ", ".join(f"`{key}`" for key in absent) or None
        if kind == "kv_quant":
            if not block.get("expected_bytes_per_token"):
                return "`expected_bytes_per_token`"
            # `exact: false` drops the identity leg, and with it the contrast.
            if block.get("exact", True) and not block.get("contrast_env"):
                return "`contrast_env`"
            return None
        if kind == "equivalence":
            # Leg B is the run's own launcher, under the environment that sets its legs apart.
            absent = [key for key in EQUIVALENCE_WITNESS_REQUIRED if not block.get(key)]
            return ", ".join(f"`{key}`" for key in absent) or None
        return None if block.get("contrast_env") else "`contrast_env`"

    def _run_witness(self, unit: str, member: str, log) -> "gates_mod.GateResult":
        """Run the witness gate `unit`'s feature needs, at `member`.

        A missing `accuracy.witness` block fails it rather than skipping it:
        boot and accuracy are blind to the feature by construction.
        """
        kind = WITNESS_GATED_UNITS[unit]
        spec = (self.task_data.get("accuracy") or {}).get("witness") or {}
        block = spec.get(unit) if isinstance(spec, dict) else None
        missing = self._witness_block_gap(unit, kind, block)
        if missing:
            return gates_mod.GateResult(
                name=kind,
                member=member,
                passed=False,
                detail=(
                    f"`accuracy.witness.{unit}` does not set {missing}; without it the "
                    f"`{unit}` unit is gated only by boot and accuracy, both of which pass "
                    f"whether or not the feature is doing anything"
                ),
            )

        # The same entry point a standalone run uses (`trtllm-modeling-v2 gate
        # witness`), over the workload the block declares.
        result = self._gate_until_verdict(
            lambda: witness_mod.run_declared_witness(
                self._member_env(member),
                kind=kind,
                unit=unit,
                member=member,
                task_data=self.task_data,
                # Leg B starts a context and a generation engine: twice the member.
                runner=self._member_runner(member, "all" if kind == "equivalence" else None),
                features=self._features_through(unit),
                workdir=self.workspace / "witness" / unit,
                boot=gates_mod.resolve_boot_spec(self.task_data, unit, None, self.boot_spec_path),
            ),
            log,
        )
        self.gate_log.record(result)
        self.journal.event("gate", result.summary())
        if not result.passed and result.stdout_tail.strip():
            # The summary quotes only the first difference; the tail (both legs
            # around each difference, or a dead leg's last lines) goes to the
            # journal and the run's log.
            self.journal.section(
                f"witness `{result.name}` failed: evidence",
                f"```\n{result.stdout_tail.rstrip()}\n```",
            )
            print_message(escape_markup(result.stdout_tail.rstrip()), log)
        return result

    def _assembler_prompt(self, unit: str) -> str:
        if variants_mod.is_variant(self.task_data, unit):
            return self._variant_prompt(unit)
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Unit: **{unit}**\n\n"
            f"Read `{self.task_path}` (the spec), `{self.plan_path}` (the plan) and "
            f"`{self.vocabulary_path}` (the computation-step list you derive against). "
            f"Earlier turns on this unit, if any, reported in `{self.progress_path}`: build "
            f"on them rather than starting over.\n\n"
            f"{self._gate_history_block()}"
            f"Write the unit. To reproduce one gate or witness while debugging, run "
            f"the workflow's own entry point, never a reconstruction of its command: "
            f"`{gate_cli.invocation(self.task_path, self.workspace / 'gates', unit, self._cli())}`.\n\n"
            + self._report_instruction("assembler")
        )

    def _cli(self) -> str:
        """The workflow's CLI as an agent calls it: by path, when preflight installed it."""
        script = self.workspace / "bin" / gate_cli.CLI_NAME
        return str(script) if script.is_file() else gate_cli.CLI_NAME

    def _variant_prompt(self, name: str) -> str:
        """An assembler turn on a configuration variant: the unit's target, served another way."""
        base = variants_mod.base_unit(self.task_data, name)
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Unit: **{base}**, configuration variant **{name}**\n\n"
            f"{variants_mod.describe(self.task_data, name)}\n\n"
            f"The `{base}` target exists. Make it serve this configuration as well, without "
            f"changing what it does in its own: every `{base}` check must keep passing. The "
            f"variant's checks run with `{base}`'s settings plus the overlay above. Read "
            f"`{self.task_path}` (the spec) and `{self.progress_path}` for earlier turns.\n\n"
            f"{self._gate_history_block()}"
            f"Reproduce one of its checks with "
            f"`{gate_cli.invocation(self.task_path, self.workspace / 'gates', name, self._cli())}`.\n\n"
            + self._report_instruction("assembler")
        )

    def _model_key(self, unit: str | None = None) -> str:
        """The HuggingFace repo name the accuracy references are keyed by.

        :func:`~.preflight.resolve_model_key`, the one preflight's anchor check
        asks too, so the two cannot resolve different keys.
        """
        return resolve_model_key(self.task_data, unit)

    def _reference_score(self, unit: str | None = None, spec: Any = None) -> float:
        # `unit` is load-bearing: the accuracy references discriminate rows by
        # `quant_algo` / `spec_dec_algo`, so core and the `mtp` increment
        # anchor to DIFFERENT numbers.
        spec = spec or self.task_data
        return gates_mod.resolve_reference(spec, model_key=resolve_model_key(spec, unit), unit=unit)

    def _gate_history_block(self) -> str:
        rendered = self.gate_log.render()
        if not rendered:
            return ""
        return f"Gates so far:\n\n```\n{rendered}\n```\n\n"
