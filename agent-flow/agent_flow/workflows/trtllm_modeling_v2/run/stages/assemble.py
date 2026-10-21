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
from ...infra.substrate.sessions import session_wait_from_task
from ...spec import variants as variants_mod
from ...spec.task_schema import (
    FEATURE_SKILLS,
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

        Returning anything wider gates a target against a feature it does
        not have yet, which either fails on the target's own fail-fast
        assert (the good case) or, if that assert is missing, passes while
        measuring something else entirely.
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
        cache = self.__dict__.setdefault("_member_runners", {})
        if want not in cache:
            executor = select_command_executor(
                env, tree=self.task_data.get("trtllm_repo_path"), gpus=want
            )
            cache[want] = self._noting_session_loss(shell_runner_for(executor))
        return cache[want]

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
        returns is not a verdict: it is not recorded, not handed to the
        assembler as a failure to fix, and costs no assembler turn. The gate
        runs again once a session heart-beats on the spool, after waiting up
        to `environment.session_wait`, at most `SESSION_LOSS_RERUNS` times in
        a row. Past either bound this raises `_SessionLost`, and the run
        closes `session-lost:<unit>`, which re-running resumes.

        A loss is what the gate says (`GateResult.infrastructure`, which
        `gates.py`'s gates set) or what the runner saw while it ran (see
        `_noting_session_loss`), so a gate that reports a lost command as an
        ordinary failure is not taken for one either. A gate that passed is a
        verdict whatever happened along the way.
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
        """Wait up to `environment.session_wait` for the gate session to heart-beat again.

        Which session that is, and whether there is one at all, is
        `execution.wait_for_session`'s to answer.
        """
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

        `gates.py`'s gates carry that reason on their verdict; a witness
        reports the same command as an ordinary failure. Noting it under every gate is what
        lets `_gate_until_verdict` tell those apart too.
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

        The schema rejects every one of these for a spec it validated, and a
        run reads its spec through the schema. This is the second line for a
        spec that changed underneath a resumed run, and it names the key,
        because "witness unconfigured" alone sends the operator to re-read the
        whole block.
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
            # Leg B is the flow's own (`checks/harness/disagg_leg_b.py`) unless the
            # block names a launcher; the run's own launcher needs the environment
            # that sets its legs apart.
            if block.get("contrast_command") and not block.get("contrast_env"):
                return "`contrast_env`"
            return None
        return None if block.get("contrast_env") else "`contrast_env`"

    def _run_witness(self, unit: str, member: str, log) -> "gates_mod.GateResult":
        """Run the witness gate `unit`'s feature needs, at `member`.

        The spec has to supply the one thing that cannot be derived: the
        environment that turns the feature OFF. A missing block is a hard
        failure rather than a skip, because skipping it would declare the unit
        green on two gates that are blind to its feature by construction --
        which is the exact outcome the witness exists to prevent.
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
            # A token comparison's tail is both legs around where each
            # differing continuation first differs; a leg that died leaves its
            # last lines. The one-line summary the gate history carries quotes
            # only the first difference, so the rest goes to the journal and
            # the run's log.
            self.journal.section(
                f"witness `{result.name}` failed: evidence",
                f"```\n{result.stdout_tail.rstrip()}\n```",
            )
            print_message(escape_markup(result.stdout_tail.rstrip()), log)
        return result

    def _assembler_prompt(self, unit: str) -> str:
        if variants_mod.is_variant(self.task_data, unit):
            return self._variant_prompt(unit)
        feature_skills = ""
        if unit != "core":
            feature_skills += (
                f" and `{FEATURE_SKILLS.get(unit, 'trtllm-modeling-v2-core')}` for the "
                f"`{unit}` feature"
            )
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Unit: **{unit}**\n\n"
            f"Load `trtllm-modeling-v2-core`{feature_skills} before anything else.\n\n"
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
        feature = f" and `{FEATURE_SKILLS[base]}`" if base in FEATURE_SKILLS else ""
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Unit: **{base}**, configuration variant **{name}**\n\n"
            f"Load `trtllm-modeling-v2-core`{feature} before anything else.\n\n"
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

    def _reviewer_prompt(self, unit: str) -> str:
        """A unit review's dispatch: the unit, its gate evidence and the assembler's report."""
        return (
            f"Workspace: {self.workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n"
            f"Unit: **{unit}**\n\n"
            "This is a **unit review**.\n\n"
            f"{self._gate_history_block()}"
            f"Read `{self.progress_path}` for the assembler's report -- its derivation "
            "table and vocabulary audit are what you check against the diff.\n\n"
            "Read the diff and decide. The orchestrator runs every gate; you do not.\n\n"
            + self._report_instruction("reviewer")
        )

    def _regression_review_prompt(self, covers: list[str], own: tuple[str, ...]) -> str:
        """A regression review's dispatch: the changes it covers, and what it runs, in order.

        ``covers`` are the landings (``<task>@<attempt>``) whose changes to
        shared code the review takes, and ``own`` the run's own paths, outside
        which a change is shared.
        """
        cli = self._cli()
        workspace = self.workspace
        changes = ", ".join(covers) or "every change to shared code since the last review"
        mine = ", ".join(f"`{p}`" for p in own) or "the target's directories"
        return (
            f"Workspace: {workspace}\n"
            f"Repository: {self.task_data['trtllm_repo_path']}\n"
            f"Target: {self._preflight.target_rel}\n\n"
            "This is a **regression review**. It covers the changes to shared code these "
            f"tasks landed: {changes}. A shared change is one outside the run's own paths "
            f"({mine}); `git diff --name-only <base>..HEAD` lists them. Run, in order:\n\n"
            f"1. `{cli} tests affected --workspace {workspace} --head <HEAD> --paths <the unit "
            "test directories of the other models that run those paths>`. Its base is the "
            "commit the last regression review cleared, else the run's starting commit. It "
            "lists the tests no recorded result holds for at the head, with why, and prints "
            "the command that runs them.\n"
            "2. That command, in a checkout at the head with the tree's python, as one GPU job "
            "on as many devices as the last one; with nothing to run, its command without a "
            "selection records the head's environment.\n"
            f"3. `{cli} tests compare`, same arguments: `CLEAN`, `REGRESSED` (each test that "
            "passed at the base and fails now, or is new and fails) or `INCOMPLETE` (what is "
            "still to run, at the head or, for a failure with no base result, at the base).\n"
            "4. Only where a changed path is engine code other models' forwards run (the "
            "executor, the runtime, a module their forward calls): boot one such model at "
            "tp1 on the head. Boot or test the base tree only for what fails at the head: "
            "`REGRESSED`, or `INCOMPLETE` asking for the base.\n\n"
            "Run nothing else of the suite, and not the contract gates.\n\n"
            + self._report_instruction("reviewer")
        )

    def _model_key(self, unit: str | None = None) -> str:
        """The HuggingFace repo name the accuracy references are keyed by.

        Delegated to :func:`~.preflight.resolve_model_key` rather than
        restated. Preflight has to ask this same question hours earlier, to
        check the anchor row exists at all, and **an anchor check that
        resolves a different key than the gate does is worse than no check**
        — it passes on a row nothing ever reads. One implementation is the
        only way those two cannot drift. It also honours the per-unit
        mapping form of `model_key`, for a feature increment anchored under
        another repo name.
        """
        return resolve_model_key(self.task_data, unit)

    def _reference_score(self, unit: str | None = None, spec: Any = None) -> float:
        # `unit` is load-bearing: the accuracy references discriminate rows by
        # `quant_algo` / `spec_dec_algo`, so core and the `mtp` increment
        # anchor to DIFFERENT numbers. Resolving without it either picks the
        # undiscriminated (bf16) row for a quantized checkpoint or raises at
        # the end of an assembly instead of at spec-validation time.
        spec = spec or self.task_data
        return gates_mod.resolve_reference(spec, model_key=resolve_model_key(spec, unit), unit=unit)

    def _gate_history_block(self) -> str:
        rendered = self.gate_log.render()
        if not rendered:
            return ""
        return f"Gates so far:\n\n```\n{rendered}\n```\n\n"
