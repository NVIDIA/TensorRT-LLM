# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight: the checks, the environment, the agent probe, host facts and the surface survey.

Toward a planned run the substrate gate, the one check that boots a GPU, runs
beside the run's first turns instead of before them (phase 0), and a failure
stops the run through the stop it was handed.

A run derived from a checkpoint (``--checkpoint``) has its intake checks here
first: every block of the request and of the spec compiled from it is
reported at once, and stops the run before anything is spent. Its
confirmation is recorded in the journal, not awaited.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule
from agent_flow.prompts import dump_prompt_bundle

from ...agents.prompts import PROMPTS_DIRNAME, build_trtllm_modeling_v2_prompts
from ...checks import gate_cli
from ...checks import gates as gates_mod
from ...infra.committer import CommitPolicy, install_committer
from ...infra.environment import RunEnvironment, image_import_warning, shares_allocation_with_spool
from ...infra.execution import (
    TURNS_DIRNAME,
    InProcessTurnExecutor,
    select_command_executor,
    shell_runner_for,
)
from ...infra.substrate.scheduler import read_json, write_json_atomic
from ...onboard.embargo import display_entries
from ...onboard.runtime_map import (
    RUNTIME_MAP_FILENAME,
    areas_for_demands,
    format_areas,
    format_kernel_sources,
    load_index,
    load_kernel_sources,
    unmatched_demands,
)
from ...onboard.survey import (
    format_surface,
    read_checkpoint_surface,
    run_inventory_probe,
    write_surface_file,
)
from ...spec.intake.checks import BLOCK, WARN, render_findings
from ...spec.intake.freeze import confirmation_status, read_confirmation
from ...spec.task_schema import assembly_units
from .. import agent_probe
from .. import contract as contract_mod
from .. import requirements as requirements_mod
from ..preflight import (
    MAX_TMPDIR_LEN,
    PreflightError,
    PreflightResult,
    SpecInconsistencyError,
    check_intake_request,
    check_processor_dependencies,
    format_preflight_report,
    run_preflight,
    substrate_gate,
)
from ..state import WorkflowState
from .common import HOST_FACTS_APPEND_MARKER, TROUBLESHOOTING_HINT

#: Where the substrate gate's last pass is recorded, under the workspace.
SUBSTRATE_RECORD = Path("gates") / "substrate" / "pass.json"
#: How long a recorded substrate pass spares a resumed run the gate, in seconds.
SUBSTRATE_FRESH_S = 12 * 3600.0


def substrate_inputs(task_data: Mapping[str, Any], environment: RunEnvironment) -> dict[str, str]:
    """What a substrate pass proves something about, one digest per part.

    - ``environment``: the launch environment as preflight resolved it, and
      ``spec_environment``, the block it was resolved from, which also names
      the proxy checkpoint;
    - ``engine``: what the boot's command line is built from -- the
      checkpoint, the parallel segment and the runtime block -- and
      ``options``, the options file it reads, by content;
    - ``checkout``: the checkout it ran in, by path.

    Not by the checkout's content: the tree changes with every integration,
    and every gate of the run measures it on this same substrate under the
    tree's own marker.
    """
    accuracy = task_data.get("accuracy") or {}
    options = gates_mod.resolve_llm_api_options(accuracy, None)
    options_text = None
    if options and Path(str(options)).is_file():
        options_text = Path(str(options)).read_text(encoding="utf-8", errors="replace")
    parts: dict[str, Any] = {
        "environment": dataclasses.asdict(environment),
        "spec_environment": task_data.get("environment"),
        "engine": {
            "checkpoint_path": task_data.get("checkpoint_path"),
            "parallel": task_data.get("parallel"),
            "runtime": gates_mod.resolve_runtime(task_data, None),
        },
        "options": [options, options_text],
        "checkout": task_data.get("trtllm_repo_path"),
    }
    return {name: _digest(value) for name, value in parts.items()}


def substrate_reuse(
    record: Mapping[str, Any] | None,
    inputs: Mapping[str, str],
    *,
    now: float,
    fresh_s: float = SUBSTRATE_FRESH_S,
) -> tuple[bool, str]:
    """Whether a recorded substrate pass stands for ``inputs`` at ``now``, and why.

    It stands when it was recorded for the same :func:`substrate_inputs`
    less than ``fresh_s`` ago. The window bounds what the inputs cannot see:
    an image re-imported under the same name, a package installed into the
    same interpreter, a driver update.
    """
    if not record:
        return False, "no pass is recorded"
    recorded = record.get("inputs") or {}
    changed = sorted(
        name for name in {*inputs, *recorded} if inputs.get(name) != recorded.get(name)
    )
    if changed:
        verb = "differs" if len(changed) == 1 else "differ"
        return False, f"the {', '.join(changed)} {verb} from the last pass's"
    try:
        age = now - float(record.get("at"))
    except (TypeError, ValueError):
        return False, "the last pass has no time"
    if not 0 <= age < fresh_s:
        return (
            False,
            f"the last pass is {age / 3600:.1f} h old, past the {fresh_s / 3600:g} h window",
        )
    return True, f"passed {age / 3600:.1f} h ago in this environment"


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class PreflightStage:
    """A stage of :class:`~..workflow.TrtllmModelingV2Workflow`; see the module docstring."""

    #: The request a ``--checkpoint`` run was derived from, whose intake
    #: checks phase 0 runs; None for a run given its spec.
    intake_request: Path | None = None

    def _run_preflight(
        self, state: WorkflowState, log, *, stop: Callable[[str], None] | None = None
    ) -> bool:
        """Resolve and prove what the run needs before its first turn; False stops the run.

        With ``stop`` the substrate gate runs beside what follows
        (:meth:`_prove_substrate_beside`) and its failure calls it; without,
        it runs here and its failure returns False.
        """
        print_rule("[bold magenta]preflight[/bold magenta]", log)
        intake = self._check_intake(log)
        if intake is None:
            return False
        kwargs: dict[str, Any] = {}
        if self._command_runner is not None:
            kwargs["runner"] = self._command_runner
        try:
            result = run_preflight(self.task_data, **kwargs)
        except PreflightError as exc:
            # A spec that contradicts itself is fixed in the spec; the
            # infrastructure guide would send the operator the wrong way.
            hint = "" if isinstance(exc, SpecInconsistencyError) else TROUBLESHOOTING_HINT
            print_message(f"[bold red]✘ preflight: {exc}[/bold red]", log)
            if hint:
                print_message(hint, log)
            self.journal.open(slug="(preflight failed)")
            self.journal.section("preflight failed", f"{exc}\n\n{hint}".rstrip())
            return False

        state.slug = result.slug
        state.branch = result.branch
        state.devices = list(result.devices)
        self.environment = result.environment
        self._preflight = result
        if isinstance(self._turns, InProcessTurnExecutor):
            # Each turn in this process leaves its prompt, activity and answer here.
            self._turns.keep_records(self.workspace / TURNS_DIRNAME)

        self.journal.open(
            slug=result.slug,
            header_lines=[
                f"- checkpoint: `{self.task_data['checkpoint_path']}`",
                f"- target: `{result.target_rel}`",
                f"- units: {', '.join(assembly_units(self.task_data))}",
            ],
        )
        if intake:
            self.journal.section(*intake)
        self.journal.section("preflight", f"```\n{format_preflight_report(result)}\n```")
        print_message(
            f"[bold green]✔ preflight[/bold green]\n{format_preflight_report(result)}", log
        )

        # Where this run's commands land, decided once, from the resolved
        # environment. An explicitly injected runner still wins.
        self._command_executor = select_command_executor(
            result.environment,
            # The checkout under test: gates run in it and the session refuses
            # a result whose ranks imported the engine from anywhere else.
            tree=self.task_data.get("trtllm_repo_path"),
        )
        if self._shell_runner_override is None:
            self._shell_runner = self._noting_session_loss(shell_runner_for(self._command_executor))

        # Legal configurations that cost hours: notes, not errors.
        for warning in (
            image_import_warning(result.environment),
            shares_allocation_with_spool(result.environment),
        ):
            if warning:
                result.notes.append(warning)
                print_message(f"[bold yellow]⚠ {warning}[/bold yellow]", log)

        # A dry run proves the agents too: whether turns work is what it asks.
        if not self._prove_agents(log):
            return False
        if not self._check_processor(result, log):
            return False
        if not self._check_coverage(log):
            return False

        # Skipped on a dry run, which only asks whether the spec resolves.
        if not self.dry_run and not self.skip_substrate_gate:
            if stop is not None:
                self._prove_substrate_beside(result, log, stop)
            elif not self._prove_substrate(result, log):
                return False

        self._take_survey(result, log)

        # Concurrent turns share one `.git/index`: the commit helper takes a lock
        # and refuses a commit whose contents are not exactly what was asked for.
        commit_script = install_committer(
            self.workspace,
            self.task_data["trtllm_repo_path"],
            policy=CommitPolicy.from_task(self.task_data, workspace=self.workspace),
        )
        gate_cli.install_cli(self.workspace)

        # Prompts can only be built now: every role's prompt carries the
        # resolved launch prefix, which did not exist until the device claim
        # was made.
        if self._prompts_override is None:
            self.prompts = build_trtllm_modeling_v2_prompts(
                environment=result.environment,
                source_embargo=display_entries(self.task_data),
                scope=str(self.task_data.get("scope") or "full"),
                commit_script=str(commit_script),
                notes=str(self.task_data.get("notes") or ""),
            )
        # What every role is told, verbatim and per launch.
        dump_prompt_bundle(self.prompts, self.workspace / PROMPTS_DIRNAME)
        self._write_host_facts(result)
        self._materialize_skills()
        self._write_contract()
        self._checkpoint(state)
        return True

    def _check_intake(self, log) -> tuple[str, ...] | None:
        """Run the intake checks on :attr:`intake_request`; None when a block stops the run.

        Every finding on the request (``spec/intake/checks.py``) and every
        file-reading check of the spec compiled from it
        (:func:`~..preflight.check_intake_request`), reported together.
        Returns the journal section (title, body) to write once the journal is
        open; an empty tuple for a run with no request.
        """
        if self.intake_request is None:
            return ()
        path = Path(self.intake_request)
        request = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        findings = check_intake_request(request, self.task_data)
        report = render_findings(findings)
        blocks = [f for f in findings if f.level == BLOCK]
        if blocks:
            print_message(
                f"[bold red]✘ intake: {len(blocks)} block(s) in {escape_markup(str(path))}"
                f"[/bold red]\n{escape_markup(report)}",
                log,
            )
            self.journal.open(slug="(intake checks failed)")
            self.journal.section(
                f"intake checks failed: {len(blocks)} block(s)",
                f"`{path}`\n\n```\n{report}\n```\n\nEdit the request and re-run the same "
                "command; a run that never got past preflight takes the edit as it is.",
            )
            return None
        status = confirmation_status(request, read_confirmation(path))
        warned = sum(1 for f in findings if f.level == WARN)
        print_message(
            f"[bold green]✔ intake[/bold green] {len(findings) - warned} pass, {warned} warn; "
            f"{escape_markup(str(path))} is {status}",
            log,
        )
        return (f"intake: {status}", f"`{path}`\n\n```\n{report}\n```")

    def _check_processor(self, result: PreflightResult, log) -> bool:
        """Load a multimodal checkpoint's processor under the run's interpreter; False on a missing import."""
        try:
            notes = check_processor_dependencies(
                self.task_data, result.environment, shell_runner=self._shell_runner
            )
        except PreflightError as exc:
            print_message(f"[bold red]✘ {escape_markup(str(exc))}[/bold red]", log)
            print_message(TROUBLESHOOTING_HINT, log)
            self.journal.section("processor probe failed", f"{exc}\n\n{TROUBLESHOOTING_HINT}")
            return False
        for note in notes:
            result.notes.append(note)
            print_message(f"→ {escape_markup(note)}", log)
        return True

    def _prove_substrate(self, result, log) -> bool:
        """Run the substrate gate, or stand on a recent pass in this environment; False when it fails.

        A resumed run -- a reload, a restart -- does not run it when the last
        pass (:data:`SUBSTRATE_RECORD`) stands for this environment
        (:func:`substrate_reuse`), and carries that pass's notes into the host
        facts. A new run always runs it. The journal says which happened and
        why; a failure removes the record, so the next start runs it again.
        """
        record_path = self.workspace / SUBSTRATE_RECORD
        inputs = substrate_inputs(self.task_data, result.environment)
        if self.resume:
            record = read_json(record_path)
            reusable, why = substrate_reuse(record, inputs, now=time.time())
        else:
            record, reusable, why = None, False, "this is a new run"
        if reusable:
            result.notes.extend(str(note) for note in (record or {}).get("notes") or ())
            self.journal.event("substrate gate not run", f"{why}; `{record_path}`")
            print_message(f"[bold green]✔ substrate gate[/bold green] not run: {why}", log)
            return True
        before = len(result.notes)
        try:
            substrate_gate(
                self.task_data,
                result.environment,
                shell_runner=self._shell_runner,
                notes=result.notes,
                workdir=record_path.parent,
            )
        except PreflightError as exc:
            record_path.unlink(missing_ok=True)
            print_message(f"[bold red]✘ {exc}[/bold red]", log)
            print_message(TROUBLESHOOTING_HINT, log)
            self.journal.section("substrate gate failed", f"{exc}\n\n{TROUBLESHOOTING_HINT}")
            return False
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            record_path, {"at": time.time(), "inputs": inputs, "notes": result.notes[before:]}
        )
        self.journal.event("substrate gate passed", f"run because {why}; `{record_path}`")
        return True

    def _prove_substrate_beside(self, result, log, stop: Callable[[str], None]) -> None:
        """Run the substrate gate on a thread of its own, while the run's first turns start.

        A pass rewrites the host facts with what it measured; a failure, or an
        error that kept it from a verdict, calls ``stop`` with why.
        """

        def prove() -> None:
            try:
                passed = self._prove_substrate(result, log)
            except Exception as exc:  # the run waits on this thread's answer, whatever it hit
                self.journal.section("substrate gate failed", f"{type(exc).__name__}: {exc}")
                passed = False
            if passed:
                self._write_host_facts(result)
                return
            stop("the substrate gate failed; the journal has its output")

        self._substrate_thread = threading.Thread(target=prove, name="mv2-substrate")
        self._substrate_thread.start()

    def _await_substrate(self) -> None:
        """Wait for a substrate gate that runs beside the run, if one does."""
        thread = getattr(self, "_substrate_thread", None)
        if thread is not None:
            thread.join()

    def _prove_agents(self, log) -> bool:
        """Probe every agent the roles resolve to, or stand on a recent pass; False when one fails.

        Only for turns in this process (:mod:`..agent_probe`): on the external
        transport a driver's session runs them, which nothing here can prove.
        A resumed run does not probe again while the last pass
        (:data:`~..agent_probe.AGENT_PROBE_RECORD`) stands for the same
        agents, CLI, host and checkout (:func:`~..agent_probe.probe_inputs`)
        and is under :data:`~..agent_probe.AGENT_PROBE_FRESH_S` old. A failure
        stops the run, says what each failed turn said and what to do, and
        removes the record, so the next start probes again.
        """
        if not isinstance(self._turns, InProcessTurnExecutor):
            return True
        if os.environ.get(agent_probe.AGENT_PROBE_SKIP_ENV, "").strip() not in ("", "0"):
            why = f"${agent_probe.AGENT_PROBE_SKIP_ENV} is set"
            self.journal.event("agent probe not run", why)
            print_message(f"[bold yellow]⚠ agent probe not run: {why}[/bold yellow]", log)
            return True
        agents = agent_probe.distinct_agents(self.task_data)
        record_path = self.workspace / agent_probe.AGENT_PROBE_RECORD
        inputs = agent_probe.probe_inputs(self.task_data, agents)
        if self.resume:
            reusable, why = substrate_reuse(
                read_json(record_path),
                inputs,
                now=time.time(),
                fresh_s=agent_probe.AGENT_PROBE_FRESH_S,
            )
        else:
            reusable, why = False, "this is a new run"
        if reusable:
            self.journal.event("agent probe not run", f"{why}; `{record_path}`")
            print_message(f"[bold green]✔ agent probe[/bold green] not run: {why}", log)
            return True
        repo = str(self.task_data.get("trtllm_repo_path") or "")
        words = (self._git(repo, "rev-parse", "HEAD") or "").split() if repo else []
        outcomes = agent_probe.probe_agents(
            agents, self.workspace, repo=repo, head=words[0] if words else ""
        )
        if not all(outcome.ok for outcome in outcomes):
            record_path.unlink(missing_ok=True)
            report = agent_probe.format_failures(outcomes)
            failed = sum(1 for outcome in outcomes if not outcome.ok)
            print_message(
                f"[bold red]✘ agent probe: {failed} of {len(outcomes)} agent(s) cannot run a "
                f"turn with Bash here[/bold red]\n{escape_markup(report)}",
                log,
            )
            self.journal.section("agent probe failed", report)
            return False
        record_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            record_path,
            {"at": time.time(), "inputs": inputs, "agents": [a.as_dict() for a in agents]},
        )
        proven = "; ".join(f"{a.label}: {', '.join(a.roles)}" for a in agents)
        self.journal.event("agent probe passed", f"run because {why}; {proven}; `{record_path}`")
        print_message(
            f"[bold green]✔ agent probe[/bold green] {len(agents)} agent(s) ran Bash in the "
            f"workspace and read the checkout's HEAD: {escape_markup(proven)}",
            log,
        )
        return True

    def _check_coverage(self, log) -> bool:
        """Write what the contract proves of the request; stop on a gap ``task.yaml`` does not accept.

        The matrix (``requirements.py``) goes to ``coverage.yaml``. A requested
        cell no check covers is a decision the spec states (``coverage.accept``),
        or the run stops here, before anything is spent on it.
        """
        contract = contract_mod.compile_contract(self.task_data, self._gate_env().world_size)
        matrix = requirements_mod.requirements_matrix(
            self.task_data, contract, self._gate_env().world_size
        )
        path = self.workspace / requirements_mod.COVERAGE_FILENAME
        path.write_text(yaml.safe_dump(matrix.as_dict(), sort_keys=False), encoding="utf-8")
        if not matrix.gaps():
            return True
        message = requirements_mod.gap_message(matrix)
        print_message(f"[bold red]✘ coverage: {escape_markup(message)}[/bold red]", log)
        self.journal.open(slug="(preflight failed)")
        self.journal.section("coverage: requested cells left uncovered", message)
        return False

    def _write_contract(self) -> None:
        """Compile the contract from the resolved spec, and say what it holds."""
        self._contract = contract_mod.compile_contract(self.task_data, self._gate_env().world_size)
        contract_mod.write_contract(self.workspace, self._contract)
        self.journal.section(
            f"contract: {len(self._contract.required())} required check(s)",
            self._contract.render(),
        )
        matrix = requirements_mod.requirements_matrix(
            self.task_data, self._contract, self._gate_env().world_size
        )
        self.journal.section("coverage: what the contract proves of the request", matrix.render())

    def _write_host_facts(self, result) -> None:
        """Seed ``host_facts.md`` from what this run already knows.

        Rewritten on every preflight, which a resume also runs, so anything
        a role appended is preserved below
        :data:`HOST_FACTS_APPEND_MARKER` and everything above it is
        regenerated.
        """
        env = result.environment
        tmpdir = str((env.env or {}).get("TMPDIR", "") or "")
        lines = [
            "# Host facts",
            "",
            "What this host does that a generic Linux box does not. **Read this before "
            "you debug an environment failure** — every fact here was learned the "
            "expensive way by an earlier dispatch on this machine.",
            "",
            f"This is the *preventive* half. `{Path(__file__).resolve().parents[2] / 'TROUBLESHOOTING.md'}` "
            "is the other half and is keyed on the verbatim error text of a failure that "
            "already happened. Keep them separate: one is read before anything breaks, "
            "the other is grepped afterwards.",
            "",
            "## Facts",
            "",
            "- **`pytest` exits 4 on this host even when every test passed.** The venv "
            "lives inside the repository root, so a session-final `sys.path` audit "
            "fires. Export `TRTLLM_SKIP_SYS_PATH_CHECK=1`, and adjudicate on the "
            "summary line (`N passed`), never on the exit code.",
            f"- **`TMPDIR` must be at most {MAX_TMPDIR_LEN} characters.** The LLM API's "
            "RPC binds a unix socket under it, the path is capped at 107 bytes and the "
            "socket basename takes 45. This run's `TMPDIR` is "
            + (f"`{tmpdir}` ({len(tmpdir)} characters)." if tmpdir else "not set in the spec."),
            "- **`UCX ERROR uct_iface_open(tcp/rdma_vf_rail0)` is noise.** UCX falls back "
            "and the run exits 0. Do not chase it; it is not why your job failed.",
            "- **`RC=$?` after a pipe is the *pipe's* exit code, not the command's.** "
            "`cmd | tee log; RC=$?` reports `tee`. Use `PIPESTATUS[0]`, or drop the pipe.",
            "- **A venv interpreter created with `--system-site-packages` resolves only "
            "inside the container it was created in.** Running it from the host finds an "
            "interpreter and none of the packages, which reads as a broken install.",
        ]
        if env.needs_llmapi_launch:
            lines.append(
                f"- **A {env.ntasks}-rank launch needs the `SLURM_*` scrub and "
                f"`--oversubscribe`.** Without them every rank reads `mpi_rank=0`, every "
                f"rank becomes a leader, and the job hangs forever with no error and no "
                f"traceback. Both are already in the launch prefix you were handed — use "
                f"it verbatim rather than writing your own `mpirun` line."
            )
        if env.container_image:
            lines.append(
                f"- **The container is `{env.container_image}`.** An interpreter path in "
                f"this run's environment block is a path *inside* it."
            )
        if result.notes:
            lines.extend(
                ["", "## What preflight measured on this host", ""]
                + [f"- {note}" for note in result.notes]
            )
        lines.extend(
            [
                "",
                "## Learned during this run",
                "",
                "This section is yours. If you spend more than a few minutes discovering "
                "something about this host that the section above would have saved you, "
                "append it here as one bullet: the symptom first, then what to do. The "
                "orchestrator never rewrites anything below the marker, so it survives "
                "every restart and reaches every later dispatch.",
                "",
                HOST_FACTS_APPEND_MARKER,
                "",
            ]
        )
        head = "\n".join(lines)

        appended = ""
        if self.host_facts_path.is_file():
            existing = self.host_facts_path.read_text(encoding="utf-8")
            if HOST_FACTS_APPEND_MARKER in existing:
                appended = existing.split(HOST_FACTS_APPEND_MARKER, 1)[1].lstrip("\n")
        self.host_facts_path.write_text(head + appended, encoding="utf-8")

    def _context_block(self) -> str:
        """The host facts every dispatch carries, by reference: a fact one turn appends reaches the next."""
        if not self.host_facts_path.is_file():
            return ""
        return (
            f"## Host facts\n\n"
            f"Read `{self.host_facts_path}` before you debug anything about this "
            f"machine. It carries what this host does differently — `pytest`'s exit "
            f"code, the `TMPDIR` bound, the launch rules, which errors are noise. If "
            f"you learn a new one, append it under `## Learned during this run`."
        )

    def _take_survey(self, result, log) -> None:
        """Measure what the checkpoint demands and what the build supplies.

        **Never fatal.** A failed probe is written into the block the surveyor
        reads, and the run goes on.
        """
        surface = read_checkpoint_surface(result.checkpoint_config)
        # Taken on a dry run too: the probe is one process, no MPI world.
        inventory = run_inventory_probe(
            self.task_data,
            result.environment,
            shell_runner=self._shell_runner,
        )
        self._surface_block = format_surface(surface, inventory)

        # The runtime map, narrowed to what this checkpoint demands.
        demands = surface.demand_names
        index = load_index()
        areas = areas_for_demands(demands, index=index)
        self._runtime_block = format_areas(
            areas, unmatched=unmatched_demands(demands, index=index), overview=index.overview
        )
        if self._runtime_block:
            # Every worker is pointed at it, not only the survey that is handed it.
            (self.workspace / RUNTIME_MAP_FILENAME).write_text(
                self._runtime_block, encoding="utf-8"
            )
        # Provenance, for the kernel writers: what each kernel source is and what
        # certifying one of its kernels requires.
        self._kernel_sources_block = format_kernel_sources(load_kernel_sources())
        write_surface_file(self.workspace / "surface.md", surface, inventory)

        headline = (
            f"{len(surface.demands)} structural demand(s), "
            f"{len(surface.unrecognized)} unrecognized key(s); "
            + (
                f"{len(inventory.spendable)}/{len(inventory.catalog)} entries certified on "
                f"{inventory.arch}, {len(inventory.unwrapped_ops)} unwrapped op(s), "
                f"{len(inventory.flashinfer_unreferenced)} FlashInfer callable(s) the engine "
                f"does not call"
                if inventory.ok
                else f"inventory probe failed: {inventory.error.splitlines()[0]}"
            )
        )
        self.journal.section("surface survey", headline + "\n\nFull survey: `surface.md`")
        # A probe that never ran must not read as an empty catalog.
        mark, colour = ("✔", "green") if inventory.ok else ("⚠", "yellow")
        print_message(f"[bold {colour}]{mark} survey[/bold {colour}] {headline}", log)
        if surface.unrecognized:
            print_message(f"  unrecognized structural keys: {', '.join(surface.unrecognized)}", log)
