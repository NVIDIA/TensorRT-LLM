# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight: the checks, the environment, the agent probe, host facts and the surface survey.

The substrate gate, the one check that boots a GPU, runs beside the run's
first turns, unless a recent pass in this environment stands for it: its
failure stops the run, which stays open for the same command to resume.
"""

from __future__ import annotations

import dataclasses
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
from ...agents.roles import packaged_skills
from ...agents.turns import TURNS_DIRNAME
from ...checks import check_config, gate_cli
from ...infra.compute.environment import (
    MAX_TMPDIR_LEN,
    RunEnvironment,
    image_import_warning,
    shares_allocation_with_spool,
)
from ...infra.compute.executor import select_command_executor, shell_runner_for
from ...infra.files import read_json, write_json_atomic
from ...infra.git import CommitPolicy
from ...operators import prior_art, tree_anchors
from ...operators.inventory import (
    format_surface,
    run_inventory_probe,
    surface_names,
    write_surface_file,
)
from ...spec.task_schema import assembly_units
from ..records.state import WorkflowState
from ..validation import contract as contract_mod
from ..validation import requirements as requirements_mod
from . import agent_probe
from .preflight import (
    PreflightError,
    PreflightResult,
    SpecInconsistencyError,
    check_processor_dependencies,
    format_preflight_report,
    run_preflight,
    substrate_gate,
)

#: Where to investigate a failed infrastructure probe.
TROUBLESHOOTING_HINT = (
    "Inspect the failed command and its log, the measured interpreter and checkout, "
    "and the compute session state. This infrastructure failure provides no verdict "
    "on the target's correctness."
)

#: The host facts written into the workspace and referenced by every dispatch prompt.
HOST_FACTS_FILENAME = "host_facts.md"
#: Everything below this line belongs to the roles; preflight preserves it on resume.
HOST_FACTS_APPEND_MARKER = "<!-- roles append below this line; nothing above it is preserved -->"

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
    options = check_config.resolve_llm_api_options(accuracy, None)
    options_text = None
    if options and Path(str(options)).is_file():
        options_text = Path(str(options)).read_text(encoding="utf-8", errors="replace")
    parts: dict[str, Any] = {
        "environment": dataclasses.asdict(environment),
        "spec_environment": task_data.get("environment"),
        "engine": {
            "checkpoint_path": task_data.get("checkpoint_path"),
            "parallel": task_data.get("parallel"),
            "runtime": check_config.resolve_runtime(task_data, None),
        },
        "options": [options, options_text],
        "checkout": task_data.get("trtllm_repo_path"),
    }
    return agent_probe.part_digests(parts)


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


def _run_preflight(
    workflow, state: WorkflowState, log, *, stop: Callable[[str], None] | None = None
) -> bool:
    """Resolve and prove what the run needs before its first turn; False stops the run.

    With ``stop`` the substrate gate runs beside what follows
    (:func:`_prove_substrate_beside`) and its failure calls it; without, it
    runs here and its failure returns False.
    """
    print_rule("[bold magenta]preflight[/bold magenta]", log)
    kwargs: dict[str, Any] = {}
    if workflow._command_runner is not None:
        kwargs["runner"] = workflow._command_runner
    try:
        result = run_preflight(workflow.task_data, **kwargs)
    except PreflightError as exc:
        # A spec that contradicts itself is fixed in the spec; the
        # infrastructure guide would send the operator the wrong way.
        hint = "" if isinstance(exc, SpecInconsistencyError) else TROUBLESHOOTING_HINT
        print_message(f"[bold red]✘ preflight: {exc}[/bold red]", log)
        if hint:
            print_message(hint, log)
        workflow.journal.open(slug="(preflight failed)")
        workflow.journal.section("preflight failed", f"{exc}\n\n{hint}".rstrip())
        return False

    state.slug = result.slug
    state.branch = result.branch
    state.devices = list(result.devices)
    workflow.environment = result.environment
    workflow._preflight = result
    # Each turn in this process leaves its prompt, activity and answer here.
    workflow._turns.keep_records(workflow.workspace / TURNS_DIRNAME)

    workflow.journal.open(
        slug=result.slug,
        header_lines=[
            f"- checkpoint: `{workflow.task_data['checkpoint_path']}`",
            f"- target: `{result.target_rel}`",
            f"- units: {', '.join(assembly_units(workflow.task_data))}",
        ],
    )
    workflow.journal.section("preflight", f"```\n{format_preflight_report(result)}\n```")
    print_message(f"[bold green]✔ preflight[/bold green]\n{format_preflight_report(result)}", log)

    # Where this run's commands land, decided once, from the resolved
    # environment. An explicitly injected runner still wins.
    workflow._command_executor = select_command_executor(
        result.environment,
        # The checkout under test: gates run in it and the session refuses
        # a result whose ranks imported the engine from anywhere else.
        tree=workflow.task_data.get("trtllm_repo_path"),
    )
    if workflow._shell_runner_override is None:
        workflow._shell_runner = shell_runner_for(workflow._command_executor)

    # Legal configurations that cost hours: notes, not errors.
    for warning in (
        image_import_warning(result.environment),
        shares_allocation_with_spool(result.environment),
    ):
        if warning:
            result.notes.append(warning)
            print_message(f"[bold yellow]⚠ {warning}[/bold yellow]", log)

    # A dry run proves the agents too: whether turns work is what it asks.
    if not _prove_agents(workflow, log):
        return False
    if not _check_processor(workflow, result, log):
        return False
    if not _check_coverage(workflow, log):
        return False

    # Skipped on a dry run, which only asks whether the spec resolves.
    if not workflow.dry_run and not workflow.skip_substrate_gate:
        if stop is not None:
            _prove_substrate_beside(workflow, result, log, stop)
        elif not _prove_substrate(workflow, result, log):
            return False

    _take_survey(workflow, result, log)

    gate_cli.install_cli(workflow.workspace)

    # Prompts can only be built now: every role's prompt carries the
    # resolved launch prefix, which did not exist until the device claim
    # was made.
    if workflow._prompts_override is None:
        workflow.prompts = build_trtllm_modeling_v2_prompts(
            environment=result.environment,
            source_embargo=[str(e) for e in workflow.task_data.get("source_embargo") or ()],
            scope=str(workflow.task_data.get("scope") or "full"),
            commit_policy=CommitPolicy.from_task(workflow.task_data, workspace=workflow.workspace),
            notes=str(workflow.task_data.get("notes") or ""),
        )
    # What every role is told, verbatim and per launch.
    dump_prompt_bundle(workflow.prompts, workflow.workspace / PROMPTS_DIRNAME)
    _write_host_facts(workflow, result)
    workflow._materialize_skills()
    _write_contract(workflow)
    workflow._checkpoint(state)
    return True


def _check_processor(workflow, result: PreflightResult, log) -> bool:
    """Load a multimodal checkpoint's processor under the run's interpreter; False on a missing import."""
    try:
        notes = check_processor_dependencies(
            workflow.task_data, result.environment, shell_runner=workflow._shell_runner
        )
    except PreflightError as exc:
        print_message(f"[bold red]✘ {escape_markup(str(exc))}[/bold red]", log)
        print_message(TROUBLESHOOTING_HINT, log)
        workflow.journal.section("processor probe failed", f"{exc}\n\n{TROUBLESHOOTING_HINT}")
        return False
    for note in notes:
        result.notes.append(note)
        print_message(f"→ {escape_markup(note)}", log)
    return True


def _prove_substrate(workflow, result, log) -> bool:
    """Run the substrate gate, or stand on a recent pass in this environment; False when it fails.

    A resumed run does not run it when the last
    pass (:data:`SUBSTRATE_RECORD`) stands for this environment
    (:func:`substrate_reuse`), and carries that pass's notes into the host
    facts. A new run always runs it. The journal says which happened and
    why; a failure removes the record, so the next start runs it again.
    """
    record_path = workflow.workspace / SUBSTRATE_RECORD
    inputs = substrate_inputs(workflow.task_data, result.environment)
    if workflow.resume:
        record = read_json(record_path)
        reusable, why = substrate_reuse(record, inputs, now=time.time())
    else:
        record, reusable, why = None, False, "this is a new run"
    if reusable:
        result.notes.extend(str(note) for note in (record or {}).get("notes") or ())
        workflow.journal.event("substrate gate not run", f"{why}; `{record_path}`")
        print_message(f"[bold green]✔ substrate gate[/bold green] not run: {why}", log)
        return True
    before = len(result.notes)
    try:
        substrate_gate(
            workflow.task_data,
            result.environment,
            shell_runner=workflow._shell_runner,
            notes=result.notes,
            workdir=record_path.parent,
        )
    except PreflightError as exc:
        record_path.unlink(missing_ok=True)
        print_message(f"[bold red]✘ {exc}[/bold red]", log)
        print_message(TROUBLESHOOTING_HINT, log)
        workflow.journal.section("substrate gate failed", f"{exc}\n\n{TROUBLESHOOTING_HINT}")
        return False
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        record_path, {"at": time.time(), "inputs": inputs, "notes": result.notes[before:]}
    )
    workflow.journal.event("substrate gate passed", f"run because {why}; `{record_path}`")
    return True


def _prove_substrate_beside(workflow, result, log, stop: Callable[[str], None]) -> None:
    """Run the substrate gate on a thread of its own, while the run's first turns start.

    A pass rewrites the host facts with what it measured; a failure, or an
    error that kept it from a verdict, calls ``stop`` with why.
    """

    def prove() -> None:
        try:
            passed = _prove_substrate(workflow, result, log)
        except Exception as exc:  # the run waits on this thread's answer, whatever it hit
            workflow.journal.section("substrate gate failed", f"{type(exc).__name__}: {exc}")
            passed = False
        if passed:
            _write_host_facts(workflow, result)
            return
        stop("the substrate gate failed; the journal has its output")

    workflow._substrate_thread = threading.Thread(target=prove, name="mv2-substrate", daemon=True)
    workflow._substrate_thread.start()


def _await_substrate(workflow) -> None:
    """Wait for a substrate gate that runs beside the run, if one does."""
    thread = getattr(workflow, "_substrate_thread", None)
    if thread is not None:
        thread.join()


def _prove_agents(workflow, log) -> bool:
    """Probe every agent the roles resolve to, or stand on a recent pass; False when one fails.

    A resumed run does not probe again while the last pass
    (:data:`~..agent_probe.AGENT_PROBE_RECORD`) stands for the same
    agents, CLI, host and checkout (:func:`~..agent_probe.probe_inputs`)
    and is under :data:`~..agent_probe.AGENT_PROBE_FRESH_S` old. A failure
    stops the run, says what each failed turn said and what to do, and
    removes the record, so the next start probes again.
    """
    if os.environ.get(agent_probe.AGENT_PROBE_SKIP_ENV, "").strip() not in ("", "0"):
        why = f"${agent_probe.AGENT_PROBE_SKIP_ENV} is set"
        workflow.journal.event("agent probe not run", why)
        print_message(f"[bold yellow]⚠ agent probe not run: {why}[/bold yellow]", log)
        return True
    agents = agent_probe.distinct_agents(workflow.task_data, resolved=workflow._role_agents())
    record_path = workflow.workspace / agent_probe.AGENT_PROBE_RECORD
    inputs = agent_probe.probe_inputs(workflow.task_data, agents)
    if workflow.resume:
        reusable, why = substrate_reuse(
            read_json(record_path),
            inputs,
            now=time.time(),
            fresh_s=agent_probe.AGENT_PROBE_FRESH_S,
        )
    else:
        reusable, why = False, "this is a new run"
    if reusable:
        recorded = (read_json(record_path) or {}).get("foreign_skills") or {}
        workflow._foreign_skills = {str(k): tuple(v) for k, v in recorded.items()}
        workflow.journal.event("agent probe not run", f"{why}; `{record_path}`")
        print_message(f"[bold green]✔ agent probe[/bold green] not run: {why}", log)
        return True
    repo = str(workflow.task_data.get("trtllm_repo_path") or "")
    outcomes = agent_probe.probe_agents(
        agents, workflow.workspace, repo=repo, head=workflow._head()
    )
    workflow._foreign_skills = agent_probe.foreign_skills(outcomes, packaged_skills())
    if not all(outcome.ok for outcome in outcomes):
        record_path.unlink(missing_ok=True)
        report = agent_probe.format_failures(outcomes)
        failed = sum(1 for outcome in outcomes if not outcome.ok)
        print_message(
            f"[bold red]✘ agent probe: {failed} of {len(outcomes)} agent(s) cannot run a "
            f"turn with Bash here[/bold red]\n{escape_markup(report)}",
            log,
        )
        workflow.journal.section("agent probe failed", report)
        return False
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        record_path,
        {
            "at": time.time(),
            "inputs": inputs,
            "agents": [a.as_dict() for a in agents],
            "foreign_skills": {k: list(v) for k, v in workflow._foreign_skills.items()},
        },
    )
    proven = "; ".join(
        outcome.probed.label
        + (f" [{outcome.settings}]" if outcome.settings else "")
        + f": {', '.join(outcome.probed.roles)}"
        for outcome in outcomes
    )
    turned_off = "; ".join(
        f"{backend} roles run without {', '.join(names)}"
        for backend, names in workflow._foreign_skills.items()
        if names
    )
    workflow.journal.event(
        "agent probe passed",
        f"run because {why}; {proven}; `{record_path}`"
        + (f"; the user's skills are off ({turned_off}), and their plugins" if turned_off else ""),
    )
    print_message(
        f"[bold green]✔ agent probe[/bold green] {len(agents)} agent(s) ran Bash in the "
        f"workspace and read the checkout's HEAD: {escape_markup(proven)}",
        log,
    )
    return True


def _check_coverage(workflow, log) -> bool:
    """Write what the contract proves of the request; stop on a gap ``task.yaml`` does not accept.

    The matrix (``requirements.py``) goes to ``coverage.yaml``. A requested
    cell no check covers is a decision the spec states (``coverage.accept``),
    or the run stops here, before anything is spent on it.
    """
    workflow._contract = contract_mod.compile_contract(
        workflow.task_data, workflow._gate_env().world_size
    )
    matrix = workflow._coverage = requirements_mod.requirements_matrix(
        workflow.task_data, workflow._contract, workflow._gate_env().world_size
    )
    path = workflow.workspace / requirements_mod.COVERAGE_FILENAME
    path.write_text(yaml.safe_dump(matrix.as_dict(), sort_keys=False), encoding="utf-8")
    if not matrix.gaps():
        return True
    message = requirements_mod.gap_message(matrix)
    print_message(f"[bold red]✘ coverage: {escape_markup(message)}[/bold red]", log)
    workflow.journal.open(slug="(preflight failed)")
    workflow.journal.section("coverage: requested cells left uncovered", message)
    return False


def _write_contract(workflow) -> None:
    """Write the contract the coverage check compiled, and say what it holds."""
    contract_mod.write_contract(workflow.workspace, workflow._contract)
    workflow.journal.section(
        f"contract: {len(workflow._contract.required())} required check(s)",
        workflow._contract.render(),
    )
    workflow.journal.section(
        "coverage: what the contract proves of the request", workflow._coverage.render()
    )
    _write_intake_lint(workflow)


def _write_intake_lint(workflow) -> None:
    """Journal the intake lint of the spec the run starts from; a finding never stops the run."""
    from ...spec import lint as lint_mod

    try:
        report = lint_mod.lint_task(
            workflow.task_data,
            contract=workflow._contract,
            world_size=workflow._gate_env().world_size,
        )
    except Exception as exc:  # the lint is advice: it must not cost the run its start
        workflow.journal.section("intake lint: not run", f"{type(exc).__name__}: {exc}")
        return
    workflow.journal.section(f"intake lint: {report.summary()}", report.render(prior_art_limit=8))


def _write_host_facts(workflow, result) -> None:
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
        "For a failure, read the affected turn's activity log or the job's receipt and log. "
        "Report the exact error, command, commit and evidence paths.",
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
    if workflow.host_facts_path.is_file():
        existing = workflow.host_facts_path.read_text(encoding="utf-8")
        if HOST_FACTS_APPEND_MARKER in existing:
            appended = existing.split(HOST_FACTS_APPEND_MARKER, 1)[1].lstrip("\n")
    workflow.host_facts_path.write_text(head + appended, encoding="utf-8")


#: What the tree already implements for this checkpoint and what the engine refuses, per run.
PRIOR_ART_FILENAME = "prior_art.md"
#: Where the checkout under test keeps what the workflow reads and the roles are sent to.
TREE_MAP_FILENAME = "tree_map.md"


def _checkpoint_config(task_data: Mapping[str, Any]) -> dict[str, Any]:
    """The checkpoint's ``config.json``, or ``{}`` when it cannot be read here."""
    import json

    try:
        path = Path(str(task_data.get("checkpoint_path") or "")) / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _take_prior_art(workflow, config: Mapping[str, Any]) -> None:
    """Read the tree's prior art and the engine's limits for this checkpoint, for the survey.

    Into ``prior_art.md`` in the workspace, for every role, and the section the
    survey dispatch carries; the leads and the limits the survey review holds
    the vocabulary to. **Never fatal**: an unreadable config or tree leaves none.
    """
    repo = workflow.task_data.get("trtllm_repo_path")
    workflow._prior_art, workflow._engine_limits = None, []
    if config and repo:
        try:
            workflow._prior_art = prior_art.find_prior_art(repo, config)
            workflow._engine_limits = prior_art.engine_limits(config, repo)
        except OSError:
            pass
    workflow._prior_art_block = prior_art.survey_block(workflow._prior_art, workflow._engine_limits)
    path = workflow.workspace / PRIOR_ART_FILENAME
    if not workflow._prior_art_block:
        path.unlink(missing_ok=True)
        return
    path.write_text(
        prior_art.survey_block(workflow._prior_art, workflow._engine_limits, limit=12),
        encoding="utf-8",
    )


def _take_tree_map(workflow) -> str:
    """Write ``tree_map.md``: every anchor's place in the checkout under test (:mod:`.tree_anchors`).

    Returns a line for the journal when an anchor moved or is missing, else
    ``""``. **Never fatal**: a checkout it cannot read leaves no map.
    """
    repo = workflow.task_data.get("trtllm_repo_path")
    path = workflow.workspace / TREE_MAP_FILENAME
    if not repo:
        path.unlink(missing_ok=True)
        return ""
    try:
        found = [tree_anchors.resolve(repo, key) for key in tree_anchors.ANCHORS]
        path.write_text(tree_anchors.render_map(repo, found), encoding="utf-8")
    except OSError:
        path.unlink(missing_ok=True)
        return ""
    parts = []
    moved = [r.anchor.concept for r in found if r.via == "search"]
    if moved:
        parts.append(f"moved: {', '.join(moved)}")
    gone = [r.anchor.concept for r in found if not r.found]
    if gone:
        parts.append(f"missing: {', '.join(gone)}")
    return f"tree map ({'; '.join(parts)}): `{TREE_MAP_FILENAME}`" if parts else ""


def _take_survey(workflow, result, log) -> None:
    """Measure what the build supplies, for the surveyor and in ``surface.md``.

    **Never fatal.** A failed probe is written into the block the surveyor
    reads, and the run goes on.
    """
    # Taken on a dry run too: the probe is one process, no MPI world.
    inventory = run_inventory_probe(
        workflow.task_data,
        result.environment,
        shell_runner=workflow._shell_runner,
    )
    config = _checkpoint_config(workflow.task_data)
    # Every list is ordered by what this checkpoint is likely to reuse before it is cut.
    rank = prior_art.ranker(config, surface_names(inventory)) if config else None
    workflow._surface_block = format_surface(inventory, rank=rank)
    write_surface_file(workflow.workspace / "surface.md", inventory, rank=rank)
    _take_prior_art(workflow, config)
    tree_note = _take_tree_map(workflow)

    headline = (
        f"{len(inventory.spendable)}/{len(inventory.catalog)} entries certified on "
        f"{inventory.arch}, {len(inventory.unwrapped_ops)} unwrapped op(s)"
        if inventory.ok
        else f"inventory probe failed: {inventory.error.splitlines()[0]}"
    )
    workflow.journal.section(
        "surface survey",
        headline + "\n\nFull survey: `surface.md`" + (f"\n\n{tree_note}" if tree_note else ""),
    )
    # A probe that never ran must not read as an empty catalog.
    mark, colour = ("✔", "green") if inventory.ok else ("⚠", "yellow")
    print_message(f"[bold {colour}]{mark} survey[/bold {colour}] {headline}", log)
