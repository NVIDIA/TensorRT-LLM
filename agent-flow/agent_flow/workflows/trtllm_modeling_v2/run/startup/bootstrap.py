# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight: the spec, checkout and checkpoint checks, the environment, the agent probe, the survey material.

It runs on every start, a resume's included, and keeps no record of a pass:
what it proves is proven again. Any failure stops the run before its first
turn; the run stays open for the same command to resume. A dry run does the
same without touching a GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule
from agent_flow.prompts import dump_prompt_bundle

from ...agents.prompts import PROMPTS_DIRNAME, build_trtllm_modeling_v2_prompts
from ...agents.roles import packaged_skills
from ...agents.turns import TURNS_DIRNAME
from ...checks import anchors as anchors_mod
from ...checks import gate_cli
from ...infra.compute.environment import MAX_TMPDIR_LEN
from ...infra.compute.executor import select_command_executor, shell_runner_for
from ...operators import prior_art, tree_anchors
from ...operators.inventory import (
    format_surface,
    run_inventory_probe,
    surface_names,
    write_surface_file,
)
from ...spec import deployments as deployments_mod
from .. import contract as contract_mod
from ..state import WorkflowState
from . import agent_probe
from .advice import spec_warnings
from .preflight import (
    PreflightError,
    PreflightResult,
    SpecInconsistencyError,
    check_processor_dependencies,
    format_preflight_report,
    run_preflight,
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


def _run_preflight(workflow, state: WorkflowState, log) -> bool:
    """Resolve and prove what the run needs before its first turn; False stops the run."""
    print_rule("[bold magenta]preflight[/bold magenta]", log)
    kwargs: dict[str, Any] = {}
    if workflow._command_runner is not None:
        kwargs["runner"] = workflow._command_runner
    try:
        # A dry run touches no GPU: it asks whether the spec, checkout and agents can run.
        result = run_preflight(workflow.task_data, claim=not workflow.dry_run, **kwargs)
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
            f"- deployments: {', '.join(d.name for d in deployments_mod.resolve(workflow.task_data))}",
        ],
    )
    workflow.journal.section("preflight", f"```\n{format_preflight_report(result)}\n```")
    print_message(f"[bold green]✔ preflight[/bold green]\n{format_preflight_report(result)}", log)

    # Where this run's commands land, decided once, from the resolved
    # environment. An explicitly injected runner still wins.
    workflow._command_executor = select_command_executor(result.environment)
    if workflow._shell_runner_override is None:
        workflow._shell_runner = shell_runner_for(workflow._command_executor)

    # A dry run proves the agents too: whether turns work is what it asks.
    if not _prove_agents(workflow, log):
        return False
    if not _check_processor(workflow, result, log):
        return False
    if not _compile_contract(workflow, log):
        return False

    _take_survey(workflow, result, log)

    gate_cli.install_cli(workflow.workspace)

    # Prompts can only be built now: every role's prompt carries the
    # resolved launch prefix, which did not exist until the device claim
    # was made.
    if workflow._prompts_override is None:
        workflow.prompts = build_trtllm_modeling_v2_prompts(
            environment=result.environment,
            scope=str(workflow.task_data.get("scope") or "full"),
            commit_policy=workflow._commit_policy(),
            notes=str(workflow.task_data.get("notes") or ""),
        )
    # What every role is told, verbatim and per launch.
    dump_prompt_bundle(workflow.prompts, workflow.workspace / PROMPTS_DIRNAME)
    _write_host_facts(workflow, result)
    workflow._materialize_skills()
    _write_contract(workflow)
    workflow._checkpoint(state)
    return True


def _cpu_probe_runner(workflow, result: PreflightResult):
    """The runner for build and processor introspection: one CPU process on this host."""
    if workflow._shell_runner_override is not None:
        return workflow._shell_runner_override
    return shell_runner_for(select_command_executor(result.environment))


def _check_processor(workflow, result: PreflightResult, log) -> bool:
    """Load a multimodal checkpoint's processor under the run's interpreter; False on a missing import."""
    try:
        notes = check_processor_dependencies(
            workflow.task_data, result.environment, shell_runner=_cpu_probe_runner(workflow, result)
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


def _prove_agents(workflow, log) -> bool:
    """Probe every agent the roles resolve to, on every start; False when one fails.

    A failure stops the run and says what each failed turn said and what to do.
    """
    if os.environ.get(agent_probe.AGENT_PROBE_SKIP_ENV, "").strip() not in ("", "0"):
        why = f"${agent_probe.AGENT_PROBE_SKIP_ENV} is set"
        workflow.journal.event("agent probe not run", why)
        print_message(f"[bold yellow]⚠ agent probe not run: {why}[/bold yellow]", log)
        return True
    agents = agent_probe.distinct_agents(workflow.task_data, resolved=workflow._role_agents())
    repo = str(workflow.task_data.get("trtllm_repo_path") or "")
    outcomes = agent_probe.probe_agents(
        agents, workflow.workspace, repo=repo, head=workflow._head()
    )
    workflow._foreign_skills = agent_probe.foreign_skills(outcomes, packaged_skills())
    if not all(outcome.ok for outcome in outcomes):
        report = agent_probe.format_failures(outcomes)
        failed = sum(1 for outcome in outcomes if not outcome.ok)
        print_message(
            f"[bold red]✘ agent probe: {failed} of {len(outcomes)} agent(s) cannot run a "
            f"turn with Bash here[/bold red]\n{escape_markup(report)}",
            log,
        )
        workflow.journal.section("agent probe failed", report)
        return False
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
        proven
        + (f"; the user's skills are off ({turned_off}), and their plugins" if turned_off else ""),
    )
    print_message(
        f"[bold green]✔ agent probe[/bold green] {len(agents)} agent(s) ran Bash in the "
        f"workspace and read the checkout's HEAD: {escape_markup(proven)}",
        log,
    )
    return True


def _compile_contract(workflow, log) -> bool:
    """Compile the checks the run owes from the spec."""
    workflow._contract = contract_mod.compile_contract(workflow.task_data)
    return True


def _write_contract(workflow) -> None:
    """Write the contract, and say what it holds."""
    contract_mod.write_contract(workflow.workspace, workflow._contract)
    workflow.journal.section(
        f"contract: {len(workflow._contract.required())} required check(s)",
        workflow._contract.render(),
    )
    _write_deployments(workflow)
    _write_spec_warnings(workflow)


def _write_deployments(workflow) -> None:
    """Freeze each benchmark's anchor, then say what the spec comes to: one line per deployment.

    Its name, its kind and servers, its benchmarks and where each one's anchor comes
    from. Only what the workspace has not frozen yet is added: a resumed run is held to
    the anchors its first start read.
    """
    found = deployments_mod.resolve(workflow.task_data)
    try:
        frozen = anchors_mod.freeze(workflow.workspace, workflow.task_data)
        moved = anchors_mod.differences(workflow.workspace, workflow.task_data)
    except Exception as exc:
        workflow.journal.section("anchors: not frozen", f"{type(exc).__name__}: {exc}")
        frozen, moved = {}, []
    sources = {
        key: f"{a.kind}" + (f" {a.value:g}" if a.value is not None else "") + f": {a.source}"
        for key, a in frozen.items()
    }
    lines = [deployments_mod.describe(found, sources)]
    lines += [f"- differs from the tree now: {line}" for line in moved]
    body = "\n".join(lines)
    workflow.journal.section(f"deployments: {len(found)}", body)
    print_message(f"deployments ({len(found)}):\n{escape_markup(body)}")


def _write_spec_warnings(workflow) -> None:
    """Journal what the spec is likely to cost or miss; a warning never stops the run."""
    warnings = spec_warnings(workflow.task_data)
    body = "\n".join(f"- {warning}" for warning in warnings) or "None."
    workflow.journal.section(f"spec warnings: {len(warnings)}", body)


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
        shell_runner=_cpu_probe_runner(workflow, result),
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
