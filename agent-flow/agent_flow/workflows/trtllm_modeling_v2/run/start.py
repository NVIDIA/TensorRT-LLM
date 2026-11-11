# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The start of a run: preflight, then what every turn and every check of the run reads.

Runs on every start: the hard blockers (``startup/preflight.py``), the agent probe, then
the journal's header, the checks each deployment owes, the survey's inventory and the
host facts. Any failure stops the run before its first turn; the same command
starts it again. A dry run does the same without touching a GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule

from ..agents import prompts
from ..agents.turns import TURNS_DIRNAME
from ..checks import anchors as anchors_mod
from ..checks import gate_cli
from ..infra.compute.environment import MAX_TMPDIR_LEN, resolve_environment
from ..infra.compute.executor import select_command_executor, shell_runner_for
from ..infra.files import write_text_atomic
from ..operators.inventory import run_inventory_probe
from ..spec import deployments as deployments_mod
from . import contract as contract_mod
from . import report as report_mod
from .startup import agent_probe
from .startup.preflight import PREFLIGHT_FILENAME, PreflightError, PreflightResult, run_preflight

#: The host facts written into the workspace and referenced by every dispatch prompt;
#: what a role appends below the marker is kept on a resume.
HOST_FACTS_FILENAME = "host_facts.md"
HOST_FACTS_APPEND_MARKER = "<!-- roles append below this line; nothing above it is preserved -->"


def start(workflow, state, log) -> bool:
    """Prove and resolve what the run needs before its first turn; ``False`` stops the run."""
    print_rule("[bold magenta]preflight[/bold magenta]", log)

    def stop(title: str, body: str) -> bool:
        print_message(f"[bold red]✘ {title}: {escape_markup(body)}[/bold red]", log)
        workflow.journal.section(title, body)
        # For the report of a run that was onboarded before this start.
        workflow._stopped_at = f"{title}: {body}"
        return False

    task = workflow.task_data
    try:
        runner = workflow._command_runner or shell_runner_for(
            select_command_executor(resolve_environment(task))
        )
        workflow._preflight_runner = runner
        result = run_preflight(
            task, runner=runner, claim=not workflow.dry_run, workspace=workflow.workspace
        )
    except (PreflightError, anchors_mod.AnchorError) as exc:
        workflow.journal.open(slug="(preflight failed)")
        unread = isinstance(exc, anchors_mod.AnchorError)
        title = "anchors: the frozen file cannot be read" if unread else "preflight failed"
        return stop(title, str(exc))
    state.slug, state.branch, state.devices = result.slug, result.branch, list(result.devices)
    workflow.environment, workflow._preflight = result.environment, result
    workflow._anchors = result.anchors
    # Each turn in this process leaves its prompt, activity and answer here.
    workflow._turns.keep_records(workflow.workspace / TURNS_DIRNAME)
    found = deployments_mod.resolve(task)
    header = [
        f"- checkpoint: `{task['checkpoint_path']}`",
        f"- target: `{result.target_rel}`",
        f"- deployments: {', '.join(d.name for d in found)}",
    ]
    workflow.journal.open(slug=result.slug, header_lines=header)
    kept = {key: getattr(result, key) for key in ("not_attempted", "config_error", "notes")}
    write_text_atomic(workflow.workspace / PREFLIGHT_FILENAME, json.dumps(kept, indent=1))
    workflow.journal.section("preflight", f"```\n{result.report()}\n```")
    print_message(f"[bold green]✔ preflight[/bold green]\n{result.report()}", log)
    # Said at the start, not found at the end: the final pass measures the checkout, and a
    # tree that is not its commit gives no verdict, whoever left the files there.
    stray = report_mod.unmeasurable(workflow, Path(str(task["trtllm_repo_path"])), "")
    if stray:
        said = (
            f"the checkout is not its commit: {stray}. A check that runs in it gives no "
            f"verdict until they are committed, removed or git-ignored"
        )
        workflow.journal.event("checkout", said)
        print_message(f"[bold yellow]! {escape_markup(said)}[/bold yellow]", log)
    # Where this run's commands land, from the resolved environment; an injected runner wins.
    workflow._command_executor = select_command_executor(result.environment)
    if workflow._shell_runner_override is None:
        workflow._shell_runner = shell_runner_for(workflow._command_executor)
    # A dry run proves the agents too: whether turns work is what it asks.
    if os.environ.get(agent_probe.AGENT_PROBE_SKIP_ENV, "").strip() not in ("", "0"):
        workflow.journal.event("agent probe not run", f"${agent_probe.AGENT_PROBE_SKIP_ENV} is set")
    else:
        agents = agent_probe.distinct_agents(task, resolved=workflow._role_agents())
        repo, head = str(task.get("trtllm_repo_path") or ""), workflow._head()
        outcomes = agent_probe.probe_agents(agents, workflow.workspace, repo=repo, head=head)
        workflow._foreign_skills = agent_probe.foreign_skills(outcomes)
        if not all(outcome.ok for outcome in outcomes):
            return stop("agent probe failed", agent_probe.format_failures(outcomes))
        proven = "; ".join(o.describe() for o in outcomes)
        workflow.journal.event("agent probe passed", proven)
        print_message(f"[bold green]✔ agent probe[/bold green] {escape_markup(proven)}", log)
    workflow._contract = contract_mod.compile_contract(task)
    # Taken on a dry run too: one read-only process, no MPI world. Never fatal.
    inventory = run_inventory_probe(task, result.environment, shell_runner=workflow._shell_runner)
    workflow._surface_block = inventory.block
    (workflow.workspace / "surface.md").write_text(inventory.surface, encoding="utf-8")
    workflow.journal.section("surface survey", f"{inventory.headline}\n\nFull survey: `surface.md`")
    print_message(f"survey: {escape_markup(inventory.headline)}", log)
    gate_cli.install_cli(workflow.workspace)
    _write_host_facts(workflow, result)
    owed = [f"- `{check.key}`" for check in workflow._contract.checks]
    owed += [f"- note: {note}" for note in workflow._contract.notes]
    workflow.journal.section(f"checks owed: {len(workflow._contract.checks)}", "\n".join(owed))
    sources = {
        key: f"{a.kind}" + (f" {a.value:g}" if a.value is not None else "") + f": {a.source}"
        for key, a in result.anchors.items()
    }
    moved = anchors_mod.differences(workflow.workspace, task)
    lines = [deployments_mod.describe(found, sources)]
    body = "\n".join(lines + [f"- differs from the tree now: {line}" for line in moved])
    workflow.journal.section(f"deployments: {len(found)}", body)
    print_message(f"deployments ({len(found)}):\n{escape_markup(body)}")
    state.libraries = state.libraries or contract_mod.library_digest(task["trtllm_repo_path"])
    workflow._checkpoint(state)
    return True


def _write_host_facts(workflow, result: PreflightResult) -> None:
    """Seed ``host_facts.md`` from what this run knows; what a role appended below the marker stays."""
    env = result.environment
    head = prompts.render(
        "host_facts.md.j2",
        max_tmpdir=MAX_TMPDIR_LEN,
        tmpdir=str((env.env or {}).get("TMPDIR") or ""),
        ranks=env.ntasks if env.needs_llmapi_launch else 0,
        container_image=env.container_image,
        notes=result.notes,
        marker=HOST_FACTS_APPEND_MARKER,
    )
    path, appended = workflow.host_facts_path, ""
    if path.is_file() and HOST_FACTS_APPEND_MARKER in path.read_text(encoding="utf-8"):
        text = path.read_text(encoding="utf-8")
        appended = text.split(HOST_FACTS_APPEND_MARKER, 1)[1].lstrip("\n")
    path.write_text(head + appended, encoding="utf-8")
