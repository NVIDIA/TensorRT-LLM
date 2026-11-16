# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight: the checks, the environment, the agent probe, host facts and the surface survey.

The allocation probe (the pool's own, for the widest shape the spec needs, with no
model loaded) runs beside the run's first turns: its failure stops the run, which
stays open for the same command to resume.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from rich.markup import escape as escape_markup

from agent_flow.console import print_message, print_rule
from agent_flow.prompts import dump_prompt_bundle

from ...agents.prompts import PROMPTS_DIRNAME, build_trtllm_modeling_v2_prompts
from ...agents.roles import packaged_skills
from ...agents.turns import TURNS_DIRNAME
from ...checks import gate_cli
from ...infra.compute import controller as controller_mod
from ...infra.compute import pool as pool_mod
from ...infra.compute.environment import MAX_TMPDIR_LEN, shares_allocation_with_spool
from ...infra.compute.executor import select_command_executor, shell_runner_for
from ...infra.files import read_json, write_json_atomic
from ...infra.git import CommitPolicy
from ...operators import prior_art, tree_anchors
from ...operators.inventory import Inventory, format_surface, run_inventory_probe
from ...spec.task_schema import assembly_units
from ..records.state import WorkflowState
from ..validation import contract as contract_mod
from ..workspace import integration as trees_mod
from . import agent_probe
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

#: How long a recorded substrate pass spares a resumed run the gate, in seconds.
SUBSTRATE_FRESH_S = 12 * 3600.0


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

    With ``stop`` the allocation probe runs beside what follows
    (:func:`_prove_allocation_beside`) and its failure calls it; without, it
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
        # A failed preflight never reaches the report that names what the run launches with.
        print_message(escape_markup(pool_mod.said_kind(workflow.task_data.get("environment"))), log)
        if hint:
            print_message(hint, log)
        workflow.journal.open(slug="(preflight failed)")
        workflow.journal.section("preflight failed", f"{exc}\n\n{hint}".rstrip())
        workflow._not_started = f"preflight: {str(exc).strip().splitlines()[0][:300]}"
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
    # The benchmarks' datasets, staged beside the first turns: never a blocker.
    from . import datasets as datasets_mod

    datasets_mod.start(
        workflow.task_data,
        workflow.workspace,
        python=getattr(result.environment, "python", None),
        note=lambda said: workflow.journal.event("datasets", said),
    )
    print_message(f"[bold green]✔ preflight[/bold green]\n{format_preflight_report(result)}", log)

    # Where this run's commands land, decided once, from the resolved
    # environment. An explicitly injected runner still wins.
    workflow._command_executor = select_command_executor(
        result.environment,
        # The checkout under test: gates run in it and the session refuses
        # a result whose ranks imported the engine from anywhere else.
        tree=workflow.task_data.get("trtllm_repo_path"),
        # What a command that names no member of its own asks for: the run's width.
        gpus=result.environment.world_size,
        note=lambda said: workflow.journal.event("waiting", said),
    )
    if workflow._shell_runner_override is None:
        workflow._shell_runner = shell_runner_for(workflow._command_executor)

    # The first holder is asked for before any agent turn: a refusal stops the start here.
    if not _start_pool(workflow, log):
        workflow._not_started = "the pool could not start (its line above says why)"
        return False

    # Legal configurations that cost hours: notes, not errors.
    for warning in (shares_allocation_with_spool(result.environment),):
        if warning:
            result.notes.append(warning)
            print_message(f"[bold yellow]⚠ {warning}[/bold yellow]", log)

    # A dry run proves the agents too: whether turns work is what it asks.
    if not _prove_agents(workflow, log):
        workflow._not_started = "the agent probe: an agent of the spec cannot run a turn"
        return False
    if not _check_processor(workflow, result, log):
        workflow._not_started = "the checkpoint's processor does not check out"
        return False
    workflow._contract = contract_mod.compile_contract(
        workflow.task_data, workflow._gate_env().world_size
    )

    # No stock model is loaded to prove the host: the pool's model-free probe runs for the
    # widest shape the spec needs. Skipped on a dry run, which only asks whether the spec
    # resolves.
    if not workflow.dry_run:
        if stop is not None:
            _prove_allocation_beside(workflow, result, log, stop)
        elif not _prove_allocation(workflow, result, log):
            workflow._not_started = "the allocation probe did not pass"
            return False

    _take_survey(workflow, result, log)

    gate_cli.install_cli(workflow.workspace)

    # Prompts can only be built now: every role's prompt carries the
    # resolved launch prefix, which did not exist until the device claim
    # was made.
    if workflow._prompts_override is None:
        workflow.prompts = build_trtllm_modeling_v2_prompts(
            environment=result.environment,
            scope=_scope(workflow.task_data, result.checkpoint_config),
            commit_policy=CommitPolicy.from_task(workflow.task_data, workspace=workflow.workspace),
            notes=str(workflow.task_data.get("notes") or ""),
        )
    # What every role is told, verbatim and per launch.
    dump_prompt_bundle(workflow.prompts, workflow.workspace / PROMPTS_DIRNAME)
    _write_host_facts(workflow, result)
    workflow._materialize_skills()
    _write_contract(workflow)
    _say_advice(workflow, log)
    workflow._checkpoint(state)
    return True


def pool_demand(task_data: Mapping[str, Any]) -> dict[int, float]:
    """What an open run owes its pool: its base shape, and the longest wall a gate of it states.

    One node. A wider shape is asked for only while a job of that shape is in the spool
    (``pool.spool_demand`` adds it), so a wide holder is not renewed once nothing needs it.
    """
    from .preflight import gate_limits

    return {1: float(max(gate_limits(task_data).values(), default=0))}


def run_demand(workflow) -> dict[int, float]:
    """What the run owes its pool now: the shapes its ready and running tasks need.

    Once the run has a task list (``GraphOrchestrator.demand``): one node for a row's task
    and the kernel fan-out, a deployment's own shape while its assembly or its session is
    due, nothing for work that is blocked or done. Before it has one (the start, the
    allocation probe), :func:`pool_demand`.
    """
    tasks = getattr(workflow, "_task_demand", None)
    if tasks is None:
        return pool_demand(workflow.task_data)
    return {int(nodes): float(wall) for nodes, wall in dict(tasks()).items()}


def _start_pool(workflow, log) -> bool:
    """Get the run its first holder. False, having said why, when the pool cannot start.

    With an injected runner no command reaches the spool, so nothing is asked for. A dry run
    asks only whether the environment resolves: the entries are rendered, nothing is submitted.
    """
    workflow._pool = workflow._pool_keeper = None
    if workflow._shell_runner_override is not None:
        return True
    try:
        pool = pool_mod.build(
            workflow.task_data,
            workspace=workflow.workspace,
            note=lambda said: workflow.journal.event("pool", said),
        )
        if workflow.dry_run:
            return True
        _state_trees(workflow)

        def demand() -> dict[int, float]:
            # Through the pool's own demand function: what the run's tasks need, and what
            # is in the spool.
            return pool_mod.spool_demand(pool.paths.root, run_demand(workflow))

        pool.start(demand())
        so_far = pool.asked()
        if so_far.bound is not None:
            # The bound of what the run may ask its scheduler for, said at every start.
            print_message(f"→ {escape_markup(so_far.render())}", log)
        if pool.exhausted:
            # At its bound with no holder: nothing of this run can run. To be continued.
            workflow._budget_stop = True
            print_message(
                f"[bold red]✘ POOL_CAPACITY  {escape_markup(pool.exhausted)}[/bold red]", log
            )
            workflow.journal.event("pool cannot start", pool.exhausted)
            return False
    except pool_mod.PoolError as exc:
        for line in str(exc).splitlines():
            print_message(f"[bold red]✘ {escape_markup(line)}[/bold red]", log)
        workflow.journal.event("pool cannot start", str(exc))
        return False

    def out_of_bound(why: str) -> bool:
        from .. import orchestrator as orchestrator_mod

        workflow._budget_stop = True
        workflow.journal.event("pool", f"the run stops at its bound: {why}")
        return orchestrator_mod.stop_run(
            workflow.workspace,
            "the run's bound of node-hours is reached and no holder is left "
            "(POOL_CAPACITY): stopped at the bound",
        )

    def to_be_continued(why: str) -> bool:
        # Nothing of the run runs or is ready any more, and what waits in the spool cannot
        # start before this host's job ends: the controller says so now, not at its limit.
        if run_demand(workflow):
            return False
        workflow.journal.event("pool", f"to be continued: {why}")
        return controller_mod.continue_now(workflow.workspace, why)

    workflow._pool = pool
    workflow._pool_keeper = pool_mod.Keeper(
        pool, demand, on_exhausted=out_of_bound, on_late=to_be_continued
    ).start()
    return True


def end_pool(workflow, state: WorkflowState, log) -> None:
    """Stop renewing. A closed run ends its pool; a local watcher never outlives the process.

    A run that stops short of its close leaves its holders: the command that resumes it finds
    them, and `trtllm-modeling-v2 pool stop` ends them. But a start that did not start
    inside a job of its own ends them: nobody types that command again there, and the first
    holder would idle to its limit.
    """
    keeper, pool = getattr(workflow, "_pool_keeper", None), getattr(workflow, "_pool", None)
    workflow._pool = workflow._pool_keeper = None
    if keeper is not None:
        keeper.stop()
    if pool is None:
        return
    local = pool.env.kind == pool_mod.KIND_LOCAL
    unattended = bool(getattr(workflow, "_not_started", "") and os.environ.get("SLURM_JOB_ID"))
    if unattended and not (state.done or local):
        print_message(
            "[bold cyan]→ the start did not start, inside a job of its own: no command "
            "resumes on its holder(s), so they are ended[/bold cyan]",
            log,
        )
    if not (state.done or local or unattended):
        held = (read_json(pool.paths.pool) or {}).get("holders") or ()
        if held:
            names = ", ".join(str(h.get("name")) for h in held)
            print_message(
                f"[bold cyan]→ the run is not closed: its holder(s) stay ({escape_markup(names)}); "
                f"the same command resumes on them, and `trtllm-modeling-v2 pool stop "
                f"--workspace {workflow.workspace}` ends them[/bold cyan]",
                log,
            )
        return
    for line in pool.done():
        print_message(
            f"[bold yellow]⚠ {escape_markup(line)}: look with `trtllm-modeling-v2 pool status "
            f"--workspace {workflow.workspace}`[/bold yellow]",
            log,
        )


#: What a dry run says in place of a measurement that only a job of the run's spool takes.
DRY_RUN_NOT_TAKEN = (
    "not taken: a dry run starts no holder, and the build's inventory (like the load of a "
    "multimodal checkpoint's processor) is a job of the run's spool. The first start takes it"
)


def _no_holder(workflow) -> bool:
    """Whether a CPU probe would wait for a holder nobody asked for: a dry run starts none."""
    return bool(workflow.dry_run) and workflow._shell_runner_override is None


def _cpu_probe_runner(workflow, result: PreflightResult):
    """Run build and processor introspection in the session without reserving a GPU."""
    if workflow._shell_runner_override is not None:
        return workflow._shell_runner_override
    return shell_runner_for(
        select_command_executor(
            result.environment,
            name_prefix="preflight-cpu",
            tree=workflow.task_data.get("trtllm_repo_path"),
            gpus="0",
        )
    )


def _check_processor(workflow, result: PreflightResult, log) -> bool:
    """Load a multimodal checkpoint's processor under the run's interpreter; False on a missing import."""
    if _no_holder(workflow):
        return True  # said once, with the survey (:data:`DRY_RUN_NOT_TAKEN`)
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


def allocation_shape(task_data: Mapping[str, Any], per_node: int | None) -> tuple[int, int]:
    """``(ranks, nodes)`` of the probe: the spec's widest deployment, whole nodes past one."""
    from ...spec.task_schema import widest_world_size

    widest = max(1, int(widest_world_size(task_data) or 1))
    if not per_node or widest <= per_node:
        return widest, 1
    nodes = -(-widest // per_node)
    return nodes * per_node, nodes


def _prove_allocation(workflow, result, log) -> bool:
    """The model-free probe of the widest shape the spec needs, before the first session.

    One world as wide as the spec's widest deployment (whole nodes, when it is wider than
    one): launch, import, a collective, the cache directories, the dataset. It loads no
    model and takes the place of the stock substrate load. What it finds is said and
    journaled: it is advice, and each deployment's own session judges its launch. It
    decides one thing on its own: a watcher that cannot read its devices stops the start,
    because no GPU job would ever start under it.

    Once a workspace and a shape: a pass is kept, and a later start of the same
    workspace asks again only when the spec's widest deployment changed, or the queue
    the shape is asked of (the account, the partition, the QoS).
    What it is for is said before hours of agents' work, not at every start: a second
    run asked for five nodes three times, and none of the probes stopped anything.
    """
    from ...infra.compute import probe as probe_mod
    from . import datasets as datasets_mod

    if getattr(workflow, "_pool", None) is None:
        return True  # an injected runner: no spool job is run
    environment = result.environment
    ranks, nodes = allocation_shape(workflow.task_data, environment.gpus_per_node)
    kept = workflow.workspace / "alloc_probe" / "pass.json"
    slurm = (workflow.task_data.get("environment") or {}).get("slurm") or {}
    asked = {
        "ranks": ranks,
        "nodes": nodes,
        "queue": [str(slurm.get(key) or "") for key in ("account", "partition", "qos")],
    }
    before = read_json(kept) or {}
    if all(before.get(key) == value for key, value in asked.items()):
        said = (
            f"allocation probe not run: {ranks} rank(s) on {nodes} node(s) passed in this "
            f"workspace before (`{kept}`)"
        )
        workflow.journal.event("allocation probe", said)
        print_message(f"[bold green]✔[/bold green] {escape_markup(said)}", log)
        return True
    caches = [
        str(environment.env[key])
        for key in ("TMPDIR", "HF_HOME", "TRITON_CACHE_DIR")
        if environment.env.get(key)
    ]
    staged = [
        str(entry["path"])
        for entry in datasets_mod.status(workflow.workspace).values()
        if entry.get("path") and Path(str(entry["path"])).exists()
    ]
    tree = workflow.task_data.get("trtllm_repo_path")
    found = probe_mod.run_allocation_probe(
        environment,
        ranks=ranks,
        nodes=nodes,
        out=workflow.workspace / "alloc_probe",
        cache_dirs=caches,
        datasets=staged,
        tree=tree,
        # The probe's job names its pin, as every job does: the checkout as it stands now,
        # pinned by the run's trees (none for a directory nothing can pin).
        pin=_pin_of(tree),
        note=lambda said: workflow.journal.event("waiting", said),
    )
    unread = probe_mod.watchers_without_reading(environment.spool)
    workflow.journal.event("allocation probe", found.render())
    if unread:
        for line in unread:
            print_message(f"[bold red]✘ {escape_markup(line)}[/bold red]", log)
        return False
    if found.ok:
        kept.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(kept, {**asked, "at": time.time()})
    mark = "[bold green]✔[/bold green]" if found.ok else "[bold yellow]⚠[/bold yellow]"
    print_message(f"{mark} {escape_markup(found.render())}", log)
    return True


def _state_trees(workflow) -> None:
    """Say in the checkout where this run's trees go, before its first job pins the checkout.

    The jobs of the start pin the checkout by what it says there, and the run's own trees are
    made after them. A check started by hand in a directory of its own leaves its place: the
    start's jobs would pin under that directory, which may be gone.
    """
    repo = Path(str(workflow.task_data.get("trtllm_repo_path") or ""))
    if (repo / ".git").exists():
        trees_mod.make(workflow.task_data, repo=repo, workspace=workflow.workspace, prefix="mv2")


def _pin_of(tree: Any) -> str | None:
    """The pin of ``tree`` as it stands now, for a job that is to run it; ``None`` when nothing pins it."""
    from ...infra.compute import pintree

    if not tree:
        return None
    try:
        return pintree.pin_tree(str(tree), "alloc-probe")
    except (pintree.PinError, OSError):
        return None


def _prove_allocation_beside(workflow, result, log, stop: Callable[[str], None]) -> None:
    """Run the allocation probe on a thread of its own, while the run's first turns start.

    A probe that stops the start (:func:`_prove_allocation`), or an error that kept it
    from an answer, calls ``stop`` with why. The first session waits for it
    (:func:`_await_substrate`).
    """

    def prove() -> None:
        try:
            passed = _prove_allocation(workflow, result, log)
        except Exception as exc:  # the run waits on this thread's answer, whatever it hit
            workflow.journal.section("allocation probe failed", f"{type(exc).__name__}: {exc}")
            passed = False
        if not passed:
            stop("the allocation probe stopped the start; the journal has its output")

    workflow._substrate_thread = threading.Thread(target=prove, name="mv2-allocation", daemon=True)
    workflow._substrate_thread.start()


def _await_substrate(workflow) -> None:
    """Wait for the allocation probe that runs beside the run's first turns, if one does."""
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


def _write_contract(workflow) -> None:
    """Write the contract the start compiled, and say what it holds."""
    contract_mod.write_contract(workflow.workspace, workflow._contract)
    workflow.journal.section(
        f"contract: {len(workflow._contract.required())} required check(s)",
        workflow._contract.render(),
    )
    _freeze_anchors(workflow)
    _write_intake_advice(workflow)


def _scope(task_data: Mapping[str, Any], checkpoint_config: Mapping[str, Any]) -> str:
    """``text_only`` for a checkpoint with a tower that no deployment sends an image to."""
    from ...spec import deployments as deployments_mod

    towers = {"vision_config", "audio_config"} & set(checkpoint_config or {})
    images = any("image" in d.inputs for d in deployments_mod.resolve(task_data))
    return "text_only" if towers and not images else "full"


def _freeze_anchors(workflow) -> None:
    """Freeze what each benchmark of each deployment is held to, the first time the run starts.

    Not on a dry run, which decides nothing a start would read as decided.

    What is frozen stays (``checks/anchors.py``): only ``resolve change-requirement``
    changes one, and records why. A file that cannot be read is journaled; the checks
    then resolve their anchors as the tree states them.
    """
    from ...checks import anchors as anchors_mod

    if workflow.dry_run:
        # The spec is still being written: a dry run with a placeholder anchor would
        # hold the run that follows to the placeholder.
        workflow.journal.section(
            "anchors: none frozen",
            "A dry run freezes none: the first start that is no dry run does, from the "
            "spec as it is then.",
        )
        return
    try:
        held = anchors_mod.freeze(workflow.workspace, workflow.task_data)
    except (anchors_mod.AnchorError, OSError) as exc:
        workflow.journal.section("anchors: not frozen", str(exc))
        return
    lines = [
        f"- `{a.key}`: {a.kind}"
        + (f" {a.value:g}" if a.value is not None else "")
        + f" ({a.source})"
        for a in held.values()
    ]
    changed = anchors_mod.changes(workflow.workspace)
    if changed:
        lines += ["", f"Changed by an explicit command {len(changed)} time(s):"]
        lines += [f"- `{c.get('anchor')}`: {c.get('reason')}" for c in changed]
    workflow.journal.section(f"anchors: {len(held)} frozen", "\n".join(lines) or "None.")


def _write_intake_advice(workflow) -> None:
    """Journal what the spec the run starts from is likely to cost or miss; advice never stops the run."""
    from ...spec import advice as advice_mod

    try:
        advice = advice_mod.advise(workflow.task_data)
    except Exception as exc:  # advice must not cost the run its start
        workflow.journal.section("intake advice: not read", f"{type(exc).__name__}: {exc}")
        return
    workflow._advice = advice
    workflow.journal.section(f"intake advice: {advice.summary()}", advice.render())


def _say_advice(workflow, log) -> None:
    """Print the spec's advice at the start, a dry run's included: said, and nothing is stopped."""
    advice = getattr(workflow, "_advice", None)
    for line in (*advice.warnings, *advice.constraints) if advice is not None else ():
        print_message(f"[yellow]advice:[/yellow] {escape_markup(line)}", log)
    # What the spec will cost in loads, and how to reach a first milestone early: one line each.
    for line in getattr(advice, "plan", ()) if advice is not None else ():
        print_message(f"[yellow]advice:[/yellow] {escape_markup(line)}", log)


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
        "What hosts this flow has run on do that a generic Linux box does not. **Read "
        "this before you debug an environment failure.** The facts below are known "
        "causes, each with when it applies: none was measured on this host unless it "
        "says so or stands under what preflight measured.",
        "",
        "For a failure, read the affected turn's activity log or the job's receipt and log. "
        "Report the exact error, command, commit and evidence paths.",
        "",
        "## Facts",
        "",
        "- **Where the engine's virtual environment lives inside the repository root, "
        "`pytest` exits 4 even when every test passed**: a session-final `sys.path` "
        "audit fires. If you see exit 4 with every test passed, export "
        "`TRTLLM_SKIP_SYS_PATH_CHECK=1` and judge on the summary line (`N passed`). "
        "Where the environment is elsewhere this does not happen: do nothing.",
        f"- **`TMPDIR` must be at most {MAX_TMPDIR_LEN} characters.** The LLM API's "
        "RPC binds a unix socket under it, the path is capped at 107 bytes and the "
        "socket basename takes 45. This run's `TMPDIR` is "
        + (f"`{tmpdir}` ({len(tmpdir)} characters)." if tmpdir else "not set in the spec."),
        "- **`UCX ERROR uct_iface_open(tcp/rdma_vf_rail0)` is noise.** UCX falls back "
        "and the run exits 0. Do not chase it; it is not why your job failed.",
        "- **`RC=$?` after a pipe is the *pipe's* exit code, not the command's.** "
        "`cmd | tee log; RC=$?` reports `tee`. Use `PIPESTATUS[0]`, or drop the pipe.",
        "- **A venv interpreter created with `--system-site-packages` resolves only "
        "inside the container it was created in.** Run from outside it, it finds an "
        "interpreter and none of the packages, which reads as a broken install: run "
        "the engine's Python in a job of the pool, where its environment is.",
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


#: What each anchor's definition found in the checkout under test, written at each start.
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
    """Read the engine's own model of this checkpoint's architecture, and the engine's limits for it.

    The section the survey dispatch carries, and the model as data: each row's card names
    it first. **Never fatal**: an unreadable config or tree leaves none.
    """
    repo = workflow.task_data.get("trtllm_repo_path")
    workflow._prior_art, workflow._engine_limits = None, []
    if config and repo:
        try:
            workflow._prior_art = prior_art.find_prior_art(repo, config)
            workflow._engine_limits = prior_art.engine_constraints(config, repo)
        except OSError:
            pass
    workflow._prior_art_block = prior_art.survey_block(workflow._prior_art, workflow._engine_limits)
    prior_art.save(workflow._prior_art, workflow.workspace / prior_art.PRIOR_ART_DATA)


def _take_tree_map(workflow) -> str:
    """Write ``tree_map.md``: what each anchor's definition found in the checkout under test.

    At each start (:mod:`.tree_anchors`): found by this definition, at this commit. Returns
    a line for the journal when a definition found nothing, else ``""``. **Never fatal**:
    a checkout it cannot read leaves no map.
    """
    repo = workflow.task_data.get("trtllm_repo_path")
    path = workflow.workspace / TREE_MAP_FILENAME
    if not repo:
        path.unlink(missing_ok=True)
        return ""
    try:
        found = list(tree_anchors.resolve_all(repo).values())
        path.write_text(tree_anchors.render_map(repo, found), encoding="utf-8")
    except OSError:
        path.unlink(missing_ok=True)
        return ""
    gone = [r.anchor.concept for r in found if not r.found]
    return f"tree map (missing: {', '.join(gone)}): `{TREE_MAP_FILENAME}`" if gone else ""


def _take_survey(workflow, result, log) -> None:
    """Measure what the shared catalog certifies on this build, for the survey's dispatch.

    **Never fatal.** A failed probe is written into the block the surveyor
    reads, and the run goes on. A dry run asks nothing of a worker: the probe is a job of
    the run's spool, a dry run starts no holder, so it is not taken and that is said.
    """
    if _no_holder(workflow):
        arch = str(workflow.task_data.get("arch", ""))
        inventory = Inventory(arch=arch, error=DRY_RUN_NOT_TAKEN)
    else:
        inventory = run_inventory_probe(
            workflow.task_data,
            result.environment,
            shell_runner=_cpu_probe_runner(workflow, result),
        )
    workflow._surface_block = format_surface(inventory)
    _take_prior_art(workflow, _checkpoint_config(workflow.task_data))
    tree_note = _take_tree_map(workflow)

    headline = (
        f"{len(inventory.spendable)}/{len(inventory.catalog)} entries certified on {inventory.arch}"
        if inventory.ok
        else inventory.error
        if inventory.error == DRY_RUN_NOT_TAKEN
        else f"inventory probe failed: {inventory.error.splitlines()[0]}"
    )
    workflow.journal.section(
        "surface survey",
        headline + (f"\n\n{tree_note}" if tree_note else ""),
    )
    # A probe that never ran must not read as an empty catalog.
    mark, colour = ("✔", "green") if inventory.ok else ("⚠", "yellow")
    print_message(f"[bold {colour}]{mark} survey[/bold {colour}] {headline}", log)
