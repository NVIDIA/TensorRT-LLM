# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One session per deployment: its servers loaded once, every check a client of them.

A session is one job with a stated device count and wall time. Inside it
``harness/session_driver.py`` starts one server per role instance on devices of its own
(and the proxy over ``ctx`` and ``gen``), then runs the deployment's checks against the
one endpoint, in the order :func:`plan` records:

==============  ===================================================================
``devices``     before any load: each server's devices, disjoint on a node
``attest``      every role, instance and rank ran the options it was given, from
                the tree it measures, on the devices it was given
``boot``        the boot workload's prompts come back completed, with their keywords
``api``         a chat completion and a streamed completion answer
``image``       when the deployment takes images: a chat completion with one
``chunk``       when asked: a prompt past ``max_num_tokens`` is served
``hop``         ``ctx`` and ``gen``: the generation side received the context's state,
                read as the measured tree defines it; else "not verified", failing nothing
``accuracy[b]`` each benchmark the endpoint evaluator carries, held to its anchor
``acceptance``  a drafter's accepted tokens per step, read from the answers
``reuse``       when asked: a prompt sent again resumes from the cache
==============  ===================================================================

``reuse`` changes the engine's state (its cache), so it runs after everything that
scores. Every check gets its own line (a
:class:`~.verdict.GateResult` under the deployment's shape), and the session one more
(:data:`GATE_NAME`): it passed when every check that ran passed and none was blocked. A
check whose anchor, dataset or protocol is missing is blocked before the load, with the
reason, and the others go on. A benchmark the endpoint evaluator cannot carry, or one the
spec marks ``through: eval``, is ``trtllm-eval`` in a load of its own after the session,
and its line says so; through ``ctx`` and ``gen`` it is blocked: an aggregated load would
not cross the hop.

A session wider than one node is placed by :func:`place` and run as ``checks/gang.py``
says: one job of whole nodes, one step per role instance, the same checks and records.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.compute.environment import EnvironmentError as NoStepLine
from ..infra.compute.environment import RunEnvironment
from ..infra.compute.scheduler import RESOLVE_HOOK_DIR
from ..infra.files import write_text_atomic
from ..spec.deployments import (
    MUTATING_PROBES,
    REQUIREMENT_OPTION,
    ROLES,
    SERVER,
    THROUGH_ENDPOINT,
    Benchmark,
    Deployment,
    ranks_of,
    requirement,
    requirement_said,
)
from . import accuracy_policy, attest, gang
from . import anchors as anchors_mod
from .boot import (
    DEFAULT_BOOT_IMAGE,
    DEFAULT_BOOT_IMAGE_KEYWORDS,
    DEFAULT_BOOT_IMAGE_PROMPT,
    BootPrompt,
    BootSpec,
    chunk_probe_tokens,
    chunk_shortfall,
    missing_keywords,
)
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    ShellRunner,
    default_shell_runner,
    gate_workdir,
    with_mode,
)
from .harness import endpoint_eval, session_driver, workload_io
from .verdict import GateResult, inspect_tree, verdict_factory

#: The session's own line in the ledger, and the gate the contract owes per deployment.
GATE_NAME = "deployment"
SESSION_DRIVER = Path(__file__).resolve().parent / "harness" / "session_driver.py"
ENDPOINT_EVAL = Path(__file__).resolve().parent / "harness" / "endpoint_eval.py"
#: The benchmarks the endpoint evaluator carries; any other is ``trtllm-eval``'s.
ENDPOINT_BENCHMARKS: tuple[str, ...] = tuple(sorted(endpoint_eval.EVALUATORS))
COMPLETIONS = "/v1/completions"
CHAT_COMPLETIONS = "/v1/chat/completions"
#: How many of the boot prompts the hop check sends again, to observe their transfer.
HOP_REQUESTS = 3
#: Seconds one endpoint benchmark adds to a session that states no ``timeout``.
BENCHMARK_TIMEOUT_S = 4 * 60 * 60
#: Seconds a session keeps for stopping its servers, past what its checks may take.
TEARDOWN_S = 600
#: What a session turns on in the servers it starts, to read what it must prove: what
#: each side of a hop did per request, and a drafter's accepted tokens per request.
HOP_OBSERVATION = {"return_perf_metrics": True, "perf_metrics_max_requests": 256}
DRAFT_OBSERVATION = {"per_request_spec_decode_stats": True}
#: The token budget of an engine whose options state none.
DEFAULT_MAX_NUM_TOKENS = 8192


class WiderThanOneNode(RuntimeError):
    """A deployment wider than one node that cannot be placed.

    No step crosses nodes in this environment, or a world of it fills no whole nodes.
    """


@dataclass(frozen=True)
class Placement:
    """What a session's one job asks of the pool."""

    gpus: int
    nodes: int = 1
    wall: int = BOOT_TIMEOUT_S
    #: Past one node: where each role instance's ranks run (``checks/gang.py``). Empty on
    #: one node, where the session's own process gives each server its devices.
    layout: Mapping[str, tuple[gang.Slot, ...]] = field(default_factory=dict)


def session_timeout(task_data: Mapping[str, Any], deployment: Deployment) -> int:
    """The seconds a session of ``deployment`` may take: its ``timeout``, else what its checks add up to."""
    if deployment.timeout:
        return int(deployment.timeout)
    accuracy = task_data.get("accuracy") or {}
    scored = sum(1 for b in deployment.benchmarks if _through_endpoint(deployment, b))
    return BOOT_TIMEOUT_S + scored * int(accuracy.get("timeout") or BENCHMARK_TIMEOUT_S)


def place(task_data: Mapping[str, Any], deployment: Deployment) -> Placement:
    """The one job a session of ``deployment`` runs as: its devices, on one node, and its wall time.

    A deployment that fits one node is one job of its devices on one node: the session's
    own process starts every role instance there (``harness/session_driver.py``). A wider
    one is one job of whole nodes, whose script the pool's lead runs once: one step per
    role instance, each on its slice of the reserved hosts (``checks/gang.py``). That
    needs an environment that can run a step across nodes (``environment.slurm``); without
    one a wider session is refused before any load.
    """
    from ..spec.task_schema import gpus_per_node

    need = deployment.ranks
    wall = session_timeout(task_data, deployment) + TEARDOWN_S
    per_node = gpus_per_node(task_data)
    if per_node is None or need <= per_node:
        return Placement(gpus=need, wall=wall)
    if not (task_data.get("environment") or {}).get("slurm"):
        from ..infra.compute.pool import OTHER_ENVIRONMENT

        raise WiderThanOneNode(
            f"`{deployment.name}` holds {need} devices and one node has {per_node}: a session "
            f"across nodes is one step per role instance on the reserved hosts, which needs "
            f"environment.slurm. {OTHER_ENVIRONMENT}"
        )
    try:
        nodes, layout = gang.plan_layout(deployment, per_node)
    except ValueError as exc:
        raise WiderThanOneNode(str(exc)) from exc
    return Placement(gpus=need, nodes=nodes, wall=wall, layout=layout)


# ------------------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class Planned:
    """One check of a session, in the order it runs."""

    name: str
    #: The driver steps it is judged from, in order; none for a check judged from the
    #: session's own records (``devices``, ``attest``).
    steps: tuple[Mapping[str, Any], ...] = ()
    #: Why it is not run; a blocked check has no steps.
    blocked: str = ""
    #: It changes the engine's state: nothing that scores runs after it begins.
    mutates: bool = False
    #: What its judging reads beside the records: the workload, the benchmark, the limits.
    reads: Mapping[str, Any] = field(default_factory=dict)


def endpoint_cannot(benchmark: Benchmark) -> str:
    """Why the endpoint evaluator cannot score ``benchmark`` as the spec states it; ``""`` when it can.

    Its name and its arguments both decide: what the evaluator's own parser does not take
    is a protocol it does not carry (``harness/endpoint_eval.py:not_carried``).
    """
    return endpoint_eval.not_carried(benchmark.name, list(benchmark.args))


def _through_endpoint(deployment: Deployment, benchmark: Benchmark) -> bool:
    # The endpoint is the default judge of what it can carry. Its score has been compared
    # with `trtllm-eval`'s on one small model only: the parity run on a mid-size model is
    # still owed (said where the default is documented: README.md, spec/INTAKE.md).
    return benchmark.through == THROUGH_ENDPOINT and not endpoint_cannot(benchmark)


def _completion(model: str, rid: str, prompt: Any, max_tokens: int, **extra: Any) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": int(max_tokens), "temperature": 0.0}
    return {"id": rid, "endpoint": COMPLETIONS, "body": {**body, **extra}}


def _chat(model: str, rid: str, content: Any, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
    }
    return {"id": rid, "endpoint": CHAT_COMPLETIONS, "body": body}


def limits_of(deployment: Deployment) -> dict[str, Any]:
    """The request limits of the engine that takes a prompt in: its token budget and bound.

    The context side of a disaggregated deployment, which is where a long prompt is split.
    """
    options = (deployment.ctx or deployment.server).options
    return {
        "max_num_tokens": int(options.get("max_num_tokens") or DEFAULT_MAX_NUM_TOKENS),
        "max_seq_len": options.get("max_seq_len"),
        "enable_chunked_prefill": bool(options.get("enable_chunked_prefill", False)),
    }


def boot_spec(task_data: Mapping[str, Any], deployment: Deployment) -> BootSpec:
    """The boot workload of ``deployment``: ``accuracy.boot`` with its own entry over it.

    A deployment that takes images and whose workload sends none gets the default image
    prompt appended.
    """
    from .boot import resolve_boot_spec

    spec = resolve_boot_spec(task_data, deployment.name)
    if "image" in deployment.inputs and not spec.has_images:
        image = BootPrompt(
            prompt=DEFAULT_BOOT_IMAGE_PROMPT,
            keywords_any=DEFAULT_BOOT_IMAGE_KEYWORDS,
            images=(DEFAULT_BOOT_IMAGE,),
        )
        spec = replace(spec, items=(*spec.items, image))
    return spec


def plan(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    *,
    directory: str | Path,
    workspace: str | Path | None = None,
    relative_scores: Mapping[str, float | None] | None = None,
    dataset_paths: Mapping[str, str] | None = None,
) -> list[Planned]:
    """Every check of one session of ``deployment``, in the order it runs.

    ``relative_scores``: for a benchmark judged against ``relative_to``'s own score, that
    score on this tree (absent or ``None``: not on record, so the benchmark is blocked).
    ``dataset_paths``: where a benchmark's dataset was staged, by benchmark.
    """
    model = str(task_data["checkpoint_path"])
    spec = boot_spec(task_data, deployment)
    asked = deployment.asked()
    found: list[Planned] = [Planned("devices"), Planned("attest")]

    # Boot: every prompt without an image, as a completion.
    requests = spec.requests(task_data.get("trtllm_repo_path"))
    text = [
        _completion(model, r["id"], r["prompt"], spec.max_tokens)
        for r in requests
        if not r.get("images")
    ]
    found.append(
        Planned(
            "boot", ({"id": "boot", "kind": "requests", "requests": text},), reads={"spec": spec}
        )
    )
    probe = spec.prompts[0] if spec.prompts else "The capital of France is"
    found.append(
        Planned(
            "api",
            (
                {
                    "id": "api.chat",
                    "kind": "requests",
                    "requests": [_chat(model, "chat", probe, spec.max_tokens)],
                },
                {
                    "id": "api.stream",
                    "kind": "stream",
                    "request": _completion(model, "stream", probe, spec.max_tokens),
                },
            ),
        )
    )
    if "image" in deployment.inputs:
        raw = [r["id"] for r in requests if r.get("images") and r.get("render") == "raw"]
        sent = [
            _chat(
                model,
                r["id"],
                [
                    {"type": "text", "text": r["prompt"]},
                    *({"type": "image_url", "image_url": {"url": i}} for i in r["images"]),
                ],
                spec.max_tokens,
            )
            for r in requests
            if r.get("images")
        ]
        blocked = (
            f"prompt(s) {', '.join(raw)} state `render: raw`: the endpoint applies the chat "
            f"template and places the images itself, so a prompt rendered by hand cannot be "
            f"sent as it is; write the question in `prompt` and drop `render`"
            if raw
            else ""
        )
        steps = () if blocked else ({"id": "image", "kind": "requests", "requests": sent},)
        found.append(Planned("image", steps, blocked=blocked, reads={"spec": spec}))
    if "chunk" in asked:
        # A prompt is split where it is taken in: by the context side of a hop.
        taking = limits_of(deployment)
        side = " of the context side" if deployment.server is None else ""
        length = chunk_probe_tokens(taking["max_num_tokens"], taking["max_seq_len"])
        blocked = ""
        if not taking["enable_chunked_prefill"]:
            blocked = (
                f"the options{side} do not turn chunked prefill on, so no prompt would be split"
            )
        elif length is None:
            blocked = (
                f"the engine's limits{side} (max_num_tokens {taking['max_num_tokens']}, "
                f"max_seq_len {taking['max_seq_len']}) leave no room for a prompt it would have "
                f"to split"
            )
        tokens = [1000 + i % 997 for i in range(length or 0)]
        steps = (
            ()
            if blocked
            else (
                {
                    "id": "chunk",
                    "kind": "requests",
                    "requests": [_completion(model, "chunk", tokens, 1)],
                },
            )
        )
        found.append(Planned("chunk", steps, blocked=blocked, reads={"limits": taking}))
    if deployment.server is None:
        # Its own requests through the proxy: how their transfer can be observed is found in
        # the measured tree by the session's job (`session_driver.hop_means`).
        crossing = [{**request, "id": f"hop.{request['id']}"} for request in text[:HOP_REQUESTS]]
        found.append(Planned("hop", ({"id": "hop", "kind": "hop", "requests": crossing},)))

    # Scoring: nothing that changes the engine's state has run yet.
    for benchmark in deployment.benchmarks:
        found.append(
            _planned_benchmark(
                task_data,
                deployment,
                benchmark,
                directory=Path(directory),
                workspace=workspace,
                relative_scores=relative_scores or {},
                dataset_path=(dataset_paths or {}).get(benchmark.name) or benchmark.dataset_path,
            )
        )
    if deployment.drafts:
        found.append(_planned_acceptance(task_data, deployment, model))

    # The probe that changes the engine's state, after everything that scores.
    if "reuse" in asked:
        options = (deployment.gen or deployment.server).options
        sends = attest.reuse_sends(options)
        requests = [_completion(model, f"reuse{n}", attest.REUSE_PROMPT, 1) for n in range(sends)]
        found.append(
            Planned(
                "reuse",
                ({"id": "reuse", "kind": "requests", "requests": requests},),
                mutates=True,
            )
        )
    assert all(p.mutates == (p.name in MUTATING_PROBES) for p in found)
    return found


def _planned_benchmark(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    benchmark: Benchmark,
    *,
    directory: Path,
    workspace: str | Path | None,
    relative_scores: Mapping[str, float | None],
    dataset_path: str | None,
) -> Planned:
    """One benchmark of the session: its step, or why it is blocked before the load."""
    name = f"accuracy[{benchmark.name}]"
    anchor = anchors_mod.frozen(workspace, deployment.name, benchmark.name) or anchors_mod.resolve(
        task_data, deployment, benchmark
    )
    reads = {"benchmark": benchmark, "anchor": anchor}
    if anchor.kind == anchors_mod.NONE:
        return Planned(
            name,
            blocked=f"no anchor to hold `{benchmark.name}` to ({anchor.source}): declare the "
            f"benchmark's `anchor` with its `source`, add the reference row, name the "
            f"deployment it is `relative_to`, or change the requirement (`resolve "
            f"change-requirement ... --anchor`)",
            reads=reads,
        )
    if anchor.kind == anchors_mod.RELATIVE:
        score = relative_scores.get(benchmark.name)
        if score is None:
            return Planned(
                name,
                blocked=f"its anchor is the score of `{anchor.relative_to}` on `{benchmark.name}`, "
                f"and that score is not on record for this tree: in a run, run `gate deployment "
                f"--deployment {anchor.relative_to}` on the same tree first (its result is "
                f"recorded in the run's ledger); outside a run, declare the benchmark's `anchor`",
                reads=reads,
            )
        reads["relative_score"] = float(score)
    if dataset_path and not Path(dataset_path).exists():
        return Planned(
            name,
            blocked=f"the dataset of `{benchmark.name}` is not at {dataset_path}, and "
            f"accuracy runs offline: put it there or state the benchmark's `dataset_path`",
            reads=reads,
        )
    # What the session's job asks on its node before it starts a server: whether the tree's
    # evaluator can load the benchmark's data where it will look for it.
    where = ["--dataset_path", str(dataset_path)] if dataset_path else []
    needs = ["{python}", str(ENDPOINT_EVAL), "--dataset_only", "--benchmark", benchmark.name]
    needs += where
    env = {**benchmark.env, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    if not _through_endpoint(deployment, benchmark):
        # Why the session's endpoint is not what scores it: the spec's own word, or what
        # the endpoint evaluator cannot carry (the benchmark, or its arguments).
        why = (
            "the spec marks it `through: eval`"
            if benchmark.through != THROUGH_ENDPOINT
            else f"the endpoint evaluator cannot carry it: {endpoint_cannot(benchmark)}"
        )
        reads["own_load_why"] = why
        if deployment.server is None:
            return Planned(
                name,
                blocked=f"{why}, and `trtllm-eval` loads one aggregated engine, which never "
                f"crosses the hop: score it on an aggregated deployment, or state the "
                f"benchmark as the endpoint evaluator takes it",
                reads=reads,
            )
        try:
            wide = place(task_data, deployment).nodes > 1
        except WiderThanOneNode:
            wide = True
        if wide:
            return Planned(
                name,
                blocked=f"`trtllm-eval` loads its engine on one node, and `{deployment.name}` "
                f"holds {deployment.ranks} devices, more than a node has: a deployment across "
                f"nodes is scored through its endpoint ({', '.join(ENDPOINT_BENCHMARKS)})",
                reads=reads,
            )
        # Its own load, after the session (see `run_session`). `trtllm-eval` builds the
        # tree's evaluator after that load: the session asks its loader before its own.
        # Only where this evaluator is the one it builds, and its own arguments (which
        # this parser may not take) name no other dataset.
        asked = ({"id": name, "kind": "needs", "needs": needs, "env": env},)
        known = benchmark.name in endpoint_eval.EVALUATORS and not any(
            str(word).startswith("--dataset_path") for word in benchmark.args
        )
        reads = {**reads, "own_load": True, "dataset_path": dataset_path}
        return Planned(name, asked if known else (), reads=reads)
    log = directory / f"accuracy.{benchmark.name}.log"
    argv = ["{python}", str(ENDPOINT_EVAL), "--url", "{url}", "--model", "{model}"]
    argv += ["--benchmark", benchmark.name, *where, *benchmark.args]
    step = {"id": name, "kind": "command", "argv": argv, "log": str(log), "env": env}
    # Asked with the benchmark's own arguments: they may name the dataset.
    step["needs"] = [*needs, *benchmark.args]
    return Planned(name, (step,), reads={**reads, "log": str(log)})


def _planned_acceptance(
    task_data: Mapping[str, Any], deployment: Deployment, model: str
) -> Planned:
    """A drafter's acceptance, read from the session's own answers."""
    from . import acceptance

    accuracy = task_data.get("accuracy") or {}
    try:
        workload = acceptance.resolve_acceptance_workload(accuracy, deployment.name)
    except ValueError as exc:
        return Planned("acceptance", blocked=f"its workload cannot be read: {exc}")
    reads: dict[str, Any] = {"workload": workload}
    source = str((workload.dataset or {}).get("source") or "")
    if source and (Path(source).is_absolute() or source.startswith(".")):
        # A dataset named by its path is either there or not: said now, before any load.
        if not Path(source).exists():
            return Planned(
                "acceptance",
                blocked=f"the dataset of its workload is not at {source}: put it there, or "
                f"state `accuracy.acceptance.dataset` (a path, or a cached dataset's id)",
                reads=reads,
            )
    requests = [
        _completion(model, r["id"], r["prompt"], r["max_tokens"]) for r in workload.requests()
    ]
    step: dict[str, Any] = {"id": "acceptance", "kind": "requests", "requests": requests}
    if workload.dataset:
        step["dataset"] = dict(workload.dataset)
        step["body"] = _completion(model, "", "", workload.max_tokens)["body"]
    return Planned("acceptance", (step,), reads=reads)


# --------------------------------------------------------------------------- the session


@dataclass(frozen=True)
class SessionResult:
    """What one session measured: a line per check, in order, and the session's own."""

    lines: tuple[GateResult, ...]
    result: GateResult
    #: How many engines the session started: one per role instance.
    loads: int = 0
    #: The benchmarks left for ``trtllm-eval`` in a load of their own, after the session.
    own_loads: tuple[Planned, ...] = ()

    def render(self) -> str:
        return "\n".join(line.summary() for line in (*self.lines, self.result))


def write_options(deployment: Deployment, directory: str | Path) -> dict[str, Path]:
    """Write each role's options where ``trtllm-serve --config`` reads them; the files by role.

    The options as the spec states them, the Modeling V2 requirement (what they state,
    else ``require``: ``spec.deployments.requirement``), and what the session turns on to
    read what it must prove (:data:`HOP_OBSERVATION`, :data:`DRAFT_OBSERVATION`).
    """
    root = gate_workdir(directory)
    paths: dict[str, Path] = {}
    for role, server in deployment.servers.items():
        written = {**server.options, REQUIREMENT_OPTION: requirement(server.options)}
        if deployment.server is None:
            written = {**HOP_OBSERVATION, **written}
        if (server.options.get("speculative_config") or {}).get("decoding_type"):
            written = {**DRAFT_OBSERVATION, **written}
        path = root / ("options.yaml" if role == SERVER else f"options.{role}.yaml")
        write_text_atomic(path, yaml.safe_dump(written, sort_keys=False))
        paths[role] = path
    return paths


def driver_spec(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    planned: Sequence[Planned],
    *,
    directory: Path,
    timeout: int,
) -> dict[str, Any]:
    """The JSON the session driver reads: the servers to start and the steps to run, in order."""
    options = write_options(deployment, directory)
    instances = [
        {
            "name": instance.name,
            "role": instance.role,
            "index": instance.index,
            "ranks": instance.server.ranks,
            "options": str(options[instance.role]),
        }
        for instance in deployment.instances()
    ]
    return {
        "deployment": deployment.name,
        "model": str(task_data["checkpoint_path"]),
        "directory": str(directory),
        # The tree the session measures: its servers run from it, and say so.
        "tree": str(task_data.get("trtllm_repo_path") or ""),
        "timeout": int(timeout),
        "env": dict(deployment.env),
        "instances": instances,
        "steps": [dict(step) for check in planned for step in check.steps],
        "order": [check.name for check in planned],
    }


def stated_ranks(deployment: Deployment) -> str:
    """The rank worlds a session's job states: ``<role instance>:<ranks>``, comma-separated.

    The pool's origin judge holds the job to it: a stated rank that reported no origin
    leaves the job with no result (``infra/compute/scheduler.py``). Each server's ranks
    name their role instance in ``MV2FLOW_ROLE`` (``session_driver.server_env``).
    """
    return ",".join(
        f"{instance.name}:{ranks_of(instance.server.options)}"
        for instance in deployment.instances()
    )


def gang_command(
    environment: RunEnvironment,
    task_data: Mapping[str, Any],
    deployment: Deployment,
    placement: Placement,
    directory: Path,
    spec_path: Path,
    options: Mapping[str, str],
) -> str:
    """The command of a session wider than one node: the script the pool's lead runs once.

    Its files are written beside the session's spec (``checks/gang.py``). The launched
    shells get what a session on one node gets from its prefix: the Modeling V2 switch,
    the rank hook first on ``PYTHONPATH``, the deployment's own variables.
    """
    import shlex

    moded = with_mode(environment, launch_mode(deployment))
    env = {k: str(v) for k, v in moded.env.items() if k != "PYTHONPATH"}
    env.update({str(k): str(v) for k, v in deployment.env.items()})
    files = gang.render(
        environment,
        deployment,
        placement.layout,
        placement.nodes,
        directory=directory,
        model=str(task_data["checkpoint_path"]),
        options=options,
        tree=str(task_data.get("trtllm_repo_path") or "") or None,
        driver=[environment.python, str(SESSION_DRIVER), "--spec", str(spec_path)],
        env=env,
        # The pool's origin hook, which a watcher puts first for a job it starts itself,
        # then the rank hook: each hands on to the next along the path.
        pythonpath=[
            str(RESOLVE_HOOK_DIR),
            *[p for p in str(moded.env.get("PYTHONPATH") or "").split(":") if p],
        ],
    )
    return f"bash {shlex.quote(str(gang.write(directory, files)))}"


def launch_mode(deployment: Deployment) -> str:
    """The requirement a session's launch states: its first server's. Each server reads its own options."""
    return requirement(next(iter(deployment.servers.values())).options)


def session_command(
    environment: RunEnvironment, spec_path: str | Path, *, mode: str = MODE_REQUIRE
) -> str:
    """The driver's launch over ``spec_path``: one process, which starts the servers itself.

    ``mode`` is the Modeling V2 requirement the launch states (:func:`launch_mode`).
    """
    import shlex

    return with_mode(environment, mode).python_command(
        f"{shlex.quote(str(SESSION_DRIVER))} --spec {shlex.quote(str(spec_path))}",
        single_process=True,
    )


def run_session(
    deployment: Deployment,
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    workspace: str | Path | None = None,
    relative_scores: Mapping[str, float | None] | None = None,
    dataset_paths: Mapping[str, str] | None = None,
    timeout: int | None = None,
) -> SessionResult:
    """Run one session of ``deployment`` as one job, and judge every check of it.

    ``runner`` submits the job (:func:`place` says what it holds). ``workdir`` is where
    the session keeps its files, on the shared filesystem when the job runs elsewhere.
    """
    verdict = verdict_factory(inspect_tree(task_data))
    member = deployment.shape
    started = time.monotonic()
    directory = gate_workdir(Path(workdir) / deployment.name)

    def line(name: str, **fields: Any) -> GateResult:
        fields.setdefault("duration_s", 0.0)
        return verdict(name=name, member=member, **fields)

    try:
        placement = place(task_data, deployment)
    except WiderThanOneNode as exc:
        result = line(GATE_NAME, passed=False, detail=str(exc), blocked=str(exc))
        return SessionResult((), result)
    planned = plan(
        task_data,
        deployment,
        directory=directory,
        workspace=workspace,
        relative_scores=relative_scores,
        dataset_paths=dataset_paths,
    )
    budget = int(timeout or session_timeout(task_data, deployment))
    spec = driver_spec(task_data, deployment, planned, directory=directory, timeout=budget)
    spec_path = directory / "session.json"
    write_text_atomic(spec_path, json.dumps(spec, indent=1))
    if placement.nodes > 1:
        try:
            command = gang_command(
                environment,
                task_data,
                deployment,
                placement,
                directory,
                spec_path,
                {str(i["role"]): str(i["options"]) for i in spec["instances"]},
            )
        except NoStepLine as exc:
            result = line(GATE_NAME, passed=False, detail=str(exc), blocked=str(exc))
            return SessionResult((), result)
    else:
        command = session_command(environment, spec_path, mode=launch_mode(deployment))
    result = runner(command, None, budget + TEARDOWN_S)
    output = result.stdout + result.stderr
    (directory / "session.out").write_text(output, encoding="utf-8")
    elapsed = time.monotonic() - started
    ran = dict(command=command, stdout_tail="\n".join(output.splitlines()[-40:]))
    if result.infrastructure:
        lost = line(
            GATE_NAME,
            passed=False,
            detail=f"did not run to a verdict: {result.infrastructure}",
            infrastructure=result.infrastructure,
            duration_s=elapsed,
            **ran,
        )
        return SessionResult((), lost)
    lines = judge(task_data, deployment, planned, output, placement=placement, line=line)
    sessions = workload_io.parse_marked(output, session_driver.SESSION_MARKER)
    ended = sessions[-1] if sessions else {}
    failed = [x for x in lines if not x.passed and not x.blocked]
    blocked = [x for x in lines if x.blocked]
    unverified = [x for x in lines if x.passed and x.unverified]
    warnings = []
    if requirement_said(deployment):
        # A stated requirement is used as stated, and the line says so.
        warnings.append(requirement_said(deployment))
    if ended.get("left"):
        warnings.append(f"pid(s) {ended['left']} still held a device after the stop")
    loads = int(ended.get("loads") or 0)
    if not sessions:
        why = "timed out" if result.timed_out else f"exited {result.returncode}"
        detail = f"the session {why} before it reported its end"
        passed = False
    elif ended.get("error"):
        detail, passed = str(ended["error"]), False
    elif failed:
        detail = f"{len(failed)} of {len(lines)} check(s) failed: " + ", ".join(
            x.name for x in failed
        )
        passed = False
    elif blocked:
        detail = f"{len(blocked)} check(s) blocked before the load: " + ", ".join(
            x.name for x in blocked
        )
        passed = False
    elif result.returncode != 0:
        # Its records are complete, and the process that wrote them (or what launched it)
        # still ended in failure: what a launcher or a lead says of its own end counts.
        detail = (
            f"{len(lines) - len(unverified)} check(s) passed, and the session's process ended "
            f"with exit {result.returncode}: a session that does not end clean does not pass"
        )
        passed = False
    else:
        detail, passed = f"{len(lines) - len(unverified)} check(s) passed", True
    if unverified:
        detail += f"; {len(unverified)} not verified: " + ", ".join(x.name for x in unverified)
    detail += (
        f"; {loads} model load(s) for {len(list(deployment.instances()))} role instance(s); "
        f"order: {', '.join(p.name for p in planned)}"
    )
    held = {x.name for x in blocked}
    own = tuple(p for p in planned if p.reads.get("own_load") and p.name not in held)
    session = line(
        GATE_NAME,
        passed=passed,
        detail=detail,
        blocked="; ".join(f"{x.name}: {x.blocked}" for x in blocked)
        if blocked and not failed
        else "",
        unverified="; ".join(f"{x.name}: {x.unverified}" for x in unverified) if passed else "",
        explained=True,
        duration_s=elapsed,
        warnings=tuple(warnings),
        engine=tuple(r for x in lines if x.name == "attest" for r in x.engine),
        **ran,
    )
    return SessionResult(tuple(lines), session, loads=loads, own_loads=own)


def run_deployment(
    deployment: Deployment,
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    workspace: str | Path | None = None,
    relative_scores: Mapping[str, float | None] | None = None,
    dataset_paths: Mapping[str, str] | None = None,
    timeout: int | None = None,
) -> SessionResult:
    """Everything ``deployment`` owes of its own engine: its session, then any load of its own.

    The registry's ``deployment:<name>``. The session is one load (one per role
    instance). A benchmark the endpoint evaluator cannot carry, or one the spec marks
    ``through: eval``, then runs ``trtllm-eval`` as a job of its own, after the session
    has stopped its servers; its line says so, and the count of loads includes it. A
    session that did not start leaves those benchmarks unrun, each with a line.
    """
    from .accuracy import run_eval

    found = run_session(
        deployment,
        environment,
        task_data=task_data,
        workdir=workdir,
        runner=runner,
        workspace=workspace,
        relative_scores=relative_scores,
        dataset_paths=dataset_paths,
        timeout=timeout,
    )
    if not found.own_loads:
        return found
    session = found.result
    lines = list(found.lines)
    loads = found.loads
    verdict = verdict_factory(inspect_tree(task_data))
    for check in found.own_loads:
        if session.infrastructure or not found.lines or not found.loads:
            lines.append(
                verdict(
                    name=check.name,
                    member=deployment.shape,
                    passed=False,
                    detail="not run: the deployment's session did not run to its end",
                )
            )
            continue
        lines.append(
            run_eval(
                environment,
                task_data=task_data,
                deployment=deployment,
                benchmark=check.reads["benchmark"],
                anchor=check.reads["anchor"],
                workdir=Path(workdir) / deployment.name,
                runner=runner,
                relative_score=check.reads.get("relative_score"),
                dataset_path=check.reads.get("dataset_path"),
            )
        )
        loads += 1
    own = lines[len(found.lines) :]
    failed = [x.name for x in own if not x.passed]
    names = ", ".join(x.name for x in own)
    whys = "; ".join(
        dict.fromkeys(
            str(check.reads.get("own_load_why") or "not carried by the session's endpoint")
            for check in found.own_loads
        )
    )
    detail = (
        f"{session.detail}; then {len(own)} load(s) of their own for {names} "
        f"(`trtllm-eval`: {whys})" + (f", of which failed: {', '.join(failed)}" if failed else "")
    )
    result = replace(
        session,
        passed=session.passed and not failed,
        detail=detail,
        duration_s=session.duration_s + sum(x.duration_s for x in own),
        infrastructure=session.infrastructure
        or next((x.infrastructure for x in own if x.infrastructure), ""),
    )
    return SessionResult(tuple(lines), result, loads=loads, own_loads=found.own_loads)


# ------------------------------------------------------------------------------- judging


def _records(output: str) -> dict[str, list[dict[str, Any]]]:
    """The driver's request records by step, each step's in request order."""
    found: dict[str, list[dict[str, Any]]] = {}
    for record in workload_io.parse_marked(output, workload_io.RECORD_MARKER):
        found.setdefault(str(record.get("step")), []).append(record)
    for records in found.values():
        records.sort(key=lambda r: int(r.get("index", 0)))
    return found


def _unanswered(records: Sequence[Mapping[str, Any]], expected: int, what: str) -> str:
    """Why ``records`` are not ``expected`` completed, non-empty answers; ``""`` when they are."""
    if len(records) < expected:
        return f"{what}: {len(records)} of {expected} request(s) came back"
    for record in records:
        where = f"{what} `{record.get('id')}` ({record.get('endpoint')})"
        if record.get("status") is None and record.get("error"):
            # Never sent, or never answered: its own words, whole (they say what to do).
            return f"{where}: {record['error']}"
        if record.get("status") != 200:
            said = str(record.get("error") or "no response")[:300]
            return f"{where} answered HTTP {record.get('status')}: {said}"
        if record.get("error"):
            return f"{where}: {record['error']}"
    return ""


def judge(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    planned: Sequence[Planned],
    output: str,
    *,
    placement: Placement,
    line: Any,
) -> list[GateResult]:
    """One line per planned check, in order, from what the session's job printed."""
    records = _records(output)
    steps = {
        str(s.get("step")): s for s in workload_io.parse_marked(output, session_driver.STEP_MARKER)
    }
    sessions = workload_io.parse_marked(output, session_driver.SESSION_MARKER)
    started = bool(sessions and sessions[-1].get("started"))
    never = "" if started else "the deployment did not start, so this check did not run"
    devices = workload_io.parse_marked(output, session_driver.DEVICES_MARKER)
    lines: list[GateResult] = []
    for check in planned:
        took = sum(
            float((steps.get(str(s["id"])) or {}).get("seconds") or 0.0) for s in check.steps
        )
        # Blocked before the job, or by the job on its node before it started a server.
        held = check.blocked or next(
            (
                str(steps[str(s["id"])]["blocked"])
                for s in check.steps
                if (steps.get(str(s["id"])) or {}).get("blocked")
            ),
            "",
        )
        if held:
            lines.append(line(check.name, passed=False, detail=held, blocked=held))
            continue
        if check.reads.get("own_load"):
            continue
        if check.name == "devices":
            fields = _judge_devices(deployment, devices)
        elif check.name == "attest":
            fields = _judge_attest(task_data, deployment, output, devices, started)
        elif never:
            fields = dict(passed=False, detail=never)
        else:
            errors = [
                f"{s['id']}: {steps[str(s['id'])]['error']}"
                for s in check.steps
                if (steps.get(str(s["id"])) or {}).get("error")
            ]
            missing = [str(s["id"]) for s in check.steps if str(s["id"]) not in steps]
            if errors or missing:
                why = "; ".join(errors) or f"step(s) {', '.join(missing)} did not run"
                fields = dict(passed=False, detail=why)
            else:
                fields = _JUDGES[check.name.partition("[")[0]](
                    task_data, deployment, check, records, steps
                )
        lines.append(line(check.name, duration_s=took, **fields))
    return lines


def _judge_devices(deployment: Deployment, devices: Sequence[Mapping[str, Any]]) -> dict:
    expected = {instance.name: instance.server.ranks for instance in deployment.instances()}
    problem = attest.devices_problem(devices, expected)
    if problem:
        return dict(passed=False, detail=f"before any load: {problem}")
    said = "; ".join(
        f"{r['instance']} on {r.get('host') or 'this host'}: {','.join(map(str, r['devices']))}"
        for r in devices
    )
    held = [f"{r['instance']}: pid(s) {r['held_before']}" for r in devices if r.get("held_before")]
    warnings = (f"device(s) already held before the load ({'; '.join(held)})",) if held else ()
    hosts = {str(r.get("host") or "") for r in devices}
    where = f" across {len(hosts)} hosts" if len(hosts) > 1 else ""
    return dict(
        passed=True,
        explained=True,
        detail=f"before any load, {len(expected)} server(s) on devices of their own{where}: {said}",
        warnings=warnings,
    )


def _judge_attest(
    task_data: Mapping[str, Any],
    deployment: Deployment,
    output: str,
    devices: Sequence[Mapping[str, Any]],
    started: bool,
) -> dict:
    hosts = {str(r.get("host") or "") for r in devices}
    given: dict[str, Any] = {}
    for r in devices:
        own = [str(d) for d in r.get("devices") or ()]
        if len(hosts) > 1:
            # Across nodes a rank is held to its instance's devices of its own host.
            given.setdefault(str(r.get("instance")), {})[str(r.get("host") or "")] = own
        else:
            given[str(r.get("instance"))] = own
    found = [
        attest.attest(
            output,
            {**instance.server.options, REQUIREMENT_OPTION: requirement(instance.server.options)},
            role=instance.role,
            instance=instance.name,
            tree=task_data.get("trtllm_repo_path"),
            devices=given.get(instance.name),
            # Only `require` holds a rank to a Modeling V2 target; the option it ran is
            # held to what the spec states either way.
            target=requirement(instance.server.options) == MODE_REQUIRE,
        )
        for instance in deployment.instances()
    ]
    said = requirement_said(deployment)
    engine = tuple(a.record() for a in found)
    failure = next((a.failure() for a in found if a.failure()), None)
    if failure:
        if not started:
            failure = f"the deployment did not start; {failure}"
        return dict(passed=False, detail=failure, engine=engine)
    ranks = sum(a.ranks for a in found)
    return dict(
        passed=True,
        explained=True,
        detail=f"{ranks} rank(s) of {len(found)} server(s) ran what their options declare, "
        f"from the tree measured, on the devices given" + (f"; {said}" if said else ""),
        engine=engine,
    )


def _judge_boot(task_data, deployment, check, records, steps) -> dict:
    spec: BootSpec = check.reads["spec"]
    answers = records.get("boot", [])
    expected = len(check.steps[0]["requests"])
    problem = _unanswered(answers, expected, "boot prompt")
    if problem:
        return dict(passed=False, detail=problem)
    by_id = {str(r.get("id")): str(r.get("text") or "") for r in answers}
    empty = [rid for rid, text in by_id.items() if not text.strip()]
    if empty:
        return dict(
            passed=False,
            detail=f"the endpoint was healthy but answered {', '.join(empty)} with nothing: "
            f"health alone is not a boot",
        )
    requests = spec.requests(None)
    texts = [None if r.get("images") else by_id.get(r["id"]) for r in requests]
    missing = missing_keywords(spec, texts)
    if missing:
        return dict(
            passed=False,
            detail=f"the answers are missing {len(missing)} expected keyword(s): {missing}. "
            f"Keywords are matched against generated text only",
        )
    return dict(
        passed=True,
        explained=True,
        detail=f"{deployment.kind} deployment `{deployment.name}` answered {len(answers)} "
        f"prompt(s) with their keywords; workload: {spec.source}",
    )


def _judge_api(task_data, deployment, check, records, steps) -> dict:
    for step, what in (("api.chat", "chat completion"), ("api.stream", "streamed completion")):
        answers = records.get(step, [])
        problem = _unanswered(answers, 1, what)
        if not problem and not str(answers[0].get("text") or "").strip():
            problem = f"the {what} came back empty"
        if not problem and step == "api.stream" and not answers[0].get("done"):
            problem = (
                f"the streamed completion sent {answers[0].get('events')} event(s) and never "
                f"its end marker"
            )
        if problem:
            return dict(passed=False, detail=problem)
    events = records["api.stream"][0].get("events")
    return dict(
        passed=True,
        detail=f"a chat completion answered, and a streamed completion in {events} event(s)",
    )


def _judge_image(task_data, deployment, check, records, steps) -> dict:
    spec: BootSpec = check.reads["spec"]
    answers = records.get("image", [])
    problem = _unanswered(answers, len(check.steps[0]["requests"]), "image prompt")
    if problem:
        return dict(passed=False, detail=f"image not checked: {problem}")
    by_id = {str(r.get("id")): str(r.get("text") or "") for r in answers}
    requests = spec.requests(None)
    texts = [by_id.get(r["id"]) if r.get("images") else None for r in requests]
    if any(text is not None and not text.strip() for text in texts):
        return dict(passed=False, detail="an image prompt came back empty")
    missing = missing_keywords(replace(spec, keywords=()), texts)
    if missing:
        return dict(
            passed=False,
            detail=f"the image answers are missing {len(missing)} expected keyword(s): {missing}",
        )
    return dict(passed=True, detail=f"{len(answers)} prompt(s) with image(s) answered")


def _judge_chunk(task_data, deployment, check, records, steps) -> dict:
    limits = check.reads["limits"]
    answers = records.get("chunk", [])
    problem = _unanswered(answers, 1, "the long prompt")
    if problem:
        return dict(passed=False, detail=problem)
    shortfall = chunk_shortfall(answers, limits["max_num_tokens"])
    if shortfall:
        return dict(passed=False, detail=shortfall)
    return dict(
        passed=True,
        detail=f"a prompt of {answers[0].get('prompt_tokens')} tokens, past max_num_tokens "
        f"{limits['max_num_tokens']}, was served in chunks",
    )


def transfer_intervals(value: Any, begin: str, end: str) -> dict[str, tuple[float, float]]:
    """Per phase, the transfer interval a timestamps header names.

    The header lists ``<phase>-<name>;ts=<seconds>`` items; a phase has an interval when
    it carries both ``<phase>-<begin>`` and ``<phase>-<end>``, the end not before the start.
    """
    stamps: dict[str, float] = {}
    for item in str(value or "").split(","):
        name, separator, stamp = item.strip().partition(";ts=")
        if separator:
            try:
                stamps[name] = float(stamp)
            except ValueError:
                continue
    found = {}
    for name, started in stamps.items():
        if name.endswith(f"-{begin}"):
            phase = name[: -len(begin) - 1]
            ended = stamps.get(f"{phase}-{end}")
            if ended is not None and ended >= started > 0:
                found[phase] = (started, ended)
    return found


def _judge_hop(task_data, deployment, check, records, steps) -> dict:
    """Whether generation started from the context's state, by what the engine lets a client see.

    The requests must be answered through the proxy: that is a failure when they are not.
    The transfer itself is read the way the measured tree defines (``session_driver.hop_means``).
    Where the engine offers no way to observe it, or its answers do not carry it, the line
    says "not verified" with the reason: it fails nothing, and the score through the hop
    is judged on its own line.
    """
    step = steps.get("hop") or {}
    answers = records.get("hop", [])
    problem = _unanswered(answers, len(check.steps[0]["requests"]), "the hop")
    if problem:
        return dict(passed=False, detail=f"the hop is not proven: {problem}")
    observe = step.get("observe") or {
        "means": "none",
        "why": "the session's job did not say how a transfer can be observed",
    }
    served = f"{len(answers)} request(s) were answered through the proxy"

    def not_verified(why: str) -> dict:
        return dict(
            passed=True,
            explained=True,
            unverified=why,
            detail=f"not verified: {why}. {served}; the score through the hop is judged on "
            f"its own line",
        )

    if observe.get("means") == "headers":
        header, begin, end = observe["header"], observe["begin"], observe["end"]
        proven, phases, carried, lacking = 0, set(), 0, ""
        for answer in answers:
            value = (answer.get("headers") or {}).get(header)
            carried += bool(value)
            # The context's own phase is not the side that receives.
            found = [
                phase
                for phase in transfer_intervals(value, begin, end)
                if not phase.startswith(ROLES[0])
            ]
            proven += bool(found)
            phases.update(found)
            if value and not found and not lacking:
                # What a response without the interval did name, for whoever reads the line.
                lacking = ", ".join(item.strip().split(";")[0] for item in str(value).split(","))
        if proven == len(answers):
            return dict(
                passed=True,
                explained=True,
                detail=f"generation received the context's state for {proven} of {proven} "
                f"request(s): each response's `{header}` names the transfer interval of "
                f"{', '.join(f'`{p}`' for p in sorted(phases))} (`{begin}` to `{end}`), as "
                f"`{observe.get('source')}` of the measured tree defines it",
            )
        if not carried:
            return not_verified(
                f"no response carried `{header}`, which `{observe.get('source')}` of the "
                f"measured tree defines for a request sent with `{observe['request_header']}: 1`"
            )
        return not_verified(
            f"{len(answers) - proven} of {len(answers)} response(s) name no transfer interval "
            f"of a generation phase (`<phase>-{begin}` to `<phase>-{end}`) in `{header}`: the "
            f"engine's answer lacks the generation side's start or end of the transfer (it "
            f"names {lacking[:300] or 'nothing'})"
        )
    return not_verified(str(observe.get("why") or "the engine offers no way to observe it"))


def _judge_accuracy(task_data, deployment, check, records, steps) -> dict:
    from .accuracy import judged, parse_eval_score, unparsed_score_detail

    benchmark: Benchmark = check.reads["benchmark"]
    step = steps[check.name]
    try:
        text = Path(check.reads["log"]).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return dict(passed=False, detail=f"the evaluator's output cannot be read: {exc}")
    fields: dict[str, Any] = dict(
        command=str(step.get("command") or ""), stdout_tail="\n".join(text.splitlines()[-40:])
    )
    if step.get("returncode"):
        return dict(passed=False, detail=f"the evaluator exited {step['returncode']}", **fields)
    measured = parse_eval_score(text, benchmark.score_pattern)
    if measured is None:
        return dict(
            passed=False, detail=unparsed_score_detail(text, benchmark.score_pattern), **fields
        )
    how = " through the hop" if deployment.server is None else " through the endpoint"
    return {
        **judged(
            measured,
            check.reads["anchor"],
            benchmark,
            accuracy_policy.resolve_floor_policy(task_data.get("accuracy") or {}),
            relative_score=check.reads.get("relative_score"),
            output=text,
            how=how,
        ),
        **fields,
    }


def _judge_acceptance(task_data, deployment, check, records, steps) -> dict:
    from . import acceptance

    return acceptance.judge_session(
        task_data, deployment, check.reads["workload"], records.get("acceptance", [])
    )


def _judge_reuse(task_data, deployment, check, records, steps) -> dict:
    answers = records.get("reuse", [])
    problem = _unanswered(answers, len(check.steps[0]["requests"]), "the repeated prompt")
    if problem:
        return dict(passed=False, detail=problem)
    cached = [r.get("cached_tokens") for r in answers]
    shortfall = attest.reuse_shortfall(cached)
    if shortfall:
        return dict(passed=False, detail=shortfall)
    side = (
        " on the context server (the proxy reports that side's usage)"
        if deployment.server is None
        else ""
    )
    return dict(
        passed=True,
        detail=f"a repeated prompt resumed {max(c or 0 for c in cached)} cached tokens{side}",
    )


_JUDGES = {
    "boot": _judge_boot,
    "api": _judge_api,
    "image": _judge_image,
    "chunk": _judge_chunk,
    "hop": _judge_hop,
    "accuracy": _judge_accuracy,
    "acceptance": _judge_acceptance,
    "reuse": _judge_reuse,
}


__all__ = [
    "ENDPOINT_BENCHMARKS",
    "GATE_NAME",
    "Placement",
    "Planned",
    "SessionResult",
    "WiderThanOneNode",
    "boot_spec",
    "driver_spec",
    "transfer_intervals",
    "judge",
    "limits_of",
    "place",
    "plan",
    "run_deployment",
    "run_session",
    "session_command",
    "session_timeout",
    "stated_ranks",
    "write_options",
]
