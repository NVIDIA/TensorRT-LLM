# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boot and accuracy of a deployment judged through its endpoint.

A disaggregated deployment has no one engine to drive in process: what serves is a
context server, a generation server and the proxy between them, so its checks ask the
proxy. Boot starts the deployment, sends the boot workload's prompts as completions and
holds the answers to the workload's keywords: an endpoint that is healthy and answers
nothing has not booted. Accuracy starts it and runs the tree's own evaluator against the
endpoint (``harness/endpoint_eval.py``), judged like any other score, against the
anchor's floor. A benchmark the tree has not driven through an endpoint is not started:
its verdict says it is not judged through the hop.

Each check is one load: start, ask, stop. A stop that leaves ranks on the devices
(:class:`~.servers.ServerError`) turns the check into no verdict.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from ..spec.deployments import Benchmark, Deployment, Server
from . import anchors as anchors_mod
from . import attest, datasets, servers
from .accuracy import judged, parse_eval_score, unparsed_score_detail
from .boot import BootSpec, missing_keywords
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner
from .harness import endpoint_eval
from .verdict import GateResult, inspect_tree, verdict_factory

#: The evaluator program, run under the interpreter of the tree being measured.
ENDPOINT_EVAL = Path(__file__).resolve().parent / "harness" / "endpoint_eval.py"
#: The benchmarks judged through an endpoint.
ENDPOINT_BENCHMARKS: tuple[str, ...] = tuple(sorted(endpoint_eval.EVALUATORS))
NOT_JUDGED = "not judged through the hop"
ENDPOINT_ACCURACY_TIMEOUT_S = 4 * 60 * 60


def _declared_env(task_data: Mapping[str, Any]) -> dict[str, str]:
    block = (task_data.get("environment") or {}).get("env") or {}
    return {str(k): str(v) for k, v in block.items()}


def _attested(
    running: servers.Running, role: str, server: Server, task_data: Mapping[str, Any]
) -> attest.Attestation:
    """What the ranks of ``role``'s server printed, against its options and the switch."""
    return attest.attest(
        running.log(role),
        {**server.options, "modeling_v2": MODE_REQUIRE},
        role=role,
        tree=task_data.get("trtllm_repo_path"),
    )


def _stopped(running: servers.Running, env: Mapping[str, str]) -> str:
    """Stop ``running``; why the check has no verdict when ranks were left, else ``""``."""
    try:
        servers.stop(running, env=env)
    except servers.ServerError as exc:
        return str(exc)
    return ""


def run_endpoint_boot(
    deployment: Deployment,
    *,
    task_data: Mapping[str, Any],
    workspace: str | Path,
    spec: BootSpec | None = None,
    timeout: int | None = None,
    devices: Sequence[int] = (),
) -> GateResult:
    """Start ``deployment``, hold its endpoint to the boot workload, and stop it.

    Every text prompt of the workload is sent as a completion. Each must come back with
    text, and the workload's keywords must be in the answers, as for a boot in process.
    """
    spec = spec or BootSpec()
    verdict = verdict_factory(inspect_tree(task_data))
    env = _declared_env(task_data)
    started = time.monotonic()

    records: list[Mapping[str, Any]] = []

    def ran(**fields: Any) -> GateResult:
        return verdict(
            name="boot",
            deployment=deployment.name,
            duration_s=time.monotonic() - started,
            engine=tuple(records),
            **fields,
        )

    try:
        running = servers.start(
            deployment,
            model=str(task_data["checkpoint_path"]),
            workspace=workspace,
            env=env,
            timeout=timeout,
            tree=task_data.get("trtllm_repo_path"),
            python=(task_data.get("environment") or {}).get("python"),
            devices=devices,
        )
    except servers.ServerError as exc:
        return ran(passed=False, detail=f"the deployment did not start: {exc}")
    texts: list[str | None] = []
    cached: list[int | None] = []
    roles = deployment.servers
    failure = ""
    try:
        prompts = [*spec.prompts, *(item.prompt for item in spec.items)]
        images = {len(spec.prompts) + n for n, item in enumerate(spec.items) if item.images}
        for index, prompt in enumerate(prompts):
            if index in images:
                texts.append(None)  # an image prompt is not sent as a text completion
                continue
            texts.append(servers.ask(running, prompt, max_tokens=spec.max_tokens))
        # With block reuse on, one more prompt is sent again and must resume from the cache.
        sends = max(
            (attest.reuse_sends(s.options) for s in roles.values() if attest.reuses(s.options)),
            default=0,
        )
        for _ in range(sends):
            reply = servers.complete(running, attest.REUSE_PROMPT, max_tokens=1)
            cached.append(servers.cached_tokens(reply))
    except servers.ServerError as exc:
        failure = str(exc)
    finally:
        attested = [_attested(running, role, server, task_data) for role, server in roles.items()]
        records += [found.record() for found in attested]
        left = _stopped(running, env)
    if left:
        return ran(passed=False, detail=left, infrastructure=left)
    if failure:
        return ran(passed=False, detail=failure)
    unmatched = next((found.failure() for found in attested if found.failure()), None)
    if unmatched:
        return ran(passed=False, detail=unmatched)
    shortfall = attest.reuse_shortfall(cached) if cached else None
    if shortfall:
        return ran(passed=False, detail=shortfall)
    asked = [text for text in texts if text is not None]
    empty = sum(1 for text in asked if not text.strip())
    if empty or not asked:
        return ran(
            passed=False,
            detail=f"the endpoint was healthy but answered {empty} of {len(asked)} prompt(s) "
            f"with nothing: health alone is not a boot",
        )
    missing = missing_keywords(spec, texts)
    if missing:
        return ran(
            passed=False,
            detail=f"the endpoint's answers are missing {len(missing)} expected keyword(s): "
            f"{missing}. Keywords are matched against generated text only",
        )
    skipped = (
        f"; {len(texts) - len(asked)} image prompt(s) not sent" if len(asked) < len(texts) else ""
    )
    return ran(
        passed=True,
        explained=True,
        detail=f"{deployment.kind} deployment `{deployment.name}` started and its endpoint "
        f"answered {len(asked)} prompt(s) with their keywords; workload: {spec.source}{skipped}",
    )


def eval_command(
    environment: RunEnvironment,
    *,
    url: str,
    model: str,
    benchmark: Benchmark,
    dataset_path: str | None = None,
) -> str:
    """The evaluator's command line against ``url``, the benchmark's own options after it."""
    words = [str(ENDPOINT_EVAL), "--url", url, "--model", model, "--benchmark", benchmark.name]
    if dataset_path:
        words += ["--dataset_path", str(dataset_path)]
    words += list(benchmark.args)
    line = f"{environment.python} {' '.join(shlex.quote(word) for word in words)}"
    return f"HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 {line}"


def run_endpoint_accuracy(
    deployment: Deployment,
    benchmark: Benchmark,
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    workspace: str | Path,
    anchor: anchors_mod.Anchor,
    relative_score: float | None = None,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
) -> GateResult:
    """Score ``benchmark`` through ``deployment``'s endpoint and hold it to ``anchor``'s floor."""
    name = f"accuracy[{benchmark.name}]"
    verdict = verdict_factory(inspect_tree(task_data))
    env = _declared_env(task_data)
    started = time.monotonic()

    def ran(**fields: Any) -> GateResult:
        return verdict(
            name=name, deployment=deployment.name, duration_s=time.monotonic() - started, **fields
        )

    if benchmark.name not in ENDPOINT_BENCHMARKS:
        return ran(
            passed=False,
            explained=True,
            detail=f"{NOT_JUDGED}: the tree's `{benchmark.name}` evaluator has not been driven "
            f"through an endpoint ({', '.join(ENDPOINT_BENCHMARKS)} have); score it on an "
            f"aggregated deployment",
        )
    dataset, said = datasets.find(benchmark.name, benchmark.dataset_path, env)
    if said and not dataset:
        return ran(passed=False, detail=said, infrastructure=said)
    try:
        running = servers.start(
            deployment,
            model=str(task_data["checkpoint_path"]),
            workspace=workspace,
            env=env,
            timeout=deployment.timeout,
            tree=task_data.get("trtllm_repo_path"),
            python=(task_data.get("environment") or {}).get("python"),
            devices=environment.devices,
        )
    except servers.ServerError as exc:
        return ran(passed=False, detail=f"the deployment did not start: {exc}")
    command = eval_command(
        environment,
        url=running.url,
        model=str(task_data["checkpoint_path"]),
        benchmark=benchmark,
        dataset_path=dataset or None,
    )
    try:
        result = runner(command, benchmark.env or None, timeout or ENDPOINT_ACCURACY_TIMEOUT_S)
    finally:
        attested = [
            _attested(running, role, server, task_data)
            for role, server in deployment.servers.items()
        ]
        left = _stopped(running, env)
    combined = result.stdout + result.stderr
    fields: dict[str, Any] = dict(
        command=command,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
        engine=tuple(found.record() for found in attested),
    )
    if left:
        return ran(passed=False, detail=left, infrastructure=left, **fields)
    if not result.ok:
        return ran(passed=False, detail=f"the evaluator exited {result.returncode}", **fields)
    unmatched = next((found.failure() for found in attested if found.failure()), None)
    if unmatched:
        return ran(passed=False, detail=unmatched, **fields)
    measured = parse_eval_score(combined, benchmark.score_pattern)
    if measured is None:
        return ran(
            passed=False, detail=unparsed_score_detail(combined, benchmark.score_pattern), **fields
        )
    how = " through the endpoint"
    return ran(
        **judged(
            measured, anchor, benchmark, relative_score=relative_score, output=combined, how=how
        ),
        **fields,
    )


__all__ = [
    "ENDPOINT_BENCHMARKS",
    "NOT_JUDGED",
    "eval_command",
    "run_endpoint_accuracy",
    "run_endpoint_boot",
]
