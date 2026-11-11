# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boot and accuracy of a deployment judged through its endpoint.

Every deployment boots this way: one server, or a context server, a generation server
and the proxy between them, started by the check and asked through the endpoint. Boot
starts the deployment, sends the boot workload's prompts as completions and
holds the answers to the workload's keywords: an endpoint that is healthy and answers
nothing has not booted. Accuracy of a disaggregated deployment starts it and runs the
tree's own evaluator against the endpoint (``harness/endpoint_eval.py``), judged like any other score, against the
anchor's floor. A benchmark the tree has not driven through an endpoint is not started:
its verdict says it is not judged through the hop.

Each check is one load: start, ask, stop. A stop that leaves ranks on the devices
(:class:`~.servers.Leftover`) turns the check into no verdict; devices that could not be
looked at are said in the verdict's line.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from ..spec.deployments import Benchmark, Deployment, Server
from ..spec.task_schema import RAW_REFUSED, resolve_gate_timeout
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
    running: servers.Running,
    role: str,
    server: Server,
    task_data: Mapping[str, Any],
    mode: str = MODE_REQUIRE,
) -> attest.Attestation:
    """What the ranks of ``role``'s server printed, against its options and the switch."""
    return attest.attest(
        running.log(role),
        {**server.options, "modeling_v2": mode},
        role=role,
        tree=task_data.get("trtllm_repo_path"),
        # A control run on the stock path builds no Modeling V2 target.
        target=mode == MODE_REQUIRE,
    )


def _stopped(running: servers.Running) -> tuple[str, str]:
    """Stop ``running``: ``(why the check has no verdict, what its line adds)``.

    Ranks left on the devices are no verdict. Devices that could not be looked at leave
    the verdict standing and are said in its line, never taken for clean.
    """
    try:
        note = servers.stop(running)
    except servers.ServerError as exc:
        return str(exc), ""
    return "", f"; {note}" if note else ""


def _started(
    deployment: Deployment,
    task_data: Mapping[str, Any],
    workspace,
    devices,
    timeout,
    mode: str = MODE_REQUIRE,
):
    """``(running, None)``, or ``(None, fields)`` of the verdict of a deployment that did not start."""
    try:
        running = servers.start(
            deployment,
            model=str(task_data["checkpoint_path"]),
            workspace=workspace,
            env=_declared_env(task_data),
            timeout=timeout,
            tree=task_data.get("trtllm_repo_path"),
            python=(task_data.get("environment") or {}).get("python"),
            devices=devices,
            mode=mode,
        )
    except servers.Leftover as exc:
        return None, dict(passed=False, detail=str(exc), infrastructure=str(exc))
    except servers.ServerError as exc:
        return None, dict(passed=False, detail=f"the deployment did not start: {exc}")
    return running, None


def run_endpoint_boot(
    deployment: Deployment,
    *,
    task_data: Mapping[str, Any],
    workspace: str | Path,
    spec: BootSpec | None = None,
    timeout: int | None = None,
    devices: Sequence[int] = (),
    mode: str = MODE_REQUIRE,
) -> GateResult:
    """Start ``deployment``, hold its endpoint to the boot workload, and stop it.

    The workload is sent as it is stated: a text prompt as a completion, a prompt with
    images as one chat completion with every image. Each must come back completed and
    with text, and the workload's keywords must be in the answers. The start and the
    requests together end at ``timeout``, else the spec's ``accuracy.boot.timeout``,
    else the deployment's own: a request takes the time that remains.
    ``mode`` other than ``require`` is a control run on the stock implementation,
    recorded as ``boot[<mode>]``.
    """
    spec = spec or BootSpec()
    declared = resolve_gate_timeout(task_data.get("accuracy") or {}, "boot", deployment.name)
    timeout = timeout or declared
    verdict = verdict_factory(inspect_tree(task_data))
    started = time.monotonic()
    records: list[Mapping[str, Any]] = []

    def ran(**fields: Any) -> GateResult:
        return verdict(
            name="boot" if mode == MODE_REQUIRE else f"boot[{mode}]",
            deployment=deployment.name,
            duration_s=time.monotonic() - started,
            engine=tuple(records),
            **fields,
        )

    running, refused = _started(deployment, task_data, workspace, devices, timeout, mode)
    if running is None:
        return ran(**refused)
    texts: list[str | None] = []
    cached: list[int | None] = []
    roles = deployment.servers
    failure = ""

    def left() -> float:
        """What remains of the check's time; a second at least, so a late request fails as one."""
        return max(1.0, started + running.budget - time.monotonic())

    try:
        for request in spec.requests(task_data.get("trtllm_repo_path")):
            asked = dict(max_tokens=spec.max_tokens, timeout=left())
            if request.get("images"):
                # The server places the images and applies the chat template itself, so a
                # prompt rendered by hand cannot go as it is; it is not replaced by another.
                if request.get("render") == "raw":
                    raise servers.ServerError(f"image not checked: {RAW_REFUSED}")
                try:
                    said = servers.ask_image(running, request["prompt"], request["images"], **asked)
                except servers.ServerError as exc:
                    raise servers.ServerError(f"image not checked: {exc}") from exc
                texts.append(said)
                continue
            texts.append(servers.ask(running, request["prompt"], **asked))
        # With block reuse on, one more prompt is sent again and must resume from the cache.
        sends = max(
            (attest.reuse_sends(s.options) for s in roles.values() if attest.reuses(s.options)),
            default=0,
        )
        for _ in range(sends):
            reply = servers.complete(running, attest.REUSE_PROMPT, max_tokens=1, timeout=left())
            cached.append(servers.cached_tokens(reply))
    except servers.ServerError as exc:
        failure = str(exc)
        if time.monotonic() - started >= running.budget:
            failure += f" (the check's {running.budget:g} s were up: start and requests share them)"
    finally:
        attested = [
            _attested(running, role, server, task_data, mode) for role, server in roles.items()
        ]
        records += [found.record() for found in attested]
        left, note = _stopped(running)
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
    empty = sum(1 for text in texts if not (text or "").strip())
    if empty or not texts:
        return ran(
            passed=False,
            detail=f"the endpoint was healthy but answered {empty} of {len(texts)} prompt(s) "
            f"with nothing: health alone is not a boot",
        )
    missing = missing_keywords(spec, texts)
    if missing:
        return ran(
            passed=False,
            detail=f"the endpoint's answers are missing {len(missing)} expected keyword(s): "
            f"{missing}. Keywords are matched against generated text only",
        )
    images = sum(1 for item in spec.items if item.images)
    # The proxy reports the context server's usage: what the repeat shows is that side's cache.
    side = " on the context server; the generation server's reuse is not shown by the proxy"
    reused = (
        f"; a repeated prompt resumed {max(c or 0 for c in cached)} cached tokens" if cached else ""
    )
    reused += side if cached and deployment.server is None else ""
    return ran(
        passed=True,
        explained=True,
        detail=f"{deployment.kind} deployment `{deployment.name}` started and its endpoint "
        f"answered {len(texts)} prompt(s) with their keywords"
        + (f", {images} of them with an image" if images else "")
        + f"; workload: {spec.source}{reused}{note}",
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
    declared = {**_declared_env(task_data), **deployment.env, **benchmark.env}
    dataset, said = datasets.find(benchmark.name, benchmark.dataset_path, declared)
    if said and not dataset:
        return ran(passed=False, detail=said, infrastructure=said)
    running, refused = _started(
        deployment, task_data, workspace, environment.devices, deployment.timeout
    )
    if running is None:
        return ran(**refused)
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
        left, note = _stopped(running)
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
    how = f" through the endpoint{note}"
    return ran(
        **judged(
            measured, anchor, benchmark, relative_score=relative_score, output=combined, how=how
        ),
        **fields,
    )
