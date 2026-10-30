# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The serve gate: ``trtllm-serve`` starts the unit's target and answers over HTTP.

Every other gate builds the engine through ``quickstart_advanced.py`` or the
LLM API in its own process, so the server's own path runs in none of them:
the CLI turning an options file into LLM arguments, what the server asks of
the model's input processor (its media defaults, for one), the OpenAI request
and response handlers, detokenization in the server's workers. A defect there
leaves every gate green and the delivered tree unable to serve, and
``trtllm-serve`` is the entry point users and disaggregated serving start from.

The gate starts ``trtllm-serve <checkpoint>`` on the tree under test with the
unit's LLM-API options (the accuracy gate's: the unit's file, the member's
overlay and the ``modeling_v2`` selection), at the member's topology, waits
for ``/health`` and sends one ``/v1/completions`` request, then the unit's
boot workload as far as the boot gate judges it: each plain prompt as a
completion when the workload names keywords, and each prompt with checks of
its own -- an image prompt as a ``/v1/chat/completions`` request carrying
its image. It passes when the server became healthy, every answer is HTTP
200 with non-empty text, and the answers hold the keywords the boot gate
holds the engine's texts to (``gates.missing_keywords``): text out of the
server that the boot gate would refuse is not served. The server and every
rank it started are stopped either way. The job is ``serve_driver.py``, run
under the tree's interpreter.

The contract owes it only when ``accuracy.serve.enabled`` says so, so a run
that resumes with a spec that never asked for it compiles the contract it
had. Turned on, it is owed per unit at the inner tier, at :func:`serve_member`,
and once more at delivery, on the last unit at :func:`delivery_member`.
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.environment import RunEnvironment
from ..spec.task_schema import DEFAULT_PARALLEL, TaskSchemaError, parse_parallel_segment
from . import serve_driver
from .gates import (
    BOOT_TIMEOUT_S,
    DEFAULT_BOOT_IMAGE_PROMPT,
    EVAL_DIALECT,
    MODE_REQUIRE,
    MODELING_V2_ENV,
    BootSpec,
    GateResult,
    ShellRunner,
    _default_shell_runner,
    _verdict_factory,
    _with_mode,
    gate_workdir,
    inspect_tree,
    launch_overlay,
    load_llm_api_options,
    member_llm_api_overlay,
    merge_options,
    missing_keywords,
    order_members_by_cost,
    resolve_boot_spec,
    resolve_llm_api_options,
    topology_flags,
)
from .harness import workload_io

GATE_NAME = "serve"

#: The driver script, run as a file under the measured tree's interpreter.
SERVE_DRIVER = Path(__file__).resolve().parent / "serve_driver.py"

COMPLETIONS = "/v1/completions"
CHAT_COMPLETIONS = "/v1/chat/completions"

DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 32
#: Seconds one request may take once the server is healthy.
REQUEST_TIMEOUT_S = 600
#: Seconds the job may spend stopping the server and reaping its ranks: the
#: launcher's grace, then the device holders' (``tune/bench.py``'s ``reap``)
#: and the driver releasing their memory.
TEARDOWN_S = 600


@dataclass(frozen=True)
class ServeSpec:
    """``accuracy.serve``: whether the contract owes the serve gate, and what it sends."""

    enabled: bool = False
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Seconds the server may take to answer ``/health``.
    timeout: int = BOOT_TIMEOUT_S

    @classmethod
    def from_task(cls, task_data: Mapping[str, Any]) -> "ServeSpec":
        block = dict((task_data.get("accuracy") or {}).get("serve") or {})
        default = cls()
        return cls(
            enabled=bool(block.get("enabled", default.enabled)),
            prompt=str(block.get("prompt") or default.prompt),
            max_tokens=int(block.get("max_tokens") or default.max_tokens),
            timeout=int(block.get("timeout") or default.timeout),
        )


def _width(member: str) -> int:
    try:
        return parse_parallel_segment(member)[1] or 0
    except TaskSchemaError:
        return 0


def serve_member(members: Sequence[str]) -> str:
    """The member a unit's serve check runs at: the widest, the cheapest to start among equals.

    The server's own path -- its CLI, the input processor, the OpenAI
    handlers -- is the same at every member, so one member covers it; the
    engine at each member is the boot gate's. The widest is the member a
    checkpoint too large for fewer devices still fits at, and it serves the
    way production and disaggregated serving do: ranks created by ``mpirun``
    before the engine, and a leader answering HTTP over a multi-rank executor,
    which a one-process server never runs.
    """
    return max(order_members_by_cost(members), key=_width)


def delivery_member(members: Sequence[str]) -> str | None:
    """The one-rank member the delivered tree is also served at, when :func:`serve_member` is wider.

    One rank serves in one process, with no ``mpirun`` and no launcher: how
    ``trtllm-serve <checkpoint>`` runs when given nothing else, and a launch
    the wider member never takes. ``None`` when the unit has no one-rank
    member, or serves at one rank already.
    """
    if _width(serve_member(members)) <= 1:
        return None
    single = [m for m in order_members_by_cost(members) if _width(m) == 1]
    return single[0] if single else None


def write_serve_options(
    path: str | Path,
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    member: str | None,
) -> Path:
    """The unit's LLM-API options as the accuracy gate serves them, as one ``--config`` file.

    The unit's file (``accuracy.llm_api_options``), the member's overlay
    (attention DP, which no topology flag carries), ``modeling_v2`` from
    the environment's switch, and beneath them the settings the unit states
    as runtime flags (:func:`~.gates.launch_overlay`): what
    :func:`~.gates.eval_command` gives ``trtllm-eval``. Written in the gate's own directory, not beside the
    unit's file where the accuracy gate writes its copy, so neither gate
    rewrites a file the other's job may be reading.
    """
    source = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    overlay = dict(member_llm_api_overlay(member))
    mode = environment.env.get(MODELING_V2_ENV)
    if mode:
        overlay["modeling_v2"] = mode
    merged = merge_options(
        merge_options(launch_overlay(task_data, unit), load_llm_api_options(source)), overlay
    )
    header = (
        f"# GENERATED by the serve gate for the `{member}` member of the `{unit}` unit. "
        f"Do not edit.\n"
        f"# {source or '(no unit options file)'} plus: {', '.join(sorted(overlay)) or 'nothing'}\n"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header + yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
    return target


def serve_args(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    member: str | None,
    options: str | Path,
) -> list[str]:
    """``trtllm-serve``'s flags after the checkpoint: the options file, topology and limits.

    ``trtllm-serve`` takes ``trtllm-eval``'s spellings, so these are the
    accuracy gate's: ``--tp_size``/``--ep_size`` from the member (attention DP
    is in the options file) and ``accuracy.max_seq_len``/``max_batch_size``,
    without which the server sizes its cache for the checkpoint's own maximum.
    """
    accuracy = task_data.get("accuracy") or {}
    segment = str(task_data.get("parallel") or DEFAULT_PARALLEL)
    args = [
        "--config",
        str(options),
        *topology_flags(segment, environment.world_size, member, dialect=EVAL_DIALECT),
    ]
    for flag in ("max_seq_len", "max_batch_size"):
        if accuracy.get(flag) is not None:
            args.append(f"--{flag}={accuracy[flag]}")
    return args


def serve_requests(
    task_data: Mapping[str, Any], spec: ServeSpec, boot: BootSpec
) -> list[dict[str, Any]]:
    """What the gate asks: one completion, then the boot workload's prompts the boot gate judges.

    A plain boot prompt goes as a completion when the workload names
    keywords; a prompt with checks of its own, or with images, goes on its
    own: as a completion, or with its first image as a chat. Each carries
    ``boot`` (``["prompt", n]`` or ``["item", n]``), its place in the
    workload, so :func:`judge` holds its answer to that prompt's checks. An
    image is resolved against the tree under test (the boot workload's
    ``repository`` asset, ``merlion.png``, by default for a multimodal unit)
    and the driver sends it inline. The chat's text is the boot prompt when
    it is plain text, and :data:`~.gates.DEFAULT_BOOT_IMAGE_PROMPT` when it
    is pre-rendered (``render: raw``): the server applies the chat template
    and places the image itself, so a prompt with placeholders of its own
    would carry the image twice.
    """
    model = str(task_data["checkpoint_path"])

    def completion(rid: str, prompt: str, max_tokens: int, place: list[Any] | None):
        request: dict[str, Any] = {
            "id": rid,
            "endpoint": COMPLETIONS,
            "body": {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
        }
        if place is not None:
            request["boot"] = place
        return request

    requests: list[dict[str, Any]] = [completion("text", spec.prompt, spec.max_tokens, None)]
    workload = boot.requests(task_data.get("trtllm_repo_path"))
    plain = len(boot.prompts)
    if boot.keywords:
        for n in range(plain):
            requests.append(completion(f"boot{n}", boot.prompts[n], spec.max_tokens, ["prompt", n]))
    for n, item in enumerate(boot.items):
        if not (item.images or item.keywords or item.keywords_any):
            continue
        request = workload[plain + n]
        if not request.get("images"):
            requests.append(completion(f"item{n}", item.prompt, spec.max_tokens, ["item", n]))
            continue
        text = item.prompt if item.render == "loader" else DEFAULT_BOOT_IMAGE_PROMPT
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": request["images"][0]}},
        ]
        requests.append(
            {
                "id": "image" if not any(r["id"] == "image" for r in requests) else f"image{n}",
                "endpoint": CHAT_COMPLETIONS,
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": spec.max_tokens,
                    "temperature": 0.0,
                },
                "boot": ["item", n],
            }
        )
    return requests


def serve_command(environment: RunEnvironment, spec_path: str | Path) -> str:
    """The driver's launch over ``spec_path``: one process, which creates the server's ranks itself."""
    return environment.python_command(
        f"{shlex.quote(str(SERVE_DRIVER))} --spec {shlex.quote(str(spec_path))}",
        single_process=True,
    )


def judge(
    output: str,
    requests: Sequence[Mapping[str, Any]],
    returncode: int = 0,
    boot: BootSpec | None = None,
) -> tuple[bool, str, tuple[str, ...]]:
    """What the driver's output says: passed, why, and what a reader should see besides.

    Read from the driver's records only, never from the server's log. With
    ``boot``, each answer to one of its prompts (``boot`` in the request) is
    held to that workload's keywords, as the boot gate holds the engine's.
    """
    servers = workload_io.parse_marked(output, serve_driver.SERVER_MARKER)
    if not servers:
        return False, f"the serve driver exited {returncode} before it reported the server", ()
    server = servers[-1]
    warnings: tuple[str, ...] = ()
    if server.get("left"):
        warnings = (f"pid(s) {server['left']} still held the server's devices after it stopped",)
    if not server.get("healthy"):
        cause = f" -- {server['cause']}" if server.get("cause") else ""
        return (
            False,
            f"trtllm-serve never became healthy: {server.get('error') or 'no reason given'}"
            f"{cause} (log {server.get('log')})",
            warnings,
        )
    answers = {
        int(r.get("index", -1)): r
        for r in workload_io.parse_marked(output, workload_io.RECORD_MARKER)
    }
    problems: list[str] = []
    for index, request in enumerate(requests):
        record = answers.get(index)
        where = f"{request['id']} ({request['endpoint']})"
        if record is None:
            problems.append(f"{where}: no answer recorded")
        elif record.get("status") != 200:
            status = record.get("status")
            got = f"HTTP {status}" if status is not None else "no response"
            problems.append(f"{where}: {got}: {str(record.get('error') or '')[:200]}")
        elif not str(record.get("text") or "").strip():
            problems.append(f"{where}: empty text (finish_reason {record.get('finish_reason')})")
    if not problems and boot is not None:
        texts: list[str | None] = [None] * (len(boot.prompts) + len(boot.items))
        for index, request in enumerate(requests):
            place = request.get("boot")
            if place:
                at = int(place[1]) + (len(boot.prompts) if place[0] == "item" else 0)
                texts[at] = str(answers[index].get("text") or "")
        missing = missing_keywords(boot, texts)
        if missing:
            problems.append(
                f"the answers are missing {len(missing)} keyword(s) the boot workload expects: "
                f"{missing} ({boot.source})"
            )
    if problems:
        return False, "trtllm-serve came up, but " + "; ".join(problems), warnings
    if returncode != 0:
        return False, f"every answer came back, but the serve driver exited {returncode}", warnings
    said = "; ".join(
        f"{request['id']}: {str(answers[i].get('text'))[:40]!r}"
        for i, request in enumerate(requests)
    )
    return (
        True,
        f"trtllm-serve healthy after {server.get('boot_s')}s and answered {len(requests)} "
        f"request(s) with text ({said})",
        warnings,
    )


def run_serve_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    runner: ShellRunner = _default_shell_runner,
    features: Any | None = None,
    unit: str | None = None,
    member: str | None = None,
    allow_dirty: bool | None = None,
    workdir: str | Path | None = None,
    spec: ServeSpec | None = None,
    boot: BootSpec | None = None,
) -> GateResult:
    """Serve the unit's target with ``trtllm-serve``, ask it, and stop it.

    ``environment`` is at the member's world size, as for every gate; the
    driver creates that many ranks. ``boot`` is the unit's boot workload:
    what is asked besides the completion, and what the answers are held to
    (default :func:`~.gates.resolve_boot_spec` over ``features``).
    """
    spec = spec or ServeSpec.from_task(task_data)
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name=GATE_NAME, member=member, passed=False, detail=state.block_detail())

    env = _with_mode(environment, MODE_REQUIRE)
    if boot is None:
        boot = resolve_boot_spec(
            task_data, unit, task_data.get("features") if features is None else features
        )
    directory = gate_workdir(workdir, GATE_NAME, task_data, environment=environment)
    tag = f"{unit or 'unit'}.{member or 'member'}"
    options = write_serve_options(
        directory / f"serve.{tag}.llm_api.yaml", task_data, env, unit=unit, member=member
    )
    requests = serve_requests(task_data, spec, boot)
    driver_spec = {
        "model": str(task_data["checkpoint_path"]),
        "serve_args": serve_args(task_data, env, member=member, options=options),
        "world_size": _width(member or "") or env.world_size,
        "server_log": str(directory / f"serve.{tag}.server.log"),
        "boot_timeout": spec.timeout,
        "request_timeout": REQUEST_TIMEOUT_S,
        "requests": requests,
    }
    spec_path = directory / f"serve.{tag}.json"
    spec_path.write_text(json.dumps(driver_spec, indent=1, ensure_ascii=False), encoding="utf-8")
    command = serve_command(env, spec_path)
    timeout = spec.timeout + len(requests) * REQUEST_TIMEOUT_S + TEARDOWN_S

    started = time.monotonic()
    result = runner(command, None, timeout)
    combined = result.stdout + result.stderr
    ran = dict(
        name=GATE_NAME,
        member=member,
        command=command,
        duration_s=time.monotonic() - started,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
    )
    if result.infrastructure:
        return _verdict(
            passed=False,
            detail=f"did not run to a verdict: {result.infrastructure}",
            infrastructure=result.infrastructure,
            **ran,
        )
    if result.timed_out:
        return _verdict(passed=False, detail=f"timed out after {timeout}s", **ran)
    passed, detail, warnings = judge(combined, requests, result.returncode, boot)
    return _verdict(passed=passed, detail=detail, warnings=warnings, **ran)


__all__ = [
    "CHAT_COMPLETIONS",
    "COMPLETIONS",
    "GATE_NAME",
    "SERVE_DRIVER",
    "ServeSpec",
    "delivery_member",
    "judge",
    "run_serve_gate",
    "serve_args",
    "serve_command",
    "serve_member",
    "serve_requests",
    "write_serve_options",
]
