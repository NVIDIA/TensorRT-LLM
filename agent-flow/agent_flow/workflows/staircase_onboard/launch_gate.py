# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch-path accuracy: a small accuracy smoke through the unit's own deployment entry.

The accuracy gate runs ``trtllm-eval``, which drives the LLM API in process.
A served model takes a different path: the OpenAI server parses the request,
renders the chat template, loads the media, and only then reaches the
engine. Every image request to ``trtllm-serve`` returned HTTP 400 for a
whole run -- the tokenizer deep-copied the server's pending media loads --
while the multimodal unit's accuracy gate, on the LLM API, passed. The disagg
unit's accuracy gates ran the aggregated engine and never the proxy.

So each unit also sends a small workload through its OWN entry:

* ``trtllm-serve`` (aggregated), started with the unit's LLM-API options and
  the member's topology, behind ``mpirun ... trtllm-llmapi-launch`` above one
  rank -- the default for every unit;
* the unit's declared launcher (``accuracy.launch.<unit>.command``) for a
  deployment ``trtllm-serve`` alone is not: the ``disagg`` unit must declare
  one, which starts the context and generation servers and the proxy and
  prints ``STAIRCASE_ENDPOINT=<base url>`` once it serves (see
  ``harness/serve_smoke.py`` for the contract).

The workload is ``num_samples`` (default 32) questions of the accuracy
dataset through ``/v1/chat/completions``, scored by the dataset's scorer,
plus -- for a unit whose target implements ``multimodal`` -- the boot
gate's image prompt sent as a ``data:`` URL. The verdict fails on any
non-200 response (the class of defect this gate exists for), on an image
answer missing its keywords, and on accuracy under the floor: declared
``min_accuracy``, else ``anchor - k * SE`` at the smoke's own sample count,
which at 32 questions sits ~10 points under a 92 % anchor -- a garbage
detector, not a second accuracy gate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import stats
from .environment import RunEnvironment
from .gates import (
    BOOT_TIMEOUT_S,
    EVAL_DIALECT,
    MODE_REQUIRE,
    GateResult,
    ShellRunner,
    _default_shell_runner,
    _verdict_factory,
    _with_mode,
    _write_member_options,
    acceptance_dataset,
    gate_workdir,
    inspect_tree,
    member_llm_api_overlay,
    resolve_boot_spec,
    resolve_llm_api_options,
    resolve_per_unit,
    resolve_reference,
    topology_flags,
)
from .harness import workload_io
from .task_schema import LAUNCH_KEYS

#: The smoke driver, run as one process that owns the claim's devices.
SERVE_SMOKE = Path(__file__).resolve().parent / "harness" / "serve_smoke.py"

DEFAULT_NUM_SAMPLES = 32
DEFAULT_MAX_TOKENS = 512
ENDPOINTS: tuple[str, ...] = ("chat", "completions")
#: The unit whose entry is not ``trtllm-serve`` alone, and must declare one.
DISAGG_UNIT = "disagg"
#: Handed to a declared launcher, so it starts the unit's own configuration.
LAUNCH_MEMBER_ENV = "STAIRCASE_LAUNCH_MEMBER"
LAUNCH_OPTIONS_ENV = "STAIRCASE_LAUNCH_LLM_API_OPTIONS"
LAUNCH_SERVE_ENV = "STAIRCASE_LAUNCH_SERVE_COMMAND"


@dataclass(frozen=True)
class LaunchConfig:
    """``accuracy.launch`` for one unit, resolved."""

    enabled: bool = True
    members: str = "all"
    num_samples: int = DEFAULT_NUM_SAMPLES
    max_tokens: int = DEFAULT_MAX_TOKENS
    endpoint: str = "chat"
    dataset: Mapping[str, Any] | None = None
    source: str = ""
    scorer: str | None = None
    min_accuracy: float | None = None
    command: str | None = None
    timeout: int = 2 * BOOT_TIMEOUT_S
    units: tuple[str, ...] | None = None


def _merged(accuracy: Mapping[str, Any], unit: str | None) -> dict[str, Any]:
    block = accuracy.get("launch")
    block = block if isinstance(block, Mapping) else {}
    merged = {k: v for k, v in block.items() if k in LAUNCH_KEYS}
    own = block.get(unit) if unit is not None else None
    if isinstance(own, Mapping):
        merged.update(own)
    return merged


def applies(accuracy: Mapping[str, Any], unit: str | None) -> bool:
    """Whether the launch-path smoke runs in ``unit`` (default: every unit)."""
    merged = _merged(accuracy, unit)
    units = merged.get("units")
    return bool(merged.get("enabled", True)) and (not units or str(unit) in map(str, units))


def resolve_launch_config(accuracy: Mapping[str, Any], unit: str | None) -> LaunchConfig:
    """Read ``accuracy.launch`` for ``unit``: run-wide keys, the unit's entry over them."""
    merged = _merged(accuracy, unit)

    def setting(key: str) -> Any:
        if key == "num_prompts":
            return merged.get("num_samples") or DEFAULT_NUM_SAMPLES
        return merged.get(key)

    dataset, source = acceptance_dataset(accuracy, unit, setting)
    scorer = merged.get("scorer")
    if scorer is None and dataset is not None:
        for scope in (unit, "default"):
            benchmark = resolve_per_unit(accuracy, "benchmark", scope)
            known = workload_io.KNOWN_DATASETS.get(str(benchmark))
            if known:
                scorer = known.get("scorer")
                break
    if dataset is not None and merged.get("answer_field"):
        dataset = {**dataset, "answer_field": merged["answer_field"]}
    units = merged.get("units")
    return LaunchConfig(
        enabled=bool(merged.get("enabled", True)),
        members=str(merged.get("members") or "all"),
        num_samples=int(merged.get("num_samples") or DEFAULT_NUM_SAMPLES),
        max_tokens=int(merged.get("max_tokens") or DEFAULT_MAX_TOKENS),
        endpoint=str(merged.get("endpoint") or "chat"),
        dataset=dataset,
        source=source,
        scorer=None if scorer in (None, "none") else str(scorer),
        min_accuracy=float(merged["min_accuracy"])
        if merged.get("min_accuracy") is not None
        else None,
        command=merged.get("command"),
        timeout=int(merged.get("timeout") or 2 * BOOT_TIMEOUT_S),
        units=tuple(str(u) for u in units) if units else None,
    )


def select_members(config: LaunchConfig, members: Sequence[str]) -> list[str]:
    """The members the smoke runs at: all of them, or the cheapest."""
    return list(members) if config.members == "all" else list(members[:1])


def serve_command(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    member: str | None,
    mode: str = MODE_REQUIRE,
) -> str:
    """``trtllm-serve`` for ``unit`` at ``member``, on ``{port}``, as engine work.

    The flags are the accuracy gate's: the topology in the eval dialect (the
    server's flag names are the same), the unit's LLM-API options with the
    member's overlay (attention DP), and the accuracy block's sequence and
    batch limits. Under ``slurm`` the server is started with ``mpirun`` inside
    the smoke's own allocation rather than with a nested ``srun``.
    """
    accuracy = dict(task_data.get("accuracy") or {})
    parts = [
        "trtllm-serve",
        f'"{task_data["checkpoint_path"]}"',
        "--host=127.0.0.1",
        "--port={port}",
        "--backend=pytorch",
        *topology_flags(
            str(task_data["parallel"]), environment.world_size, member, dialect=EVAL_DIALECT
        ),
    ]
    options = resolve_llm_api_options(accuracy, unit)
    overlay = member_llm_api_overlay(member)
    if overlay:
        options = _write_member_options(options, overlay, unit=unit, member=member)
    if options:
        parts.append(f'--extra_llm_api_options="{options}"')
    for flag in ("max_seq_len", "max_batch_size"):
        if accuracy.get(flag) is not None:
            parts.append(f"--{flag}={accuracy[flag]}")
    env = _with_mode(environment, mode)
    if env.kind == "slurm":
        env = replace(env, kind="local")
    return env.command(" ".join(parts))


def smoke_reference(
    task_data: Mapping[str, Any], unit: str | None, model_key: Callable[[str | None], str]
) -> float | None:
    """The anchor the smoke's text questions are judged against.

    The unit's own anchor when its benchmark is the dataset the smoke draws
    from; otherwise (a multimodal unit anchored on an image benchmark) the
    run-wide one. ``None`` when no anchor resolves; the smoke then judges
    HTTP status and keywords only.
    """
    accuracy = task_data.get("accuracy") or {}
    own = resolve_per_unit(accuracy, "benchmark", unit)
    scope = unit if str(own) in workload_io.KNOWN_DATASETS else "default"
    try:
        return resolve_reference(task_data, model_key=model_key(scope), unit=scope)
    except (KeyError, FileNotFoundError, ValueError):
        return None


def run_launch_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    reference: float | None = None,
    runner: ShellRunner = _default_shell_runner,
    unit: str | None = None,
    member: str | None = None,
    features: Sequence[str] | None = None,
    workdir: str | Path | None = None,
    timeout: int | None = None,
    config: LaunchConfig | None = None,
    allow_dirty: bool | None = None,
) -> GateResult:
    """Send the smoke workload through the unit's own entry and judge it; see the module docstring."""
    name = "launch"
    state = inspect_tree(task_data, allow_dirty=allow_dirty)
    _verdict = _verdict_factory(state)
    if state.blocked:
        return _verdict(name=name, member=member, passed=False, detail=state.block_detail())
    accuracy = task_data.get("accuracy") or {}
    config = config or resolve_launch_config(accuracy, unit)
    features = list(task_data.get("features") or () if features is None else features)
    if unit == DISAGG_UNIT and not config.command:
        return _verdict(
            name=name,
            member=member,
            passed=False,
            detail="the disagg unit's deployment entry is the disaggregated proxy, which "
            "trtllm-serve alone is not: declare `accuracy.launch.disagg.command`, a launcher "
            "that prints STAIRCASE_ENDPOINT=<base url> once it serves",
        )
    requests: list[dict[str, Any]] = []
    boot = resolve_boot_spec(task_data, unit, features)
    for item in boot.items:
        if item.images:
            request = next(
                r
                for r in boot.requests(task_data.get("trtllm_repo_path"))
                if r["prompt"] == item.prompt
            )
            requests.append(
                {
                    "id": f"image{len(requests)}",
                    "prompt": item.prompt,
                    "images": request["images"],
                    "keywords": list(item.keywords),
                    "keywords_any": list(item.keywords_any),
                    "max_tokens": 64,
                }
            )
    if config.dataset is None and not requests:
        return _verdict(
            name=name,
            member=member,
            passed=False,
            detail="no launch-path workload: the benchmark has no dataset this gate knows; "
            "declare `accuracy.launch.dataset` (and `scorer`)",
        )
    directory = gate_workdir(workdir, "launch", task_data) / str(member or "member")
    directory.mkdir(parents=True, exist_ok=True)
    spec: dict[str, Any] = {
        "endpoint": config.endpoint,
        "requests": requests,
        "dataset": dict(config.dataset) if config.dataset else None,
        "scorer": config.scorer,
        "max_tokens": config.max_tokens,
        "model": str(task_data["checkpoint_path"]),
        "startup_timeout_s": int(timeout or config.timeout),
        "log_dir": str(directory),
    }
    serve = serve_command(task_data, environment, unit=unit, member=member)
    launch_env = _with_mode(environment, MODE_REQUIRE)
    if config.command:
        spec["launcher_command"] = config.command
        options = resolve_llm_api_options(accuracy, unit)
        launch_env = replace(
            launch_env,
            env={
                **launch_env.env,
                LAUNCH_MEMBER_ENV: str(member or ""),
                LAUNCH_OPTIONS_ENV: str(options or ""),
                LAUNCH_SERVE_ENV: serve,
            },
        )
    else:
        spec["server_command"] = serve
    spec_path = directory / f"launch.{unit or 'unit'}.json"
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")

    from .witness import contrast_leg_command

    command = contrast_leg_command(launch_env, f"{SERVE_SMOKE} --spec {spec_path}")
    budget = int(timeout or config.timeout) + 600
    started = time.monotonic()
    result = runner(command, None, budget)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(name=name, member=member, command=command, duration_s=elapsed)
    summaries = workload_io.parse_marked(combined, workload_io.SERVE_SUMMARY_MARKER)
    summary = summaries[-1] if summaries else {}
    records = workload_io.parse_marked(combined, workload_io.SERVE_MARKER)
    entry = "the declared launcher" if config.command else "trtllm-serve"
    leftovers = summary.get("leftover_pids") or []
    left = (
        f"; {len(leftovers)} process(es) still alive after a graceful stop, not killed: {leftovers}"
        if leftovers
        else ""
    )
    if not summary.get("started"):
        server_tail = "\n".join(summary.get("server_tail") or ()) or tail
        return _verdict(
            passed=False,
            detail=f"{entry} never became healthy"
            + (" (timed out)" if result.timed_out else f" (smoke exited {result.returncode})")
            + left,
            stdout_tail=server_tail,
            **ran,
        )
    failures: list[str] = []
    bad = [r for r in records if int(r.get("status", 0)) != 200]
    if bad:
        codes = sorted({int(r.get("status", 0)) for r in bad})
        first = str(bad[0].get("error", ""))[:300]
        failures.append(
            f"{len(bad)} of {len(records)} request(s) through {entry} failed with HTTP "
            f"{codes} (first: {first!r}). The accuracy gate drives the LLM API in process, "
            f"so it cannot see the server's request parsing, chat rendering or media loading"
        )
    unseen = [r for r in records if r.get("keywords_missing")]
    if unseen:
        failures.append(
            f"{len(unseen)} image answer(s) missing their keywords: "
            f"{unseen[0].get('keywords_missing')} in {str(unseen[0].get('text', ''))[:120]!r}"
        )
    if len(records) < int(summary.get("requests", len(records))):
        failures.append(f"{summary.get('requests')} request(s) planned, {len(records)} recorded")
    measured = summary.get("accuracy")
    floor = None
    floor_rule = ""
    scored = int(summary.get("scored") or 0)
    if measured is not None and scored:
        if config.min_accuracy is not None:
            floor, floor_rule = (
                config.min_accuracy,
                f"declared min_accuracy {config.min_accuracy:.2f}",
            )
        elif reference is not None:
            policy = stats.resolve_floor_policy(accuracy)
            computed = stats.statistical_floor(reference, scored, policy.k)
            floor, floor_rule = computed.value, computed.describe()
        if floor is not None and float(measured) < floor:
            failures.append(
                f"accuracy {float(measured):.2f} over {scored} question(s) is under {floor:.2f} "
                f"({floor_rule}): the served path answers worse than the engine does"
            )
    images = sum(int(r.get("images") or 0) for r in records)
    head = (
        f"{len(records)} request(s) through {entry} at {config.endpoint} endpoint"
        + (f", {images} image(s)" if images else "")
        + (
            f"; accuracy {float(measured):.2f} over {scored} ({config.source})"
            if measured is not None
            else ""
        )
    )
    return _verdict(
        passed=not failures,
        detail="; ".join([head, *failures]) + left,
        measured=float(measured) if measured is not None else None,
        reference=reference,
        floor=floor,
        floor_rule=floor_rule,
        stdout_tail="" if not failures else tail,
        explained=True,
        **ran,
    )


__all__ = [
    "DISAGG_UNIT",
    "LAUNCH_MEMBER_ENV",
    "LAUNCH_OPTIONS_ENV",
    "LAUNCH_SERVE_ENV",
    "LaunchConfig",
    "SERVE_SMOKE",
    "applies",
    "resolve_launch_config",
    "run_launch_gate",
    "select_members",
    "serve_command",
    "smoke_reference",
]
