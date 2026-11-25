# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A drafter's acceptance on a declared workload, held to 1.0 and to a declared floor."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

from ..infra.compute.commands import ended_early
from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import (
    SPECULATORS,
    TaskSchemaError,
    load_prompts_file,
    resolve_gate_timeout,
    resolve_per_unit,
)
from .check_config import check_configuration, resolve_llm_api_options
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    ShellRunner,
    default_shell_runner,
    engine_driver_command,
    gate_workdir,
    parse_driver_records,
    with_mode,
    write_driver_spec,
)
from .harness import workload_io
from .runtime_evidence import reuse_failure
from .verdict import GateResult, inspect_tree, verdict_factory

#: Tokens each acceptance prompt may generate when the task declares none.
DEFAULT_ACCEPTANCE_MAX_TOKENS = 256

#: What the engine refuses a request with when its prompt is longer than one step's token
#: budget and chunked prefill is off (``LLM._check_arguments``).
_PROMPT_OVER_BUDGET = re.compile(
    r"prompt length \(([0-9.]+)\) should not exceed max_num_tokens \((\d+)\)"
)


def prompt_over_budget(output: str) -> str:
    """Why the engine refused the workload, when a prompt outgrew ``max_num_tokens``; else ``""``."""
    found = _PROMPT_OVER_BUDGET.findall(output)
    if not found:
        return ""
    length, budget = found[-1]
    return (
        f"the engine refused a prompt of {float(length):.0f} tokens, longer than "
        f"max_num_tokens {budget} with chunked prefill off"
    )


def _workload_refused(why: str, workload: "AcceptanceWorkload") -> str:
    return (
        f"the engine could not run the acceptance workload: {why}. Workload: {workload.source}. "
        "This is a property of the workload, not of the target: use shorter prompts, raise "
        "`max_num_tokens`, or turn chunked prefill on for this check."
    )


#: Generation iterations the acceptance workload must reach, summed over its
#: prompts, before the gate reads a rate from it. A few short-answer prompts
#: that each end in EOS after 2-3 tokens reach only a handful of iterations,
#: too few for a rate to mean anything. A task may raise it through
#: ``accuracy.acceptance.min_draft_iterations``.
DEFAULT_MIN_DRAFT_ITERATIONS = 32


@dataclass(frozen=True)
class AcceptanceReading:
    """One acceptance measurement, with the requests it came from."""

    #: The gate's number: iteration-weighted over the requests (:func:`parse_acceptance`).
    value: float
    #: ``(tokens, iterations)`` per request, in request order.
    requests: tuple[tuple[int, int], ...]

    @property
    def draft_iterations(self) -> int:
        """Generation iterations over the whole workload.

        The first streamed iteration of every request is its context step,
        which verifies no draft, so it is not counted.
        """
        return sum(max(0, iterations - 1) for _, iterations in self.requests)

    def breakdown(self) -> str:
        """Per-prompt rates and their plain mean, beside the weighted value."""
        rates = [(n, t, i) for n, (t, i) in enumerate(self.requests) if i]
        if not rates:
            return ""
        per_prompt = ", ".join(f"[{n}] {t / i:.2f} ({t} tok/{i} it)" for n, t, i in rates)
        mean = sum(t / i for _, t, i in rates) / len(rates)
        return (
            f"per prompt {per_prompt}; prompt mean {mean:.2f}, "
            f"iteration-weighted {self.value:.2f} over {self.draft_iterations} "
            f"generation iterations"
        )


def parse_acceptance(stdout: str) -> AcceptanceReading | None:
    """The acceptance the engine driver's records show; ``None`` when they show none.

    Requests combine **iteration-weighted**, as ``sum(tokens) / sum(iterations)``,
    not as the mean of their ratios, which would let one short request dominate
    a long one.
    """
    records = parse_driver_records(stdout)
    parts = tuple((int(r.get("tokens", 0)), int(r.get("iterations", 0))) for r in records)
    iterations = sum(i for _, i in parts)
    if not iterations:
        return None
    return AcceptanceReading(value=sum(t for t, _ in parts) / iterations, requests=parts)


#: Prompts taken from the accuracy dataset when the task declares none.
DEFAULT_ACCEPTANCE_NUM_PROMPTS = 32


#: The share of a measured reference a floor derived from it keeps.
MIN_ACCEPTANCE_RATIO = 0.95


@dataclass(frozen=True)
class AcceptanceFloor:
    """An absolute acceptance floor, and the measurement it came from, if any."""

    value: float
    #: Where the reference drafter this floor derives from was measured;
    #: ``None`` when the task declared a number with no measurement behind it.
    source: str | None = None
    #: The measured reference value the floor was derived from, when it was.
    measured: float | None = None

    def warning(self) -> str | None:
        """What a verdict says when the floor is not tied to a measurement."""
        if self.source or self.measured is not None:
            return None
        return (
            f"the acceptance floor {self.value:.2f} is not tied to a measured reference: "
            f"declare `floor_source` naming the reference-drafter measurement it was "
            f"derived from, or `measured_reference` with the value itself. A floor chosen "
            f"below every real drafter's acceptance passes a wrong drafter too"
        )


def resolve_acceptance_floor_spec(
    accuracy: Mapping[str, Any], unit: str | None, features: Any = ()
) -> AcceptanceFloor | None:
    """The absolute floor ``accuracy.acceptance`` sets for ``unit``, if any.

    ``unit``'s own entry; else that of the last speculator among ``features``
    (the units it builds on), whose drafter it runs; else ``default``.
    ``min_absolute`` is the floor itself. ``measured_reference`` is what a
    reference drafter measured on this workload; with no ``min_absolute``
    the floor is that times :data:`MIN_ACCEPTANCE_RATIO`. ``floor_source``
    names where the measurement was taken.
    """
    block = accuracy.get("acceptance")
    if not isinstance(block, dict):
        return None
    drafters = [str(f) for f in features or () if str(f) in SPECULATORS]
    chosen = next(
        (
            b
            for name in (unit, *reversed(drafters), "default")
            if (b := block.get(name)) is not None
        ),
        None,
    )
    if isinstance(chosen, dict):
        source = chosen.get("floor_source")
        measured = chosen.get("measured_reference")
        measured = float(measured) if isinstance(measured, (int, float)) else None
        if chosen.get("min_absolute") is not None:
            try:
                return AcceptanceFloor(float(chosen["min_absolute"]), source, measured)
            except (TypeError, ValueError):
                return None
        if measured is not None:
            return AcceptanceFloor(measured * MIN_ACCEPTANCE_RATIO, source, measured)
        return None
    try:
        return AcceptanceFloor(float(chosen))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AcceptanceWorkload:
    """What the acceptance gate sends, and how many generation iterations it needs back."""

    #: Declared prompts; empty when ``dataset`` supplies them.
    prompts: tuple[str, ...]
    max_tokens: int
    min_draft_iterations: int
    #: Where the prompts came from, named in a verdict that blames them.
    source: str
    #: The engine driver's ``dataset`` block, when the prompts come from one.
    dataset: Mapping[str, Any] | None = None

    def requests(self) -> list[dict[str, Any]]:
        """The declared prompts as engine-driver requests."""
        return [
            {"id": f"p{n}", "prompt": prompt, "max_tokens": self.max_tokens}
            for n, prompt in enumerate(self.prompts)
        ]


def acceptance_dataset(
    accuracy: Mapping[str, Any], unit: str | None, setting: Callable[[str], Any]
) -> tuple[dict[str, Any] | None, str]:
    """The dataset an undeclared acceptance workload is drawn from, and its name.

    ``accuracy.acceptance.dataset`` when declared; otherwise the accuracy
    benchmark's own dataset -- ``accuracy.dataset_path`` when the run keeps
    it locally, else the benchmark's Hugging Face id, read from the local
    cache only (:data:`.harness.workload_io.KNOWN_DATASETS`). A unit whose
    benchmark has no text dataset uses the run-wide benchmark's.
    """
    known: dict[str, Any] = {}
    for scope in (unit, "default"):
        benchmark = resolve_per_unit(accuracy, "benchmark", scope)
        if isinstance(benchmark, str) and benchmark in workload_io.KNOWN_DATASETS:
            known = dict(workload_io.KNOWN_DATASETS[benchmark])
            break
    declared = setting("dataset")
    if declared:
        source, where = str(declared), "accuracy.acceptance.dataset"
    elif accuracy.get("dataset_path") and known:
        source, where = str(accuracy["dataset_path"]), "accuracy.dataset_path"
    elif known:
        source, where = str(known["source"]), "the accuracy benchmark's dataset"
    else:
        return None, ""
    dataset = {
        "source": source,
        "num": int(setting("num_prompts") or DEFAULT_ACCEPTANCE_NUM_PROMPTS),
        "text_field": setting("text_field") or known.get("text_field"),
        "split": setting("split") or known.get("split", "test"),
        "config": setting("dataset_config") or (None if declared else known.get("config")),
        "template": setting("prompt_template") or "{text}",
    }
    return dataset, f"{dataset['num']} prompts from {where} ({source})"


def resolve_acceptance_workload(
    accuracy: Mapping[str, Any], unit: str | None
) -> AcceptanceWorkload:
    """The acceptance workload ``unit`` declares, or the accuracy dataset's prompts.

    Read from ``accuracy.acceptance``, most specific level first: the unit's
    own entry, then its ``default`` entry, then the block's top level.
    ``prompts`` and ``prompts_file`` are one setting, so the level that states
    either wins for both. With neither, the workload is the first
    ``num_prompts`` questions of the accuracy dataset (:func:`acceptance_dataset`).

    Raises :class:`~.task_schema.TaskSchemaError` when a declared prompts
    file cannot be read, or when nothing declares a workload and the
    benchmark has no dataset to draw one from.
    """
    block = accuracy.get("acceptance")
    block = block if isinstance(block, dict) else {}
    levels = [
        (f"accuracy.acceptance.{name}", block.get(name))
        for name in (unit, "default")
        if name is not None
    ]
    levels.append(("accuracy.acceptance", block))
    levels = [(where, entry) for where, entry in levels if isinstance(entry, dict)]

    def setting(key: str) -> Any:
        for _, entry in levels:
            if entry.get(key) is not None:
                return entry[key]
        return None

    prompts: tuple[str, ...] = ()
    source = ""
    dataset: dict[str, Any] | None = None
    for where, entry in levels:
        if entry.get("prompts") is not None:
            prompts = tuple(str(p) for p in entry["prompts"])
            source = f"{where}.prompts"
            break
        if entry.get("prompts_file") is not None:
            prompts = load_prompts_file(entry["prompts_file"])
            source = f"{where}.prompts_file ({entry['prompts_file']})"
            break
    if not prompts:
        dataset, source = acceptance_dataset(accuracy, unit, setting)
        if dataset is None:
            raise TaskSchemaError(
                "no acceptance workload: declare `accuracy.acceptance.prompts`, "
                "`prompts_file` or `dataset` (a local path or a cached dataset id); the "
                "benchmark has no dataset this gate knows how to draw prompts from"
            )
    declared_tokens = setting("max_tokens")
    declared_minimum = setting("min_draft_iterations")
    return AcceptanceWorkload(
        prompts=prompts,
        max_tokens=int(
            DEFAULT_ACCEPTANCE_MAX_TOKENS if declared_tokens is None else declared_tokens
        ),
        min_draft_iterations=int(
            DEFAULT_MIN_DRAFT_ITERATIONS if declared_minimum is None else declared_minimum
        ),
        source=source,
        dataset=dataset,
    )


def run_acceptance_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
    unit: str | None = None,
    features: Any | None = None,
    member: str | None = None,
) -> GateResult:
    """Measure a drafter's acceptance in one load, and hold it to 1.0 and to a declared floor.

    A workload under ``min_draft_iterations`` generation iterations fails as too
    short to measure, not as a low number. Then: against 1.0 (every draft was
    rejected), and against the floor ``accuracy.acceptance`` declares for the unit
    (``min_absolute``, or ``measured_reference`` times :data:`MIN_ACCEPTANCE_RATIO`).
    With no floor the verdict says it stands on 1.0 alone. The load may run
    ``timeout`` seconds: ``accuracy.acceptance``'s ``timeout`` unless the caller
    says, else :data:`BOOT_TIMEOUT_S`.
    """
    _verdict = partial(verdict_factory(inspect_tree(task_data)), name="acceptance", member=member)
    accuracy = task_data.get("accuracy") or {}
    if timeout is None:
        timeout = resolve_gate_timeout(accuracy, "acceptance", unit) or BOOT_TIMEOUT_S
    try:
        workload = resolve_acceptance_workload(accuracy, unit)
    except TaskSchemaError as exc:
        return _verdict(
            passed=False,
            detail=f"the declared acceptance workload cannot be read: {exc}",
        )
    config = check_configuration(
        task_data,
        environment,
        unit=unit,
        features=task_data.get("features") if features is None else features,
        max_tokens=workload.max_tokens,
        member=member,
    )
    spec_path = write_driver_spec(
        gate_workdir(workdir) / f"acceptance.{unit or 'unit'}.{member or 'member'}.json",
        config=config,
        requests=workload.requests(),
        dataset=workload.dataset,
        max_tokens=workload.max_tokens,
        reuse_probe=True,
    )
    command = engine_driver_command(with_mode(environment, MODE_REQUIRE), spec_path)
    started = time.monotonic()
    result = runner(command, None, timeout)
    combined = result.stdout + result.stderr
    ran: dict[str, Any] = dict(
        command=command,
        duration_s=time.monotonic() - started,
        stdout_tail="\n".join(combined.splitlines()[-40:]),
    )
    if result.infrastructure:
        return _verdict(
            passed=False,
            detail=f"the acceptance load did not run to a verdict: {result.infrastructure}",
            infrastructure=result.infrastructure,
            **ran,
        )
    refused = "" if result.ok or result.timed_out else prompt_over_budget(combined)
    if refused:
        return _verdict(passed=False, detail=_workload_refused(refused, workload), **ran)
    if not result.ok:
        ended = ended_early(result, timeout) if result.timed_out else f"exited {result.returncode}"
        return _verdict(passed=False, detail=f"the acceptance load {ended}", **ran)
    failure = reuse_failure(
        combined, config["llm"], options=resolve_llm_api_options(accuracy, unit)
    )
    if failure:
        return _verdict(passed=False, detail=failure, **ran)
    reading = parse_acceptance(combined)
    if reading is None:
        return _verdict(
            passed=False,
            detail="could not read acceptance from the engine driver's records",
            **ran,
        )
    ran.update(measured=reading.value, breakdown=reading.breakdown())
    said = f"Workload: {workload.source}, max_tokens {workload.max_tokens}"
    if reading.draft_iterations < workload.min_draft_iterations:
        return _verdict(
            passed=False,
            detail=f"workload too short to measure acceptance: {reading.draft_iterations} "
            f"generation iteration(s) over {len(reading.requests)} prompt(s), fewer than the "
            f"{workload.min_draft_iterations} this gate needs, so its rate "
            f"({reading.value:.2f}) says nothing about the draft path -- the answers end "
            f"before drafts are verified. {said}. This is a property of the workload, not of "
            f"the target.",
            **ran,
        )
    if reading.value <= 1.0:
        return _verdict(
            passed=False,
            detail=f"acceptance_length {reading.value:.4f} at or below 1.0 means every draft was "
            f"rejected — the draft path is computing the wrong thing",
            **ran,
        )
    floor = resolve_acceptance_floor_spec(
        accuracy, unit, task_data.get("features") if features is None else features
    )
    if floor is None:
        return _verdict(
            passed=True,
            detail=f"acceptance_length {reading.value:.4f} over {reading.draft_iterations} "
            f"generation iteration(s) of {len(reading.requests)} prompt(s), held to 1.0 alone. "
            f"{said}",
            warnings=(
                "held to 1.0 alone: `accuracy.acceptance` declares no floor for this unit "
                "(`min_absolute`, or `measured_reference` with its `floor_source`), so a "
                "drafter that accepts little still passes",
            ),
            explained=True,
            **ran,
        )
    source = f" (from {floor.source})" if floor.source else ""
    derived = floor.measured is not None and math.isclose(
        floor.value, floor.measured * MIN_ACCEPTANCE_RATIO
    )
    if derived:
        source = f" ({MIN_ACCEPTANCE_RATIO} x measured reference {floor.measured}){source}"
    passed = reading.value >= floor.value
    return _verdict(
        passed=passed,
        detail=f"acceptance_length {reading.value:.4f} over {reading.draft_iterations} "
        f"generation iteration(s) of {len(reading.requests)} prompt(s) vs declared floor "
        f"{floor.value:.4f}{source}: {'met' if passed else 'NOT met'}. {said}",
        reference=floor.value,
        warnings=tuple(w for w in (floor.warning(),) if w),
        explained=True,
        **ran,
    )


__all__ = [
    "AcceptanceFloor",
    "AcceptanceReading",
    "AcceptanceWorkload",
    "DEFAULT_ACCEPTANCE_MAX_TOKENS",
    "DEFAULT_ACCEPTANCE_NUM_PROMPTS",
    "DEFAULT_MIN_DRAFT_ITERATIONS",
    "MIN_ACCEPTANCE_RATIO",
    "acceptance_dataset",
    "parse_acceptance",
    "prompt_over_budget",
    "resolve_acceptance_floor_spec",
    "resolve_acceptance_workload",
    "run_acceptance_gate",
]
