# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A drafter's acceptance: its workload, its counters, and what the reading is held to.

Speculative decoding preserves the output distribution by construction, so a draft path
that computes the wrong thing passes boot and accuracy and shows only as speed: accepted
tokens per step are the detector. A deployment that drafts is asked the acceptance
workload in its own session, and each answer says how it was drafted
(:func:`reading_of`); nothing here loads a model. The reading is held to 1.0 and to the
floor ``accuracy.acceptance`` declares for the deployment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..spec.task_schema import TaskSchemaError, load_prompts_file, resolve_per_unit
from .harness import workload_io

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


def _workload_refused(why: str, leg: str, workload: "AcceptanceWorkload") -> str:
    return (
        f"the {leg} leg could not run its workload: {why}. Workload: {workload.source}. "
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
    """One leg's acceptance measurement, with the requests it came from."""

    #: The check's number: iteration-weighted over the requests, as
    #: ``sum(tokens) / sum(iterations)``, not as the mean of their ratios, which would
    #: let one short request dominate a long one (:func:`reading_of`).
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


def reading_of(records: Sequence[Mapping[str, Any]]) -> AcceptanceReading | None:
    """The acceptance a session's answers show; ``None`` when none of them drafted.

    Each answer states how its request was drafted (``speculative_decoding``: the verify
    steps it took and the draft tokens they accepted). A request's tokens are its first,
    from the context step, and each verify step's own plus what that step accepted; its
    iterations are the context step and the verify steps.
    """
    parts: list[tuple[int, int]] = []
    for record in records:
        drafted = record.get("speculative_decoding")
        if not isinstance(drafted, Mapping):
            continue
        steps = int(drafted.get("num_spec_steps") or 0)
        accepted = int(drafted.get("total_accepted_draft_tokens") or 0)
        parts.append((1 + steps + accepted, 1 + steps))
    iterations = sum(i for _, i in parts)
    if not iterations:
        return None
    return AcceptanceReading(value=sum(t for t, _ in parts) / iterations, requests=tuple(parts))


def judge_session(
    task_data: Mapping[str, Any],
    deployment: Any,
    workload: "AcceptanceWorkload",
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The verdict fields of a drafter's acceptance, read from one session's answers.

    The session sent ``workload`` to the deployment's own endpoint, so no engine is
    loaded here. A workload under ``min_prompts`` prompts or ``min_draft_iterations``
    generation iterations fails as too small to measure, not as a low number. Then:
    against 1.0 (every draft rejected); against the floor ``accuracy.acceptance``
    declares for the deployment (``min_absolute``, or ``measured_reference`` times
    :data:`MIN_ACCEPTANCE_RATIO`). With no floor the verdict says it stands on 1.0 alone.
    """
    accuracy = task_data.get("accuracy") or {}
    answered = [r for r in records if r.get("status") == 200 and not r.get("error")]
    if len(answered) < len(records) or not records:
        bad = next((r for r in records if r not in answered), None)
        why = (
            f"`{bad.get('id')}` answered HTTP {bad.get('status')}: "
            f"{str(bad.get('error') or '')[:300]}"
            if bad
            else "no request came back"
        )
        refused = prompt_over_budget(str((bad or {}).get("error") or ""))
        if refused:
            return dict(passed=False, detail=_workload_refused(refused, "session's", workload))
        return dict(passed=False, detail=f"the acceptance workload was not answered: {why}")
    reading = reading_of(records)
    if reading is None:
        return dict(
            passed=False,
            detail="no answer says how it was drafted (`speculative_decoding`): the acceptance "
            "of this deployment is unknown, and unknown is not a pass",
        )
    base = dict(measured=reading.value, breakdown=reading.breakdown())
    said = f"Workload: {workload.source}, max_tokens {workload.max_tokens}"
    if len(reading.requests) < workload.min_prompts:
        return dict(
            passed=False,
            detail=f"workload too small to compare acceptance: {len(reading.requests)} "
            f"prompt(s) drafted, fewer than the {workload.min_prompts} a comparison needs. "
            f"{said}. This is a property of the workload, not of the target.",
            **base,
        )
    if reading.draft_iterations < workload.min_draft_iterations:
        return dict(
            passed=False,
            detail=f"workload too short to measure acceptance: {reading.draft_iterations} "
            f"generation iteration(s) over {len(reading.requests)} prompt(s), fewer than the "
            f"{workload.min_draft_iterations} this check needs, so its rate "
            f"({reading.value:.2f}) says nothing about the draft path. {said}. This is a "
            f"property of the workload, not of the target.",
            **base,
        )
    if reading.value <= 1.0:
        return dict(
            passed=False,
            detail=f"acceptance_length {reading.value:.4f} at or below 1.0 means every draft "
            f"was rejected: the draft path is computing the wrong thing",
            **base,
        )
    warnings: list[str] = []
    notes: list[str] = []
    floor = resolve_acceptance_floor_spec(accuracy, getattr(deployment, "name", None))
    if floor is not None:
        if floor.warning():
            warnings.append(floor.warning())
        if reading.value < floor.value:
            return dict(
                passed=False,
                reference=floor.value,
                detail=f"acceptance_length {reading.value:.4f} is under the declared floor "
                f"{floor.value:.4f}" + (f" ({floor.source})" if floor.source else ""),
                warnings=tuple(warnings),
                **base,
            )
        notes.append(f"floor {floor.value:.2f}")
    if floor is None:
        warnings.append(
            "held to 1.0 alone: `accuracy.acceptance` declares no floor for this deployment "
            "(`min_absolute`, or `measured_reference` with its `floor_source`), so a drafter "
            "that accepts little still passes"
        )
    return dict(
        passed=True,
        explained=True,
        reference=floor.value if floor is not None else None,
        detail=f"acceptance_length {reading.value:.4f} over {reading.draft_iterations} "
        f"generation iteration(s) of {len(reading.requests)} prompt(s), read from the "
        f"session's answers" + (f"; {', '.join(notes)}" if notes else "") + f". {said}",
        warnings=tuple(warnings),
        **base,
    )


#: Prompts an acceptance workload must hold before its rate is read: on a handful, a
#: change that only moves the generated text moves the rate.
#: ``accuracy.acceptance.min_prompts`` may lower it.
DEFAULT_ACCEPTANCE_MIN_PROMPTS = 32


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
    accuracy: Mapping[str, Any], unit: str | None
) -> AcceptanceFloor | None:
    """The absolute floor ``unit`` declares in ``accuracy.acceptance``, if any.

    ``min_absolute`` is the floor itself. ``measured_reference`` is what a
    reference drafter measured on this workload; with no ``min_absolute``
    the floor is that times :data:`MIN_ACCEPTANCE_RATIO`. ``floor_source``
    names where the measurement was taken.
    """
    block = accuracy.get("acceptance")
    if not isinstance(block, dict):
        return None
    chosen = block.get(unit)
    if chosen is None:
        chosen = block.get("default")
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
    """What the acceptance gate sends, and how much of it it needs back."""

    #: Declared prompts; empty when ``dataset`` supplies them.
    prompts: tuple[str, ...]
    max_tokens: int
    min_draft_iterations: int
    #: Where the prompts came from, named in a verdict that blames them.
    source: str
    #: The engine driver's ``dataset`` block, when the prompts come from one.
    dataset: Mapping[str, Any] | None = None
    min_prompts: int = DEFAULT_ACCEPTANCE_MIN_PROMPTS

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
    min_prompts = setting("min_prompts")
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
        min_prompts=int(DEFAULT_ACCEPTANCE_MIN_PROMPTS if min_prompts is None else min_prompts),
    )


__all__ = [
    "AcceptanceFloor",
    "AcceptanceReading",
    "AcceptanceWorkload",
    "DEFAULT_ACCEPTANCE_MAX_TOKENS",
    "DEFAULT_ACCEPTANCE_MIN_PROMPTS",
    "DEFAULT_ACCEPTANCE_NUM_PROMPTS",
    "DEFAULT_MIN_DRAFT_ITERATIONS",
    "MIN_ACCEPTANCE_RATIO",
    "acceptance_dataset",
    "judge_session",
    "prompt_over_budget",
    "reading_of",
    "resolve_acceptance_floor_spec",
    "resolve_acceptance_workload",
]
