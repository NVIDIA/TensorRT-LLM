# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A speculator's acceptance: the workload an ``acceptance`` dataset sends, and its floor.

The tree's runner measures it (``checks/e2e.py``): iteration-weighted
acceptance length over the workload's streamed requests, held to 1.0, to the
workload being long enough to measure (``min_draft_iterations``), and to the
floor ``accuracy.acceptance`` declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..spec.task_schema import SPECULATORS, TaskSchemaError, load_prompts_file
from .harness import workload_io

#: Tokens each acceptance prompt may generate when the task declares none.
DEFAULT_ACCEPTANCE_MAX_TOKENS = 256
#: Generation iterations a workload must reach before its acceptance says anything.
DEFAULT_MIN_DRAFT_ITERATIONS = 32
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
    """What an ``acceptance`` dataset sends, and how many generation iterations it needs back."""

    #: Declared prompts; empty when ``dataset`` supplies them.
    prompts: tuple[str, ...]
    max_tokens: int
    min_draft_iterations: int
    #: Where the prompts came from, named in a verdict that blames them.
    source: str
    #: Where the prompts are drawn from, when no prompts are declared.
    dataset: Mapping[str, Any] | None = None

    def requests(self) -> list[dict[str, Any]]:
        """The declared prompts as engine-driver requests."""
        return [
            {"id": f"p{n}", "prompt": prompt, "max_tokens": self.max_tokens}
            for n, prompt in enumerate(self.prompts)
        ]


def acceptance_dataset(
    accuracy: Mapping[str, Any],
    unit: str | None,
    setting: Callable[[str], Any],
    *,
    benchmark: str = "gsm8k",
) -> tuple[dict[str, Any] | None, str]:
    """The dataset an undeclared acceptance workload is drawn from, and its name.

    ``accuracy.acceptance.dataset`` when declared; otherwise ``benchmark``'s own
    dataset -- ``accuracy.dataset_path`` when the run keeps it locally, else the
    benchmark's Hugging Face id, read from the local cache only
    (:data:`.harness.workload_io.KNOWN_DATASETS`).
    """
    known: dict[str, Any] = {}
    if isinstance(benchmark, str) and benchmark in workload_io.KNOWN_DATASETS:
        known = dict(workload_io.KNOWN_DATASETS[benchmark])
    elif "gsm8k" in workload_io.KNOWN_DATASETS:
        known = dict(workload_io.KNOWN_DATASETS["gsm8k"])
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
    accuracy: Mapping[str, Any], unit: str | None, *, benchmark: str = "gsm8k"
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
        dataset, source = acceptance_dataset(accuracy, unit, setting, benchmark=benchmark)
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


__all__ = [
    "AcceptanceFloor",
    "AcceptanceWorkload",
    "DEFAULT_ACCEPTANCE_MAX_TOKENS",
    "DEFAULT_ACCEPTANCE_NUM_PROMPTS",
    "DEFAULT_MIN_DRAFT_ITERATIONS",
    "MIN_ACCEPTANCE_RATIO",
    "acceptance_dataset",
    "resolve_acceptance_floor_spec",
    "resolve_acceptance_workload",
]
