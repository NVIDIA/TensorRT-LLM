# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative acceptance workloads and comparisons against stock, depth and prior trees."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import TaskSchemaError, load_prompts_file, resolve_per_unit
from . import stats
from .check_config import (
    check_configuration,
    engine_declaration,
    merge_options,
    resolve_llm_api_options,
    resolve_max_draft_len,
    speculation_flags,
)
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_OFF,
    MODE_REQUIRE,
    MODELING_V2_ENV,
    ShellRunner,
    default_shell_runner,
    engine_driver_command,
    gate_workdir,
    parse_driver_records,
    with_mode,
    write_driver_spec,
)
from .gate_records import KIND_ACCEPTANCE, KIND_STOCK_UNAVAILABLE, GateRecord, GateRecords
from .harness import workload_io
from .runtime_evidence import (
    EngineAttestation,
    attest_engine,
    engine_failure,
    engine_records,
    reports_unsupported_architecture,
)
from .verdict import GateResult, inspect_tree, tree_commit, verdict_factory

#: Tokens each acceptance prompt may generate when the task declares none.
DEFAULT_ACCEPTANCE_MAX_TOKENS = 256


#: Generation iterations the acceptance workload must reach, summed over its
#: prompts, before the gate reads a rate from it. A few short-answer prompts
#: that each end in EOS after 2-3 tokens reach only a handful of iterations,
#: too few for a rate to mean anything. A task may raise it through
#: ``accuracy.acceptance.min_draft_iterations``.
DEFAULT_MIN_DRAFT_ITERATIONS = 32


@dataclass(frozen=True)
class AcceptanceReading:
    """One leg's acceptance measurement, with the requests it came from."""

    #: The gate's number: iteration-weighted over the requests (:func:`parse_acceptance`).
    value: float
    #: ``(tokens, iterations)`` per request, in request order.
    requests: tuple[tuple[int, int], ...]
    #: A digest of each request's prompt, parallel to ``requests``: what pairs a
    #: prompt's rate with the same prompt's rate in another leg or on another tree.
    prompts: tuple[str, ...]

    def rates(self) -> dict[str, float]:
        """Accepted tokens per iteration, per prompt, keyed by prompt digest."""
        return {name: t / i for name, (t, i) in zip(self.prompts, self.requests) if i}

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
    return AcceptanceReading(
        value=sum(t for t, _ in parts) / iterations,
        requests=parts,
        prompts=tuple(str(r.get("prompt_sha", n)) for n, r in enumerate(records)),
    )


#: Prompts an acceptance workload must hold before its rates are compared: on
#: a handful, a change that only moves the generated text reads as a
#: regression. ``accuracy.acceptance.min_prompts`` may lower it.
DEFAULT_ACCEPTANCE_MIN_PROMPTS = 32


#: Prompts taken from the accuracy dataset when the task declares none.
DEFAULT_ACCEPTANCE_NUM_PROMPTS = 32


#: The minimum drop, in accepted tokens per iteration, a paired comparison
#: must find before it calls a significant change a regression.
DEFAULT_ACCEPTANCE_MIN_EFFECT = 0.1


#: The share of stock's acceptance a target must reach unless the paired test
#: finds its drop insignificant; a floor derived from a measured reference
#: keeps the same share of it.
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

    @property
    def fingerprint(self) -> str:
        """Names this workload, so two runs of it can be paired."""
        payload = json.dumps(
            {
                "prompts": list(self.prompts),
                "dataset": dict(self.dataset or {}),
                "max_tokens": self.max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

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


@dataclass(frozen=True)
class RegressionPolicy:
    """How a paired acceptance comparison is judged (``accuracy.acceptance.regression``)."""

    alpha: float = stats.DEFAULT_ALPHA
    min_effect: float = DEFAULT_ACCEPTANCE_MIN_EFFECT

    def compare(
        self, baseline: Sequence[float], candidate: Sequence[float]
    ) -> stats.PairedComparison:
        """:func:`.stats.paired_test` under this policy."""
        return stats.paired_test(baseline, candidate, alpha=self.alpha, min_effect=self.min_effect)


def resolve_regression_policy(accuracy: Mapping[str, Any]) -> RegressionPolicy:
    """Read ``accuracy.acceptance.regression``; every key optional."""
    block = accuracy.get("acceptance")
    block = (block if isinstance(block, dict) else {}).get("regression")
    block = block if isinstance(block, dict) else {}
    return RegressionPolicy(
        alpha=float(block.get("alpha", stats.DEFAULT_ALPHA)),
        min_effect=float(block.get("min_effect", DEFAULT_ACCEPTANCE_MIN_EFFECT)),
    )


def acceptance_key(
    workload: AcceptanceWorkload,
    *,
    member: str | None,
    speculation: Sequence[str],
) -> dict[str, Any]:
    """What makes two acceptance verdicts comparable across trees.

    The member, the workload and the speculator flags -- not the unit: a later
    unit that keeps speculation on is compared against the earlier unit's verdict.
    """
    return {
        "gate": "acceptance",
        "member": str(member),
        "workload": workload.fingerprint,
        "speculation": " ".join(speculation),
    }


#: Speculators whose draft depth is a setting of the run, so one depth shallower is a
#: configuration the same checkpoint serves (``--spec_decode_algo``, upper-cased).
VARIABLE_DEPTH_SPECULATORS: frozenset[str] = frozenset({"MTP", "EAGLE3"})


def depth_rise(
    deeper: AcceptanceReading, shallower: AcceptanceReading, policy: RegressionPolicy
) -> tuple[stats.PairedComparison | None, str]:
    """Whether ``deeper`` accepts significantly more than ``shallower``, paired per prompt.

    One-sided at ``policy.alpha`` with no minimum effect: a deep position is
    accepted rarely, and what is asked is that it is at all. ``None`` when
    fewer than two prompts were measured by both.
    """
    deep, shallow = deeper.rates(), shallower.rates()
    shared = [p for p in deep if p in shallow]
    if len(shared) < 2:
        return None, f"{len(shared)} prompt(s) measured by both, too few to compare"
    rise = stats.paired_test(
        [deep[p] for p in shared], [shallow[p] for p in shared], alpha=policy.alpha
    )
    se = "n/a" if math.isnan(rise.se_diff) else f"{rise.se_diff:.3f}"
    return rise, (
        f"mean per-prompt rise {-rise.mean_diff:+.3f} (SE {se}, one-sided p {rise.p_value:.3g} "
        f"against alpha {rise.alpha}, {rise.down} prompt(s) up, {rise.up} down over {rise.n})"
    )


def run_acceptance_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    features: Any | None = None,
    member: str | None = None,
    records: GateRecords | None = None,
) -> GateResult:
    """Measure per-prompt acceptance, and judge it against stock, a floor and the last tree.

    A workload under ``min_prompts`` prompts or ``min_draft_iterations``
    generation iterations fails as too small to measure, not as a low number.
    Then, each where it can be made: against stock (``TRTLLM_MODELING_V2=off``),
    failing when the ratio is under :data:`MIN_ACCEPTANCE_RATIO` and the paired
    test finds the drop significant; where the stock path cannot load the
    checkpoint (recorded in ``records``), against one draft depth shallower
    (:func:`depth_rise`) and a declared floor; against 1.0; and against the
    previous passing verdict in ``records`` (:func:`acceptance_key`).
    """
    tree = inspect_tree(task_data)
    _verdict = partial(verdict_factory(tree), name="acceptance", member=member)
    accuracy = task_data.get("accuracy") or {}
    try:
        workload = resolve_acceptance_workload(accuracy, unit)
    except TaskSchemaError as exc:
        return _verdict(
            passed=False,
            detail=f"the declared acceptance workload cannot be read: {exc}",
        )
    features = task_data.get("features") if features is None else features
    draft_len = resolve_max_draft_len(accuracy, unit)
    speculation = speculation_flags(features, max_draft_len=draft_len)
    config = check_configuration(
        task_data,
        environment,
        unit=unit,
        features=features,
        max_tokens=workload.max_tokens,
        member=member,
    )
    directory = gate_workdir(workdir)
    spec_path = write_driver_spec(
        directory / f"acceptance.{unit or 'unit'}.{member or 'member'}.json",
        config=config,
        requests=workload.requests(),
        dataset=workload.dataset,
        max_tokens=workload.max_tokens,
        reuse_probe=True,
    )
    policy = resolve_regression_policy(accuracy)
    floor_spec = resolve_acceptance_floor_spec(accuracy, unit)
    # One draft depth shallower is a leg of its own only where the depth is a setting:
    # a DSpark drafter's block is the checkpoint's (`resolve_max_draft_len`).
    algorithm = str((config["llm"].get("speculative_config") or {}).get("decoding_type", ""))
    depth_curve = draft_len > 1 and algorithm.upper() in VARIABLE_DEPTH_SPECULATORS

    options = resolve_llm_api_options(accuracy, unit)
    started = time.monotonic()
    readings: dict[str, AcceptanceReading | None] = {}
    commands: dict[str, str] = {}
    attested: list[EngineAttestation] = []
    stock_unavailable = False
    stock_key = {
        "checkpoint": str(task_data.get("checkpoint_path") or ""),
        "arch": str(task_data.get("arch") or ""),
    }
    known = (
        records.latest(KIND_STOCK_UNAVAILABLE, stock_key, passed_only=False)
        if records is not None
        else None
    )
    skipped_stock = ""
    for mode in (MODE_REQUIRE, MODE_OFF):
        if mode == MODE_OFF and known is not None and (floor_spec is not None or depth_curve):
            # Asked once per checkpoint and architecture: the stock path does not load it.
            readings[mode] = None
            stock_unavailable = True
            skipped_stock = (
                f" (not run again: on tree {known.tree} the stock leg could not load this "
                f"checkpoint)"
            )
            break
        command = engine_driver_command(with_mode(environment, mode), spec_path)
        commands[mode] = command
        result = runner(command, None, timeout)
        combined = result.stdout + result.stderr
        if result.infrastructure:
            return _verdict(
                passed=False,
                detail=(
                    f"the {MODELING_V2_ENV}={mode} leg did not run to a verdict: "
                    f"{result.infrastructure}"
                ),
                infrastructure=result.infrastructure,
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
            )
        if not result.ok:
            # A genuinely new architecture has no built-in modeling: the stock
            # leg is impossible rather than broken.
            unavailable = mode == MODE_OFF and reports_unsupported_architecture(
                f"exited {result.returncode}", "\n".join(combined.splitlines()[-200:])
            )
            if unavailable and records is not None:
                records.append(
                    GateRecord(
                        kind=KIND_STOCK_UNAVAILABLE,
                        key=stock_key,
                        tree=tree,
                        passed=False,
                        values={"tail": "\n".join(combined.splitlines()[-5:])[:500]},
                    )
                )
            if unavailable and (floor_spec is not None or depth_curve):
                readings[mode] = None
                stock_unavailable = True
                break
            return _verdict(
                passed=False,
                detail=(
                    f"the {MODELING_V2_ENV}={mode} leg "
                    + ("timed out" if result.timed_out else f"exited {result.returncode}")
                    + (
                        " -- this checkpoint has no built-in implementation, so the "
                        "stock leg cannot run, and there is no shallower depth to compare "
                        "with (draft depth 1, or a drafter whose depth is the checkpoint's). "
                        "Declare `accuracy.acceptance.<unit>."
                        "min_absolute` (or `measured_reference`) to gate on an absolute "
                        "floor instead."
                        if unavailable
                        else ""
                    )
                ),
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
            )
        declared = engine_declaration(config, mode=mode)
        attested.append(attest_engine(combined, declared, leg=mode))
        failure = engine_failure(
            combined,
            attested[-1],
            options=options,
            asks=bool(declared.get("block_reuse")),
            prefix=f"the {MODELING_V2_ENV}={mode} leg: ",
        )
        if failure:
            return _verdict(
                passed=False,
                detail=failure,
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(combined.splitlines()[-40:]),
                engine=engine_records(attested),
            )
        readings[mode] = parse_acceptance(combined)
    shallower = draft_len - 1 if stock_unavailable and depth_curve else None
    if shallower is not None:
        leg = f"depth {shallower}"
        shallow_config = merge_options(
            config, {"llm": {"speculative_config": {"max_draft_len": shallower}}}
        )
        shallow_spec = write_driver_spec(
            directory / f"acceptance.{unit or 'unit'}.{member or 'member'}.depth{shallower}.json",
            config=shallow_config,
            requests=workload.requests(),
            dataset=workload.dataset,
            max_tokens=workload.max_tokens,
        )
        command = engine_driver_command(
            with_mode(environment, MODE_REQUIRE),
            shallow_spec,
        )
        commands[leg] = command
        result = runner(command, None, timeout)
        combined = result.stdout + result.stderr
        ran = dict(
            command=command,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(combined.splitlines()[-40:]),
        )
        if result.infrastructure:
            return _verdict(
                passed=False,
                detail=f"the {leg} leg did not run to a verdict: {result.infrastructure}",
                infrastructure=result.infrastructure,
                **ran,
            )
        if not result.ok:
            return _verdict(
                passed=False,
                detail=f"the {leg} leg (the target one draft depth shallower, which this "
                f"depth's acceptance is compared with where no stock speculator exists) "
                + ("timed out" if result.timed_out else f"exited {result.returncode}"),
                **ran,
            )
        declared = engine_declaration(shallow_config, mode=MODE_REQUIRE)
        attested.append(attest_engine(combined, declared, leg=leg))
        if attested[-1].mismatches:
            return _verdict(
                passed=False,
                detail=attested[-1].difference(),
                engine=engine_records(attested),
                **ran,
            )
        readings[leg] = parse_acceptance(combined)
    elapsed = time.monotonic() - started

    target_reading = readings.get(MODE_REQUIRE)
    stock_reading = readings.get(MODE_OFF)
    target = target_reading.value if target_reading is not None else None
    stock = stock_reading.value if stock_reading is not None else None
    breakdown = target_reading.breakdown() if target_reading is not None else ""
    base = dict(duration_s=elapsed, engine=engine_records(attested))

    def too_small(mode: str) -> GateResult | None:
        """The verdict for a leg whose workload cannot carry a comparison."""
        reading = readings.get(mode)
        if reading is None:
            return None
        counted = reading.draft_iterations
        if len(reading.requests) < workload.min_prompts:
            why = (
                f"workload too small to compare acceptance: the {MODELING_V2_ENV}={mode} leg "
                f"measured {len(reading.requests)} prompt(s), fewer than the "
                f"{workload.min_prompts} a paired comparison needs; on a handful of prompts "
                f"a change that only moves the generated text reads as a regression"
            )
        elif counted < workload.min_draft_iterations:
            why = (
                f"workload too short to measure acceptance: the {MODELING_V2_ENV}={mode} "
                f"leg ran {counted} generation iteration(s) over {len(reading.requests)} "
                f"prompt(s), fewer than the {workload.min_draft_iterations} this gate "
                f"needs, so its rate ({reading.value:.2f}) says nothing about the draft "
                f"path -- the answers end before drafts are verified"
            )
        else:
            return None
        return _verdict(
            passed=False,
            detail=(
                f"{why}. Workload: {workload.source}, max_tokens {workload.max_tokens}. "
                f"This is a property of the workload, not of the target."
            ),
            command=commands[mode],
            breakdown=reading.breakdown(),
            **base,
        )

    if target is None:
        return _verdict(
            passed=False,
            detail=f"could not read acceptance from the require leg (off={stock})",
            command=commands[MODE_REQUIRE],
            **base,
        )
    for mode in (MODE_REQUIRE, MODE_OFF):
        short = too_small(mode)
        if short is not None:
            return short

    warnings: list[str] = []
    notes: list[str] = []
    if target <= 1.0:
        return _verdict(
            passed=False,
            detail=f"acceptance_length {target:.4f} at or below 1.0 means every draft was "
            f"rejected — the draft path is computing the wrong thing",
            command=commands[MODE_REQUIRE],
            measured=target,
            reference=stock if stock is not None else (floor_spec.value if floor_spec else None),
            breakdown=breakdown,
            **base,
        )

    reference: float | None
    shallow_reading = readings.get(f"depth {shallower}") if shallower is not None else None
    if shallower is not None:
        if shallow_reading is None:
            return _verdict(
                passed=False,
                detail=f"could not read per-prompt acceptance from the depth {shallower} leg",
                command=commands[f"depth {shallower}"],
                **base,
            )
        reference = shallow_reading.value
        rise, rise_note = depth_rise(target_reading, shallow_reading, policy)
        passed = rise is not None and rise.significant
        notes.append(
            f"acceptance_length {target:.4f} at draft depth {draft_len} vs "
            f"{shallow_reading.value:.4f} at depth {shallower}, the same target on the same "
            f"prompts ({rise_note}): "
            + (
                f"depth {draft_len}'s last draft position is accepted"
                if passed
                else f"depth {draft_len} accepts no more than depth {shallower}: its last draft "
                f"position is never accepted, or too rarely for this workload to show (more "
                f"prompts: accuracy.acceptance.num_prompts)"
            )
            + f". The {MODELING_V2_ENV}=off leg could not run: this checkpoint has no built-in "
            f"implementation{skipped_stock}"
        )
        if floor_spec is not None:
            floor_passed = target >= floor_spec.value
            passed = passed and floor_passed
            notes.append(
                f"declared floor {floor_spec.value:.4f}: "
                + ("met" if floor_passed else f"NOT met by {target:.4f}")
            )
            if floor_spec.warning():
                warnings.append(floor_spec.warning())
    elif stock_unavailable:
        assert floor_spec is not None
        reference = floor_spec.value
        passed = target >= floor_spec.value
        if floor_spec.warning():
            warnings.append(floor_spec.warning())
        derived = (
            f" (= measured reference {floor_spec.measured:.2f} x {MIN_ACCEPTANCE_RATIO:.2f})"
            if floor_spec.measured is not None and floor_spec.source is None
            else ""
        )
        notes.append(
            f"acceptance_length {target:.4f} vs declared floor {floor_spec.value:.4f}{derived}"
            + (f" (from {floor_spec.source})" if floor_spec.source else "")
            + f". The {MODELING_V2_ENV}=off leg could not run: this checkpoint has no "
            f"built-in implementation{skipped_stock}. THIS IS THE WEAKER CLAIM -- it shows the "
            f"draft path accepts at a real rate, not that it matches stock"
        )
    else:
        if stock is None:
            return _verdict(
                passed=False,
                detail=f"could not read acceptance from both legs (require={target}, off={stock})",
                command=commands[MODE_REQUIRE],
                **base,
            )
        stock_breakdown = stock_reading.breakdown() if stock_reading is not None else ""
        if breakdown and stock_breakdown:
            breakdown = f"require: {breakdown}; off: {stock_breakdown}"
        reference = stock
        ratio = target / stock if stock else 0.0
        paired = None
        own, theirs = target_reading.rates(), stock_reading.rates()
        shared = [p for p in own if p in theirs]
        if len(shared) >= 2:
            paired = policy.compare([theirs[p] for p in shared], [own[p] for p in shared])
        passed = ratio >= MIN_ACCEPTANCE_RATIO or (paired is not None and not paired.significant)
        notes.append(
            f"acceptance_length {target:.4f} vs stock {stock:.4f} (ratio {ratio:.3f}, floor "
            f"{MIN_ACCEPTANCE_RATIO:.2f})"
            + (f"; {paired.describe()}" if paired is not None else "")
            + (
                ". Clearly above 1.0 but below the reference means subtly wrong, not merely slower"
                if not passed
                else ""
            )
        )

    key = acceptance_key(workload, member=member, speculation=speculation)
    if records is not None:
        baseline = records.latest(KIND_ACCEPTANCE, key, exclude_tree=tree)
        if baseline is None:
            notes.append(
                "no earlier verdict for this member and workload; this one is the baseline"
            )
        else:
            before = {
                str(k): float(v) for k, v in (baseline.values.get("per_prompt") or {}).items()
            }
            now = target_reading.rates()
            shared = [p for p in now if p in before]
            if len(shared) < 2:
                warnings.append(
                    f"the earlier verdict on tree {baseline.tree} shares {len(shared)} prompt(s) "
                    f"with this one, so the two were not compared"
                )
            else:
                compared = policy.compare([before[p] for p in shared], [now[p] for p in shared])
                regressed = compared.regressed
                notes.append(
                    f"against tree {baseline.tree}: "
                    + ("REGRESSED, " if regressed else "")
                    + compared.describe()
                )
                passed = passed and not regressed
        records.append(
            GateRecord(
                kind=KIND_ACCEPTANCE,
                key=key,
                tree=tree,
                commit=tree_commit(task_data.get("trtllm_repo_path")),
                passed=passed,
                values={
                    "unit": unit,
                    "weighted": target,
                    "per_prompt": target_reading.rates(),
                    "prompts": len(target_reading.requests),
                },
            )
        )

    return _verdict(
        passed=passed,
        detail="; ".join(notes),
        command=commands[MODE_REQUIRE],
        measured=target,
        reference=reference,
        breakdown=breakdown,
        warnings=tuple(warnings),
        explained=True,
        **base,
    )


__all__ = [
    "AcceptanceFloor",
    "AcceptanceReading",
    "AcceptanceWorkload",
    "DEFAULT_ACCEPTANCE_MAX_TOKENS",
    "DEFAULT_ACCEPTANCE_MIN_EFFECT",
    "DEFAULT_ACCEPTANCE_MIN_PROMPTS",
    "DEFAULT_ACCEPTANCE_NUM_PROMPTS",
    "DEFAULT_MIN_DRAFT_ITERATIONS",
    "MIN_ACCEPTANCE_RATIO",
    "RegressionPolicy",
    "VARIABLE_DEPTH_SPECULATORS",
    "acceptance_dataset",
    "acceptance_key",
    "depth_rise",
    "parse_acceptance",
    "resolve_acceptance_floor_spec",
    "resolve_acceptance_workload",
    "resolve_regression_policy",
    "run_acceptance_gate",
]
