# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract: every check a run must pass, compiled from the resolved task spec.

The contract is the run's definition of "done". Its checks are the run's tests
(``spec/verification.py``): the ones the flow derives from the spec
(:func:`flow_tests`), as the spec's ``verification`` changes them, and the
spec's own. Each test at each of its members is one check, keyed
``<feature>:<test>@<member>``, run at the test's stage:

- ``landing``: when a landing claims it;
- ``delivery``: once every other required check has passed. Its pass stands
  like any other, and it runs again when a task that claims it lands.

A check runs when the task that claims it lands, a delivery check with its stage. Verdicts are appended to
``contract_status.jsonl`` as gates return, keyed by unit. Delivery evaluates
the contract (:func:`evaluate`): every required check's latest verdict must
have passed. A check that passed stands, whatever landed after it. A verdict
that failed counts on the state of the target it measured (its ``marker``):
once the target has changed the check is missing, and runs again. A verdict
recorded with other ``inputs`` (:func:`check_inputs`) is dropped. A check
passes by its own verdict only. Before delivery one check has passed on the
final tree (:func:`confirmed`).

The first state of the target on which the first unit's bring-up checks hold
(:func:`bringup_checks`) is where the bring-up was delivered. The first on
which every required check has passed and the tree is confirmed is where
correctness was delivered. Both are appended to ``correctness.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from ...checks import check_config, serve
from ...checks.accuracy import resolve_discriminators
from ...checks.check_config import (
    load_llm_api_options,
    order_members_by_cost,
    resolve_llm_api_options,
    resolve_runtime,
)
from ...infra.files import write_text_atomic
from ...spec import variants as variants_mod
from ...spec import verification as verification_mod
from ...spec.task_schema import (
    DISAGG_FEATURE,
    PROFILE_BRINGUP,
    PROFILE_CERTIFIED,
    PROFILE_SMOKE,
    SPECULATORS,
    TaskSchemaError,
    assembly_units,
    members_for,
    parse_parallel_segment,
    profile_of,
    resolve_per_unit,
)
from ...spec.verification import (
    BOOT,
    DISAGG,
    E2E,
    ORIGIN_FLOW,
    RUN_DATASETS,
    RUN_DISAGG,
    RUN_SERVE,
    SERVE,
    SERVE_ONE_RANK,
    STAGE_DELIVERY,
    STAGE_LANDING,
)
from ...spec.verification import DATASET_ACCEPTANCE as ACCEPTANCE
from ...spec.verification import DATASET_GENERATE as GENERATE

CONTRACT_FILENAME = "contract.yaml"
STATUS_FILENAME = "contract_status.jsonl"
#: One line per milestone: each state of the target the contract first held on,
#: and the bring-up; the last of each tier is where it stands.
MILESTONE_FILENAME = "correctness.jsonl"
#: A milestone's tier: the first unit's checks at its narrowest accuracy member
#: (its benchmark, and its boot where it owes one), or every required check.
MILESTONE_BRINGUP = "bringup"
MILESTONE_CORRECTNESS = "correctness"
#: The benchmark of a unit whose spec names none.
DEFAULT_BENCHMARK = "gsm8k"
#: Seconds one of the flow's tests may take at one member, unless ``accuracy.timeout`` says.
E2E_TIMEOUT_S = 4 * 60 * 60

#: What each of the flow's tests proves, as the contract and the report say it.
_PURPOSE: dict[str, str] = {
    E2E: "the engine starts, answers the boot prompts and scores its benchmark (and accepts "
    "its drafts, on a speculator) on one load",
    BOOT: "the engine starts and answers the boot prompts (and accepts its drafts, on a "
    "speculator)",
    SERVE: "trtllm-serve starts and answers the boot prompts",
    SERVE_ONE_RANK: "trtllm-serve on one rank starts and answers the boot prompts",
    DISAGG: "the tree's disaggregated-serving test passes (accuracy.disagg)",
}


@dataclass(frozen=True)
class Check:
    """One check: a test (``gate`` is its name), at one member, for one feature, run at one stage."""

    unit: str
    gate: str
    member: str
    stage: str
    required: bool = True
    #: What the check reads from outside the checkout, as a digest; see :func:`check_inputs`.
    inputs: str = field(default="", compare=False)
    #: The check whose verdict on the same target this one needs first: the unit's
    #: benchmark at the same member, for a variant whose anchor is relative.
    after: str = field(default="", compare=False)
    #: How its test runs (``verification.RUN_*``).
    runner: str = field(default=RUN_DATASETS, compare=False)
    #: Whether it judges the engine's answers by a score: a benchmark, or the disaggregated test.
    scores: bool = field(default=False, compare=False)
    #: Who declared its test (``verification.ORIGIN_*``).
    origin: str = field(default=ORIGIN_FLOW, compare=False)
    #: The check whose pass on the same target answers this one: it measures this
    #: one's datasets, and more, on the same engine with the same settings
    #: (:func:`answering`). Empty for one that always runs itself.
    answered_by: str = field(default="", compare=False)
    #: The speculator's ``e2e`` at the same member that stands in for this ``e2e``
    #: at a landing that names both (:func:`screening`); this one then runs once the
    #: work is done. Empty for one a landing runs whenever it names it.
    screened_by: str = field(default="", compare=False)

    @property
    def key(self) -> str:
        return f"{self.unit}:{self.gate}@{self.member}"


@dataclass(frozen=True)
class Contract:
    checks: tuple[Check, ...]
    #: Why a unit or variant owes less than every check at every member, one line each.
    notes: tuple[str, ...] = ()
    #: The key of the check that confirms the final tree when none of its kind
    #: has passed on it (:func:`confirmed`): the benchmark of the last unit that
    #: owes one, at its cheapest member, else the last variant's that does; the
    #: last boot where the run owes no benchmark at all.
    confirm: str = ""

    def required(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.required)

    def by_stage(self, stage: str) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.stage == stage)

    def as_dict(self) -> dict[str, Any]:
        sparse = ("inputs", "after", "scores", "answered_by", "screened_by")
        out: dict[str, Any] = {
            "checks": [
                {k: v for k, v in asdict(c).items() if k not in sparse or v} for c in self.checks
            ]
        }
        if self.notes:
            out["notes"] = list(self.notes)
        if self.confirm:
            out["confirm"] = self.confirm
        return out

    def render(self) -> str:
        """A markdown table, one row per check."""
        lines = [
            "| feature | test | member | stage | from | required |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.checks:
            lines.append(
                f"| {c.unit} | {c.gate} | {c.member} | {c.stage} | {c.origin} | "
                f"{'yes' if c.required else 'no'} |"
            )
        if self.notes:
            lines += ["", *(f"- {note}" for note in self.notes)]
        return "\n".join(lines)


@dataclass(frozen=True)
class LandingBudget:
    """Advisory breakdown of ``landing`` vs ``delivery`` checks and reference run time."""

    landing_checks: int
    delivery_checks: int
    #: Sum of estimated pure run time (minutes, excluding model load) across required
    #: ``landing`` checks that run on their own (not answered by another check).
    estimated_minutes: float
    #: Required ``landing`` checks with no catalog timing estimate (custom ``pytest`` / disagg test).
    unestimated_checks: int = 0
    #: Non-blocking hints for keeping ``landing`` lean (heavy datasets, thinking mode, over budget).
    warnings: tuple[str, ...] = ()

    def summary_line(self) -> str:
        """One-line stage breakdown with the reference ``landing`` run-time estimate."""
        details: list[str] = []
        if self.estimated_minutes > 0:
            details.append(
                f"~{round(self.estimated_minutes)} min est. run time on ~1T-class MoE ref, "
                "excl. model load"
            )
        if self.unestimated_checks:
            prefix = "+ " if details else ""
            details.append(f"{prefix}{self.unestimated_checks} unestimated")
        landing = f"{self.landing_checks} landing"
        if details:
            landing += f" ({'; '.join(details)})"
        return f"stages: {landing}, {self.delivery_checks} delivery"

    def render(self) -> str:
        """The summary line followed by any advisory warnings."""
        lines = [self.summary_line()]
        for warning in self.warnings:
            lines.append(f"- warning: {warning}")
        return "\n".join(lines)


def _effective_evaluator_kwargs(
    spec: Mapping[str, Any], test: verification_mod.Test
) -> dict[str, Any]:
    """``accuracy.evaluator_kwargs`` with ``test.evaluator_kwargs`` laid over it."""
    merged = dict((spec.get("accuracy") or {}).get("evaluator_kwargs") or {})
    merged.update(test.evaluator_kwargs or {})
    return merged


def landing_budget(
    task_data: Mapping[str, Any], contract: Contract, world_size: int
) -> LandingBudget:
    """Summarize ``landing`` vs ``delivery`` checks and warn when ``landing`` looks heavy.

    Estimates use :data:`verification.DATASET_REF_MINUTES` (pure run time excluding model
    load on a ~1T-class MoE reference) and skip checks answered by another check.
    Advisory only: never rejects a spec or blocks launch.
    """
    tests = {(t.feature, t.name): t for t in run_tests(task_data, world_size)}
    specs: dict[str, Mapping[str, Any]] = {}
    required = contract.required()
    landing = [c for c in required if c.stage == STAGE_LANDING]
    delivery = [c for c in required if c.stage == STAGE_DELIVERY]

    estimated = 0.0
    unestimated = 0
    heavy_hits: list[str] = []
    thinking_hits: list[str] = []

    for check in landing:
        if check.answered_by:
            continue
        test = tests.get((check.unit, check.gate))
        if test is None:
            unestimated += 1
            continue
        if variants_mod.is_variant(task_data, check.unit) and check.unit not in specs:
            specs[check.unit] = check_config.variant_spec(
                task_data, check.unit, world_size=world_size
            )
        spec = specs.get(check.unit, task_data)
        kwargs = _effective_evaluator_kwargs(spec, test)

        if test.runner == RUN_DISAGG:
            accuracy = spec.get("accuracy") or {}
            disagg_block = accuracy.get("disagg") or {}
            if isinstance(disagg_block, Mapping) and disagg_block.get("test"):
                unestimated += 1
                continue
            benchmark = str(
                resolve_per_unit(accuracy, "benchmark", check.unit) or DEFAULT_BENCHMARK
            )
            minutes = verification_mod.DATASET_REF_MINUTES.get(benchmark)
            if minutes is None:
                unestimated += 1
                continue
            if benchmark in verification_mod.HEAVY_DATASETS:
                heavy_hits.append(f"{check.key} ({benchmark})")
            if (
                verification_mod.thinking_enabled(kwargs)
                and benchmark not in verification_mod.LONG_OUTPUT_BENCHMARKS
            ):
                minutes *= verification_mod.THINKING_MULTIPLIER
                thinking_hits.append(check.key)
            estimated += minutes
            continue

        minutes = verification_mod.estimate_run_minutes(test, evaluator_kwargs=kwargs)
        if minutes is None:
            unestimated += 1
            continue
        estimated += minutes

        heavy = [d for d in test.datasets if d in verification_mod.HEAVY_DATASETS]
        if heavy:
            heavy_hits.append(f"{check.key} ({', '.join(heavy)})")
        if verification_mod.thinking_enabled(kwargs) and any(
            b not in verification_mod.LONG_OUTPUT_BENCHMARKS for b in test.benchmarks
        ):
            thinking_hits.append(check.key)

    warnings: list[str] = []
    if heavy_hits:
        warnings.append(
            "heavy benchmark(s) at landing: "
            + ", ".join(heavy_hits)
            + " — consider setting stage: delivery"
        )
    if thinking_hits:
        warnings.append(
            "enable_thinking: true on landing benchmark(s): "
            + ", ".join(thinking_hits)
            + " (5-20x longer decode) — consider disabling thinking at landing or setting "
            "stage: delivery"
        )
    if estimated > verification_mod.LANDING_BUDGET_MINUTES:
        warnings.append(
            f"landing checks estimate ~{round(estimated)} min of pure run time on the ~1T-class "
            f"MoE reference (excluding model load), past the "
            f"{verification_mod.LANDING_BUDGET_MINUTES} min landing budget — consider moving "
            "heavier or multi-member checks to stage: delivery"
        )
    return LandingBudget(
        landing_checks=len(landing),
        delivery_checks=len(delivery),
        estimated_minutes=round(estimated, 1),
        unestimated_checks=unestimated,
        warnings=tuple(warnings),
    )


def _per_unit(key: str) -> Callable[[Mapping[str, Any], str], Any]:
    return lambda accuracy, unit: resolve_per_unit(accuracy, key, unit)


def _options(accuracy: Mapping[str, Any], unit: str) -> Any:
    """The unit's LLM-API options as their content; the path when the file cannot be read."""
    path = resolve_llm_api_options(accuracy, unit)
    if not path:
        return None
    if not Path(path).is_file():
        return {"missing": str(path)}
    try:
        return load_llm_api_options(path)
    except (OSError, ValueError, yaml.YAMLError):
        return {"unreadable": str(path)}


#: The accuracy-block keys a check reads through a resolver, and
#: that resolver.
_ACCURACY_RESOLVERS: dict[str, Callable[[Mapping[str, Any], str], Any]] = {
    "llm_api_options": _options,
    "discriminators": resolve_discriminators,
    **{key: _per_unit(key) for key in ("benchmark", "reference_yaml", "model_key")},
}


#: File digests by ``(path, size, mtime)``: an input is read once per change.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def _with_contents(value: Any, checkout: Path | None) -> Any:
    """``value`` with every path of a file outside ``checkout`` paired with its content's digest.

    A file in the checkout is part of the marker, its anchor tables by the
    entries the gates read; repeating it here would make an edit to another
    model's entry stale a verdict the marker keeps.
    """
    if isinstance(value, Mapping):
        return {str(k): _with_contents(v, checkout) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_with_contents(v, checkout) for v in value]
    if not isinstance(value, str) or not value or "\n" in value or len(value) > 4096:
        return value
    path = Path(value)
    try:
        if not path.is_absolute() or not path.is_file():
            return value
        resolved = path.resolve()
        if checkout is not None and resolved.is_relative_to(checkout):
            return value
        return {"path": value, "sha1": _file_digest(resolved)}
    except (OSError, ValueError):
        return value


def _for_unit(value: Any, unit: str, others: set[str]) -> list[Any]:
    """``value`` as it applies to ``unit``: its own entry, else the value less other units' entries."""
    if isinstance(value, Mapping) and unit in value:
        return ["unit", value[unit]]
    if isinstance(value, Mapping):
        return ["run", {k: v for k, v in value.items() if k not in others}]
    return ["run", value]


def check_inputs(
    task_data: Mapping[str, Any],
    test: verification_mod.Test,
    member: str,
    features: Sequence[str],
) -> str:
    """What ``test``'s check at ``member`` reads from outside the checkout, as one digest.

    The test as the run declares it (less its purpose and who declared it),
    and every file outside the checkout the spec names, by content.
    Conservative: a change anywhere in what applies to its feature makes its
    checks stale; one another feature's entry alone does not.
    """
    unit = test.feature
    accuracy = task_data.get("accuracy") or {}
    repo = str(task_data.get("trtllm_repo_path") or "")
    try:
        checkout = Path(repo).resolve() if repo else None
    except OSError:
        checkout = None
    others = (set(assembly_units(task_data)) | set(variants_mod.variants(task_data))) - {unit}
    entries = {
        key: (
            _ACCURACY_RESOLVERS[key](accuracy, unit)
            if key in _ACCURACY_RESOLVERS
            else _for_unit(accuracy[key], unit, others)
        )
        for key in sorted(accuracy)
    }
    # A block holding other units' entries alone says nothing to this one.
    entries = {k: v for k, v in entries.items() if v != ["run", {}]}
    environment = task_data.get("environment") or {}
    # A setting the test leaves unset is not part of what it reads, so specs that set
    # none of them keep their digests.
    declared = {
        k: v
        for k, v in asdict(test).items()
        if k not in ("purpose", "origin")
        and not (k in ("config", "evaluator_kwargs", "extra_checks") and not v)
    }
    payload = {
        "test": declared,
        "member": member,
        "features": sorted(str(f) for f in features),
        "checkpoint": task_data.get("checkpoint_path"),
        "parallel": task_data.get("parallel"),
        "runtime": resolve_runtime(task_data, unit),
        "accuracy": entries,
        "env": environment.get("env") if isinstance(environment, Mapping) else None,
        "variant": variants_mod.variants(task_data).get(unit),
    }
    text = json.dumps(_with_contents(payload, checkout), sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _width(member: str) -> int:
    """How many ranks ``member`` serves on; 0 when it does not say."""
    try:
        return parse_parallel_segment(member)[1] or 0
    except TaskSchemaError:
        return 0


def benchmark_members(task_data: Mapping[str, Any], unit: str, members: Sequence[str]) -> list[str]:
    """Where ``unit``'s accuracy benchmark runs: every member, or the one ``benchmark_members`` names."""
    named = (task_data.get("accuracy") or {}).get("benchmark_members")
    return [str(named)] if named in members else list(members)


def benchmark_member(task_data: Mapping[str, Any], members: Sequence[str]) -> tuple[str, str]:
    """The one member of ``members`` a configuration variant's benchmark runs at, and why.

    The member ``accuracy.benchmark_members`` names; else an attention-DP member,
    whose served engine takes the most paths (the router, dummy requests, per-rank
    caches); else the widest, the cheapest to start among equals.
    """
    named = (task_data.get("accuracy") or {}).get("benchmark_members")
    if named in members:
        return str(named), "named in accuracy.benchmark_members"
    ranked = order_members_by_cost(list(members))
    for member in ranked:
        if parse_parallel_segment(member)[0] == "dep":
            return member, "attention DP serves through the most paths"
    return max(ranked, key=_width), "the widest"


def test_members(
    task_data: Mapping[str, Any], test: verification_mod.Test, world_size: int
) -> list[str]:
    """Where ``test`` runs, one check each.

    Its members; every member of its feature for ``all``; the feature's cheapest
    for a test of datasets that names none. A module test (pytest with
    ``devices``) runs once, recorded at ``tp<N>`` for its devices.
    """
    if not test.whole_model:
        return [verification_mod.module_member(test.devices or 1)]
    if test.members and list(test.members) != [verification_mod.ALL_MEMBERS]:
        return list(test.members)
    if variants_mod.is_variant(task_data, test.feature):
        spec = check_config.variant_spec(task_data, test.feature, world_size=world_size)
        everywhere = order_members_by_cost(members_for(spec, test.feature, world_size))
    else:
        everywhere = order_members_by_cost(members_for(task_data, test.feature, world_size))
    return everywhere if test.members else everywhere[:1]


def _test(
    name: str,
    feature: str,
    purpose: str,
    runner: str,
    members: Sequence[str],
    *,
    datasets: Sequence[str] = (),
    stage: str = STAGE_LANDING,
    timeout: int = verification_mod.DEFAULT_TIMEOUT_S,
) -> verification_mod.Test:
    return verification_mod.Test(
        name=name,
        feature=feature,
        purpose=purpose,
        runner=runner,
        datasets=tuple(datasets),
        members=tuple(members),
        stage=stage,
        timeout=timeout,
        origin=verification_mod.ORIGIN_FLOW,
    )


def flow_tests(
    task_data: Mapping[str, Any], world_size: int, notes: list[str] | None = None
) -> list[verification_mod.Test]:
    """The tests the flow derives from the spec, before ``verification`` changes them.

    Per unit, by its profile (``task_schema.profile_of``):

    - ``e2e``: one engine per member that boots, answers the boot prompts and
      scores the benchmark (``accuracy.benchmark``, at ``accuracy.benchmark_members``),
      and, on a target that implements a speculator, measures its acceptance;
      ``boot`` at the members the benchmark skips (with the acceptance there);
    - ``serve`` when the spec serves (``accuracy.serve``), at one member;
    - ``disagg`` for the ``disagg`` feature alone: its disaggregated-serving
      test, once (``checks/disagg.py``); its engines are aggregated ones the
      units before it prove.

    ``smoke`` owes ``boot`` alone, ``bringup`` its tests at the cheapest member,
    ``certified`` each variant's benchmark at every member. Each configuration
    variant runs its ``gates`` (``spec/variants.py``): ``e2e`` with the benchmark at
    one member (:func:`benchmark_member`) unless it names its members, ``boot``
    at its other members and, for a storage or scheduling variant, at every
    member of its unit; a variant of ``disagg`` owes nothing of its own. A
    ``smoke`` or ``bringup`` run owes no variant and no delivery test. Last,
    ``serve_one_rank`` at delivery: the last unit served on one rank.
    """
    notes = notes if notes is not None else []
    accuracy = task_data.get("accuracy") or {}
    serving = serve.ServeSpec.from_task(task_data).enabled
    units = assembly_units(task_data)
    lean = profile_of(task_data) in (PROFILE_SMOKE, PROFILE_BRINGUP)
    timeout = int(accuracy.get("timeout") or E2E_TIMEOUT_S)
    out: list[verification_mod.Test] = []

    def add(
        name: str, feature: str, purpose: str, runner: str, members: Sequence[str], **kw
    ) -> None:
        if members:
            out.append(_test(name, feature, purpose, runner, members, timeout=timeout, **kw))

    for index, unit in enumerate(units):
        level = profile_of(task_data, unit)
        members = order_members_by_cost(members_for(task_data, unit, world_size))
        if level == PROFILE_BRINGUP and len(members) > 1:
            notes.append(
                f"{unit}: the bringup profile gates it at {members[0]} only, not at "
                f"{', '.join(members[1:])}"
            )
            members = members[:1]
        implemented = set(units[1 : index + 1])
        if unit == DISAGG_FEATURE:
            member, why = benchmark_member(task_data, members)
            add(DISAGG, unit, _PURPOSE[DISAGG], RUN_DISAGG, [member])
            notes.append(
                f"{unit}: its disaggregated-serving test alone proves it, once at {member} "
                f"({why}); its boot and accuracy would run one aggregated engine"
            )
            continue
        if level == PROFILE_SMOKE:
            add(BOOT, unit, _PURPOSE[BOOT], RUN_DATASETS, members, datasets=[GENERATE])
            notes.append(f"{unit}: the smoke profile owes its boot alone")
            continue
        benchmark = str(resolve_per_unit(accuracy, "benchmark", unit) or DEFAULT_BENCHMARK)
        # Speculative decoding preserves the output distribution by construction, so a
        # draft path that computes the wrong thing passes boot and accuracy and shows
        # only as speed: per-prompt acceptance is the detector.
        drafts = [ACCEPTANCE] if SPECULATORS & implemented else []
        scored = benchmark_members(task_data, unit, members)
        add(E2E, unit, _PURPOSE[E2E], RUN_DATASETS, scored, datasets=[GENERATE, benchmark, *drafts])
        rest = [m for m in members if m not in scored]
        add(BOOT, unit, _PURPOSE[BOOT], RUN_DATASETS, rest, datasets=[GENERATE, *drafts])
        if serving:
            add(SERVE, unit, _PURPOSE[SERVE], RUN_SERVE, [serve.serve_member(members)],
                datasets=[GENERATE])  # fmt: skip
    for name in () if lean else variants_mod.variants(task_data):
        level = profile_of(task_data, name)
        if level == PROFILE_BRINGUP:
            notes.append(f"{name}: the bringup profile gates no configuration variant")
            continue
        base = variants_mod.base_unit(task_data, name)
        if base == DISAGG_FEATURE:
            notes.append(f"{name}: a variant of {base} owes no check; its axis is proven on core")
            continue
        spec = check_config.variant_spec(task_data, name, world_size=world_size)
        members = order_members_by_cost(members_for(spec, name, world_size))
        everywhere = order_members_by_cost(members_for(task_data, base, world_size))
        benchmark = str(
            resolve_per_unit(spec.get("accuracy") or {}, "benchmark", name) or DEFAULT_BENCHMARK
        )
        gates = ("boot",) if level == PROFILE_SMOKE else variants_mod.variant_gates(task_data, name)
        drafts = [ACCEPTANCE] if "acceptance" in gates else []
        if variants_mod.anchor_state(task_data, name) == variants_mod.ANCHOR_RELATIVE and (
            "accuracy" in gates
        ):
            notes.append(f"{name}: its anchor is {base}'s own score at the same member and target")
        scored = members if "accuracy" in gates else []
        named = variants_mod.variants(task_data)[name].get("members")
        if scored and not named and level != PROFILE_CERTIFIED and len(scored) > 1:
            member, why = benchmark_member(spec, members)
            scored = [member]
            notes.append(f"{name}: accuracy at {member} ({why}); its other members boot")
        if scored:
            add(E2E, name, _PURPOSE[E2E], RUN_DATASETS, scored,
                datasets=[GENERATE, benchmark, *drafts])  # fmt: skip
            rest = [m for m in members if m not in scored]
        else:
            add(E2E, name, _PURPOSE[E2E], RUN_DATASETS, members if drafts else [],
                datasets=[GENERATE, *drafts])  # fmt: skip
            rest = [] if drafts else list(members)
        if not drafts:
            rest += [m for m in everywhere if m not in members]
        if drafts or "boot" in gates:
            add(BOOT, name, _PURPOSE[BOOT], RUN_DATASETS, rest, datasets=[GENERATE, *drafts])
    if lean and variants_mod.variants(task_data):
        notes.append(
            f"the {profile_of(task_data)} profile gates no configuration variant: "
            + ", ".join(variants_mod.variants(task_data))
        )
    if units and serving and not lean:
        single = serve.delivery_member(members_for(task_data, units[-1], world_size))
        if single is not None:
            add(SERVE_ONE_RANK, units[-1], _PURPOSE[SERVE_ONE_RANK], RUN_SERVE, [single],
                datasets=[GENERATE], stage=STAGE_DELIVERY)  # fmt: skip
    return out


def run_tests(
    task_data: Mapping[str, Any], world_size: int, notes: list[str] | None = None
) -> list[verification_mod.Test]:
    """The run's tests: the flow's (:func:`flow_tests`) as ``verification`` changes them, then the spec's own."""
    try:
        return verification_mod.apply(flow_tests(task_data, world_size, notes), task_data)
    except ValueError as exc:
        raise TaskSchemaError(str(exc)) from exc


def _built(task_data: Mapping[str, Any], units: Sequence[str], feature: str) -> list[str]:
    """The features the target of ``feature`` implements (a variant's unit's)."""
    unit = (
        variants_mod.base_unit(task_data, feature)
        if variants_mod.is_variant(task_data, feature)
        else feature
    )
    return list(units[1 : units.index(unit) + 1]) if unit in units else []


def compile_contract(task_data: Mapping[str, Any], world_size: int) -> Contract:
    """Every check the run owes: each test (:func:`run_tests`) at each of its members, in order.

    Every check carries its :func:`check_inputs`. A configuration variant's
    benchmark whose anchor is relative runs after its unit's at the same
    member (:attr:`Check.after`).
    """
    notes: list[str] = []
    tests = run_tests(task_data, world_size, notes)
    units = assembly_units(task_data)
    checks: list[Check] = []
    specs: dict[str, Mapping[str, Any]] = {}
    for test in tests:
        feature = test.feature
        if variants_mod.is_variant(task_data, feature) and feature not in specs:
            specs[feature] = check_config.variant_spec(task_data, feature, world_size=world_size)
        spec = specs.get(feature, task_data)
        built = _built(task_data, units, feature)
        relative = bool(relative_benchmark(task_data, test, tests))
        base = variants_mod.base_unit(task_data, feature) if relative else ""
        members = test_members(task_data, test, world_size)
        for member in members:
            after = ""
            if relative:
                after = next(
                    (
                        c.key
                        for c in checks
                        if (c.unit, c.gate, c.member) == (base, verification_mod.E2E, member)
                    ),
                    "",
                )
            checks.append(
                Check(
                    feature,
                    test.name,
                    member,
                    test.stage,
                    inputs=check_inputs(spec, test, member, built),
                    after=after,
                    runner=test.runner,
                    scores=test.scores,
                    origin=test.origin,
                )
            )
        if test.origin != verification_mod.ORIGIN_FLOW:
            where = ", ".join(members) if test.whole_model else f"{test.devices} device(s), once"
            whose = (
                "the spec's own"
                if test.origin == verification_mod.ORIGIN_SPEC
                else ("the flow's, changed by the spec")
            )
            notes.append(
                f"{feature}: `{test.name}`, {whose}, at {test.stage} ({where})"
                + (f": {test.purpose}" if test.purpose else "")
            )
    by_name = {(t.feature, t.name): t for t in tests}
    checks = answering(task_data, checks, by_name, notes)
    checks = screening(task_data, checks, by_name, units, notes)
    return Contract(
        checks=tuple(checks),
        notes=tuple(notes),
        confirm=_confirm(checks, units, list(variants_mod.variants(task_data))),
    )


def relative_benchmark(
    task_data: Mapping[str, Any],
    test: verification_mod.Test,
    tests: Iterable[verification_mod.Test],
) -> str:
    """The benchmark of ``test`` judged against its unit's ``e2e`` score at the same member; ``""``.

    Only a configuration variant whose anchor is relative has one: its first
    benchmark, where it is the one its unit's ``e2e`` scores first, under the
    same protocol. Any other is judged against its references row.
    """
    feature = test.feature
    if not (
        test.benchmarks
        and variants_mod.is_variant(task_data, feature)
        and variants_mod.anchor_state(task_data, feature) == variants_mod.ANCHOR_RELATIVE
    ):
        return ""
    base = variants_mod.base_unit(task_data, feature)
    unit_e2e = next((t for t in tests if (t.feature, t.name) == (base, E2E)), None)
    if unit_e2e is None or not unit_e2e.benchmarks:
        return ""
    first = test.benchmarks[0]
    same_protocol = dict(test.evaluator_kwargs or {}) == dict(unit_e2e.evaluator_kwargs or {})
    return first if first == unit_e2e.benchmarks[0] and same_protocol else ""


def answering(
    task_data: Mapping[str, Any],
    checks: Sequence[Check],
    tests: Mapping[tuple[str, str], verification_mod.Test],
    notes: list[str],
) -> list[Check]:
    """``checks`` with :attr:`Check.answered_by` set: one measurement runs once.

    A check of datasets is answered by another of the same feature at the same
    member that measures every dataset it does, and may measure more, on an
    engine of the same settings (:attr:`~verification.Test.settings`), each
    benchmark judged against the same anchor, and scores first what it scores
    first (the score its verdict carries). Of two that measure the same, the
    first answers the second. One that answers another is answered by none, so
    an answer is never an answer's.
    """
    out = list(checks)
    answers: set[str] = set()
    every = list(tests.values())
    for index, check in enumerate(out):
        test = tests.get((check.unit, check.gate))
        if check.runner != RUN_DATASETS or test is None or check.key in answers:
            continue
        anchored = relative_benchmark(task_data, test, every)
        for other_index, other in enumerate(out):
            them = tests.get((other.unit, other.gate))
            if (
                other is check
                or them is None
                or other.runner != RUN_DATASETS
                or (other.unit, other.member) != (check.unit, check.member)
                or other.answered_by
                or them.settings != test.settings
                or relative_benchmark(task_data, them, every) != anchored
                or not set(test.datasets) <= set(them.datasets)
                or bool(test.benchmarks)
                and test.benchmarks[0] != them.benchmarks[0]
                or (set(test.datasets) == set(them.datasets) and other_index > index)
            ):
                continue
            out[index] = replace(check, answered_by=other.key)
            answers.add(other.key)
            notes.append(
                f"{check.key}: answered by {other.key}'s pass on the same tree, which measures "
                f"{', '.join(test.datasets)} on an engine of the same settings; it runs "
                "itself only where that check has not passed"
            )
            break
    return out


def screening(
    task_data: Mapping[str, Any],
    checks: Sequence[Check],
    tests: Mapping[tuple[str, str], verification_mod.Test],
    units: Sequence[str],
    notes: list[str],
) -> list[Check]:
    """``checks`` with :attr:`Check.screened_by` set: a speculator's ``e2e`` screens its base's.

    A speculator builds on the unit before it with the same weights; at the
    same member its engine adds only the drafts, which a target that verifies
    them greedily cannot change the answers of. Its ``e2e`` then scores what
    the base's does on the base's model, and a landing that names both runs
    the speculator's alone. The base's own decode path, without drafts, is not
    reached that way, so the base's ``e2e`` stays owed and runs once the work is
    done (``run/orchestrator.py``). Not for a speculator whose acceptance is
    relaxed, which changes the answers, nor where the two tests' datasets or
    settings differ.
    """
    out = list(checks)
    index = {c.key: i for i, c in enumerate(out)}
    accuracy = task_data.get("accuracy") or {}
    for position, unit in enumerate(units[1:], start=1):
        if unit not in SPECULATORS or _relaxed(accuracy, unit):
            continue
        base = units[position - 1]
        lead, follow = tests.get((unit, E2E)), tests.get((base, E2E))
        if lead is None or follow is None or lead.settings != follow.settings:
            continue
        if not set(follow.datasets) - {ACCEPTANCE} <= set(lead.datasets):
            continue
        screened = []
        for check in list(out):
            if (check.unit, check.gate) != (unit, E2E) or check.stage != STAGE_LANDING:
                continue
            key = f"{base}:{E2E}@{check.member}"
            if key not in index or out[index[key]].stage != STAGE_LANDING:
                continue
            out[index[key]] = replace(out[index[key]], screened_by=check.key)
            screened.append(check.member)
        if screened:
            notes.append(
                f"{base}:{E2E}: at a landing that names it with {unit}:{E2E} at the same member "
                f"({', '.join(screened)}), {unit}'s stands in for it (the same weights, the "
                f"target verifying its drafts); it runs once the work is done, unless what it "
                "reads is unchanged since it passed"
            )
    return out


def _relaxed(accuracy: Mapping[str, Any], unit: str) -> bool:
    """Whether ``unit``'s options accept drafts the target would not have answered with."""
    options = _options(accuracy, unit)
    speculative = options.get("speculative_config") if isinstance(options, Mapping) else None
    return isinstance(speculative, Mapping) and bool(
        speculative.get("use_relaxed_acceptance_for_thinking")
    )


def _confirm(checks: Sequence[Check], units: Sequence[str], variants: Sequence[str]) -> str:
    """The check that confirms the final tree (:attr:`Contract.confirm`).

    The first benchmark of the last unit that owes one, else of the last
    variant that does; the first engine check of the last unit where the run
    owes no benchmark at all.
    """
    order = {name: index for index, name in enumerate([*units, *variants])}
    scored = [c for c in checks if c.required and c.scores and c.runner == RUN_DATASETS]
    engines = [c for c in checks if c.required and c.runner == RUN_DATASETS and c.unit in units]
    for pool in (
        [c for c in scored if c.unit in units],
        [c for c in scored if c.unit not in units],
        engines,
    ):
        if pool:
            last = max(pool, key=lambda c: order.get(c.unit, -1)).unit
            return next(c.key for c in pool if c.unit == last)
    return ""


def write_contract(workspace: Path, contract: Contract) -> Path:
    path = Path(workspace) / CONTRACT_FILENAME
    path.write_text(yaml.safe_dump(contract.as_dict(), sort_keys=False), encoding="utf-8")
    return path


@dataclass
class Verdict:
    unit: str
    gate: str
    member: str
    passed: bool
    detail: str = ""
    tree_state: str = ""
    #: The target's identity when the check ran (``workflow._target_marker``).
    marker: str = ""
    #: The commit each command of the check imported the package under test
    #: from (``GateResult.imported``): one per leg or engine job that said.
    imported: list[str] = field(default_factory=list)
    #: Why the verdict says nothing about one tree (``GateResult.void``); such a
    #: verdict never passed.
    void: str = ""
    #: The check's :attr:`Check.inputs` when it ran.
    inputs: str = ""
    #: What the check read of the tree: the id of its list of Python files
    #: (``workflow._narrowed``), and the target's marker narrowed to them on
    #: the commit it measured. With both, the verdict holds on any later state
    #: of the target whose narrowed marker is the same (:func:`evaluate`).
    reads: str = ""
    reads_marker: str = ""
    #: How long the check's gate ran, in seconds; 0 when it did not say.
    duration_s: float = 0.0
    #: The score a benchmark measured (``GateResult.measured``), pass or fail.
    measured: float | None = None


#: Fields written only when they say something.
_UNSAID: dict[str, Any] = {
    "imported": [],
    "void": "",
    "inputs": "",
    "reads": "",
    "reads_marker": "",
    "duration_s": 0.0,
    "measured": None,
}
_VERDICT_FIELDS = frozenset(f.name for f in fields(Verdict))


def _score(value: Any) -> float | None:
    """A benchmark score a gate reported; ``None`` for one it did not."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _seconds(value: Any) -> float:
    """A duration a gate reported, as seconds; 0 for one it did not."""
    try:
        return round(float(value or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class ContractStatus:
    """The verdicts recorded against a contract, append-only on disk."""

    path: Path
    verdicts: list[Verdict] = field(default_factory=list)

    @classmethod
    def load(cls, workspace: Path) -> "ContractStatus":
        path = Path(workspace) / STATUS_FILENAME
        verdicts: list[Verdict] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    verdicts.append(
                        Verdict(**{k: v for k, v in data.items() if k in _VERDICT_FIELDS})
                    )
        return cls(path=path, verdicts=verdicts)

    def record(
        self,
        unit: str,
        result: Any,
        marker: str = "",
        inputs: str = "",
        *,
        reads: tuple[str, str] = ("", ""),
    ) -> Verdict:
        """Append one gate result, as returned by any gate, for ``unit``.

        ``marker`` is the target's state when the check was started; a result
        measured on a snapshot names the commit's own (``GateResult.marker``),
        and is recorded under that. ``inputs`` is the check's
        (:attr:`Check.inputs`). ``reads`` is what it read of the tree, as
        :attr:`Verdict.reads` and :attr:`Verdict.reads_marker`.
        """
        void = str(getattr(result, "void", "") or "")
        verdict = Verdict(
            unit=unit,
            gate=str(result.name),
            member=str(result.member or ""),
            passed=bool(result.passed) and not void,
            detail=str(result.detail)[:500],
            tree_state=str(getattr(result, "tree_state", "") or ""),
            marker=str(getattr(result, "marker", "") or "") or marker,
            imported=[str(i) for i in getattr(result, "imported", ()) or ()],
            void=void,
            inputs=inputs,
            reads=reads[0],
            reads_marker=reads[1],
            duration_s=_seconds(getattr(result, "duration_s", 0.0)),
            measured=_score(getattr(result, "measured", None)),
        )
        self.verdicts.append(verdict)
        line = {k: v for k, v in asdict(verdict).items() if k not in _UNSAID or v != _UNSAID[k]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        return verdict

    def latest(self) -> dict[str, Verdict]:
        """The last verdict per check key."""
        last: dict[str, Verdict] = {}
        for v in self.verdicts:
            last[f"{v.unit}:{v.gate}@{v.member}"] = v
        return last


@dataclass(frozen=True)
class Evaluation:
    passed: tuple[Check, ...]
    failed: tuple[tuple[Check, Verdict], ...]
    missing: tuple[Check, ...]

    @property
    def holds(self) -> bool:
        return not self.failed and not self.missing

    def render(self) -> str:
        lines = [
            f"{len(self.passed)} passed, {len(self.failed)} failed, {len(self.missing)} missing"
        ]
        lines += [f"- {'VOID' if v.void else 'FAILED'} {c.key}: {v.detail}" for c, v in self.failed]
        lines += [f"- MISSING {c.key}" for c in self.missing]
        return "\n".join(lines)


def evaluate(
    contract: Contract,
    verdicts: Iterable[Verdict] | ContractStatus,
    marker: str | None = None,
    holds: Callable[[Verdict], bool] | None = None,
) -> Evaluation:
    """Each required check against its latest verdict.

    A check that passed stands, whatever the target has become since. With
    ``marker``, the target's state now, a verdict that failed or is void counts
    on the state it measured only (or where ``holds`` says what it read of the
    tree is unchanged: :attr:`Verdict.reads`): once the target has changed, the
    check is missing and runs again. A verdict whose :attr:`Verdict.inputs` are
    not the check's is dropped too (one without inputs counts as the check's).
    """
    latest = (
        verdicts.latest()
        if isinstance(verdicts, ContractStatus)
        else {f"{v.unit}:{v.gate}@{v.member}": v for v in verdicts}
    )
    if marker is not None:
        latest = {k: v for k, v in latest.items() if v.passed or _here(v, marker, holds)}
    checks = {c.key: c for c in contract.checks}
    latest = {
        k: v
        for k, v in latest.items()
        if not v.inputs or k not in checks or not checks[k].inputs or v.inputs == checks[k].inputs
    }
    passed, failed, missing = [], [], []
    for check in contract.required():
        verdict = latest.get(check.key)
        if verdict is None:
            missing.append(check)
        elif verdict.passed:
            passed.append(check)
        else:
            failed.append((check, verdict))
    return Evaluation(tuple(passed), tuple(failed), tuple(missing))


def _here(verdict: Verdict, marker: str, holds: Callable[[Verdict], bool] | None) -> bool:
    """Whether ``verdict`` is about the target at ``marker``."""
    return verdict.marker == marker or bool(marker and holds is not None and holds(verdict))


def passed_here(
    contract: Contract,
    status: ContractStatus,
    check: Check,
    marker: str,
    holds: Callable[[Verdict], bool] | None = None,
) -> bool:
    """Whether ``check`` passed on the target at ``marker``, by its own verdict."""
    verdict = status.latest().get(check.key)
    if verdict is None or not _here(verdict, marker, holds):
        return False
    same_inputs = not verdict.inputs or not check.inputs or verdict.inputs == check.inputs
    return same_inputs and verdict.passed


def confirmed(
    contract: Contract,
    status: ContractStatus,
    marker: str,
    holds: Callable[[Verdict], bool] | None = None,
) -> bool:
    """Whether the target at ``marker`` was measured: a check that can see what a late change breaks passed here.

    The checks that passed earlier stand (:func:`evaluate`); this is the one
    measurement the final tree owes before delivery. Where the confirmation
    check (:attr:`Contract.confirm`) is a benchmark, any check that scores the
    engine's answers counts, the last landing's own included; one that only
    shows the engine answering (a boot, a serve, the acceptance of drafts)
    does not say it still scores. Where the run owes boots alone, any required
    check counts. A contract that names no confirmation check owes none.
    """
    if not contract.confirm:
        return True
    named = next((c for c in contract.checks if c.key == contract.confirm), None)
    scored = named is not None and named.scores
    return any(
        passed_here(contract, status, check, marker, holds)
        for check in contract.required()
        if not scored or check.scores
    )


def measured_where(
    evaluation: Evaluation,
    status: ContractStatus,
    marker: str,
    since: Callable[[str], str] = lambda tree: "",
) -> str:
    """A table of where each passed check was measured, for the delivery report.

    A pass stands whatever landed after it (:func:`evaluate`), so the report
    says which commit each check measured (:attr:`Verdict.tree_state`) and,
    through ``since``, how many commits the target has taken on top of it.
    """
    latest = status.latest()
    lines = ["| check | measured at | commits since |", "|---|---|---|"]
    for check in evaluation.passed:
        verdict = latest.get(check.key)
        if verdict is None:
            continue
        tree = verdict.tree_state or "unknown"
        after = "0, the final tree" if verdict.marker == marker else since(verdict.tree_state)
        lines.append(f"| {check.key} | {tree} | {after or 'unknown'} |")
    return "\n".join(lines)


def verdict_on(
    status: ContractStatus,
    key: str,
    marker: str,
    holds: Callable[[Verdict], bool] | None = None,
) -> Verdict | None:
    """``key``'s latest verdict if it is about the target at ``marker``, as :func:`evaluate` reads it."""
    verdict = status.latest().get(key)
    return verdict if verdict is not None and _here(verdict, marker, holds) else None


def measured_commit(
    status: ContractStatus, evaluation: Evaluation, checks: Iterable[Check], marker: str
) -> str | None:
    """The one commit every check of ``checks`` that passed on ``marker`` imported; ``None``.

    ``None`` when one of them names no commit, or two name different ones: a
    milestone names a commit only where its checks say which.
    """
    latest = status.latest()
    commits: set[str] = set()
    for check in checks:
        verdict = latest.get(check.key)
        if (
            verdict is None
            or not verdict.passed
            or verdict.marker != marker
            or not verdict.imported
        ):
            return None
        commits.update(verdict.imported)
    return commits.pop() if len(commits) == 1 else None


@dataclass(frozen=True)
class Milestone:
    """A tier delivered on one commit of the target.

    ``correctness``: every required check passed. ``bringup``: the first unit's
    benchmark, and its boot where it owes one, passed at its narrowest member
    (:func:`bringup_checks`).
    ``marker`` is the state of the target the checks measured, and ``sha`` the
    commit that holds exactly that state.
    """

    sha: str
    marker: str
    #: The branch that names ``sha`` (``<target branch>-correct``, ``-bringup``);
    #: ``""`` when none was written.
    ref: str = ""
    #: When it was recorded, and how long after the run opened (``None``: not known).
    at: float = 0.0
    elapsed_s: float | None = None
    #: How many required checks held.
    checks: int = 0
    #: :data:`MILESTONE_CORRECTNESS` or :data:`MILESTONE_BRINGUP`.
    tier: str = MILESTONE_CORRECTNESS

    def describe(self, marker: str | None = None, evaluation: Evaluation | None = None) -> str:
        """One sentence: where the tier was delivered and, given the target now, whether it still is."""
        what = "Bring-up" if self.tier == MILESTONE_BRINGUP else "Correctness"
        line = f"{what} delivered at {self.sha[:10]}"
        if self.ref:
            line += f" (`{self.ref}`)"
        if self.elapsed_s is not None:
            line += f", {self.elapsed_s / 3600:.1f} h after the run started"
        if marker is None or self.tier == MILESTONE_BRINGUP:
            return line + "."
        if marker == self.marker:
            return line + "; the target is still there."
        line += "; the target has moved since"
        if evaluation is not None and not evaluation.holds:
            line += (
                f", and on it {len(evaluation.failed)} required checks failed and "
                f"{len(evaluation.missing)} are stale or missing"
            )
        return line + ". It moves forward once the contract holds on a later commit."


def milestones(workspace: Path, tier: str | None = MILESTONE_CORRECTNESS) -> list[Milestone]:
    """Every milestone of ``tier`` the run recorded, in order; every one with ``tier=None``."""
    path = Path(workspace) / MILESTONE_FILENAME
    if not path.is_file():
        return []
    known = {f.name for f in fields(Milestone)}
    recorded = [
        Milestone(**{k: v for k, v in json.loads(line).items() if k in known})
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [m for m in recorded if tier is None or m.tier == tier]


def latest_milestone(workspace: Path) -> Milestone | None:
    """Where correctness stands delivered, or None before the contract first held."""
    recorded = milestones(workspace)
    return recorded[-1] if recorded else None


def bringup_milestone(workspace: Path) -> Milestone | None:
    """Where the bring-up was delivered, or None before it was."""
    recorded = milestones(workspace, MILESTONE_BRINGUP)
    return recorded[0] if recorded else None


def record_milestone(workspace: Path, milestone: Milestone) -> None:
    """Append ``milestone`` to ``correctness.jsonl``."""
    path = Path(workspace) / MILESTONE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(milestone)) + "\n")


def standing(evaluation: Evaluation, workspace: Path, marker: str | None = None) -> str:
    """``evaluation`` rendered, then where the bring-up and correctness stand delivered once they have been.

    ``marker`` is the state of the target ``evaluation`` was read on; ``""`` or ``None``
    when it is not established, and then nothing is said of whether the target moved.
    """
    rendered = evaluation.render()
    bringup = bringup_milestone(workspace)
    if bringup is not None:
        rendered = f"{rendered}\n\n{bringup.describe()}"
    milestone = latest_milestone(workspace)
    if milestone is None:
        return rendered
    return f"{rendered}\n\n{milestone.describe(marker or None, evaluation)}"


# ------------------------------------------------------------------- the bring-up


def bringup_checks(contract: Contract) -> tuple[Check, ...]:
    """The checks whose passes deliver the bring-up: the first unit's benchmark at one member.

    The flow's engine checks of the first unit at that member (its ``e2e``, and
    its ``boot`` where it owes one); where its benchmark runs at several
    members, the narrowest of them. The spec's own tests are not among them.

    Empty for a contract with no such checks.
    """
    if not contract.checks:
        return ()
    first = contract.checks[0].unit
    own = [
        c
        for c in contract.required()
        if c.unit == first
        and c.stage == STAGE_LANDING
        and c.runner == RUN_DATASETS
        and c.origin != verification_mod.ORIGIN_SPEC
    ]
    measured = {c.member for c in own if c.scores}
    if not measured:
        return ()
    narrowest = min(order_members_by_cost(sorted(measured)), key=_width)
    return tuple(c for c in own if c.member == narrowest)


def bringup_holds(contract: Contract, evaluation: Evaluation) -> bool:
    """Whether every bring-up check passed in ``evaluation``."""
    owed = bringup_checks(contract)
    passed = {c.key for c in evaluation.passed}
    return bool(owed) and all(c.key in passed for c in owed)


# ------------------------------------------------------------------- the cells

#: Where each cell's latest holding commit is kept, and the commit holding the most at once.
CELLS_FILENAME = "cells.json"


def cell_name(cell: tuple[str, str]) -> str:
    """``core@tp1`` for ``("core", "tp1")``."""
    return f"{cell[0]}@{cell[1]}"


def cells(contract: Contract) -> dict[tuple[str, str], tuple[Check, ...]]:
    """The required checks by cell: a unit or a variant at one member, in the contract's order."""
    out: dict[tuple[str, str], list[Check]] = {}
    for check in contract.required():
        out.setdefault((check.unit, check.member), []).append(check)
    return {cell: tuple(checks) for cell, checks in out.items()}


def cells_holding(contract: Contract, evaluation: Evaluation) -> list[tuple[str, str]]:
    """The cells every one of whose checks passed in ``evaluation``, in the contract's order."""
    passed = {c.key for c in evaluation.passed}
    return [cell for cell, owed in cells(contract).items() if all(c.key in passed for c in owed)]


def load_cells(workspace: Path) -> dict[str, Any]:
    """``cells.json``: ``{"cells": {name: {sha, at}}, "best": {sha, cells, ref, at} | None}``."""
    path = Path(workspace) / CELLS_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {"cells": dict(data.get("cells") or {}), "best": data.get("best")}


def save_cells(workspace: Path, record: Mapping[str, Any]) -> None:
    write_text_atomic(
        Path(workspace) / CELLS_FILENAME, json.dumps(record, indent=1, sort_keys=True)
    )


__all__ = [
    "CELLS_FILENAME",
    "CONTRACT_FILENAME",
    "Check",
    "Contract",
    "ContractStatus",
    "Evaluation",
    "LandingBudget",
    "MILESTONE_BRINGUP",
    "MILESTONE_CORRECTNESS",
    "MILESTONE_FILENAME",
    "Milestone",
    "STATUS_FILENAME",
    "STAGE_DELIVERY",
    "STAGE_LANDING",
    "Verdict",
    "benchmark_members",
    "bringup_checks",
    "bringup_holds",
    "bringup_milestone",
    "cell_name",
    "cells",
    "cells_holding",
    "check_inputs",
    "compile_contract",
    "DEFAULT_BENCHMARK",
    "E2E_TIMEOUT_S",
    "flow_tests",
    "run_tests",
    "test_members",
    "evaluate",
    "landing_budget",
    "latest_milestone",
    "load_cells",
    "measured_commit",
    "milestones",
    "record_milestone",
    "save_cells",
    "standing",
    "write_contract",
]
