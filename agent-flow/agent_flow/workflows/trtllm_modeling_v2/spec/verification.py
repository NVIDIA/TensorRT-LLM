# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The run's tests (``verification``): what the target must pass, by feature, at a stage.

Every test of a run is one entry of one list. The flow derives a default list
from the spec (``run/validation/contract.flow_tests``): per feature (core, each
feature, each configuration variant) an ``e2e`` test, and the serve and
disaggregated-serving tests where the spec asks for them. ``verification``
changes that list, feature by feature: an entry named like one of the flow's
tests (:data:`FLOW_TEST_NAMES`) changes it, ``enabled: false`` drops it, and
any other name adds a test::

    verification:
      core:
        e2e:                        # the flow's own: run MMLU too, at delivery
          datasets: [generate, gsm8k, mmlu]
          stage: delivery
        moe_routing:                # a test of the spec's own
          purpose: the router picks the experts the HF reference picks
          pytest: tests/unittest/_torch/modeling_v2/test_moe_routing.py
          devices: 1
        sparse_long:                # one that needs inputs long enough to reach its path
          purpose: past the TopK budget the indexer runs, and LongBench v2 holds its reference
          datasets: [longbench_v2]
          config: {max_seq_len: 131072, enable_chunked_prefill: true}
          extra_checks: [sparse_topk_exceeded]
          stage: delivery

``e2e`` is each feature's foundation test, at each of its members: one engine
of the feature's whole configuration answers the boot prompts, scores the full
benchmark against its reference and, on a speculator, measures its acceptance.
It says the model is right in that configuration, and it is the target's CI
test. It is not cut down (no subset, no lowered threshold), and it promises no
particular path: what runs only at some length, shape or switch (sparse
attention past its TopK budget, chunked prefill, a parallel shape) is a test of
the spec's own, which shows that its path ran (``extra_checks``).

Every entry is flat. A test runs one of two ways:

- ``datasets``: a list of names from the catalog (:data:`DATASETS`), run in
  order on one engine at each member by the tree's own runner
  (``tests/integration/defs/accuracy/modeling_v2/_e2e.py``, through the test the
  flow generates for the target: ``checks/e2e.py``). ``generate`` sends the
  boot prompts, ``acceptance`` measures a speculator's acceptance length, and
  every other name is an ``accuracy_core`` benchmark judged against its
  references row. With ``serve: true`` the prompts go through ``trtllm-serve``
  instead (``generate`` only).
- ``pytest``: pytest nodes of the tree, run from their pytest root as CI runs
  them and judged by their JUnit report: every selected test passed and at
  least one was selected; a skip fails it. With ``members`` the test runs once
  per member, handed that member's configuration of its feature
  (:data:`HANDOVER_ENV`); with ``devices`` (one by default) it runs once.

What a test changes of its feature's settings is its own, so tests that need
other settings are other entries:

- ``config``: LLM-API options laid over the feature's configuration at the
  member (``max_seq_len``, ``enable_chunked_prefill``,
  ``sparse_attention_config``, ...), last, as ``LLM(**options)`` takes them. A
  test of datasets runs its engine with them and a whole-model pytest test is
  handed them. What the member and the feature decide (:data:`FLOW_CONFIG_KEYS`)
  is not a test's to change: a configuration the run delivers is a variant.
- ``evaluator_kwargs``: laid over ``accuracy.evaluator_kwargs`` for this test's
  benchmarks (``num_fewshot``, ``chat_template_kwargs``, ...).
- ``extra_checks``: names of functions of the tree that judge what the test's
  datasets did, beyond each dataset's own judgment (that sparse attention's top-k
  selection ran, ...). Each is ``f(result)``: it takes the case's result and nothing
  else, and derives what it holds the run to from it. The tree's runner defines one
  and runs it after the datasets (``tests/integration/defs/accuracy/modeling_v2/_e2e.py``,
  "Extra checks"). A name is looked up in two files: the target's own
  ``extra_checks.py`` beside its generated test, then the shared ``_extra_checks.py``
  (``checks/e2e.extra_check_files``); a name both define is refused. The flow only
  finds the function (:func:`extra_check_function`) and carries where it is into the
  case. A new kind of evidence is a new function, not a change here.

The flow reads none of these: it passes them to ``LLM``, to ``accuracy_core`` and
to the tree's runner, and two tests that differ in them measure differently.

Each test runs at a stage:

- ``landing``: when a landing names it (the agent loop), and once more after the
  work if it has not passed;
- ``delivery``: after the work is done and every other required test has passed.

A test × member is one check of the contract, keyed ``<feature>:<name>@<member>``.
The contract runs one measurement once (``run/validation/contract.py``): a
test of datasets that another test of the same feature and settings measures at
the same member is answered by that test's pass on the same tree.
A pytest test is *defined* when the tree's sources hold its file, class and
function (:func:`defined`); nothing is imported, collected or run to say so.

This module reads the spec alone: the schema calls :func:`validate` with what it
knows of the spec's features and members, and the contract merges the entries
into the flow's list (:func:`apply`).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

VERIFICATION_FIELD = "verification"

STAGE_LANDING = "landing"
STAGE_DELIVERY = "delivery"
STAGES: tuple[str, ...] = (STAGE_LANDING, STAGE_DELIVERY)

TEST_KEYS: tuple[str, ...] = (
    "purpose",
    "datasets",
    "pytest",
    "serve",
    "members",
    "devices",
    "stage",
    "timeout",
    "env",
    "enabled",
    "config",
    "evaluator_kwargs",
    "extra_checks",
)

#: LLM-API options the member and the feature decide; a test's ``config`` leaves them.
FLOW_CONFIG_KEYS: tuple[str, ...] = (
    "model",
    "modeling_v2",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "moe_expert_parallel_size",
    "enable_attention_dp",
    "speculative_config",
)

#: How a test runs: its datasets on the tree's runner, its prompts through
#: ``trtllm-serve``, its pytest nodes, or the disaggregated-serving test.
RUN_DATASETS = "datasets"
RUN_SERVE = "serve"
RUN_PYTEST = "pytest"
RUN_DISAGG = "disagg"

#: The boot prompts (``accuracy.boot``): every answer must be non-empty.
DATASET_GENERATE = "generate"
#: A speculator's acceptance length (``accuracy.acceptance``).
DATASET_ACCEPTANCE = "acceptance"
#: The benchmarks, by their ``accuracy_core`` dataset name (the stem of their
#: ``references/<name>.yaml``), and their ``accuracy_core`` task.
BENCHMARKS: dict[str, str] = {
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
    "gpqa_diamond": "GPQADiamond",
    "mmmu": "MMMU",
    "cnn_dailymail": "CnnDailymail",
    "json_mode_eval": "JsonModeEval",
    "longbench_v1": "LongBenchV1",
    "longbench_v2": "LongBenchV2",
    "passkey_retrieval_64k": "PassKeyRetrieval64k",
    "passkey_retrieval_128k": "PassKeyRetrieval128k",
}
DATASETS: tuple[str, ...] = (DATASET_GENERATE, DATASET_ACCEPTANCE, *BENCHMARKS)

#: Who declared a test: the flow (``contract.flow_tests``), the flow with the
#: spec changing it, or the spec alone.
ORIGIN_FLOW = "flow"
ORIGIN_CHANGED = "flow, changed by the spec"
ORIGIN_SPEC = "spec"

#: The names of the tests the flow derives; an entry of one of them changes the flow's test.
E2E = "e2e"
BOOT = "boot"
SERVE = "serve"
SERVE_ONE_RANK = "serve_one_rank"
DISAGG = "disagg"
FLOW_TEST_NAMES: tuple[str, ...] = (E2E, BOOT, SERVE, SERVE_ONE_RANK, DISAGG)

#: A test's name: it names the test's checks (``core:e2e@tep4``).
TEST_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
#: ``members: all``: every member the feature is gated at.
ALL_MEMBERS = "all"
#: Seconds one test may take at one member when it states none.
DEFAULT_TIMEOUT_S = 3600
#: In a node id, the member and the feature of the check that runs it.
MEMBER_PLACEHOLDER = "{member}"
FEATURE_PLACEHOLDER = "{feature}"

#: What a whole-model pytest test is handed, as environment variables of its pytest process.
FEATURE_ENV = "MV2FLOW_FEATURE"
MEMBER_ENV = "MV2FLOW_MEMBER"
MODEL_ENV = "MV2FLOW_MODEL"
LLM_OPTIONS_ENV = "MV2FLOW_LLM_OPTIONS"
HANDOVER_ENV: tuple[str, ...] = (FEATURE_ENV, MEMBER_ENV, MODEL_ENV, LLM_OPTIONS_ENV)

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Test:
    """One test of the run: the flow's, as the spec left it, or the spec's own."""

    name: str
    #: The unit or variant it proves, and whose target and settings it runs with.
    feature: str
    purpose: str
    #: How it runs (:data:`RUN_DATASETS`, :data:`RUN_SERVE`, :data:`RUN_PYTEST`, :data:`RUN_DISAGG`).
    runner: str
    #: Catalog names, run in order on one engine (:data:`DATASETS`).
    datasets: tuple[str, ...] = ()
    #: Repository-relative pytest node ids; each may name ``{member}`` and ``{feature}``.
    nodes: tuple[str, ...] = ()
    stage: str = STAGE_LANDING
    #: Where it runs, one check each: members, or :data:`ALL_MEMBERS`; ``()`` for a
    #: module test (pytest with ``devices``), and for a spec's datasets test that
    #: names none (the feature's cheapest member).
    members: tuple[str, ...] = ()
    #: The devices a module test runs on; ``0`` for a test that runs per member.
    devices: int = 0
    timeout: int = DEFAULT_TIMEOUT_S
    env: Mapping[str, str] | None = None
    origin: str = ORIGIN_SPEC
    #: LLM-API options laid over its feature's configuration at each member.
    config: Mapping[str, Any] | None = None
    #: Laid over ``accuracy.evaluator_kwargs`` for its benchmarks.
    evaluator_kwargs: Mapping[str, Any] | None = None
    #: Names of functions of the tree that judge what its datasets did (``f(result)``).
    extra_checks: tuple[str, ...] = ()

    @property
    def settings(self) -> tuple[Any, ...]:
        """What it changes of its feature's settings: two tests that differ here measure differently."""
        return (
            dict(self.config or {}),
            dict(self.evaluator_kwargs or {}),
            tuple(self.extra_checks),
            dict(self.env or {}),
        )

    @property
    def whole_model(self) -> bool:
        """Whether it runs per member with that member's configuration (every test but a module test)."""
        return self.runner != RUN_PYTEST or bool(self.members)

    @property
    def of_the_flow(self) -> bool:
        """Whether the flow derived it (the spec may have changed it)."""
        return self.origin != ORIGIN_SPEC

    @property
    def gate(self) -> str:
        """The name its checks carry: the test's own."""
        return self.name

    @property
    def benchmarks(self) -> tuple[str, ...]:
        """The benchmarks among its datasets."""
        return tuple(d for d in self.datasets if d in BENCHMARKS)

    @property
    def scores(self) -> bool:
        """Whether it judges the engine's answers by a score: a benchmark, or the disaggregated test."""
        return bool(self.benchmarks) or self.runner == RUN_DISAGG

    def node_ids(self, member: str | None) -> tuple[str, ...]:
        """Its node ids at ``member``, the placeholders filled in."""
        return tuple(
            node.replace(MEMBER_PLACEHOLDER, str(member or "")).replace(
                FEATURE_PLACEHOLDER, self.feature
            )
            for node in self.nodes
        )


def module_member(devices: int) -> str:
    """The member a module test's check is recorded at: ``tp<N>``, its devices."""
    return f"tp{int(devices)}"


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    return ()


def entries(data: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """``verification``'s entries by ``(feature, name)``, in declaration order."""
    by_feature = data.get(VERIFICATION_FIELD)
    if not isinstance(by_feature, Mapping):
        return {}
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for feature, named in by_feature.items():
        if not isinstance(named, Mapping):
            continue
        for name, entry in named.items():
            if isinstance(entry, Mapping):
                out[(str(feature), str(name))] = entry
    return out


def changes(entry: Mapping[str, Any], base: Test) -> Test | None:
    """``base``, a test of the flow's, as ``entry`` changes it; ``None`` when it turns it off.

    Each key the entry gives replaces the test's. ``pytest`` makes it a pytest
    test, ``datasets`` a datasets test (``serve`` keeps it on ``trtllm-serve``).
    """
    if entry.get("enabled") is False:
        return None
    test = _with_fields(base, entry)
    return test if test == base else replace(test, origin=ORIGIN_CHANGED)


def declared(feature: str, name: str, entry: Mapping[str, Any]) -> Test:
    """The spec's own test ``name`` of ``feature``, from its validated entry."""
    runner = RUN_PYTEST if entry.get("pytest") else RUN_DATASETS
    base = Test(name=name, feature=feature, purpose="", runner=runner, origin=ORIGIN_SPEC)
    return _with_fields(base, entry)


def _with_fields(base: Test, entry: Mapping[str, Any]) -> Test:
    fields: dict[str, Any] = {}
    if entry.get("purpose"):
        fields["purpose"] = " ".join(str(entry["purpose"]).split())
    if entry.get("pytest"):
        fields.update(runner=RUN_PYTEST, nodes=_strings(entry["pytest"]), datasets=())
    if entry.get("datasets"):
        fields.update(datasets=_strings(entry["datasets"]), nodes=())
        fields["runner"] = (
            RUN_SERVE if entry.get("serve", base.runner == RUN_SERVE) else RUN_DATASETS
        )
    elif "serve" in entry and base.runner in (RUN_DATASETS, RUN_SERVE):
        fields["runner"] = RUN_SERVE if entry["serve"] else RUN_DATASETS
    if entry.get("members") is not None:
        fields.update(members=_strings(entry["members"]), devices=0)
    if entry.get("devices") is not None:
        fields.update(devices=int(entry["devices"]), members=())
    runner = fields.get("runner", base.runner)
    if runner == RUN_PYTEST and not fields.get("members", base.members):
        fields["devices"] = fields.get("devices") or base.devices or 1
    if entry.get("stage"):
        fields["stage"] = str(entry["stage"])
    if entry.get("timeout"):
        fields["timeout"] = int(entry["timeout"])
    if entry.get("env") is not None:
        fields["env"] = {str(k): str(v) for k, v in (entry.get("env") or {}).items()} or None
    for key in ("config", "evaluator_kwargs"):
        if entry.get(key) is not None:
            fields[key] = dict(entry[key]) or None
    if entry.get("extra_checks") is not None:
        fields["extra_checks"] = tuple(str(name) for name in entry["extra_checks"])
    return replace(base, **fields)


def apply(flow: list[Test], data: Mapping[str, Any]) -> list[Test]:
    """The run's tests: ``flow``, the flow's own, as ``verification`` changes them, then the spec's own.

    An entry named like one of ``flow``'s tests of its feature changes it
    (:func:`changes`); any other is a test of the spec's own (:func:`declared`),
    and must say what it proves and how it runs. Raises ``ValueError`` naming
    every entry that does neither.
    """
    declared_entries = entries(data)
    out: list[Test] = []
    for test in flow:
        entry = declared_entries.get((test.feature, test.name))
        if entry is None:
            out.append(test)
            continue
        changed = changes(entry, test)
        if changed is not None:
            out.append(changed)
    flow_keys = {(t.feature, t.name) for t in flow}
    problems: list[str] = []
    for (feature, name), entry in declared_entries.items():
        if (feature, name) in flow_keys:
            continue
        where = f"{VERIFICATION_FIELD}.{feature}.{name}"
        if entry.get("enabled") is False:
            problems.append(f"'{where}' turns off a test the flow does not derive for {feature}")
        elif not (entry.get("purpose") and (entry.get("pytest") or entry.get("datasets"))):
            problems.append(
                f"'{where}' changes a test the flow does not derive for {feature}; as a test of "
                "the spec's own it needs 'purpose' and 'datasets' or 'pytest'"
            )
        else:
            out.append(declared(feature, name, entry))
    for test in out:
        where = f"{VERIFICATION_FIELD}.{test.feature}.{test.name}"
        if test.runner == RUN_SERVE and test.datasets != (DATASET_GENERATE,):
            problems.append(
                f"'{where}' serves through trtllm-serve, which sends the boot prompts only: its "
                f"datasets are [{DATASET_GENERATE}], not {list(test.datasets)}"
            )
        if test.config and not (
            test.runner == RUN_DATASETS or (test.runner == RUN_PYTEST and test.members)
        ):
            problems.append(
                f"'{where}.config' lays options over an engine of its feature: a test of "
                "datasets, or a pytest test with 'members', has one; this test has none"
            )
        if test.evaluator_kwargs and not (test.runner == RUN_DATASETS and test.benchmarks):
            problems.append(
                f"'{where}.evaluator_kwargs' applies to the test's benchmarks, and it runs none "
                f"({list(BENCHMARKS)})"
            )
        if test.extra_checks and test.runner != RUN_DATASETS:
            problems.append(
                f"'{where}.extra_checks' judge what a test's datasets did on the tree's runner; "
                "this test runs none there"
            )
        clash = sorted(set(test.datasets) & set(test.extra_checks))
        if clash:
            problems.append(f"'{where}.extra_checks' {clash} are named like its datasets")
    if problems:
        raise ValueError("; ".join(problems))
    return out


def validate(
    data: Mapping[str, Any],
    errors: list[str],
    *,
    features: set[str],
    member_error: Callable[[str], str | None],
) -> None:
    """``verification``: every entry names a feature of the spec and says something a test can be.

    ``features`` are the unit and variant names a test may prove; ``member_error``
    says what is wrong with a member's spelling, ``None`` when nothing is. Whether
    an entry of a flow test's name finds that test is the contract's to say
    (:func:`apply`): the flow's tests depend on the profile and the features.
    """
    by_feature = data.get(VERIFICATION_FIELD)
    if by_feature is None:
        return
    if not isinstance(by_feature, Mapping):
        errors.append(
            f"'{VERIFICATION_FIELD}' must map a feature (core, a feature or a variant) to its tests"
        )
        return
    for feature, named in by_feature.items():
        where = f"{VERIFICATION_FIELD}.{feature}"
        if str(feature) not in features:
            errors.append(f"'{where}' names no feature of this spec ({sorted(features)})")
        if not isinstance(named, Mapping) or not named:
            errors.append(f"'{where}' must map a test's name to its entry")
            continue
        for name, entry in named.items():
            _validate_entry(f"{where}.{name}", str(name), entry, errors, member_error)


def _validate_entry(
    where: str,
    name: str,
    entry: Any,
    errors: list[str],
    member_error: Callable[[str], str | None],
) -> None:
    if not TEST_NAME.match(name):
        errors.append(f"'{where}': a test's name is letters, digits, '_', '.' and '-'")
    if not isinstance(entry, Mapping):
        errors.append(f"'{where}' must be a mapping with keys {list(TEST_KEYS)}")
        return
    for key in entry:
        if key not in TEST_KEYS:
            errors.append(f"'{where}.{key}' is not a key of a test ({list(TEST_KEYS)})")
    flows = name in FLOW_TEST_NAMES
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        errors.append(f"'{where}.enabled' must be true or false")
    purpose = entry.get("purpose")
    if purpose is not None and (not isinstance(purpose, str) or not purpose.strip()):
        errors.append(f"'{where}.purpose' must say what the test proves and what passing means")
    elif purpose is None and not flows and enabled is not False:
        errors.append(f"'{where}.purpose' must say what the test proves and what passing means")
    nodes, datasets = entry.get("pytest"), entry.get("datasets")
    if nodes is not None and datasets is not None:
        errors.append(f"'{where}' sets both 'datasets' and 'pytest'; a test runs one way")
    elif nodes is None and datasets is None and not flows and enabled is not False:
        errors.append(
            f"'{where}' must say how it runs: 'datasets' (from {list(DATASETS)}) or 'pytest'"
        )
    if nodes is not None:
        _validate_nodes(where, nodes, entry, errors, per_member=flows)
    if datasets is not None:
        _validate_datasets(where, datasets, entry, errors)
    serve = entry.get("serve")
    if serve is not None and not isinstance(serve, bool):
        errors.append(f"'{where}.serve' must be true or false")
    elif serve and nodes is not None:
        errors.append(f"'{where}.serve' sends prompts through trtllm-serve; a pytest test has none")
    members, devices = entry.get("members"), entry.get("devices")
    if members is not None and devices is not None:
        errors.append(
            f"'{where}' sets both 'members' (once per member) and 'devices' (a module test, "
            "once); say one"
        )
    if members is not None:
        listed = [members] if isinstance(members, str) else members
        if not isinstance(listed, list) or not listed:
            errors.append(f"'{where}.members' must be a member, a list of them, or '{ALL_MEMBERS}'")
        elif listed != [ALL_MEMBERS]:
            for member in listed:
                said = member_error(str(member)) if isinstance(member, str) else "not a string"
                if said:
                    errors.append(f"'{where}.members': {member!r}: {said}")
    if devices is not None:
        if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
            errors.append(f"'{where}.devices' must be an integer >= 1")
        elif nodes is None:
            errors.append(
                f"'{where}.devices' makes a pytest test a module test; a test of datasets runs "
                "at members"
            )
    stage = entry.get("stage")
    if stage is not None and stage not in STAGES:
        errors.append(f"'{where}.stage' must be one of {list(STAGES)}, got {stage!r}")
    timeout = entry.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0
    ):
        errors.append(f"'{where}.timeout' must be a positive number of seconds")
    env = entry.get("env")
    if env is not None:
        if not isinstance(env, Mapping):
            errors.append(f"'{where}.env' must be a mapping from variable to value")
        else:
            for var, value in env.items():
                if not isinstance(var, str) or not _ENV_NAME.fullmatch(var):
                    errors.append(
                        f"'{where}.env' names {var!r}, which is not a shell variable name"
                    )
                elif var in HANDOVER_ENV:
                    errors.append(f"'{where}.env.{var}' is set by the flow for a whole-model test")
                if not isinstance(value, (str, int, float, bool)):
                    errors.append(f"'{where}.env.{var}' must be a scalar the shell can carry")
    config = entry.get("config")
    if config is not None:
        if not isinstance(config, Mapping) or not config:
            errors.append(f"'{where}.config' must map LLM-API options to their values")
        else:
            for key in config:
                if key in FLOW_CONFIG_KEYS:
                    errors.append(
                        f"'{where}.config.{key}' is decided by the member and the feature; a "
                        "configuration the run delivers is a variant"
                    )
    kwargs = entry.get("evaluator_kwargs")
    if kwargs is not None and (not isinstance(kwargs, Mapping) or not kwargs):
        errors.append(
            f"'{where}.evaluator_kwargs' must map accuracy_core evaluator arguments to their values"
        )
    if nodes is not None and kwargs is not None:
        errors.append(f"'{where}.evaluator_kwargs' applies to benchmarks; a pytest test runs none")
    checks = entry.get("extra_checks")
    if checks is not None:
        _validate_extra_checks(where, checks, errors)
        if nodes is not None:
            errors.append(
                f"'{where}.extra_checks' judge what a test's datasets did; a pytest test "
                "asserts in its own code"
            )


def _validate_nodes(
    where: str, nodes: Any, entry: Mapping[str, Any], errors: list[str], *, per_member: bool
) -> None:
    """``pytest``'s node ids; ``per_member``: the test runs at members whatever the entry says."""
    listed = [nodes] if isinstance(nodes, str) else nodes
    if not isinstance(listed, list) or not listed:
        errors.append(f"'{where}.pytest' must be a pytest node id, or a nonempty list of them")
        return
    for node in listed:
        if (
            not isinstance(node, str)
            or not node.strip()
            or Path(node.partition("::")[0]).is_absolute()
            or not node.partition("::")[0].endswith(".py")
        ):
            errors.append(
                f"'{where}.pytest' must name repository-relative test files "
                f"(`tests/.../test_x.py` or `tests/.../test_x.py::TestY::test_z`), got {node!r}"
            )
        elif entry.get("members") is None and not per_member and MEMBER_PLACEHOLDER in node:
            errors.append(
                f"'{where}.pytest' names {MEMBER_PLACEHOLDER}, which only a test with 'members' has"
            )


#: An extra check's name: a public function's (one starting with ``_`` is a helper).
EXTRA_CHECK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _validate_extra_checks(where: str, checks: Any, errors: list[str]) -> None:
    if not isinstance(checks, list) or not checks:
        errors.append(f"'{where}.extra_checks' must be a nonempty list of extra checks' names")
        return
    for name in checks:
        if not isinstance(name, str) or not EXTRA_CHECK_NAME.fullmatch(name):
            errors.append(
                f"'{where}.extra_checks' names {name!r}; an extra check is named by its "
                "function, a public one of the target's extra_checks.py or the shared "
                "_extra_checks.py, and takes no arguments"
            )
    if len(set(map(str, checks))) != len(checks):
        errors.append(f"'{where}.extra_checks' names an extra check twice")


def _validate_datasets(
    where: str, datasets: Any, entry: Mapping[str, Any], errors: list[str]
) -> None:
    if not isinstance(datasets, list) or not datasets:
        errors.append(f"'{where}.datasets' must be a nonempty list of names from {list(DATASETS)}")
        return
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        errors.append(
            f"'{where}.datasets' names {unknown}, not in the catalog {list(DATASETS)}; another "
            "evaluation is a test of the tree, named with 'pytest'"
        )
    if len(set(map(str, datasets))) != len(datasets):
        errors.append(f"'{where}.datasets' names a dataset twice")
    if entry.get("serve") and list(datasets) != [DATASET_GENERATE]:
        errors.append(
            f"'{where}.serve' sends the boot prompts through trtllm-serve: its datasets are "
            f"[{DATASET_GENERATE}]"
        )


# ----------------------------------------------------------------------------- definition


def extra_check_function(
    repo: str | Path, files: Sequence[str], name: str
) -> tuple[str | None, str]:
    """Where extra check ``name`` is defined among ``files``, ``<file>::<name>``; else why not.

    Read from the source, nothing imported. ``files`` are where the run looks, in order
    (``checks/e2e.extra_check_files``); a file that is not there is passed over. The
    check is a public module-level function of exactly one of them, of one parameter,
    the case's result: the runner calls it as ``f(result)``. ``None`` and the reason
    when it is in none, in two, of another signature, or a file cannot be read.
    """
    found: list[str] = []
    for file in files:
        path = Path(repo) / file
        if not path.is_file():
            continue
        tree = _parse(path)
        if tree is None:
            return None, f"{file} could not be read as Python"
        function = next(
            (
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
            ),
            None,
        )
        if function is None:
            continue
        spec = function.args
        parameters = [*spec.posonlyargs, *spec.args]
        if len(parameters) != 1 or spec.defaults or spec.vararg or spec.kwonlyargs or spec.kwarg:
            return None, (
                f"{file}::{name} is not f(result): an extra check takes the case's result and "
                "nothing else, and derives what it holds the run to from it"
            )
        found.append(f"{file}::{name}")
    if len(found) > 1:
        return None, f"{name} is defined in {' and '.join(found)}; one name, one check"
    if not found:
        return None, f"{name} is defined in none of {', '.join(files)}"
    return found[0], found[0]


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None


def _imported(tree: ast.Module, path: Path, repo: Path, name: str) -> tuple[Path, str] | None:
    """Where module ``path`` imports ``name`` from: the file, and its name there; else ``None``."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        for alias in node.names:
            if (alias.asname or alias.name) != name:
                continue
            base = repo
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
            target = base.joinpath(*node.module.split("."))
            for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                if candidate.is_file():
                    return candidate, alias.name
    return None


def _binds(tree: ast.Module, name: str) -> bool:
    """Whether ``tree`` binds ``name`` at module level by an import or an assignment."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.partition(".")[0]) == name for alias in node.names):
                return True
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return True
    return False


def _has_method(path: Path, repo: Path, cls: str, method: str, depth: int = 4) -> bool | None:
    """Whether class ``cls`` of module ``path`` has ``method``, its own or a base's.

    Bases are followed through the tree's sources: one the module defines, or
    imports from a module of the tree. ``None`` when one cannot be followed, so
    the method may be inherited from it.
    """
    tree = _parse(path)
    if tree is None or depth < 0:
        return None
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None)
    if node is None:
        source = _imported(tree, path, repo, cls)
        return _has_method(source[0], repo, source[1], method, depth - 1) if source else None
    if any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method
        for n in node.body
    ):
        return True
    unknown = False
    for base in node.bases:
        if not isinstance(base, ast.Name):
            unknown = True
            continue
        found = _has_method(path, repo, base.id, method, depth - 1)
        if found:
            return True
        unknown = unknown or found is None
    return None if unknown else False


def defined(repo: str | Path, node: str) -> tuple[bool | None, str]:
    """Whether the tree at ``repo`` defines pytest node ``node``, and what says so.

    Defined when its file holds its class and the class its function, its own or
    one a base of the tree gives it. A parametrization (``[tep4]``) is not looked
    at, and a base the sources do not show counts as giving it. ``None`` when the
    tree cannot be read for it.
    """
    file, _, rest = node.partition("::")
    path = Path(repo) / file
    if not file or not path.is_file():
        return False, f"{file or '(no file)'} is not in the tree"
    names = [part.partition("[")[0] for part in rest.split("::") if part]
    if not names:
        return True, file
    tree = _parse(path)
    if tree is None:
        return None, f"{file} could not be read as Python"
    first = names[0]
    local = any(
        isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == first
        for n in tree.body
    )
    if not local and not _binds(tree, first):
        return False, f"{file} defines no `{first}`"
    if len(names) > 1 and _has_method(path, Path(repo), first, names[1]) is False:
        return False, f"{file} defines no `{first}::{names[1]}`"
    return True, node


__all__ = [
    "ALL_MEMBERS",
    "BENCHMARKS",
    "BOOT",
    "DATASETS",
    "DATASET_ACCEPTANCE",
    "DATASET_GENERATE",
    "DEFAULT_TIMEOUT_S",
    "DISAGG",
    "E2E",
    "FEATURE_ENV",
    "FEATURE_PLACEHOLDER",
    "FLOW_TEST_NAMES",
    "HANDOVER_ENV",
    "LLM_OPTIONS_ENV",
    "MEMBER_ENV",
    "MEMBER_PLACEHOLDER",
    "MODEL_ENV",
    "ORIGIN_CHANGED",
    "ORIGIN_FLOW",
    "ORIGIN_SPEC",
    "RUN_DATASETS",
    "RUN_DISAGG",
    "RUN_PYTEST",
    "RUN_SERVE",
    "SERVE",
    "SERVE_ONE_RANK",
    "STAGES",
    "STAGE_DELIVERY",
    "STAGE_LANDING",
    "TEST_KEYS",
    "TEST_NAME",
    "Test",
    "VERIFICATION_FIELD",
    "apply",
    "changes",
    "declared",
    "defined",
    "entries",
    "module_member",
    "validate",
]
