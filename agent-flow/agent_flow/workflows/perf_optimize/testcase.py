# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read a pytest test-case id as a workload definition.

A campaign normally names its workload with an ``extra_llm_api_options``
YAML and drives it through ``trtllm-serve`` + ``benchmark_serving.py``.
The alternative this module supports is to name a **test case** instead,
from one of two families:

``tests/integration/defs/perf/test_perf_sanity.py::test_e2e[...]``
    The CI perf-sanity suite. The id selects a config YAML under
    ``tests/scripts/perf-sanity/`` and, for the plain aggregated shape,
    which ``server_configs`` entry of it to run.

``tests/integration/defs/perf/test_perf.py::test_perf[...]``
    The QA perf suite. The id *is* the configuration: ``PerfTestConfig``
    serialises itself into the id with ``to_string()`` and reads it back
    with ``load_from_str()``, so every knob the id spells out is pinned by
    it and the rest comes from ``pytorch_model_config.py``.

Why the id is worth parsing at all: it is the identity CI, the regression
report and the waive lists already use, so a campaign that accepts it
needs no translation step and cannot disagree with the case CI measured.

**The perf-sanity grammar here is a mirror.** Its owner is
:data:`GRAMMAR_SOURCE` in the checkout under test, which means a campaign
whose checkout changes the grammar leaves this parser stale. That is the
accepted cost of keeping this module pure and unit-testable; the tests pin
every shape the owning docstring documents, so a drift shows up as a test
to update rather than as a run against the wrong config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The function this module mirrors. Named rather than imported: it lives in
# the repo under test, it is only importable with pytest present, and
# constructing its caller shells out to ``nvidia-smi``.
GRAMMAR_SOURCE = "tests/integration/defs/perf/test_perf_sanity.py::parse_test_string"

FAMILY_PERF_SANITY = "perf_sanity"
FAMILY_PERF = "perf"

_PERF_SANITY_MODULE = "test_perf_sanity.py"
_PERF_MODULE = "test_perf.py"

# Mirrors ``test_perf_sanity.py``. ``AGG_CONFIG_FOLDER`` and
# ``DISAGG_CONFIG_FOLDER`` are environment-overridable there; the defaults
# are what a campaign resolves against, and an override is a deliberate
# act the campaign would have to opt into.
AGG_CONFIG_SUBDIR = "tests/scripts/perf-sanity/aggregated"
DISAGG_CONFIG_SUBDIR = "tests/scripts/perf-sanity/disaggregated"

AGG_PREFIX_TOKEN = "aggr"
DISAGG_PREFIX_TOKEN = "disagg"

DISAGG_BENCHMARK_MODES = ("e2e", "gen_only")
AGGREGATED_DISAGG_YAML_MODES = ("ctx_only", "gen_only_no_context")
#: Modes whose config is read from :data:`DISAGG_CONFIG_SUBDIR`. Note that
#: the two ``AGGREGATED_DISAGG_YAML_MODES`` run the *aggregated* launch
#: path while reading a *disagg* config, so the folder follows the mode and
#: never the prefix.
DISAGG_CONFIG_MODES = DISAGG_BENCHMARK_MODES + AGGREGATED_DISAGG_YAML_MODES

TIME_BREAKDOWN_MODIFIER = "time_breakdown"
#: Closed vocabulary that makes ``<prefix>-<mode>[-<modifier>]-<stem>``
#: decidable: a third segment is a modifier only if it is in here, and no
#: config stem's first segment may collide with one.
TEST_ID_MODIFIERS = (TIME_BREAKDOWN_MODIFIER,)

RUNTIME_AGGREGATED = "aggregated"
RUNTIME_DISAGGREGATED = "disaggregated"


class TestCaseError(ValueError):
    """Raised when a test-case id cannot be read, or its config is missing.

    Deliberately an exception rather than a fallback. A perf-sanity id
    whose config cannot be found is the one failure that must never
    degrade quietly: the executor sizes an unresolved perf-sanity case at
    a single GPU, so a silent miss turns a multi-node case into a one-GPU
    job that runs, produces numbers, and means nothing.
    """


@dataclass(frozen=True)
class PerfSanityCase:
    """A parsed ``test_perf_sanity.py`` id.

    ``config_stem`` names a YAML; ``select_pattern`` names the
    ``server_configs`` entry within it, and is ``None`` both for the disagg
    shapes (which carry no entry) and for a plain aggregated id that omits
    it (which means "every entry").
    """

    name: str
    prefix: str
    config_stem: str
    select_pattern: str | None
    runtime: str
    benchmark_mode: str | None
    time_breakdown: bool

    family = FAMILY_PERF_SANITY

    @property
    def config_subdir(self) -> str:
        """Which of the two config folders this id's YAML lives in."""
        if self.benchmark_mode in DISAGG_CONFIG_MODES:
            return DISAGG_CONFIG_SUBDIR
        return AGG_CONFIG_SUBDIR

    @property
    def uploads_to_db(self) -> bool:
        """Whether the id asks the harness to publish to OpenSearch.

        A campaign measuring a candidate should not be writing to the
        series the regression was detected in, so this is worth reading
        rather than ignoring.
        """
        return "upload" in self.prefix


@dataclass(frozen=True)
class PerfCase:
    """A parsed ``test_perf.py`` id.

    ``labels`` is the id's ``-``-separated tail in order, and ``knobs``
    maps the ``key:value`` ones to their raw string values. Values are left
    as text: this module reports what the id pins, and interpreting a knob
    is the harness's job.
    """

    name: str
    labels: tuple[str, ...]
    knobs: dict[str, str]

    family = FAMILY_PERF


TestCase = PerfSanityCase | PerfCase


def _split_node_id(name: str) -> tuple[str, str]:
    """Return ``(module_basename, bracketed_params)`` for a pytest node id.

    Accepts any path to the module so a caller may pass the id exactly as
    CI, a regression report or a waive list spells it.
    """
    text = name.strip()
    if "::" not in text:
        raise TestCaseError(f"not a pytest node id (expected '<path>::<func>[<params>]'): {name!r}")

    path_part, _, rest = text.partition("::")
    module = Path(path_part).name
    if "[" not in rest or not rest.endswith("]"):
        raise TestCaseError(f"test-case id carries no '[parameters]': {name!r}")
    params = rest[rest.index("[") + 1 : -1]
    if not params:
        raise TestCaseError(f"test-case id has empty '[parameters]': {name!r}")
    return module, params


def _split_modifiers(rest: list[str], name: str) -> tuple[bool, str]:
    """Peel the optional modifier segment off the front of a config stem."""
    time_breakdown = bool(rest) and rest[0] in TEST_ID_MODIFIERS
    if time_breakdown:
        rest = rest[1:]
    if not rest:
        raise TestCaseError(f"test-case id has a modifier but no config: {name!r}")
    return time_breakdown, "-".join(rest)


def _parse_perf_sanity(name: str, params: str) -> PerfSanityCase:
    labels = params.split("-")
    if len(labels) <= 1:
        raise TestCaseError(f"perf-sanity id must name a config: {name!r}")

    prefix = labels[0]
    if DISAGG_PREFIX_TOKEN in prefix:
        if len(labels) <= 2:
            raise TestCaseError(f"disagg id must carry a benchmark mode and a config: {name!r}")
        benchmark_mode = labels[1]
        if benchmark_mode not in DISAGG_BENCHMARK_MODES:
            expected = ", ".join(repr(mode) for mode in DISAGG_BENCHMARK_MODES)
            raise TestCaseError(
                f"invalid disagg benchmark mode {benchmark_mode!r} in {name!r}; "
                f"expected one of {expected}"
            )
        time_breakdown, config_stem = _split_modifiers(labels[2:], name)
        return PerfSanityCase(
            name=name,
            prefix=prefix,
            config_stem=config_stem,
            select_pattern=None,
            runtime=RUNTIME_DISAGGREGATED,
            benchmark_mode=benchmark_mode,
            time_breakdown=time_breakdown,
        )

    if AGG_PREFIX_TOKEN in prefix:
        # A disagg config on the aggregated launch path keeps the aggr
        # prefix, so the mode segment is what distinguishes it.
        if len(labels) > 2 and labels[1] in AGGREGATED_DISAGG_YAML_MODES:
            time_breakdown, config_stem = _split_modifiers(labels[2:], name)
            return PerfSanityCase(
                name=name,
                prefix=prefix,
                config_stem=config_stem,
                select_pattern=None,
                runtime=RUNTIME_AGGREGATED,
                benchmark_mode=labels[1],
                time_breakdown=time_breakdown,
            )
        return PerfSanityCase(
            name=name,
            prefix=prefix,
            config_stem=labels[1],
            select_pattern="-".join(labels[2:]) if len(labels) > 2 else None,
            runtime=RUNTIME_AGGREGATED,
            benchmark_mode=None,
            time_breakdown=False,
        )

    raise TestCaseError(f"invalid perf-sanity id prefix {prefix!r} in {name!r}")


def _parse_perf(name: str, params: str) -> PerfCase:
    labels = tuple(params.split("-"))
    knobs = {}
    for label in labels:
        key, sep, value = label.partition(":")
        if sep:
            knobs[key] = value
    return PerfCase(name=name, labels=labels, knobs=knobs)


def parse(name: str) -> TestCase:
    """Read a test-case id, or raise :class:`TestCaseError`.

    The family is decided by the module the node id names, not by the
    shape of the parameters: the two families' parameter grammars are
    unrelated, and guessing between them would turn a typo into a
    confident parse of the wrong thing.
    """
    module, params = _split_node_id(name)
    if module == _PERF_SANITY_MODULE:
        return _parse_perf_sanity(name, params)
    if module == _PERF_MODULE:
        return _parse_perf(name, params)
    raise TestCaseError(
        f"unsupported test module {module!r} in {name!r}; expected "
        f"{_PERF_SANITY_MODULE} or {_PERF_MODULE}"
    )


def resolve_config(case: TestCase, repo_dir: str | Path) -> Path:
    """Absolute path to ``case``'s perf-sanity config YAML.

    Resolved against ``repo_dir`` — the checkout under test — because the
    perf-sanity harness and its configs are part of the code being
    measured, not neutral tooling beside it. Reading them from anywhere
    else measures a different test than the one named.

    There is deliberately **no second search location**: a missing config
    raises. The one previous attempt at a fallback folder in this area
    silently emptied a real reproduce run.
    """
    if not isinstance(case, PerfSanityCase):
        raise TestCaseError(
            f"only {FAMILY_PERF_SANITY} ids resolve to a config file; "
            f"{case.name!r} generates its configuration from the id itself"
        )
    stem = case.config_stem
    filename = stem if stem.endswith(".yaml") else f"{stem}.yaml"
    path = Path(repo_dir) / case.config_subdir / filename
    if not path.is_file():
        raise TestCaseError(
            f"perf-sanity config not found for {case.name!r}: {path} does not "
            f"exist. The id resolves to {case.config_subdir}/{filename} under "
            f"the checkout under test; check the id against that checkout "
            f"rather than against main."
        )
    return path


def pinned_labels(case: TestCase) -> frozenset[str]:
    """Knob names the id itself fixes, which a fix should leave alone.

    Changing one of these changes what is measured rather than how it
    performs, so a candidate that moves the number by moving a pinned knob
    has not addressed the regression the id names.

    The two families differ in how firmly this holds, and the difference
    is worth stating. For :class:`PerfCase` it is mechanical: ``to_string``
    and ``load_from_str`` are inverses, so a knob in the id *is* the id,
    and editing it produces a different test case — one that need not even
    exist in the test lists. For :class:`PerfSanityCase` it is a naming
    convention: the harness parses only the config stem and the entry
    selection, so tokens like ``con128`` inside the stem are not read at
    all. Editing the matching YAML field is therefore mechanically
    harmless and still wrong, because it leaves the id describing a run it
    no longer performs and breaks comparability with the stored series
    keyed to that id.
    """
    if isinstance(case, PerfCase):
        return frozenset(case.knobs)
    pinned = {"config_stem"}
    if case.select_pattern is not None:
        pinned.add("select_pattern")
    if case.benchmark_mode is not None:
        pinned.add("benchmark_mode")
    return frozenset(pinned)


__all__ = [
    "AGG_CONFIG_SUBDIR",
    "AGGREGATED_DISAGG_YAML_MODES",
    "DISAGG_BENCHMARK_MODES",
    "DISAGG_CONFIG_MODES",
    "DISAGG_CONFIG_SUBDIR",
    "FAMILY_PERF",
    "FAMILY_PERF_SANITY",
    "GRAMMAR_SOURCE",
    "RUNTIME_AGGREGATED",
    "RUNTIME_DISAGGREGATED",
    "TEST_ID_MODIFIERS",
    "PerfCase",
    "PerfSanityCase",
    "TestCase",
    "TestCaseError",
    "parse",
    "pinned_labels",
    "resolve_config",
]
