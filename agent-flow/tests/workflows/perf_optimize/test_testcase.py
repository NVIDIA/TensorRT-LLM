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
"""Tests for reading a test-case id as a workload definition.

Two properties carry the module. First, the mirrored perf-sanity grammar
agrees with its owner on every shape that owner documents — these tests are
what a grammar drift shows up as. Second, an id that cannot be resolved
raises instead of degrading, because the executor sizes an unresolved
perf-sanity case at one GPU and a silent miss would run a multi-node case
on a single device and publish the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.perf_optimize import testcase

PERF_SANITY = "tests/integration/defs/perf/test_perf_sanity.py::test_e2e"
PERF = "tests/integration/defs/perf/test_perf.py::test_perf"


def _sanity(params: str) -> testcase.PerfSanityCase:
    case = testcase.parse(f"{PERF_SANITY}[{params}]")
    assert isinstance(case, testcase.PerfSanityCase)
    return case


def _perf(params: str) -> testcase.PerfCase:
    case = testcase.parse(f"{PERF}[{params}]")
    assert isinstance(case, testcase.PerfCase)
    return case


# --------------------------------------------------------------------------- #
# The shapes the owning docstring documents
# --------------------------------------------------------------------------- #


def test_disagg_id_carries_mode_and_stem():
    case = _sanity("disagg-e2e-b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4")
    assert case.family == testcase.FAMILY_PERF_SANITY
    assert case.benchmark_mode == "e2e"
    assert case.runtime == testcase.RUNTIME_DISAGGREGATED
    # The stem keeps every remaining '-' segment: a disagg id has no entry
    # selection, so nothing after the mode is anything else.
    assert case.config_stem == "b200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4"
    assert case.select_pattern is None
    assert case.time_breakdown is False


def test_disagg_upload_prefix_is_recognised_and_reported():
    case = _sanity("disagg_upload-gen_only-b200_cfg")
    assert case.benchmark_mode == "gen_only"
    assert case.runtime == testcase.RUNTIME_DISAGGREGATED
    assert case.uploads_to_db is True
    assert _sanity("disagg-gen_only-b200_cfg").uploads_to_db is False


@pytest.mark.parametrize("mode", testcase.AGGREGATED_DISAGG_YAML_MODES)
def test_aggregated_path_with_disagg_config(mode):
    """These keep the aggr prefix but read a disagg config.

    The folder has to follow the *mode*; following the prefix would send
    both of these to the aggregated folder and miss every time.
    """
    case = _sanity(f"aggr-{mode}-b200_cfg_with-dashes")
    assert case.runtime == testcase.RUNTIME_AGGREGATED
    assert case.benchmark_mode == mode
    assert case.config_stem == "b200_cfg_with-dashes"
    assert case.select_pattern is None
    assert case.config_subdir == testcase.DISAGG_CONFIG_SUBDIR


def test_regular_aggr_without_entry_selection():
    case = _sanity("aggr-deepseek_r1_fp4_v2_2_nodes_blackwell")
    assert case.runtime == testcase.RUNTIME_AGGREGATED
    assert case.benchmark_mode is None
    assert case.config_stem == "deepseek_r1_fp4_v2_2_nodes_blackwell"
    assert case.select_pattern is None
    assert case.config_subdir == testcase.AGG_CONFIG_SUBDIR


def test_regular_aggr_with_entry_selection():
    case = _sanity("aggr-deepseek_r1_fp4_v2_2_nodes_blackwell-r1_fp4_v2_dep16_mtp1_1k1k")
    assert case.config_stem == "deepseek_r1_fp4_v2_2_nodes_blackwell"
    assert case.select_pattern == "r1_fp4_v2_dep16_mtp1_1k1k"


def test_regular_aggr_entry_selection_may_contain_dashes():
    case = _sanity("aggr-cfg-entry-with-dashes")
    assert case.config_stem == "cfg"
    assert case.select_pattern == "entry-with-dashes"


@pytest.mark.parametrize(
    "params,stem",
    [
        ("disagg-e2e-time_breakdown-b200_cfg", "b200_cfg"),
        ("aggr-ctx_only-time_breakdown-b200_cfg", "b200_cfg"),
    ],
)
def test_modifier_is_peeled_off_the_stem(params, stem):
    case = _sanity(params)
    assert case.time_breakdown is True
    assert case.config_stem == stem


def test_a_stem_segment_is_not_mistaken_for_a_modifier():
    case = _sanity("disagg-e2e-b200_cfg")
    assert case.time_breakdown is False
    assert case.config_stem == "b200_cfg"


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "tests/integration/defs/perf/test_perf_sanity.py",  # no ::func
        f"{PERF_SANITY}",  # no [params]
        f"{PERF_SANITY}[]",  # empty params
        f"{PERF_SANITY}[aggr",  # unterminated
    ],
)
def test_malformed_node_ids_raise(name):
    with pytest.raises(testcase.TestCaseError):
        testcase.parse(name)


def test_unsupported_module_raises():
    with pytest.raises(testcase.TestCaseError, match="unsupported test module"):
        testcase.parse("tests/unittest/test_llm_api.py::test_something[x]")


@pytest.mark.parametrize(
    "params,match",
    [
        ("nonsense-cfg", "prefix"),
        ("aggr", "must name a config"),
        ("disagg-cfg", "benchmark mode and a config"),
        ("disagg-not_a_mode-cfg", "invalid disagg benchmark mode"),
        ("disagg-e2e-time_breakdown", "modifier but no config"),
    ],
)
def test_grammar_violations_raise(params, match):
    with pytest.raises(testcase.TestCaseError, match=match):
        testcase.parse(f"{PERF_SANITY}[{params}]")


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("metadata: {}\n", encoding="utf-8")


def test_resolve_config_finds_the_aggregated_yaml(tmp_path):
    _write(tmp_path / testcase.AGG_CONFIG_SUBDIR / "cfg.yaml")
    case = _sanity("aggr-cfg-entry")
    assert testcase.resolve_config(case, tmp_path) == (
        tmp_path / testcase.AGG_CONFIG_SUBDIR / "cfg.yaml"
    )


def test_resolve_config_follows_the_mode_not_the_prefix(tmp_path):
    """An aggr-prefixed ctx_only id must find its YAML in the disagg folder."""
    _write(tmp_path / testcase.DISAGG_CONFIG_SUBDIR / "cfg.yaml")
    case = _sanity("aggr-ctx_only-cfg")
    assert testcase.resolve_config(case, tmp_path).parent.name == "disaggregated"


def test_resolve_config_raises_rather_than_falling_back(tmp_path):
    """The config exists, but in the other folder: this must still raise.

    A fallback search is what makes an unresolved case look resolved, and
    the executor's own default then sizes it at a single GPU.
    """
    _write(tmp_path / testcase.AGG_CONFIG_SUBDIR / "cfg.yaml")
    case = _sanity("disagg-e2e-cfg")
    with pytest.raises(testcase.TestCaseError, match="config not found"):
        testcase.resolve_config(case, tmp_path)


def test_resolve_config_tolerates_a_stem_that_already_has_the_suffix(tmp_path):
    _write(tmp_path / testcase.AGG_CONFIG_SUBDIR / "cfg.yaml")
    case = _sanity("aggr-cfg.yaml")
    assert testcase.resolve_config(case, tmp_path).name == "cfg.yaml"


def test_resolve_config_rejects_the_perf_family():
    case = _perf("glm_5_fp8-bench-pytorch-float8-maxbs:512")
    with pytest.raises(testcase.TestCaseError, match="generates its configuration"):
        testcase.resolve_config(case, "/nonexistent")


def test_resolution_against_the_real_checkout():
    """The mirror agrees with a config that actually exists in this repo.

    Skipped when the suite runs somewhere without the TensorRT-LLM tree
    around it, so vendoring agent-flow elsewhere does not fail here.
    """
    repo = Path(__file__).resolve().parents[4]
    folder = repo / testcase.DISAGG_CONFIG_SUBDIR
    if not folder.is_dir():
        pytest.skip("TensorRT-LLM perf-sanity configs not present")
    configs = sorted(folder.glob("*.yaml"))
    if not configs:
        pytest.skip("no disagg perf-sanity configs to resolve against")
    case = _sanity(f"disagg-e2e-{configs[0].stem}")
    assert testcase.resolve_config(case, repo) == configs[0]


# --------------------------------------------------------------------------- #
# The QA perf family, and what each id pins
# --------------------------------------------------------------------------- #


def test_perf_id_knobs_are_read_as_text():
    case = _perf(
        "glm_5_fp8-bench-pytorch-float8-maxbs:512-maxnt:2048-input_output_len:500,2000-ep:8-gpus:8"
    )
    assert case.family == testcase.FAMILY_PERF
    assert case.knobs["maxbs"] == "512"
    assert case.knobs["maxnt"] == "2048"
    assert case.knobs["ep"] == "8"
    assert case.knobs["gpus"] == "8"
    # A comma-bearing value survives intact; interpreting it is the
    # harness's job, not this module's.
    assert case.knobs["input_output_len"] == "500,2000"
    # Flag-shaped labels carry no value and so pin nothing by name.
    assert "bench" not in case.knobs
    assert "bench" in case.labels


def test_perf_flag_label_without_a_value():
    case = _perf("model-bench-mp-maxbs:64")
    assert "mp" in case.labels
    assert "mp" not in case.knobs


def test_pinned_labels_for_the_perf_family_are_the_id_knobs():
    case = _perf("glm_5_fp8-bench-maxbs:512-gpus:8")
    assert testcase.pinned_labels(case) == frozenset({"maxbs", "gpus"})


def test_pinned_labels_omit_a_knob_the_id_does_not_spell_out():
    """An id without ``maxbs`` leaves max batch size to a class default.

    That default is editable without changing the id, which is why the
    pinned set is derived per id rather than per family.
    """
    case = _perf("glm_5_fp8-bench-gpus:8")
    assert "maxbs" not in testcase.pinned_labels(case)


def test_pinned_labels_for_perf_sanity_cover_stem_and_selection():
    case = _sanity("aggr-cfg-entry")
    assert testcase.pinned_labels(case) == frozenset({"config_stem", "select_pattern"})
    assert testcase.pinned_labels(_sanity("aggr-cfg")) == frozenset({"config_stem"})
    assert testcase.pinned_labels(_sanity("disagg-e2e-cfg")) == frozenset(
        {"config_stem", "benchmark_mode"}
    )
