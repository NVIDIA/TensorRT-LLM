# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPOT reporting and regression comparisons use latency semantics."""

import ast
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only
_PERF_DIR = Path(__file__).resolve().parents[2] / "integration" / "defs" / "perf"
_TPOT_METRICS = ("d_mean_tpot", "d_median_tpot", "d_p99_tpot")


@pytest.fixture
def regression_utils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the real comparison code without GPU or OpenSearch dependencies."""
    alternative = ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    db_utils = ModuleType("defs.perf.open_search_db_utils")
    db_utils.add_id = lambda *args, **kwargs: None
    db_utils.get_history_data = lambda *args, **kwargs: None
    db_utils.post_new_perf_data = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "defs.trt_test_alternative", alternative)
    monkeypatch.setitem(sys.modules, "defs.perf.open_search_db_utils", db_utils)
    spec = importlib.util.spec_from_file_location(
        "defs.perf.perf_regression_utils", _PERF_DIR / "perf_regression_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def metric_lists() -> dict[str, list[str]]:
    """Read the harness lists without importing its serving dependencies."""
    # The unrelated time-breakdown metrics are not exercised by these tests.
    namespace = {"TIME_BREAKDOWN_METRICS": ()}
    names = {"MAXIMIZE_METRICS", "MINIMIZE_METRICS", "REGRESSION_METRICS"}
    tree = ast.parse((_PERF_DIR / "test_perf_sanity.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "test_perf_sanity", "exec"),
                namespace,
            )
    return {name: namespace[name] for name in names}


@pytest.mark.parametrize("metric", _TPOT_METRICS)
@pytest.mark.parametrize("gate_tpot", [False, True])
@pytest.mark.parametrize(
    "value, expected_diff, is_slowdown",
    [(15.24, -150.6578947368421, True), (3.04, 50.0, False)],
)
def test_tpot_report_and_gate(
    regression_utils: ModuleType,
    metric_lists: dict[str, list[str]],
    metric: str,
    gate_tpot: bool,
    value: float,
    expected_diff: float,
    is_slowdown: bool,
) -> None:
    """Honor a stored baseline while applying latency signs and the selected gate."""
    data = {0: {metric: value}}
    regression_utils.prepare_regressive_test_cases(
        latest_history_data_dict={},
        latest_baseline_threshold_dict={0: {f"d_baseline_{metric[2:]}": 6.08}},
        history_data_dict={},
        new_data_dict=data,
        maximize_metrics=metric_lists["MAXIMIZE_METRICS"],
        minimize_metrics=metric_lists["MINIMIZE_METRICS"],
        regression_metrics=[metric] if gate_tpot else metric_lists["REGRESSION_METRICS"],
    )
    assert f"diff={expected_diff:+.2f}%" in data[0]["s_regression_info"]
    assert data[0]["b_is_regression"] is (gate_tpot and is_slowdown)


@pytest.mark.parametrize("metric", _TPOT_METRICS)
def test_tpot_baseline_uses_low_latency_percentile(
    regression_utils: ModuleType, metric_lists: dict[str, list[str]], metric: str
) -> None:
    """Compute P5 from historical latency measurements, including older samples."""
    history = [
        {"ts_created": "2026-10-01", metric: 6.0},
        {"ts_created": "2026-10-02", metric: 12.0},
        {"ts_created": "2026-10-03", metric: 18.0},
    ]
    baseline = regression_utils.calculate_baseline_metrics(
        history, None, metric_lists["MAXIMIZE_METRICS"], metric_lists["MINIMIZE_METRICS"]
    )
    # The trailing means are 6, 9, 12; their P5 is 6.3 (P95 would be 11.7).
    assert baseline[metric] == pytest.approx(6.3)


@pytest.mark.parametrize("metric", _TPOT_METRICS)
def test_calculated_tpot_baseline_is_not_persisted_as_an_override(
    regression_utils: ModuleType,
    metric_lists: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
) -> None:
    """Reclassifying old measurements recomputes P5 instead of inheriting a calculated P95."""
    history = [
        {"ts_created": "2026-10-01", metric: 6.0},
        {"ts_created": "2026-10-02", metric: 12.0},
        {"ts_created": "2026-10-03", metric: 18.0},
    ]
    uploaded = []
    monkeypatch.setattr(regression_utils, "get_job_info", lambda: {"b_is_post_merge": True})
    monkeypatch.setattr(
        regression_utils, "post_new_perf_data", lambda payload: uploaded.append(deepcopy(payload))
    )
    stored_fields = {}
    for maximize, expected_baseline in ((True, 11.7), (False, 6.3)):
        monkeypatch.setattr(
            regression_utils,
            "get_history_data",
            lambda *args: ({}, {0: stored_fields}, {0: history}),
        )
        data = {0: {metric: 9.0}}
        regression_utils.process_and_upload_test_results(
            new_data_dict=data,
            match_keys=[],
            maximize_metrics=[metric] if maximize else metric_lists["MAXIMIZE_METRICS"],
            minimize_metrics=[] if maximize else metric_lists["MINIMIZE_METRICS"],
            regression_metrics=metric_lists["REGRESSION_METRICS"] if not maximize else [],
        )
        assert uploaded[-1] == data
        uploaded_record = uploaded[-1][0]
        assert f"baseline={expected_baseline:.4f}" in uploaded_record["s_regression_info"]
        assert uploaded_record["b_is_regression"] is False
        stored_fields = {
            key: value for key, value in uploaded_record.items() if key.startswith("d_baseline_")
        }
        assert not stored_fields
    assert "diff=-42.86%" in uploaded[-1][0]["s_regression_info"]
