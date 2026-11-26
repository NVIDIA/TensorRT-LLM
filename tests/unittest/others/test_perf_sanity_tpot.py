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
"""Latency reporting and regression comparisons use latency semantics."""

import ast
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

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


@pytest.fixture
def perf_sanity_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[Callable[..., tuple[list[dict], RuntimeError | None]]]:
    """Run test_e2e with real parsing, history matching and verdicts, without servers."""
    repo_root = _PERF_DIR.parents[3]
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.delenv("PERF_BASELINE_BRANCH", raising=False)
    monkeypatch.delenv("BOLT_ITER_MULT", raising=False)
    monkeypatch.setenv("AGG_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("DISAGG_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("OPEN_SEARCH_DB_BASE_URL", "https://example.test/unused")
    monkeypatch.setenv("DISAGG_SERVING_TYPE", "BENCHMARK")
    monkeypatch.setenv("stageName", "")

    def unused_boundary(*args, **kwargs) -> None:
        raise AssertionError("The CPU verdict test must not start native execution")

    definitions = ModuleType("defs")
    definitions.__path__ = [str(_PERF_DIR.parent)]
    performance = ModuleType("defs.perf")
    performance.__path__ = [str(_PERF_DIR)]
    configuration = ModuleType("defs.conftest")
    configuration.get_llm_root = lambda: str(repo_root)
    configuration.get_device_count = unused_boundary
    configuration.llm_models_root = unused_boundary
    common = ModuleType("defs.common")
    common.wait_for_reported_addr = unused_boundary
    alternative = ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    native_utils = ModuleType("tensorrt_llm._torch.disaggregation.native.utils")
    native_utils.get_local_ip = lambda: "127.0.0.1"
    opensearch = ModuleType("jenkins.scripts.open_search_db")
    opensearch.PERF_SANITY_PROJECT_NAME = "perf-tests"
    opensearch.OpenSearchDB = SimpleNamespace(
        add_id_of_json=lambda data: None,
        queryPerfDataFromOpenSearchDB=unused_boundary,
        postToOpenSearchDB=unused_boundary,
    )
    for module in (
        definitions,
        performance,
        configuration,
        common,
        alternative,
        native_utils,
        opensearch,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    module_names = [
        f"defs.perf.{name}"
        for name in (
            "_model_paths",
            "open_search_db_utils",
            "perf_regression_utils",
            "time_breakdown_metrics",
            "warmup_pass",
            "test_perf_sanity",
        )
    ]
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        harness = importlib.import_module("defs.perf.test_perf_sanity")
        database = importlib.import_module("defs.perf.open_search_db_utils")
        monkeypatch.setattr(database, "time", SimpleNamespace(time=lambda: 1767398400.0))
        monkeypatch.setattr(
            harness.subprocess, "check_output", lambda *args, **kwargs: "Product Name : B200\n"
        )

        def replay(
            history: list[dict],
            *,
            ttft: float = 125.0,
            throughput: float = 100.0,
            is_post_merge: bool = False,
            config: dict | None = None,
            test_case: str = "aggr_upload-latency_control",
        ) -> tuple[list[dict], RuntimeError | None]:
            configuration_data = config or {
                "metadata": {"model_name": "latency-control"},
                "hardware": {"gpus_per_node": 1},
                "server_configs": [
                    {"name": "plain", "client_configs": [{"name": "con1", "concurrency": 1}]}
                ],
            }
            (tmp_path / "latency_control.yaml").write_text(yaml.safe_dump(configuration_data))
            merge_type = "PostMerge" if is_post_merge else "MergeRequest_PR"
            monkeypatch.setenv(
                "globalVars",
                json.dumps(
                    {
                        "build_branch": "main" if is_post_merge else "latency-check",
                        "action_info": {
                            "parents": [{"url": f"https://example.test/L0_{merge_type}/1/"}]
                        },
                    }
                ),
            )
            output = "\n".join(
                [
                    "Successful requests: 1",
                    "Failed requests: 0",
                    "Request throughput (req/s): 1.0",
                    f"Output token throughput (tok/s): {throughput}",
                    "Total Token throughput (tok/s): 1000.0",
                    "User throughput (tok/s): 1.0",
                    f"Mean TTFT (ms): {ttft}",
                    "Median TTFT (ms): N/A",
                    "P99 TTFT (ms): N/A",
                    "Mean ITL (ms): N/A",
                    "Median ITL (ms): N/A",
                    "P99 ITL (ms): N/A",
                    "Mean TPOT (ms): N/A",
                    "Median TPOT (ms): N/A",
                    "P99 TPOT (ms): N/A",
                    "Mean E2EL (ms): 1.0",
                    "Median E2EL (ms): 1.0",
                    "P99 E2EL (ms): 1.0",
                    "Mean Avg Decoded Tokens per Iter: 2.0",
                ]
            )

            def run_cmd(commands, server_idx: int) -> list[str]:
                return [output for _ in commands.client_cmds[server_idx]]

            monkeypatch.setattr(harness.AggrTestCmds, "run_cmd", run_cmd)
            monkeypatch.setattr(harness.DisaggTestCmds, "run_cmd", run_cmd)
            uploaded = []
            monkeypatch.setattr(
                database.OpenSearchDB,
                "queryPerfDataFromOpenSearchDB",
                lambda *args, **kwargs: deepcopy(history),
            )

            def post_to_database(values: list[dict], project: str) -> bool:
                uploaded.extend(deepcopy(values))
                return True

            monkeypatch.setattr(database.OpenSearchDB, "postToOpenSearchDB", post_to_database)
            regression_error = None
            try:
                harness.test_e2e(str(tmp_path / "output"), test_case)
            except RuntimeError as error:
                regression_error = error
            assert uploaded, f"test_e2e did not reach the upload/verdict path: {regression_error}"
            return uploaded, regression_error

        yield replay
    finally:
        for name in module_names:
            sys.modules.pop(name, None)


def _ttft_history(**overrides: object) -> dict:
    record = {
        "@timestamp": "2026-01-02T00:00:00.000Z",
        "ts_created": 1767312000000,
        "b_is_post_merge": True,
        "b_is_valid": True,
        "b_is_baseline": False,
        "s_test_case_name": "plain-con1",
        "s_gpu_type": "b200",
        "s_runtime": "aggr_server",
        "s_branch": "main",
        "d_mean_ttft": 100.0,
        "d_token_throughput": 100.0,
        "d_total_token_throughput": 1000.0,
        "d_baseline_mean_ttft": 100.0,
        "d_baseline_token_throughput": 100.0,
        "d_baseline_total_token_throughput": 1000.0,
    }
    record.update(overrides)
    return record


def _metric_comparison(record: dict, metric: str) -> tuple[float, float]:
    comparison = next(
        (
            line
            for line in record["s_regression_info"].splitlines()
            if line.strip().startswith(f"{metric}:")
        ),
        None,
    )
    assert comparison is not None, f"Missing matched comparison for {metric}"
    fields = dict(field.split("=", 1) for field in comparison.split()[1:])
    return float(fields["value"]), float(fields["baseline"])


@pytest.mark.parametrize("is_post_merge", [False, True], ids=["pre-merge", "post-merge"])
def test_ttft_caller_default_tolerances(perf_sanity_caller: Callable, is_post_merge: bool) -> None:
    """An 8% TTFT increase only crosses the default post-merge threshold."""
    uploaded, error = perf_sanity_caller([_ttft_history()], ttft=108.0, is_post_merge=is_post_merge)
    assert error is None
    assert uploaded[0]["b_is_regression"] is is_post_merge


def test_ttft_caller_configured_tolerance(perf_sanity_caller: Callable) -> None:
    """The selected latency gate honors the case's stored pre-merge tolerance."""
    uploaded, error = perf_sanity_caller([_ttft_history(d_threshold_pre_merge_mean_ttft=0.3)])
    assert error is None
    assert uploaded[0]["b_is_regression"] is False


@pytest.mark.parametrize("has_ttft_history", [True, False], ids=["healthy-history", "no-history"])
def test_ttft_caller_without_stored_override(
    perf_sanity_caller: Callable, has_ttft_history: bool
) -> None:
    """Only absent TTFT measurements, not an absent override, leave TTFT ungated."""
    history = _ttft_history()
    history.pop("d_baseline_mean_ttft")
    if not has_ttft_history:
        history.pop("d_mean_ttft")
    uploaded, error = perf_sanity_caller([history])
    assert uploaded[0]["b_is_regression"] is has_ttft_history
    assert (error is not None) is has_ttft_history
    if has_ttft_history:
        assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))


def test_ttft_caller_selects_each_client(perf_sanity_caller: Callable) -> None:
    """Only plain single-node concurrency-one clients gain the latency gate."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 4},
        "server_configs": [
            {
                "name": "plain",
                "tensor_parallel_size": 4,
                "client_configs": [
                    {"name": "con1", "concurrency": 1},
                    {"name": "con2", "concurrency": 2},
                ],
            },
            {
                "name": "partial-node",
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
            {
                "name": "speculative",
                "tensor_parallel_size": 4,
                "speculative_config": {"decoding_type": "MTP"},
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
            {
                "name": "multi-node",
                "tensor_parallel_size": 8,
                "client_configs": [{"name": "con1", "concurrency": 1}],
            },
        ],
    }
    expected = {
        "plain-con1": True,
        "plain-con2": False,
        "partial-node-con1": True,
        "speculative-con1": False,
        "multi-node-con1": False,
    }
    history = [
        _ttft_history(
            s_test_case_name=name,
            s_runtime="multi_node_aggr_server"
            if name.startswith(("multi-node", "partial-node"))
            else "aggr_server",
        )
        for name in expected
    ]
    uploaded, error = perf_sanity_caller(history, config=config)
    for row in uploaded:
        assert _metric_comparison(row, "d_mean_ttft") == pytest.approx((125.0, 100.0))
    assert {row["s_test_case_name"]: row["b_is_regression"] for row in uploaded} == expected
    assert error is not None
    assert "Regression in plain-con1:" in str(error)


@pytest.mark.parametrize("mode", ["e2e", "ctx_only"])
def test_ttft_caller_preserves_other_modes(perf_sanity_caller: Callable, mode: str) -> None:
    """Disaggregated e2e and context-only cases retain their throughput gates."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1, "num_ctx_servers": 1, "num_gen_servers": 1},
        "benchmark": {"concurrency_list": [1], "input_length": 16, "output_length": 8},
        "worker_config": {"ctx": {}, "gen": {}},
    }
    prefix = "disagg_upload" if mode == "e2e" else "aggr_upload"
    history = _ttft_history(
        s_test_case_name=f"{mode}-latency_control-con1_iter1_isl16_osl{8 if mode == 'e2e' else 1}",
        s_runtime="multi_node_disagg_server" if mode == "e2e" else "aggr_server",
    )
    uploaded, error = perf_sanity_caller(
        [history], config=config, test_case=f"{prefix}-{mode}-latency_control"
    )
    assert error is None
    assert uploaded[0]["b_is_regression"] is False
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((125.0, 100.0))


def test_ttft_caller_preserves_legacy_exemption(perf_sanity_caller: Callable) -> None:
    """An excluded speculative client retains the whole-case mainline exemption."""
    config = {
        "metadata": {"model_name": "latency-control"},
        "hardware": {"gpus_per_node": 1},
        "server_configs": [
            {
                "name": "speculative",
                "speculative_config": {"decoding_type": "MTP"},
                "client_configs": [{"name": "con1", "concurrency": 1}],
            }
        ],
    }
    history = _ttft_history(
        s_test_case_name="speculative-con1",
        d_al=1.5,
        d_baseline_al=2.0,
        b_is_regression=True,
    )
    uploaded, error = perf_sanity_caller([history], ttft=100.0, throughput=75.0, config=config)
    assert uploaded[0]["b_is_regression"] is True
    assert _metric_comparison(uploaded[0], "d_mean_ttft") == pytest.approx((100.0, 100.0))
    assert _metric_comparison(uploaded[0], "d_al") == pytest.approx((2.0, 2.0))
    assert _metric_comparison(uploaded[0], "d_token_throughput") == pytest.approx((75.0, 100.0))
    assert error is None
    assert "Not failing this stage" in uploaded[0]["s_regression_info"]


@pytest.mark.parametrize(
    "main_ttft,main_throughput,ttft,throughput,should_fail",
    [
        pytest.param(125.0, 100.0, 100.0, 75.0, True, id="recovered-ttft-new-throughput"),
        pytest.param(125.0, 100.0, 125.0, 75.0, True, id="inherited-ttft-new-throughput"),
        pytest.param(100.0, 75.0, 125.0, 75.0, True, id="inherited-throughput-new-ttft"),
        pytest.param(125.0, 100.0, 125.0, 100.0, False, id="inherited-ttft"),
        pytest.param(100.0, 75.0, 100.0, 75.0, False, id="inherited-throughput"),
    ],
)
def test_ttft_caller_keeps_independent_regressions(
    perf_sanity_caller: Callable,
    main_ttft: float,
    main_throughput: float,
    ttft: float,
    throughput: float,
    should_fail: bool,
) -> None:
    """Main only exempts a row when every current regressive metric already misses its gate."""
    latest = _ttft_history(d_mean_ttft=main_ttft, d_token_throughput=main_throughput)
    older = _ttft_history(**{"@timestamp": "2026-01-01T00:00:00.000Z", "ts_created": 1767225600000})
    uploaded, error = perf_sanity_caller([older, latest], ttft=ttft, throughput=throughput)
    assert uploaded[0]["b_is_regression"] is True
    assert (error is not None) is should_fail
    if should_fail:
        assert "Not failing this stage" not in str(error)
    else:
        assert "Not failing this stage" in uploaded[0]["s_regression_info"]


def test_per_case_regression_metrics_are_validated(
    regression_utils: ModuleType, metric_lists: dict[str, list[str]]
) -> None:
    """Per-case overrides must use the same metric definitions as default gates."""
    with pytest.raises(ValueError, match="d_unknown_metric"):
        regression_utils.process_and_upload_test_results(
            new_data_dict={0: {}},
            match_keys=[],
            maximize_metrics=metric_lists["MAXIMIZE_METRICS"],
            minimize_metrics=metric_lists["MINIMIZE_METRICS"],
            regression_metrics=metric_lists["REGRESSION_METRICS"],
            regression_metrics_by_cmd_idx={0: ["d_unknown_metric"]},
        )


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
