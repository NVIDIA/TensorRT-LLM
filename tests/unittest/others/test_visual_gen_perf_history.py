# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for VisualGen's client-timing baseline migration."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from test_common.visual_gen_perf_history import (
    VISUAL_GEN_METRIC_CONTRACT,
    transform_visual_gen_history,
)

pytestmark = pytest.mark.cpu_only
_PERF_DIR = Path(__file__).resolve().parents[3] / "tests/integration/defs/perf"
_CLIENT_COMMIT = "b" * 40
_SERVER_COMMIT = "a" * 40
_Modules = tuple[types.ModuleType, types.ModuleType, types.SimpleNamespace]


def _row(commit: str, value: float, timestamp: str, **fields: object) -> dict:
    return {
        "s_runtime": "visual_gen",
        "s_test_case_name": "flux2",
        "s_gpu_type": "b200",
        "s_commit": commit,
        "d_median_generation": value,
        "@timestamp": timestamp,
        "ts_created": timestamp,
        **fields,
    }


def _client_row(value: float, timestamp: str, **fields: object) -> dict:
    return _row(
        _CLIENT_COMMIT,
        value,
        timestamp,
        s_visual_gen_metric_contract=VISUAL_GEN_METRIC_CONTRACT,
        **fields,
    )


def test_old_window_is_excluded_even_with_a_new_date() -> None:
    record = _row(_SERVER_COMMIT, 0.8554, "2026-10-14T00:00:00Z")
    assert transform_visual_gen_history(record) is None


def test_unversioned_client_history_needs_explicit_backfill() -> None:
    record = _row(_CLIENT_COMMIT, 0.9433, "2026-09-14T00:00:00Z")
    assert transform_visual_gen_history(record) is None
    record["s_visual_gen_metric_contract"] = VISUAL_GEN_METRIC_CONTRACT
    assert transform_visual_gen_history(record) == record


@pytest.mark.parametrize("contract", [None, "", "server_generation", "v3_other"])
def test_missing_or_different_contract_is_excluded(contract: str | None) -> None:
    assert transform_visual_gen_history(_row("f" * 40, 0.94, "2026-10-14T00:00:00Z")) is None
    assert (
        transform_visual_gen_history(
            _row(
                _CLIENT_COMMIT,
                0.94,
                "2026-10-14T00:00:00Z",
                s_visual_gen_metric_contract=contract,
            )
        )
        is None
    )


def test_new_metadata_works_without_git_history() -> None:
    record = _row(
        "f" * 40,
        0.94,
        "2026-10-14T00:00:00Z",
        s_visual_gen_metric_contract=VISUAL_GEN_METRIC_CONTRACT,
    )
    assert transform_visual_gen_history(record) == record


def test_mixed_baselines_are_removed_but_thresholds_and_raw_records_survive() -> None:
    record = _client_row(
        0.9433,
        "2026-09-14T00:00:00Z",
        d_baseline_median_generation=0.8554,
        d_threshold_post_merge_median_generation=0.05,
        d_threshold_pre_merge_median_generation=0.1,
    )
    clean = transform_visual_gen_history(record)
    assert "d_baseline_median_generation" not in clean
    assert clean["d_threshold_post_merge_median_generation"] == 0.05
    assert clean["d_threshold_pre_merge_median_generation"] == 0.1
    assert record["d_baseline_median_generation"] == 0.8554


def test_other_runtimes_keep_their_baseline() -> None:
    record = {"s_runtime": "aggr_server", "d_baseline_token_throughput": 123.0}
    assert transform_visual_gen_history(record) == record


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> _Modules:
    """Load the real CPU logic while replacing only GPU/DB import dependencies."""
    defs = types.ModuleType("defs")
    defs.__path__ = []
    perf = types.ModuleType("defs.perf")
    perf.__path__ = []
    alternative = types.ModuleType("defs.trt_test_alternative")
    alternative.print_info = lambda *args, **kwargs: None
    alternative.print_warning = lambda *args, **kwargs: None
    search = types.ModuleType("jenkins.scripts.open_search_db")
    search.PERF_SANITY_PROJECT_NAME = "test-only"
    search.OpenSearchDB = types.SimpleNamespace(
        queryPerfDataFromOpenSearchDB=lambda *args, **kwargs: [],
        add_id_of_json=lambda record: None,
    )
    for name, module in {
        "defs": defs,
        "defs.perf": perf,
        "defs.trt_test_alternative": alternative,
        "jenkins.scripts.open_search_db": search,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    loaded = []
    for name in ("open_search_db_utils", "perf_regression_utils"):
        fullname = "defs.perf." + name
        spec = importlib.util.spec_from_file_location(fullname, _PERF_DIR / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded[0], loaded[1], search.OpenSearchDB


def test_policy_applies_before_latest_baseline_and_exemption_selection(modules: _Modules) -> None:
    db, _, search = modules
    old = _row(_SERVER_COMMIT, 0.8554, "2026-10-14T00:00:00Z", d_baseline_median_generation=0.8554)
    new = _client_row(
        0.9433,
        "2026-09-14T00:00:00Z",
        d_baseline_median_generation=0.8554,
        d_threshold_post_merge_median_generation=0.05,
    )
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [old, new]
    latest, baseline, history = db.get_history_data(
        {0: new},
        ["s_test_case_name", "s_gpu_type", "s_runtime"],
        {},
        history_transform=transform_visual_gen_history,
    )
    assert latest[0]["s_commit"] == _CLIENT_COMMIT
    assert [r["s_commit"] for r in history[0]] == [_CLIENT_COMMIT]
    assert baseline[0] == {"d_threshold_post_merge_median_generation": 0.05}


def test_default_history_lookup_is_unchanged(modules: _Modules) -> None:
    db, _, search = modules
    old = _row(_SERVER_COMMIT, 0.8554, "2026-10-14T00:00:00Z", d_baseline_median_generation=0.8554)
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [old]
    latest, baseline, history = db.get_history_data({0: old}, ["s_test_case_name"], {})
    assert latest[0] == old
    assert baseline[0] == {"d_baseline_median_generation": 0.8554}
    assert history[0] == [old]


def test_empty_compatible_history_and_db_failure_are_distinct(modules: _Modules) -> None:
    db, _, search = modules
    old = _row(_SERVER_COMMIT, 0.8554, "2026-10-14T00:00:00Z")
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [old]
    assert db.get_history_data(
        {0: old}, [], {}, history_transform=transform_visual_gen_history
    ) == ({0: None}, {0: None}, {0: []})
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: None
    assert db.get_history_data(
        {0: old}, [], {}, history_transform=transform_visual_gen_history
    ) == (None, None, None)


@pytest.mark.parametrize("post_merge", [False, True])
@pytest.mark.parametrize("old_value", [0.8554, 1.2])
def test_end_to_end_false_regression_is_removed_and_real_regression_remains(
    modules: _Modules, monkeypatch: pytest.MonkeyPatch, post_merge: bool, old_value: float
) -> None:
    _, pipeline, search = modules
    old = _row(_SERVER_COMMIT, old_value, "2026-10-14T00:00:00Z")
    good = _client_row(0.9433, "2026-09-14T00:00:00Z", d_baseline_median_generation=0.8554)
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [old, good]
    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": post_merge})
    for value, should_fail in [(0.9433, False), (1.2, True)]:
        current = _row(
            "f" * 40,
            value,
            "2026-10-14T00:00:00Z",
            s_visual_gen_metric_contract=VISUAL_GEN_METRIC_CONTRACT,
        )
        kwargs = dict(
            new_data_dict={0: current},
            match_keys=["s_test_case_name", "s_gpu_type", "s_runtime"],
            maximize_metrics=[],
            minimize_metrics=["d_median_generation"],
            regression_metrics=["d_median_generation"],
            upload_to_db=False,
            fail_on_regression=True,
            history_transform=transform_visual_gen_history,
        )
        if should_fail and not post_merge:
            with pytest.raises(RuntimeError, match="Regression"):
                pipeline.process_and_upload_test_results(**kwargs)
        else:
            pipeline.process_and_upload_test_results(**kwargs)
        assert current["b_is_regression"] == should_fail


@pytest.mark.parametrize("post_merge", [False, True])
def test_untagged_history_bootstraps_without_mutating_stored_results(
    modules: _Modules, monkeypatch: pytest.MonkeyPatch, post_merge: bool
) -> None:
    _, pipeline, search = modules
    old = _row(_SERVER_COMMIT, 0.8554, "2026-09-11T00:00:00Z")
    untagged_client = _row(_CLIENT_COMMIT, 0.9433, "2026-09-14T00:00:00Z")
    history = [old, untagged_client]
    originals = [dict(record) for record in history]
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: history
    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": post_merge})
    current = _client_row(0.9433, "2026-10-14T00:00:00Z")
    pipeline.process_and_upload_test_results(
        new_data_dict={0: current},
        match_keys=["s_test_case_name", "s_gpu_type", "s_runtime"],
        maximize_metrics=[],
        minimize_metrics=["d_median_generation"],
        regression_metrics=["d_median_generation"],
        upload_to_db=False,
        history_transform=transform_visual_gen_history,
    )
    assert current["b_is_regression"] is False
    assert "baseline=0.9433" in current["s_regression_info"]
    assert history == originals


def test_producer_stamps_metadata_without_changing_match_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = types.ModuleType("tensorrt_llm.commands.utils")
    commands.get_visual_gen_num_gpus = lambda config: 1
    monkeypatch.setitem(sys.modules, "tensorrt_llm.commands.utils", commands)
    spec = importlib.util.spec_from_file_location(
        "visual_gen_perf_utils", _PERF_DIR / "visual_gen_perf_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "extract_visual_gen_metrics", lambda data: {"d_request_throughput": 2}
    )
    row = module.build_visual_gen_db_entry(
        gpu_type="b200",
        model_name="flux2",
        server_name="flux2",
        server_config={},
        client_config={"backend": "openai-images", "num_inference_steps": 50, "max_concurrency": 1},
        result_data={},
    )
    assert row["s_visual_gen_metric_contract"] == VISUAL_GEN_METRIC_CONTRACT
    assert "s_visual_gen_metric_contract" not in module.get_visual_gen_match_keys()
