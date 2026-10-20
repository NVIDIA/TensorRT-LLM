# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for VisualGen's client-timing baseline migration."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from test_common import visual_gen_perf_history
from test_common.visual_gen_perf_history import (
    get_generation_timing_source,
    transform_visual_gen_history,
)

pytestmark = pytest.mark.cpu_only
_PERF_DIR = Path(__file__).resolve().parents[3] / "tests/integration/defs/perf"
_CLIENT_COMMIT = "b" * 40
_SERVER_COMMIT = "a" * 40
_Modules = tuple[types.ModuleType, types.ModuleType, types.SimpleNamespace]


def _benchmark_result(backend: str) -> dict:
    def latency(value: float) -> dict:
        return {"mean": value, "median": value, "percentiles": {"p90": value, "p99": value}}

    return {
        "backend": backend,
        "model": "test-model",
        "config": {"num_gpus": 1},
        "total_requests": 1,
        "completed": 1,
        "timings": {},
        "request_throughput": 2,
        "e2e_latency": latency(0.94),
        "gen_latency": latency(0.8),
    }


def _row(commit: str, value: float, timestamp: str, **fields: object) -> dict:
    return {
        "s_runtime": "visual_gen",
        "s_backend": "openai-images",
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
        s_generation_timing_source=(
            "client_gen_latency"
            if fields.get("s_backend", "openai-images") == "openai-videos"
            else "client_e2e_latency"
        ),
        **fields,
    )


def _entry(producer: types.ModuleType, backend: str) -> dict:
    return producer.build_visual_gen_db_entry(
        gpu_type="b200",
        model_name="flux2",
        server_name="flux2",
        server_config={},
        client_config={"backend": backend, "num_inference_steps": 50, "max_concurrency": 1},
        result_data=_benchmark_result(backend),
    )


@pytest.fixture
def pipeline_kwargs() -> dict:
    return dict(
        match_keys=["s_test_case_name", "s_gpu_type", "s_runtime", "s_backend"],
        maximize_metrics=[],
        minimize_metrics=["d_median_generation"],
        regression_metrics=["d_median_generation"],
        history_transform=transform_visual_gen_history,
    )


def test_untagged_client_history_needs_explicit_backfill() -> None:
    record = _row(_CLIENT_COMMIT, 0.9433, "2026-09-14T00:00:00Z")
    assert transform_visual_gen_history(record) is None
    record["s_generation_timing_source"] = "client_e2e_latency"
    assert transform_visual_gen_history(record) == record


@pytest.mark.parametrize("source", [None, "server_generation", "unknown_source"])
def test_missing_or_different_source_is_excluded(source: str | None) -> None:
    record = _row(
        _CLIENT_COMMIT,
        0.94,
        "2026-10-14T00:00:00Z",
        s_generation_timing_source=source,
    )
    assert transform_visual_gen_history(record) is None


@pytest.mark.parametrize("backend", [None, "future-backend"])
def test_missing_or_unknown_backend_cannot_establish_compatibility(backend: str | None) -> None:
    record = _client_row(0.94, "2026-10-14T00:00:00Z", s_backend=backend)
    assert transform_visual_gen_history(record) is None


def test_unknown_measurement_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported VisualGen benchmark backend"):
        get_generation_timing_source("future-backend")


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


def test_history_selection_preserves_compatible_baselines_thresholds_and_raw_rows(
    modules: _Modules,
) -> None:
    db, _, search = modules
    old = _row(_SERVER_COMMIT, 0.8554, "2026-10-14T00:00:00Z", d_baseline_median_generation=0.8554)
    new = _client_row(
        0.9433,
        "2026-09-14T00:00:00Z",
        d_baseline_median_generation=0.94,
        d_threshold_post_merge_median_generation=0.05,
        d_threshold_pre_merge_median_generation=0.1,
    )
    original = dict(new)
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [old, new]
    latest, baseline, history = db.get_history_data(
        {0: new},
        ["s_test_case_name", "s_gpu_type", "s_runtime"],
        {},
        history_transform=transform_visual_gen_history,
    )
    assert latest[0]["s_commit"] == _CLIENT_COMMIT
    assert [r["s_commit"] for r in history[0]] == [_CLIENT_COMMIT]
    assert baseline[0] == {
        "d_baseline_median_generation": 0.94,
        "d_threshold_post_merge_median_generation": 0.05,
        "d_threshold_pre_merge_median_generation": 0.1,
    }
    assert latest[0]["d_baseline_median_generation"] == 0.94
    assert latest[0] is not new
    assert new == original


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
def test_end_to_end_false_regression_is_removed_and_real_regression_remains(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
    post_merge: bool,
    pipeline_kwargs: dict,
) -> None:
    _, pipeline, search = modules
    old = _row(_SERVER_COMMIT, 1.2, "2026-10-14T00:00:00Z", d_baseline_median_generation=0.8554)
    good = _client_row(0.9433, "2026-09-14T00:00:00Z")
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [old, good]
    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": post_merge})
    for value, should_fail in [(0.9433, False), (1.2, True)]:
        current = _client_row(value, "2026-10-14T00:00:00Z")
        kwargs = dict(
            new_data_dict={0: current},
            upload_to_db=False,
            fail_on_regression=True,
            **pipeline_kwargs,
        )
        if should_fail and not post_merge:
            with pytest.raises(RuntimeError, match="Regression"):
                pipeline.process_and_upload_test_results(**kwargs)
        else:
            pipeline.process_and_upload_test_results(**kwargs)
        assert current["b_is_regression"] == should_fail


def test_fixed_client_baseline_and_custom_threshold_remain_effective(
    modules: _Modules, monkeypatch: pytest.MonkeyPatch, pipeline_kwargs: dict
) -> None:
    _, pipeline, search = modules
    good = _client_row(
        1.03,
        "2026-10-14T00:00:00Z",
        d_baseline_median_generation=1.0,
        d_threshold_post_merge_median_generation=0.02,
    )
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [good]
    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": True})
    current = _client_row(1.03, "2026-10-15T00:00:00Z")
    pipeline.process_and_upload_test_results(
        new_data_dict={0: current},
        upload_to_db=False,
        **pipeline_kwargs,
    )
    assert current["b_is_regression"] is True
    assert current["d_baseline_median_generation"] == 1.0
    assert current["d_threshold_post_merge_median_generation"] == 0.02
    assert "baseline=1.0000" in current["s_regression_info"]
    assert "threshold=2.00%" in current["s_regression_info"]


def test_first_post_merge_upload_becomes_compatible_history_for_next_run(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
    producer: types.ModuleType,
    pipeline_kwargs: dict,
) -> None:
    db, pipeline, search = modules
    first = _entry(producer, "openai-images")
    history = [
        _row(
            _SERVER_COMMIT,
            0.8554,
            "2026-09-11T00:00:00Z",
            s_test_case_name=first["s_test_case_name"],
        )
    ]
    original = dict(history[0])
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: history

    def upload(rows: list[dict], project: str) -> bool:
        history.extend(
            {**row, "@timestamp": "2026-10-14T00:00:00Z", "ts_created": 1791936000000}
            for row in rows
        )
        return True

    search.postToOpenSearchDB = upload
    monkeypatch.setattr(
        pipeline, "get_job_info", lambda: {"b_is_post_merge": True, "s_commit": _CLIENT_COMMIT}
    )
    pipeline.process_and_upload_test_results(new_data_dict={0: first}, **pipeline_kwargs)
    assert first["b_is_regression"] is False
    assert "@timestamp" not in first and "ts_created" not in first
    assert "baseline=" not in first["s_regression_info"]
    assert history[0] == original
    assert len(history) == 2
    assert history[-1]["s_generation_timing_source"] == "client_e2e_latency"
    latest, _, compatible = db.get_history_data(
        {0: first},
        pipeline_kwargs["match_keys"],
        {},
        history_transform=transform_visual_gen_history,
    )
    assert latest[0]["s_commit"] == _CLIENT_COMMIT
    assert len(compatible[0]) == 1

    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": False})
    slower = {**first, "d_median_generation": 1.2}
    with pytest.raises(RuntimeError, match="Regression"):
        pipeline.process_and_upload_test_results(
            new_data_dict={0: slower}, upload_to_db=False, **pipeline_kwargs
        )
    assert slower["b_is_regression"] is True
    assert "baseline=0.9400" in slower["s_regression_info"]


@pytest.fixture
def producer(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Load the actual metric mapping and producer without importing GPU code."""
    commands = types.ModuleType("tensorrt_llm.commands.utils")
    commands.get_visual_gen_num_gpus = lambda config: 1
    monkeypatch.setitem(sys.modules, "tensorrt_llm.commands.utils", commands)
    spec = importlib.util.spec_from_file_location(
        "visual_gen_perf_utils", _PERF_DIR / "visual_gen_perf_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(
    modules: _Modules, producer: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """Import the actual harness with only environment/GPU dependencies replaced."""
    config = types.ModuleType("defs.conftest")
    config.get_llm_root = lambda: str(_PERF_DIR.parents[3])
    config.llm_models_root = lambda: "/unused-test-models"
    gpu = types.ModuleType("tensorrt_llm._utils")
    gpu.get_free_port = lambda: 0
    monkeypatch.setitem(sys.modules, "defs.conftest", config)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._utils", gpu)
    monkeypatch.setitem(sys.modules, "defs.perf.visual_gen_perf_utils", producer)
    spec = importlib.util.spec_from_file_location(
        "defs.perf.test_visual_gen_perf_sanity", _PERF_DIR / "test_visual_gen_perf_sanity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("backend", "source", "generation"),
    [
        ("openai-videos", "client_gen_latency", 0.8),
        ("openai-images", "client_e2e_latency", 0.94),
        ("openai-image-edits", "client_e2e_latency", 0.94),
    ],
)
def test_producer_source_agrees_with_extracted_metrics_without_changing_match_keys(
    producer: types.ModuleType, backend: str, source: str, generation: float
) -> None:
    row = _entry(producer, backend)
    assert get_generation_timing_source(backend) == source
    assert row["s_generation_timing_source"] == source
    assert row["d_median_generation"] == generation
    assert transform_visual_gen_history(row) == row
    assert "s_generation_timing_source" not in producer.get_visual_gen_match_keys()
    row["s_generation_timing_source"] = (
        "client_e2e_latency" if source == "client_gen_latency" else "client_gen_latency"
    )
    assert transform_visual_gen_history(row) is None


@pytest.mark.parametrize(
    ("backend", "new_path", "new_value"),
    [
        ("openai-videos", "e2e_latency", 0.94),
        ("openai-images", "gen_latency", 0.8),
    ],
)
def test_one_path_change_updates_extraction_metadata_and_history_selection(
    producer: types.ModuleType,
    harness: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    new_path: str,
    new_value: float,
) -> None:
    old = _client_row(0.9, "2026-10-14T00:00:00Z", s_backend=backend)
    assert transform_visual_gen_history(old) == old
    monkeypatch.setitem(visual_gen_perf_history._GENERATION_METRIC_PATHS, backend, new_path)
    result = _benchmark_result(backend)
    validate = harness.VisualGenPerfSanityTestConfig._validate_benchmark_result
    validation_kwargs = dict(
        model_path="test-model",
        expected_num_gpus=1,
        result_data=result,
        client_config={"backend": backend},
        result_path="test-only.json",
    )
    validate(None, **validation_kwargs)
    row = _entry(producer, backend)
    assert producer.result_metric_paths(backend)["d_median_generation"] == f"{new_path}.median"
    assert row["d_median_generation"] == new_value
    assert row["s_generation_timing_source"] == f"client_{new_path}"
    assert transform_visual_gen_history(row) == row
    assert transform_visual_gen_history(old) is None
    for invalid in (float("nan"), 0, -1):
        result[new_path]["median"] = invalid
        with pytest.raises(ValueError, match=f"Invalid median {new_path}"):
            validate(None, **validation_kwargs)
    del result[new_path]
    with pytest.raises(ValueError, match="Missing keys"):
        validate(None, **validation_kwargs)


def test_producer_rejects_mismatched_result_backend(producer: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="backend does not match"):
        producer.build_visual_gen_db_entry(
            gpu_type="b200",
            model_name="flux2",
            server_name="flux2",
            server_config={},
            client_config={"backend": "openai-images"},
            result_data={"backend": "openai-videos"},
        )
