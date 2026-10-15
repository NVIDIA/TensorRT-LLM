# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for VisualGen's client-timing baseline migration."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from test_common.visual_gen_perf_history import (
    get_generation_timing_source,
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


def test_old_window_is_excluded_even_with_a_new_date() -> None:
    record = _row(_SERVER_COMMIT, 0.8554, "2026-10-14T00:00:00Z")
    assert transform_visual_gen_history(record) is None


def test_untagged_client_history_needs_explicit_backfill() -> None:
    record = _row(_CLIENT_COMMIT, 0.9433, "2026-09-14T00:00:00Z")
    assert transform_visual_gen_history(record) is None
    record["s_generation_timing_source"] = "client_e2e_latency"
    assert transform_visual_gen_history(record) == record


@pytest.mark.parametrize("source", [None, "", "server_generation", "unknown_source"])
def test_missing_or_different_source_is_excluded(source: str | None) -> None:
    assert transform_visual_gen_history(_row("f" * 40, 0.94, "2026-10-14T00:00:00Z")) is None
    assert (
        transform_visual_gen_history(
            _row(
                _CLIENT_COMMIT,
                0.94,
                "2026-10-14T00:00:00Z",
                s_generation_timing_source=source,
            )
        )
        is None
    )


def test_new_metadata_works_without_git_history() -> None:
    record = _row(
        "f" * 40,
        0.94,
        "2026-10-14T00:00:00Z",
        s_generation_timing_source="client_e2e_latency",
    )
    assert transform_visual_gen_history(record) == record


@pytest.mark.parametrize(
    ("backend", "source"),
    [
        ("openai-videos", "client_gen_latency"),
        ("openai-images", "client_e2e_latency"),
        ("openai-image-edits", "client_e2e_latency"),
    ],
)
def test_source_must_match_backend(backend: str, source: str) -> None:
    assert get_generation_timing_source(backend) == source
    record = _row(
        _CLIENT_COMMIT,
        0.94,
        "2026-10-14T00:00:00Z",
        s_backend=backend,
        s_generation_timing_source=source,
    )
    assert transform_visual_gen_history(record) == record
    record["s_generation_timing_source"] = (
        "client_e2e_latency" if source == "client_gen_latency" else "client_gen_latency"
    )
    assert transform_visual_gen_history(record) is None


@pytest.mark.parametrize("backend", [None, "", "future-backend"])
def test_missing_or_unknown_backend_cannot_establish_compatibility(backend: str | None) -> None:
    record = _client_row(0.94, "2026-10-14T00:00:00Z", s_backend=backend)
    assert transform_visual_gen_history(record) is None


def test_unknown_producer_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported VisualGen benchmark backend"):
        get_generation_timing_source("future-backend")


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
@pytest.mark.parametrize("backend", ["openai-images", "openai-image-edits", "openai-videos"])
def test_end_to_end_false_regression_is_removed_and_real_regression_remains(
    modules: _Modules,
    monkeypatch: pytest.MonkeyPatch,
    post_merge: bool,
    old_value: float,
    backend: str,
) -> None:
    _, pipeline, search = modules
    old = _row(_SERVER_COMMIT, old_value, "2026-10-14T00:00:00Z", s_backend=backend)
    good = _client_row(
        0.9433, "2026-09-14T00:00:00Z", s_backend=backend, d_baseline_median_generation=0.8554
    )
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: [old, good]
    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": post_merge})
    for value, should_fail in [(0.9433, False), (1.2, True)]:
        current = _row(
            "f" * 40,
            value,
            "2026-10-14T00:00:00Z",
            s_backend=backend,
            s_generation_timing_source=get_generation_timing_source(backend),
        )
        kwargs = dict(
            new_data_dict={0: current},
            match_keys=["s_test_case_name", "s_gpu_type", "s_runtime", "s_backend"],
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


def test_first_post_merge_upload_becomes_compatible_history_for_next_run(
    modules: _Modules, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, pipeline, search = modules
    history = [_row(_SERVER_COMMIT, 0.8554, "2026-09-11T00:00:00Z")]
    search.queryPerfDataFromOpenSearchDB = lambda *args, **kwargs: history

    def upload(rows: list[dict], project: str) -> bool:
        history.extend(dict(row) for row in rows)
        return True

    search.postToOpenSearchDB = upload
    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": True})
    first = _client_row(0.9433, "2026-10-14T00:00:00Z")
    kwargs = dict(
        match_keys=["s_test_case_name", "s_gpu_type", "s_runtime", "s_backend"],
        maximize_metrics=[],
        minimize_metrics=["d_median_generation"],
        regression_metrics=["d_median_generation"],
        history_transform=transform_visual_gen_history,
    )
    pipeline.process_and_upload_test_results(new_data_dict={0: first}, **kwargs)
    assert len(history) == 2
    assert history[-1]["s_generation_timing_source"] == "client_e2e_latency"
    latest, _, compatible = db.get_history_data(
        {0: first}, kwargs["match_keys"], {}, history_transform=transform_visual_gen_history
    )
    assert latest[0]["s_commit"] == _CLIENT_COMMIT
    assert len(compatible[0]) == 1

    monkeypatch.setattr(pipeline, "get_job_info", lambda: {"b_is_post_merge": False})
    slower = _client_row(1.2, "2026-10-15T00:00:00Z")
    with pytest.raises(RuntimeError, match="Regression"):
        pipeline.process_and_upload_test_results(
            new_data_dict={0: slower}, upload_to_db=False, **kwargs
        )
    assert slower["b_is_regression"] is True


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
    def latency(value: float) -> dict:
        return {"mean": value, "median": value, "percentiles": {"p90": value, "p99": value}}

    row = producer.build_visual_gen_db_entry(
        gpu_type="b200",
        model_name="flux2",
        server_name="flux2",
        server_config={},
        client_config={"backend": backend, "num_inference_steps": 50, "max_concurrency": 1},
        result_data={
            "backend": backend,
            "request_throughput": 2,
            "e2e_latency": latency(0.94),
            "gen_latency": latency(0.8),
        },
    )
    assert row["s_generation_timing_source"] == source
    assert row["d_median_generation"] == generation
    assert "s_generation_timing_source" not in producer.get_visual_gen_match_keys()


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
