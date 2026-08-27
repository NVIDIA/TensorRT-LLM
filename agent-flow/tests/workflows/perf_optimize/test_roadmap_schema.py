"""Tests for the perf-optimize roadmap.yaml schema and mutators."""

from __future__ import annotations

import copy

import pytest
import yaml

from agent_flow.workflows.perf_optimize import roadmap_schema

VALID = {
    "version": 1,
    "target_metric": "output_throughput",
    "baseline": {"value": 1234.5, "source": "baseline/benchmark_results.md"},
    "current_best": {"value": 1234.5, "source": "baseline/benchmark_results.md"},
    "items": [
        {
            "id": "opt-001",
            "title": "Enable CUDA graphs for decode",
            "category": "launch-host",
            "approach": "config",
            "evidence": ["nsys: 31% GPU idle from per-launch gaps (nsys_stats.txt)"],
            "casebook_ref": "launch storm at decode -> cuda-graph capture",
            "expected_gain_pct": 12.0,
            "expected_gain_rationale": "idle share x casebook-typical recovery",
            "how_to_apply": "add cuda_graph_config to tuning/extra_llm_api_options.yaml",
            "status": "pending",
            "attempts": 0,
            "measured_gain_pct": None,
        },
        {
            "id": "opt-002",
            "title": "Raise KV cache fraction",
            "category": "kv-capacity",
            "approach": "config",
            "evidence": ["requests queued at the concurrency limit (serve.log)"],
            "expected_gain_pct": 5.0,
            "expected_gain_rationale": "queueing share",
            "how_to_apply": "raise kv_cache_config.free_gpu_memory_fraction",
            "status": "pending",
            "attempts": 0,
            "measured_gain_pct": None,
        },
    ],
}


def _write(tmp_path, data) -> str:
    path = tmp_path / "roadmap.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(path)


def _invalid_errors(tmp_path, mutate) -> str:
    """Apply ``mutate`` to a deep copy of VALID and return the error text."""
    data = copy.deepcopy(VALID)
    mutate(data)
    path = _write(tmp_path, data)
    with pytest.raises(roadmap_schema.RoadmapError) as exc_info:
        roadmap_schema.load_roadmap(path)
    return str(exc_info.value)


# ------------------------------------------------------------------ validation


def test_valid_roadmap_loads(tmp_path):
    data = roadmap_schema.load_roadmap(_write(tmp_path, VALID))
    assert [i["id"] for i in data["items"]] == ["opt-001", "opt-002"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(roadmap_schema.RoadmapError, match="not found"):
        roadmap_schema.load_roadmap(tmp_path / "roadmap.yaml")


def test_non_mapping_top_level_raises(tmp_path):
    path = tmp_path / "roadmap.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(roadmap_schema.RoadmapError, match="mapping"):
        roadmap_schema.load_roadmap(path)


def test_errors_are_batched(tmp_path):
    def mutate(data):
        data["version"] = 2
        data["target_metric"] = ""
        data["items"][0]["category"] = "not-a-category"
        data["items"][0]["approach"] = "magic"
        data["items"][1]["evidence"] = []

    message = _invalid_errors(tmp_path, mutate)
    for fragment in ("version", "target_metric", "category", "approach", "evidence"):
        assert fragment in message, fragment


def test_duplicate_ids_rejected(tmp_path):
    message = _invalid_errors(tmp_path, lambda d: d["items"][1].__setitem__("id", "opt-001"))
    assert "duplicates" in message


def test_bad_status_rejected(tmp_path):
    message = _invalid_errors(tmp_path, lambda d: d["items"][0].__setitem__("status", "done"))
    assert "status" in message


def test_expected_gain_must_be_positive_number(tmp_path):
    assert "expected_gain_pct" in _invalid_errors(
        tmp_path, lambda d: d["items"][0].__setitem__("expected_gain_pct", 0)
    )
    # bool is an int subclass — must be rejected explicitly.
    assert "expected_gain_pct" in _invalid_errors(
        tmp_path, lambda d: d["items"][0].__setitem__("expected_gain_pct", True)
    )


def test_empty_casebook_ref_normalized_to_absent(tmp_path):
    # Analyzers write `casebook_ref: ""` for items with no casebook row
    # (e.g. prior-campaign carry-overs); that must not fail the roadmap.
    data = copy.deepcopy(VALID)
    data["items"][1]["casebook_ref"] = ""
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, data))
    assert loaded["items"][1]["casebook_ref"] is None

    assert "casebook_ref" in _invalid_errors(
        tmp_path, lambda d: d["items"][0].__setitem__("casebook_ref", ["not", "a", "string"])
    )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("memory", "memory-bw"),
        ("Memory-BW", "memory-bw"),
        ("host", "launch-host"),
        ("comm", "communication"),
        ("kv-cache", "kv-capacity"),
    ],
)
def test_category_alias_normalized_on_load(tmp_path, alias, canonical):
    # Analyzers occasionally write shorthand ("memory") for the category
    # enum; unambiguous aliases must load as the canonical value instead
    # of failing the roadmap after the stage has finished (which costs a
    # full analyzer re-run).
    data = copy.deepcopy(VALID)
    data["items"][0]["category"] = alias
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, data))
    assert loaded["items"][0]["category"] == canonical


def test_unknown_category_still_rejected(tmp_path):
    message = _invalid_errors(
        tmp_path, lambda d: d["items"][0].__setitem__("category", "gpu-stuff")
    )
    assert "category" in message
    assert "gpu-stuff" in message


def test_baseline_shape_enforced(tmp_path):
    assert "baseline" in _invalid_errors(tmp_path, lambda d: d.pop("baseline"))
    assert "baseline.value" in _invalid_errors(
        tmp_path, lambda d: d["baseline"].__setitem__("value", "fast")
    )


def test_current_best_optional_but_validated_when_present(tmp_path):
    data = copy.deepcopy(VALID)
    del data["current_best"]
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, data))
    assert "current_best" not in loaded

    assert "current_best.source" in _invalid_errors(
        tmp_path, lambda d: d["current_best"].__setitem__("source", "")
    )


def test_lifecycle_defaults_filled(tmp_path):
    data = copy.deepcopy(VALID)
    for field in ("status", "attempts", "measured_gain_pct"):
        data["items"][0].pop(field, None)
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, data))
    item = loaded["items"][0]
    assert item["status"] == "pending"
    assert item["attempts"] == 0
    assert item["measured_gain_pct"] is None


# ------------------------------------------------------------------ curve mode

_CURVE = [
    {"concurrency": 8, "value": 812.0, "tok_s_user": 21.4, "tok_s_gpu": 101.5},
    {"concurrency": 32, "value": 1657.0, "tok_s_user": 12.9, "tok_s_gpu": 207.1},
    {"concurrency": 128, "value": 2210.0, "tok_s_user": 6.1, "tok_s_gpu": 276.3},
]


def _with_curves(data):
    data["baseline"]["curve"] = copy.deepcopy(_CURVE)
    data["current_best"]["curve"] = copy.deepcopy(_CURVE)


def test_curve_augmented_roadmap_loads(tmp_path):
    data = copy.deepcopy(VALID)
    _with_curves(data)
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, data))
    assert loaded["baseline"]["curve"] == _CURVE
    assert loaded["current_best"]["curve"] == _CURVE


def test_curve_errors_are_batched(tmp_path):
    def mutate(data):
        _with_curves(data)
        del data["baseline"]["curve"][0]["tok_s_user"]  # missing field
        data["current_best"]["curve"][1]["concurrency"] = 8  # not ascending

    message = _invalid_errors(tmp_path, mutate)
    assert "baseline.curve[0].tok_s_user" in message
    assert "strictly ascending" in message


@pytest.mark.parametrize(
    "bad_curve, fragment",
    [
        ([], "non-empty list"),
        ("oops", "non-empty list"),
        (["not-a-mapping"], "must be a mapping"),
        (
            [{"concurrency": True, "value": 1.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0}],
            "curve[0].concurrency",
        ),
        (
            [{"concurrency": 0, "value": 1.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0}],
            "curve[0].concurrency",
        ),
        (
            [{"concurrency": 8, "value": "fast", "tok_s_user": 1.0, "tok_s_gpu": 1.0}],
            "curve[0].value",
        ),
        (
            [
                {"concurrency": 8, "value": 1.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0},
                {"concurrency": 8, "value": 2.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0},
            ],
            "strictly ascending",
        ),
    ],
)
def test_curve_rejects_bad_shapes(tmp_path, bad_curve, fragment):
    def mutate(data):
        data["baseline"]["curve"] = bad_curve

    assert fragment in _invalid_errors(tmp_path, mutate)


# ----------------------------------------------------------------- selection


def test_top_pending_item_honors_list_order_and_status(tmp_path):
    data = copy.deepcopy(VALID)
    data["items"][0]["status"] = "accepted"
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, data))
    assert roadmap_schema.top_pending_item(loaded)["id"] == "opt-002"


def test_top_pending_item_skips_items_below_noise_floor(tmp_path):
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, VALID))
    # opt-001 promises 12%, opt-002 promises 5%.
    assert roadmap_schema.top_pending_item(loaded, 6.0)["id"] == "opt-001"
    loaded["items"][0]["status"] = "failed"
    assert roadmap_schema.top_pending_item(loaded, 6.0) is None
    assert roadmap_schema.top_pending_item(loaded, 5.0)["id"] == "opt-002"


def test_top_pending_item_filters_disallowed_approaches(tmp_path):
    data = copy.deepcopy(VALID)
    data["items"][1]["approach"] = "code"
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, data))
    # opt-001 (config, 12%) leads, but a code-only run must skip it.
    assert roadmap_schema.top_pending_item(loaded)["id"] == "opt-001"
    assert roadmap_schema.top_pending_item(loaded, 0.0, ("code",))["id"] == "opt-002"
    assert roadmap_schema.top_pending_item(loaded, 0.0, ("config",))["id"] == "opt-001"
    # None means no filter; both approaches allowed behaves the same.
    assert roadmap_schema.top_pending_item(loaded, 0.0, ("config", "code"))["id"] == "opt-001"
    # The filter composes with the noise floor: opt-002 promises 5%.
    assert roadmap_schema.top_pending_item(loaded, 6.0, ("code",)) is None


def test_find_item(tmp_path):
    loaded = roadmap_schema.load_roadmap(_write(tmp_path, VALID))
    assert roadmap_schema.find_item(loaded, "opt-002")["title"] == "Raise KV cache fraction"
    assert roadmap_schema.find_item(loaded, "opt-999") is None


# ------------------------------------------------------------------ mutators


def test_mark_in_progress_touches_only_status(tmp_path):
    path = _write(tmp_path, VALID)
    roadmap_schema.mark_in_progress(path, "opt-001")
    data = roadmap_schema.load_roadmap(path)
    changed = roadmap_schema.find_item(data, "opt-001")
    assert changed["status"] == "in_progress"
    # Everything else — including the sibling item — is untouched.
    assert changed["expected_gain_pct"] == 12.0
    assert changed["attempts"] == 0
    assert roadmap_schema.find_item(data, "opt-002") == VALID["items"][1]
    assert data["baseline"] == VALID["baseline"]


def test_apply_evaluation_records_outcome(tmp_path):
    path = _write(tmp_path, VALID)
    roadmap_schema.apply_evaluation(
        path, "opt-001", status="accepted", attempts=2, measured_gain_pct=8.4
    )
    item = roadmap_schema.find_item(roadmap_schema.load_roadmap(path), "opt-001")
    assert item["status"] == "accepted"
    assert item["attempts"] == 2
    assert item["measured_gain_pct"] == pytest.approx(8.4)


def test_apply_evaluation_without_gain_leaves_it_null(tmp_path):
    path = _write(tmp_path, VALID)
    roadmap_schema.apply_evaluation(path, "opt-001", status="failed", attempts=3)
    item = roadmap_schema.find_item(roadmap_schema.load_roadmap(path), "opt-001")
    assert item["status"] == "failed"
    assert item["measured_gain_pct"] is None


def test_apply_evaluation_rejects_bad_status_and_missing_item(tmp_path):
    path = _write(tmp_path, VALID)
    with pytest.raises(roadmap_schema.RoadmapError, match="invalid status"):
        roadmap_schema.apply_evaluation(path, "opt-001", status="done", attempts=1)
    with pytest.raises(roadmap_schema.RoadmapError, match="opt-999"):
        roadmap_schema.apply_evaluation(path, "opt-999", status="failed", attempts=1)


def test_set_current_best_advances_watermark(tmp_path):
    path = _write(tmp_path, VALID)
    roadmap_schema.set_current_best(path, 1298.7, "rounds/round_1/attempt_1/evaluation.md")
    data = roadmap_schema.load_roadmap(path)
    assert data["current_best"] == {
        "value": 1298.7,
        "source": "rounds/round_1/attempt_1/evaluation.md",
    }
    # Baseline and items are untouched.
    assert data["baseline"] == VALID["baseline"]
    assert [i["id"] for i in data["items"]] == ["opt-001", "opt-002"]


def test_set_current_best_stores_curve(tmp_path):
    path = _write(tmp_path, VALID)
    roadmap_schema.set_current_best(
        path, 1559.7, "rounds/round_1/attempt_1/evaluation.md", curve=_CURVE
    )
    data = roadmap_schema.load_roadmap(path)
    assert data["current_best"]["value"] == pytest.approx(1559.7)
    assert data["current_best"]["curve"] == _CURVE


def test_set_current_best_without_curve_drops_stale_curve(tmp_path):
    data = copy.deepcopy(VALID)
    _with_curves(data)
    path = _write(tmp_path, data)
    roadmap_schema.set_current_best(path, 1400.0, "rounds/round_2/attempt_1/evaluation.md")
    reloaded = roadmap_schema.load_roadmap(path)
    # value and curve never desync: no curve supplied means no curve kept.
    assert "curve" not in reloaded["current_best"]
    # Baseline's curve is untouched.
    assert reloaded["baseline"]["curve"] == _CURVE


def test_set_current_best_rejects_malformed_curve(tmp_path):
    path = _write(tmp_path, VALID)
    with pytest.raises(roadmap_schema.RoadmapError, match="strictly ascending"):
        roadmap_schema.set_current_best(
            path,
            1400.0,
            "rounds/round_1/attempt_1/evaluation.md",
            curve=[
                {"concurrency": 32, "value": 1.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0},
                {"concurrency": 8, "value": 2.0, "tok_s_user": 1.0, "tok_s_gpu": 1.0},
            ],
        )
    # The failed call must not have touched the file.
    assert roadmap_schema.load_roadmap(path)["current_best"] == VALID["current_best"]
