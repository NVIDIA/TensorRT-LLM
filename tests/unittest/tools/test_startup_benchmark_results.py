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
"""CPU-only result pairing and exclusion contracts; no TRT-LLM imports."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_SPEC = importlib.util.spec_from_file_location(
    "startup_results",
    Path(__file__).resolve().parents[3] / "jenkins/scripts/startup_benchmark/results.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_RESULTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RESULTS)


def _trial(variant: str = "native", repetition: int = 1, duration: float = 100) -> dict:
    native = variant == "native"
    return {
        "schema_version": 1,
        "case": "model_tp8",
        "variant": variant,
        "repetition": repetition,
        "profile": "application_cold",
        "status": "passed",
        "launch_to_ready_seconds": duration,
        "metric_scope": "model_loader_worker_rank_0",
        "metrics": {
            "total_model_loading_seconds": duration / 2,
            "checkpoint_pipeline_seconds": duration / 4,
        },
        "cache": {"verified": True},
        "policy": {
            "requested": "native" if native else "auto",
            "effective": "native" if native else "rank_striped_read_ahead",
            "activated": not native,
            "complete": True,
            "fallback_reason": "none",
            "scope": "rank_0",
        },
        "identity": {
            "runtime_image": "registry/image@sha256:abc",
            "runner_git_commit": "commit",
            "runner_source_fingerprint": "runner-script-hash",
            "runtime_version": "1.0",
            "hostname": "node",
            "checkpoint_fingerprint": "weights",
            "case_config_fingerprint": "config",
            "gpu_inventory": ["B300", "driver:1"],
        },
        "error": None,
    }


def _write(root: Path, name: str, record: object) -> None:
    path = root / name / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_matched_medians_use_pairwise_reductions(tmp_path: Path) -> None:
    for repetition, before, after in [(1, 100, 50), (2, 300, 270)]:
        _write(tmp_path, f"native{repetition}", _trial(repetition=repetition, duration=before))
        _write(tmp_path, f"striped{repetition}", _trial("rank_striped", repetition, after))
    summary = _RESULTS.collect_results(tmp_path)
    assert summary["matched_pair_count"] == 2
    row = next(row for row in summary["comparisons"] if row["metric"] == "launch_to_ready_seconds")
    assert row["native_median"] == 200
    assert row["candidate_median"] == 160
    assert row["median_paired_reduction_percent"] == pytest.approx(30)  # Not 20% of medians.
    assert len(summary["comparisons"]) == 3
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert "no pooled fleet speedup" in (tmp_path / "report.md").read_text()
    assert not summary["exclusions"]


@pytest.mark.parametrize("field", _RESULTS._IDENTITY_FIELDS)
def test_each_identity_dimension_must_match(tmp_path: Path, field: str) -> None:
    _write(tmp_path, "native", _trial())
    candidate = _trial("rank_striped")
    candidate["identity"][field] = "different"
    _write(tmp_path, "striped", candidate)
    summary = _RESULTS.collect_results(tmp_path)
    assert not summary["pairs"]
    assert len(summary["exclusions"]) == 2


def test_source_hash_allows_unavailable_mounted_git_metadata(tmp_path: Path) -> None:
    for variant in ("native", "rank_striped"):
        record = _trial(variant)
        record["identity"]["runner_git_commit"] = "unknown"
        _write(tmp_path, variant, record)
    assert _RESULTS.collect_results(tmp_path)["matched_pair_count"] == 1
    candidate = _trial("rank_striped")
    candidate["identity"]["runner_git_commit"] = "unknown"
    del candidate["identity"]["runner_source_fingerprint"]
    _write(tmp_path, "rank_striped", candidate)
    summary = _RESULTS.collect_results(tmp_path)
    assert not summary["pairs"]
    assert any(
        entry["reason"] == "incomplete comparison identity" for entry in summary["exclusions"]
    )


@pytest.mark.parametrize(
    "field,value", [("case", "other"), ("profile", "loader_isolation"), ("repetition", 2)]
)
def test_trial_dimensions_are_not_pooled(tmp_path: Path, field: str, value: object) -> None:
    _write(tmp_path, "native", _trial())
    candidate = _trial("rank_striped")
    candidate[field] = value
    _write(tmp_path, "candidate", candidate)
    assert not _RESULTS.collect_results(tmp_path)["pairs"]


@pytest.mark.parametrize(
    "kind",
    [
        "failed",
        "invalid",
        "cache",
        "fallback",
        "unknown",
        "activation",
        "identity",
        "scope",
        "incomplete",
        "metric_scope",
    ],
)
def test_untrusted_trials_remain_in_csv(tmp_path: Path, kind: str) -> None:
    _write(tmp_path, "native", _trial())
    candidate = _trial("rank_striped")
    if kind in {"failed", "invalid"}:
        candidate.update(status=kind, error="startup failed")
    elif kind == "cache":
        candidate["cache"]["verified"] = False
    elif kind in {"fallback", "unknown"}:
        candidate["policy"]["effective"] = "native" if kind == "fallback" else "unknown"
    elif kind == "activation":
        candidate["policy"]["activated"] = False
    elif kind == "identity":
        del candidate["identity"]["runtime_version"]
    elif kind == "incomplete":
        candidate["policy"]["complete"] = False
    elif kind == "metric_scope":
        candidate["metric_scope"] = "all_ranks_max"
    else:
        candidate["policy"]["scope"] = "all_ranks"
    _write(tmp_path, "candidate", candidate)
    summary = _RESULTS.collect_results(tmp_path)
    assert not summary["pairs"]
    assert len(summary["exclusions"]) == 2
    with (tmp_path / "results.csv").open(newline="") as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 2
    assert all(row["exclusion"] for row in rows)


def test_duplicates_are_excluded_without_pseudoreplication(tmp_path: Path) -> None:
    _write(tmp_path, "native1", _trial())
    _write(tmp_path, "native2", _trial())
    _write(tmp_path, "candidate", _trial("rank_striped"))
    summary = _RESULTS.collect_results(tmp_path)
    assert not summary["pairs"]
    assert (
        sum(entry["reason"] == "duplicate trial identity" for entry in summary["exclusions"]) == 2
    )


def test_common_finite_metrics_only_and_zero_baseline(tmp_path: Path) -> None:
    native, candidate = _trial(), _trial("rank_striped")
    native["metrics"].update(zero=0, missing=10, flag=True, nonfinite=float("nan"))
    candidate["metrics"].update(zero=1, flag=2, nonfinite=5)
    _write(tmp_path, "native", native)
    _write(tmp_path, "candidate", candidate)
    summary = _RESULTS.collect_results(tmp_path)
    metrics = summary["pairs"][0]["metrics"]
    assert not ({"missing", "flag", "nonfinite"} & metrics.keys())
    assert metrics["zero"]["reduction_percent"] is None
    assert "NaN" not in (tmp_path / "summary.json").read_text()


def test_future_explicit_candidate_policy(tmp_path: Path) -> None:
    _write(tmp_path, "native", _trial())
    candidate = _trial("future_streaming")
    candidate["policy"].update(requested="future_streaming", effective="future_streaming")
    _write(tmp_path, "candidate", candidate)
    assert _RESULTS.collect_results(tmp_path)["matched_pair_count"] == 1


def test_malformed_artifacts_and_empty_root(tmp_path: Path) -> None:
    assert _RESULTS.collect_results(tmp_path)["trial_count"] == 0
    _write(tmp_path, "nonobject", [1, 2])
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/result.json").write_text("{")
    summary = _RESULTS.collect_results(tmp_path)
    assert summary["status_counts"] == {"invalid": 2}
    assert len(summary["exclusions"]) == 2
    assert all(entry["error"] for entry in summary["exclusions"])


def test_missing_planned_trials_retain_run_error(tmp_path: Path) -> None:
    run = tmp_path / "nested_run"
    _write(run, "model_tp8/repeat_00/native", _trial(repetition=0))
    manifest = {
        "cases": [{"name": "model_tp8"}],
        "variants": [{"name": "native"}, {"name": "rank_striped"}],
        "repeats": 2,
        "profile": "application_cold",
        "identity": {"runtime_image": "image"},
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "run_error.json").write_text(json.dumps({"error": "qualification failure"}))
    summary = _RESULTS.collect_results(tmp_path)
    assert summary["trial_count"] == 4
    assert summary["status_counts"] == {"passed": 1, "missing": 3}
    missing = [
        entry for entry in summary["exclusions"] if entry["reason"] == "missing planned trial"
    ]
    assert len(missing) == 3
    assert all(entry["error"] == {"error": "qualification failure"} for entry in missing)
    with (tmp_path / "results.csv").open(newline="") as output:
        assert sum(row["status"] == "missing" for row in csv.DictReader(output)) == 3
    assert _RESULTS.collect_results(tmp_path) == summary


def test_missing_trials_without_run_error_or_valid_manifest(tmp_path: Path) -> None:
    manifest = {
        "cases": [{"name": "small"}],
        "variants": [{"name": "native"}],
        "repeats": 1,
        "profile": "loader_isolation",
        "identity": {},
    }
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    summary = _RESULTS.collect_results(tmp_path)
    assert summary["status_counts"] == {"missing": 1}
    assert summary["exclusions"][0]["error"] == "Planned trial has no result.json"
    path.write_text("{")
    assert _RESULTS.collect_results(tmp_path)["status_counts"] == {"invalid": 1}


@pytest.mark.parametrize(
    "submission_status,job_status,expected",
    [
        ("submitted", "submitted", "missing"),
        ("submission_failed", "planned", "not_submitted"),
        ("dry_run", "planned", "not_executed"),
    ],
)
def test_submission_without_runner_manifest_is_not_silent(
    tmp_path: Path, submission_status: str, job_status: str, expected: str
) -> None:
    submission = {
        "status": submission_status,
        "jobs": [
            {"case": "started", "status": "submitted", "job_id": "123"},
            {
                "case": "unstarted",
                "status": job_status,
                "job_id": "124" if job_status == "submitted" else None,
            },
        ],
        "variants": ["native", "rank_striped"],
        "repeats": 1,
        "profile": "application_cold",
        "image_reference": "image",
    }
    (tmp_path / "submission.json").write_text(json.dumps(submission))
    started = tmp_path / "results/started"
    started.mkdir(parents=True)
    (started / "run_manifest.json").write_text(
        json.dumps(
            {
                "cases": [{"name": "started"}],
                "variants": [{"name": "native"}, {"name": "rank_striped"}],
                "repeats": 1,
                "profile": "application_cold",
                "identity": {},
            }
        )
    )
    summary = _RESULTS.collect_results(tmp_path)
    assert summary["trial_count"] == 4  # Existing runner manifest is not double-counted.
    unstarted = [entry for entry in summary["exclusions"] if "unstarted" in entry["source"]]
    assert len(unstarted) == 2
    assert all(entry["error"]["job_status"] == job_status for entry in unstarted)
    assert summary["status_counts"][expected] == (4 if expected == "missing" else 2)
    assert "failed" not in summary["status_counts"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("launch_to_ready_seconds", True),
        ("launch_to_ready_seconds", float("inf")),
        ("metrics", []),
        ("repetition", True),
    ],
)
def test_malformed_fields_cannot_enter_comparison(
    tmp_path: Path, field: str, value: object
) -> None:
    record = _trial()
    record[field] = value
    _write(tmp_path, "bad", record)
    assert _RESULTS.collect_results(tmp_path)["exclusions"]
