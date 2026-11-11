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
"""Tests of the checkpoint I/O CI experiment report, on synthetic rows without OpenSearch."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = REPO_ROOT / "jenkins" / "scripts" / "perf" / "checkpoint_io_report.py"
X86_BUILD_URL = "https://ci/job/LLM/job/main/job/L0_Test-x86_64-Multi-GPU/7/"
SBSA_BUILD_URL = "https://ci/job/LLM/job/main/job/L0_Test-SBSA-Multi-Node/7/"
METRIC = "checkpoint_pipeline_seconds"


@pytest.fixture(scope="module")
def report() -> ModuleType:
    spec = importlib.util.spec_from_file_location("checkpoint_io_report", REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_row(build: int, arm: str, classification: str, seconds: float, **fields: object) -> dict:
    row = {
        "ts_created": build * 1000,
        "s_commit": "0123abcd",
        "s_branch": "main",
        "s_build_url": X86_BUILD_URL,
        "s_gpu_type": "h200",
        "s_stage_name": "DGX_H200-8_GPUs-PyTorch-PerfSanity-Post-Merge-1",
        "s_host_node_name": "node-1",
        "s_test_case_name": "qwen3_32b_fp8_tp2_6k1k-con96_iter10_6k1k",
        "s_model_name": "qwen3_32b_fp8",
        "s_startup_observation_id": f"startup-{build}",
        "s_checkpoint_io_experiment_assigned_arm": arm,
        "s_checkpoint_io_experiment_assignment_source": "postmerge_build_number",
        "l_checkpoint_io_experiment_root_build_number": build,
        "s_checkpoint_io_experiment_classification": classification,
        "s_checkpoint_io_fallback_category": "none",
        "s_checkpoint_io_fallback_reason": "none",
        "b_startup_metrics_collection_complete": True,
        "d_checkpoint_pipeline_seconds": seconds,
        "d_total_model_loading_seconds": seconds + 1.0,
    }
    row.update(fields)
    return row


def alternating_rows(builds: range) -> list[dict]:
    """Odd builds load with rank-striped read-ahead in 8 s, even builds natively in 10 s."""
    rows = []
    for build in builds:
        if build % 2:
            rows.append(make_row(build, "auto", "rank_striped_activated", 8.0))
        else:
            rows.append(make_row(build, "native", "native_control", 10.0))
    return rows


def observe(report: ModuleType, rows: list[dict]) -> list:
    return [obs for row in rows for obs in report.observations_from_row(row)]


def test_aggregated_row_gives_one_server_observation(report: ModuleType) -> None:
    (observation,) = report.observations_from_row(
        make_row(7, "auto", "rank_striped_activated", 12.5)
    )

    assert observation.role == "server"
    assert observation.root_build == 7
    assert observation.pipeline == "main"
    assert observation.model == "qwen3_32b_fp8"
    assert observation.post_merge and observation.timing_complete
    assert observation.metrics == {
        "checkpoint_pipeline_seconds": 12.5,
        "total_model_loading_seconds": 13.5,
    }
    assert (observation.image, observation.image_source) == ("unknown", "unknown")


def test_disaggregated_row_gives_ctx_and_gen_observations(report: ModuleType) -> None:
    row = make_row(
        8,
        "native",
        "unknown",
        0.0,
        s_test_case_name="e2e-gb300_glm-5-fp4_8k1k_con64_ctx1_dep4_gen1_dep8_eplb0_mtp1",
        s_model_name="",
        d_ctx_checkpoint_pipeline_seconds=30.0,
        d_gen_checkpoint_pipeline_seconds=40.0,
        b_ctx_startup_metrics_collection_complete=True,
        b_gen_startup_metrics_collection_complete=True,
        s_gen_checkpoint_io_experiment_classification="native_control",
    )
    del row["d_checkpoint_pipeline_seconds"], row["d_total_model_loading_seconds"]

    observations = {obs.role: obs for obs in report.observations_from_row(row)}

    assert set(observations) == {"ctx", "gen"}
    assert observations["ctx"].metrics == {"checkpoint_pipeline_seconds": 30.0}
    assert observations["ctx"].classification == "unknown"
    assert observations["gen"].classification == "native_control"
    assert observations["gen"].model == "glm-5-fp4"
    assert (
        report._model_from_test_case("e2e-gb300_deepseek-v4-pro-dspark_agentx_con1156_ctx2_dep8")
        == "deepseek-v4-pro-dspark"
    )


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf, True, "12", None])
def test_invalid_timings_are_dropped(report: ModuleType, bad: object) -> None:
    row = make_row(9, "auto", "rank_striped_activated", 1.0)
    row["d_checkpoint_pipeline_seconds"] = bad
    row["d_total_model_loading_seconds"] = bad

    assert report.observations_from_row(row) == []


@pytest.mark.parametrize(
    ("fields", "complete"),
    [
        ({}, True),
        ({"b_startup_metrics_collection_complete": False}, False),
        ({"b_startup_metrics_timing_complete": False}, False),
        ({"b_startup_metrics_timing_complete": True}, True),
    ],
)
def test_timing_completeness(report: ModuleType, fields: dict, complete: bool) -> None:
    row = make_row(9, "auto", "rank_striped_activated", 1.0, **fields)

    assert report.observations_from_row(row)[0].timing_complete is complete


def test_resolve_images_prefers_the_row_then_the_commit(report: ModuleType) -> None:
    calls = []

    def lookup(commit: str, architecture: str) -> str:
        calls.append((commit, architecture))
        return f"image-of-{commit}-{architecture}"

    rows = [
        make_row(1, "auto", "rank_striped_activated", 1.0, s_runtime_image="repo/img:a@sha256:0"),
        make_row(2, "native", "native_control", 1.0, s_build_url=SBSA_BUILD_URL),
        make_row(3, "auto", "rank_striped_activated", 1.0, s_build_url="", s_gpu_type="gb300"),
        make_row(
            4,
            "auto",
            "rank_striped_activated",
            1.0,
            **{"s_checkpoint_io_experiment_assignment_source": "premerge_default"},
        ),
    ]

    images = [
        (obs.image, obs.image_source)
        for obs in report.resolve_images(observe(report, rows), lookup)
    ]

    assert images == [
        ("repo/img:a", "row"),
        ("image-of-0123abcd-sbsa", "commit"),
        ("image-of-0123abcd-sbsa", "commit"),
        ("unknown", "unknown"),
    ]
    assert calls == [("0123abcd", "sbsa"), ("0123abcd", "sbsa")]


def test_git_image_lookup_reads_the_tag_file_at_a_commit(
    report: ModuleType, tmp_path: Path
) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    tags = tmp_path / "jenkins" / "current_image_tags.properties"
    tags.parent.mkdir()
    tags.write_text("# comment\nLLM_DOCKER_IMAGE=repo/x86:1\nLLM_SBSA_DOCKER_IMAGE=repo/sbsa:1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "tags")
    commit = git("rev-parse", "HEAD")

    lookup = report.git_image_lookup(tmp_path)

    assert lookup(commit, "x86_64") == "repo/x86:1"
    assert lookup(commit, "sbsa") == "repo/sbsa:1"
    assert lookup("f" * 40, "sbsa") == "unknown"
    assert lookup("--output=/dev/null", "sbsa") == "unknown"


def test_pairs_are_one_to_one_nearest_and_within_the_window(report: ModuleType) -> None:
    treatment = observe(
        report, [make_row(b, "auto", "rank_striped_activated", 8.0) for b in (11, 13, 21)]
    )
    control = observe(report, [make_row(b, "native", "native_control", 10.0) for b in (10, 12, 30)])

    pairs = report.pair_observations(treatment, control, window=3)

    assert [(t.root_build, c.root_build) for t, c in pairs] == [(11, 10), (13, 12)]


def test_compare_effective_policy_and_intent_to_treat(report: ModuleType) -> None:
    rows = alternating_rows(range(1, 21))
    rows[4] = make_row(5, "auto", "auto_fallback", 10.0)
    rows.append(
        make_row(
            41,
            "auto",
            "rank_striped_activated",
            1.0,
            **{"s_checkpoint_io_experiment_assignment_source": "premerge_default"},
        )
    )
    observations = observe(report, rows)
    effective, intent_to_treat = report.COMPARISONS

    (summary,), pairs = report.compare(observations, effective, METRIC, 3, 200, 0)
    assert (summary.n_treatment, summary.n_control, summary.n_pairs) == (9, 10, 9)
    assert summary.change == pytest.approx(-0.2)
    assert (summary.ci_low, summary.ci_high) == pytest.approx((-0.2, -0.2))
    assert summary.share_treatment_faster == 1.0
    assert all(abs(p.treatment.root_build - p.control.root_build) <= 3 for p in pairs)

    (summary,), _ = report.compare(observations, intent_to_treat, METRIC, 3, 200, 0)
    assert (summary.n_treatment, summary.n_control, summary.n_pairs) == (10, 10, 10)
    assert summary.change == pytest.approx(-0.2)
    assert summary.share_treatment_faster == pytest.approx(0.9)


def test_pairs_stay_within_one_image_and_pipeline(report: ModuleType) -> None:
    rows = [
        make_row(1, "auto", "rank_striped_activated", 8.0, s_runtime_image="img:a"),
        make_row(2, "native", "native_control", 10.0, s_runtime_image="img:b"),
        make_row(
            3, "auto", "rank_striped_activated", 8.0, s_build_url="https://ci/job/LLM/job/r/x"
        ),
    ]

    (summary,), pairs = report.compare(
        observe(report, rows), report.COMPARISONS[0], METRIC, 3, 10, 0
    )

    assert pairs == [] and summary.n_pairs == 0 and math.isnan(summary.change)


def test_bootstrap_interval_is_deterministic(report: ModuleType) -> None:
    rows = [
        make_row(b, "auto", "rank_striped_activated", 8.0 + b % 5)
        if b % 2
        else make_row(b, "native", "native_control", 10.0)
        for b in range(1, 41)
    ]
    observations = observe(report, rows)

    runs = [report.compare(observations, report.COMPARISONS[0], METRIC, 3, 500, 1)[0] for _ in "ab"]

    assert runs[0] == runs[1]
    assert runs[0][0].ci_low <= runs[0][0].change <= runs[0][0].ci_high


def test_repeat_loads(report: ModuleType) -> None:
    rows = [
        make_row(5, "auto", "rank_striped_activated", 1.0, ts_created=1000),
        make_row(
            5, "auto", "rank_striped_activated", 1.0, ts_created=2000, s_startup_observation_id="b"
        ),
        make_row(
            5,
            "auto",
            "rank_striped_activated",
            1.0,
            s_host_node_name="node-2",
            **{"s_startup_observation_id": "c"},
        ),
        make_row(-1, "auto", "rank_striped_activated", 1.0, s_startup_observation_id="d"),
        make_row(-1, "auto", "rank_striped_activated", 1.0, s_startup_observation_id="e"),
    ]

    marked = report.mark_repeat_loads(observe(report, rows))

    assert [obs.repeat_load for obs in marked] == [False, True, False, False, False]


def test_fallback_summary(report: ModuleType) -> None:
    reason = (
        "rank 0: checkpoint bytes ({}) exceed available host memory ({}) after startup headroom"
    )
    rows = [
        make_row(1, "auto", "rank_striped_activated", 1.0),
        make_row(
            3,
            "auto",
            "auto_fallback",
            1.0,
            s_checkpoint_io_fallback_category="host_memory",
            s_checkpoint_io_fallback_reason=reason.format(30, 10),
        ),
        make_row(
            5,
            "auto",
            "auto_fallback",
            1.0,
            s_checkpoint_io_fallback_category="host_memory",
            s_checkpoint_io_fallback_reason=reason.format(30, 12),
        ),
        make_row(
            7,
            "auto",
            "auto_fallback",
            1.0,
            **{
                "s_checkpoint_io_fallback_category": "model_specific_loader",
                "s_checkpoint_io_fallback_reason": "lazy",
                "s_checkpoint_io_experiment_assignment_source": "premerge_default",
            },
        ),
        make_row(2, "native", "native_control", 1.0),
    ]

    (summary,) = report.summarize_fallbacks(observe(report, rows))

    assert (summary.n_auto, summary.n_activated, summary.n_fallback) == (4, 1, 3)
    assert summary.n_post_merge_fallback == 2
    assert summary.categories == "host_memory 2, model_specific_loader 1"
    assert summary.top_reason == reason.format("N", "N").replace("rank 0", "rank N")
    assert summary.example_reason == reason.format(30, 10)


def test_main_writes_the_report_from_a_rows_cache(report: ModuleType, tmp_path: Path) -> None:
    cache = tmp_path / "rows.json"
    query = {"version": report.DEFAULT_EXPERIMENT_VERSION, "since": None, "until": None}
    cache.write_text(json.dumps({"query": query, "rows": alternating_rows(range(1, 21))}))
    out_dir = tmp_path / "out"
    args = ["--out-dir", str(out_dir), "--rows-json", str(cache), "--repo-root", str(tmp_path)]

    assert report.main([*args, "--bootstrap", "50"]) == 0

    text = (out_dir / "report.md").read_text()
    assert "## Effective policy: RANK-STRIPED activated vs NATIVE control" in text
    assert "| qwen3_32b_fp8_tp2_6k1k-con96_iter10_6k1k | h200 | server | 10 | -20.0% |" in text
    assert (out_dir / "pairs.csv").read_text().count("\n") == 21
    assert (out_dir / "per_test.csv").exists() and (out_dir / "fallbacks.csv").exists()

    with pytest.raises(ValueError, match="pass --refresh"):
        report.main([*args, "--since", "2026-01-01"])
