# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for checkpoint-startup policy attribution."""

from pathlib import Path

import pytest

pytest.importorskip("torch._inductor")

__extra_import_path__ = ["~/tests/integration"]
from defs.perf import test_perf_sanity as perf_sanity  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _assignment() -> perf_sanity.CheckpointIoExperimentAssignment:
    return perf_sanity.CheckpointIoExperimentAssignment(
        version=perf_sanity.CHECKPOINT_IO_EXPERIMENT_VERSION,
        bucket=1,
        assigned_arm="auto",
        assignment_source="postmerge_build_number",
        pr_number=None,
        root_build_number=101,
    )


def test_make_startup_observation_uses_legacy_logs_only_when_field_is_absent(
    tmp_path: Path,
) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Checkpoint I/O policy: requested=native, selected=native, activated=False, "
        "effective=native, fallback_reason=none.\n",
        encoding="utf-8",
    )

    observation = perf_sanity.make_startup_observation({}, [str(server_log)], "aggregate")

    assert observation["metadata"]["checkpoint_io_policy_source"] == "legacy_logs"
    assert observation["checkpoint_io_policies"][0]["effective"] == "native"


@pytest.mark.parametrize(
    ("selected", "activated", "effective", "reason"),
    [
        ("native", False, "native", None),
        ("native", False, "native", "insufficient host memory"),
        ("rank_striped_read_ahead", True, "rank_striped_read_ahead", None),
        ("rank_striped_read_ahead", True, "native", "reader failed after activation"),
    ],
)
def test_make_startup_observation_prefers_final_primary_policy(
    tmp_path: Path, selected: str, activated: bool, effective: str, reason: str | None
) -> None:
    policy = {
        "requested": "auto",
        "selected": selected,
        "activated": activated,
        "effective": effective,
        "fallback_reason": reason,
    }
    draft_policy = dict(
        policy, requested="native", selected="native", activated=False, effective="native"
    )
    server_info = {
        "startup_metrics": {
            "model_loader": {
                "checkpoint_io_policy": policy,
                "draft_checkpoint_io_policy": draft_policy,
            },
            "draft_model_loader": {"checkpoint_io_policy": draft_policy},
        }
    }
    # A stale/independent draft log must not override the primary snapshot.
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Checkpoint I/O policy: requested=native, selected=native, activated=False, "
        "effective=native, fallback_reason=none.\n",
        encoding="utf-8",
    )

    observation = perf_sanity.make_startup_observation(server_info, [str(server_log)], "aggregate")

    assert observation["metadata"]["checkpoint_io_policy_source"] == "server_info"
    assert observation["checkpoint_io_policies"] == [
        {
            **policy,
            "fallback_reason": reason or "none",
            "fallback_category": perf_sanity.checkpoint_io_fallback_category(reason or "none"),
        }
    ]


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        {
            "requested": "auto",
            "selected": "native",
            "activated": "False",
            "effective": "native",
            "fallback_reason": None,
        },
        {
            "requested": "auto",
            "selected": "rank_striped_read_ahead",
            "activated": False,
            "effective": "rank_striped_read_ahead",
            "fallback_reason": None,
        },
        {
            "requested": [],
            "selected": "native",
            "activated": False,
            "effective": "native",
            "fallback_reason": None,
        },
    ],
)
def test_make_startup_observation_does_not_replace_unknown_structured_policy(
    tmp_path: Path, policy: object
) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Checkpoint I/O policy: requested=auto, selected=native, activated=False, "
        "effective=native, fallback_reason=none.\n",
        encoding="utf-8",
    )
    observation = perf_sanity.make_startup_observation(
        {"startup_metrics": {"model_loader": {"checkpoint_io_policy": policy}}},
        [str(server_log)],
        "aggregate",
    )
    new_data = {}
    perf_sanity.add_startup_metric_values(new_data, [observation], _assignment())

    assert observation["checkpoint_io_policies"] == []
    assert new_data["s_checkpoint_io_policy_effective"] == "unknown"
    assert new_data["b_checkpoint_io_policy_complete"] is False
    assert new_data["s_checkpoint_io_experiment_classification"] == "unknown"


@pytest.mark.parametrize("missing", [None, "timing", "policy", "server", "legacy_logs"])
def test_add_startup_metric_values_checks_timing_and_policy_completeness(
    missing: str | None,
) -> None:
    loader_metrics = {
        "checkpoint_preparation_seconds": 1.0,
        "weight_population_seconds": 2.0,
        "checkpoint_finalization_seconds": 0.0,
        "total_model_loading_seconds": 4.0,
        "checkpoint_io_policy": {
            "requested": "auto",
            "selected": "rank_striped_read_ahead",
            "activated": True,
            "effective": "rank_striped_read_ahead",
            "fallback_reason": None,
        },
    }
    if missing == "timing":
        del loader_metrics["weight_population_seconds"]
    if missing == "policy":
        loader_metrics["checkpoint_io_policy"] = None
    observation = perf_sanity.make_startup_observation(
        {"startup_metrics": {"model_loader": loader_metrics}}, [], "gen"
    )
    if missing == "legacy_logs":
        observation["metadata"]["checkpoint_io_policy_source"] = "legacy_logs"
    new_data = {}

    perf_sanity.add_startup_metric_values(
        new_data,
        [observation],
        _assignment(),
        role="gen",
        expected_server_count=2 if missing == "server" else 1,
    )

    assert new_data["b_gen_startup_metrics_collection_complete"] is (missing != "server")
    assert new_data["b_gen_startup_metrics_timing_complete"] is (
        missing not in ("timing", "server")
    )
    assert new_data["b_gen_checkpoint_io_policy_complete"] is (
        missing not in ("policy", "server", "legacy_logs")
    )
    expected_classification = (
        "unknown" if missing in ("policy", "server") else "rank_striped_activated"
    )
    assert new_data["s_gen_checkpoint_io_experiment_classification"] == expected_classification


@pytest.mark.parametrize("value", [True, -1.0, float("nan"), float("inf")])
def test_make_startup_observation_rejects_invalid_timing(value: object) -> None:
    observation = perf_sanity.make_startup_observation(
        {"startup_metrics": {"model_loader": {"weight_population_seconds": value}}}, [], "aggregate"
    )
    assert "weight_population_seconds" not in observation["metrics"]


def test_add_startup_metric_values_does_not_classify_partial_policy_coverage() -> None:
    policy = {
        "requested": "auto",
        "selected": "rank_striped_read_ahead",
        "activated": True,
        "effective": "rank_striped_read_ahead",
        "fallback_reason": None,
    }
    observations = [
        perf_sanity.make_startup_observation(
            {"startup_metrics": {"model_loader": {"checkpoint_io_policy": value}}}, [], "gen"
        )
        for value in (policy, None)
    ]
    new_data = {}
    perf_sanity.add_startup_metric_values(new_data, observations, _assignment(), role="gen")

    assert new_data["b_gen_startup_metrics_collection_complete"] is True
    assert new_data["b_gen_checkpoint_io_policy_complete"] is False
    assert new_data["s_gen_checkpoint_io_experiment_classification"] == "unknown"
