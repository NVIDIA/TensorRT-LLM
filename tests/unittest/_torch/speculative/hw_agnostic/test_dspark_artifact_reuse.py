# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost reuse changes predictions/activation, never target acceptance."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from tensorrt_llm._torch.speculative.dspark_planner import (
    ExactSpsCostRow,
    ExactSpsCostTable,
    ExactSpsDrainGuard,
    load_runtime_sps_cost_table,
    select_exact_sps_candidate,
)
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig
from tensorrt_llm.usage.llmapi_config import collect_llm_api_config_payloads, manifest_rows

pytestmark = pytest.mark.cpu_only


def _payload():
    fingerprint = {
        "gpu": "B300",
        "gpu_count": 8,
        "gpu_snapshot_sha256": "a" * 64,
        "global_graph_batch_sizes": [512],
        "max_draft_len": 3,
        "rank_local_graph_batch_sizes": [64],
        "runtime_snapshot": "original-runtime",
        "source_diff_sha256": "b" * 64,
        "source_head": "original-source",
        "topology": "DEP8",
    }
    return {
        "schema_version": 2,
        "minimum_predicted_gain": 0.01,
        "engine_fingerprint": fingerprint,
        "engine_fingerprint_sha256": hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "cost_tables": {"64": {"token_counts": [0, 128], "step_time_ms": [10.0, 8.0]}},
        "measurements": [
            {
                "rank_local_graph_batch_size": 64,
                "rank_local_verifier_budget": budget,
                "step_time_ms": step,
                "source_result_sha256": "c" * 64,
            }
            for budget, step in ((0, 10.0), (128, 8.0))
        ],
    }


def _load(tmp_path, payload, *, strict=False, live=None, graph_batch_sizes=(64,)):
    path = tmp_path / "sps.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    live_path = None
    if live is not None:
        live_path = tmp_path / "live.json"
        live_path.write_text(json.dumps(live), encoding="utf-8")
    return load_runtime_sps_cost_table(
        path,
        graph_batch_sizes=graph_batch_sizes,
        max_draft_len=3,
        live_engine_fingerprint_path=live_path,
        validate_freshness=strict,
    )


def _select(table, *, yield_value=9.0):
    return select_exact_sps_candidate(
        graph_batch_size=64,
        native_expected_yield=10.0,
        compact_expected_yields={128: yield_value},
        compact_max_yield_losses_per_request={128: 10.0 - yield_value},
        cost_table=table,
    )


def test_reuse_without_live_receipt_or_drain_can_genuinely_select_compact(tmp_path):
    payload = _payload()
    before = copy.deepcopy(payload)
    table, metadata = _load(tmp_path, payload)
    assert metadata == before
    assert table.validate_artifact_freshness is False
    assert table.iteration_drain_guard is None
    assert _select(table) == 128
    assert _select(table, yield_value=8.0) == 0
    assert table.minimum_predicted_gain == 0.01


@pytest.mark.parametrize(
    "field", ["source_head", "runtime_snapshot", "source_diff_sha256", "gpu_snapshot_sha256"]
)
def test_reuse_retains_original_metadata_but_strict_rejects_staleness(tmp_path, field):
    payload = _payload()
    live = copy.deepcopy(payload["engine_fingerprint"])
    live[field] = "d" * 64
    table, metadata = _load(tmp_path, payload, live=live)
    assert _select(table) == 128
    assert metadata["engine_fingerprint"] == payload["engine_fingerprint"]
    assert metadata["engine_fingerprint"][field] != live[field]
    with pytest.raises(ValueError, match="does not match active runtime"):
        _load(tmp_path, payload, strict=True, live=live)


def test_strict_requires_independent_live_and_retains_absent_drain_refusal(tmp_path):
    payload = _payload()
    with pytest.raises(ValueError, match="independently generated"):
        _load(tmp_path, payload, strict=True)
    table, _ = _load(
        tmp_path, payload, strict=True, live=copy.deepcopy(payload["engine_fingerprint"])
    )
    assert _select(table) == 0


@pytest.mark.parametrize(
    "fault",
    [
        "nonfinite",
        "missing_native",
        "illegal_window",
        "missing_graph",
        "wrong_k",
        "bad_hash",
        "missing_provenance",
    ],
)
def test_reuse_keeps_all_structural_guards(tmp_path, fault):
    payload = _payload()
    if fault == "nonfinite":
        payload["cost_tables"]["64"]["step_time_ms"][1] = float("nan")
    if fault == "missing_native":
        payload["cost_tables"]["64"]["token_counts"][0] = 64
    if fault == "illegal_window":
        payload["cost_tables"]["64"]["token_counts"][1] = 257
    if fault == "missing_graph":
        payload["cost_tables"] = {"32": payload["cost_tables"]["64"]}
    if fault == "wrong_k":
        payload["engine_fingerprint"]["max_draft_len"] = 5
        payload["engine_fingerprint_sha256"] = hashlib.sha256(
            json.dumps(
                payload["engine_fingerprint"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    if fault == "bad_hash":
        payload["engine_fingerprint_sha256"] = "0" * 64
    if fault == "missing_provenance":
        payload["measurements"].pop()
    with pytest.raises((TypeError, ValueError)):
        _load(tmp_path, payload)


def test_provided_drain_policy_prices_and_margin_are_unchanged_in_both_modes():
    kwargs = {
        "tables": {64: ExactSpsCostRow((0, 128), (10.0, 8.0))},
        "max_draft_len": 3,
        "minimum_predicted_gain": 0.01,
        "iteration_drain_guard": ExactSpsDrainGuard(1.0, 1.0, 0.0, 64, "a" * 64),
    }
    strict = ExactSpsCostTable(**kwargs)
    reuse = ExactSpsCostTable(**kwargs, validate_artifact_freshness=False)
    assert _select(strict) == _select(reuse) == 0
    assert strict.iteration_drain_guard == reuse.iteration_drain_guard
    assert strict.tables == reuse.tables
    assert strict.minimum_predicted_gain == reuse.minimum_predicted_gain
    assert strict.collective_identity_words != reuse.collective_identity_words
    assert len(strict.collective_identity_words) == len(reuse.collective_identity_words) == 8


@pytest.mark.parametrize("strict", [False, True])
def test_freshness_mode_serialization_and_telemetry_capture_only_the_bool(strict):
    class Args(BaseModel):
        speculative_config: DSparkDecodingConfig

    config = DSparkDecodingConfig(
        max_draft_len=3,
        enable_confidence_scheduling=True,
        validate_confidence_artifact_freshness=strict,
        confidence_sps_table_path="/private/original/sps.json",
        confidence_sts_path="/private/original/sts.json",
    )
    args = Args.model_validate_json(Args(speculative_config=config).model_dump_json())
    field_path = "speculative_config.validate_confidence_artifact_freshness"
    expected = {"path": field_path, "kind": "value", "capture_policy": "bool"}
    assert expected in manifest_rows(Args)
    captured, metadata = collect_llm_api_config_payloads(args)
    assert json.loads(metadata)["capture_succeeded"] is True
    assert json.loads(captured)[field_path] is strict
    assert "private" not in captured
    golden_path = Path(__file__).parents[5] / "tensorrt_llm/usage/llm_args_golden_manifest.json"
    assert expected in json.loads(golden_path.read_text(encoding="utf-8"))["TorchLlmArgs"]


def _multigraph_payload():
    payload = _payload()
    fingerprint = payload["engine_fingerprint"]
    fingerprint["rank_local_graph_batch_sizes"] = [32, 64]
    fingerprint["global_graph_batch_sizes"] = [256, 512]
    payload["engine_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["cost_tables"]["32"] = {
        "token_counts": [0, 64],
        "step_time_ms": [9.0, 7.0],
    }
    payload["measurements"].extend(
        {
            "rank_local_graph_batch_size": 32,
            "rank_local_verifier_budget": budget,
            "step_time_ms": step,
            "source_result_sha256": "c" * 64,
        }
        for budget, step in ((0, 9.0), (64, 7.0))
    )
    return payload


def test_reuse_graph_subset_installs_only_captured_tiers_and_preserves_artifact(tmp_path):
    payload = _multigraph_payload()
    before = copy.deepcopy(payload)
    table, metadata = _load(tmp_path, payload)
    assert metadata == before
    assert set(table.tables) == {64}
    assert table.candidate_cells() == ((64, 128),)
    assert _select(table) == 128
    with pytest.raises(ValueError, match="no direct measurements for G=32"):
        table.for_graph_batch_size(32)


def test_strict_freshness_keeps_exact_original_graph_ladder_requirement(tmp_path):
    payload = _multigraph_payload()
    live = copy.deepcopy(payload["engine_fingerprint"])
    with pytest.raises(ValueError, match="original runtime graph ladder"):
        _load(tmp_path, payload, strict=True, live=live)
    table, metadata = _load(tmp_path, payload, strict=True, live=live, graph_batch_sizes=(32, 64))
    assert metadata == payload
    assert set(table.tables) == {32, 64}
    assert table.candidate_cells() == ((32, 64), (64, 128))
    assert _select(table) == 0


@pytest.mark.parametrize(
    "fault", ["nonfinite", "missing_native", "missing_provenance", "mismatched_provenance"]
)
def test_reuse_graph_subset_still_validates_unused_measurements(tmp_path, fault):
    payload = _multigraph_payload()
    if fault == "nonfinite":
        payload["cost_tables"]["32"]["step_time_ms"][1] = float("nan")
    elif fault == "missing_native":
        payload["cost_tables"]["32"]["token_counts"][0] = 32
    elif fault == "missing_provenance":
        payload["measurements"].pop()
    else:
        payload["measurements"][-1]["step_time_ms"] = 6.0
    with pytest.raises(ValueError):
        _load(tmp_path, payload)


@pytest.mark.parametrize("fault", ["original_ladder", "global_ladder", "unmeasured_runtime"])
def test_reuse_graph_subset_keeps_original_fingerprint_and_runtime_coverage(tmp_path, fault):
    payload = _multigraph_payload()
    graph_batch_sizes = (64,)
    if fault == "original_ladder":
        payload["engine_fingerprint"]["rank_local_graph_batch_sizes"] = [64]
    elif fault == "global_ladder":
        payload["engine_fingerprint"]["global_graph_batch_sizes"] = [512]
    else:
        graph_batch_sizes = (128,)
    payload["engine_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(payload["engine_fingerprint"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        _load(tmp_path, payload, graph_batch_sizes=graph_batch_sizes)


@pytest.mark.parametrize("tail_graph_batch_size", [32, 64])
def test_reuse_graph_subset_requires_captured_and_measured_drain_tail(
    tmp_path, tail_graph_batch_size
):
    payload = _multigraph_payload()
    payload["iteration_drain_guard"] = {
        "loss_multiplier": 1.0,
        "mean_output_tokens_per_request_iteration": 1.0,
        "minimum_group_value_ms": 0.0,
        "tail_graph_batch_size": tail_graph_batch_size,
        "source_result_sha256": "a" * 64,
    }
    if tail_graph_batch_size == 32:
        with pytest.raises(ValueError, match="captured, measured native tail"):
            _load(tmp_path, payload)
    else:
        table, metadata = _load(tmp_path, payload)
        assert metadata == payload
        assert table.iteration_drain_guard.tail_graph_batch_size == 64
        assert set(table.tables) == {64}
