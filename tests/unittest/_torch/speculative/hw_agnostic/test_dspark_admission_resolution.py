# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration-level native fallback, before any worker snapshots its flags."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.speculative import dspark_planner
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, TorchLlmArgs

_ARTIFACT_FIELDS = (
    "confidence_sts_path",
    "confidence_sps_table_path",
    "confidence_sps_live_fingerprint_path",
    "confidence_admission_receipt_path",
    "confidence_admission_receipt_sha256",
)


def _config(physical_k=3, **kwargs):
    return DSparkDecodingConfig(
        max_draft_len=physical_k,
        enable_confidence_scheduling=True,
        enable_fused_confidence_scheduler=True,
        **kwargs,
    )


@pytest.mark.parametrize("outcome", ["missing", "negative", "stale", "unreadable", "positive"])
@pytest.mark.parametrize("physical_k", [2, 3, 4, 5, 8])
def test_admission_resolves_before_worker_construction(monkeypatch, physical_k, outcome):
    artifacts = {name: "/sealed/artifact.json" for name in _ARTIFACT_FIELDS}
    artifacts["confidence_admission_receipt_sha256"] = "a" * 64
    config = _config(physical_k, **({} if outcome == "missing" else artifacts))
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        if outcome == "stale":
            raise ValueError("source identity mismatch")
        if outcome == "unreadable":
            raise OSError("artifact unavailable")
        return SimpleNamespace(admitted=outcome == "positive")

    monkeypatch.setattr(dspark_planner, "load_confidence_workload_admission", load)
    TorchLlmArgs._resolve_dspark_confidence_workload_admission(config)
    assert config.max_draft_len == physical_k
    assert config.tokens_per_gen_step == physical_k + 1
    assert config.enable_confidence_scheduling is (outcome == "positive")
    assert config.enable_fused_confidence_scheduler is (outcome == "positive")
    # Experiment classification must use the resolved snapshot, not the input
    # enable flag. A successful fallback is not an enabled-native experiment.
    resolved = config.model_dump()
    assert resolved["enable_confidence_scheduling"] is (outcome == "positive")
    assert resolved["enable_fused_confidence_scheduler"] is (outcome == "positive")
    for name in _ARTIFACT_FIELDS:
        assert getattr(config, name) == (artifacts[name] if outcome == "positive" else None)
        assert resolved[name] == (artifacts[name] if outcome == "positive" else None)
    if outcome == "missing":
        assert not calls
    else:
        assert calls[0][1]["physical_k"] == physical_k
        assert calls[0][1]["expected_receipt_sha256"] == "a" * 64


def test_k1_never_needs_admission_or_confidence_artifacts():
    config = _config(1)
    assert config.enable_confidence_scheduling is False
    assert config.enable_fused_confidence_scheduler is False
    assert config.tokens_per_gen_step == 2
    assert all(getattr(config, name) is None for name in _ARTIFACT_FIELDS)


@pytest.mark.parametrize("field_name", _ARTIFACT_FIELDS)
def test_feature_off_rejects_confidence_only_inputs(field_name):
    with pytest.raises(ValueError, match="enable_confidence_scheduling=True"):
        DSparkDecodingConfig(max_draft_len=3, **{field_name: "unconsumed"})
