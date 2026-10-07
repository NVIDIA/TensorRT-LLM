# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup-boundary tests for DSpark workload admission."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, DSparkDecodingConfig, TorchLlmArgs


@pytest.mark.cpu_only
@pytest.mark.parametrize("outcome", ["missing", "negative", "stale"])
def test_dspark_workload_admission_disables_before_runtime_construction(tmp_path, outcome):
    (tmp_path / "config.json").write_text('{"dspark_block_size": 2}')
    spec_cfg = DSparkDecodingConfig(
        max_draft_len=2,
        speculative_model=str(tmp_path),
        enable_confidence_scheduling=True,
        confidence_sps_table_path=(None if outcome == "missing" else "/sealed/sps.json"),
        confidence_sps_live_fingerprint_path=(
            None if outcome == "missing" else "/sealed/live.json"
        ),
        confidence_admission_receipt_path=(
            None if outcome == "missing" else "/sealed/admission.json"
        ),
        confidence_admission_receipt_sha256=(None if outcome == "missing" else "a" * 64),
    )
    if outcome == "negative":
        patch_kwargs = {"return_value": SimpleNamespace(admitted=False)}
    elif outcome == "stale":
        patch_kwargs = {"side_effect": ValueError("stale receipt")}
    else:
        patch_kwargs = {"side_effect": TypeError("missing receipt")}
    with patch(
        "tensorrt_llm._torch.speculative.dspark_planner.load_confidence_workload_admission",
        **patch_kwargs,
    ):
        # No CUDA-graph configuration is supplied. Construction succeeds only
        # if admission resolves the feature off before the confidence-specific
        # environment validator observes it.
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            speculative_config=spec_cfg,
        )

    assert args.speculative_config.max_draft_len == 2
    assert args.speculative_config.enable_confidence_scheduling is False
    assert args.speculative_config.confidence_sps_table_path is None
    assert args.speculative_config.confidence_sps_live_fingerprint_path is None
    assert args.speculative_config.confidence_admission_receipt_path is None
    assert args.speculative_config.confidence_admission_receipt_sha256 is None


@pytest.mark.cpu_only
def test_dspark_positive_workload_admission_preserves_confidence_path(tmp_path):
    (tmp_path / "config.json").write_text('{"dspark_block_size": 2}')
    spec_cfg = DSparkDecodingConfig(
        max_draft_len=2,
        speculative_model=str(tmp_path),
        enable_confidence_scheduling=True,
        confidence_sps_table_path="/sealed/sps.json",
        confidence_sps_live_fingerprint_path="/sealed/live.json",
        confidence_admission_receipt_path="/sealed/admission.json",
        confidence_admission_receipt_sha256="a" * 64,
    )
    with patch(
        "tensorrt_llm._torch.speculative.dspark_planner.load_confidence_workload_admission",
        return_value=SimpleNamespace(admitted=True),
    ):
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            skip_tokenizer_init=True,
            speculative_config=spec_cfg,
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=True),
        )

    assert args.speculative_config.enable_confidence_scheduling is True
    assert args.speculative_config.confidence_admission_receipt_path == "/sealed/admission.json"
    assert args.speculative_config.confidence_admission_receipt_sha256 == "a" * 64
