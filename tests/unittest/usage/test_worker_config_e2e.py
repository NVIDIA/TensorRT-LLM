# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify worker observations through real two-GPU startup and local HTTP delivery."""

import json

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.usage import usage_lib

from .test_e2e_capture import CaptureHandler, _assert_event_matches_sms_schema, _get_model_path
from .test_e2e_capture import capture_server as capture_server
from .test_e2e_capture import reset_usage_state as reset_usage_state


# Fresh workers must inherit this test's opt-in, not another test's opt-out.
@pytest.mark.gpu2
@pytest.mark.private_mpi_session
@pytest.mark.threadleak(enabled=False)
@pytest.mark.usefixtures("reset_usage_state", "enable_telemetry")
@pytest.mark.parametrize("orchestrator", ["mpi", "rpc", pytest.param("ray", marks=pytest.mark.ray)])
def test_worker_config_tp2(
    orchestrator: str,
    capture_server: str,  # noqa: F811 - imported pytest fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only external delivery is redirected; startup and worker UDP transport are real.
    monkeypatch.setattr(usage_lib, "_get_stats_server", lambda: capture_server)
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "0")
    model_path = _get_model_path()
    with LLM(
        model=model_path,
        skip_tokenizer_init=True,
        tensor_parallel_size=2,
        orchestrator_type=None if orchestrator == "mpi" else orchestrator,
        max_batch_size=1,
        max_seq_len=64,
        max_num_tokens=64,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.05,
            host_cache_size=0,
            disk_cache_size=0,
            enable_partial_reuse=False,
        ),
    ) as llm:
        outputs = llm.generate(
            [[1, 3, 4]], SamplingParams(max_tokens=2, temperature=0, end_id=2, pad_id=0)
        )
        assert len(outputs) == 1 and outputs[0].outputs[0].token_ids
        assert CaptureHandler.capture_event.wait(30), "No initial telemetry report received"

    initial_reports = [
        event
        for payload in CaptureHandler.captured_payloads
        for event in payload["events"]
        if event["name"] == "trtllm_initial_report"
    ]
    assert len(initial_reports) == 1
    event = initial_reports[0]
    parameters = event["parameters"]
    assert parameters["tensorParallelSize"] == 2
    config = json.loads(parameters["llmApiConfigJson"])
    metadata = json.loads(parameters["llmApiConfigMetaJson"])
    expected_values = {
        "kv_cache_config.host_cache_size": 0,
        "kv_cache_config.disk_cache_size": 0,
        "kv_cache_config.enable_partial_reuse": False,
    }
    assert metadata["capture_version"] == "3"
    assert metadata["worker_capture"] == {
        "expected": 2,
        "received": 2,
        "status": "complete",
        "verified_fields": list(expected_values),
        "conflicting_fields": [],
        "unavailable_fields": [],
    }
    assert {path: config[path] for path in expected_values} == expected_values
    assert model_path not in json.dumps(parameters)
    assert "_worker_config_endpoint" not in json.dumps(parameters)
    _assert_event_matches_sms_schema(event)
