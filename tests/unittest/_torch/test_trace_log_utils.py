# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from tensorrt_llm._torch.pyexecutor import trace_log_utils


def _mock_cuda_counters(monkeypatch) -> None:
    gib = 1 << 30
    monkeypatch.setattr(trace_log_utils.torch.cuda, "current_device", Mock(return_value=2))
    monkeypatch.setattr(
        trace_log_utils.torch.cuda, "mem_get_info", Mock(return_value=(6 * gib, 10 * gib))
    )
    monkeypatch.setattr(trace_log_utils.torch.cuda, "memory_allocated", Mock(return_value=1 * gib))
    monkeypatch.setattr(trace_log_utils.torch.cuda, "memory_reserved", Mock(return_value=2 * gib))
    monkeypatch.setattr(
        trace_log_utils.torch.cuda, "max_memory_allocated", Mock(return_value=3 * gib)
    )
    monkeypatch.setattr(
        trace_log_utils.torch.cuda, "max_memory_reserved", Mock(return_value=4 * gib)
    )


def test_mem_snapshot_disabled_makes_no_cuda_calls(monkeypatch) -> None:
    monkeypatch.delenv("TLLM_LOG_MEM_PROFILE", raising=False)
    cuda_calls = []
    for name in (
        "current_device",
        "mem_get_info",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        call = Mock()
        monkeypatch.setattr(trace_log_utils.torch.cuda, name, call)
        cuda_calls.append(call)

    trace_log_utils.log_mem_snapshot("disabled")

    assert all(not call.called for call in cuda_calls)


def test_mem_snapshot_logs_compact_counters(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    monkeypatch.setattr(trace_log_utils.logger, "rank", 3)
    info = Mock()
    monkeypatch.setattr(trace_log_utils.logger, "info", info)
    _mock_cuda_counters(monkeypatch)

    trace_log_utils.log_mem_snapshot("stage/model")

    assert info.call_args.args[0] == (
        "[mem-profile/stage/model] rank=3 device=2 "
        "torch_alloc=1.00GiB torch_reserved=2.00GiB "
        "torch_alloc_peak=3.00GiB torch_reserved_peak=4.00GiB "
        "device_used=4.00GiB device_free=6.00GiB "
        "device_total=10.00GiB device_gap_estimate=2.00GiB"
    )


def test_forced_mem_snapshot_bypasses_gate(monkeypatch) -> None:
    monkeypatch.delenv("TLLM_LOG_MEM_PROFILE", raising=False)
    warning = Mock()
    monkeypatch.setattr(trace_log_utils.logger, "warning", warning)
    _mock_cuda_counters(monkeypatch)

    trace_log_utils.log_mem_snapshot("oom/model", force=True)

    assert warning.call_count == 1
    assert warning.call_args.args[0].startswith("[mem-profile/oom/model]")


def test_mem_snapshot_collection_failure_does_not_raise(monkeypatch) -> None:
    warning = Mock()
    monkeypatch.setattr(trace_log_utils.logger, "warning", warning)
    monkeypatch.setattr(
        trace_log_utils.torch.cuda,
        "current_device",
        Mock(side_effect=RuntimeError("CUDA unavailable")),
    )

    trace_log_utils.log_mem_snapshot("oom/model", force=True)

    warning.assert_called_once_with("[mem-profile/oom/model] snapshot unavailable: RuntimeError")
