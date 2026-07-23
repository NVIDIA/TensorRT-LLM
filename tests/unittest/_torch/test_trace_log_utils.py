# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor import trace_log_utils


@pytest.fixture(autouse=True)
def _reset_mem_history():
    trace_log_utils.reset_mem_history()
    yield
    trace_log_utils.reset_mem_history()


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
    assert not trace_log_utils._MEM_HISTORY


def test_mem_snapshot_logs_compact_counters(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    monkeypatch.setattr(trace_log_utils.logger, "rank", 3)
    info = Mock()
    nvml_query = Mock()
    monkeypatch.setattr(trace_log_utils.logger, "info", info)
    monkeypatch.setattr(trace_log_utils, "_format_nvml_process_fields", nvml_query)
    _mock_cuda_counters(monkeypatch)

    trace_log_utils.log_mem_snapshot("stage/model", iter=7, active_requests=2)

    assert info.call_args.args[0] == (
        "[mem-profile/stage/model] rank=3 device=2 "
        "torch_alloc=1.00GiB torch_reserved=2.00GiB "
        "torch_alloc_peak=3.00GiB torch_reserved_peak=4.00GiB "
        "device_used=4.00GiB device_free=6.00GiB "
        "device_total=10.00GiB device_gap_estimate=2.00GiB "
        "iter=7 active_requests=2"
    )
    assert len(trace_log_utils._MEM_HISTORY) == 1
    assert trace_log_utils._MEM_HISTORY[0][1] == info.call_args.args[0]
    nvml_query.assert_not_called()


def test_forced_mem_snapshot_bypasses_gate(monkeypatch) -> None:
    monkeypatch.delenv("TLLM_LOG_MEM_PROFILE", raising=False)
    warning = Mock()
    nvml_query = Mock()
    monkeypatch.setattr(trace_log_utils.logger, "warning", warning)
    monkeypatch.setattr(trace_log_utils, "_format_nvml_process_fields", nvml_query)
    _mock_cuda_counters(monkeypatch)

    trace_log_utils.log_mem_snapshot("oom/model", force=True)

    assert warning.call_count == 1
    assert warning.call_args.args[0].startswith("[mem-profile/oom/model]")
    assert not trace_log_utils._MEM_HISTORY
    nvml_query.assert_not_called()


def test_forced_mem_snapshot_adds_nvml_fields_when_profile_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    warning = Mock()
    nvml_query = Mock(return_value=" nvml_status=ok nvml_process_count=2")
    monkeypatch.setattr(trace_log_utils.logger, "warning", warning)
    monkeypatch.setattr(trace_log_utils, "_format_nvml_process_fields", nvml_query)
    _mock_cuda_counters(monkeypatch)

    trace_log_utils.log_mem_snapshot("oom/model", force=True)

    nvml_query.assert_called_once_with(2)
    assert warning.call_args.args[0].endswith("nvml_status=ok nvml_process_count=2")
    assert not trace_log_utils._MEM_HISTORY


def test_nvml_process_fields_use_uuid_and_split_self_from_nonself(monkeypatch) -> None:
    gib = 1 << 30
    handle = object()
    pynvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(return_value=handle),
        nvmlDeviceGetComputeRunningProcesses=Mock(
            return_value=[
                SimpleNamespace(pid=101, usedGpuMemory=3 * gib),
                SimpleNamespace(pid=202, usedGpuMemory=5 * gib),
                SimpleNamespace(pid=303, usedGpuMemory=2 * gib),
            ]
        ),
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setattr(trace_log_utils.os, "getpid", Mock(return_value=101))
    monkeypatch.setattr(
        trace_log_utils.torch.cuda,
        "get_device_properties",
        Mock(return_value=SimpleNamespace(uuid="01234567-89ab-cdef-0123-456789abcdef")),
    )

    fields = trace_log_utils._format_nvml_process_fields(2)

    pynvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(
        "GPU-01234567-89ab-cdef-0123-456789abcdef"
    )
    pynvml.nvmlDeviceGetComputeRunningProcesses.assert_called_once_with(handle)
    pynvml.nvmlShutdown.assert_called_once_with()
    assert fields == (
        " nvml_status=ok nvml_self_found=1 nvml_self_used=3.00GiB"
        " nvml_nonself_used=7.00GiB nvml_process_count=3"
        " nvml_processes=202:5.00GiB,101:3.00GiB,303:2.00GiB"
    )


def test_nvml_process_fields_bound_output_and_report_partial_data(monkeypatch) -> None:
    gib = 1 << 30
    processes = [
        SimpleNamespace(pid=100 + index, usedGpuMemory=(10 - index) * gib) for index in range(10)
    ]
    processes.append(SimpleNamespace(pid=999, usedGpuMemory=None))
    pynvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(return_value=object()),
        nvmlDeviceGetComputeRunningProcesses=Mock(return_value=processes),
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)
    monkeypatch.setattr(trace_log_utils.os, "getpid", Mock(return_value=100))
    monkeypatch.setattr(
        trace_log_utils.torch.cuda,
        "get_device_properties",
        Mock(return_value=SimpleNamespace(uuid="GPU-device-uuid")),
    )

    fields = trace_log_utils._format_nvml_process_fields(0)

    assert "nvml_status=partial" in fields
    assert "nvml_process_count=10" in fields
    assert "nvml_processes_omitted=2" in fields
    assert "nvml_processes_unavailable=1" in fields
    assert "100:10.00GiB" in fields
    assert "107:3.00GiB" in fields
    assert "108:2.00GiB" not in fields


def test_nvml_process_fields_report_query_failure_without_raising(monkeypatch) -> None:
    pynvml = SimpleNamespace(
        nvmlInit=Mock(side_effect=RuntimeError("NVML unavailable")),
        nvmlShutdown=Mock(),
    )
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)

    fields = trace_log_utils._format_nvml_process_fields(0)

    assert fields == " nvml_status=unavailable nvml_error=RuntimeError"
    pynvml.nvmlShutdown.assert_not_called()


def test_mem_history_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    monkeypatch.setattr(trace_log_utils.logger, "info", Mock())
    _mock_cuda_counters(monkeypatch)

    for index in range(trace_log_utils._MEM_HISTORY_CAPACITY + 1):
        trace_log_utils.log_mem_snapshot(f"stage/{index}")

    assert len(trace_log_utils._MEM_HISTORY) == trace_log_utils._MEM_HISTORY_CAPACITY
    assert trace_log_utils._MEM_HISTORY[0][1].startswith("[mem-profile/stage/1]")
    assert trace_log_utils._MEM_HISTORY[-1][1].startswith("[mem-profile/stage/64]")


def test_mem_history_logs_capture_age(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_LOG_MEM_PROFILE", "1")
    monkeypatch.setattr(trace_log_utils.logger, "rank", 3)
    monkeypatch.setattr(trace_log_utils.logger, "info", Mock())
    warning = Mock()
    monkeypatch.setattr(trace_log_utils.logger, "warning", warning)
    monkeypatch.setattr(
        trace_log_utils.time,
        "monotonic_ns",
        Mock(side_effect=[1_000_000_000, 3_500_000_000]),
    )
    _mock_cuda_counters(monkeypatch)

    trace_log_utils.log_mem_snapshot("stage/model")
    trace_log_utils.log_mem_history("oom/kv_cache")

    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert message.startswith(
        "[mem-history/oom/kv_cache] index=0 entries=1 "
        "age_ms=2500.00 snapshot=[mem-profile/stage/model]"
    )
    assert "rank=3 device=2" in message
    assert "torch_reserved=2.00GiB" in message
    assert "device_gap_estimate=2.00GiB" in message


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
