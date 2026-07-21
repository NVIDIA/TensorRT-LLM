# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, call

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType


def test_startup_monitor_warns_about_visible_gpu_processes(monkeypatch) -> None:
    gib = 1 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(50 * gib, 100 * gib)))
    monkeypatch.setattr(
        torch.cuda,
        "list_gpu_processes",
        Mock(return_value="GPU: 0\nprocess 123 uses 4096.000 MB GPU memory"),
    )
    warning = Mock()
    snapshot = Mock()
    monkeypatch.setattr(py_executor_creator.logger, "warning", warning)
    monkeypatch.setattr(py_executor_creator, "log_mem_snapshot", snapshot)

    py_executor_creator._ExecutorMemoryMonitor()

    snapshot.assert_called_once_with("startup/baseline")
    message = warning.call_args.args[0]
    assert "50.00 GiB already used" in message
    assert "process 123 uses 4096.000 MB GPU memory" in message
    assert "\n" not in message


def test_startup_monitor_skips_process_query_for_clean_gpu(monkeypatch) -> None:
    gib = 1 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(95 * gib, 100 * gib)))
    process_query = Mock()
    warning = Mock()
    monkeypatch.setattr(torch.cuda, "list_gpu_processes", process_query)
    monkeypatch.setattr(py_executor_creator.logger, "warning", warning)
    monkeypatch.setattr(py_executor_creator, "log_mem_snapshot", Mock())

    py_executor_creator._ExecutorMemoryMonitor()

    process_query.assert_not_called()
    warning.assert_not_called()


def test_startup_monitor_logs_successful_stage(monkeypatch) -> None:
    gib = 1 << 30
    mem_get_info = Mock(
        side_effect=[
            (100 * gib, 100 * gib),
            (80 * gib, 100 * gib),
            (70 * gib, 100 * gib),
        ]
    )
    snapshot = Mock()
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(py_executor_creator, "log_mem_snapshot", snapshot)
    monitor = py_executor_creator._ExecutorMemoryMonitor()

    with monitor.observe_creation_stage(ExecutorMemoryType.SAMPLER):
        pass

    assert snapshot.call_args_list == [
        call("startup/baseline"),
        call("stage/sampler"),
    ]
    assert len(monitor._samples) == 1
    assert monitor._samples[0].free_gpu_memory_bytes_pre == 80 * gib
    assert monitor._samples[0].free_gpu_memory_bytes_post == 70 * gib


def test_startup_monitor_forces_snapshot_without_masking_oom(monkeypatch) -> None:
    gib = 1 << 30
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        Mock(
            side_effect=[
                (100 * gib, 100 * gib),
                (1 * gib, 100 * gib),
            ]
        ),
    )
    snapshot = Mock()
    monkeypatch.setattr(py_executor_creator, "log_mem_snapshot", snapshot)
    monitor = py_executor_creator._ExecutorMemoryMonitor()
    oom = torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError) as raised:
        with monitor.observe_creation_stage(ExecutorMemoryType.MODEL_ENGINE_MAIN):
            raise oom

    assert raised.value.__cause__ is oom
    assert snapshot.call_args_list == [
        call("startup/baseline"),
        call("oom/model", force=True),
    ]
