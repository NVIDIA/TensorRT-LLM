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
"""Periodic device/host memory snapshot of the executor iteration log."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_memory_log as dml

_GIB = 1 << 30


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, dml.DEFAULT_DEVICE_MEM_LOG_INTERVAL),
        ("50", 50),
        ("0", 0),
        ("-1", -1),
        ("x", dml.DEFAULT_DEVICE_MEM_LOG_INTERVAL),
    ],
)
def test_interval_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(dml.DEVICE_MEM_LOG_INTERVAL_ENV, raising=False)
    else:
        monkeypatch.setenv(dml.DEVICE_MEM_LOG_INTERVAL_ENV, raw)
    assert dml.device_mem_log_interval() == expected


@pytest.mark.cpu_only
def test_host_mem_available(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  4194304 kB\nMemFree:  1024 kB\nMemAvailable:  2097152 kB\n")
    assert dml.host_mem_available_bytes(str(meminfo)) == 2 * _GIB
    assert dml.host_mem_available_bytes(str(tmp_path / "missing")) == -1


@pytest.mark.cpu_only
def test_snapshot_fields(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (5 * _GIB, 100 * _GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 60 * _GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 62 * _GIB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 40 * _GIB)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda device=None: {"num_device_alloc": 5200, "num_device_free": 1200},
    )
    monkeypatch.setattr(dml, "host_mem_available_bytes", lambda: 300 * _GIB)
    line = dml.device_memory_snapshot()
    assert line == (
        "device_free = 5.00GiB, device_total = 100.00GiB, torch_reserved = 60.00GiB, "
        "torch_max_reserved = 62.00GiB, torch_allocated = 40.00GiB, "
        "non_torch_used = 35.00GiB, torch_mappings = 4000, host_mem_available = 300.0GiB"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_snapshot_on_device():
    line = dml.device_memory_snapshot()
    fields = dict(item.split(" = ") for item in line.split(", "))
    assert set(fields) == {
        "device_free",
        "device_total",
        "torch_reserved",
        "torch_max_reserved",
        "torch_allocated",
        "non_torch_used",
        "torch_mappings",
        "host_mem_available",
    }
    assert float(fields["device_total"].removesuffix("GiB")) > 0
    assert int(fields["torch_mappings"]) >= 0
