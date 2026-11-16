# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visibility, privacy and deadline checks for optional startup hardware discovery."""

import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm.usage import _startup

pytestmark = pytest.mark.cpu_only
UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in (1, 2)]


@pytest.mark.parametrize(
    "count,visible,container,mig,expected",
    [
        (1, None, None, False, 1),
        (1, "0", "all", False, 1),
        (2, "0", "all", False, None),
        (2, None, "all", False, None),
        (2, UUIDS[1], "all", False, 1),
        (2, ",".join(reversed(UUIDS)), "all", False, 2),
        (2, UUIDS[1], UUIDS[0], False, None),
        (1, "0", "0", False, None),
        (1, None, None, True, None),
        (1, "MIG-private", None, False, None),
        (1, "", None, False, 0),
        (1, None, "void", False, None),
    ],
)
def test_nvml_resolves_only_known_visibility(monkeypatch, count, visible, container, mig, expected):
    for name, value in (("CUDA_VISIBLE_DEVICES", visible), ("NVIDIA_VISIBLE_DEVICES", container)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: count,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda index: UUIDS[index],
        nvmlDeviceGetMigMode=lambda handle: (int(mig), 0),
        nvmlDeviceGetName=lambda handle: f"GPU {handle}",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=1024**3),
        NVMLError_NotSupported=NotImplementedError,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    fields = _startup._nvml_gpu_fields()
    assert fields.get("gpuCount") == expected
    assert not any(uuid in str(fields) for uuid in UUIDS)
    if expected and visible and visible.startswith(UUIDS[1]):
        assert fields["gpuName"] == "GPU 1"


def test_nvml_timeout_discards_late_result(monkeypatch):
    release = threading.Event()
    finished = threading.Event()

    def slow():
        try:
            release.wait(2)
            return {"gpuCount": 99}
        finally:
            finished.set()

    monkeypatch.setattr(_startup, "_gpu_fields", slow)
    monkeypatch.setattr(_startup, "_GPU_DISCOVERY_TIMEOUT", 0.01)
    start = time.monotonic()
    result = _startup.bounded_gpu_fields(start + 0.5)
    try:
        assert result == {}
        assert time.monotonic() - start < 0.25
    finally:
        release.set()
        assert finished.wait(1)
        for thread in threading.enumerate():
            if thread.name == "trtllm-usage-nvml":
                thread.join(1)
    assert result == {}


def test_missing_nvml_and_expired_budget(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    assert _startup.bounded_gpu_fields(time.monotonic() + 0.5) == {}
    assert _startup.bounded_gpu_fields(time.monotonic() - 1) == {}


def test_existing_cuda_discovery_has_priority(monkeypatch):
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda index: SimpleNamespace(
            name="Assigned GPU", total_memory=1024**3
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    fields = _startup.bounded_gpu_fields(time.monotonic() + 0.5)
    assert fields == {
        "gpuCount": 1,
        "gpuName": "Assigned GPU",
        "gpuMemoryMB": 1024,
        "_source": "cuda_initialized",
    }
