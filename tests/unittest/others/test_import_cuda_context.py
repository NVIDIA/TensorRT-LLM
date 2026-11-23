# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import must leave visible GPUs unused until an engine is constructed."""

import subprocess
import sys

import pytest
import torch

_DRIVER_PROBE = r"""
import ctypes
import sys

cuda = ctypes.CDLL("libcuda.so.1")

def check(status):
    assert status == 0, f"CUDA driver error {status}"

check(cuda.cuInit(0))
count = ctypes.c_int()
check(cuda.cuDeviceGetCount(ctypes.byref(count)))
assert count.value > 0, "A visible CUDA device is required"
devices = []
for ordinal in range(count.value):
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), ordinal))
    devices.append(device)

def context_state():
    active_devices = []
    for device in devices:
        flags = ctypes.c_uint()
        active = ctypes.c_int()
        check(cuda.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(active)))
        active_devices.append(bool(active.value))
    current = ctypes.c_void_p()
    check(cuda.cuCtxGetCurrent(ctypes.byref(current)))
    return active_devices, bool(current.value)

before, current = context_state()
assert not any(before) and not current, "The fresh process already has a CUDA context"
if sys.argv[1] == "allocate":
    import torch
    allocations = [torch.empty(1, device=f"cuda:{ordinal}") for ordinal in range(count.value)]
    active, current = context_state()
    assert all(active), "Observer missed a deliberately allocated CUDA primary context"
    assert current, "Observer missed the allocation thread's current CUDA context"
else:
    import tensorrt_llm
    active, current = context_state()
    assert not any(active) and not current, "Package import created a CUDA context on a visible device"
    from tensorrt_llm import LLM
    assert callable(LLM)
    active, current = context_state()
    assert not any(active) and not current, "LLM import created a CUDA context on a visible device"
print("CUDA-CONTEXT-PROBE-OK")
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a visible CUDA device")
def test_import_does_not_create_cuda_context_with_visible_device() -> None:
    # Driver state is process-local, so this also works in PID namespaces where
    # NVML reports host PIDs. Use a separate child for the positive control to
    # keep the import under test entirely cold.
    for mode in ("allocate", "import"):
        result = subprocess.run(
            [sys.executable, "-c", _DRIVER_PROBE, mode],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"{mode} probe failed:\n{result.stdout}\n{result.stderr}"
        assert "CUDA-CONTEXT-PROBE-OK" in result.stdout
