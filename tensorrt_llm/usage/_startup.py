# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort startup context; never initializes CUDA or loads model files."""

import os
import re
import sys
import threading
import time
from collections.abc import Mapping

_GPU_DISCOVERY_TIMEOUT = 0.05
REPORT_CONTEXT = "pre_initialization_exit"
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_PARALLEL_FIELDS = {
    "tensor_parallel_size": "tensorParallelSize",
    "pipeline_parallel_size": "pipelineParallelSize",
    "context_parallel_size": "contextParallelSize",
    "moe_expert_parallel_size": "moeExpertParallelSize",
    "moe_tensor_parallel_size": "moeTensorParallelSize",
}


def requested_fields(values: Mapping) -> dict:
    """Copy only independently valid, low-cardinality requested settings."""
    result = {}
    for name, alias in _PARALLEL_FIELDS.items():
        value = values.get(name)
        if type(value) is int and 0 < value <= 4_294_967_295:
            result[alias] = value
    for name, allowed in (
        ("backend", {"pytorch", "_autodeploy", "tensorrt"}),
        ("dtype", {"auto", "float16", "bfloat16", "float32"}),
    ):
        value = values.get(name)
        if type(value) is str and value in allowed:
            result[name] = value
    return result


def _nvml_gpu_fields() -> dict:
    """Resolve unambiguous visible physical GPUs without treating NVML indices as CUDA indices."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    container_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if visible in ("", "-1"):
        return {"gpuCount": 0, "_source": "visibility_mask"}
    if container_visible in ("", "none", "void"):
        return {}
    # MPS can remap devices independently of the client's visibility settings.
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        return {}

    import pynvml

    pynvml.nvmlInit()
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        uuids = [pynvml.nvmlDeviceGetUUID(handle) for handle in handles]
        by_uuid = dict(zip(uuids, handles))
        # Container runtimes may expose more devices to NVML than to CUDA.
        if container_visible not in (None, "all"):
            selected = container_visible.split(",")
            if not all(_GPU_UUID.fullmatch(item) and item in by_uuid for item in selected):
                return {}
            handles = [by_uuid[item] for item in selected]
        if visible is not None:
            selected = visible.split(",")
            if all(_GPU_UUID.fullmatch(item) and item in by_uuid for item in selected):
                if len(set(selected)) != len(selected):
                    return {}
                allowed = {pynvml.nvmlDeviceGetUUID(handle) for handle in handles}
                if not set(selected).issubset(allowed):
                    return {}
                handles = [by_uuid[item] for item in selected]
            elif not (visible == "0" and len(handles) == 1):
                return {}
        elif len(handles) != 1:
            # CUDA's default FASTEST_FIRST does not establish a first NVML device.
            return {}
        if not handles:
            return {}
        for handle in handles:
            try:
                if pynvml.nvmlDeviceGetMigMode(handle)[0] != 0:
                    return {}
            except pynvml.NVMLError_NotSupported:
                pass
        name = pynvml.nvmlDeviceGetName(handles[0])
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return {
            "gpuCount": len(handles),
            "gpuName": name[:256],
            "gpuMemoryMB": pynvml.nvmlDeviceGetMemoryInfo(handles[0]).total // (1024 * 1024),
        }
    finally:
        pynvml.nvmlShutdown()


def _gpu_fields() -> dict:
    """Prefer existing CUDA discovery; NVML must not initialize CUDA on its behalf."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return _nvml_gpu_fields()
    count = torch.cuda.device_count()
    fields = {"gpuCount": count, "_source": "cuda_initialized"}
    if count:
        properties = torch.cuda.get_device_properties(0)
        fields.update(
            gpuName=properties.name[:256], gpuMemoryMB=properties.total_memory // (1024 * 1024)
        )
    return fields


def bounded_gpu_fields(deadline: float) -> dict:
    """Spend at most 50 ms of the terminal deadline on optional hardware discovery."""
    discovery_deadline = min(time.monotonic() + _GPU_DISCOVERY_TIMEOUT, deadline)
    if discovery_deadline <= time.monotonic():
        return {}
    result = {}
    finished = threading.Event()

    def collect() -> None:
        try:
            result.update(_gpu_fields())
        except Exception:
            # Includes optional binding/driver failures; telemetry must not affect exit.
            pass
        finally:
            finished.set()

    threading.Thread(target=collect, daemon=True, name="trtllm-usage-nvml").start()
    # A timed-out native call cannot be cancelled. Its private result is discarded.
    return result.copy() if finished.wait(max(0, discovery_deadline - time.monotonic())) else {}
