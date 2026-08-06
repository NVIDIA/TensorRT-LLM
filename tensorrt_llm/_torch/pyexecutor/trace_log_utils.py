# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated trace/log utilities for pyexecutor.

Leaf module — no other pyexecutor file is imported here, so any consumer
(``_util``, ``model_engine``, ``model_loader``, ``resource_manager``)
can import freely without creating circular dependencies.
"""

import os
import time
from collections import deque

import torch

from tensorrt_llm.logger import logger

_GIB = 1 << 30
_MEM_HISTORY_CAPACITY = 64
_MAX_NVML_PROCESSES = 8
_MEM_HISTORY: deque[tuple[int, str]] = deque(maxlen=_MEM_HISTORY_CAPACITY)


def reset_mem_history() -> None:
    """Clear snapshots left by a previous executor in this process."""
    _MEM_HISTORY.clear()


def log_mem_history(reason: str) -> None:
    """Log bounded pre-failure snapshots from oldest to newest."""
    try:
        history = tuple(_MEM_HISTORY)
        if not history:
            return
        now_ns = time.monotonic_ns()
        for index, (timestamp_ns, snapshot) in enumerate(history):
            age_ms = (now_ns - timestamp_ns) / 1_000_000
            logger.warning(
                f"[mem-history/{reason}] index={index} entries={len(history)} "
                f"age_ms={age_ms:.2f} snapshot={snapshot}"
            )
    except Exception as error:
        try:
            logger.warning(f"[mem-history/{reason}] history unavailable: {type(error).__name__}")
        except Exception:
            pass


def _format_nvml_process_fields(device: int) -> str:
    """Return best-effort per-process GPU memory fields for an OOM log."""
    initialized = False
    try:
        import pynvml

        pynvml.nvmlInit()
        initialized = True

        raw_uuid = str(torch.cuda.get_device_properties(device).uuid)
        if not raw_uuid or raw_uuid == "None":
            raise RuntimeError("CUDA device UUID is unavailable")
        device_uuid = raw_uuid if raw_uuid.startswith(("GPU-", "MIG-")) else f"GPU-{raw_uuid}"
        handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

        process_memory: dict[int, int] = {}
        unavailable_count = 0
        for process in processes:
            used = process.usedGpuMemory
            if used is None or used < 0:
                unavailable_count += 1
                continue
            pid = int(process.pid)
            process_memory[pid] = max(process_memory.get(pid, 0), int(used))

        self_pid = os.getpid()
        self_used = process_memory.get(self_pid, 0)
        nonself_used = sum(used for pid, used in process_memory.items() if pid != self_pid)
        sorted_processes = sorted(process_memory.items(), key=lambda item: (-item[1], item[0]))
        shown_processes = sorted_processes[:_MAX_NVML_PROCESSES]
        process_list = ",".join(f"{pid}:{used / _GIB:.2f}GiB" for pid, used in shown_processes)
        if not process_list:
            process_list = "none"

        status = "partial" if unavailable_count else "ok"
        fields = (
            f" nvml_status={status}"
            f" nvml_self_found={int(self_pid in process_memory)}"
            f" nvml_self_used={self_used / _GIB:.2f}GiB"
            f" nvml_nonself_used={nonself_used / _GIB:.2f}GiB"
            f" nvml_process_count={len(process_memory)}"
            f" nvml_processes={process_list}"
        )
        omitted_count = len(sorted_processes) - len(shown_processes)
        if omitted_count:
            fields += f" nvml_processes_omitted={omitted_count}"
        if unavailable_count:
            fields += f" nvml_processes_unavailable={unavailable_count}"
        return fields
    except Exception as error:
        return f" nvml_status=unavailable nvml_error={type(error).__name__}"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def log_mem_snapshot(tag: str, *, force: bool = False, **extra) -> None:
    """Log a compact snapshot of Torch and whole-device GPU memory.

    Routine snapshots are gated by ``TLLM_LOG_MEM_PROFILE=1`` and retained in
    a bounded history. ``force=True`` bypasses the gate for one current
    failure-path snapshot and is not retained. When profiling is enabled, a
    forced snapshot also adds best-effort NVML process memory fields.

    Prints these fields:

    - ``torch_alloc``         = :func:`torch.cuda.memory_allocated`
    - ``torch_reserved``      = :func:`torch.cuda.memory_reserved`
    - ``torch_alloc_peak``    = :func:`torch.cuda.max_memory_allocated`
    - ``torch_reserved_peak`` = :func:`torch.cuda.max_memory_reserved`
    - ``device_free``         = ``cuMemGetInfo().free``
    - ``device_total``        = ``cuMemGetInfo().total``

    ``device_gap_estimate`` is the signed difference between whole-device used
    memory and this process's Torch reserved memory. It can include other
    processes and non-Torch allocations, so it is not an ownership ledger.
    """
    profile_enabled = os.environ.get("TLLM_LOG_MEM_PROFILE", "") == "1"
    if not force and not profile_enabled:
        return
    try:
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        alloc_peak = torch.cuda.max_memory_allocated()
        reserved_peak = torch.cuda.max_memory_reserved()
        device_used = total - free
        device_gap_estimate = device_used - reserved
        extras = "".join(f" {key}={value}" for key, value in extra.items())
        nvml_fields = _format_nvml_process_fields(device) if force and profile_enabled else ""
        message = (
            f"[mem-profile/{tag}] "
            f"rank={logger.rank} device={device} "
            f"torch_alloc={alloc / _GIB:.2f}GiB "
            f"torch_reserved={reserved / _GIB:.2f}GiB "
            f"torch_alloc_peak={alloc_peak / _GIB:.2f}GiB "
            f"torch_reserved_peak={reserved_peak / _GIB:.2f}GiB "
            f"device_used={device_used / _GIB:.2f}GiB "
            f"device_free={free / _GIB:.2f}GiB "
            f"device_total={total / _GIB:.2f}GiB "
            f"device_gap_estimate={device_gap_estimate / _GIB:.2f}GiB"
            f"{extras}"
            f"{nvml_fields}"
        )
        if profile_enabled and not force:
            _MEM_HISTORY.append((time.monotonic_ns(), message))
        if force:
            logger.warning(message)
        else:
            logger.info(message)
    except Exception as error:
        # A diagnostic must not replace the failure it is trying to explain.
        try:
            logger.warning(f"[mem-profile/{tag}] snapshot unavailable: {type(error).__name__}")
        except Exception:
            pass


def log_tensor_size(tag: str, tensor: torch.Tensor, **extra) -> None:
    """Log a single tensor's footprint (shape / dtype / bytes) at a tag.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).

    Bytes = ``numel * element_size``. Any keyword arguments are appended
    as ``key=value`` for caller-specific context (e.g. routing config).
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    size_bytes = tensor.numel() * tensor.element_size()
    extras = "".join(f" {k}={v}" for k, v in extra.items())
    logger.info(
        f"[mem-profile/{tag}] "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"size={size_bytes / 1024 / 1024:.2f}MiB{extras}"
    )
