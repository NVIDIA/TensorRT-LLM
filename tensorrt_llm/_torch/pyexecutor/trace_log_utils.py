# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated trace/log utilities for pyexecutor.

Leaf module — no other pyexecutor file is imported here, so any consumer
(``_util``, ``model_engine``, ``model_loader``, ``resource_manager``)
can import freely without creating circular dependencies.
"""

import os

import torch

from tensorrt_llm.logger import logger

_GIB = 1 << 30


def log_mem_snapshot(tag: str, *, force: bool = False) -> None:
    """Log a compact snapshot of Torch and whole-device GPU memory.

    Routine snapshots are gated by ``TLLM_LOG_MEM_PROFILE=1``. ``force=True``
    bypasses the gate for failure-path diagnostics.

    Prints these fields:

    - ``torch_alloc``         = :func:`torch.cuda.memory_allocated`
    - ``torch_reserved``      = :func:`torch.cuda.memory_reserved`
    - ``torch_alloc_peak``    = :func:`torch.cuda.max_memory_allocated`
    - ``torch_reserved_peak`` = :func:`torch.cuda.max_memory_reserved`
    - ``free``                = ``cuMemGetInfo().free``
    - ``total``               = ``cuMemGetInfo().total``

    ``device_gap_estimate`` is the signed difference between whole-device used
    memory and this process's Torch reserved memory. It can include other
    processes and non-Torch allocations, so it is not an ownership ledger.
    """
    if not force and os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
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
        )
        if force:
            logger.warning(message)
        else:
            logger.info(message)
    except Exception as error:
        # A diagnostic must not replace the failure it is trying to explain.
        logger.warning(f"[mem-profile/{tag}] snapshot unavailable: {type(error).__name__}")


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
