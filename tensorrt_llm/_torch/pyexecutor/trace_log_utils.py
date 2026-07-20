"""Gated trace/log utilities for pyexecutor.

Leaf module — no other pyexecutor file is imported here, so any consumer
(``_util``, ``model_engine``, ``model_loader``, ``resource_manager``)
can import freely without creating circular dependencies.
"""

import os

import torch

from tensorrt_llm.logger import logger

_GIB = 1 << 30


def log_mem_snapshot(tag: str) -> None:
    """Log Torch alloc/reserved + alloc/reserved peak + free/total GPU memory.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).

    Prints these fields:

    - ``torch_alloc``         = :func:`torch.cuda.memory_allocated`
    - ``torch_reserved``      = :func:`torch.cuda.memory_reserved`
    - ``torch_alloc_peak``    = :func:`torch.cuda.max_memory_allocated`
    - ``torch_reserved_peak`` = :func:`torch.cuda.max_memory_reserved`
    - ``free``                = ``cuMemGetInfo().free``
    - ``total``               = ``cuMemGetInfo().total``

    Derived quantities the reader may need:

    - ``used      = total - free`` — whole-process GPU consumption
    - ``slack     = reserved - alloc`` — Torch caching allocator free blocks
    - ``non_torch = used - reserved`` — bytes outside Torch (KV pool C++
      cudaMalloc, NCCL buffers, cuBLAS workspace, CUDA driver context,
      CUDA graph mempool, etc.)
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    alloc_peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    logger.info(
        f"[mem-profile/{tag}] "
        f"torch_alloc={alloc / _GIB:.2f}GiB "
        f"torch_reserved={reserved / _GIB:.2f}GiB "
        f"torch_alloc_peak={alloc_peak / _GIB:.2f}GiB "
        f"torch_reserved_peak={reserved_peak / _GIB:.2f}GiB "
        f"free={free / _GIB:.2f}GiB total={total / _GIB:.2f}GiB"
    )


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
