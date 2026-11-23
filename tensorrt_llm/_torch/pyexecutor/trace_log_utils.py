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

    - ``used      = total - free`` — **device-wide** consumption. This is not
      this process's usage: TensorRT-LLM supports colocated processes (Ray
      fractional GPUs, GMS weight sharing, RLHF alongside a trainer,
      disaggregated ctx+gen on one device), whose memory is included here.
      The memory monitor's full mode (``memory_monitor_config.mode='full'``
      or ``TLLM_MEM_MONITOR=1``) adds the NVML per-process split.
    - ``slack     = reserved - alloc`` — reserved but not handed out. Besides
      fragmented space this covers blocks already requested for release that
      are waiting on other streams, so it is not all immediately reusable.
    - ``non_torch = used - reserved`` — bytes outside Torch (KV pool C++
      ``cuMemCreate``, NCCL buffers, cuBLAS workspace, CUDA driver context,
      lazily loaded module code, etc.). CUDA-graph pools are **not** in here:
      they use the native allocator and are already inside ``reserved``.
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
