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
"""Host-side helpers of the kernel launches: workspace buffers, pointers, the stream, SM count and compile cache."""

import functools
import math
import os
import time
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

from tensorrt_llm.logger import logger

from .constants import ALIGN

Layout = dict[str, tuple[torch.dtype, tuple[int, ...]]]


def _aligned_bytes(dtype: torch.dtype, shape: tuple[int, ...]) -> int:
    return -(-dtype.itemsize * math.prod(shape) // ALIGN) * ALIGN


def workspace_bytes(layout: Layout) -> int:
    """Return the bytes of a workspace that holds the buffers of a layout, name -> (dtype, shape) in workspace order."""
    return sum(_aligned_bytes(dtype, shape) for dtype, shape in layout.values())


def carve(workspace: torch.Tensor, layout: Layout) -> dict[str, torch.Tensor]:
    """Return the buffers of a layout as views of a workspace: name -> tensor of the buffer's dtype and shape.

    * The 1-D uint8 workspace is read through one view per dtype; each buffer is the contiguous block of that view at
      its offset. The content of the workspace is left as it is.
    * ValueError for a workspace that is too small or whose address is not a multiple of ALIGN.
    """
    sizes = [_aligned_bytes(dtype, shape) for dtype, shape in layout.values()]
    total = sum(sizes)
    if workspace.numel() < total or workspace.data_ptr() % ALIGN:
        raise ValueError(
            f"the workspace holds {workspace.numel()} bytes and this call needs {total} bytes (workspace_size) at an "
            f"address that is a multiple of {ALIGN}: the workspace is too small or not aligned to {ALIGN} bytes"
        )
    whole = workspace[:total]
    start = whole.storage_offset()
    typed, views, offset = {}, {}, 0
    for (name, (dtype, shape)), size in zip(layout.items(), sizes):
        base = typed.get(dtype)
        if base is None:
            base = typed[dtype] = whole.view(dtype)
        strides, step = [], 1
        for n in reversed(shape):
            strides.append(step)
            step *= n
        strides.reverse()
        views[name] = base.as_strided(shape, strides, (start + offset) // dtype.itemsize)
        offset += size
    return views


def gmem_ptr(dtype: type, t: torch.Tensor, align: int = ALIGN) -> cute.Pointer:
    """Return a pointer to tensor t's data, assumed aligned to `align` bytes."""
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def current_stream(device: torch.device) -> cuda.CUstream:
    """Return the current PyTorch stream of `device` as a driver stream: every launch goes there."""
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def token_pitches(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, ...]:
    """Return the bytes between consecutive tokens of q, k and v: 256 H for a contiguous tensor, the row of the fused
    projection for a view into one."""
    return tuple(t.stride(1) * t.element_size() for t in (q, k, v))


def pdl_enabled() -> bool:
    """Return whether the launches use programmatic dependent launch: TRTLLM_ENABLE_PDL, on unless set to 0, read
    at every call as the kernels of TensorRT-LLM read it."""
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


def num_sms(device: torch.device | str) -> int:
    """Return the SM count of a device ("cuda": the current device), cached per device index."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _num_sms(index)


@functools.cache
def _num_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def compiled(
    cache: dict, key: tuple, make: Callable[[], object], *args: object, **kwargs: object
) -> Callable:
    """Return cache[key], compiled for args from make() on first use (make builds the kernel object, or returns the jit
    function, only on a miss). A miss logs the key and the compile time; a miss under CUDA graph capture raises."""
    fn = cache.get(key)
    if fn is None:
        not_under_capture(torch.device("cuda"), f"kernel {key}")
        start = time.perf_counter()
        fn = cache[key] = cute.compile(make(), *args, **kwargs)
        logger.info(f"Compiled DeCo-Attention kernel {key} in {time.perf_counter() - start:.1f} s")
    return fn


def not_under_capture(device: torch.device, what: str) -> None:
    """Raise when `what` is first needed while the current CUDA stream is capturing a graph."""
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"DeCo-Attention {what} first needed during CUDA graph capture; "
            "run the call eagerly once before capturing"
        )
