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
"""Host-side helpers of the kernel launches: the buffers of a call in the caller's workspace, global-memory pointers,
the current stream, the SM count and the compile cache."""

import functools
import math

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_ptr

# Every buffer of a workspace starts at a multiple of ALIGN bytes of the workspace, whose own address is a multiple
# of ALIGN: the alignment that the launches assume of their pointers (gmem_ptr)
ALIGN = 256


def _aligned_bytes(dtype, shape):
    return -(-dtype.itemsize * math.prod(shape) // ALIGN) * ALIGN


def workspace_bytes(layout):
    """Bytes of a workspace that holds the buffers of a layout (name -> (dtype, shape), a dict in the order of the
    buffers in the workspace)."""
    return sum(_aligned_bytes(dtype, shape) for dtype, shape in layout.values())


def carve(workspace, layout):
    """The buffers of a layout as views of a workspace, name -> tensor of the buffer's dtype and shape: the 1-D uint8
    tensor is read through one view per dtype, and each buffer is the contiguous block of that view at its offset.
    The content of the workspace is left as it is. ValueError for a workspace that is too small or whose address is
    not a multiple of ALIGN."""
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


def gmem_ptr(dtype, t, align=256):
    """Pointer to tensor t's data, assumed aligned to `align` bytes."""
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def current_stream(device):
    """The current PyTorch stream of `device` as a driver stream: every launch goes there."""
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


@functools.cache
def num_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def compiled(cache, key, make, *args, **kwargs):
    """cache[key], compiled for args from make() on first use (make builds the kernel object, or returns the jit
    function, only on a miss)."""
    fn = cache.get(key)
    if fn is None:
        fn = cache[key] = cute.compile(make(), *args, **kwargs)
    return fn
