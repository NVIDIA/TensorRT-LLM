# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The current CUDA stream's raw handle, for the launch wrappers.

`torch.cuda.current_stream(device).cuda_stream` builds a `torch.cuda.Stream` (device-index
lookup, lazy-init check, a Python object) to read one integer: ~3 us per call on the serving
host, and a context step makes about twenty such calls per layer. With
TRTLLM_GLM52_CTX_HOST_DIET=1 the wrappers read the same handle from torch's raw getter
(`torch._C._cuda_getCurrentRawStream`, the call TorchInductor launches with): the current
stream of the same device, so every launch lands on the stream it landed on before, under
CUDA-graph capture too. Default on; TRTLLM_GLM52_CTX_HOST_DIET=0 restores the torch.cuda call.
"""

import os

import cuda.bindings.driver as cuda
import torch

#: Read once at import; the engine sets the switch before it loads the model.
DIET = os.environ.get("TRTLLM_GLM52_CTX_HOST_DIET", "1") == "1"
_raw = torch._C._cuda_getCurrentRawStream


def raw_stream(device=None) -> int:
    """`torch.cuda.current_stream(device).cuda_stream`. `device` is a CUDA `torch.device`
    (with or without an index), a device index, or None for the current device."""
    if not DIET:
        return torch.cuda.current_stream(device).cuda_stream
    if device is None:
        return _raw(torch.cuda.current_device())
    if isinstance(device, int):
        return _raw(device)
    index = device.index
    return _raw(torch.cuda.current_device() if index is None else index)


def cu_stream(device=None) -> cuda.CUstream:
    """`cuda.CUstream` of `raw_stream(device)`: what `cutlass.torch.current_stream()` returns
    for the current device."""
    return cuda.CUstream(raw_stream(device))
