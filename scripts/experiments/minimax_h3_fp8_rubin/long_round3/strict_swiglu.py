# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the strict SwiGLU candidate for the restricted static-FP8 H3 experiment."""

import hashlib
import os
from pathlib import Path

import torch

CANDIDATE = "7a97c80df0921df22222c894743f4b7a7a96fd3f919645d0584ee0a7d5af884b"
_module = None


def load():
    """Compile for an actual SM107 device, preserving the measured math flags."""
    global _module
    if _module is None:
        import tvm_ffi.cpp

        if torch.cuda.get_device_capability() != (10, 7):
            raise RuntimeError("This research candidate requires an actual SM107 GPU")
        os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.7")
        if os.environ["TVM_FFI_CUDA_ARCH_LIST"] != "10.7":
            raise ValueError("The strict H3 reproduction requires TVM_FFI_CUDA_ARCH_LIST=10.7")
        source = Path(__file__).with_suffix(".cu")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _module = tvm_ffi.cpp.load(
            name=f"h3_long_round3_strict_swiglu_{digest}",
            cuda_files=[str(source)],
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
            extra_ldflags=["-lcuda"],
        )
    return _module


def run(x, scale):
    """Map contiguous CUDA BF16 ``[M,28672]`` and one FP32 scale to FP8 ``[M,14336]``."""
    if not (
        x.ndim == 2
        and x.shape[1] == 28672
        and x.is_contiguous()
        and x.dtype == torch.bfloat16
        and x.is_cuda
    ):
        raise ValueError("Expected a contiguous CUDA BF16 [M,28672] activation")
    if not (
        scale.device == x.device
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and scale.is_contiguous()
    ):
        raise ValueError("Expected one contiguous FP32 scale on the activation's CUDA device")
    output = torch.empty((x.shape[0], 14336), device=x.device, dtype=torch.float8_e4m3fn)
    load().run(x, scale, output)
    return output
