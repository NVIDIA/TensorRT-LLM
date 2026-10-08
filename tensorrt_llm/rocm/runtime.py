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
"""RDNA4 runtime checks, without NVIDIA device discovery or architecture spoofing."""

import os
import shutil
from typing import Literal

import torch

RDNA4_ARCHITECTURES = frozenset(("gfx1200", "gfx1201"))


def architecture_name(name: str) -> str:
    """Remove ROCm feature suffixes from a GCN architecture name."""
    return name.split(":", 1)[0]


def resolve_device(device: str | torch.device = "cuda:0") -> torch.device:
    """Require real RDNA4 hardware for GPU execution; CPU is an explicit test mode.

    PyTorch deliberately uses the ``cuda`` device namespace on HIP builds. It
    does not mean CUDA libraries or NVIDIA hardware are used in this backend.
    """
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type != "cuda":
        raise ValueError(
            "ROCm execution supports an RDNA4 GPU ('cuda:N') or explicit 'cpu' testing"
        )
    if not torch.version.hip:
        raise RuntimeError(
            "Install an RDNA4-compatible ROCm PyTorch wheel; this PyTorch build has no HIP support"
        )
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        raise RuntimeError(
            "Remove HSA_OVERRIDE_GFX_VERSION: RDNA4 must use its real gfx1200/gfx1201 ISA"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "HIP cannot access a GPU. Check the ROCm driver, /dev/kfd, /dev/dri and group permissions"
        )
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"HIP logical device {index} does not exist")
    properties = torch.cuda.get_device_properties(index)
    arch = architecture_name(getattr(properties, "gcnArchName", "unknown"))
    if arch not in RDNA4_ARCHITECTURES:
        raise RuntimeError(f"Expected RDNA4 gfx1200/gfx1201, got {arch} ({properties.name})")
    return torch.device("cuda", index)


def resolve_dtype(
    dtype: str | torch.dtype,
    device: torch.device,
) -> torch.dtype:
    """Select a supported storage dtype, with FP32 accumulation in reference kernels."""
    if isinstance(dtype, torch.dtype):
        selected = dtype
    elif dtype == "auto":
        selected = torch.float32 if device.type == "cpu" else torch.bfloat16
    else:
        names = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype not in names:
            raise ValueError("dtype must be auto, float32, float16 or bfloat16")
        selected = names[dtype]
    if selected not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("RDNA4 inference supports FP32, FP16 and BF16 storage")
    return selected


def diagnostics() -> dict:
    """Return actionable installation/device information, even on a CPU-only host."""
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            arch = architecture_name(getattr(properties, "gcnArchName", "unknown"))
            devices.append(
                {
                    "logical_index": index,
                    "name": properties.name,
                    "architecture": arch,
                    "rdna4": arch in RDNA4_ARCHITECTURES,
                    "vram_total_bytes": properties.total_memory,
                }
            )
    return {
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "hipcc": shutil.which("hipcc"),
        "devices": devices,
        "ready": bool(
            torch.version.hip
            and devices
            and all(device["rdna4"] for device in devices)
            and not os.environ.get("HSA_OVERRIDE_GFX_VERSION")
        ),
        "device_namespace": "cuda (PyTorch HIP compatibility namespace)",
        "supported_architectures": sorted(RDNA4_ARCHITECTURES),
        "visible_devices": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "architecture_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
        "kernel_choices": ["torch", "hip"],
    }


KernelBackend = Literal["torch", "hip"]
