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
"""Which path the native ``enorm``/``hnorm`` take, and whether draft_front mirrors it.

draft_front is bit-identical to the native draft front because its kernel reproduces the
FlashInfer CuTe DSL ``RMSNormKernel`` that TensorRT-LLM's ``RMSNorm.forward``
reaches for a 2-D BF16 input without a residual. ``RMSNorm.forward`` can take
other paths: the fused NVFP4-quantize epilogue, ``cuda.tile``, the gemma form
``x * rstd * (w + 1)``, or the torch fallback when FlashInfer is unavailable.
FlashInfer itself can select its CUDA JIT norm instead of the CuTe DSL kernel.
Each of those rounds differently from draft_front (gemma also changes the result), so
draft_front must decline before launching anything whenever one of them would run.

* :func:`norm_module_reason` covers the per-module attributes; only a caller
  holding the ``RMSNorm`` modules can check them (the ``DeepseekV3MTP`` adapter,
  once at construction).
* :func:`flashinfer_path_reason` covers the per-process and per-device
  conditions; :func:`~.draft_front.unsupported_reason` applies it on every call.
* :func:`native_norm_reason` combines both.
"""

import functools

import torch
from torch import nn

# FlashInfer releases whose CuTe DSL ``RMSNormKernel`` the draft_front kernel was written
# against and verified bit for bit (thread-value layout, reduction tree, rsqrt).
VERIFIED_FLASHINFER_VERSIONS = ("0.6.18",)


def norm_module_reason(name: str, norm: nn.Module) -> str | None:
    """Return why ``norm`` would not run the plain FlashInfer RMSNorm, or ``None``.

    Args:
        name: label used in the returned reason (``"enorm"`` or ``"hnorm"``).
        norm: the module the native draft front calls as ``norm(x)``.
    """
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not isinstance(norm, RMSNorm) or type(norm).forward is not RMSNorm.forward:
        return f"{name} is {type(norm).__name__}, not the TensorRT-LLM RMSNorm"
    if norm.use_cuda_tile:
        return f"{name} runs the cuda.tile RMSNorm"
    if norm.use_gemma:
        return f"{name} is a gemma RMSNorm (weight + 1)"
    if norm.is_nvfp4:
        return f"{name} can take the fused NVFP4-quantize RMSNorm"
    if norm.return_hp_output:
        return f"{name} requests a high-precision auxiliary output"
    return None


@functools.cache
def _flashinfer_path_reason(device_index: int) -> str | None:
    # Every input is fixed for the life of the process: the import-time
    # FlashInfer availability, its version, FLASHINFER_USE_CUDA_NORM (read when
    # flashinfer.norm is imported) and the device's compute capability.
    from tensorrt_llm._torch.modules import rms_norm

    if not rms_norm.IS_FLASHINFER_AVAILABLE:
        return "FlashInfer unavailable: native RMSNorm takes the torch path"
    import flashinfer
    import flashinfer.norm as flashinfer_norm

    version = getattr(flashinfer, "__version__", None)
    if version not in VERIFIED_FLASHINFER_VERSIONS:
        return (
            f"FlashInfer {version} not verified; the draft_front kernel mirrors the RMSNorm of "
            f"{', '.join(VERIFIED_FLASHINFER_VERSIONS)}"
        )
    use_cuda_norm = getattr(flashinfer_norm, "_use_cuda_norm", None)
    if use_cuda_norm is None:
        return "flashinfer.norm has no _use_cuda_norm dispatch"
    if use_cuda_norm(torch.device("cuda", device_index)):
        return (
            "FlashInfer selects its CUDA JIT RMSNorm (FLASHINFER_USE_CUDA_NORM=1, CuTe DSL "
            "unavailable, or device not supported by the CuTe DSL)"
        )
    return None


def flashinfer_path_reason(device: torch.device) -> str | None:
    """Return why a native BF16 ``RMSNorm`` on ``device`` would not run FlashInfer's
    CuTe DSL ``RMSNormKernel``, or ``None`` if it would.

    Host-side only; the result is cached per device for the life of the process.
    """
    if device.type != "cuda":
        return f"device {device} is not a CUDA device"
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _flashinfer_path_reason(index)


def native_norm_reason(
    enorm: nn.Module, hnorm: nn.Module, device: torch.device | None = None
) -> str | None:
    """Return why the native ``enorm``/``hnorm`` would not take the path draft_front mirrors.

    Args:
        enorm: the draft layer's embedding norm module.
        hnorm: the draft layer's hidden-state norm module.
        device: the CUDA device the norms run on; defaults to the current device.

    Returns:
        A reason string when draft_front must leave the draft front to the native path,
        else ``None``.
    """
    for name, norm in (("enorm", enorm), ("hnorm", hnorm)):
        reason = norm_module_reason(name, norm)
        if reason is not None:
            return reason
    if device is None:
        if not torch.cuda.is_available():
            return "CUDA unavailable"
        device = torch.device("cuda", torch.cuda.current_device())
    return flashinfer_path_reason(device)


__all__ = [
    "VERIFIED_FLASHINFER_VERSIONS",
    "flashinfer_path_reason",
    "native_norm_reason",
    "norm_module_reason",
]
