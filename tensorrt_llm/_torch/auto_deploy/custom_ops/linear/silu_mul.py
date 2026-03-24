# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fused SiLU+Mul custom operations for graph transformation.

Replaces the pattern: silu(narrow(x, 0, half)) * narrow(x, half, half)
with a single fused kernel that avoids materializing the narrow views.

Two backends are available:
- ``flashinfer``: Uses FlashInfer's fused kernel (default).
- ``trtllm``: Uses TRT-LLM's Triton kernel (``torch.ops.trtllm.silu_and_mul``),
  which natively supports fused FP8 quantization via ``scale`` and ``dtype`` params.
"""

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul
except ImportError:
    _flashinfer_silu_and_mul = None


@torch.library.custom_op("auto_deploy::flashinfer_silu_and_mul", mutates_args=())
def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SiLU+Mul using FlashInfer's kernel (fallback to manual implementation)."""
    if _flashinfer_silu_and_mul is not None:
        return _flashinfer_silu_and_mul(x)
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


@flashinfer_silu_and_mul.register_fake
def _flashinfer_fake(x: torch.Tensor) -> torch.Tensor:
    half_size = x.shape[-1] // 2
    output_shape = list(x.shape[:-1]) + [half_size]
    return x.new_empty(output_shape, dtype=x.dtype)


@torch.library.custom_op("auto_deploy::trtllm_silu_and_mul", mutates_args=())
def trtllm_silu_and_mul(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    dtype: Optional[int] = None,
) -> torch.Tensor:
    """Fused SiLU+Mul using TRT-LLM's Triton kernel.

    Supports optional fused FP8 quantization: when ``scale`` and ``dtype`` are provided,
    the output is quantized to the given dtype in-kernel (avoiding a separate quant pass).

    Args:
        x: Input tensor of shape ``(..., 2*D)``.
        scale: Optional quantization scale tensor for fused output quantization.
        dtype: Optional output dtype encoded as int (use ``_DTYPE_TO_INT``/``_INT_TO_DTYPE``).
              When set with ``scale``, output is quantized in-kernel.
    """
    torch_dtype = _INT_TO_DTYPE.get(dtype) if dtype is not None else None
    # trtllm::silu_and_mul expects 2D (rows, 2*D). Flatten higher dims and restore.
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    out_2d = torch.ops.trtllm.silu_and_mul(x_2d, scale=scale, dtype=torch_dtype)
    return out_2d.reshape(*orig_shape[:-1], out_2d.shape[-1])


@trtllm_silu_and_mul.register_fake
def _trtllm_fake(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    dtype: Optional[int] = None,
) -> torch.Tensor:
    half_size = x.shape[-1] // 2
    output_shape = list(x.shape[:-1]) + [half_size]
    out_dtype = _INT_TO_DTYPE.get(dtype, x.dtype) if dtype is not None else x.dtype
    return x.new_empty(output_shape, dtype=out_dtype)


# Legacy alias — kept so that existing graph references continue to work.
@torch.library.custom_op("auto_deploy::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Legacy fused SiLU+Mul (delegates to FlashInfer backend)."""
    return flashinfer_silu_and_mul(x)


@silu_and_mul.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    half_size = x.shape[-1] // 2
    output_shape = list(x.shape[:-1]) + [half_size]
    return x.new_empty(output_shape, dtype=x.dtype)


# Dtype encoding for custom op (torch.dtype is not supported in custom op schemas).
_DTYPE_TO_INT = {
    torch.float8_e4m3fn: 1,
    torch.bfloat16: 2,
    torch.float16: 3,
}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_TO_INT.items()}
