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


def _resolve_out_dtype(out_dtype: Optional[str]) -> Optional[torch.dtype]:
    """Resolve a torch dtype name (e.g. ``"float8_e4m3fn"``) to ``torch.dtype``."""
    if out_dtype is None:
        return None
    try:
        return getattr(torch, out_dtype)
    except AttributeError as e:
        raise RuntimeError(
            f"Unsupported out_dtype={out_dtype!r}; expected a valid torch dtype name."
        ) from e


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
    out_dtype: Optional[str] = None,
) -> torch.Tensor:
    """Fused SiLU+Mul using TRT-LLM's Triton kernel.

    Supports optional fused FP8 output quantization: when ``scale`` and
    ``out_dtype`` are provided, the kernel quantizes the output to the given
    dtype in-place (avoiding a separate quant pass).

    Args:
        x: Input tensor of shape ``(..., 2*D)``.
        scale: Optional per-tensor quantization scale.
        out_dtype: Optional output dtype as a ``torch`` attribute name
            (e.g. ``"float8_e4m3fn"``).  When set together with ``scale``, the
            kernel produces a quantized output of this dtype.
    """
    torch_dtype = _resolve_out_dtype(out_dtype)
    # trtllm::silu_and_mul expects 2D (rows, 2*D). Flatten higher dims and restore.
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    out_2d = torch.ops.trtllm.silu_and_mul(x_2d, scale=scale, dtype=torch_dtype)
    return out_2d.reshape(*orig_shape[:-1], out_2d.shape[-1])


@trtllm_silu_and_mul.register_fake
def _trtllm_fake(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    out_dtype: Optional[str] = None,
) -> torch.Tensor:
    half_size = x.shape[-1] // 2
    output_shape = list(x.shape[:-1]) + [half_size]
    return x.new_empty(output_shape, dtype=_resolve_out_dtype(out_dtype) or x.dtype)
