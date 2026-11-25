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
"""Gated residual + LayerNorm / AdaLN / RMSNorm + static FP8 quantization in one row kernel
(``trtllm::fused_residual_norm_quant``)."""

from enum import IntEnum
from typing import Optional

import torch


class ResidualNormMode(IntEnum):
    """Modes of residual_norm_quant (values match the C++ ResidualNormMode)."""

    RESIDUAL_ONLY = 0
    LAYER_NORM = 1
    ADA_LAYER_NORM = 2
    RMS_NORM = 3


def residual_norm_quant(
    x: torch.Tensor,
    mode: ResidualNormMode,
    rows_per_batch: int,
    *,
    y: Optional[torch.Tensor] = None,
    y_bias: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    inv_out_scale: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Gated residual and LayerNorm / AdaLN + static FP8 quant of [..., hidden] rows in one pass.

    RESIDUAL_ONLY returns (x + gate * (y + y_bias), None).
    LAYER_NORM: affine weight / bias [hidden]; ADA_LAYER_NORM: weight (scale) / bias (shift) of
    shape [batch, hidden]; RMS_NORM: weight [hidden] (bf16 values), no bias. Returns (updated x
    or None without y, e4m3 normalized rows).
    """
    shape = x.shape
    hidden = shape[-1]

    def rows(
        t: Optional[torch.Tensor], dtype: Optional[torch.dtype] = None
    ) -> Optional[torch.Tensor]:
        """Contiguous [rows, hidden] view, optionally cast; None passes through."""
        return None if t is None else t.to(dtype or t.dtype).reshape(-1, hidden).contiguous()

    x_out, normed = torch.ops.trtllm.fused_residual_norm_quant(
        rows(x),
        rows(y),
        y_bias,
        rows(gate, torch.float32),
        mode,
        rows(weight, torch.float32),
        rows(bias, torch.float32),
        rows_per_batch,
        eps,
        None if inv_out_scale is None else inv_out_scale.reshape(1).to(torch.float32),
    )
    return (
        x_out.view(shape) if y is not None else None,
        normed.view(shape) if mode != ResidualNormMode.RESIDUAL_ONLY else None,
    )
