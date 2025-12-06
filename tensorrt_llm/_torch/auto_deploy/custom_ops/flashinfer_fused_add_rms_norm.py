# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flashinfer
import torch

from ...flashinfer_utils import get_env_enable_pdl


@torch.library.custom_op(
    "auto_deploy::flashinfer_fused_add_rms_norm_inplace", mutates_args={"x", "residual"}
)
def flashinfer_fused_add_rms_norm_inplace(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """
    Fused Add + RMSNorm operation using FlashInfer (In-place).
    Computes in-place:
        residual = x + residual (sum)
        x = rms_norm(residual, weight, eps) (normalized)

    Returns None.
    """
    # FlashInfer expects 2D inputs (batch*seq_len, hidden_size)
    x_shape = x.shape
    residual_shape = residual.shape
    x_flat = x.view(-1, x.shape[-1])
    residual_flat = residual.view(-1, residual.shape[-1])

    flashinfer.norm.fused_add_rmsnorm(
        x_flat, residual_flat, weight, eps, enable_pdl=get_env_enable_pdl()
    )
    x_flat.view(x_shape)
    residual_flat.view(residual_shape)
    return


@flashinfer_fused_add_rms_norm_inplace.register_fake
def _(x, residual, weight, eps):
    return


def flashinfer_fused_add_rms_norm(x, residual, weight, eps):
    """Wrapper that calls the in-place op and returns the modified tensors."""
    torch.ops.auto_deploy.flashinfer_fused_add_rms_norm_inplace(x, residual, weight, eps)
    return x, residual
