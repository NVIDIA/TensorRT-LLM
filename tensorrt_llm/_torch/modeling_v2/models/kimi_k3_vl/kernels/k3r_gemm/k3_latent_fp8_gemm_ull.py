# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatcher entry for the ULL FP8-weight latent-down GEMM with NVFP4 output."""

import torch

from .k3_latent_fp8_gemm import FP4_VEC
from .k3_latent_fp8_gemm import k3_latent_down_fp4 as _latent_down


@torch.library.custom_op("trtllm::k3_latent_down_fp4", mutates_args=())
def k3_latent_down_fp4(
    x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _latent_down(x, w, w_scale, global_sf)


@k3_latent_down_fp4.register_fake
def _(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, global_sf: torch.Tensor):
    m = x.shape[0]
    n = w.shape[0]
    return (
        x.new_empty((m, n // 2), dtype=torch.uint8),
        x.new_empty((m, n // FP4_VEC), dtype=torch.uint8),
    )
