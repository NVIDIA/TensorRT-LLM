# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatcher entry for the ULL low-M FP8-weight GEMM."""

import torch

from . import k3_latent_fp8_gemm_ull  # noqa: F401 - registers the sibling dispatcher op
from .k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm as _gemm


@torch.library.custom_op("trtllm::k3_fp8w_lowm_gemm", mutates_args=())
def k3_fp8w_lowm_gemm(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return _gemm(x, w, w_scale)


@k3_fp8w_lowm_gemm.register_fake
def _(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)
