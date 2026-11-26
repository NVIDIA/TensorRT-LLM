# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm


class PerHeadRMSNorm(nn.Module):
    allreduce = None
    use_gemma = False

    def __init__(
        self, *, hidden_size: int, eps: float, dtype: torch.dtype, has_weights: bool = True
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        weight = torch.ones(hidden_size, dtype=dtype)
        if has_weights:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * normed.to(x.dtype)


class RMSNorm(PerHeadRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if IS_FLASHINFER_AVAILABLE and x.dtype in (torch.float16, torch.bfloat16):
            return flashinfer_rmsnorm(x, self.weight, self.variance_epsilon)
        return super().forward(x)
