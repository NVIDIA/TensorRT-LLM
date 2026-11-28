# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ULL low-M GEMM with the dispatcher entry for the allocating path."""

from __future__ import annotations

import torch

from ...kernels.k3r_gemm import k3_fp8w_lowm_gemm_ull as _dispatch
from .k3_fp8w_lowm_gemm import K3Fp8wLowmGemm as _SharedK3Fp8wLowmGemm


class K3Fp8wLowmGemm(_SharedK3Fp8wLowmGemm):
    def __call__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        sigmoid: bool = False,
    ) -> torch.Tensor:
        if gate is None and not sigmoid and out is None:
            return _dispatch.k3_fp8w_lowm_gemm(x, w, w_scale)
        return super().__call__(x, w, w_scale, out, gate, sigmoid)


k3_fp8w_lowm_gemm = K3Fp8wLowmGemm()
