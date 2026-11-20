# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL backend for the shared k3 fp8w lowm mxfp8 gemm operation."""

from __future__ import annotations

import torch

from ...kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as _mx
from .k3_fp8w_lowm_mxfp8_gemm import K3Fp8wLowmMxfp8Gemm as _SharedK3Fp8wLowmMxfp8Gemm


class K3Fp8wLowmMxfp8Gemm(_SharedK3Fp8wLowmMxfp8Gemm):
    def __call__(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        stage_weights: bool = False,
    ) -> torch.Tensor:
        return _mx.run_k3_fp8w_lowm_gemm_mxfp8(xq, xs, w, w_scale, out, stage_weights=stage_weights)

    def is_valid(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        out: torch.Tensor,
        stage_weights: bool = False,
    ) -> None:
        # The launcher passes raw pointers; can_run_mxfp8 is the contract it does not re-check.
        assert _mx.can_run_mxfp8(xq, xs, w, w_scale, allow_partial_n_tile=True, above_cutover=True)
        assert out.dtype == torch.bfloat16 and out.is_contiguous()
        assert out.shape == (xq.shape[0], w.shape[0])


k3_fp8w_lowm_mxfp8_gemm = K3Fp8wLowmMxfp8Gemm()
