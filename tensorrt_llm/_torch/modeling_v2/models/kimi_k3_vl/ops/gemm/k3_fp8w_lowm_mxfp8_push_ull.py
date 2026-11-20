# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL backend for the shared k3 fp8w lowm mxfp8 push operation."""

from __future__ import annotations

from typing import Optional

import torch

from ...kernels.k3r_comm import k3_lamport_ar
from ...kernels.k3r_gemm import k3_fp8w_lowm_mxfp8_ull as _mx
from .k3_fp8w_lowm_mxfp8_push import K3Fp8wLowmMxfp8Push as _SharedK3Fp8wLowmMxfp8Push


class K3Fp8wLowmMxfp8Push(_SharedK3Fp8wLowmMxfp8Push):
    def __call__(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        ws: "k3_lamport_ar.LamportWorkspace",
        one_shot: bool = True,
        k0: int = 0,
        k1: Optional[int] = None,
        addend: Optional[torch.Tensor] = None,
        col0: int = 0,
        hidden: Optional[int] = None,
        tactic: Optional["_mx.K3LowMMxTactic"] = None,
    ) -> None:
        if hidden is not None:
            _mx.run_k3_fp8w_lowm_gemm_mxfp8_ag_push(
                xq, xs, w, w_scale, ws, col0, hidden, tactic=tactic
            )
        elif k1 is not None:
            _mx.run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push(
                xq, xs, w, w_scale, k0, k1, ws, one_shot, addend=addend, tactic=tactic
            )
        else:
            _mx.run_k3_fp8w_lowm_gemm_mxfp8_ar_push(xq, xs, w, w_scale, ws, one_shot, tactic=tactic)


k3_fp8w_lowm_mxfp8_push = K3Fp8wLowmMxfp8Push()
