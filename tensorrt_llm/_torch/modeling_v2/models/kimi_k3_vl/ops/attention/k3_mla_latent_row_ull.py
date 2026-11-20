# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL backend for the shared k3 mla latent row operation."""

from __future__ import annotations

import torch

from ...kernels.k3r_mla_q import k3_mla_q_fused_ull as _kernels
from .k3_mla_latent_row import K3MlaLatentRow as _SharedK3MlaLatentRow


class K3MlaLatentRow(_SharedK3MlaLatentRow):
    def __call__(
        self,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_pdl: bool = True,
    ) -> torch.Tensor:
        return _kernels.k3_mla_latent_row(c_kv, k_pe, weight, eps, use_pdl=use_pdl)

    def is_valid(
        self,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_pdl: bool = True,
    ) -> None:
        # The launcher checks nothing: it reads 512 / 64 columns at the given row strides with
        # 16-byte loads whatever the shapes, so an operand outside these is read wrongly.
        assert _kernels.latent_row_supported(c_kv, k_pe, weight), (
            "c_kv [T, 512] / k_pe [T, 64] / weight [512] BF16, row strides % 8, 16-byte aligned"
        )


k3_mla_latent_row = K3MlaLatentRow()
