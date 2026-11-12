# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent row of the MLA KV half: kv_a RMSNorm of ``c_kv`` and the ``k_pe`` copy, one launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_mla_q import k3_mla_q_fused as _kernels


class K3MlaLatentRow(OpWrapper):
    """``torch.cat([rmsnorm(c_kv, weight, eps), k_pe], -1)``: BF16 ``c_kv`` ``[T, 512]`` and
    ``k_pe`` ``[T, 64]`` (row strides multiples of 8, unit column stride) into a new contiguous
    BF16 ``[T, 576]``."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # Spec: `tokens` T <= 16 (the fused query's batches: generation-only Helix, FP4 KV); c_kv and
    # k_pe are the kv_a and K-pe thirds of the projection output [T, 2112]; eps 1e-6.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(tokens=tokens, row_stride=2112, eps=1e-6))
        for why, tokens in (
            ("helix16 ULL c1: one token, one warp of one CTA", 1),
            ("helix16 ULL c2 with draft tokens: a partial last CTA", 6),
            ("helix8 / helix16 fused-query cap: 16 tokens, four full CTAs", 16),
        )
    )

    note = """
    One warp per token with the arithmetic of FlashInfer's CuTe DSL `RMSNormKernel` for a
    512-wide row, so the output is bitwise FlashInfer's norm followed by the copy. The caller
    takes it only where `latent_row_supported` holds and FlashInfer's norm is its CuTe DSL kernel.
    PDL: the weight is read before `griddepcontrol.wait` and dependents are released right after
    it. A process cache holds one compiled kernel per `use_pdl`; compile before graph capture.
    """

    def __call__(
        self,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_pdl: bool = True,
    ) -> torch.Tensor:
        return _kernels.k3_mla_latent_row(c_kv, k_pe, weight, eps, use_pdl=use_pdl)

    def reference(
        self,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_pdl: bool = True,
    ) -> torch.Tensor:
        """FP32 RMSNorm rounded once, then ``k_pe``; ``use_pdl`` does not change the result."""
        x = c_kv.float()
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()
        return torch.cat([normed.to(c_kv.dtype), k_pe], dim=-1)

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
