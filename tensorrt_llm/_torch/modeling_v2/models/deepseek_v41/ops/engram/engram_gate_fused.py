# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram gate in one launch: the engine's fused Triton kernel (`modules/engram/functional.py::engram_gate`) over the
WKV output as it comes out of the projection (`kv [n, (hc + 1) * d]`, keys first, then the shared value), for the dep
and tep8 targets; stock runs the same kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class EngramGateFused(OpWrapper):
    """`h [n, hc, d]` bf16, `kv [n, (hc + 1) * d]` bf16 -> `h + gate * value` `[n, hc, d]` bf16, where per (token, copy)
    `dot = sum_d(h * q_weight * k_weight * key) * rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps) / sqrt(d)`,
    `gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))`, key = `kv[:, c*d:(c+1)*d]`, value = `kv[:, hc*d:]`.

    The residual add is fp32 and rounds once (as the family's plain-torch `engram_gate`). `weight_product` is the
    cached fp32 `q_weight * k_weight` `[hc, d]` (the target builds it at load time); the kernel reads it instead of
    multiplying the two norms per call (bit-identical: the same fp32 product)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="decode token at the model's width (hc 4, d 5120)", spec=dict(n=1, hc=4, d=5120)),
        Cell(why="tep8 verify step: one request, draft 5 (6 rows)", spec=dict(n=6, hc=4, d=5120)),
        Cell(why="dep8 c7000 decode: 160 rows", spec=dict(n=160, hc=4, d=5120)),
        Cell(why="dep32 decode: 150 rows", spec=dict(n=150, hc=4, d=5120)),
        Cell(why="tep8 verify of four requests (24 rows)", spec=dict(n=24, hc=4, d=5120)),
    )
    note = (
        "The precomputed-product variant of the kernel is compiled for "
        "hc 4 / d 5120 on sm_103; other shapes take the two-norm form."
    )

    def __call__(
        self,
        h: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        weight_product: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.modules.engram.functional import engram_gate

        return engram_gate(
            h, kv, q_weight, k_weight, eps, add_residual=True, norm_weight_product=weight_product
        )

    def reference(
        self,
        h: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        weight_product: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Per (token, copy), float64."""
        n, hc, d = h.shape
        hd = h.double()
        key = kv[:, : hc * d].double().view(n, hc, d)
        value = kv[:, hc * d :].double()
        w = q_weight.double() * k_weight.double()
        rstd = (hd.square().mean(-1) + eps).rsqrt() * (key.square().mean(-1) + eps).rsqrt()
        dot = (hd * w * key).sum(-1) * rstd * d**-0.5
        g = torch.sigmoid(torch.sign(dot) * dot.abs().clamp(min=1e-6).sqrt())
        return (hd + g.unsqueeze(-1) * value.unsqueeze(1)).to(h.dtype)

    def is_valid(
        self,
        h: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        weight_product: torch.Tensor,
        eps: float,
    ) -> None:
        n, hc, d = h.shape
        assert h.dtype == kv.dtype == torch.bfloat16 and h.is_contiguous() and kv.is_contiguous()
        assert kv.shape == (n, (hc + 1) * d)
        assert q_weight.shape == k_weight.shape == weight_product.shape == (hc, d)
        assert weight_product.dtype == torch.float32 and weight_product.is_contiguous()
        assert eps > 0


engram_gate_fused = EngramGateFused()
