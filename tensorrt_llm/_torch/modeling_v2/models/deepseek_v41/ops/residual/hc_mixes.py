# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-Connections mixing weights (pre, post, Sinkhorn comb) of one sub-layer (plain torch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_TILE = 4096  # tokens per tile: the fp32 intermediates are hc times the residual stream


def _split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class HcMixes(OpWrapper):
    """`x [n, hc, d]` -> `(pre [n, hc], post [n, hc], comb [n, hc, hc])`, fp32.

    `mixes = (flatten(x) @ hc_fn.T) * rsqrt(mean(x^2) + norm_eps)`; `pre = sigmoid(.) + hc_eps`,
    `post = 2 * sigmoid(.)`, `comb` the row-softmax followed by `sinkhorn_iters` alternating
    column / row normalizations (a doubly stochastic mix).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="decode: one token at the model's width (hc 4, d 5120)",
            spec=dict(n=1, hc=4, d=5120, iters=20),
        ),
        Cell(
            why="a 5-token verify step",
            spec=dict(n=5, hc=4, d=5120, iters=20),
        ),
        Cell(
            why="prefill tile edge: 4097 rows, one past the 4096-row tile",
            spec=dict(n=4097, hc=4, d=1024, iters=20),
        ),
        Cell(
            why="few Sinkhorn iterations: the loop's boundary (1 iteration is the softmax pass only)",
            spec=dict(n=7, hc=4, d=5120, iters=1),
        ),
    )
    note = """
    `hc_fn` is fp32 and the product is a plain `F.linear`: the entry inherits the caller's matmul
    precision. The test pins IEEE fp32 for its tight band and separately bounds the TF32 default.
    """

    def __call__(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        pre = x.new_empty(n, hc_mult, dtype=torch.float32)
        post = x.new_empty(n, hc_mult, dtype=torch.float32)
        comb = x.new_empty(n, hc_mult, hc_mult, dtype=torch.float32)
        for a in range(0, n, _TILE):
            t = slice(a, min(a + _TILE, n))
            flat = x[t].flatten(1).float()
            rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
            mixes = F.linear(flat, hc_fn) * rsqrt
            pre[t], post[t], comb[t] = _split_sinkhorn(
                mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
            )
        return pre, post, comb

    def reference(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Untiled, in float64: the whole batch at once, the Sinkhorn loop written out."""
        hc = hc_mult
        flat = x.flatten(1).double()
        rs = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = (flat @ hc_fn.double().T) * rs
        sc, base = hc_scale.double(), hc_base.double()
        pre = torch.sigmoid(mixes[:, :hc] * sc[0] + base[:hc]) + hc_eps
        post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * sc[1] + base[hc : 2 * hc])
        comb = (mixes[:, 2 * hc :] * sc[2] + base[2 * hc :]).reshape(-1, hc, hc)
        comb = torch.softmax(comb, dim=-1) + hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
        return pre.float(), post.float(), comb.float()

    def is_valid(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
    ) -> None:
        assert x.dim() == 3 and x.shape[1] == hc_mult, "x is [n, hc_mult, d]"
        mix = (2 + hc_mult) * hc_mult
        assert hc_fn.shape == (mix, hc_mult * x.shape[2]) and hc_fn.dtype == torch.float32
        assert hc_scale.shape == (3,) and hc_base.shape == (mix,)
        assert sinkhorn_iters >= 1, "zero iterations would still run the first normalization"


hc_mixes = HcMixes()
