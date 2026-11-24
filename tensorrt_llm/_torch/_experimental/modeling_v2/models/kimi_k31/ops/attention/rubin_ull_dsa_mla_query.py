# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's MLA query for this rank's 6 heads, published into every rank's q mailbox (K3's MR8
``MlaQuery``, ``kernels/vendor/vllm_mr8/.../mla/query.py``, DCP16, publish-only)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

LOCAL, HEAD_DIM, NOPE, LATENT, ROW = 6, 192, 128, 512, 576


def _cell(why: str, rows: int, rank: int) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, rank=rank))


class DsaMlaQuery(OpWrapper):
    """For each row ``t < rows`` of the DSA hidden projection ``hidden`` (BF16, pitch 3040; columns
    0..1535 are q_a) and each of this rank's 6 heads ``j`` (global head ``6 * query.rank + j``):
    ``qr = rmsnorm(q_a[t]; q_norm)``, ``q = qr @ q_b[j]^T`` (192: 128 NoPE, 64 PE; the rank's
    rows of the layer's packed query bank), the latent half ``q_nope @ W_UK[j]`` (512) and the PE
    half (64; the K3.1 MLA layers have no RoPE), quantized to E4M3 at the per-tensor scale 1 --
    576 bytes, multicast into head block ``query.rank`` of every rank's q mailbox
    (``mailbox[parity, t, 6 * rank + j]``) and stored into ``query.local[t, j]``. The KV insert is
    disabled (every slot -1). Returns None.

    ``query``: the model's ``FlatMlaQuery`` (the ULL target's ``modeling.py``; the compiled kernel of ``rows``, the
    layer's weight views, the mailbox views and its multicast address, this rank).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1: 8 rows (the tiled kernel, one 16-request tile)", 8, 5),
        _cell("served b2: 16 rows", 16, 0),
        _cell("served b4: 32 rows as two 16-request tiles (96 CTAs)", 32, 15),
        _cell("q1 row (the small-batch kernel, FP32 FMAs)", 1, 9),
    )

    note = """
    The K3.1 ULL generation target's DSA query (``dsa_front.MAX_ROWS`` = 32 rows, 16 ranks x 6
    heads), publish-only (``QueryWait`` polls the gathered rows before the core reads them).
    Static weights load before the PDL wait; the current rows after it. On the forked stream layout
    the launch is not PDL (it starts once IndexFront completed; ``dsa_front.query_pdl``).
    """

    def __call__(self, query, rows: int, parity: int, layer: int, hidden: torch.Tensor) -> None:
        x = query._cute(hidden[:, :2112])
        qnorm, kvnorm, qb, uk, qinv, kinv = query.views[layer]
        query.kernels[rows](
            x,
            qnorm,
            kvnorm,
            qb,
            uk,
            query.cache_view,
            query.slots_view,
            qinv,
            kinv,
            query.eps,
            query.local_views[rows],
            query.box_views[parity],
            query._mc(parity),
            query._rank(),
            query.host.stream(query.device),
        )

    def reference(self, query, rows: int, parity: int, layer: int, hidden: torch.Tensor):
        """The FP32 query values [rows, 6, 576] this rank publishes (before the E4M3 rounding):
        FP32 RMSNorm rounded to BF16, FP32 Q_B products rounded to BF16, the FP32 absorption."""
        q_norm, _, q_b, k_b = query.weights[layer]
        x = hidden[:rows, :1536].float()
        normed = x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + float(getattr(query.eps, "value", query.eps))
        )
        normed = (normed * q_norm.float()).bfloat16().float()
        with true_fp32_matmul():
            q = (normed @ q_b.float().t()).bfloat16().float().view(rows, LOCAL, HEAD_DIM)
            latent = torch.einsum("rjn,jnl->rjl", q[..., :NOPE], k_b.float())
        return torch.cat((latent, q[..., NOPE:]), dim=-1)

    def is_valid(self, query, rows: int, parity: int, layer: int, hidden: torch.Tensor) -> None:
        """``FlatMlaQuery.run`` checks ``supports``; the kernel does not check that ``rows`` is the
        rows of ``hidden``, or the parity."""
        if not query.supports(hidden) or hidden.shape[0] != rows:
            raise ValueError("dsa_mla_query: hidden rows the query kernel was not built for")
        if parity not in (0, 1) or query.parity[layer] != parity:
            raise ValueError("dsa_mla_query: the layer's mailbox parity is fixed by its DSA order")


dsa_mla_query = DsaMlaQuery()
