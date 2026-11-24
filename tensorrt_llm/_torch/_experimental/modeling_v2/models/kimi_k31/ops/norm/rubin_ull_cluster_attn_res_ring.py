# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MR8's seven-CTA cluster AttnRes on a Lamport mailbox delta (``ClusterAttnRes`` with ``slots``,
launched by ``PreparedAttnRes.run_ring``): the MoE tail's output ring finished, re-armed and fed
through the residual add, block-attention residual mixture and the next layer's RMSNorm."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _rubin_ull_attn_res_reference as _ref
from .rubin_ull_cluster_attn_res import cluster_attn_res as _plain


def _cell(why: str, rows: int, nb: int, write: int = -1, final: bool = False) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, key=(nb, True, write, final), aux=True, slots=1))


class ClusterAttnResRing(OpWrapper):
    """``cluster_attn_res`` whose delta is the sum of the ``[slots, cap, 7168]`` Lamport mailbox
    bound to ``prepared`` (``bind_ring``; the MoE tail's output box: one slot): each thread polls
    its 16-byte fragments until none holds the BF16x2 sentinel ``0x80000000``, sums the slots in
    slot order (rounded to BF16 past one slot) and re-arms what it read. At the ``final`` boundary
    the mixture is rounded to BF16 before the output RMSNorm. Other operands as
    ``cluster_attn_res``. Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The ring sites: before every layer from L2 on (the previous layer's MoE tail finishes into
    # the box) and the final norm (8 blocks). nb = ceil(layer / 12).
    CELLS: tuple[Cell, ...] = (
        _cell("L2 before: one saved block, b1 q8", 8, 1),
        _cell("L50 before: five saved blocks, b4 q8", 32, 5),
        _cell("L60 before: a block boundary (write 5)", 8, 5, write=5),
        _cell("L90 before: eight saved blocks, b8 q8 (64 rows, the box's capacity)", 64, 8),
        _cell(
            "final norm: eight blocks, the mixture rounded before the RMSNorm, b1", 8, 8, final=True
        ),
        _cell("final norm, b8 q8 (64 rows)", 64, 8, final=True),
    )

    note = """
    The K3.1 ULL target's ``before_ring`` and ``final_ring`` sites, the consumer the MoE tail of
    the previous layer pushes into (``consume``). Rows past the bound ring's capacity (64) have no
    ring here. The box is re-armed by this launch, which every rank's next MoE tail relies on: the
    next push into it is ordered by the collectives in between, so exactly one ring launch must
    consume each push. Launch order and residency as ``cluster_attn_res``.
    """

    def __call__(
        self,
        prepared,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        out: torch.Tensor,
        aux: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return prepared.run_ring(prefix, blocks, out, aux=aux)

    def reference(
        self,
        prepared,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        out: torch.Tensor,
        aux: Optional[torch.Tensor],
        *,
        weights: Sequence[torch.Tensor],
        eps: Sequence[float],
        mailbox: torch.Tensor,
    ) -> dict:
        """As ``cluster_attn_res.reference``, the delta read from ``mailbox`` (the tensor bound to
        ``prepared``, as it was before the launch)."""
        rows = prefix.shape[0]
        delta = _ref.mailbox_sum(mailbox, rows, prepared.sanitize)
        return _ref.attn_res(prepared.key, weights, eps, prepared.aux_mix, delta, prefix, blocks)

    def is_valid(
        self,
        prepared,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        out: torch.Tensor,
        aux: Optional[torch.Tensor],
    ) -> None:
        """``cluster_attn_res``'s guard, and a bound ring that holds the rows."""
        if prepared.ring is None:
            raise ValueError("cluster_attn_res_ring: no ring bound (bind_ring)")
        if prefix.shape[0] > prepared.ring[2][1]:
            raise ValueError("cluster_attn_res_ring: more rows than the bound ring's capacity")
        _plain.is_valid(prepared, prefix, prefix, blocks, out, aux)


cluster_attn_res_ring = ClusterAttnResRing()
