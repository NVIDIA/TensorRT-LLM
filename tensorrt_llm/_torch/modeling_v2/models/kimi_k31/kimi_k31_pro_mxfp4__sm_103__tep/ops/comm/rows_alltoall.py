# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 all-to-all of per-peer row blocks over the TP group (the engine's NCCL send / recv group).

The token-sharded sparse MLA prefill of the SM 10.3 tep target moves the projected queries from head shards to token
shards (``[tokens of peer j, my heads, 192]`` to peer ``j``) and the V-projected outputs back (``[my tokens, heads of
peer j, 128]`` to peer ``j``). Peers' blocks may differ by a row (an uneven token split), so every block is padded to
``rows`` before the exchange.
"""

from typing import List, Sequence

import torch

from tensorrt_llm._torch.distributed.ops import alltoall_helix
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

#: Kimi K3.1 MLA: query head width (qk_nope 128 + rope 64) and V head width; 96 heads.
_QK = 192
_V = 128
_HEADS = 96


class RowsAllToAll(OpWrapper):
    """Send ``pieces[j]`` (``[n_j, ...]``, ``n_j <= rows``) to TP rank ``j``; return ``[tp, rows, ...]`` whose block
    ``i`` is what rank ``i`` sent here (its first ``n`` rows meaningful, the padding rows unspecified).

    Every rank passes ``tp`` pieces with the same trailing shape and dtype. A one-rank group returns ``pieces[0]``
    padded to ``rows`` with a leading axis of 1.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="ShardQ query exchange of a 16k TEP8 step, head shards to token shards (TP4 tray: 24 heads/rank)",
            spec=dict(world=4, rows=4096, sizes=(4096,) * 4, tail=(_HEADS // 4, _QK)),
        ),
        Cell(
            why="ShardQ output exchange back to head shards, uneven token split",
            spec=dict(world=4, rows=257, sizes=(256, 257, 256, 257), tail=(_HEADS // 4, _V)),
        ),
    )
    note = """
    Engine-function-internal: `tensorrt_llm._torch.distributed.ops.alltoall_helix` (NCCL group of send / recv,
    existing C++ op). A copy: the meaningful rows are bit-exact. Not driven: groups wider than one GB300 tray (TEP8
    spans two; same NCCL calls), CUDA-graph capture (the prefill runs it eagerly).
    """

    def __call__(self, pieces: List[torch.Tensor], rows: int, mapping: Mapping) -> torch.Tensor:
        send = []
        for p in pieces:
            if p.shape[0] != rows or not p.is_contiguous():
                buf = p.new_empty((rows,) + tuple(p.shape[1:]))
                buf[: p.shape[0]].copy_(p)
                p = buf
            send.append(p)
        return alltoall_helix(send, mapping.tp_group)[0]

    def reference(
        self, pieces_in_rank_order: Sequence[Sequence[torch.Tensor]], rows: int, mapping: Mapping
    ) -> torch.Tensor:
        """Every rank's pieces, in rank order (the collective exception); returns ``mapping.tp_rank``'s
        ``[tp, rows, ...]`` with zero padding rows (the test compares the meaningful rows only)."""
        me = mapping.tp_rank
        out = []
        for sender in pieces_in_rank_order:
            p = sender[me]
            buf = p.new_zeros((rows,) + tuple(p.shape[1:]))
            buf[: p.shape[0]].copy_(p)
            out.append(buf)
        return torch.stack(out)


rows_alltoall = RowsAllToAll()
