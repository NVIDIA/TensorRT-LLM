# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Helix all-to-all of the attention partials over the MNNVL FIFO workspace: the stock
``torch.ops.trtllm.alltoall_helix_native`` (fifo v2)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _rubin_ull_helix_reference as _ref


def _cell(why: str, tokens: int, cp_size: int, masked: bool = False) -> Cell:
    return Cell(why=why, spec=dict(tokens=tokens, cp_size=cp_size, heads=_ref.HEADS, masked=masked))


class HelixAlltoallNative(OpWrapper):
    """Exchange of every rank's ``partial_o`` [T, P, L * V] and ``softmax_stats`` [T, P, L * 2]
    (slot ``d`` for rank ``d``'s heads): returns new ``(o, s)`` of the same shapes whose slot
    ``s`` holds what rank ``s`` sent to this rank. ``zero_kv_mask`` bool [T] or None: the sender
    sends zeros with stats ``(-inf, 0)`` for its tokens marked True. ``workspace`` is the FIFO
    workspace of the group (``HelixCpMnnvlMemory``, initialized by ``initialize_helix_workspace``)."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("4 ranks of one tray, b1 q8 (served at P = 16 over 4 trays)", 8, 4),
        _cell("4 ranks, b8 q8 (64 rows)", 64, 4),
        _cell("4 ranks, b1 with zero-KV rows", 8, 4, True),
    )

    note = """
    ``comm.FlatHelix``'s FIFO transport (``cp_config``: ``use_nccl_for_alltoall: false``,
    ``fifo_version: 2``), followed by ``helix_post_process_native``. On the served path it runs
    only where the fused Helix tail declines. The FIFO's per-pair step counters live in the
    workspace: every rank must make the same calls in the same order. Not CUDA-graph safe to
    rebuild; the workspace is built once, collectively, outside capture.
    """

    def __call__(
        self,
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
        workspace: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        zero_kv_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.alltoall_helix_native(
            partial_o, softmax_stats, workspace, cp_rank, cp_size, zero_kv_mask
        )

    def reference(
        self,
        partial_o: Sequence[torch.Tensor],
        softmax_stats: Sequence[torch.Tensor],
        workspace: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        zero_kv_mask: Sequence[Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The collective exception: ``partial_o``, ``softmax_stats`` and ``zero_kv_mask`` of every
        rank in rank order. ``workspace`` is not read."""
        outs, stats = [], []
        for o, s, mask in zip(partial_o, softmax_stats, zero_kv_mask):
            o, s = o[:, cp_rank], s[:, cp_rank].float()
            if mask is not None:
                rows = mask[: o.shape[0]].view(-1, 1)
                o = o.masked_fill(rows, 0.0)
                pair = s.view(s.shape[0], -1, 2)
                pair = _ref.masked_stats(pair, mask)
                s = pair.view(s.shape[0], -1)
            outs.append(o)
            stats.append(s)
        return torch.stack(outs, dim=1), torch.stack(stats, dim=1)


helix_alltoall_native = HelixAlltoallNative()
