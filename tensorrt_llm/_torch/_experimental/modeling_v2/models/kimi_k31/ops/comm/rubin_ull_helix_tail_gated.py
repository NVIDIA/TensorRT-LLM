# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA layers' gated Helix tail in one launch: ``kernels/comm/rubin_ull_helix_tail_gate.py`` — the Helix
tail (``ops.comm.helix_tail``) whose receivers store ``bf16(combined * sigmoid(gate))``."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.comm import rubin_ull_helix_tail_gate as _gate
from . import _rubin_ull_helix_reference as _ref
from .rubin_ull_helix_tail import helix_tail as _ungated


def _cell(why: str, tokens: int, cp_size: int, masked: bool = False) -> Cell:
    return Cell(why=why, spec=dict(tokens=tokens, cp_size=cp_size, heads=_ref.HEADS, masked=masked))


class HelixTailGated(OpWrapper):
    """``helix_tail`` with the DSA output gate in its receive loop: ``out`` is
    ``bf16(combined * bf16(sigmoid(gate)))``, bit for bit the two PyTorch BF16 kernels applied to
    ``helix_tail``'s rows, except that a row no rank holds a key for is zeros (``helix_tail``'s is
    NaN). ``gate`` BF16 [T, H / P * 128], unit column stride, row stride a multiple of 8, 16-byte
    aligned. Other arguments as ``helix_tail``. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("one rank (P = 1), b1 q8: the gate on a one-source combine", 8, 1),
        _cell("one rank, b8 q8 (64 rows)", 64, 1),
        _cell("4 ranks of one tray, b1 q8 (served at P = 16 over 4 trays)", 8, 4),
        _cell("4 ranks, b4 q8 (32 rows)", 32, 4),
        _cell("4 ranks, b1 with zero-KV rows", 8, 4, True),
    )

    note = """
    The served gated DSA tail of the K3.1 ULL target (``comm.FlatHelix.fused_tail(gate=...)``):
    the gate is the DSA layer's packed gate projection columns. Workspace, ordering and residency
    as ``helix_tail``, on the same workspace and launch counter: gated and ungated launches
    alternate the same two halves.
    """

    def __call__(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
        gate: torch.Tensor,
        out: torch.Tensor,
        zero_kv_mask: Optional[torch.Tensor],
        cp_rank: int,
        cp_size: int,
        ws_addr: int,
        rank_stride: int,
        half_bytes: int,
        state: torch.Tensor,
        max_ctas: int,
        use_pdl: bool,
    ) -> None:
        _gate.k3_helix_tail_gated(
            rows,
            stats,
            w_uv,
            gate,
            out,
            zero_kv_mask=zero_kv_mask,
            cp_rank=cp_rank,
            cp_size=cp_size,
            ws_addr=ws_addr,
            rank_stride=rank_stride,
            half_bytes=half_bytes,
            state=state,
            max_ctas=max_ctas,
            use_pdl=use_pdl,
        )

    def reference(
        self,
        rows: Sequence[torch.Tensor],
        stats: Sequence[torch.Tensor],
        w_uv: torch.Tensor,
        gate: torch.Tensor,
        out: torch.Tensor,
        zero_kv_mask: Sequence[Optional[torch.Tensor]],
        cp_rank: int,
        cp_size: int,
    ) -> torch.Tensor:
        """``helix_tail.reference`` gated as the two PyTorch BF16 kernels do (rows with no key: 0).
        The collective exception: ``rows``, ``stats`` and ``zero_kv_mask`` in rank order."""
        combined = _ref.tail(rows, stats, w_uv, zero_kv_mask, cp_rank, cp_size)
        return (combined * gate.sigmoid()).nan_to_num(nan=0.0)

    def is_valid(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
        gate: torch.Tensor,
        out: torch.Tensor,
        zero_kv_mask: Optional[torch.Tensor],
        cp_rank: int,
        cp_size: int,
        ws_addr: int,
        rank_stride: int,
        half_bytes: int,
        state: torch.Tensor,
        max_ctas: int,
        use_pdl: bool,
    ) -> None:
        """``helix_tail``'s guard; the gate's layout the launcher checks itself."""
        _ungated.is_valid(
            rows,
            stats,
            w_uv,
            out,
            zero_kv_mask,
            cp_rank,
            cp_size,
            ws_addr,
            rank_stride,
            half_bytes,
            state,
            max_ctas,
            use_pdl,
        )


helix_tail_gated = HelixTailGated()
