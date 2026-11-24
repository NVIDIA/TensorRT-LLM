# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA layers' Helix tail in one launch: ``kernels/comm/rubin_ull_k3_helix_tail.py`` (no split fold) — the
BF16 V up-projection of every head, the push of each head's values to its owner's Lamport receive
buffer and the owner's softmax-weighted combine."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.comm import rubin_ull_k3_helix_tail as _tail
from . import _rubin_ull_helix_reference as _ref


def _cell(why: str, tokens: int, cp_size: int, masked: bool = False) -> Cell:
    return Cell(why=why, spec=dict(tokens=tokens, cp_size=cp_size, heads=_ref.HEADS, masked=masked))


def receive_check(rows, num_heads, cp_size, half_bytes):
    """Whether ``rows`` latent rows of ``num_heads`` heads fit one receive half at ``cp_size``."""
    tokens = rows.shape[0] // num_heads
    return (
        tokens * cp_size * _tail.record_bytes(num_heads // cp_size, _tail.V_HEAD_DIM) <= half_bytes
    )


class HelixTail(OpWrapper):
    """Rank ``cp_rank`` of a Helix group of ``cp_size``: ``out[t, j * 128 : (j + 1) * 128]`` for
    each of its ``H / P`` heads ``j`` is the combine over every rank ``s`` of
    ``bf16(latent_s[t, h] @ w_uv[h]^T)`` weighted by ``sum_s * exp(max_s - max)``.

    ``rows`` BF16 [T * H, 512] latent rows (row ``t * H + h``), ``stats`` FP32 [T * H, 2] their
    ``(max, sum)``, ``w_uv`` BF16 [H, 128, 512], ``out`` BF16 contiguous [T, H / P * 128] (written).
    ``zero_kv_mask`` bool [>= T] or None: True where this rank holds no key of the token (its
    weight is then 0). ``ws_addr`` / ``rank_stride`` / ``half_bytes`` locate the ranks' receive
    buffers in this process's address space (rank ``r``'s at ``ws_addr + r * rank_stride``, two
    halves, every word armed with the sentinel); ``state`` int64 [8] is the workspace's launch
    counter (it picks the half and is advanced on device). Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("one rank (P = 1): the V projection and a one-source combine, b1 q8", 8, 1),
        _cell("one rank, b8 q8: 64 rows, four 16-row tiles per head", 64, 1),
        _cell("4 ranks of one tray, b1 q8 (served at P = 16 over 4 trays)", 8, 4),
        _cell("4 ranks, b4 q8 (32 rows)", 32, 4),
        _cell("4 ranks, b1 with zero-KV rows (a rank holds no key of some tokens)", 8, 4, True),
    )

    note = """
    The served DSA tail of the K3.1 ULL target (``comm.FlatHelix.fused_tail``) at Helix CP16 with
    96 heads: 6 heads, 768 output columns per rank. The receive workspace is the target's
    ``HelixCpMnnvlMemory`` of 16 MiB per rank, armed with 0xFF bytes once at build; each launch
    re-arms what it read, and the launches alternate the two halves through ``state``, so every
    rank must launch the tail the same number of times in the same order (the drafter shares the
    workspace). Every CTA polls after its own pushes: the grid (``max_ctas``, one CTA per SM at
    most) must be resident at once. The first call per shape compiles (outside graph capture).
    """

    def __call__(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
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
        _tail.k3_helix_tail(
            rows,
            stats,
            w_uv,
            out,
            fold=None,
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
        out: torch.Tensor,
        zero_kv_mask: Sequence[Optional[torch.Tensor]],
        cp_rank: int,
        cp_size: int,
    ) -> torch.Tensor:
        """The collective exception: ``rows``, ``stats`` and ``zero_kv_mask`` of every rank in rank
        order. FP64 V projection rounded to BF16 (what the senders push), FP64 combine rounded
        once. ``out`` is not read."""
        return _ref.tail(rows, stats, w_uv, zero_kv_mask, cp_rank, cp_size)

    def is_valid(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
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
        """The kernel reads its operands through raw pointers: raise where their layout differs
        from the one it assumes, or where the group or grid is one it cannot run."""
        heads = w_uv.shape[0]
        if not _tail.can_run(cp_size, heads, _tail.LATENT_DIM, _tail.V_HEAD_DIM):
            raise ValueError(f"helix_tail: no kernel for P={cp_size}, {heads} heads")
        if not 0 <= cp_rank < cp_size:
            raise ValueError("helix_tail: cp_rank outside the group")
        tokens = rows.shape[0] // heads
        width = heads // cp_size * _tail.V_HEAD_DIM
        if (
            rows.dtype != torch.bfloat16
            or w_uv.dtype != torch.bfloat16
            or stats.dtype != torch.float32
            or out.dtype != torch.bfloat16
            or rows.shape != (tokens * heads, _tail.LATENT_DIM)
            or stats.shape != (tokens * heads, 2)
            or w_uv.shape != (heads, _tail.V_HEAD_DIM, _tail.LATENT_DIM)
            or out.shape != (tokens, width)
            or not all(t.is_contiguous() for t in (rows, stats, w_uv, out))
        ):
            raise ValueError(
                "helix_tail: operands must be contiguous [T*H,512] / [T*H,2] / [H,128,512] / [T,H/P*128]"
            )
        if zero_kv_mask is not None and (
            zero_kv_mask.dtype != torch.bool or zero_kv_mask.numel() < tokens
        ):
            raise ValueError("helix_tail: zero_kv_mask must be bool with a row per token")
        if not receive_check(rows, heads, cp_size, half_bytes):
            raise ValueError("helix_tail: the receive half is too small for these tokens")
        if (
            state.dtype != torch.int64
            or max_ctas > torch.cuda.get_device_properties(rows.device).multi_processor_count
        ):
            raise ValueError("helix_tail: state must be int64 and the grid at most one CTA per SM")


helix_tail = HelixTail()
