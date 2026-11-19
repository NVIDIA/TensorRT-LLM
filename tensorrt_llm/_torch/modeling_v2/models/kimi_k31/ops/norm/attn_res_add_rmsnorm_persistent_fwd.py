# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 (residual add +) attention-residual mix + RMSNorm through the engine's persistent attn_res op."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _attn_res_common as _c


def _cell(rows: int, candidates: int, add: bool, why: str) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, candidates=candidates, add=add))


class AttnResAddRmsnormPersistentFwd(OpWrapper):
    """``(updated, out)`` for ``[M, 7168]`` BF16 rows: ``updated = bf16(prefix + addend)`` (or
    ``prefix`` itself without an addend), ``out`` the attention-residual mix of ``[block...,
    updated]`` followed by the trailing RMSNorm, as ``k31_attn_res_add_rmsnorm`` computes it.

    The engine's ``trtllm::attn_res_add_rmsnorm_persistent_fwd`` (C++, sm_100 family): a persistent
    grid that streams the rows, so unlike the per-token kernels its cost follows the bytes, with no
    row ceiling. Without an addend the first result is ``prefix`` itself (not a copy).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            _cell(rows, n, add, why)
            for rows, why in (
                (128, "DEP16 gen, 128 rows per rank"),
                (224, "DEP16 gen, 32 requests x 7 rows per rank"),
                (2048, "TEP8 prefill, a 16384-token chunk split over 8 ranks"),
                (8192, "DEP16 prefill, 8192 tokens per rank"),
            )
            for n in (2, 9)
            for add in (False, True)
        ),
        _cell(16384, 5, True, "a 16384-token prefill chunk on one rank"),
    )
    note = """
    Requires B == 1 (rows as [M, 1, 7168] internally), H == 7168 and 2..9 candidates (Kimi K3.1:
    block size 12 over 93 layers gives 2..9). Not bitwise against the target's current path
    (attn_res_fwd + flashinfer RMSNorm, which rounds the normalized row once with the weight): the
    output is one BF16 step away in about a quarter of the elements; `updated` has the bits of the
    torch add. GB300 (CUDA graph, chain of dependent calls): faster than attn_res_fwd + RMSNorm
    from 256 rows on (8192 rows, 5 candidates: 162 -> 119 us, with the add 213 -> 141 us) and
    faster than the PDL attn_res kernel from ~99 rows at 7+ candidates and from ~192 rows at any
    count.
    """

    def __call__(
        self,
        prefix: torch.Tensor,
        addend: Optional[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = prefix.shape[0]
        updated, out = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix.reshape(m, 1, _c.HIDDEN),
            None if addend is None else addend.reshape(m, 1, _c.HIDDEN),
            block.reshape(-1, m, 1, _c.HIDDEN),
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
        )
        if addend is None:
            return prefix, out.reshape(m, _c.HIDDEN)
        return updated.reshape(m, _c.HIDDEN), out.reshape(m, _c.HIDDEN)

    def reference(
        self,
        prefix: torch.Tensor,
        addend: Optional[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated = prefix if addend is None else prefix + addend
        out = _c.reference(updated, block, res_weight, rms_weight, out_weight, rms_eps, out_eps)
        return updated, out

    def is_valid(
        self,
        prefix: torch.Tensor,
        addend: Optional[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> None:
        assert prefix.dim() == 2 and prefix.shape[1] == _c.HIDDEN, "prefix must be [M, 7168]"
        assert 2 <= block.shape[0] + 1 <= 9, "the persistent op takes 2..9 candidates"
        assert tuple(block.shape[1:]) == tuple(prefix.shape), "block must be [N - 1, M, 7168]"
        for t in (prefix, block, res_weight, rms_weight, out_weight) + (
            () if addend is None else (addend,)
        ):
            assert t.dtype == torch.bfloat16 and t.is_contiguous() and t.is_cuda, (
                "contiguous BF16 CUDA tensors"
            )
        if addend is not None:
            assert addend.shape == prefix.shape, "the addend must be [M, 7168]"
        for w in (res_weight, rms_weight, out_weight):
            assert w.numel() == _c.HIDDEN, "each weight has 7168 elements"


attn_res_add_rmsnorm_persistent_fwd = AttnResAddRmsnormPersistentFwd()
