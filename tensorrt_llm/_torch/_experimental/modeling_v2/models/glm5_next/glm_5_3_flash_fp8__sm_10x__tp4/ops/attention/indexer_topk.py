# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool top-k of the GLM-5.3-Flash indexer: the CuTe DSL radix decode top-k with one candidate
count per row.

Row ``r`` picks the ``select_k`` highest of its ``num_cand[r]`` complete-pool scores
``scores[r, :num_cand[r]]``. A row with ``num_cand[r] <= select_k`` keeps all its pools,
``0 .. num_cand[r] - 1`` in order and ``-1`` after them; a longer row gets ``select_k`` pool ids
in an unspecified order. Columns at and past ``num_cand[r]`` are not read. The target selects
the pools of its sequence-parallel context rows with it (``MlaLayer.context_rows``), on the
scores of ``kpool_score_fp8``; generation rows select with ``kpool_select``. The op is the CuTe
DSL radix decode top-k (``trtllm::cute_dsl_indexer_topk_decode``) with ``next_n = 1``.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _ordered_key(values: torch.Tensor) -> torch.Tensor:
    """FP32 values as int64 keys in the order of their ordered bit patterns (-0 below +0)."""
    bits = values.contiguous().view(torch.int32).long()
    return torch.where(bits < 0, -(bits & 0x7FFFFFFF) - 1, bits)


class IndexerTopk(OpWrapper):
    """``scores`` FP32 ``[R, C]``, ``num_cand`` int32 ``[R]`` -> ``selected`` int32 ``[R,
    select_k]``, written in place and returned.

    ``selected`` is the caller's buffer, exactly ``select_k`` columns wide and contiguous.
    ``scores`` needs unit-stride columns and may have padded rows (the FP8 scorer's strided
    view); ``num_cand[r] <= C``.
    """

    ARCHS = frozenset({Arch.SM_100})
    # select_k = index_topk / index_kpool = 512. The target's context rows score the FP8
    # scorer's [rows, max_kv_len // 4] view, max_kv_len being the largest visible length of
    # the scored rows, and DeepGEMM pads its rows (`row_pad` extra columns); context cells give
    # a rank's sequence-parallel row range, whose candidate counts run up from (row + 1) // 4.
    # The generation cells (`kv` the KV length after the step) drive the op at the 32,800-pool
    # width of a full block table. `levels` rounds the scores to that many values per unit, so
    # the k-th value is tied many times.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP-3 verify of one 13K-token request, BF16-scorer width",
            spec=dict(requests=1, tokens=4, kv=13312, width=32800),
        ),
        Cell(
            why="MTP-3 verify of eight 13K-token requests, BF16-scorer width",
            spec=dict(requests=8, tokens=4, kv=13312, width=32800),
        ),
        Cell(
            why="MTP-3 verify of 64 requests, the largest CUDA-graph batch",
            spec=dict(requests=64, tokens=4, kv=13312, width=32800),
        ),
        Cell(
            why="single-token decode of eight 13K-token requests",
            spec=dict(requests=8, tokens=1, kv=13312, width=32800),
        ),
        Cell(
            why="rank 0's quarter of a 13,312-token prompt (rows 0-3327, 832 pools)",
            spec=dict(rows=(0, 3328), width=832, row_pad=64),
        ),
        Cell(
            why="last sequence-parallel quarter of a 13,312-token prompt (rows 9984-13311)",
            spec=dict(rows=(9984, 13312), width=3328, row_pad=256),
        ),
        Cell(
            why="rank 1's quarter (rows 10-19) of a 37-token prompt: five pools, all short rows",
            spec=dict(rows=(10, 20), width=5, row_pad=3),
        ),
        Cell(
            why="heavy ties at the k-th score",
            spec=dict(requests=8, tokens=4, kv=13312, width=32800, levels=4),
        ),
    )

    note = """
    The kernel's choice among pools of equal score at the k-th value is unspecified, and so
    is the order of a long row's winners: the test compares each row's selected scores as a
    sorted multiset, which every valid top-k shares, and the pool sets where no tie decides.

    JIT: the first call of a compile key compiles a CuTe DSL kernel for seconds. The key is
    the score dtype, next_power_of_2(C), select_k and, through the dispatch on the row count,
    the occupancy band (the cluster size or the large-occupancy variant). The target's C is
    the FP8 context scorer's width, which follows the largest visible length of a scoring call,
    so each power-of-two bucket of it, and each row band, compiles on its first prefill step
    unless a warmup compiled it: `warmup_cute_dsl_radix_topk_decode(top_k=select_k,
    num_cols=C, next_n=1)` in `tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops` compiles
    every row band of one width. Registered only where the CuTe DSL is available.

    FP32 scores only: BF16 and FP16 scores take other kernels of the dispatch.
    """

    def __call__(
        self,
        scores: torch.Tensor,
        num_cand: torch.Tensor,
        selected: torch.Tensor,
        *,
        select_k: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(scores, num_cand, selected, select_k, 1)
        return selected

    def reference(
        self,
        scores: torch.Tensor,
        num_cand: torch.Tensor,
        selected: torch.Tensor,
        *,
        select_k: int,
    ) -> torch.Tensor:
        """A fresh ``[R, select_k]`` int32 selection; ``selected`` is not read.

        A long row's winners are its ``select_k`` highest scores, ties at the k-th score broken
        toward the lower pool id, listed in ascending pool order (the op's order is
        unspecified). A short row is ``0 .. num_cand[r] - 1`` then ``-1``, as the op writes it.
        """
        rows = scores.shape[0]
        dev = scores.device
        out = torch.full((rows, select_k), -1, dtype=torch.int32, device=dev)
        count = num_cand.long()
        ident = torch.arange(select_k, device=dev)
        short = count <= select_k
        out[short] = torch.where(ident < count[short, None], ident, -1).to(torch.int32)
        for chunk in (~short).nonzero().flatten().split(64):
            if chunk.numel() == 0:
                continue
            width = int(count[chunk].max())
            key = _ordered_key(scores[chunk, :width].float())
            col = torch.arange(width, device=dev)
            key = torch.where(col < count[chunk, None], key, torch.iinfo(torch.int64).min)
            # A stable descending sort puts the lower column first among equal keys.
            order = torch.sort(key, dim=1, descending=True, stable=True).indices[:, :select_k]
            out[chunk] = order.sort(dim=1).values.to(torch.int32)
        return out

    def is_valid(
        self,
        scores: torch.Tensor,
        num_cand: torch.Tensor,
        selected: torch.Tensor,
        *,
        select_k: int,
    ) -> None:
        assert scores.dtype == torch.float32, f"FP32 scores only; got {scores.dtype}"
        assert scores.dim() == 2 and scores.stride(1) == 1, f"scores strides {scores.stride()}"
        # The compiled kernel assumes a 32-byte-aligned base for its vector loads.
        assert scores.data_ptr() % 32 == 0, "scores must start 32-byte aligned"
        rows = scores.shape[0]
        # One int32 length per row: the kernel reads num_cand through an int32 pointer.
        assert num_cand.dtype == torch.int32 and num_cand.shape == (rows,)
        assert num_cand.is_contiguous()
        # The kernel writes select_k columns per row at a row stride of select_k.
        assert selected.dtype == torch.int32 and selected.shape == (rows, select_k)
        assert selected.is_contiguous()
        if rows:
            # A count past the row reads the next row's scores.
            assert 0 <= int(num_cand.min()) and int(num_cand.max()) <= scores.shape[1], (
                f"candidate counts up to {int(num_cand.max())} for {scores.shape[1]} columns"
            )


indexer_topk = IndexerTopk()
