# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepGEMM schedule of the candidate consumers' sparse indexer logits, once per step.

Every consumer layer of a step scores the same candidates (`cand_in` of the one candidate-source layer) over the same
owner's block table, so the schedule `consume_candidates` derives per layer is the same tensor each time; stock
derives it once per step. The dep and tep8 targets compute it here and hand it to
`csa2_indexer_topk_decode(cand_sched=...)`."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

SPARSE_BLOCK = (
    8  # keys per block of the sparse logits kernel (`csa2_indexer_candidates.SPARSE_BLOCK`)
)
_NATIVE = 64  # rows per native index page


class Csa2ConsumerSchedule(OpWrapper):
    """`(counts [n] int32, blocks [n, pages] int32, ids [n, c] int32)` -> the `fp8_fp4_paged_sparse_mqa_logits`
    schedule `get_paged_sparse_mqa_logits_metadata(counts, blocks, arange(n), 64, ids, int8, 8)`, the same call
    `consume_candidates` makes when not handed one."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 c7000 decode: 160 rows of 40 requests, 40 candidate blocks",
            spec=dict(n=160, blocks=40, pages=64),
        ),
        Cell(why="dep32 decode: 150 rows of 25 requests", spec=dict(n=150, blocks=40, pages=64)),
        Cell(
            why="tep8 decode: one request, six rows, 40 candidate blocks",
            spec=dict(n=6, blocks=40, pages=16),
        ),
        Cell(why="tep8 verify of four requests (24 rows)", spec=dict(n=24, blocks=40, pages=16)),
    )
    note = "The schedule is DeepGEMM's opaque buffer; the tests gate the consumer picks made with it, bit for bit."

    def __call__(
        self, counts: torch.Tensor, blocks: torch.Tensor, ids: torch.Tensor
    ) -> torch.Tensor:
        from tensorrt_llm.deep_gemm import get_paged_sparse_mqa_logits_metadata

        rows = torch.arange(blocks.shape[0], dtype=torch.int32, device=blocks.device)
        return get_paged_sparse_mqa_logits_metadata(
            counts, blocks, rows, _NATIVE, ids, torch.int8, SPARSE_BLOCK
        )

    def reference(
        self, counts: torch.Tensor, blocks: torch.Tensor, ids: torch.Tensor
    ) -> torch.Tensor:
        """The same call inline, as `consume_candidates` makes it (the schedule's format is DeepGEMM's)."""
        from tensorrt_llm.deep_gemm import get_paged_sparse_mqa_logits_metadata

        rows = torch.arange(blocks.shape[0], dtype=torch.int32, device=blocks.device)
        return get_paged_sparse_mqa_logits_metadata(
            counts, blocks, rows, _NATIVE, ids, torch.int8, SPARSE_BLOCK
        )

    def is_valid(self, counts: torch.Tensor, blocks: torch.Tensor, ids: torch.Tensor) -> None:
        n = blocks.shape[0]
        assert counts.dtype == blocks.dtype == ids.dtype == torch.int32
        assert counts.shape == (n,) and ids.shape[0] == n and blocks.is_contiguous()


csa2_consumer_schedule = Csa2ConsumerSchedule()
