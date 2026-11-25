# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepGEMM schedule of a decode step's paged indexer logits (`fp8_fp4_paged_mqa_logits`), built ahead of the call.

`csa2_indexer_topk_decode` derives it from the rows' visible lengths right before the logits; the lengths are the
owner's per-step descriptors, known when the layer starts, so the target builds it on the attention side stream next
to the query projections and hands it over (`logits_sched`), as the engine derives its indexer metadata on a side
stream."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_BLOCK_KV = 64  # native index page rows (`csa2_indexer_decode._logits`)


class Csa2LogitsSchedule(OpWrapper):
    """`context [n] int32` (each row's visible compressed length, at least 1) -> the schedule
    `deep_gemm.get_paged_mqa_logits_metadata(context.view(n / L, L), 64, SM count)` for requests of
    `L = rows_per_request` rows, the same call
    `csa2_indexer_topk_decode(..., rows_per_request=L)` makes when not handed one."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="tep8 decode: six rows of one request", spec=dict(n=6, longest=400000)),
        Cell(why="tep8 verify of four requests (24 rows)", spec=dict(n=24, longest=20000)),
        Cell(
            why="tep8 decode: one request of six rows grouped (rows_per_request 6), 700K context",
            spec=dict(n=6, longest=350000, rpr=6),
        ),
    )
    note = "The schedule is DeepGEMM's buffer; the indexer test gates the picks made with it bit for bit."

    def __call__(self, context: torch.Tensor, rows_per_request: int = 1) -> torch.Tensor:
        from tensorrt_llm import deep_gemm

        sms = torch.cuda.get_device_properties(context.device).multi_processor_count
        return deep_gemm.get_paged_mqa_logits_metadata(
            context.view(-1, rows_per_request), _BLOCK_KV, sms
        )

    def reference(self, context: torch.Tensor, rows_per_request: int = 1) -> torch.Tensor:
        """The same call (the schedule's format is DeepGEMM's)."""
        from tensorrt_llm import deep_gemm

        sms = torch.cuda.get_device_properties(context.device).multi_processor_count
        return deep_gemm.get_paged_mqa_logits_metadata(
            context.view(-1, rows_per_request), _BLOCK_KV, sms
        )

    def is_valid(self, context: torch.Tensor, rows_per_request: int = 1) -> None:
        assert context.dtype == torch.int32 and context.dim() == 1 and context.is_contiguous()
        assert rows_per_request >= 1 and context.numel() % rows_per_request == 0


csa2_logits_schedule = Csa2LogitsSchedule()
