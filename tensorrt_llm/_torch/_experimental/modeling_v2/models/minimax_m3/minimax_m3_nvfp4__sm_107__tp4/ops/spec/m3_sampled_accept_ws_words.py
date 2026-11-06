# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Size of the sampled-verify exchange workspace (``m3_sampled_accept.SampledAcceptWorkspace.create``)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Exchange rounds per row, verify rows per call, CTAs per row (at most), uint64 words per record.
MAX_ATTEMPTS = 16
MAX_ROWS = 48
MAX_SPLITS = 8
RECORD_WORDS = 4
#: SA-1R (one_round_k 9 / 82): 9 listed bf16 values ride each record, 3 per extra word.
LIST_WORDS = {0: 0, 9: 3, 82: 3}


class M3SampledAcceptWsWords(OpWrapper):
    """uint64 words of the multicast workspace ``minimax_m3_sm107tp4_sampled_accept`` exchanges through for
    ``world`` ranks: two call halves of ``[attempts][rows][world * splits][record words]``."""

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="the tp4 target's workspace", spec=dict(world=4)),
        Cell(why="a single rank (local workspace, no multicast)", spec=dict(world=1)),
        Cell(
            why="the tp4 target's workspace with SA-1R pooled (82)",
            spec=dict(world=4, one_round_k=82),
        ),
        Cell(
            why="the tp4 target's workspace with SA-1R top-9 (9)", spec=dict(world=4, one_round_k=9)
        ),
    )

    note = """
    A host computation (no device work); it is an op so the workspace size and the kernel's indexing come from one
    definition.
    """

    def __call__(self, world: int, one_round_k: int = 0) -> int:
        return torch.ops.trtllm.minimax_m3_sm107tp4_sampled_accept_ws_words(world, one_round_k)

    def reference(self, world: int, one_round_k: int = 0) -> int:
        """Two halves of attempts x rows x (world x splits) records x record words."""
        return (
            2
            * MAX_ATTEMPTS
            * MAX_ROWS
            * world
            * MAX_SPLITS
            * (RECORD_WORDS + LIST_WORDS[one_round_k])
        )


m3_sampled_accept_ws_words = M3SampledAcceptWsWords()
