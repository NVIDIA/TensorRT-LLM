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


class M3SampledAcceptWsWords(OpWrapper):
    """uint64 words of the multicast workspace ``minimax_m3_sampled_accept`` exchanges through for ``world`` ranks:
    two call halves of ``[attempts][rows][world * splits][record words]``."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="the tp8 target's workspace", spec=dict(world=8)),
        Cell(why="a single rank (local workspace, no multicast)", spec=dict(world=1)),
    )

    note = """
    A host computation (no device work); it is an op so the workspace size and the kernel's indexing come from one
    definition.
    """

    def __call__(self, world: int) -> int:
        return torch.ops.trtllm.minimax_m3_sampled_accept_ws_words(world)

    def reference(self, world: int) -> int:
        """Two halves of attempts x rows x (world x splits) records x record words."""
        return 2 * MAX_ATTEMPTS * MAX_ROWS * world * MAX_SPLITS * RECORD_WORDS


m3_sampled_accept_ws_words = M3SampledAcceptWsWords()
