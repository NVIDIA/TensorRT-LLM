# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's part of a vocab-sharded embedding lookup (Triton), before the all-reduce."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import vocab_shard_lookup


class K3VocabShardEmbedding(OpWrapper):
    """``out[t] = weight[ids[t] - vocab_start]`` where ``vocab_start <= ids[t] < vocab_end``,
    zero elsewhere: ``[len(ids), H]`` in ``weight``'s dtype, new. ``weight`` is this rank's
    contiguous ``[shard, H]`` rows, ``ids`` 1-D of any integer dtype. The all-reduce (sum) of the
    group's results is the full-vocab lookup; the caller runs it."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="helix8 c75/c174: 8 bonus ids and the mask id on rank 3 of 8 (20480-row shard)",
            spec=dict(tp=8, rank=3, num_ids=9),
        ),
        Cell(
            why="helix8 c24/c42: the last of 8 ranks, which holds the mask id",
            spec=dict(tp=8, rank=7, num_ids=3),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4): rank 0 of 16 (10240-row shard)",
            spec=dict(tp=16, rank=0, num_ids=5),
        ),
    )

    note = """
    The DFlash generation step's lookup of ``[bonus..., mask]`` in a vocab-parallel drafter
    embedding where the owner-multicast site (``k3_draft_embed``) does not apply. Grid
    (len(ids), cdiv(H, 1024)); rows outside the shard are stored as zeros, not skipped.
    """

    def __call__(
        self, ids: torch.Tensor, weight: torch.Tensor, vocab_start: int, vocab_end: int
    ) -> torch.Tensor:
        return vocab_shard_lookup(ids, weight, vocab_start, vocab_end)

    def reference(
        self, ids: torch.Tensor, weight: torch.Tensor, vocab_start: int, vocab_end: int
    ) -> torch.Tensor:
        ids = ids.to(torch.int64)
        local = (ids >= vocab_start) & (ids < vocab_end)
        rows = weight[torch.where(local, ids - vocab_start, 0)]
        return torch.where(local[:, None], rows, torch.zeros_like(rows))

    def is_valid(
        self, ids: torch.Tensor, weight: torch.Tensor, vocab_start: int, vocab_end: int
    ) -> None:
        # The kernel addresses rows as weight + row * H.
        assert weight.is_contiguous(), "weight must be contiguous"
        assert vocab_end - vocab_start <= weight.shape[0], "the shard is wider than the table"


k3_vocab_shard_embedding = K3VocabShardEmbedding()
