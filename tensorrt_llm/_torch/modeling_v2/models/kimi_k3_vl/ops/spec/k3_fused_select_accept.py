# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vocab-parallel target selection, strict acceptance, the KDA replay record and the drafter's
KV-length preparation in one Triton launch (one program)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import fused_select_accept
from ._reference import argmax_pairs, interleaved_pairs, strict_accept


class K3FusedSelectAccept(OpWrapper):
    """``(accepted [batch, K + 1], num_accepted [batch], rewind [num_gens] or None)``, int32.

    ``pairs`` is ``(buffer, row stride, slot stride == 2, slots)``: every rank's interleaved
    ``(value, global index)`` float32 candidates of the ``num_contexts + num_gens * (K + 1)``
    rows (``K3VocabShardedLogits.select_pairs``). Each row's token is the candidate the
    ``torch.argmax`` order selects; acceptance and ``force`` are those of ``k3_fused_strict_accept``.
    ``kda_replay = (prev_num_accepted_tokens, state_indices, dummy_mask)`` also writes
    ``prev[state_indices[c + g]] = max(num_accepted - 1, 0)`` for non-dummy generation rows
    (``KDAReplayState.record_acceptance``); ``kv_lens`` also adds 1 to its generation rows and
    returns ``rewind = 1 - num_accepted``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="helix8 c24/c42: 2 rows of DSpark 7, 8 ranks x 10 splits = 80 pair slots, with the "
            "KDA replay record and the KV lengths",
            spec=dict(
                num_contexts=0, num_gens=2, draft_len=7, tp=8, splits=10, replay=True, force=0.0
            ),
        ),
        Cell(
            why="helix8 c75/c174: 8 rows of DSpark 7, 8 ranks x 4 splits = 32 pair slots",
            spec=dict(
                num_contexts=0, num_gens=8, draft_len=7, tp=8, splits=4, replay=True, force=0.0
            ),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4): 4 rows of DSpark 7, 16 ranks x 5 splits = 80 slots",
            spec=dict(
                num_contexts=0, num_gens=4, draft_len=7, tp=16, splits=5, replay=True, force=0.0
            ),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c1): one generation row",
            spec=dict(
                num_contexts=0, num_gens=1, draft_len=7, tp=16, splits=5, replay=True, force=0.0
            ),
        ),
        Cell(
            why="helix16 shape with a context row ahead, without the replay record or KV lengths",
            spec=dict(
                num_contexts=1, num_gens=4, draft_len=7, tp=16, splits=5, replay=False, force=0.0
            ),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4) under a fractional forced acceptance (2.5)",
            spec=dict(
                num_contexts=0, num_gens=4, draft_len=7, tp=16, splits=5, replay=True, force=2.5
            ),
        ),
    )

    note = """
    One program of 4 warps covers every row, so the caller limits it to TAIL_V2_MAX_GENS (16)
    generation requests. All replay slots are read before any is written; generation rows must
    map to distinct slots. The fractional force advances its device counter by one. No host
    synchronization; launch shape fixed by host batch sizes.
    """

    def __call__(
        self,
        pairs: tuple,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
        kda_replay: Optional[tuple] = None,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return fused_select_accept(
            pairs, draft_tokens, num_contexts, batch_size, force, kda_replay, kv_lens
        )

    def reference(
        self,
        pairs: tuple,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
        kda_replay: Optional[tuple] = None,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> tuple:
        """``(accepted, num_accepted, rewind, prev_num_accepted_tokens, kv_lens)``: the op's
        outputs (context rows hold 0 past column 0) and the expected post-state of the replay
        count and of ``kv_lens`` (None when not passed). Nothing is written."""
        buf, stride_row, stride_slot, n_slots = pairs
        num_gens = batch_size - num_contexts
        k = draft_tokens.shape[-1]
        rows = num_contexts + num_gens * (k + 1)
        vals, idx = interleaved_pairs(buf, stride_row, stride_slot, n_slots, rows)
        _, target = argmax_pairs(vals, idx)
        accepted, num_accepted = strict_accept(
            target, draft_tokens, num_contexts, batch_size, force
        )
        n = num_accepted[num_contexts:]
        prev_out = None
        if kda_replay is not None:
            prev, state_indices, dummy = kda_replay
            prev_out = prev.clone()
            slots = state_indices[num_contexts:batch_size].to(torch.int64)
            keep = dummy[num_contexts:batch_size].to(torch.bool)
            prev_out[slots] = torch.where(keep, prev[slots], (n - 1).clamp(min=0).to(prev.dtype))
        rewind = kv_out = None
        if kv_lens is not None:
            rewind = (1 - n).to(torch.int32)
            kv_out = kv_lens.clone()
            kv_out[num_contexts:batch_size] += 1
        return accepted, num_accepted, rewind, prev_out, kv_out

    def is_valid(
        self,
        pairs: tuple,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        force: Optional[tuple] = None,
        kda_replay: Optional[tuple] = None,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> None:
        num_gens = batch_size - num_contexts
        assert draft_tokens.shape[0] >= num_gens, "one draft row per generation request"
        if kda_replay is not None and num_gens > 0:
            slots = kda_replay[1][num_contexts:batch_size]
            # Two rows on one slot race on the same store.
            assert slots.unique().numel() == num_gens, "generation rows share a replay slot"
        if force is not None and len(force) > 1:
            pool = force[3]
            assert pool.numel() & (pool.numel() - 1) == 0, "the force pool must be a power of two"


k3_fused_select_accept = K3FusedSelectAccept()
