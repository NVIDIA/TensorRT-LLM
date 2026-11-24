# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store each slot's convolution window after its accepted drafts back into the live convolution pool (one Triton
launch)."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.kda_replay_conv_window import (
    store_kda_replay_conv_window as _store_kda_replay_conv_window,
)


class StoreKdaReplayConvWindow(OpWrapper):
    """Write `{q,k,v}_cache[s, :, a : a + W - 1]` into `conv_pool[s, section]` for each `s` in `state_indices`,
    `a` its `num_accepted_tokens` entry clamped to `[0, num_spec]`. In place; returns None.

    Layouts as in `copy_kda_replay_conv_window`; `num_spec` is the caches' width minus `W - 1`.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # The cache manager calls it once per KDA layer for one suspended slot, right after
    # kda_mtp_commit_pending_drafts. The batched form and the clamp are the op's own surface.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP1 / attention DP, 96 heads, w8 caches: one slot with all seven drafts accepted",
            spec=dict(heads=96, num_spec=7, slots=(5,), accepted=(7,), pool_slots=8),
        ),
        Cell(
            why="TP2, 48 heads, w5 caches: one slot, two drafts accepted",
            spec=dict(heads=48, num_spec=4, slots=(1,), accepted=(2,), pool_slots=8),
        ),
        Cell(
            why="TP4, 24 heads, w8 caches: four slots, 0 / 1 / 7 / 3 accepted",
            spec=dict(
                heads=24, num_spec=7, slots=(6, 1, 4, 2), accepted=(0, 1, 7, 3), pool_slots=8
            ),
        ),
        Cell(
            why="TP8, 12 heads, w5 caches: one slot with nothing accepted restores the committed window",
            spec=dict(heads=12, num_spec=4, slots=(3,), accepted=(0,), pool_slots=8),
        ),
        Cell(
            why="TP16, 6 heads, w8 caches: eight slots, every count 0..7",
            spec=dict(
                heads=6,
                num_spec=7,
                slots=(9, 7, 5, 3, 1, 0, 2, 4),
                accepted=tuple(range(8)),
                pool_slots=10,
            ),
        ),
        Cell(
            why="TP32, 3 heads, w5 caches: one slot, four drafts accepted",
            spec=dict(heads=3, num_spec=4, slots=(0,), accepted=(4,), pool_slots=8),
        ),
        Cell(
            why="counts outside [0, num_spec] are clamped like the verify kernel clamps them",
            spec=dict(heads=6, num_spec=4, slots=(0, 5), accepted=(-3, 9), pool_slots=8),
        ),
        Cell(
            why="Rubin helix8, 12 heads, w6 caches: one slot, five drafts accepted",
            spec=dict(heads=12, num_spec=5, slots=(2,), accepted=(5,), pool_slots=8),
        ),
        Cell(
            why="Rubin TEP8, 12 heads, w8 caches: one slot, two drafts accepted",
            spec=dict(heads=12, num_spec=7, slots=(4,), accepted=(2,), pool_slots=8),
        ),
    )

    note = """
    The cache manager's suspend path (`commit_kda_pending_drafts` after the replay-only commit): conv_pool
    the layer's live bf16 [slots, 3 * H * 128, 3] pool, q / k / v_cache its replay conv caches (bf16
    transposed views of per-slot [3 + num_spec, H * 128] storage), state_indices int32 [N] and
    num_accepted_tokens int32 [N] (request-indexed, the slot's gathered count). Together with the commit it
    leaves the live pools where the next verify would start, with nothing pending; the caller then clears
    the count. Not per step.

    A pure copy of raw bf16 convolution inputs, compared bit for bit: only the selected slots' live windows
    change; the replay caches are read only, other slots and the slot padding untouched. A slot outside the
    pool is written without a bounds check, and two requests naming one slot race. One Triton launch over
    (requests, ceil(H * 128 * 3 / 256)). ARCHS lists only what the cells were driven on.
    """

    def __call__(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        _store_kda_replay_conv_window(
            conv_pool, q_cache, k_cache, v_cache, state_indices, num_accepted_tokens
        )

    def reference(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        dim = conv_pool.shape[1] // 3
        committed = conv_pool.shape[2]
        num_spec = q_cache.shape[2] - committed
        for slot, accepted in zip(state_indices.tolist(), num_accepted_tokens.tolist()):
            shift = min(max(accepted, 0), num_spec)
            for section, cache in enumerate((q_cache, k_cache, v_cache)):
                conv_pool[slot, section * dim : (section + 1) * dim] = cache[
                    slot, :, shift : shift + committed
                ]

    def is_valid(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> None:
        # Geometry, layouts and the one-count-per-slot shape the op checks itself, loudly.
        slots = state_indices.tolist()
        assert all(0 <= slot < conv_pool.shape[0] for slot in slots), (
            "a slot index outside the pool is read and written without a bounds check"
        )
        assert len(set(slots)) == len(slots), (
            "state_indices must name each slot once: two requests storing into one slot race"
        )


store_kda_replay_conv_window = StoreKdaReplayConvWindow()
