# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed the KDA verify's replay caches of transferred requests in every local KDA layer at once (one launch per
buffer kind and section)."""

from collections.abc import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class SeedKdaReplayLayers(OpWrapper):
    """For each `s` in `state_indices` and every layer `l`: copy `conv_pool[l, s, section]` into columns
    `[0, W - 1)` of `{q,k,v}_cache[l, s]`, zero their draft columns `[W - 1, W - 1 + num_spec)`, and zero
    `zeroed[i][l, s]` for every buffer of `zeroed`. In place; returns None.

    Every operand is one `[layers, slots, ...]` view over all local KDA layers (the per-layer pool views stacked
    by their constant layer spacing). `conv_pool` is the live packed `[L, slots, 3 * D, W - 1]` window (sections
    q | k | v); the caches are `[L, slots, D, W - 1 + num_spec]`; `zeroed` holds the other replay buffers (k / g /
    v / beta caches), `[L, slots, ...]` each. `state_indices` is a rank-1 int64 tensor of slots.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The cache manager calls it when the KV transfer of disaggregated generation requests completes and when
    # suspended requests resume, with all 69 local KDA layers of Kimi K3.1 stacked. Slot counts and order are
    # runtime; the geometry follows the attention parallelism (heads per rank) and the drafter's width.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16 generation, 6 heads, w5 caches, 69 layers: one transferred request",
            spec=dict(heads=6, num_spec=4, layers=69, slots=(5,), pool_slots=8),
        ),
        Cell(
            why="TP16 generation, 6 heads, w5 caches, 69 layers: four requests in one step, unsorted",
            spec=dict(heads=6, num_spec=4, layers=69, slots=(7, 0, 3, 6), pool_slots=8),
        ),
        Cell(
            why="TP8 generation, 12 heads, w8 caches, 69 layers: three scattered slots",
            spec=dict(heads=12, num_spec=7, layers=69, slots=(6, 1, 4), pool_slots=8),
        ),
        Cell(
            why="attention DP, 96 heads, w5 caches, 69 layers: two slots",
            spec=dict(heads=96, num_spec=4, layers=69, slots=(2, 0), pool_slots=4),
        ),
        Cell(
            why="one local KDA layer (the stacked view of a single layer), 6 heads, w8 caches",
            spec=dict(heads=6, num_spec=7, layers=1, slots=(1, 3), pool_slots=4),
        ),
    )

    note = """
    The family KDA cache manager's batched replay seeding (`K31KDAReplayState._plan_batched_seed`, run by the
    engine's `seed_transferred_slots`): conv_pool the bf16 live conv pools, the caches and `zeroed` the replay
    buffers of the KV cache manager V2 per-slot coalesced pools, each kind stacked over the local KDA layers with
    `_stack_state_views` (a view with a layer dimension; the manager seeds layer by layer when a kind's views are
    not evenly spaced). state_indices the int64 slots of the requests whose KV transfer just completed.

    The same torch launches as the engine's per-layer loop with a layer dimension in front: one index_fill per
    buffer kind, one index_select of the live window and one index_copy per section, so the result is the
    per-layer loop's bit for bit (copies and zero fills only; eleven launches with the k / g / v / beta caches,
    whatever the layer count). Only the selected slots of the viewed buffers change; every other slot, the conv
    pool and the slot padding are untouched. Eager: it runs on the executor thread between steps. ARCHS lists
    only what the cells were driven on.
    """

    def __call__(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        zeroed: Sequence[torch.Tensor],
        state_indices: torch.Tensor,
    ) -> None:
        caches = (q_cache, k_cache, v_cache)
        for buffer in (*caches, *zeroed):
            buffer.index_fill_(1, state_indices, 0)
        dim = q_cache.shape[2]
        committed = conv_pool.shape[3]
        selected = conv_pool.index_select(1, state_indices)
        for section, cache in enumerate(caches):
            cache[..., :committed].index_copy_(
                1,
                state_indices,
                selected[:, :, section * dim : (section + 1) * dim].to(cache.dtype),
            )

    def reference(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        zeroed: Sequence[torch.Tensor],
        state_indices: torch.Tensor,
    ) -> None:
        dim = q_cache.shape[2]
        committed = conv_pool.shape[3]
        for slot in state_indices.tolist():
            for buffer in zeroed:
                buffer[:, slot] = 0
            for section, cache in enumerate((q_cache, k_cache, v_cache)):
                cache[:, slot, :, :committed] = conv_pool[
                    :, slot, section * dim : (section + 1) * dim
                ].to(cache.dtype)
                cache[:, slot, :, committed:] = 0

    def is_valid(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        zeroed: Sequence[torch.Tensor],
        state_indices: torch.Tensor,
    ) -> None:
        # Slot and section geometry mismatches fail loudly inside the index ops; a buffer stacked over fewer layers
        # than the conv pool is zeroed in those layers only.
        assert all(
            buffer.shape[0] == conv_pool.shape[0] for buffer in (q_cache, k_cache, v_cache, *zeroed)
        ), "every operand needs the same layer dimension"


seed_kda_replay_layers = SeedKdaReplayLayers()
