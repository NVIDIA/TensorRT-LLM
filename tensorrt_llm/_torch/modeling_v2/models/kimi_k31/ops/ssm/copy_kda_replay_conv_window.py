# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed the KDA verify's replay convolution caches from the live convolution pool (one Triton launch)."""

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm.kda_replay_conv_window import (
    copy_kda_replay_conv_window as _copy_kda_replay_conv_window,
)


class CopyKdaReplayConvWindow(OpWrapper):
    """Copy `conv_pool[s, section]` into columns `[0, W - 1)` of `{q,k,v}_cache[s]` for each `s` in
    `state_indices`. In place; returns None.

    `conv_pool` is the live packed `[slots, 3 * D, W - 1]` window (sections q | k | v); the caches are
    `[slots, D, W - 1 + num_spec]`, dim-contiguous (`stride(1) == 1`), sharing one layout.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # The layer calls it with the prefill requests' slots after every prefill chunk and with the decode
    # batch's slots after every plain one-token decode, while the replay caches exist. The Triton kernel
    # specializes on the section width (heads) and the committed width; slot counts and order are runtime.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP1 / attention DP, 96 heads, w8 caches: one prefill request",
            spec=dict(heads=96, num_spec=7, slots=(5,), pool_slots=8),
        ),
        Cell(
            why="attention DP, 96 heads, w5 caches: a 32-request decode batch in reverse slot order",
            spec=dict(heads=96, num_spec=4, slots=tuple(range(35, 3, -1)), pool_slots=40),
        ),
        Cell(
            why="TP2, 48 heads, w5 caches: three scattered slots",
            spec=dict(heads=48, num_spec=4, slots=(6, 1, 4), pool_slots=8),
        ),
        Cell(
            why="TP4, 24 heads, w8 caches: three scattered slots",
            spec=dict(heads=24, num_spec=7, slots=(6, 1, 4), pool_slots=8),
        ),
        Cell(
            why="TP8, 12 heads, w5 caches: eight slots",
            spec=dict(heads=12, num_spec=4, slots=(7, 0, 2, 9, 4, 1, 8, 3), pool_slots=10),
        ),
        Cell(
            why="TP16, 6 heads, w8 caches: two slots",
            spec=dict(heads=6, num_spec=7, slots=(0, 7), pool_slots=8),
        ),
        Cell(
            why="TP32, 3 heads, w5 caches, int64 slot indices (accepted like int32)",
            spec=dict(heads=3, num_spec=4, slots=(2, 5, 0), pool_slots=8, index_dtype="int64"),
        ),
        Cell(
            why="sm_107 DEP16 GEN, 96 heads, w8 caches: a 12-request decode batch",
            spec=dict(heads=96, num_spec=7, slots=tuple(range(12)), pool_slots=16),
        ),
        Cell(
            why="sm_107 Helix CP 8, 12 heads, w6 caches: a 16-request batch in reverse slot order",
            spec=dict(heads=12, num_spec=5, slots=tuple(range(15, -1, -1)), pool_slots=20),
        ),
    )

    note = """
    The KDA layer's `_sync_kda_replay_conv_window`: conv_pool the layer's
    live bf16 [slots, 3 * H * 128, 3] pool, q / k / v_cache its replay conv caches (bf16 transposed views of
    per-slot [3 + num_spec, H * 128] storage), state_indices the int32 slots of the requests that just ran
    prefill or a plain decode. Both pools follow the V2 manager's per-slot coalesced layout (padded slot
    strides).

    A pure copy of raw bf16 convolution inputs, compared bit for bit: only columns [0, 3) of the selected
    slots change; the pending-draft tail columns, every other slot and the slot padding are untouched.
    A slot outside the pool is written without a bounds check. One Triton launch over (requests,
    ceil(H * 128 * 3 / 256)); graph-capturable once the specialization is compiled. ARCHS lists only what
    the cells were driven on.
    """

    def __call__(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> None:
        _copy_kda_replay_conv_window(conv_pool, q_cache, k_cache, v_cache, state_indices)

    def reference(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> None:
        dim = conv_pool.shape[1] // 3
        committed = conv_pool.shape[2]
        for slot in state_indices.tolist():
            for section, cache in enumerate((q_cache, k_cache, v_cache)):
                cache[slot, :, :committed] = conv_pool[slot, section * dim : (section + 1) * dim]

    def is_valid(
        self,
        conv_pool: torch.Tensor,
        q_cache: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> None:
        # Geometry, layouts, dtypes and devices the op checks itself, loudly.
        assert all(0 <= slot < conv_pool.shape[0] for slot in state_indices.tolist()), (
            "a slot index outside the pool is read and written without a bounds check"
        )


copy_kda_replay_conv_window = CopyKdaReplayConvWindow()
