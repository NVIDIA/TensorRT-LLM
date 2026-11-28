# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the recurrent (and optionally convolution) state rows of the fresh requests of a prefill batch."""

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.ssm.reset_recurrent_state_rows import (
    reset_recurrent_state_rows as _reset_recurrent_state_rows,
)


class ResetRecurrentStateRows(OpWrapper):
    """Zero `recurrent_states[state_indices[i]]` (and `conv_states[...]`) where `has_initial_states[i]` is False.

    In place; returns None. Slots outside `[0, recurrent_states.shape[0])` are skipped.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads} KDA heads per rank, {dtype_name} pool with V2 slot padding: indexed-prefill fresh rows",
                spec=dict(
                    heads=heads,
                    slots=(5, 0, 9, 3),
                    fresh=(True, False, True, False),
                    state_dtype=dtype_name,
                    with_conv=False,
                    pool_slots=12,
                ),
            )
            for heads in (96, 48, 24, 12, 6, 3)
            for dtype_name in ("float32",)
        ),
        Cell(
            why="empty prefill chunk: recurrent and convolution rows reset together",
            spec=dict(
                heads=12,
                slots=(1, 4, 6),
                fresh=(True, False, True),
                state_dtype="float32",
                with_conv=True,
                pool_slots=8,
            ),
        ),
        Cell(
            why="padding slot -1 and an out-of-range slot are skipped",
            spec=dict(
                heads=6,
                slots=(-1, 3, 8),
                fresh=(True, True, True),
                state_dtype="float32",
                with_conv=True,
                pool_slots=8,
            ),
        ),
        Cell(
            why="sm_107 DEP16 context: 96 heads, recurrent and convolution rows reset together",
            spec=dict(
                heads=96,
                slots=(3, 1, 6),
                fresh=(True, True, False),
                state_dtype="float32",
                with_conv=True,
                pool_slots=8,
            ),
        ),
    )
    note = """
    One Triton launch, grid (requests, ceil(row elements / 256)); stores zeros
    with the pool's dtype into each selected row, nothing else is read or
    written. Rows must be contiguous within a slot (V-first [H, V, K] state,
    [3D, W - 1] convolution window); the slot stride may be padded, as the V2
    hybrid manager lays out its per-slot coalesced pools. Bitwise: the result
    is exact zeros. The prefill path calls it on the recurrent pool before the
    indexed CuTe DSL prefill (the fused convolution resets its own rows), and
    on both pools for a zero-token prefill chunk. Ordered on the current
    stream; graph-capturable after the Triton specialization is warm.
    ARCHS lists only the architectures the cells were driven on: sm_103 (GB300)
    and sm_107 (VR200).
    """

    def __call__(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        conv_states: Optional[torch.Tensor] = None,
    ) -> None:
        _reset_recurrent_state_rows(
            recurrent_states, state_indices, has_initial_states, conv_states
        )

    def reference(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        conv_states: Optional[torch.Tensor] = None,
    ) -> None:
        for slot, has_init in zip(state_indices.tolist(), has_initial_states.tolist()):
            if has_init or not 0 <= slot < recurrent_states.shape[0]:
                continue
            recurrent_states[slot].zero_()
            if conv_states is not None:
                conv_states[slot].zero_()


reset_recurrent_state_rows = ResetRecurrentStateRows()
