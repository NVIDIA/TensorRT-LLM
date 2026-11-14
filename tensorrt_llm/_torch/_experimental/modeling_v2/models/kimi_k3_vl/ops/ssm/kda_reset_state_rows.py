# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the KDA state rows of a prefill's fresh requests before the recurrence reads them."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.kda_state_rows.reset import (
    reset_state_rows,
)


class KdaResetStateRows(OpWrapper):
    """Zero ``state_pool[state_indices[i]]`` and, when given, ``conv_pool[state_indices[i]]`` for every request i
    without ``has_initial_states[i]``; every other row is left as is. Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a prefill batch mixing fresh and continuing requests on the manager's fp32 state pool",
            spec=dict(fresh=(True, False, True, False, True), conv=False),
        ),
        Cell(
            why="an empty prefill: the conv rows of the fresh requests are zeroed too",
            spec=dict(fresh=(True, False, True), conv=True),
        ),
        Cell(
            why="every request continuing: nothing is written",
            spec=dict(fresh=(False, False), conv=True),
        ),
    )
    note = """
    Kimi K3's prefill zeroes the fresh requests' fp32 V-first state rows
    [slots, H, 128, 128] before kda_prefill, which reads every selected row
    (kda_prefill_conv zeroes the conv windows itself); an empty prefill zeroes
    both pools. Rows are the manager's per-layer views: each slot row
    contiguous, slots strided by the manager's coalescing, padding untouched.
    state_indices is int32 [N], has_initial_states bool [N]. Slot indices
    outside [0, slots) are skipped (the manager's padding convention).

    One launch on the current stream, so the next kernel on the stream sees
    the zeros; no host sync, CUDA-graph safe.
    """

    def __call__(
        self,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        conv_pool: Optional[torch.Tensor] = None,
    ) -> None:
        reset_state_rows(state_pool, state_indices, has_initial_states, conv_pool)

    def reference(
        self,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        conv_pool: Optional[torch.Tensor] = None,
    ) -> None:
        for slot, init in zip(state_indices.tolist(), has_initial_states.tolist()):
            if init or not 0 <= slot < state_pool.shape[0]:
                continue
            state_pool[slot] = 0
            if conv_pool is not None:
                conv_pool[slot] = 0


kda_reset_state_rows = KdaResetStateRows()
