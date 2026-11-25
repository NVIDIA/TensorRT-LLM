# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy a step's routing (and quantized activation) into the MegaMoE bucket buffers in one launch."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class MegaMoeStageInputs(OpWrapper):
    """Write the `n` live rows into the bucket's buffers; rows `n..max_T` get slot -1, weight 0.

    Wraps `stage_megamoe_inputs` (CuTe-DSL), one launch. `topk_slots` `[n, top_k]` int32 (slot ids: the
    router's expert ids looked up in `megamoe_slot_table`'s table), `topk_weights` `[n, top_k]` f32 ->
    `idx_buffer` `[max_T, top_k]` (int64 on sm_103) and `weights_buffer` `[max_T, top_k]` f32. With `x`
    `[n, hidden/2]` u8 and `x_sf` `[n, sf_bytes]` u8 (NVFP4 data and linear scales) also `act_buffer` /
    `sf_buffer`, the bucket's symmetric activation regions;
    activation rows past `n` are not written. `n` may be 0 without a host sync: graph-capturable (a replay
    re-reads the inputs). Inputs must be contiguous and byte rows whole 32-bit words.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="an idle rank: no rows", spec=dict(rows=0, max_t=256)),
        Cell(why="17 live rows in the 256 bucket", spec=dict(rows=17, max_t=256)),
        Cell(why="dep32 bucket: all 150 rows live", spec=dict(rows=150, max_t=150)),
    )
    note = "The routing-only form (no activation) is driven by the test beside the cells."

    def __call__(
        self,
        topk_slots: torch.Tensor,
        topk_weights: torch.Tensor,
        idx_buffer: torch.Tensor,
        weights_buffer: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        x_sf: Optional[torch.Tensor] = None,
        act_buffer: Optional[torch.Tensor] = None,
        sf_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        from tensorrt_llm._torch.moe.kernels.blackwell.megamoe_stage_inputs import (
            stage_megamoe_inputs,
        )

        stage_megamoe_inputs(
            topk_slots,
            topk_weights,
            idx_buffer,
            weights_buffer,
            x=x,
            x_sf=x_sf,
            act_buffer=act_buffer,
            sf_buffer=sf_buffer,
        )

    def reference(self, topk_slots: torch.Tensor, topk_weights: torch.Tensor, max_t: int) -> tuple:
        """(idx `[max_T, K]` int64, weights `[max_T, K]` f32): live rows copied, the rest -1 / 0.

        Returns what `__call__` leaves in `idx_buffer` / `weights_buffer`; `max_t` is their row count.
        """
        n, k = topk_slots.shape
        idx = torch.full((max_t, k), -1, dtype=torch.int64, device=topk_slots.device)
        w = torch.zeros((max_t, k), dtype=torch.float32, device=topk_slots.device)
        idx[:n], w[:n] = topk_slots.long(), topk_weights
        return idx, w

    def is_valid(
        self,
        topk_slots: torch.Tensor,
        topk_weights: torch.Tensor,
        idx_buffer: torch.Tensor,
        weights_buffer: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        x_sf: Optional[torch.Tensor] = None,
        act_buffer: Optional[torch.Tensor] = None,
        sf_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        for name, t in (
            ("topk_slots", topk_slots),
            ("topk_weights", topk_weights),
            ("x", x),
            ("x_sf", x_sf),
        ):
            if t is not None:
                assert t.is_contiguous(), f"{name} must be contiguous; the kernel reads packed rows"
        assert topk_slots.dtype == torch.int32 and topk_weights.dtype == torch.float32
        assert (x is None) == (x_sf is None) == (act_buffer is None) == (sf_buffer is None), (
            "x, x_sf, act_buffer and sf_buffer go together"
        )
        assert (
            topk_slots.shape[0] <= idx_buffer.shape[0] and idx_buffer.shape == weights_buffer.shape
        )


megamoe_stage_inputs = MegaMoeStageInputs()
