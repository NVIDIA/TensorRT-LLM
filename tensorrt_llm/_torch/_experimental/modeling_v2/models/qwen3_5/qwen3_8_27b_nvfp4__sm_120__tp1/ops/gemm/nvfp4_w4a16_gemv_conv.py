# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN in_proj_qkvz decode GEMV with the causal conv1d update in its epilogue
(``native_ops.nvfp4_w4a16_gemv_conv``, built from the target's ``csrc/``).

The `nvfp4_w4a16_gemv` kernel, plus, for output columns ``[0, conv_dim)`` (the
``q|k|v`` channels the GDN conv runs over), the single-step width-4 causal conv
with SiLU against a slot-indexed BF16 conv-state pool: the thread that rounds an
output column reads the slot's three state taps, applies the conv weights in the
order ``causal_conv1d_update``'s seqlen-1 kernel uses, writes the activated value
in place of the projection and shifts the state by the new token. One launch
replaces the projection plus the conv update at pure-decode rows 1-3.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ... import native_ops
from .nvfp4_w4a16_gemv import nvfp4_w4a16_gemv

#: This checkpoint's GDN conv width: 2 * 16 key heads * 128 + 48 value heads * 128.
CONV_DIM = 10240


def _cell(why: str, m: int, slots: tuple[int, ...]) -> Cell:
    return Cell(why=why, spec=dict(n=16384, k=5120, m=m, slots=slots))


class Nvfp4W4a16GemvConv(OpWrapper):
    """``out = gemv(x)``; for ``c < conv_dim`` and a non-pad row,
    ``out[m, c] = silu(sum_i w[c, i] * [s0, s1, s2, gemv(x)[m, c]]_i)`` and the slot's
    state becomes ``[s1, s2, gemv(x)[m, c]]``. Returns the fresh ``[M, N]`` BF16 output;
    ``conv_state`` is updated in place. A row whose slot is ``pad_slot_id`` keeps the
    plain projection and touches no state.
    """

    ARCHS = frozenset({Arch.SM_120})

    # GDN in_proj_qkvz (16384, 5120) at pure-decode rows 1-3 against a strided pool; the
    # slot patterns include a padded row (-1) and non-contiguous slot sets.
    CELLS: tuple[Cell, ...] = (
        _cell("decode M=1, one live slot", 1, (3,)),
        _cell("decode M=1, a CUDA-graph padding row only", 1, (-1,)),
        _cell("decode M=2, slots out of order", 2, (4, 1)),
        _cell("decode M=2, padding row first", 2, (-1, 2)),
        _cell("decode M=3, the cap, non-contiguous slots", 3, (2, 0, 5)),
        _cell("decode M=3, a padding row in the middle", 3, (1, -1, 3)),
    )

    note = """
    Metadata consumed: `conv_state_indices` (the step's state slots, int32 `[M]`,
    `pad_slot_id` for CUDA-graph padding rows), `pad_slot_id`.

    Preconditions: as `nvfp4_w4a16_gemv` (BF16 contiguous 16-byte-aligned `x`, K in
    {5120, 6144}), plus `1 <= M <= 3`, `conv_dim` even and `<= N`, `conv_state` BF16
    `[slots, conv_dim, 3]` with strides `(any, 3, 1)`, `conv_weight` BF16 contiguous
    `[conv_dim, 4]` and 8-byte aligned, `conv_state_indices` int32 contiguous `[M]` (the
    op rejects every violation of these loudly). Not checked by the op: slots in range
    and distinct among live rows (`is_valid`).

    Numerics: the projection is `nvfp4_w4a16_gemv`'s, bitwise. The conv is bitwise
    `causal_conv1d_update(gemv(x)[:, :conv_dim], conv_state, conv_weight, None, "silu",
    conv_state_indices=..., pad_slot_id=...)`: `__fmaf_rn` over the taps from 0, then
    `v * __frcp_rn(1 + __expf(-v))`, rounded to BF16; the state stores the BF16
    projection.

    PDL: weight, scale and conv-weight loads before `griddepcontrol.wait`; `x`, the slot
    indices and the state after it.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_sf: torch.Tensor,
        global_scale: float,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        conv_dim: int,
        pad_slot_id: int,
    ) -> torch.Tensor:
        return native_ops.nvfp4_w4a16_gemv_conv(
            x,
            weight,
            weight_sf,
            global_scale,
            conv_state,
            conv_weight,
            conv_state_indices,
            conv_dim,
            pad_slot_id,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_sf: torch.Tensor,
        global_scale: float,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        conv_dim: int,
        pad_slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(out, conv_state)`` as the call leaves them; mutates nothing.

        The projection is `nvfp4_w4a16_gemv.reference`; the conv sums the four taps in
        fp32 and applies torch's SiLU, one rounding.
        """
        proj = nvfp4_w4a16_gemv.reference(x, weight, weight_sf, global_scale)
        out = proj.clone()
        new_state = conv_state.clone()
        w = conv_weight.float()
        for row, slot in enumerate(conv_state_indices.tolist()):
            if slot == pad_slot_id:
                continue
            state = conv_state[slot].float()
            token = proj[row, :conv_dim].float()
            window = torch.cat([state, token.unsqueeze(1)], dim=1)
            acc = (w * window).sum(dim=1)
            out[row, :conv_dim] = (acc * torch.sigmoid(acc)).to(out.dtype)
            new_state[slot] = torch.cat([state[:, 1:], token.unsqueeze(1)], dim=1).to(
                conv_state.dtype
            )
        return out, new_state

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_sf: torch.Tensor,
        global_scale: float,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        conv_dim: int,
        pad_slot_id: int,
    ) -> None:
        assert x.data_ptr() % 16 == 0, "x must be 16-byte aligned (16-byte vector loads)"
        # Slots index the pool unchecked: out of range writes another tensor's memory,
        # and two live rows on one slot race on its state.
        live = [s for s in conv_state_indices.tolist() if s != pad_slot_id]
        assert all(0 <= s < conv_state.shape[0] for s in live), (
            f"slot outside the pool's {conv_state.shape[0]} slots: {live}"
        )
        assert len(set(live)) == len(live), f"live rows share a conv-state slot: {live}"


nvfp4_w4a16_gemv_conv = Nvfp4W4a16GemvConv()
