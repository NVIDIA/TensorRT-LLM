# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert weights and scale scalars in the MegaMoE kernel's layout (plain torch; load time only).

`megamoe_pack_experts` is the entry: the layout transform, bit-identical to the engine's
`NVFP4MegaMoECuteDslMethod` (the test compares against it at H=5120 / I=2304 and small shapes).
`megamoe_expert_scalars` is its companion helper: the kernel's per-slot scalars from the checkpoint's
`input_scale` / `weight_scale_2`. A slot holding a copy of an expert gets the same bytes and scalars as the
expert's other copies.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fc1_scale_fold import (
    fc1_global_scale,
    fold_block_scales,
)

_SLOT_CHUNK = 16  # slots transformed at a time: bounds the transient copies to a few chunks


def _blocked(sf: torch.Tensor) -> torch.Tensor:
    """`[S, rows, cols]` scale bytes -> `[S, flat]` in 128x4 atoms (32x4x4 order), zero-padded to
    whole atoms (`rows % 128`, `cols % 4`)."""
    s, rows, cols = sf.shape
    padded = F.pad(sf, (0, -cols % 4, 0, -rows % 128))
    blocked = padded.view(s, (rows + 127) // 128, 4, 32, (cols + 3) // 4, 4)
    return blocked.permute(0, 1, 4, 3, 2, 5).reshape(s, -1).contiguous()


def _interleave(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`[S, I, K]` gate and up -> `[S, 2I, K]` as gate 0:16, up 0:16, gate 16:32, ..."""
    s, inter, k = gate.shape
    g = gate.view(s, inter // 16, 16, k)
    u = up.view(s, inter // 16, 16, k)
    return torch.stack([g, u], dim=2).view(s, 2 * inter, k)


class MegaMoePackExperts(OpWrapper):
    """NVFP4 expert stacks as the kernel reads them: `(fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf)`,
    one row per slot, all uint8.

    Inputs, per slot `s`: `w1` (gate) and `w3` (up) `[S, I, H/2]` packed e2m1 (two codes per byte, element 2i in
    the low nibble) with `w1_sf`, `w3_sf` `[S, I, H/16]` e4m3 block scales (any 1-byte dtype); `w2` (down)
    `[S, H, I/2]` with `w2_sf` `[S, H, I/16]`. `w3_scale_fold` / `w1_scale_fold` `[S]` (optional): the up / gate
    block scales are multiplied by it and rounded to e4m3 (`fold_block_scales`). The kernel has one FC1 alpha, built
    from the larger of w1's and w3's `weight_scale_2`; `megamoe_expert_scalars` gives the factors (`<= 1`, 1 for the
    half holding the larger) that put the other half on it.

    - `fc1_weight` `[S, 2I, H/2]`: gate and up interleaved in groups of 16 rows;
    - `fc1_weight_sf` `[S, round128(2I) * round4(H/16)]`: the same interleave, then 128x4 atoms;
    - `fc2_weight` `[S, H, I/2]` (a copy of `w2`) and `fc2_weight_sf` `[S, round128(H) * round4(I/16)]` in
      128x4 atoms.

    Slots are transformed 16 at a time, which bounds the transient copies.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 geometry (H=5120, I=2304) at 96 slots, twice the 48 per rank deployed: six 16-slot chunks",
            spec=dict(slots=96, inter=2304, hidden=5120),
        ),
        Cell(
            why="dep32: 13 slots per rank (416 slots over 32 ranks)",
            spec=dict(slots=13, inter=2304, hidden=5120),
        ),
        Cell(
            why="small geometry (H=512, I=256), 35 slots: two full chunks of 16 and a partial one",
            spec=dict(slots=35, inter=256, hidden=512),
        ),
    )
    note = (
        "Any bytes are accepted (it is a layout transform); the test "
        "feeds random bytes and compares against the engine's loader."
    )

    def __call__(
        self,
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w3: torch.Tensor,
        w3_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        w3_scale_fold: Optional[torch.Tensor] = None,
        w1_scale_fold: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s, inter, h_bytes = w1.shape
        assert w3.shape == w1.shape and w1_sf.shape == w3_sf.shape == (s, inter, h_bytes // 8)
        assert inter % 16 == 0
        dev = w1.device
        fc1_cols = -(-(h_bytes // 8) // 4) * 4
        fc1_sf_flat = -(-2 * inter // 128) * 128 * fc1_cols
        fc2_rows, fc2_cols = w2.shape[1], w2_sf.shape[2]
        fc2_sf_flat = -(-fc2_rows // 128) * 128 * (-(-fc2_cols // 4) * 4)
        fc1 = torch.empty((s, 2 * inter, h_bytes), dtype=torch.uint8, device=dev)
        fc1_sf = torch.empty((s, fc1_sf_flat), dtype=torch.uint8, device=dev)
        fc2 = w2.view(torch.uint8).clone()
        fc2_sf = torch.empty((s, fc2_sf_flat), dtype=torch.uint8, device=dev)
        for lo in range(0, s, _SLOT_CHUNK):
            hi = min(lo + _SLOT_CHUNK, s)
            up_sf, gate_sf = w3_sf[lo:hi], w1_sf[lo:hi]
            if w3_scale_fold is not None:
                up_sf = fold_block_scales(up_sf, w3_scale_fold[lo:hi])
            if w1_scale_fold is not None:
                gate_sf = fold_block_scales(gate_sf, w1_scale_fold[lo:hi])
            fc1[lo:hi] = _interleave(w1[lo:hi].view(torch.uint8), w3[lo:hi].view(torch.uint8))
            fc1_sf[lo:hi] = _blocked(
                _interleave(gate_sf.view(torch.uint8), up_sf.view(torch.uint8))
            )
            fc2_sf[lo:hi] = _blocked(w2_sf[lo:hi].view(torch.uint8))
        return fc1, fc1_sf, fc2, fc2_sf

    def reference(
        self,
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w3: torch.Tensor,
        w3_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        w3_scale_fold: Optional[torch.Tensor] = None,
        w1_scale_fold: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same four tensors from plain index arithmetic: the 16-row interleave group by group, and the
        128x4 scale atoms by the destination offset of every scale element, not by view and permute."""
        s, inter, hb = w1.shape
        dev = w1.device
        up_sf, gate_sf = w3_sf, w1_sf
        if w3_scale_fold is not None:
            fold = w3_scale_fold.float()[:, None, None]
            up_sf = (w3_sf.view(torch.float8_e4m3fn).float() * fold).clamp(-448, 448)
            up_sf = up_sf.to(torch.float8_e4m3fn)
        if w1_scale_fold is not None:
            fold = w1_scale_fold.float()[:, None, None]
            gate_sf = (w1_sf.view(torch.float8_e4m3fn).float() * fold).clamp(-448, 448)
            gate_sf = gate_sf.to(torch.float8_e4m3fn)
        fc1 = torch.empty((s, 2 * inter, hb), dtype=torch.uint8, device=dev)
        fc1_sf = torch.empty((s, 2 * inter, hb // 8), dtype=torch.uint8, device=dev)
        for g in range(inter // 16):
            lo, mid, hi = 32 * g, 32 * g + 16, 32 * g + 32
            fc1[:, lo:mid] = w1.view(torch.uint8)[:, 16 * g : 16 * g + 16]
            fc1[:, mid:hi] = w3.view(torch.uint8)[:, 16 * g : 16 * g + 16]
            fc1_sf[:, lo:mid] = gate_sf.view(torch.uint8)[:, 16 * g : 16 * g + 16]
            fc1_sf[:, mid:hi] = up_sf.view(torch.uint8)[:, 16 * g : 16 * g + 16]
        return (
            fc1,
            _atoms_reference(fc1_sf),
            w2.view(torch.uint8).clone(),
            _atoms_reference(w2_sf.view(torch.uint8)),
        )

    def is_valid(
        self,
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w3: torch.Tensor,
        w3_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        w3_scale_fold: Optional[torch.Tensor] = None,
        w1_scale_fold: Optional[torch.Tensor] = None,
    ) -> None:
        s, inter, hb = w1.shape
        assert inter % 16 == 0, "gate and up interleave in groups of 16 rows"
        assert w3.shape == w1.shape and w1_sf.shape == w3_sf.shape == (s, inter, hb // 8), (
            "w1 / w3 are [S, I, H/2] with scales [S, I, H/16]"
        )
        assert w2.shape[:2] == (s, hb * 2) and w2_sf.shape == (s, hb * 2, w2.shape[2] // 8), (
            "w2 is [S, H, I/2] with scales [S, H, I/16]"
        )
        for t in (w1, w1_sf, w3, w3_sf, w2, w2_sf):
            assert t.element_size() == 1, (
                "the layout transform moves bytes: every operand is 1-byte"
            )
        for fold in (w3_scale_fold, w1_scale_fold):
            if fold is not None:
                assert fold.shape == (s,), "a scale fold is one factor per slot"


def _atoms_reference(sf: torch.Tensor) -> torch.Tensor:
    """`[S, rows, cols]` scale bytes -> `[S, flat]` in 128x4 atoms, element (r, c) at
    `(((r // 128) * ceil(cols / 4) + c // 4) * 32 + r % 32) * 16 + (r % 128) // 32 * 4 + c % 4`, zero padded."""
    s, rows, cols = sf.shape
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    r = torch.arange(rows, device=sf.device)[:, None]
    c = torch.arange(cols, device=sf.device)[None, :]
    dest = (((r // 128) * col_tiles + c // 4) * 32 + r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    out = torch.zeros((s, row_tiles * 128 * col_tiles * 4), dtype=torch.uint8, device=sf.device)
    out[:, dest.reshape(-1)] = sf.reshape(s, -1)
    return out


def megamoe_expert_scalars(
    input_scale_fc1: torch.Tensor,
    input_scale_fc2: torch.Tensor,
    weight_scale_2_w1: torch.Tensor,
    weight_scale_2_w3: torch.Tensor,
    weight_scale_2_w2: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """The kernel's per-slot scalars from a modelopt NVFP4 checkpoint's `input_scale` and
    `weight_scale_2` (both are dequant factors `amax / (448*6)`).

    `input_scale_fc1`: ONE scalar for the whole layer, the maximum over **every** expert's w1 and w3
    `input_scale` (not just this rank's slots): the activation is quantized once, before dispatch,
    for all experts. `input_scale_fc2`, `weight_scale_2_*`: `[S]`, one per slot, of the expert the
    slot holds. Returns, all float32:

    - `input_global_scale` `[1]` = `1 / input_scale_fc1`, the `global_scale` of `fp4_quantize`;
    - `fc1_alpha` `[S]` = `input_scale_fc1 * ws13`, with `ws13 = max(weight_scale_2_w1, weight_scale_2_w3)` the
      one weight scalar of the fused FC1 (`fc1_global_scale`; stock's CUTLASS path also takes the larger), evaluated
      as the engine's NVFP4 loader does, `1 / ((1 / input_scale_fc1) * (1 / ws13))` (the float32 rounding differs
      from the plain product);
    - `fc2_alpha` `[S]` = `input_scale_fc2 * weight_scale_2_w2`, likewise `1 / ((1 / input_scale_fc2) *
      (1 / weight_scale_2_w2))`;
    - `fc1_norm_const` `[S]` = `1 / input_scale_fc2`, the global scale of the FC1-output NVFP4
      requantization (so FC2's activation scale is the expert's own);
    - `w1_scale_fold`, `w3_scale_fold` `[S]` = `weight_scale_2_w1 / ws13`, `weight_scale_2_w3 / ws13` (`<= 1`),
      for `megamoe_pack_experts`.
    """
    ws13, w1_fold, w3_fold = fc1_global_scale(weight_scale_2_w1, weight_scale_2_w3)
    g1 = input_scale_fc1.float().reshape(()).reciprocal()
    g2 = input_scale_fc2.float().reciprocal()
    return {
        "input_global_scale": g1.reshape(1).contiguous(),
        "fc1_alpha": (1.0 / (g1 * (1.0 / ws13))).contiguous(),
        "fc2_alpha": (1.0 / (g2 * (1.0 / weight_scale_2_w2.float()))).contiguous(),
        "fc1_norm_const": g2.contiguous(),
        "w1_scale_fold": w1_fold,
        "w3_scale_fold": w3_fold,
    }


megamoe_pack_experts = MegaMoePackExperts()
