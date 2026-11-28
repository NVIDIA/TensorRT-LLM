# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared expert as extra tiles of the MegaMoE kernel: MXFP8 weight packing (plain torch, load time)
and the buffers of its MXFP8 input (written by the routing launch, `dsv4_route_nvfp4_quant(..., mxfp8=)`).

Both helpers run when the target is built: `megamoe_pack_shared_expert` (FP8 weights with UE8M0 block scales into the
kernel's layout, every dequantized weight unchanged; the checkpoint's 32x32 blocks, and 128x128 bit-equal to the
engine's packing; needs `H % 512 == 0` and `2I % 512 == 0`; `fc1_weight` gate/up interleaved by 16 rows, scales per
row per 32 in 128x4 atoms, `gate_up_clamp` None when the limit is missing or <= 0) and
`megamoe_shared_activation_buffers` (per-bucket E4M3 rows and scale atoms shared by all layers; allocate every bucket
at build time, outside any graph capture).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

_GATE_UP_INTERLEAVE = 16  # the routed FC1 interleaves gate and up rows in groups of this many
_MXFP8_BLOCK = 32  # elements per UE8M0 scale along K
_CLUSTER_ROWS = 512  # gate/up and down outputs tile clusters of up to four 128-row CTAs


def _row_scales(scale: torch.Tensor, rows: int, cols: int, row_block: int, col_block: int):
    """`[rows, cols/32]` UE8M0 bytes of a `[ceil(rows/rb), ceil(cols/cb)]` block-scale matrix."""
    if row_block <= 0 or col_block <= 0 or col_block % _MXFP8_BLOCK:
        # a column block that is not whole 32-column groups would hand a group the next block's exponent
        raise ValueError(
            f"scale block {(row_block, col_block)}: both positive, columns a multiple of {_MXFP8_BLOCK}"
        )
    s = scale.contiguous().view(torch.uint8)
    s = s.repeat_interleave(row_block, dim=0)[:rows]
    s = s.repeat_interleave(col_block // _MXFP8_BLOCK, dim=1)[:, : cols // _MXFP8_BLOCK]
    return s


def _interleave_gate_up(t: torch.Tensor, intermediate: int) -> torch.Tensor:
    """`[gate | up]` rows -> `[gate 0:16, up 0:16, gate 16:32, ...]`."""
    groups = intermediate // _GATE_UP_INTERLEAVE
    gate = t[:intermediate].reshape(groups, _GATE_UP_INTERLEAVE, *t.shape[1:])
    up = t[intermediate:].reshape(groups, _GATE_UP_INTERLEAVE, *t.shape[1:])
    return torch.stack((gate, up), dim=1).reshape(t.shape).contiguous()


def _blocked(row_scales: torch.Tensor) -> torch.Tensor:
    """`[rows, cols]` scale bytes in 128x4 atoms (32x4x4 order), zero-padded to whole atoms, flat."""
    rows, cols = row_scales.shape
    padded = F.pad(row_scales, (0, -cols % 4, 0, -rows % 128))
    blocked = padded.view((rows + 127) // 128, 4, 32, (cols + 3) // 4, 4)
    return blocked.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def megamoe_pack_shared_expert(
    w1: torch.Tensor,
    s1: torch.Tensor,
    w3: torch.Tensor,
    s3: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    swiglu_limit: Optional[float],
    block: Tuple[int, int] = (32, 32),
) -> Dict[str, object]:
    """MegaMoE-format tensors of one gated FP8 shared expert, with the dequantized weights unchanged.

    `w1` (gate), `w3` (up) `[I, H]` and `w2` (down) `[H, I]` are `float8_e4m3fn`; `s1`, `s3`, `s2`
    their UE8M0 block scales (any 1-byte dtype, the exponent byte `e` meaning `2**(e-127)`) of one
    scale per `block` = (rows, cols) elements, e.g. `[I/32, H/32]` for the checkpoint's 32x32. Every
    scale must be a power of two, which UE8M0 is. `H` and `2*I` must be multiples of 512.

    Returns `fc1_weight` `[2I, H]` fp8 (gate and up interleaved by 16 rows), `fc1_weight_sf` (flat
    u8, per-row per-32 scales in 128x4 atoms), `fc2_weight` `[H, I]`, `fc2_weight_sf`,
    `intermediate_size` and `gate_up_clamp` (None for a non-positive or missing limit).
    """
    inter, hidden = w1.shape
    if (
        w3.shape != (inter, hidden)
        or w2.shape != (hidden, inter)
        or (2 * inter) % _CLUSTER_ROWS
        or hidden % _CLUSTER_ROWS
    ):
        raise ValueError(
            f"shared expert shapes {tuple(w1.shape)} {tuple(w3.shape)} {tuple(w2.shape)} are not "
            f"fusable: H and 2*I must be multiples of {_CLUSTER_ROWS}"
        )
    for w in (w1, w3, w2):
        assert w.dtype == torch.float8_e4m3fn, "shared expert weights must be float8_e4m3fn"
    rb, cb = block
    gate_up = torch.cat([w1, w3], dim=0)
    gate_up_s = torch.cat(
        [_row_scales(s1, inter, hidden, rb, cb), _row_scales(s3, inter, hidden, rb, cb)], dim=0
    )
    fc1 = _interleave_gate_up(gate_up.view(torch.uint8), inter).view(torch.float8_e4m3fn)
    fc1_s = _interleave_gate_up(gate_up_s, inter)
    fc2_s = _row_scales(s2, hidden, inter, rb, cb)
    limit = None if swiglu_limit is None or swiglu_limit <= 0 else float(swiglu_limit)
    return {
        "fc1_weight": fc1.contiguous(),
        "fc1_weight_sf": _blocked(fc1_s),
        "fc2_weight": w2.contiguous(),
        "fc2_weight_sf": _blocked(fc2_s),
        "intermediate_size": inter,
        "gate_up_clamp": limit,
    }


def megamoe_shared_activation_buffers(
    max_tokens: int, hidden: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(E4M3 rows `[max_T, H]`, UE8M0 scales `[pad128(max_T) * H/32]` in 128x4 atoms) of a launch
    bucket's MXFP8 shared-expert input. One pair per (device, bucket, H), shared by every layer;
    allocate every bucket's at build time, outside any graph capture."""
    from tensorrt_llm._torch.moe.fused_moe.mega_moe.shared_expert import shared_activation_buffers

    return shared_activation_buffers(max_tokens, hidden, torch.device(device))
