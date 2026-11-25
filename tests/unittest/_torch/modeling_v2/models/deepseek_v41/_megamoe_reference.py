# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch references of what the MegaMoE routing launch (`dsv4_route_nvfp4_quant(..., slot_of_expert=,
mxfp8=)`) writes, for the deepseek_v41 MegaMoE tests (not a test module; imports torch only, so CPU tests can use it).

* `megamoe_slots`: router expert ids -> slot ids through `megamoe_slot_table`'s table (device torch, graph-safe).
* `megamoe_slots_by_placement`: the same ids -> slots from the placement itself, as a host loop.
* `mxfp8_rows` / `mxfp8_scale_atoms`: the MXFP8 of the fused shared expert's input (E4M3 rows, UE8M0 scales per 32
  elements in 128x4 atoms).
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

_COPIES_SHIFT = 16
_HASH = 0x9E3779B1


def megamoe_slots(table: torch.Tensor, ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Slot ids `[rows, top_k]` int32 of router expert ids `ids` `[rows, top_k]` through `megamoe_slot_table`'s table.

    One copy per expert (`table` `[E]`): `table[e]`. Copies (`[E + slots]`, words `(copies << 16) | start` then the
    slots): row `r`'s pick takes copy `h(r) % copies`, `h(r) = ((r * 0x9E3779B1) mod 2**32) >> 16`.
    """
    if table.numel() == num_experts:
        return table[ids.long()]
    rows = torch.arange(ids.shape[0], device=ids.device, dtype=torch.int64)[:, None]
    meta = table[ids.long()].long()
    copy = (((rows * _HASH) & 0xFFFFFFFF) >> _COPIES_SHIFT) % (meta >> _COPIES_SHIFT)
    return table[num_experts + (meta & 0xFFFF) + copy].to(torch.int32)


def megamoe_slots_by_placement(
    assignments: List[int], ids: torch.Tensor, ep_rank: int
) -> torch.Tensor:
    """`megamoe_slots` from the placement (`assignments`, slot -> expert) and the rank the table was built for: copy
    `h(r) % copies` of the expert's slots, rotated by `ep_rank`, as a host loop over rows."""
    slots_of = {}
    for slot, e in enumerate(assignments):
        slots_of.setdefault(int(e), []).append(slot)
    out = []
    for r, row in enumerate(ids.cpu().tolist()):
        picks = []
        for e in row:
            s = slots_of[e]
            s = s[ep_rank % len(s) :] + s[: ep_rank % len(s)]
            picks.append(s[(((r * _HASH) & 0xFFFFFFFF) >> _COPIES_SHIFT) % len(s)])
        out.append(picks)
    return torch.tensor(out, dtype=torch.int32, device=ids.device)


def mxfp8_rows(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(E4M3 rows `[M, H]`, UE8M0 exponent bytes per 32 elements `[M, H/32]`) of `x` `[M, H]`: scale
    `2**ceil(log2(amax / 448))`, capped at `2**127`."""
    rows, k = x.shape
    b = x.float().view(rows, k // 32, 32)
    bits = (b.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    e8 = torch.clamp((bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32), max=254)
    scale = torch.exp2(e8.float() - 127.0)[..., None]
    return (b / scale).to(torch.float8_e4m3fn).view(rows, k), e8.to(torch.uint8)


def mxfp8_scale_atoms(e8: torch.Tensor) -> torch.Tensor:
    """`[M, C]` scale bytes in 128x4 atoms (32x4x4 order), zero-padded to whole atoms, flat."""
    rows, cols = e8.shape
    padded = F.pad(e8, (0, -cols % 4, 0, -rows % 128))
    blocked = padded.view((rows + 127) // 128, 4, 32, (cols + 3) // 4, 4)
    return blocked.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()
