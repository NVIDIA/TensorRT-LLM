# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert placement for the MegaMoE kernel: which expert each slot holds, and the table that maps the
router's expert ids onto slots (plain torch, host side, run when the target is built).

`megamoe_default_placement` (the engine's placement when nothing is configured: rank `r` holds
`num_slots // ep_size` consecutive experts from `r * E // ep_size`, wrapping modulo `E`, so with extra slots
each rank also holds the first experts of the next rank's window), `megamoe_check_placement` (every expert
held, never twice on one rank) and `megamoe_slot_table` (the int32 device table the routing launch,
`dsv4_route_nvfp4_quant(..., slot_of_expert=)`, reads; rewrite it in place with `table.copy_` after moving
experts so CUDA graphs that captured it route to the new slots).
"""

from __future__ import annotations

from typing import List

import torch

_COPIES_SHIFT = 16
_MAX_SLOTS = 1 << 16


def megamoe_default_placement(num_experts: int, num_slots: int, ep_size: int) -> List[int]:
    """Slot -> expert list of the engine's default (no learned statistics) placement.

    Rank `r` holds `num_slots // ep_size` consecutive experts starting at expert
    `r * num_experts // ep_size`, wrapping modulo `num_experts`. With `num_slots == num_experts` that
    is the contiguous one-copy split; with extra slots (`num_slots = num_experts + ep_size * x`) each
    rank also holds the first experts of the next rank's window, so every expert is held and no rank
    holds an expert twice.
    """
    if num_slots % ep_size or num_slots < num_experts:
        raise ValueError(
            f"num_slots {num_slots} must be a multiple of ep_size {ep_size} and >= {num_experts}"
        )
    local = num_slots // ep_size
    return [
        (r * num_experts // ep_size + j) % num_experts for r in range(ep_size) for j in range(local)
    ]


def megamoe_check_placement(assignments: List[int], num_experts: int, ep_size: int) -> None:
    """Raise unless `assignments` (slot -> expert) holds every expert, never twice on one rank."""
    if len(assignments) % ep_size:
        raise ValueError("number of slots must be a multiple of ep_size")
    if len(assignments) >= _MAX_SLOTS:
        raise ValueError("MegaMoE replica placement must hold every expert in < 65536 slots")
    if set(assignments) != set(range(num_experts)):
        raise ValueError("placement must hold every expert at least once")
    local = len(assignments) // ep_size
    for r in range(ep_size):
        mine = assignments[r * local : (r + 1) * local]
        if len(set(mine)) != len(mine):
            raise ValueError(f"rank {r} holds an expert twice")


def megamoe_slot_table(
    assignments: List[int], num_experts: int, ep_rank: int, device="cuda"
) -> torch.Tensor:
    """The int32 device table the routing launch reads, for the placement `assignments`.

    One copy per expert (`len(assignments) == num_experts`): `table[e]` is `e`'s slot, `[num_experts]`.
    Copies: `num_experts` words `(copies << 16) | start` followed by every expert's slots, rotated by
    `ep_rank` so ranks start on different copies, `[num_experts + len(assignments)]`. The routing launch sends row `r`'s
    pick of an expert to copy `h(r) % copies`, `h(r) = ((r * 0x9E3779B1) mod 2**32) >> 16`. Built on the host
    once; rewrite it in place (`table.copy_(megamoe_slot_table(...))`) when experts move, so CUDA
    graphs that captured it route to the new slots.
    """
    if len(assignments) == num_experts:
        table = [0] * num_experts
        for slot, expert in enumerate(assignments):
            table[expert] = slot
        return torch.tensor(table, dtype=torch.int32, device=device)
    slots_of: List[List[int]] = [[] for _ in range(num_experts)]
    for slot, expert in enumerate(assignments):
        slots_of[int(expert)].append(slot)
    meta, flat = [], []
    for slots in slots_of:
        if not slots or len(assignments) >= _MAX_SLOTS:
            raise ValueError("MegaMoE replica placement must hold every expert in < 65536 slots")
        k = ep_rank % len(slots)
        meta.append((len(slots) << _COPIES_SHIFT) | len(flat))
        flat.extend(slots[k:] + slots[:k])
    return torch.tensor(meta + flat, dtype=torch.int32, device=device)
