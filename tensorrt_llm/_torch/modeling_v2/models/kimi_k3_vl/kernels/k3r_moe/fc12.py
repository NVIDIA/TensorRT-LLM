# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Counter layout and die-local claiming of the fused FC12 NVFP4 MoE kernel (SM107).

The kernel (``Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel``) takes three int32 atomic
counters that must be zero at launch: ``fc1_ready`` (one word per M tile), then one scheduler
counter per phase, each on its own 128-byte line. With die-local claiming each phase keeps one
counter per die, die 1's on the line after die 0's. :func:`fc12_counter_layout` is that layout;
:func:`fc12_counter_words` the words a caller zeroing them ahead of a call allocates.

Die-local claiming (:func:`fc12_die_sm_words`) applies to a call of a 1-CTA cluster tactic on
die-alternating expert weights (``die_locality.die_alternating_copy``) of a device whose
SM -> die table is known; each CTA then takes the tiles whose weight rows sit on its own die
before the other die's. Only the CTA that computes a tile changes.

These helpers import no CuTe DSL; the launch wrapper is :mod:`.fc12_runner`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

#: Int32 words per 128-byte line.
_INTS_PER_LINE = 32
#: Rows of the routing tile the most tiles come from (the 1-CTA tile).
_MIN_TILE = 128


def fc12_counter_layout(num_tiles: int, die_sched: bool = False) -> Tuple[int, int, int]:
    """Int32 word layout of the fused FC12 kernel's atomic counters for ``num_tiles`` M tiles.

    Returns ``(fc1_scheduler_counter, fc2_scheduler_counter, total)``: ``fc1_ready`` takes the
    first ``num_tiles`` words and each scheduler counter starts its own 128-byte line. With
    ``die_sched`` (die-local claiming) each phase has one counter per die, die 1's on the line
    after die 0's. Every word must be zero when the kernel starts.
    """
    lines_per_phase = 2 if die_sched else 1
    fc1_at = (num_tiles + _INTS_PER_LINE - 1) // _INTS_PER_LINE * _INTS_PER_LINE
    fc2_at = fc1_at + lines_per_phase * _INTS_PER_LINE
    return fc1_at, fc2_at, fc2_at + lines_per_phase * _INTS_PER_LINE


def fc12_max_tiles(num_local_experts: int, num_tokens: int, top_k: int) -> int:
    """``moe_sort``'s tile count bound at the 128-row tile (more tiles than at 256 rows) for
    ``num_tokens`` tokens with ``top_k`` experts each over ``num_local_experts``."""
    expanded = num_tokens * top_k
    filled = min(num_local_experts, expanded)
    return filled + (expanded - filled) // _MIN_TILE


def fc12_counter_words(num_local_experts: int, num_tokens: int, top_k: int, die_sched: bool) -> int:
    """Counter words of any FC12 call on ``num_tokens`` tokens: enough for every tile count the
    routing can produce, with one counter per die and phase where ``die_sched``."""
    return fc12_counter_layout(
        fc12_max_tiles(num_local_experts, num_tokens, top_k), die_sched=die_sched
    )[2]


def fc12_die_sm_words(
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    cluster_size: int,
    num_tokens: Optional[int],
    min_tokens: Optional[int],
) -> Optional[Tuple[int, ...]]:
    """The fused FC12 kernel's ``die_sm_words`` for a call of ``num_tokens`` tokens on these
    expert weights, or ``None`` (die-oblivious claiming).

    Die-local claiming applies with ``min_tokens`` set and at least that many tokens (at one
    decode request the claim overhead outweighs the locality gain), a 1-CTA cluster, both
    weights die-alternating copies and a known SM -> die table of their device.
    """
    if min_tokens is None or cluster_size != 1:
        return None
    if num_tokens is not None and num_tokens < min_tokens:
        return None
    from .die_locality import is_die_alternating, pack_sm_die_words, sm_die_table

    if not (is_die_alternating(fc1_weight) and is_die_alternating(fc2_weight)):
        return None
    table = sm_die_table(fc1_weight.device.index)
    return None if table is None else pack_sm_die_words(table)
