# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs shared by the tp4 target's spec entry tests: TP4 dimensions, single-device stand-ins for the cross-rank
exchanges, and the comparison bands."""

from __future__ import annotations

import torch

__extra_import_path__ = ["../.."]
from _m3_test_inputs import assert_bf16_close as assert_bf16_close
from _m3_test_inputs import craft_drafts as craft_drafts
from _m3_test_inputs import gen as gen
from _m3_test_inputs import num_sms as num_sms
from _m3_test_inputs import randn as randn
from _m3_test_inputs import seed_of as seed_of  # noqa: E402

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.spec.m3_shard_accept import (  # noqa: E501
    SHARD_ROWS,
    make_keys,
)

#: MiniMax-M3 at TP4: hidden size, vocabulary, the target head's shard and RMSNorm epsilon.
HIDDEN = 6144
VOCAB = 200064
WORLD = 2
TARGET_SHARD = VOCAB // WORLD
EPS = 1e-6


def head_workspace(world: int, call: int = 0, slot_rows: int = SHARD_ROWS):
    """(ws_uc int64 [2 * slot_rows * world], keys int64 [slot_rows], counters int32 [2]) of a fresh call ``call``."""
    uc = torch.zeros(2 * slot_rows * world, dtype=torch.int64, device="cuda")
    keys = torch.zeros(slot_rows, dtype=torch.int64, device="cuda")
    counters = torch.tensor([0, call], dtype=torch.int32, device="cuda")
    return uc, keys, counters


def plant_peers(
    uc: torch.Tensor,
    counters: torch.Tensor,
    own_best: torch.Tensor,
    rank: int,
    world: int,
    shard: int,
    g: torch.Generator,
    slot_rows: int = SHARD_ROWS,
    ties: bool = False,
) -> None:
    """Write the other ranks' keys for each row into this rank's slots, as their multicast pushes would.

    Each peer's maximum is the own row maximum shifted by +-0.25 (well clear of the accumulation-order band), so
    some rows are won by a peer and some by this rank; with ``ties`` a peer on an even row reports exactly the own
    maximum, and the lower global index must win.
    """
    if world == 1:
        return
    m = own_best.shape[0]
    half = int(counters[1].item()) & 1
    for r in range(world):
        if r == rank:
            continue
        sign = torch.where(torch.rand(m, generator=g, device="cuda") < 0.5, -1.0, 1.0)
        value = own_best.float() + 0.25 * sign
        if ties:
            value = torch.where(torch.arange(m, device="cuda") % 2 == 0, own_best.float(), value)
        index = r * shard + torch.randint(0, shard, (m,), generator=g, device="cuda")
        keys = make_keys(value, index)
        for t in range(m):
            uc[(half * slot_rows + t) * world + r] = keys[t]
