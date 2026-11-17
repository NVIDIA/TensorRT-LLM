# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs shared by the tp4 target's spec entry tests: TP4 dimensions, single-device stand-ins for the cross-rank
exchanges, and the comparison bands."""

from __future__ import annotations

import zlib

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_shard_accept import (  # noqa: E501
    SHARD_ROWS,
    make_keys,
)

#: MiniMax-M3 at TP4: hidden size, vocabulary, the target head's shard and RMSNorm epsilon.
HIDDEN = 6144
VOCAB = 200064
WORLD = 4
TARGET_SHARD = VOCAB // WORLD
EPS = 1e-6


def seed_of(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def randn(shape, g: torch.Generator, std: float = 1.0, dtype=torch.bfloat16) -> torch.Tensor:
    return (torch.randn(shape, generator=g, device="cuda") * std).to(dtype)


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


def assert_bf16_close(out: torch.Tensor, ref: torch.Tensor, max_frac: float = 1e-3) -> None:
    """At most one bf16 ulp per element, and only on a ``max_frac`` share of the elements.

    Kernel and reference compute the same fp32 expression and round once; they differ only where a different
    summation order (or rsqrtf) puts the fp32 value on the other side of a bf16 rounding boundary.
    """
    assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16
    o, r = out.float(), ref.float()
    ulp = torch.ldexp(torch.ones_like(r), torch.frexp(r).exponent - 8)
    ulp = torch.where(r == 0, torch.full_like(r, 2.0**-133), ulp)
    diff = (o - r).abs()
    assert bool((diff <= ulp * 1.0001).all()), f"max |diff| / ulp = {(diff / ulp).max().item():.2f}"
    frac = (diff > 0).float().mean().item()
    assert frac <= max_frac, f"{frac:.2e} of the elements moved by one ulp (allowed {max_frac})"


def craft_drafts(tokens: torch.Tensor, nreq: int, d: int) -> torch.Tensor:
    """Drafts matching request b's verify tokens for its first (b + 2) % (d + 1) positions and differing from
    there on, so the acceptance counts vary across requests."""
    t = tokens.view(nreq, d + 1)[:, :d].clone()
    for b in range(nreq):
        n = (b + 2) % (d + 1)
        t[b, n:] += 1
    return t.contiguous()
