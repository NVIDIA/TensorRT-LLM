# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch pieces the vocab-parallel selection entries' references share: the pair argmax and
Triton's Philox4x32-10 with the local sampler's Gumbel noise."""

from __future__ import annotations

import torch

_M32 = 0xFFFFFFFF
_ROUND_A, _ROUND_B = 0xD2511F53, 0xCD9E8D57
_KEY_A, _KEY_B = 0x9E3779B9, 0xBB67AE85


def _limbs(a: torch.Tensor, b: torch.Tensor):
    a_lo, a_hi, b_lo, b_hi = a & 0xFFFF, a >> 16, b & 0xFFFF, b >> 16
    return a_lo * b_lo, a_hi * b_lo, a_lo * b_hi, a_hi * b_hi


def _mulhi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The high 32 bits of the 64-bit product of two uint32 (held in int64)."""
    ll, hl, lh, hh = _limbs(a, b)
    cross = (ll >> 16) + (hl & 0xFFFF) + (lh & 0xFFFF)
    return (hh + (hl >> 16) + (lh >> 16) + (cross >> 16)) & _M32


def _mullo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ll, hl, lh, _ = _limbs(a, b)
    return (ll + ((hl + lh) << 16)) & _M32


def philox(seed: torch.Tensor, c0, c1, c2, c3, rounds: int = 10) -> tuple[torch.Tensor, ...]:
    """``tl.philox(seed, c0, c1, c2, c3)``: uint32 words as int64 tensors (broadcast), the 64-bit
    seed split into the low (k0) and high (k1) key words."""
    seed = seed.to(torch.int64)
    k0, k1 = seed & _M32, (seed >> 32) & _M32
    c0, c1, c2, c3 = (
        torch.as_tensor(c, dtype=torch.int64, device=seed.device) & _M32 for c in (c0, c1, c2, c3)
    )
    for _ in range(rounds):
        p0, p2 = c0, c2
        c0 = _mulhi(torch.full_like(p2, _ROUND_B), p2) ^ c1 ^ k0
        c2 = _mulhi(torch.full_like(p0, _ROUND_A), p0) ^ c3 ^ k1
        c1 = _mullo(torch.full_like(p2, _ROUND_B), p2)
        c3 = _mullo(torch.full_like(p0, _ROUND_A), p0)
        k0, k1 = (k0 + _KEY_A) & _M32, (k1 + _KEY_B) & _M32
    return c0, c1, c2, c3


def gumbel_noise(seed: int, offset: int, cols: torch.Tensor) -> torch.Tensor:
    """The local sampler's FP32 Gumbel noise of global vocab ids ``cols``: lane ``v % 4`` of
    ``Philox(seed, (v // 4, offset lo, offset hi, 0))``, ``u = ((r >> 9) + 0.5) / 2^23``."""
    dev = cols.device
    words = philox(
        torch.tensor(seed, dtype=torch.int64, device=dev),
        cols // 4,
        offset & _M32,
        (offset >> 32) & _M32,
        0,
    )
    r = torch.stack(torch.broadcast_tensors(*words), dim=-1).gather(-1, (cols % 4)[..., None])[
        ..., 0
    ]
    u = ((r >> 9).to(torch.float32) + 0.5) * (1.0 / 8388608.0)
    return -torch.log(-torch.log(u))


def pair_argmax(vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Per row of ``[rows, slots]`` (value, index) candidates the index ``torch.argmax`` of the
    gathered logits picks: a NaN beats every number (the lowest index among NaNs), else the
    largest value and the lowest index among equal ones (``-0 == +0``)."""
    big = torch.iinfo(torch.int64).max
    idx = idx.to(torch.int64)
    nan = vals != vals
    any_nan = nan.any(-1, keepdim=True)
    vmax = torch.where(nan, float("-inf"), vals).amax(-1, keepdim=True)
    cand = torch.where(any_nan, nan, (vals == vmax) & ~nan)
    return torch.where(cand, idx, big).amin(-1)
