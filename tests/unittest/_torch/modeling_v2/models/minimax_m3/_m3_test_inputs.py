# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic inputs and unchanged numerical oracles for M3 tests."""

import zlib

import torch


def seed_of(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def randn(shape, g: torch.Generator, std: float = 1.0, dtype=torch.bfloat16) -> torch.Tensor:
    return (torch.randn(shape, generator=g, device="cuda") * std).to(dtype)


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
