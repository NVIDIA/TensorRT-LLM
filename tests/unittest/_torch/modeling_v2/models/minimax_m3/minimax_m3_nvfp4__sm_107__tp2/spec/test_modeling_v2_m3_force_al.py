# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the per-request forced-acceptance draw (``FORCE_AL_PER_REQUEST``, the ``m3_force_vec_req`` entry,
synth only): each request's forced count comes from its own key (its KV length at the verify), so it does not depend
on the call history, on the request's slot in the batch, or on the other requests."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.spec.m3_force_vec import (  # noqa: E501
    m3_force_vec_req,
)
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SEED,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _pool():
    g = torch.Generator(device="cpu")
    g.manual_seed(_FORCE_ACCEPT_RNG_SEED)
    return torch.rand(_FORCE_ACCEPT_RNG_POOL_SIZE, dtype=torch.float32, generator=g).cuda()


def _draw(pool, keys, base_total=2, max_total=4, frac=0.6):
    out = torch.empty(keys.numel(), dtype=torch.int32, device="cuda")
    m3_force_vec_req(
        pool,
        keys,
        out,
        keys.numel(),
        base_total,
        max_total,
        frac,
        _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
        _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
        max(16, 1 << (keys.numel() - 1).bit_length()),
    )
    return out


def test_a_request_draws_the_same_whatever_ran_before_or_beside_it() -> None:
    pool = _pool()
    keys = torch.tensor([165_521, 40_003, 323_007, 7], dtype=torch.int32, device="cuda")
    first = _draw(pool, keys)
    for _ in range(5):  # no shared counter: repeated calls draw the same
        assert torch.equal(_draw(pool, keys), first)
    perm = torch.tensor([2, 0, 3, 1], device="cuda")
    assert torch.equal(_draw(pool, keys[perm]), first[perm])  # slot-independent
    alone = torch.cat([_draw(pool, keys[i : i + 1]) for i in range(4)])
    assert torch.equal(alone, first)  # independent of the other requests


def test_the_counts_follow_the_forced_fraction() -> None:
    pool = _pool()
    keys = torch.arange(100_000, 100_000 + 64 * 400, dtype=torch.int32, device="cuda")
    counts = torch.cat([_draw(pool, keys[i : i + 64]) for i in range(0, keys.numel(), 64)])
    assert set(counts.tolist()) <= {2, 3}
    extra = (counts == 3).float().mean().item()
    assert abs(extra - 0.6) < 0.02, extra
