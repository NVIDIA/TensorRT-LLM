# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_force_vec_req entry of the MiniMax-M3 tp4 target: the per-request fractional forced-acceptance
draw (FORCE_AL_PER_REQUEST: one draw per request at its own key, its KV length at the verify, no shared counter), so
a request's forced count does not depend on the call history, on its slot in the batch, or on the other requests."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.spec.m3_force_vec_req import (  # noqa: E501
    m3_force_vec_req as op,
)
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SEED,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_force_vec_req requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(7)
    pool = torch.rand(_FORCE_ACCEPT_RNG_POOL_SIZE, generator=g, device="cuda")
    # Each request's key is its KV length at the verify: long, distinct, arbitrary.
    keys = torch.randint(
        1, 1 << 20, (spec["batch"],), generator=g, device="cuda", dtype=torch.int64
    ).to(torch.int32)
    out = torch.empty(spec["batch"], dtype=torch.int32, device="cuda")
    args = (
        pool,
        keys,
        out,
        spec["batch"],
        2,
        4,
        spec["frac"],
        _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
        _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
        spec["block"],
    )
    ref = op.reference(*args)
    with validating(op):
        op(*args)
    assert torch.equal(out, ref)


def _pool():
    g = torch.Generator(device="cpu")
    g.manual_seed(_FORCE_ACCEPT_RNG_SEED)
    return torch.rand(_FORCE_ACCEPT_RNG_POOL_SIZE, dtype=torch.float32, generator=g).cuda()


def _draw(pool, keys, base_total=2, max_total=4, frac=0.6):
    out = torch.empty(keys.numel(), dtype=torch.int32, device="cuda")
    return op(
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
