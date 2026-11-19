# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's runner sort (``runner_sort``).

The kernel's claim is bitwise: the pre-routed layout of one rank's experts (local experts ascending,
each expert's rows in ascending token order, ceil(count / tile) tiles per touched expert, -1 for
other ranks' picks and for padding rows) equals a torch reference, on the one-CTA launch (the
shared-memory p2t) and a cluster.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the runner sort is a CuTe DSL kernel", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("the runner sort runs on SM100 / SM103", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import (  # noqa: E402
    runner_sort as rs,
)

E, K = 896, 16


def _ids(rows: int, seed: int, experts=None) -> torch.Tensor:
    """Top-16 distinct experts per row, drawn from ``experts`` (all 896 by default)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pool = torch.arange(E, device="cuda") if experts is None else experts.cuda()
    pick = torch.rand(rows, pool.numel(), generator=g, device="cuda").topk(K, dim=1).indices
    return pool[pick].int().contiguous()


def _ref(ids: torch.Tensor, offset: int, num: int, tile: int):
    """Torch runner sort: ``[e2p, p2t[:total], total, t2e[:nt], t2l[:nt], nt]``."""
    rows = ids.shape[0]
    loc = ids.reshape(-1).long() - offset
    x = torch.nonzero(
        (loc >= 0) & (loc < num)
    ).flatten()  # expanded index = token order within an expert
    exp = loc[x]
    counts = torch.bincount(exp, minlength=num)[:num]
    tiles = (counts + tile - 1) // tile
    start = (torch.cumsum(tiles, 0) - tiles) * tile
    nt = int(tiles.sum())
    order = torch.argsort(exp * (rows * K) + x)
    xs, es = x[order], exp[order]
    perm = (
        start[es]
        + torch.arange(xs.numel(), device=ids.device)
        - (torch.cumsum(counts, 0) - counts)[es]
    )
    e2p = torch.full((rows * K,), -1, dtype=torch.int32, device=ids.device)
    e2p[xs] = perm.int()
    p2t = torch.full((nt * tile,), -1, dtype=torch.int32, device=ids.device)
    p2t[perm] = (xs // K).int()
    t2e = torch.repeat_interleave(torch.arange(num, device=ids.device), tiles)
    t2l = torch.minimum((torch.arange(nt, device=ids.device) + 1) * tile, start[t2e] + counts[t2e])
    return e2p, p2t, nt * tile, t2e.int(), t2l.int(), nt


def _assert_equal(outs, ref) -> None:
    e2p, p2t, total, t2e, t2l, nt = outs
    r_e2p, r_p2t, r_total, r_t2e, r_t2l, r_nt = ref
    assert int(nt[0]) == r_nt and int(total[0]) == r_total
    assert torch.equal(e2p, r_e2p)
    assert torch.equal(p2t[:r_total], r_p2t)
    assert torch.equal(t2e[:r_nt], r_t2e)
    assert torch.equal(t2l[:r_nt], r_t2l)


def _run(ids, offset, num, tile):
    outs = rs.runner_sort(ids, offset, num, tile)
    assert outs[1].numel() == rs.max_tiles_for(ids.shape[0], num, tile) * tile
    _assert_equal(outs, _ref(ids, offset, num, tile))


@pytest.mark.parametrize(
    "rows,num,offset,tile",
    [
        (24, 224, 0, 8),  # one-CTA decode (224 local experts)
        (2688, 56, 448, 128),  # DEP16 receive side (56 local experts)
    ],
)
def test_bitwise(rows, num, offset, tile) -> None:
    for seed in range(2):
        _run(_ids(rows, seed), offset, num, tile)
