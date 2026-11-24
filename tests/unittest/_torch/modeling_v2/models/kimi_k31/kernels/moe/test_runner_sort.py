# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's runner sort (``runner_sort``).

The kernel's claim is bitwise: the pre-routed layout of one rank's experts (local experts ascending,
each expert's rows in ascending token order, ceil(count / tile) tiles per touched expert, -1 for
other ranks' picks and for padding rows) equals a torch reference, over the TEP / DEP16 / prefill
shapes, every launch configuration (one CTA with the shared-memory p2t, clusters of 2..16 CTAs, the
two-launch path), the edge cases and a CUDA-graph replay.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the runner sort is a CuTe DSL kernel", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("the runner sort runs on SM100 / SM103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (  # noqa: E402
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
        # TEP32 (112 local experts) / TEP16 (224) decode
        (2, 112, 0, 8),
        (8, 112, 784, 8),
        (24, 112, 336, 8),
        (24, 224, 0, 8),
        (72, 224, 224, 16),
        (144, 224, 448, 16),
        (192, 224, 672, 16),
        (192, 112, 0, 8),
        (256, 224, 0, 128),
        # DEP16 receive side (56 local experts)
        (1536, 56, 0, 64),
        (2240, 56, 280, 64),
        (2688, 56, 448, 128),
        (3584, 56, 840, 128),
        # cluster sizes 2 / 4 / 8 and the two-launch prefill path
        (257, 112, 112, 8),
        (700, 224, 0, 16),
        (1500, 112, 0, 32),
        (8192, 112, 0, 128),
        (16384, 56, 56, 128),
        (16385, 112, 0, 64),
        (65536, 112, 336, 128),
        (65536, 56, 840, 128),
    ],
)
def test_bitwise(rows, num, offset, tile) -> None:
    for seed in range(2):
        _run(_ids(rows, seed), offset, num, tile)


@pytest.mark.parametrize("offset", [0, 448])
@pytest.mark.parametrize("rows", list(range(1, 33)) + [33, 47, 64, 100, 256])
def test_tep8_decode(rows, offset) -> None:
    """TEP8 (448 local experts) decode and verify rows on the one-CTA layout, with and without the early release."""
    tile = 8 if rows < 48 else 16
    for seed in range(2):
        ids = _ids(rows, 100 + seed)
        ref = _ref(ids, offset, 448, tile)
        _assert_equal(rs.runner_sort(ids, offset, 448, tile), ref)
        _assert_equal(rs.runner_sort(ids, offset, 448, tile, early_release=True), ref)


@pytest.mark.parametrize("rows", [1, 3, 200, 3000, 20000])
def test_edge_cases(rows) -> None:
    # No pick in the window: no tiles, e2p all -1.
    outside = torch.cat([torch.arange(0, 112), torch.arange(224, E)])
    ids = _ids(rows, 1, outside)
    outs = rs.runner_sort(ids, 112, 112, 8)
    _assert_equal(outs, _ref(ids, 112, 112, 8))
    assert int(outs[5][0]) == 0 and bool((outs[0] == -1).all())
    # Every pick local (16 of the window's 56 experts), so experts get many tiles of rows.
    ids = _ids(rows, 2, torch.arange(840, 896))
    _run(ids, 840, 56, 64)
    # All 896 experts local.
    _run(_ids(rows, 3), 0, E, 16)
    # One expert in every row (> tile rows for it), the rest random.
    ids = _ids(rows, 4, torch.arange(1, E))
    ids[:, 7] = 0
    _run(ids, 0, 112, 8)


def test_graph_replay_and_reuse() -> None:
    """Captured launches re-read the ids on replay; the two-launch path's workspace stays reusable."""
    dev = torch.device("cuda")
    rs.runner_sort_workspace(dev)
    cases = [(24, 224, 0, 8), (3584, 56, 840, 128), (20000, 112, 112, 128)]
    bufs = []
    for rows, num, offset, tile in cases:
        mt = rs.max_tiles_for(rows, num, tile)
        ids = _ids(rows, 0)
        outs = [
            torch.empty(n, dtype=torch.int32, device=dev)
            for n in (rows * K, mt * tile, 1, mt, mt, 1)
        ]
        bufs.append((ids, outs))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for (rows, num, offset, tile), (ids, outs) in zip(cases, bufs):
            rs.run_runner_sort(ids, offset, num, tile, *outs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for (rows, num, offset, tile), (ids, outs) in zip(cases, bufs):
            rs.run_runner_sort(ids, offset, num, tile, *outs)
    for seed in range(1, 4):
        for (rows, _num, _offset, _tile), (ids, outs) in zip(cases, bufs):
            ids.copy_(_ids(rows, seed))
            for o in outs:
                o.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        for (rows, num, offset, tile), (ids, outs) in zip(cases, bufs):
            _assert_equal(outs, _ref(ids, offset, num, tile))
    assert int(rs.runner_sort_workspace(dev)[0]) == 0


@pytest.mark.parametrize(
    "rows, num, offset, tile", [(200, 56, 0, 128), (1536, 56, 840, 128), (4096, 56, 280, 128)]
)
def test_expanded_p2t(rows, num, offset, tile) -> None:
    """``expanded``: p2t holds ``token * 16 + slot`` (the token sort's rows otherwise unchanged, padding -1), and every
    expanded index maps back to its row through e2p."""
    ids = _ids(rows, 5)
    e2p, p2t, total, t2e, t2l, nt = rs.runner_sort(ids, offset, num, tile)
    dev = torch.device("cuda")
    mt = rs.max_tiles_for(rows, num, tile)
    outs = [
        torch.empty(n, dtype=torch.int32, device=dev) for n in (rows * K, mt * tile, 1, mt, mt, 1)
    ]
    rs.run_runner_sort(ids, offset, num, tile, *outs, expanded=True)
    torch.cuda.synchronize()
    tot, ntl = int(total[0]), int(nt[0])
    assert int(outs[2][0]) == tot and int(outs[5][0]) == ntl
    assert torch.equal(outs[0], e2p)
    assert torch.equal(outs[3][:ntl], t2e[:ntl]) and torch.equal(outs[4][:ntl], t2l[:ntl])
    p2x, p2t = outs[1][:tot].long(), p2t[:tot].long()
    pad = p2t < 0
    assert torch.equal(p2x[pad], p2t[pad])
    assert torch.equal(p2x[~pad] // K, p2t[~pad])
    assert torch.equal(e2p.long()[p2x[~pad]], torch.arange(tot, device=dev)[~pad])


def test_rejects() -> None:
    ids = _ids(8, 0)
    with pytest.raises(ValueError, match="tile"):
        rs.runner_sort(ids, 0, 112, 12)
    with pytest.raises(ValueError, match="int32"):
        rs.runner_sort(ids.long(), 0, 112, 8)
    with pytest.raises(ValueError, match="out of range"):
        rs.runner_sort(ids, 800, 112, 8)
