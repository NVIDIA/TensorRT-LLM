# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm::fp4_block_scale_moe_prerouted against the stock FP4 block-scale MoE op.

Both run FC1 / FC2 of the same pre-routed call (do_finalize=False) at the same tile and config: the stock
op builds its routing tables with its routing kernel, the prerouted op takes tables built here in token
order (the order ``ring_buddy_assign``'s ``meta_tile`` tables use). An expert's rows may sit in another
order in the permuted space, so the check maps every (token, slot) through each op's own
expanded->permuted table: the slots each op runs must agree, and every slot's FC2 row must be bitwise
equal. The geometries are MuseSpark's: the TEP8 two-window call (64 local experts of 256, 9..64 rows,
ring_buddy_assign's tables) and the TEP4 half-window call (96 local experts, 16..32 rows, the wide
pool_direct_route kernel's tables), top 8.
"""

import pytest
import torch

__extra_import_path__ = ["."]
from test_modeling_v2_fp4_block_scale_moe_runner import (  # noqa: E402 - needs the path declared above
    DEV,
    _build,
    _rand_activation,
    _scalars,
)

EXPERTS, TOP_K = 256, 8
HIDDEN, INTER = 2048, 1024
TILES = (8, 16, 32, 64, 128, 256)

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops.trtllm, "fp4_block_scale_moe_prerouted"),
    reason="this build registers no trtllm::fp4_block_scale_moe_prerouted",
)


def _stock_tile(rows: int, window: int) -> int:
    """run_moe's fallback tile: nextPowerOfTwo(rows * top_k / local experts), clamped."""
    avg = rows * TOP_K / window
    tile = 1
    while tile < avg:
        tile *= 2
    return min(max(tile, TILES[0]), TILES[-1])


def _tables(ids: torch.Tensor, tile: int, window: int):
    """The routing tables for ``ids`` ([rows, top_k], local ids < window run here), token order."""
    rows = ids.shape[0]
    flat = ids.reshape(-1).tolist()
    counts = [0] * window
    for e in flat:
        if 0 <= e < window:
            counts[e] += 1
    tiles = [(c + tile - 1) // tile for c in counts]
    offs, run = [], 0
    for t in tiles:
        offs.append(run)
        run += t
    ctas = run
    slots = rows * TOP_K
    bound = (
        min(window, slots) + (slots - min(window, slots)) // tile
    )  # the producers' bound on CTAs
    e2p = [-1] * slots
    p2t = [0] * (max(bound, ctas) * tile + 1)
    pos = [0] * window
    for s, e in enumerate(flat):
        if 0 <= e < window:
            e2p[s] = offs[e] * tile + pos[e]
            p2t[e2p[s]] = s // TOP_K
            pos[e] += 1
    c2b, c2m = [0] * max(bound, ctas), [0] * max(bound, ctas)
    for e in range(window):
        for t in range(tiles[e]):
            c2b[offs[e] + t] = e
            c2m[offs[e] + t] = min((offs[e] + t + 1) * tile, offs[e] * tile + counts[e])

    def i32(v):
        return torch.tensor(v, dtype=torch.int32, device=DEV)

    return dict(
        e2p=i32(e2p).view(rows, TOP_K),
        p2t=i32(p2t),
        c2b=i32(c2b),
        c2m=i32(c2m),
        nnc=i32([ctas]),
        tnp=i32([ctas * tile]),
    )


@pytest.mark.parametrize(
    "window,rows", [(64, 9), (64, 16), (64, 24), (64, 40), (64, 64), (96, 16), (96, 24), (96, 32)]
)
def test_prerouted_matches_stock(window, rows):
    args, ref = _build(window, HIDDEN, INTER, seed=rows)
    gen = ref["gen"]
    g1, g2 = 1.0, 1.0
    data, sf, _ = _rand_activation(rows, HIDDEN, g1, gen)
    sf = sf.view(torch.float8_e4m3fn)
    # Global ids, distinct per row: the slots with an id below window run here, the rest on other ranks.
    ids = torch.stack(
        [torch.randperm(EXPERTS, device=DEV, generator=gen)[:TOP_K] for _ in range(rows)]
    ).to(torch.int32)
    wts = torch.rand(rows, TOP_K, device=DEV, generator=gen).to(torch.bfloat16)
    o1, o1g, o2 = _scalars(window, g1, g2)
    tile = _stock_tile(rows, window)
    config = torch.ops.trtllm.fp4_block_scale_moe_default_config(
        tile, TOP_K, HIDDEN, INTER, window, rows, 0
    )

    # Every argument positionally, the trailing output (None) too: a custom class method takes no defaults.
    stock = torch.classes.trtllm.FP4BlockScaleMoERunner(0).run_moe(
        None,
        None,
        data,
        sf,
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        None,
        None,
        None,
        None,
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        None,
        o1,
        o1g,
        o2,
        EXPERTS,
        TOP_K,
        None,
        None,
        INTER,
        0,
        window,
        None,
        1,
        False,
        [tile, config],
        wts,
        ids,
        None,
    )
    t = _tables(ids, tile, window)
    zeros = torch.zeros(rows * TOP_K, dtype=torch.int32, device=DEV)
    pre = torch.ops.trtllm.fp4_block_scale_moe_prerouted(
        data,
        sf,
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        o1,
        o1g,
        o2,
        ids,
        wts,
        t["e2p"],
        t["p2t"],
        t["c2b"],
        t["c2m"],
        t["nnc"],
        t["tnp"],
        zeros,
        EXPERTS,
        TOP_K,
        INTER,
        0,
        window,
        tile,
        config,
        0,
    )
    torch.cuda.synchronize()
    s_out, s_e2p = stock[0], stock[2].reshape(-1)
    p_out, p_e2p = pre[0], pre[2].reshape(-1)
    assert torch.equal(s_e2p >= 0, p_e2p >= 0), "the two ops run different slots"
    local = (s_e2p >= 0).nonzero().flatten()
    assert torch.equal(
        s_out[s_e2p[local].long()].view(torch.int16), p_out[p_e2p[local].long()].view(torch.int16)
    ), "a slot's FC2 row differs from the stock op's"
