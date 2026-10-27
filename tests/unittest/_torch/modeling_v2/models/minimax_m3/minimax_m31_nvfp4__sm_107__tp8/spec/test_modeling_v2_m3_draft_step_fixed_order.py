# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D-FIX: the draft step's tile pieces and their summation order (kernels/m3_draft_step.FIXED_ORDER).

A CTA's (tile, chunk) range is cut into 16 equal warp segments; the shared-memory atomics add a tile's pieces in
arrival order, which is deterministic only while no tile takes more than 2 pieces. ``fixed_order`` sums them in warp
order instead. The geometry tests enumerate the real grid (the shard on 212 SMs) so a grid or shard change cannot
silently bring back an order-dependent sum; the GPU tests hold the fixed order to the atomics on exact data and to the
warp-order oracle on a planted tile whose sum depends on the order.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (
    m3_draft_step as kds,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.spec.m3_draft_step import (  # noqa: E501
    DRAFT_SHARD,
    MAX_TOKENS,
    bf16_fragments,
    max_tile_pieces,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import gen, head_workspace  # noqa: E402

OP = "minimax_m3_sm107tp8_draft_step"
SMS = 212  # Rubin's SMs: the grid the draft step launches on
K = 6144
CHUNKS = K // 16
# This target's shard on Rubin: 7-8 tiles per CTA; a 7-tile CTA's tiles take up to 4 pieces (the atomics' sum is
# order-dependent, so the fixed order is on).
ORDER_DEPENDENT = True

gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)


@pytest.mark.parametrize("grid", [SMS, kds.GRID])
def test_real_grid_pieces(grid) -> None:
    """One CTA per SM and the DS-GRID launch (kernels/m3_draft_step.GRID)."""
    pieces = max_tile_pieces(DRAFT_SHARD // 16, grid, CHUNKS)
    assert (pieces > 2) == ORDER_DEPENDENT, (grid, pieces)


def test_ds_grid_launches_the_configured_grid(monkeypatch) -> None:
    monkeypatch.setattr(kds, "_num_sms", lambda: SMS)
    assert kds._grid() == min(kds.GRID, SMS)
    monkeypatch.setattr(kds, "GRID", 0)
    assert kds._grid() == SMS


def test_max_tile_pieces_counts_warp_segments() -> None:
    # One tile per CTA: all 16 warps stream it. 7 tiles per CTA: a 168-chunk segment can sit inside a tile (TP8).
    assert max_tile_pieces(16, 16, CHUNKS) == 16
    assert max_tile_pieces(7, 1, CHUNKS) == 4
    assert max_tile_pieces(8, 1, CHUNKS) == 2
    assert max_tile_pieces(15, 1, CHUNKS) == 2


@pytest.mark.parametrize("flag", [True, False])
def test_the_kernel_engages_the_fixed_order_where_the_geometry_needs_it(flag, monkeypatch) -> None:
    monkeypatch.setattr(kds, "FIXED_ORDER", flag)
    x = torch.empty(1, K)

    def head(n_tiles):
        return types.SimpleNamespace(wf=torch.empty(n_tiles, 1), exc_off=None, bnd=None)

    assert kds._fixed_order(1, head(DRAFT_SHARD // 16), x, SMS) == (flag and ORDER_DEPENDENT)
    assert kds._fixed_order(1, head(DRAFT_SHARD // 16), x, kds.GRID) == (flag and ORDER_DEPENDENT)
    assert kds._fixed_order(1, head(16), x, 16) == flag  # 16 pieces per tile


def _launcher(wf, x, fixed, nctas):
    """(launch, ids, keys): one pass's op launch on buffers made here (outside any capture), world 1."""
    uc, keys, counters = head_workspace(1, slot_rows=MAX_TOKENS)
    m = x.shape[0]
    i32 = dict(dtype=torch.int32, device="cuda")
    draft_out = torch.zeros(m, 4, **i32)
    ids = torch.zeros(m, **i32)
    pos_src, pos_next, kv_lens = (
        torch.full((m,), 100, **i32),
        torch.zeros(m, **i32),
        torch.full((m,), 500, **i32),
    )
    out = torch.zeros(m, dtype=torch.int64, device="cuda")

    def launch():
        getattr(torch.ops.trtllm, OP)(x, None, m, wf, 0, keys, counters, uc, uc.data_ptr(), 0, 1, draft_out, 1, ids,
                                      pos_src, pos_next, kv_lens, None, None, 3, nctas, False, out,
                                      fixed_order=fixed)  # fmt: skip

    return launch, ids, out


@gpu
@pytest.mark.parametrize("nctas,vs", [(SMS, DRAFT_SHARD), (16, 256)])
@pytest.mark.parametrize("m", [1, 4, 8])
def test_fixed_order_equals_the_atomics_on_exact_data(nctas, vs, m) -> None:
    """Integer data: every fp32 sum is exact in any order, so both summations give the same bits."""
    g = gen(17 + m)
    w = torch.randint(-1, 2, (vs, K), generator=g, device="cuda").to(torch.bfloat16)
    x = torch.randint(-1, 2, (m, K), generator=g, device="cuda").to(torch.bfloat16)
    wf = bf16_fragments(w)
    out = {}
    for fixed in (False, True):
        launch, ids, keys = _launcher(wf, x, fixed, nctas)
        launch()
        torch.cuda.synchronize()
        out[fixed] = (keys, ids)
    assert torch.equal(out[False][0], out[True][0])
    ref = (x.double() @ w.double().t()).float().bfloat16().float().argmax(dim=1).int()
    assert torch.equal(out[True][1], ref)


@gpu
@pytest.mark.parametrize("graph", [False, True])
def test_fixed_order_is_the_warp_order_sum_on_a_planted_tile(graph) -> None:
    """One tile per CTA (16 pieces): row 0's pieces are +2^25 (warp 0), +1 (warps 1..14), -2^25 (warp 15). In warp
    order the +1s are absorbed by 2^25 and the row sums to 0, so the competitor (row 255, logit 0.5) wins; arrival
    order could give up to 14. Every launch must give the warp-order result."""
    w = torch.zeros(256, K, device="cuda")
    seg = CHUNKS // 16 * 16  # one warp segment in K
    w[0, 0] = 2.0**25
    for s in range(1, 15):
        w[0, s * seg] = 1.0
    w[0, 15 * seg] = -(2.0**25)
    w[255, 0] = 0.5
    wf = bf16_fragments(w.to(torch.bfloat16))
    x = torch.ones(1, K, device="cuda", dtype=torch.bfloat16)
    launch, ids, _ = _launcher(wf, x, True, 16)
    run = launch
    if graph:
        launch()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        cs = torch.cuda.Stream()
        with torch.cuda.stream(cs), torch.cuda.graph(gr, stream=cs):
            launch()
        run = gr.replay
    for _ in range(16):
        ids.zero_()
        run()
        torch.cuda.synchronize()
        assert int(ids[0]) == 255
