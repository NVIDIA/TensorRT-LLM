# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`moe_route_front` (`ops/moe/moe_route_front__decode`): routing, pre-routed tables and NVFP4
activation of a decode MoE block's gathered rows in one launch.

* **GPU, every cell**, eagerly and replayed from a CUDA graph: the codes and block scales are
  bitwise `fp4_quantize`'s, the experts and weights bitwise `renorm_moe_routing_op`'s, and the
  tables hold their meaning (`check_tables`) for the cell's window and tile. Logits with
  repeated values exercise the top-k's ties.
* **CPU** (`check_tables` itself): a table built the way the runner reads it passes, and each
  way of breaking it is named.

The window output through the runner entered pre-routed is held bitwise to the runner's own
routing by the target's build check (`modeling._route_front_check`) on every layer.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import moe_route_front__decode as op

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the route front is certified on SM103",
)


def _operands(spec, seed: int, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, hidden = spec["rows"], spec["hidden"]
    x = torch.randn([rows, hidden], generator=gen, device=device).to(torch.bfloat16)
    logits = torch.randn([rows, op.NUM_EXPERTS], generator=gen, device=device)
    # Coarse logits: bf16 ties inside the top-10 and at its edge.
    logits = (logits * 4).round().div(4).to(torch.bfloat16)
    global_scale = torch.tensor([448.0 * 6.0 / 8.0], dtype=torch.float32, device=device)
    n = sum(op.table_sizes(rows, op.TOP_K, spec["num_local"], spec["tile"]))
    tables = (torch.arange(n, device=device) % rows).to(torch.int32)
    return logits, x, global_scale, tables


def _assert_matches(spec, logits, x, global_scale, tables, got):
    xq, xsf, ids, wts = got
    rxq, rxsf, rids, rwts = op.moe_route_front.reference(
        logits, x, global_scale, tables, spec["tile"], spec["local_offset"], spec["num_local"]
    )
    for name, a, b in (("codes", xq, rxq), ("scales", xsf, rxsf), ("experts", ids, rids)):
        assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)), name
    assert torch.equal(wts.view(torch.int16), rwts.view(torch.int16)), "weights"
    views = op.table_views(tables, x.shape[0], op.TOP_K, spec["num_local"], spec["tile"])
    why = op.check_tables(views, ids, spec["local_offset"], spec["num_local"], spec["tile"])
    assert not why, why


@_GPU
@pytest.mark.parametrize("cell", op.moe_route_front.CELLS, ids=lambda c: c.why[:40])
def test_certified_cells(cell):
    spec, device = cell.spec, torch.device("cuda")
    seed = zlib.crc32(cell.why.encode()) % 2**31
    logits, x, global_scale, tables = _operands(spec, seed, device)
    fast2 = spec.get("fast2", False)
    assert op.compile_front(
        spec["rows"], spec["hidden"], x.stride(0), spec["num_local"], spec["tile"], fast2
    )
    args = (spec["tile"], spec["local_offset"], spec["num_local"], fast2)
    got = op.moe_route_front(logits, x, global_scale, tables, *args)
    _assert_matches(spec, logits, x, global_scale, tables, got)

    # Captured once, replayed over fresh inputs in the static buffers.
    s_logits, s_x = logits.clone(), x.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = op.moe_route_front(s_logits, s_x, global_scale, tables, *args)
    torch.cuda.current_stream().wait_stream(stream)
    for k in range(3):
        logits, x, _, _ = _operands(spec, seed + 1 + k, device)
        s_logits.copy_(logits)
        s_x.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches(spec, s_logits, s_x, global_scale, tables, out)


@_GPU
@pytest.mark.parametrize("rows, local", [(513, 64), (544, 64), (768, 64), (1024, 64), (640, 128)])
def test_fast2_is_bitwise_the_two_token_sort(rows, local):
    """`fast2` routes a warp's two tokens on the one-token fast path: the experts, weights,
    codes, scales and tables are the same bits as the full two-token sort's."""
    device = torch.device("cuda")
    spec = dict(rows=rows, hidden=2560, local_offset=local, num_local=local, tile=32)
    logits, x, global_scale, tables = _operands(spec, 7000 + rows, device)
    for fast2 in (False, True):
        assert op.compile_front(rows, 2560, x.stride(0), local, 32, fast2)
    base = op.moe_route_front(logits, x, global_scale, tables.clone(), 32, local, local, False)
    fast_tables = tables.clone()
    fast = op.moe_route_front(logits, x, global_scale, fast_tables, 32, local, local, True)
    torch.cuda.synchronize()
    for a, b in zip(base, fast):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    _assert_matches(spec, logits, x, global_scale, fast_tables, fast)


@_GPU
@pytest.mark.parametrize("cell", op.moe_route_front.CELLS, ids=lambda c: c.why[:40])
def test_route_only_is_the_full_launchs_routing(cell):
    """`route_front_only` (the routing cluster alone, for rows the packed gather quantized) writes
    the experts, weights and tables of the full launch over the same logits, bit for bit."""
    spec, device = cell.spec, torch.device("cuda")
    seed = zlib.crc32(cell.why.encode()) % 2**31 + 17
    logits, x, global_scale, tables = _operands(spec, seed, device)
    fast2 = spec.get("fast2", False)
    rows, hidden = spec["rows"], spec["hidden"]
    assert op.compile_front(rows, hidden, x.stride(0), spec["num_local"], spec["tile"], fast2)
    assert op.compile_front(rows, hidden, hidden, spec["num_local"], spec["tile"], fast2, True)
    args = (spec["tile"], spec["local_offset"], spec["num_local"])
    full_tables = tables.clone()
    _, _, ids, wts = op.moe_route_front(logits, x, global_scale, full_tables, *args, fast2)
    ids2, wts2 = op.route_front_only(logits, tables, *args, hidden, fast2)
    torch.cuda.synchronize()
    assert torch.equal(ids2, ids) and torch.equal(wts2.view(torch.int16), wts.view(torch.int16))
    assert torch.equal(tables, full_tables)


def _tables(ids: torch.Tensor, local_offset: int, num_local: int, tile: int) -> list:
    """The pre-routed tables the runner reads, built in plain torch: each expert's slots in
    token order from its first CTA's first row, whole tiles a CTA."""
    rows, top_k = ids.shape
    sizes = op.table_sizes(rows, top_k, num_local, tile)
    flat = torch.full([sum(sizes)], -7, dtype=torch.int32)
    e2p, p2t, cta_expert, cta_limit, n_ctas, n_padded = op.table_views(
        flat, rows, top_k, num_local, tile
    )
    local = ids.to(torch.int64) - local_offset
    e2p.fill_(-1)
    cta = 0
    for e in range(num_local):
        slots = (local == e).nonzero().tolist()  # token order, then slot order
        if not slots:
            continue
        for r, (t, k) in enumerate(slots):
            perm = cta * tile + r
            e2p[t * top_k + k] = perm
            p2t[perm] = t
        tiles = -(-len(slots) // tile)
        for c in range(tiles):
            cta_expert[cta + c] = e
            cta_limit[cta + c] = min((cta + c + 1) * tile, cta * tile + len(slots))
        cta += tiles
    n_ctas[0], n_padded[0] = cta, cta * tile
    return [e2p, p2t, cta_expert, cta_limit, n_ctas, n_padded]


def _ids(rows: int = 12, seed: int = 3) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randperm(op.NUM_EXPERTS, generator=gen)[: op.TOP_K] for _ in range(rows)]
    ).to(torch.int32)


@pytest.mark.cpu_only
def test_check_tables_passes_the_runners_layout():
    ids = _ids()
    assert op.check_tables(_tables(ids, 128, 128, 8), ids, 128, 128, 8) == ""


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "breaks, message",
    [
        (lambda v, ids: v[0].__setitem__(int((v[0] >= 0).nonzero()[0]), -1), "distinct permuted"),
        (lambda v, ids: v[1].__setitem__(int(v[0][v[0] >= 0][0]), 17), "another token"),
        (lambda v, ids: v[4].__setitem__(0, int(v[4][0]) + 1), "CTA count"),
        (lambda v, ids: v[2].__setitem__(0, int(v[2][0]) + 1), "another expert's CTA"),
        (lambda v, ids: v[3].__setitem__(0, int(v[3][0]) - 8), "row limit"),
    ],
)
def test_check_tables_names_a_broken_table(breaks, message):
    ids = _ids()
    views = _tables(ids, 128, 128, 8)
    breaks(views, ids)
    assert message in op.check_tables(views, ids, 128, 128, 8)


@pytest.mark.cpu_only
def test_check_tables_rejects_a_route_into_spare_capacity():
    """One live route in a window of tile 8: one CTA, 8 padded rows. Moved to permuted row 8,
    with the spare entries naming its token, its expert and a limit of 9, every per-entry check
    holds, but the runner launches one CTA and never serves row 8."""
    ids = torch.tensor(
        [[130, 0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32
    )  # one slot in 128..255
    views = _tables(ids, 128, 128, 8)
    assert op.check_tables(views, ids, 128, 128, 8) == ""
    e2p, p2t, cta_expert, cta_limit, n_ctas, n_padded = views
    assert int(n_ctas[0]) == 1 and int(n_padded[0]) == 8 and int(e2p[0]) == 0
    e2p[0], p2t[8], cta_expert[1], cta_limit[1] = 8, 0, int(cta_expert[0]), 9
    assert "past the padded rows" in op.check_tables(views, ids, 128, 128, 8)
