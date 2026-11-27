# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`moe_route_front` (`ops/moe/moe_route_front__decode`): routing, pre-routed tables and NVFP4
activation of a decode MoE block's gathered rows in one launch.

* **GPU, every cell**, eagerly and replayed from a CUDA graph: the codes and block scales are
  bitwise `fp4_quantize`'s, the experts and weights bitwise `renorm_moe_routing_op`'s, and the
  tables hold their meaning (`check_tables`) for the cell's window and tile. Logits with
  repeated values exercise the top-k's ties.
* **GPU**: `route_front_only` alone, and with `ids_in` over rows routed by `routed_rows`, writes
  the full launch's experts, weights and tables.
* **GPU**: `fast2=2` (the second token's logits loaded first) and `early` (the release right
  after the wait) write the same bits as the plain launch; `early` under PDL has exactly one
  `griddepcontrol.wait`, first, with no global access, branch or label above it in the SASS / PTX
  and no `ld.global.nc` anywhere (compiled for the running SM103, no launch).
* **CPU** (`check_tables` itself): a table built the way the runner reads it passes, and each
  way of breaking it is named; the `ids_in` row layout.

The window output through the runner entered pre-routed is held bitwise to the runner's own
routing by the target's build check (`modeling._route_front_check`) on every layer.
"""

import glob
import os
import subprocess
import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import moe_route_front__decode as op

# `_sass_pdl` is imported inside the SASS test (a test helper of the modeling_v2 test dir).
__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]

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
    fast2, early = spec.get("fast2", False), spec.get("early", False)
    assert op.compile_front(
        spec["rows"],
        spec["hidden"],
        x.stride(0),
        spec["num_local"],
        spec["tile"],
        fast2,
        early=early,
    )
    args = (spec["tile"], spec["local_offset"], spec["num_local"], fast2, early)
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
@pytest.mark.parametrize("rows, local, tile", [(528, 128, 32), (640, 128, 32), (832, 128, 64)])
@pytest.mark.parametrize("ties", [False, True])
def test_preload_and_early_are_bitwise_the_plain_launch(rows, local, tile, ties):
    """dep4's measured shapes (33..52 requests a rank of 16 rows): `fast2=2` and `early`, alone
    and together, write the codes, scales, experts, weights and tables of the two-token sort with
    the release at the end, bit for bit; with `ties` every row's top-10 sits in one lane's 16
    experts (the fast path's fallback to the full sort)."""
    device = torch.device("cuda")
    spec = dict(rows=rows, hidden=2560, local_offset=local, num_local=local, tile=tile)
    logits, x, global_scale, tables = _operands(spec, 9100 + rows, device)
    if ties:
        logits[:, 16:32] = (logits.abs().amax(dim=1, keepdim=True) + 1).to(torch.bfloat16)
    forms = [(0, False), (1, False), (2, False), (0, True), (2, True)]
    outs = []
    for fast2, early in forms:
        assert op.compile_front(rows, 2560, x.stride(0), local, tile, fast2, early=early)
        t = tables.clone()
        got = op.moe_route_front(logits, x, global_scale, t, tile, local, local, fast2, early)
        outs.append((got, t))
    torch.cuda.synchronize()
    (base, base_tables) = outs[0]
    for (got, t), form in zip(outs[1:], forms[1:]):
        for a, b in zip(base, got):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), form
        assert torch.equal(t, base_tables), form
    _assert_matches(spec, logits, x, global_scale, base_tables, base)


_COMPILE_EARLY = """
import sys
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import moe_route_front__decode as op
rows, local, tile, fast2, route_only, ids_in = (int(v) for v in sys.argv[1:7])
assert op.compile_front(rows, 2560, 2560, local, tile, fast2, bool(route_only), bool(ids_in), True)
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="compiles for the running SM103 GPU",
)
@pytest.mark.parametrize(
    "rows, local, tile, fast2, route_only, ids_in",
    [
        (288, 128, 32, 0, 0, 0),
        (448, 64, 32, 0, 0, 0),
        (640, 128, 32, 2, 0, 0),
        (832, 128, 64, 2, 0, 0),
        (768, 64, 32, 1, 0, 0),
        (576, 64, 32, 1, 1, 0),  # the packed gather's routing-only front
        (896, 64, 64, 1, 1, 1),  # over owner-routed rows
    ],
)
def test_early_reads_nothing_before_its_wait(
    tmp_path, rows, local, tile, fast2, route_only, ids_in
):
    """`early` releases FC1 right after the wait, so the wait must come before every global access:
    in the SASS exactly one `ACQBULK` (`griddepcontrol.wait`) with no global access, branch or label
    before it, in the PTX no global load above it, and no `ld.global.nc` anywhere (ptxas may move one
    above the wait). The release itself is not a memory access."""
    from _sass_pdl import nc_loads, pre_wait_violations

    env = dict(
        os.environ,
        CUTE_DSL_KEEP="ptx,cubin,sass",
        CUTE_DSL_DUMP_DIR=str(tmp_path),
        CUTE_DSL_CACHE_DIR=str(tmp_path / "cache"),
        CUTE_DSL_DISABLE_FILE_CACHING="1",
        TRTLLM_ENABLE_PDL="1",
    )
    args = [str(v) for v in (rows, local, tile, fast2, route_only, ids_in)]
    subprocess.run([sys.executable, "-c", _COMPILE_EARLY, *args], env=env, check=True, timeout=1800)
    sass = glob.glob(str(tmp_path / "*.sass"))
    ptx = glob.glob(str(tmp_path / "*.ptx"))
    assert len(sass) == 1 and len(ptx) == 1, (sass, ptx)
    sass_text, ptx_text = open(sass[0]).read(), open(ptx[0]).read()
    assert not pre_wait_violations(sass_text, ptx_text)
    assert nc_loads(sass_text) == 0
    assert ptx_text.count("griddepcontrol.launch_dependents") >= 1


@_GPU
@pytest.mark.parametrize("cell", op.moe_route_front.CELLS, ids=lambda c: c.why[:40])
def test_route_only_is_the_full_launchs_routing(cell):
    """`route_front_only` (the routing cluster alone, for rows the packed gather quantized) writes
    the experts, weights and tables of the full launch over the same logits, bit for bit."""
    spec, device = cell.spec, torch.device("cuda")
    seed = zlib.crc32(cell.why.encode()) % 2**31 + 17
    logits, x, global_scale, tables = _operands(spec, seed, device)
    fast2, early = spec.get("fast2", False), spec.get("early", False)
    rows, hidden = spec["rows"], spec["hidden"]
    assert op.compile_front(rows, hidden, x.stride(0), spec["num_local"], spec["tile"], fast2)
    assert op.compile_front(
        rows, hidden, hidden, spec["num_local"], spec["tile"], fast2, True, early=early
    )
    args = (spec["tile"], spec["local_offset"], spec["num_local"])
    full_tables = tables.clone()
    _, _, ids, wts = op.moe_route_front(logits, x, global_scale, full_tables, *args, fast2)
    ids2, wts2 = op.route_front_only(logits, tables, *args, hidden, fast2, early=early)
    torch.cuda.synchronize()
    assert torch.equal(ids2, ids) and torch.equal(wts2.view(torch.int16), wts.view(torch.int16))
    assert torch.equal(tables, full_tables)


@_GPU
@pytest.mark.parametrize(
    "cell",
    [c for c in op.moe_route_front.CELLS if c.spec["rows"] > op.LEAN_MAX_ROWS],
    ids=lambda c: c.why[:40],
)
def test_ids_in_is_the_routing_fronts_routing(cell):
    """`route_front_only(ids_in=True)` over the rows' routing (`routed_rows` of the routing-only
    front's experts and weights) writes the same experts, weights and tables, bit for bit; the
    lean rows decline it."""
    spec, device = cell.spec, torch.device("cuda")
    seed = zlib.crc32(cell.why.encode()) % 2**31 + 29
    logits, _, _, tables = _operands(spec, seed, device)
    fast2, early = spec.get("fast2", False), spec.get("early", False)
    rows, hidden = spec["rows"], spec["hidden"]
    args = (spec["tile"], spec["local_offset"], spec["num_local"])
    assert op.compile_front(rows, hidden, hidden, spec["num_local"], spec["tile"], fast2, True)
    assert op.compile_front(
        rows, hidden, hidden, spec["num_local"], spec["tile"], fast2, True, True, early
    )
    ref_tables = tables.clone()
    ids, wts = op.route_front_only(logits, ref_tables, *args, hidden, fast2)
    routed = op.routed_rows(ids, wts)
    ids2, wts2 = op.route_front_only(routed, tables, *args, hidden, fast2, ids_in=True, early=early)
    torch.cuda.synchronize()
    assert torch.equal(ids2, ids) and torch.equal(wts2.view(torch.int16), wts.view(torch.int16))
    assert torch.equal(tables, ref_tables)
    assert not op.compile_front(256, hidden, hidden, 64, 32, False, True, ids_in=True)


@pytest.mark.cpu_only
def test_routed_rows_layout():
    """`ids_in`'s row: the experts in slots 0..9, the weights' bf16 bits in 10..19, 4 zeros."""
    ids = _ids(3)
    wts = torch.rand(3, op.TOP_K).to(torch.bfloat16)
    rows = op.routed_rows(ids, wts)
    assert rows.dtype is torch.int16 and tuple(rows.shape) == (3, op.IDS_ROW_SLOTS)
    assert torch.equal(rows[:, : op.TOP_K].to(torch.int32), ids)
    assert torch.equal(rows[:, op.TOP_K : 2 * op.TOP_K], wts.view(torch.int16))
    assert not bool(rows[:, 2 * op.TOP_K :].any())


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
