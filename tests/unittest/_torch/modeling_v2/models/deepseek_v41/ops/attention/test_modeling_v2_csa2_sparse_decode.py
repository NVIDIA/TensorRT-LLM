# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_sparse_decode entries (fused kernel on the packed pools) against the plain-torch
`csa2_sparse_attn` reference.

`csa2_sparse_decode` (64 heads, generation and context-chunk forms) and `csa2_direct_decode` (8 / 64
heads; rows beyond the kernel's 16 run as chunked launches) accumulate in their own order and round the softmax
probabilities to bf16 in-kernel; the reference rounds them to bf16 too but sums in fp32 in another order. Gate: max abs
difference below 0.08 and mean abs difference below 4e-3 on outputs of magnitude ~1 (a bf16 ulp
of the largest values is 2^-8 ~ 4e-3; the max band is 20 of them).
"""

import pytest
import torch
from _v41_util import DEV

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    csa2_pack_rows,
    csa2_unpack_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rows_io import (  # noqa: E501
    csa2_row_addr,
    csa2_scatter_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_decode import (  # noqa: E501
    csa2_direct_decode,
    csa2_direct_decode_supported,
    csa2_sparse_decode,
    csa2_sparse_decode_supported,
    csa2_sparse_decode_workspace,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_swa_slots import (  # noqa: E501
    csa2_swa_slots,
    csa2_swa_table_descriptors,
)

# Gate on the hardware only: on an SM100-family GPU the fused kernel is part of the required engine build, so a
# failed support probe there fails `test_fused_kernel_serves_the_sm100_family` instead of skipping the module.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the fused and direct decode kernels are the SM100 family's",
)
H, D, WIN = 64, 512, 128


def _build(reqs, ratio, topk, gen, heads=H):
    """reqs: [(history, new rows)] per request. Everything both implementations need, pools included.

    `gen`: the persistent SWA pool also holds this step's own rows (the generation form reads the window
    from it); otherwise only the history before the step (the context form gets the step's rows apart).
    """
    torch.manual_seed(sum(h * 7 + n for h, n in reqs) + ratio)
    nreq = len(reqs)
    maxlen = max(h + n for h, n in reqs)
    nblk = maxlen // 128 + 2
    swa_pool = torch.zeros(nreq * nblk + 3, 128, 528, dtype=torch.uint8, device=DEV)
    swa_table = (
        torch.arange(nreq * nblk, dtype=torch.int32, device=DEV).view(nreq, nblk).flip(1) + 1
    )
    kv, pos, rid, cs = [], [], [], []
    for hist, n in reqs:
        kv.append(
            csa2_unpack_rows(csa2_pack_rows(torch.randn(hist + n, D, device=DEV), "swa"), "swa")
        )
    for r, (hist, n) in enumerate(reqs):
        p = torch.arange(hist, hist + n, device=DEV)
        pos.append(p), rid.append(torch.full_like(p, r)), cs.append(torch.full_like(p, hist))
        lo = max(0, hist - WIN)  # earlier window rows (and, for generation, the step's own rows)
        pp = torch.arange(lo, hist + (n if gen else 0), device=DEV)
        if len(pp) == 0:
            continue
        a = csa2_row_addr(swa_table, torch.full_like(pp, r), pp, 128)
        csa2_scatter_rows(
            swa_pool, a, csa2_pack_rows(kv[r][pp], "swa"), torch.ones_like(a, dtype=torch.bool)
        )
    pos, rid, cs = torch.cat(pos), torch.cat(rid), torch.cat(cs)
    step_kv = torch.cat([kv[r][h:] for r, (h, n) in enumerate(reqs)])
    main_pool = main_table = idx = None
    if ratio:
        rpp = 128 // ratio
        ncomp = maxlen // ratio + 1
        mblk = ncomp // rpp + 2
        main_pool = torch.zeros(nreq * mblk + 3, rpp, 288, dtype=torch.uint8, device=DEV)
        main_table = (
            torch.arange(nreq * mblk, dtype=torch.int32, device=DEV).view(nreq, mblk).flip(1) + 2
        )
        for r in range(nreq):
            rows = torch.arange(ncomp, device=DEV)
            a = csa2_row_addr(main_table, torch.full_like(rows, r), rows, rpp)
            csa2_scatter_rows(
                main_pool,
                a,
                csa2_pack_rows(torch.randn(ncomp, D, device=DEV), "main"),
                torch.ones_like(a, dtype=torch.bool),
            )
        vis = ((pos + 1) // ratio).cpu()
        idx = torch.full((len(pos), topk), -1, dtype=torch.int32)
        for t in range(len(pos)):
            v = int(vis[t])
            k = min(topk, v)
            if k:
                idx[t, :k] = torch.randperm(v)[:k].sort().values.int()
        idx = idx.to(DEV)
    q = torch.randn(len(pos), heads, D, device=DEV).bfloat16()
    sink = torch.randn(heads, device=DEV)
    return dict(
        q=q, sink=sink, pos=pos.int(), rid=rid, cs=cs, swa_pool=swa_pool, swa_table=swa_table,
        main_pool=main_pool, main_table=main_table, idx=idx, step_kv=step_kv,
    )  # fmt: skip


def _check(got, ref):
    err = (got.float() - ref.float()).abs()
    assert err.max() < 0.08 and err.mean() < 4e-3, (float(err.max()), float(err.mean()))


def _args(c, ratio, workspace):
    return (
        workspace, c["q"], c["sink"], c["pos"], c["rid"], c["swa_pool"], c["swa_table"], WIN, D**-0.5,
        c["main_pool"], c["main_table"], c["idx"], ratio,
    )  # fmt: skip


@pytest.mark.parametrize(
    "cell", csa2_sparse_decode.CELLS, ids=[c.why[:44] for c in csa2_sparse_decode.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_sparse_decode)
    s = cell.spec
    c = _build(s["reqs"], s["ratio"], s["topk"], gen=not s["ctx"])
    args = _args(c, s["ratio"], csa2_sparse_decode_workspace(DEV))
    # the context form gets this step's packed window rows and each request's first position
    extra = (csa2_pack_rows(c["step_kv"], "swa"), c["cs"]) if s["ctx"] else ()
    kw = {}
    if "floor" in s:  # per-request first readable position -> per row
        kw["floor"] = torch.tensor(s["floor"], device=DEV)[c["rid"]]
    got = csa2_sparse_decode(*args, *extra, **kw)
    ref = csa2_sparse_decode.reference(*args, *extra, **kw)
    assert got.dtype == torch.bfloat16 and got.shape == c["q"].shape
    _check(got, ref)
    if "floor" in s:  # the floor changes the answer: rows below it are really masked
        unmasked = csa2_sparse_decode(*args, *extra)
        assert (got.float() - unmasked.float()).abs().max() > 0.01


@pytest.mark.parametrize(
    "cell", [c for c in csa2_sparse_decode.CELLS if not c.spec["ctx"]], ids=lambda c: c.why[:44]
)
def test_fp8_query(cell):
    """Generation steps with a unit-scale E4M3 query (stock's FP8 decode query): the reference on the same (exactly
    widened) query within the bf16 band, and the bf16 launch over that widened query within the same band."""
    s = cell.spec
    c = _build(s["reqs"], s["ratio"], s["topk"], gen=True)
    q8 = c["q"].to(torch.float8_e4m3fn)
    c["q"] = q8
    ws = csa2_sparse_decode_workspace(DEV)
    args = _args(c, s["ratio"], ws)
    kw = {"floor": torch.tensor(s["floor"], device=DEV)[c["rid"]]} if "floor" in s else {}
    got = csa2_sparse_decode(*args, **kw)
    assert got.dtype == torch.bfloat16 and got.shape == q8.shape
    _check(got, csa2_sparse_decode.reference(*args, **kw))
    c["q"] = q8.bfloat16()
    _check(got, csa2_sparse_decode(*_args(c, s["ratio"], ws), **kw))


@pytest.mark.parametrize("ratio", [0, 2])
def test_mxfp8_output_feeds_the_o_projection(ratio):
    """`mxfp8_out`: the combine writes the grouped O-A input (inverse RoPE + MXFP8, 8 groups) itself. The O projection
    over it equals the O projection over the bf16 output (which runs the engine's inverse_rope_mxfp8 itself) up to the
    rounding of the MXFP8 values (computed from fp32 in the kernel, from the bf16 output otherwise)."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
        mxfp8_row_scales,
        mxfp8_swizzled_scales,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (
        csa2_o_a_mxfp8_scales,
        csa2_o_projection,
        requantize_block128,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
        csa2_rope_table,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
        csa2_stock_rope_table,
    )

    c = _build([(300, 8), (1000, 8), (50, 8)], ratio, 64 if ratio else 0, gen=True)
    n = c["q"].shape[0]
    args = _args(c, ratio, csa2_sparse_decode_workspace(DEV))
    cos_sin = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV))
    with validating(csa2_sparse_decode):
        data, scales = csa2_sparse_decode(*args, mxfp8_out=(c["pos"], cos_sin))
    bf16 = csa2_sparse_decode(*args)
    assert data.dtype == torch.float8_e4m3fn and data.shape == (8, n, 4096)
    gen = torch.Generator(device=DEV).manual_seed(ratio)
    q, sc = requantize_block128(
        (torch.randn(8 * 1024, 4096, generator=gen, device=DEV) * 0.05).bfloat16()
    )
    wo_a, wo_a_scale = q.view(8, 1024, 4096), sc.view(8, 8, 32)
    wo_b = (torch.randn(5120, 8192, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    wo_b_s = mxfp8_row_scales(
        torch.randint(121, 127, (160, 256), generator=gen, device=DEV, dtype=torch.uint8),
        (5120, 8192),
    )
    o_args = (c["pos"], cos_sin, wo_a, wo_a_scale, csa2_o_a_mxfp8_scales(wo_a_scale), wo_b,
              mxfp8_swizzled_scales(wo_b_s), torch.ones(1, device=DEV))  # fmt: skip
    via_combine = csa2_o_projection(None, *o_args, att_mxfp8=(data, scales))
    via_bf16 = csa2_o_projection(bf16.reshape(n, -1), *o_args)
    rel = (
        (via_combine.float() - via_bf16.float()).square().mean().sqrt()
        / via_bf16.float().square().mean().sqrt()
    ).item()
    assert rel < 2e-2, rel


def test_cuda_graph_replay_follows_device_data():
    reqs, ratio, topk = [(300, 4), (1000, 4), (50, 4)], 2, 64
    a = _build(reqs, ratio, topk, gen=True)
    b = _build([(h + 11, n) for h, n in reqs], ratio, topk, gen=True)
    ws = csa2_sparse_decode_workspace(DEV)
    static = {k: v.clone() for k, v in a.items() if torch.is_tensor(v)}

    def run(c):
        return csa2_sparse_decode(*_args(c, ratio, ws))

    run(static)  # eager warm-up
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run(static)
    same_shape = {k: v for k, v in b.items() if torch.is_tensor(v) and v.shape == static[k].shape}
    for k, v in same_shape.items():  # new step: same shapes, new contents
        static[k].copy_(v)
    g.replay()
    torch.cuda.synchronize()
    want = run({**b, **{k: static[k] for k in same_shape}})
    assert torch.equal(out, want)


@pytest.mark.parametrize(
    "cell", csa2_direct_decode.CELLS, ids=[c.why[:44] for c in csa2_direct_decode.CELLS]
)
def test_direct_decode_certified_cells(cell, certifying):  # noqa: F811
    s = cell.spec
    if not csa2_direct_decode_supported(s["heads"], sum(n for _, n in s["reqs"]), DEV):
        pytest.skip("direct decode unsupported here")
    certifying(csa2_direct_decode)
    c = _build(s["reqs"], s["ratio"], s["topk"], gen=True, heads=s["heads"])
    args = _args(c, s["ratio"], None)[1:]
    got = csa2_direct_decode(*args)
    ref = csa2_direct_decode.reference(*args)
    assert got.dtype == torch.bfloat16 and got.shape == c["q"].shape
    _check(got, ref)


def _step_slots(c, ratio):
    """`csa2_swa_slots` of the fixture's one layer: (reads [n, 128], visible [n])."""
    tables = c["swa_table"].unsqueeze(0).contiguous()
    floors = torch.zeros(tables.shape[1], dtype=torch.int64, device=DEV)
    reads, _, visible = csa2_swa_slots(
        c["pos"], c["rid"], tables, csa2_swa_table_descriptors(tables),
        torch.tensor([ratio], dtype=torch.int32, device=DEV), floors, WIN, 128,
    )  # fmt: skip
    return reads[0], visible[0]


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_precomputed_window_slots_match_the_derived_ones(ratio):
    """The step's slots from `csa2_swa_slots` give the same attention as the ones the op derives itself
    (tp8 direct decode with 8 heads and the 64-head fused kernel)."""
    reqs = [(300, 4), (1000, 4), (50, 4)]
    for heads in (8, 64):
        if heads == 8 and not csa2_direct_decode_supported(8, 12, DEV):
            continue
        c = _build(reqs, ratio, 64 if ratio else 0, gen=True, heads=heads)
        reads, visible = _step_slots(c, ratio)
        if heads == 64:
            args = _args(c, ratio, csa2_sparse_decode_workspace(DEV))
            op = csa2_sparse_decode
            tail = (None, None)
        else:
            args = _args(c, ratio, None)[1:]
            op = csa2_direct_decode
            tail = ()
        with validating(op):
            got = op(*args, *tail, reads, visible)
        assert torch.equal(got, op(*args, *tail))


@pytest.mark.parametrize("ratio", [0, 2])
def test_direct_decode_rotates_the_query_in_the_launch(ratio):
    """`q_rope`: the launch rotates the projected query itself, as
    `csa2_rope_inplace` followed by the plain launch does."""
    if not csa2_direct_decode_supported(8, 18, DEV):
        pytest.skip("direct decode unsupported here")
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
        csa2_rope_table,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
        csa2_rope_inplace,
        csa2_stock_rope_table,
    )

    reqs = [(300, 6), (1000, 6), (50, 6)]  # 18 rows: a full launch of 16 and a two-row tail
    c = _build(reqs, ratio, 64 if ratio else 0, gen=True, heads=8)
    table = csa2_stock_rope_table(csa2_rope_table(64, 2048, 0, 10000.0, 1.0, 32, 1, DEV))
    args = _args(c, ratio, None)[1:]
    rotated = csa2_rope_inplace(c["q"].clone(), c["pos"], table)
    want = csa2_direct_decode(rotated, *args[1:])
    with validating(csa2_direct_decode):
        got = csa2_direct_decode(c["q"], *args[1:], None, None, (c["pos"], table))
    _check(got, want)
    _check(got, csa2_direct_decode.reference(c["q"], *args[1:], None, None, (c["pos"], table)))


@pytest.mark.parametrize("ratio", [1, 2])
def test_direct_decode_exported_slots_feed_a_later_layer(ratio):
    """`slots_out` / `main_slots` (the engine's index-source / Reuse layers): the launch that maps the selections stores
    the pool rows they resolve to, and a later launch over the same selections reading those rows unmapped attends
    exactly as the mapped launch does (18 rows: a full launch of 16 and a two-row tail)."""
    if not csa2_direct_decode_supported(8, 18, DEV):
        pytest.skip("direct decode unsupported here")
    reqs = [(1500, 6), (300, 6), (40, 6)]
    c = _build(reqs, ratio, 96, gen=True, heads=8)
    args = _args(c, ratio, None)[1:]
    reads, visible = _step_slots(c, ratio)
    want = csa2_direct_decode(*args, reads, visible)
    slots = torch.full(
        (c["q"].shape[0], 128), -2, dtype=torch.int64, device=DEV
    )  # 96 selections padded to 128
    with validating(csa2_direct_decode):
        exported = csa2_direct_decode(*args, reads, visible, slots_out=slots)
    assert torch.equal(exported, want)
    assert int(slots.min()) >= -1 and bool((slots >= 0).any()), (
        "every selection resolves to a row or -1"
    )
    later = torch.randn_like(c["q"])  # another layer's queries over the same selections
    with validating(csa2_direct_decode):
        got = csa2_direct_decode(later, *args[1:], reads, visible, main_slots=slots)
    assert torch.equal(got, csa2_direct_decode(later, *args[1:], reads, visible))


def test_fused_kernel_serves_the_sm100_family():
    """The module runs on every SM100-family GPU; there the fused op must load and report support (a broken or
    missing build fails here rather than skipping the module's tests)."""
    assert torch.ops.trtllm.csa2_sparse_decode_supported()
    assert csa2_sparse_decode_supported(64, "cuda")
    assert not csa2_sparse_decode_supported(8, "cuda"), (
        "8 heads are the direct decode's, not the fused kernel's"
    )


def test_rejections():
    cell = csa2_sparse_decode.CELLS[1]
    s = cell.spec
    c = _build(s["reqs"], s["ratio"], s["topk"], gen=True)
    ws = csa2_sparse_decode_workspace(DEV)
    args = list(_args(c, s["ratio"], ws))
    csa2_sparse_decode.is_valid(*args)
    csa2_sparse_decode.is_valid(
        *args, c["step_kv"].new_zeros(len(c["pos"]), 528, dtype=torch.uint8), c["cs"]
    )
    with pytest.raises(
        AssertionError, match="go together"
    ):  # step rows without each request's first position
        csa2_sparse_decode.is_valid(
            *args, c["step_kv"].new_zeros(len(c["pos"]), 528, dtype=torch.uint8)
        )
    bad = list(args)
    bad[1] = c["q"][:, :8].contiguous()  # 8 heads: only the direct decode serves them
    with pytest.raises(AssertionError):
        csa2_sparse_decode.is_valid(*bad)
    bad = list(args)
    bad[5] = torch.zeros(4, 64, 528, dtype=torch.uint8, device=DEV)  # 64-row SWA pages
    with pytest.raises(AssertionError, match="128"):
        csa2_sparse_decode.is_valid(*bad)
    bad = list(args)
    bad[11] = None  # a main pool with no selected rows is ignored
    with pytest.raises(AssertionError, match="ignored"):
        csa2_sparse_decode.is_valid(*bad)
