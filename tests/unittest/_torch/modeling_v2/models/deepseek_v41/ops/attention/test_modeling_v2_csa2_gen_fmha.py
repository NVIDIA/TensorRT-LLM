# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_gen_fmha_decode entry: stock's `gen_fmha` generation core (E4M3 staging into one bank, GB300 sparse
MLA) on the packed pools, against the plain-torch reference and against the fused decode over the same rows.

The query and the staged rows are E4M3 at a shared power-of-two scale and the core's probabilities E4M3, so the gate is
FP8-level: relative RMS below 4e-2 against the reference and against `csa2_sparse_decode` (FP8 query, bf16 staging) on
outputs of magnitude ~0.1; the MXFP8 epilogue is checked bit for bit against `inverse_rope_mxfp8` of the core's own bf16
output, and a staging group's later layer (reusing the first layer's compaction and staged main rows) against the same
layer staged from scratch.
"""

import pytest
import torch
from _v41_util import DEV
from test_modeling_v2_csa2_sparse_decode import WIN, _build

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_gen_fmha import (  # noqa: E501
    Csa2GenFmhaTile,
    csa2_gen_fmha_decode,
    csa2_gen_fmha_supported,
    csa2_gen_fmha_tile,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_decode import (  # noqa: E501
    csa2_sparse_decode,
    csa2_sparse_decode_workspace,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not csa2_gen_fmha_supported(64, 512, "cuda"),
    reason="the gen_fmha core (SM103)",
)
TOPK = 512


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()


def _case(spec, seed=0):
    gen = torch.Generator().manual_seed(spec["reqs"] * 31 + seed)
    lo, hi = spec["hist"]
    hist = torch.randint(lo, hi + 1, (spec["reqs"],), generator=gen).tolist()
    return _build([(h, spec["rows"]) for h in hist], spec["ratio"], spec["topk"], gen=True)


def _word(q8):
    return q8.float().abs().amax().reshape(1).view(torch.int32).clone()


def _args(c, tile, q8, ratio, **kw):
    return (tile, q8, _word(q8), c["sink"], c["pos"], c["rid"], c["swa_pool"], c["swa_table"], WIN, 512**-0.5,
            c["main_pool"], c["main_table"], c["idx"], ratio), kw  # fmt: skip


@pytest.mark.parametrize(
    "cell", csa2_gen_fmha_decode.CELLS, ids=[c.why[:44] for c in csa2_gen_fmha_decode.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_gen_fmha_decode)
    s = cell.spec
    c = _case(s)
    n = c["q"].shape[0]
    q8 = c["q"].to(torch.float8_e4m3fn)
    tile = csa2_gen_fmha_tile(DEV, n, n, TOPK)
    args, kw = _args(c, tile, q8.clone(), s["ratio"], group=("cell", n))
    ref = csa2_gen_fmha_decode.reference(*args, **kw)
    with validating(csa2_gen_fmha_decode):
        got = csa2_gen_fmha_decode(*args, **kw)
    assert got.dtype == torch.bfloat16 and got.shape == (n, 64, 512)
    assert int(args[2].item()) == 0, "the staging consumes and clears Q's amax word"
    fused = csa2_sparse_decode(
        csa2_sparse_decode_workspace(DEV),
        q8,
        c["sink"],
        c["pos"],
        c["rid"],
        c["swa_pool"],
        c["swa_table"],
        WIN,
        512**-0.5,
        c["main_pool"],
        c["main_table"],
        c["idx"],
        s["ratio"],
    )
    rel, rel_fused = _rel(got, ref), _rel(got, fused)
    print(f"gen_fmha {cell.why}: vs reference {rel:.4f}, vs fused decode {rel_fused:.4f}")
    assert rel < 4e-2 and rel_fused < 4e-2, (rel, rel_fused)


def test_bf16_query_of_an_eager_step():
    """An eager step's bf16 query (stock projects it in bf16 outside the decode graphs): the scale derivation reads its
    amax and the staging quantizes it at the live scale. Against the reference over the unquantized bf16 query the band
    also carries the query's E4M3 rounding (6e-2); against the E4M3-query launch over the same values, the FP8 band."""
    s = csa2_gen_fmha_decode.CELLS[0].spec
    c = _case(s, 4)
    n = c["q"].shape[0]
    tile = csa2_gen_fmha_tile(DEV, n, n, TOPK)
    args, kw = _args(c, tile, c["q"].clone(), 2, group="bf16")
    args = (args[0], args[1], None) + args[3:]
    ref = csa2_gen_fmha_decode.reference(*args, **kw)
    with validating(csa2_gen_fmha_decode):
        got = csa2_gen_fmha_decode(*args, **kw)
    fp8, kw8 = _args(c, tile, c["q"].to(torch.float8_e4m3fn), 2, group="bf16")
    via_fp8 = csa2_gen_fmha_decode(*fp8, **kw8)
    rel, rel_fp8 = _rel(got, ref), _rel(got, via_fp8)
    print(f"gen_fmha bf16 query: vs reference {rel:.4f}, vs the E4M3 query {rel_fp8:.4f}")
    assert rel < 6e-2 and rel_fp8 < 4e-2, (rel, rel_fp8)


def test_mxfp8_output_is_inverse_rope_mxfp8_of_the_bf16_output():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import inverse_rope_mxfp8
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
        csa2_rope_table,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
        csa2_stock_rope_table,
    )

    s = csa2_gen_fmha_decode.CELLS[0].spec
    c = _case(s, 1)
    n = c["q"].shape[0]
    q8 = c["q"].to(torch.float8_e4m3fn)
    tile = csa2_gen_fmha_tile(DEV, n, n, TOPK)
    cos_sin = csa2_stock_rope_table(csa2_rope_table(64, 8192, 0, 10000.0, 1.0, 32, 1, DEV))
    args, kw = _args(c, tile, q8.clone(), 2, group="mx")
    bf16 = csa2_gen_fmha_decode(*args, **kw)
    args, kw = _args(c, tile, q8.clone(), 2, group="mx", mxfp8_out=(c["pos"], cos_sin))
    with validating(csa2_gen_fmha_decode):
        data, scales = csa2_gen_fmha_decode(*args, **kw)
    want_data, want_scales = inverse_rope_mxfp8(bf16.reshape(n, -1), c["pos"], cos_sin, 8)
    assert torch.equal(data.view(torch.uint8), want_data.view(torch.uint8).view(data.shape))
    assert torch.equal(scales.view(torch.uint8), want_scales.view(torch.uint8).reshape(-1))


def test_group_layers_reuse_the_first_layers_staging():
    """A staging group's later layer (`first=False`: the first layer's compaction, recorded scale bits and, with no
    other group staged in between, its staged main rows) equals that layer staged from scratch; with another group
    staged in between, it restages the main rows and still equals it."""
    s = csa2_gen_fmha_decode.CELLS[0].spec
    c = _case(s, 2)
    n = c["q"].shape[0]
    q8 = c["q"].to(torch.float8_e4m3fn)
    q8b = (c["q"].float() * 0.5).to(
        torch.float8_e4m3fn
    )  # a second layer's query: another amax, another scale cap
    tile = csa2_gen_fmha_tile(DEV, n, n, TOPK)
    a, kw = _args(c, tile, q8.clone(), 2, group="g", first=True)
    csa2_gen_fmha_decode(*a, **kw)
    b, kw = _args(c, tile, q8b.clone(), 2, group="g", first=False)
    later = csa2_gen_fmha_decode(*b, **kw)
    b, kw = _args(c, tile, q8b.clone(), 2, group="fresh", first=True)
    scratch = csa2_gen_fmha_decode(*b, **kw)
    assert torch.equal(later, scratch)
    a, kw = _args(c, tile, q8.clone(), 2, group="g", first=True)
    csa2_gen_fmha_decode(*a, **kw)
    oc = _case(s, 4)  # another group stages other main rows (same geometry) in between
    o, kw = _args(oc, tile, oc["q"].to(torch.float8_e4m3fn), 2, group="other", first=True)
    csa2_gen_fmha_decode(*o, **kw)
    b, kw = _args(c, tile, q8b.clone(), 2, group="g", first=False)
    assert torch.equal(csa2_gen_fmha_decode(*b, **kw), scratch)


def test_cuda_graph_replay_follows_device_data():
    s = csa2_gen_fmha_decode.CELLS[0].spec
    c = _case(s, 3)
    n = c["q"].shape[0]
    tile = csa2_gen_fmha_tile(DEV, n, n, TOPK)
    q_static = c["q"].to(torch.float8_e4m3fn)
    word = torch.zeros(1, dtype=torch.int32, device=DEV)
    args = (tile, q_static, word, c["sink"], c["pos"], c["rid"], c["swa_pool"], c["swa_table"], WIN, 512**-0.5,
            c["main_pool"], c["main_table"], c["idx"], 2)  # fmt: skip
    q_live = (c["q"].float() * 0.7).to(torch.float8_e4m3fn)
    word.copy_(_word(q_static))
    csa2_gen_fmha_decode(*args, group="graph")  # eager warm-up: compiles every launch
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    q_static.copy_(c["q"].to(torch.float8_e4m3fn))
    with torch.cuda.graph(g):
        out = csa2_gen_fmha_decode(*args, group="graph")
    q_static.copy_(q_live)
    word.copy_(_word(q_live))
    g.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    eager = csa2_gen_fmha_decode(tile, q_live.clone(), _word(q_live), *args[3:], group="graph")
    assert torch.equal(replayed, eager)


def test_tiles():
    a = csa2_gen_fmha_tile(DEV, 1024, 1280, TOPK)
    assert (
        a.capacity == 1280 and csa2_gen_fmha_tile(DEV, 1280, 1280, TOPK) is a
    )  # graph steps share one tile
    big = csa2_gen_fmha_tile(DEV, 2000, 1280, TOPK)
    assert big.capacity == 2000 and big is not a
    assert isinstance(a, Csa2GenFmhaTile) and a.num_sparse_topk == 640
    assert a.gen_fmha_bank[1] == 1280 * 128


def test_rejections():
    s = dict(reqs=8, rows=4, hist=(200, 600), ratio=2, topk=TOPK)
    c = _case(s)
    n = c["q"].shape[0]
    tile = csa2_gen_fmha_tile(DEV, n, n, TOPK)
    q8 = c["q"].to(torch.float8_e4m3fn)
    args, _ = _args(c, tile, q8, 2)
    csa2_gen_fmha_decode.is_valid(*args)
    bad = list(args)
    bad[1] = c["q"]  # a bf16 query
    with pytest.raises(AssertionError):
        csa2_gen_fmha_decode.is_valid(*bad)
    small = csa2_gen_fmha_tile(DEV, n - 1, n - 1, TOPK)
    bad = list(args)
    bad[0] = small
    with pytest.raises(AssertionError, match="rows"):
        csa2_gen_fmha_decode.is_valid(*bad)
