# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_indexer_decode entries: `csa2_indexer_topk_decode` (device-length radix top-k over
paged FP4 index keys, with the candidate-source / consumer pair) and `csa2_indexer_topk_prefill` (its
row-tiled context form) against the dense per-request oracle.

The kernels score with DeepGEMM's FP4 paged MQA logits in their own accumulation order, so the gate is
the one the paged indexer entry uses: the unreachable (-1) picks agree exactly and more than 97% of the
reference's picks are in the op's set. The candidate source is gated on its position set against the
oracle's block mask (same 97%), the consumer on the score band of `check_consumer_picks`.
"""

import pytest
import torch
from _v41_util import DEV, IS_SM103, check_consumer_picks, overlap, request_index_pools

__extra_import_path__ = ["../../../.."]
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (  # noqa: E501
    csa2_indexer_topk_decode,
    csa2_indexer_topk_prefill,
    dequantize_queries,
    pack_queries,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_descriptors import (  # noqa: E501
    csa2_indexer_descriptors,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    csa2_pack_rows,
    csa2_unpack_rows,
)

pytestmark = pytest.mark.skipif(
    not IS_SM103, reason="the paged FP4 MQA-logits kernels are certified on sm_103 only"
)
HEADS, TOPK = 32, 64


def _qdq(q):
    """`q` on the index fp4 grid, as the model hands it to the indexer."""
    return csa2_unpack_rows(csa2_pack_rows(q.reshape(-1, 128), "index"), "index", q.dtype).view_as(
        q
    )


def _inputs(ratio, positions_per_req, lens, seed=0):
    pool, table = request_index_pools(ratio, lens, seed)
    pos = torch.tensor([p for ps in positions_per_req for p in ps], device=DEV, dtype=torch.int32)
    rid = torch.tensor([r for r, ps in enumerate(positions_per_req) for _ in ps], device=DEV)
    q = _qdq(torch.randn(len(pos), HEADS, 128, device=DEV).bfloat16())
    w = torch.rand(len(pos), HEADS, device=DEV) * 0.1
    return pool, table, pos, rid, q, w


def _decode_cell(spec):
    """(inputs, max_positions) of a decode-shaped cell: `nreq` requests of `rows` trailing positions."""
    ratio, nreq, rows = spec["ratio"], spec["nreq"], spec["rows"]
    if nreq == 3:
        lens = [(100 + 37 * (i % 11)) * ratio for i in range(nreq)]
    elif spec["cand"]:
        lens = [2500 + 101 * i for i in range(nreq)]
    else:
        lens = [(3000 + 17 * i) * ratio for i in range(nreq)]
    pp = [[ln - rows + j for j in range(rows)] for ln in lens]
    return _inputs(ratio, pp, lens, seed=5 if spec["cand"] else 0), max(lens) // ratio + 8


@pytest.mark.parametrize(
    "cell", csa2_indexer_topk_decode.CELLS, ids=[c.why[:44] for c in csa2_indexer_topk_decode.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_indexer_topk_decode)
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    want = (8, 40) if s["cand"] else None
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    got, cand = csa2_indexer_topk_decode(*args, None, want)
    ref, ref_masks = csa2_indexer_topk_decode.reference(*args, None, want)
    k = min(got.shape[1], ref.shape[1])
    assert got.shape[0] == ref.shape[0]
    assert torch.equal(got[:, :k] < 0, ref[:, :k] < 0), "unreachable picks must agree exactly"
    assert overlap(got[:, :k], ref[:, :k]) > 0.97
    if not s["cand"]:
        return
    # source: candidate positions == the oracle's kept-position mask (up to block-max ties)
    positions, rows = cand[0], s["rows"]
    agree = total = 0
    for r, mask in enumerate(ref_masks):  # [rows, span] per request
        for j in range(rows):
            a = set(positions[rows * r + j][positions[rows * r + j] >= 0].tolist())
            b = set(torch.nonzero(mask[j]).flatten().tolist())
            agree += len(a & b)
            total += len(b)
    assert agree / total > 0.97
    # consumer: only the source's candidates are scored; the oracle gets the op's own candidate set
    q2 = _qdq(torch.randn_like(q.float()).bfloat16())
    cargs = (q2, w, pos, rid, pool, table, ratio, TOPK, mp)
    picks, _ = csa2_indexer_topk_decode(*cargs, cand, None)
    cref, _ = csa2_indexer_topk_decode.reference(*cargs, cand, None)
    span = mp
    c_len = (pos.long() + 1) // ratio
    with true_fp32_matmul():
        check_consumer_picks(picks, q2, w, rid, pool, table, ratio, TOPK, positions, c_len, span)
        check_consumer_picks(cref, q2, w, rid, pool, table, ratio, TOPK, positions, c_len, span)


@pytest.mark.parametrize(
    "cell", csa2_indexer_topk_decode.CELLS, ids=[c.why[:44] for c in csa2_indexer_topk_decode.CELLS]
)
def test_step_descriptors_give_the_same_picks(cell):
    """With the owner's per-step descriptors (`csa2_indexer_descriptors`, built once for all its layers) the op gives
    the picks, candidate set and consumer picks it gives when it derives the tables itself."""
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    want = (8, 40) if s["cand"] else None
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    with validating(csa2_indexer_topk_decode):
        got, cand = csa2_indexer_topk_decode(*args, None, want, desc)
    plain, pcand = csa2_indexer_topk_decode(*args, None, want)
    assert torch.equal(got, plain)
    if not s["cand"]:
        return
    assert all(torch.equal(a, b) for a, b in zip(cand, pcand))
    q2 = _qdq(torch.randn_like(q.float()).bfloat16())
    cargs = (q2, w, pos, rid, pool, table, ratio, TOPK, mp)
    picks, _ = csa2_indexer_topk_decode(*cargs, cand, None, desc)
    c_len = (pos.long() + 1) // ratio
    with true_fp32_matmul():
        check_consumer_picks(picks, q2, w, rid, pool, table, ratio, TOPK, cand[0], c_len, mp)


@pytest.mark.parametrize("cand", [False, True], ids=["owner", "candidate-source"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_split_rows_give_deepgemm_bits_and_picks(ratio, cand):
    """`split_rows` on a dep32 decode step (25 requests of six rows): the engine's split-row CuTe DSL logits (three
    64-head rows per request, one pass over its keys) carry DeepGEMM's bits on every row's visible columns, so the
    picks and a candidate source's candidates are the DeepGEMM path's, bit for bit."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (  # noqa: E501
        _logits,
        _split_row_heads,
    )

    nreq, rows = 25, 6
    assert _split_row_heads(rows, HEADS, nreq) == HEADS, "the engine runs split rows at this shape"
    lens = [(3000 + 37 * i) * ratio for i in range(nreq)]
    (pool, table, pos, rid, q, w) = _inputs(
        ratio, [[ln - rows + j for j in range(rows)] for ln in lens], lens, seed=3
    )
    mp = max(lens) // ratio + 8
    c_len = ((pos.long() + 1) // ratio).contiguous()
    desc = csa2_indexer_descriptors(table, rid.long().contiguous(), c_len, ratio)
    width = min(table.shape[1] * (128 // ratio), -(-mp // 8) * 8)
    ctx = (q, w, desc[1], pool, table, rid, ratio, width, desc[0], None, rows)
    deep, split = _logits(*ctx), _logits(*ctx, split_rows=True)
    assert deep.shape == split.shape
    visible = torch.arange(width, device=DEV) < c_len.unsqueeze(-1)
    assert torch.equal(deep[visible].view(torch.int32), split[visible].view(torch.int32))
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    want = (8, 40) if cand else None
    ref, rcand = csa2_indexer_topk_decode(*args, None, want, desc, rows_per_request=rows)
    with validating(csa2_indexer_topk_decode):
        got, gcand = csa2_indexer_topk_decode(
            *args, None, want, desc, rows_per_request=rows, split_rows=True
        )
    assert torch.equal(got, ref)
    if cand:
        assert all(torch.equal(a, b) for a, b in zip(gcand, rcand))


@pytest.mark.parametrize(
    "cell",
    [c for c in csa2_indexer_topk_decode.CELLS if not c.spec["cand"]],
    ids=lambda c: c.why[:44],
)
def test_heuristic_topk_gives_the_radix_picks(cell):
    """`heuristic=True` (the engine's self-sampling GVR top-k, K = 512 as in the model) selects what the exact radix
    top-k selects: unreachable picks agree exactly, the sets agree up to ties at the k-th score."""
    s = cell.spec
    (pool, table, pos, rid, q, w), mp = _decode_cell(s)
    ratio = s["ratio"]
    desc = csa2_indexer_descriptors(
        table, rid.long().contiguous(), ((pos.long() + 1) // ratio).contiguous(), ratio
    )
    args = (q, w, pos, rid, pool, table, ratio, 512, mp, None, None, desc)
    with validating(csa2_indexer_topk_decode):
        got, _ = csa2_indexer_topk_decode(*args, True)
    want, _ = csa2_indexer_topk_decode(*args, False)
    assert torch.equal(got < 0, want < 0)
    assert overlap(got, want) > 0.99


@pytest.mark.parametrize(
    "cell",
    csa2_indexer_topk_prefill.CELLS,
    ids=[c.why[:44] for c in csa2_indexer_topk_prefill.CELLS],
)
def test_prefill_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_indexer_topk_prefill)
    s = cell.spec
    ratio, hist, n, tile = s["ratio"], s["hist"], s["n"], s["tile"]
    pool, table = request_index_pools(ratio, [hist + n], seed=9)
    pos = torch.arange(hist, hist + n, device=DEV, dtype=torch.int32)
    rid = torch.zeros(n, dtype=torch.long, device=DEV)
    q = _qdq(torch.randn(n, HEADS, 128, device=DEV).bfloat16())
    w = torch.rand(n, HEADS, device=DEV) * 0.1
    span = -(-(hist + n) // ratio)
    mp = -(-span // 8) * 8
    want = (8, 50)
    desc = None
    if s.get(
        "desc"
    ):  # as the dep2 target calls it: the table cut to the columns the span reaches, the step's descriptors
        table = table[:, : -(-(hist + n) // 128)]
        desc = csa2_indexer_descriptors(table, rid, ((pos.long() + 1) // ratio).contiguous(), ratio)
    args = (q, w, pos, rid, pool, table, ratio, TOPK, mp)
    with validating(csa2_indexer_topk_prefill):
        got, cand = csa2_indexer_topk_prefill(*args, None, want, tile, desc)
    ref, _ = csa2_indexer_topk_prefill.reference(*args, None, want, tile, desc)
    k = min(got.shape[1], ref.shape[1])
    assert overlap(got[:, :k], ref[:, :k]) > 0.97
    # a consumer layer over the source's candidates, tile by tile
    q2 = _qdq(torch.randn_like(q.float()).bfloat16())
    picks, _ = csa2_indexer_topk_prefill(
        q2, w, pos, rid, pool, table, ratio, TOPK, mp, cand, None, tile, desc
    )
    c_len = (pos.long() + 1) // ratio
    with true_fp32_matmul():
        check_consumer_picks(picks, q2, w, rid, pool, table, ratio, TOPK, cand[0], c_len, mp)


def test_quantized_queries_pass_through():
    """The (data, scales) pair of `csa2_store_rows` and the bf16 queries it came from give the same picks."""
    ratio = 1
    lens = [800, 1200]
    pp = [[ln - 2, ln - 1] for ln in lens]
    pool, table, pos, rid, q, w = _inputs(ratio, pp, lens, seed=2)
    data, scale = pack_queries(q)
    pair = (data, scale.contiguous().view(torch.uint8).view(*scale.shape, 4))
    assert torch.equal(dequantize_queries(pair), _qdq(q))
    a, _ = csa2_indexer_topk_decode(q, w, pos, rid, pool, table, ratio, TOPK, 1216)
    b, _ = csa2_indexer_topk_decode(pair, w, pos, rid, pool, table, ratio, TOPK, 1216)
    assert torch.equal(a, b)


@pytest.mark.parametrize("cand", [False, True])
def test_cuda_graph_replay_follows_device_data(cand):
    ratio, nreq = 1, 16
    lens = [3000 + 53 * i for i in range(nreq)]
    pp = [[ln - 1] for ln in lens]
    pool, table, pos, rid, q, w = _inputs(ratio, pp, lens, seed=3)
    mp = max(lens) + 64
    want = (8, 40) if cand else None

    def run(pos_):
        return csa2_indexer_topk_decode(q, w, pos_, rid, pool, table, ratio, TOPK, mp, None, want)

    run(pos)  # eager warm-up (kernel JIT)
    torch.cuda.synchronize()
    static_pos = pos.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, cands = run(static_pos)
    static_pos.copy_(pos - 7)  # lengths are device data: the replay must follow them
    g.replay()
    torch.cuda.synchronize()
    ref, rcands = run(pos - 7)
    assert torch.equal(out, ref)
    if cand:
        assert all(torch.equal(a, b) for a, b in zip(cands, rcands))


def test_rejections():
    ratio, lens = 1, [300]
    pool, table, pos, rid, q, w = _inputs(ratio, [[299]], lens)
    csa2_indexer_topk_decode.is_valid(q, w, pos, rid, pool, table, ratio, TOPK)
    with pytest.raises(AssertionError, match="ratio"):
        csa2_indexer_topk_decode.is_valid(q, w, pos, rid, pool, table, 3, TOPK)
    with pytest.raises(AssertionError, match="fp32"):
        csa2_indexer_topk_decode.is_valid(q, w.bfloat16(), pos, rid, pool, table, ratio, TOPK)
    with pytest.raises(AssertionError, match="32 or 64"):
        csa2_indexer_topk_decode.is_valid(
            q[:, :16].contiguous(), w[:, :16].contiguous(), pos, rid, pool, table, ratio, TOPK
        )
    with pytest.raises(AssertionError, match="8 keys"):
        csa2_indexer_topk_decode.is_valid(
            q, w, pos, rid, pool, table, ratio, TOPK, None, None, (16, 8)
        )
    with pytest.raises(AssertionError, match="multiple of 8"):
        csa2_indexer_topk_prefill.is_valid(q, w, pos, rid, pool, table, ratio, TOPK, 100)
    # a prefill candidate consumer takes bf16 head weights (forwarded unchanged to the consumer), as the decode form
    cand = tuple(torch.zeros(w.shape[0], 8, dtype=torch.int32, device=DEV) for _ in range(3))
    csa2_indexer_topk_prefill.is_valid(
        q, w.bfloat16(), pos, rid, pool, table, ratio, TOPK, 304, cand
    )
    with pytest.raises(AssertionError, match="fp32"):
        csa2_indexer_topk_prefill.is_valid(q, w.bfloat16(), pos, rid, pool, table, ratio, TOPK, 304)


@pytest.mark.skipif(not IS_SM103, reason="the engine's CuTe-DSL TopK kernels")
@pytest.mark.parametrize("heuristic", [True, False], ids=["heuristic", "exact"])
@pytest.mark.parametrize("width", [4504, 9000])
def test_context_rows_select_as_the_engine(width, heuristic):
    """Context rows (`csa2_indexer_topk_prefill` -> `prefill_select`) select with the engine's context TopK
    (`select_prepared_scores(is_prefill=True)`: its exact radix prefill kernel below 8192 columns, and from there
    its self-sampling GVR prefill engine under `enable_heuristic_topk`, the exact radix kernel without it). Those
    kernels (as every top-k kernel here) pick among scores tied at the k-th in no fixed order from run to run, so
    the check is the engine's: the same top-k scores, and the same picks on every row without a tie at the k-th."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention import (
        csa2_indexer_decode as mod,
    )
    from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

    gen = torch.Generator(device=DEV).manual_seed(width)
    rows, k = 96, 512
    # coarse scores: a few hundred distinct values per row, so most k-th scores are tied
    logits = (
        torch.randint(-300, 300, (rows, width), generator=gen, device=DEV).float() / 64
    ).contiguous()
    logits[: rows // 2] += (
        torch.rand(rows // 2, width, generator=gen, device=DEV) * 1e-3
    )  # untied rows
    c_len = torch.randint(k + 1, width + 1, (rows,), generator=gen, device=DEV, dtype=torch.int32)
    c_len[0] = width
    got = mod._engine_prefill_topk(logits, c_len, k, heuristic)
    impl = (
        TopKImplementation.CUTE_DSL_RADIX
        if width < mod._SELF_SAMPLING_PREFILL_MIN_COLUMNS or not heuristic
        else TopKImplementation.CUTE_DSL_GVR
    )
    want = torch.full((rows, k), -1, dtype=torch.int32, device=DEV)
    TopK(k, prefill_implementation=impl)(
        logits, want, is_prefill=True, row_starts=torch.zeros_like(c_len), row_ends=c_len
    )
    tied = 0
    for r in range(rows):
        g, wv = got[r][got[r] >= 0].long(), want[r][want[r] >= 0].long()
        assert torch.equal(logits[r, g].sort().values, logits[r, wv].sort().values), r
        kth = logits[r, : c_len[r]].topk(k).values[-1]
        if int((logits[r, : c_len[r]] == kth).sum()) == 1:
            assert set(g.tolist()) == set(wv.tolist()), r
        else:
            tied += 1
    assert 0 < tied < rows


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("gvr", [False, True])
def test_deepgemm_masked_table_replaces_the_mask_launch(ratio, gvr):
    """Attention TP (the built-in `_deepgemm_page_mask` and TP selection chain): with the descriptors' DeepGEMM-masked
    table the paged logits score the unallocated pages -inf themselves; the picks are those of the mask launch
    (`mask_unmapped`), bit for bit, and none lands on an unallocated page. One request of six draft rows (tep8
    decode), pages unmapped inside its visible range."""
    lens = [2600 * ratio]
    pp = [[lens[0] - 6 + j for j in range(6)]]
    pool, table, pos, rid, q, w = _inputs(ratio, pp, lens, seed=3)
    table = table.clone()
    table[0, 2:9:3] = -1  # unallocated pages below the visible length
    mp = lens[0] // ratio + 8
    visible = ((pos.long() + 1) // ratio).contiguous()
    full = csa2_indexer_descriptors(table, rid.long().contiguous(), visible, ratio, masked=True)
    desc, masked = full[:4], full[4]
    args = (q, w, pos, rid, pool, table, ratio, 512, mp, None, None, desc, gvr)
    kw = dict(mask_unmapped=True, tp_chain=gvr, rows_per_request=6)
    with validating(csa2_indexer_topk_decode):
        got, _ = csa2_indexer_topk_decode(*args, masked_blocks=masked, **kw)
    ref, _ = csa2_indexer_topk_decode(*args, **kw)
    assert torch.equal(got, ref)
    rpp = 128 // ratio
    unmapped = {int(p) for p in torch.nonzero(table[0] < 0).flatten().tolist()}
    picked_pages = {int(c) // rpp for c in got[got >= 0].tolist()}
    assert not (picked_pages & unmapped), "a pick on an unallocated page"
    assert (got >= 0).sum() > 0
