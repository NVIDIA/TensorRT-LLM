# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for mv2_minimax_m3::minimax_m3_sm107_tp2_sparse_decode_lc (LC2: one-launch long-context sparse decode, TP2)
as the MiniMax-M3.1 sm_107 tp2 target takes it (TP2WIDE_LC2), ported from the M3 sm_107 tp2 target's test.

Contracts:
- the BF16 rows sit in the same band around a float64 reference as the stock Triton sparse decode, and every variant
  (linear softmax, pre-wait block-table staging, MXFP8 hand-off on or off) writes the same rows as the stock
  log-domain recurrence where the arithmetic is shared;
- the MXFP8 hand-off (q8, sf) is bitwise torch.ops.trtllm.mxfp8_quantize(out, True) of the rows the kernel wrote: the
  E4M3 bytes row-major and the UE8M0 scales in its 128x4-swizzled layout. The o_proj consumes this as its
  prequantized activation, so this test is the interface contract;
- the target dispatch takes LC2 for selected tables at lc_min_rows(qlen)..LC_MAX_ROWS rows (9 for K7, 33 otherwise).
  Every narrower step (and every
  step under TRTLLM_M31_DISABLE_TP2WIDE_LC2=1, emulated by LC_KEYS = frozenset()) is bitwise the Triton producer +
  merge, and the qlen-8 (K7) wide steps and the K3 steps from 12 rows sit in the band.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
        m3_sparse_decode_lc as op,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
        mxfp8_handoff_reference,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402, F401 — needs the path declared above
    assert_within_ulp,
    certifying,
)

PAGE, DIM, TOPK, GROUP, KV_HEADS = 128, 128, 16, 16, 2
WIDTH = KV_HEADS * GROUP * DIM
SM_SCALE = DIM**-0.5
# Band, in BF16 ulps (2^-8) of each output row's largest magnitude, and of the relative RMS. Same as the sm_103 TP4
# kernel; on SM107 the stock Triton decode and this kernel both measure at most 1.03 ulps and 0.26 RMS.
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

_E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _op():
    return family_ops().minimax_m3_sm107_tp2_sparse_decode_lc


def _e4m3_bytes(shape, gen, lo=0x28, hi=0x48):
    mag = torch.randint(lo, hi, shape, dtype=torch.int32, device="cuda", generator=gen)
    sign = torch.randint(0, 2, shape, dtype=torch.int32, device="cuda", generator=gen) << 7
    return (mag | sign).to(torch.uint8)


def _case(seq_lens, seed, dql=4, overlap=14, q_hi=0x48, tie=False):
    """A packed-NVFP4 paged cache, ascending top-16 tables with a -1 tail, and FP8 queries for TP2 geometry."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    reqs, total_q = len(seq_lens), len(seq_lens) * dql
    nblk = [(s + PAGE - 1) // PAGE for s in seq_lens]
    pages = sum(nblk) + 8
    k = torch.randint(
        0, 256, (pages, KV_HEADS, PAGE, DIM // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    v = torch.randint(
        0, 256, (pages, KV_HEADS, PAGE, DIM // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    if tie:
        k[:] = k[:1, :, :1, :]
    ksc = _e4m3_bytes((pages, KV_HEADS, PAGE, DIM // 16), gen)
    vsc = _e4m3_bytes((pages, KV_HEADS, PAGE, DIM // 16), gen)
    perm = torch.randperm(pages, device="cuda", generator=gen).int()
    bt = torch.full((reqs, max(nblk)), -1, dtype=torch.int32, device="cuda")
    start = 0
    for r, n in enumerate(nblk):
        bt[r, :n] = perm[start : start + n]
        start += n
    topk = torch.full((KV_HEADS, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    for r, s in enumerate(seq_lens):
        for h in range(KV_HEADS):
            last = (s + PAGE - 1) // PAGE
            base = torch.randperm(max(last - 1, 1), device="cuda", generator=gen)[
                : TOPK - 1
            ].tolist()
            for qo in range(dql):
                nb = (s - dql + qo + 1 + PAGE - 1) // PAGE
                sel = [b for b in base[:overlap] if b < nb - 1]
                for b in torch.randperm(max(nb - 1, 1), device="cuda", generator=gen).tolist():
                    if len(sel) >= min(TOPK, nb) - 1:
                        break
                    if b not in sel and b < nb - 1:
                        sel.append(b)
                sel = sorted(set(sel + [nb - 1]))[: min(TOPK, nb)]
                topk[h, r * dql + qo, : len(sel)] = torch.tensor(
                    sel, dtype=torch.int32, device="cuda"
                )
    q = _e4m3_bytes((total_q, KV_HEADS * GROUP, DIM), gen, lo=0x20, hi=q_hi).view(
        torch.float8_e4m3fn
    )
    return dict(
        q=q,
        k=k,
        v=v,
        ksc=ksc,
        vsc=vsc,
        kgs=torch.tensor([0.03], device="cuda"),
        vgs=torch.tensor([0.05], device="cuda"),
        topk=topk,
        bt=bt,
        seq=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        dql=dql,
    )


def _dequant(packed, sc, quad):
    lut = torch.tensor(_E2M1, dtype=torch.float64, device="cuda")
    vals = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1).reshape(
        *packed.shape[:3], DIM
    )
    if (
        quad
    ):  # V scales: token t, column c at byte (t // 4) * 32 + t % 4 + 4 * c within the page/head
        t = torch.arange(PAGE, device="cuda")[:, None]
        idx = (t // 4) * 32 + t % 4 + 4 * torch.arange(8, device="cuda")[None, :]
        sc = sc.reshape(*sc.shape[:2], PAGE * 8)[:, :, idx]
    return vals * sc.contiguous().view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(
        16, dim=-1
    )


def _reference(c):
    q = c["q"].to(torch.float64)
    out = torch.zeros(q.shape, dtype=torch.float64, device="cuda")
    dql = c["dql"]
    for tok in range(q.shape[0]):
        r, qo = divmod(tok, dql)
        kv_len = max(int(c["seq"][r]) - dql + qo + 1, 0)
        for h in range(KV_HEADS):
            blks = [b for b in c["topk"][h, tok].tolist() if b >= 0][
                : min(TOPK, (kv_len + PAGE - 1) // PAGE)
            ]
            if not blks:
                continue
            pg = c["bt"][r, blks].long()
            kk = _dequant(c["k"][pg, h : h + 1], c["ksc"][pg, h : h + 1], False)[:, 0].reshape(
                -1, DIM
            )
            vv = _dequant(c["v"][pg, h : h + 1], c["vsc"][pg, h : h + 1], True)[:, 0].reshape(
                -1, DIM
            )
            pos = (
                torch.tensor(blks, device="cuda")[:, None] * PAGE
                + torch.arange(PAGE, device="cuda")
            ).reshape(-1)
            s = (q[tok, h * GROUP : (h + 1) * GROUP] @ kk.T) * (SM_SCALE * c["kgs"].item())
            s[:, pos >= kv_len] = float("-inf")
            out[tok, h * GROUP : (h + 1) * GROUP] = (torch.softmax(s, dim=-1) @ vv) * c[
                "vgs"
            ].item()
    return out.reshape(q.shape[0], WIDTH)


def _band(x, ref):
    x = x.to(torch.float64).reshape(-1, DIM)
    ref = ref.reshape(-1, DIM)
    rowmax = ref.abs().amax(dim=-1, keepdim=True)
    live = rowmax[:, 0] > 0
    # A row whose reference is all zero (a padding request, a query row with no key yet) must be written as zero:
    # the band below measures the live rows only.
    assert bool((x[~live] == 0).all()), "a row with an all-zero reference is not zero"
    unit = (rowmax * 2.0**-8).clamp_min(1e-300)
    err = ((x - ref).abs() / unit)[live]
    rms = (((x - ref) ** 2).mean(dim=-1).sqrt() / unit[:, 0])[live]
    return (float(err.max()) if err.numel() else 0.0), (float(rms.max()) if rms.numel() else 0.0)


def _run(c, linear=False, prewait=False, mxfp8=False, n1=None, use_pdl=False, out=None):
    total_q = c["q"].shape[0]
    if out is None:
        out = torch.empty(total_q, WIDTH, dtype=torch.bfloat16, device="cuda")
    q8 = torch.zeros(total_q, WIDTH, dtype=torch.uint8, device="cuda") if mxfp8 else None
    sf_bytes = 128 * (WIDTH // 32) * ((total_q + 127) // 128)
    sf = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda") if mxfp8 else None
    _op()(
        c["q"].view(torch.uint8),
        c["k"],
        c["v"],
        c["ksc"],
        c["vsc"],
        c["kgs"],
        c["vgs"],
        c["topk"],
        c["bt"],
        c["seq"],
        out,
        q8,
        sf,
        WIDTH // 128,
        c["dql"],
        SM_SCALE,
        linear,
        use_pdl,
        prewait,
        n1,
    )
    return out, q8, sf


_CASES = {
    "b1-165k": dict(seq_lens=[165521]),
    "b2-mixed": dict(seq_lens=[165521, 98765]),
    "b2-short": dict(seq_lens=[130, 5000]),
    "b1-one-token-tail": dict(seq_lens=[128 * 40 + 1]),
    "b1-padded-topk": dict(seq_lens=[700]),
    "b1-tiny": dict(seq_lens=[4]),
    "b2-large-logits": dict(seq_lens=[165521, 64000], q_hi=0x70),
    "b1-ties": dict(seq_lens=[20000], tie=True),
    "b2-disjoint": dict(seq_lens=[165521, 165521], overlap=0),
    "b4": dict(seq_lens=[165521, 120000, 80000, 30000]),
}

# The TP2WIDE_LC2 steps of the K7 (qlen 8) verify: 5 / 8 / 12 / 16 requests = 40 / 64 / 96 / 128 rows, mixed lengths
# with 128-token edges inside the verify rows and a zero-length padding request.
_WIDE_CASES = {
    "k7-c5": dict(seq_lens=[165521, 20000, 1285, 129, 0], dql=8),
    "k7-c8": dict(seq_lens=[165521, 98765, 20000, 2048, 1285, 136, 4, 0], dql=8),
    "k7-c12": dict(
        seq_lens=[165521] + [(40000, 20000, 1285, 129, 2056)[i % 5] for i in range(10)] + [0], dql=8
    ),
    "k7-c16": dict(
        seq_lens=[165521] + [(40000, 20000, 1285, 129, 2056, 136)[i % 6] for i in range(14)] + [0],
        dql=8,
    ),
    # TP2WIDE_LC2_448: the agg TP2 Y steps at K3 (4 rows / request: c46 / c56 / c64 = 184 / 224 / 256 rows)
    # and the K7 steps at 24 / 46 / 56 requests (192 / 368 / 448 rows); 184 and 368 end in a partial scale tile.
    **{
        f"k{dql - 1}-c{reqs}": dict(
            seq_lens=[165521]
            + [(40000, 20000, 1285, 129, 2056, 136)[i % 6] for i in range(reqs - 2)]
            + [0],
            dql=dql,
        )
        for dql, reqs in ((4, 46), (4, 56), (4, 64), (8, 24), (8, 46), (8, 56))
    },
}


# Boundary rows: long / short / one-token-tail / padded top-k / tiny / ties / 4 requests, and the wide K3 / K7 steps
# (including the 184-row step that ends in a partial scale tile).
_BOUNDARY = [
    "b1-165k",
    "b2-mixed",
    "b1-one-token-tail",
    "b1-padded-topk",
    "b1-tiny",
    "b1-ties",
    "b4",
    "k7-c5",
    "k7-c16",
    "k3-c46",
    "k3-c56",
    "k7-c56",
]


@pytest.mark.parametrize("name", _BOUNDARY)
@pytest.mark.parametrize("seed", [0])
def test_band_vs_float64(name, seed) -> None:
    c = _case(seed=seed, **{**_CASES, **_WIDE_CASES}[name])
    out, _, _ = _run(c)
    elem, rms = _band(out, _reference(c))
    assert torch.isfinite(out.float()).all()
    assert elem <= _ELEMENT_ULP and rms <= _RMS_ULP, (elem, rms)


@pytest.mark.parametrize("name", ["b1-165k", "b2-mixed", "b1-padded-topk", "b4"])
def test_variants_write_the_same_rows(name) -> None:
    c = _case(seed=7, **_CASES[name])
    base, _, _ = _run(c)
    for prewait in (False, True):
        for mxfp8 in (False, True):
            out, _, _ = _run(c, prewait=prewait, mxfp8=mxfp8)
            assert torch.equal(out.view(torch.int16), base.view(torch.int16)), (prewait, mxfp8)
    lin, _, _ = _run(c, linear=True)
    elem, rms = _band(lin, _reference(c))
    assert elem <= _ELEMENT_ULP and rms <= _RMS_ULP, (elem, rms)


@pytest.mark.parametrize(
    "name,linear",
    [(n, False) for n in ("b1-165k", "b1-padded-topk", "b4")]
    + [(n, lin) for n in ("k7-c16", "k3-c46", "k7-c56") for lin in (False, True)],
)
def test_mxfp8_handoff_is_stock_mxfp8_quantize(name, linear) -> None:
    """The o_proj interface: (q8, sf) == mxfp8_quantize(out, is_sf_swizzled_layout=True), bitwise."""
    c = _case(seed=3, **{**_CASES, **_WIDE_CASES}[name])
    out, q8, sf = _run(c, linear=linear, mxfp8=True)
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(out, True)
    assert torch.equal(q8, ref_q.view(torch.uint8).reshape(q8.shape))
    ref_sf = ref_sf.view(torch.uint8).reshape(-1)
    assert torch.equal(sf[: ref_sf.numel()], ref_sf)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """The catalog entry's cells through its own __call__ (guarded by is_valid) and its own reference."""
    spec = cell.spec
    c = _case(
        list(spec["seq_lens"]),
        seed=zlib.crc32(cell.why.encode()) % 2**31,
        dql=spec["decode_query_len"],
    )
    total_q = c["q"].shape[0]
    # the hand-off buffers hold whole 128-row tiles (MergeMxfp8.MAX_ROWS)
    alloc = (total_q + 127) // 128 * 128
    out = torch.empty(total_q, WIDTH, dtype=torch.bfloat16, device="cuda")
    q8 = torch.zeros(alloc, WIDTH, dtype=torch.uint8, device="cuda") if spec["mxfp8"] else None
    sf = (
        torch.zeros(alloc * (WIDTH // 32), dtype=torch.uint8, device="cuda")
        if spec["mxfp8"]
        else None
    )
    args = (
        c["q"].view(torch.uint8),
        c["k"],
        c["v"],
        c["ksc"],
        c["vsc"],
        c["kgs"],
        c["vgs"],
        c["topk"],
        c["bt"],
        c["seq"],
        out,
        q8,
        sf,
        WIDTH // 128,
        c["dql"],
        SM_SCALE,
        spec["linear_softmax"],
        False,
        spec["prewait"],
    )
    certifying(op)
    op(*args)
    ref = op.reference(*args)
    assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
    assert torch.isfinite(out.float()).all()
    if spec["mxfp8"]:
        ref_q8, ref_sf = mxfp8_handoff_reference(out, alloc)
        assert torch.equal(q8[:total_q], ref_q8)
        assert torch.equal(sf, ref_sf)
        stock_q, stock_sf = torch.ops.trtllm.mxfp8_quantize(out, True)
        assert torch.equal(q8[:total_q], stock_q.view(torch.uint8).reshape(total_q, WIDTH))
        assert torch.equal(sf[: stock_sf.numel()], stock_sf.view(torch.uint8).reshape(-1))


# LC on vs the killed range (SPARSE_DECODE_LC = frozenset(), TRTLLM_M31_DISABLE_TP2WIDE_LC2=1: this target's Triton
# producer + merge) through the target's
# minimax_m3_sparse_attn_decode on the same inputs. Not bitwise: the split-K merge sums the 16 partials in Triton's
# reduction order there and in chunk order here (the per-block math is the same), so the rows differ by summation
# order only. Band, in BF16 ulps of each row's largest magnitude: measured on SM107 at most 1.92 per element and
# 0.18 RMS, so 2 / 0.25.
# The K7 wide cells (40..128 rows; this target's Triton route at 1-2 top-k chunks, unlike the cells above): LC and the
# killed Triton path each sit within the float64-reference band (_ELEMENT_ULP / _RMS_ULP, asserted below for both), so
# their distance is bounded by twice that band. Measured on SM107: at most 1.97 per element and 0.263 RMS.


# N1 (n1_ids): the layer's previous-step block ids, prefetched into L2 before the grid wait and overwritten with this
# step's. A hint: the rows must be bitwise the rows without it, whatever the buffer holds.


def _n1_expected(c, rows, before=None):
    """The buffer after a launch: row ``tok`` < T holds topk[h, tok, r] for r below the token's block count, else -1;
    rows at or past T keep what they held (``before``, default -1)."""
    total_q, dql = c["q"].shape[0], c["dql"]
    exp = (
        before.clone()
        if before is not None
        else torch.full((KV_HEADS, rows, TOPK), -1, dtype=torch.int32, device="cuda")
    )
    for tok in range(min(total_q, rows)):
        r, qo = divmod(tok, dql)
        kv_len = max(int(c["seq"][r]) - dql + qo + 1, 0)
        real = min(TOPK, (kv_len + PAGE - 1) // PAGE)
        exp[:, tok] = -1
        exp[:, tok, :real] = c["topk"][:, tok, :real]
    return exp


def _n1_garbage(rows, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(
        -5, 1 << 20, (KV_HEADS, rows, TOPK), dtype=torch.int32, device="cuda", generator=gen
    )
    ids[0, :, 0] = 2**31 - 1
    ids[1, :, 0] = -(2**31)
    return ids


# In-model request counts (1, 2, 4, 5, 8 Eagle3 verify requests of 4 rows), a full 128-row step, a plain decode.
_N1_CASES = {
    "m1": dict(seq_lens=[165521]),
    "m2": dict(seq_lens=[165521, 98765]),
    "m4": dict(seq_lens=[165521, 120000, 700, 30000]),
    "m5": dict(seq_lens=[165521, 4, 130, 98765, 5000]),
    "m8": dict(seq_lens=[165521, 98765, 64000, 30000, 20000, 5000, 700, 130]),
    "m32": dict(seq_lens=[4096 * (i + 1) + 77 * i for i in range(32)]),
    "m2-dql1": dict(seq_lens=[165521, 5000], dql=1),
    "k7-c5": _WIDE_CASES["k7-c5"],
    "k7-c16": _WIDE_CASES["k7-c16"],
}


@pytest.mark.parametrize("name", list(_N1_CASES))
def test_n1_rows_bitwise_and_ids_recorded(name) -> None:
    c = _case(seed=11, **_N1_CASES[name])
    total_q = c["q"].shape[0]
    for prewait in (False, True):
        for mxfp8 in (False, True):
            base, bq8, bsf = _run(c, prewait=prewait, mxfp8=mxfp8)
            ids = torch.full((KV_HEADS, 128, TOPK), -1, dtype=torch.int32, device="cuda")
            # First step (all -1: no prefetch), second (last step's ids: prefetch), then stale garbage ids.
            for step, init in enumerate((None, None, _n1_garbage(128, 3))):
                if init is not None:
                    ids.copy_(init)
                exp = _n1_expected(c, 128, ids)
                out, q8, sf = _run(c, prewait=prewait, mxfp8=mxfp8, n1=ids)
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int16), base.view(torch.int16)), (
                    prewait,
                    mxfp8,
                    step,
                )
                if mxfp8:
                    assert torch.equal(q8, bq8) and torch.equal(sf, bsf), (prewait, mxfp8, step)
                assert torch.equal(ids, exp), (prewait, mxfp8, step)
    if total_q > 1:
        # A buffer with fewer rows than the step: those rows record, the rest skip both prefetch and record.
        rows = total_q // 2
        ids = torch.full((KV_HEADS, rows, TOPK), -1, dtype=torch.int32, device="cuda")
        out, _, _ = _run(c, n1=ids)
        assert torch.equal(out.view(torch.int16), _run(c)[0].view(torch.int16))
        assert torch.equal(ids, _n1_expected(c, rows))


def test_n1_pdl_stress_and_graph_replay() -> None:
    """Back-to-back PDL launches of two layers (each with its own id buffer, alternating steps), eager and replayed
    from a CUDA graph: every launch writes the rows of the launch without N1, and the buffers end at the last step's
    ids."""
    cases = [_case(seed=21, **_N1_CASES["m2"]), _case(seed=22, **_N1_CASES["m5"])]
    bases = [_run(c, mxfp8=True)[0] for c in cases]
    ids = [torch.full((KV_HEADS, 128, TOPK), -1, dtype=torch.int32, device="cuda") for _ in cases]
    iters = 200
    outs = [[torch.empty_like(b) for _ in range(iters)] for b in bases]
    for i in range(iters):
        for j, c in enumerate(cases):
            _run(c, mxfp8=True, n1=ids[j], use_pdl=True, out=outs[j][i])
    torch.cuda.synchronize()
    for j in range(len(cases)):
        assert all(torch.equal(o.view(torch.int16), bases[j].view(torch.int16)) for o in outs[j]), j
        assert torch.equal(ids[j], _n1_expected(cases[j], 128)), j
    for t in ids:
        t.fill_(-1)
    gouts = [torch.empty_like(b) for b in bases]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for j, c in enumerate(cases):  # warm the launch configuration outside capture
            _run(c, mxfp8=True, n1=ids[j], use_pdl=True, out=gouts[j])
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(4):
                for j, c in enumerate(cases):
                    _run(c, mxfp8=True, n1=ids[j], use_pdl=True, out=gouts[j])
    garbage = _n1_garbage(128, 5)
    for t in ids:
        t.copy_(garbage)
    for _ in range(25):
        for o in gouts:
            o.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for j in range(len(cases)):
            assert torch.equal(gouts[j].view(torch.int16), bases[j].view(torch.int16)), j
            assert torch.equal(ids[j], _n1_expected(cases[j], 128, garbage)), j


# ---------------------------------------------------------------------------------- the target's TP2WIDE_LC2 dispatch


def _kernels():
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        m3_attn_lc,
        m3_prequant,
        m3_report,
        triton_sparse_decode,
    )

    return m3_attn_lc, m3_prequant, m3_report, triton_sparse_decode


# LC2_ALLROWS (LC2 from one row and past LC_MAX_ROWS) and LC2_IDENTITY (LC2 on identity steps) are default ON; the
# band / Triton-route contracts below are tested with both killed, their defaults in
# test_allrows_and_identity_defaults.


# Below lc_min_rows: 1 / 2 / 4 requests of the K7 verify (8 / 16 / 32 rows), 1 / 2 requests of the K3 verify (4 / 8
# rows), a plain decode at 32 requests (32 rows).
# TP2_LC2_SMALL: the K3 verify (4 query rows per request) at 3 / 4 / 8 requests (12 / 16 / 32 rows) takes LC2.


def test_handoff_buffers_cover_whole_scale_tiles() -> None:
    """The hand-off buffers hold every row and every whole 128-row scale tile up to LC_MAX_ROWS (448 is not a multiple
    of 128: the last tile is written in full), and each step's hand-off slice is its whole tiles, never truncated."""
    _, _, _, tsd = _kernels()
    q8, sf = tsd.MERGE_MXFP8.buffers(WIDTH, torch.device("cuda"))
    assert q8.shape[0] >= tsd.LC_MAX_ROWS
    assert sf.numel() >= (tsd.LC_MAX_ROWS + 127) // 128 * 128 * (WIDTH // 32)
    for rows in (1, 33, 128, 129, 184, 224, 256, 368, tsd.LC_MAX_ROWS):
        hq, hs = tsd.MergeMxfp8.handoff((q8, sf), rows)
        assert hq.shape[0] == rows and hs.numel() == (rows + 127) // 128 * 128 * (WIDTH // 32), rows
