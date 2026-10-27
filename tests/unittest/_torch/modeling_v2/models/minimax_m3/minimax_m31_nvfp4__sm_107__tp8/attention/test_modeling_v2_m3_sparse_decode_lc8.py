# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode_lc8 entry (one-launch long-context sparse decode at the TP8 8 query heads,
``torch.ops.trtllm.minimax_m3_tp8_sparse_decode_lc8``).

Every cell is held to the float64 reference within the derived band, its MXFP8 hand-off to the recipe bit for bit, and
the two ``prewait`` variants (staging before or after the grid wait) to each other bit for bit.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.attention.m3_sparse_decode_lc8 import (  # noqa: E501
    NUM_HEADS,
    WIDTH,
    dequant_nvfp4_pages,
    mxfp8_handoff_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.attention.m3_sparse_decode_lc8 import (  # noqa: E501
    m3_sparse_decode_lc8 as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import NUM_HEADS as CASE_HEADS  # noqa: E402 — needs the path declared above
from _m3_nvfp4_cache import build_step  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

assert torch.cuda.is_available(), "m3_sparse_decode_lc8 requires a CUDA device"

#: Hand-off buffers are sized as MERGE_MXFP8 sizes them.
_HANDOFF_ROWS = 128

# Band, in BF16 ulps (2^-8) of each output row's largest magnitude, and of the relative RMS.
#
# The reference is float64 attention over the exactly dequantized blocks, rounded once to BF16. The kernel's
# operands are exact too (E4M3 q and E2M1 x E4M3 K/V are exact in FP16), so what separates the two is: the final
# BF16 rounding on both sides (up to one ulp of the element, at most one of the row's scale), the softmax
# probabilities rounded to FP16 before PV (2^-11 relative each, averaging down over the block), ex2.approx /
# lg2.approx (~2^-22) and FP32 accumulation order. Hence 2 ulps per element; the RMS is dominated by the two
# roundings (each at most half an ulp, about 0.3 in aggregate), so half an ulp. None of these depends on the head
# count or on the MMA orientation: the tp4 target's 16-head kernel measured at most 1.91 ulps per element and 0.25
# ulps RMS on GB300 over the same cells.
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5


def _args(case, out, q8, sf, linear_softmax, prewait, num_sf_k_tiles=WIDTH // 128):
    return (
        case.q.view(torch.uint8),
        case.k_paged,
        case.v_paged,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        case.topk[0],
        case.block_table,
        case.seq_lens,
        out,
        q8,
        sf,
        num_sf_k_tiles,
        case.decode_query_len,
        case.sm_scale,
        linear_softmax,
        False,
        prewait,
    )


def _buffers(total_q, mxfp8):
    out = torch.empty(total_q, WIDTH, dtype=torch.bfloat16, device="cuda")
    if not mxfp8:
        return out, None, None
    q8 = torch.zeros(_HANDOFF_ROWS, WIDTH, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(_HANDOFF_ROWS * (WIDTH // 32), dtype=torch.uint8, device="cuda")
    return out, q8, sf


def _case(spec, seed):
    return build_step(spec["seq_lens"], spec["decode_query_len"], identity=False, seed=seed)


def test_the_cases_are_the_tp8_geometry() -> None:
    assert CASE_HEADS == NUM_HEADS == 8 and WIDTH == 1024


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    case = _case(spec, zlib.crc32(cell.why.encode()) % 2**31)
    total_q = case.q.shape[0]
    out, q8, sf = _buffers(total_q, spec["mxfp8"])
    args = _args(case, out, q8, sf, spec["linear_softmax"], spec["prewait"])
    certifying(op)
    op(*args)
    ref = op.reference(*args)
    assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
    # Padding rows attend nothing and must be exact zeros, not NaN.
    assert torch.isfinite(out.float()).all()
    if spec["mxfp8"]:
        # The hand-off is a deterministic function of the op's own BF16 rows: exact.
        ref_q8, ref_sf = mxfp8_handoff_reference(out, _HANDOFF_ROWS)
        assert torch.equal(q8[:total_q], ref_q8), (
            "MXFP8 rows differ from the recipe applied to the BF16 rows"
        )
        assert torch.equal(sf, ref_sf), "UE8M0 scales differ from the recipe / swizzle"

    # The other staging variant: the same arithmetic, so the same bits (rows and hand-off).
    other_out, other_q8, other_sf = _buffers(total_q, spec["mxfp8"])
    op(*_args(case, other_out, other_q8, other_sf, spec["linear_softmax"], not spec["prewait"]))
    assert torch.equal(other_out.view(torch.int16), out.view(torch.int16)), (
        "prewait and plain staging wrote different rows"
    )
    if spec["mxfp8"]:
        assert torch.equal(other_q8, q8) and torch.equal(other_sf, sf)


def test_reference_dequant_matches_the_quantizer() -> None:
    """The reference reads the paged bytes through its own layout claims (K scales token-major, V scales in
    4x4 token quads). Hold them to the quantizer's own flat output: every slot was written in order, so slot
    ``s`` is page ``s // 128``, token ``s % 128``, and the flat scales are token-major by construction."""
    case = build_step((3000,), 1, identity=False, seed=5)
    pages = torch.arange(case.k_paged.shape[0], device="cuda")
    views = ((case.k_paged, case.k_block_scale, False), (case.v_paged, case.v_block_scale, True))
    for role, (data, scale, quad) in enumerate(views):
        flat_data, flat_scale = torch.ops.trtllm.fp4_quantize(
            case.source[role].reshape(case.num_slots, 1, -1).contiguous(),
            case.quant[role + 1 : role + 2],
            16,
            False,
            False,
        )
        expected = dequant_nvfp4_pages(
            flat_data.view(torch.uint8).reshape(-1, 1, 128, 64),
            flat_scale.view(torch.uint8).reshape(-1, 1, 128, 8),
            pages,
            quad_scales=False,
        )
        got = dequant_nvfp4_pages(data, scale, pages, quad_scales=quad)
        assert torch.equal(got, expected), (
            f"role {role}: paged dequantization disagrees with the quantizer"
        )


def test_the_variants_agree() -> None:
    """The two softmax recurrences, the hand-off switch and the staging must not change the BF16 rows beyond
    rounding; the hand-off switch and the staging not at all."""
    case = build_step((5000, 12345), 4, identity=False, seed=11)
    outs = {}
    for linear in (False, True):
        for mxfp8 in (False, True):
            for prewait in (False, True):
                out, q8, sf = _buffers(case.q.shape[0], mxfp8)
                op(*_args(case, out, q8, sf, linear, prewait))
                outs[(linear, mxfp8, prewait)] = out
    base = outs[(False, False, False)]
    for key, other in outs.items():
        assert_within_ulp(other, base, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
        same_recurrence = outs[(key[0], False, False)]
        torch.testing.assert_close(other, same_recurrence, rtol=0, atol=0)


def test_guard_refuses_a_block_past_the_request() -> None:
    """A selected id at or past the request's block count reads a page the token cannot attend; with every
    position masked the CTA's softmax is NaN."""
    case = build_step((3000,), 1, identity=False, seed=4)
    case.topk[0, 0, 0] = 40
    out, q8, sf = _buffers(1, True)
    with pytest.raises(AssertionError, match="block count"):
        with validating(op):
            op(*_args(case, out, q8, sf, False, True))


def test_guard_refuses_strided_page_rows() -> None:
    """The kernel walks a page's token rows at a fixed 64-byte pitch; a padded row stride is misread."""
    case = build_step((3000,), 1, identity=False, seed=6)
    wide = torch.zeros(*case.k_paged.shape[:3], 80, dtype=torch.uint8, device="cuda")[..., :64]
    wide.copy_(case.k_paged)
    case.k_paged = wide
    out, q8, sf = _buffers(1, True)
    with pytest.raises(AssertionError, match="64-byte rows"):
        with validating(op):
            op(*_args(case, out, q8, sf, False, True))


def test_guard_refuses_a_narrow_handoff() -> None:
    """The hand-off rows are 1024 wide (8 scale k-tiles); a smaller count overlaps them."""
    case = build_step((3000,), 1, identity=False, seed=7)
    out, q8, sf = _buffers(1, True)
    with pytest.raises(AssertionError, match="num_sf_k_tiles"):
        with validating(op):
            op(*_args(case, out, q8, sf, False, True, num_sf_k_tiles=4))


@pytest.mark.parametrize("prewait", [True, False], ids=["prewait", "plain"])
@pytest.mark.parametrize("mxfp8", [True, False], ids=["handoff", "no_handoff"])
def test_run_to_run_deterministic(prewait, mxfp8) -> None:
    """The exact-class reference for the TP8 sparse decode (band vs fp64, as the TP4 LC): every launch of the same
    step writes the same bits, in eager repeats and in a CUDA-graph replay."""
    spec = dict(seq_lens=(165521, 5000), decode_query_len=4)
    case = _case(spec, seed=91)
    for linear in (False, True):
        runs = []
        for _ in range(3):
            out, q8, sf = _buffers(case.q.shape[0], mxfp8)
            op(*_args(case, out, q8, sf, linear, prewait))
            runs.append((out, q8, sf))
        g_out, g_q8, g_sf = _buffers(case.q.shape[0], mxfp8)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(*_args(case, g_out, g_q8, g_sf, linear, prewait))
        graph.replay()
        torch.cuda.synchronize()
        runs.append((g_out, g_q8, g_sf))
        base = runs[0]
        for other in runs[1:]:
            assert torch.equal(other[0].view(torch.int16), base[0].view(torch.int16)), (
                linear,
                prewait,
                mxfp8,
            )
            if mxfp8:
                assert torch.equal(other[1], base[1]) and torch.equal(other[2], base[2]), (
                    linear,
                    prewait,
                )


@pytest.mark.parametrize(
    "seq_lens, decode_query_len",
    [((165521,), 4), ((5000, 12345), 4), ((700,), 1), ((200000, 3000, 64, 70000), 4)],
    ids=["1x165k", "2req", "short-1row", "4req-incl-64"],
)
def test_merge_once_is_bitwise_the_image_op(seq_lens, decode_query_len) -> None:
    """kernels/m3_attn_lc8.MERGE_ONCE: the target's op (merge weights formed once per CTA) writes exactly the
    image op's rows and hand-off, at both softmax recurrences and both staging variants, including requests with fewer
    than 16 selected blocks (empty chunks, -inf LSE). With SPARSE_DECODE_LC8_PREFETCH_OPROJ it also pulls two ranges
    the size of the TP8 o_proj weight and scales (one not a multiple of 16 bytes) into L2: the same bits again. With
    SPARSE_DECODE_LC8_PREFETCH_NEXT_KV it records each row's selection (the first min(16, blocks) ids, -1 after) and
    prefetches a next layer's recorded blocks, stale ids past a request's blocks included: the same bits, and the
    record."""
    assert hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp8_sparse_decode_lc8"), (
        "the target's LC8 op is not loaded (its TP8 extension)"
    )
    case = build_step(
        seq_lens, decode_query_len, identity=False, seed=zlib.crc32(repr(seq_lens).encode())
    )
    o_proj_weight = torch.empty(6144, 1024, dtype=torch.uint8, device="cuda")
    o_proj_scale = torch.empty(6144 * 32 + 9, dtype=torch.uint8, device="cuda")
    rows = case.q.shape[0]
    n1_next = torch.randint(
        -1, 2 * (max(seq_lens) // 128 + 1), (128, 16), dtype=torch.int32, device="cuda"
    )
    for linear in (False, True):
        for prewait in (False, True):
            for mxfp8 in (False, True):
                n1_self = torch.full((128, 16), -7, dtype=torch.int32, device="cuda")
                next_kv = (
                    n1_self,
                    n1_next,
                    case.k_paged.view(torch.uint8),
                    case.v_paged.view(torch.uint8),
                    case.k_block_scale.view(torch.uint8),
                    case.v_block_scale.view(torch.uint8),
                )
                variants = (
                    (False, None, None),
                    (True, None, None),
                    (True, (o_proj_weight, o_proj_scale), None),
                    (True, (o_proj_weight, o_proj_scale), next_kv),
                )
                got = []
                for merge_once, prefetch, n1 in variants:
                    out, q8, sf = _buffers(case.q.shape[0], mxfp8)
                    op(
                        *_args(case, out, q8, sf, linear, prewait),
                        merge_once=merge_once,
                        prefetch=prefetch,
                        next_kv=n1,
                    )
                    got.append((out, q8, sf))
                torch.cuda.synchronize()
                a, aq, asf = got[0]
                for i, (b, bq, bsf) in enumerate(got[1:], 1):
                    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
                        linear,
                        prewait,
                        mxfp8,
                        i,
                    )
                    if mxfp8:
                        assert torch.equal(aq, bq) and torch.equal(asf, bsf), (linear, prewait, i)
                # The record: each row's first min(16, blocks) selected ids, -1 past them; rows past the step untouched.
                want = case.topk[0].clone()
                for tok in range(rows):
                    req, qoff = divmod(tok, case.decode_query_len)
                    kv_len = max(int(case.seq_lens[req]) - case.decode_query_len + qoff + 1, 0)
                    want[tok, min(16, (kv_len + 127) // 128) :] = -1
                assert torch.equal(n1_self[:rows], want), "the recorded selection"
                assert bool((n1_self[rows:] == -7).all()), "rows past the step"


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("prewait", [True, False], ids=["prewait", "plain"])
@pytest.mark.parametrize(
    "seq_lens, decode_query_len",
    [((165521,), 4), ((5000, 12345), 4), ((700,), 1), ((200000, 3000, 64, 70000), 4)],
    ids=["1x165k", "2req", "short-1row", "4req-incl-64"],
)
def test_one_div_merge_is_deterministic_and_in_band(
    seq_lens, decode_query_len, prewait, linear
) -> None:
    """kernels/m3_attn_lc8.ONE_DIV_MERGE: the one-division merge writes the same rows on
    every run (5 eager runs, PDL on and off) and under CUDA-graph replay (3 replays == eager), MXFP8 hand-off included;
    the rows are within the float64 reference's band, and within summation-order distance of the per-chunk merge."""
    case = build_step(
        seq_lens, decode_query_len, identity=False, seed=zlib.crc32(repr(seq_lens).encode()) + 3
    )
    total_q = case.q.shape[0]

    def run(out, q8, sf, use_pdl, one_div):
        args = list(_args(case, out, q8, sf, linear, prewait))
        args[18] = use_pdl
        op(*args, merge_once=True, one_div=one_div)

    runs = []
    for i in range(6):
        out, q8, sf = _buffers(total_q, True)
        run(out, q8, sf, i == 5, True)
        runs.append((out, q8, sf))
    torch.cuda.synchronize()
    a, aq, asf = runs[0]
    for b, bq, bsf in runs[1:]:
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        assert torch.equal(aq, bq) and torch.equal(asf, bsf)
    out = torch.empty_like(a)
    run(out, None, None, True, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out, None, None, True, True)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), a.view(torch.int16))
    per, _, _ = _buffers(total_q, False)
    run(per, None, None, False, False)
    ref = op.reference(*_args(case, a, None, None, linear, prewait))
    assert torch.isfinite(a.float()).all()
    assert_within_ulp(a, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
    assert_within_ulp(a, per, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
