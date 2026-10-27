# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode_lc entry (one-launch long-context sparse decode)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
    WIDTH,
    dequant_nvfp4_pages,
    mxfp8_handoff_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
    m3_sparse_decode_lc as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import build_step  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)

assert torch.cuda.is_available(), "m3_sparse_decode_lc requires a CUDA device"

#: Hand-off buffers are sized as MERGE_MXFP8 sizes them.
_HANDOFF_ROWS = 128

# Band, in BF16 ulps (2^-8) of each output row's largest magnitude, and of the relative RMS.
#
# The reference is float64 attention over the exactly dequantized blocks, rounded once to BF16. The kernel's
# operands are exact too (E4M3 q and E2M1 x E4M3 K/V are exact in FP16), so what separates the two is: the final
# BF16 rounding on both sides (up to one ulp of the element, at most one of the row's scale), the softmax
# probabilities rounded to FP16 before PV (2^-11 relative each, averaging down over the block), ex2.approx /
# lg2.approx (~2^-22) and FP32 accumulation order. Hence 2 ulps per element; the RMS is dominated by the two
# roundings (each at most half an ulp, about 0.3 in aggregate), so half an ulp. Measured on GB300 over the cells:
# at most 1.91 ulps per element and 0.25 ulps RMS.
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5


def _call(case, spec, out, q8, sf):
    return op(
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
        WIDTH // 128,
        case.decode_query_len,
        case.sm_scale,
        spec["linear_softmax"],
        False,
        True,
    )


def _buffers(total_q, mxfp8):
    out = torch.empty(total_q, WIDTH, dtype=torch.bfloat16, device="cuda")
    if not mxfp8:
        return out, None, None
    q8 = torch.zeros(_HANDOFF_ROWS, WIDTH, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(_HANDOFF_ROWS * (WIDTH // 32), dtype=torch.uint8, device="cuda")
    return out, q8, sf


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    case = build_step(
        spec["seq_lens"],
        spec["decode_query_len"],
        identity=False,
        seed=zlib.crc32(cell.why.encode()) % 2**31,
    )
    total_q = case.q.shape[0]
    out, q8, sf = _buffers(total_q, spec["mxfp8"])
    certifying(op)
    _call(case, spec, out, q8, sf)
    ref = op.reference(
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
        WIDTH // 128,
        case.decode_query_len,
        case.sm_scale,
        spec["linear_softmax"],
        False,
        True,
    )
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
    """The two softmax recurrences and the hand-off switch must not change the BF16 rows beyond rounding."""
    spec = dict(seq_lens=(5000, 12345), decode_query_len=4)
    case = build_step(spec["seq_lens"], 4, identity=False, seed=11)
    outs = []
    for linear in (False, True):
        for mxfp8 in (False, True):
            out, q8, sf = _buffers(case.q.shape[0], mxfp8)
            _call(case, dict(linear_softmax=linear), out, q8, sf)
            outs.append(out)
    for other in outs[1:]:
        assert_within_ulp(other, outs[0], element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
    torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)  # hand-off on/off: same rows


def test_op_refuses_the_uninstantiated_late_trigger() -> None:
    case = build_step((3000,), 1, identity=False, seed=3)
    out, q8, sf = _buffers(1, True)
    with pytest.raises(RuntimeError, match="early_trigger"):
        op(
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
            WIDTH // 128,
            1,
            case.sm_scale,
            True,
            False,
            False,
        )


def test_guard_refuses_a_block_past_the_request() -> None:
    """A selected id at or past the request's block count reads a page the token cannot attend; with every
    position masked the CTA's softmax is NaN."""
    case = build_step((3000,), 1, identity=False, seed=4)
    case.topk[0, 0, 0] = 40
    out, q8, sf = _buffers(1, True)
    with pytest.raises(AssertionError, match="block count"):
        with validating(op):
            _call(case, dict(linear_softmax=False), out, q8, sf)


def test_guard_refuses_strided_page_rows() -> None:
    """The kernel walks a page's token rows at a fixed 64-byte pitch; a padded row stride is misread."""
    case = build_step((3000,), 1, identity=False, seed=6)
    wide = torch.zeros(*case.k_paged.shape[:3], 80, dtype=torch.uint8, device="cuda")[..., :64]
    wide.copy_(case.k_paged)
    case.k_paged = wide
    out, q8, sf = _buffers(1, True)
    with pytest.raises(AssertionError, match="64-byte rows"):
        with validating(op):
            _call(case, dict(linear_softmax=False), out, q8, sf)


def test_guard_refuses_a_narrow_handoff() -> None:
    case = build_step((3000,), 1, identity=False, seed=7)
    out, q8, sf = _buffers(1, True)
    with pytest.raises(AssertionError, match="num_sf_k_tiles"):
        with validating(op):
            op(
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
                8,
                1,
                case.sm_scale,
                False,
                False,
                True,
            )


@pytest.mark.parametrize(
    "seq_lens, decode_query_len",
    [
        ((165521,), 4),
        ((5000, 12345), 4),
        ((700,), 1),
        ((1500, 300, 90000), 2),
        ((200000, 3000, 64, 70000), 4),
    ],
    ids=["1x165k", "2req", "short-1row", "3req-mixed", "4req-incl-64"],
)
def test_merge_once_is_bitwise_the_image_op(seq_lens, decode_query_len) -> None:
    """dispatch.SPARSE_DECODE_LC_MERGE_ONCE: the target's op (merge weights formed once per CTA) writes exactly the
    image op's rows and hand-off, at both softmax recurrences, including requests with fewer than 16 selected blocks
    (empty chunks, -inf LSE). With SPARSE_DECODE_LC_PREFETCH_OPROJ it also pulls two ranges the size of the TP4
    o_proj weight and scales (one not a multiple of 16 bytes) into L2: the same bits again. With
    SPARSE_DECODE_LC_PREFETCH_NEXT_KV it records each row's selection (the first min(16, blocks) ids, -1 after) and
    prefetches a next layer's recorded blocks, stale ids past a request's blocks included: the same bits, and the
    record."""
    assert hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp4_sparse_decode_lc"), (
        "the target's sparse decode op is not loaded (its TP4 extension)"
    )
    case = build_step(
        seq_lens, decode_query_len, identity=False, seed=zlib.crc32(repr(seq_lens).encode())
    )
    o_proj_weight = torch.empty(6144, 2048, dtype=torch.uint8, device="cuda")
    o_proj_scale = torch.empty(6144 * 64 + 9, dtype=torch.uint8, device="cuda")
    rows = case.q.shape[0]
    n1_self = torch.full((128, 16), -7, dtype=torch.int32, device="cuda")
    n1_next = torch.randint(
        -1, 2 * (max(seq_lens) // 128 + 1), (128, 16), dtype=torch.int32, device="cuda"
    )
    next_kv = (n1_self, n1_next, case.k_paged, case.v_paged, case.k_block_scale, case.v_block_scale)
    variants = (
        (False, None, None),
        (True, None, None),
        (True, (o_proj_weight, o_proj_scale), None),
        (True, (o_proj_weight, o_proj_scale), next_kv),
    )
    for linear in (False, True):
        for mxfp8 in (False, True):
            got = []
            for merge_once, prefetch, n1 in variants:
                out, q8, sf = _buffers(case.q.shape[0], mxfp8)
                op(
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
                    WIDTH // 128,
                    case.decode_query_len,
                    case.sm_scale,
                    linear,
                    False,
                    True,
                    merge_once=merge_once,
                    prefetch=prefetch,
                    next_kv=n1,
                )
                got.append((out, q8, sf))
            torch.cuda.synchronize()
            a, aq, asf = got[0]
            for i, (b, bq, bsf) in enumerate(got[1:], 1):
                assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (linear, mxfp8, i)
                if mxfp8:
                    assert torch.equal(aq, bq) and torch.equal(asf, bsf), (linear, mxfp8, i)
            # The record: each row's first min(16, blocks) selected ids, -1 past them; rows past the step untouched.
            want = case.topk[0].clone()
            for tok in range(rows):
                req, qoff = divmod(tok, case.decode_query_len)
                kv_len = max(int(case.seq_lens[req]) - case.decode_query_len + qoff + 1, 0)
                want[tok, min(16, (kv_len + 127) // 128) :] = -1
            assert torch.equal(n1_self[:rows], want), "the recorded selection"
            assert bool((n1_self[rows:] == -7).all()), "rows past the step"


def _tp4_op(case, out, q8, sf, linear, use_pdl, one_div):
    torch.ops.trtllm.minimax_m3_sm107_tp4_sparse_decode_lc(
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
        WIDTH // 128,
        case.decode_query_len,
        case.sm_scale,
        linear,
        use_pdl,
        True,
        one_div=one_div,
    )


_ONE_DIV_STEPS = [
    ((165521,), 4),
    ((5000, 12345), 4),
    ((700,), 1),
    ((1500, 300, 90000), 2),
    ((200000, 3000, 64, 70000), 4),
]


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize(
    "seq_lens, decode_query_len",
    _ONE_DIV_STEPS,
    ids=["1x165k", "2req", "short-1row", "3req-mixed", "4req-incl-64"],
)
def test_one_div_merge_is_deterministic_and_in_band(seq_lens, decode_query_len, linear) -> None:
    """The one-division merge (on in this target): it writes the same rows on every run
    (5 eager runs, PDL on and off) and under CUDA-graph replay (3 replays == eager), MXFP8 hand-off included; the rows
    are within the float64 reference's band, and within summation-order distance of the per-chunk merge."""
    case = build_step(
        seq_lens, decode_query_len, identity=False, seed=zlib.crc32(repr(seq_lens).encode()) + 1
    )
    total_q = case.q.shape[0]
    runs = []
    for i in range(6):
        out, q8, sf = _buffers(total_q, True)
        _tp4_op(case, out, q8, sf, linear, i == 5, True)
        runs.append((out, q8, sf))
    torch.cuda.synchronize()
    a, aq, asf = runs[0]
    for b, bq, bsf in runs[1:]:
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        assert torch.equal(aq, bq) and torch.equal(asf, bsf)
    out = torch.empty_like(a)
    _tp4_op(case, out, None, None, linear, True, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _tp4_op(case, out, None, None, linear, True, True)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), a.view(torch.int16))
    per, _, _ = _buffers(total_q, False)
    _tp4_op(case, per, None, None, linear, False, False)
    ref = op.reference(
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
        a,
        None,
        None,
        WIDTH // 128,
        case.decode_query_len,
        case.sm_scale,
        linear,
        False,
        True,
    )
    assert torch.isfinite(a.float()).all()
    assert_within_ulp(a, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)
    assert_within_ulp(a, per, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


_BT_PREWAIT_STEPS = [
    ((165521,), 4),
    ((5000, 12345), 4),
    ((700,), 1),
    ((1500, 300, 90000), 2),
    ((200000, 3000, 64, 70000), 4),
    ((524288,), 4),  # exactly the 4096-entry staging cap
    ((524289, 2000), 4),  # one block past the cap: the global lookup, beside a staged request
    ((600000,), 1),
]


@pytest.mark.parametrize(
    "seq_lens, decode_query_len",
    _BT_PREWAIT_STEPS,
    ids=[
        "1x165k",
        "2req",
        "short-1row",
        "3req-mixed",
        "4req-incl-64",
        "at-cap",
        "past-cap+short",
        "600k-1row",
    ],
)
def test_bt_prewait_is_bitwise(seq_lens, decode_query_len) -> None:
    """The op's ``bt_prewait`` option (not used by this target): staging the request's block-table row in shared memory
    before the grid wait writes exactly the rows and MXFP8 hand-off of the global lookup, at both softmax recurrences,
    with and without the one-division merge, PDL on and off, eager and under CUDA-graph replay; rows at the 4096-entry
    cap are staged, longer ones keep the global lookup; the o_proj / next-KV prefetch hints alongside change nothing."""
    case = build_step(
        seq_lens, decode_query_len, identity=False, seed=zlib.crc32(repr(seq_lens).encode()) + 7
    )
    total_q = case.q.shape[0]
    n1_self = torch.full((128, 16), -7, dtype=torch.int32, device="cuda")
    n1_next = torch.randint(
        -1, 2 * (max(seq_lens) // 128 + 1), (128, 16), dtype=torch.int32, device="cuda"
    )
    next_kv = (n1_self, n1_next, case.k_paged, case.v_paged, case.k_block_scale, case.v_block_scale)

    def run(out, q8, sf, linear, one_div, bt, use_pdl, hints=False):
        torch.ops.trtllm.minimax_m3_sm107_tp4_sparse_decode_lc(
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
            WIDTH // 128,
            case.decode_query_len,
            case.sm_scale,
            linear,
            use_pdl,
            True,
            *((None, None) + next_kv if hints else ()),
            one_div=one_div,
            bt_prewait=bt,
        )

    for linear in (False, True):
        for one_div in (False, True):
            base = _buffers(total_q, True)
            run(*base, linear, one_div, False, False)
            arms = []
            for bt, use_pdl, hints in (
                (True, False, False),
                (True, True, False),
                (True, True, True),
            ):
                bufs = _buffers(total_q, True)
                run(*bufs, linear, one_div, bt, use_pdl, hints)
                arms.append(bufs)
            torch.cuda.synchronize()
            a, aq, asf = base
            for i, (b, bq, bsf) in enumerate(arms):
                assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (linear, one_div, i)
                assert torch.equal(aq, bq) and torch.equal(asf, bsf), (linear, one_div, i)
            out = torch.empty_like(a)
            run(out, None, None, linear, one_div, True, True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(out, None, None, linear, one_div, True, True)
            for _ in range(3):
                out.zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int16), a.view(torch.int16)), (
                    linear,
                    one_div,
                    "graph",
                )
