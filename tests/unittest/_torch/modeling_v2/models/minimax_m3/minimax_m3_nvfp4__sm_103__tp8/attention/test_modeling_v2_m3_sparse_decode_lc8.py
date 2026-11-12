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

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_sparse_decode_lc8 import (  # noqa: E501
    NUM_HEADS,
    WIDTH,
    dequant_nvfp4_pages,
    mxfp8_handoff_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_sparse_decode_lc8 import (  # noqa: E501
    m3_sparse_decode_lc8 as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import NUM_HEADS as CASE_HEADS  # noqa: E402 — needs the path declared above
from _m3_nvfp4_cache import build_step  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

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
