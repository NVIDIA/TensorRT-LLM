# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode_lc entry (one-launch long-context sparse decode)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
    WIDTH,
    dequant_nvfp4_pages,
    mxfp8_handoff_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
    m3_sparse_decode_lc as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import build_step  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

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
