# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode entry (the target's Triton block-sparse decode and its routes)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_sparse_decode import (  # noqa: E501
    m3_sparse_decode as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import (  # noqa: E402 — needs the path declared above
    HEAD_DIM,
    NUM_HEADS,
    build_step,
)
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_sparse_decode requires a CUDA device"

# Band, in BF16 ulps (2^-8) of each [8, 128] head row's largest magnitude, and of the relative RMS. Every route
# computes from exact operands (E4M3 q and E2M1 x E4M3 K/V are exact in FP16), so the gap to the float64 reference
# is the BF16 rounding on both sides (one ulp of the row at most), the FP16 probabilities before PV, the split-K
# partials kept in FP32 and merged by their log2-sum-exp, and approximate exp2/log2: 2 ulps per element, the two
# roundings (each at most half an ulp) dominating the RMS, so half an ulp. None of these depends on the head count:
# measured on GB300 over the same cells at the tp4 target's 16 heads, at most 1.80 ulps per element and 0.20 ulps
# RMS.
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5


def _call(case, output, identity):
    return op(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        output,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        identity,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    case = build_step(
        spec["seq_lens"],
        spec["decode_query_len"],
        identity=spec["identity"],
        seed=zlib.crc32(cell.why.encode()) % 2**31,
    )
    output = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    certifying(op)
    _call(case, output, spec["identity"])
    ref = op.reference(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        output,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        spec["identity"],
    )
    assert torch.isfinite(output.float()).all(), "padding rows must be zeros, not NaN"
    assert_within_ulp(output, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_identity_flag_does_not_change_the_answer() -> None:
    """On an identity table the identity routes (the producer, or the multi-query producer from 17 rows) and the
    selected-table route (the long-context kernel) compute the same attention."""
    case = build_step((2048, 130, 5, 0, 1777, 900, 2000, 0), 4, identity=True, seed=17)
    outs = []
    for identity in (True, False):
        output = torch.empty(
            case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        _call(case, output, identity)
        outs.append(output)
    assert_within_ulp(outs[0], outs[1], element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_guard_refuses_a_padded_selection_slot() -> None:
    """The Triton producers index the block table with every one of a token's first min(16, blocks) ids,
    -1 included."""
    case = build_step((1500,), 1, identity=True, seed=23)
    case.topk[0, 0, 3] = -1
    output = torch.empty(1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="blocks of the request"):
        with validating(op):
            _call(case, output, True)


def test_guard_refuses_a_false_identity_promise() -> None:
    """identity_blocks lets the multi-query producer read one table for the whole request."""
    case = build_step((1500, 1600, 1700, 1800, 1500, 1600, 1700, 1800), 4, identity=True, seed=29)
    case.topk[0, 5, :2] = case.topk[0, 5, :2].flip(0)
    output = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="identity_blocks promises"):
        with validating(op):
            _call(case, output, True)


def test_a_selected_step_the_lc8_kernel_declines_is_a_miss() -> None:
    """A selected table at the TP8 heads is the one-launch kernel's (modeling's SPARSE_DECODE_LC8): an output it
    cannot write (here a strided view) raises under require instead of silently taking the split-K chain."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8 import (  # noqa: E501
        modeling as dispatch,
    )

    case = build_step((9000, 40000), 4, identity=False, seed=31)
    wide = torch.empty(
        case.q.shape[0], NUM_HEADS, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    output = wide[..., :HEAD_DIM]
    assert not output.is_contiguous()
    dispatch.reset_logged()
    dispatch.set_require(True)
    try:
        with pytest.raises(dispatch.FastPathUnavailable, match="m3_sparse_decode_lc8"):
            _call(case, output, False)
    finally:
        dispatch.set_require(False)
