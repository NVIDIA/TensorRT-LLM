# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_attn_c1 entry (fused decode attention + o_proj GEMV in one 96-CTA launch)."""

import inspect
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_attn_c1 import (  # noqa: E501
    m3_attn_c1 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_fused_decode_attn import (  # noqa: E501
    mx_gemm,
    mxfp8_rows,
    unswizzle_scales,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.m3_attn_mega_c1 import (  # noqa: E501
    MegaWorkspace,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_fused_attn_step import (  # noqa: E402
    ATTN_ELEMENT_ULP,
    ATTN_RMS_ULP,
    attention_args,
    build_step,
    check_insert,
    decode_over_written_caches,
    mxfp8_operand,
)
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_attn_c1 requires a CUDA device"

QLEN = 6

# Attention and the cache insert: the bands of m3_fused_decode_attn (the same kernel code).
# o_proj GEMV, on the op's own MXFP8 hand-off: exact operands, fp32 split-K accumulation over 2048 products against
# an fp64 sum, one bf16 rounding on each side: at most one bf16 step apart, which is 2 of these ulps (2^-8 of the row
# scale) in the row maximum's binade. Measured: at most 1.12 ulps.
_GEMV_ELEMENT_ULP = 2.0
_GEMV_RMS_ULP = 0.5


def _args(s, ws, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    wo, wosf = mxfp8_operand(6144, 2048, 120, 122, g, 6144)
    out = torch.empty(s.num_tokens, 2048, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(s.num_tokens, 6144, dtype=torch.bfloat16, device="cuda")
    return [
        s.packed,
        ws.flags,
        *attention_args(s),
        1,
        1,
        out,
        ws.q8,
        ws.sf,
        True,
        wo,
        wosf,
        y,
        0,
        0,
        0,
        1,
        8,
        0,
    ]


def _arg(name):
    """Position of a parameter in the op's argument list (``is_valid`` mirrors ``__call__``)."""
    return list(inspect.signature(type(op).is_valid).parameters)[1:].index(name)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- the fixture from _validating
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    s = build_step([cell.spec["seq_len"]], QLEN, seed=seed)
    ws = MegaWorkspace(torch.device("cuda"))
    for _ in range(cell.spec["calls"]):
        args = _args(s, ws, seed + 1)
        out, y = args[_arg("output")], args[_arg("y")]
        wo, wosf = args[_arg("o_weight")], args[_arg("o_weight_scale")]
        parity = int(ws.flags[32]) & 1
        before = (s.kv_data.clone(), s.kv_scale.clone(), s.idx_k.clone())
        ref = op.reference(*args)
        op(*args)
        check_insert(s, before, ref)
        want = decode_over_written_caches(s, s.packed)
        assert_within_ulp(out.view(-1, 128), want.view(-1, 128), ATTN_ELEMENT_ULP, ATTN_RMS_ULP)
        # The o_proj units read the attention's MXFP8 hand-off, which is the merge's recipe on the bf16 output.
        q8 = ws.q8.view(2, 8, 2048)[parity, :QLEN]
        ex = unswizzle_scales(ws.sf.view(2, -1)[parity], QLEN, 64)
        q8_want, ex_want = mxfp8_rows(out)
        assert torch.equal(q8, q8_want) and torch.equal(ex, ex_want), "MXFP8 hand-off"
        y_staged = mx_gemm(q8, ex, wo.view(torch.uint8), unswizzle_scales(wosf, 6144, 64))
        assert_within_ulp(y.view(-1, 128), y_staged.view(-1, 128), _GEMV_ELEMENT_ULP, _GEMV_RMS_ULP)
        assert bool((ws.q8.view(2, -1)[parity ^ 1] == 255).all()), (
            "the next call's buffer is at rest"
        )
        assert int(ws.flags[32]) & 1 == parity ^ 1, "the call counter advanced"


def test_guard_refuses_a_handoff_not_at_rest() -> None:
    """The o_proj units poll the hand-off for codes other than 0xFF: stale codes read as this call's."""
    s = build_step([300], QLEN, seed=5)
    ws = MegaWorkspace(torch.device("cuda"))
    ws.q8.zero_()
    args = _args(s, ws, 6)
    with pytest.raises(AssertionError, match="not at rest"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_slot_the_block_table_does_not_give() -> None:
    s = build_step([300], QLEN, seed=5)
    args = _args(s, MegaWorkspace(torch.device("cuda")), 6)
    loc = s.out_cache_loc.clone()
    loc[2] = (loc[2] + 7) % (s.kv_data.shape[0] * 128)
    args[_arg("out_cache_loc")] = loc
    with pytest.raises(AssertionError, match="out_cache_loc must be the slot"):
        with validating(op):
            op(*args)


def test_op_rejects_another_query_length() -> None:
    """Compiled for one request of 6 query tokens (96 CTAs, one o_proj unit each) only."""
    s = build_step([300], 4, seed=5)
    args = _args(s, MegaWorkspace(torch.device("cuda")), 6)
    args[_arg("decode_query_len")] = 4
    with pytest.raises(RuntimeError, match="six query tokens"):
        op(*args)
