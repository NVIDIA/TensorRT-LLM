# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_attn_mega entry (QKV GEMV + fused decode attention + o_proj GEMV in one launch)."""

import inspect
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_attn_mega import (  # noqa: E501
    m3_attn_mega as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_fused_decode_attn import (  # noqa: E501
    mx_gemm,
    mxfp8_rows,
    reference_attention,
    unswizzle_scales,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.m3_attn_mega import (  # noqa: E501
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

assert torch.cuda.is_available(), "m3_attn_mega requires a CUDA device"

# Each stage is gated against its own reference fed the op's own input to that stage, so a band states one
# stage's error, not the propagation of an earlier stage's rounding through a quantizer.
#
# QKV GEMV and o_proj GEMV: exact MXFP8 operands, fp32 accumulation over 6144 / 2048 products in split-K partials
# against an fp64 sum, one bf16 rounding on each side: at most one bf16 step apart, which is 2 of these ulps (2^-8
# of the row scale) in the row maximum's binade. Measured: bit-exact but for a few elements (0.43 ulp at most).
_GEMV_ELEMENT_ULP = 2.0
_GEMV_RMS_ULP = 0.5
# Attention and the cache insert: the bands of m3_fused_decode_attn (the same kernel code), on the op's own rows.


def _operands(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xq, xsf = mxfp8_operand(1, 6144, 126, 128, g, 128)
    w, wsf = mxfp8_operand(2560, 6144, 119, 121, g, 2560)
    wo, wosf = mxfp8_operand(6144, 2048, 120, 122, g, 6144)
    return xq, xsf, w, wsf, wo, wosf


def _args(s, ops, ws):
    xq, xsf, w, wsf, wo, wosf = ops
    out = torch.empty(1, 2048, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(1, 6144, dtype=torch.bfloat16, device="cuda")
    return [
        xq,
        xsf,
        w,
        wsf,
        ws.qkv_out,
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
    s = build_step([cell.spec["seq_len"]], 1, seed=seed)
    ops = _operands(seed + 1)
    wo, wosf = ops[4], ops[5]
    ws = MegaWorkspace(torch.device("cuda"))
    for _ in range(cell.spec["calls"]):
        args = _args(s, ops, ws)
        out, y = args[_arg("output")], args[_arg("y")]
        parity = int(ws.flags[32]) & 1
        caches = (s.kv_data.clone(), s.kv_scale.clone(), s.idx_k.clone())
        ref = op.reference(*args)
        op(*args)
        packed = ws.qkv_out[parity, :1]
        assert_within_ulp(
            packed.view(-1, 128), ref.packed.view(-1, 128), _GEMV_ELEMENT_ULP, _GEMV_RMS_ULP
        )
        check_insert(s, caches, reference_attention(packed, *caches, *attention_args(s)[3:]))
        want = decode_over_written_caches(s, packed)
        assert_within_ulp(out.view(-1, 128), want.view(-1, 128), ATTN_ELEMENT_ULP, ATTN_RMS_ULP)
        # The o_proj units read the attention's MXFP8 hand-off, which is the merge's recipe on the bf16 output.
        q8 = ws.q8.view(2, 8, 2048)[parity, :1]
        ex = unswizzle_scales(ws.sf.view(2, -1)[parity], 1, 64)
        q8_want, ex_want = mxfp8_rows(out)
        assert torch.equal(q8, q8_want) and torch.equal(ex, ex_want), "MXFP8 hand-off"
        y_staged = mx_gemm(q8, ex, wo.view(torch.uint8), unswizzle_scales(wosf, 6144, 64))
        assert_within_ulp(y.view(-1, 128), y_staged.view(-1, 128), _GEMV_ELEMENT_ULP, _GEMV_RMS_ULP)
        # The next call's buffers are back at rest.
        assert bool((ws.qkv_out.view(torch.int16)[parity ^ 1] == -1).all())
        assert bool((ws.q8.view(2, -1)[parity ^ 1] == 255).all())
        assert int(ws.flags[32]) & 1 == parity ^ 1, "the call counter advanced"


def test_guard_refuses_a_workspace_not_at_rest() -> None:
    """The attention CTAs poll the QKV rows for the 0xFFFF marker: stale rows read as this call's."""
    s = build_step([300], 1, seed=5)
    ws = MegaWorkspace(torch.device("cuda"))
    ws.qkv_out.zero_()
    args = _args(s, _operands(6), ws)
    with pytest.raises(AssertionError, match="not at rest"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_request_past_the_identity_window() -> None:
    s = build_step([300], 1, seed=5)
    args = _args(s, _operands(6), MegaWorkspace(torch.device("cuda")))
    seq_lens = s.seq_lens.clone()
    seq_lens[0] = 2049
    args[_arg("seq_lens")] = seq_lens
    with pytest.raises(AssertionError, match="identity selection covers 16 blocks"):
        with validating(op):
            op(*args)


def test_op_rejects_more_than_one_token() -> None:
    """Compiled for one request with one query token only."""
    s = build_step([300, 400], 1, seed=5)
    args = _args(s, _operands(6), MegaWorkspace(torch.device("cuda")))
    args[_arg("xq")] = torch.cat([args[_arg("xq")]] * 2)
    args[_arg("output")] = torch.empty(2, 2048, dtype=torch.bfloat16, device="cuda")
    args[_arg("y")] = torch.empty(2, 6144, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="one request with one query token"):
        op(*args)
