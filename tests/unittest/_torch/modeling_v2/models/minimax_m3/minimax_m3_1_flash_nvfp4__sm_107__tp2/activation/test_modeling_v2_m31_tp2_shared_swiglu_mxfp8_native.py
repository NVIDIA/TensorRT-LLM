# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWQ_CPP at TP2: the exact native shared-expert SwiGLU + MXFP8 producer with measured-row dispatch.

With finite_code0 (the TP2 call) the op is bit for bit the Triton m3_shared_swiglu_mxfp8 on every q and scale byte,
padding included, scale-code-0 blocks too; without it, bit for bit the stock silu_and_mul + mxfp8_quantize pair on every
byte. The dispatch takes every SWQ row (17..32) by default; TRTLLM_M31_DISABLE_SHARED_SWIGLU_MXFP8_NATIVE=1 (or the
native glue switch TRTLLM_M31_DISABLE_GEN_NATIVE_GLUE=1) uses Triton."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.activation.m3_shared_swiglu_mxfp8 import (
        m3_shared_swiglu_mxfp8,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops

OAI = (7.0, 1.702, 1.0)
U8 = torch.uint8


def _op():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs an SM100-family GPU")
    return family_ops().minimax_m31_swiglu_oai_mxfp8_decode


def _inputs(rows, inter, seed, scale):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * inter, generator=g, device="cuda") * scale
    if rows >= 4:
        x[1, :inter] = x[1, :inter].abs() + OAI[0]  # gates past the clamp
        x[2] = 0.0
        x[3] = -0.0
    return x.to(torch.bfloat16).contiguous()


def _run(op, x, finite_code0):
    rows, inter = x.shape[0], x.shape[1] // 2
    q = torch.full((rows, inter), 0x7F, dtype=U8, device="cuda").view(torch.float8_e4m3fn)
    sf = torch.full(((rows + 127) // 128 * 128 * inter // 32,), 0xAB, dtype=U8, device="cuda")
    op(x, q, sf, finite_code0)
    return q.view(U8), sf


@pytest.mark.parametrize("inter", [256, 1536])
@pytest.mark.parametrize("rows", [1, 17, 129])
@pytest.mark.parametrize("scale", [12.0])
def test_bitwise_vs_triton_swq_and_stock_pair(rows, inter, scale) -> None:
    op = _op()
    x = _inputs(rows, inter, rows * 131 + inter + int(scale * 7), scale)
    q, sf = _run(op, x, True)
    tq, tsf = m3_shared_swiglu_mxfp8(x, *OAI)
    assert torch.equal(q, tq.view(U8)), "q bytes = the Triton swq"
    assert torch.equal(sf, tsf.view(U8).flatten()), "scale bytes (padding zeroed) = the Triton swq"
    sq, ssf = _run(op, x, False)
    h = torch.ops.trtllm.silu_and_mul(
        x, swiglu_limit=OAI[0], swiglu_alpha=OAI[1], swiglu_beta=OAI[2]
    )
    rq, rsf = torch.ops.trtllm.mxfp8_quantize(h, True)
    assert torch.equal(sq, rq.view(U8).view(rows, inter)), "q bytes = the stock pair"
    assert torch.equal(ssf, rsf.view(U8).flatten()), "scale bytes = the stock pair"


@pytest.mark.parametrize("rows", [17, 32])
def test_scale_code0_blocks_follow_the_triton_swq(rows) -> None:
    """Planted tiny blocks (scale code 0 and its neighbours): finite_code0 is the Triton swq on every byte."""
    op = _op()
    inter, nb = 1536, 48
    g = torch.Generator(device="cuda").manual_seed(151 + rows)
    gate = torch.randn(rows, inter, generator=g, device="cuda")
    up = torch.randn(rows, inter, generator=g, device="cuda")
    pattern = torch.tensor([1.0, -0.75, 0.5, -0.25, 0.375, 0.0, -1.0, 0.625] * 4, device="cuda")
    for r in range(rows):
        for j, c in enumerate([2.0**-130, 2.0**-127, 448 * 2.0**-128, 448 * 2.0**-127, 2.0**-118]):
            blk = (r * 5 + j * 3) % nb
            gate[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
            up[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
    x = torch.cat([gate, up], 1).to(torch.bfloat16)
    q, sf = _run(op, x, True)
    tq, tsf = m3_shared_swiglu_mxfp8(x, *OAI)
    assert torch.equal(q, tq.view(U8)) and torch.equal(sf, tsf.view(U8).flatten())
