# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_swq entry of the MiniMax-M3.1 TP8 target (shared expert intermediate 384): SwiGLU-OAI
written as MXFP8, bitwise to silu_and_mul + mxfp8_quantize."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.activation.m3_swq import (  # noqa: E501
    m3_swq as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_swq requires a CUDA device"

OAI = (7.0, 1.702, 1.0)


def _assert_bitwise(x):
    q, sf = op(x, *OAI)
    rq, rsf = op.reference(x, *OAI)
    assert q.shape == rq.shape and q.dtype == torch.float8_e4m3fn
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert sf.numel() <= rsf.numel() and torch.equal(sf, rsf.view(-1)[: sf.numel()])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = (torch.randn(cell.spec["m"], 2 * cell.spec["inter"], generator=g, device="cuda") * 4.0).to(
        torch.bfloat16
    )
    with validating(op):
        _assert_bitwise(x)


@pytest.mark.parametrize("m", [9, 16, 32, 48])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_bitwise_across_magnitudes_and_edges(m, scale) -> None:
    """Scaled inputs, gates past the clamp, an all-zero row and a -0.0 row: q bytes and every scale byte."""
    g = torch.Generator(device="cuda").manual_seed(m * 1000 + int(scale * 10) % 997)
    x = torch.randn(m, 2 * 384, generator=g, device="cuda") * scale
    x[1, :384] = x[1, :384].abs() + OAI[0]
    x[2] = 0.0
    x[3] = -0.0
    _assert_bitwise(x.to(torch.bfloat16))


def _scale_bytes(sf, m, nb):
    """Un-swizzle the 128x4 scale layout to [m, nb] bytes."""
    nbp = (nb + 3) // 4 * 4
    r = torch.arange(m, device=sf.device).view(-1, 1)
    c = torch.arange(nb, device=sf.device).view(1, -1)
    idx = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * nbp)
    )
    return sf.view(-1)[idx]


@pytest.mark.parametrize("m", [9, 48])
def test_tiny_amax_blocks_stay_finite(m) -> None:
    """A block whose scale code is 0 (amax / 448 <= 2^-127) is native's inf/NaN; SWQ keeps native's scale
    byte, writes data quantized with the exact reciprocal 2^127, and stays bitwise to native on every other block."""
    g = torch.Generator(device="cuda").manual_seed(151 + m)
    gate = torch.randn(m, 384, generator=g, device="cuda")
    up = torch.randn(m, 384, generator=g, device="cuda")
    pattern = torch.tensor([1.0, -0.75, 0.5, -0.25, 0.375, 0.0, -1.0, 0.625] * 4, device="cuda")
    for r in range(m):
        for j, c in enumerate([2.0**-130, 2.0**-127, 448 * 2.0**-128, 448 * 2.0**-127, 2.0**-118]):
            blk = (r * 5 + j * 3) % 12
            gate[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
            up[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
    x = torch.cat([gate, up], 1).to(torch.bfloat16)
    q, sf = op(x, *OAI)
    rq, rsf = op.reference(x, *OAI)
    h = torch.ops.trtllm.silu_and_mul(
        x, swiglu_limit=OAI[0], swiglu_alpha=OAI[1], swiglu_beta=OAI[2]
    ).float()
    byte, rbyte = _scale_bytes(sf, m, 12), _scale_bytes(rsf, m, 12)
    assert torch.equal(byte, rbyte), "scale bytes are native's on every block"
    deq = q.float().view(m, 12, 32) * torch.exp2(byte.float() - 127.0).unsqueeze(-1)
    rdeq = rq.float().view(m, 12, 32) * torch.exp2(rbyte.float() - 127.0).unsqueeze(-1)
    native_finite = torch.isfinite(rdeq).all(-1)
    code0 = (byte == 0) & (h.view(m, 12, 32).abs().amax(-1) != 0)
    assert bool(code0.any()), "the planted blocks reach scale code 0"
    assert not bool(native_finite[code0].any()), (
        "native is inf/NaN on code-0 blocks (the bug this guards)"
    )
    assert bool(torch.isfinite(deq).all()), "SWQ is finite everywhere"
    qb, rqb = q.view(torch.uint8).view(m, 12, 32), rq.view(torch.uint8).view(m, 12, 32)
    assert bool((qb == rqb).all(-1)[native_finite].all()), (
        "bitwise to native wherever native is finite"
    )
    exact = (h.view(m, 12, 32) * 2.0**127).to(torch.float8_e4m3fn).view(torch.uint8)
    assert bool((qb == exact).all(-1)[code0].all()), "code-0 data = RN-e4m3(h * 2^127)"
