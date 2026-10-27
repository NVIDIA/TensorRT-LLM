# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWQ at TP2 (the sm_107 TP8 lever, ported): the m3_swq entry writes the shared expert's SwiGLU-OAI as the down
projection's MXFP8 activation, bitwise to silu_and_mul + mxfp8_quantize, at the TP2 shared width 1536 (the cells of
17 and 32 rows: the rows the shared down's GEMV does not serve) and at the TP8 width 384 it was certified on; the
The scale-code-0 exception keeps the code-0 blocks finite. The dispatch takes it on SHARED_DOWN (6144, 1536) at
17..32 rows; TRTLLM_M31_DISABLE_SWQ=1 empties SWQ_ROWS."""

import importlib
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    m31_switches,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.activation.m3_swq import (  # noqa: E501
    m3_swq as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

OAI = (7.0, 1.702, 1.0)
_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)


def test_swq_rows_default_and_kill(monkeypatch) -> None:
    """CPU: the shared down at the TP2 width, 17..32 rows, none of them a GEMV row; the kill switch empties it."""
    assert modeling.SWQ is True
    assert modeling.SWQ_SHARED_DOWN == (6144, 1536) == modeling.SHARED_DOWN
    assert modeling.SWQ_ROWS == {(6144, 1536): frozenset(range(17, 33))}
    assert not modeling.SWQ_ROWS[(6144, 1536)] & modeling.MX_GEMV_ROWS[modeling.SHARED_DOWN]
    monkeypatch.setenv("TRTLLM_M31_DISABLE_SWQ", "1")
    try:
        importlib.reload(m31_switches)
        killed = importlib.reload(modeling)
        assert killed.SWQ is False and killed.SWQ_ROWS == {}
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_SWQ", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)


def test_the_cells_cover_the_tp2_rows() -> None:
    """CPU: the entry's cells include the TP2 width at the first and last SWQ row."""
    cells = {(c.spec["m"], c.spec["inter"]) for c in op.CELLS}
    assert {(17, 1536), (32, 1536)} <= cells


def _assert_bitwise(x):
    q, sf = op(x, *OAI)
    rq, rsf = op.reference(x, *OAI)
    assert q.shape == rq.shape and q.dtype == torch.float8_e4m3fn
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert sf.numel() <= rsf.numel() and torch.equal(sf, rsf.view(-1)[: sf.numel()])


@_gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = (torch.randn(cell.spec["m"], 2 * cell.spec["inter"], generator=g, device="cuda") * 4.0).to(
        torch.bfloat16
    )
    with validating(op):
        _assert_bitwise(x)


@_gpu
@pytest.mark.parametrize("inter", [1536, 384])
@pytest.mark.parametrize("m", [17, 24, 32])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_bitwise_across_magnitudes_and_edges(inter, m, scale) -> None:
    """Scaled inputs, gates past the clamp, an all-zero row and a -0.0 row: q bytes and every scale byte."""
    g = torch.Generator(device="cuda").manual_seed(m * 1000 + inter + int(scale * 10) % 997)
    x = torch.randn(m, 2 * inter, generator=g, device="cuda") * scale
    x[1, :inter] = x[1, :inter].abs() + OAI[0]
    x[2] = 0.0
    x[3] = -0.0
    _assert_bitwise(x.to(torch.bfloat16))


@_gpu
@pytest.mark.parametrize("m", [17, 32])
def test_bitwise_under_block_scale_stress(m) -> None:
    """Every 32-element block of gate and up scaled by its own 2^-12..2^+12 at the TP2 width: q and scale bytes."""
    inter = 1536
    g = torch.Generator(device="cuda").manual_seed(4242 + m)
    x = torch.randn(m, 2 * inter, generator=g, device="cuda")
    block = torch.randint(-12, 13, (m, 2 * inter // 32), generator=g, device="cuda").float()
    x = x * torch.exp2(block).repeat_interleave(32, dim=1)
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


@_gpu
@pytest.mark.parametrize("m", [17, 32])
def test_tiny_amax_blocks_stay_finite(m) -> None:
    """Scale code 0 at TP2 width: a block whose scale code is 0 (amax / 448 <= 2^-127) is native's inf/NaN; SWQ keeps
    native's scale byte, writes data quantized with the exact reciprocal 2^127, and stays bitwise to native on every
    other block."""
    inter = 1536
    nb = inter // 32
    g = torch.Generator(device="cuda").manual_seed(151 + m)
    gate = torch.randn(m, inter, generator=g, device="cuda")
    up = torch.randn(m, inter, generator=g, device="cuda")
    pattern = torch.tensor([1.0, -0.75, 0.5, -0.25, 0.375, 0.0, -1.0, 0.625] * 4, device="cuda")
    for r in range(m):
        for j, c in enumerate([2.0**-130, 2.0**-127, 448 * 2.0**-128, 448 * 2.0**-127, 2.0**-118]):
            blk = (r * 5 + j * 3) % nb
            gate[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
            up[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
    x = torch.cat([gate, up], 1).to(torch.bfloat16)
    q, sf = op(x, *OAI)
    rq, rsf = op.reference(x, *OAI)
    h = torch.ops.trtllm.silu_and_mul(
        x, swiglu_limit=OAI[0], swiglu_alpha=OAI[1], swiglu_beta=OAI[2]
    ).float()
    byte, rbyte = _scale_bytes(sf, m, nb), _scale_bytes(rsf, m, nb)
    assert torch.equal(byte, rbyte), "scale bytes are native's on every block"
    deq = q.float().view(m, nb, 32) * torch.exp2(byte.float() - 127.0).unsqueeze(-1)
    rdeq = rq.float().view(m, nb, 32) * torch.exp2(rbyte.float() - 127.0).unsqueeze(-1)
    native_finite = torch.isfinite(rdeq).all(-1)
    code0 = (byte == 0) & (h.view(m, nb, 32).abs().amax(-1) != 0)
    assert bool(code0.any()), "the planted blocks reach scale code 0"
    assert not bool(native_finite[code0].any()), (
        "native is inf/NaN on code-0 blocks (the bug this guards)"
    )
    assert bool(torch.isfinite(deq).all()), "SWQ is finite everywhere"
    qb, rqb = q.view(torch.uint8).view(m, nb, 32), rq.view(torch.uint8).view(m, nb, 32)
    assert bool((qb == rqb).all(-1)[native_finite].all()), (
        "bitwise to native wherever native is finite"
    )
    exact = (h.view(m, nb, 32) * 2.0**127).to(torch.float8_e4m3fn).view(torch.uint8)
    assert bool((qb == exact).all(-1)[code0].all()), "code-0 data = RN-e4m3(h * 2^127)"
