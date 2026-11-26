# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP8 shared-expert SwiGLU-OAI + MXFP8 producer as mv2_minimax_m3::minimax_m31_swiglu_oai_mxfp8_decode.

Bitwise oracle. With finite_code0=True (the TP8 call) every q and scale byte, padding included, is the Triton
m3_shared_swiglu_mxfp8's, scale-code-0 blocks too. With finite_code0=False every byte is the in-tree silu_and_mul +
mxfp8_quantize(h, True) pair's. Covered: the TP8 width 384 and 416 (K32-block padding), rows at the 128-row scale
padding edge, magnitudes 1e-3 and 1e3, gates past the clamp, zero and -0.0 rows, planted scale-code-0 blocks, and
CUDA-graph replay of refreshed inputs."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.activation.m3_shared_swiglu_mxfp8 import (
    m3_shared_swiglu_mxfp8,
)

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 TP8 target runs on SM107 only", allow_module_level=True)

OAI = (7.0, 1.702, 1.0)
ROWS = [1, 128, 129]  # one row, the 128-row scale tile, one row past it


def _op(x, finite_code0=True):
    rows, d = x.shape[0], x.shape[1] // 2
    nb_pad = (d // 32 + 3) // 4 * 4
    q = torch.empty(rows, d, dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.full(((rows + 127) // 128 * 128 * nb_pad,), 0xA5, dtype=torch.uint8, device=x.device)
    family_ops().minimax_m31_swiglu_oai_mxfp8_decode(x, q, sf, finite_code0)
    return q, sf


def _native(x):
    h = torch.ops.trtllm.silu_and_mul(
        x.contiguous(), swiglu_limit=OAI[0], swiglu_alpha=OAI[1], swiglu_beta=OAI[2]
    )
    q, sf = torch.ops.trtllm.mxfp8_quantize(h, True)
    return q.view(h.shape), sf.view(-1), h


def _assert_bytes(q, sf, rq, rsf, what):
    assert q.shape == rq.shape, what
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8)), f"{what}: q bytes"
    assert sf.numel() <= rsf.numel(), what
    assert torch.equal(sf, rsf.view(-1)[: sf.numel()]), f"{what}: scale bytes (padding included)"


def _inputs(rows, d, seed, scale):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * d, generator=g, device="cuda") * scale
    if rows > 3:
        x[1, :d] = x[1, :d].abs() + OAI[0]
        x[2] = 0.0
        x[3] = -0.0
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("d", [384, 416])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_bitwise_vs_triton_swq_and_native(d, rows, scale) -> None:
    x = _inputs(rows, d, rows * 7919 + d + int(scale * 10), scale)
    q, sf = _op(x, True)
    rq, rsf = m3_shared_swiglu_mxfp8(x, *OAI)
    _assert_bytes(q, sf, rq, rsf, f"finite_code0 vs m3_shared_swiglu_mxfp8 rows={rows} d={d}")
    q0, sf0 = _op(x, False)
    nq, nsf, _ = _native(x)
    _assert_bytes(
        q0, sf0, nq, nsf, f"native mode vs silu_and_mul + mxfp8_quantize rows={rows} d={d}"
    )


def _unswizzle(sf, m, nb):
    nbp = (nb + 3) // 4 * 4
    r = torch.arange(m, device=sf.device).view(-1, 1)
    c = torch.arange(nb, device=sf.device).view(1, -1)
    idx = (
        (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * (128 * nbp)
    )
    return sf.view(-1)[idx]


@pytest.mark.parametrize("m", [9, 48])
def test_scale_code0_blocks(m) -> None:
    """Planted blocks at and around scale code 0: finite_code0 is m3_shared_swiglu_mxfp8's bytes (RN-e4m3(h * 2^127));
    the native mode
    is the in-tree pair's bytes (zero data) wherever that pair is finite."""
    g = torch.Generator(device="cuda").manual_seed(151 + m)
    gate = torch.randn(m, 384, generator=g, device="cuda")
    up = torch.randn(m, 384, generator=g, device="cuda")
    pattern = torch.tensor([1.0, -0.75, 0.5, -0.25, 0.375, 0.0, -1.0, 0.625] * 4, device="cuda")
    for r in range(m):
        for j, c in enumerate([2.0**-130, 2.0**-127, 448 * 2.0**-128, 448 * 2.0**-127, 2.0**-118]):
            blk = (r * 5 + j * 2) % 12  # 5 distinct blocks per row (j * 3 put j = 4 on j = 0)
            gate[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
            up[r, blk * 32 : (blk + 1) * 32] = pattern * (2 * c)
    x = torch.cat([gate, up], 1).to(torch.bfloat16)
    q, sf = _op(x, True)
    rq, rsf = m3_shared_swiglu_mxfp8(x, *OAI)
    _assert_bytes(q, sf, rq, rsf, "code-0 finite mode vs m3_shared_swiglu_mxfp8")
    nq, nsf, h = _native(x)
    byte = _unswizzle(sf, m, 12)
    code0 = (byte == 0) & (h.float().view(m, 12, 32).abs().amax(-1) != 0)
    assert bool(code0.any()), "the planted blocks reach scale code 0"
    q0, sf0 = _op(x, False)
    assert torch.equal(sf0, sf), "both modes publish the same scale bytes"
    ndeq = nq.float().view(m, 12, 32) * torch.exp2(
        _unswizzle(nsf, m, 12).float() - 127.0
    ).unsqueeze(-1)
    native_finite = torch.isfinite(ndeq).all(-1)
    qb0, nqb = q0.view(torch.uint8).view(m, 12, 32), nq.view(torch.uint8).view(m, 12, 32)
    assert bool((qb0 == nqb).all(-1)[native_finite].all()), (
        "native mode = the pair wherever it is finite"
    )
    # The pair's mxfp8_quantize scales by 0 on a code-0 block, so a negative element is E4M3 -0 (0x80): zero data
    # means a zero magnitude, sign kept, as the pair check above pins on finite blocks.
    assert bool(((qb0[code0] & 0x7F) == 0).all()), "native mode: code-0 data is (signed) zero"
    qb = q.view(torch.uint8).view(m, 12, 32)
    assert bool((qb == qb0).all(-1)[~code0].all()), "the modes differ only on code-0 blocks"


@pytest.mark.parametrize("rows", [9, 32])
def test_graph_replay_refreshed_inputs(rows) -> None:
    x = _inputs(rows, 384, 5, 1.0)
    static = x.clone()
    _op(static, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, sf = _op(static, True)
    for seed in (11, 12, 13):
        fresh = _inputs(rows, 384, seed, 10.0 ** (seed - 12))
        static.copy_(fresh)
        graph.replay()
        rq, rsf = m3_shared_swiglu_mxfp8(fresh, *OAI)
        _assert_bytes(q, sf, rq, rsf, f"graph replay seed={seed}")
