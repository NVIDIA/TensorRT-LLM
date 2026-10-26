# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""minimax_m31_ctx_residual_rmsnorm_mxfp8_out (the TP2 CTXNORMQ boundary writing into caller buffers, e.g. the owned
rows of the SP gather views) is bit for bit the allocating ops it replaces.

The out= op launches the same kernel as minimax_m31_ctx_residual_rmsnorm_mxfp8 (write_norm on / off) and its _f32 fold
with caller output pointers. Every output (norm, MXFP8 + its swizzled scales, NVFP4 + its linear scales, the FP32 norm)
and the in-place residual update must equal the allocating op's bytes, and no byte outside the destination rows may
change. GPU (an sm_10x build carrying the ops).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401

H = 6144
EPS = 1e-6
SENT = 0xA5

assert torch.cuda.is_available(), "the CTXNORMQ out= op requires a CUDA device"
_ops = torch.ops.trtllm
pytestmark = pytest.mark.skipif(
    not hasattr(_ops, "minimax_m31_ctx_residual_rmsnorm_mxfp8_out"),
    reason="the CTXNORMQ out= op is not built",
)


def _bf16(rows, cols, kind, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda", generator=gen)
    if kind == "stress":
        # 2^-12..2^12 per 32-element block: every MXFP8 / NVFP4 scale exponent the boundary meets, and then some
        e = torch.randint(-12, 13, (rows, cols // 32, 1), device="cuda", generator=gen).float()
        x = (x.view(rows, cols // 32, 32) * torch.exp2(e)).view(rows, cols)
    return x.to(torch.bfloat16)


def _u8(t):
    return t.contiguous().view(torch.uint8).reshape(-1)


def _sentinel(shape, dtype):
    return torch.full(shape, SENT, dtype=torch.uint8, device="cuda").view(dtype)


def _reference(x, r, w, gs, f32, write_norm):
    inter = x.clone()
    if f32:
        out = _ops.minimax_m31_ctx_residual_rmsnorm_mxfp8_f32(inter, r, w, EPS, gs)
    else:
        out = _ops.minimax_m31_ctx_residual_rmsnorm_mxfp8(inter, r, w, EPS, write_norm, gs)
    return inter, out


@pytest.mark.parametrize("T", [6144, 8192, 16384])
@pytest.mark.parametrize("kind", ["randn", "stress"])
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("f32", [False, True])
@pytest.mark.parametrize("half", [0, 1])
def test_out_into_the_owned_gather_rows(T, kind, fp4, f32, half):
    n = T // 2
    rows = slice(half * n, (half + 1) * n)
    x, r = _bf16(n, H, kind, T + 3 * half + fp4), _bf16(n, H, "randn", T + 7)
    w = (torch.rand(H, device="cuda") + 0.5).to(torch.bfloat16)
    gs = torch.tensor([448.0 * 6.0 / 64.0], device="cuda") if fp4 else None
    ref_inter, ref = _reference(x, r, w, gs, f32, True)
    # full-T gather-style destinations prefilled with a sentinel; the op writes the owned rows only
    norm = _sentinel((T, H * 2), torch.uint8).view(torch.bfloat16)
    q = _sentinel((T, H), torch.uint8)
    per = _u8(ref[2]).numel()
    sf = _sentinel((2 * per,), torch.uint8)
    fq = _sentinel((T, H // 2), torch.uint8)
    fsf = _sentinel((T, H // 16), torch.uint8)
    f32o = _sentinel((T, H * 4), torch.uint8).view(torch.float32)
    inter = x.clone()
    _ops.minimax_m31_ctx_residual_rmsnorm_mxfp8_out(
        inter,
        r,
        w,
        EPS,
        norm[rows],
        q[rows],
        sf[half * per : (half + 1) * per],
        fq[rows] if fp4 else None,
        fsf[rows] if fp4 else None,
        gs,
        f32o[rows] if f32 else None,
    )
    torch.cuda.synchronize()
    assert torch.equal(_u8(inter), _u8(ref_inter)), "in-place residual update"
    pairs = [("norm_out", norm[rows], ref[0]), ("mxfp8", q[rows], ref[1])]
    pairs.append(("sf", sf[half * per : (half + 1) * per], ref[2]))
    if fp4:
        pairs += [("fp4", fq[rows], ref[3]), ("fp4_sf", fsf[rows], ref[4])]
    if f32:
        pairs.append(("f32", f32o[rows], ref[5]))
    for name, got, want in pairs:
        assert torch.equal(_u8(got), _u8(want)), f"{name} differs from the allocating op"
    peer = slice((1 - half) * n, (2 - half) * n)
    untouched = [_u8(norm[peer]), _u8(q[peer]), _u8(sf[(1 - half) * per : (2 - half) * per])]
    untouched += [_u8(fq[peer]), _u8(fsf[peer])] if fp4 else [_u8(fq), _u8(fsf)]
    untouched.append(_u8(f32o[peer]) if f32 else _u8(f32o))
    assert all(bool((u == SENT).all()) for u in untouched), "the op wrote outside the owned rows"


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 4097])
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("write_norm", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_out_edge_rows_and_norm_off(rows, fp4, write_norm, pdl):
    """Partial 128-row scale tiles, no norm output (norm_out None) and the programmatic launch."""
    x, r = _bf16(rows, H, "stress", rows), _bf16(rows, H, "randn", rows + 1)
    w = (torch.rand(H, device="cuda") + 0.5).to(torch.bfloat16)
    gs = torch.tensor([448.0 * 6.0 / 64.0], device="cuda") if fp4 else None
    ref_inter, ref = _reference(x, r, w, gs, False, write_norm)
    norm = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda") if write_norm else None
    q = torch.empty(rows, H, dtype=torch.uint8, device="cuda")
    sf = _sentinel((_u8(ref[2]).numel(),), torch.uint8)
    fq = torch.empty(rows, H // 2, dtype=torch.uint8, device="cuda") if fp4 else None
    fsf = torch.empty(rows, H // 16, dtype=torch.uint8, device="cuda") if fp4 else None
    inter = x.clone()
    _ops.minimax_m31_ctx_residual_rmsnorm_mxfp8_out(
        inter, r, w, EPS, norm, q, sf, fq, fsf, gs, None, pdl
    )
    torch.cuda.synchronize()
    assert torch.equal(_u8(inter), _u8(ref_inter)), "in-place residual update"
    if write_norm:
        assert torch.equal(_u8(norm), _u8(ref[0])), "norm_out differs"
    assert torch.equal(_u8(q), _u8(ref[1])), "mxfp8 differs"
    assert torch.equal(_u8(sf), _u8(ref[2])), "mxfp8 scales differ (incl. the padded tile rows)"
    if fp4:
        assert torch.equal(_u8(fq), _u8(ref[3])) and torch.equal(_u8(fsf), _u8(ref[4])), (
            "nvfp4 differs"
        )
