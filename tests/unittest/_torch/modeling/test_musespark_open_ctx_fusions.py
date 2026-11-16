# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise checks for the classic Muse Spark Open model's MUSESPARK_OPEN_CTX_FUSIONS paths."""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_V2 = "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep"


def _same_bits(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[a.element_size()]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


@pytest.mark.parametrize("rope", [False, True])
def test_gate_view_matches_gate_copy(rope):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.rmsnorm_gate import rmsnorm_gate
    from tensorrt_llm._torch.models import modeling_musespark_open as mso

    v2_prep = importlib.import_module(f"{_V2}.ops.attention.preattn_prep").preattn_prep
    g = torch.Generator(device="cuda").manual_seed(3)
    rows, nq, nkv = 2048, 128, 16
    qkv = (torch.randn((rows, 2 * (nq + nkv) * 64), device="cuda", generator=g) * 3).to(
        torch.bfloat16
    )
    cos_sin = torch.randn((1 << 16, 2, 32), device="cuda", generator=g).float() if rope else None
    pos = (
        torch.randint(0, 1 << 16, (rows,), device="cuda", generator=g, dtype=torch.int32)
        if rope
        else None
    )
    attn = (torch.randn((rows, nq * 64), device="cuda", generator=g) * 2).to(torch.bfloat16)
    args = (qkv, cos_sin, pos, nq, nkv, 1e-5, 5.47, False)
    q0, g0 = v2_prep(*args, enable_pdl=True)
    q1, g1 = mso._vectorized_preattn_prep(v2_prep)(*args, enable_pdl=True)
    assert _same_bits(q0, q1)
    assert _same_bits(rmsnorm_gate(attn, g0, 1e-5), rmsnorm_gate(attn, g1, 1e-5))


def test_capture_join_matches_cast_and_slice_copy():
    from tensorrt_llm._torch.custom_ops import inplace_slice_copy
    from tensorrt_llm._torch.models import modeling_musespark_open as mso
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    g = torch.Generator(device="cuda").manual_seed(4)
    rows, hidden = 1024, 8192
    join = mso.MuseSparkOpenGatedResidual(hidden, torch.bfloat16).cuda()
    with torch.no_grad():
        join.gate.copy_(
            (torch.randn((hidden,), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        )
    join.cache_derived_state()
    norm = RMSNorm(hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
    with torch.no_grad():
        norm.weight.copy_(
            (torch.randn((hidden,), device="cuda", generator=g) * 0.1).to(torch.bfloat16)
        )
    gain = ((torch.randn((hidden,), device="cuda", generator=g) * 0.1) + 1.0).to(torch.bfloat16)
    residual = torch.randn((rows, hidden), device="cuda", generator=g) * 40
    branch = (torch.randn((rows, hidden), device="cuda", generator=g) * 2).to(torch.bfloat16)
    buf0 = torch.zeros((rows + 7, 5 * hidden), dtype=torch.bfloat16, device="cuda")
    buf1 = torch.zeros_like(buf0)
    u0, n0 = join._add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm, gain)
    inplace_slice_copy(buf0, u0.to(torch.bfloat16), 3 * hidden, 4 * hidden)
    u1, n1, captured = join._add_after_rmsnorm_and_norm(
        residual, branch, 1e-5, norm, gain, capture_out=buf1[:rows, 3 * hidden : 4 * hidden]
    )
    assert captured
    assert _same_bits(u0, u1) and _same_bits(n0, n1) and _same_bits(buf0, buf1)


@pytest.mark.parametrize("kind", ["normal", "ties", "equal"])
@pytest.mark.parametrize("rows", [600, 4096])
def test_top8_routing_matches_sorted_routing(kind, rows):
    from tensorrt_llm._torch.models import modeling_musespark_open as mso

    g = torch.Generator(device="cuda").manual_seed(rows)
    logits = torch.randn((rows, 256), device="cuda", generator=g) * 2
    bias = torch.randn((256,), device="cuda", generator=g) * 0.1
    if kind == "ties":
        logits, bias = torch.round(logits), torch.zeros_like(bias)
    elif kind == "equal":
        logits[:], bias[:] = 0.5, 0.0
    out = []
    for kernel, extra in (
        (mso._full_routing_kernel, {"TORCH_ORDER": True}),
        (mso._top8_routing_kernel, {}),
    ):
        sel = torch.empty((rows, 8), device="cuda", dtype=torch.int32)
        wts = torch.empty((rows, 8), device="cuda", dtype=torch.float32)
        kernel[(rows,)](
            logits, bias, sel, wts, 1e-20, EXPERTS=256, num_warps=1 if rows >= 512 else 4, **extra
        )
        out.append((sel, wts))
    assert torch.equal(out[0][0], out[1][0]) and _same_bits(out[0][1], out[1][1])


@pytest.mark.parametrize("fused", [False, True])
def test_join_with_router_input_matches_join_and_cast(fused, monkeypatch):
    """The join hands its MoE the router's FP32 input; with the switch on both joins run row-batched.

    1001 rows: a partial last program.
    """
    from tensorrt_llm._torch.models import modeling_musespark_open as mso
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    g = torch.Generator(device="cuda").manual_seed(1)
    rows, hidden = 1001, 8192
    join = mso.MuseSparkOpenGatedResidual(hidden, torch.bfloat16).cuda()
    with torch.no_grad():
        join.gate.copy_(
            (torch.randn((hidden,), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        )
    join.cache_derived_state()
    norm = RMSNorm(hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
    with torch.no_grad():
        norm.weight.copy_(
            (torch.randn((hidden,), device="cuda", generator=g) * 0.1).to(torch.bfloat16)
        )
    residual = torch.randn((rows, hidden), device="cuda", generator=g) * 4
    branch = (torch.randn((rows, hidden), device="cuda", generator=g) * 2).to(torch.bfloat16)
    monkeypatch.setattr(mso, "_CTX_FUSIONS", False)
    updated, ffn_input = join._add_after_rmsnorm_and_norm(residual.clone(), branch, 1e-5, norm)
    monkeypatch.setattr(mso, "_CTX_FUSIONS", fused)
    moe = SimpleNamespace(ctx_router_input=None)
    join._router_input_moe = moe
    joined = join._add_after_rmsnorm_and_norm(residual.clone(), branch, 1e-5, norm)
    join._router_input_moe = None
    assert moe.ctx_router_input[0] is joined[1]
    assert _same_bits(updated, joined[0]) and _same_bits(ffn_input, joined[1])
    assert _same_bits(ffn_input.to(torch.float32), moe.ctx_router_input[1])
    rows_updated, rows_input = join._add_after_rmsnorm_and_norm(
        residual.clone(), branch, 1e-5, norm
    )
    assert _same_bits(updated, rows_updated) and _same_bits(ffn_input, rows_input)


@pytest.mark.parametrize("heads", [(128, 16), (32, 4)])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("rows", [600, 2048])
def test_vectorized_context_prep_matches_modeling_v2(rows, rope, pdl, heads):
    from tensorrt_llm._torch.models import modeling_musespark_open as mso

    v2_prep = importlib.import_module(f"{_V2}.ops.attention.preattn_prep").preattn_prep
    num_q, num_kv = heads
    g = torch.Generator(device="cuda").manual_seed(2)
    qkv = torch.randn((rows, 2 * (num_q + num_kv) * 64), device="cuda", generator=g).to(
        torch.bfloat16
    )
    cos_sin = torch.randn((1 << 16, 2, 32), device="cuda", generator=g).float() if rope else None
    positions = (
        torch.randint(0, 1 << 16, (rows,), device="cuda", generator=g, dtype=torch.int32)
        if rope
        else None
    )
    args = (qkv, cos_sin, positions, num_q, num_kv, 1e-5, 0.3, False)
    ref = v2_prep(*args, enable_pdl=pdl)
    out = mso._vectorized_preattn_prep(v2_prep)(*args, enable_pdl=pdl)
    assert _same_bits(ref[0], out[0]) and _same_bits(ref[1], out[1].reshape(ref[1].shape))


def test_combine_gain_reload_matches_held_gain(monkeypatch):
    from tensorrt_llm._torch.models import modeling_musespark_open as mso

    g = torch.Generator(device="cuda").manual_seed(0)
    tokens, top_k, hidden = 1024, 8, 4096
    gemm2 = (torch.randn((tokens * top_k + 256, hidden), device="cuda", generator=g) * 0.7).to(
        torch.bfloat16
    )
    idx = (
        torch.randperm(tokens * top_k, device="cuda", generator=g)
        .to(torch.int32)
        .reshape(tokens, top_k)
    )
    idx[::7, 3] = -1  # a non-local slot
    weights = torch.softmax(torch.randn((tokens, top_k), device="cuda", generator=g), dim=-1)
    norm_weight = (torch.randn((hidden,), device="cuda", generator=g) * 0.1).to(torch.bfloat16)

    def combine():
        return mso._sparse_post_expert_norm_combine(
            gemm2,
            idx,
            weights,
            norm_weight,
            1e-8,
            num_tokens=tokens,
            top_k=top_k,
            hidden_size=hidden,
        )

    monkeypatch.setattr(mso, "_CTX_FUSIONS", False)
    held = combine()
    monkeypatch.setattr(mso, "_CTX_FUSIONS", True)
    assert _same_bits(held, combine())
