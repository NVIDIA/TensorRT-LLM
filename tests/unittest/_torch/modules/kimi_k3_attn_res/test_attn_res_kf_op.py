# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for trtllm::attn_res_kf_fwd (TRTLLM_K3_ATTN_RES_KF), the generated attn_res + RMSNorm kernel.

The op is NOT bit-identical to the persistent kernel it replaces (both keep the mix in FP32 and round
once, in a different reduction order). Its output is checked against the FP32 reference and against the
persistent kernel with the acceptance bounds of the kernel search (no element beyond about 2 bf16 ulp, relative
L2 <= 1e-3), the updated prefix sum bit-exactly. The tests are skipped where the op is not built in.
"""

from unittest import mock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

HIDDEN_SIZE = 7168
EPS = 1e-5
OUTPUT_EPS = 1e-5


@pytest.fixture(scope="module")
def kf_op():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 10:
        pytest.skip("the KF attn_res kernel requires an SM10x GPU")
    try:
        return torch.ops.trtllm.attn_res_kf_fwd
    except (AttributeError, RuntimeError) as err:
        pytest.skip(f"trtllm::attn_res_kf_fwd unavailable: {err}")


def _make_inputs(num_tokens: int, snapshots: int, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    row_scale = torch.exp(randn(num_tokens, 1, scale=1.5))
    block = randn(snapshots, num_tokens, HIDDEN_SIZE) * row_scale
    block[:, :, [5, 1031, 3584]] *= 30.0  # massive-activation channels
    x = {
        "layer": (randn(num_tokens, HIDDEN_SIZE) * row_scale).bfloat16(),
        "add": (randn(num_tokens, HIDDEN_SIZE, scale=0.3) * row_scale).bfloat16(),
        "block": block.bfloat16().contiguous(),
        "res_w": randn(HIDDEN_SIZE, scale=0.05).bfloat16(),
        "rms_w": (1 + randn(HIDDEN_SIZE, scale=0.1)).bfloat16(),
        "out_w": (1 + randn(HIDDEN_SIZE, scale=0.1)).bfloat16(),
    }
    if num_tokens > 2:
        x["layer"][1].zero_()  # an all-zero current row
        x["add"][1].zero_()
    return x


def _call(op, x: dict, has_add: bool):
    return op(
        x["layer"],
        x["add"] if has_add else None,
        x["block"],
        x["res_w"],
        x["rms_w"],
        x["out_w"],
        EPS,
        OUTPUT_EPS,
    )


def _reference(x: dict, has_add: bool):
    """The FP32 reference: softmax mix, rounded to BF16 once after the output RMSNorm."""
    updated = x["layer"] + x["add"] if has_add else x["layer"].clone()
    v = torch.cat([x["block"].float(), updated.float().unsqueeze(0)], dim=0)
    rs = (v.square().mean(dim=-1) + EPS).rsqrt()
    score_w = x["rms_w"].float() * x["res_w"].float()
    p = torch.softmax(((v * rs.unsqueeze(-1)) * score_w).sum(dim=-1), dim=0)
    mixed = (p.unsqueeze(-1) * v).sum(dim=0)
    r2 = (mixed.square().mean(dim=-1, keepdim=True) + OUTPUT_EPS).rsqrt()
    return updated, (mixed * r2 * x["out_w"].float()).bfloat16()


def _persistent(x: dict, has_add: bool):
    tokens, snapshots = x["layer"].shape[0], x["block"].shape[0]
    updated, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
        x["layer"].reshape(tokens, 1, HIDDEN_SIZE),
        x["add"].reshape(tokens, 1, HIDDEN_SIZE) if has_add else None,
        x["block"].reshape(snapshots, tokens, 1, HIDDEN_SIZE),
        x["res_w"],
        x["rms_w"],
        x["out_w"],
        EPS,
        OUTPUT_EPS,
    )
    return updated.reshape(tokens, HIDDEN_SIZE), output.reshape(tokens, HIDDEN_SIZE)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """The acceptance metrics (check_attn_res)."""
    r, u = expected.float(), actual.float()
    row_rms = r.square().mean(dim=-1, keepdim=True).sqrt()
    err = (u - r).abs()
    return {
        "finite": bool(torch.isfinite(u).all()),
        "beyond_2ulp": int((err > 2.0**-6 * r.abs() + 2.0**-8 * row_rms).sum()),
        "frac_beyond_1ulp": float((err > 2.0**-7 * r.abs() + 2.0**-8 * row_rms).float().mean()),
        "rel_l2": float(err.square().sum().sqrt() / (r.square().sum().sqrt() + 1e-30)),
    }


def _same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return actual.shape == expected.shape and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@torch.no_grad()
@pytest.mark.parametrize("has_add", [True, False])
@pytest.mark.parametrize("snapshots", range(1, 9))
@pytest.mark.parametrize("num_tokens", [1, 18, 36, 84, 100, 129])
def test_kf_op_matches_reference(kf_op, num_tokens, snapshots, has_add):
    x = _make_inputs(num_tokens, snapshots, seed=1000 * num_tokens + 10 * snapshots + has_add)
    updated, output = _call(kf_op, x, has_add)
    ref_updated, ref_output = _reference(x, has_add)
    assert _same_bytes(updated, ref_updated)
    err = _errors(output, ref_output)
    assert err["finite"] and err["beyond_2ulp"] == 0, err
    assert err["frac_beyond_1ulp"] <= 1e-3 and err["rel_l2"] <= 1e-3, err
    per_updated, per_output = _persistent(x, has_add)
    if has_add:
        assert _same_bytes(updated, per_updated)
    err = _errors(output, per_output)
    assert err["finite"] and err["beyond_2ulp"] == 0 and err["rel_l2"] <= 1e-3, err


@torch.no_grad()
def test_kf_op_cuda_graph_replay(kf_op):
    x = _make_inputs(36, 5, seed=1)
    _call(kf_op, x, True)  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _call(kf_op, x, True)
    fresh = _make_inputs(36, 5, seed=2)
    for key in ("layer", "add", "block"):
        x[key].copy_(fresh[key])
    graph.replay()
    torch.cuda.synchronize()
    for replayed, eager in zip(graph_out, _call(kf_op, x, True)):
        assert _same_bytes(replayed, eager)


@torch.no_grad()
def test_kf_op_rejects_unsupported_inputs(kf_op):
    x = _make_inputs(4, 3, seed=0)
    with pytest.raises(RuntimeError, match="7168"):
        _call(kf_op, dict(x, layer=x["layer"][:, :4096].contiguous()), False)
    for snapshots in (0, 9):
        bad = torch.zeros(snapshots, 4, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match=r"\[1, 8\]"):
            _call(kf_op, dict(x, block=bad), False)
    with pytest.raises(RuntimeError, match="bf16"):
        _call(kf_op, dict(x, add=x["add"].half()), True)
    # A row-contiguous view 16 bytes into its storage: the 32-byte row accesses need alignment.
    storage = torch.zeros(4 * HIDDEN_SIZE + 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="32-byte aligned"):
        _call(kf_op, dict(x, add=storage[8:].view(4, HIDDEN_SIZE)), True)


def _modules(seed: int):
    torch.manual_seed(seed)
    proj = nn.Linear(HIDDEN_SIZE, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    nn.init.normal_(proj.weight, std=0.05)
    norm = mkl.KimiK3RMSNorm(HIDDEN_SIZE, eps=EPS, dtype=torch.bfloat16).cuda()
    out_norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=OUTPUT_EPS, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN_SIZE, device="cuda"))
        out_norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN_SIZE, device="cuda"))
    return proj, norm, out_norm


def _kf_flags(kf: bool, pdl: bool = False):
    """The TP8 decode stack (persistent kernel above 32 tokens), with or without TRTLLM_K3_ATTN_RES_KF."""
    return mock.patch.multiple(
        mkl,
        _K3_ATTN_RES_PDL=pdl,
        _K3_ATTN_RES_PDL_MODE="2" if pdl else "0",
        _K3_ATTN_RES_PDL_MAX_M=32,
        _FUSED_ATTN_RES_ENABLED=True,
        _ATTN_RES_TOPOLOGY="split",
        _FUSED_ATTN_RES_MAX_TOKENS=32,
        _K3_ATTN_RES_KF=kf,
        _K3_ATTN_RES_KF_MAX_M=84,
    )


@torch.no_grad()
@pytest.mark.parametrize("has_add", [True, False])
@pytest.mark.parametrize("num_tokens", [36, 84, 85])
def test_plain_mixes_dispatch(kf_op, num_tokens, has_add):
    """TRTLLM_K3_ATTN_RES_KF takes the persistent kernel's mixes up to 84 tokens, with the op's outputs."""
    proj, norm, out_norm = _modules(num_tokens)
    x = _make_inputs(num_tokens, 4, seed=7)
    x.update(res_w=proj.weight.reshape(-1), rms_w=norm.weight, out_w=out_norm.weight)
    sites = []

    def run(kf: bool):
        with (
            _kf_flags(kf),
            mock.patch.object(mkl, "_note_attn_res_fusion", lambda site, *_: sites.append(site)),
        ):
            if has_add:
                return mkl._apply_attn_res_add_rmsnorm_fused(
                    x["layer"], x["add"], x["block"], proj, norm, out_norm
                )
            return x["layer"], mkl._apply_attn_res_rmsnorm_fused(
                x["layer"], x["block"], proj, norm, out_norm
            )

    with_kf, without_kf = run(True), run(False)
    kind = "add+attn_res+norm" if has_add else "attn_res+norm"
    taken = "kf" if num_tokens <= 84 else "persistent"
    assert sites == [f"{kind}/{taken}", f"{kind}/persistent"]
    expected = _call(kf_op, x, has_add) if num_tokens <= 84 else _persistent(x, has_add)
    assert _same_bytes(with_kf[1], expected[1])
    assert _same_bytes(with_kf[0], without_kf[0])  # the prefix sum stays bit-identical


@torch.no_grad()
def test_kf_gate_declines(kf_op):
    proj, norm, out_norm = _modules(3)
    x = _make_inputs(48, 2, seed=3)
    with _kf_flags(True):
        # A view that is not row-contiguous, and a compile trace: the persistent kernel keeps them.
        strided = x["layer"].t().contiguous().t()
        assert mkl._k3_attn_res_kf(strided, None, x["block"], proj, norm, out_norm) is None
        with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            assert mkl._k3_attn_res_kf(x["layer"], None, x["block"], proj, norm, out_norm) is None
        assert mkl._k3_attn_res_kf(x["layer"], None, x["block"], proj, norm, out_norm) is not None
    with _kf_flags(True), mock.patch.object(mkl, "_ATTN_RES_TOPOLOGY", "per_token"):
        with pytest.raises(ValueError, match="KIMI_K3_ATTN_RES_TOPOLOGY"):
            mkl._check_attn_res_kf_flag()
    with _kf_flags(True):
        mkl._check_attn_res_kf_flag()
