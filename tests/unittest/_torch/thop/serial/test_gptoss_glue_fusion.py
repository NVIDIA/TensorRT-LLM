# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise tests of the GPT-OSS glue fusions (tensorrt_llm/_torch/cute_dsl_kernels/gptoss_glue_fusion.py).

moe_finalize_add_rmsnorm is compared with the MoE finalize followed by FlashInfer's fused add+RMSNorm. The finalize is
written in PyTorch here: a bf16 x bf16 product is exact in fp32, so ``acc + w * x`` in fp32 from +0 in expert order and
one round to bf16 is the finalize kernel's ``fma(w, x, acc)`` for normal-range values. fused_add_rmsnorm_mxfp8 is
compared with FlashInfer's fused add+RMSNorm followed by mxfp8_quantize.
"""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() // 10 != 10,
    reason="the TRTLLM-Gen MXFP4 MoE path these fuse runs on the SM100 family only",
)

H, HP, TOPK, EPS = 2880, 2944, 4, 1e-5


def _glue():
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
    from tensorrt_llm._torch.cute_dsl_kernels import gptoss_glue_fusion as glue

    reason = glue.flashinfer_check()
    if reason:
        pytest.skip(reason)
    if not glue.flashinfer_uses_cute_norm(torch.device("cuda")):
        pytest.skip("FlashInfer runs its CUDA JIT norm here")
    return glue


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _moe_triple(m, seed, invalid_frac=0.0, sigma=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = m * TOPK + 128
    gemm2 = (torch.randn((rows, HP), generator=gen, device="cuda") * sigma).to(torch.bfloat16)
    idx = (
        torch.randperm(rows, generator=gen, device="cuda")[: m * TOPK].to(torch.int32).view(m, TOPK)
    )
    if invalid_frac > 0:
        drop = torch.rand((m, TOPK), generator=gen, device="cuda") < invalid_frac
        idx = torch.where(drop, torch.full_like(idx, -1), idx).contiguous()
    weights = torch.softmax(torch.randn((m, TOPK), generator=gen, device="cuda"), dim=-1).to(
        torch.bfloat16
    )
    residual = torch.randn((m, H), generator=gen, device="cuda").to(torch.bfloat16)
    norm_w = (1.0 + 0.1 * torch.randn((H,), generator=gen, device="cuda")).to(torch.bfloat16)
    return gemm2, weights, idx, residual, norm_w


def _reference_finalize_add_rmsnorm(gemm2, weights, idx, residual, norm_w):
    m = residual.shape[0]
    acc = torch.zeros((m, H), dtype=torch.float32, device="cuda")
    for k in range(TOPK):
        valid = idx[:, k] >= 0
        rows = gemm2[idx[:, k].clamp(min=0).long(), :H].float()
        acc = torch.where(valid[:, None], acc + weights[:, k : k + 1].float() * rows, acc)
    x = acc.to(torch.bfloat16)
    r = residual.clone()
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(x, r, norm_w, EPS)
    return x, r


@pytest.mark.parametrize("m", [1, 3, 18, 97, 99, 1536, 7918])
@pytest.mark.parametrize("invalid_frac", [0.0, 0.2])
def test_moe_finalize_add_rmsnorm_bitwise(m, invalid_frac):
    _glue()
    gemm2, weights, idx, residual, norm_w = _moe_triple(m, seed=m, invalid_frac=invalid_frac)
    out, res = torch.ops.trtllm.moe_finalize_add_rmsnorm(gemm2, weights, idx, residual, norm_w, EPS)
    x_ref, r_ref = _reference_finalize_add_rmsnorm(gemm2, weights, idx, residual, norm_w)
    assert torch.equal(_bits(out), _bits(x_ref))
    assert torch.equal(_bits(res), _bits(r_ref))


def test_moe_finalize_add_rmsnorm_cuda_graph():
    glue = _glue()
    m = 64
    gemm2, weights, idx, residual, norm_w = _moe_triple(m, seed=1)
    out = torch.empty_like(residual)
    res = torch.empty_like(residual)
    kernel = glue.compiled_finalize_norm(
        H,
        HP,
        TOPK,
        glue.FINALIZE_USES_FMA,
        glue.enable_pdl(residual.device),
        glue.sm_version_of(residual.device),
    )
    kernel(gemm2, idx, weights, residual, norm_w, out, res, m, EPS)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(gemm2, idx, weights, residual, norm_w, out, res, m, EPS)
    new = _moe_triple(m, seed=2)
    for dst, src in zip((gemm2, weights, idx, residual), new[:4]):
        dst.copy_(src)
    graph.replay()
    x_ref, r_ref = _reference_finalize_add_rmsnorm(gemm2, weights, idx, residual, norm_w)
    assert torch.equal(_bits(out), _bits(x_ref))
    assert torch.equal(_bits(res), _bits(r_ref))


@pytest.mark.parametrize("m", [1, 9, 18, 1536, 7918])
@pytest.mark.parametrize("sigma", [1.0, 0.003, 50.0])
def test_fused_add_rmsnorm_mxfp8_bitwise(m, sigma):
    glue = _glue()
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = (torch.randn((m, H), generator=gen, device="cuda") * sigma).to(torch.bfloat16)
    r = (torch.randn((m, H), generator=gen, device="cuda") * sigma).to(torch.bfloat16)
    norm_w = (1.0 + 0.1 * torch.randn((H,), generator=gen, device="cuda")).to(torch.bfloat16)
    ref = glue.unfused_add_rmsnorm_mxfp8(x, r, norm_w, EPS, 512)
    got = torch.ops.trtllm.fused_add_rmsnorm_mxfp8(x, r, norm_w, EPS, 512)
    for a, b in zip(got, ref):
        assert torch.equal(_bits(a), _bits(b))
