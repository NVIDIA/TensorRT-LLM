# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FFN input quantized once for both consumers: nvfp4_fp8_quantize is bitwise the two separate
quantizers, dsv4_route is bitwise the routing half of dsv4_route_nvfp4_quant, and the shared expert from the
prequantized input is bitwise the one that quantizes it itself."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (  # noqa: E501
    dsv4_route,
    dsv4_route_nvfp4_quant,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fp8_quantize import (  # noqa: E501
    nvfp4_fp8_quantize,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import (  # noqa: E501
    deep_gemm_weight_scale,
    requantize_block128,
    shared_expert_fp8_deep_gemm,
    shared_expert_fp8_deep_gemm_prequant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103 kernels",
)


@pytest.mark.parametrize(
    "cell", nvfp4_fp8_quantize.CELLS, ids=[c.why[:44] for c in nvfp4_fp8_quantize.CELLS]
)
def test_quantize_cells(cell, certifying):  # noqa: F811
    certifying(nvfp4_fp8_quantize)
    s = cell.spec
    gen = gen_for(cell)
    x = (torch.randn(s["n"], s["hidden"], generator=gen, device=DEV) * 3).bfloat16()
    g = torch.tensor([448.0 * 6.0 / x.float().abs().max().item()], device=DEV)
    fp4, sf, fp8, fp8_sf = nvfp4_fp8_quantize(x, g)
    r4, rsf, r8, r8sf = nvfp4_fp8_quantize.reference(x, g)
    # the FP8 half is bitwise the engine's quantizer; the NVFP4 half differs from the torch reference in < 0.2% of bytes
    # (its fp32 scale arithmetic; measured 0.075% at 32300 rows)
    assert torch.equal(fp8.view(torch.uint8), r8.view(torch.uint8)) and torch.equal(fp8_sf, r8sf)
    assert (fp4.view(torch.uint8) != r4.view(torch.uint8)).float().mean().item() < 2e-3
    assert (sf.view(torch.uint8) != rsf.view(torch.uint8)).float().mean().item() < 2e-3
    # and bitwise what the routing launch's own NVFP4 quantization (the path it replaces) writes
    logits = torch.randn(s["n"], 384, generator=gen, device=DEV)
    bias = torch.randn(384, generator=gen, device=DEV) * 0.1
    _, _, q4, qsf = dsv4_route_nvfp4_quant(logits, bias, x, g, 6, 1.5)
    assert torch.equal(fp4.view(torch.uint8), q4.view(torch.uint8).view(fp4.shape))
    assert torch.equal(sf.view(torch.uint8), qsf.view(torch.uint8).view(sf.shape))


@pytest.mark.parametrize("wdt", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("cell", dsv4_route.CELLS, ids=[c.why[:44] for c in dsv4_route.CELLS])
def test_route_cells(cell, wdt, certifying):  # noqa: F811
    certifying(dsv4_route)
    s = cell.spec
    gen = gen_for(cell)
    logits = torch.randn(s["n"], s["experts"], generator=gen, device=DEV)
    bias = torch.randn(s["experts"], generator=gen, device=DEV) * 0.1
    x = torch.randn(s["n"], s["hidden"], generator=gen, device=DEV).bfloat16()
    g = torch.ones(1, device=DEV)
    ids, w = dsv4_route(logits, bias, s["top_k"], s["scale"], s["hidden"], wdt)
    fids, fw, _, _ = dsv4_route_nvfp4_quant(logits, bias, x, g, s["top_k"], s["scale"], wdt)
    assert w.dtype == wdt and torch.equal(ids, fids) and torch.equal(w, fw)
    rids, rw = dsv4_route.reference(logits, bias, s["top_k"], s["scale"], s["hidden"], wdt)
    assert torch.equal(ids.sort(-1).values, rids.sort(-1).values)
    order, rorder = ids.long().argsort(-1), rids.long().argsort(-1)
    # fp32: the weights the CuTe-DSL experts take (stock's routing_scales_dtype), within fp32 rounding of the reference
    tol = 1e-6 if wdt == torch.float32 else 8e-3
    torch.testing.assert_close(
        w.gather(1, order).float(), rw.gather(1, rorder).float(), rtol=tol, atol=tol
    )


@pytest.mark.parametrize(
    "cell",
    shared_expert_fp8_deep_gemm_prequant.CELLS,
    ids=[c.why[:40] for c in shared_expert_fp8_deep_gemm_prequant.CELLS],
)
def test_shared_expert_prequant_matches(cell, certifying):  # noqa: F811
    certifying(shared_expert_fp8_deep_gemm_prequant)
    n, h, i = cell.spec["n"], cell.spec["hidden"], cell.spec["inter"]
    gen = torch.Generator(device=DEV).manual_seed(n)
    w13, s13 = requantize_block128(torch.randn(2 * i, h, generator=gen, device=DEV) * 0.02)
    w2, s2 = requantize_block128(torch.randn(h, i, generator=gen, device=DEV) * 0.02)
    sf13, sf2 = deep_gemm_weight_scale(s13, *w13.shape), deep_gemm_weight_scale(s2, *w2.shape)
    x = (torch.randn(n, h, generator=gen, device=DEV) * 2).bfloat16()
    _, _, a, a_sf = nvfp4_fp8_quantize(x, torch.ones(1, device=DEV))
    lim = cell.spec["limit"]
    got = shared_expert_fp8_deep_gemm_prequant(a, a_sf, w13, sf13, w2, sf2, lim)
    assert torch.equal(got, shared_expert_fp8_deep_gemm(x, w13, sf13, s13, w2, sf2, s2, lim))
    ref = shared_expert_fp8_deep_gemm_prequant.reference(
        a, a_sf, w13, sf13, w2, sf2, lim, x=x, s13=s13, s2=s2
    )
    rel = (
        (got.float() - ref.float()).pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()
    ).item()
    assert rel < 0.06, rel
