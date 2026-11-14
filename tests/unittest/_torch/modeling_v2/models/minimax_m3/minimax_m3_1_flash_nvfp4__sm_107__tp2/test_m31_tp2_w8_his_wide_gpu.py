# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W8 HIS_WIDE GPU oracle (SM 10.7): the drafter projections' quantize + MXFP8-input GEMV rows.

1. The decode quantize (m3_mxfp8_pinned_quantize at the decode tactic) is bitwise mxfp8_quantize(x, True).
2. Where both GEMVs serve (all four drafter shapes at 7 rows, qkv / o at 14), quantize + m3_mx_gemv is bitwise the
   bf16-input GEMV (the in-kernel quantize): the same operands, the same config.
3. At the rows W8 adds (gate_up / down at 14, all four at 21 / 28) the output matches the fp32 reference of the
   MXFP8 operands (dequantized activation x dequantized weight) within the bf16 output rounding, and replays the same
   inside a CUDA graph.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import (  # noqa: F401
        minimax_m3_1_flash_nvfp4__sm_107__tp2,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

_ON_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(not _ON_SM107, reason="needs an SM 10.7 GPU")

_TP2 = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2"
SHAPES = {
    "qkv": (4608, 6144),  # the served drafter qkv (served dispatch lines)
    "o": (6144, 4096),
    "gate_up": (12288, 6144),
    "down": (6144, 6144),
}


def _mods():
    import importlib

    drafter_mxfp8 = importlib.import_module(f"{_TP2}.drafter_mxfp8")
    gemv = importlib.import_module(f"{_TP2}.kernels.mx_gemv.op")
    return drafter_mxfp8, gemv


def _weight(n, k, seed):
    """A bf16 weight that is exactly an MXFP8 dequantization, and its recovered (E4M3, swizzled scales)."""
    drafter_mxfp8, _ = _mods()
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = (torch.randn(n, k, generator=g, device="cuda") * 64).to(torch.float8_e4m3fn)
    exp = torch.randint(-12, -4, (n, k // 32, 1), generator=g, device="cuda").float()
    w = (codes.float().view(n, k // 32, 32) * torch.exp2(exp)).view(n, k).to(torch.bfloat16)
    rec = drafter_mxfp8.mxfp8_from_bf16(w)
    assert rec is not None
    return w, rec[0], drafter_mxfp8.swizzle_sf(rec[1])


def _dequant_x(x):
    """The fp32 values of mxfp8_quantize(x)'s operand (row-major scales from the unswizzled quantize)."""
    q, sf = torch.ops.trtllm.mxfp8_quantize(x, False)
    m, k = x.shape
    scale = torch.exp2(sf.view(torch.uint8)[: m * (k // 32)].view(m, k // 32).float() - 127)
    return (q.float().view(m, k // 32, 32) * scale.unsqueeze(-1)).view(m, k)


@pytest.mark.parametrize("rows", [7, 14, 21, 28, 48])
@pytest.mark.parametrize("k", [4096, 6144])
def test_decode_quantize_is_mxfp8_quantize(rows, k) -> None:
    drafter_mxfp8, _ = _mods()
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 3
    q, sf = drafter_mxfp8._his_quantize(x)
    rq, rsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8).reshape(-1), rsf.view(torch.uint8).reshape(-1))


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("rows", [7, 14])
def test_quantize_gemv_is_the_bf16_gemv_where_both_serve(name, rows) -> None:
    drafter_mxfp8, gemv = _mods()
    n, k = SHAPES[name]
    cfg = gemv.get_config(n, k)
    if gemv._declines(n, k, rows, True, cfg):
        pytest.skip(f"the bf16-input GEMV declines {name} at {rows} rows (a W8 row)")
    _, w8, sf = _weight(n, k, seed=rows)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    ref = gemv.m3_mx_gemv_bf16(x, w8, sf, True, cfg)
    out = gemv.m3_mx_gemv(*drafter_mxfp8._his_quantize(x), w8, sf, True, cfg)
    assert ref is not None and out is not None
    assert torch.equal(out, ref)


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("rows", [14, 21, 28])
def test_w8_rows_match_the_mxfp8_reference(name, rows) -> None:
    drafter_mxfp8, gemv = _mods()
    n, k = SHAPES[name]
    cfg = gemv.get_config(n, k)
    if not gemv._declines(n, k, rows, True, cfg):
        pytest.skip(f"{name} at {rows} rows stays on the bf16-input GEMV")
    assert rows <= drafter_mxfp8.HIS_WIDE_MAX_ROWS and not gemv._declines(n, k, rows, False, cfg)
    w, w8, sf = _weight(n, k, seed=100 + rows)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    out = gemv.m3_mx_gemv(*drafter_mxfp8._his_quantize(x), w8, sf, True, cfg)
    assert out is not None and tuple(out.shape) == (rows, n)
    ref = _dequant_x(x) @ w.float().t()
    err = (out.float() - ref).abs()
    tol = ref.abs() * 2.0**-7 + 1e-3
    assert bool((err <= tol).all()), f"max abs err {err.max().item():.4g}"
    # CUDA graph: the same bits on replay.
    xs = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gemv.m3_mx_gemv(*drafter_mxfp8._his_quantize(xs), w8, sf, True, cfg)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gemv.m3_mx_gemv(*drafter_mxfp8._his_quantize(xs), w8, sf, True, cfg)
    for _ in range(2):
        captured.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, out)
