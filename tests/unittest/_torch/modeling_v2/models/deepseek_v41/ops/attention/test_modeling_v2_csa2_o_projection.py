# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_o_projection entry: inverse RoPE + grouped O-A + O-B over quantized latents.

Gate: relative RMS against the fp32 product with the same (requantized) weights and unquantized activations: 3e-2 for
the small-M launch (one MXFP8 rounding, of the latent), 5e-2 beyond 16 rows (the activation and the latent are both
quantized), 1e-2 for the low-M pair (cells marked `lowm` and `graph`: O-B's per-row scales passed, a bf16 latent; the
pair runs only in a decode-graph step, so those cells run under `flashinfer_mxfp8_decode_graph_capture`, and a `lowm`
cell without `graph` takes the small-M launch, as the engine's eager steps do).
csa2_o_projection_fp8_in (the O-A input already E4M3 with per-128 scales): 3e-2 against the product of the dequantized
input (one MXFP8 rounding, of the latent).
"""

import contextlib

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (  # noqa: E501
    mxfp8_row_scales,
    mxfp8_swizzled_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (  # noqa: E501
    csa2_o_a_mxfp8_scales,
    csa2_o_projection,
    csa2_o_projection_fp8_in,
    requantize_block128,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
    csa2_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_stock_rope_table,
)
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="CuTe-DSL / Triton kernels, SM100 family",
)


def _rel(a, b):
    return (
        (a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt()
    ).item()


def _inputs(s, gen):
    m, g, r, d, n = s["m"], s["g"], s["r"], s["d"], s["n"]
    dense = (torch.randn(g * r, d, generator=gen, device=DEV) * 0.05).bfloat16()
    q, sc = requantize_block128(dense)
    wo_a, wo_a_scale = q.view(g, r, d), sc.view(g, r // 128, d // 128)
    wo_b = (torch.randn(n, g * r, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    blocks = torch.randint(
        121, 127, ((n + 31) // 32, g * r // 32), generator=gen, device=DEV, dtype=torch.uint8
    )
    wo_b_scale = mxfp8_row_scales(blocks, (n, g * r))
    att = torch.randn(m, g * d, generator=gen, device=DEV).bfloat16()
    pos = torch.randint(0, 4000, (m,), generator=gen, device=DEV, dtype=torch.int32)
    cos_sin = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV))
    args = (
        att,
        pos,
        cos_sin,
        wo_a,
        wo_a_scale,
        csa2_o_a_mxfp8_scales(wo_a_scale),
        wo_b,
        mxfp8_swizzled_scales(wo_b_scale),
        torch.ones(1, device=DEV),
    )
    return args, wo_b_scale


@pytest.mark.parametrize(
    "cell", csa2_o_projection.CELLS, ids=[c.why[:48] for c in csa2_o_projection.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_o_projection)
    args, wo_b_scale = _inputs(cell.spec, gen_for(cell))
    lowm, graph = cell.spec.get("lowm", False), cell.spec.get("graph", False)
    with flashinfer_mxfp8_decode_graph_capture() if graph else contextlib.nullcontext():
        native = dict(native_eager=True) if cell.spec.get("native") else {}
        got = (
            csa2_o_projection(*args, wo_b_scale=wo_b_scale, **native)
            if lowm
            else csa2_o_projection(*args, **native)
        )
    ref = csa2_o_projection.reference(*args, wo_b_scale=wo_b_scale)
    assert (
        got.shape == ref.shape == (cell.spec["m"], cell.spec["n"]) and got.dtype == torch.bfloat16
    )
    gate = 1e-2 if lowm and graph else 3e-2 if cell.spec["m"] <= 16 else 5e-2
    assert _rel(got, ref) < gate, _rel(got, ref)


@pytest.mark.parametrize("m,groups,lowm", [(6, 8, False), (160, 8, False), (6, 1, True)])
def test_cuda_graph_replay(m, groups, lowm):
    """`lowm`: a decode-graph step (warm-up and capture under the engine's decode-graph flag) runs the low-M pair."""
    with flashinfer_mxfp8_decode_graph_capture() if lowm else contextlib.nullcontext():
        _graph_replay(m, groups, lowm)


def _graph_replay(m, groups, lowm):
    gen = torch.Generator(device=DEV).manual_seed(5)
    args, wo_b_scale = _inputs(dict(m=m, g=groups, r=1024, d=4096, n=5120), gen)
    kw = dict(wo_b_scale=wo_b_scale) if lowm else {}
    csa2_o_projection(*args, **kw)  # eager: compiles the grouped GEMM / the low-M kernels
    torch.cuda.synchronize()
    static = args[0].clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = csa2_o_projection(static, *args[1:], **kw)
    static.copy_(args[0] * 0.5)
    g.replay()
    want = csa2_o_projection(static, *args[1:], **kw)
    assert _rel(out, want) < 1e-6


def test_low_m_pair_only_in_a_decode_graph():
    """Six rows with O-B's per-row scales: the low-M pair (bf16 latent) under the decode-graph flag, the small-M launch
    (MXFP8 latent) outside it, as the engine's `_lowm_o_a` gates them. The two differ (a quantized latent)."""
    gen = torch.Generator(device=DEV).manual_seed(9)
    args, wo_b_scale = _inputs(dict(m=6, g=1, r=1024, d=4096, n=5120), gen)
    with flashinfer_mxfp8_decode_graph_capture():
        pair = csa2_o_projection(*args, wo_b_scale=wo_b_scale)
    eager = csa2_o_projection(*args, wo_b_scale=wo_b_scale)
    assert torch.equal(
        eager, csa2_o_projection(*args)
    )  # the scales select nothing outside a decode graph
    assert not torch.equal(pair, eager)


@pytest.mark.parametrize("m", [6, 24, 2048])
def test_native_eager_o_b_is_stocks_native_gemm(m, monkeypatch):
    """`native_eager`: outside a decode graph O-B over the same MXFP8 latent is stock's native autotuned GEMM
    (`trtllm::mxfp8_mxfp8_gemm_autotuned`); O-A and the latent are unchanged, so the two outputs differ by O-B's kernel
    only (FP8 accumulation order), and under the decode-graph flag the option selects nothing. A spy on the O-B dispatch
    checks that the flag reaches it."""
    import tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection as o_projection

    flags = []
    o_b = o_projection._quantized_gemm

    def spy(*a, **k):
        flags.append(k.get("native_eager", False))
        return o_b(*a, **k)

    monkeypatch.setattr(o_projection, "_quantized_gemm", spy)
    gen = torch.Generator(device=DEV).manual_seed(21)
    args, wo_b_scale = _inputs(dict(m=m, g=1, r=1024, d=4096, n=5120), gen)
    native = csa2_o_projection(*args, native_eager=True)
    assert flags == [True], flags
    plain = csa2_o_projection(*args)
    assert flags == [True, False], flags
    assert _rel(native, plain) < 1e-2, _rel(native, plain)
    assert torch.equal(native, csa2_o_projection(*args, native_eager=True))
    with flashinfer_mxfp8_decode_graph_capture():
        assert torch.equal(
            csa2_o_projection(*args, wo_b_scale=wo_b_scale, native_eager=True),
            csa2_o_projection(*args, wo_b_scale=wo_b_scale),
        )


def test_rejections():
    cell = csa2_o_projection.CELLS[0]
    args, _ = _inputs(cell.spec, gen_for(cell))
    csa2_o_projection.is_valid(*args)
    bad = list(args)
    bad[1] = args[1].long()
    with pytest.raises(AssertionError):
        csa2_o_projection.is_valid(*bad)
    # the dep8 decode call: the O-A input prequantized by the sparse decode, `att` None
    g, m, d = 8, cell.spec["m"], cell.spec["d"]
    data = torch.zeros(g, m, d, dtype=torch.float8_e4m3fn, device=DEV)
    scales = torch.zeros(g * 128 * (d // 32), dtype=torch.uint8, device=DEV)
    csa2_o_projection.is_valid(None, *args[1:], att_mxfp8=(data, scales))
    with pytest.raises(AssertionError, match="swizzled"):
        csa2_o_projection.is_valid(None, *args[1:], att_mxfp8=(data, scales[:-1]))


@pytest.mark.parametrize(
    "cell", csa2_o_projection_fp8_in.CELLS, ids=[c.why[:48] for c in csa2_o_projection_fp8_in.CELLS]
)
def test_fp8_in_cells(cell, certifying):  # noqa: F811
    certifying(csa2_o_projection_fp8_in)
    s, gen = cell.spec, gen_for(cell)
    (_, _, _, wo_a, wo_a_scale, _, wo_b, wo_b_sw, alpha), wo_b_scale = _inputs(s, gen)
    m, g, d = s["m"], s["g"], s["d"]
    data = torch.randn(g, m, d, generator=gen, device=DEV).to(torch.float8_e4m3fn)
    scale = torch.rand(g, d // 128, (m + 3) // 4 * 4, generator=gen, device=DEV) * 0.02 + 0.001
    args = (data, scale, wo_a, wo_a_scale, wo_b, wo_b_sw, alpha)
    with validating(csa2_o_projection_fp8_in):
        got = csa2_o_projection_fp8_in(*args)
    ref = csa2_o_projection_fp8_in.reference(*args, wo_b_scale=wo_b_scale)
    assert got.shape == ref.shape == (m, s["n"]) and got.dtype == torch.bfloat16
    print(f"csa2_o_projection_fp8_in rel RMS {cell.why}: {_rel(got, ref):.4f}")
    assert _rel(got, ref) < 3e-2, _rel(got, ref)
