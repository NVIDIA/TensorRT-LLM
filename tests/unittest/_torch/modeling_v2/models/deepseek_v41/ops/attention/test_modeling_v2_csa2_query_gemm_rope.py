# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_query_gemm_rope entry: q_norm + MXFP8 quantization in one launch, then Q-B with the RoPE in the
GEMM epilogue (stock's context query path).

Rows carry RMS spread over two decades (the regression of a projection that folded the norm and the RoPE together: the
rotated column read its pair partner without that row's 1/rms). Gates: 3e-2 relative RMS against the fp32 product of the
unquantized normalized rows (the latent is MXFP8, a 3-bit mantissa), per row as well (so no row can hide behind the
others); 1e-2 against `csa2_mxfp8_linear` over the same latent (the same MXFP8 operands, another GEMM and a separate
RoPE: bf16 rounding). The MXFP8 latent it returns is checked to be the one `csa2_mxfp8_linear` quantizes.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (  # noqa: E501
    csa2_mxfp8_linear,
    mxfp8_row_scales,
    mxfp8_swizzled_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_query_gemm_rope import (  # noqa: E501
    csa2_query_gemm_fp8,
    csa2_query_gemm_rope,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
    csa2_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_stock_rope_table,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="CuTe-DSL kernels, SM103",
)


def _rel(a, b, dim=None):
    d = (a.float() - b.float()).square()
    r = b.float().square()
    if dim is None:
        return (d.mean().sqrt() / r.mean().sqrt()).item()
    return d.mean(dim).sqrt() / r.mean(dim).sqrt()


def _inputs(s, gen):
    m, n, k = s["m"], s["n"], s["k"]
    w = (torch.randn(n, k, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    blocks = torch.randint(
        121, 127, ((n + 31) // 32, k // 32), generator=gen, device=DEV, dtype=torch.uint8
    )
    scale = mxfp8_row_scales(blocks, (n, k))
    x = torch.randn(m, k, generator=gen, device=DEV)
    if s.get("rms_spread"):  # row RMS from 0.1 to 10
        x = x * torch.logspace(-1, 1, m, device=DEV)[
            torch.randperm(m, generator=gen, device=DEV)
        ].unsqueeze(1)
    x = x.bfloat16()
    gamma = (1.0 + 0.2 * torch.randn(k, generator=gen, device=DEV)).bfloat16()
    pos = torch.randint(0, 4000, (m,), generator=gen, device=DEV, dtype=torch.int32)
    cos_sin = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV))
    return (
        x,
        (gamma, 1e-6),
        w,
        mxfp8_swizzled_scales(scale),
        torch.ones(1, device=DEV),
        pos,
        cos_sin,
    ), scale


@pytest.mark.parametrize(
    "cell", csa2_query_gemm_rope.CELLS, ids=[c.why[:48] for c in csa2_query_gemm_rope.CELLS]
)
def test_cells(cell, certifying):  # noqa: F811
    certifying(csa2_query_gemm_rope)
    args, scale = _inputs(cell.spec, gen_for(cell))
    with validating(csa2_query_gemm_rope):
        q, (data, a_sf) = csa2_query_gemm_rope(*args)
    ref = csa2_query_gemm_rope.reference(*args, weight_scale=scale)
    assert q.shape == ref.shape and q.dtype == torch.bfloat16
    rel = _rel(q, ref)
    worst = float(_rel(q, ref, dim=1).max())
    x, norm, w, sw, alpha, pos, cos_sin = args
    same = csa2_mxfp8_linear(x, w, scale, sw, alpha, 0, (pos, cos_sin.view(-1, 2, 32), 512), norm)
    rel_same = _rel(q, same)
    print(
        f"query_gemm_rope {cell.why}: vs fp32 {rel:.4f} (worst row {worst:.4f}), vs mxfp8_linear {rel_same:.2e}"
    )
    assert rel < 3e-2 and worst < 6e-2, (rel, worst)
    assert rel_same < 1e-2, rel_same
    # the returned latent is the one the projections read: the index query reuses it
    again = csa2_mxfp8_linear(x, w, scale, sw, alpha, 0, None, None, (data, a_sf))
    plain = csa2_mxfp8_linear(x, w, scale, sw, alpha, 0, None, norm)
    assert _rel(again, plain) < 1e-6


def test_rejections():
    cell = csa2_query_gemm_rope.CELLS[0]
    args, _ = _inputs(cell.spec, gen_for(cell))
    csa2_query_gemm_rope.is_valid(*args)
    bad = list(args)
    bad[5] = args[5].long()
    with pytest.raises(AssertionError):
        csa2_query_gemm_rope.is_valid(*bad)
    bad = list(args)
    bad[4] = torch.ones(2, device=DEV)  # the GEMMs take exactly one fp32 alpha
    with pytest.raises(AssertionError, match="alpha"):
        csa2_query_gemm_rope.is_valid(*bad)


@pytest.mark.parametrize(
    "cell", csa2_query_gemm_fp8.CELLS, ids=[c.why[:48] for c in csa2_query_gemm_fp8.CELLS]
)
def test_fp8_cells(cell, certifying):  # noqa: F811
    """The unit-scale E4M3 query: against the fp32 product (7%: E4M3 has a 3-bit mantissa), and against the E4M3
    rounding of `csa2_query_gemm_rope`'s bf16 query (the context kernel's on-load quantization at unit scale): the same
    values up to one E4M3 step (the GEMM rounds its fp32 accumulator directly, the bf16 path rounds twice)."""
    certifying(csa2_query_gemm_fp8)
    args, scale = _inputs(cell.spec, gen_for(cell))
    with validating(csa2_query_gemm_fp8):
        q8, (data, a_sf) = csa2_query_gemm_fp8(*args)
    assert q8.dtype == torch.float8_e4m3fn and q8.shape == (cell.spec["m"], cell.spec["n"])
    ref = csa2_query_gemm_fp8.reference(*args, weight_scale=scale)
    rel = _rel(q8.float(), ref)
    bf16, (data_b, a_sf_b) = csa2_query_gemm_rope(*args)
    rounded = bf16.to(torch.float8_e4m3fn).float()
    rel_rounded = _rel(q8.float(), rounded)
    print(f"query_gemm_fp8 {cell.why}: vs fp32 {rel:.4f}, vs e4m3(bf16 path) {rel_rounded:.4f}")
    assert rel < 7e-2, rel
    assert rel_rounded < 2e-2, rel_rounded
    # the same MXFP8 latent as the bf16 path (the index query reads it)
    assert torch.equal(data.view(torch.uint8), data_b.view(torch.uint8)) and torch.equal(
        a_sf, a_sf_b
    )


@pytest.mark.parametrize("m", [32, 150, 1280])
def test_fp8_decode_query_from_the_shared_latent(m):
    """The decode form: `x_mxfp8` (the latent `csa2_rms_norm_mxfp8` normalized once for q_b and the index q_b) gives the
    same E4M3 query bit for bit, and once an eager call compiled the tile a capture replays the same query."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rms_norm_mxfp8 import (
        csa2_rms_norm_mxfp8,
    )

    args, _ = _inputs(
        dict(m=m, n=32768, k=1280, rms_spread=True), torch.Generator(device=DEV).manual_seed(m)
    )
    x, norm = args[0], args[1]
    latent = csa2_rms_norm_mxfp8(x, norm[0], norm[1])
    q8, _ = csa2_query_gemm_fp8(*args)
    shared, got_latent = csa2_query_gemm_fp8(*args, x_mxfp8=latent)
    assert torch.equal(q8.view(torch.uint8), shared.view(torch.uint8))
    assert got_latent[0] is latent[0]
    # `amax`: the same query, and the word raised to the FP32 bits
    # of amax(|q|) (before the E4M3 rounding: within a step)
    word = torch.zeros(1, dtype=torch.int32, device=DEV)
    worded, _ = csa2_query_gemm_fp8(*args, x_mxfp8=latent, amax=word)
    assert torch.equal(worded.view(torch.uint8), q8.view(torch.uint8))
    amax, coded = word.view(torch.float32).item(), q8.float().abs().max().item()
    assert abs(amax - coded) <= coded / 16, (amax, coded)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = csa2_query_gemm_fp8(*args, x_mxfp8=latent)  # compiled by the eager calls above
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[0].view(torch.uint8), q8.view(torch.uint8))


def test_fp8_decode_query_refuses_to_compile_in_a_capture():
    """A tile not compiled yet: inside a capture the entry raises (nothing is launched; the target's graph warm-up
    compiles every tile eagerly first). 300 rows take the 128x128 tile; other tests of the same pytest process (e.g.
    the dep attention-stream tests, 320-row decode steps) may already have compiled it, so the engine's
    compiled 128x128 FP8-query kernels are set aside for this test and restored after it."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import query_projection
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rms_norm_mxfp8 import (
        csa2_rms_norm_mxfp8,
    )

    args, _ = _inputs(dict(m=300, n=32768, k=1280), torch.Generator(device=DEV).manual_seed(300))
    latent = csa2_rms_norm_mxfp8(
        args[0], args[1][0], args[1][1]
    )  # eager: the capture launches nothing
    torch.cuda.synchronize()
    compiled = query_projection._COMPILED
    aside = {
        key: compiled.pop(key) for key in list(compiled) if key[0] == "fp8" and key[3] == (128, 128)
    }
    try:
        with pytest.raises(RuntimeError, match="eager step"):
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                csa2_query_gemm_fp8(*args, x_mxfp8=latent)
    finally:
        compiled.update(aside)
