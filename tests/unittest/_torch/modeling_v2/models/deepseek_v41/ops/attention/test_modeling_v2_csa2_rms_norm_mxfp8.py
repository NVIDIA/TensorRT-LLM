# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_rms_norm_mxfp8 entry: the q_norm + MXFP8 quantization of the q_a latent in one launch.

Gates: (1) the quantized GEMM over this entry's output is bit for bit `csa2_mxfp8_linear(rms_norm=...)` (the launch the
projection runs itself), for a column slice of the fused q_a | kv output read in place; (2) against
`mxfp8_quantize` of the float64 reference rounded to bf16: the scales agree and at most 1% of the E4M3 bytes differ
(one step on a rounding tie of the bf16 normalized value).
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
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rms_norm_mxfp8 import (  # noqa: E501
    csa2_rms_norm_mxfp8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


def _latent(m, k, gen):
    """A column slice of a fused [m, k + 512] projection output, rows of RMS spread over two decades."""
    fused = (
        torch.randn(m, k + 512, generator=gen, device=DEV)
        * torch.logspace(-1, 1, m, device=DEV)[:, None]
    )
    return fused.bfloat16()[:, :k]


@pytest.mark.parametrize(
    "cell", csa2_rms_norm_mxfp8.CELLS, ids=[c.why[:48] for c in csa2_rms_norm_mxfp8.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_rms_norm_mxfp8)
    s, gen = cell.spec, gen_for(cell)
    x = _latent(s["m"], s["k"], gen)
    g = (1 + 0.1 * torch.randn(s["k"], generator=gen, device=DEV)).bfloat16()
    data, scales = csa2_rms_norm_mxfp8(x, g, 1e-6)
    assert data.shape == (s["m"], s["k"]) and data.dtype == torch.float8_e4m3fn
    want_data, want_scales = torch.ops.trtllm.mxfp8_quantize(
        csa2_rms_norm_mxfp8.reference(x, g, 1e-6).bfloat16().contiguous(), True
    )
    # the live scales at their offsets in the 128x4 swizzled tiles (not a flat prefix, which mixes in padding)
    m, cols = s["m"], s["k"] // 32
    r, c = torch.meshgrid(
        torch.arange(m, device=DEV), torch.arange(cols, device=DEV), indexing="ij"
    )
    off = (
        (r // 128) * ((cols + 3) // 4) * 512
        + (c // 4) * 512
        + (r % 32) * 16
        + (r % 128 // 32) * 4
        + c % 4
    )
    got_sf = scales.view(torch.uint8).reshape(-1)[off]  # [m, cols]
    want_sf = want_scales.view(torch.uint8).reshape(-1)[off]
    assert (got_sf != want_sf).float().mean() < 1e-2
    assert (data.view(torch.uint8) != want_data.view(torch.uint8)).float().mean() < 1e-2

    # and every differing value is finite and within one E4M3 step of the other (3 mantissa bits, or the block's
    # smallest subnormal)
    def decode(d, sf):
        return d.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)

    a, b = decode(data, got_sf), decode(want_data, want_sf)
    tiny = torch.exp2(torch.maximum(got_sf, want_sf).float() - 127 - 9).repeat_interleave(32, dim=1)
    assert torch.isfinite(a).all()
    assert bool(((a - b).abs() <= 2.0**-3 * torch.maximum(a.abs(), b.abs()) + tiny).all())


@pytest.mark.parametrize(
    "m,n", [(24, 32768), (24, 4096), (160, 4096)], ids=["q_b-24", "index-q_b-24", "index-q_b-160"]
)
def test_the_gemm_over_it_is_the_projections_own(m, n):
    """q_b / index q_b over the shared quantized latent equal each projection's own norm + quantize, bit for bit."""
    gen = torch.Generator(device=DEV).manual_seed(m + n)
    k = 1280
    x = _latent(m, k, gen)
    g = (1 + 0.1 * torch.randn(k, generator=gen, device=DEV)).bfloat16()
    w = (torch.randn(n, k, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    s = mxfp8_row_scales(
        torch.randint(121, 127, (n // 32, k // 32), generator=gen, device=DEV, dtype=torch.uint8),
        (n, k),
    )
    sw, alpha = mxfp8_swizzled_scales(s), torch.ones(1, device=DEV)
    own = csa2_mxfp8_linear(x, w, s, sw, alpha, 0, None, (g, 1e-6))
    shared = csa2_mxfp8_linear(x, w, s, sw, alpha, 0, None, None, csa2_rms_norm_mxfp8(x, g, 1e-6))
    assert torch.equal(own, shared)


def test_rejections():
    x = torch.zeros(4, 1280, dtype=torch.bfloat16, device=DEV)
    g = torch.ones(1280, dtype=torch.bfloat16, device=DEV)
    csa2_rms_norm_mxfp8.is_valid(x, g, 1e-6)
    with pytest.raises(AssertionError):
        csa2_rms_norm_mxfp8.is_valid(x[:, :1000], g[:1000], 1e-6)
    with pytest.raises(AssertionError):
        csa2_rms_norm_mxfp8.is_valid(x.float(), g, 1e-6)
    with pytest.raises(AssertionError, match="device"):  # a CPU weight with a CUDA activation
        csa2_rms_norm_mxfp8.is_valid(x, g.cpu(), 1e-6)
