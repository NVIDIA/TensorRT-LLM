# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness of the FP8 block-scale low-M GEMM (``trtllm::fp8_block_scale_low_m_gemm``).

The kernel quantizes activations in-kernel to MXFP8 (one power-of-two scale per 1x32
group) and consumes the same resmoothed FP8 weight + DeepGEMM-packed UE8M0 scale as
``fp8_swap_ab_gemm``. Its error against an FP32 reference must not exceed the error of the
architecture's default FP8 block-scale GEMM, which quantizes activations with one scale per
1x128 (``fp8_swap_ab_gemm`` on SM100/SM103, the CuTe MXFP8 GEMM on SM107).
"""

import dataclasses
import functools

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)


def _capability():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


def _is_sm107():
    return _capability() == (10, 7)


pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not (
        _capability() in ((10, 0), (10, 3)) or (_is_sm107() and IS_CUTLASS_DSL_RUBIN_AVAILABLE)
    ),
    reason="needs an SM100/SM103 GPU, or SM107 with a Rubin-capable CuTe DSL",
)
# The default GEMM next to which the low-M GEMM runs: DeepGEMM on SM100/SM103 (on the same
# DeepGEMM-packed scale), the CuTe MXFP8 GEMM on SM107.
sm100_only = pytest.mark.skipif(_is_sm107(), reason="SM100/SM103 DeepGEMM path")
sm107_only = pytest.mark.skipif(not _is_sm107(), reason="needs an SM107 GPU")

# (N, K): Kimi-K3 decode projections plus small shapes with K % 512 != 0.
SHAPES = [(3072, 7168), (7168, 768), (384, 1280), (256, 640)]
MS = [1, 7, 8, 14, 28, 32, 56, 64, 65, 112, 128, 168, 224, 256]


def _kernel_module():
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm


def _make_weight(n, k, seed):
    """N(0, 0.02) weight of ceil(N/128) * 128 rows quantized per 128x128 block with
    power-of-two scales."""
    n_pad = -(-n // 128) * 128
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((n_pad, k), device="cuda", generator=gen) * 0.02
    blocks = w.view(n_pad // 128, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n_pad, k)
    return w_fp8, scale.float()


def _pack(scale, n, k):
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    return transform_sf_into_required_layout(scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False)


def _make_activation(m, k, seed):
    """N(0, 1) activations with a few large outliers (up to ~1e3)."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    x = torch.randn((m, k), device="cuda", generator=gen)
    idx = torch.randint(0, m * k, (max(1, m * k // 2048),), device="cuda", generator=gen)
    mag = torch.rand(idx.shape, device="cuda", generator=gen) * 900.0 + 100.0
    x.view(-1)[idx] = mag * torch.sign(torch.randn(idx.shape, device="cuda", generator=gen))
    return x.to(torch.bfloat16)


def _reference(x, w_fp8, scale):
    n, k = w_fp8.shape
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return x.float() @ (w_fp8.float() * expanded).t()


def _mxfp8_emulated_reference(x, w_fp8, scale):
    """FP32 GEMM on the exact MXFP8 activation the kernel produces (1x32 UE8M0 + E4M3)."""
    m, k = x.shape
    groups = x.float().view(m, k // 32, 32)
    amax = groups.abs().amax(-1, keepdim=True).clamp(min=1.0014e-4)
    exp = torch.ceil(torch.log2(amax / 448.0))
    xq = (groups / torch.exp2(exp)).to(torch.float8_e4m3fn).float() * torch.exp2(exp)
    return _reference(xq.view(m, k), w_fp8, scale)


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _run(x, w, packed, tactic=None):
    out = torch.empty((x.shape[0], w.shape[0]), device="cuda", dtype=torch.bfloat16)
    _kernel_module().run_low_m_gemm(x, w, packed, out, tactic=tactic)
    return out


def _swap_ab(x, w, packed):
    return torch.ops.trtllm.fp8_swap_ab_gemm(x, w, packed, disable_ue8m0_cast=True)


@sm100_only
@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", MS)
def test_matches_reference_and_swap_ab_accuracy(m, n, k):
    w_fp8, scale = _make_weight(n, k, seed=n + k)
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=m)
    ref = _reference(x, w_fp8, scale)

    out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w_fp8, packed)
    out_swap_ab = _swap_ab(x, w_fp8, packed)
    torch.cuda.synchronize()

    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    err = _rel_err(out, ref)
    assert err < 5e-2, f"m={m} n={n} k={k}: rel err {err:.3e}"
    # The MMA and the split-K reduction against the emulated MXFP8 activation.
    err_emu = _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale))
    assert err_emu < 5e-3, f"m={m} n={n} k={k}: rel err vs MXFP8 emulation {err_emu:.3e}"
    # 1x32 activation scales are never coarser than 1x128 ones; only the FP32 summation order
    # and the BF16 output rounding may differ, hence the small relative slack.
    err_swap_ab = _rel_err(out_swap_ab, ref)
    assert err <= err_swap_ab * (1.0 + 2e-3), (
        f"m={m} n={n} k={k}: {err:.3e} > fp8_swap_ab_gemm {err_swap_ab:.3e}"
    )


@pytest.mark.parametrize("split_k", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [5, 16, 40])
def test_split_k_tactics(m, split_k):
    mod = _kernel_module()
    n, k = 384, 1280
    w_fp8, scale = _make_weight(n, k, seed=11)
    x = _make_activation(m, k, seed=5)
    tactic = mod.LowMTactic(mma_n=mod.mma_n_for(m), split_k=split_k, a_stages=2, x_stages=4)
    out = _run(x, w_fp8, _pack(scale, n, k), tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale)) < 5e-3


@pytest.mark.parametrize("split_k", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "m, mma_n", [(40, 64), (65, 96), (128, 128), (150, 160), (170, 192), (200, 224), (250, 256)]
)
def test_mid_m_tactics(m, mma_n, split_k):
    """Quantization kernel + TMA-fed GEMM, including the split-K reduce-scatter."""
    mod = _kernel_module()
    n, k = 384, 1280
    w_fp8, scale = _make_weight(n, k, seed=13)
    x = _make_activation(m, k, seed=9)
    base = mod.MidMTactic(mma_n=mma_n, split_k=split_k, a_stages=1, b_stages=2)
    tactic = dataclasses.replace(base, a_stages=mod.max_mid_a_stages(n, k, base))
    out = _run(x, w_fp8, _pack(scale, n, k), tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale)) < 5e-3


def test_low_m_cutover_dispatch(monkeypatch):
    """Shapes with a lower low-M cut-over run M above it on the mid-M kernel."""
    mod = _kernel_module()
    n, k, m = 384, 1280, 40
    monkeypatch.setitem(mod._tables().low_m_cutover, (n, k), 32)
    assert mod.low_m_max_for(n, k) == 32
    w_fp8, scale = _make_weight(n, k, seed=17)
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=5)
    out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w_fp8, packed)
    mid = _run(x, w_fp8, packed, mod.default_mid_tactic(m, n, k))
    torch.cuda.synchronize()
    assert torch.equal(out, mid)


@pytest.mark.parametrize("n, k", [(7168, 768), (2112, 7168)])
@pytest.mark.parametrize("m", [1, 14, 64, 200])
def test_cuda_graph_capture(m, n, k):
    mod = _kernel_module()
    w_fp8, scale = _make_weight(n, k, seed=7)
    w = w_fp8[:n].contiguous()
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=1)
    eager = torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w, packed)
    out = torch.empty_like(eager)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.run_low_m_gemm(x, w, packed, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.run_low_m_gemm(x, w, packed, out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_rejects_unsupported_inputs():
    mod = _kernel_module()
    n, k = 256, 640
    w_fp8, scale = _make_weight(n, k, seed=2)
    packed = _pack(scale, n, k)
    with pytest.raises(ValueError):
        torch.ops.trtllm.fp8_block_scale_low_m_gemm(
            torch.zeros((mod.MAX_M + 1, k), device="cuda", dtype=torch.bfloat16), w_fp8, packed
        )
    with pytest.raises(ValueError):
        torch.ops.trtllm.fp8_block_scale_low_m_gemm(
            torch.zeros((4, k), device="cuda", dtype=torch.bfloat16), w_fp8, packed.contiguous()
        )
    assert 1 <= mod.max_m_for(n, k) <= mod.MAX_M
    assert mod.max_m_for(n + 8, k) == 0
    assert mod.max_m_for(n, k + 64) == 0


# ---------------------------------------------------------------------------
# Partial last N tile (N % 128 != 0), e.g. the MLA fused [q_a; kv_a; k_pe] projection.
# ---------------------------------------------------------------------------

#: (N, K): the MLA fused-A projection (1536 + 512 + 64 rows) and small shapes whose last tile
#: keeps 80 / 16 rows.
PARTIAL_SHAPES = [(2112, 7168), (336, 1280), (144, 640)]
PARTIAL_MS = [1, 7, 8, 13, 32, 63, 64, 65, 100, 128, 129, 192, 224]


@pytest.mark.parametrize("n, k", PARTIAL_SHAPES)
@pytest.mark.parametrize("m", PARTIAL_MS)
def test_partial_n_tile_matches_padded_weight(m, n, k):
    """The N real columns equal the zero-padded weight's run and ignore the padding rows."""
    mod = _kernel_module()
    w_pad, scale = _make_weight(n, k, seed=n + k)
    n_pad = w_pad.shape[0]
    x = _make_activation(m, k, seed=m)
    # The partial and padded weights share the tactic (same tiles, same table entry).
    if m <= mod.low_m_max_for(n, k):
        assert mod.default_tactic(m, n, k) == mod.default_tactic(m, n_pad, k)
    else:
        assert mod.default_mid_tactic(m, n, k) == mod.default_mid_tactic(m, n_pad, k)

    zero_pad = w_pad.clone()
    zero_pad[n:] = 0
    out = _run(x, w_pad[:n].contiguous(), _pack(scale, n, k))
    out_zero = _run(x, zero_pad, _pack(scale, n_pad, k))
    out_garbage = _run(x, w_pad, _pack(scale, n_pad, k))
    torch.cuda.synchronize()

    assert out.shape == (m, n)
    assert torch.isfinite(out).all()
    # Rows of a tile are independent: zero or random padding rows change nothing.
    assert torch.equal(out, out_zero[:, :n])
    assert torch.equal(out, out_garbage[:, :n])
    assert not out_zero[:, n:].any()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_pad[:n], scale)) < 5e-3


@pytest.mark.parametrize("m", [1, 8, 32, 64, 65, 128, 224, 256])
def test_partial_n_tile_stays_in_bounds(m):
    """No store lands past the [M, N] output (a guard band after it stays untouched)."""
    mod = _kernel_module()
    n, k = 2112, 7168
    w_pad, scale = _make_weight(n, k, seed=21)
    x = _make_activation(m, k, seed=m + 3)
    guard = 128 * 64
    buf = torch.full((m * n + guard,), -7.0, device="cuda", dtype=torch.bfloat16)
    out = buf[: m * n].view(m, n)
    mod.run_low_m_gemm(x, w_pad[:n].contiguous(), _pack(scale, n, k), out)
    torch.cuda.synchronize()
    assert (buf[m * n :] == -7.0).all()


@pytest.mark.parametrize("split_k", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [5, 16, 40, 64])
def test_partial_n_tile_split_k_tactics(m, split_k):
    mod = _kernel_module()
    n, k = 336, 1280
    w_pad, scale = _make_weight(n, k, seed=11)
    x = _make_activation(m, k, seed=5)
    tactic = mod.LowMTactic(mma_n=mod.mma_n_for(m), split_k=split_k, a_stages=2, x_stages=4)
    out = _run(x, w_pad[:n].contiguous(), _pack(scale, n, k), tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_pad[:n], scale)) < 5e-3


@pytest.mark.parametrize("split_k", [1, 2, 3, 5])
@pytest.mark.parametrize("m, mma_n", [(40, 64), (65, 96), (128, 128), (200, 224), (250, 256)])
def test_partial_n_tile_mid_m_tactics(m, mma_n, split_k):
    mod = _kernel_module()
    n, k = 336, 1280
    w_pad, scale = _make_weight(n, k, seed=13)
    x = _make_activation(m, k, seed=9)
    base = mod.MidMTactic(mma_n=mma_n, split_k=split_k, a_stages=1, b_stages=2)
    tactic = dataclasses.replace(base, a_stages=mod.max_mid_a_stages(n, k, base))
    out = _run(x, w_pad[:n].contiguous(), _pack(scale, n, k), tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_pad[:n], scale)) < 5e-3


def test_partial_n_tile_limits_follow_whole_tiles():
    mod = _kernel_module()
    k = 7168
    assert mod.max_m_for(2112, k) == mod.max_m_for(2176, k) == (192 if _is_sm107() else 224)
    assert mod.low_m_max_for(2112, k) == mod.low_m_max_for(2176, k)


@pytest.mark.parametrize("split_k", [9, 12, 16])
@pytest.mark.parametrize("m", [8, 40, 100])
def test_wide_split_k_tactics(m, split_k):
    """Clusters of more than 8 CTAs (the rule's split for few weight tiles and long K)."""
    mod = _kernel_module()
    n, k = 768, 7168
    w_fp8, scale = _make_weight(n, k, seed=19)
    x = _make_activation(m, k, seed=3)
    tactic = mod.split_tactic(m, n, k, split_k, mid=m > mod.LOW_M_MAX)
    out = _run(x, w_fp8, _pack(scale, n, k), tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale)) < 5e-3


@pytest.mark.parametrize(
    "m, k, quant_warps, quant_groups",
    [(8, 20480, 4, 4), (16, 36864, 8, 4)],
    ids=["160_tiles_128_threads", "288_tiles_256_threads"],
)
def test_k_slice_longer_than_quant_threads(m, k, quant_warps, quant_groups):
    """A CTA whose K slice has more tiles than it has quantization threads folds the weight
    exponent of every one of its K tiles."""
    mod = _kernel_module()
    n = 128
    w_fp8, scale = _make_weight(n, k, seed=23)
    # Distinct block exponents along K, so a missing or wrong exponent changes the output.
    scale = scale * torch.exp2(torch.arange(k // 128, device="cuda") % 7 - 3.0)
    x = _make_activation(m, k, seed=29)
    base = mod.LowMTactic(
        mma_n=mod.mma_n_for(m),
        split_k=1,
        a_stages=1,
        x_stages=4,
        b_stages=4,
        quant_warps=quant_warps,
        quant_groups=quant_groups,
    )
    tactic = dataclasses.replace(base, a_stages=mod.max_a_stages(n, k, base))
    assert k // 128 > 32 * quant_warps
    out = _run(x, w_fp8, _pack(scale, n, k), tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale)) < 5e-3


@pytest.mark.parametrize("m, n, k", [(1, 12288, 28672), (8, 12288, 28672), (16, 16384, 53248)])
def test_long_k_shapes_default_tactics(m, n, k):
    """One-wave weights with long K, which callers select on SM100 / SM103 up to the low-M
    kernel's range: its default tactics run whole-K slices (split-K 1, 224 / 416 K tiles per
    CTA)."""
    w_fp8, scale = _make_weight(n, k, seed=31)
    x = _make_activation(m, k, seed=37)
    out = _run(x, w_fp8, _pack(scale, n, k))
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale)) < 5e-3


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7),
    reason="SM100 / SM103 selection rule",
)
@pytest.mark.parametrize("sms, max_m_7168_18432", [(148, 32), (152, 128)])
def test_low_m_range_and_mid_m_k_slice_limit(monkeypatch, sms, max_m_7168_18432):
    """Without measured limits, the low-M kernel serves 64 tokens while a CTA streams at most
    12 K tiles (32 otherwise), and the mid-M path is selected only for K slices of at most 96
    tiles. The split-K rule depends on the SM count, which is pinned: (7168, 18432) gets a
    mid-M split of 1 (144 K tiles per CTA, low-M range only) at 148 SMs and 2 (72) at 152."""
    mod = _kernel_module()
    monkeypatch.setattr(mod, "_num_sms", lambda: sms)
    monkeypatch.setattr(mod, "_max_active_clusters", lambda cluster_size: sms // cluster_size)
    mod._rule_split.cache_clear()
    try:
        assert mod.low_m_max_for(2112, 7168) == mod.LOW_M_MAX
        assert mod.low_m_max_for(7168, 1536) == mod.LOW_M_MAX
        assert mod.low_m_max_for(12288, 7168) == 32
        assert mod.max_m_for(7168, 18432) == max_m_7168_18432
        assert mod.max_m_for(12288, 28672) == mod.low_m_max_for(12288, 28672) == 32
    finally:
        mod._rule_split.cache_clear()


@pytest.mark.parametrize("quant_warps", [1, 2])
def test_rejects_fewer_than_four_quantization_warps(quant_warps):
    """The first four quantization warps run the epilogue."""
    mod = _kernel_module()
    tactic = mod.LowMTactic(
        mma_n=8, split_k=1, a_stages=2, quant_warps=quant_warps, quant_groups=quant_warps
    )
    with pytest.raises(ValueError):
        mod.validate_tactic(128, 1280, tactic)


def test_selection_rule():
    """One-wave shapes without a measured limit are selected up to the default limit on
    SM100/SM103 (never on SM107); larger grids never; the low-M kernel's range is the part of
    the selection below the cut-over."""
    mod = _kernel_module()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    one_wave, multi_wave = (256, 640), ((sms + 1) * 128, 1280)
    assert mod.max_m_for(*one_wave) == (0 if _is_sm107() else 128)
    assert mod.max_m_for(*multi_wave) == 0
    for n, k in [one_wave, (3072, 7168), (7168, 768), (2112, 7168)]:
        assert mod.kernel_max_m_for(n, k) == min(mod.max_m_for(n, k), mod.low_m_max_for(n, k))
        tactic = mod.default_tactic(8, n, k)
        assert tactic.split_k * -(-n // 128) <= sms
        mod.validate_tactic(n, k, tactic)
        mod.validate_mid_tactic(n, k, mod.default_mid_tactic(128, n, k))


@sm100_only
@pytest.mark.parametrize("m", [1, 32, 64, 128, 129, 256])
def test_fp8_block_scales_linear_selects_low_m_gemm(m):
    """The generic FP8 block-scale ``Linear`` runs this GEMM where its range serves the rows
    and ``fp8_swap_ab_gemm`` elsewhere, on the same operands."""
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    n, k = 3072, 7168
    w_pad, scale = _make_weight(n, k, seed=43)
    linear = Linear(
        k,
        n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    ).cuda()
    linear.load_weights([{"weight": w_pad[:n].cpu(), "weight_scale": scale.cpu()}])
    linear.post_load_weights()
    assert linear.fp8_low_m_gemm.max_m == _kernel_module().max_m_for(n, k) > 0
    x = _make_activation(m, k, seed=m)
    out = linear(x)
    torch.cuda.synchronize()
    if m <= linear.fp8_low_m_gemm.max_m:
        expected = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
            x, linear.weight, linear.weight_scale
        )
    else:
        expected = _swap_ab(x, linear.weight, linear.weight_scale)
    assert torch.equal(out, expected)


@sm100_only
def test_fp8_block_scales_linear_skips_low_m_gemm_on_sm107(monkeypatch):
    """On SM107 the generic FP8 block-scale ``Linear`` keeps ``fp8_swap_ab_gemm``: the GEMM's
    SM107 limits are cross-overs with the CuTe MXFP8 GEMM."""
    from tensorrt_llm._torch.modules import linear as linear_module
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    n, k = 3072, 7168
    w_pad, scale = _make_weight(n, k, seed=43)
    linear = linear_module.Linear(
        k,
        n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    ).cuda()
    linear.load_weights([{"weight": w_pad[:n].cpu(), "weight_scale": scale.cpu()}])
    monkeypatch.setattr(linear_module, "get_sm_version", lambda: 107)
    linear.post_load_weights()
    assert linear.fp8_low_m_gemm.max_m == 0


@pytest.mark.parametrize("rubin_dsl", [False, True], ids=["stock_dsl", "rubin_dsl"])
def test_sm107_selected_only_with_rubin_cute_dsl(monkeypatch, rubin_dsl):
    """On SM107 callers select neither the GEMM nor its variants unless the CuTe DSL is
    Rubin-capable; their default GEMM then runs unchanged."""
    from tensorrt_llm._torch import cute_dsl_utils

    mod = _kernel_module()
    monkeypatch.setattr(mod, "_device_capability", lambda device_index: (10, 7))
    monkeypatch.setattr(cute_dsl_utils, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", rubin_dsl)
    n, k = 3584, 7168
    assert (mod.max_m_for(n, k) > 0) == rubin_dsl
    assert (mod.kernel_max_m_for(n, k) > 0) == rubin_dsl


# ---------------------------------------------------------------------------
# Selection inside the Kimi-K3 FP8 weight-read linear.
# ---------------------------------------------------------------------------


@sm100_only
@pytest.mark.parametrize("n, k", [(3072, 7168), (2112, 7168)])
@pytest.mark.parametrize("m", [1, 28, 64, 65, 224, 225, 256, 257])
def test_weight_read_linear_selects_low_m_gemm(m, n, k):
    """``Fp8BlockScaleWeightReadLinear`` runs this GEMM up to ``max_m_for(N, K)`` rows and
    ``fp8_swap_ab_gemm`` above, on the same operands."""
    from tensorrt_llm._torch.modules.fp8_block_scale_weight_read_linear import (
        Fp8BlockScaleWeightReadLinear,
    )

    w_pad, scale = _make_weight(n, k, seed=41)
    linear = Fp8BlockScaleWeightReadLinear(w_pad[:n].contiguous(), scale.clone(), n)
    linear._weights_transformed = False
    linear.post_load_weights()
    x = _make_activation(m, k, seed=m)

    out = linear(x)
    swap_ab = _swap_ab(x, linear.weight, linear.weight_scale)
    torch.cuda.synchronize()
    assert out.shape == (m, n) and out.is_contiguous()
    if m <= _kernel_module().max_m_for(n, k):
        direct = torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, linear.weight, linear.weight_scale)
        assert torch.equal(out, direct)
    else:
        assert torch.equal(out, swap_ab)
    ref = _reference(x, w_pad[:n], scale)
    assert _rel_err(out, ref) <= _rel_err(swap_ab, ref) * (1.0 + 2e-3)


# ---------------------------------------------------------------------------
# SM107: the low-M GEMM next to the CuTe MXFP8 GEMM of the weight-read projections.
# ---------------------------------------------------------------------------

#: (N, K) of the Kimi-K3 FP8 weight-read decode projections: KDA fused qkvg, o_proj (KDA/MLA),
#: MLA fused q_a/kv_a (partial last tile), q_b, g / shared gate_up, MoE latent down / up,
#: shared down, and the helix16 / helix8 shards (KDA qkvg, o_proj, MLA g).
K3_SHAPES = [
    (49152, 7168),
    (7168, 12288),
    (2112, 7168),
    (18432, 1536),
    (12288, 7168),
    (3584, 7168),
    (7168, 3584),
    (7168, 6144),
    (3072, 7168),
    (7168, 768),
    (768, 7168),
    (6144, 7168),
    (7168, 1536),
    (1536, 7168),
]
#: Decode token counts (batch x (1 + MTP draft tokens) per rank) and the low-M edges.
K3_MS = [1, 8, 16, 32, 48, 64, 96, 128, 144, 192]


def test_pack_from_mxfp8_layout_matches_block_scale_pack():
    """The low-M scale derived from the CuTe MXFP8 (R128c4) scale equals the FP32 pack."""
    from tensorrt_llm.quantization.utils.fp8_utils import (
        transform_k128_scales_to_cutedsl_mxfp8_layout,
    )

    mod = _kernel_module()
    for n, k in [(2112, 7168), (768, 7168), (7168, 768), (336, 1280), (256, 640)]:
        _, scale = _make_weight(n, k, seed=n)
        sf = transform_k128_scales_to_cutedsl_mxfp8_layout(scale, mn=n, k=k)
        packed = mod.pack_block_scale_from_mxfp8_layout(sf, n, k)
        expected = _pack(scale, n, k)
        assert packed.shape == expected.shape and packed.stride() == expected.stride()
        assert torch.equal(packed, expected)


@functools.lru_cache(maxsize=None)
def _k3_weight(n, k):
    """Checkpoint-style FP8 codes + 128x128 scales (zero-padded past N for a partial tile)."""
    w_pad, scale = _make_weight(n, k, seed=n + k)
    return w_pad[:n].contiguous(), scale


def _loaded_weight_read_linear(w_fp8, scale):
    """``Fp8BlockScaleWeightReadLinear`` loaded like the checkpoint path."""
    from tensorrt_llm._torch.modules.fp8_block_scale_weight_read_linear import (
        Fp8BlockScaleWeightReadLinear,
    )

    linear = Fp8BlockScaleWeightReadLinear(w_fp8.clone(), scale.clone(), w_fp8.shape[0])
    linear._weights_transformed = False
    linear.post_load_weights()
    return linear


def _cute_mxfp8_path(linear, x):
    act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
    return torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
        act, linear.weight, act_scale, linear.weight_scale, fine_grained_m=True
    )


@sm107_only
@pytest.mark.parametrize("n, k", K3_SHAPES)
@pytest.mark.parametrize("m", K3_MS + [65, 97, 129, 161, 193, 225, 256])
def test_sm107_weight_read_linear_selects_low_m_gemm(m, n, k):
    """The low-M GEMM up to ``max_m_for(N, K)`` rows, the CuTe MXFP8 GEMM above; never less
    accurate than the CuTe MXFP8 GEMM."""
    w_fp8, scale = _k3_weight(n, k)
    linear = _loaded_weight_read_linear(w_fp8, scale)
    x = _make_activation(m, k, seed=m + k)
    out = linear(x)
    cute = _cute_mxfp8_path(linear, x)
    torch.cuda.synchronize()

    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    if m <= _kernel_module().max_m_for(n, k):
        direct = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
            x, linear.weight, linear._low_m_weight_scale
        )
        assert torch.equal(out, direct)
        assert _rel_err(out, _mxfp8_emulated_reference(x, w_fp8, scale)) < 5e-3
    else:
        assert torch.equal(out, cute)
    ref = _reference(x, w_fp8, scale)
    assert _rel_err(out, ref) <= _rel_err(cute, ref) * (1.0 + 2e-3)


@sm107_only
@pytest.mark.parametrize(
    "n, k, m",
    [
        (3584, 7168, 8),
        (3584, 7168, 64),
        (3584, 7168, 144),
        (2112, 7168, 8),
        (2112, 7168, 192),
        (7168, 768, 64),
        (1536, 7168, 224),
    ],
)
def test_sm107_weight_read_linear_cuda_graph(m, n, k):
    w_fp8, scale = _k3_weight(n, k)
    linear = _loaded_weight_read_linear(w_fp8, scale)
    x = _make_activation(m, k, seed=m)
    eager = linear(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        linear(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = linear(x)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


# ---------------------------------------------------------------------------
# Gated variant: out = bf16(x * gate) @ W^T (an output gate fused into o_proj).
# ---------------------------------------------------------------------------

#: (N, K): the helix o_proj, a split-K shape, and a partial last N tile.
GATED_SHAPES = [(7168, 768), (384, 1280), (2112, 7168), (7168, 12288)]
GATED_MS = [1, 7, 8, 14, 28, 32, 56, 64]


def _make_gate(m, k, seed, row_stride=None):
    """Sigmoid gate in BF16 (with exact 0 / 1 entries), optionally a column slice of a wider
    tensor (row stride ``row_stride``)."""
    gen = torch.Generator(device="cuda").manual_seed(seed + 2)
    width = k if row_stride is None else row_stride
    g = (torch.randn((m, width), device="cuda", generator=gen) * 4).to(torch.bfloat16).sigmoid()
    g.view(-1)[::97] = 0
    g.view(-1)[1::89] = 1
    return g[:, width - k :] if row_stride is not None else g


@pytest.mark.parametrize("n, k", GATED_SHAPES)
@pytest.mark.parametrize("m", GATED_MS)
@pytest.mark.parametrize("strided", [False, True], ids=["contig", "strided"])
def test_gated_matches_multiply_then_gemm(m, n, k, strided):
    """Bitwise equal to PyTorch's ``x * gate`` followed by the plain low-M kernel in the
    shape's low-M range; the gated GEMM declines above it."""
    mod = _kernel_module()
    w_pad, scale = _make_weight(n, k, seed=n + k)
    w, packed = w_pad[:n].contiguous(), _pack(scale, n, k)
    x = _make_activation(m, k, seed=m)
    gate = _make_gate(m, k, seed=m, row_stride=k + 1024 if strided else None)
    if m > mod.low_m_max_for(n, k):
        with pytest.raises(ValueError):
            torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w, packed, gate=gate)
        return
    assert mod.gate_fits(x, gate, n)
    plain = _run(x * gate, w, packed, mod.default_tactic(m, n, k))
    out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w, packed, gate=gate)
    torch.cuda.synchronize()
    assert torch.equal(out, plain)
    assert _rel_err(out, _mxfp8_emulated_reference((x * gate).contiguous(), w, scale)) < 5e-3


@pytest.mark.parametrize("m", [7, 14, 32])
def test_gated_cuda_graph_after_side_stream(m):
    """Graph replay with the gate written on another stream right before the launch."""
    mod = _kernel_module()
    n, k = 7168, 768
    w, scale = _make_weight(n, k, seed=7)
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=1)
    logits = _make_activation(m, k, seed=2)
    eager = torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w, packed, gate=logits.sigmoid())
    out = torch.empty_like(eager)
    gate = torch.empty_like(x)
    side = torch.cuda.Stream()
    fork, join = torch.cuda.Event(), torch.cuda.Event()

    def step():
        fork.record()
        with torch.cuda.stream(side):
            fork.wait()
            torch.sigmoid(logits, out=gate)
            join.record()
        join.wait()
        mod.run_low_m_gemm(x, w, packed, out, gate=gate)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(3):
        out.zero_()
        gate.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


def test_gated_rejects_unsupported_inputs():
    mod = _kernel_module()
    n, k = 7168, 768
    w, scale = _make_weight(n, k, seed=2)
    packed = _pack(scale, n, k)
    x = torch.zeros((8, k), device="cuda", dtype=torch.bfloat16)
    assert mod.gate_fits(x, torch.zeros_like(x), n)
    # Wrong dtype / shape, unaligned row stride; M above the low-M kernel.
    assert not mod.gate_fits(x, torch.zeros_like(x, dtype=torch.float32), n)
    assert not mod.gate_fits(x, torch.zeros((8, k - 128), device="cuda").bfloat16(), n)
    wide = torch.zeros((8, k + 4), device="cuda", dtype=torch.bfloat16)
    assert not mod.gate_fits(x, wide[:, 4:], n)
    big = torch.zeros((mod.low_m_max_for(n, k) + 1, k), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        torch.ops.trtllm.fp8_block_scale_low_m_gemm(big, w, packed, gate=torch.zeros_like(big))


# ---------------------------------------------------------------------------
# Sigmoid variant: torch.sigmoid of the BF16 output (an output-gate projection).
# ---------------------------------------------------------------------------


def _sigmoid_all_bf16_kernel():
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute

    mod = _kernel_module()

    @cute.kernel
    def kernel(src: cute.Tensor, dst: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 256 + tidx
        dst[i] = mod.sigmoid_bf16(src[i])

    @cute.jit
    def launch(src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream):
        kernel(src, dst).launch(grid=(256, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def test_sigmoid_bf16_matches_torch_on_every_bf16():
    """``sigmoid_bf16`` equals ``torch.sigmoid`` on all 65536 BF16 bit patterns."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32, device="cuda").to(torch.int16)
    src = bits.view(torch.bfloat16)
    dst = torch.empty_like(src)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _sigmoid_all_bf16_kernel()(from_dlpack(src), from_dlpack(dst), stream)
    torch.cuda.synchronize()
    ref = torch.sigmoid(src)
    same = dst.view(torch.int16) == ref.view(torch.int16)
    both_nan = torch.isnan(dst) & torch.isnan(ref)
    assert bool((same | both_nan).all()), f"{int((~(same | both_nan)).sum())} BF16 inputs differ"


@pytest.mark.parametrize("n, k", [(768, 7168), (384, 1280), (2112, 7168)])
@pytest.mark.parametrize("m", [1, 7, 14, 32, 64])
def test_sigmoid_matches_torch(m, n, k):
    """``torch.sigmoid`` of the plain kernel, bitwise, in the shape's low-M range; declined
    above it."""
    mod = _kernel_module()
    w_pad, scale = _make_weight(n, k, seed=n + k)
    w, packed = w_pad[:n].contiguous(), _pack(scale, n, k)
    x = _make_activation(m, k, seed=m) * 0.05
    if m > mod.low_m_max_for(n, k):
        with pytest.raises(ValueError):
            torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w, packed, sigmoid=True)
        return
    plain = _run(x, w, packed, mod.default_tactic(m, n, k))
    out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, w, packed, sigmoid=True)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.sigmoid(plain))


def _bf16_weight_read_linear(n, k):
    from tensorrt_llm._torch.modules.fp8_block_scale_weight_read_linear import (
        Fp8BlockScaleWeightReadLinear,
    )

    torch.manual_seed(0)
    weight = (torch.randn((n, k), device="cuda") * 0.02).to(torch.bfloat16)
    linear = Fp8BlockScaleWeightReadLinear(
        *Fp8BlockScaleWeightReadLinear.quantize_weight(weight), n
    )
    linear.post_load_weights()
    return linear


@pytest.mark.parametrize("n, k", [(7168, 768), (768, 7168)])
@pytest.mark.parametrize("m", [7, 14, 64, 65, 200])
def test_weight_read_linear_forward_gated_and_sigmoid(m, n, k):
    """``forward_gated`` / ``forward_sigmoid`` equal ``forward(x * gate)`` /
    ``forward(x).sigmoid()`` bitwise on every path."""
    linear = _bf16_weight_read_linear(n, k)
    x = _make_activation(m, k, seed=m)
    gate = _make_gate(m, k, seed=m)
    gated = linear.forward_gated(x, gate)
    sigmoid = linear.forward_sigmoid(x)
    torch.cuda.synchronize()
    assert torch.equal(gated, linear(x * gate))
    assert torch.equal(sigmoid, linear(x).sigmoid())


# ---------------------------------------------------------------------------
# Prequantized low-M kernel: the MXFP8 operand a producer hands over (E4M3 + UE8M0 1x32 scales
# in the R128c4 layout), bitwise the low-M kernel on the BF16 rows it quantizes.
# ---------------------------------------------------------------------------


def _mxfp8_operand(x):
    """``(xq, xs)`` of ``x`` with the scales in the R128c4 layout (rows padded to 128)."""
    mod = _kernel_module()
    m, k = x.shape
    xq, xs_rows = mod.mxfp8_quantize_reference(x)
    xs = torch.full((-(-m // 128) * 128 * (k // 32),), 0x7F, dtype=torch.uint8, device="cuda")
    r = torch.arange(m, device="cuda").unsqueeze(1)
    c = torch.arange(k // 32, device="cuda").unsqueeze(0)
    offsets = ((r // 128) * (k // 128) + c // 4) * 512 + (r % 32) * 16 + (r % 128) // 32 * 4 + c % 4
    xs[offsets] = xs_rows
    return xq, xs


@pytest.mark.parametrize(
    "n, k", [(3072, 7168), (3584, 7168), (7168, 768), (2112, 7168), (18432, 1536), (384, 1280)]
)
@pytest.mark.parametrize("m", [1, 7, 8, 14, 32, 56, 64])
def test_prequantized_matches_bf16_kernel(m, n, k):
    mod = _kernel_module()
    if m > mod.low_m_max_for(n, k):
        pytest.skip("above the low-M kernel's range")
    w_pad, scale = _make_weight(n, k, seed=n + k)
    w, packed = w_pad[:n].contiguous(), _pack(scale, n, k)
    x = _make_activation(m, k, seed=m)
    xq, xs = _mxfp8_operand(x)
    out = torch.ops.trtllm.fp8_block_scale_low_m_gemm_prequantized(xq, xs, w, packed)
    ref = _run(x, w, packed, mod.default_tactic(m, n, k))
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "n, k", [(3072, 7168), (3584, 7168), (7168, 3584), (7168, 768), (2112, 7168), (18432, 1536)]
)
def test_prequantized_tactic_follows_selected_low_m_tactic(n, k):
    """Over the prequantized range of the measured shapes, the prequantized kernel runs the
    low-M kernel's ``mma_n`` and split-K (the condition for bitwise equality)."""
    mod = _kernel_module()
    kernel_max_m = mod.kernel_max_m_for(n, k)
    assert 32 <= kernel_max_m <= mod.LOW_M_MAX
    for m in sorted({1, 8, 9, 16, 17, 32, kernel_max_m}):
        pq = mod.default_prequantized_tactic(m, n, k)
        low = mod.default_tactic(m, n, k)
        assert (pq.mma_n, pq.split_k) == (low.mma_n, low.split_k), m


@pytest.mark.parametrize("split_k", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [5, 16, 40])
def test_prequantized_split_k_tactics(m, split_k):
    mod = _kernel_module()
    n, k = 336, 768
    w_pad, scale = _make_weight(n, k, seed=11)
    w, packed = w_pad[:n].contiguous(), _pack(scale, n, k)
    x = _make_activation(m, k, seed=5)
    xq, xs = _mxfp8_operand(x)
    mma_n = mod.mma_n_for(m)
    tactic = mod.MidMTactic(mma_n=mma_n, split_k=split_k, a_stages=2, b_stages=2)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    mod.run_low_m_gemm_prequantized(xq, xs, w, packed, out, tactic=tactic)
    torch.cuda.synchronize()
    assert _rel_err(out, _mxfp8_emulated_reference(x, w_pad[:n], scale)) < 5e-3


@pytest.mark.parametrize("m", [7, 64])
def test_prequantized_cuda_graph(m):
    mod = _kernel_module()
    n, k = 7168, 768
    w, scale = _make_weight(n, k, seed=7)
    packed = _pack(scale, n, k)
    xq, xs = _mxfp8_operand(_make_activation(m, k, seed=1))
    eager = torch.ops.trtllm.fp8_block_scale_low_m_gemm_prequantized(xq, xs, w, packed)
    out = torch.empty_like(eager)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.run_low_m_gemm_prequantized(xq, xs, w, packed, out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.run_low_m_gemm_prequantized(xq, xs, w, packed, out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize("n, k", [(3072, 7168), (2112, 7168)])
@pytest.mark.parametrize("m", [1, 7, 32, 64, 65])
def test_weight_read_linear_forward_prequantized(m, n, k):
    """``forward_prequantized`` on the MXFP8 form of the rows is bitwise ``forward`` where
    ``takes_prequantized_input``."""
    linear = _bf16_weight_read_linear(n, k)
    x = _make_activation(m, k, seed=m)
    takes = linear.takes_prequantized_input(m)
    mod = _kernel_module()
    assert takes == (m <= min(mod.max_m_for(n, k), mod.low_m_max_for(n, k)))
    if not takes:
        return
    out = linear.forward_prequantized(*_mxfp8_operand(x))
    torch.cuda.synchronize()
    assert torch.equal(out, linear(x))


# ---------------------------------------------------------------------------
# NVFP4 output: fp4_quantize(y, global_sf, 16) of the BF16 output y of the low-M / mid-M kernel
# at the same tactic, bitwise (the routed experts' NVFP4 input straight from the MoE latent down
# projection).
# ---------------------------------------------------------------------------

#: (N, K): MoE latent down, a split-K helix shape and a small shape (all whole 128-row tiles).
NVFP4_SHAPES = [(3584, 7168), (7168, 768), (384, 1280)]
NVFP4_MS = [1, 7, 8, 14, 28, 32, 56, 64, 65, 96, 128, 160, 200, 256]


def _global_sf(y):
    """The usual NVFP4 global scale ``448 * 6 / amax`` of a BF16 tensor, as an FP32 [1]."""
    return (448.0 * 6.0 / y.float().abs().amax().clamp(min=1e-6)).reshape(1)


def _bf16_default_tactic(m, n, k):
    mod = _kernel_module()
    if m <= mod.low_m_max_for(n, k):
        return mod.default_tactic(m, n, k)
    return mod.default_mid_tactic(m, n, k)


def _nvfp4_run(x, w, packed, global_sf, tactic=None):
    m, n = x.shape[0], w.shape[0]
    codes = torch.empty((m, n // 2), device="cuda", dtype=torch.uint8)
    scales = torch.empty((m * n // 16,), device="cuda", dtype=torch.uint8)
    _kernel_module().run_low_m_gemm_nvfp4(x, w, packed, global_sf, codes, scales, tactic=tactic)
    return codes, scales


def _gemm_then_fp4_quantize(x, w, packed, global_sf, tactic):
    y = _run(x, w, packed, tactic)
    codes, scales = torch.ops.trtllm.fp4_quantize(y, global_sf, 16, False, False)
    return codes.view(torch.uint8), scales.view(torch.uint8)


def _assert_nvfp4_matches(x, w, packed, tactic, global_sf=None):
    """NVFP4 output at ``tactic`` equals the BF16 GEMM at ``tactic`` + ``fp4_quantize``."""
    if global_sf is None:
        global_sf = _global_sf(_run(x, w, packed, tactic))
    ref_codes, ref_scales = _gemm_then_fp4_quantize(x, w, packed, global_sf, tactic)
    codes, scales = _nvfp4_run(x, w, packed, global_sf, tactic)
    torch.cuda.synchronize()
    assert codes.shape == ref_codes.shape and scales.shape == ref_scales.shape
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scales, ref_scales)


@pytest.mark.parametrize("n, k", NVFP4_SHAPES)
@pytest.mark.parametrize("m", NVFP4_MS)
def test_nvfp4_out_matches_gemm_then_fp4_quantize(m, n, k):
    """Low-M kernel up to the shape's cut-over, quantization + mid-M kernel above."""
    mod = _kernel_module()
    w_fp8, scale = _make_weight(n, k, seed=n + k)
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=m)
    assert mod.can_run_nvfp4(x, w_fp8, packed, torch.ones(1, device="cuda"))
    _assert_nvfp4_matches(x, w_fp8, packed, _bf16_default_tactic(m, n, k))


@pytest.mark.parametrize("split_k", [1, 2, 3, 4])
@pytest.mark.parametrize("m", [5, 16, 40, 64])
def test_nvfp4_out_low_m_split_k_tactics(m, split_k):
    """Every split-K reduce-scatter owner of the low-M kernel quantizes the rows it owns."""
    mod = _kernel_module()
    n, k = 384, 1280
    w_fp8, scale = _make_weight(n, k, seed=11)
    x = _make_activation(m, k, seed=5)
    tactic = mod.LowMTactic(mma_n=mod.mma_n_for(m), split_k=split_k, a_stages=2, x_stages=4)
    _assert_nvfp4_matches(x, w_fp8, _pack(scale, n, k), tactic)


@pytest.mark.parametrize("split_k", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "m, mma_n", [(17, 64), (40, 64), (65, 96), (128, 128), (150, 160), (200, 224), (250, 256)]
)
def test_nvfp4_out_mid_m_split_k_tactics(m, mma_n, split_k):
    """Same for the mid-M kernel's split-K reduction."""
    mod = _kernel_module()
    n, k = 384, 1280
    w_fp8, scale = _make_weight(n, k, seed=13)
    x = _make_activation(m, k, seed=9)
    base = mod.MidMTactic(mma_n=mma_n, split_k=split_k, a_stages=1, b_stages=2)
    tactic = dataclasses.replace(base, a_stages=mod.max_mid_a_stages(n, k, base))
    _assert_nvfp4_matches(x, w_fp8, _pack(scale, n, k), tactic)


@pytest.mark.parametrize(
    "m, split_k, mid",
    [(1, 3, False), (8, 3, False), (16, 3, False), (17, 4, True), (64, 4, True), (96, 4, True)],
)
def test_nvfp4_out_split_tactic(m, split_k, mid):
    """The SM107 latent-down route's tactics (``split_tactic``), run on any architecture."""
    mod = _kernel_module()
    n, k = 3584, 7168
    tactic = mod.split_tactic(m, n, k, split_k, mid=mid)
    assert isinstance(tactic, mod.MidMTactic if mid else mod.LowMTactic)
    assert tactic.split_k == split_k
    if not mid:
        assert tactic.b_stages >= 2
        assert tactic.x_stages % tactic.quant_groups == 0
    w_fp8, scale = _make_weight(n, k, seed=n + k)
    x = _make_activation(m, k, seed=m)
    _assert_nvfp4_matches(x, w_fp8, _pack(scale, n, k), tactic)


def test_nvfp4_out_zero_rows_and_fixed_global_scale():
    """All-zero token rows (zero scales and codes) and a global scale unrelated to the data."""
    n, k = 3584, 7168
    w_fp8, scale = _make_weight(n, k, seed=3)
    packed = _pack(scale, n, k)
    gsf = torch.tensor([37.5], device="cuda", dtype=torch.float32)
    for m in (16, 100):
        x = _make_activation(m, k, seed=4)
        x[3] = 0
        x[11] = 0
        _assert_nvfp4_matches(x, w_fp8, packed, _bf16_default_tactic(m, n, k), gsf)


@pytest.mark.parametrize("m", [8, 64, 128])
def test_nvfp4_out_cuda_graph(m):
    mod = _kernel_module()
    n, k = 3584, 7168
    w_fp8, scale = _make_weight(n, k, seed=7)
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=1)
    gsf = _global_sf(_run(x, w_fp8, packed))
    eager = _nvfp4_run(x, w_fp8, packed, gsf)
    codes, scales = torch.empty_like(eager[0]), torch.empty_like(eager[1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.run_low_m_gemm_nvfp4(x, w_fp8, packed, gsf, codes, scales)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.run_low_m_gemm_nvfp4(x, w_fp8, packed, gsf, codes, scales)
    codes.zero_()
    scales.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(codes, eager[0])
    assert torch.equal(scales, eager[1])


def test_nvfp4_out_declines_unsupported_inputs():
    mod = _kernel_module()
    gsf = torch.ones(1, device="cuda", dtype=torch.float32)
    # Partial last weight tile.
    n, k = 2112, 7168
    w_pad, scale = _make_weight(n, k, seed=2)
    w, packed = w_pad[:n].contiguous(), _pack(scale, n, k)
    x = torch.zeros((8, k), device="cuda", dtype=torch.bfloat16)
    assert not mod.can_run_nvfp4(x, w, packed, gsf)
    assert mod.nvfp4_out_max_m_for(n, k) == 0
    # M above MAX_M, a global scale of the wrong dtype / size.
    n = 3584
    w, scale = _make_weight(n, k, seed=2)
    packed = _pack(scale, n, k)
    assert mod.can_run_nvfp4(x, w, packed, gsf)
    big = torch.zeros((mod.MAX_M + 1, k), device="cuda", dtype=torch.bfloat16)
    assert not mod.can_run_nvfp4(big, w, packed, gsf)
    assert not mod.can_run_nvfp4(x, w, packed, gsf.double())
    assert not mod.can_run_nvfp4(x, w, packed, torch.ones(2, device="cuda"))
    with pytest.raises(ValueError):
        _nvfp4_run(big, w, packed, gsf)
    # The low-M kernel stages its BF16 output tile in two quantized-activation stages.
    # The tactic is valid for the BF16 kernel (quant_groups=1 keeps b_stages=1 legal
    # there) and violates only the NVFP4 staging rule, so the test fails if that check
    # is removed or weakened.
    tactic = mod.LowMTactic(
        mma_n=32, split_k=1, a_stages=2, x_stages=4, b_stages=1, quant_warps=4, quant_groups=1
    )
    mod.validate_tactic(n, k, tactic)
    with pytest.raises(ValueError, match="b_stages"):
        mod.validate_nvfp4_out_tactic(n, k, tactic)


@pytest.mark.skipif(
    not hasattr(torch.ops.trtllm, "fp8_block_scale_low_m_gemm_nvfp4"),
    reason="trtllm::fp8_block_scale_low_m_gemm_nvfp4 is not registered",
)
@pytest.mark.parametrize("m", [1, 32, 64, 96, 200])
def test_nvfp4_out_op(m):
    """The op runs the default tactic of :func:`run_low_m_gemm_nvfp4` (the route's on SM107)."""
    mod = _kernel_module()
    n, k = 3584, 7168
    w_fp8, scale = _make_weight(n, k, seed=n)
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=m)
    tactic = mod.nvfp4_out_tactic(m, n, k) or _bf16_default_tactic(m, n, k)
    gsf = _global_sf(_run(x, w_fp8, packed, tactic))
    codes, scales = torch.ops.trtllm.fp8_block_scale_low_m_gemm_nvfp4(x, w_fp8, packed, gsf)
    ref_codes, ref_scales = _gemm_then_fp4_quantize(x, w_fp8, packed, gsf, tactic)
    torch.cuda.synchronize()
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scales, ref_scales)


@sm107_only
def test_sm107_nvfp4_out_route_for_latent_down():
    """SM107 selects the NVFP4 output for the MoE latent down projection up to 256 tokens: the
    low-M kernel at split-K 3 up to 16 tokens, the mid-M kernel at split-K 4 up to 96 and at its
    default tactic above; no other shape."""
    mod = _kernel_module()
    n, k = 3584, 7168
    assert mod.nvfp4_out_max_m_for(n, k) == 256
    for m in (1, 8, 16):
        assert mod.nvfp4_out_tactic(m, n, k) == mod.split_tactic(m, n, k, 3, mid=False)
    for m in (17, 64, 96):
        assert mod.nvfp4_out_tactic(m, n, k) == mod.split_tactic(m, n, k, 4, mid=True)
    for m in (97, 160, 256):
        assert mod.nvfp4_out_tactic(m, n, k) == mod.default_mid_tactic(m, n, k)
    for nk in K3_SHAPES:
        if nk != (n, k):
            assert mod.nvfp4_out_max_m_for(*nk) == 0
            assert mod.nvfp4_out_tactic(8, *nk) is None


@sm107_only
@pytest.mark.parametrize("m", [1, 8, 16, 17, 64, 96, 97, 160, 224, 256])
def test_sm107_nvfp4_out_route_matches_gemm_then_fp4_quantize(m):
    """The routed default of ``run_low_m_gemm_nvfp4`` is bitwise the BF16 GEMM at the route's
    tactic + ``fp4_quantize`` over the whole route (beyond the BF16 GEMM's own ``max_m``)."""
    mod = _kernel_module()
    n, k = 3584, 7168
    w_fp8, scale = _make_weight(n, k, seed=n + k)
    packed = _pack(scale, n, k)
    x = _make_activation(m, k, seed=m)
    tactic = mod.nvfp4_out_tactic(m, n, k)
    gsf = _global_sf(_run(x, w_fp8, packed, tactic))
    ref_codes, ref_scales = _gemm_then_fp4_quantize(x, w_fp8, packed, gsf, tactic)
    codes, scales = _nvfp4_run(x, w_fp8, packed, gsf)
    torch.cuda.synchronize()
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scales, ref_scales)


@sm100_only
def test_sm100_nvfp4_out_not_selected():
    """SM100 / SM103 keep the GEMM followed by ``fp4_quantize`` for every shape."""
    mod = _kernel_module()
    for n, k in K3_SHAPES:
        assert mod.nvfp4_out_max_m_for(n, k) == 0
        assert mod.nvfp4_out_tactic(8, n, k) is None
