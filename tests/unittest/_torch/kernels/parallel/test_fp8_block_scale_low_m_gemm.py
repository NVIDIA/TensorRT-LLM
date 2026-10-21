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
``fp8_swap_ab_gemm``. Its error against an FP32 reference must not exceed the error of
``fp8_swap_ab_gemm``, which quantizes activations with one scale per 1x128.
"""

import dataclasses

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE


def _is_sm100f():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE or not _is_sm100f(), reason="needs an SM100/SM103 GPU"
)

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
    assert mod.max_m_for(2112, k) == mod.max_m_for(2176, k) == 224
    assert mod.low_m_max_for(2112, k) == mod.low_m_max_for(2176, k)


@pytest.mark.parametrize("split_k", [9, 12, 16])
@pytest.mark.parametrize("m", [8, 40, 100])
def test_wide_split_k_tactics(m, split_k):
    """Clusters of more than 8 CTAs (the rule's split for few weight tiles and long K)."""
    mod = _kernel_module()
    n, k = 768, 7168
    w_fp8, scale = _make_weight(n, k, seed=19)
    x = _make_activation(m, k, seed=3)
    if m > mod.LOW_M_MAX:
        base = mod.MidMTactic(mma_n=mod.mid_mma_n_for(m), split_k=split_k, a_stages=1, b_stages=2)
        tactic = dataclasses.replace(base, a_stages=mod.max_mid_a_stages(n, k, base))
    else:
        tactic = mod.LowMTactic(mma_n=mod.mma_n_for(m), split_k=split_k, a_stages=2, x_stages=4)
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
    assert mod.max_m_for(*one_wave) == 128
    assert mod.max_m_for(*multi_wave) == 0
    for n, k in [one_wave, (3072, 7168), (7168, 768), (2112, 7168)]:
        assert mod.kernel_max_m_for(n, k) == min(mod.max_m_for(n, k), mod.low_m_max_for(n, k))
        tactic = mod.default_tactic(8, n, k)
        assert tactic.split_k * -(-n // 128) <= sms
        mod.validate_tactic(n, k, tactic)
        mod.validate_mid_tactic(n, k, mod.default_mid_tactic(128, n, k))


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


# ---------------------------------------------------------------------------
# Selection inside the Kimi-K3 FP8 weight-read linear.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n, k", [(3072, 7168), (2112, 7168)])
@pytest.mark.parametrize("m", [1, 28, 64, 65, 224, 225, 256, 257])
def test_weight_read_linear_selects_low_m_gemm(m, n, k):
    """``_Fp8BlockScaleWeightReadLinear`` runs this GEMM up to ``max_m_for(N, K)`` rows and
    ``fp8_swap_ab_gemm`` above, on the same operands."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    w_pad, scale = _make_weight(n, k, seed=41)
    linear = _Fp8BlockScaleWeightReadLinear(w_pad[:n].contiguous(), scale.clone(), n)
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
