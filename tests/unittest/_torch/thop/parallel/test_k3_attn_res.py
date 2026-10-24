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
"""``k3_attn_res`` (CuTe DSL) against the C++ per-token attention-residual ops.

* The exact kernel reproduces the arithmetic of ``trtllm::attn_res_rmsnorm_fwd`` /
  ``trtllm::attn_res_add_rmsnorm_fwd`` instruction for instruction: every output is
  bit-identical.
* The fast (vectorized) kernel sums the statistics in another order: its updated prefix sum is
  bit-identical and its normalized output may differ from the C++ op by one BF16 ulp in a small
  fraction of the elements.
* Both run inside a CUDA graph behind PDL producers and consumers (DSL and C++).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)

HIDDEN = 7168
EPS = 1e-6
OUT_EPS = 1e-5
# Fast kernel vs C++: an element differs by at most two BF16 rounding steps of its magnitude
# (plus a tiny absolute slack for values near zero), and few elements differ at all.
FAST_MAX_DIFF_FRACTION = 2e-2


def _k3():
    import tensorrt_llm  # noqa: F401  (registers the C++ trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_attn_res

    return k3_attn_res


def _inputs(m: int, n: int, scale: float, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return dict(
        block=rn(n - 1, m, HIDDEN, sc=scale),
        layer=rn(m, HIDDEN, sc=scale),
        addend=rn(m, HIDDEN, sc=0.5 * scale),
        shared=rn(m, HIDDEN, sc=0.5 * scale),
        proj=rn(HIDDEN, sc=0.05),
        norm_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(HIDDEN, sc=0.1).float()).to(torch.bfloat16),
    )


def _weights(t):
    return (t["proj"], t["norm_w"], t["out_w"], EPS, OUT_EPS)


def _cpp(t, mode: int):
    """``(updated prefix or None, output)`` of the C++ path for add mode 0 / 1 / 2."""
    _k3()
    m, k = t["layer"].shape[0], t["block"].shape[0]
    block = t["block"].view(k, m, 1, HIDDEN)
    if mode == 1:
        updated, out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
            t["layer"].view(m, 1, HIDDEN), t["addend"].view(m, 1, HIDDEN), block, *_weights(t)
        )
        return updated.view(m, HIDDEN), out.view(m, HIDDEN)
    updated = None
    row = t["layer"]
    if mode == 2:
        from tensorrt_llm._torch.modules.kimi_k3_moe_residual import _fused_moe_residual_add

        updated = row = _fused_moe_residual_add(t["addend"], t["shared"], t["layer"])
    out = torch.ops.trtllm.attn_res_rmsnorm_fwd(row.view(m, 1, HIDDEN), block, *_weights(t))
    return updated, out.view(m, HIDDEN)


def _dsl(t, mode: int, exact: bool, cluster=None):
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    out = torch.empty_like(t["layer"])
    updated = torch.empty_like(t["layer"]) if mode else None
    _k3().run_k3_attn_res(
        t["layer"], addends, t["block"], *_weights(t), out, updated, exact=exact, cluster=cluster
    )
    return updated, out


def _mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


def _assert_fast_close(actual: torch.Tensor, expected: torch.Tensor, what):
    a, e = actual.float(), expected.float()
    diff = (a - e).abs()
    # 2^-6 relative is two BF16 ulps (a one-ulp change of the mixture can move the normalized
    # and the weighted rounding by one each). Where the mixture cancels to near zero, the two
    # summation orders differ by fp32 rounding of the (much larger) terms: 2^-8 of the row RMS.
    bound = (
        torch.maximum(a.abs(), e.abs()) * 2.0**-6 + 2.0**-8 * e.pow(2).mean(-1, keepdim=True).sqrt()
    )
    assert bool((diff <= bound).all()), (what, float((diff - bound).max()))
    frac = float((diff > 0).float().mean())
    assert frac <= FAST_MAX_DIFF_FRACTION, (what, frac)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", list(range(2, 13)))
@pytest.mark.parametrize("m", [1, 7, 32, 64, 96])
def test_exact_bitwise(m: int, n: int, mode: int):
    for scale in (1.0, 30.0):
        t = _inputs(m, n, scale)
        ref_updated, ref_out = _cpp(t, mode)
        updated, out = _dsl(t, mode, exact=True)
        assert _mismatches(out, ref_out) == 0, f"scale {scale}"
        if mode:
            assert _mismatches(updated, ref_updated) == 0, f"scale {scale}"


@pytest.mark.parametrize("n", [5, 6, 7, 8, 12])
@pytest.mark.parametrize("m", [128, 512])
def test_cluster_grids_beyond_one_wave(m: int, n: int):
    """More clusters than are resident at once. On SM107 the BF16-output exact kernel faulted in
    such grids at some dynamic shared memory sizes (see ``K3AttnResKernel.__call__``); the
    FP8-output exact kernel runs at two of them (3488 / 4000 B for N = 6 / 7)."""
    t = _inputs(m, n, 1.0, seed=m + n)
    _, ref_out = _cpp(t, 0)
    _, out = _dsl(t, 0, exact=True)
    assert _mismatches(out, ref_out) == 0
    _, out, fp8, sf = _dsl_fp8(t, 0, exact=True)
    assert _mismatches(out, ref_out) == 0
    _assert_quantized_like(out, fp8, sf, "exact")
    for cluster in (None, 8):
        _, out = _dsl(t, 0, exact=False, cluster=cluster)
        _assert_fast_close(out, ref_out, cluster)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 9, 12])
@pytest.mark.parametrize("m", [1, 7, 32, 56, 64, 96])
def test_fast_against_cpp(m: int, n: int, mode: int):
    clusters = (None, 2, 4, 8) if n > 8 else (None, 1, 2, 4, 8)
    for scale in (1.0, 30.0):
        t = _inputs(m, n, scale, seed=m * 13 + n)
        ref_updated, ref_out = _cpp(t, mode)
        for cluster in clusters:
            updated, out = _dsl(t, mode, exact=False, cluster=cluster)
            if mode:
                assert _mismatches(updated, ref_updated) == 0
            _assert_fast_close(out, ref_out, (cluster, scale))


@pytest.mark.parametrize("exact", [True, False])
def test_ops(exact: bool, monkeypatch):
    """The custom ops run the kernel ``TRTLLM_K3_ATTN_RES_PDL_EXACT`` selects."""
    k3 = _k3()
    monkeypatch.setattr(k3, "_EXACT", exact)
    t = _inputs(7, 9, 1.0)
    w = _weights(t)
    out = torch.ops.trtllm.k3_attn_res_rmsnorm(t["layer"], t["block"], *w)
    assert _mismatches(out, _dsl(t, 0, exact=exact)[1]) == 0
    updated, out = torch.ops.trtllm.k3_attn_res_add_rmsnorm(t["layer"], t["addend"], t["block"], *w)
    ref_updated, ref_out = _dsl(t, 1, exact=exact)
    assert _mismatches(updated, ref_updated) == 0 and _mismatches(out, ref_out) == 0
    prefix, out = torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    ref_prefix, ref_out = _dsl(t, 2, exact=exact)
    assert _mismatches(prefix, ref_prefix) == 0 and _mismatches(out, ref_out) == 0


def _dsl_fp8(t, mode: int, exact: bool, cluster=None):
    """``(updated, out, fp8, scale)`` of the FP8-output kernel; scale filled with a canary."""
    k3 = _k3()
    m = t["layer"].shape[0]
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    out = torch.empty_like(t["layer"])
    updated = torch.empty_like(t["layer"]) if mode else None
    fp8 = torch.empty((m, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda")
    # A canary in every scale byte: padding rows must come back zero, as the quantizers write them.
    scale = torch.full((k3.fp8_scale_numel(m),), 0xA5, dtype=torch.uint8, device="cuda")
    k3.run_k3_attn_res(
        t["layer"],
        addends,
        t["block"],
        *_weights(t),
        out,
        updated,
        exact=exact,
        cluster=cluster,
        fp8_output=fp8,
        fp8_scale=scale,
    )
    return updated, out, fp8, scale


def _assert_quantized_like(out, fp8, scale, what):
    """Codes and scales equal both production 1x128 quantizers applied to the BF16 output."""
    for quantize in (
        torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0,
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0,
    ):
        ref_fp8, ref_scale = quantize(out)
        assert ref_scale.numel() == scale.numel(), (what, quantize, ref_scale.shape, scale.shape)
        codes = int((fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).sum())
        scales = int((scale.view(torch.uint8) != ref_scale.view(torch.uint8).reshape(-1)).sum())
        assert codes == 0 and scales == 0, (what, quantize, codes, scales)


def _check_fp8_output(m: int, n: int, mode: int, exact: bool, cluster=None):
    # The output is RMS-normalized, so its magnitude (and the UE8M0 exponents) follow the output
    # weight: tiny weights reach E4M3 subnormals, large ones the top of the range.
    for scale, out_scale in ((1.0, 1.0), (30.0, 1e-4), (1e-3, 300.0)):
        t = _inputs(m, n, scale, seed=m * 31 + n)
        t["out_w"] = (t["out_w"].float() * out_scale).to(torch.bfloat16)
        ref_updated, ref_out = _dsl(t, mode, exact=exact, cluster=cluster)
        updated, out, fp8, sf = _dsl_fp8(t, mode, exact=exact, cluster=cluster)
        what = (scale, cluster)
        assert _mismatches(out, ref_out) == 0, what
        if mode:
            assert _mismatches(updated, ref_updated) == 0, what
        _assert_quantized_like(out, fp8, sf, what)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 5, 12])
@pytest.mark.parametrize("m", [1, 7, 64, 129])
def test_fp8_output(m: int, n: int, mode: int, exact: bool):
    """The FP8 output is the production 1x128 quantization of the BF16 output (bit for bit, padding
    rows of the R128c4 scales zero); the BF16 outputs equal those of the BF16-only kernel."""
    _check_fp8_output(m, n, mode, exact)


@pytest.mark.parametrize("cluster", [1, 2, 4, 8])
def test_fp8_output_fast_clusters(cluster: int):
    """Every cluster size of the fast kernel (a CTA slice is a whole number of 128-blocks)."""
    _check_fp8_output(7, 5, 1, exact=False, cluster=cluster)


@pytest.mark.parametrize("exact", [True, False])
def test_fp8_ops(exact: bool, monkeypatch):
    """The FP8-output custom ops equal run_k3_attn_res with FP8 buffers."""
    k3 = _k3()
    monkeypatch.setattr(k3, "_EXACT", exact)
    t = _inputs(9, 5, 1.0)
    w = _weights(t)
    out, fp8, sf = torch.ops.trtllm.k3_attn_res_rmsnorm_fp8(t["layer"], t["block"], *w)
    _, ref_out, ref_fp8, ref_sf = _dsl_fp8(t, 0, exact=exact)
    assert _mismatches(out, ref_out) == 0 and torch.equal(
        fp8.view(torch.uint8), ref_fp8.view(torch.uint8)
    )
    assert torch.equal(sf, ref_sf)
    updated, out, fp8, sf = torch.ops.trtllm.k3_attn_res_add_rmsnorm_fp8(
        t["layer"], t["addend"], t["block"], *w
    )
    ref_updated, ref_out, ref_fp8, ref_sf = _dsl_fp8(t, 1, exact=exact)
    assert _mismatches(updated, ref_updated) == 0 and _mismatches(out, ref_out) == 0
    assert torch.equal(fp8.view(torch.uint8), ref_fp8.view(torch.uint8)) and torch.equal(sf, ref_sf)
    prefix, out, fp8, sf = torch.ops.trtllm.k3_attn_res_moe_add_rmsnorm_fp8(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    ref_prefix, ref_out, ref_fp8, ref_sf = _dsl_fp8(t, 2, exact=exact)
    assert _mismatches(prefix, ref_prefix) == 0 and _mismatches(out, ref_out) == 0
    assert torch.equal(fp8.view(torch.uint8), ref_fp8.view(torch.uint8)) and torch.equal(sf, ref_sf)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("n", [4, 9])
def test_graph_replay_pdl_chain(n: int, exact: bool):
    """DSL -> DSL -> C++ -> DSL chain in one CUDA graph: every PDL hand-off is observed."""
    m = 32
    t = _inputs(m, n, 1.0)
    block4 = t["block"].view(n - 1, m, 1, HIDDEN)
    layer3 = t["layer"].view(m, 1, HIDDEN)
    addend = torch.empty_like(t["addend"])

    def dsl(a):
        out = torch.empty_like(a)
        updated = torch.empty_like(a)
        _k3().run_k3_attn_res(t["layer"], (a,), t["block"], *_weights(t), out, updated, exact=exact)
        return updated, out

    def cpp(a):
        u, o = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
            layer3, a.view(m, 1, HIDDEN), block4, *_weights(t)
        )
        return u.view(m, HIDDEN), o.view(m, HIDDEN)

    def chain(first, second, third, fourth):
        r1 = first(addend)
        r2 = second(r1[1])
        r3 = third(r2[1])
        r4 = fourth(r3[1])
        return [r1, r2, r3, r4]

    outs = []

    def body():
        outs[:] = chain(dsl, dsl, cpp, dsl)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        addend.copy_(t["addend"])
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        addend.copy_(torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        # The same chain run eagerly is the reference (same kernels, same inputs).
        refs = chain(dsl, dsl, cpp, dsl)
        for i, ((u, o), (ru, ro)) in enumerate(zip(outs, refs)):
            assert _mismatches(o, ro) == 0, f"replay {seed} stage {i} output"
            assert _mismatches(u, ru) == 0, f"replay {seed} stage {i} updated prefix"
        if exact:
            for i, ((u, o), (ru, ro)) in enumerate(zip(outs, chain(cpp, cpp, cpp, cpp))):
                assert _mismatches(o, ro) == 0, f"replay {seed} stage {i} output vs C++"
