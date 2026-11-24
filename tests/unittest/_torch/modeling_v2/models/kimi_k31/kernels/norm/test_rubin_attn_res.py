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
"""``rubin_attn_res`` (CuTe DSL) against the C++ per-token attention-residual ops.

* The (vectorized) kernel sums the statistics in another order than the C++ ops: its updated
  prefix sum is bit-identical and its normalized output may differ from the C++ op by one BF16
  ulp in a small fraction of the elements.
* It runs inside a CUDA graph behind PDL producers and consumers (DSL and C++).
"""

import pytest
import torch
from _torch.modeling_v2.models.kimi_k31.kernels.gemm.rubin_mxfp8_test_utils import (
    mxfp8_quantize_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

HIDDEN = 7168
EPS = 1e-6
OUT_EPS = 1e-5
# Fast kernel vs C++: an element differs by at most two BF16 rounding steps of its magnitude
# (plus a tiny absolute slack for values near zero), and few elements differ at all.
FAST_MAX_DIFF_FRACTION = 2e-2


def _k3():
    import tensorrt_llm  # noqa: F401  (registers the C++ trtllm ops)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.norm import (
        rubin_attn_res,
    )

    return rubin_attn_res


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
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
            rubin_moe_residual,
        )

        _fused_moe_residual_add = rubin_moe_residual._fused_moe_residual_add

        updated = row = _fused_moe_residual_add(t["addend"], t["shared"], t["layer"])
    out = torch.ops.trtllm.attn_res_rmsnorm_fwd(row.view(m, 1, HIDDEN), block, *_weights(t))
    return updated, out.view(m, HIDDEN)


def _dsl(t, mode: int, cluster=None):
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    out = torch.empty_like(t["layer"])
    updated = torch.empty_like(t["layer"]) if mode else None
    _k3().run_k3_attn_res(
        t["layer"], addends, t["block"], *_weights(t), out, updated, cluster=cluster
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
@pytest.mark.parametrize("n", [2, 3, 4, 5, 9, 12])
@pytest.mark.parametrize("m", [1, 7, 32, 56, 64])
def test_fast_against_cpp(m: int, n: int, mode: int):
    clusters = (None, 2, 4, 8) if n > 8 else (None, 1, 2, 4, 8)
    for scale in (1.0, 30.0):
        t = _inputs(m, n, scale, seed=m * 13 + n)
        ref_updated, ref_out = _cpp(t, mode)
        for cluster in clusters:
            updated, out = _dsl(t, mode, cluster=cluster)
            if mode:
                assert _mismatches(updated, ref_updated) == 0
            _assert_fast_close(out, ref_out, (cluster, scale))


def test_ops():
    """The launchers return what :func:`run_k3_attn_res` computes."""
    _k3()
    t = _inputs(7, 9, 1.0)
    w = _weights(t)
    out = _k3().k3_attn_res_rmsnorm(t["layer"], t["block"], *w)
    assert _mismatches(out, _dsl(t, 0)[1]) == 0
    updated, out = _k3().k3_attn_res_add_rmsnorm(t["layer"], t["addend"], t["block"], *w)
    ref_updated, ref_out = _dsl(t, 1)
    assert _mismatches(updated, ref_updated) == 0 and _mismatches(out, ref_out) == 0
    prefix, out = _k3().k3_attn_res_moe_add_rmsnorm(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    ref_prefix, ref_out = _dsl(t, 2)
    assert _mismatches(prefix, ref_prefix) == 0 and _mismatches(out, ref_out) == 0


@pytest.mark.parametrize("n", [2, 6, 12])
@pytest.mark.parametrize("m", [1, 7, 14, 32, 64])
def test_mxfp8_side_output(m: int, n: int):
    """The ``_mx`` launcher of the MoE residual update returns the vectorized kernel's BF16 results
    and the MXFP8 form of the normalized output that the low-M GEMM's in-kernel quantization makes
    of it."""
    _k3()
    t = _inputs(m, n, 1.0, seed=m + n)
    t["layer"][0, :64] = 0  # an all-zero 1x32 group of the mixture input
    w = _weights(t)
    ops = _k3()
    ref = ops.k3_attn_res_moe_add_rmsnorm(t["layer"], t["addend"], t["shared"], t["block"], *w)
    got = ops.k3_attn_res_moe_add_rmsnorm_mx(t["layer"], t["addend"], t["shared"], t["block"], *w)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert _mismatches(b, a) == 0, f"BF16 output {i}"
    xq, xs = got[len(ref) :]
    ref_q, ref_s = mxfp8_quantize_reference(ref[-1])
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(xs, ref_s)


def _lamport_workspace(rows: torch.Tensor, dirty: int):
    """A one-rank stand-in for an MNNVL Lamport workspace (``rubin_lamport_ar`` layout: three
    buffers of two stages) whose last all-reduce left ``rows`` in the broadcast stage of buffer
    ``dirty``; the rest holds the empty-slot sentinel."""
    m = rows.shape[0]
    stage_bytes = -(-m * HIDDEN * 2 // 1024) * 1024
    bytes_per_buffer = 2 * stage_bytes
    uc = torch.full((3 * bytes_per_buffer // 4,), -(2**31), dtype=torch.int32, device="cuda")
    flags = torch.zeros(9, dtype=torch.int32, device="cuda")
    flags[0], flags[1], flags[2] = (dirty + 1) % 3, dirty, bytes_per_buffer
    start = (dirty * 2 + 1) * stage_bytes // 4
    stage = uc[start : start + m * HIDDEN // 2]
    return uc, flags, stage


@pytest.mark.parametrize("n", [2, 6, 12])
@pytest.mark.parametrize("m", [1, 7, 14, 32])
def test_lamport_addend(m: int, n: int):
    """ADD_PREFIX with the addend polled from a Lamport workspace (no finish: the all-reduce that
    completed last) equals the tensor addend."""
    _k3()
    t = _inputs(m, n, 1.0, seed=3 * m + n)
    addend = t["addend"].clone()
    # A Lamport row never holds -0 (the all-reduce stores +0), so no word is the sentinel.
    addend[addend == 0] = 0
    uc, flags, stage = _lamport_workspace(addend, dirty=2)
    stage.copy_(addend.view(torch.int32).flatten())
    w = _weights(t)
    ops = _k3()
    ref = ops.k3_attn_res_add_rmsnorm(t["layer"], addend, t["block"], *w)
    finish = (0, 0, False, 0, 0)  # no finish: the all-reduce that completed last
    got = ops.k3_attn_res_lamport_add_rmsnorm(
        t["layer"], uc.data_ptr(), flags.data_ptr(), t["block"], *w, *finish, False
    )
    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"output {i}"


@pytest.mark.parametrize("n", [4, 9])
def test_graph_replay_pdl_chain(n: int):
    """DSL -> DSL -> C++ -> DSL chain in one CUDA graph: every PDL hand-off is observed."""
    m = 32
    t = _inputs(m, n, 1.0)
    block4 = t["block"].view(n - 1, m, 1, HIDDEN)
    layer3 = t["layer"].view(m, 1, HIDDEN)
    addend = torch.empty_like(t["addend"])

    def dsl(a):
        out = torch.empty_like(a)
        updated = torch.empty_like(a)
        _k3().run_k3_attn_res(t["layer"], (a,), t["block"], *_weights(t), out, updated)
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


@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("m", [1, 72, 96, 128])
def test_mxfp8_side_output_mid_layout(m: int, n: int):
    """``mx_atom``: the MXFP8 side output's scales in the mid-M FP8 GEMM's layout (one 128-row
    block of tcgen05 scale atoms): the row-major side output's exponents, the other outputs
    unchanged."""
    _k3()
    t = _inputs(m, n, 1.0, seed=m + n)
    args = (t["layer"], t["addend"], t["shared"], t["block"], *_weights(t))
    ops = _k3()
    ref = ops.k3_attn_res_moe_add_rmsnorm_mx(*args)
    got = ops.k3_attn_res_moe_add_rmsnorm_mx(*args, False, True)
    for i in range(2):
        assert _mismatches(got[i], ref[i]) == 0, f"BF16 output {i}"
    assert torch.equal(got[2].view(torch.uint8), ref[2].view(torch.uint8))
    assert tuple(got[3].shape) == ((HIDDEN // 128) * 512,)
    tok = torch.arange(m, device="cuda").view(m, 1)
    grp = torch.arange(HIDDEN // 32, device="cuda").view(1, -1)
    off = (grp // 4) * 512 + (tok % 32) * 16 + (tok // 32) * 4 + grp % 4
    assert torch.equal(got[3][off], ref[3])


@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("m", [129, 144, 168, 192, 256])
def test_mxfp8_side_output_mid_layout_two_blocks(m: int, n: int):
    """``mx_atom_blocks=2``: above 128 rows the side output's scales go to the mid-M FP8 GEMM's
    two 128-row scale blocks per 128-column K tile (``[K / 128][2][512 B]``): the row-major side
    output's exponents, the other outputs unchanged; one block refuses more than 128 rows."""
    _k3()
    t = _inputs(m, n, 1.0, seed=m + n)
    args = (t["layer"], t["addend"], t["shared"], t["block"], *_weights(t))
    ops = _k3()
    ref = ops.k3_attn_res_moe_add_rmsnorm_mx(*args)
    got = ops.k3_attn_res_moe_add_rmsnorm_mx(*args, False, True, 2)
    for i in range(2):
        assert _mismatches(got[i], ref[i]) == 0, f"BF16 output {i}"
    assert torch.equal(got[2].view(torch.uint8), ref[2].view(torch.uint8))
    assert tuple(got[3].shape) == ((HIDDEN // 128) * 2 * 512,)
    tok = torch.arange(m, device="cuda").view(m, 1)
    grp = torch.arange(HIDDEN // 32, device="cuda").view(1, -1)
    off = ((grp // 4) * 2 + tok // 128) * 512 + (tok % 32) * 16 + (tok % 128 // 32) * 4 + grp % 4
    assert torch.equal(got[3][off], ref[3])
    with pytest.raises(ValueError, match="covers at most 128 rows"):
        ops.k3_attn_res_moe_add_rmsnorm_mx(*args, False, True)
