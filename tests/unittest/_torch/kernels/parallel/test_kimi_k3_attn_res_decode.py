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
"""The Kimi-K3 decode attention-residual kernel (``kimi_k3_attn_res``, CuTe DSL) against the C++
per-token attention-residual ops.

The kernel sums the statistics in another order than the C++ kernels: its updated prefix sum is
bit-identical and its normalized output may differ from the C++ op by one or two BF16 rounding
steps in a small fraction of the elements. It runs inside a CUDA graph behind and in front of PDL
producers and consumers (DSL and C++).
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not torch.cuda.is_available()
    or not (
        torch.cuda.get_device_capability() in ((10, 0), (10, 3))
        or (torch.cuda.get_device_capability() == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE)
    ),
    reason="needs an SM100/SM103 GPU, or SM107 with a Rubin-capable CuTe DSL",
)

HIDDEN = 7168
EPS = 1e-6
OUT_EPS = 1e-5
# Largest fraction of output elements that may differ from the C++ op at all.
MAX_DIFF_FRACTION = 2e-2


def _kernel():
    import tensorrt_llm  # noqa: F401  (registers the C++ trtllm ops)
    from tensorrt_llm._torch.kernels.blackwell import kimi_k3_attn_res

    return kimi_k3_attn_res


def _inputs(m: int, n: int, scale: float, seed: int = 0, hidden: int = HIDDEN):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return dict(
        block=rn(n - 1, m, hidden, sc=scale),
        layer=rn(m, hidden, sc=scale),
        addend=rn(m, hidden, sc=0.5 * scale),
        shared=rn(m, hidden, sc=0.5 * scale),
        proj=rn(hidden, sc=0.05),
        norm_w=(1.0 + rn(hidden, sc=0.1).float()).to(torch.bfloat16),
        out_w=(1.0 + rn(hidden, sc=0.1).float()).to(torch.bfloat16),
    )


def _weights(t):
    return (t["proj"], t["norm_w"], t["out_w"], EPS, OUT_EPS)


def _cpp(t, mode: int):
    """``(updated prefix or None, output)`` of the C++ path for add mode 0 / 1 / 2."""
    _kernel()
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
        # The MoE layer's residual update: two BF16 adds.
        updated = row = t["layer"] + (t["addend"] + t["shared"])
    out = torch.ops.trtllm.attn_res_rmsnorm_fwd(row.view(m, 1, HIDDEN), block, *_weights(t))
    return updated, out.view(m, HIDDEN)


def _torch_reference(t, mode: int):
    """``(updated prefix or None, output)`` in FP32 PyTorch for any hidden size."""
    updated = None
    row = t["layer"]
    if mode == 1:
        updated = row = t["layer"] + t["addend"]
    elif mode == 2:
        updated = row = t["layer"] + (t["addend"] + t["shared"])
    v = torch.cat((t["block"], row.unsqueeze(0))).float()
    rsigma = torch.rsqrt(v.pow(2).mean(-1) + EPS)
    q = t["proj"].float() * t["norm_w"].float()
    p = torch.softmax(rsigma * (v @ q), dim=0)
    mixed = (p.unsqueeze(-1) * v).sum(0).to(torch.bfloat16).float()
    normed = (mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + OUT_EPS)).to(torch.bfloat16)
    return updated, (normed.float() * t["out_w"].float()).to(torch.bfloat16)


def _dsl(t, mode: int):
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    out = torch.empty_like(t["layer"])
    updated = torch.empty_like(t["layer"]) if mode else None
    _kernel().run_attn_res(t["layer"], addends, t["block"], *_weights(t), out, updated)
    return updated, out


def _mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


def _assert_within_two_ulps(actual: torch.Tensor, expected: torch.Tensor, what=None):
    a, e = actual.float(), expected.float()
    diff = (a - e).abs()
    # 2^-6 relative is two BF16 ulps (a one-ulp change of the mixture can move the normalized
    # and the weighted rounding by one each). Where the mixture cancels to near zero, the two
    # summation orders differ by fp32 rounding of the (much larger) terms: 2^-7 of the row RMS.
    bound = (
        torch.maximum(a.abs(), e.abs()) * 2.0**-6 + 2.0**-7 * e.pow(2).mean(-1, keepdim=True).sqrt()
    )
    assert bool((diff <= bound).all()), (what, float((diff - bound).max()))
    return diff


def assert_close_to_cpp(actual: torch.Tensor, expected: torch.Tensor, what=None):
    diff = _assert_within_two_ulps(actual, expected, what)
    frac = float((diff > 0).float().mean())
    assert frac <= MAX_DIFF_FRACTION, (what, frac)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 9, 12])
@pytest.mark.parametrize("m", [1, 7, 16, 17, 32, 56, 64])
def test_against_cpp(m: int, n: int, mode: int):
    """Both cluster sizes :func:`cluster_size` picks (eight CTAs for up to 16 tokens with four or
    more candidates, else four)."""
    for scale in (1.0, 30.0):
        t = _inputs(m, n, scale, seed=m * 13 + n)
        ref_updated, ref_out = _cpp(t, mode)
        updated, out = _dsl(t, mode)
        if mode:
            assert _mismatches(updated, ref_updated) == 0
        assert_close_to_cpp(out, ref_out, scale)


def test_hidden_sizes():
    k = _kernel()
    assert all(k.supports_hidden_size(h) for h in (256, 4096, 7168, 8192))
    assert not any(k.supports_hidden_size(h) for h in (0, 128, 7040, 8448))


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("hidden", [4096, 7168, 8192])
@pytest.mark.parametrize("m, n", [(3, 5), (32, 9)])
def test_against_torch_reference_across_hidden_sizes(m: int, n: int, hidden: int, mode: int):
    """The hidden size is a compile-time parameter (strides, ``1 / H`` of the mean, MXFP8
    scale atoms); the C++ ops cover only 7168, so an FP32 PyTorch reference covers the rest."""
    t = _inputs(m, n, 1.0, seed=hidden + m, hidden=hidden)
    ref_updated, ref_out = _torch_reference(t, mode)
    updated, out = _dsl(t, mode)
    if mode:
        assert _mismatches(updated, ref_updated) == 0
    _assert_within_two_ulps(out, ref_out, hidden)


def test_ops():
    """The custom ops run the kernel."""
    t = _inputs(7, 9, 1.0)
    w = _weights(t)
    out = torch.ops.trtllm.kimi_k3_attn_res_rmsnorm(t["layer"], t["block"], *w)
    assert _mismatches(out, _dsl(t, 0)[1]) == 0
    updated, out = torch.ops.trtllm.kimi_k3_attn_res_add_rmsnorm(
        t["layer"], t["addend"], t["block"], *w
    )
    ref_updated, ref_out = _dsl(t, 1)
    assert _mismatches(updated, ref_updated) == 0 and _mismatches(out, ref_out) == 0
    prefix, out = torch.ops.trtllm.kimi_k3_attn_res_moe_add_rmsnorm(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    ref_prefix, ref_out = _dsl(t, 2)
    assert _mismatches(prefix, ref_prefix) == 0 and _mismatches(out, ref_out) == 0


def r128c4_offsets(m: int, k: int) -> torch.Tensor:
    """Byte offset of the UE8M0 scale of (row r, K32 group c) in the R128c4 layout, ``[m, k / 32]``."""
    r = torch.arange(m, device="cuda").unsqueeze(1)
    c = torch.arange(k // 32, device="cuda").unsqueeze(0)
    return ((r // 128) * (k // 128) + c // 4) * 512 + (r % 32) * 16 + (r % 128) // 32 * 4 + c % 4


@pytest.mark.parametrize("n", [2, 6, 12])
@pytest.mark.parametrize("m", [1, 7, 14, 32, 64])
def test_mxfp8_side_output(m: int, n: int):
    """The MXFP8 op returns the plain op's BF16 results and the MXFP8 form of the normalized
    output that the FP8 low-M GEMM's in-kernel quantization makes of it."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    t = _inputs(m, n, 1.0, seed=m + n)
    w = _weights(t)
    ops = torch.ops.trtllm
    ref = ops.kimi_k3_attn_res_moe_add_rmsnorm(t["layer"], t["addend"], t["shared"], t["block"], *w)
    prefix, out, xq, xs = ops.kimi_k3_attn_res_moe_add_rmsnorm_mxfp8(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    assert _mismatches(prefix, ref[0]) == 0 and _mismatches(out, ref[1]) == 0
    ref_q, ref_s = low_m.mxfp8_quantize_reference(ref[1])
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(xs[r128c4_offsets(m, HIDDEN)], ref_s)


@pytest.mark.parametrize("hidden", [4096, 8192])
def test_mxfp8_side_output_other_hidden_sizes(hidden: int):
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    m, n = 9, 6
    t = _inputs(m, n, 1.0, seed=hidden, hidden=hidden)
    w = _weights(t)
    ops = torch.ops.trtllm
    prefix, out, xq, xs = ops.kimi_k3_attn_res_moe_add_rmsnorm_mxfp8(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    assert xs.numel() == _kernel().mxfp8_scale_numel(m, hidden)
    ref_q, ref_s = low_m.mxfp8_quantize_reference(out)
    assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(xs[r128c4_offsets(m, hidden)], ref_s)


def test_can_run():
    k = _kernel()
    t = _inputs(k.MAX_TOKENS, 4, 1.0)
    w = (t["proj"], t["norm_w"], t["out_w"])
    assert k.can_run(t["layer"], (), t["block"], *w)
    assert k.can_run(t["layer"], (t["addend"], t["shared"]), t["block"], *w)
    big = torch.zeros((k.MAX_TOKENS + 1, HIDDEN), device="cuda", dtype=torch.bfloat16)
    assert not k.can_run(big, (), torch.zeros((3, *big.shape), **_like(big)), *w)
    assert not k.can_run(t["layer"].float(), (), t["block"], *w)
    assert not k.can_run(t["layer"], (), t["block"][:, :, :4096], *w)
    narrow = _inputs(k.MAX_TOKENS, 4, 1.0, hidden=4096)
    assert k.can_run(
        narrow["layer"], (), narrow["block"], narrow["proj"], narrow["norm_w"], narrow["out_w"]
    )
    assert not k.can_run(narrow["layer"], (), narrow["block"], *w)
    assert not k.can_run(t["layer"], (), torch.zeros((12, *t["layer"].shape), **_like(big)), *w)


def _like(x):
    return dict(device=x.device, dtype=x.dtype)


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
        _kernel().run_attn_res(t["layer"], (a,), t["block"], *_weights(t), out, updated)
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


def test_can_run_rejects_misaligned_rows():
    """The kernel reads 16-byte vectors: a view off a 16-byte boundary is rejected (and the op
    raises instead of launching)."""
    k = _kernel()
    m = 8
    t = _inputs(m, 4, 1.0)
    w = (t["proj"], t["norm_w"], t["out_w"])
    storage = torch.empty(m * HIDDEN + 1, device="cuda", dtype=torch.bfloat16)
    misaligned = storage[1:].view(m, HIDDEN)
    misaligned.copy_(t["layer"])
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16
    assert not k.can_run(misaligned, (), t["block"], *w)
    with pytest.raises(ValueError, match="unsupported inputs"):
        torch.ops.trtllm.kimi_k3_attn_res_rmsnorm(misaligned, t["block"], *_weights(t))


@pytest.mark.parametrize("m", [1, 16, 64])
def test_snapshots_after_wait_matches(m: int):
    """The variant that loads the snapshots after the PDL wait computes the same bits."""
    t = _inputs(m, 9, 1.0, seed=m)
    w = _weights(t)
    ops = torch.ops.trtllm
    ref = ops.kimi_k3_attn_res_rmsnorm(t["layer"], t["block"], *w)
    out = ops.kimi_k3_attn_res_rmsnorm(t["layer"], t["block"], *w, True)
    assert _mismatches(out, ref) == 0


@pytest.mark.parametrize("n", [4, 9])
def test_graph_replay_bank_scattered_right_before_the_mix(n: int):
    """helix_a2a's first token-sliced layer: the copy that slices this rank's rows out of the
    snapshot bank is the kernel launched right before the pre-attention mix, inside a CUDA graph
    (PDL edge). With ``snapshots_after_wait`` the mix reads the rows that copy wrote."""
    world, rank, m = 4, 2, 16
    full = _inputs(world * m, n, 1.0, seed=n)
    w = _weights(full)
    bank = torch.empty_like(full["block"])
    rows = torch.empty_like(full["layer"])
    outs = []

    def body():
        # KimiK3HelixA2A.scatter_rows: a view of the hidden states, a copy of the bank.
        layer = rows.narrow(0, rank * m, m).contiguous()
        block = bank.narrow(1, rank * m, m).contiguous()
        outs[:] = [torch.ops.trtllm.kimi_k3_attn_res_rmsnorm(layer, block, *w, True)]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bank.copy_(full["block"])
        rows.copy_(full["layer"])
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for seed in range(4):
        step = _inputs(world * m, n, 1.0, seed=100 + seed)
        bank.copy_(step["block"])
        rows.copy_(step["layer"])
        graph.replay()
        torch.cuda.synchronize()
        # The graph's weights are ``full``'s; only the bank and the rows change per replay.
        local = dict(full)
        local["block"] = step["block"][:, rank * m : (rank + 1) * m].contiguous()
        local["layer"] = step["layer"][rank * m : (rank + 1) * m].contiguous()
        ref = _dsl(local, 0)[1]
        assert _mismatches(outs[0], ref) == 0, f"replay {seed}"
        _assert_within_two_ulps(outs[0], _torch_reference(local, 0)[1])


def pdl_switch_outputs() -> list:
    """Outputs of the decode attention-residual ops and of the FP8 low-M GEMMs fed by the MXFP8
    side output, on fixed inputs (compared with and without ``TRTLLM_ENABLE_PDL``)."""
    from torch import nn

    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    m, n = 12, 9
    t = _inputs(m, n, 1.0, seed=7)
    w = _weights(t)
    ops = torch.ops.trtllm
    results = [ops.kimi_k3_attn_res_rmsnorm(t["layer"], t["block"], *w)]
    results.append(ops.kimi_k3_attn_res_rmsnorm(t["layer"], t["block"], *w, True))
    results.extend(ops.kimi_k3_attn_res_add_rmsnorm(t["layer"], t["addend"], t["block"], *w))
    prefix, out, xq, xs = ops.kimi_k3_attn_res_moe_add_rmsnorm_mxfp8(
        t["layer"], t["addend"], t["shared"], t["block"], *w
    )
    # The scale buffer is padded to 128 rows; only this batch's scales are written.
    results.extend((prefix, out, xq.view(torch.uint8), xs[r128c4_offsets(m, HIDDEN)]))
    torch.manual_seed(3)
    dense = nn.Linear(HIDDEN, 3072, bias=False, dtype=torch.bfloat16, device="cuda")
    proj = _Fp8BlockScaleWeightReadLinear.from_linear(dense)
    proj.post_load_weights()
    results.append(proj(out))
    if proj.takes_prequantized_input(m):
        results.append(proj.forward_prequantized(xq, xs))
    torch.cuda.synchronize()
    return [r.cpu() for r in results]


def test_pdl_disabled_matches(tmp_path):
    """``TRTLLM_ENABLE_PDL=0`` launches the decode kernel and the low-M GEMMs without PDL and
    yields the same bits."""
    import os
    import subprocess
    import sys
    import textwrap

    result = tmp_path / "no_pdl.pt"
    script = textwrap.dedent(
        f"""
        import sys
        import torch
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from tensorrt_llm._torch.kernels.blackwell import utils
        assert not utils.TRTLLM_ENABLE_PDL
        import test_kimi_k3_attn_res_decode as t
        torch.save(t.pdl_switch_outputs(), {str(result)!r})
        """
    )
    from tensorrt_llm._torch.kernels.blackwell import utils

    if not utils.TRTLLM_ENABLE_PDL:
        pytest.skip("compares against PDL launches; TRTLLM_ENABLE_PDL=0 is set")
    env = dict(os.environ, TRTLLM_ENABLE_PDL="0")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
    expected = pdl_switch_outputs()
    actual = torch.load(result)
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert torch.equal(a, e), i
