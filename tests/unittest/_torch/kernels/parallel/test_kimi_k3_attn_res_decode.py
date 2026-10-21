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

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
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


def _dsl(t, mode: int, cluster=None):
    addends = ((), (t["addend"],), (t["addend"], t["shared"]))[mode]
    out = torch.empty_like(t["layer"])
    updated = torch.empty_like(t["layer"]) if mode else None
    _kernel().run_attn_res(
        t["layer"], addends, t["block"], *_weights(t), out, updated, cluster=cluster
    )
    return updated, out


def _mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


def assert_close_to_cpp(actual: torch.Tensor, expected: torch.Tensor, what=None):
    a, e = actual.float(), expected.float()
    diff = (a - e).abs()
    # 2^-6 relative is two BF16 ulps (a one-ulp change of the mixture can move the normalized
    # and the weighted rounding by one each). Where the mixture cancels to near zero, the two
    # summation orders differ by fp32 rounding of the (much larger) terms: 2^-7 of the row RMS.
    bound = (
        torch.maximum(a.abs(), e.abs()) * 2.0**-6 + 2.0**-7 * e.pow(2).mean(-1, keepdim=True).sqrt()
    )
    assert bool((diff <= bound).all()), (what, float((diff - bound).max()))
    frac = float((diff > 0).float().mean())
    assert frac <= MAX_DIFF_FRACTION, (what, frac)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 9, 12])
@pytest.mark.parametrize("m", [1, 7, 32, 56, 64])
def test_against_cpp(m: int, n: int, mode: int):
    clusters = (None, 2, 4, 8) if n > 8 else (None, 1, 2, 4, 8)
    for scale in (1.0, 30.0):
        t = _inputs(m, n, scale, seed=m * 13 + n)
        ref_updated, ref_out = _cpp(t, mode)
        for cluster in clusters:
            updated, out = _dsl(t, mode, cluster=cluster)
            if mode:
                assert _mismatches(updated, ref_updated) == 0
            assert_close_to_cpp(out, ref_out, (cluster, scale))


def test_ops():
    """The custom ops run the kernel with its default cluster size."""
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
