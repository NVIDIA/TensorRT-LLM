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
"""Correctness of the fused Kimi-K3 KDA ``[f_a | b] -> f_b`` projection (``trtllm::k3_kda_fab_proj``).

The kernel accumulates in FP32 in a different order than the unfused reference
(``bf16(x @ W_bfa^T)``, then ``bf16(f_a @ W_fb^T)``), so ``beta`` and the BF16 ``f_a`` it feeds to
the second GEMM can differ from the reference by one BF16 rounding step; ``forget_gate`` is
checked against the reference with a tolerance covering that, and against the product of the
reference ``f_a`` at a few BF16 ulps. Results must be bit-identical from run to run and across
CUDA-graph replays.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103/SM107 GPU",
)

HIDDEN = 7168
FA = 128
HEADS = 6
N1 = 136
N2 = HEADS * 128
# Covers every heuristic bucket (1, 2-4, 5-8 token tiles, 1-4 UMMA token groups) and edges.
MS = [1, 3, 7, 8, 9, 14, 16, 17, 28, 32, 33, 56, 64, 65, 100, 112, 128, 129, 168, 193, 224, 256]
MS += [257, 300, 384, 385, 448, 511, 512]


def _module():
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_kda_fab

    return k3_kda_fab


def _weights(seed=0, n1=N1, heads=HEADS, n2=N2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w_bfa = torch.randn(n1, HIDDEN, device="cuda", generator=g) * 0.02
    w_bfa[FA + heads :] = 0
    w_fb = torch.randn(n2, FA, device="cuda", generator=g) * 0.05
    return w_bfa.bfloat16(), w_fb.bfloat16()


def _input(m, seed=1, outliers=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, device="cuda", generator=g)
    if outliers:
        chan = torch.ones(HIDDEN, device="cuda")
        chan[torch.randperm(HIDDEN, device="cuda", generator=g)[:16]] = 30.0
        x = x * chan
    return x.bfloat16()


def _reference(x, w_bfa, w_fb, heads=HEADS):
    bfa = (x.float() @ w_bfa.float().t()).bfloat16()
    fa = bfa[:, :FA]
    return bfa[:, FA : FA + heads], (fa.float() @ w_fb.float().t()).bfloat16(), fa


def _check(beta, fg, x, w_bfa, w_fb, heads=HEADS):
    ref_beta, ref_fg, ref_fa = _reference(x, w_bfa, w_fb, heads)
    # One BF16 rounding step of the FP32 sum (2^-8 relative), plus FP32 reordering noise.
    torch.testing.assert_close(beta.float(), ref_beta.float(), rtol=2**-7, atol=1e-3)
    # f_a may round to a neighbouring BF16 value; its effect on f_b is bounded by
    # 2^-8 * |f_a| @ |W_fb| per element.
    bound = (ref_fa.float().abs() @ w_fb.float().abs().t()) * 2**-8
    err = (fg.float() - ref_fg.float()).abs()
    assert bool((err <= bound + ref_fg.float().abs() * 2**-7 + 1e-4).all()), err.max()
    # Nearly every element must agree with the reference to one BF16 ulp.
    close = err <= ref_fg.float().abs() * 2**-7 + 1e-4
    assert close.float().mean().item() > 0.99


@pytest.mark.parametrize("m", MS)
def test_matches_reference(m):
    mod = _module()
    w_bfa, w_fb = _weights()
    x = _input(m)
    beta, fg = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, HEADS)
    assert beta.shape == (m, HEADS) and fg.shape == (m, N2)
    _check(beta, fg, x, w_bfa, w_fb)
    # Deterministic: a second launch is bit-identical.
    beta2, fg2 = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, HEADS)
    assert torch.equal(beta, beta2) and torch.equal(fg, fg2)
    assert mod.can_use_k3_kda_fab(x, w_bfa, w_fb, HEADS)


@pytest.mark.parametrize("heads,n1,n2", [(1, 136, N2), (8, 136, N2), (16, 144, N2), (8, 136, 1024)])
def test_other_shapes(heads, n1, n2):
    mod = _module()
    w_bfa, w_fb = _weights(seed=3, n1=n1, heads=heads, n2=n2)
    for m in (5, 40, 150):
        x = _input(m, seed=m)
        if not mod.can_use_k3_kda_fab(x, w_bfa, w_fb, heads):
            # Larger W_fb blocks than fit in shared memory are rejected, never mis-run.
            assert n2 > N2
            continue
        beta, fg = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, heads)
        _check(beta, fg, x, w_bfa, w_fb, heads)


def test_small_hidden_size():
    """K below the tuned split count (the KDA module tests use hidden size 512)."""
    w_bfa = (torch.randn(N1, 512, device="cuda") * 0.05).bfloat16()
    w_fb = (torch.randn(1024, FA, device="cuda") * 0.05).bfloat16()
    for m in (3, 6, 24):
        x = torch.randn(m, 512, device="cuda").bfloat16()
        beta, fg = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, 8)
        _check(beta, fg, x, w_bfa, w_fb, 8)


def test_back_to_back_launches():
    """Consecutive PDL launches of different sizes on one stream (shared counters/workspace)."""
    w_bfa, w_fb = _weights()
    sizes = [7, 224, 28, 512, 28, 1, 384, 112, 56, 7]
    cases = [(m, _input(m, seed=10 + i)) for i, m in enumerate(sizes)]
    outs = [torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, HEADS) for _, x in cases]
    torch.cuda.synchronize()
    for (_, x), (beta, fg) in zip(cases, outs):
        _check(beta, fg, x, w_bfa, w_fb)


def test_cuda_graph_replay():
    w_bfa, w_fb = _weights()
    graphs = []
    for m in (7, 28, 224, 384):
        x = _input(m, seed=m)
        eager = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, HEADS)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, HEADS)
        graphs.append((m, x, g, out, eager))
    # Interleave replays of different sizes with fresh inputs and eager launches.
    for it in range(4):
        for m, x, g, out, _ in graphs:
            x.copy_(_input(m, seed=100 * it + m))
            g.replay()
            torch.cuda.synchronize()
            beta, fg = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, HEADS)
            assert torch.equal(out[0], beta) and torch.equal(out[1], fg)
            _check(out[0], out[1], x, w_bfa, w_fb)


def test_concurrent_with_main_stream_gemm():
    """Fork/join next to a large GEMM on the main stream, as in the KDA verify projection."""
    w_bfa, w_fb = _weights()
    w_big = torch.randn(3072, HIDDEN, device="cuda").bfloat16() * 0.02
    aux = torch.cuda.Stream()
    for m in (7, 56, 224, 384, 512):
        x = _input(m, seed=m)
        fork, join = torch.cuda.Event(), torch.cuda.Event()

        def body():
            fork.record()
            big = x @ w_big.t()
            with torch.cuda.stream(aux):
                aux.wait_event(fork)
                res = torch.ops.trtllm.k3_kda_fab_proj(x, w_bfa, w_fb, HEADS)
                join.record()
            torch.cuda.current_stream().wait_event(join)
            return big, res

        body()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _, (beta, fg) = body()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        _check(beta, fg, x, w_bfa, w_fb)


def test_rejects_unsupported_inputs():
    mod = _module()
    w_bfa, w_fb = _weights()
    x = _input(4)
    assert not mod.can_use_k3_kda_fab(_input(mod.MAX_M + 1), w_bfa, w_fb, HEADS)
    assert not mod.can_use_k3_kda_fab(x.float(), w_bfa, w_fb, HEADS)
    assert not mod.can_use_k3_kda_fab(x, w_bfa[:128], w_fb, HEADS)
    assert not mod.can_use_k3_kda_fab(x, w_bfa, w_fb[:, :64].contiguous(), HEADS)
    assert not mod.can_use_k3_kda_fab(x, w_bfa, w_fb, 9)
    assert not mod.can_use_k3_kda_fab(x[:, :7000], w_bfa, w_fb, HEADS)
    assert mod.can_use_k3_kda_fab(x, w_bfa, w_fb, HEADS)


def _run_cfg(mod, x, w_bfa, w_fb, cfg):
    beta = torch.full((x.shape[0], HEADS), -7.0, device="cuda").bfloat16()
    fg = torch.full((x.shape[0], N2), -7.0, device="cuda").bfloat16()
    mod.run_k3_kda_fab(x, w_bfa, w_fb, HEADS, beta, fg, config=cfg)
    return beta, fg


@pytest.mark.parametrize("staged", [1, 2])
@pytest.mark.parametrize("m", [65, 100, 128, 129, 200, 256, 300, 384, 448, 512])
def test_staged_tail_bitwise(staged, m):
    """The two-level tail reproduces the one-level tail bit for bit (same sums and MMAs)."""
    mod = _module()
    w_bfa, w_fb = _weights(seed=5)
    n_tiles = (m + 7) // 8
    base = mod.default_config(n_tiles, N2, HIDDEN, staged=0)
    cfg = mod.default_config(n_tiles, N2, HIDDEN, staged=staged)
    assert cfg.staged == staged and cfg.s == base.s
    for seed in range(2):
        x = _input(m, seed=100 + seed)
        ref = _run_cfg(mod, x, w_bfa, w_fb, base)
        out = _run_cfg(mod, x, w_bfa, w_fb, cfg)
        torch.cuda.synchronize()
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    _check(out[0], out[1], x, w_bfa, w_fb)


@pytest.mark.parametrize("staged", [1, 2])
def test_staged_tail_cuda_graph_mixed(staged):
    """Staged launches of several sizes interleaved with one-level launches under graph replay
    (shared counters, workspace halves and their sentinel refill)."""
    mod = _module()
    w_bfa, w_fb = _weights(seed=6)
    cases = []
    for i, m in enumerate((256, 100, 384, 7, 256, 512)):
        x = _input(m, seed=200 + i)
        n_tiles = (m + 7) // 8
        st = staged if i % 2 == 0 else 0
        cfg = mod.default_config(n_tiles, N2, HIDDEN, staged=st)
        cases.append(
            (
                x,
                cfg,
                _run_cfg(mod, x, w_bfa, w_fb, mod.default_config(n_tiles, N2, HIDDEN, staged=0)),
            )
        )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [_run_cfg(mod, x, w_bfa, w_fb, cfg) for x, cfg, _ in cases]
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        for (x, cfg, ref), out in zip(cases, outs):
            assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
