# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's fused KDA ``[f_a | b] -> f_b`` projection (``kda_fab.k31_kda_fab_proj``).

The kernel accumulates in FP32 in another order than the unfused GEMMs the general target runs today
(``bfa = bf16(x @ W_bfa^T)``, then ``bf16(f_a @ W_fb^T)``), so ``beta`` and the BF16 ``f_a`` it feeds to the
second GEMM can differ by one BF16 rounding step; ``forget_gate`` is bounded by the effect of that step. Results
are bit-identical from run to run and across CUDA-graph replays.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the fab projection is a CuTe DSL kernel", allow_module_level=True)
if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
    (10, 7),
):
    pytest.skip("the fab projection runs on SM100 / SM103 / SM107", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.ssm import (  # noqa: E402
    kda_fab as fab,
)

HIDDEN = 7168
FA = 128


def _n1(heads):
    return (FA + heads + 7) // 8 * 8


def _weights(heads, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n1, n2 = _n1(heads), heads * 128
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


def _unfused(x, w_bfa, w_fb, heads):
    """What the general target computes today: cuBLAS [f_a | b] GEMM, slice, cuBLAS f_b GEMM."""
    bfa = torch.nn.functional.linear(x, w_bfa)
    return bfa[:, FA : FA + heads], torch.nn.functional.linear(bfa[:, :FA], w_fb), bfa[:, :FA]


def _check(beta, fg, x, w_bfa, w_fb, heads):
    ref_beta, ref_fg, ref_fa = _unfused(x, w_bfa, w_fb, heads)
    # One BF16 rounding step of the FP32 sum (2^-8 relative), plus FP32 reordering noise.
    torch.testing.assert_close(beta.float(), ref_beta.float(), rtol=2**-7, atol=1e-3)
    # f_a may round to a neighbouring BF16 value; its effect on f_b is bounded by 2^-8 |f_a| @ |W_fb|.
    bound = (ref_fa.float().abs() @ w_fb.float().abs().t()) * 2**-8
    err = (fg.float() - ref_fg.float()).abs()
    assert bool((err <= bound + ref_fg.float().abs() * 2**-7 + 1e-4).all()), err.max()
    close = err <= ref_fg.float().abs() * 2**-7 + 1e-4
    assert close.float().mean().item() > 0.99
    return (
        (beta != ref_beta).float().mean().item(),
        (fg != ref_fg).float().mean().item(),
    )


# TEP16 (6 heads, N1 136, N2 768) and TEP8 (12 heads, N1 144, N2 1536) at verify batches of up to 4 requests
# of 1 + 4..7 rows and the token-tile edges.
@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("m", [1, 7, 8, 9, 16, 20, 28, 32, 40, 64])
def test_matches_the_unfused_gemms(heads, m):
    w_bfa, w_fb = _weights(heads)
    x = _input(m, seed=m)
    if not fab.can_use_k3_kda_fab(x, w_bfa, w_fb, heads):
        pytest.skip(f"{heads} heads x {m} rows: rejected by can_use (the target keeps the GEMMs)")
    beta, fg = fab.k31_kda_fab_proj(x, w_bfa, w_fb, heads)
    assert beta.shape == (m, heads) and fg.shape == (m, heads * 128)
    beta_ne, fg_ne = _check(beta, fg, x, w_bfa, w_fb, heads)
    print(
        f"\n[fab vs cuBLAS] heads={heads} m={m}: beta differing={beta_ne:.2%} f_b differing={fg_ne:.2%}"
    )
    beta2, fg2 = fab.k31_kda_fab_proj(x, w_bfa, w_fb, heads)
    assert torch.equal(beta, beta2) and torch.equal(fg, fg2)


def test_tep4_heads_do_not_fit():
    """24 heads need 152 [f_a | b] rows, past the kernel's 144: TEP4 keeps the GEMMs."""
    w_bfa, w_fb = _weights(24)
    assert not fab.can_use_k3_kda_fab(_input(8), w_bfa, w_fb, 24)


def test_back_to_back_launches():
    """Consecutive launches of different sizes on one stream (shared counter and workspace)."""
    w_bfa, w_fb = _weights(6)
    cases = [_input(m, seed=10 + i) for i, m in enumerate([7, 28, 28, 1, 56, 7])]
    outs = [fab.k31_kda_fab_proj(x, w_bfa, w_fb, 6) for x in cases]
    torch.cuda.synchronize()
    for x, (beta, fg) in zip(cases, outs):
        _check(beta, fg, x, w_bfa, w_fb, 6)


def test_cuda_graph_replay():
    w_bfa, w_fb = _weights(6)
    graphs = []
    for m in (8, 32):
        x = _input(m, seed=m)
        fab.k31_kda_fab_proj(x, w_bfa, w_fb, 6)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fab.k31_kda_fab_proj(x, w_bfa, w_fb, 6)
        graphs.append((m, x, g, out))
    for it in range(3):
        for m, x, g, out in graphs:
            x.copy_(_input(m, seed=100 * it + m))
            g.replay()
            torch.cuda.synchronize()
            beta, fg = fab.k31_kda_fab_proj(x, w_bfa, w_fb, 6)
            assert torch.equal(out[0], beta) and torch.equal(out[1], fg)
            _check(out[0], out[1], x, w_bfa, w_fb, 6)


@pytest.mark.parametrize("m", [5, 8, 10, 12])
def test_launch_without_pdl_gives_the_same_bits(m):
    """TEP8 launches the kernel without PDL (12 heads, 1-2 requests of DFlash2 width 5-6): eager and CUDA-graph
    launches equal the PDL launch bit for bit, also when the two kinds alternate on one stream."""
    w_bfa, w_fb = _weights(12)
    xs = [_input(m, seed=1000 + 10 * m + i) for i in range(3)]
    ref = [fab.k31_kda_fab_proj(x, w_bfa, w_fb, 12) for x in xs]
    for x, (beta, fg) in zip(xs, ref):
        beta2, fg2 = fab.k31_kda_fab_proj(x, w_bfa, w_fb, 12, False)
        assert torch.equal(beta, beta2) and torch.equal(fg, fg2)
    x = xs[0].clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        first = fab.k31_kda_fab_proj(x, w_bfa, w_fb, 12, False)
        second = fab.k31_kda_fab_proj(x, w_bfa, w_fb, 12, True)
    for src, (beta, fg) in zip(xs, ref):
        x.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        for out in (first, second):
            assert torch.equal(out[0], beta) and torch.equal(out[1], fg)


def test_rejects_unsupported_inputs():
    w_bfa, w_fb = _weights(6)
    x = _input(4)
    assert fab.can_use_k3_kda_fab(x, w_bfa, w_fb, 6)
    assert not fab.can_use_k3_kda_fab(_input(fab.MAX_M + 1), w_bfa, w_fb, 6)
    assert not fab.can_use_k3_kda_fab(x.float(), w_bfa, w_fb, 6)
    assert not fab.can_use_k3_kda_fab(x, w_bfa[:128], w_fb, 6)
    assert not fab.can_use_k3_kda_fab(x, w_bfa, w_fb[:, :64].contiguous(), 6)
    assert not fab.can_use_k3_kda_fab(x[:, :7000], w_bfa, w_fb, 6)
    with pytest.raises(ValueError):
        fab.k31_kda_fab_proj(_input(fab.MAX_M + 1), w_bfa, w_fb, 6)
    with pytest.raises(ValueError):
        fab.k31_kda_fab_proj(x.float(), w_bfa, w_fb, 6)
