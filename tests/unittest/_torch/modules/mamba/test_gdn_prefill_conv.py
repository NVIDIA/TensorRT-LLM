# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gdn_prefill_conv_post against the stock GDN prefill pipeline.

extract_transpose_prefill_slice + causal_conv1d_fn + fused_gdn_post_conv: g, beta and the new conv states are
bitwise equal; q / k differ in the FP32 summation order of their L2 norms, and q / k / v in the SiLU, which the
stock CUDA conv takes through tanh.approx above -7 (causalConv1d.cu) and the fused kernel exactly. The linear-space
gate alpha = exp(g) is libdevice expf (torch.exp's bits), a few FP32 ulp from the stock kernel's tl.exp
(fused_gdn_post_conv(g_linear=True)). The value-head counts include ones that are not powers of two (24 / 48 per
rank).
"""

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)
from tensorrt_llm._torch.modules.mamba.gdn_prefill_conv import (
    gdn_prefill_conv_post,
    gdn_prefill_conv_supported,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HEAD_DIM = 128


def _inputs(lens, num_k_heads, num_v_heads, *, pad_seqs=(), extra_tokens=0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    num_tokens = sum(lens) + extra_tokens
    num_seqs = len(lens)
    num_slots = 2 * num_seqs + 1
    ab = torch.randn(num_tokens, 2 * num_v_heads, device="cuda", generator=g).to(torch.bfloat16)
    return dict(
        mixed_qkv=torch.randn(num_tokens, conv_dim, device="cuda", generator=g).to(torch.bfloat16),
        weight=(torch.randn(conv_dim, 1, 4, device="cuda", generator=g) * 0.5).to(torch.bfloat16),
        conv_states=torch.randn(num_slots, conv_dim, 3, device="cuda", generator=g).to(
            torch.bfloat16
        ),
        query_start_loc=torch.tensor(
            [0] + torch.tensor(lens).cumsum(0).tolist(), device="cuda", dtype=torch.int32
        ),
        cache_indices=torch.tensor(
            [-1 if i in pad_seqs else 2 * i + 1 for i in range(num_seqs)],
            device="cuda",
            dtype=torch.int32,
        ),
        has_initial_state=torch.rand(num_seqs, device="cuda", generator=g) < 0.5,
        a=ab[:, :num_v_heads],
        b=ab[:, num_v_heads:],
        A_log=torch.randn(num_v_heads, device="cuda", generator=g),
        dt_bias=torch.randn(num_v_heads, device="cuda", generator=g),
    )


def _stock(x, num_k_heads, num_v_heads):
    conv_states = x["conv_states"].clone()
    mixed = x["mixed_qkv"]
    mixed_t = extract_transpose_prefill_slice(mixed, mixed.shape[0], 0, mixed.shape[1])
    mixed_t = causal_conv1d_fn(
        mixed_t,
        x["weight"].squeeze(1),
        None,
        activation="silu",
        conv_states=conv_states,
        has_initial_state=x["has_initial_state"],
        cache_indices=x["cache_indices"],
        query_start_loc=x["query_start_loc"],
    )
    q, k, v, g, beta = fused_gdn_post_conv(
        mixed_t,
        None,
        x["a"],
        x["b"],
        x["A_log"],
        x["dt_bias"],
        num_k_heads,
        HEAD_DIM,
        num_v_heads,
        HEAD_DIM,
    )
    return q, k, v, g, beta, conv_states


def _stock_alpha(x, num_k_heads, num_v_heads):
    """alpha of the stock pipeline with the FlashInfer gate convention: fused_gdn_post_conv(g_linear=True)."""
    conv_states = x["conv_states"].clone()
    mixed = x["mixed_qkv"]
    mixed_t = extract_transpose_prefill_slice(mixed, mixed.shape[0], 0, mixed.shape[1])
    mixed_t = causal_conv1d_fn(
        mixed_t,
        x["weight"].squeeze(1),
        None,
        activation="silu",
        conv_states=conv_states,
        has_initial_state=x["has_initial_state"],
        cache_indices=x["cache_indices"],
        query_start_loc=x["query_start_loc"],
    )
    return fused_gdn_post_conv(
        mixed_t,
        None,
        x["a"],
        x["b"],
        x["A_log"],
        x["dt_bias"],
        num_k_heads,
        HEAD_DIM,
        num_v_heads,
        HEAD_DIM,
        g_linear=True,
    )[3]


def _fused(x, num_k_heads, num_v_heads, out=None):
    conv_states = x["conv_states"].clone()
    alpha = torch.empty((1, x["mixed_qkv"].shape[0], num_v_heads), device="cuda")
    q, k, v, g, beta = gdn_prefill_conv_post(
        x["mixed_qkv"],
        x["weight"],
        None,
        conv_states,
        x["query_start_loc"],
        x["cache_indices"],
        x["has_initial_state"],
        x["a"],
        x["b"],
        x["A_log"],
        x["dt_bias"],
        num_k_heads,
        HEAD_DIM,
        num_v_heads,
        HEAD_DIM,
        alpha_out=alpha,
        out=out,
    )
    return q, k, v, g, beta, conv_states, alpha


@pytest.mark.parametrize(
    "num_k_heads,num_v_heads",
    [(8, 32), (16, 64), (8, 24), (16, 48)],
    ids=["tp2_397b", "tp1_397b", "48v_tp2", "48v_tp1"],
)
@pytest.mark.parametrize(
    "lens,pad_seqs,extra_tokens",
    [
        ([4096], (), 0),
        ([700, 3, 0, 1, 1300], (2,), 0),
        ([1, 2, 65, 64, 130], (0, 3), 7),
    ],
    ids=["one", "mixed_pad", "short_tail"],
)
def test_matches_stock_pipeline(num_k_heads, num_v_heads, lens, pad_seqs, extra_tokens):
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    assert gdn_prefill_conv_supported(HEAD_DIM, HEAD_DIM, 4, "silu", conv_dim, num_k_heads)
    x = _inputs(lens, num_k_heads, num_v_heads, pad_seqs=pad_seqs, extra_tokens=extra_tokens)
    q_ref, k_ref, v_ref, g_ref, beta_ref, cs_ref = _stock(x, num_k_heads, num_v_heads)
    q, k, v, g, beta, cs, alpha = _fused(x, num_k_heads, num_v_heads)
    # tanh.approx (max relative error 2^-11) in the stock SiLU: |error| <= |pre-activation| / 2 * 2^-11 on top of
    # the bf16 rounding.
    torch.testing.assert_close(v.float(), v_ref.float(), atol=4e-3, rtol=2**-7)
    assert torch.equal(g, g_ref)
    assert torch.equal(beta, beta_ref)
    assert torch.equal(cs, cs_ref)
    assert torch.equal(alpha, torch.exp(g))
    # alpha in (0, 1]: the int32 bit patterns of two such floats differ by their ulp distance.
    alpha_ref = _stock_alpha(x, num_k_heads, num_v_heads)
    ulps = (alpha.view(torch.int32) - alpha_ref.view(torch.int32)).abs().max().item()
    assert ulps <= _ALPHA_MAX_ULPS, ulps
    torch.testing.assert_close(q.float(), q_ref.float(), atol=8e-3, rtol=0)
    torch.testing.assert_close(k.float(), k_ref.float(), atol=8e-3, rtol=0)


# libdevice expf vs the stock kernel's tl.exp (ex2.approx of g * log2(e), whose error grows with |g|) on the gates
# of these inputs: at most 25 ulp measured on GB300, 50 on B200 / B300 and 41 on H100 (CI 70254); 128 ulp is a
# relative difference below 1.6e-5.
_ALPHA_MAX_ULPS = 128


@pytest.mark.parametrize("lens,pad_seqs", [([700, 3, 0, 1, 1300], (2,)), ([1, 65], ())])
def test_out_buffers_take_the_same_values(lens, pad_seqs):
    """``out=`` (the mixer's shared scratch views) is the buffers the kernel writes and returns: the same launches
    give bit-identical q / k / v / g / beta, alpha and conv states as freshly allocated outputs."""
    num_k_heads, num_v_heads = 8, 32
    x = _inputs(lens, num_k_heads, num_v_heads, pad_seqs=pad_seqs)
    ref = _fused(x, num_k_heads, num_v_heads)
    T = x["mixed_qkv"].shape[0]
    # Prefix views of larger flat buffers filled with NaN, as the scratch hands them out.
    out = (
        torch.full((2 * T * num_k_heads * HEAD_DIM,), float("nan"), device="cuda")
        .to(torch.bfloat16)[: T * num_k_heads * HEAD_DIM]
        .view(1, T, num_k_heads, HEAD_DIM),
        torch.full((T * num_k_heads * HEAD_DIM,), float("nan"), device="cuda")
        .to(torch.bfloat16)
        .view(1, T, num_k_heads, HEAD_DIM),
        torch.full((T * num_v_heads * HEAD_DIM,), float("nan"), device="cuda")
        .to(torch.bfloat16)
        .view(1, T, num_v_heads, HEAD_DIM),
        torch.full((3 * T * num_v_heads,), float("nan"), device="cuda")[: T * num_v_heads].view(
            1, T, num_v_heads
        ),
        torch.full((T, num_v_heads), float("nan"), device="cuda").view(1, T, num_v_heads),
    )
    got = _fused(x, num_k_heads, num_v_heads, out=out)
    torch.cuda.synchronize()
    for t, o in zip(got[:5], out):
        assert t.data_ptr() == o.data_ptr()
    for name, a, b in zip(("q", "k", "v", "g", "beta", "conv_states", "alpha"), got, ref):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name


def test_out_buffers_are_checked():
    num_k_heads, num_v_heads = 8, 32
    x = _inputs([5], num_k_heads, num_v_heads)
    T = x["mixed_qkv"].shape[0]
    q = torch.empty((1, T, num_k_heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    v = torch.empty((1, T, num_v_heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    g = torch.empty((1, T, num_v_heads), device="cuda")
    with pytest.raises(ValueError, match="out must be"):
        _fused(x, num_k_heads, num_v_heads, out=(q, q.clone(), v, g, g.to(torch.bfloat16)))


@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 6), (4, 12)])
def test_partial_program_head_counts_unsupported(num_k_heads, num_v_heads):
    """Head counts whose q / k or v channels do not fill whole programs keep the stock kernels."""
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    assert not gdn_prefill_conv_supported(HEAD_DIM, HEAD_DIM, 4, "silu", conv_dim, num_k_heads)
