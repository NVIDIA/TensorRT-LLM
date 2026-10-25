# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`spec_decode/dspark_markov_bias` (TAILBK_SPEC S4L): its two launches around the bias GEMM equal, bit for bit, the
in-tree batched biased-logits lines (prevs cat / casts, markov_prev_embeddings, F.linear, fp32 widen + add), on the
M3.1 TP8 shard (25008 columns, rank 256), K = 3 / 7, batches 1..8, with out-of-vocab anchors (-1, V, a large id),
extreme logits, eagerly and replayed from a CUDA graph; ``supports`` declines what the kernels do not take."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
    dspark_markov_bias as mb,
)
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias

assert torch.cuda.is_available(), "dspark_markov_bias requires a CUDA device"

V = 200064
R = 256
N = 25008


def _weights(seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w1 = (torch.randn(V, R, device="cuda", generator=g) * 0.05).bfloat16()
    w2 = (torch.randn(N, R, device="cuda", generator=g) * 0.05).bfloat16()
    return w1, w2


def _case(seed, B, K, kind="random"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(B, K, N, device="cuda", generator=g) * 4).bfloat16()
    if kind == "extremes":
        pool = torch.tensor([3.0e38, -3.0e38, 2.0**-133, -(2.0**-133), 0.0, -0.0, float("-inf"), 1.0],
                            device="cuda")  # fmt: skip
        logits = pool[
            torch.randint(0, pool.numel(), logits.shape, device="cuda", generator=g)
        ].bfloat16()
    first_prev = torch.randint(0, V, (B,), device="cuda", generator=g)
    tokens = torch.randint(0, V, (B, K), device="cuda", generator=g, dtype=torch.int32)
    if kind == "oov":
        first_prev[0] = -1
        if B > 1:
            first_prev[1] = V
        tokens[0, 0] = -1
        tokens[-1, K - 2] = V + 12345
    return logits, first_prev, tokens


def _reference(logits, first_prev, tokens, w1, w2):
    """The in-tree batched form (tp targets' _biased_shard_logits, fp32 + batched)."""
    shard = logits.float()
    B, K, n = shard.shape
    prevs = torch.cat([first_prev.view(B, 1), tokens[:, : K - 1].long()], dim=1)
    bias = dspark_markov_step_bias(prevs.reshape(-1), w1, w2).view(B, K, n)
    return shard + bias.to(shard.dtype)


def _fused(logits, first_prev, tokens, w1, w2):
    B, K, n = logits.shape
    bias = F.linear(mb.prev_embeddings(first_prev, tokens, w1), w2)
    return mb.widen_add(logits, bias.view(B, K, n))


def _bits_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.view(torch.int32), b.view(torch.int32))
    )


@pytest.mark.parametrize("K", [3, 7])
@pytest.mark.parametrize("B", [1, 2, 8])
@pytest.mark.parametrize("kind", ["random", "oov", "extremes"])
def test_bitwise_in_tree(B, K, kind):
    w1, w2 = _weights()
    for seed in range(3):
        logits, first_prev, tokens = _case(100 * seed + B * 10 + K, B, K, kind)
        assert mb.supports(logits, first_prev, tokens, w1)
        got = _fused(logits, first_prev, tokens, w1, w2)
        want = _reference(logits, first_prev, tokens, w1, w2)
        assert _bits_equal(got, want), f"B {B} K {K} {kind} seed {seed}"


def test_prev_embeddings_rows():
    """Each row is w1[prev] bit for bit, or +0.0 for an anchor outside [0, V); int32 and int64 ids."""
    w1, _ = _weights()
    B, K = 3, 7
    first_prev = torch.tensor([5, -1, V], device="cuda")
    tokens = torch.tensor([[1, 2, 3, -7, V - 1, 0, 9]] * B, device="cuda", dtype=torch.int32)
    for fp, tk in ((first_prev, tokens), (first_prev.int(), tokens.long())):
        e = mb.prev_embeddings(fp, tk, w1)
        for b in range(B):
            for k in range(K):
                prev = int(fp[b]) if k == 0 else int(tk[b, k - 1])
                row = e[b * K + k]
                if 0 <= prev < V:
                    assert torch.equal(row.view(torch.int16), w1[prev].view(torch.int16)), (b, k)
                else:
                    assert torch.equal(
                        row.view(torch.int16), torch.zeros_like(row).view(torch.int16)
                    ), (b, k)


def test_cuda_graph_replay():
    w1, w2 = _weights()
    B, K = 1, 7
    logits, first_prev, tokens = _case(1, B, K)
    _fused(logits, first_prev, tokens, w1, w2)  # compile outside capture
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            out = _fused(logits, first_prev, tokens, w1, w2)
    torch.cuda.current_stream().wait_stream(s)
    for rep in range(6):
        l2, f2, t2 = _case(500 + rep, B, K, "oov" if rep % 2 else "random")
        logits.copy_(l2)
        first_prev.copy_(f2)
        tokens.copy_(t2)
        g.replay()
        torch.cuda.synchronize()
        assert _bits_equal(out, _reference(logits, first_prev, tokens, w1, w2)), f"replay {rep}"


def test_supports_declines():
    w1, _ = _weights()
    logits, first_prev, tokens = _case(3, 2, 7)
    assert mb.supports(logits, first_prev, tokens, w1)
    assert not mb.supports(logits.float(), first_prev, tokens, w1)
    assert not mb.supports(logits.transpose(0, 1), first_prev, tokens, w1)
    assert not mb.supports(logits, first_prev.float(), tokens, w1)
    assert not mb.supports(logits, first_prev, tokens[:, :3], w1)
    assert not mb.supports(logits, first_prev[:1], tokens, w1)
    assert not mb.supports(logits, first_prev, tokens, w1[:, :200])
    assert not mb.supports(logits, first_prev, tokens, w1.float())
