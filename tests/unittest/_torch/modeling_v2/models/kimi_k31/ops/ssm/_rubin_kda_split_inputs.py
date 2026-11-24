# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the sequence-split KDA split-error tests share (``test_rubin_kda_relay.py`` and
``rubin_multirank/test_rubin_kda_handoff.py``): one request from a cached state, its FP32
reference recurrence, and the error metric."""

import torch
from _rubin_kda_prefill_inputs import HEAD_DIM, LOWER_BOUND, SCALE, stacked_columns

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill_conv import (
    rubin_kda_prefill_conv,
)

WIDTH = 4  # convolution taps
#: The split-error case: one request of SPLIT_TOKENS new tokens from a cached state, SPLIT_HEADS
#: heads, cut into k = 1, 2, 4, 8 equal parts (one per rank).
SPLIT_TOKENS = 2000
SPLIT_HEADS = 16
SPLIT_KS = (1, 2, 4, 8)


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The largest absolute difference relative to the largest magnitude of ``expected``."""
    scale = expected.float().abs().max().clamp_min(1e-12)
    return ((actual.float() - expected.float()).abs().max() / scale).item()


def split_case(seed: int = 3, device="cuda") -> dict:
    """One request of ``SPLIT_TOKENS`` tokens from a cached state at real scales: the fused
    projection, the forget gate, the layer's recurrence weights (slow forgetting, so a start state
    still matters thousands of tokens later), a non-zero cached FP32 state and window; the
    one-rank convolution (``q, k, v``, ``[1, T, H, D]``), the beta logits (``[T, H]``) and the
    FP32 reference recurrence over it (``ref_out``
    ``[T, H, D]``, ``ref_state`` ``[H, D, D]``)."""
    heads, tokens = SPLIT_HEADS, SPLIT_TOKENS
    dim = heads * HEAD_DIM
    gen = torch.Generator(device=device).manual_seed(seed)
    cols = stacked_columns(heads)
    y = torch.randn(tokens, cols["N"], generator=gen, device=device)
    y[:, cols["pad"] :] = 0
    y = y.to(torch.bfloat16)
    g = torch.randn(tokens, dim, generator=gen, device=device).to(torch.bfloat16)
    weight = (torch.randn(3 * dim, WIDTH, generator=gen, device=device) * 0.3).to(torch.bfloat16)
    a_log = torch.empty(heads, device=device).uniform_(1, 16, generator=gen).log()
    dt_bias = torch.empty(dim, device=device).uniform_(-6.9, -2.3, generator=gen)
    state0 = torch.randn(heads, HEAD_DIM, HEAD_DIM, generator=gen, device=device) * 0.05
    conv0 = (torch.randn(3 * dim, WIDTH - 1, generator=gen, device=device) * 0.5).to(torch.bfloat16)
    beta = y[:, cols["b"] : cols["b"] + heads]
    window = conv0.clone().unsqueeze(0)
    starts = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    q, k, v = rubin_kda_prefill_conv(
        y,
        weight,
        window,
        starts,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.bool, device=device),
        tokens,
    )
    ref_out, ref_state = fp32_recurrence(q[0], k[0], v[0], g, beta, state0, a_log, dt_bias)
    return dict(
        heads=heads,
        tokens=tokens,
        y=y,
        g=g,
        weight=weight,
        a_log=a_log,
        dt_bias=dt_bias,
        state0=state0,
        conv0=conv0,
        cols=cols,
        q=q,
        k=k,
        v=v,
        beta=beta,
        ref_out=ref_out,
        ref_state=ref_state,
        ref_window=window[0],
    )


def fp32_recurrence(q, k, v, g, beta, state, a_log, dt_bias):
    """The gated delta rule token by token in FP32 from ``state`` (``[H, V, K]``, V-first) over
    ``[T, H, D]`` q / k / v, ``[T, H D]`` g and ``[T, H]`` beta logits; ``(out, end state)``."""
    heads = q.shape[1]
    qf = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-12) * SCALE
    kf = torch.nn.functional.normalize(k.float(), dim=-1, eps=1e-12)
    vf = v.float()
    gate = g.float().view(-1, heads, HEAD_DIM) + dt_bias.float().view(heads, HEAD_DIM)
    decay = (LOWER_BOUND * torch.sigmoid(a_log.float().exp().view(-1, 1) * gate)).exp()
    bf = beta.float().sigmoid()
    s = state.float().clone()
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for t in range(q.shape[0]):
        s = s * decay[t].unsqueeze(-2)
        err = vf[t] - (s * kf[t].unsqueeze(-2)).sum(-1)
        s = s + bf[t, :, None, None] * err.unsqueeze(-1) * kf[t].unsqueeze(-2)
        out[t] = (s * qf[t].unsqueeze(-2)).sum(-1)
    return out, s
