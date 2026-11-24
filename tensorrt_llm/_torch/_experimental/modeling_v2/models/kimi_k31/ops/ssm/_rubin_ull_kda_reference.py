# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA layer core of the ``ssm`` entries in FP64, token by token: what MR8's ``KdaSplit`` (and
the fused heads around it) compute from the packed in-projection rows.

Per request, head ``h`` and token ``t`` (``D = 128``; ``x`` the packed BF16 row
``[q | k | v | g | f_a | beta]`` of ``4 H D + 128 + H`` columns):

* short convolution: ``c_t = sum_i w[s, i] * window[t + i]`` over the window
  ``[3 taps of the conv page, x_0 .. x_t]`` (taps at the accepted position), then SiLU;
* ``q = scale * q / sqrt(|q|^2 + 1e-6)``, ``k = k / sqrt(|k|^2 + 1e-6)``;
* forget gate: ``f = bf16(f_a . f_b^T)`` (the kernels round the FP32 product to BF16),
  ``decay = exp(lower * sigmoid(exp(A_log[h]) * (f + dt_bias)))`` per key;
* ``S <- S * decay``; ``S <- S + sigmoid(beta) * (v - S k) k^T``; ``o = S q``;
* ``out = o / sqrt(mean(o^2) + eps) * norm * sigmoid(g)``.

The conv page of a request is rewritten to ``[taps[1], taps[2], x_0 .. x_{q-1}]`` (q + 2 columns).
"""

from __future__ import annotations

import torch

L2_EPS = 1e-6  # the kernels' q / k normalization epsilon (a constant, not the norm epsilon)


def scalar(value) -> float:
    """A Python float of one of ``PreparedKda.scalars`` (CuTe DSL ``Float32``)."""
    return float(getattr(value, "value", value))


def kda_core_reference(core, projected, caches, slots, accepted, tokens):
    """FP64 KDA core of the packed rows ``projected`` [requests * tokens, pitch].

    ``core`` is MR8's ``PreparedKda`` (its ``weights``, ``scalars``, ``heads``, ``dim`` and
    ``conv_dim_first`` are read); ``caches`` its (conv, state) views. With ``accepted``
    (verify) ``slots`` is [requests, >= tokens] and request ``b`` reads state
    ``slots[b, accepted[b] - 1]`` and conv page ``slots[b, 0]``; without (q = 1) ``slots`` is
    [requests] and the request reads and writes slot ``slots[b]``.

    Returns ``(out, states, pages)``: ``out`` FP64 [requests * tokens, dim] (zero rows for
    padded requests, whose read slot is <= 0), ``states[b]`` FP64 [tokens, H, 128, 128] (the
    state after each token) and ``pages[b]`` the rewritten conv page [3 dim, tokens + 2]
    (both None for padded requests). Nothing is written.
    """
    conv_w, _bias, f_b, a_log, dt_bias, norm = core.weights
    heads, dim = core.heads, core.dim
    lower, scale, eps = (scalar(s) for s in core.scalars)
    conv, state = caches[:2]
    conv = conv if core.conv_dim_first else conv.transpose(-1, -2)
    requests = projected.shape[0] // tokens
    rows = projected.double()
    weights = conv_w.double().permute(0, 2, 1).reshape(3 * dim, -1)  # [3 dim, width]
    width = weights.shape[1]
    f_b, a_log = f_b.double(), a_log.double()
    dt_bias, norm = dt_bias.double().view(heads, 128), norm.double()
    out = torch.zeros(projected.shape[0], dim, dtype=torch.float64, device=projected.device)
    states, pages = [], []
    for b in range(requests):
        if accepted is None:
            slot = int(slots[b])
            conv_slot, offset = slot, 0
        else:
            offset = int(accepted[b]) - 1
            slot, conv_slot = int(slots[b, offset]), int(slots[b, 0])
        if slot <= 0:
            states.append(None)
            pages.append(None)
            continue
        x = rows[b * tokens : (b + 1) * tokens]
        taps = conv[conv_slot, :, offset : offset + width - 1].double()  # [3 dim, width - 1]
        window = torch.cat([taps, x[:, : 3 * dim].T], dim=1)
        conv_out = torch.stack(
            [(window[:, t : t + width] * weights).sum(1) for t in range(tokens)]
        )  # [tokens, 3 dim]
        act = conv_out * torch.sigmoid(conv_out)
        q, k, v = (act[:, i * dim : (i + 1) * dim].view(tokens, heads, 128) for i in range(3))
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2_EPS) * scale
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2_EPS)
        g_out = x[:, 3 * dim : 4 * dim].view(tokens, heads, 128)
        f_a = x[:, 4 * dim : 4 * dim + 128]
        beta = torch.sigmoid(x[:, 4 * dim + 128 : 4 * dim + 128 + heads])  # [tokens, heads]
        f = (f_a @ f_b.T).float().bfloat16().double().view(tokens, heads, 128)
        gate = torch.exp(a_log)[None, :, None] * (f + dt_bias)
        decay = torch.exp(lower * torch.sigmoid(gate))
        s = state[slot].double()  # [heads, value, key]
        after = []
        for t in range(tokens):
            s = s * decay[t][:, None, :]
            err = v[t] - torch.einsum("hvk,hk->hv", s, k[t])
            s = s + beta[t][:, None, None] * err[:, :, None] * k[t][:, None, :]
            o = torch.einsum("hvk,hk->hv", s, q[t])
            rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
            out[b * tokens + t] = (o * rstd * norm * torch.sigmoid(g_out[t])).reshape(dim)
            after.append(s)
        states.append(torch.stack(after))
        pages.append(torch.cat([taps[:, 1:], x[:, : 3 * dim].T], dim=1))
    return out, states, pages
