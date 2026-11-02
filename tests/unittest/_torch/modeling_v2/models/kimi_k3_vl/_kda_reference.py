# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA prefill inputs and an fp64 token-by-token reference, for the ssm/ prefill entry tests.

The two prefill paths of Kimi K3's KDA layer (``ssm/kda_prefill`` from four 64-token chunks up, ``ssm/chunk_kda``
below) compute the same recurrence; both tests check against this one.
"""

import math
from typing import Dict

import torch

HEAD_DIM = 128  # Kimi K3's KDA head_dim (K = V)
LOWER_BOUND = -5.0  # Kimi K3's gate_lower_bound
SCALE = HEAD_DIM**-0.5
CHUNK = 64


def kimi_k3_inputs(heads: int, tokens: int, gen: torch.Generator, batch_rows: int = 1) -> Dict:
    """q / k / v / g / beta / A_log / dt_bias as Kimi K3's layer hands them to either prefill path."""
    shape = (batch_rows, tokens, heads, HEAD_DIM)

    def unit(t):  # q and k arrive L2-normalized per head (the layer's post-conv step)
        return (t / t.norm(dim=-1, keepdim=True)).to(torch.bfloat16)

    return dict(
        q=unit(torch.randn(*shape, generator=gen, device="cuda")),
        k=unit(torch.randn(*shape, generator=gen, device="cuda")),
        v=(torch.randn(*shape, generator=gen, device="cuda") * 0.5).to(torch.bfloat16),
        g=torch.randn(*shape, generator=gen, device="cuda").to(torch.bfloat16),
        beta=torch.randn(batch_rows, tokens, heads, generator=gen, device="cuda"),
        A_log=torch.log(torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen)),
        dt_bias=torch.empty(heads * HEAD_DIM, device="cuda").uniform_(
            math.log(1e-3), math.log(1e-1), generator=gen
        ),
    )


def cu_seqlens_of(seq_lens, dtype=torch.int64) -> torch.Tensor:
    """Kimi K3's layer passes the metadata's int64 ``query_start_loc_long``."""
    starts = [0]
    for n in seq_lens:
        starts.append(starts[-1] + n)
    return torch.tensor(starts, dtype=dtype, device="cuda")


def recurrent_reference(q, k, v, g, beta, state, A_log, dt_bias):
    """fp64 token-by-token gated delta rule for one sequence: (o [T, H, V], final state [H, V, K]).

    decay = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias))) per key; S <- S * decay;
    S <- S + sigmoid(beta) * (v - S k) k^T; o = S (scale * q).
    """
    heads = q.shape[1]
    s = state.double()
    gate = LOWER_BOUND * torch.sigmoid(
        torch.exp(A_log.double())[:, None] * (g.double() + dt_bias.double().view(heads, HEAD_DIM))
    )
    decay = torch.exp(gate)
    b = torch.sigmoid(beta.double())
    qd, kd, vd = q.double() * SCALE, k.double(), v.double()
    outs = []
    for t in range(q.shape[0]):
        s = s * decay[t][:, None, :]
        err = vd[t] - torch.einsum("hvk,hk->hv", s, kd[t])
        s = s + b[t][:, None, None] * err[:, :, None] * kd[t][:, None, :]
        outs.append(torch.einsum("hvk,hk->hv", s, qd[t]))
    return torch.stack(outs), s


def assert_scaled_close(got: torch.Tensor, ref: torch.Tensor, atol: float) -> None:
    """Within ``atol`` of ``ref``'s largest magnitude (chunked bf16 evaluation against an fp64 recurrence)."""
    scale = ref.abs().max().clamp_min(1e-6)
    torch.testing.assert_close(got.double() / scale, ref.double() / scale, rtol=0.0, atol=atol)
