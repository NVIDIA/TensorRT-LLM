# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block verification against a dense per-request reference of Algorithm 2 (Sun et al.
2024, as the serving implementation's kernel runs it)."""

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_verify import (
    dflash2_block_verify,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _dense_reference(p, d, cand, q, coins, final_coin):
    """One request, float64, dense q: the kernel's Algorithm 2 literally."""
    g = d.shape[0]
    vocab = p.shape[-1]
    qd = torch.zeros(g, vocab, dtype=torch.float64)
    qd.scatter_(-1, cand, q)
    tau, scale = 0, 1.0
    prefix, r = 1.0, []
    for i in range(g):
        qv = float(qd[i, d[i]])
        prefix = min(prefix * float(p[i, d[i]]) / qv, 1.0) if 0.0 < qv <= 1.0 else 0.0
        r.append(prefix)
    for step in range(g, 0, -1):
        rs = r[step - 1]
        if coins[step - 1] < rs:
            accept_prob, mass = rs, -1.0
            if step < g and 0.0 < rs < 1.0:
                mass = float(torch.clamp(rs * p[step] - qd[step], min=0).sum())
                accept_prob = mass / (mass + 1.0 - rs)
            if coins[step - 1] < accept_prob:
                tau, scale = step, rs
                break
    if tau == g:
        w = p[g].clone()
    else:
        w = torch.clamp(scale * p[tau] - qd[tau], min=0)
    cdf = w.cumsum(-1)
    idx = int(
        torch.searchsorted(
            cdf, torch.tensor([final_coin * float(cdf[-1])], dtype=torch.float64), right=True
        )
    )
    return tau, min(idx, vocab - 1), r


def _problem(n, g, vocab, k, seed):
    gen = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn(n, g + 1, vocab, generator=gen)
    # Top-k style zeros: drop the lowest third of every row.
    cut = logits.kthvalue(vocab // 3, dim=-1, keepdim=True).values
    p = torch.softmax(logits.masked_fill(logits <= cut, float("-inf")), dim=-1)
    # Distinct candidates per slot: the target's top ids, then random others.
    cand = torch.empty(n, g, k, dtype=torch.int64)
    for b in range(n):
        for i in range(g):
            top = p[b, i].topk(k // 2).indices
            rest = torch.randperm(vocab, generator=gen)
            rest = rest[~torch.isin(rest, top)][: k - k // 2]
            cand[b, i] = torch.cat((top, rest))
    q = torch.softmax(
        1.5 * torch.randn(n, g, k, generator=gen) + 2.0 * (torch.arange(k) < k // 2), dim=-1
    )
    pick = torch.multinomial(q.view(-1, k), 1, generator=gen).view(n, g)
    d = cand.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
    coins = torch.rand(n, g, generator=gen)
    final = torch.rand(n, generator=gen)
    return p, d, cand, q, coins, final


def test_verify_matches_dense_reference():
    n, g, vocab, k = 96, 7, 200, 32
    p, d, cand, q, coins, final = _problem(n, g, vocab, k, seed=g)
    # A few rows draft a token outside the candidates (q = 0 there: rejected at once).
    d[::17, 1] = (d[::17, 1] + 1) % vocab
    tau, bonus = dflash2_block_verify(
        p.to(_DEVICE),
        d.to(_DEVICE),
        cand.to(_DEVICE),
        q.to(_DEVICE),
        coins.to(_DEVICE),
        final.to(_DEVICE),
    )
    tau, bonus = tau.cpu(), bonus.cpu()
    for b in range(n):
        ref_tau, ref_bonus, _ = _dense_reference(
            p[b].double(), d[b], cand[b], q[b].double(), coins[b].tolist(), float(final[b])
        )
        assert int(tau[b]) == ref_tau, (b, int(tau[b]), ref_tau)
        assert int(bonus[b]) == ref_bonus, (b, int(bonus[b]), ref_bonus)
    assert 0 < float(tau.float().mean()) < g
