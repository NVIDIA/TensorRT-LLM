# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block verification: against a dense per-request reference of Algorithm 2 (Sun et al.
2024, as the serving implementation's kernel runs it), its losslessness on a small vocabulary
(sampled selector walk + verification reproduce the target's joint law of the emitted tokens,
with top-k / top-p style zeros), the strict greedy rule, and the engine glue
(context rows, slot store, greedy rows, output layout) on a stub worker."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    dflash2_selector_walk,
    dflash2_uniforms,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_verify import (
    dflash2_block_verify,
    dflash2_greedy_verify,
    dflash2_sample_and_accept,
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


@pytest.mark.parametrize("g", [4, 7])
def test_verify_matches_dense_reference(g):
    n, vocab, k = 96, 200, 32
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


def test_all_accepted_when_draft_equals_target():
    n, g, vocab = 64, 5, 50
    gen = torch.Generator().manual_seed(5)
    p = torch.softmax(torch.randn(n, g + 1, vocab, generator=gen), dim=-1)
    cand = torch.arange(vocab).expand(n, g, vocab).contiguous()
    q = p[:, :g].clone()
    d = torch.multinomial(q.reshape(-1, vocab), 1, generator=gen).view(n, g)
    tau, bonus = dflash2_block_verify(
        p.to(_DEVICE),
        d.to(_DEVICE),
        cand.to(_DEVICE),
        q.to(_DEVICE),
        torch.rand(n, g, generator=gen).to(_DEVICE),
        torch.rand(n, generator=gen).to(_DEVICE),
    )
    assert bool((tau == g).all())


def test_greedy_verify_is_strict():
    logits = torch.randn(6, 5, 30, device=_DEVICE)
    best = logits.argmax(-1)
    d = best[:, :4].clone()
    d[1, 0] += 1
    d[2, 2] += 1
    d[3, 3] += 1
    tau, bonus = dflash2_greedy_verify(logits, d)
    assert tau.tolist() == [4, 0, 2, 3, 4, 4]
    assert bonus.tolist() == [int(best[b, int(tau[b])]) for b in range(6)]


def _markov_lossless_run(n: int, seed: int):
    """Generate L tokens per row by repeated draft (selector walk) + verify steps over a
    Markov target; return the first L tokens, and the accepted drafts per step."""
    vocab, k, g, length, temp = 6, 3, 3, 3, 0.9
    gen = torch.Generator().manual_seed(seed)
    trans = torch.softmax(1.5 * torch.randn(vocab, vocab, generator=gen), dim=-1)
    # Top-p style zeros in the target.
    trans = trans * (trans > 0.08)
    trans = (trans / trans.sum(-1, keepdim=True)).to(_DEVICE)
    cand_tab = torch.stack([torch.randperm(vocab, generator=gen)[:k] for _ in range(g)]).to(_DEVICE)
    scores = (1.2 * torch.randn(g, k, k, generator=gen)).to(_DEVICE)
    scores[0] = scores[0, :1].expand(k, -1)
    torch.manual_seed(seed)
    out = torch.full((n, length + g + 1), -1, dtype=torch.int64, device=_DEVICE)
    lens = torch.zeros(n, dtype=torch.int64, device=_DEVICE)
    last = torch.zeros(n, dtype=torch.int64, device=_DEVICE)
    cols = torch.arange(g + 1, device=_DEVICE)
    taus = []
    for _ in range(length):
        cand = cand_tab.expand(n, g, k)
        d, q, _ = dflash2_selector_walk(
            cand,
            scores.expand(n, g, k, k),
            torch.rand(n, g, device=_DEVICE),
            torch.full((n,), temp, device=_DEVICE),
            torch.zeros(n, dtype=torch.bool, device=_DEVICE),
        )
        prefix = torch.cat((last.unsqueeze(1), d), dim=1)  # [n, g + 1]
        p = trans[prefix]  # p_i = p(. | prefix token i)
        tau, bonus = dflash2_block_verify(
            p,
            d,
            cand,
            q,
            torch.rand(n, g, device=_DEVICE),
            torch.rand(n, device=_DEVICE),
        )
        taus.append(tau)
        emitted = torch.cat((d, d[:, -1:]), dim=1).scatter(1, tau.unsqueeze(1), bonus.unsqueeze(1))
        valid = cols.unsqueeze(0) <= tau.unsqueeze(1)
        pos = (lens.unsqueeze(1) + cols.unsqueeze(0)).clamp_max(out.shape[1] - 1)
        out.scatter_(1, pos, torch.where(valid, emitted, out.gather(1, pos)))
        lens = lens + tau + 1
        last = bonus
    first = out[:, :length].cpu()
    exact = torch.zeros([vocab] * length, dtype=torch.float64)
    t = trans.double().cpu()
    for a in range(vocab):
        for b in range(vocab):
            for c in range(vocab):
                exact[a, b, c] = t[0, a] * t[a, b] * t[b, c]
    return first, exact, torch.cat(taus).float().mean().item()


def test_verify_is_lossless():
    n, vocab = 400000, 6
    first, exact, mean_tau = _markov_lossless_run(n, seed=21)
    assert bool((first >= 0).all())
    index = first[:, 0] * vocab * vocab + first[:, 1] * vocab + first[:, 2]
    counts = torch.bincount(index, minlength=vocab**3).double()
    expected = exact.flatten() * n
    support = expected > 0
    assert float(counts[~support].sum()) == 0.0, "emitted a token the target gives no mass"
    chi2 = float(((counts[support] - expected[support]) ** 2 / expected[support]).sum())
    dof = int(support.sum()) - 1
    assert chi2 < dof + 6 * math.sqrt(2 * dof), (chi2, dof)
    assert mean_tau > 0.2


class _StubWorker:
    """The engine hooks dflash2_sample_and_accept calls, as plain torch."""

    max_draft_len = 4

    def __init__(self, top_k):
        self._draft_model_ref = lambda: SimpleNamespace(
            has_candidate_selector=True, candidate_selector=SimpleNamespace(top_k=top_k)
        )

    def _reshape_draft_tokens_for_accept(self, spec_metadata, num_gens, device):
        return spec_metadata.draft_tokens.reshape(num_gens, spec_metadata.runtime_draft_len)

    def _reshape_logits_for_accept(self, logits, num_contexts, num_gens, spec_metadata):
        return logits

    def _apply_occurrence_penalties(self, logits, *args):
        return logits

    def _commit_occurrence_counts(self, accepted, *args):
        return accepted

    def _apply_force_accepted_tokens(self, num_accepted, *args, **kwargs):
        return num_accepted

    def _sample_tokens_for_batch(self, logits, *args):
        return logits.argmax(-1)

    @staticmethod
    def _zero_padding_rows(logits, *args, **kwargs):
        return logits

    @staticmethod
    def _compute_probs_from_logits(mode, logits, temperatures, top_ks, top_ps, min_ps):
        return torch.softmax(logits / temperatures.unsqueeze(-1), dim=-1)

    def _rng_state_per_request(self, spec_metadata, start, end, slot):
        return spec_metadata.request_seeds[start:end], spec_metadata.request_offsets[
            start:end
        ] + slot


def test_sample_and_accept_glue():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
        dflash2_proposal_store,
    )

    K, k, vocab, num_ctx, num_gens = 4, 8, 64, 2, 5
    bs = num_ctx + num_gens
    worker = _StubWorker(top_k=k)
    gen = torch.Generator().manual_seed(2)
    req_t = torch.tensor([1.0, 1e-5, 1.0, 0.7, 1e-5, 1.3, 1.0], device=_DEVICE)  # 1e-5: greedy
    tok_t = torch.cat((req_t[:num_ctx], req_t[num_ctx:].repeat_interleave(K + 1)))
    spec_metadata = SimpleNamespace(
        is_ragged_verify=False,
        runtime_draft_len=K,
        use_rejection_sampling=True,
        is_all_greedy_sample=False,
        draft_probs=None,
        num_seq_slots=16,
        max_num_requests=8,
        batch_slot_ids=torch.tensor([3, 9, 0, 4, 12, 6, 1], device=_DEVICE),
        request_temperatures=req_t,
        temperatures=tok_t,
        top_ks=torch.zeros_like(tok_t, dtype=torch.int32),
        top_ps=torch.ones_like(tok_t),
        min_ps=torch.zeros_like(tok_t),
        request_seeds=torch.full((bs,), 42, dtype=torch.int64, device=_DEVICE),
        request_offsets=torch.arange(bs, dtype=torch.int64, device=_DEVICE) * 1000,
        rng_accept_slot=10,
        advanced_sampling_mode=None,
    )
    store = dflash2_proposal_store(worker, spec_metadata)
    assert store.num_rows == 17
    # The previous step's proposals: gens 0-3 walked (gen 1 greedy-walked a one-hot q), gen 4's
    # slot holds a stale block.
    gen_slots = spec_metadata.batch_slot_ids[num_ctx:]
    cand = torch.stack(
        [torch.randperm(vocab, generator=gen)[:k] for _ in range(num_gens * K)]
    ).view(num_gens, K, k)
    q = torch.softmax(torch.randn(num_gens, K, k, generator=gen), dim=-1)
    pick = torch.multinomial(q.view(-1, k), 1, generator=gen).view(num_gens, K)
    drafts = cand.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
    stored = drafts.clone()
    stored[4, 0] = (stored[4, 0] + 1) % vocab
    store.store(gen_slots, cand.to(_DEVICE), q.to(_DEVICE), stored.to(_DEVICE))
    store.store(
        spec_metadata.batch_slot_ids[:num_ctx],
        cand[:2].to(_DEVICE),
        q[:2].to(_DEVICE),
        drafts[:2].to(_DEVICE),
    )
    spec_metadata.draft_tokens = drafts.to(_DEVICE).flatten().to(torch.int32)
    logits = torch.randn(num_ctx + num_gens * (K + 1), vocab, generator=gen).to(_DEVICE)
    # Make gen 0's drafts the target's likely tokens, so something gets accepted.
    rows = logits[num_ctx : num_ctx + K + 1]
    rows[torch.arange(K), drafts[0].to(_DEVICE)] += 8.0
    attn_metadata = SimpleNamespace(num_seqs=bs, num_contexts=num_ctx)
    accepted, num_accepted = dflash2_sample_and_accept(worker, logits, attn_metadata, spec_metadata)
    assert accepted.shape == (bs, K + 1) and accepted.dtype == torch.int32
    assert num_accepted.dtype == torch.int32
    # Context rows: their first token; their store rows are invalidated.
    assert torch.equal(accepted[:num_ctx, 0], logits[:num_ctx].argmax(-1).to(torch.int32))
    assert num_accepted[:num_ctx].tolist() == [1, 1]
    assert bool((store.tokens[spec_metadata.batch_slot_ids[:num_ctx]] == -1).all())
    # Generation rows: the direct verification with the same uniforms.
    gl = logits[num_ctx:].view(num_gens, K + 1, vocab)
    p = torch.softmax(gl / tok_t[num_ctx:].view(num_gens, K + 1, 1), dim=-1)
    lc, lq = store.load(gen_slots, drafts.to(_DEVICE))
    u = dflash2_uniforms(
        spec_metadata.request_seeds[num_ctx:], spec_metadata.request_offsets[num_ctx:] + 10, K + 1
    )
    tau, bonus = dflash2_block_verify(p, drafts.to(_DEVICE), lc, lq, u[:, :K], u[:, K])
    g_tau, g_bonus = dflash2_greedy_verify(gl, drafts.to(_DEVICE))
    greedy = req_t[num_ctx:] <= 1e-4
    tau = torch.where(greedy, g_tau, tau)
    bonus = torch.where(greedy, g_bonus, bonus)
    assert torch.equal(num_accepted[num_ctx:].long(), tau + 1)
    for b in range(num_gens):
        t = int(tau[b])
        assert accepted[num_ctx + b, :t].tolist() == drafts[b, :t].tolist()
        assert int(accepted[num_ctx + b, t]) == int(bonus[b])
    assert int(tau[0]) >= 1
    # The stale row verified against the point mass on its drafts.
    assert torch.equal(lq[4, :, 0], torch.ones(K, device=_DEVICE))
