# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 sampled selector walk: the counter-based uniforms, the walk against a per-row reference
and against the vectorized reference walk of the DFlash2 serving implementation, its path
distribution, greedy rows, and the slot-indexed proposal store."""

import math

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    DFlash2ProposalStore,
    dflash2_score_edges,
    dflash2_selector_walk,
    dflash2_uniforms,
    dflash2_walk_candidate_paths,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MASK = (1 << 64) - 1


def _mix64_py(z: int) -> int:
    z &= _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _uniforms_py(seed: int, offset: int, count: int) -> list:
    key = _mix64_py(_mix64_py(seed) ^ (offset & _MASK))
    return [
        (_mix64_py(key + (j + 1) * 0x9E3779B97F4A7C15) >> 40) / float(1 << 24) for j in range(count)
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_score_and_walk_are_the_engine_functions(dtype):
    """The selector's edge scores and greedy walk are bitwise the engine DFlash2 functions."""
    from tensorrt_llm._torch.models.modeling_dflash import dflash2_score_edges as engine_score_edges
    from tensorrt_llm._torch.models.modeling_dflash import (
        dflash2_walk_candidate_paths as engine_walk,
    )

    g = torch.Generator(device=_DEVICE).manual_seed(5)
    batch, slots, top_k, vocab, rank = 3, 7, 32, 1000, 256
    pred = torch.randn(vocab, rank, generator=g, device=_DEVICE).to(dtype)
    succ = torch.randn(vocab, rank, generator=g, device=_DEVICE).to(dtype)
    cand = torch.randint(0, vocab, (batch, slots, top_k), generator=g, device=_DEVICE)
    unary = torch.randn(batch, slots, top_k, generator=g, device=_DEVICE).to(dtype)
    gate = torch.randn(batch, slots, rank, generator=g, device=_DEVICE).to(dtype)
    anchors = torch.randint(0, vocab, (batch,), generator=g, device=_DEVICE)
    scores = dflash2_score_edges(pred, succ, cand, unary, gate, anchors)
    assert torch.equal(scores, engine_score_edges(pred, succ, cand, unary, gate, anchors))
    assert torch.equal(dflash2_walk_candidate_paths(cand, scores), engine_walk(cand, scores))


def test_uniforms_match_unsigned_reference():
    seeds = [0, 1, 42, 2**31 - 1, 2**62 + 12345, 123456789]
    offsets = [0, 7, 1 << 62, (1 << 62) + 32 * 41, 3, 99991]
    got = dflash2_uniforms(
        torch.tensor(seeds, dtype=torch.int64, device=_DEVICE),
        torch.tensor(offsets, dtype=torch.int64, device=_DEVICE),
        9,
    ).cpu()
    ref = torch.tensor([_uniforms_py(s, o, 9) for s, o in zip(seeds, offsets)], dtype=torch.float32)
    assert torch.equal(got, ref)
    assert float(got.min()) >= 0.0 and float(got.max()) < 1.0


def test_uniforms_are_uniform_and_keyed():
    n, count = 20000, 8
    seed = torch.full((n,), 42, dtype=torch.int64, device=_DEVICE)
    offset = (1 << 62) + torch.arange(n, dtype=torch.int64, device=_DEVICE) * 32
    u = dflash2_uniforms(seed, offset, count)
    assert torch.equal(u, dflash2_uniforms(seed, offset, count))
    flat = u.flatten().double()
    assert abs(float(flat.mean()) - 0.5) < 0.005
    assert abs(float(flat.var()) - 1.0 / 12.0) < 0.002
    # Kolmogorov-Smirnov distance against U(0, 1).
    s = flat.sort().values
    m = s.numel()
    ks = torch.maximum(
        torch.arange(1, m + 1, device=s.device) / m - s, s - torch.arange(m, device=s.device) / m
    ).max()
    assert float(ks) < 1.95 / math.sqrt(m)
    # Neighbouring draws and neighbouring offsets are uncorrelated.
    assert abs(float(torch.corrcoef(u[:, :2].T.double())[0, 1])) < 0.03
    assert abs(float(torch.corrcoef(torch.stack((u[:-1, 0], u[1:, 0])).double())[0, 1])) < 0.03
    other = dflash2_uniforms(seed + 1, offset, count)
    assert not torch.equal(u, other)


def _walk_reference(cand, scores, uniforms, temps, greedy):
    """One request and slot at a time, the walk as specified."""
    n, slots, k = cand.shape
    tokens = torch.zeros(n, slots, dtype=torch.int64)
    q = torch.zeros(n, slots, k)
    path = torch.zeros(n, slots, dtype=torch.int64)
    for b in range(n):
        prev = 0
        for e in range(slots):
            row = scores[b, e, prev].float()
            if greedy[b]:
                idx = int(row.argmax())
                q[b, e, idx] = 1.0
            else:
                probs = torch.softmax(row / max(float(temps[b]), 1e-5), dim=-1)
                idx = min(int((uniforms[b, e] >= probs.cumsum(-1)).sum()), k - 1)
                q[b, e] = probs
            path[b, e] = idx
            tokens[b, e] = cand[b, e, idx]
            prev = idx
    return tokens, q, path


def _walk_vectorized_reference(cand, scores, uniforms, temps, greedy):
    """The serving implementation's torch reference walk (CandidateSelector.sample_path):
    a sampled local map per slot and predecessor, then the path follows the maps."""
    top_k = cand.shape[-1]
    t = temps.view(-1, 1)
    initial_probs = torch.softmax(scores[:, 0, 0].float() / t, dim=-1)
    initial = uniforms[:, :1].ge(initial_probs.cumsum(dim=-1)).sum(dim=-1).clamp_max(top_k - 1)
    transition_probs = torch.softmax(scores[:, 1:].float() / t[:, :, None, None], dim=-1)
    local_maps = (
        uniforms[:, 1:, None, None]
        .ge(transition_probs.cumsum(dim=-1))
        .sum(dim=-1)
        .clamp_max(top_k - 1)
    )
    initial = torch.where(greedy, scores[:, 0, 0].argmax(dim=-1), initial)
    local_maps = torch.where(greedy[:, None, None], scores[:, 1:].argmax(dim=-1), local_maps)
    index = initial
    path = [index]
    for edge in range(scores.shape[1] - 1):
        index = local_maps[:, edge].gather(-1, index[:, None])[:, 0]
        path.append(index)
    path = torch.stack(path, dim=1)
    tokens = cand.gather(-1, path.unsqueeze(-1))[:, :, 0]
    realized = transition_probs.gather(2, path[:, :-1, None, None].expand(-1, -1, 1, top_k))[
        :, :, 0
    ]
    q = torch.cat((initial_probs.unsqueeze(1), realized), dim=1)
    q = torch.where(greedy[:, None, None], F.one_hot(path, top_k).float(), q)
    return tokens, q, path


def _lattice(n, slots, k, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    cand = torch.stack([torch.randperm(vocab, generator=g)[:k] for _ in range(n * slots)]).view(
        n, slots, k
    )
    scores = 2.0 * torch.randn(n, slots, k, k, generator=g)
    # Slot 0's predecessor is the anchor for every row.
    scores[:, 0] = scores[:, 0, :1].expand(-1, k, -1)
    uniforms = torch.rand(n, slots, generator=g)
    temps = torch.tensor([0.6, 1.0, 1.7, 1e-6, 1.0, 0.3, 2.5, 1.0])[torch.arange(n) % 8]
    greedy = torch.arange(n) % 5 == 4
    return cand, scores, uniforms, temps, greedy


@pytest.mark.parametrize("slots", [4, 7])
def test_walk_matches_references(slots):
    cand, scores, uniforms, temps, greedy = _lattice(40, slots, 32, 1000)
    tokens, q, path = dflash2_selector_walk(
        cand.to(_DEVICE),
        scores.to(_DEVICE),
        uniforms.to(_DEVICE),
        temps.to(_DEVICE),
        greedy.to(_DEVICE),
    )
    ref_tokens, ref_q, ref_path = _walk_reference(cand, scores, uniforms, temps, greedy)
    assert torch.equal(path.cpu(), ref_path)
    assert torch.equal(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(q.cpu(), ref_q, rtol=1e-5, atol=1e-6)
    vec_tokens, vec_q, vec_path = _walk_vectorized_reference(
        cand, scores, uniforms, temps.clamp(min=1e-5), greedy
    )
    assert torch.equal(path.cpu(), vec_path)
    assert torch.equal(tokens.cpu(), vec_tokens)
    torch.testing.assert_close(q.cpu(), vec_q, rtol=1e-5, atol=1e-6)


def test_greedy_rows_follow_the_greedy_walk():
    cand, scores, uniforms, temps, _ = _lattice(16, 7, 32, 500, seed=3)
    greedy = torch.ones(16, dtype=torch.bool)
    tokens, q, path = dflash2_selector_walk(
        cand.to(_DEVICE),
        scores.to(_DEVICE),
        uniforms.to(_DEVICE),
        temps.to(_DEVICE),
        greedy.to(_DEVICE),
    )
    realized = dflash2_walk_candidate_paths(cand.to(_DEVICE), scores.to(_DEVICE))
    assert torch.equal(path, realized.argmax(dim=-1))
    assert torch.equal(q, F.one_hot(path, 32).float())
    assert torch.equal(tokens, cand.to(_DEVICE).gather(-1, path.unsqueeze(-1)).squeeze(-1))


def test_walk_path_distribution():
    """The sampled path's law is the chain of conditional softmaxes, and q is its row."""
    slots, k, temp = 3, 4, 0.8
    g = torch.Generator().manual_seed(7)
    scores = 1.5 * torch.randn(1, slots, k, k, generator=g)
    scores[:, 0] = scores[:, 0, :1].expand(-1, k, -1)
    n = 400000
    cand = torch.arange(k).view(1, 1, k).expand(n, slots, k).contiguous()
    uniforms = torch.rand(n, slots, generator=g)
    tokens, q, path = dflash2_selector_walk(
        cand.to(_DEVICE),
        scores.expand(n, -1, -1, -1).contiguous().to(_DEVICE),
        uniforms.to(_DEVICE),
        torch.full((n,), temp, device=_DEVICE),
        torch.zeros(n, dtype=torch.bool, device=_DEVICE),
    )
    path = path.cpu()
    cond = torch.softmax(scores[0] / temp, dim=-1)  # [slot, prev, cand]
    exact = torch.zeros(k, k, k, dtype=torch.float64)
    for a in range(k):
        for b in range(k):
            for c in range(k):
                exact[a, b, c] = cond[0, 0, a] * cond[1, a, b] * cond[2, b, c]
    index = path[:, 0] * k * k + path[:, 1] * k + path[:, 2]
    counts = torch.bincount(index, minlength=k**3).double()
    expected = exact.flatten() * n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = k**3 - 1
    assert chi2 < dof + 6 * math.sqrt(2 * dof), chi2
    # q of slot e is the conditional row under the realized predecessor.
    rows = torch.cat((torch.zeros(n, 1, dtype=torch.int64), path[:, :-1]), dim=1)
    want = cond[torch.arange(slots).expand(n, -1), rows]
    torch.testing.assert_close(q.cpu(), want, rtol=1e-5, atol=1e-6)


def test_proposal_store_round_trip():
    store = DFlash2ProposalStore(num_rows=9, max_draft_len=7, top_k=4, device=_DEVICE)
    cand = torch.randint(0, 100, (3, 4, 4), device=_DEVICE)
    q = torch.softmax(torch.randn(3, 4, 4, device=_DEVICE), dim=-1)
    tokens = cand[:, :, 1]
    slots = torch.tensor([5, 0, 8], device=_DEVICE)
    store.store(slots, cand, q, tokens)
    got_c, got_q = store.load(slots, tokens)
    assert torch.equal(got_c, cand) and torch.equal(got_q, q)
    # A row verifying other tokens (here the first) gets the point mass on them.
    other = tokens.clone()
    other[0, 2] += 1
    got_c, got_q = store.load(slots, other)
    assert torch.equal(got_c[1:], cand[1:]) and torch.equal(got_q[1:], q[1:])
    assert torch.equal(got_c[0], other[0].unsqueeze(-1).expand(4, 4))
    assert torch.equal(got_q[0], torch.tensor([1.0, 0, 0, 0], device=_DEVICE).expand(4, 4))
    store.invalidate(slots[1:2])
    got_c, got_q = store.load(slots, tokens)
    assert torch.equal(got_q[1, :, 0], torch.ones(4, device=_DEVICE))
    assert torch.equal(got_c[2], cand[2])
