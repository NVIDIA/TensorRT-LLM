# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 sampled selector walk: the walk against a per-row reference and against the vectorized
reference walk of the DFlash2 serving implementation."""

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    dflash2_selector_walk,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def test_walk_matches_references():
    cand, scores, uniforms, temps, greedy = _lattice(40, 7, 32, 1000)
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


def test_arriving_generation_uses_fresh_proposal_probabilities(monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec import dflash2_sampling
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.unified_draft_cache_worker import (
        UnifiedDraftCacheWorkerMixin,
    )

    worker = UnifiedDraftCacheWorkerMixin()
    store = dflash2_sampling.DFlash2ProposalStore(2, 1, 2, "cpu")
    worker._dflash2_proposals = store
    worker._dflash2_proposal_requests = {5}
    slots = torch.tensor([0])
    tokens = torch.tensor([[7]])
    store.store(slots, torch.tensor([[[7, 8]]]), torch.tensor([[[0.3, 0.7]]]), tokens)
    manager = SimpleNamespace(is_disagg=True, kv_cache_map={6: object()})
    attn = SimpleNamespace(kv_cache_manager=manager, num_contexts=0)
    spec = SimpleNamespace(
        request_ids=[6], batch_slot_ids=slots, max_draft_len=1, use_rejection_sampling=True
    )
    monkeypatch.setattr(worker, "get_draft_kv_cache_manager", lambda _: None)
    monkeypatch.setattr(
        dflash2_sampling,
        "async_tensor_h2d",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype, device=device),
    )
    worker.prepare_managed_draft_cache(None, spec, attn, None)
    expected_ids, expected_q = store.load(torch.tensor([1]), tokens)
    ids, q = store.load(slots, tokens)
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(q, expected_q)
