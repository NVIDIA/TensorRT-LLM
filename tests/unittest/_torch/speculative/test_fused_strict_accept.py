# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The strict acceptance of a CUDA-graph verify step (`fused_accept.strict_accept`) must
return what the composed ops of an eager step return: the same accepted tokens and counts,
including the synthetic acceptance override and its counter, replay after replay."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative import fused_accept
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB = 248320


class _Worker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


def _worker(force: float) -> _Worker:
    worker = _Worker()
    worker.force_num_accepted_tokens = force
    return worker


def _meta(is_cuda_graph: bool) -> SimpleNamespace:
    return SimpleNamespace(
        is_cuda_graph=is_cuda_graph,
        is_all_greedy_sample=True,
        is_ragged_verify=False,
        enable_penalty=False,
    )


def _fill(logits, draft, gen, d, generator):
    """Random logits with one planted maximum a row; draft tokens that repeat a random number
    of the target's leading tokens, then differ (or repeat all of them)."""
    rows = gen * (d + 1)
    logits.copy_(torch.randn(rows, VOCAB, generator=generator).to(logits.device))
    target = torch.randint(0, VOCAB, (rows,), generator=generator)
    logits[torch.arange(rows, device=logits.device), target.to(logits.device)] = 100.0
    target = target.reshape(gen, d + 1)
    match = torch.randint(0, d + 1, (gen,), generator=generator)
    proposal = torch.where(
        torch.arange(d)[None, :] < match[:, None], target[:, :d], (target[:, :d] + 1) % VOCAB
    )
    draft.copy_(proposal.to(draft.device, torch.int32))


def _composed(worker, logits, draft, gen):
    return worker._sample_and_accept_draft_tokens_base(logits, draft, 0, gen, _meta(False))


def _fused(worker, logits, draft, gen):
    meta = _meta(True)
    assert worker._can_use_fused_strict_accept(logits, 0, gen, draft.shape[1], meta)
    return worker._sample_and_accept_draft_tokens_base(logits, draft, 0, gen, meta)


@pytest.mark.parametrize("gen", [1, 4, 33])
@pytest.mark.parametrize("d", [1, 3, 7])
def test_verified_counts_match(gen, d):
    g = torch.Generator().manual_seed(gen * 10 + d)
    logits = torch.empty(gen * (d + 1), VOCAB, device="cuda")
    draft = torch.empty(gen, d, dtype=torch.int32, device="cuda")
    worker = _worker(0.0)
    for _ in range(4):
        _fill(logits, draft, gen, d, g)
        ref = _composed(worker, logits, draft, gen)
        got = _fused(worker, logits, draft, gen)
        assert got[0].dtype == ref[0].dtype == torch.int32
        assert got[1].dtype == ref[1].dtype == torch.int32
        assert torch.equal(got[0], ref[0])
        assert torch.equal(got[1], ref[1])


@pytest.mark.parametrize("force", [0.0, 2.3, 2.0, 1.7, 5.0, 3.5])
@pytest.mark.parametrize("gen", [1, 4, 64])
def test_cuda_graph_replays_match_eager_steps(force, gen):
    """The fused step captured once and replayed against the composed eager step of a second
    worker: both counters start at zero and advance once a step."""
    d = 3
    g = torch.Generator().manual_seed(int(force * 10) + gen)
    logits = torch.empty(gen * (d + 1), VOCAB, device="cuda")
    draft = torch.empty(gen, d, dtype=torch.int32, device="cuda")
    eager, graphed = _worker(force), _worker(force)
    graphed._ensure_force_accept_rng_state(logits.device)
    _fill(logits, draft, gen, d, g)
    # The graph warmup step: synthetic acceptance off, kernels compiled and loaded.
    _fused(graphed, logits, draft, gen)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fused(graphed, logits, draft, gen)
    for _ in range(12):
        _fill(logits, draft, gen, d, g)
        ref = _composed(eager, logits, draft, gen)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[0], ref[0])
        assert torch.equal(out[1], ref[1])
    if eager._force_accept_rng_counter is not None:
        assert torch.equal(graphed._force_accept_rng_counter, eager._force_accept_rng_counter)


def test_admission():
    meta = _meta(True)
    logits = torch.empty(8, 16, device="cuda")
    assert SpecWorkerBase._can_use_fused_strict_accept(logits, 0, 2, 3, meta)
    # Context rows, eager steps, non-greedy and ragged batches keep the composed ops.
    assert not SpecWorkerBase._can_use_fused_strict_accept(logits, 1, 2, 3, meta)
    assert not SpecWorkerBase._can_use_fused_strict_accept(logits, 0, 2, 3, _meta(False))
    for key in ("is_all_greedy_sample",):
        other = _meta(True)
        setattr(other, key, False)
        assert not SpecWorkerBase._can_use_fused_strict_accept(logits, 0, 2, 3, other)
    other = _meta(True)
    other.is_ragged_verify = True
    assert not SpecWorkerBase._can_use_fused_strict_accept(logits, 0, 2, 3, other)
    big = torch.empty((fused_accept.MAX_REQUESTS + 1) * 4, 16, device="cuda")
    assert not SpecWorkerBase._can_use_fused_strict_accept(
        big, 0, fused_accept.MAX_REQUESTS + 1, 3, meta
    )
