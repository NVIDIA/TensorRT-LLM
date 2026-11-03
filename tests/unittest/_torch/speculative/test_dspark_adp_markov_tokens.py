# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSpark greedy draft tokens straight from a full-vocab LM head output on one rank
(drafting Kimi K3.1 engines; the attention-DP drafter without a vocab-parallel
selection): the tokens equal the logits path's (float32 logits, the Markov chain's biased
logits, then the per-position argmax), and the worker takes the path only where that path
would run the same refinement."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor import input_commit
from tensorrt_llm._torch.speculative.vocab_parallel import markov_chain_logits, markov_chain_tokens

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _logits_path_tokens(block_logits, anchors, w1, w2):
    biased = markov_chain_logits(block_logits.float(), anchors, w1, w2)
    return biased.argmax(-1).to(torch.int32)


@needs_cuda
@pytest.mark.parametrize("logit_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "num_reqs,block,vocab,rank",
    [(1, 1, 4097, 16), (4, 6, 5000, 32), (16, 6, 6001, 64), (3, 7, 163840, 256)],
)
def test_tokens_equal_the_logits_path(logit_dtype, num_reqs, block, vocab, rank):
    gen = torch.Generator(device="cuda").manual_seed(num_reqs * 31 + block)
    logits = (torch.randn(num_reqs, block, vocab, device="cuda", generator=gen) * 3).to(logit_dtype)
    w1 = torch.randn(vocab, rank, device="cuda", generator=gen).bfloat16()
    w2 = (torch.randn(vocab, rank, device="cuda", generator=gen) * 0.3).bfloat16()
    anchors = torch.randint(0, vocab, (num_reqs,), device="cuda", generator=gen)
    if num_reqs > 1:
        anchors[0] = -1  # out of vocab: no bias at step 0
    got = markov_chain_tokens(logits, anchors, w1, w2)
    assert got.dtype == torch.int32 and got.shape == (num_reqs, block)
    assert torch.equal(got, _logits_path_tokens(logits, anchors, w1, w2))
    if block > 1:
        # The chain must actually steer the tokens for the check to mean something.
        assert not torch.equal(got, logits.float().argmax(-1).to(torch.int32))


@needs_cuda
def test_tokens_ties_nan_inf_and_graph_replay():
    num_reqs, block, vocab, rank = 4, 6, 4096, 16
    logits = torch.randn(num_reqs, block, vocab, device="cuda").bfloat16()
    logits[0] = 1.0  # every column ties: lowest index wins
    logits[1, 2, 17] = float("nan")
    logits[2, 3, 1000] = float("inf")
    logits[3, 0] = float("-inf")
    w1 = torch.randn(vocab, rank, device="cuda").bfloat16()
    w2 = torch.zeros(vocab, rank, device="cuda").bfloat16()  # no bias: keeps the ties exact
    anchors = torch.tensor([1, 2, 3, 4], device="cuda")
    expected = _logits_path_tokens(logits, anchors, w1, w2)
    assert torch.equal(markov_chain_tokens(logits, anchors, w1, w2), expected)

    w2.copy_(torch.randn(vocab, rank, device="cuda").bfloat16())
    markov_chain_tokens(logits, anchors, w1, w2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokens = markov_chain_tokens(logits, anchors, w1, w2)
    logits.copy_(torch.randn_like(logits.float()).bfloat16())
    anchors.copy_(torch.tensor([7, 8, 9, 10], device="cuda"))
    graph.replay()
    assert torch.equal(tokens, _logits_path_tokens(logits, anchors, w1, w2))


def _worker(**spec_config):
    from tensorrt_llm._torch.speculative.dspark import DSparkWorker

    worker = DSparkWorker.__new__(DSparkWorker)
    nn.Module.__init__(worker)
    config = dict(enable_penalty=False)
    config.update(spec_config)
    worker.spec_config = SimpleNamespace(**config)
    worker.guided_decoder = None
    worker._d2t = None
    worker._vocab_parallel = None
    return worker


def _drafter(vocab=64, rank=8, **overrides):
    attrs = dict(
        has_markov_head=True,
        markov_w1=torch.zeros(vocab, rank),
        markov_w2=torch.zeros(vocab, rank),
        lm_head=SimpleNamespace(out_features=vocab),
        logits_processor=LogitsProcessor(),
        lilicorr=None,
        has_candidate_selector=False,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_worker_takes_the_tokens_path_only_where_the_logits_path_matches(monkeypatch):
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = _worker()
    monkeypatch.setattr(input_commit, "_K31_DRAFTS", False)
    assert worker._local_draft_tokens(_drafter()) is None  # other models keep the logits path

    monkeypatch.setattr(input_commit, "_K31_DRAFTS", True)
    assert worker._local_draft_tokens(_drafter()) == worker._markov_local_tokens
    # Plain DFlash never has the path.
    assert DFlashWorker._local_draft_tokens(worker, _drafter()) is None

    assert worker._local_draft_tokens(_drafter(has_markov_head=False)) is None
    assert worker._local_draft_tokens(_drafter(lm_head=SimpleNamespace(out_features=32))) is None
    assert worker._local_draft_tokens(_drafter(lm_head=None)) is None
    assert worker._local_draft_tokens(_drafter(logits_processor=nn.Identity())) is None
    assert worker._local_draft_tokens(_drafter(lilicorr=object())) is None
    assert worker._local_draft_tokens(_drafter(has_candidate_selector=True)) is None
    assert worker._local_draft_tokens(_drafter(markov_w1=torch.zeros(8, 64).t())) is None

    worker._d2t = torch.zeros(64, dtype=torch.long)
    assert worker._local_draft_tokens(_drafter()) is None
    worker._d2t = None
    worker.guided_decoder = object()
    assert worker._local_draft_tokens(_drafter()) is None
    assert _worker(enable_penalty=True)._local_draft_tokens(_drafter()) is None
