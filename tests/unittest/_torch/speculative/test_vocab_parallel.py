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
"""Single-shard checks of the vocab-parallel selection kernels.

The multi-rank exchange is covered by
``tests/unittest/_torch/multi_gpu/test_vocab_parallel_selection.py``; here one
shard holds the whole vocab, which isolates the kernels: argmax order, the
Gumbel-max sampler's distribution and the Markov chain, including the
full-vocab chain that also returns the biased logits.
"""

import pytest
import torch
from _torch.helpers import dspark_markov_chain_logits_reference

from tensorrt_llm._torch.speculative.vocab_parallel import (
    _local_argmax_kernel,
    _local_sample_kernel,
    _markov_chain_tokens,
    _PairExchange,
    _select_tokens,
    markov_chain_logits,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_SINGLE = _PairExchange(None, 1, 0)


def _argmax(logits: torch.Tensor) -> torch.Tensor:
    return _select_tokens(_local_argmax_kernel, logits, (), 0, _SINGLE, torch.int64)


def _sample(logits, temperatures, seeds, offsets, threshold=1e-4):
    return _select_tokens(
        _local_sample_kernel,
        logits,
        (temperatures, seeds, offsets, threshold),
        0,
        _SINGLE,
        torch.int64,
    )


def _special_rows(logits: torch.Tensor) -> torch.Tensor:
    """Ties, all -inf, NaN and +inf rows, with the extremes far apart."""
    vocab = logits.shape[-1]
    logits[0] = 1.0
    logits[1] = -float("inf")
    logits[2, vocab - 1] = float("nan")
    logits[2, 2500] = float("nan")
    logits[3, 2047] = float("inf")
    logits[3, vocab - 2] = float("inf")
    logits[4] = 0.0
    logits[4, vocab - 1] = 2.0
    logits[4, 1] = 2.0
    return logits


@pytest.mark.parametrize("rows", [1, 7, 64, 300])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_argmax_matches_torch_argmax(rows, dtype):
    # 6001 columns: several 2048-wide tiles, the last one partial, and several
    # column splits per row when rows are few.
    vocab = 6001
    gen = torch.Generator(device="cuda").manual_seed(rows)
    logits = torch.randint(-4, 4, (rows, vocab), device="cuda", generator=gen).to(dtype)
    if rows >= 5:
        logits = _special_rows(logits)
    assert torch.equal(_argmax(logits), torch.argmax(logits.float(), dim=-1))


def test_sample_greedy_rows_take_the_argmax():
    rows, vocab = 33, 4100
    logits = torch.randn(rows, vocab, device="cuda")
    temperatures = torch.full((rows,), 1e-5, device="cuda")
    seeds = torch.arange(rows, dtype=torch.int64, device="cuda")
    offsets = torch.zeros(rows, dtype=torch.int64, device="cuda")
    assert torch.equal(_sample(logits, temperatures, seeds, offsets), logits.argmax(-1))


def test_sample_is_keyed_by_seed_and_offset_only():
    """The draw depends on (seed, offset) and the logits, not on the batch
    position or on how the row is split over programs."""
    vocab = 3 * 2048 + 100
    logits = torch.randn(1, vocab, device="cuda").expand(256, vocab).contiguous()
    temperatures = torch.full((256,), 0.8, device="cuda")
    seeds = torch.full((256,), 1234, dtype=torch.int64, device="cuda")
    offsets = torch.arange(256, dtype=torch.int64, device="cuda") * 32
    batched = _sample(logits, temperatures, seeds, offsets)
    for row in (0, 17, 255):
        single = _sample(logits[row : row + 1], temperatures[:1], seeds[:1], offsets[row : row + 1])
        assert single.item() == batched[row].item()
    # Repeatable, and a different seed gives a different stream.
    assert torch.equal(_sample(logits, temperatures, seeds, offsets), batched)
    assert not torch.equal(_sample(logits, temperatures, seeds + 1, offsets), batched)


@pytest.mark.parametrize("temperature", [0.6, 1.0, 1.7])
def test_sample_follows_the_softmax_distribution(temperature):
    """Gumbel-max draws are distributed as softmax(logits / T)."""
    vocab, draws = 64, 1 << 18
    gen = torch.Generator(device="cuda").manual_seed(3)
    row = torch.randn(vocab, device="cuda", generator=gen) * 2.0
    logits = row.expand(draws, vocab).contiguous()
    temperatures = torch.full((draws,), temperature, device="cuda")
    seeds = torch.full((draws,), 42, dtype=torch.int64, device="cuda")
    offsets = torch.arange(draws, dtype=torch.int64, device="cuda") * 32
    tokens = _sample(logits, temperatures, seeds, offsets)
    freq = torch.bincount(tokens, minlength=vocab).double() / draws
    probs = torch.softmax(row.double() / temperature, dim=-1)
    # Five standard errors per bin, plus a floor for the near-empty bins.
    tolerance = 5.0 * torch.sqrt(probs * (1 - probs) / draws) + 1e-4
    assert torch.all((freq - probs).abs() <= tolerance), (freq - probs).abs().max()
    # Pearson chi-square over the bins with enough expected mass.
    expected = probs * draws
    keep = expected >= 5
    chi2 = (((freq * draws - expected) ** 2) / expected)[keep].sum().item()
    dof = int(keep.sum().item()) - 1
    assert chi2 < dof + 6.0 * (2.0 * dof) ** 0.5


class _FakeVocabParallelHead:
    """Stands in for VocabParallelHead, which needs a TP > 1 LM head."""

    def __init__(self, lm_head):
        self.lm_head = lm_head

    @staticmethod
    def supports(lm_head):
        return True


def _bare_worker(worker_cls, monkeypatch, **spec_config):
    """A DFlash-family worker with only the state the vocab-parallel selection reads."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.speculative import dflash
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    monkeypatch.setattr(dflash, "VocabParallelHead", _FakeVocabParallelHead)
    worker = worker_cls.__new__(worker_cls)
    nn.Module.__init__(worker)
    config = dict(
        enable_penalty=False,
        use_rejection_sampling=False,
        advanced_sampling_mode=AdvancedSamplingMode.NO_TOPK_NO_TOPP,
    )
    config.update(spec_config)
    worker.spec_config = SimpleNamespace(**config)
    worker.guided_decoder = None
    worker._d2t = None
    worker._vocab_parallel = None
    worker._target_model_ref = None
    worker._draft_model_ref = None
    return worker


def _plain_logits_processor():
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

    return LogitsProcessor()


def test_vocab_parallel_selection_decisions(monkeypatch):
    """Everything but the per-batch sampling mode is decided once: the target
    side when the head is bound (and dropped by a guided decoder), and both
    sides by the worker's post-load hook."""
    import weakref

    from torch import nn

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm._torch.speculative.dspark import DSparkWorker

    lm_head = nn.Linear(4, 4)
    worker = _bare_worker(DFlashWorker, monkeypatch)
    worker.bind_target_lm_head(lm_head, _plain_logits_processor())
    assert worker.target_vocab_parallel_head.lm_head is lm_head
    assert worker._vocab_parallel.samples
    assert worker._vocab_parallel.draft_tokens is None

    class _Drafter:
        def __init__(self, **attrs):
            self.__dict__.update(attrs)

    def drafter(**overrides):
        attrs = dict(lm_head=lm_head, lilicorr=None, has_candidate_selector=False)
        attrs.update(overrides)
        return _Drafter(**attrs)

    # The model loader's post-load walk binds the draft side, after the
    # drafter took over the target's head.
    shared = drafter()
    worker._draft_model_ref = weakref.ref(shared)
    worker.post_load_weights()
    assert worker._vocab_parallel.draft_tokens == worker._argmax_block_tokens
    own_head = drafter(lm_head=nn.Linear(4, 4))
    worker._draft_model_ref = weakref.ref(own_head)
    worker.setup_aliases()
    assert worker._vocab_parallel.draft_tokens is None

    assert worker._vocab_parallel_draft_tokens(drafter()) == worker._argmax_block_tokens
    assert worker._vocab_parallel_draft_tokens(drafter(lm_head=nn.Linear(4, 4))) is None
    assert worker._vocab_parallel_draft_tokens(drafter(lilicorr=object())) is None
    assert worker._vocab_parallel_draft_tokens(drafter(has_candidate_selector=True)) is None
    worker._d2t = torch.zeros(4, dtype=torch.long)
    assert worker._vocab_parallel_draft_tokens(drafter()) is None

    dspark = _bare_worker(DSparkWorker, monkeypatch)
    dspark.bind_target_lm_head(lm_head, _plain_logits_processor())
    assert dspark._vocab_parallel_draft_tokens(drafter()) == dspark._argmax_block_tokens
    markov = drafter(has_markov_head=True)
    assert dspark._vocab_parallel_draft_tokens(markov) == dspark._markov_block_tokens

    # A guided decoder edits the full target logits, also if the post-load
    # hook runs again.
    worker.set_guided_decoder(object())
    assert worker.target_vocab_parallel_head is None
    worker.bind_target_lm_head(lm_head, _plain_logits_processor())
    assert worker.target_vocab_parallel_head is None

    # Occurrence penalties rewrite the full target logits too.
    penalized = _bare_worker(DFlashWorker, monkeypatch, enable_penalty=True)
    penalized.bind_target_lm_head(lm_head, _plain_logits_processor())
    assert penalized.target_vocab_parallel_head is None

    # Rejection sampling needs the full target distribution; greedy batches
    # still take the shard.
    rejection = _bare_worker(DFlashWorker, monkeypatch, use_rejection_sampling=True)
    rejection.bind_target_lm_head(lm_head, _plain_logits_processor())
    assert rejection.target_vocab_parallel_head is not None
    assert not rejection._vocab_parallel.samples


def test_target_with_a_custom_logits_processor_keeps_gathered_logits(monkeypatch):
    """The shard logits are ``lm_head(hidden_states)``: a target whose logits
    processor does more (e.g. a final norm before the head) keeps the gathered
    logits. The decision is made after loading, once a model's own processor
    has replaced the base one."""
    from torch import nn

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    class _NormThenHead(nn.Module):
        def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
            return lm_head(hidden_states / hidden_states.norm(dim=-1, keepdim=True))

    target = nn.Module()
    target.lm_head = nn.Linear(4, 4)
    target.logits_processor = _plain_logits_processor()
    worker = _bare_worker(DFlashWorker, monkeypatch)
    worker.set_target_model(target)
    assert worker._target_model_ref() is target
    # Nothing is decided at construction.
    assert worker.target_vocab_parallel_head is None
    worker.setup_aliases()
    assert worker.target_vocab_parallel_head.lm_head is target.lm_head
    # Running the hook again keeps the selection (and its all-reduce).
    selection = worker._vocab_parallel
    worker.setup_aliases()
    assert worker._vocab_parallel is selection

    target.logits_processor = _NormThenHead()
    worker.setup_aliases()
    assert worker.target_vocab_parallel_head is None
    worker.bind_target_lm_head(target.lm_head, _NormThenHead())
    assert worker.target_vocab_parallel_head is None


@pytest.mark.parametrize("walk", ["_setup_aliases", "_walk_full_post_load"])
def test_loader_post_load_walk_binds_the_draft_side(monkeypatch, walk):
    """The worker is a module of the engine, registered after the drafter, so
    the model loader's post-load walks bind its target side and its draft side
    once the drafter shares the target's head."""
    import weakref

    from torch import nn

    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    lm_head = nn.Linear(4, 4)
    engine = nn.Module()
    engine.lm_head = lm_head
    engine.logits_processor = _plain_logits_processor()
    engine.draft_model = nn.Module()
    engine.draft_model.lilicorr = None
    engine.draft_model.has_candidate_selector = False
    worker = _bare_worker(DFlashWorker, monkeypatch)
    worker.set_target_model(engine)
    worker._draft_model_ref = weakref.ref(engine.draft_model)
    engine.spec_worker = worker
    # The drafter takes over the target's head while the weights load.
    engine.draft_model.lm_head = lm_head
    getattr(ModelLoader, walk)(engine)
    assert worker.target_vocab_parallel_head.lm_head is lm_head
    assert worker._vocab_parallel.draft_tokens == worker._argmax_block_tokens


def test_markov_tp_exchange_only_for_a_drafter_with_shard_logits(monkeypatch):
    """The DSpark Markov chain's TP exchange is built only for a Markov drafter
    whose own column-parallel head does not gather its output."""
    from types import SimpleNamespace

    from torch import nn

    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
    from tensorrt_llm._torch.speculative import dspark
    from tensorrt_llm._torch.speculative.dspark import DSparkWorker

    monkeypatch.setattr(dspark, "tp_pair_exchange", lambda mapping, strategy: "exchange")

    def head(gather_output):
        linear = Linear.__new__(Linear)
        nn.Module.__init__(linear)
        linear.tp_mode = TensorParallelMode.COLUMN
        linear.gather_output = gather_output
        return linear

    def exchange_for(lm_head, has_markov_head=True, tp_size=2, attention_dp=False):
        worker = _bare_worker(DSparkWorker, monkeypatch)
        worker.mapping = SimpleNamespace(tp_size=tp_size, enable_attention_dp=attention_dp)
        worker._markov_tp_exchange = None
        worker.bind_draft_lm_head(SimpleNamespace(lm_head=lm_head, has_markov_head=has_markov_head))
        return worker._markov_tp_exchange

    assert exchange_for(head(gather_output=False)) == "exchange"
    # The usual drafter shares the target's gathered head: full-vocab logits.
    assert exchange_for(head(gather_output=True)) is None
    assert exchange_for(head(gather_output=False), has_markov_head=False) is None
    assert exchange_for(head(gather_output=False), tp_size=1) is None
    assert exchange_for(head(gather_output=False), attention_dp=True) is None
    assert exchange_for(nn.Linear(4, 4)) is None


def test_target_logits_are_the_shard_once_the_head_is_bound(monkeypatch):
    """The worker's target-logits hook hands its forward this rank's vocab
    shard once the target head is bound, and the gathered logits before."""
    from torch import nn

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    class _RecordingLogitsProcessor:
        def __init__(self):
            self.calls = 0

        def forward(self, *args):
            self.calls += 1
            return "gathered"

    class _ShardHead(_FakeVocabParallelHead):
        def shard_logits(self, hidden_states):
            return ("shard", hidden_states)

    worker = _bare_worker(DFlashWorker, monkeypatch)
    processor = _RecordingLogitsProcessor()
    hidden = torch.zeros(2, 4)
    lm_head = nn.Linear(4, 4)
    args = (hidden, lm_head, processor, None, None, None)
    assert worker.target_logits(*args) == "gathered"
    assert processor.calls == 1

    from tensorrt_llm._torch.speculative import dflash

    monkeypatch.setattr(dflash, "VocabParallelHead", _ShardHead)
    worker.bind_target_lm_head(lm_head, _plain_logits_processor())
    shard = worker.target_logits(*args)
    assert shard[0] == "shard" and shard[1] is hidden
    assert processor.calls == 1


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
def test_markov_chain_matches_the_reference_chain(w_dtype):
    num_reqs, block, vocab, rank = 6, 7, 5000, 32
    gen = torch.Generator(device="cuda").manual_seed(11)
    base = torch.randn(num_reqs, block, vocab, device="cuda", generator=gen)
    w1 = (torch.randn(vocab, rank, device="cuda", generator=gen)).to(w_dtype)
    w2 = (torch.randn(vocab, rank, device="cuda", generator=gen) * 0.5).to(w_dtype)
    # Out-of-vocab anchors contribute no bias.
    anchors = torch.tensor([0, 17, vocab - 1, -1, vocab, 4096], device="cuda")
    expected = (
        dspark_markov_chain_logits_reference(base, anchors, w1, w2).argmax(-1).to(torch.int32)
    )
    tokens = _markov_chain_tokens(base, anchors, w1, w2, 0, _SINGLE)
    assert torch.equal(tokens, expected)
    # The bias must actually steer the chain for the check to mean something.
    assert not torch.equal(tokens, base.argmax(-1).to(torch.int32))


def test_markov_chain_cuda_graph_replay():
    num_reqs, block, vocab, rank = 3, 5, 4097, 16
    base = torch.randn(num_reqs, block, vocab, device="cuda")
    w1 = torch.randn(vocab, rank, device="cuda").to(torch.bfloat16)
    w2 = torch.randn(vocab, rank, device="cuda").to(torch.bfloat16)
    anchors = torch.tensor([1, 2, 3], device="cuda")
    _markov_chain_tokens(base, anchors, w1, w2, 0, _SINGLE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokens = _markov_chain_tokens(base, anchors, w1, w2, 0, _SINGLE)
    base.copy_(torch.randn_like(base))
    anchors.copy_(torch.tensor([7, 8, 9], device="cuda"))
    graph.replay()
    expected = (
        dspark_markov_chain_logits_reference(base, anchors, w1, w2).argmax(-1).to(torch.int32)
    )
    assert torch.equal(tokens, expected)


@pytest.mark.parametrize("num_reqs,block", [(1, 1), (4, 7), (128, 6)])
@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
def test_markov_chain_logits_bit_identical(num_reqs, block, w_dtype):
    vocab, rank = 6001, 64
    gen = torch.Generator(device="cuda").manual_seed(num_reqs)
    base = torch.randn(num_reqs, block, vocab, device="cuda", generator=gen) * 3
    w1 = torch.randn(vocab, rank, device="cuda", generator=gen).to(w_dtype)
    w2 = (torch.randn(vocab, rank, device="cuda", generator=gen) * 0.2).to(w_dtype)
    anchors = torch.randint(0, vocab, (num_reqs,), device="cuda", generator=gen)
    if num_reqs > 1:
        anchors[0] = -1
    expected = dspark_markov_chain_logits_reference(base, anchors, w1, w2)
    biased = markov_chain_logits(base, anchors, w1, w2)
    assert torch.equal(biased.view(torch.int32), expected.view(torch.int32))


def test_markov_chain_logits_ties_and_nan():
    """The chain follows torch.argmax order through ties and NaN."""
    num_reqs, block, vocab, rank = 3, 4, 4100, 8
    base = torch.zeros(num_reqs, block, vocab, device="cuda")
    base[1, 0, 3000] = float("nan")
    base[2, :, 2049] = 5.0
    base[2, :, 4099] = 5.0
    w1 = torch.randn(vocab, rank, device="cuda").to(torch.bfloat16)
    w2 = torch.zeros(vocab, rank, device="cuda").to(torch.bfloat16)
    anchors = torch.tensor([5, 6, 7], device="cuda")
    expected = dspark_markov_chain_logits_reference(base, anchors, w1, w2)
    biased = markov_chain_logits(base, anchors, w1, w2)
    assert torch.equal(biased.view(torch.int32), expected.view(torch.int32))
