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
"""The vocab-parallel greedy argmax of the stock one-model speculative workers.

The greedy target tokens of a TP worker come from this rank's vocab shard of
the target logits: each rank reduces its shard to (value, global id) pairs and
the ranks agree on the global argmax. Here TP ranks are simulated on one GPU
(every rank's pair buffer is computed, then summed as the all-reduce would),
which checks the selected ids against ``torch.argmax`` over the gathered
logits, ties included, and the host-side decision of when the worker takes
the shard and when it keeps the gathered logits. The real exchanges over a TP
group are covered by
``tests/unittest/_torch/multi_gpu/test_vocab_parallel_selection.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.vocab_parallel import (
    VOCAB_PARALLEL_ARGMAX_ENV_VAR,
    _PairExchange,
    is_vocab_sharded_lm_head,
    shard_argmax,
    vocab_parallel_argmax_enabled,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN = 64


class _Worker(SpecWorkerBase):
    verifies_on_target_shard = True

    @property
    def max_draft_len(self):
        return 4

    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError


class _SimulatedAllReduce:
    """The pair all-reduce of one simulated rank: records the rank's buffer in
    the first pass, returns the sum over the ranks in the second."""

    def __init__(self, rank: int, buffers: dict):
        self.rank = rank
        self.buffers = buffers
        self.total = None

    def __call__(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.total is None:
            self.buffers[self.rank] = buffer.clone()
            return buffer
        return self.total


def _run_ranks(tp: int, select):
    """``select(rank, exchange)`` on every simulated rank of a TP group; every
    rank must reach the same result, which is returned."""
    buffers = {}
    reducers = [_SimulatedAllReduce(r, buffers) for r in range(tp)]
    for r in range(tp):
        select(r, _PairExchange(reducers[r], tp, r))
    total = sum(buffers[r] for r in range(tp))
    outs = []
    for r in range(tp):
        reducers[r].total = total
        outs.append(select(r, _PairExchange(reducers[r], tp, r)))
    for out in outs[1:]:
        assert torch.equal(out, outs[0])
    return outs[0]


def _shard(full: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    width = full.shape[-1] // tp
    return full[..., rank * width : (rank + 1) * width].contiguous()


def _tie_rows(full: torch.Tensor, tp: int) -> torch.Tensor:
    """Equal maxima across shards, inside a shard, -0.0 against +0.0, all
    -inf and the last column, at the head of ``full``."""
    vocab = full.shape[-1]
    width = vocab // tp
    full[0] = 0.0
    full[0, 3::width] = 1.0  # one equal maximum per shard: id 3
    full[1] = -1.0
    full[1, width - 1] = full[1, vocab - 1] = 5.0  # tied across the first and last shard
    full[2] = -float("inf")
    full[3] = -1.0
    full[3, width + 2] = 0.0
    full[3, 7] = -0.0  # equal to +0.0: id 7
    full[4] = -1.0
    full[4, vocab - 1] = 3.0
    full[5] = 2.0  # the whole row ties: id 0
    return full


# The per-shard argmax kernel itself (rows, dtypes) is covered by
# test_vocab_parallel.py; these cases cover composing the shards' ids by
# their vocab offsets, ties within and across shards included.
@pytest.mark.parametrize(
    "tp, rows, dtype",
    [(2, 1, torch.bfloat16), (2, 64, torch.float32), (8, 300, torch.bfloat16)],
)
def test_shard_argmax_matches_gathered_argmax(tp, rows, dtype):
    # 3000-wide shards: one 2048-wide tile and a partial one per shard.
    vocab = 3000 * tp
    gen = torch.Generator(device="cuda").manual_seed(rows * tp)
    # Few distinct values make many ties within and across shards.
    full = torch.randint(-3, 3, (rows, vocab), device="cuda", generator=gen).to(dtype)
    if rows >= 6:
        full = _tie_rows(full, tp)
    expected = torch.argmax(full.float(), dim=-1)
    if rows >= 6:
        assert expected[:6].tolist() == [3, 3000 - 1, 0, 7, vocab - 1, 0]
    width = vocab // tp
    tokens = _run_ranks(
        tp, lambda r, ex: shard_argmax(_shard(full, tp, r), r * width, ex, torch.int32)
    )
    assert tokens.dtype == torch.int32
    assert torch.equal(tokens.long(), expected)


def _greedy_md(**overrides):
    md = dict(
        is_all_greedy_sample=True,
        temperature_only=False,
        enable_penalty=False,
        sharded_target_logits=True,
        is_cuda_graph=False,
        is_ragged_verify=False,
    )
    md.update(overrides)
    return SimpleNamespace(**md)


def _lm_head(tp=2, rank=0, vocab=4096, adp=False, gather_output=True, padded=False, hidden=HIDDEN):
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank, enable_attention_dp=adp)
    head = LMHead(
        vocab + (1 if padded else 0),
        hidden,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=gather_output,
        reduce_output=False,
    ).cuda()
    torch.nn.init.normal_(head.weight, std=0.05)
    return head


@pytest.fixture
def gathered(monkeypatch):
    """Replace the gathering logits processor (a TP all-gather) by a marker."""
    marker = object()

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        return marker

    monkeypatch.setattr(LogitsProcessor, "forward", forward)
    return marker


@pytest.fixture
def enabled_worker(monkeypatch):
    """A worker that went through enable_vocab_parallel_argmax (no key
    exchange: single process), with the NCCL pair exchange stubbed out."""
    monkeypatch.setattr(interface, "tp_pair_exchange", lambda mapping, strategy: "pairs")
    worker = _Worker()
    assert worker.enable_vocab_parallel_argmax(_lm_head(), key_exchange=None)
    assert worker._target_pair_exchange == "pairs"
    return worker


def _target_logits(worker, lm_head, md, processor=None, rows=5):
    hidden = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = worker.target_logits(
        hidden, lm_head, processor or LogitsProcessor(), None, md, draft_model=None
    )
    return out, hidden


def test_greedy_batch_takes_the_shard(enabled_worker, gathered):
    lm_head = _lm_head()
    md = _greedy_md(sharded_target_logits=False)
    out, hidden = _target_logits(enabled_worker, lm_head, md)
    assert md.sharded_target_logits
    assert out is not gathered
    assert out.shape == (5, lm_head.out_features)
    torch.testing.assert_close(out, lm_head(hidden, gather_output=False), rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "temperature_only",  # PR scope: greedy only for the newly covered workers
        "top_k_top_p",  # any other sampled batch (also rejection sampling)
        "penalties",
        "guided_decoding",
        "custom_logits_processor",
        "relaxed_acceptance_mtp",
        "relaxed_acceptance_eagle3",
        "no_rows",
    ],
)
def test_fallback_keeps_the_gathered_logits(case, enabled_worker, gathered):
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
    from tensorrt_llm._torch.speculative.mtp import MTPWorker

    lm_head = _lm_head()
    md = _greedy_md(sharded_target_logits=True)
    worker = enabled_worker
    processor = None
    rows = 5
    if case.startswith("relaxed_acceptance"):
        # Relaxed acceptance reads the target's top-k, so the worker's own
        # acceptance check turns the shard off for every batch.
        cls = MTPWorker if case == "relaxed_acceptance_mtp" else Eagle3OneModelWorker
        worker = _bare_covered_worker(cls)
        assert worker.enable_vocab_parallel_argmax(lm_head, key_exchange=None)
        assert worker._verifies_target_by_argmax()
        worker.spec_config.use_relaxed_acceptance_for_thinking = True
        assert not worker._verifies_target_by_argmax()
    elif case == "temperature_only":
        md.is_all_greedy_sample, md.temperature_only = False, True
    elif case == "top_k_top_p":
        md.is_all_greedy_sample = False
    elif case == "penalties":
        md.enable_penalty = True
    elif case == "guided_decoding":
        enabled_worker.guided_decoder = object()
    elif case == "custom_logits_processor":

        class _Scaled(LogitsProcessor):
            pass

        processor = _Scaled()
    elif case == "no_rows":
        rows = 0
    out, _ = _target_logits(worker, lm_head, md, processor, rows)
    assert out is gathered
    assert not md.sharded_target_logits


@pytest.mark.parametrize("case", ["attention_dp", "no_gather", "padded_vocab", "tp1", "embedding"])
def test_heads_that_are_not_vocab_shards(case, monkeypatch, gathered):
    monkeypatch.setattr(interface, "tp_pair_exchange", lambda mapping, strategy: "pairs")
    if case == "attention_dp":
        head = _lm_head(tp=2, adp=True)
    elif case == "no_gather":
        head = _lm_head(gather_output=False)
    elif case == "padded_vocab":
        head = _lm_head(padded=True)
    elif case == "tp1":
        head = _lm_head(tp=1)
    else:
        from tensorrt_llm._torch.modules.embedding import Embedding

        head = Embedding(
            4096,
            HIDDEN,
            dtype=torch.bfloat16,
            mapping=Mapping(world_size=2, tp_size=2, rank=0),
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            reduce_output=False,
        )
    assert not is_vocab_sharded_lm_head(head)
    worker = _Worker()
    assert not worker.enable_vocab_parallel_argmax(head, key_exchange=None)
    if case != "embedding":
        md = _greedy_md(sharded_target_logits=False)
        out, _ = _target_logits(worker, head, md)
        assert out is gathered and not md.sharded_target_logits


def test_a_worker_whose_acceptance_reads_the_logits_does_not_enable(monkeypatch):
    monkeypatch.setattr(interface, "tp_pair_exchange", lambda mapping, strategy: "pairs")

    class _TreeWorker(_Worker):
        verifies_on_target_shard = False

    assert not _TreeWorker().enable_vocab_parallel_argmax(_lm_head(), key_exchange=None)


def test_covered_workers():
    """The stock workers whose greedy verification reads only the target's
    argmax take the path; tree and block drafters keep their own."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm._torch.speculative.draft_target import DraftTargetOneModelWorker
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker, MTPEagleWorker
    from tensorrt_llm._torch.speculative.eagle3_dynamic_tree import Eagle3OneModelDynamicTreeWorker
    from tensorrt_llm._torch.speculative.mtp import MTPWorker
    from tensorrt_llm._torch.speculative.mtp_dynamic_tree import MTPEagleDynamicTreeWorker
    from tensorrt_llm._torch.speculative.pard import PARDWorker
    from tensorrt_llm._torch.speculative.sa_worker import SAWorker

    for cls in (
        MTPWorker,
        Eagle3OneModelWorker,
        MTPEagleWorker,
        DraftTargetOneModelWorker,
        PARDWorker,
        SAWorker,
    ):
        assert cls.verifies_on_target_shard, cls.__name__
    for cls in (Eagle3OneModelDynamicTreeWorker, MTPEagleDynamicTreeWorker, DFlashWorker):
        assert not cls.verifies_on_target_shard, cls.__name__


def _bare_covered_worker(cls):
    """A covered worker with the base state and what its acceptance reads."""
    worker = object.__new__(cls)
    SpecWorkerBase.__init__(worker)
    worker.spec_config = SimpleNamespace(use_relaxed_acceptance_for_thinking=False)
    worker.is_thop = False
    worker.sa_enhancer = None
    return worker


def _accept(kind, worker, logits, md, attn_metadata):
    """The covered worker's own strict acceptance of one step."""
    if kind in ("mtp", "sa"):
        input_ids = torch.zeros(1, dtype=torch.int32, device=logits.device)
        if kind == "mtp":
            return worker.sample_and_accept_draft_tokens(input_ids, logits, md, attn_metadata)
        return worker._sample_and_accept_draft_tokens(input_ids, logits, md, attn_metadata)
    if kind in ("eagle3", "mtp_eagle"):
        input_ids = torch.zeros(1, dtype=torch.int32, device=logits.device)
        return worker.sample_and_accept_draft_tokens(input_ids, logits, attn_metadata, md)
    return worker.sample_and_accept_draft_tokens(logits, attn_metadata, md)


@pytest.mark.parametrize("kind", ["mtp", "eagle3", "mtp_eagle", "draft_target", "pard", "sa"])
def test_covered_worker_acceptance_on_shard_matches_gathered(kind):
    """Each covered worker's own acceptance (including PARD's 2K-row layout)
    accepts the same tokens and counts from the target logits' vocab shard as
    from the gathered logits, including the draft tokens accepted on tied
    maxima."""
    from tensorrt_llm._torch.speculative.draft_target import DraftTargetOneModelWorker
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker, MTPEagleWorker
    from tensorrt_llm._torch.speculative.mtp import MTPWorker
    from tensorrt_llm._torch.speculative.pard import PARDWorker
    from tensorrt_llm._torch.speculative.sa_worker import SAWorker

    cls = {
        "mtp": MTPWorker,
        "eagle3": Eagle3OneModelWorker,
        "mtp_eagle": MTPEagleWorker,
        "draft_target": DraftTargetOneModelWorker,
        "pard": PARDWorker,
        "sa": SAWorker,
    }[kind]
    tp = 4
    num_ctx, num_gens, k = 2, 5, 3
    rows_per_gen = 2 * k if kind == "pard" else k + 1
    vocab = 512 * tp
    gen = torch.Generator(device="cuda").manual_seed(tp * 7 + len(kind))
    full = torch.randint(
        0, 4, (num_ctx + num_gens * rows_per_gen, vocab), device="cuda", generator=gen
    )
    full = _tie_rows(full.to(torch.bfloat16), tp)
    # The target tokens acceptance compares: per generation request the first
    # k + 1 of its rows (all of them but PARD's mask positions).
    target = torch.argmax(full.float(), dim=-1)[num_ctx:].view(num_gens, rows_per_gen)
    drafts = target[:, :k].clone().to(torch.int32)
    # A different accepted length per request. The first request, whose rows
    # are the tie rows, accepts every draft token.
    for g in range(1, k + 1):
        drafts[g, g - 1] = (drafts[g, g - 1] + 1) % vocab
    if kind == "pard":
        # PARD's draft buffer holds 2K - 1 entries per request, the K drafts first.
        buffer = torch.zeros(num_gens, 2 * k - 1, dtype=torch.int32, device="cuda")
        buffer[:, :k] = drafts
        draft_tokens = buffer.flatten()
    else:
        draft_tokens = drafts.flatten()
    batch = num_ctx + num_gens
    attn_metadata = SimpleNamespace(num_seqs=batch, num_contexts=num_ctx)

    def metadata(sharded):
        return _greedy_md(
            sharded_target_logits=sharded,
            use_rejection_sampling=False,
            runtime_draft_len=k,
            draft_tokens=draft_tokens,
            mtp_hidden_states_manager=SimpleNamespace(sa_manager=None),
        )

    def flatten(accepted, num_accepted):
        return torch.cat(
            [accepted[:num_ctx, 0], accepted[num_ctx:, : k + 1].flatten(), num_accepted]
        ).to(torch.int64)

    ref = flatten(
        *_accept(kind, _bare_covered_worker(cls), full.float(), metadata(False), attn_metadata)
    )

    def accept_on_rank(rank, exchange):
        worker = _bare_covered_worker(cls)
        worker._vocab_parallel_argmax = True
        worker._target_pair_exchange = exchange
        worker._target_head_mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
        return flatten(
            *_accept(kind, worker, _shard(full, tp, rank), metadata(True), attn_metadata)
        )

    out = _run_ranks(tp, accept_on_rank)
    assert torch.equal(out, ref)
    num_accepted = ref[-batch:]
    assert num_accepted[num_ctx:].tolist() == [4, 1, 2, 3, 4]


def test_forward_marks_the_shard_logits():
    """``forward`` tells the engine the returned logits are the shard, so it
    gathers them for a logits post-processor."""

    class _Echo(_Worker):
        def _forward_impl(self, logits, spec_metadata):
            return {"logits": logits}

    md = _greedy_md(sharded_target_logits=True)
    assert _Echo()(logits=torch.zeros(1), spec_metadata=md)["logits_vocab_shard"]
    md.sharded_target_logits = False
    assert "logits_vocab_shard" not in _Echo()(logits=torch.zeros(1), spec_metadata=md)


@pytest.mark.parametrize("value, enabled", [(None, True), ("0", False)])
def test_kill_switch(value, enabled, monkeypatch):
    """``TLLM_SPEC_VOCAB_PARALLEL_ARGMAX=0`` keeps the model from enabling the
    path: no key exchange is built and the worker keeps the gathered logits."""
    from tensorrt_llm._torch.distributed import ll_exchange
    from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM

    if value is None:
        monkeypatch.delenv(VOCAB_PARALLEL_ARGMAX_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(VOCAB_PARALLEL_ARGMAX_ENV_VAR, value)
    assert vocab_parallel_argmax_enabled() == enabled

    created, enabled_with = [], []
    monkeypatch.setattr(
        ll_exchange,
        "create_tp_key_exchange",
        lambda mapping, max_rows: created.append(max_rows) or "key exchange",
    )
    worker = SimpleNamespace(
        verifies_on_target_shard=True,
        enable_vocab_parallel_argmax=lambda lm_head, ex: enabled_with.append(ex),
    )
    lm_head = _lm_head()
    model = SimpleNamespace(spec_worker=worker, lm_head=lm_head)
    # Sized by the batch bound: 16 requests x (1 + 3 drafts) rows.
    spec_config = SimpleNamespace(_max_batch_size=16, tokens_per_gen_step=4)
    model_config = SimpleNamespace(
        mapping=lm_head.mapping, max_num_tokens=8192, spec_config=spec_config
    )
    SpecDecOneEngineForCausalLM._enable_vocab_parallel_argmax(model, model_config, None)
    assert created == ([64] if enabled else [])
    assert enabled_with == (["key exchange"] if enabled else [])
    if enabled:
        # Without a batch bound in the spec config, by max_num_tokens.
        created.clear()
        model_config.spec_config = SimpleNamespace(tokens_per_gen_step=4)
        SpecDecOneEngineForCausalLM._enable_vocab_parallel_argmax(model, model_config, None)
        assert created == [8192]

    # A key exchange the fast paths already built is reused.
    created.clear()
    enabled_with.clear()
    model_config.spec_config = spec_config
    SpecDecOneEngineForCausalLM._enable_vocab_parallel_argmax(model, model_config, "fast path")
    assert created == []
    assert enabled_with == (["fast path"] if enabled else [])


def test_eagle3_draft_head_shard_decision():
    """An Eagle3 draft step projects only its head's vocab shard for greedy
    draft tokens of a vocab-sharded draft head with the plain processor."""
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

    def decide(advanced=False, guided=None, head=None, processor=None, enabled=True):
        worker = SimpleNamespace(_vocab_parallel_argmax=enabled, guided_decoder=guided)
        draft_model = SimpleNamespace(
            lm_head=head if head is not None else _lm_head(),
            logits_processor=processor or LogitsProcessor(),
        )
        return Eagle3OneModelWorker._draft_head_shards(worker, draft_model, advanced)

    assert decide()
    assert not decide(enabled=False)
    assert not decide(advanced=True)
    assert not decide(guided=object())
    assert not decide(head=_lm_head(tp=1))
    assert not decide(head=_lm_head(adp=True))

    class _Custom(LogitsProcessor):
        pass

    assert not decide(processor=_Custom())


def test_eagle3_draft_tokens_with_d2t(monkeypatch):
    """Greedy draft tokens of a vocab-sharded reduced-vocab draft head: the
    global argmax over the draft vocab, then d2t, the same as the gathered
    draft logits give."""
    from tensorrt_llm._torch.distributed import ops

    tp = 4
    draft_vocab = 1000 * tp
    rows = 9
    gen = torch.Generator(device="cuda").manual_seed(tp)
    full = torch.randint(-3, 3, (rows, draft_vocab), device="cuda", generator=gen).float()
    full = _tie_rows(full, tp)
    d2t = torch.randint(0, 5000, (draft_vocab,), device="cuda", generator=gen)
    reference = _Worker()
    reference._d2t = d2t
    md = SimpleNamespace(
        wants_advanced_draft_sampling=False,
        draft_vocab_size=draft_vocab,
        vocab_size=128000,
        is_cuda_graph=False,
    )
    reference.mapping = Mapping(world_size=1, tp_size=1, rank=0)
    expected = reference.sample_draft_tokens(full, md, rows)
    assert torch.equal(expected.long(), torch.argmax(full, -1) + d2t[torch.argmax(full, -1)])

    # The NCCL pair all-gather of the draft side, simulated: every rank's
    # (index, value) pairs concatenated in rank order.
    pairs = [None] * tp

    def fake_allgather(combined, mapping, dim=-1):
        return torch.cat(pairs, dim=-1)

    workers = []
    for r in range(tp):
        worker = _Worker()
        worker._d2t = d2t
        worker.mapping = Mapping(world_size=tp, tp_size=tp, rank=r)
        pairs[r] = worker._get_local_max_and_combined(_shard(full, tp, r))
        workers.append(worker)
    monkeypatch.setattr(ops, "allgather", fake_allgather)
    for r, worker in enumerate(workers):
        out = worker.sample_draft_tokens(_shard(full, tp, r), md, rows)
        assert torch.equal(out, expected), r


class _StubKeyExchange:
    """What BF16DraftHead reads from a TPKeyExchange (no peer memory; nothing is launched)."""

    transport = "stub"

    def __init__(self, nranks=2):
        self.nranks = nranks
        self.epoch = torch.ones(1, dtype=torch.int32, device="cuda")

    def channel_protocol_args(self, channel):
        assert channel == 0
        return self.epoch, [0] * self.nranks, 1024


def _fused_head_worker(monkeypatch, lm_head):
    monkeypatch.setattr(interface, "tp_pair_exchange", lambda mapping, strategy: "pairs")
    worker = _Worker()
    worker._fused_draft_lm_head = lambda draft_model: draft_model.lm_head
    assert worker.enable_vocab_parallel_argmax(lm_head, key_exchange=_StubKeyExchange())
    return worker


def _fused_capable():
    from tensorrt_llm._torch.modules.bf16_draft_head import max_rows

    return max_rows(1024, torch.cuda.current_device()) > 0


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "kill_switch",
        "no_vocab_parallel",
        "no_lm_head",
        "tp1",
        "attention_dp",
        "padded",
        "hidden_size",
        "fp32_weight",
        "small_shard",
    ],
)
def test_fused_draft_head_build(case, monkeypatch):
    """The worker builds the fused BF16 draft head only for a vocab-sharded BF16 draft head the
    kernel supports, with the vocab-parallel argmax on and the head's kill switch not set."""
    from tensorrt_llm._torch.modules import bf16_draft_head
    from tensorrt_llm._torch.modules.bf16_draft_head import FUSED_DRAFT_HEAD_ENV_VAR

    if case == "enabled" and not _fused_capable():
        pytest.skip("trtllm::lm_head_argmax needs SM 90+")
    head = _lm_head(hidden=1024)
    worker = _fused_head_worker(monkeypatch, head)
    if case != "small_shard":  # the test head is far below the production minimum
        monkeypatch.setattr(bf16_draft_head, "MIN_SHARD_BYTES", 0)
    draft_lm_head = head
    if case == "kill_switch":
        monkeypatch.setenv(FUSED_DRAFT_HEAD_ENV_VAR, "0")
    elif case == "no_vocab_parallel":
        worker._vocab_parallel_argmax = False
    elif case == "no_lm_head":
        worker._fused_draft_lm_head = lambda draft_model: None
    elif case == "tp1":
        draft_lm_head = _lm_head(tp=1, hidden=1024)
    elif case == "attention_dp":
        draft_lm_head = _lm_head(adp=True, hidden=1024)
    elif case == "padded":
        draft_lm_head = _lm_head(padded=True, hidden=1024)
    elif case == "hidden_size":
        draft_lm_head = _lm_head()  # 64: not a multiple of the kernel's 256
    elif case == "fp32_weight":
        draft_lm_head.weight.data = draft_lm_head.weight.data.float()
    worker.build_fused_draft_head(SimpleNamespace(lm_head=draft_lm_head))
    fused = worker.fused_draft_head
    if case != "enabled":
        assert fused is None
        return
    assert fused is not None and fused.max_rows == 8
    assert fused.vocab_offset == 0 and fused.round_logits
    md = SimpleNamespace(wants_advanced_draft_sampling=False)
    assert all(worker._fused_draft_head_for(rows, md) is fused for rows in range(1, 9))
    # Fallbacks decided per step.
    assert worker._fused_draft_head_for(9, md) is None
    assert worker._fused_draft_head_for(0, md) is None
    assert (
        worker._fused_draft_head_for(1, SimpleNamespace(wants_advanced_draft_sampling=True)) is None
    )
    worker.guided_decoder = object()
    assert worker._fused_draft_head_for(1, md) is None
    # A rebuild (weights reloaded) replaces it; the kill switch then drops it.
    monkeypatch.setenv(FUSED_DRAFT_HEAD_ENV_VAR, "0")
    worker.build_fused_draft_head(SimpleNamespace(lm_head=draft_lm_head))
    assert worker.fused_draft_head is None


def test_fused_draft_head_draft_lm_head_choice():
    """Which LM head the stock draft loops may fuse: MTP (vanilla and Eagle) through a plain
    DeepseekV3MTPHead projection of the shared head, Eagle3 through its own head and the plain
    logits processor."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3MTPHead
    from tensorrt_llm._torch.modules.bf16_draft_head import is_plain_mtp_shared_head
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
    from tensorrt_llm._torch.speculative.mtp import MTPWorker

    class _InheritsForward(DeepseekV3MTPHead):
        pass

    class _NormalizesInput(DeepseekV3MTPHead):
        def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
            raise NotImplementedError

    plain = object.__new__(_InheritsForward)
    custom = object.__new__(_NormalizesInput)
    assert is_plain_mtp_shared_head(plain) and not is_plain_mtp_shared_head(custom)

    head = object()

    def mtp_model(*shared_heads):
        return SimpleNamespace(
            lm_head=head, mtp_layers=[SimpleNamespace(shared_head=h) for h in shared_heads]
        )

    def eagle(is_mtp_eagle, draft_model):
        worker = SimpleNamespace(is_mtp_eagle=is_mtp_eagle)
        return Eagle3OneModelWorker._fused_draft_lm_head(worker, draft_model)

    assert MTPWorker._fused_draft_lm_head(None, mtp_model(plain, plain)) is head
    assert MTPWorker._fused_draft_lm_head(None, mtp_model(plain, custom)) is None
    assert eagle(True, mtp_model(plain)) is head
    assert eagle(True, mtp_model(custom)) is None

    class _Custom(LogitsProcessor):
        pass

    assert eagle(False, SimpleNamespace(lm_head=head, logits_processor=LogitsProcessor())) is head
    assert eagle(False, SimpleNamespace(lm_head=head, logits_processor=_Custom())) is None
