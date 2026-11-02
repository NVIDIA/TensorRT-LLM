# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA tests of the pieces the Beam NVFP4 target composes for DFlash 2 speculative decoding.

Generated inputs, one CUDA device, no model files. The pieces are the native ones the target
reuses -- the DFlash worker's greedy verifier and candidate merge, the DFlash 2 selector, the
one-model speculative sampler, the DFlash capture buffer -- plus the drafter's logit softcap
the target installs (``DFlashSoftcapLogitsProcessor``). What is checked:

* the drafter's unary logits are ``cap * tanh(raw / cap)`` exactly once, and the candidate
  path the selector picks discriminates an omitted and a doubled cap (two cases whose
  bilinear bonus sits between the once-capped and the faulty scores), with the candidates on
  different TP vocabulary shards merged by the worker's global top-k;
* greedy verification on identical logits: zero, partial and full acceptance of seven
  proposals in one four-request step with unequal counts, the target's argmax as correction
  or bonus token, ties to the lowest id, and the committed length ``1 + accepted``;
* stops inside a verification block: an end id among the accepted tokens or as the bonus
  token, and completion budgets 1, 7, 8, 9 and 64 against a full block, emit nothing past
  the stop or budget;
* the capture buffer: each capture layer's rows land in its own 5120-wide slice in ascending
  layer order, and a capture at a neighbouring layer is detected.
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch.distributed import ops as dist_ops
from tensorrt_llm._torch.models.modeling_dflash import (
    DFlash2CandidateSelector,
    DFlashForCausalLM,
    DFlashSoftcapLogitsProcessor,
    dflash_final_logit_softcap,
)
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.bindings.executor import SamplingConfig as ExecutorSamplingConfig
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

K = 7  # proposals per block: the anchor plus seven rows of a verification block
CAP = 30.0


def _worker(tp_rank: int = 0, tp_size: int = 1) -> DFlashWorker:
    config = DFlashDecodingConfig(max_draft_len=K, attention_backend="TRTLLM")
    return DFlashWorker(config, Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_rank))


def _spec_metadata(num_requests: int = 8) -> DFlashSpecMetadata:
    md = DFlashSpecMetadata(
        max_draft_len=K,
        max_total_draft_tokens=K,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        max_num_requests=num_requests,
    )
    md.runtime_draft_len = K
    md.runtime_tokens_per_gen_step = K + 1
    return md


# ---------------------------------------------------------------------------------------------
# Draft logit softcap and the DFlash 2 candidate selector
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def test_softcap_processor_caps_the_head_output_once():
    head = torch.nn.Linear(16, 24, bias=False, device="cuda", dtype=torch.bfloat16)
    hidden = 8 * torch.randn(5, 16, device="cuda", dtype=torch.bfloat16)
    raw = head(hidden).float()
    capped = DFlashSoftcapLogitsProcessor(CAP)(hidden, head, None, True)
    assert capped.dtype == torch.float32
    assert torch.equal(capped, CAP * torch.tanh(raw / CAP))
    assert float(capped.abs().max()) < CAP
    assert torch.equal(LogitsProcessor()(hidden, head, None, True), raw)


def test_softcap_value_comes_from_the_drafter_config():
    config = types.SimpleNamespace(dflash_config={"final_logit_softcapping": 30})
    assert dflash_final_logit_softcap(config) == 30.0
    assert dflash_final_logit_softcap(types.SimpleNamespace(dflash_config={})) is None
    for bad in (0, -1.0, "30", True):
        with pytest.raises(ValueError):
            dflash_final_logit_softcap(
                types.SimpleNamespace(dflash_config={"final_logit_softcapping": bad})
            )
    with pytest.raises(ValueError):
        DFlashSoftcapLogitsProcessor(0.0)


VOCAB, SHARDS = 32, 4
SHARD = VOCAB // SHARDS
FIRST, SECOND, ANCHOR = 5, 27, 12  # candidates on shards 0 and 3; the anchor on shard 1
HIDDEN = 16


def _selector(bonus: float) -> DFlash2CandidateSelector:
    """A rank-2 selector whose bilinear term adds ``bonus`` to SECOND after the anchor only."""
    predecessor = torch.zeros(VOCAB, 2, dtype=torch.bfloat16, device="cuda")
    predecessor[ANCHOR, 0] = 1.0
    successor = torch.zeros(VOCAB, 2, dtype=torch.bfloat16, device="cuda")
    successor[SECOND, 0] = bonus
    projection = torch.zeros(2, HIDDEN, dtype=torch.bfloat16, device="cuda")
    projection[0, 0] = 1.0
    return DFlash2CandidateSelector(predecessor, successor, projection, 2, VOCAB, 2)


class _ShardHead(torch.nn.Module):
    """Stands for one TP rank's vocabulary-parallel draft head: returns that rank's raw slice."""

    def __init__(self, raw: torch.Tensor, rank: int):
        super().__init__()
        self.raw = raw[..., rank * SHARD : (rank + 1) * SHARD]

    def forward(self, hidden):
        return self.raw.expand(hidden.shape[0], SHARD)


def _selected_token(processor_layers: int, bonus: float, monkeypatch) -> int:
    """The draft token the native path picks for one block position, emulating four TP ranks.

    Each rank applies the draft logits processor ``processor_layers`` times (0 = the uncapped
    native processor, 1 = the target's capped one, 2 = a doubled cap) to its vocabulary shard;
    the worker's global top-k merges the shards through an all-gather of every rank's
    (id, value) pairs; the selector scores the merged candidates after the anchor; the worker's
    greedy draft sampler reads the result.
    """
    raw = torch.full((VOCAB,), -100.0, device="cuda")
    raw[FIRST], raw[SECOND] = 60.0, 20.0
    hidden = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    hidden[0, 0] = 1.0
    spec = types.SimpleNamespace(draft_vocab_size=VOCAB, vocab_size=VOCAB)

    def shard_logits(rank):
        processor = (
            LogitsProcessor() if processor_layers == 0 else DFlashSoftcapLogitsProcessor(CAP)
        )
        logits = processor(hidden, _ShardHead(raw, rank), None, True)
        if processor_layers == 2:
            logits = CAP * torch.tanh(logits / CAP)
        return logits.view(1, 1, SHARD)

    gathered = []
    monkeypatch.setattr(dist_ops, "allgather", lambda t, mapping, dim=-1: gathered.append(t) or t)
    for rank in range(SHARDS):
        _worker(rank, SHARDS)._dflash2_global_top_k(shard_logits(rank), spec, 2, VOCAB)
    merged = torch.cat(gathered, dim=-1)
    monkeypatch.setattr(dist_ops, "allgather", lambda t, mapping, dim=-1: merged)
    picks = set()
    for rank in range(SHARDS):
        worker = _worker(rank, SHARDS)
        candidate_ids, unary, block_logits = worker._dflash2_global_top_k(
            shard_logits(rank), spec, 2, VOCAB
        )
        assert sorted(candidate_ids.view(-1).tolist()) == [FIRST, SECOND]
        selector = types.SimpleNamespace(candidate_selector=_selector(bonus))
        block_logits = DFlashForCausalLM.select_candidate_path(
            selector,
            block_logits,
            candidate_ids,
            unary,
            hidden.view(1, 1, HIDDEN),
            torch.tensor([ANCHOR], device="cuda"),
        )
        draft_spec = types.SimpleNamespace(
            wants_advanced_draft_sampling=False, draft_vocab_size=VOCAB, vocab_size=VOCAB
        )
        picks.add(int(worker.sample_draft_tokens(block_logits, draft_spec, 1).view(-1)[0]))
    assert len(picks) == 1, "the TP ranks disagree on the draft token"
    return picks.pop()


def test_selector_path_detects_an_omitted_cap(monkeypatch):
    """Bonus 20: capped [28.92, 17.48 + 20] picks SECOND; uncapped [60, 20 + 20] picks FIRST."""
    assert _selected_token(1, 20.0, monkeypatch) == SECOND
    assert _selected_token(0, 20.0, monkeypatch) == FIRST
    assert _selected_token(2, 20.0, monkeypatch) == SECOND


def test_selector_path_detects_a_doubled_cap(monkeypatch):
    """Bonus 9: capped [28.92, 17.48 + 9] picks FIRST; doubled [22.38, 15.74 + 9] picks SECOND."""
    assert _selected_token(1, 9.0, monkeypatch) == FIRST
    assert _selected_token(2, 9.0, monkeypatch) == SECOND
    assert _selected_token(0, 9.0, monkeypatch) == FIRST


# ---------------------------------------------------------------------------------------------
# Greedy verification on identical logits
# ---------------------------------------------------------------------------------------------

V = 64


def _rows(argmaxes) -> torch.Tensor:
    """FP32 logits rows whose argmax is each of ``argmaxes``."""
    logits = torch.linspace(-1.0, 1.0, V, device="cuda").repeat(len(argmaxes), 1)
    for row, token in enumerate(argmaxes):
        logits[row, token] = 50.0
    return logits


def _verify(contexts, blocks):
    """Run the worker's acceptance on a step of ``contexts`` (argmax of each context's last
    row) and ``blocks`` (per generation request: (row argmaxes [8], proposals [7]))."""
    worker = _worker()
    md = _spec_metadata()
    rows = list(contexts) + [t for argmaxes, _ in blocks for t in argmaxes]
    md.draft_tokens = torch.tensor(
        [d for _, proposals in blocks for d in proposals], dtype=torch.int32, device="cuda"
    )
    attn = types.SimpleNamespace(num_seqs=len(contexts) + len(blocks), num_contexts=len(contexts))
    accepted, counts = worker.sample_and_accept_draft_tokens(_rows(rows), attn, md)
    return accepted.cpu(), counts.cpu()


def test_zero_partial_and_full_acceptance_in_one_step():
    """Four requests with 0, 3, 7 and 5 accepted proposals beside one context request."""
    target = [list(range(10 + 8 * r, 18 + 8 * r)) for r in range(4)]
    proposals = [
        [63] + target[0][1:7],  # first proposal wrong: nothing accepted
        target[1][:3] + [63, 63, 63, 63],  # three accepted, then the correction
        target[2][:7],  # all seven accepted, then the bonus
        target[3][:5] + [63, target[3][6]],  # a later match after a mismatch does not count
    ]
    accepted, counts = _verify([40], list(zip(target, proposals)))
    assert counts.tolist() == [1, 1, 4, 8, 6]
    assert accepted[0, 0] == 40
    for r, n in zip(range(4), (1, 4, 8, 6)):
        # The emitted tokens are the target's own argmaxes: accepted proposals, then the
        # correction (or, after full acceptance, the bonus), nothing past them.
        assert accepted[1 + r, :n].tolist() == target[r][:n]


def test_ties_take_the_lowest_token_id():
    logits = _rows([7, 7, 7, 7, 7, 7, 7, 7])
    logits[0, 3] = 50.0  # equal maximum at a lower id than 7
    worker, md = _worker(), _spec_metadata()
    md.draft_tokens = torch.tensor([7] * K, dtype=torch.int32, device="cuda")
    attn = types.SimpleNamespace(num_seqs=1, num_contexts=0)
    accepted, counts = worker.sample_and_accept_draft_tokens(logits, attn, md)
    assert counts.tolist() == [1] and accepted[0, 0] == 3


# ---------------------------------------------------------------------------------------------
# Stops inside a verification block (the one-model speculative sampler)
# ---------------------------------------------------------------------------------------------

EOS = 60


def _sampler() -> SpecSampler:
    return SpecSampler(
        TorchSampler.Args(
            max_seq_len=8192,
            max_draft_len=K,
            max_num_sequences=8,
            max_beam_width=1,
            max_total_draft_tokens=K,
            disable_overlap_scheduler=True,
        )
    )


class _Request(LlmRequest):
    """An ``LlmRequest`` that keeps the finish reason the sampler records on it."""

    py_finish_reason = None

    def finish_by(self, reason: FinishReason, beam: int) -> None:
        super().finish_by(reason, beam)
        self.py_finish_reason = reason


def _request(slot: int, budget: int) -> LlmRequest:
    request = _Request(
        request_id=slot + 1,
        max_new_tokens=budget,
        input_tokens=[1, 2, 3],
        sampling_config=ExecutorSamplingConfig(beam_width=1, top_k=1, temperature=0.0),
        is_streaming=False,
        end_id=EOS,
    )
    request.py_seq_slot = slot
    return request


def _step(sampler, requests, outputs, *, is_context):
    batch = ScheduledRequests()
    if is_context:
        batch.context_requests_last_chunk = list(requests)
    else:
        batch.generation_requests = list(requests)
    state = sampler.sample_async(batch, outputs, [0])
    if is_context:
        for request in requests:
            request.context_chunk_size = request.prompt_len
            request.move_to_next_context_chunk()
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
    sampler.update_requests(state)


def _outputs(worker, md, accepted, counts):
    """The worker's result dictionary for a step whose drafting produced no proposals."""
    batch = accepted.shape[0]
    next_draft = torch.zeros(batch, K, dtype=torch.int32, device="cuda")
    return {
        "new_tokens": accepted,
        "new_tokens_lens": counts,
        "next_draft_tokens": next_draft,
        "next_new_tokens": worker._prepare_next_new_tokens(
            accepted, next_draft, md.batch_indices_cuda, batch, counts
        ),
    }


def _step_metadata(num_requests: int) -> DFlashSpecMetadata:
    md = _spec_metadata()
    md.request_ids = list(range(num_requests))
    md.prepare()  # batch_indices_cuda = arange(num_requests)
    return md


def _run_block(budgets, block_argmaxes, proposals):
    """A context step (first token 9 for every request), then one verification step."""
    sampler, worker = _sampler(), _worker()
    requests = [_request(i, b) for i, b in enumerate(budgets)]
    md = _step_metadata(len(requests))
    attn = types.SimpleNamespace(num_seqs=len(requests), num_contexts=len(requests))
    accepted, counts = worker.sample_and_accept_draft_tokens(_rows([9] * len(requests)), attn, md)
    _step(sampler, requests, _outputs(worker, md, accepted, counts), is_context=True)
    live = [r for r in requests if r.state != LlmRequestState.GENERATION_COMPLETE]
    if live:
        md = _step_metadata(len(live))
        md.draft_tokens = torch.tensor(
            [d for r in live for d in proposals[r.py_seq_slot]], dtype=torch.int32, device="cuda"
        )
        attn = types.SimpleNamespace(num_seqs=len(live), num_contexts=0)
        rows = [t for r in live for t in block_argmaxes[r.py_seq_slot]]
        accepted, counts = worker.sample_and_accept_draft_tokens(_rows(rows), attn, md)
        _step(sampler, live, _outputs(worker, md, accepted, counts), is_context=False)
    return requests


def _generated(request):
    return request.get_tokens(0)[request.prompt_len :]


def test_end_id_inside_and_at_the_end_of_a_block():
    block = list(range(20, 28))
    inside = block[:2] + [EOS] + block[3:]  # the end id is the third accepted token
    bonus = block[:7] + [EOS]  # all seven proposals accepted, the bonus is the end id
    requests = _run_block(
        [64, 64, 64],
        [inside, bonus, block],
        [inside[:7], bonus[:7], block[:4] + [63, 63, 63]],
    )
    assert _generated(requests[0]) == [9, 20, 21, EOS]
    assert requests[0].py_finish_reason == FinishReason.END_ID
    assert _generated(requests[1]) == [9] + block[:7] + [EOS]
    assert requests[1].py_finish_reason == FinishReason.END_ID
    # Four accepted then the correction: five tokens, still running.
    assert _generated(requests[2]) == [9] + block[:5]
    assert requests[2].state != LlmRequestState.GENERATION_COMPLETE
    assert requests[2].py_num_accepted_draft_tokens == 4


@pytest.mark.parametrize("budget", [1, 7, 8, 9, 64])
def test_completion_budget_inside_a_full_block(budget):
    """One token from the context step, then a fully accepted block of eight."""
    block = list(range(30, 38))
    (request,) = _run_block([budget], [block], [block[:7]])
    expected = ([9] + block)[:budget]
    assert _generated(request) == expected
    finished = request.state == LlmRequestState.GENERATION_COMPLETE
    assert finished == (budget <= 9)
    assert request.py_finish_reason == (FinishReason.LENGTH if finished else None)


# ---------------------------------------------------------------------------------------------
# The DFlash capture buffer
# ---------------------------------------------------------------------------------------------

LAYERS = [4, 16, 28, 40, 46]
WIDTH = 5120


def _tagged(layer: int, rows: int) -> torch.Tensor:
    """Rows ``(layer, row)``-distinguishable in BF16: layer in column 0, row in column 1."""
    x = torch.zeros(rows, WIDTH, dtype=torch.bfloat16, device="cuda")
    x[:, 0] = layer
    x[:, 1] = torch.arange(rows, device="cuda", dtype=torch.bfloat16)
    return x


def test_capture_slices_follow_the_layer_order():
    md = DFlashSpecMetadata(
        max_draft_len=K,
        max_total_draft_tokens=K,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        max_num_requests=4,
        layers_to_capture=list(reversed(LAYERS)),
        hidden_size=WIDTH,
        max_num_tokens=64,
        dtype=torch.bfloat16,
    )
    for layer in range(52):
        md.maybe_capture_hidden_states(layer, _tagged(layer, 24))
    captured = md.get_hidden_states(24)
    assert captured.shape == (24, 5 * WIDTH)
    for i, layer in enumerate(LAYERS):
        assert torch.equal(captured[:, i * WIDTH : (i + 1) * WIDTH], _tagged(layer, 24))


@pytest.mark.parametrize("offset", [-1, 1])
def test_capture_at_a_neighbouring_layer_is_detected(offset):
    """A target that taps layer l +/- 1 instead of l leaves a slice that is not layer l's."""
    md = DFlashSpecMetadata(
        max_draft_len=K,
        max_total_draft_tokens=K,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        max_num_requests=4,
        layers_to_capture=LAYERS,
        hidden_size=WIDTH,
        max_num_tokens=64,
        dtype=torch.bfloat16,
    )
    md.captured_hidden_states.fill_(-1)
    for layer in range(52):
        md.maybe_capture_hidden_states(layer, _tagged(layer + offset, 16))
    captured = md.get_hidden_states(16)
    wrong = [
        i
        for i, layer in enumerate(LAYERS)
        if not torch.equal(captured[:, i * WIDTH : (i + 1) * WIDTH], _tagged(layer, 16))
    ]
    assert wrong == list(range(5))
