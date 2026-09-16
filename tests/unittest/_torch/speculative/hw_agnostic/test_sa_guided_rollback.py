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
"""GPU unit tests: suffix-automaton speculation under guided decoding.

``CapturableGuidedDecoder.execute`` advances every grammar matcher through the
golden position and all draft tokens (as far as the grammar allows) to mask the
target logits. Drafters with a draft forward undo the rejected tail in
``execute_draft_batch(draft_step=0)``; the suffix automaton drafts from the
token history and never enters that loop, so ``SAWorker`` has to roll the
matchers back itself through ``rollback_rejected_batch``. Without it the next
verification masks from a grammar state that still contains rejected draft
tokens, and grammar-forbidden tokens get through.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.guided_decoder import (
    CapturableGuidedDecoder,
    GuidedRequest,
    GuidedRequests,
)
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.sa_worker import SASpecMetadata, SAWorker
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi import SADecodingConfig
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the guided decoder runs host callbacks on a CUDA stream",
    ),
    # The CUDA runtime executes host callbacks on its own thread, which Python
    # registers as a dummy thread the first time it runs Python code.
    pytest.mark.threadleak(enabled=False),
]

# A raw vocabulary just large enough for the built-in JSON grammar.
VOCAB = ["<eos>", "{", "}", '"', "a", "b", ":", ",", "1", " ", "[", "]", "t", "r", "u", "e"]
TOK = {text: i for i, text in enumerate(VOCAB)}
EOS, LBRACE, QUOTE, A, COLON, ONE = (TOK[t] for t in ("<eos>", "{", '"', "a", ":", "1"))
VOCAB_PADDED = 64
K = 3  # draft tokens per step


def _make_decoder() -> CapturableGuidedDecoder:
    pytest.importorskip("xgrammar")
    config = GuidedDecodingConfig(
        backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        encoded_vocab=VOCAB,
        stop_token_ids=[EOS],
    )
    return CapturableGuidedDecoder(
        config, max_num_sequences=2, vocab_size_padded=VOCAB_PADDED, max_num_draft_tokens=K
    )


def _json_request(**overrides) -> GuidedRequest:
    fields = dict(
        guided_decoding_params=GuidedDecodingParams(guide_type=GuidedDecodingParams.GuideType.JSON),
        request_id=7,
        seq_slot=0,
        draft_tokens=[],
    )
    fields.update(overrides)
    return GuidedRequest(**fields)


def _target_step(decoder: CapturableGuidedDecoder, request: GuidedRequest, new_tokens=None):
    """One target verification for a hand-built request.

    Mirrors ``CapturableGuidedDecoder.add_batch`` (requests snapshot, new-token
    column, queue hand-off) and then runs ``execute`` on zero logits, so the
    applied mask is readable as ``-inf`` entries.
    """
    requests = GuidedRequests(
        [request],
        num_contexts=int(request.is_context_init_state),
        num_generations=int(request.is_generation_in_progress_state),
        max_num_draft_tokens=K,
    )
    decoder.requests = requests
    if new_tokens is not None:
        decoder.new_tokens[:, request.seq_slot].copy_(torch.tensor(new_tokens, dtype=torch.int32))
    decoder.queue.put((requests, new_tokens is not None))
    logits = torch.zeros(requests.num_bitmask_tokens, VOCAB_PADDED, device="cuda")
    # Model-engine preprocessing records this event before target execution.
    # Record it again after capture, where events represent graph dependencies.
    decoder.token_event.record()
    decoder.execute(logits)
    torch.cuda.synchronize()
    return logits


def _generation_request() -> GuidedRequest:
    return _json_request(is_generation_in_progress_state=True, prev_seq_slot=0)


def _open_object_then_verify_drafts(decoder: CapturableGuidedDecoder) -> None:
    """Context step, then a verification whose drafts `"`, `a`, `"` all pass the grammar."""
    _target_step(decoder, _json_request(is_context_init_state=True, is_last_context_chunk=True))
    assert decoder.grammar_matchers[0] is not None
    _target_step(decoder, _generation_request(), [LBRACE, QUOTE, A, QUOTE])
    # Golden `{` plus three grammatical drafts: the matcher sits after `{"a"`.
    assert decoder.num_advanced_tokens[0] == 1 + K


@pytest.mark.parametrize("num_accepted", [1, 2, 3, 4])
def test_rollback_rejected_batch_restores_the_accepted_prefix(num_accepted: int) -> None:
    decoder = _make_decoder()
    _open_object_then_verify_drafts(decoder)

    decoder.rollback_rejected_batch(torch.tensor([num_accepted], dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()

    # Continue from each possible accepted prefix of the JSON object.
    continuation = [LBRACE, QUOTE, A, QUOTE, COLON, ONE, ONE, ONE]
    logits = _target_step(
        decoder, _generation_request(), continuation[num_accepted : num_accepted + 4]
    )
    assert decoder.num_advanced_tokens[0] == 1 + K
    assert logits[0, EOS] == float("-inf")


@pytest.mark.parametrize("num_accepted", [1, 2, 3, 4])
def test_rollback_after_terminal_draft(num_accepted: int) -> None:
    decoder = _make_decoder()
    _target_step(decoder, _json_request(is_context_init_state=True, is_last_context_chunk=True))
    _target_step(decoder, _generation_request(), [LBRACE, TOK["}"], EOS, A])
    matcher = decoder.grammar_matchers[0]
    assert matcher.is_terminated()
    assert decoder.num_advanced_tokens[0] == 3

    decoder.rollback_rejected_batch(torch.tensor([num_accepted], dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    assert matcher.is_terminated() == (num_accepted >= 3)
    assert (
        decoder.requests_hostfunc.requests[0].num_accepted_draft_tokens == min(num_accepted, 3) - 1
    )
    if num_accepted == 1:
        logits = _target_step(decoder, _generation_request(), [QUOTE, A, QUOTE, COLON])
        assert decoder.num_advanced_tokens[0] == 4
        assert logits[0, EOS] == float("-inf")
        assert logits[0, A] == 0.0
    elif num_accepted == 2:
        # The completed object accepts EOS, not trailing whitespace. Replaying
        # EOS must advance the restored matcher and terminate it again.
        _target_step(decoder, _generation_request(), [EOS, A, A, A])
        assert decoder.num_advanced_tokens[0] == 1
        assert matcher.is_terminated()


def test_rollback_rejected_batch_joins_the_captured_stream() -> None:
    decoder = _make_decoder()
    _open_object_then_verify_drafts(decoder)
    counts = torch.tensor([2], dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        decoder.rollback_rejected_batch(counts)
    graph.replay()
    torch.cuda.synchronize()
    logits = _target_step(decoder, _generation_request(), [A, QUOTE, COLON, ONE])
    assert decoder.num_advanced_tokens[0] == 4
    assert logits[0, EOS] == float("-inf")
    assert logits[0, QUOTE] == 0.0


def test_without_the_rollback_the_next_verification_rejects_the_golden_token():
    """The control for the test above: the pre-fix state machine fails here."""
    decoder = _make_decoder()
    _open_object_then_verify_drafts(decoder)

    # No rollback: the matcher still sits after `{"a"`, where another `a` is
    # not grammatical, so the build rejects the golden token and masks nothing.
    logits = _target_step(decoder, _generation_request(), [A, QUOTE, COLON, ONE])
    assert decoder.num_advanced_tokens[0] == 0
    assert torch.all(logits == 0.0)


class _RecordingGuidedDecoder:
    def __init__(self):
        self.calls = []

    def execute(self, logits, d2t=None):
        self.calls.append(("execute", None))

    def rollback_rejected_batch(self, num_accepted_tokens):
        self.calls.append(("rollback_rejected_batch", num_accepted_tokens.clone()))


def _worker_inputs(batch_size, num_contexts, runtime_draft_len):
    meta = SASpecMetadata(
        max_num_requests=8,
        max_draft_len=K,
        max_total_draft_tokens=K,
        spec_dec_mode=SpeculativeDecodingMode.SA,
        runtime_draft_len=runtime_draft_len,
    )
    attn = types.SimpleNamespace(
        num_seqs=batch_size,
        num_contexts=num_contexts,
        kv_cache_manager=None,
        has_spec_dec_saved_state=False,
    )
    input_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    hidden = torch.zeros(batch_size, 8, device="cuda")
    logits = torch.zeros(batch_size, VOCAB_PADDED, device="cuda")
    return meta, attn, input_ids, position_ids, hidden, logits


def test_sa_worker_rolls_back_after_verification(monkeypatch):
    worker = SAWorker(SADecodingConfig(max_draft_len=K))
    guided = _RecordingGuidedDecoder()
    worker.guided_decoder = guided

    batch_size, num_contexts = 3, 1
    meta, attn, *inputs = _worker_inputs(batch_size, num_contexts, runtime_draft_len=K)
    accepted = torch.zeros(batch_size, K + 1, dtype=torch.int32, device="cuda")
    num_accepted = torch.tensor([1, 2, K + 1], dtype=torch.int32, device="cuda")
    drafts = torch.zeros(batch_size, K, dtype=torch.int32, device="cuda")
    next_new = torch.zeros(batch_size, K + 1, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        worker, "_sample_and_accept_draft_tokens", lambda *a, **k: (accepted, num_accepted)
    )

    def generate_drafts(*args, **kwargs):
        guided.calls.append(("draft", None))
        return drafts

    monkeypatch.setattr(worker, "_generate_draft_tokens", generate_drafts)
    monkeypatch.setattr(worker, "_prepare_next_new_tokens", lambda *a, **k: next_new)

    out = worker.forward(*inputs, attn, meta)

    # Target mask first, then exactly one rollback fed the verification's own counts.
    assert [name for name, _ in guided.calls] == ["execute", "draft", "rollback_rejected_batch"]
    assert torch.equal(guided.calls[2][1], num_accepted)
    assert torch.equal(out["new_tokens_lens"], num_accepted)


def test_sa_worker_skips_the_rollback_when_not_drafting(monkeypatch):
    """A zero runtime draft length verifies nothing, so there is nothing to roll back."""
    worker = SAWorker(SADecodingConfig(max_draft_len=K))
    guided = _RecordingGuidedDecoder()
    worker.guided_decoder = guided

    batch_size = 2
    meta, attn, *inputs = _worker_inputs(batch_size, 0, runtime_draft_len=0)
    sampled = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(worker, "_sample_tokens_for_batch", lambda *a, **k: sampled)

    out = worker.forward(*inputs, attn, meta)

    assert [name for name, _ in guided.calls] == ["execute"]
    assert out["next_draft_tokens"].shape == (batch_size, 0)
