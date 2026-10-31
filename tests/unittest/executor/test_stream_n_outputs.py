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
"""Streaming with n > 1 must not repeat text in the outputs a response does not update.

Without beam search every response carries one sequence (``sequence_index``),
but detokenization runs over all outputs. An output whose previous diff was
not marked as consumed was decoded again, so its text repeated decoded pieces.
CPU-only: the responses are stubs that go through the real
``GenerationResultBase._handle_response`` and
``DetokenizedGenerationResultBase._handle_response``.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor.result import DetokenizedGenerationResultBase, Logprob
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(t) for t in token_ids)

    def decode_incrementally(self, token_ids, prev_text=None, states=None, *, flush=False,
                             stream_interval=1, **kwargs):
        return (prev_text or "") + self.decode(token_ids), states


class _Result(SimpleNamespace):
    """Fields the handler reads that these tests do not set are absent (None)."""

    def __getattr__(self, name):
        return None


def _response(seq_idx, ids, finish_reason, is_final, cum_log_prob=None):
    result = _Result(sequence_index=seq_idx, output_token_ids=[ids], finish_reasons=[finish_reason],
                     is_final=is_final,
                     log_probs=[[{t: Logprob(logprob=-0.5, rank=1)} for t in ids]],
                     cum_log_probs=None if cum_log_prob is None else [cum_log_prob])
    return SimpleNamespace(has_error=lambda: False, result=result)


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("n,best_of", [(2, 2), (2, 3)])
def test_outputs_match_their_tokens(streaming, n, best_of):
    texts = ["alpha beta gamma", "one two three four five", "x y"][:best_of]
    sp = SamplingParams(n=n, best_of=best_of, temperature=1.0, max_tokens=64, logprobs=1)
    tok = CharTokenizer()
    sp._setup(tok, None, None)
    result = DetokenizedGenerationResultBase(0, sp, tokenizer=tok, streaming=streaming)
    ids = [tok.encode(t) for t in texts]
    pos = [0] * best_of
    streamed = [""] * best_of
    streamed_ids = [[] for _ in range(best_of)]
    streamed_logprobs = [0] * best_of
    step = 3 if streaming else 10**6
    order = []
    while any(pos[k] < len(ids[k]) for k in range(best_of)):
        for k in range(best_of):
            if pos[k] < len(ids[k]):
                order.append((k, ids[k][pos[k]:pos[k] + step]))
                pos[k] += step
    for i, (k, chunk) in enumerate(order):
        last_of_k = all(kk != k for kk, _ in order[i + 1:])
        reason = tllm.FinishReason.LENGTH if last_of_k else tllm.FinishReason.NOT_FINISHED
        result._handle_response(_response(k, chunk, reason, i == len(order) - 1))
        for j, out in enumerate(result._outputs):
            if j != k:
                # Outputs this response did not update report nothing new.
                assert out.text_diff == "" and out.token_ids_diff == [] and len(out.logprobs_diff) == 0
            streamed[j] += out.text_diff
            streamed_ids[j] += out.token_ids_diff
            streamed_logprobs[j] += len(out.logprobs_diff)
    for j, out in enumerate(result._outputs):
        assert out.token_ids == ids[j]
        assert len(out.logprobs) == len(ids[j])
        if streaming:
            assert streamed_ids[j] == ids[j]
            assert streamed_logprobs[j] == len(ids[j])
    # Only the first n of best_of sequences are returned, and only those are detokenized.
    for j, out in enumerate(result.outputs):
        assert out.text == texts[j]
        if streaming:
            assert streamed[j] == texts[j]


# (sequence, chunk, cumulative logprob after the chunk). Sequence 1 starts outside the top n=1 and is
# selected later: in the first case after two responses of its own, in the second because the other
# sequence's score drops.
_RERANK_CASES = {
    "selected_after_own_responses": [(0, "aaa", -1.0), (1, "bbb", -5.0), (0, "ccc", -3.0),
                                      (1, "ddd", -2.0), (0, "eee", -9.0)],
    "selected_by_other_response": [(0, "aaa", -1.0), (1, "bbb", -5.0), (0, "ccc", -9.0),
                                   (1, "ddd", -6.0)],
}


@pytest.mark.parametrize("case", sorted(_RERANK_CASES))
def test_streaming_best_of_rerank_keeps_text(case):
    steps = _RERANK_CASES[case]
    sp = SamplingParams(n=1, best_of=2, temperature=1.0, max_tokens=64, logprobs=1)
    tok = CharTokenizer()
    sp._setup(tok, None, None)
    result = DetokenizedGenerationResultBase(0, sp, tokenizer=tok, streaming=True)
    for i, (k, chunk, cum) in enumerate(steps):
        last_of_k = all(kk != k for kk, _, _ in steps[i + 1:])
        reason = tllm.FinishReason.LENGTH if last_of_k else tllm.FinishReason.NOT_FINISHED
        result._handle_response(
            _response(k, tok.encode(chunk), reason, i == len(steps) - 1, cum_log_prob=cum))
    (selected,) = result.outputs
    assert selected is result._outputs[1]
    assert selected.text == tok.decode(selected.token_ids)
