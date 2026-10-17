# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-token beam stops do not need prompt or generated-token history."""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.bindings.executor import FinishReason

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _handler(beam_width: int = 4, history_length: int = 20) -> FinishReasonsHandler:
    return FinishReasonsHandler(
        max_stop_word_length=history_length,
        max_num_stop_words=1,
        max_num_sequences=5,
        max_beam_width=beam_width,
        max_tokens=1,
        max_seq_len=64,
    )


def _request(
    slot: int,
    words: list[list[int]] | None,
    *,
    beam_width: int = 4,
    prompt: list[int] | None = None,
    max_new_tokens: int = 10,
    end_id: int = 17,
    streaming: bool = False,
) -> LlmRequest:
    return LlmRequest(
        request_id=slot,
        seq_slot=slot,
        input_tokens=[2, 7] if prompt is None else prompt,
        max_new_tokens=max_new_tokens,
        end_id=end_id,
        sampling_config=SamplingConfig(beam_width=beam_width),
        is_streaming=streaming,
        stop_words_list=words,
    )


def _admit(
    handler: FinishReasonsHandler,
    new_requests: list[LlmRequest],
    all_requests: list[LlmRequest] | None = None,
) -> None:
    handler.setup_new_request_handling()
    for request in new_requests:
        handler.prepare_for_new_request(request)
    slots = torch.tensor([r.py_seq_slot for r in new_requests], dtype=torch.int64, device="cpu")
    handler.update_for_new_request(
        seq_slots_cuda_long=slots.cuda(),
        max_lengths_cuda=torch.tensor(handler.new_max_lens, dtype=torch.int32, device="cuda"),
        end_ids_cuda=torch.tensor(handler.new_end_ids, dtype=torch.int32, device="cuda"),
        seq_slots_host=slots,
        all_sampling_requests=new_requests if all_requests is None else all_requests,
    )


def _step(
    handler: FinishReasonsHandler,
    requests: list[LlmRequest],
    tokens: list[list[int]],
    step: int = 0,
    first_reasons: torch.Tensor | None = None,
) -> torch.Tensor:
    slots = torch.tensor([r.py_seq_slot for r in requests], dtype=torch.int64, device="cpu")
    new_tokens = torch.zeros((1, 6, handler._max_beam_width), dtype=torch.int32, device="cuda")
    new_tokens[0, slots.cuda()] = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    result = handler.write_finish_reasons(
        seq_slots_host=slots,
        is_draft_batch=False,
        seq_slots_cuda=slots.cuda(),
        seq_lens_cuda=torch.tensor(
            [r.py_orig_prompt_len + step for r in requests], dtype=torch.int32, device="cuda"
        ),
        new_tokens_cuda=new_tokens,
        first_finish_reasons_cuda=first_reasons,
    )
    return result[0, slots.cuda()].cpu()


@pytest.mark.parametrize("beam_width", [2, 500])
@pytest.mark.parametrize("history_length", [1, 20])
@pytest.mark.parametrize("words", [[[0]], [[13], [17]]])
def test_single_token_beam_history_does_not_read_tokens(
    beam_width: int, history_length: int, words: list[list[int]]
) -> None:
    handler = _handler(beam_width, history_length)
    request = _request(3, words, beam_width=beam_width)
    with patch.object(LlmRequest, "get_tokens", side_effect=AssertionError("history is unused")):
        history = handler._get_past_tokens(request)
    assert history.shape == (history_length - 1, beam_width)
    assert history.dtype == torch.int32
    assert history.is_cuda
    assert torch.count_nonzero(history).item() == 0


@pytest.mark.parametrize("beam_width", [1, 4])
@pytest.mark.parametrize("words", [None, [], [[7, 8]], [[13], [7, 8]]])
@pytest.mark.parametrize("history_length", [1, 5])
def test_history_fallback_preserves_prompt_suffix(
    beam_width: int, words: list[list[int]] | None, history_length: int
) -> None:
    handler = _handler(beam_width, history_length)
    request = _request(0, words, beam_width=beam_width)
    history = handler._get_past_tokens(request)
    expected = torch.zeros((history_length - 1, beam_width), dtype=torch.int32)
    if history_length > 1:
        expected[-2:] = torch.tensor([2, 7], dtype=torch.int32)[:, None]
    torch.testing.assert_close(history.cpu(), expected, rtol=0, atol=0)


def test_single_beam_single_token_stop_keeps_history_path() -> None:
    handler = _handler(1, 5)
    history = handler._get_past_tokens(_request(0, [[13]], beam_width=1))
    assert history[:, 0].tolist() == [0, 0, 2, 7]


@pytest.mark.parametrize("history_length", [1, 20])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_stop_matching_and_finish_precedence(
    history_length: int, mixed: bool, streaming: bool
) -> None:
    handler = _handler(history_length=history_length)
    requests = [
        _request(3, [[13], [17], [0]], max_new_tokens=3, streaming=streaming),
        _request(0, [[7, 8], [9, 10, 11], [17]] if mixed else [[8], [11], [17]]),
        _request(4, None, max_new_tokens=2),
    ]
    _admit(handler, requests)
    histories = [r.get_tokens() for r in requests]
    first_reasons = torch.full(
        (5, 4), FinishReason.NOT_FINISHED.value, device="cuda", dtype=torch.int32
    )
    expected_first = torch.full((3, 4), FinishReason.NOT_FINISHED.value, dtype=torch.int32)
    steps = [
        [[13, 5, 17, 0], [8, 9, 17, 5], [13, 0, 17, 5]],
        [[8, 17, 5, 13], [5, 10, 4, 7], [13, 17, 8, 0]],
        [[13, 17, 0, 5], [5, 11, 8, 8], [17, 5, 0, 13]],
    ]
    for step, tokens in enumerate(steps):
        expected = torch.full((3, 4), FinishReason.NOT_FINISHED.value, dtype=torch.int32)
        for row, request in enumerate(requests):
            for beam, token in enumerate(tokens[row]):
                history = histories[row][beam]
                history.append(token)
                if any(history[-len(word) :] == word for word in request.py_stop_words_list or []):
                    expected[row, beam] = FinishReason.STOP_WORDS.value
                if len(history) >= request.py_orig_prompt_len + request.py_max_new_tokens:
                    expected[row, beam] = FinishReason.LENGTH.value
                if token == request.py_end_id:
                    expected[row, beam] = FinishReason.END_ID.value
        actual = _step(handler, requests, tokens, step, first_reasons)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_first = torch.where(
            expected_first == FinishReason.NOT_FINISHED.value, expected, expected_first
        )
        torch.testing.assert_close(first_reasons[[3, 0, 4]].cpu(), expected_first, rtol=0, atol=0)


def test_live_buffer_growth_reprocesses_single_and_multi_token_requests() -> None:
    handler = _handler(history_length=1)
    single = _request(3, [[13]])
    multi = _request(0, [[1, 2, 3, 4], [8]], prompt=[1, 2, 3])
    native_get_tokens = LlmRequest.get_tokens
    read_slots = []

    def get_tokens(request: LlmRequest) -> list[list[int]]:
        read_slots.append(request.py_seq_slot)
        return native_get_tokens(request)

    with patch.object(LlmRequest, "get_tokens", get_tokens):
        _admit(handler, [single])
        _admit(handler, [multi], [single, multi])
    assert 3 not in read_slots
    assert 0 in read_slots
    assert handler._max_stop_word_length == 4
    assert handler._max_num_stop_words == 2
    assert handler.store.past_tokens_cuda.shape == (4, 5, 4)
    result = _step(handler, [single, multi], [[13, 5, 13, 5], [4, 8, 5, 17]])
    assert result.tolist() == [
        [FinishReason.STOP_WORDS.value, 0, FinishReason.STOP_WORDS.value, 0],
        [
            FinishReason.STOP_WORDS.value,
            FinishReason.STOP_WORDS.value,
            0,
            FinishReason.END_ID.value,
        ],
    ]


@pytest.mark.parametrize("history_length", [1, 20])
def test_slot_reuse_drops_stale_rules_and_restores_multi_token_history(history_length: int) -> None:
    handler = _handler(history_length=history_length)
    cases = [
        ([[7, 8]], 8, FinishReason.STOP_WORDS),
        ([[13]], 8, FinishReason.NOT_FINISHED),
        ([[13]], 13, FinishReason.STOP_WORDS),
        (None, 13, FinishReason.NOT_FINISHED),
        ([[7, 8]], 8, FinishReason.STOP_WORDS),
    ]
    for words, token, reason in cases:
        request = _request(3, words)
        _admit(handler, [request])
        result = _step(handler, [request], [[token] * 4])
        assert result.tolist() == [[reason.value] * 4]
