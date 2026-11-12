# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TorchSampler consuming greedy tokens a forward sampled in place of its LM head."""

from typing import Any, Callable

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.handle_logits import HandleLogits
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.sampler import GREEDY_TOKENS_KEY, greedy_tokens_usable
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

VOCAB_SIZE = 16


def _request(
    request_id: int = 0,
    seq_slot: int = 0,
    sampling_params: SamplingParams | None = None,
    **kwargs: Any,
) -> LlmRequest:
    params = sampling_params if sampling_params is not None else SamplingParams()
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=[1, 2, 3],
        sampling_config=SamplingConfig(params._get_sampling_config()),
        seq_slot=seq_slot,
        is_streaming=False,
        **kwargs,
    )


def _set(name: str, value: Any) -> Callable[[LlmRequest], None]:
    def apply(request: LlmRequest) -> None:
        setattr(request, name, value)

    return apply


class TestGreedyTokensUsable:
    """Whether a generation request is served by the forward's greedy token alone."""

    pytestmark = pytest.mark.cpu_only

    def test_plain_greedy_request_is_usable(self):
        assert greedy_tokens_usable(_request())

    @pytest.mark.parametrize(
        ("kwargs", "mutate"),
        [
            pytest.param({}, _set("py_draft_tokens", [5, 6]), id="draft_tokens"),
            pytest.param(
                {}, _set("py_embedding_bias", torch.zeros(VOCAB_SIZE)), id="embedding_bias"
            ),
            pytest.param({}, _set("py_bad_words", [[4]]), id="bad_words"),
            pytest.param({}, _set("py_no_repeat_ngram_size", 2), id="no_repeat_ngram"),
            pytest.param(
                {}, _set("py_logits_post_processors", [print]), id="logits_post_processor"
            ),
            pytest.param({"return_log_probs": True}, None, id="log_probs"),
            pytest.param({"return_generation_logits": True}, None, id="generation_logits"),
            pytest.param({"stop_words_list": [[13, 14]]}, None, id="multi_token_stop_word"),
        ],
    )
    def test_a_logits_reader_makes_the_request_unusable(self, kwargs, mutate):
        request = _request(**kwargs)
        if mutate is not None:
            mutate(request)

        assert not greedy_tokens_usable(request)

    @pytest.mark.parametrize(
        "sampling_params",
        [
            pytest.param(SamplingParams(temperature=0.8, top_k=50), id="top_k"),
            pytest.param(SamplingParams(temperature=0.7, top_p=0.9), id="top_p"),
            pytest.param(SamplingParams(repetition_penalty=1.3), id="repetition_penalty"),
            pytest.param(SamplingParams(presence_penalty=0.5), id="presence_penalty"),
            pytest.param(SamplingParams(frequency_penalty=0.5), id="frequency_penalty"),
            pytest.param(SamplingParams(min_tokens=2), id="min_length"),
        ],
    )
    def test_non_greedy_or_penalized_sampling_is_unusable(self, sampling_params):
        assert not greedy_tokens_usable(_request(sampling_params=sampling_params))

    def test_single_token_stop_words_stay_usable(self):
        """Single-token stop words are checked on the host, like the end id."""
        assert greedy_tokens_usable(_request(stop_words_list=[[13], [17]]))

    def test_guided_decoding_makes_the_request_unusable(self):
        """Guided decoding masks the logits after the forward, so a greedy token cannot serve it."""
        request = _request(guided_decoding_params=GuidedDecodingParams(regex="[0-9]+"))
        assert request.py_guided_decoding_params is not None

        assert not greedy_tokens_usable(request)

    @pytest.mark.parametrize("dummy_flag", ["is_attention_dp_dummy", "is_cuda_graph_dummy"])
    def test_a_dummy_request_is_never_usable(self, dummy_flag):
        """The stable greedy path takes no batch with a dummy, so such a batch needs logits."""
        request = _request()
        setattr(request, dummy_flag, True)

        assert not greedy_tokens_usable(request)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_handle_logits_takes_greedy_only_outputs() -> None:
    """The executor hands HandleLogits the outputs' logits before sampling; a greedy-only batch (no request asks for
    logits) has none, and HandleLogits must return without reading them."""
    HandleLogits()([], [_request()], None, 1, [0], True)


class TestStableGreedyFromGreedyTokens:
    """The stable greedy path takes the forward's tokens instead of an argmax over logits."""

    @staticmethod
    def _sampler(max_num_sequences: int = 3) -> TorchSampler:
        return TorchSampler(
            TorchSampler.Args(
                max_seq_len=16,
                max_draft_len=0,
                max_num_sequences=max_num_sequences,
                max_beam_width=1,
                max_total_draft_tokens=0,
                disable_overlap_scheduler=True,
            )
        )

    @staticmethod
    def _batch(*requests: LlmRequest) -> ScheduledRequests:
        scheduled = ScheduledRequests()
        scheduled.generation_requests = list(requests)
        return scheduled

    @staticmethod
    def _one_hot_logits(tokens: list[int]) -> torch.Tensor:
        logits = torch.zeros((len(tokens), VOCAB_SIZE), device="cuda")
        logits[torch.arange(len(tokens)), torch.tensor(tokens)] = 10.0
        return logits

    def test_greedy_tokens_match_the_logits_argmax(self):
        tokens = [7, 3]
        slots = [2, 0]
        results = {}
        for name, outputs in (
            ("logits", {"logits": self._one_hot_logits(tokens)}),
            (
                "greedy_tokens",
                {
                    # No logits key, as the engine returns greedy-only outputs.
                    # A padded graph replays more rows than the batch has requests.
                    GREEDY_TOKENS_KEY: torch.tensor(tokens + [9], dtype=torch.int64, device="cuda"),
                },
            ),
        ):
            sampler = self._sampler()
            sampler.store.new_tokens.zero_()
            batch = self._batch(*(_request(i, slot) for i, slot in enumerate(slots)))

            (
                requests,
                seq_slots_host,
                _,
                _,
                _,
                new_tokens_host,
                single_step_greedy,
            ) = sampler._process_requests(batch, outputs, sampler.store.new_tokens, [0])
            torch.cuda.synchronize()

            assert single_step_greedy
            assert requests == batch.generation_requests
            assert seq_slots_host.tolist() == slots
            results[name] = (
                new_tokens_host.reshape(-1).tolist(),
                sampler.store.new_tokens[0, :, 0].tolist(),
            )

        assert results["greedy_tokens"] == results["logits"]
        assert results["greedy_tokens"][0] == tokens
        assert [results["greedy_tokens"][1][slot] for slot in slots] == tokens

    def test_missing_logits_for_a_batch_that_needs_them_raises(self):
        sampler = self._sampler()
        request = _request()
        request.py_no_repeat_ngram_size = 2
        outputs = {GREEDY_TOKENS_KEY: torch.tensor([4], dtype=torch.int64, device="cuda")}

        with pytest.raises(RuntimeError, match="greedy tokens without logits"):
            sampler._process_requests(self._batch(request), outputs, sampler.store.new_tokens, [0])

    def test_a_batch_padded_with_an_attention_dp_dummy_asks_for_logits(self):
        """A dummy makes the batch unusable, so the engine computes logits rather than keep greedy tokens."""
        live = _request(0, 0)
        dummy = _request(1, 1)
        dummy.is_attention_dp_dummy = True
        batch = self._batch(live, dummy)

        assert not all(greedy_tokens_usable(request) for request in batch.generation_requests)
