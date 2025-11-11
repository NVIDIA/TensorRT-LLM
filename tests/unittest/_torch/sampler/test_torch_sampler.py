# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import Callable, Generator, Optional, cast

import pytest
import torch
from utils.util import assert_no_cuda_sync, force_ampere

from tensorrt_llm._torch.pyexecutor.llm_request import convert_wordlist
from tensorrt_llm._torch.pyexecutor.sampler import (
    GREEDY,
    LlmRequest,
    ScheduledRequests,
    TorchSampler,
    _request_strategy,
)
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.sampling_params import SamplingParams


@force_ampere
class TestStrategySelection:
    VOCAB_SIZE = 1000
    TOP_K_VALS = [None, 0, 1, 42, 1000]
    TOP_P_VALS = [None, 0, 0.42, 1]
    TEMPERATURE_VALS = [None, 0, 1.42]

    # For non-greedy sampling, the following choices have no effect.
    TOP_P_NEUTRAL_VALS = [None, 1]
    TOP_K_NEUTRAL_VALS = [None, 0, VOCAB_SIZE]
    TEMPERATURE_NEUTRAL_VALS = [None, 1]

    TEMPERATURE_NOT_GREEDY = [0.42] + [t for t in TEMPERATURE_NEUTRAL_VALS if t is not None]

    class MockLlmRequest:
        sampling_config: SamplingConfig

    def _check_params(self, params: SamplingParams):
        # cf. description of 'top_p' in doc-string of SamplingParams and
        # test_top_p_0_disallowed below.
        if params.top_p == 0:
            pytest.skip("top_p = 0 disallowed by tensorrt_llm::executor::SamplingConfig")

    # If this xpasses, update _check_params and doc-string of SamplingParams.
    @pytest.mark.xfail(reason="top_p = 0 disallowed by tensorrt_llm::executor::SamplingConfig")
    def test_top_p_0_disallowed(self):
        params = SamplingParams(top_p=0)
        params._get_sampling_config()

    def _build_mock_llm_request(self, params: SamplingParams) -> LlmRequest:
        request = self.MockLlmRequest()
        request.sampling_config = SamplingConfig(params._get_sampling_config())
        return cast(LlmRequest, request)

    def test_defaults(self):
        # NB: The code in _request_strategy relies on the default values below.
        default_params = SamplingParams()
        assert default_params.top_k is None
        assert default_params.top_p is None
        assert default_params.temperature is None

    def test_defaults_config(self):
        # NB: The code in _request_strategy relies on the default values below.
        default_config = SamplingParams()._get_sampling_config()
        assert default_config.top_k is None
        assert default_config.top_p is None
        assert default_config.temperature is None

    def test_defaults_request(self):
        # NB: The code in _request_strategy relies on the default values below.
        request = self._build_mock_llm_request(SamplingParams())
        default_config = request.sampling_config
        assert default_config.top_k is None
        assert default_config.top_p is None
        assert default_config.temperature is None

    def test_default_is_greedy(self):
        request = self._build_mock_llm_request(SamplingParams())
        assert _request_strategy(request, vocab_size=self.VOCAB_SIZE) is GREEDY

    @pytest.mark.parametrize(
        "top_p, top_k",
        [
            pytest.param(top_p, top_k)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (top_k, top_p) in product(TOP_K_VALS, TOP_P_VALS)
        ],
    )
    def test_temperature_0_is_greedy(self, top_p: Optional[float], top_k: Optional[int]):
        params = SamplingParams(temperature=0, top_p=top_p, top_k=top_k)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        assert _request_strategy(request, vocab_size=self.VOCAB_SIZE) is GREEDY

    @pytest.mark.parametrize(
        "temperature, top_k",
        [
            pytest.param(temperature, top_k)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_k) in product(TEMPERATURE_VALS, TOP_K_VALS)
        ],
    )
    def test_top_p_0_is_greedy(self, temperature: Optional[float], top_k: Optional[int]):
        params = SamplingParams(top_p=0, temperature=temperature, top_k=top_k)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        assert _request_strategy(request, vocab_size=self.VOCAB_SIZE) is GREEDY

    @pytest.mark.parametrize(
        "temperature, top_p",
        [
            pytest.param(temperature, top_p)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_p) in product(TEMPERATURE_VALS, TOP_P_VALS)
        ],
    )
    def test_top_k_1_is_greedy(self, temperature: Optional[float], top_p: Optional[float]):
        params = SamplingParams(top_p=top_p, temperature=temperature, top_k=1)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        assert _request_strategy(request, vocab_size=self.VOCAB_SIZE) is GREEDY

    @pytest.mark.parametrize(
        "temperature, trivial_top_p, trivial_top_k",
        [
            pytest.param(temperature, top_p, top_k)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_k, top_p) in product(
                TEMPERATURE_NOT_GREEDY, TOP_K_NEUTRAL_VALS, TOP_P_NEUTRAL_VALS
            )
        ],
    )
    def test_temperature_only(
        self, temperature: float, trivial_top_p: Optional[float], trivial_top_k: Optional[int]
    ):
        params = SamplingParams(temperature=temperature, top_p=trivial_top_p, top_k=trivial_top_k)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 2
        assert strat[0] == "temperature"
        assert strat[1] == pytest.approx(temperature)

    @pytest.mark.parametrize(
        "trivial_temperature, trivial_top_k",
        [
            pytest.param(temperature, top_k)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_k) in product(TEMPERATURE_NEUTRAL_VALS, TOP_K_NEUTRAL_VALS)
        ],
    )
    def test_top_p_only(self, trivial_temperature: Optional[float], trivial_top_k: Optional[int]):
        params = SamplingParams(top_p=0.42, temperature=trivial_temperature, top_k=trivial_top_k)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 3
        assert strat[0] == "top_p"
        assert strat[1] == pytest.approx(0.42)
        assert strat[2] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "trivial_top_k",
        [
            pytest.param(top_k)
            for top_k in TOP_K_NEUTRAL_VALS  # https://stackoverflow.com/a/75421799
        ],
    )
    def test_top_p_with_temperature(self, trivial_top_k: Optional[int]):
        params = SamplingParams(top_p=0.42, temperature=0.9, top_k=trivial_top_k)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 3
        assert strat[0] == "top_p"
        assert strat[1] == pytest.approx(0.42)
        assert strat[2] == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "trivial_temperature, trivial_top_p",
        [
            pytest.param(temperature, top_p)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_p) in product(TEMPERATURE_NEUTRAL_VALS, TOP_P_NEUTRAL_VALS)
        ],
    )
    def test_top_k_only(self, trivial_temperature: Optional[float], trivial_top_p: Optional[float]):
        params = SamplingParams(top_k=42, temperature=trivial_temperature, top_p=trivial_top_p)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 3
        assert strat[0] == "top_k"
        assert strat[1] == 42
        assert strat[2] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "trivial_top_p",
        [
            pytest.param(top_p)
            for top_p in TOP_P_NEUTRAL_VALS  # https://stackoverflow.com/a/75421799
        ],
    )
    def test_top_k_with_temperature(self, trivial_top_p: Optional[float]):
        params = SamplingParams(top_k=42, temperature=0.9, top_p=trivial_top_p)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 3
        assert strat[0] == "top_k"
        assert strat[1] == 42
        assert strat[2] == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "trivial_temperature",
        [
            pytest.param(temperature)
            for temperature in TEMPERATURE_NEUTRAL_VALS  # https://stackoverflow.com/a/75421799
        ],
    )
    def test_top_k_top_p(self, trivial_temperature: Optional[float]):
        params = SamplingParams(top_k=42, top_p=0.7, temperature=trivial_temperature)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 4
        assert strat[0] == "top_k_top_p"
        assert strat[1] == 42
        assert strat[2] == pytest.approx(0.7)
        assert strat[3] == pytest.approx(1.0)

    def test_top_k_top_p_with_temperature(self):
        params = SamplingParams(top_k=42, top_p=0.7, temperature=0.9)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 4
        assert strat[0] == "top_k_top_p"
        assert strat[1] == 42
        assert strat[2] == pytest.approx(0.7)
        assert strat[3] == pytest.approx(0.9)

    def test_param_validation(self):
        with pytest.raises(ValueError, match="require temperature >= 0, got temperature=-1"):
            SamplingParams(temperature=-1)

        with pytest.raises(ValueError, match="require 0 <= top_p <= 1, got top_p=-1"):
            SamplingParams(top_p=-1)

        with pytest.raises(ValueError, match="require 0 <= top_p <= 1, got top_p=2"):
            SamplingParams(top_p=2)

        with pytest.raises(ValueError, match="require top_k >= 0, got top_k=-1"):
            SamplingParams(top_k=-1)

    @pytest.mark.parametrize(
        "top_k, top_p",
        [
            pytest.param(top_k, top_p)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (top_k, top_p) in product(TOP_K_NEUTRAL_VALS, TOP_P_NEUTRAL_VALS)
            if (top_k, top_p) != (None, None)
        ],
    )
    def test_trivial_top_k_top_p_not_greedy(self, top_k: Optional[int], top_p: Optional[float]):
        params = SamplingParams(top_k=top_k, top_p=top_p)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 2
        assert strat[0] == "temperature"
        assert strat[1] == pytest.approx(1.0)

    @pytest.fixture
    def torch_sampler(self) -> TorchSampler:
        return TorchSampler(
            TorchSampler.Args(
                max_seq_len=123,
                max_draft_len=3,
                max_num_sequences=12,
                max_beam_width=1,
                max_total_draft_tokens=3,
            )
        )

    @pytest.mark.parametrize(
        "temperature, top_p, top_k",
        [
            pytest.param(temperature, top_p, top_k)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_p, top_k) in product(TEMPERATURE_VALS, TOP_P_VALS, TOP_K_VALS)
        ],
    )
    def test_should_provide_draft_probs_consistency(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        torch_sampler: TorchSampler,
    ):
        params = SamplingParams(top_k=top_k, top_p=top_p, temperature=temperature)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        is_greedy = strat is GREEDY

        assert torch_sampler.should_provide_draft_probs(request) == (not is_greedy)


@force_ampere
@pytest.mark.parametrize(
    "draft_len, with_ctx, with_gen",
    [
        pytest.param(draft_len, with_ctx, with_gen)
        for (draft_len, with_ctx, with_gen) in product(
            [0, 3],
            [False, True],
            [False, True],
        )
        if with_ctx or with_gen
    ],
)
def test_select_generated_logits(draft_len: int, with_ctx: bool, with_gen: bool):
    # Currently only checks that this works and does not sync

    device = torch.device("cuda")

    @contextmanager
    def _test_runner() -> Generator[Callable[[], None], None, None]:
        class ContextRequestMock:
            def __init__(self, return_context_logits: bool):
                self._return_context_logits = return_context_logits

            @property
            def py_return_context_logits(self) -> bool:
                return self._return_context_logits

        class GenRequestMock:
            pass

        class ScheduledRequestsMock:
            @property
            def context_requests(self) -> list[LlmRequest]:
                return (
                    [
                        # NB: One request with py_return_context_logits is enough
                        #     to trigger tested code.
                        cast(LlmRequest, ContextRequestMock(True)),
                        cast(LlmRequest, ContextRequestMock(False)),
                        cast(LlmRequest, ContextRequestMock(True)),
                    ]
                    if with_ctx
                    else []
                )

            @property
            def generation_requests(self) -> list[LlmRequest]:
                # NB: Currently this list is not inspected, UUT only checks that this
                #     is not empty.
                return (
                    [
                        cast(LlmRequest, GenRequestMock()),
                        cast(LlmRequest, GenRequestMock()),
                    ]
                    if with_gen
                    else []
                )

        vocab_size = 12

        num_context_logits_prefix_sum = [
            0,
            *(
                [
                    100 + 1,  # context req. 1 (assume context len. 100)
                    (100 + 1) + (0 + 1),  # context req. 2 (not returning context)
                    (100 + 1) + (0 + 1) + (50 + 1),  # context req. 3 (assume context len. 50)
                ]
                if with_ctx
                else []
            ),
        ]
        draft_len_req1 = draft_len
        draft_len_req2 = draft_len + 1  # test with different draft lens
        req_num_generation_steps = [
            *(
                [
                    1,  # context req. 1
                    1,  # context req. 2
                    1,  # context req. 3
                ]
                if with_ctx
                else []
            ),
            *(
                [
                    draft_len_req1 + 1,  # gen. req. 1
                    draft_len_req2 + 1,  # gen. req. 2
                ]
                if with_gen
                else []
            ),
        ]
        req_num_generation_steps_tensor = torch.tensor(req_num_generation_steps, dtype=torch.int32)
        num_logits_to_keep = cast(int, req_num_generation_steps_tensor.sum().item())
        generation_requests_total_steps = (draft_len_req1 + 1) + (
            draft_len_req2 + 1
        )  # cf. req_num_generation_steps

        num_total_steps = num_context_logits_prefix_sum[-1] + generation_requests_total_steps
        all_logits = torch.empty((num_total_steps, vocab_size))

        for i in range(all_logits.size(0)):
            all_logits[i, :] = torch.arange(i, i + vocab_size)

        all_logits_cuda = all_logits.to(device=device)

        expected_logit_indices = []
        if with_ctx:
            expected_logit_indices += [
                100,  # gen logits from context req. 1
                101,  # gen logits from context req. 2
                152,  # gen logits from context req. 3
            ]
        if with_gen:
            gen_logit_offset = num_context_logits_prefix_sum[-1]
            expected_logit_indices += [
                *range(
                    gen_logit_offset, gen_logit_offset + draft_len_req1 + 1
                ),  # gen logits from gen. req. 1
                *range(
                    gen_logit_offset + draft_len_req1 + 1,
                    gen_logit_offset + generation_requests_total_steps,
                ),  # gen logits from gen. req. 2
            ]

        @dataclass
        class UutResult:
            selected_logits: torch.Tensor

        @dataclass
        class UutResultWrapper:
            result: Optional[UutResult] = None

        res = UutResultWrapper()

        def _uut(res=res):
            selected_logits = TorchSampler._select_generated_logits(
                cast(ScheduledRequests, ScheduledRequestsMock()),
                all_logits_cuda,
                req_num_generation_steps=req_num_generation_steps_tensor,
                num_context_logits_prefix_sum=num_context_logits_prefix_sum,
                generation_requests_total_steps=generation_requests_total_steps,
                num_logits_to_keep=num_logits_to_keep,
            )
            res.result = UutResult(selected_logits=selected_logits)

        yield _uut

        # Check logits
        assert res.result is not None
        selected_logits = res.result.selected_logits
        torch.testing.assert_close(selected_logits.to("cpu"), all_logits[expected_logit_indices])

    with _test_runner() as uut:
        # Pre-allocates a large chunk of memory, because PyTorch caching memory allocator
        # can sync otherwise.
        buf = torch.ones((2**30,), device=device)
        del buf
        # Warmup to avoid syncs due to lazy loading of kernels
        uut()

    with torch.cuda.Stream():
        with _test_runner() as uut:
            with assert_no_cuda_sync():
                uut()


MAX_NUM_SEQUENCES = 128
NOT_FINISHED = FinishReason.NOT_FINISHED
STOP_WORDS = FinishReason.STOP_WORDS
END_ID = FinishReason.END_ID
LENGTH = FinishReason.LENGTH
BEAM = 0


class RequestCase:
    MAX_NEW_TOKENS = 10
    seq_slots = torch.randperm(MAX_NUM_SEQUENCES).tolist()

    def __init__(
        self,
        *,
        prompt: list[int],
        new_tokens: list[int],
        finish_reasons: list[FinishReason],
        max_new_tokens: int = MAX_NEW_TOKENS,
        end_id: int = None,
        stop_words_list: list[list[int]] = None,
    ):
        seq_slot = self.seq_slots.pop()  # random seq slot in MAX_NUM_SEQUENCES
        self.prompt = prompt
        self.request = LlmRequest(
            request_id=seq_slot,
            seq_slot=seq_slot,
            input_tokens=prompt,
            max_new_tokens=max_new_tokens,
            stop_words_list=convert_wordlist(stop_words_list)
            if stop_words_list is not None
            else None,
            end_id=end_id,
            sampling_config=SamplingConfig(),
            is_streaming=False,
            draft_tokens=new_tokens[:-1],
        )
        assert len(new_tokens) == len(finish_reasons)
        self.new_tokens = new_tokens
        self.finish_reasons = finish_reasons

    def __repr__(self):
        return f"RequestCase({self.prompt=}, {self.new_tokens=}, {self.finish_reasons=}, \
        {self.request.max_new_tokens=}, {self.request.end_id=}, {self.request.stop_words_list=})"

    @staticmethod
    def setup(requests: list["RequestCase"]):
        max_tokens = set(len(req.new_tokens) for req in requests)
        assert len(max_tokens) == 1
        max_draft_len = max_tokens.pop() - 1
        sampler_args = TorchSampler.Args(
            max_seq_len=20,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_draft_len,
            # Fill with many more max requests than below,
            # so we can test that write_finish_reasons uses seq_slots correctly
            max_num_sequences=MAX_NUM_SEQUENCES,
            max_beam_width=1,
        )
        sampler = TorchSampler(args=sampler_args)

        # fill with garbage value so we can observe that finish reasons are filled
        # with NOT_FINISHED before we write to them.
        sampler.store.finish_reasons.fill_(205)
        seq_slots = torch.tensor(
            [req.request.py_seq_slot for req in requests], device="cuda", dtype=torch.int64
        )
        new_tokens = torch.tensor(
            [req.new_tokens for req in requests], dtype=torch.int32, device="cuda"
        ).T
        sampler.store.new_tokens[:, seq_slots, BEAM] = new_tokens

        def run():
            sampler._write_finish_reasons(
                [req.request for req in requests],
                finish_reasons=sampler.store.finish_reasons,
                new_tokens=sampler.store.new_tokens,
                seq_slots=seq_slots,
            )

            reasons = sampler.store.finish_reasons[:, seq_slots, BEAM].T.tolist()

            for actual, request in zip(reasons, requests, strict=True):
                expected = request.finish_reasons
                msg = f"actual={[FinishReason(reason) for reason in actual]} != expected={expected}\nFor {request}"
                assert actual == [reason.value for reason in expected], msg

        return run, sampler


def test_write_finish_reasons():
    """We don't really care about the finish reason past the first infraction, because we're not going to use it,
    although in some instance it is written anyway."""
    run, _ = RequestCase.setup(
        [
            RequestCase(
                prompt=[13, 14],
                new_tokens=[60, 61, 62],
                # We pre-fill the finish reasons with NOT_FINISHED.
                finish_reasons=[NOT_FINISHED, NOT_FINISHED, NOT_FINISHED],
            ),
            RequestCase(
                prompt=[7, 8, 6],
                stop_words_list=[[12, 13]],
                new_tokens=[12, 13, 60],
                finish_reasons=[NOT_FINISHED, STOP_WORDS, NOT_FINISHED],
            ),
            RequestCase(
                prompt=[1, 2, 3, 4],
                end_id=99,
                new_tokens=[55, 99, 58],
                finish_reasons=[NOT_FINISHED, END_ID, NOT_FINISHED],
            ),
            RequestCase(
                prompt=[4, 5, 6],
                max_new_tokens=2,
                new_tokens=[56, 57, 59],
                # The LENGTH check happens to not have an early exit
                finish_reasons=[NOT_FINISHED, LENGTH, LENGTH],
            ),
            RequestCase(
                prompt=[1, 12],
                stop_words_list=[[12, 13], [14, 15]],
                new_tokens=[13, 14, 15],
                # We have an early exit specifically for stop words
                finish_reasons=[STOP_WORDS, NOT_FINISHED, NOT_FINISHED],
            ),
            RequestCase(
                prompt=[1, 12],
                stop_words_list=[[12, 13, 14, 15], [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
                new_tokens=[13, 14, 15],
                # Stop words of different lengths are handled correctly with respect to padding of stop words and tokens
                finish_reasons=[NOT_FINISHED, NOT_FINISHED, STOP_WORDS],
            ),
            RequestCase(
                prompt=[1],
                max_new_tokens=2,
                end_id=99,
                stop_words_list=[[1, 12]],
                new_tokens=[12, 99, 63],
                # Different infractions are written to different places as
                # we don't have an early exit between infractions
                finish_reasons=[STOP_WORDS, END_ID, LENGTH],
            ),
            RequestCase(
                prompt=[1, 12, 56, 67, 68, 234, 678],
                stop_words_list=[[12, 56, 67, 68, 234, 678, 129, 182]],
                new_tokens=[129, 182, 600],
                # Notice the offending stop sequence is concatenated, as we lookback
                finish_reasons=[NOT_FINISHED, STOP_WORDS, NOT_FINISHED],
            ),
            RequestCase(
                prompt=[1, 12],
                end_id=99,
                max_new_tokens=1,
                stop_words_list=[[1, 12, 99]],
                new_tokens=[99, 100, 101],
                # The latest infraction check overrides the earlier infraction checks,
                # hence the first finish_reason is END_ID
                finish_reasons=[END_ID, LENGTH, LENGTH],
            ),
        ]
    )
    run()


def test_are_stop_words_isnt_called_when_no_stop_words():
    """We don't want to call are_stop_words when there are no stop words because it's expensive"""

    def stop_words_that_raises(*args, **kwargs):
        raise AssertionError

    run_with_stop_words, sampler = RequestCase.setup(
        [
            RequestCase(
                prompt=[1], stop_words_list=[[1]], new_tokens=[4], finish_reasons=[NOT_FINISHED]
            )
        ]
    )
    sampler._are_stop_words = stop_words_that_raises
    with pytest.raises(AssertionError):
        run_with_stop_words()

    run_without_stop_words, sampler = RequestCase.setup(
        [RequestCase(prompt=[1], new_tokens=[4], finish_reasons=[NOT_FINISHED])]
    )
    sampler._are_stop_words = stop_words_that_raises
    _ = run_without_stop_words()
