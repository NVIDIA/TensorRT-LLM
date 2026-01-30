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

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import product
from typing import (
    Callable,
    ContextManager,
    Final,
    Generator,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
)

import flashinfer.sampling
import numpy as np
import pytest
import torch
from scipy.stats import power_divergence
from utils.util import assert_no_cuda_sync, force_ampere

from tensorrt_llm._torch.pyexecutor import sampling_utils_flashinfer
from tensorrt_llm._torch.pyexecutor.llm_request import convert_wordlist
from tensorrt_llm._torch.pyexecutor.sampler import (
    GREEDY,
    LlmRequest,
    ScheduledRequests,
    SimpleGroupedStrategySampler,
    StrategyMetadata,
    TorchSampler,
    _BatchedSamplingResult,
    _request_get_sampling_params,
    _request_strategy,
    get_draft_token_length,
)
from tensorrt_llm._torch.pyexecutor.sampling_utils import (
    BeamSearch,
    Greedy,
    Strategy,
    TemperatureOnly,
    TopK,
    TopKTopP,
    TopP,
    UtilsSamplingParams,
)
from tensorrt_llm._torch.pyexecutor.sampling_utils_flashinfer import (
    FlashInferGroupedStrategySampler,
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
        is_context_init_state: bool  # Torch sampler accesses this, but it does not affect this test

        def get_beam_width_by_iter(
            self, for_next_iteration: bool = False
        ) -> int:  # Torch sampler accesses this, but it does not affect this test
            return self.sampling_config.beam_width

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
        request.is_context_init_state = False  # Not used in this test
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
                disable_overlap_scheduler=False,
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


class UutProvider(Protocol):
    def __call__(self, is_warmup: bool) -> ContextManager[Callable[[], None]]: ...


def _run_test_with_warmup(
    uut_provider: UutProvider,
    warmup_sizes_bytes: tuple[int] = (4 * 2**30,),
    max_sync_s: Optional[float] = None,
):
    """Run UUT including setup and warmup.

    This is mainly used to check that the UUT does not CUDA device sync. Thus,
    given that PyTorch's caching memory allocator can device sync when it runs
    out of cached GPU memory segments, the warmup allocates some GPU memory.

    The warmup also runs the test once. This avoids issues with things like lazy loading
    of device code. The UUT provider can use the 'is_warmup' argument to adapt its
    behavior to the warmup and final test runs.

    If max_sync_s is provided, this helper checks that the UUT does not device sync,
    assuming that the sync (CPU) part of the code takes no longer than max_sync_s
    seconds to complete.

    It is the user's responsibility to ensure that the amount of submitted work
    does not exceed the CUDA driver/device queue capacity, which would make
    the execution appear synchronous.
    """
    with torch.cuda.Stream():
        with uut_provider(is_warmup=True) as uut:
            bufs = []
            for warmup_size in warmup_sizes_bytes:
                bufs.append(
                    torch.ones(warmup_size, device=torch.cuda.current_device(), dtype=torch.int8)
                )
            del bufs
            uut()

        with uut_provider(is_warmup=False) as uut:
            with (
                assert_no_cuda_sync(sync_timeout_s=max_sync_s)
                if max_sync_s is not None
                else nullcontext()
            ):
                uut()


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
    def _test_runner(is_warmup: bool) -> Generator[Callable[[], None], None, None]:
        draft_len_req1 = draft_len
        draft_len_req2 = draft_len + 1  # test with different draft lens

        class ContextRequestMock:
            def __init__(self, is_last_context_chunk: bool, return_context_logits: bool):
                self.is_context_init_state = True
                self.is_last_context_chunk = is_last_context_chunk
                self.py_draft_tokens = torch.tensor([], dtype=torch.int32, device=device)
                self.sampling_config = SamplingConfig(beam_width=1)
                self._return_context_logits = return_context_logits

            @property
            def py_return_context_logits(self) -> bool:
                return self._return_context_logits

            def get_beam_width_by_iter(
                self, for_next_iteration: bool = False
            ) -> int:  # Torch sampler accesses this, but it does not affect this test
                return self.sampling_config.beam_width

        class GenRequestMock:
            def __init__(self, draft_len: int):
                self.is_context_init_state = False
                self.py_draft_tokens = torch.empty(draft_len, dtype=torch.int32, device=device)
                self.sampling_config = SamplingConfig(beam_width=1)

            def get_beam_width_by_iter(
                self, for_next_iteration: bool = False
            ) -> int:  # Torch sampler accesses this, but it does not affect this test
                return self.sampling_config.beam_width

        class ScheduledRequestsMock:
            @property
            def context_requests(self) -> list[LlmRequest]:
                return (
                    [
                        # NB: One request with py_return_context_logits is enough
                        #     to trigger tested code.
                        cast(
                            LlmRequest,
                            ContextRequestMock(
                                is_last_context_chunk=True, return_context_logits=True
                            ),
                        ),
                        cast(
                            LlmRequest,
                            ContextRequestMock(
                                is_last_context_chunk=True, return_context_logits=False
                            ),
                        ),
                        cast(
                            LlmRequest,
                            ContextRequestMock(
                                is_last_context_chunk=True, return_context_logits=True
                            ),
                        ),
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
                        cast(LlmRequest, GenRequestMock(draft_len=draft_len_req1)),
                        cast(LlmRequest, GenRequestMock(draft_len=draft_len_req2)),
                    ]
                    if with_gen
                    else []
                )

            def all_requests(self) -> list[LlmRequest]:
                return self.context_requests + self.generation_requests

        expected_num_requests = with_ctx * 3 + with_gen * 2
        expected_req_num_beams = torch.tensor([1] * expected_num_requests, dtype=torch.int32)

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
        expected_req_num_generation_steps = [
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
        expected_req_num_generation_steps_tensor = torch.tensor(
            expected_req_num_generation_steps, dtype=torch.int32
        )

        expected_req_offsets = torch.cumsum(expected_req_num_generation_steps_tensor, dim=0).roll(1)
        expected_req_offsets[0] = 0

        # num_logits_to_keep = cast(int, req_num_generation_steps_tensor.sum().item())
        generation_requests_total_steps = (draft_len_req1 + 1) + (
            draft_len_req2 + 1
        )  # cf. req_num_generation_steps

        vocab_size = 12

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

        expected_logits = all_logits[expected_logit_indices]

        @dataclass
        class UutResult:
            req_num_generated_tokens: torch.Tensor
            req_num_beams: torch.Tensor
            req_num_steps: torch.Tensor
            req_offsets: torch.Tensor
            selected_logits: torch.Tensor

        @dataclass
        class UutResultWrapper:
            result: Optional[UutResult] = None

        res = UutResultWrapper()

        def _uut(res=res):
            (
                sampling_requests_metadata,
                selected_logits,
            ) = TorchSampler._select_generated_logits(
                cast(ScheduledRequests, ScheduledRequestsMock()),
                all_logits_cuda,
                num_context_logits_prefix_sum=num_context_logits_prefix_sum,
            )
            res.result = UutResult(
                req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
                req_num_beams=sampling_requests_metadata.req_num_beams,
                req_num_steps=sampling_requests_metadata.req_num_steps,
                req_offsets=sampling_requests_metadata.req_offsets,
                selected_logits=selected_logits,
            )

        yield _uut

        # Check results
        assert res.result is not None

        torch.testing.assert_close(
            res.result.req_num_generated_tokens.to("cpu"), expected_req_num_generation_steps_tensor
        )
        torch.testing.assert_close(res.result.req_num_beams.to("cpu"), expected_req_num_beams)
        torch.testing.assert_close(
            res.result.req_num_steps.to("cpu"), expected_req_num_generation_steps_tensor
        )
        torch.testing.assert_close(res.result.req_offsets.to("cpu"), expected_req_offsets)
        torch.testing.assert_close(res.result.selected_logits.to("cpu"), expected_logits)

    _run_test_with_warmup(_test_runner, max_sync_s=0.3)


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
        end_id: Optional[int] = None,
        stop_words_list: Optional[list[list[int]]] = None,
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
            disable_overlap_scheduler=False,
        )
        sampler = TorchSampler(args=sampler_args)

        # fill with garbage value so we can observe that finish reasons are filled
        # with NOT_FINISHED before we write to them.
        sampler.store.finish_reasons.fill_(205)
        seq_slots = torch.tensor(
            [req.request.py_seq_slot for req in requests], device="cuda", dtype=torch.int64
        )
        seq_lens = torch.tensor(
            [req.request.max_beam_num_tokens for req in requests], dtype=torch.int32, device="cuda"
        )
        new_tokens = torch.tensor(
            [req.new_tokens for req in requests], dtype=torch.int32, device="cuda"
        ).T
        sampler.store.new_tokens[:, seq_slots, BEAM] = new_tokens
        max_seq_lens = torch.tensor(
            [
                min(
                    sampler.max_seq_len, req.request.orig_prompt_len + req.request.py_max_new_tokens
                )
                for req in requests
            ],
            dtype=torch.int32,
            device="cuda",
        )
        end_ids = torch.tensor(
            [
                req.request.py_end_id if req.request.py_end_id is not None else -1
                for req in requests
            ],
            dtype=torch.int32,
            device="cuda",
        )
        sampler.store.max_lengths_tensor[seq_slots] = max_seq_lens
        sampler.store.end_ids[seq_slots] = end_ids

        def run():
            sampler._write_finish_reasons(
                [req.request for req in requests],
                finish_reasons=sampler.store.finish_reasons,
                new_tokens=sampler.store.new_tokens,
                seq_lens=seq_lens,
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
                # We don't use early exit to avoid stream synchronization for stop words
                finish_reasons=[STOP_WORDS, NOT_FINISHED, STOP_WORDS],
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


class TestBatchedSampling:
    """Validate batched/mixed sampling.

    This test class focuses on the functionality implemented in `_sample_batched_by_strategy`
    and `_unbatch_sampling_results`, as invoked from `sample_async` (via `_process_requests`).
    """

    VOCAB_SIZE = 123

    @staticmethod
    def _build_test_cases(
        *,
        vocab_size: int,
        allow_greedy: bool = True,
        include_all: bool = True,
        include_uniform: bool = False,  # include_all takes precedence
        include_mixed: bool = False,  # include_all takes precedence
    ) -> list[tuple[list[SamplingParams], str]]:
        """Return test cases for testing batched sampling.

        Each test case consists of a list of sampling parameters and a human-readable
        test case name.
        """

        BASE_CASES = {  # one entry per sampling strategy
            Greedy: SamplingParams(),
            TemperatureOnly: SamplingParams(temperature=0.7),
            TopP: SamplingParams(top_p=0.42, temperature=0.2),
            TopK: SamplingParams(top_k=27, temperature=0.5),
            TopKTopP: SamplingParams(top_k=27, top_p=0.6, temperature=0.5),
        }

        # Check that all relevant strategies are covered
        # Beam search is tested in test_beam_search.py instead of here.
        # It's added here to pass the assert statement, without testing it.
        assert Union[*BASE_CASES.keys(), BeamSearch] == Strategy

        test_cases = []

        def _get_strategy_name(strategy_type: Type[Strategy]) -> str:
            return strategy_type.__args__[0].__args__[0]  # type: ignore

        if include_all or include_uniform:
            # Base cases (single-request batches)
            for strategy_type, params in BASE_CASES.items():
                if strategy_type == Greedy and not allow_greedy:
                    continue
                strategy_name = _get_strategy_name(strategy_type)
                test_cases.append(
                    (
                        [params],
                        f"single_{strategy_name}",
                    )
                )

        rng = np.random.default_rng(seed=42)

        if include_all or include_uniform:
            # Homogeneous batches (all requests use the same sampling params)
            max_batch_size: Final = 24
            for strategy_type, params in BASE_CASES.items():
                batch_size = rng.integers(low=1, high=max_batch_size)
                if strategy_type == Greedy and not allow_greedy:
                    continue
                strategy_name = _get_strategy_name(strategy_type)
                test_cases.append(
                    (
                        [params] * batch_size,
                        f"uniform_batch_{strategy_name}",
                    )
                )

        if include_all or include_mixed:

            class OneContinguous:
                pass

            class Shuffle:
                pass

            class VaryParams:
                pass

            # Batches containing requests with different sampling params
            max_sub_batch_size: Final = 6
            type_to_constrain = TopK
            mixed_params_list = None
            for constraint_value in [
                None,  # all sub-batches have at least two requests
                0,  # one sub-batch omitted
                1,  # one size-1 sub-batch
                OneContinguous(),  # one contiguous sub-batch, rest shuffled
                Shuffle(),  # random ordering
                VaryParams(),  # random ordering + randomized request parameter values
            ]:
                mixed_params_list = []
                constrained_indices = None
                for strategy_type, params in BASE_CASES.items():
                    sub_batch_size = rng.integers(low=2, high=max_sub_batch_size)
                    if strategy_type == Greedy and not allow_greedy:
                        continue
                    if strategy_type == type_to_constrain and constraint_value is not None:
                        if isinstance(constraint_value, int):
                            sub_batch_size = constraint_value
                        else:
                            constrained_indices = (
                                len(mixed_params_list),
                                len(mixed_params_list) + sub_batch_size,
                            )
                    strategy_name = _get_strategy_name(strategy_type)
                    mixed_params_list += [params] * sub_batch_size
                label = "mixed_batch"
                if isinstance(constraint_value, OneContinguous):
                    assert constrained_indices is not None
                    no_shuffle_start_idx, no_shuffle_end_idx = constrained_indices
                    head_shuffled = mixed_params_list[:no_shuffle_start_idx]
                    rng.shuffle(head_shuffled)  # inplace
                    tail_shuffled = mixed_params_list[no_shuffle_end_idx:]
                    rng.shuffle(tail_shuffled)  # inplace
                    mixed_params_list = (
                        head_shuffled
                        + mixed_params_list[no_shuffle_start_idx:no_shuffle_end_idx]
                        + tail_shuffled
                    )
                    label += "_oneContiguous"
                elif isinstance(constraint_value, Shuffle):
                    rng.shuffle(mixed_params_list)  # inplace
                    label += "_shuffled"
                elif isinstance(constraint_value, VaryParams):
                    rng.shuffle(mixed_params_list)  # inplace

                    def _perturb_params(param: SamplingParams):
                        top_k = param.top_k
                        if top_k is not None:
                            top_k = rng.integers(2, vocab_size // 3)
                        top_p = param.top_p
                        if top_p is not None:
                            top_p *= max(rng.random(), 1e-6)
                        temperature = param.temperature
                        if temperature is not None:
                            temperature *= max(rng.random(), 1e-6)
                        return SamplingParams(
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                        )

                    mixed_params_list = [_perturb_params(params) for params in mixed_params_list]
                    label += "_randomized"
                else:
                    label += f"_one{constraint_value}" if constraint_value is not None else ""
                test_cases.append((mixed_params_list, label))

        return test_cases

    @pytest.fixture(scope="function")
    def draft_lens(
        self,
        max_draft_len: int,
        sampling_params_list: list[SamplingParams],
        allow_zero_draft_len: bool,
    ) -> list[int]:
        """Generate per-request draft lengths.

        Currently drawn at random, every draft length is between 0
        (1) and max_draft_len if allow_zero_draft_len is True (False).
        """
        draft_len = list(
            np.random.default_rng(seed=42).integers(
                1 if (max_draft_len > 0 and not allow_zero_draft_len) else 0,
                max_draft_len + 1,
                size=(
                    len(
                        sampling_params_list,
                    )
                ),
            )
        )
        return draft_len

    @pytest.fixture(scope="function")
    def seq_slot_assignment(
        self, sampling_params_list: list[SamplingParams]
    ) -> tuple[list[int], int]:
        # Returns list of seq slots associated with each request and
        # total number of seq slots.
        #
        # Assumes a dense packing of requests in the sample state buffer.
        #
        # This choice only affects _unbatch_sampling_results, which is tested in
        # test_unbatch_sampling_results, overriding 'seq_slot_assignment' via test
        # parametrization.
        seq_slots = list(range(len(sampling_params_list)))
        return seq_slots, len(seq_slots)

    @pytest.fixture(scope="function")
    def mock_requests(
        self,
        sampling_params_list: list[SamplingParams],
        seq_slot_assignment: tuple[list[int], int],
        draft_lens: list[int],
    ) -> ScheduledRequests:
        return self._build_mock_requests(
            sampling_params_list=sampling_params_list,
            seq_slot_assignment=seq_slot_assignment,
            draft_lens=draft_lens,
        )

    def _build_mock_requests(
        self,
        sampling_params_list: list[SamplingParams],
        *,
        seq_slot_assignment: tuple[list[int], int],
        draft_lens: list[int],
    ) -> ScheduledRequests:
        """Build a batch of test requests consumable by sample_async."""
        seq_slots, num_seq_slots = seq_slot_assignment

        class ScheduledRequestsMock:
            def __init__(
                self,
                sampling_params_list: list[SamplingParams],
                *,
                draft_lens: list[int],
            ):
                self._sampling_params_list = sampling_params_list

                # NB:
                #   -  stop words are tested in test_write_finish_reasons
                #   -  'end_id' is tested in test_write_finish_reasons
                #   -  embedding bias is tested elsewhere
                #   -  py_min_length is tested elsewhere
                #   -  py_return_log_probs is tested elsewhere
                #   -  code paths gated by py_return_context_logits tested in test_select_generated_logits
                self._gen_requests = [
                    LlmRequest(
                        request_id=seq_slot,
                        max_new_tokens=(2 * draft_len),  # not used by tested code
                        input_tokens=[12],  # not used by tested code
                        sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
                        seq_slot=seq_slot,
                        is_streaming=False,  # not relevant for tested code
                        draft_tokens=(  # 'len(.py_draft_tokens)' is inspected by get_draft_token_length
                            torch.testing.make_tensor(
                                (draft_len,),
                                dtype=torch.int32,
                                device="cpu",
                            ).tolist()
                            if draft_len
                            else None
                        ),
                    )
                    for sampling_params, seq_slot, draft_len in zip(
                        sampling_params_list, seq_slots, draft_lens
                    )
                ]

            @property
            def context_requests(self) -> list[LlmRequest]:
                # Code paths excluded by this choice are addressed by test_select_generated_logits
                return []

            @property
            def generation_requests(self) -> list[LlmRequest]:
                # The batched sampling code in sample_async only checks that this is not empty
                return self._gen_requests

            def all_requests(self) -> list[LlmRequest]:
                # The sampling code relies on this ordering assumption
                return self.context_requests + self.generation_requests

        with torch.inference_mode(True):
            return cast(
                ScheduledRequests,
                ScheduledRequestsMock(sampling_params_list, draft_lens=draft_lens),
            )

    @pytest.fixture(scope="function")
    def model_outputs(
        self,
        mock_requests: ScheduledRequests,
        vocab_size: int,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Provide a batch of random logits for use as input to sample_async.

        This fixture also validates that the logits are not altered by the UUT.
        """
        total_steps = sum(get_draft_token_length(req) + 1 for req in mock_requests.all_requests())
        logits = torch.testing.make_tensor(
            (total_steps, vocab_size),
            dtype=torch.float32,
            device="cuda",
        )
        logits_orig = logits.clone()
        try:
            yield {
                # No 'd2t': Non-greedy sampling with 'd2t' is currently not
                #           supported and greedy case is tested elsewhere
                "logits": logits,
            }
        finally:
            torch.testing.assert_close(logits, logits_orig)

    @pytest.fixture(scope="function")
    def sampler(
        self,
        use_flashinfer: bool,
        max_draft_len: int,
        seq_slot_assignment: tuple[list[int], int],
    ) -> TorchSampler:
        return self._build_sampler(
            use_flashinfer=use_flashinfer,
            max_draft_len=max_draft_len,
            seq_slot_assignment=seq_slot_assignment,
        )

    def _build_sampler(
        self,
        *,
        use_flashinfer: bool,
        max_draft_len: int,
        seq_slot_assignment: tuple[list[int], int],
    ) -> TorchSampler:
        _, num_seq_slots = seq_slot_assignment
        return TorchSampler(
            TorchSampler.Args(
                max_seq_len=321,  # only used for stop criteria, tested separately
                max_draft_len=42,  # not used by TorchSampler
                max_beam_width=1,  # currently the only supported value
                max_num_sequences=num_seq_slots,
                max_total_draft_tokens=max_draft_len,
                disable_flashinfer_sampling=(not use_flashinfer),
                disable_overlap_scheduler=False,
            )
        )

    def _sample(
        self,
        sampler: TorchSampler,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, torch.Tensor],
        *,
        num_repeats: Optional[int] = None,
        allow_sync: bool = True,
    ) -> torch.Tensor:
        """Call sample_async.

        Optionally, run sampling repeatedly, e.g., to gather statistics.
        """
        assert not scheduled_requests.context_requests
        num_actual_repeats = num_repeats if num_repeats is not None else 1

        T = TypeVar("T")
        is_first = True

        def maybe_check_no_sync(func: Callable[[], T]) -> T:
            # The device-side sleep submitted by assert_no_cuda_sync blocks CUDA operations
            # once the amount of enqueued work becomes large enough.
            # Only checking the first sampling repetition to avoid this.
            nonlocal is_first
            with (
                assert_no_cuda_sync(sync_timeout_s=0.25)
                if (not allow_sync and is_first)
                else nullcontext()
            ):
                is_first = False
                return func()

        sample_states = [
            maybe_check_no_sync(
                lambda: sampler.sample_async(
                    scheduled_requests,
                    model_outputs=model_outputs,
                    num_context_logits_prefix_sum=[0],
                    resource_manager=None,  #  only used for tree sampling, which is not tested here
                )
            )
            for _ in range(num_actual_repeats)
        ]
        new_tokens_tensors = []
        for sample_state in sample_states:
            assert sample_state.sampler_event is not None
            sample_state.sampler_event.synchronize()
            assert sample_state.host is not None
            new_tokens_tensors.append(sample_state.host.new_tokens.unsqueeze(-1))
        new_tokens = torch.cat(new_tokens_tensors, dim=-1)
        if num_repeats is None:
            new_tokens = new_tokens.squeeze(-1)
        return new_tokens

    @pytest.mark.parametrize(
        "use_flashinfer, max_draft_len, sampling_params_list",
        [
            pytest.param(use_flashinfer, max_draft_len, [])
            for (use_flashinfer, max_draft_len) in product(
                [False, True],
                [0, 3],
            )
        ],
    )
    def test_backend_selection(
        self,
        sampler: TorchSampler,
        use_flashinfer: bool,
    ):
        """Check that TorchSampler uses the correct sampling backend."""
        expected_cls = (
            FlashInferGroupedStrategySampler if use_flashinfer else SimpleGroupedStrategySampler
        )
        assert sampler._grouped_sampler_cls == expected_cls

    @pytest.mark.parametrize(
        (
            "use_flashinfer",
            "max_draft_len",
            "draft_lens",
            "sampling_params_list",
            "params_label",
            "allow_zero_draft_len",
            "vocab_size",
        ),
        [
            # NB: non-zero draft len ensures that LlmRequest.py_target_probs is set.
            pytest.param(
                use_flashinfer,
                3,
                [3] * len(sampling_params_list),
                sampling_params_list,
                params_label,
                False,
                vocab_size,
                id=f"{'FlashInfer' if use_flashinfer else 'Torch'}-{params_label}",
            )
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (use_flashinfer, (sampling_params_list, params_label), vocab_size) in product(
                [False, True],
                _build_test_cases(
                    vocab_size=VOCAB_SIZE,
                    allow_greedy=False,  # Greedy does not return probs
                ),
                [VOCAB_SIZE],
            )
        ],
    )
    def test_probs(
        self,
        sampler: TorchSampler,
        mock_requests: ScheduledRequests,
        model_outputs: dict[str, torch.Tensor],
        draft_lens: list[int],
        vocab_size: int,
        params_label: str,
        allow_zero_draft_len: bool,  # used by fixtures
        sampling_params_list: list[SamplingParams],
        seq_slot_assignment: tuple[list[int], int],
    ):
        """Validate probabilities returned by sample_async.

        For suitable inputs, sample_async populates the py_target_probs attribute, storing
        the distribution from which the sampler has drawn the new tokens (typically these
        are the probabilities computed after applying temperature, top-p/k masking, etc.).
        This test checks that the presence of py_target_probs behaves as expected and
        validates the values of this attribute (when present).
        """

        @contextmanager
        def _uut_provider(is_warmup: bool) -> Generator[Callable[[], None], None, None]:
            torch.manual_seed(42)  # torch.testing.make_tensor does not accept Generator

            strategy_tags = {
                strategy_type: strategy_type.__args__[0].__args__[0]  # type: ignore
                for strategy_type in [
                    TemperatureOnly,
                    TopP,
                    TopK,
                    TopKTopP,
                ]
            }

            if is_warmup:
                # Use separate requests for warmup, because prob outputs are attached to
                # requests.
                uut_mock_requests = self._build_mock_requests(
                    sampling_params_list=sampling_params_list,
                    seq_slot_assignment=seq_slot_assignment,
                    draft_lens=draft_lens,
                )
            else:
                uut_mock_requests = mock_requests

            def _uut():
                _ = self._sample(
                    sampler,
                    scheduled_requests=uut_mock_requests,
                    model_outputs=model_outputs,
                    allow_sync=is_warmup,
                )

            yield _uut

            logit_offset = 0
            for req, draft_len in zip(uut_mock_requests.all_requests(), draft_lens):
                assert req.py_target_probs is not None
                probs = req.py_target_probs.cpu()
                assert probs.shape == (draft_len + 1, vocab_size)
                # NB: _request_strategy tested in TestStrategySelection
                strategy = _request_strategy(req, vocab_size=vocab_size)

                steps = draft_len + 1

                assert strategy is not GREEDY
                temperature = strategy[-1]
                assert temperature is not None
                req_logits = model_outputs["logits"][logit_offset : (logit_offset + steps), :].cpu()
                expected_probs_after_temperature = torch.softmax(req_logits / temperature, dim=-1)

                # check normalization
                torch.testing.assert_close(
                    probs.sum(dim=-1), torch.tensor(1.0).broadcast_to(probs.shape[:-1])
                )

                # Do not compare tiny probs (ignore floating point accuracy differences)
                prob_threshold = 1e-10
                expected_probs_after_temperature = torch.where(
                    expected_probs_after_temperature >= prob_threshold,
                    expected_probs_after_temperature,
                    0,
                )
                probs = torch.where(probs >= prob_threshold, probs, 0)
                expected_probs_after_temperature /= expected_probs_after_temperature.sum(
                    dim=-1, keepdim=True
                )
                probs /= probs.sum(dim=-1, keepdim=True)

                if strategy[0] == strategy_tags[TemperatureOnly]:
                    torch.testing.assert_close(probs, expected_probs_after_temperature)
                else:
                    if strategy[0] not in [
                        strategy_tags[strategy_type] for strategy_type in [TopP, TopK, TopKTopP]
                    ]:
                        raise ValueError(f"Unknown strategy: {strategy}")

                    top_k = None
                    if strategy[0] in [
                        strategy_tags[strategy_type] for strategy_type in [TopK, TopKTopP]
                    ]:
                        # Validate top-k
                        top_k = strategy[1]
                        assert top_k is not None

                        # Correct for possible zero probs in input
                        input_nnz = torch.count_nonzero(expected_probs_after_temperature, dim=-1)
                        top_k = torch.where(input_nnz < top_k, input_nnz, top_k)

                        nnz = torch.count_nonzero(probs, dim=-1)
                        if strategy[0] == strategy_tags[TopKTopP]:
                            # when top-k is followed by top-p, the result set is smaller
                            assert torch.le(nnz, top_k).all()
                        else:
                            torch.testing.assert_close(nnz, top_k)

                    if strategy[0] in [
                        strategy_tags[strategy_type] for strategy_type in [TopP, TopKTopP]
                    ]:
                        # Validate top-p
                        top_p = strategy[-2]
                        assert top_p is not None

                        if top_k is not None:
                            expected_probs_before_top_p, indices = (
                                expected_probs_after_temperature.topk(
                                    cast(int, top_k.amax().item()), dim=-1
                                )
                            )
                            expected_probs_before_top_p /= expected_probs_before_top_p.sum(
                                dim=-1, keepdim=True
                            )
                            probs_sorted = probs.gather(-1, indices)
                        else:
                            expected_probs_before_top_p = expected_probs_after_temperature
                            probs_sorted = probs

                            if params_label.startswith("single_"):
                                # top_p is chosen to cover possible edge cases
                                assert 1 in probs.count_nonzero(dim=-1)
                                assert len(set(probs.count_nonzero(dim=-1))) > 1

                        # Check that probs make top-p and that no index can be omitted without missing top-p
                        probs_sorted_pre_norm = torch.where(
                            probs_sorted != 0, expected_probs_before_top_p, 0.0
                        )
                        probs_sorted_pre_norm_nz = torch.where(
                            probs_sorted_pre_norm != 0, probs_sorted_pre_norm, float("inf")
                        )
                        assert torch.ge(
                            probs_sorted_pre_norm.sum(dim=-1),
                            cast(float, top_p),
                        ).all()
                        assert torch.lt(
                            probs_sorted_pre_norm.sum(dim=-1)
                            - probs_sorted_pre_norm_nz.amin(dim=-1),
                            cast(float, top_p),
                        ).all()

                    # All indices not selected must have logits less or equal
                    # to the smallest selected logit.
                    probs_selected_min = torch.where(
                        probs == 0.0, float("inf"), expected_probs_after_temperature
                    ).amin(dim=-1)
                    probs_other_max = torch.where(
                        probs == 0.0, expected_probs_after_temperature, 0.0
                    )
                    assert torch.le(probs_other_max.amax(dim=-1), probs_selected_min).all()

                    # Check selected probs agree up to normalization
                    expected_probs = torch.where(
                        probs != 0.0, expected_probs_after_temperature, 0.0
                    )
                    expected_probs /= expected_probs.sum(keepdim=True, dim=-1)
                    torch.testing.assert_close(probs, expected_probs)

                logit_offset += steps

        _run_test_with_warmup(_uut_provider)

    def _compute_probs(
        self,
        *,
        use_flashinfer: bool,
        model_outputs: dict[str, torch.Tensor],
        sampling_params_list: list[SamplingParams],
        seq_slot_assignment: tuple[list[int], int],
        vocab_size: int,
        max_draft_len: int,
        draft_lens: list[int],
    ) -> ScheduledRequests:
        """Construct a batch of requests with given sampling params and invoke sampler to compute probs.

        The probs (PMFs) corresponding to the provided model_outputs and sampling_params_list are returned
        in the py_target_probs attribute of the returned requests.

        Used by test_samples.
        """
        # Because max_draft_len can be zero and probs are not computed in this case,
        # a separate sampler instance (with larger max_draft_len) is needed to
        # compute probs in general.
        draft_len_with_probs = max(1, max_draft_len)
        sampler_with_probs = self._build_sampler(
            use_flashinfer=use_flashinfer,
            max_draft_len=draft_len_with_probs,
            seq_slot_assignment=seq_slot_assignment,
        )
        mock_requests_with_probs = self._build_mock_requests(
            sampling_params_list=sampling_params_list,
            seq_slot_assignment=seq_slot_assignment,
            # NB: non-zero draft len ensures that LlmRequest.py_target_probs is set.
            draft_lens=([draft_len_with_probs] * len(sampling_params_list)),
        )
        # zero-pad logits to draft_len_with_probs
        logits = model_outputs["logits"]
        logits_offset = 0
        steps_with_probs = draft_len_with_probs + 1
        logits_with_probs = torch.zeros(
            (steps_with_probs * len(mock_requests_with_probs.all_requests()), vocab_size),
            dtype=logits.dtype,
            device=logits.device,
        )
        for req_idx, draft_len in enumerate(draft_lens):
            steps = draft_len + 1
            logits_with_probs[
                (req_idx * steps_with_probs) : (req_idx * steps_with_probs + steps)
            ] = logits[logits_offset : (logits_offset + steps)]
            logits_offset += steps
        model_outputs_with_probs = model_outputs.copy()
        model_outputs_with_probs["logits"] = logits_with_probs
        _ = self._sample(
            sampler_with_probs,
            scheduled_requests=mock_requests_with_probs,
            model_outputs=model_outputs_with_probs,
        )
        return mock_requests_with_probs

    @staticmethod
    def _inject_batching_check(
        patch_ctx: pytest.MonkeyPatch,
        *,
        sampler: TorchSampler,
        use_flashinfer: bool,
    ):
        """Setup interception of sample_async and request grouping.

        If FlashInfer.sampling is used, this validates that at every
        invocation of sample_async, the sampling backend is called at most
        once for any given sampling strategy (if FlashInfer.sampling is used).

        Used by test_samples.
        """
        # FlashInfer sampling batches requests of the same kind (e.g. top-p)
        # together even if they have different parameter values (e.g. probability thresholds).
        # This variable tracks which request types have been encountered.
        flashinfer_keys_seen = set()

        if use_flashinfer:
            sample_grouped_strategies_orig = sampler._grouped_sampler_cls.sample_grouped_strategies

            def _sample_grouped_strategies(
                group_key: FlashInferGroupedStrategySampler.STRATEGY_KEY_TYPE,
                strategies: list[Strategy],
                logits: torch.Tensor,
                *,
                group_logit_indices: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None,
                return_probs: bool,
                group_metadata: StrategyMetadata | None = None,
            ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
                assert issubclass(group_key, sampling_utils_flashinfer._StrategyImpls.StrategyImpl)
                assert generator is sampler.get_generator(logits.device)
                nonlocal flashinfer_keys_seen
                assert group_key not in flashinfer_keys_seen
                flashinfer_keys_seen.add(group_key)
                return sample_grouped_strategies_orig(
                    group_key,
                    strategies,
                    logits,
                    group_logit_indices=group_logit_indices,
                    generator=generator,
                    return_probs=return_probs,
                )

            patch_ctx.setattr(
                sampler._grouped_sampler_cls,
                "sample_grouped_strategies",
                _sample_grouped_strategies,
            )

            sample_async_orig = sampler.sample_async

            def _sample_async(
                scheduled_requests: ScheduledRequests,
                model_outputs: dict[str, torch.Tensor],
                num_context_logits_prefix_sum: list[int],
                resource_manager=None,
            ):
                nonlocal flashinfer_keys_seen
                flashinfer_keys_seen.clear()
                res = sample_async_orig(
                    scheduled_requests,
                    model_outputs,
                    num_context_logits_prefix_sum,
                    resource_manager,
                )

                # Fast greedy path bypasses flashinfer sampling, so flashinfer_keys_seen
                # will be empty when all requests are greedy
                all_greedy = all(
                    _request_strategy(req, vocab_size=2**31) == GREEDY
                    for req in scheduled_requests.all_requests()
                )
                assert flashinfer_keys_seen or all_greedy
                return res

            patch_ctx.setattr(sampler, "sample_async", _sample_async)

    @dataclass(frozen=True, kw_only=True)
    class _TorchUtilsSamplingParams:
        """Variant of UtilsSamplingParams which stores torch.Tensor, to avoid device syncs.

        Used by test_samples.
        """

        temperature: Optional[torch.Tensor]
        top_p: Optional[torch.Tensor]
        top_k: Optional[torch.Tensor]

    @dataclass(frozen=True, kw_only=True)
    class _MockSamplingLogEntry:
        probs: torch.Tensor
        sampling_params: "TestBatchedSampling._TorchUtilsSamplingParams"

    @staticmethod
    def _instrument_sampling_backend(
        patch_ctx: pytest.MonkeyPatch,
        *,
        sampler: TorchSampler,
    ) -> list["TestBatchedSampling._MockSamplingLogEntry"]:
        """Setup interception of sampling routines.

        This patches the sampling backend. The added instrumentation records observed
        sampling parameters and input probs in the returned log. Instead of tokens, the
        patched sampling routines return indices into the log, permitting to retrieve the
        captured sampling inputs.

        Used by test_samples.
        """
        mock_sampling_log: list[TestBatchedSampling._MockSamplingLogEntry] = []

        def _mock_flashinfer_top_k_top_p(
            logits: torch.Tensor,
            *,
            top_k: torch.Tensor,
            top_p: torch.Tensor,
            filter_apply_order: str,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
        ) -> torch.Tensor:
            assert filter_apply_order == "top_k_first"
            assert deterministic
            assert not check_nan, "check_nan syncs"
            assert generator is sampler.get_generator(logits.device)
            nonlocal mock_sampling_log
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=torch.softmax(logits[row_idx], dim=-1),
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=top_k[row_idx],
                        top_p=top_p[row_idx],
                        temperature=None,
                    ),
                )
                for row_idx in range(logits.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log), len(mock_sampling_log) + len(new_entries)
            )
            mock_sampling_log += new_entries
            return mock_tokens

        patch_ctx.setattr(
            flashinfer.sampling,
            "top_k_top_p_sampling_from_logits",
            _mock_flashinfer_top_k_top_p,
        )

        def _mock_flashinfer_from_logits(
            logits: torch.Tensor,
            *,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
        ) -> torch.Tensor:
            assert deterministic
            assert not check_nan, "check_nan syncs"
            assert generator is sampler.get_generator(logits.device)
            nonlocal mock_sampling_log
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=torch.softmax(logits[row_idx], dim=-1),
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=None,
                        top_p=None,
                        temperature=None,
                    ),
                )
                for row_idx in range(logits.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log), len(mock_sampling_log) + len(new_entries)
            )
            mock_sampling_log += new_entries
            return mock_tokens

        patch_ctx.setattr(flashinfer.sampling, "sampling_from_logits", _mock_flashinfer_from_logits)

        def _mock_flashinfer_top_k(
            probs: torch.Tensor,
            *,
            top_k: torch.Tensor,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
        ) -> torch.Tensor:
            assert deterministic
            assert not check_nan, "check_nan syncs"
            assert generator is sampler.get_generator(probs.device)
            nonlocal mock_sampling_log
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=probs[row_idx],
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=top_k[row_idx],
                        top_p=None,
                        temperature=None,
                    ),
                )
                for row_idx in range(probs.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log), len(mock_sampling_log) + len(new_entries)
            )
            mock_sampling_log += new_entries
            return mock_tokens

        patch_ctx.setattr(flashinfer.sampling, "top_k_sampling_from_probs", _mock_flashinfer_top_k)

        def _mock_flashinfer_top_p(
            probs: torch.Tensor,
            *,
            top_p: torch.Tensor,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
        ) -> torch.Tensor:
            assert deterministic
            assert not check_nan, "check_nan syncs"
            assert generator is sampler.get_generator(probs.device)
            nonlocal mock_sampling_log
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=probs[row_idx],
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=None,
                        top_p=top_p[row_idx],
                        temperature=None,
                    ),
                )
                for row_idx in range(probs.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log), len(mock_sampling_log) + len(new_entries)
            )
            mock_sampling_log += new_entries
            return mock_tokens

        patch_ctx.setattr(flashinfer.sampling, "top_p_sampling_from_probs", _mock_flashinfer_top_p)

        def _mock_flashinfer_from_probs(
            probs: torch.Tensor,
            *,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
        ) -> torch.Tensor:
            assert deterministic
            assert not check_nan, "check_nan syncs"
            assert generator is sampler.get_generator(probs.device)
            nonlocal mock_sampling_log
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=probs[row_idx],
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=None,
                        top_p=None,
                        temperature=None,
                    ),
                )
                for row_idx in range(probs.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log), len(mock_sampling_log) + len(new_entries)
            )
            mock_sampling_log += new_entries
            return mock_tokens

        patch_ctx.setattr(flashinfer.sampling, "sampling_from_probs", _mock_flashinfer_from_probs)

        def _mock_torch_multinomial(
            probs: torch.Tensor,
            num_samples: int,
            generator: torch.Generator,
        ) -> torch.Tensor:
            assert generator is sampler.get_generator(probs.device)
            assert num_samples == 1
            nonlocal mock_sampling_log
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=probs[row_idx],
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=None,
                        top_p=None,
                        temperature=None,
                    ),
                )
                for row_idx in range(probs.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log), len(mock_sampling_log) + len(new_entries)
            )
            mock_sampling_log += new_entries
            return mock_tokens.unsqueeze(-1)

        patch_ctx.setattr(torch, "multinomial", _mock_torch_multinomial)

        return mock_sampling_log

    @staticmethod
    def _validate_intercepted_probs(
        log_entry: "TestBatchedSampling._MockSamplingLogEntry",
        *,
        vocab_size: int,
        expected_probs: torch.Tensor,
        req_params: UtilsSamplingParams,
    ):
        """Validate sampling inputs captured by the code injected via _instrument_sampling_backend.

        Used by test_samples.
        """
        # Tests rely on UUT handling temperature outside the sampling routines
        assert log_entry.sampling_params.temperature is None

        req_has_top_p = (
            log_entry.sampling_params.top_p is not None
            and log_entry.sampling_params.top_p.item() != 1
        )
        req_has_top_k = (
            log_entry.sampling_params.top_k is not None
            and log_entry.sampling_params.top_k.item() != vocab_size
        )
        if req_has_top_k:
            assert req_params.top_k is not None
            assert log_entry.sampling_params.top_k is not None
            assert req_params.top_k == log_entry.sampling_params.top_k.item()
        if req_has_top_p:
            assert req_params.top_p is not None
            assert log_entry.sampling_params.top_p is not None
            assert np.allclose(req_params.top_p, log_entry.sampling_params.top_p.item())
        if req_has_top_k or req_has_top_p:
            # for top-k and/or top-p _sampling_, probs contains only the top probs,
            # whereas log_entry.probs contains all probs passed to the sampling code.

            # validate selection in 'probs' is consistent with log_entry.probs
            log_entry_probs_selected = torch.where(expected_probs != 0, log_entry.probs.cpu(), 1)
            log_entry_probs_masked = torch.where(expected_probs == 0, log_entry.probs.cpu(), 0)
            assert torch.all(
                log_entry_probs_masked.amax(dim=-1) <= log_entry_probs_selected.amin(dim=-1)
            )

            # validate non-zero probs
            log_entry_probs_selected = torch.where(expected_probs != 0, log_entry.probs.cpu(), 0)
            log_entry_probs_selected /= log_entry_probs_selected.sum(-1)
            torch.testing.assert_close(log_entry_probs_selected, expected_probs)
        else:
            torch.testing.assert_close(log_entry.probs.cpu(), expected_probs)

    @staticmethod
    def _validate_token_frequencies(
        *,
        test_token_counts: torch.Tensor,
        test_expected_counts: torch.Tensor,
        num_samples: int,
    ):
        """Check consistency of observed and expected token frequencies.

        Used by test_samples.
        """
        # NB: G-test yields NaN if expected count is 0
        #     -> check those entries separately and mask them
        #     (https://stats.stackexchange.com/a/668064)
        #
        test_token_counts_for_zero_prob = torch.where(
            test_expected_counts != 0, 0, test_token_counts
        )
        assert (test_token_counts_for_zero_prob == 0).all()
        test_expected_counts_ma = np.ma.masked_array(
            test_expected_counts.numpy(),
            mask=(test_expected_counts.numpy() == 0),
        )
        test_token_counts_ma = np.ma.masked_array(
            test_token_counts.numpy(),
            mask=test_expected_counts_ma.mask,
        )

        # FlashInfer normalization is numerically inaccurate enough to
        # yield a tiny p-value in the test below, despite passing the
        # test's normalization check. Most likely, this mainly
        # affects the 'delta' distributions handled explicitly below.
        assert np.allclose(test_expected_counts_ma.sum(axis=-1), num_samples)
        test_expected_counts_ma /= test_expected_counts_ma.sum(axis=-1, keepdims=True)
        test_expected_counts_ma *= num_samples

        # Skip entries with exact agreement. Needed, because
        # 'power_divergence' generates NaN p-values otherwise.
        mask = ~(
            np.round(test_expected_counts_ma).astype(np.int64)
            == test_token_counts_ma.astype(np.int64)
        ).all(axis=-1)
        test_expected_counts_ma = test_expected_counts_ma[mask]
        test_token_counts_ma = test_token_counts_ma[mask]

        # Perform G-test (asymptotically approximated by Pearson's chi-square test) to
        # check that sampled tokens are consistent with the expected probs.
        #
        # NB: Need to use FP64 to avoid negative test statistic values.
        test_token_counts_ma = test_token_counts_ma.astype(np.float64)
        test_expected_counts_ma = test_expected_counts_ma.astype(np.float64)
        test_expected_counts_ma /= test_expected_counts_ma.sum(axis=-1, keepdims=True)
        test_expected_counts_ma *= num_samples
        test_result = power_divergence(
            f_obs=test_token_counts_ma,
            f_exp=test_expected_counts_ma,
            axis=-1,
            lambda_="log-likelihood",  # = KL divergence
        )
        if hasattr(test_result.pvalue, "mask"):
            assert test_result.pvalue.mask
            pvalue = test_result.pvalue.data
        else:
            pvalue = test_result.pvalue
        if not np.all(pvalue > 0.1):  # This can happen by "chance" (many test instances)
            # Fail test if sampled data are highly unlikely
            assert np.all(pvalue > 0.001)
            prob_delta = np.abs(test_token_counts_ma - test_expected_counts_ma) / num_samples
            # accept small prob differences
            prob_delta = np.where(prob_delta > 5e-2, prob_delta, 0)  # NB: this is rather liberal
            # bound relative differences on remaining probs
            prob_delta_rel = (
                np.ma.masked_array(num_samples * prob_delta, mask=test_expected_counts_ma.mask)
                / test_expected_counts_ma.data
            )
            assert prob_delta_rel.max() < 0.05

    @pytest.mark.parametrize(
        (
            "use_flashinfer",
            "max_draft_len",
            "sampling_params_list",
            "allow_zero_draft_len",
            "bypass_sampling",
            "vocab_size",
        ),
        [
            pytest.param(
                use_flashinfer,
                max_draft_len,
                sampling_params_list,
                allow_zero_draft_len,
                # Run full sampling test only for uniform batches, with/without probs, but skip
                # sampling statistics when varying draft lens etc. to validate batch handling:
                not (
                    (not is_mixed) and (not allow_zero_draft_len) and max_draft_len > 0
                ),  # bypass_sampling
                vocab_size,
                id=(
                    f"{'FlashInfer' if use_flashinfer else 'Torch'}"
                    f"-draft_len={0 if allow_zero_draft_len else 1}..{max_draft_len}"
                    f"-{params_label}"
                ),
            )
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (
                use_flashinfer,
                is_mixed,
                max_draft_len,
                allow_zero_draft_len,
                _build_test_cases,
                vocab_size,
            ) in product(
                [False, True],
                [False, True],
                [0, 3],
                [False, True],
                [_build_test_cases],
                [VOCAB_SIZE],
            )
            for (sampling_params_list, params_label) in _build_test_cases(
                vocab_size=vocab_size,
                include_all=False,
                include_uniform=(not is_mixed),
                include_mixed=is_mixed,
            )
            if allow_zero_draft_len or max_draft_len > 0
        ],
    )
    def test_samples(
        self,
        sampler: TorchSampler,
        mock_requests: ScheduledRequests,
        model_outputs: dict[str, torch.Tensor],
        draft_lens: list[int],
        vocab_size: int,
        sampling_params_list: list[SamplingParams],
        seq_slot_assignment: tuple[list[int], int],
        max_draft_len: int,
        use_flashinfer: bool,
        allow_zero_draft_len: bool,  # used by fixtures
        bypass_sampling: bool,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Validate tokens sampled by the sampler.

        This test validates the token generation by running many sampling iterations and comparing
        the frequencies of the sampled tokens against the distributions (PMFs) computed separately
        using the mechanism validated by `test_probs`.

        To save time, for test cases (cf. sampling_params_list) which mainly mix requests with
        sampling strategies validated by repeated sampling, repeated sampling is omitted. Instead,
        the sampling routines of the sampling backend are patched to capture their input
        logits / probs and return a pseudo-token identifying the capture result. Thus, the corresponding
        observed PMFs can be directly compared with the expected ones.
        """

        @contextmanager
        def _uut_provider(is_warmup: bool) -> Generator[Callable[[], None], None, None]:
            torch.manual_seed(42)  # torch.testing.make_tensor does not accept Generator

            # Compute sampling probabilities for the given sampling_params_list and
            # model_outputs. These probs, the computation of which is validated by 'test_probs',
            # are used to validate the batched sampling process later in this test.
            mock_requests_with_probs = self._compute_probs(
                use_flashinfer=use_flashinfer,
                model_outputs=model_outputs,
                sampling_params_list=sampling_params_list,
                seq_slot_assignment=seq_slot_assignment,
                vocab_size=vocab_size,
                max_draft_len=max_draft_len,
                draft_lens=draft_lens,
            )

            num_samples = 5000 if not (bypass_sampling or is_warmup) else 1

            # filled when bypass_sampling=True
            mock_sampling_log: Optional[list[TestBatchedSampling._MockSamplingLogEntry]] = None

            with monkeypatch.context() as patch_ctx:
                self._inject_batching_check(
                    patch_ctx, sampler=sampler, use_flashinfer=use_flashinfer
                )
                if bypass_sampling:
                    mock_sampling_log = self._instrument_sampling_backend(
                        patch_ctx, sampler=sampler
                    )

                @dataclass
                class UutResult:
                    new_tokens_repeats: torch.Tensor

                @dataclass
                class UutResultWrapper:
                    result: Optional[UutResult] = None

                res = UutResultWrapper()

                def _uut(res=res):
                    new_tokens_repeats = self._sample(
                        sampler,
                        scheduled_requests=mock_requests,
                        model_outputs=model_outputs,
                        num_repeats=num_samples,
                        allow_sync=is_warmup,
                    )
                    res.result = UutResult(new_tokens_repeats=new_tokens_repeats)

                yield _uut

                assert res.result is not None
                new_tokens_repeats = res.result.new_tokens_repeats

            # remove 'beam' dimension
            assert new_tokens_repeats.size(-2) == 1
            new_tokens_repeats = new_tokens_repeats.squeeze(-2)

            # compute token frequencies
            new_tokens_repeats_safe = new_tokens_repeats.clamp(
                min=0, max=vocab_size - 1
            )  # tame uninitialized memory
            token_counts = torch.zeros(
                (*new_tokens_repeats_safe.shape[:-1], vocab_size),
                device=new_tokens_repeats_safe.device,
                dtype=torch.int32,
            )
            token_counts = (
                token_counts.view((-1, vocab_size))
                .scatter_add_(
                    dim=1,
                    index=new_tokens_repeats_safe.view((-1, new_tokens_repeats_safe.size(-1))),
                    src=torch.ones_like(
                        new_tokens_repeats_safe.view((-1, new_tokens_repeats_safe.size(-1)))
                    ),
                )
                .view(token_counts.shape)
            )
            token_counts = token_counts.to(dtype=torch.float32)
            assert (token_counts.sum(-1, keepdim=True) == num_samples).all()

            logits = model_outputs["logits"]
            for req_idx, (req, req_with_probs, draft_len) in enumerate(
                zip(
                    mock_requests.all_requests(),
                    mock_requests_with_probs.all_requests(),
                    draft_lens,
                )
            ):
                strategy = _request_strategy(req, vocab_size=vocab_size)
                assert strategy == _request_strategy(req_with_probs, vocab_size=vocab_size)
                if strategy is GREEDY:  # handle Greedy case explicitly
                    # greedy never returns probs
                    assert getattr(req_with_probs, "py_target_probs", None) is None
                    assert getattr(req, "py_target_probs", None) is None
                    req_logit_offset = sum(draft_len + 1 for draft_len in draft_lens[:req_idx])
                    req_logits = logits[req_logit_offset : (req_logit_offset + draft_len + 1)]
                    tokens_expected = torch.argmax(req_logits, dim=-1, keepdim=True)
                    tokens_sampled = new_tokens_repeats[: (draft_len + 1), req.py_seq_slot, :]
                    assert torch.all(tokens_expected.cpu() == tokens_sampled)
                    continue  # nothing else to check

                assert req_with_probs.py_target_probs is not None
                probs = req_with_probs.py_target_probs.cpu()
                assert probs.size(0) >= draft_len + 1
                probs = probs[: (draft_len + 1)]

                # check probs are returned only when needed
                should_return_probs = bool(draft_len)
                assert (
                    hasattr(req, "py_target_probs") and req.py_target_probs is not None
                ) == should_return_probs
                # check probs
                if should_return_probs:
                    assert req.py_target_probs is not None
                    torch.testing.assert_close(req.py_target_probs.cpu(), probs)

                if bypass_sampling:  # fast path (mock sampling)
                    assert mock_sampling_log is not None
                    for step_idx in range(draft_len + 1):
                        log_idx = new_tokens_repeats[step_idx, req.py_seq_slot, 0]
                        log_entry = mock_sampling_log[log_idx]
                        req_params = _request_get_sampling_params(req)
                        expected_probs = probs[step_idx]
                        self._validate_intercepted_probs(
                            log_entry,
                            vocab_size=vocab_size,
                            expected_probs=expected_probs,
                            req_params=req_params,
                        )
                else:
                    test_token_counts = token_counts[: (draft_len + 1), req.py_seq_slot]
                    test_expected_counts = num_samples * probs.cpu()
                    self._validate_token_frequencies(
                        test_token_counts=test_token_counts,
                        test_expected_counts=test_expected_counts,
                        num_samples=num_samples,
                    )

        _run_test_with_warmup(_uut_provider)

    @staticmethod
    def _build_seq_slot_assignments() -> list[tuple[list[int], int, str]]:
        """Build seq_slot assignments.

        This constructs various seq_slot assignments (see seq_slot_assignment method
        for details), which are useful for validating the unbatching of sampling results.
        """
        rng = np.random.default_rng(seed=42)

        max_seq_slots: Final = 2048
        margin: Final = 12

        seq_slot_assignments = []
        for include_first, include_last in product([False, True], [False, True]):
            total_seq_slots = rng.integers(max_seq_slots // 2, max_seq_slots)
            start = 0 if include_first else rng.integers(margin)
            end = total_seq_slots - (0 if include_last else rng.integers(margin))
            for dense in [False, True]:
                if dense:
                    seq_slots = range(start, end)
                else:
                    allowed_slots = np.arange(start, end)
                    num_seq_slots = rng.integers(len(allowed_slots) // 2, len(allowed_slots))
                    seq_slots = list(rng.choice(allowed_slots, num_seq_slots, replace=False))
                seq_slot_assignments.append(
                    (
                        seq_slots,
                        total_seq_slots,
                        (
                            f"lo_{'in' if include_first else 'out'}"
                            f"_hi_{'in' if include_last else 'out'}"
                            f"_{'dense' if dense else 'sparse'}"
                        ),
                    )
                )

        return seq_slot_assignments

    @pytest.mark.parametrize(
        (
            "use_flashinfer",
            "max_draft_len",
            "allow_zero_draft_len",
            "vocab_size",
            "seq_slot_assignment",
            "ordered",
        ),
        [
            pytest.param(
                False,  # NB: _unbatch_sampling_results does not depend on backend
                max_draft_len,
                allow_zero_draft_len,
                vocab_size,
                (seq_slots, total_seq_slots),
                ordered,
                id=(
                    f"draft_len={0 if allow_zero_draft_len else 1}..{max_draft_len}"
                    f"-{label}-{'ordered' if ordered else 'permuted'}"
                ),
            )
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (
                is_mixed,
                max_draft_len,
                allow_zero_draft_len,
                _build_seq_slot_assignments,
                vocab_size,
                ordered,
            ) in product(
                [False, True],
                [0, 3],
                [False, True],
                [_build_seq_slot_assignments],
                [VOCAB_SIZE],
                [False, True],
            )
            for (seq_slots, total_seq_slots, label) in _build_seq_slot_assignments()
        ],
    )
    def test_unbatch_sampling_results(
        self,
        sampler: TorchSampler,
        vocab_size: int,  # used by fixtures
        seq_slot_assignment: tuple[list[int], int],
        max_draft_len: int,
        use_flashinfer: bool,  # used by fixtures
        allow_zero_draft_len: bool,  # used by fixtures
        ordered: bool,
    ):
        """Validate _unbatch_sampling_results.

        Considers variable numbers of generated tokens per request and varying seq_slot
        assignments. By using unique integers as fictitious "token" values, the test
        validates that the sampling results are copied into the correct locations in
        the output buffers.
        """

        @contextmanager
        def _uut_provider(is_warmup: bool) -> Generator[Callable[[], None], None, None]:
            seq_slots, total_seq_slots = seq_slot_assignment
            seq_slots_tensor = torch.tensor(seq_slots, dtype=torch.int32)

            torch.manual_seed(42)  # torch.testing.make_tensor does not accept Generator
            rng = np.random.default_rng(seed=42)

            draft_lens = list(
                rng.integers(
                    1 if (max_draft_len > 0 and not allow_zero_draft_len) else 0,
                    max_draft_len + 1,
                    size=(
                        len(
                            seq_slots,
                        )
                    ),
                )
            )

            req_num_steps = torch.tensor(draft_lens, dtype=torch.int32) + 1
            total_steps = cast(int, req_num_steps.sum())

            new_tokens_cuda = torch.testing.make_tensor(
                (max_draft_len + 1, total_seq_slots, 1),
                device="cuda",
                dtype=torch.int32,
            )
            new_tokens_cuda_snapshot = new_tokens_cuda.clone()

            batch_req_indices = torch.arange(0, len(seq_slots), dtype=torch.int32)
            if not ordered:
                batch_req_indices = batch_req_indices[torch.randperm(batch_req_indices.numel())]

            first_token = rng.integers(123456)
            batch_next_tokens_cuda_int = torch.arange(
                first_token, first_token + total_steps, dtype=torch.int32, device="cuda"
            ).unsqueeze(1)  # Add a dimension for beam width

            batched_sampling_result = _BatchedSamplingResult(
                batch_req_indices=batch_req_indices.clone(),
                batch_next_tokens_cuda_int=batch_next_tokens_cuda_int.clone(),
            )
            seq_slots_tensor_snapshot = seq_slots_tensor.clone()

            @dataclass
            class UutResult:
                new_tokens_host: torch.Tensor

            @dataclass
            class UutResultWrapper:
                result: Optional[UutResult] = None

            res = UutResultWrapper()

            def _uut(res=res):
                new_tokens_host = sampler._unbatch_sampling_results(
                    batched_sampling_result=batched_sampling_result,
                    new_tokens_cuda=new_tokens_cuda,
                    req_num_generated_tokens=req_num_steps,
                    seq_slots=seq_slots_tensor,
                )
                res.result = UutResult(new_tokens_host=new_tokens_host)

            yield _uut

            torch.cuda.synchronize()
            assert res.result is not None
            new_tokens_host = res.result.new_tokens_host
            assert new_tokens_host.device == torch.device("cpu")

            # check for unwanted side effects
            for slot in range(total_seq_slots):
                if slot in seq_slots:
                    continue
                torch.testing.assert_close(
                    new_tokens_cuda_snapshot[:, slot], new_tokens_cuda[:, slot]
                )
            torch.testing.assert_close(
                batch_next_tokens_cuda_int, batched_sampling_result.batch_next_tokens_cuda_int
            )
            torch.testing.assert_close(batch_req_indices, batched_sampling_result.batch_req_indices)
            torch.testing.assert_close(seq_slots_tensor, seq_slots_tensor_snapshot)

            # validate tokens returned
            input_offset = 0
            for req_idx in batch_req_indices.tolist():
                steps = draft_lens[req_idx] + 1
                seq_slot = seq_slots[req_idx]
                req_tokens = batch_next_tokens_cuda_int[input_offset : (input_offset + steps)]
                torch.testing.assert_close(new_tokens_cuda[:steps, seq_slot], req_tokens)
                torch.testing.assert_close(new_tokens_host[:steps, seq_slot], req_tokens.cpu())
                input_offset += steps

        _run_test_with_warmup(_uut_provider, max_sync_s=0.2)
