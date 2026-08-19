# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    ContextManager,
    Final,
    Generator,
    Optional,
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
from utils.util import UutProvider, assert_no_cuda_sync, force_ampere, run_test_with_warmup

from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    convert_wordlist,
    get_draft_token_length,
)
from tensorrt_llm._torch.pyexecutor.sampler import (
    SampleStateTensorsHostTorch,
    SampleStateTorch,
    TorchSampler,
    _BatchedSamplingResult,
    _request_get_sampling_params,
    _request_strategy,
    _SeedManager,
)
from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import min_p_renorm_probs
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import UtilsSamplingParams
from tensorrt_llm._torch.pyexecutor.sampler.sampler_strategy import (
    GREEDY,
    BeamSearch,
    FlashInferGroupedStrategySampler,
    Greedy,
    MinP,
    RequestSeeds,
    Strategy,
    StrategyMetadata,
    TemperatureOnly,
    TopK,
    TopKTopP,
    TopP,
    TopPDecayMetadata,
    resolve_sampling_strategy,
    sample,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.sampling_params import SamplingParams


class TestSetupSamplerStepRequestSelection:
    @staticmethod
    def _make_request(
        *,
        is_attention_dp_dummy: bool = False,
        is_finished: bool = False,
        py_is_draft: bool = False,
    ) -> LlmRequest:
        return cast(
            LlmRequest,
            SimpleNamespace(
                is_attention_dp_dummy=is_attention_dp_dummy,
                is_finished=is_finished,
                py_is_draft=py_is_draft,
            ),
        )

    def test_collect_new_requests_for_setup_includes_adp_dummy_generation_requests(self):
        scheduled_requests = ScheduledRequests()

        context_request = self._make_request()
        finished_context_request = self._make_request(is_finished=True)
        draft_context_request = self._make_request(py_is_draft=True)
        adp_dummy_generation_request = self._make_request(is_attention_dp_dummy=True)
        regular_generation_request = self._make_request()
        finished_adp_dummy_generation_request = self._make_request(
            is_attention_dp_dummy=True, is_finished=True
        )
        draft_adp_dummy_generation_request = self._make_request(
            is_attention_dp_dummy=True, py_is_draft=True
        )

        scheduled_requests.context_requests_last_chunk = [
            context_request,
            finished_context_request,
            draft_context_request,
        ]
        scheduled_requests.generation_requests = [
            adp_dummy_generation_request,
            regular_generation_request,
            finished_adp_dummy_generation_request,
            draft_adp_dummy_generation_request,
        ]

        collected = TorchSampler._collect_new_requests_for_setup(scheduled_requests)
        assert len(collected) == 4
        assert collected[0] is context_request
        assert collected[1] is adp_dummy_generation_request
        assert collected[2] is finished_adp_dummy_generation_request
        assert collected[3] is draft_adp_dummy_generation_request


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
        py_sampling_strategy: Strategy | None
        # Read by the row_stride query in sampler_common; these tests are
        # single-beam, so the static admission width is 1.
        py_beam_width: int

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
        request.py_sampling_strategy = None  # used for caching
        request.py_beam_width = 1
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

    # --- min_p ---
    # A min_p strategy is ("min_p", top_k, top_p, min_p, temperature). When
    # unset, top_k carries the disabled sentinel 0 ("keep all"; sanitized to
    # vocab_size downstream) and top_p carries 1.0, so min_p composes with any
    # subset of temperature/top_k/top_p.

    @pytest.mark.parametrize(
        "trivial_temperature, trivial_top_p, trivial_top_k",
        [
            pytest.param(temperature, top_p, top_k)
            for (temperature, top_k, top_p) in product(
                TEMPERATURE_NEUTRAL_VALS, TOP_K_NEUTRAL_VALS, TOP_P_NEUTRAL_VALS
            )
        ],
    )
    def test_min_p_only(
        self,
        trivial_temperature: Optional[float],
        trivial_top_p: Optional[float],
        trivial_top_k: Optional[int],
    ):
        params = SamplingParams(
            min_p=0.1, temperature=trivial_temperature, top_p=trivial_top_p, top_k=trivial_top_k
        )
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 5
        assert strat[0] == "min_p"
        assert strat[1] == 0  # top_k disabled sentinel (0 == "keep all")
        assert strat[2] == pytest.approx(1.0)  # top_p disabled sentinel
        assert strat[3] == pytest.approx(0.1)
        assert strat[4] == pytest.approx(1.0)  # temperature default

    def test_min_p_with_temperature(self):
        params = SamplingParams(min_p=0.1, temperature=0.8)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert strat[0] == "min_p"
        assert strat[3] == pytest.approx(0.1)
        assert strat[4] == pytest.approx(0.8)

    def test_min_p_with_top_k_top_p(self):
        params = SamplingParams(min_p=0.1, top_k=42, top_p=0.7, temperature=0.8)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert len(strat) == 5
        assert strat[0] == "min_p"
        assert strat[1] == 42
        assert strat[2] == pytest.approx(0.7)
        assert strat[3] == pytest.approx(0.1)
        assert strat[4] == pytest.approx(0.8)

    def test_min_p_0_not_selected(self):
        # min_p == 0 disables min_p; a plain temperature strategy is chosen.
        params = SamplingParams(min_p=0.0, temperature=0.7)
        request = self._build_mock_llm_request(params)
        strat = _request_strategy(request, vocab_size=self.VOCAB_SIZE)
        assert strat[0] == "temperature"

    def test_min_p_1_is_greedy(self):
        # min_p == 1 keeps only the row max, i.e. an explicit greedy control
        # (like top_p == 0), so it must not reach the min_p sampling path.
        params = SamplingParams(min_p=1.0, temperature=0.7)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        assert _request_strategy(request, vocab_size=self.VOCAB_SIZE) is GREEDY

    @pytest.mark.parametrize(
        "greedy_kwargs",
        [
            pytest.param({"top_k": 1}, id="top_k_1"),
            pytest.param({"temperature": 0}, id="temperature_0"),
        ],
    )
    def test_min_p_greedy_triggers_win(self, greedy_kwargs: dict[str, Any]):
        # An explicit greedy trigger collapses to a single token even with min_p.
        params = SamplingParams(min_p=0.1, **greedy_kwargs)
        self._check_params(params)
        request = self._build_mock_llm_request(params)
        assert _request_strategy(request, vocab_size=self.VOCAB_SIZE) is GREEDY

    def test_param_validation(self):
        with pytest.raises(ValueError, match="require temperature >= 0, got temperature=-1"):
            SamplingParams(temperature=-1)

        with pytest.raises(ValueError, match="require 0 <= top_p <= 1, got top_p=-1"):
            SamplingParams(top_p=-1)

        with pytest.raises(ValueError, match="require 0 <= top_p <= 1, got top_p=2"):
            SamplingParams(top_p=2)

        with pytest.raises(ValueError, match="require top_k >= 0, got top_k=-1"):
            SamplingParams(top_k=-1)

        with pytest.raises(ValueError, match="require 0 <= min_p <= 1, got min_p=-1"):
            SamplingParams(min_p=-1)

        with pytest.raises(ValueError, match="require 0 <= min_p <= 1, got min_p=2"):
            SamplingParams(min_p=2)

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


@force_ampere
@pytest.mark.parametrize(
    "draft_len, with_ctx_chunking, with_ctx_last_chunk, with_gen",
    [
        pytest.param(draft_len, with_ctx_chunking, with_ctx_last_chunk, with_gen)
        for (draft_len, with_ctx_chunking, with_ctx_last_chunk, with_gen) in product(
            [0, 3],
            [False, True],
            [False, True],
            [False, True],
        )
        if with_ctx_chunking or with_ctx_last_chunk or with_gen
    ],
)
def test_select_generated_logits(
    draft_len: int, with_ctx_chunking: bool, with_ctx_last_chunk: bool, with_gen: bool
):
    # Currently only checks that this works and does not sync

    device = torch.device("cuda")

    @contextmanager
    def _test_runner(is_warmup: bool) -> Generator[Callable[[], None], None, None]:
        draft_len_req1 = draft_len
        draft_len_req2 = draft_len + 1  # test with different draft lens

        class ContextRequestMock:
            def __init__(self, is_last_context_chunk: bool, return_context_logits: bool):
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
                self.py_draft_tokens = torch.empty(draft_len, dtype=torch.int32, device=device)
                self.sampling_config = SamplingConfig(beam_width=1)
                # Read by the row_stride query in sampler_common.
                self.py_beam_width = 1

            def get_beam_width_by_iter(
                self, for_next_iteration: bool = False
            ) -> int:  # Torch sampler accesses this, but it does not affect this test
                return self.sampling_config.beam_width

        def _build_scheduled_requests() -> ScheduledRequests:
            scheduled_requests = ScheduledRequests()
            scheduled_requests.context_requests_chunking = (
                [
                    # This request is expected to be skipped
                    cast(
                        LlmRequest,
                        ContextRequestMock(is_last_context_chunk=False, return_context_logits=True),
                    )
                ]
                if with_ctx_chunking
                else []
            )
            scheduled_requests.context_requests_last_chunk = (
                [
                    # NB: One request with py_return_context_logits is enough
                    #     to trigger tested code.
                    cast(
                        LlmRequest,
                        ContextRequestMock(is_last_context_chunk=True, return_context_logits=True),
                    ),
                    cast(
                        LlmRequest,
                        ContextRequestMock(is_last_context_chunk=True, return_context_logits=False),
                    ),
                    cast(
                        LlmRequest,
                        ContextRequestMock(is_last_context_chunk=True, return_context_logits=True),
                    ),
                ]
                if with_ctx_last_chunk
                else []
            )

            # NB: Currently this list is not inspected, UUT only checks that this
            #     is not empty.
            scheduled_requests.generation_requests = (
                [
                    cast(LlmRequest, GenRequestMock(draft_len=draft_len_req1)),
                    cast(LlmRequest, GenRequestMock(draft_len=draft_len_req2)),
                ]
                if with_gen
                else []
            )
            return scheduled_requests

        expected_num_requests = with_ctx_last_chunk * 3 + with_gen * 2
        expected_req_num_beams = torch.tensor([1] * expected_num_requests, dtype=torch.int32)

        num_context_logits_prefix_sum = [0]
        if with_ctx_chunking:
            # context req. 1 (assume context len. 10)
            num_context_logits_prefix_sum.append(num_context_logits_prefix_sum[-1] + 10 + 1)
        if with_ctx_last_chunk:
            # context req. 2 (assume context len. 100)
            num_context_logits_prefix_sum.append(num_context_logits_prefix_sum[-1] + 100 + 1)
            # context req. 3 (not returning context)
            num_context_logits_prefix_sum.append(num_context_logits_prefix_sum[-1] + 0 + 1)
            # context req. 4 (assume context len. 50)
            num_context_logits_prefix_sum.append(num_context_logits_prefix_sum[-1] + 50 + 1)

        expected_req_num_generation_steps = [
            *(
                [
                    1,  # context req. 2
                    1,  # context req. 3
                    1,  # context req. 4
                ]
                if with_ctx_last_chunk
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

        if expected_req_num_generation_steps_tensor.numel() > 0:
            expected_req_offsets = torch.cumsum(
                expected_req_num_generation_steps_tensor, dim=0
            ).roll(1)
            expected_req_offsets[0] = 0
        else:
            expected_req_offsets = torch.empty_like(expected_req_num_generation_steps_tensor)

        generation_requests_total_steps = (
            (draft_len_req1 + 1) + (draft_len_req2 + 1) if with_gen else 0
        )

        vocab_size = 12

        num_total_steps = num_context_logits_prefix_sum[-1] + generation_requests_total_steps
        all_logits = torch.empty((num_total_steps, vocab_size))

        for i in range(all_logits.size(0)):
            all_logits[i, :] = torch.arange(i, i + vocab_size)

        all_logits_cuda = all_logits.to(device=device)

        expected_logit_indices = []
        if with_ctx_last_chunk:
            if with_ctx_chunking:
                begin_offset = 11
            else:
                begin_offset = 0
            expected_logit_indices += [
                begin_offset + 100,  # gen logits from context req. 2
                begin_offset + 101,  # gen logits from context req. 3
                begin_offset + 152,  # gen logits from context req. 4
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
            selected_requests: list[LlmRequest]
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
                selected_requests,
                sampling_requests_metadata,
                selected_logits,
            ) = TorchSampler._select_generated_logits(
                _build_scheduled_requests(),
                all_logits_cuda,
                num_context_logits_prefix_sum=num_context_logits_prefix_sum,
            )
            res.result = UutResult(
                selected_requests=selected_requests,
                req_num_generated_tokens=sampling_requests_metadata.req_num_generated_tokens,
                req_num_beams=sampling_requests_metadata.req_num_beams,
                req_num_steps=sampling_requests_metadata.req_num_steps,
                req_offsets=sampling_requests_metadata.req_offsets,
                selected_logits=selected_logits,
            )

        yield _uut

        # Check results
        assert res.result is not None

        assert len(res.result.selected_requests) == expected_num_requests
        torch.testing.assert_close(
            res.result.req_num_generated_tokens.to("cpu"), expected_req_num_generation_steps_tensor
        )
        torch.testing.assert_close(res.result.req_num_beams.to("cpu"), expected_req_num_beams)
        torch.testing.assert_close(
            res.result.req_num_steps.to("cpu"), expected_req_num_generation_steps_tensor
        )
        torch.testing.assert_close(res.result.req_offsets.to("cpu"), expected_req_offsets)
        torch.testing.assert_close(res.result.selected_logits.to("cpu"), expected_logits)

    run_test_with_warmup(_test_runner, max_sync_s=0.3)


def test_stable_greedy_cache_key_includes_sequence_slots(monkeypatch: pytest.MonkeyPatch):
    sampler = object.__new__(TorchSampler)
    sampler.max_beam_width = 1
    sampler._stable_greedy_request_ids = []
    sampler._stable_greedy_seq_slots = []
    sampler._stable_greedy_seq_slots_host = None
    sampler._stable_greedy_seq_slots_cuda = None
    monkeypatch.setattr(sampler, "_copy_to_host", lambda tensor: tensor.clone())
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.sampler.sampler.prefer_pinned", lambda: False
    )

    original_tensor_to = torch.Tensor.to

    def copy_without_cuda(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copy_without_cuda)

    logits = torch.tensor([[0.0, 1.0, 2.0]])
    new_tokens = torch.zeros((1, 2, 1), dtype=torch.int32)
    requests = [
        LlmRequest(
            request_id=0,
            max_new_tokens=4,
            input_tokens=[1],
            sampling_config=SamplingConfig(),
            seq_slot=seq_slot,
            is_streaming=False,
            is_draft=is_draft,
        )
        for seq_slot, is_draft in ((0, False), (1, True))
    ]

    for seq_slot, request in enumerate(requests):
        scheduled_requests = ScheduledRequests()
        scheduled_requests.generation_requests = [request]

        (
            _,
            seq_slots_host,
            _,
            seq_slots_cuda,
            _,
            _,
            single_step_greedy,
        ) = sampler._process_requests(
            scheduled_requests,
            {"logits": logits},
            new_tokens,
            [0],
        )

        assert single_step_greedy
        assert seq_slots_host.tolist() == [seq_slot]
        assert seq_slots_cuda.tolist() == [seq_slot]
        assert new_tokens[0, seq_slot, 0].item() == 2


@force_ampere
def test_greedy_no_repeat_ngram_uses_token_ban_path():
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=16,
            max_draft_len=0,
            max_num_sequences=1,
            max_beam_width=1,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=True,
        )
    )
    request = LlmRequest(
        request_id=0,
        max_new_tokens=4,
        input_tokens=[1, 2, 1],
        sampling_config=SamplingConfig(),
        seq_slot=0,
        is_streaming=False,
    )
    setattr(request, "py_no_repeat_ngram_size", 2)
    scheduled_requests = ScheduledRequests()
    scheduled_requests.generation_requests = [request]
    logits = torch.tensor([[0.0, 0.0, 10.0, 9.0]], device="cuda")

    *_, new_tokens_host, single_step_greedy = sampler._process_requests(
        scheduled_requests,
        {"logits": logits},
        sampler.store.new_tokens,
        [0],
    )
    torch.cuda.synchronize()

    assert not single_step_greedy
    assert new_tokens_host.reshape(-1)[0].item() == 3


@force_ampere
@pytest.mark.parametrize(
    ("penalty_name", "penalty_value"),
    [
        pytest.param("repetition_penalty", 100.0, id="repetition"),
        pytest.param("presence_penalty", 2.0, id="presence"),
        pytest.param("frequency_penalty", 2.0, id="frequency"),
    ],
)
def test_greedy_occurrence_penalties_bypass_stable_path(penalty_name: str, penalty_value: float):
    if penalty_name == "repetition_penalty":
        sampling_params = SamplingParams(repetition_penalty=penalty_value)
    elif penalty_name == "presence_penalty":
        sampling_params = SamplingParams(presence_penalty=penalty_value)
    else:
        assert penalty_name == "frequency_penalty"
        sampling_params = SamplingParams(frequency_penalty=penalty_value)

    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=16,
            max_draft_len=0,
            max_num_sequences=1,
            max_beam_width=1,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=True,
        )
    )
    request = LlmRequest(
        request_id=0,
        max_new_tokens=4,
        input_tokens=[1],
        sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
        seq_slot=0,
        is_streaming=False,
    )

    admission = ScheduledRequests()
    admission.context_requests_last_chunk = [request]
    sampler.setup_sampler_step(admission)

    scheduled_requests = ScheduledRequests()
    scheduled_requests.generation_requests = [request]
    logits = torch.tensor([[0.0, 10.0, 9.0]], device="cuda")

    *_, new_tokens_host, single_step_greedy = sampler._process_requests(
        scheduled_requests,
        {"logits": logits},
        sampler.store.new_tokens,
        [0],
    )
    torch.cuda.synchronize()

    assert not single_step_greedy
    assert new_tokens_host.reshape(-1)[0].item() == 2


class TestFinishReasons:
    NOT_FINISHED = FinishReason.NOT_FINISHED
    STOP_WORDS = FinishReason.STOP_WORDS
    END_ID = FinishReason.END_ID
    LENGTH = FinishReason.LENGTH

    def test_single_step_greedy_updates_finish_reasons_and_filters_completed_requests(self):
        sampler = object.__new__(TorchSampler)
        sampler.max_seq_len = 20
        sampler._track_pending_steps = False
        requests = [
            LlmRequest(
                request_id=0,
                seq_slot=0,
                input_tokens=[2, 0],
                max_new_tokens=1,
                end_id=2,
                sampling_config=SamplingConfig(),
                is_streaming=False,
            ),
            LlmRequest(
                request_id=1,
                seq_slot=1,
                input_tokens=[2, 0],
                max_new_tokens=1,
                end_id=2,
                sampling_config=SamplingConfig(),
                is_streaming=False,
            ),
            LlmRequest(
                request_id=2,
                seq_slot=2,
                input_tokens=[2, 0],
                max_new_tokens=10,
                end_id=2,
                sampling_config=SamplingConfig(),
                is_streaming=False,
            ),
        ]
        requests[2].finish_by(FinishReason.LENGTH, 0)
        new_tokens = torch.tensor([2, 7, 99], dtype=torch.int32)
        state = SampleStateTorch(
            requests=requests,
            device=None,
            host=SampleStateTensorsHostTorch(
                new_tokens=new_tokens,
                finish_reasons=None,
                first_finish_reasons=None,
                single_step_greedy=True,
            ),
        )

        sampler.update_requests(state)

        assert all(request.is_finished for request in requests)
        # The first request reaches EOS and length together; EOS takes precedence.
        assert not requests[0].is_finished_due_to_length
        assert requests[1].is_finished_due_to_length
        assert requests[0].get_tokens(0)[-1] == 2
        assert requests[1].get_tokens(0)[-1] == 7
        assert requests[2].get_tokens(0) == [2, 0]

    class RequestCase:
        MAX_NEW_TOKENS = 10
        MAX_NUM_SEQUENCES = 128
        seq_slots = torch.randperm(MAX_NUM_SEQUENCES).tolist()
        BEAM = 0

        def __init__(
            self,
            *,
            prompt: list[int],
            new_tokens: list[int],
            finish_reasons: list[FinishReason],
            max_new_tokens: int = MAX_NEW_TOKENS,
            end_id: Optional[int] = None,
            num_draft_tokens: int | None = None,
            stop_words_list: Optional[list[list[int]]] = None,
        ):
            seq_slot = self.seq_slots.pop()  # random seq slot in MAX_NUM_SEQUENCES
            self.prompt = prompt
            if num_draft_tokens is None:
                num_draft_tokens = len(new_tokens) - 1
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
                draft_tokens=new_tokens[:num_draft_tokens],
            )
            assert len(new_tokens) == len(finish_reasons)
            self.new_tokens = new_tokens
            self.finish_reasons = finish_reasons

        def __repr__(self):
            return f"RequestCase({self.prompt=}, {self.new_tokens=}, {self.finish_reasons=}, \
            {self.request.max_new_tokens=}, {self.request.end_id=}, {self.request.stop_words_list=})"

        @classmethod
        def build(
            cls,
            requests: list["TestFinishReasons.RequestCase"],
            *,
            check_no_cuda_sync: bool = True,
            extra_context: Callable[[], ContextManager[Any]] | None = None,
            expect_result: bool = True,
        ) -> UutProvider:
            @contextmanager
            def _uut_provider(is_warmup: bool) -> Generator[Callable[[], None], None, None]:
                max_tokens = set(len(req.new_tokens) for req in requests)
                assert len(max_tokens) == 1
                max_draft_len = max_tokens.pop() - 1
                sampler_args = TorchSampler.Args(
                    max_seq_len=20,
                    max_draft_len=max_draft_len,
                    max_total_draft_tokens=max_draft_len,
                    # Fill with many more max requests than below,
                    # so we can test that write_finish_reasons uses seq_slots correctly
                    max_num_sequences=cls.MAX_NUM_SEQUENCES,
                    max_beam_width=1,
                    disable_overlap_scheduler=False,
                )
                sampler = TorchSampler(args=sampler_args)
                finish_reasons_store = sampler._finish_reasons_handler.store
                # setup the sampler store for the requests
                scheduled_requests = ScheduledRequests()
                scheduled_requests.context_requests_last_chunk = [req.request for req in requests]
                sampler.setup_sampler_step(scheduled_requests)

                # fill with garbage value so we can observe that finish reasons are filled
                # with NOT_FINISHED before we write to them.
                finish_reasons_store.finish_reasons_cuda.fill_(205)
                seq_slots_host = torch.tensor(
                    [req.request.py_seq_slot for req in requests], device="cpu", dtype=torch.int64
                )
                seq_slots_cuda = seq_slots_host.to(device="cuda", non_blocking=True)
                seq_lens_cuda = torch.tensor(
                    [req.request.max_beam_num_tokens for req in requests],
                    dtype=torch.int32,
                    device="cuda",
                )
                new_tokens_cuda = torch.tensor(
                    [req.new_tokens for req in requests], dtype=torch.int32, device="cuda"
                ).T
                sampler.store.new_tokens[:, seq_slots_cuda, cls.BEAM] = new_tokens_cuda

                is_draft_batch = False
                # Capture return value of write_finish_reasons for use after _uut() runs.
                write_finish_reasons_result: list[torch.Tensor] = []

                def _uut():
                    with extra_context() if extra_context is not None else nullcontext():
                        result = sampler._finish_reasons_handler.write_finish_reasons(
                            seq_slots_host=seq_slots_host,
                            is_draft_batch=is_draft_batch,
                            seq_slots_cuda=seq_slots_cuda,
                            seq_lens_cuda=seq_lens_cuda,
                            new_tokens_cuda=sampler.store.new_tokens,
                            first_finish_reasons_cuda=None,
                        )
                        write_finish_reasons_result.append(result)

                yield _uut

                if not expect_result:
                    assert len(write_finish_reasons_result) == 0, (
                        f"Expected no results, got {len(write_finish_reasons_result)}"
                    )
                else:
                    assert len(write_finish_reasons_result) > 0, "Expected results, got none"
                    # write_finish_reasons_result[0] is the return value from write_finish_reasons.
                    reasons = write_finish_reasons_result[0][:, seq_slots_cuda, cls.BEAM].T.tolist()

                    for actual, request in zip(reasons, requests, strict=True):
                        expected = request.finish_reasons
                        msg = f"actual={[FinishReason(reason) for reason in actual]} \
                            != expected={expected}\nFor {request}"
                        assert actual == [reason.value for reason in expected], msg

            return _uut_provider

    @classmethod
    def test_write_finish_reasons(cls):
        """We don't really care about the finish reason past the first infraction, because we're not going to use it,
        although in some instance it is written anyway."""
        uut_provider = cls.RequestCase.build(
            [
                cls.RequestCase(
                    prompt=[13, 14],
                    new_tokens=[60, 61, 62],
                    # We pre-fill the finish reasons with NOT_FINISHED.
                    finish_reasons=[cls.NOT_FINISHED, cls.NOT_FINISHED, cls.NOT_FINISHED],
                ),
                cls.RequestCase(
                    prompt=[7, 8, 6],
                    stop_words_list=[[12, 13]],
                    new_tokens=[12, 13, 60],
                    finish_reasons=[cls.NOT_FINISHED, cls.STOP_WORDS, cls.NOT_FINISHED],
                ),
                cls.RequestCase(
                    prompt=[1, 2, 3, 4],
                    end_id=99,
                    new_tokens=[55, 99, 58],
                    finish_reasons=[cls.NOT_FINISHED, cls.END_ID, cls.NOT_FINISHED],
                ),
                cls.RequestCase(
                    prompt=[4, 5, 6],
                    max_new_tokens=2,
                    new_tokens=[56, 57, 59],
                    # The LENGTH check happens to not have an early exit
                    finish_reasons=[cls.NOT_FINISHED, cls.LENGTH, cls.LENGTH],
                ),
                cls.RequestCase(
                    prompt=[1, 12],
                    stop_words_list=[[12, 13], [14, 15]],
                    new_tokens=[13, 14, 15],
                    # We don't use early exit to avoid stream synchronization for stop words
                    finish_reasons=[cls.STOP_WORDS, cls.NOT_FINISHED, cls.STOP_WORDS],
                ),
                cls.RequestCase(
                    prompt=[1, 12],
                    stop_words_list=[
                        [12, 13, 14, 15],
                        [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                    ],
                    new_tokens=[13, 14, 15],
                    # Stop words of different lengths are handled correctly with respect to padding of stop words
                    # and tokens
                    finish_reasons=[cls.NOT_FINISHED, cls.NOT_FINISHED, cls.STOP_WORDS],
                ),
                cls.RequestCase(
                    prompt=[1],
                    max_new_tokens=2,
                    end_id=99,
                    stop_words_list=[[1, 12]],
                    new_tokens=[12, 99, 63],
                    # Different infractions are written to different places as
                    # we don't have an early exit between infractions
                    finish_reasons=[cls.STOP_WORDS, cls.END_ID, cls.LENGTH],
                ),
                cls.RequestCase(
                    prompt=[1, 12, 56, 67, 68, 234, 678],
                    stop_words_list=[[12, 56, 67, 68, 234, 678, 129, 182]],
                    new_tokens=[129, 182, 600],
                    # Notice the offending stop sequence is concatenated, as we lookback
                    finish_reasons=[cls.NOT_FINISHED, cls.STOP_WORDS, cls.NOT_FINISHED],
                ),
                cls.RequestCase(
                    prompt=[1, 12],
                    end_id=99,
                    max_new_tokens=1,
                    stop_words_list=[[1, 12, 99]],
                    new_tokens=[99, 100, 101],
                    # The latest infraction check overrides the earlier infraction checks,
                    # hence the first finish_reason is END_ID
                    finish_reasons=[cls.END_ID, cls.LENGTH, cls.LENGTH],
                ),
            ]
        )

        run_test_with_warmup(uut_provider, max_sync_s=0.5)

    @classmethod
    def test_are_stop_words_isnt_called_when_no_stop_words(cls, monkeypatch: pytest.MonkeyPatch):
        """We don't want to call are_stop_words when there are no stop words because it's expensive"""

        def stop_words_that_raises(*args, **kwargs):
            raise AssertionError

        @contextmanager
        def raising_stop_words_ctx(expect_raise: bool) -> Generator[None, None, None]:
            with monkeypatch.context() as patch_ctx:
                patch_ctx.setattr(FinishReasonsHandler, "_are_stop_words", stop_words_that_raises)
                patch_ctx.setattr(
                    FinishReasonsHandler,
                    "_are_stop_words_single_token",
                    stop_words_that_raises,
                )
                with pytest.raises(AssertionError) if expect_raise else nullcontext():
                    yield

        uut_provider_with_stop_words = cls.RequestCase.build(
            [
                cls.RequestCase(
                    prompt=[1],
                    stop_words_list=[[1, 2]],
                    new_tokens=[4],
                    finish_reasons=[cls.NOT_FINISHED],
                ),
                cls.RequestCase(
                    prompt=[1],
                    stop_words_list=[[1]],
                    new_tokens=[4],
                    finish_reasons=[cls.NOT_FINISHED],
                ),
            ],
            extra_context=lambda: raising_stop_words_ctx(True),
            expect_result=False,
        )
        run_test_with_warmup(uut_provider_with_stop_words, max_sync_s=0.5)

        uut_provider_with_stop_words = cls.RequestCase.build(
            [
                cls.RequestCase(
                    prompt=[1],
                    new_tokens=[4],
                    finish_reasons=[cls.NOT_FINISHED],
                )
            ],
            extra_context=lambda: raising_stop_words_ctx(False),
        )
        run_test_with_warmup(uut_provider_with_stop_words, max_sync_s=0.5)

    @classmethod
    def test_are_stop_words_single_token_is_called_when_single_token_stop_words_are_present(
        cls, monkeypatch: pytest.MonkeyPatch
    ):
        """We don't want to call are_stop_words when there are only single token stop words because it's expensive"""

        def stop_words_that_raises(*args, **kwargs):
            raise AssertionError

        @contextmanager
        def raising_single_token_stop_words_ctx(expect_raise: bool) -> Generator[None, None, None]:
            with monkeypatch.context() as patch_ctx:
                patch_ctx.setattr(
                    FinishReasonsHandler,
                    "_are_stop_words_single_token",
                    stop_words_that_raises,
                )
                with pytest.raises(AssertionError) if expect_raise else nullcontext():
                    yield

        uut_provider_with_stop_words = cls.RequestCase.build(
            [
                cls.RequestCase(
                    prompt=[1],
                    stop_words_list=[[1]],
                    new_tokens=[4],
                    finish_reasons=[cls.NOT_FINISHED],
                )
            ],
            extra_context=lambda: raising_single_token_stop_words_ctx(True),
            expect_result=False,
        )
        run_test_with_warmup(uut_provider_with_stop_words, max_sync_s=0.5)

        uut_provider_with_stop_words = cls.RequestCase.build(
            [
                cls.RequestCase(
                    prompt=[1],
                    stop_words_list=[[1, 2]],
                    new_tokens=[4],
                    finish_reasons=[cls.NOT_FINISHED],
                )
            ],
            extra_context=lambda: raising_single_token_stop_words_ctx(False),
        )
        run_test_with_warmup(uut_provider_with_stop_words, max_sync_s=0.5)

    @classmethod
    def check_resize_and_update_stop_words_buffer(
        cls,
        old_stop_words: torch.Tensor,
        old_past_tokens: torch.Tensor,
        new_stop_words: torch.Tensor,
        new_past_tokens: torch.Tensor,
        seq_slots_to_compare: torch.Tensor,
        num_draft_tokens: int,
    ):
        old_num_stop_words = old_stop_words.shape[0]
        old_stop_word_length = old_stop_words.shape[1]

        old_past_token_length = old_past_tokens.shape[0]

        # These sizes should not change after the resize
        assert old_stop_words.shape[2] == new_stop_words.shape[2]
        assert old_past_tokens.shape[1] == new_past_tokens.shape[1]
        assert old_past_tokens.shape[2] == new_past_tokens.shape[2]

        assert (
            new_stop_words[-old_num_stop_words:, -old_stop_word_length:, seq_slots_to_compare]
            == old_stop_words[..., seq_slots_to_compare]
        ).all()
        # initial fill has an offset of 1, as there will be a shift happening during sample_async
        assert (
            new_past_tokens[-old_past_token_length:-num_draft_tokens, seq_slots_to_compare, :]
            == old_past_tokens[:-num_draft_tokens, seq_slots_to_compare, :]
        ).all()

    @classmethod
    def test_stop_words_buffer_resize(cls, monkeypatch: pytest.MonkeyPatch):
        @contextmanager
        def check_resize_ctx() -> Generator[None, None, None]:
            with monkeypatch.context() as patch_ctx:
                setup_sampler_step_orig = TorchSampler.setup_sampler_step

                def setup_sampler_step_with_size_check(self, scheduled_requests: ScheduledRequests):
                    # RequestCase.build calls setup_sampler_step to fill the buffers for all context requests
                    # Move the context requests to the generation requests
                    scheduled_requests.generation_requests = scheduled_requests.context_requests
                    # Add a request that enforces a resize
                    scheduled_requests.context_requests_last_chunk = [
                        cls.RequestCase(
                            prompt=[1],
                            stop_words_list=[
                                [x for x in range(2 * TorchSampler.DEFAULT_MAX_STOP_WORD_LENGTH)]
                            ],
                            new_tokens=[4],
                            finish_reasons=[cls.NOT_FINISHED],
                        ).request
                    ]
                    # Store the old stop words and past tokens for comparison
                    old_stop_words = self.store.stop_words.clone()
                    old_past_tokens = self.store.past_tokens.clone()
                    # Call setup sampler step to trigger the resize
                    setup_sampler_step_orig(self, scheduled_requests)

                    # Check if sizes are correct
                    assert self.store.stop_words.shape[0] == TorchSampler.DEFAULT_MAX_STOP_WORDS
                    assert (
                        self.store.stop_words.shape[1]
                        == 2 * TorchSampler.DEFAULT_MAX_STOP_WORD_LENGTH
                    )
                    assert self.max_tokens == 1
                    assert (
                        self.store.past_tokens.shape[0]
                        == 2 * TorchSampler.DEFAULT_MAX_STOP_WORD_LENGTH - 1 + self.max_tokens
                    )
                    # Check if values are added correctly
                    seq_slots_to_compare_cuda = torch.Tensor(
                        [
                            scheduled_requests.generation_requests[x].py_seq_slot
                            for x in range(len(scheduled_requests.generation_requests))
                        ]
                    ).to(device="cuda", dtype=torch.int32, non_blocking=True)
                    cls.check_resize_and_update_stop_words_buffer(
                        old_stop_words,
                        old_past_tokens,
                        self.store.stop_words,
                        self.store.past_tokens,
                        seq_slots_to_compare_cuda,
                        self.max_draft_len,
                    )

                patch_ctx.setattr(
                    TorchSampler, "setup_sampler_step", setup_sampler_step_with_size_check
                )
                yield

        # The test adds one more request
        num_requests = 8
        uut_provider_with_resize_on_demand = cls.RequestCase.build(
            [
                cls.RequestCase(
                    prompt=[1 + x],
                    stop_words_list=[[x + 100]],
                    new_tokens=[x + 4, x + 3, x + 2],
                    finish_reasons=[cls.NOT_FINISHED, cls.NOT_FINISHED, cls.NOT_FINISHED],
                )
                for x in range(num_requests)
            ],
            extra_context=lambda: check_resize_ctx(),
        )
        run_test_with_warmup(uut_provider_with_resize_on_demand, max_sync_s=0.5)

    @classmethod
    def test_stop_words_buffer_resizes_on_demand(cls, monkeypatch: pytest.MonkeyPatch):
        @contextmanager
        def check_resize_ctx() -> Generator[None, None, None]:
            with monkeypatch.context() as patch_ctx:
                setup_sampler_step_orig = TorchSampler.setup_sampler_step

                def setup_sampler_step_with_size_check(self, scheduled_requests: ScheduledRequests):
                    setup_sampler_step_orig(self, scheduled_requests)
                    assert self.store.stop_words.shape[0] == TorchSampler.DEFAULT_MAX_STOP_WORDS
                    assert (
                        self.store.stop_words.shape[1]
                        == 2 * TorchSampler.DEFAULT_MAX_STOP_WORD_LENGTH
                    )
                    assert self.max_tokens == 1
                    assert (
                        self.store.past_tokens.shape[0]
                        == 2 * TorchSampler.DEFAULT_MAX_STOP_WORD_LENGTH - 1 + self.max_tokens
                    )

                patch_ctx.setattr(
                    TorchSampler, "setup_sampler_step", setup_sampler_step_with_size_check
                )
                yield

        uut_provider_with_resize_on_demand = cls.RequestCase.build(
            [
                cls.RequestCase(
                    prompt=[1],
                    stop_words_list=[
                        [x for x in range(2 * TorchSampler.DEFAULT_MAX_STOP_WORD_LENGTH)]
                    ],
                    new_tokens=[4],
                    finish_reasons=[cls.NOT_FINISHED],
                )
            ],
            extra_context=lambda: check_resize_ctx(),
        )
        run_test_with_warmup(uut_provider_with_resize_on_demand, max_sync_s=None)

    @staticmethod
    def _all_beams_finished_reference(row: torch.Tensor, beam_width: int) -> bool:
        """The per-request reduction the batched prefix count replaces."""
        return bool(
            (row[:beam_width] != FinishReason.NOT_FINISHED.value).sum().item() == beam_width
        )

    def test_finished_beam_prefix_lengths_matches_per_request_reduction(self):
        """The batched prefix count answers the per-request question for every width."""
        store_width = 4
        reasons = [
            FinishReason.NOT_FINISHED.value,
            FinishReason.END_ID.value,
            FinishReason.STOP_WORDS.value,
            FinishReason.LENGTH.value,
        ]
        rows = list(product(reasons, repeat=store_width))
        finish_reasons = torch.tensor(rows, dtype=torch.int32)

        prefix_lengths = TorchSampler._finished_beam_prefix_lengths(finish_reasons)

        assert len(prefix_lengths) == len(rows)
        for row, prefix_length in zip(finish_reasons, prefix_lengths):
            for beam_width in range(1, store_width + 1):
                assert (prefix_length >= beam_width) == self._all_beams_finished_reference(
                    row, beam_width
                ), f"row={row.tolist()} beam_width={beam_width}"

    def test_finished_beam_prefix_lengths_ignores_columns_past_beam_width(self):
        """Reasons beyond a request's beam width must not complete it, or vice versa."""
        # Slot 0 uses 2 beams and both finished; the padding columns are unfinished.
        # Slot 1 uses 2 beams, only the second finished; the padding columns are set.
        finish_reasons = torch.tensor(
            [
                [FinishReason.END_ID.value, FinishReason.LENGTH.value, 0, 0],
                [
                    0,
                    FinishReason.END_ID.value,
                    FinishReason.END_ID.value,
                    FinishReason.END_ID.value,
                ],
            ],
            dtype=torch.int32,
        )

        prefix_lengths = TorchSampler._finished_beam_prefix_lengths(finish_reasons)

        assert prefix_lengths[0] >= 2
        assert prefix_lengths[1] < 2

    def test_handle_first_finish_reasons_completes_only_fully_finished_requests(self):
        """Requests are completed, and their per-beam reasons recorded, only when all
        of their own beams finished -- across differing beam widths in one batch."""
        sampler = object.__new__(TorchSampler)
        store_width = 4
        # Slot 0: beam_width 2, both finished -> completes.
        # Slot 1: beam_width 4, first beam unfinished -> stays running.
        # Slot 2: beam_width 1, finished -> completes.
        finish_reasons = torch.tensor(
            [
                [FinishReason.END_ID.value, FinishReason.LENGTH.value, 0, 0],
                [
                    0,
                    FinishReason.END_ID.value,
                    FinishReason.END_ID.value,
                    FinishReason.END_ID.value,
                ],
                [FinishReason.STOP_WORDS.value, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        assert finish_reasons.size(1) == store_width
        prefix_lengths = TorchSampler._finished_beam_prefix_lengths(finish_reasons)
        finish_reasons_list = finish_reasons.tolist()

        class RecordingLlmRequest(LlmRequest):
            """LlmRequest that records the per-beam reasons the sampler sets."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.recorded_reasons: list[tuple[int, FinishReason]] = []

            def set_finished_reason(self, finish_reason: FinishReason, beam: int) -> None:
                self.recorded_reasons.append((beam, finish_reason))
                super().set_finished_reason(finish_reason, beam)

        requests = []
        for seq_slot, beam_width in enumerate([2, 4, 1]):
            # The beam width must come from the sampling config: it sizes the
            # request's C++ per-beam state, which set_finished_reason indexes.
            request = RecordingLlmRequest(
                request_id=seq_slot,
                seq_slot=seq_slot,
                input_tokens=[1],
                max_new_tokens=10,
                end_id=2,
                sampling_config=SamplingConfig(beam_width=beam_width),
                is_streaming=False,
            )
            assert request.py_beam_width == beam_width
            requests.append(request)

        completed = [
            sampler._handle_first_finish_reasons(request, prefix_lengths, finish_reasons_list)
            for request in requests
        ]

        assert completed == [True, False, True]
        assert requests[0].state == LlmRequestState.GENERATION_COMPLETE
        assert requests[1].state != LlmRequestState.GENERATION_COMPLETE
        assert requests[2].state == LlmRequestState.GENERATION_COMPLETE
        # Only the request's own beams are reported, in beam order.
        assert requests[0].recorded_reasons == [
            (0, FinishReason.END_ID),
            (1, FinishReason.LENGTH),
        ]
        assert requests[1].recorded_reasons == []
        assert requests[2].recorded_reasons == [(0, FinishReason.STOP_WORDS)]


@pytest.mark.parametrize("min_p", [0.0, 0.1, 0.5, 0.9])
def test_min_p_renorm_probs(min_p: float):
    """min_p_renorm_probs keeps tokens with p >= min_p * max and renormalizes."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 16), dim=-1)

    got = min_p_renorm_probs(probs.clone(), min_p)

    max_probs = probs.max(dim=-1, keepdim=True).values
    kept = probs >= (min_p * max_probs)
    expected = torch.where(kept, probs, torch.zeros_like(probs))
    expected = expected / expected.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(got, expected)
    # every row still sums to 1 and the argmax token always survives
    torch.testing.assert_close(got.sum(dim=-1), torch.ones(probs.size(0)))
    assert (got.gather(1, probs.argmax(dim=-1, keepdim=True)) > 0).all()


def test_min_p_renorm_probs_per_request_tensor():
    """A per-request min_p tensor applies a distinct threshold per row."""
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(3, 16), dim=-1)
    min_p = torch.tensor([0.0, 0.3, 0.95])

    got = min_p_renorm_probs(probs.clone(), min_p)

    max_probs = probs.max(dim=-1, keepdim=True).values
    kept = probs >= (min_p.reshape(-1, 1) * max_probs)
    expected = torch.where(kept, probs, torch.zeros_like(probs))
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(got, expected)
    # row 0 (min_p=0) keeps everything; row 2 (min_p=0.95) prunes more aggressively
    assert (got[0] > 0).all()
    assert (got[2] > 0).sum() <= (got[0] > 0).sum()


def test_min_p_sample_top_k_disabled_sentinel():
    """min_p + unset top_k must survive the standalone sample() dispatch.

    Draft-model rejection sampling resolves strategies with vocab_size=2**31
    (the greedy probe), so a min_p request with an unset top_k carries the
    disabled-top_k sentinel 0. That 0 flows straight into sample() ->
    top_k_top_p_sampling_batch without sanitize_top_k, so the vanilla path must
    treat it as "keep all" instead of tripping ``assert top_k > 1``. Regression
    test for min_p under speculative decoding with rejection sampling.
    """
    min_p = 0.5
    # ("min_p", top_k, top_p, min_p, temperature) with the top_k=0 sentinel.
    strategy: MinP = ("min_p", 0, 1.0, min_p, 1.0)

    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    # Must not raise (top_k=0 previously hit ``assert top_k > 1``).
    tokens, _, _ = sample(strategy, logits.clone())

    assert tokens.shape == (4,)
    # min_p filtering was applied: every sampled token clears the min_p mask.
    probs = torch.softmax(logits, dim=-1)
    kept = probs >= (min_p * probs.max(dim=-1, keepdim=True).values)
    assert kept.gather(1, tokens.unsqueeze(-1)).all()


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
            MinP: SamplingParams(min_p=0.02, top_k=40, top_p=0.9, temperature=1.0),
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
                strategy_name = _get_strategy_name(
                    cast(Type[Strategy], strategy_type),
                )
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
                strategy_name = _get_strategy_name(
                    cast(Type[Strategy], strategy_type),
                )
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
            for constraint_value in [
                None,  # all sub-batches have at least two requests
                0,  # one sub-batch omitted
                1,  # one size-1 sub-batch
                OneContinguous(),  # one contiguous sub-batch, rest shuffled
                Shuffle(),  # random ordering
                VaryParams(),  # random ordering + randomized request parameter values
            ]:
                mixed_params_list: list[SamplingParams] = []
                constrained_indices = None
                for strategy_type, params in BASE_CASES.items():
                    sub_batch_size = rng.integers(low=2, high=max_sub_batch_size).item()
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
                    strategy_name = _get_strategy_name(cast(Type[Strategy], strategy_type))
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
                            top_k = int(rng.integers(2, vocab_size // 3))
                        top_p = param.top_p
                        if top_p is not None:
                            top_p *= max(rng.random(), 1e-6)
                        temperature = param.temperature
                        if temperature is not None:
                            temperature *= max(rng.random(), 1e-6)
                        min_p = param.min_p
                        if min_p is not None:
                            min_p *= max(rng.random(), 1e-6)
                        return SamplingParams(
                            top_p=top_p,
                            top_k=top_k,
                            min_p=min_p,
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

        with torch.inference_mode(True):
            scheduled_requests = ScheduledRequests()
            # Code paths excluded by this choice are addressed by test_select_generated_logits
            scheduled_requests.context_requests_chunking = []
            # Code paths excluded by this choice are addressed by test_select_generated_logits
            scheduled_requests.context_requests_last_chunk = []
            # NB:
            #   -  stop words are tested in test_write_finish_reasons
            #   -  'end_id' is tested in test_write_finish_reasons
            #   -  embedding bias is tested elsewhere
            #   -  py_min_length is tested elsewhere
            #   -  py_return_log_probs is tested elsewhere
            #   -  code paths gated by py_return_context_logits tested in test_select_generated_logits
            scheduled_requests.generation_requests = [
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
            return scheduled_requests

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
        max_draft_len: int,
        seq_slot_assignment: tuple[list[int], int],
    ) -> TorchSampler:
        return self._build_sampler(
            max_draft_len=max_draft_len,
            seq_slot_assignment=seq_slot_assignment,
        )

    def _build_sampler(
        self,
        *,
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> torch.Tensor:
        """Call sample_async.

        Optionally, run sampling repeatedly, e.g., to gather statistics.
        """
        assert scheduled_requests.num_context_requests == 0

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

        with monkeypatch.context() as patcher:
            # Ensure that internal sampler data structures are set up for all requests
            # (production code only considers context requests and examines other not mocked
            # LlmRequest fields)
            def _mock_filter(self, requests: ScheduledRequests) -> list[LlmRequest]:
                return requests.all_requests()

            patcher.setattr(TorchSampler, "_collect_new_requests_for_setup", _mock_filter)

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
            host_new_tokens = sample_state.host.new_tokens
            if sample_state.host.single_step_greedy:
                # The stable greedy path copies one token per active request instead of
                # the full [step, slot, beam] buffer. This fixture uses dense sequence
                # slots, so restore that layout before comparing sampling results.
                assert host_new_tokens.shape == (len(sample_state.requests),)
                host_new_tokens = host_new_tokens.reshape(1, -1, 1)
            new_tokens_tensors.append(host_new_tokens.unsqueeze(-1))
        new_tokens = torch.cat(new_tokens_tensors, dim=-1)
        if num_repeats is None:
            new_tokens = new_tokens.squeeze(-1)
        return new_tokens

    @pytest.mark.parametrize(
        (
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
                3,
                [3] * len(sampling_params_list),
                sampling_params_list,
                params_label,
                False,
                vocab_size,
                id=f"FlashInfer-{params_label}",
            )
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for ((sampling_params_list, params_label), vocab_size) in product(
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
        monkeypatch: pytest.MonkeyPatch,
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
                    MinP,
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
                    monkeypatch=monkeypatch,
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
                        strategy_tags[strategy_type]
                        for strategy_type in [TopP, TopK, TopKTopP, MinP]
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

                    if strategy[0] == strategy_tags[MinP]:
                        # Renorm preserves the ratio, so every kept token satisfies
                        # prob >= min_p * max (holds with top_k/top_p also applied).
                        min_p_val = cast(float, strategy[3])
                        kept = probs != 0.0
                        ratio = probs / probs.amax(dim=-1, keepdim=True)
                        assert torch.all((ratio >= min_p_val - 1e-6)[kept])

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

        run_test_with_warmup(
            _uut_provider,
            max_sync_s=None,  # NB: assert_no_cuda_sync called in TestBatchedSampler._sample
        )

    def _compute_probs(
        self,
        *,
        model_outputs: dict[str, torch.Tensor],
        sampling_params_list: list[SamplingParams],
        seq_slot_assignment: tuple[list[int], int],
        vocab_size: int,
        max_draft_len: int,
        draft_lens: list[int],
        monkeypatch: pytest.MonkeyPatch,
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
            monkeypatch=monkeypatch,
        )
        return mock_requests_with_probs

    @staticmethod
    def _inject_batching_check(
        patch_ctx: pytest.MonkeyPatch,
        *,
        sampler: TorchSampler,
    ):
        """Setup interception of sample_async and request grouping.

        Validates that at every invocation of sample_async, the FlashInfer
        sampling backend is called at most once for any given sampling strategy.

        Used by test_samples.
        """
        # FlashInfer sampling batches requests of the same kind (e.g. top-p)
        # together even if they have different parameter values (e.g. probability thresholds).
        # This variable tracks which request types have been encountered.
        flashinfer_keys_seen: set[Any] = set()

        assert sampler._grouped_sampler_cls == FlashInferGroupedStrategySampler
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
            seeds: Optional[RequestSeeds] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor] | float]:
            assert generator is sampler.get_generator(logits.device)
            if isinstance(group_key, tuple):
                assert isinstance(group_key[0], str)
            else:
                assert isinstance(group_key, str)
            nonlocal flashinfer_keys_seen
            assert (group_key, return_probs) not in flashinfer_keys_seen
            flashinfer_keys_seen.add((group_key, return_probs))
            result: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor] | float] = (
                sample_grouped_strategies_orig(
                    group_key,
                    strategies,
                    logits,
                    group_logit_indices=group_logit_indices,
                    generator=generator,
                    return_probs=return_probs,
                    group_metadata=group_metadata,
                    seeds=seeds,
                )
            )
            return result

        # _grouped_sampler_cls is a class; point the instance at a subclass
        # that overrides the callable, rather than mutating the shared class.
        instrumented_cls = type(
            "InstrumentedFlashInferGroupedStrategySampler",
            (FlashInferGroupedStrategySampler,),
            {"sample_grouped_strategies": staticmethod(_sample_grouped_strategies)},
        )
        patch_ctx.setattr(sampler, "_grouped_sampler_cls", instrumented_cls)

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
        min_p: Optional[torch.Tensor] = None

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
            seed: Optional[Union[int, torch.Tensor]] = None,
            offset: Optional[Union[int, torch.Tensor]] = None,
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

        def _mock_flashinfer_top_k_top_p_from_probs(
            probs: torch.Tensor,
            *,
            top_k: torch.Tensor,
            top_p: torch.Tensor,
            filter_apply_order: str,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
            seed: Optional[Union[int, torch.Tensor]] = None,
            offset: Optional[Union[int, torch.Tensor]] = None,
        ) -> torch.Tensor:
            # The min_p strategy terminates its renorm chain here, so the probs
            # recorded below already have min_p applied; min_p itself never
            # reaches a flashinfer kernel and thus cannot be captured as a param.
            # Patching this is not optional: unpatched, the real flashinfer
            # implementation delegates to the *patched* top_p_sampling_from_probs
            # with kwargs its mock does not accept.
            assert filter_apply_order == "top_k_first"
            assert deterministic
            assert not check_nan, "check_nan syncs"
            assert generator is sampler.get_generator(probs.device)
            nonlocal mock_sampling_log
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=probs[row_idx],
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=top_k[row_idx],
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

        patch_ctx.setattr(
            flashinfer.sampling,
            "top_k_top_p_sampling_from_probs",
            _mock_flashinfer_top_k_top_p_from_probs,
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
            seed: Optional[Union[int, torch.Tensor]] = None,
            offset: Optional[Union[int, torch.Tensor]] = None,
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
            seed: Optional[Union[int, torch.Tensor]] = None,
            offset: Optional[Union[int, torch.Tensor]] = None,
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

        def _mock_flashinfer_min_p(
            probs: torch.Tensor,
            min_p: torch.Tensor,
            *,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
            seed: Optional[Union[int, torch.Tensor]] = None,
            offset: Optional[Union[int, torch.Tensor]] = None,
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
                        min_p=min_p[row_idx],
                    ),
                )
                for row_idx in range(probs.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log), len(mock_sampling_log) + len(new_entries)
            )
            mock_sampling_log += new_entries
            return mock_tokens

        patch_ctx.setattr(flashinfer.sampling, "min_p_sampling_from_probs", _mock_flashinfer_min_p)

        def _mock_flashinfer_from_probs(
            probs: torch.Tensor,
            *,
            deterministic: bool,
            check_nan: bool,
            generator: torch.Generator,
            seed: Optional[Union[int, torch.Tensor]] = None,
            offset: Optional[Union[int, torch.Tensor]] = None,
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
        req_has_min_p = (
            log_entry.sampling_params.min_p is not None
            and log_entry.sampling_params.min_p.item() != 0
        )
        if req_has_top_k:
            assert req_params.top_k is not None
            assert log_entry.sampling_params.top_k is not None
            assert req_params.top_k == log_entry.sampling_params.top_k.item()
        if req_has_top_p:
            assert req_params.top_p is not None
            assert log_entry.sampling_params.top_p is not None
            assert np.allclose(req_params.top_p, log_entry.sampling_params.top_p.item())
        if req_has_min_p:
            assert req_params.min_p is not None
            assert log_entry.sampling_params.min_p is not None
            assert np.allclose(req_params.min_p, log_entry.sampling_params.min_p.item())
        # min_p also filters to a top-prefix subset, so it reuses the validation below.
        if req_has_top_k or req_has_top_p or req_has_min_p:
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
        test_expected_counts_ma = np.ma.MaskedArray(
            test_expected_counts.numpy(),
            mask=(test_expected_counts.numpy() == 0),
        )
        test_token_counts_ma = np.ma.MaskedArray(
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
        pvalue: np.ndarray | float
        if hasattr(test_result.pvalue, "mask"):
            assert test_result.pvalue.mask  # pyright: ignore
            pvalue = test_result.pvalue.data  # pyright: ignore
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
                np.ma.MaskedArray(num_samples * prob_delta, mask=test_expected_counts_ma.mask)
                / test_expected_counts_ma.data
            )
            assert prob_delta_rel.max() < 0.05

    @pytest.mark.parametrize(
        (
            "max_draft_len",
            "sampling_params_list",
            "allow_zero_draft_len",
            "bypass_sampling",
            "vocab_size",
        ),
        [
            pytest.param(
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
                    f"FlashInfer"
                    f"-draft_len={0 if allow_zero_draft_len else 1}..{max_draft_len}"
                    f"-{params_label}"
                ),
            )
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (
                is_mixed,
                max_draft_len,
                allow_zero_draft_len,
                _build_test_cases,
                vocab_size,
            ) in product(
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
                model_outputs=model_outputs,
                sampling_params_list=sampling_params_list,
                seq_slot_assignment=seq_slot_assignment,
                vocab_size=vocab_size,
                max_draft_len=max_draft_len,
                draft_lens=draft_lens,
                monkeypatch=monkeypatch,
            )

            num_samples = 5000 if not (bypass_sampling or is_warmup) else 1

            # filled when bypass_sampling=True
            mock_sampling_log: Optional[list[TestBatchedSampling._MockSamplingLogEntry]] = None

            with monkeypatch.context() as patch_ctx:
                self._inject_batching_check(patch_ctx, sampler=sampler)
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
                        monkeypatch=monkeypatch,
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

        run_test_with_warmup(
            _uut_provider,
            max_sync_s=None,  # NB: assert_no_cuda_sync called in TestBatchedSampler._sample
        )

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
            total_seq_slots = rng.integers(max_seq_slots // 2, max_seq_slots).item()
            start = 0 if include_first else rng.integers(margin).item()
            end = total_seq_slots - (0 if include_last else rng.integers(margin).item())
            for dense in [False, True]:
                if dense:
                    seq_slots = list(range(start, end))
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
            "max_draft_len",
            "allow_zero_draft_len",
            "vocab_size",
            "seq_slot_assignment",
            "ordered",
        ),
        [
            pytest.param(
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
            total_steps = cast(int, req_num_steps.sum().item())

            new_tokens_cuda = torch.testing.make_tensor(
                (max_draft_len + 1, total_seq_slots, 1),
                device="cuda",
                dtype=torch.int32,
            )
            new_tokens_cuda_snapshot = new_tokens_cuda.clone()

            batch_req_indices = torch.arange(0, len(seq_slots), dtype=torch.int32)
            if not ordered:
                batch_req_indices = batch_req_indices[torch.randperm(batch_req_indices.numel())]

            first_token = rng.integers(123456).item()
            batch_next_tokens_cuda_int = torch.arange(
                first_token, first_token + total_steps, dtype=torch.int32, device="cuda"
            ).unsqueeze(1)  # Add a dimension for beam width

            batched_sampling_result = _BatchedSamplingResult(
                req_indices=batch_req_indices.clone(),
                next_tokens_cuda_int=batch_next_tokens_cuda_int.clone(),
            )
            seq_slots_tensor_snapshot = seq_slots_tensor.clone()

            @dataclass
            class UutResult:
                new_tokens_host: torch.Tensor

            @dataclass
            class UutResultWrapper:
                result: Optional[UutResult] = None

            res = UutResultWrapper()
            # Precomputed outside the no-sync region (mirrors the production
            # resident device copy of seq_slots).
            seq_slots_tensor_cuda = (
                seq_slots_tensor.to(torch.int64).pin_memory().to("cuda", non_blocking=True)
            )

            def _uut(res=res):
                new_tokens_host = sampler._unbatch_sampling_results(
                    batched_sampling_result=batched_sampling_result,
                    new_tokens_cuda=new_tokens_cuda,
                    req_num_generated_tokens=req_num_steps,
                    seq_slots=seq_slots_tensor,
                    seq_slots_cuda=seq_slots_tensor_cuda,
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
                batch_next_tokens_cuda_int, batched_sampling_result.next_tokens_cuda_int
            )
            torch.testing.assert_close(batch_req_indices, batched_sampling_result.req_indices)
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

        run_test_with_warmup(_uut_provider, max_sync_s=0.2)


class TestRequestSeed:
    """Functional guards for per-request ``SamplingParams.seed``.

    The property that matters is reproducibility that does not depend on batch
    composition: a seeded request must draw the same tokens whether it runs
    alone or beside unrelated requests, which is exactly what a single
    batch-wide generator cannot provide.
    """

    VOCAB_SIZE = 128
    NUM_STEPS = 8

    @staticmethod
    def _sampling_params(seed: Optional[int]) -> SamplingParams:
        # Temperature+top_k keeps sampling stochastic, so matching tokens across
        # runs indicate the seed is being honored rather than coincidence.
        return SamplingParams(temperature=1.0, top_k=64, seed=seed)

    def _run(
        self,
        sampling_params_list: list[SamplingParams],
        *,
        logits: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> torch.Tensor:
        """Sample ``NUM_STEPS`` steps; returns tokens indexed by [step, seq slot]."""
        harness = TestBatchedSampling()
        seq_slot_assignment = (list(range(len(sampling_params_list))), len(sampling_params_list))
        scheduled_requests = harness._build_mock_requests(
            sampling_params_list=sampling_params_list,
            seq_slot_assignment=seq_slot_assignment,
            draft_lens=[0] * len(sampling_params_list),
        )
        sampler = TorchSampler(
            TorchSampler.Args(
                max_seq_len=321,
                max_draft_len=42,
                max_beam_width=1,
                max_num_sequences=len(sampling_params_list),
                max_total_draft_tokens=0,
                disable_overlap_scheduler=False,
            )
        )
        return harness._sample(
            sampler,
            scheduled_requests,
            {"logits": logits},
            num_repeats=self.NUM_STEPS,
            monkeypatch=monkeypatch,
        )

    def _logits(self, num_requests: int) -> torch.Tensor:
        return torch.testing.make_tensor(
            (num_requests, self.VOCAB_SIZE), dtype=torch.float32, device="cuda"
        )

    def test_seed_is_independent_of_batch_composition(self, monkeypatch: pytest.MonkeyPatch):
        """The core guarantee: batching must not perturb a seeded stream."""
        logits = self._logits(3)
        seeded = self._sampling_params(1234)

        alone = self._run([seeded], logits=logits[:1], monkeypatch=monkeypatch)

        # Same seeded request, now in slot 0 of a batch whose other members
        # draw from the same strategy group and would advance a shared
        # generator's state.
        batched = self._run(
            [seeded, self._sampling_params(None), self._sampling_params(999)],
            logits=logits,
            monkeypatch=monkeypatch,
        )

        torch.testing.assert_close(alone[:, 0], batched[:, 0])

        # Negative control, disabled until FlashInfer honors per-row seeds.
        #
        # The assertion above passes even if the seed path is entirely inert,
        # because slot 0 is the one row whose seed is read either way. This
        # check would catch that -- but flashinfer-python 0.6.15 reads only
        # seed[0]/offset[0] for the whole call and distinguishes rows by
        # blockIdx.x, so rows 1..N sample from row 0's seed and this assertion
        # fails for reasons outside this code. Re-enable once the pinned
        # FlashInfer supports per-row seeds -- tracked upstream in
        # https://github.com/flashinfer-ai/flashinfer/pull/2345 (note it lands
        # the feature as generator=(seed_arr, offset_arr), so the sampler call
        # sites change with it).
        #
        # assert not torch.equal(batched[:, 0], batched[:, 2])

    def test_draft_batch_does_not_disturb_target_seed_state(self):
        """Draft slots come from a different SeqSlotManager over the same range.

        Observing a draft batch must not look like a change of occupant for the
        target request holding that slot number, which would reset its offset
        and make it replay part of its stream.
        """
        manager = _SeedManager(max_num_sequences=4, global_seed=42)

        target = cast(
            LlmRequest,
            SimpleNamespace(
                py_seq_slot=0,
                py_request_id=100,
                py_is_draft=False,
                sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            ),
        )
        manager.observe([target])
        manager.advance([0, 0, 0])
        assert manager.any_seeded is False  # target carries no user seed
        offset_before = manager._offsets[0].item()

        # A draft request lands on slot 0, owned by `target` above.
        draft = cast(
            LlmRequest,
            SimpleNamespace(
                py_seq_slot=0,
                py_request_id=200,
                py_is_draft=True,
                sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            ),
        )
        manager.observe([draft])
        assert manager.any_seeded is False  # draft never uses the per-row path
        assert manager._offsets[0].item() == offset_before
        assert manager._slot_owner[0] == 100  # still the target request

        # The target is unchanged when it comes back around.
        manager.observe([target])
        assert manager._offsets[0].item() == offset_before

    def test_multi_row_offsets_do_not_overlap(self):
        """Speculative decoding draws several rows per request per step.

        Those rows must be assigned distinct stretches of the request's stream,
        and the next step must resume past all of them -- otherwise a request
        would replay the same random numbers across steps.

        This asserts the offsets ``_SeedManager`` produces, not what the kernel
        does with them: the pinned flashinfer reads only ``offset[0]``, so the
        per-row values are not yet honored downstream.
        """
        manager = _SeedManager(max_num_sequences=4, global_seed=42)
        manager._seeds[0] = 1234
        manager._seeds[1] = 999
        manager._any_seeded = True

        rows = [0, 0, 0, 1]  # slot 0 draws 3 tokens this step, slot 1 draws 1
        device = torch.device("cpu")

        first = manager.make_row_seeds(rows, device=device)
        assert first.seed.tolist() == [1234, 1234, 1234, 999]

        manager.advance(rows)
        second = manager.make_row_seeds(rows, device=device)

        # Each row reserves a stretch of the stream; the invariant is that no
        # two stretches overlap, within a step or across steps. Asserted as a
        # property rather than against OFFSET_STRIDE, so that a stride too small
        # for the kernel's per-row consumption fails here instead of silently
        # rescaling the expected values.
        #
        # flashinfer 0.6.15 reserves 32 offset units per row for the top-k/top-p
        # rejection samplers, so a stride below that would replay random values.
        # Only within a slot: distinct slots carry distinct seeds, so they are
        # independent streams and may legitimately share offsets.
        assert _SeedManager.OFFSET_STRIDE >= 32
        per_slot: dict[int, set[int]] = {}
        for offsets in (first.offset.tolist(), second.offset.tolist()):
            for slot, off in zip(rows, offsets):
                stretch = range(off, off + _SeedManager.OFFSET_STRIDE)
                used = per_slot.setdefault(slot, set())
                assert used.isdisjoint(stretch), (
                    f"slot {slot} reuses offsets {off}..{off + _SeedManager.OFFSET_STRIDE - 1}; "
                    "the stream is replayed"
                )
                used.update(stretch)


class TestTopPDecay:
    """Minimal functional guards for Top-P Decay in TorchSampler.

    Covers strategy routing, the post-sample runtime update (parity with the
    C++ computeToppDecay recurrence; cases ported from
    topPSamplingLayerTest.cpp), and per-request rejection of unsupported
    combinations.
    """

    VOCAB_SIZE = 1000

    @staticmethod
    def _params(**kw) -> UtilsSamplingParams:
        base = dict(temperature=None, top_p=None, top_k=None, use_beam_search=False)
        base.update(kw)
        return UtilsSamplingParams(**base)

    @staticmethod
    def _make_sampler(*, max_draft_len=0):
        return TorchSampler(
            TorchSampler.Args(
                max_seq_len=128,
                max_draft_len=max_draft_len,
                max_num_sequences=8,
                max_beam_width=1,
                max_total_draft_tokens=max_draft_len,
                disable_overlap_scheduler=True,
            )
        )

    def test_strategy_routing(self):
        # Active decay (set and < 1.0) forces a top-p-capable strategy even for
        # an otherwise-greedy request (initial top-p defaults to 1.0), so the
        # decayed runtime value can take effect on later steps.
        s = resolve_sampling_strategy(self._params(top_p_decay=0.5), vocab_size=self.VOCAB_SIZE)
        assert s[0] == "top_p" and s[1] == pytest.approx(1.0)
        s = resolve_sampling_strategy(
            self._params(top_k=50, top_p=0.9, top_p_decay=0.8), vocab_size=self.VOCAB_SIZE
        )
        assert s[0] == "top_k_top_p"
        # decay == 1.0 (the C++ default) is a no-op and does not activate...
        s = resolve_sampling_strategy(self._params(top_p_decay=1.0), vocab_size=self.VOCAB_SIZE)
        assert s is GREEDY
        # ...and an explicit greedy control wins over an active decay.
        s = resolve_sampling_strategy(
            self._params(top_p_decay=0.5, top_k=1), vocab_size=self.VOCAB_SIZE
        )
        assert s is GREEDY
        # min_p wins the strategy choice, but the request keeps carrying top_p,
        # so decay stays applicable (see test_decay_metadata_dispatch).
        s = resolve_sampling_strategy(
            self._params(min_p=0.1, top_p=0.9, top_p_decay=0.8), vocab_size=self.VOCAB_SIZE
        )
        assert s[0] == "min_p"

    # Every strategy a decay-active request can resolve to carries a per-row
    # top-p, so all of them must be offered the decay metadata. A strategy
    # missing from the dispatch silently drops decay: the request is still
    # admitted and its runtime top-p still decays, but sampling keeps reading
    # the static initial value.
    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(dict(top_p_decay=0.8), id="top_p"),
            pytest.param(dict(top_k=50, top_p=0.9, top_p_decay=0.8), id="top_k_top_p"),
            pytest.param(dict(min_p=0.1, top_p=0.9, top_p_decay=0.8), id="min_p"),
        ],
    )
    def test_decay_metadata_dispatch(self, params):
        strategy = resolve_sampling_strategy(self._params(**params), vocab_size=self.VOCAB_SIZE)
        group_key = FlashInferGroupedStrategySampler.strategy_grouping_key(strategy)
        assert (
            FlashInferGroupedStrategySampler.get_metadata_type_for_group(group_key)
            is TopPDecayMetadata
        )

    # Companion to the dispatch test: the metadata must not just be handed over
    # but actually override the per-row top-p. Logits are chosen so the static
    # top_p=1.0 leaves every token samplable (min_p=0.1 keeps them all too),
    # while the decayed runtime top-p of 0.3 is below the argmax's own
    # probability and collapses the nucleus onto it.
    @pytest.mark.parametrize(
        "strategy",
        [
            pytest.param(("top_p", 1.0, 1.0), id="top_p"),
            pytest.param(("top_k_top_p", 5, 1.0, 1.0), id="top_k_top_p"),
            pytest.param(("min_p", 0, 1.0, 0.1, 1.0), id="min_p"),
        ],
    )
    @pytest.mark.parametrize("return_probs", [True, False], ids=["with_probs", "sample_only"])
    def test_decay_override_reaches_sampling(self, strategy, return_probs):
        num_rows, vocab, decayed_top_p = 64, 5, 0.3
        logits = torch.zeros(num_rows, vocab, device="cuda")
        logits[:, 0] = 1.0
        argmax = 0

        def run(is_decay_slot: bool) -> set[int]:
            metadata = TopPDecayMetadata(
                # All rows share slot 0, so a single store entry gates them all.
                slots=torch.zeros(num_rows, dtype=torch.int64, device="cuda"),
                runtime_top_p=torch.tensor([decayed_top_p], dtype=torch.float32, device="cuda"),
                is_decay_slot=torch.tensor([is_decay_slot], dtype=torch.bool, device="cuda"),
            )
            tokens, _, _ = FlashInferGroupedStrategySampler.sample_grouped_strategies(
                FlashInferGroupedStrategySampler.strategy_grouping_key(strategy),
                [cast(Strategy, strategy)] * num_rows,
                logits,
                generator=torch.Generator(device="cuda").manual_seed(0),
                return_probs=return_probs,
                group_metadata=metadata,
            )
            return set(tokens.flatten().tolist())

        # Gate off: the static top-p applies and sampling spreads over the vocab.
        assert len(run(is_decay_slot=False)) > 1
        # Gate on: the decayed runtime top-p replaces it and only the argmax survives.
        assert run(is_decay_slot=True) == {argmax}

    def test_runtime_update_parity(self):
        # Post-sample update parity with the C++ computeToppDecay recurrence
        # (a negative reset_id never matches, since token ids are non-negative):
        #   runtime = initial                    if token == reset_id
        #           = max(runtime * decay, min)  otherwise
        sampler = self._make_sampler()
        store = sampler._top_p_decay.store
        configs = [
            dict(initial=0.8, decay=0.3, top_p_min=0.5, reset_id=2),  # decay, then reset
            dict(initial=0.2, decay=0.9, top_p_min=0.1, reset_id=-1),  # plain decay, floored
            dict(initial=0.3, decay=0.5, top_p_min=0.6, reset_id=-1),  # min > initial: rises
        ]
        token_steps = [[1, 2, 3], [9, 9, 9], [9, 9, 9]]
        slots = list(range(len(configs)))
        for slot, cfg in zip(slots, configs):
            sampler._top_p_decay._slots.add(slot)
            store.runtime_top_p_decay_cuda[slot] = cfg["initial"]
            store.initial_top_p_decay_cuda[slot] = cfg["initial"]
            store.top_p_decay_cuda[slot] = cfg["decay"]
            store.top_p_decay_min_cuda[slot] = cfg["top_p_min"]
            store.top_p_decay_reset_ids_cuda[slot] = cfg["reset_id"]
            store.is_top_p_decay_slot_cuda[slot] = True

        runtime = [cfg["initial"] for cfg in configs]
        slots_cuda = torch.tensor(slots, dtype=torch.int64, device="cuda")
        for step in range(3):
            for slot in slots:
                sampler.store.new_tokens[0, slot, 0] = token_steps[slot][step]
            sampler._top_p_decay.update_after_sample(
                step_tokens=sampler.store.new_tokens[0, :, 0], sampled_slots_cuda=slots_cuda
            )
            got = store.runtime_top_p_decay_cuda.cpu()
            for slot, cfg in zip(slots, configs):
                tok = token_steps[slot][step]
                if tok == cfg["reset_id"]:
                    runtime[slot] = cfg["initial"]
                else:
                    runtime[slot] = max(runtime[slot] * cfg["decay"], cfg["top_p_min"])
                assert got[slot].item() == pytest.approx(runtime[slot], abs=1e-6), (step, slot)

    @staticmethod
    def _mock_request(params: SamplingParams, *, draft_tokens=None):
        params._validate()
        req = SimpleNamespace(
            sampling_config=SamplingConfig(params._get_sampling_config()),
            is_context_init_state=False,
            py_sampling_strategy=None,
            py_draft_tokens=draft_tokens,
            # Read by the row_stride query in sampler_common; these tests are
            # single-beam, so the static admission width is 1.
            py_beam_width=1,
        )
        req.get_beam_width_by_iter = lambda for_next_iteration=False: 1
        return cast(LlmRequest, req)

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"top_p_decay": 1.5},
            {"top_p_decay": -0.5},
            {"top_p_decay": 0.0},
            {"top_p_min": 0.0},
            {"top_p_min": 1.5},
            {"top_p_reset_ids": -1},
        ],
    )
    def test_out_of_range_decay_params_rejected(self, bad_kwargs):
        # Out-of-range decay params raise (mirroring the executor::SamplingConfig
        # constructor's hard checks) instead of the former warn-and-default.
        with pytest.raises(ValueError):
            SamplingParams(**bad_kwargs)

    def test_reject_speculative_draft_tokens(self):
        # Decay + draft tokens through TorchSampler is rejected per-request at
        # admission (validate_request), so only the offending request fails.
        sampler = self._make_sampler(max_draft_len=4)
        with pytest.raises(ValueError, match="speculative"):
            sampler.validate_request(
                self._mock_request(SamplingParams(top_p=0.9, top_p_decay=0.5), draft_tokens=[1, 2])
            )
        # Same request without decay is accepted.
        sampler.validate_request(self._mock_request(SamplingParams(top_p=0.9), draft_tokens=[1, 2]))
