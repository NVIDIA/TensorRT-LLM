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

from itertools import product
from typing import Optional, cast

import pytest
from utils.util import force_ampere

from tensorrt_llm._torch.pyexecutor.sampler import (
    GREEDY,
    LlmRequest,
    TorchSampler,
    _request_strategy,
)
from tensorrt_llm.bindings import SamplingConfig
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
