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
from typing import Any, Callable, ContextManager, Final, Generator, Optional, TypeVar, cast

import numpy as np
import pytest
import torch
from scipy.stats import power_divergence
from utils.util import UutProvider, assert_no_cuda_sync, force_ampere, run_test_with_warmup

import tensorrt_llm._torch.pyexecutor.sampler.sampler as sampler_module
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    get_draft_token_length,
)
from tensorrt_llm._torch.pyexecutor.sampler import (
    SampleStateTensorsHostTorch,
    SampleStateTorch,
    TorchSampler,
    _BatchedSamplingResult,
    _SeedManager,
)
from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
from tensorrt_llm._torch.pyexecutor.sampler.ops.custom import (
    UNSEEDED_OFFSET_BASE,
    fused_compute_probs_from_logits,
    fused_sample_from_logits_with_probs,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import min_p_renorm_probs
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import (
    SampleType,
    UtilsSamplingParams,
    _get_max_beam_width,
    _request_get_sampling_params,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import AsyncWorkerMixin
from tensorrt_llm._torch.pyexecutor.sampler.sampling_resolution import (
    RowSamplingParams,
    SamplingMode,
    SlotSamplingParams,
    request_row_params,
    resolve_row_params,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.sampling_params import LogprobMode, SamplingParams


@pytest.fixture
def host_wait_event_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    event_factory = torch.cuda.Event
    calls: list[dict[str, Any]] = []

    def create_event(*args: Any, **kwargs: Any) -> torch.cuda.Event:
        calls.append(dict(kwargs))
        return event_factory(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Event", create_event)
    return calls


class TestSamplerHostWaitEvents:
    def test_main_event_waits_for_device_result(
        self, host_wait_event_calls: list[dict[str, Any]]
    ) -> None:
        worker = AsyncWorkerMixin()
        worker._async_worker_init(False)
        values = torch.arange(16, dtype=torch.float32, device="cuda").mul_(2)
        event = worker._record_sampler_event()
        event.synchronize()
        torch.testing.assert_close(values.cpu(), torch.arange(16).float() * 2)
        assert host_wait_event_calls == [{"blocking": True}]

    def test_side_stream_copy_is_ready_after_sampler_wait(
        self, host_wait_event_calls: list[dict[str, Any]]
    ) -> None:
        worker = AsyncWorkerMixin()
        worker._async_worker_init(False)
        values = torch.arange(16, dtype=torch.float32, device="cuda").add_(7)
        with worker._make_side_stream_copier() as copier:
            host_values = copier.stage_copy_to_host(values)
        event = worker._record_sampler_event(side_stream_event=copier.event)
        event.synchronize()
        torch.testing.assert_close(host_values, torch.arange(16).float() + 7)
        assert host_wait_event_calls == [{"blocking": True}, {"blocking": True}]

    def test_async_copy_keeps_snapshot_until_future_completes(
        self, host_wait_event_calls: list[dict[str, Any]]
    ) -> None:
        worker = AsyncWorkerMixin()
        worker._async_worker_init(True)
        worker.async_worker_start()
        try:
            values = torch.arange(16, dtype=torch.float32, device="cuda").add_(9)
            host_values = worker._copy_to_host(values)
            values.add_(100)
            event = worker._record_sampler_event()
            event.synchronize()
            assert event.worker_futures
            for future in event.worker_futures:
                future.result(timeout=10)
            torch.testing.assert_close(host_values, torch.arange(16).float() + 9)
            assert host_wait_event_calls == [{"blocking": True}, {"blocking": True}]
        finally:
            worker.async_worker_stop()


class TestSetupSamplerStepRequestSelection:
    @staticmethod
    def _make_request(
        *,
        is_attention_dp_dummy: bool = False,
        is_finished: bool = False,
    ) -> LlmRequest:
        return cast(
            LlmRequest,
            SimpleNamespace(
                is_attention_dp_dummy=is_attention_dp_dummy,
                is_finished=is_finished,
            ),
        )

    def test_collect_new_requests_for_setup_includes_adp_dummy_generation_requests(self):
        scheduled_requests = ScheduledRequests()

        context_request = self._make_request()
        finished_context_request = self._make_request(is_finished=True)
        adp_dummy_generation_request = self._make_request(is_attention_dp_dummy=True)
        regular_generation_request = self._make_request()
        finished_adp_dummy_generation_request = self._make_request(
            is_attention_dp_dummy=True, is_finished=True
        )

        scheduled_requests.context_requests_last_chunk = [
            context_request,
            finished_context_request,
        ]
        scheduled_requests.generation_requests = [
            adp_dummy_generation_request,
            regular_generation_request,
            finished_adp_dummy_generation_request,
        ]

        collected = TorchSampler._collect_new_requests_for_setup(scheduled_requests)
        assert len(collected) == 3
        assert collected[0] is context_request
        assert collected[1] is adp_dummy_generation_request
        assert collected[2] is finished_adp_dummy_generation_request


@force_ampere
class TestRowParamsResolution:
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
        request.py_beam_width = 1
        return cast(LlmRequest, request)

    def _resolve(self, params: SamplingParams) -> RowSamplingParams:
        """The request's row params; a top_k at or above the vocab size reads as
        disabled (0), as the fused kernel treats it."""
        self._check_params(params)
        row = request_row_params(self._build_mock_llm_request(params))
        return row._replace(top_k=0) if row.top_k >= self.VOCAB_SIZE else row

    @staticmethod
    def _sampling(
        *, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0
    ) -> Any:
        return pytest.approx(
            RowSamplingParams(SamplingMode.SAMPLING, temperature, top_k, top_p, min_p)
        )

    def _assert_greedy(self, params: SamplingParams) -> None:
        assert self._resolve(params) == RowSamplingParams(SamplingMode.GREEDY, 1.0, 0, 1.0, 0.0)

    def test_defaults(self):
        # NB: The code in resolve_row_params relies on the default values below.
        default_params = SamplingParams()
        assert default_params.top_k is None
        assert default_params.top_p is None
        assert default_params.temperature is None

    def test_defaults_config(self):
        # NB: The code in resolve_row_params relies on the default values below.
        default_config = SamplingParams()._get_sampling_config()
        assert default_config.top_k is None
        assert default_config.top_p is None
        assert default_config.temperature is None

    def test_defaults_request(self):
        # NB: The code in resolve_row_params relies on the default values below.
        request = self._build_mock_llm_request(SamplingParams())
        default_config = request.sampling_config
        assert default_config.top_k is None
        assert default_config.top_p is None
        assert default_config.temperature is None

    def test_default_is_greedy(self):
        self._assert_greedy(SamplingParams())

    @pytest.mark.parametrize(
        "top_p, top_k",
        [
            pytest.param(top_p, top_k)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (top_k, top_p) in product(TOP_K_VALS, TOP_P_VALS)
        ],
    )
    def test_temperature_0_is_greedy(self, top_p: Optional[float], top_k: Optional[int]):
        self._assert_greedy(SamplingParams(temperature=0, top_p=top_p, top_k=top_k))

    @pytest.mark.parametrize(
        "temperature, top_k",
        [
            pytest.param(temperature, top_k)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_k) in product(TEMPERATURE_VALS, TOP_K_VALS)
        ],
    )
    def test_top_p_0_is_greedy(self, temperature: Optional[float], top_k: Optional[int]):
        self._assert_greedy(SamplingParams(top_p=0, temperature=temperature, top_k=top_k))

    @pytest.mark.parametrize(
        "temperature, top_p",
        [
            pytest.param(temperature, top_p)
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (temperature, top_p) in product(TEMPERATURE_VALS, TOP_P_VALS)
        ],
    )
    def test_top_k_1_is_greedy(self, temperature: Optional[float], top_p: Optional[float]):
        self._assert_greedy(SamplingParams(top_p=top_p, temperature=temperature, top_k=1))

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
        assert self._resolve(params) == self._sampling(temperature=temperature)

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
        assert self._resolve(params) == self._sampling(top_p=0.42)

    @pytest.mark.parametrize(
        "trivial_top_k",
        [
            pytest.param(top_k)
            for top_k in TOP_K_NEUTRAL_VALS  # https://stackoverflow.com/a/75421799
        ],
    )
    def test_top_p_with_temperature(self, trivial_top_k: Optional[int]):
        params = SamplingParams(top_p=0.42, temperature=0.9, top_k=trivial_top_k)
        assert self._resolve(params) == self._sampling(top_p=0.42, temperature=0.9)

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
        assert self._resolve(params) == self._sampling(top_k=42)

    @pytest.mark.parametrize(
        "trivial_top_p",
        [
            pytest.param(top_p)
            for top_p in TOP_P_NEUTRAL_VALS  # https://stackoverflow.com/a/75421799
        ],
    )
    def test_top_k_with_temperature(self, trivial_top_p: Optional[float]):
        params = SamplingParams(top_k=42, temperature=0.9, top_p=trivial_top_p)
        assert self._resolve(params) == self._sampling(top_k=42, temperature=0.9)

    @pytest.mark.parametrize(
        "trivial_temperature",
        [
            pytest.param(temperature)
            for temperature in TEMPERATURE_NEUTRAL_VALS  # https://stackoverflow.com/a/75421799
        ],
    )
    def test_top_k_top_p(self, trivial_temperature: Optional[float]):
        params = SamplingParams(top_k=42, top_p=0.7, temperature=trivial_temperature)
        assert self._resolve(params) == self._sampling(top_k=42, top_p=0.7)

    def test_top_k_top_p_with_temperature(self):
        params = SamplingParams(top_k=42, top_p=0.7, temperature=0.9)
        assert self._resolve(params) == self._sampling(top_k=42, top_p=0.7, temperature=0.9)

    # --- min_p: composes with any subset of temperature/top_k/top_p ---

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
        assert self._resolve(params) == self._sampling(min_p=0.1)

    def test_min_p_with_temperature(self):
        params = SamplingParams(min_p=0.1, temperature=0.8)
        assert self._resolve(params) == self._sampling(min_p=0.1, temperature=0.8)

    def test_min_p_with_top_k_top_p(self):
        params = SamplingParams(min_p=0.1, top_k=42, top_p=0.7, temperature=0.8)
        assert self._resolve(params) == self._sampling(
            min_p=0.1, top_k=42, top_p=0.7, temperature=0.8
        )

    def test_min_p_0_is_disabled(self):
        params = SamplingParams(min_p=0.0, temperature=0.7)
        assert self._resolve(params) == self._sampling(temperature=0.7)

    def test_min_p_1_is_greedy(self):
        # min_p == 1 keeps only the row max, i.e. an explicit greedy control
        # (like top_p == 0), so it must not reach the sampling path.
        self._assert_greedy(SamplingParams(min_p=1.0, temperature=0.7))

    @pytest.mark.parametrize(
        "greedy_kwargs",
        [
            pytest.param({"top_k": 1}, id="top_k_1"),
            pytest.param({"temperature": 0}, id="temperature_0"),
        ],
    )
    def test_min_p_greedy_triggers_win(self, greedy_kwargs: dict[str, Any]):
        # An explicit greedy trigger collapses to a single token even with min_p.
        self._assert_greedy(SamplingParams(min_p=0.1, **greedy_kwargs))

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
        assert self._resolve(SamplingParams(top_k=top_k, top_p=top_p)) == self._sampling()

    def test_beam_search(self):
        row = request_row_params(
            self._build_mock_llm_request(SamplingParams(n=2, use_beam_search=True))
        )
        assert row.mode is SamplingMode.BEAM


@force_ampere
class TestSlotSamplingParams:
    """Admission writes each request's resolved params into its slot."""

    NUM_SLOTS = 4  # the last one is the scratch row

    @staticmethod
    def _request(params: SamplingParams, slot: int, **kwargs: Any) -> LlmRequest:
        return LlmRequest(
            request_id=slot,
            max_new_tokens=4,
            input_tokens=[1],
            sampling_config=SamplingConfig(params._get_sampling_config()),
            seq_slot=slot,
            is_streaming=False,
            **kwargs,
        )

    def _admit(self, store: SlotSamplingParams, requests: list[LlmRequest]) -> None:
        slots = torch.tensor([r.py_seq_slot for r in requests], dtype=torch.int64, device="cuda")
        store.admit(requests, slots)

    @staticmethod
    def _gathered(store: SlotSamplingParams, slots: list[int]) -> list[tuple[float, ...]]:
        temperature, top_k, top_p, min_p = store.gather(
            torch.tensor(slots, dtype=torch.int64, device="cuda")
        )
        return list(zip(*(t.tolist() for t in (temperature, top_k, top_p, min_p))))

    def test_unadmitted_slots_are_neutral(self):
        store = SlotSamplingParams(self.NUM_SLOTS, device="cuda")
        assert store.modes == [SamplingMode.GREEDY] * self.NUM_SLOTS
        assert self._gathered(store, list(range(self.NUM_SLOTS))) == [(1.0, 0, 1.0, 0.0)] * 4

    def test_admit_and_slot_reuse(self):
        store = SlotSamplingParams(self.NUM_SLOTS, device="cuda")
        self._admit(
            store,
            [
                self._request(SamplingParams(temperature=0.5, top_k=7), 0),
                self._request(SamplingParams(), 1),
                self._request(SamplingParams(top_p=0.5, min_p=0.25), 2),
            ],
        )
        assert store.modes[:3] == [
            SamplingMode.SAMPLING,
            SamplingMode.GREEDY,
            SamplingMode.SAMPLING,
        ]
        assert self._gathered(store, [2, 0, 0, 1, 3]) == [
            (1.0, 0, 0.5, 0.25),
            (0.5, 7, 1.0, 0.0),
            (0.5, 7, 1.0, 0.0),
            (1.0, 0, 1.0, 0.0),
            (1.0, 0, 1.0, 0.0),  # scratch row
        ]

        # A new request on a slot replaces everything its previous occupant set.
        self._admit(store, [self._request(SamplingParams(), 0)])
        assert store.modes[0] is SamplingMode.GREEDY
        assert self._gathered(store, [0, 2]) == [(1.0, 0, 1.0, 0.0), (1.0, 0, 0.5, 0.25)]

    def test_admit_records_logprob_needs(self):
        store = SlotSamplingParams(self.NUM_SLOTS, device="cuda")
        params = SamplingParams(temperature=0.5)
        requests = [
            self._request(params, 0),
            self._request(params, 1, return_log_probs=True, logprobs_mode=LogprobMode.PROCESSED),
            self._request(params, 2, return_log_probs=True, logprobs_mode=LogprobMode.RAW),
        ]
        self._admit(store, requests)
        assert store.need_processed_logprobs[:3] == [False, True, False]
        assert store.need_raw_logprobs[:3] == [False, False, True]


@force_ampere
def test_stochastic_requests_share_one_group():
    """Every stochastic request lands in one group, whatever filters it uses, so
    a batch mixing them is sampled in a single fused call."""
    params_list = [
        SamplingParams(temperature=0.8),
        SamplingParams(top_k=50, temperature=0.8),
        SamplingParams(),
        SamplingParams(top_p=0.9, temperature=0.8),
        SamplingParams(top_k=50, top_p=0.9, temperature=0.8),
        SamplingParams(min_p=0.05, temperature=0.8),
    ]
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=16,
            max_draft_len=0,
            max_num_sequences=len(params_list),
            max_beam_width=1,
            max_total_draft_tokens=0,
        )
    )
    requests = TestSlotSamplingParams._request
    batch = [requests(params, slot) for slot, params in enumerate(params_list)]
    sampler._slot_params.admit(batch, torch.arange(len(batch), dtype=torch.int64, device="cuda"))
    groups = sampler._group_requests(batch, list(range(len(batch))))
    assert [(g.mode, g.req_indices) for g in groups] == [
        (SamplingMode.SAMPLING, [0, 1, 3, 4, 5]),
        (SamplingMode.GREEDY, [2]),
    ]


@force_ampere
def test_grouping_rejects_a_request_not_admitted_to_its_slot():
    """A request whose slot was last admitted for another request fails loudly
    instead of sampling with that request's params."""
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=16,
            max_draft_len=0,
            max_num_sequences=1,
            max_beam_width=1,
            max_total_draft_tokens=0,
        )
    )
    make_request = TestSlotSamplingParams._request
    admitted = make_request(SamplingParams(temperature=0.8), 0)
    sampler._slot_params.admit([admitted], torch.tensor([0], dtype=torch.int64, device="cuda"))
    sampler._group_requests([admitted], [0])

    unadmitted = make_request(SamplingParams(), 0)
    unadmitted.py_request_id = admitted.py_request_id + 1
    with pytest.raises(AssertionError, match="last admitted for request"):
        sampler._group_requests([unadmitted], [0])


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
    # Bypassing __init__ skips the in-graph sampling state; the batch is sampled
    # eagerly here, which is what FULL means.
    sampler._current_sample_type = SampleType.FULL
    sampler._stable_greedy_request_ids = []
    sampler._stable_greedy_seq_slots = []
    sampler._stable_greedy_seq_slots_host = None
    sampler._stable_greedy_seq_slots_cuda = None
    sampler._slot_params = SlotSamplingParams(2, device="cpu")
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
        )
        for seq_slot in (0, 1)
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


@pytest.mark.parametrize(
    ("target_tokens", "finish_step", "force_num_accepted", "expected_tokens", "expected_accepted"),
    [
        pytest.param([5, 6, 7, 8], None, 0, [5, 6, 7, 8], 3, id="all_accepted"),
        pytest.param([9, 6, 7, 8], None, 0, [9], 0, id="first_rejected"),
        pytest.param([5, 6, 9, 8], None, 0, [5, 6, 9], 2, id="mid_chain_rejected"),
        pytest.param([5, 6, 7, 8], 1, 0, [5, 6], 1, id="finished_mid_chain"),
        pytest.param([9, 9, 9, 9], None, 2, [9, 9, 9], 2, id="force_num_accepted"),
    ],
)
def test_draft_tokens_accepted_until_first_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    target_tokens: list[int],
    finish_step: Optional[int],
    force_num_accepted: int,
    expected_tokens: list[int],
    expected_accepted: int,
):
    draft_tokens = [5, 6, 7]
    max_num_sequences = 4
    monkeypatch.setenv("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", str(force_num_accepted))
    sampler = TorchSampler(
        args=TorchSampler.Args(
            max_seq_len=32,
            max_draft_len=len(draft_tokens),
            max_total_draft_tokens=len(draft_tokens),
            max_num_sequences=max_num_sequences,
            max_beam_width=1,
            disable_overlap_scheduler=True,
        )
    )
    prompt = [1, 2]
    request = LlmRequest(
        request_id=0,
        seq_slot=2,
        input_tokens=prompt,
        max_new_tokens=16,
        sampling_config=SamplingConfig(),
        is_streaming=False,
        draft_tokens=draft_tokens,
    )
    scheduled_requests = ScheduledRequests()
    scheduled_requests.context_requests_last_chunk = [request]
    sampler.setup_sampler_step(scheduled_requests)

    new_tokens = torch.zeros((len(draft_tokens) + 1, max_num_sequences, 1), dtype=torch.int32)
    new_tokens[:, request.py_seq_slot, 0] = torch.tensor(target_tokens, dtype=torch.int32)
    finish_reasons = torch.full_like(new_tokens, FinishReason.NOT_FINISHED.value)
    if finish_step is not None:
        finish_reasons[finish_step, request.py_seq_slot, 0] = FinishReason.END_ID.value
    state = SampleStateTorch(
        requests=[request],
        device=None,
        host=SampleStateTensorsHostTorch(
            new_tokens=new_tokens,
            finish_reasons=finish_reasons,
            first_finish_reasons=None,
        ),
    )

    sampler.update_requests(state)

    assert request.get_tokens(0) == prompt + expected_tokens
    assert request.py_num_accepted_draft_tokens == expected_accepted
    assert request.py_rewind_len == len(draft_tokens) - expected_accepted
    assert request.is_finished == (finish_step is not None)


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

    # 13 and 17 are the two stop words; 99 is neither, and pins that the check
    # stays selective rather than finishing on any token.
    @pytest.mark.parametrize("new_token, expect_finished", [(13, True), (17, True), (99, False)])
    def test_single_step_greedy_honors_every_single_token_stop_word(
        self, new_token: int, expect_finished: bool
    ) -> None:
        """Each single-token stop word must stop generation, not just the first.

        The harmony / GPT-OSS serving path passes two of them (``<|return|>``
        and ``<|call|>``); matching only ``py_stop_words_list[0]`` let a tool
        call run on past ``<|call|>`` into fabricated content (nvbugs/6751484).
        """
        sampler = object.__new__(TorchSampler)
        sampler.max_seq_len = 20
        sampler._track_pending_steps = False
        request = LlmRequest(
            request_id=0,
            seq_slot=0,
            input_tokens=[2, 0],
            # Ample budget, so LENGTH cannot fire and mask a missed stop word.
            max_new_tokens=10,
            end_id=2,
            stop_words_list=[[13], [17]],
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        state = SampleStateTorch(
            requests=[request],
            device=None,
            host=SampleStateTensorsHostTorch(
                new_tokens=torch.tensor([new_token], dtype=torch.int32),
                finish_reasons=None,
                first_finish_reasons=None,
                single_step_greedy=True,
            ),
        )

        sampler.update_requests(state)

        assert request.is_finished == expect_finished
        if expect_finished:
            # Pins STOP_WORDS as the reason: the budget above rules out LENGTH,
            # and none of the parametrized tokens is ``end_id``, ruling out END_ID.
            assert not request.is_finished_due_to_length

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
                stop_words_list=stop_words_list,
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
            {self.request.max_new_tokens=}, {self.request.py_end_id=}, {self.request.py_stop_words_list=})"

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

                # Capture return value of write_finish_reasons for use after _uut() runs.
                write_finish_reasons_result: list[torch.Tensor] = []

                def _uut():
                    with extra_context() if extra_context is not None else nullcontext():
                        result = sampler._finish_reasons_handler.write_finish_reasons(
                            seq_slots_host=seq_slots_host,
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

        # One entry per combination of filters; beam search is tested in
        # test_beam_search.py instead.
        BASE_CASES = {
            "greedy": SamplingParams(),
            "temperature": SamplingParams(temperature=0.7),
            "top_p": SamplingParams(top_p=0.42, temperature=0.2),
            "top_k": SamplingParams(top_k=27, temperature=0.5),
            "top_k_top_p": SamplingParams(top_k=27, top_p=0.6, temperature=0.5),
            "min_p": SamplingParams(min_p=0.02, top_k=40, top_p=0.9, temperature=1.0),
        }

        test_cases = []

        if include_all or include_uniform:
            # Base cases (single-request batches)
            for strategy_name, params in BASE_CASES.items():
                if strategy_name == "greedy" and not allow_greedy:
                    continue
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
            for strategy_name, params in BASE_CASES.items():
                batch_size = rng.integers(low=1, high=max_batch_size)
                if strategy_name == "greedy" and not allow_greedy:
                    continue
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
            type_to_constrain = "top_k"
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
                for strategy_name, params in BASE_CASES.items():
                    sub_batch_size = rng.integers(low=2, high=max_sub_batch_size).item()
                    if strategy_name == "greedy" and not allow_greedy:
                        continue
                    if strategy_name == type_to_constrain and constraint_value is not None:
                        if isinstance(constraint_value, int):
                            sub_batch_size = constraint_value
                        else:
                            constrained_indices = (
                                len(mixed_params_list),
                                len(mixed_params_list) + sub_batch_size,
                            )
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
            # NB: non-zero draft len gives every request several rows.
            pytest.param(
                3,
                [3] * len(sampling_params_list),
                sampling_params_list,
                params_label,
                False,
                vocab_size,
                id=f"Fused-{params_label}",
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
        """Validate the probs the sampler draws from.

        The fused sampling call is made to return its probs (the distribution
        after applying temperature, top-p/k masking, etc.), which this test
        checks against the request's sampling params.
        """

        @contextmanager
        def _uut_provider(is_warmup: bool) -> Generator[Callable[[], None], None, None]:
            torch.manual_seed(42)  # torch.testing.make_tensor does not accept Generator

            with monkeypatch.context() as patch_ctx:
                recorded_probs = self._record_probs(patch_ctx, sampler=sampler)

                def _uut():
                    _ = self._sample(
                        sampler,
                        scheduled_requests=mock_requests,
                        model_outputs=model_outputs,
                        allow_sync=is_warmup,
                        monkeypatch=monkeypatch,
                    )

                yield _uut

            logit_offset = 0
            for req, draft_len in zip(mock_requests.all_requests(), draft_lens):
                probs = recorded_probs[cast(int, req.py_seq_slot)].cpu()
                assert probs.shape == (draft_len + 1, vocab_size)
                # NB: request_row_params tested in TestRowParamsResolution
                row = request_row_params(req)
                assert row.mode is SamplingMode.SAMPLING
                has_top_k = 0 < row.top_k < vocab_size
                has_top_p = row.top_p < 1
                has_min_p = row.min_p > 0

                steps = draft_len + 1

                req_logits = model_outputs["logits"][logit_offset : (logit_offset + steps), :].cpu()
                expected_probs_after_temperature = torch.softmax(
                    req_logits / row.temperature, dim=-1
                )

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

                if not (has_top_k or has_top_p or has_min_p):
                    torch.testing.assert_close(probs, expected_probs_after_temperature)
                else:
                    top_k = None
                    if has_top_k:
                        # Validate top-k
                        # Correct for possible zero probs in input
                        input_nnz = torch.count_nonzero(expected_probs_after_temperature, dim=-1)
                        top_k = torch.where(input_nnz < row.top_k, input_nnz, row.top_k)

                        nnz = torch.count_nonzero(probs, dim=-1)
                        if has_top_p or has_min_p:
                            # when top-k is followed by another filter, the result set is smaller
                            assert torch.le(nnz, top_k).all()
                        else:
                            torch.testing.assert_close(nnz, top_k)

                    if has_top_p and not has_min_p:
                        # Validate top-p
                        top_p = row.top_p

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
                            top_p,
                        ).all()
                        assert torch.lt(
                            probs_sorted_pre_norm.sum(dim=-1)
                            - probs_sorted_pre_norm_nz.amin(dim=-1),
                            top_p,
                        ).all()

                    if has_min_p:
                        # Renorm preserves the ratio, so every kept token satisfies
                        # prob >= min_p * max (holds with top_k/top_p also applied).
                        kept = probs != 0.0
                        ratio = probs / probs.amax(dim=-1, keepdim=True)
                        assert torch.all((ratio >= row.min_p - 1e-6)[kept])

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

    @staticmethod
    def _record_probs(
        patch_ctx: pytest.MonkeyPatch,
        *,
        sampler: TorchSampler,
    ) -> dict[int, torch.Tensor]:
        """Make the sampler's fused sampling calls also return probs, and record them.

        Returns a dict, filled as the sampler runs, from sequence slot to the probs
        of that slot's rows (one per step) in the latest call.
        """
        recorded: dict[int, torch.Tensor] = {}
        sample_fused_orig = sampler._sample_fused

        def _sample_fused(
            logits: torch.Tensor, slots_per_row: list[int], *, return_probs: bool
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
            tokens, probs, temperature = sample_fused_orig(logits, slots_per_row, return_probs=True)
            assert probs is not None
            # A request's rows are consecutive; slice them out without a device sync.
            start = 0
            for end in range(1, len(slots_per_row) + 1):
                if end == len(slots_per_row) or slots_per_row[end] != slots_per_row[start]:
                    recorded[slots_per_row[start]] = probs[start:end]
                    start = end
            return tokens, (probs if return_probs else None), temperature

        patch_ctx.setattr(sampler, "_sample_fused", _sample_fused)
        return recorded

    def _compute_probs(
        self,
        *,
        model_outputs: dict[str, torch.Tensor],
        sampling_params_list: list[SamplingParams],
        seq_slot_assignment: tuple[list[int], int],
        max_draft_len: int,
        draft_lens: list[int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[int, torch.Tensor]:
        """Compute the probs (PMFs) the sampler draws from, per sequence slot.

        Uses its own sampler and requests, so that the sampler under test keeps its
        RNG state. Greedy requests have no entry.

        Used by test_samples.
        """
        sampler_with_probs = self._build_sampler(
            max_draft_len=max_draft_len,
            seq_slot_assignment=seq_slot_assignment,
        )
        mock_requests_with_probs = self._build_mock_requests(
            sampling_params_list=sampling_params_list,
            seq_slot_assignment=seq_slot_assignment,
            draft_lens=draft_lens,
        )
        with monkeypatch.context() as patch_ctx:
            recorded_probs = self._record_probs(patch_ctx, sampler=sampler_with_probs)
            _ = self._sample(
                sampler_with_probs,
                scheduled_requests=mock_requests_with_probs,
                model_outputs=model_outputs,
                monkeypatch=monkeypatch,
            )
        return recorded_probs

    @staticmethod
    def _inject_batching_check(
        patch_ctx: pytest.MonkeyPatch,
        *,
        sampler: TorchSampler,
    ):
        """Setup interception of sample_async and the fused sampling call.

        Validates that every invocation of sample_async samples all stochastic
        requests in at most one fused call per probs variant, whatever their
        filters and parameter values.

        Used by test_samples.
        """
        fused_calls: list[bool] = []
        sample_fused_orig = sampler._sample_fused

        def _sample_fused(
            logits: torch.Tensor, slots_per_row: list[int], *, return_probs: bool
        ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
            assert return_probs not in fused_calls
            fused_calls.append(return_probs)
            tokens, probs, temperature = sample_fused_orig(
                logits, slots_per_row, return_probs=return_probs
            )
            return tokens, probs, temperature

        patch_ctx.setattr(sampler, "_sample_fused", _sample_fused)

        sample_async_orig = sampler.sample_async

        def _sample_async(
            scheduled_requests: ScheduledRequests,
            model_outputs: dict[str, torch.Tensor],
            num_context_logits_prefix_sum: list[int],
            resource_manager=None,
        ):
            fused_calls.clear()
            res = sample_async_orig(
                scheduled_requests,
                model_outputs,
                num_context_logits_prefix_sum,
                resource_manager,
            )
            all_greedy = all(
                request_row_params(req).mode is SamplingMode.GREEDY
                for req in scheduled_requests.all_requests()
            )
            assert bool(fused_calls) != all_greedy
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

        def _log_fused_rows(
            logits: torch.Tensor,
            temperatures: torch.Tensor,
            top_ks: torch.Tensor,
            top_ps: torch.Tensor,
            min_ps: torch.Tensor,
            seed: Optional[torch.Tensor],
            offset: Optional[torch.Tensor],
        ) -> torch.Tensor:
            # TorchSampler must hand every row its own RNG state, independent of
            # the row's position in the batch.
            assert seed is not None and offset is not None
            assert seed.numel() == offset.numel() == logits.size(0)
            nonlocal mock_sampling_log
            # Log the kernel's own tempered, unfiltered distribution, so that it
            # compares exactly against the filtered probs the kernel returns.
            probs = fused_compute_probs_from_logits(
                logits,
                temperatures,
                torch.zeros_like(top_ks),
                torch.ones_like(top_ps),
                torch.zeros_like(min_ps),
            )
            new_entries = [
                TestBatchedSampling._MockSamplingLogEntry(
                    probs=probs[row_idx],
                    sampling_params=TestBatchedSampling._TorchUtilsSamplingParams(
                        top_k=top_ks[row_idx],
                        top_p=top_ps[row_idx],
                        min_p=min_ps[row_idx],
                        temperature=None,
                    ),
                )
                for row_idx in range(logits.size(0))
            ]
            mock_tokens = torch.arange(
                len(mock_sampling_log),
                len(mock_sampling_log) + len(new_entries),
                dtype=torch.int32,
                device=logits.device,
            )
            mock_sampling_log += new_entries
            return mock_tokens

        def _mock_fused_sample_from_logits(
            logits: torch.Tensor,
            temperatures: torch.Tensor,
            top_ks: torch.Tensor,
            top_ps: torch.Tensor,
            min_ps: torch.Tensor,
            *,
            seed: Optional[torch.Tensor] = None,
            offset: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return _log_fused_rows(logits, temperatures, top_ks, top_ps, min_ps, seed, offset)

        patch_ctx.setattr(
            sampler_module, "fused_sample_from_logits", _mock_fused_sample_from_logits
        )

        def _mock_fused_sample_from_logits_with_probs(
            logits: torch.Tensor,
            temperatures: torch.Tensor,
            top_ks: torch.Tensor,
            top_ps: torch.Tensor,
            min_ps: torch.Tensor,
            *,
            seed: Optional[torch.Tensor] = None,
            offset: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # The probs feed logprobs, so they come from the real kernel; only the
            # tokens are replaced by log indices.
            _, probs = fused_sample_from_logits_with_probs(
                logits,
                temperatures,
                top_ks,
                top_ps,
                min_ps,
                seed=seed,
                offset=offset,
            )
            mock_tokens = _log_fused_rows(
                logits, temperatures, top_ks, top_ps, min_ps, seed, offset
            )
            return mock_tokens, probs

        patch_ctx.setattr(
            sampler_module,
            "fused_sample_from_logits_with_probs",
            _mock_fused_sample_from_logits_with_probs,
        )

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
        # The mocks log the tempered distribution, so no temperature remains here
        assert log_entry.sampling_params.temperature is None

        req_has_top_p = (
            log_entry.sampling_params.top_p is not None
            and log_entry.sampling_params.top_p.item() < 1
        )
        req_has_top_k = (
            log_entry.sampling_params.top_k is not None
            and 0 < log_entry.sampling_params.top_k.item() < vocab_size
        )
        req_has_min_p = (
            log_entry.sampling_params.min_p is not None
            and log_entry.sampling_params.min_p.item() > 0
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
                    f"Fused"
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
            probs_by_slot = self._compute_probs(
                model_outputs=model_outputs,
                sampling_params_list=sampling_params_list,
                seq_slot_assignment=seq_slot_assignment,
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
            for req_idx, (req, draft_len) in enumerate(
                zip(mock_requests.all_requests(), draft_lens)
            ):
                seq_slot = cast(int, req.py_seq_slot)
                if request_row_params(req).mode is SamplingMode.GREEDY:
                    # greedy is not sampled by the fused kernel
                    assert seq_slot not in probs_by_slot
                    req_logit_offset = sum(draft_len + 1 for draft_len in draft_lens[:req_idx])
                    req_logits = logits[req_logit_offset : (req_logit_offset + draft_len + 1)]
                    tokens_expected = torch.argmax(req_logits, dim=-1, keepdim=True)
                    tokens_sampled = new_tokens_repeats[: (draft_len + 1), seq_slot, :]
                    assert torch.all(tokens_expected.cpu() == tokens_sampled)
                    continue  # nothing else to check

                probs = probs_by_slot[seq_slot].cpu()
                assert probs.size(0) == draft_len + 1

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
        logits = self._logits(1).expand(3, -1).contiguous()
        seeded = self._sampling_params(1234)

        alone = self._run([seeded], logits=logits[:1], monkeypatch=monkeypatch)
        alone_other = self._run(
            [self._sampling_params(999)], logits=logits[2:], monkeypatch=monkeypatch
        )

        # The same seeded requests in a batch whose other members draw from the
        # same strategy group, one of them away from row 0.
        batched = self._run(
            [seeded, self._sampling_params(None), self._sampling_params(999)],
            logits=logits,
            monkeypatch=monkeypatch,
        )

        torch.testing.assert_close(alone[:, 0], batched[:, 0])
        torch.testing.assert_close(alone_other[:, 0], batched[:, 2])
        # Negative control: distinct seeds on identical logits draw distinct streams.
        assert not torch.equal(batched[:, 0], batched[:, 2])

    def test_seed_is_independent_of_strategy_mix(self, monkeypatch: pytest.MonkeyPatch):
        """Requests of other strategy kinds share the seeded request's group, so
        they must not perturb its stream either."""
        logits = self._logits(1).expand(4, -1).contiguous()
        seeded = self._sampling_params(1234)

        alone = self._run([seeded], logits=logits[:1], monkeypatch=monkeypatch)
        mixed = self._run(
            [
                SamplingParams(temperature=0.9, top_p=0.8),
                seeded,
                SamplingParams(temperature=1.1, min_p=0.05, seed=7),
                SamplingParams(temperature=0.7),
            ],
            logits=logits,
            monkeypatch=monkeypatch,
        )

        torch.testing.assert_close(alone[:, 0], mixed[:, 1])

    def test_seeded_row_position_does_not_matter(self, monkeypatch: pytest.MonkeyPatch):
        logits = self._logits(1).expand(3, -1).contiguous()
        seeded = self._sampling_params(1234)

        first = self._run(
            [seeded, self._sampling_params(None), self._sampling_params(999)],
            logits=logits,
            monkeypatch=monkeypatch,
        )
        last = self._run(
            [self._sampling_params(None), self._sampling_params(999), seeded],
            logits=logits,
            monkeypatch=monkeypatch,
        )

        torch.testing.assert_close(first[:, 0], last[:, 2])

    def test_unseeded_requests_do_not_replay_on_a_reused_slot(self):
        manager = _SeedManager(max_num_sequences=1, global_seed=42)
        device = torch.device("cpu")

        def unseeded(request_id: int) -> LlmRequest:
            return cast(
                LlmRequest,
                SimpleNamespace(
                    py_seq_slot=0,
                    py_request_id=request_id,
                    sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
                ),
            )

        manager.observe([unseeded(1)])
        first = manager.take_row_seeds([0], device=device)
        manager.observe([unseeded(2)])
        second = manager.take_row_seeds([0], device=device)

        assert first.seed.tolist() == second.seed.tolist() == [42]
        assert first.offset.item() != second.offset.item()

    def test_multi_row_offsets_do_not_overlap(self):
        """Speculative decoding draws several rows per request per step.

        Those rows must be assigned distinct offsets of the request's stream,
        and the next step must resume past all of them -- otherwise a request
        would replay the same random numbers across steps.
        """
        manager = _SeedManager(max_num_sequences=4, global_seed=42)
        manager._seeds[0] = 1234
        manager._seeds[1] = 999
        manager._slot_seeded[0] = manager._slot_seeded[1] = True

        rows = [0, 0, 0, 1]  # slot 0 draws 3 tokens this step, slot 1 draws 1
        device = torch.device("cpu")

        first = manager.take_row_seeds(rows, device=device)
        assert first.seed.tolist() == [1234, 1234, 1234, 999]
        second = manager.take_row_seeds(rows, device=device)

        # Distinct slots carry distinct seeds, so they are independent streams
        # and may share offsets; within a slot no offset may repeat.
        per_slot: dict[int, list[int]] = {}
        for offsets in (first.offset.tolist(), second.offset.tolist()):
            for slot, off in zip(rows, offsets):
                per_slot.setdefault(slot, []).append(off)
        assert per_slot == {0: [0, 1, 2, 3, 4, 5], 1: [0, 1]}

    def test_unseeded_rows_take_distinct_offsets(self):
        manager = _SeedManager(max_num_sequences=4, global_seed=42)
        device = torch.device("cpu")

        first = manager.take_row_seeds([0, 1, 2], device=device)
        second = manager.take_row_seeds([2, 3], device=device)

        assert first.seed.tolist() + second.seed.tolist() == [42] * 5
        assert first.offset.tolist() + second.offset.tolist() == [
            UNSEEDED_OFFSET_BASE + i for i in range(5)
        ]

    def test_unseeded_offsets_continue_across_mixed_batches(self):
        manager = _SeedManager(max_num_sequences=4, global_seed=42)
        manager._seeds[0] = 1234
        manager._slot_seeded[0] = True
        device = torch.device("cpu")

        before = manager.take_row_seeds([1], device=device)
        mixed = manager.take_row_seeds([0, 1], device=device)
        after = manager.take_row_seeds([1], device=device)

        assert mixed.seed.tolist() == [1234, 42]
        unseeded = before.offset.tolist() + mixed.offset.tolist()[1:] + after.offset.tolist()
        assert unseeded == [UNSEEDED_OFFSET_BASE + i for i in range(3)]

    def test_user_seed_equal_to_global_seed_does_not_share_unseeded_streams(self):
        manager = _SeedManager(max_num_sequences=4, global_seed=42)
        manager._seeds[0] = 42
        manager._slot_seeded[0] = True
        device = torch.device("cpu")

        for _ in range(4):
            seeds = manager.take_row_seeds([0, 1], device=device)
            assert seeds.seed.tolist() == [42, 42]
            seeded_offset, unseeded_offset = seeds.offset.tolist()
            assert seeded_offset < UNSEEDED_OFFSET_BASE <= unseeded_offset


class TestTopPDecay:
    """Minimal functional guards for Top-P Decay in TorchSampler.

    Covers resolution, the override of the sampled top-p, the post-sample
    runtime update (parity with the recurrence the former C++ computeToppDecay
    implemented), and per-request rejection of unsupported combinations.
    """

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

    def test_resolution(self):
        # Active decay (set and < 1.0) makes an otherwise-greedy request sample
        # (initial top-p defaults to 1.0), so the decayed runtime value can take
        # effect on later steps.
        def resolve(**kw: Any) -> RowSamplingParams:
            return resolve_row_params(
                temperature=kw.get("temperature"),
                top_k=kw.get("top_k"),
                top_p=kw.get("top_p"),
                min_p=kw.get("min_p"),
                top_p_decay=kw.get("top_p_decay"),
            )

        assert resolve(top_p_decay=0.5) == (SamplingMode.SAMPLING, 1.0, 0, 1.0, 0.0)
        assert resolve(top_k=50, top_p=0.9, top_p_decay=0.8) == pytest.approx(
            (SamplingMode.SAMPLING, 1.0, 50, 0.9, 0.0)
        )
        # decay == 1.0 (the C++ default) is a no-op and does not activate...
        assert resolve(top_p_decay=1.0).mode is SamplingMode.GREEDY
        # ...and an explicit greedy control wins over an active decay.
        assert resolve(top_p_decay=0.5, top_k=1).mode is SamplingMode.GREEDY

    # Every sampled row is offered the decayed top-p, whatever other filters it
    # uses. Logits are chosen so the static top_p=1.0 leaves every token
    # samplable (min_p=0.1 keeps them all too), while the decayed runtime top-p
    # of 0.3 is below the argmax's own probability and collapses the nucleus
    # onto it.
    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(dict(top_p=1.0), id="top_p"),
            pytest.param(dict(top_k=5, top_p=1.0), id="top_k_top_p"),
            pytest.param(dict(min_p=0.1), id="min_p"),
        ],
    )
    @pytest.mark.parametrize("return_probs", [True, False], ids=["with_probs", "sample_only"])
    def test_decay_override_reaches_sampling(self, params, return_probs):
        num_rows, vocab, decayed_top_p = 64, 5, 0.3
        logits = torch.zeros(num_rows, vocab, device="cuda")
        logits[:, 0] = 1.0
        argmax = 0
        slot, other_slot = 0, 1

        def run(is_decay_slot: bool) -> set[int]:
            sampler = self._make_sampler()
            sampler._slot_params.admit(
                [self._mock_request(SamplingParams(top_p_decay=0.8, **params), slot=slot)],
                torch.tensor([slot], dtype=torch.int64, device="cuda"),
            )
            # Some slot is always decay-active, so it is the device gate that
            # decides whether the override applies to this one.
            decay_slot = slot if is_decay_slot else other_slot
            sampler._top_p_decay._slots.add(decay_slot)
            store = sampler._top_p_decay.store
            store.runtime_top_p_decay_cuda[decay_slot] = decayed_top_p
            store.is_top_p_decay_slot_cuda[decay_slot] = True
            tokens, _, _ = sampler._sample_fused(
                logits, [slot] * num_rows, return_probs=return_probs
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
    def _mock_request(params: SamplingParams, *, draft_tokens=None, slot=None):
        params._validate()
        req = SimpleNamespace(
            sampling_config=SamplingConfig(params._get_sampling_config()),
            is_context_init_state=False,
            py_draft_tokens=draft_tokens,
            py_request_id=0,
            py_seq_slot=slot,
            is_dummy=False,
            py_return_log_probs=False,
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

    @pytest.mark.parametrize(
        "beam_width_array, expected_max_width",
        [([], 1), (None, 1), ([1], 1), ([1, 2], 2), ([3, 5, 7], 7)],
    )
    def test_beam_width_array_max_handles_empty(
        self, beam_width_array: list[int] | None, expected_max_width: int
    ) -> None:
        # An empty beam_width_array reaches the sampler: the executor's
        # checkBeamWidthArray bounds only the array's length, so [] passes
        # admission. _get_max_beam_width must fall back to beam_width instead of
        # reducing over the empty array, and must still take the array's maximum
        # when it has entries.
        #
        # Build the request through SamplingParams, the way production does.
        request = self._mock_request(SamplingParams(top_p=0.9, beam_width_array=beam_width_array))
        assert _get_max_beam_width(request) == expected_max_width
        # The same request must survive admission: top_p_decay is unset, but
        # TopPDecayHandler.validate_request resolves the sampling params -- and
        # with them the beam width -- before it checks whether decay is active.
        self._make_sampler().validate_request(request)
