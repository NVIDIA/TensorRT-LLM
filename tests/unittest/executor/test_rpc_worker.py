# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time

import pytest

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc_worker import RpcWorker
from tensorrt_llm.executor.utils import ErrorResponse
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from utils.llm_data import llm_models_root
# isort: on

model_path = llm_models_root() / "Qwen3/Qwen3-0.6B"
assert model_path.exists()


class TestRpcWorkerTP1:

    def setup_method(self):
        self.llm_args = TorchLlmArgs(
            model=model_path,
            tensor_parallel_size=1,
            backend='pytorch',
            enable_iter_perf_stats=True,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5, ),
        )
        # Create RpcWorker instance
        self.worker = RpcWorker(
            llm_args=self.llm_args,
            hf_model_dir=model_path,
        )
        # Initialize the engine
        self.worker.setup_engine()

    def teardown_method(self):
        # Clean up the worker
        self.worker.shutdown()

    def test_fetch_responses_async(self):
        """Test that fetch_responses_async can be called and returns a list."""
        # Submit a request first
        sampling_params = SamplingParams(max_tokens=10)
        request = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params)
        self.worker.submit(request)

        # Sleep a bit to let the request start processing
        time.sleep(0.5)

        # Fetch responses with a timeout to prevent hanging
        responses = asyncio.run(self.worker.fetch_responses_async(timeout=1.0))
        assert isinstance(responses, list)

    def test_fetch_stats_async(self):
        """Test that fetch_stats_async can be called and returns a list."""
        # Submit a request first to generate some stats
        sampling_params = SamplingParams(max_tokens=10)
        request = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params)
        self.worker.submit(request)

        # Sleep a bit to let the request start processing
        time.sleep(0.5)

        # Fetch stats
        stats = asyncio.run(self.worker.fetch_stats_async())
        assert isinstance(stats, list)

    def test_fetch_kv_cache_events_async(self):
        """Test that fetch_kv_cache_events_async can be called and returns a list."""
        # Submit a request first to generate some kv cache events
        sampling_params = SamplingParams(max_tokens=10)
        request = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params)
        self.worker.submit(request)

        # Sleep a bit to let the request start processing
        time.sleep(0.5)

        # Fetch kv cache events
        events = asyncio.run(self.worker.fetch_kv_cache_events_async())
        assert isinstance(events, list)

    def test_fetch_responses_after_event_loop_crash(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self.worker.engine
        start_iteration = engine._start_response_staging_iteration
        crash_message = "injected RPC executor event-loop crash"
        sampling_params = SamplingParams(max_tokens=10, ignore_eos=True)
        pending = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params).set_id(1)

        def crash_with_pending_request() -> None:
            if pending.id in self.worker._client_id_to_request_id:
                raise RuntimeError(crash_message)
            start_iteration()

        monkeypatch.setattr(engine, "_start_response_staging_iteration",
                            crash_with_pending_request)
        self.worker.submit(pending)

        deadline = time.monotonic() + 10
        responses = []
        while not responses and time.monotonic() < deadline:
            responses = self.worker.fetch_responses(timeout=0.1)
            if not responses:
                time.sleep(0.01)
        assert responses, "Timed out waiting for the pending request's crash response"
        assert responses == [
            ErrorResponse(pending.id,
                          f"Event loop terminated with error: {crash_message}",
                          -1)
        ]

        late = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                 sampling_params=sampling_params).set_id(2)
        self.worker.submit(late)

        assert self.worker.fetch_responses(timeout=0.1) == [
            ErrorResponse(late.id,
                          f"Event loop terminated with error: {crash_message}",
                          -1)
        ]
        assert not engine._event_loop_error_delivered.is_set()
