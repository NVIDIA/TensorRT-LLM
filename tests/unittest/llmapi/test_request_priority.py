# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for request priority support in the LLM API.

Tests cover priority propagation through:
  - GenerationRequest storage
  - GenerationExecutor.generate_async -> GenerationRequest
  - BaseLLM.generate (scalar and list priority)
  - BaseLLM.generate_async -> executor.generate_async
  - BaseWorker._enqueue_request (default fallback to 0.5)
  - executor_request_to_llm_request (no override)
"""

from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY, GenerationRequest
from tensorrt_llm.sampling_params import SamplingParams

# ---------------------------------------------------------------------------
# GenerationRequest
# ---------------------------------------------------------------------------


class TestGenerationRequestPriority:
    def _make_request(self, **kwargs):
        return GenerationRequest(
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
            **kwargs,
        )

    def test_priority_defaults_to_half(self):
        req = self._make_request()
        assert req.priority == DEFAULT_REQUEST_PRIORITY

    def test_priority_stored_correctly(self):
        req = self._make_request(priority=0.8)
        assert req.priority == 0.8

    def test_priority_zero(self):
        req = self._make_request(priority=0.0)
        assert req.priority == 0.0

    def test_priority_one(self):
        req = self._make_request(priority=1.0)
        assert req.priority == 1.0

    def test_priority_midpoint(self):
        req = self._make_request(priority=DEFAULT_REQUEST_PRIORITY)
        assert req.priority == DEFAULT_REQUEST_PRIORITY

    @pytest.mark.parametrize("bad_priority", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
    def test_invalid_priority_raises(self, bad_priority):
        with pytest.raises(ValueError):
            self._make_request(priority=bad_priority)


# ---------------------------------------------------------------------------
# GenerationExecutor.generate_async  ->  GenerationRequest
# ---------------------------------------------------------------------------


class TestGenerationExecutorPriorityPropagation:
    """Verify that GenerationExecutor.generate_async builds a GenerationRequest.

    Checks that the correct priority is set and the request is passed to submit().
    """

    def _make_executor(self):
        """Return a concrete subclass of GenerationExecutor with a spy submit."""
        from tensorrt_llm.executor.executor import GenerationExecutor

        class _FakeExecutor(GenerationExecutor):
            def __init__(self):
                super().__init__()
                self.submitted = []

            def submit(self, request):
                self.submitted.append(request)
                result = MagicMock()
                result.request_id = 0
                return result

            def abort_request(self, request_id):
                pass

            def shutdown(self):
                pass

        return _FakeExecutor()

    def test_priority_propagated_to_request(self):
        executor = self._make_executor()
        executor.generate_async(
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=5),
            priority=0.9,
        )
        assert len(executor.submitted) == 1
        assert executor.submitted[0].priority == 0.9

    def test_no_priority_defaults_to_half(self):
        executor = self._make_executor()
        executor.generate_async(
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=5),
        )
        assert executor.submitted[0].priority == DEFAULT_REQUEST_PRIORITY

    def test_priority_zero_propagated(self):
        executor = self._make_executor()
        executor.generate_async(
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=5),
            priority=0.0,
        )
        assert executor.submitted[0].priority == 0.0


# ---------------------------------------------------------------------------
# BaseWorker._enqueue_request  -  priority default of 0.5
# ---------------------------------------------------------------------------


class TestBaseWorkerPriorityDefault:
    """Verify that BaseWorker passes request.priority to tllm.Request."""

    def _captured_tllm_request_priority(self, request_priority):
        """Run _enqueue_request with a mocked engine.

        Returns the priority value that was passed to tllm.Request.
        """
        import tensorrt_llm.executor.base_worker as bw_mod

        captured = {}

        class CapturingRequest:
            def __init__(self, *args, **kwargs):
                captured["priority"] = kwargs.get("priority")
                # Mimic minimal interface used by _enqueue_request
                self.py_num_logprobs = None
                self.py_lora_path = None
                self.py_logprobs_mode = None

            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)

        req = GenerationRequest(
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=5),
            priority=request_priority,
        )
        req.set_id(42)

        # Build a minimal mock worker.
        # max_seq_len=None causes _deduce_max_tokens to return max_tokens early
        # (via the "cannot be deduced" path), avoiding arithmetic on MagicMocks.
        worker = MagicMock()
        worker.llm_args = MagicMock()
        worker.llm_args.max_beam_width = 1
        worker.llm_args.return_perf_metrics = False
        worker._executor_config = None
        worker._is_pytorch_backend = False
        worker.max_seq_len = None
        worker.engine = MagicMock()
        worker.engine.enqueue_request = MagicMock(return_value=42)

        with patch.object(bw_mod.tllm, "Request", CapturingRequest):
            # Call the unbound method directly with our mock worker as self
            from tensorrt_llm.executor.base_worker import BaseWorker

            BaseWorker._enqueue_request(worker, req, result_wait_queue=None)

        return captured.get("priority")

    def test_explicit_priority_is_forwarded(self):
        priority = self._captured_tllm_request_priority(0.8)
        assert priority == 0.8

    def test_default_priority_is_half(self):
        priority = self._captured_tllm_request_priority(DEFAULT_REQUEST_PRIORITY)
        assert priority == DEFAULT_REQUEST_PRIORITY

    def test_low_priority_is_forwarded(self):
        priority = self._captured_tllm_request_priority(0.1)
        assert priority == 0.1

    def test_high_priority_is_forwarded(self):
        priority = self._captured_tllm_request_priority(1.0)
        assert priority == 1.0


# ---------------------------------------------------------------------------
# BaseLLM.generate - scalar and list priority
# ---------------------------------------------------------------------------


class TestBaseLLMGeneratePriority:
    """Verify that BaseLLM.generate forwards the correct priority value(s).

    Covers both scalar and list forms passed to generate_async.
    """

    def _make_llm_spy(self):
        """Return (llm, calls) where calls accumulates generate_async kwargs."""
        from tensorrt_llm.llmapi.llm import BaseLLM

        calls = []

        class SpyLLM(BaseLLM):
            def generate_async(self, inputs, **kwargs):
                calls.append(kwargs)
                result = MagicMock()
                result.result.return_value = MagicMock()
                return result

            # Satisfy abstract-ish requirements without a real model
            def _preprocess(self, *a, **kw):
                raise NotImplementedError

            def shutdown(self, *a, **kw):
                pass

        llm = SpyLLM.__new__(SpyLLM)
        llm._executor = MagicMock()
        llm._executor.is_shutdown.return_value = False
        llm.args = MagicMock()
        llm.args.return_perf_metrics = False
        return llm, calls

    def test_scalar_priority_passed_to_each_request(self):
        llm, calls = self._make_llm_spy()
        prompts = ["hello", "world"]
        llm.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=5),
            priority=0.7,
            use_tqdm=False,
        )
        assert len(calls) == 2
        assert all(c["priority"] == 0.7 for c in calls)

    def test_list_priority_per_request(self):
        llm, calls = self._make_llm_spy()
        prompts = ["hello", "world"]
        llm.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=5),
            priority=[0.2, 0.9],
            use_tqdm=False,
        )
        assert calls[0]["priority"] == 0.2
        assert calls[1]["priority"] == 0.9

    def test_no_priority_passes_default(self):
        llm, calls = self._make_llm_spy()
        llm.generate(
            ["hello"],
            sampling_params=SamplingParams(max_tokens=5),
            use_tqdm=False,
        )
        assert calls[0]["priority"] == DEFAULT_REQUEST_PRIORITY

    @pytest.mark.parametrize("bad_priority", [-0.1, 1.5, float("nan")])
    def test_invalid_scalar_priority_raises(self, bad_priority):
        llm, _ = self._make_llm_spy()
        # generate delegates to generate_async which validates the scalar value
        with pytest.raises((ValueError, Exception)):
            llm.generate(
                ["hello"],
                sampling_params=SamplingParams(max_tokens=5),
                priority=bad_priority,
                use_tqdm=False,
            )

    def test_invalid_list_priority_raises(self):
        llm, _ = self._make_llm_spy()
        with pytest.raises(ValueError):
            llm.generate(
                ["hello", "world"],
                sampling_params=SamplingParams(max_tokens=5),
                priority=[0.2, 1.5],
                use_tqdm=False,
            )


# ---------------------------------------------------------------------------
# executor_request_to_llm_request - priority not hard-coded
# ---------------------------------------------------------------------------


class TestExecutorRequestToLlmRequestPriority:
    """Verify that executor_request_to_llm_request passes executor_request.priority.

    Ensures no hard-coded value is substituted. Because the function interacts
    heavily with C++ pybind types, we verify the source-code contract instead of
    calling the function end-to-end.
    """

    def test_priority_uses_executor_request_not_hardcoded(self):
        """The function must read priority from executor_request, not use 0.5."""
        import inspect

        import tensorrt_llm._torch.pyexecutor.llm_request as lr_mod

        source = inspect.getsource(lr_mod.executor_request_to_llm_request)
        assert "priority=executor_request.priority" in source, (
            "executor_request_to_llm_request must pass executor_request.priority "
            "to LlmRequest, not a hard-coded value."
        )


# ---------------------------------------------------------------------------
# PriorityWaitingQueue - requests served in priority order
# ---------------------------------------------------------------------------


class TestPriorityWaitingQueueSchedulingOrder:
    """Verify that PriorityWaitingQueue serves requests in descending priority order.

    This is the key integration point: the waiting queue must dequeue
    higher-priority requests before lower-priority ones so that the
    downstream C++ capacity scheduler sees them in the right order.
    """

    def _make_queue_item(self, req_id: int, priority: float = 0.5):
        from unittest.mock import MagicMock

        from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem

        mock_req = MagicMock()
        mock_req.priority = priority
        return RequestQueueItem(id=req_id, request=mock_req)

    def test_high_priority_served_before_low(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import PriorityWaitingQueue

        q = PriorityWaitingQueue()
        q.add_request(self._make_queue_item(1, priority=0.1))
        q.add_request(self._make_queue_item(2, priority=0.9))
        q.add_request(self._make_queue_item(3, priority=DEFAULT_REQUEST_PRIORITY))
        served = [q.pop_request().id for _ in range(3)]
        assert served == [2, 3, 1], "Requests must be served in descending priority order"

    def test_equal_priority_falls_back_to_fcfs(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import PriorityWaitingQueue

        q = PriorityWaitingQueue()
        q.add_request(self._make_queue_item(10, priority=0.7))
        q.add_request(self._make_queue_item(20, priority=0.7))
        q.add_request(self._make_queue_item(30, priority=0.7))
        served = [q.pop_request().id for _ in range(3)]
        assert served == [10, 20, 30], (
            "Requests with equal priority must be served in arrival (FCFS) order"
        )

    def test_create_waiting_queue_priority_policy(self):
        """create_waiting_queue(PRIORITY) returns a PriorityWaitingQueue."""
        from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import (
            PriorityWaitingQueue,
            create_waiting_queue,
        )
        from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy

        q = create_waiting_queue(WaitingQueuePolicy.PRIORITY)
        assert isinstance(q, PriorityWaitingQueue)

        q.add_request(self._make_queue_item(1, priority=0.2))
        q.add_request(self._make_queue_item(2, priority=0.8))
        assert q.pop_request().id == 2
        assert q.pop_request().id == 1
