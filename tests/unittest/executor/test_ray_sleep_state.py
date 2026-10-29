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

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from utils.sleep import V2SleepWorkerExtension, assert_v2_gpu_pool_bytes

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestAdmissionState,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.executor.ray.executor import RayExecutor
from tensorrt_llm.executor.ray.gpu_worker import RayGPUWorker
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    CacheTier,
    GpuCacheTierConfig,
    HostCacheTierConfig,
)
from tensorrt_llm.sampling_params import SamplingParams


class _Args:
    sleep_config = object()
    parallel_config = SimpleNamespace(world_size=1)


def _make_worker(state="running", parked_tags=None):
    worker = object.__new__(RayGPUWorker)
    worker.doing_shutdown = True
    worker.llm_args = _Args()
    worker.engine = MagicMock()
    worker.engine.get_memory_status.return_value = {
        "state": state,
        "parked_tags": parked_tags or [],
    }
    return worker


def _run_method(method, worker, tags):
    getattr(worker, method)(tags)


@pytest.mark.parametrize("world_size", [1, 2])
def test_ray_release_recovery_requires_single_rank(world_size):
    from tensorrt_llm._torch.virtual_memory import MemoryReleaseAbortedError

    worker = _make_worker()
    worker.llm_args.parallel_config = SimpleNamespace(world_size=world_size)
    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize"),
        patch(
            "tensorrt_llm.executor.ray.gpu_worker.release_with_tag",
            side_effect=MemoryReleaseAbortedError("recovered"),
        ),
        pytest.raises(MemoryReleaseAbortedError),
    ):
        worker.sleep(["kv_cache"])
    assert worker.engine.fail_sleep_wakeup_transition.call_count == int(world_size > 1)
    worker.engine.abort_sleep_transition.assert_called_once_with()


@pytest.mark.parametrize("world_size", [1, 2])
def test_ray_capacity_failure_after_drain_respects_group_contract(world_size: int) -> None:
    worker = _make_worker()
    worker.llm_args.parallel_config = SimpleNamespace(world_size=world_size)
    worker.engine.validate_memory_release.side_effect = RuntimeError("capacity")
    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize"),
        patch("tensorrt_llm.executor.ray.gpu_worker.release_with_tag") as release,
    ):
        worker.validate_sleep(["kv_cache"])
        worker.engine.control_action.assert_not_called()
        worker.engine.validate_memory_release.assert_not_called()
        with pytest.raises(RuntimeError, match="capacity"):
            worker.sleep(["kv_cache"])
    worker.engine.control_action.assert_called_once_with()
    worker.engine.validate_memory_release.assert_called_once_with([ExecutorMemoryType.KV_CACHE])
    assert worker.engine.fail_sleep_wakeup_transition.call_count == int(world_size > 1)
    worker.engine.abort_sleep_transition.assert_called_once_with()
    worker.engine.complete_sleep_transition.assert_not_called()
    release.assert_not_called()


@pytest.mark.parametrize(
    "method, operation",
    [
        ("sleep", "release_with_tag"),
        ("wakeup", "materialize_with_tag"),
    ],
)
def test_ray_worker_publishes_persistent_admission_transition(method, operation):
    state = "running" if method == "sleep" else "parked"
    parked_tags = [] if method == "sleep" else ["model"]
    worker = _make_worker(state, parked_tags)

    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch(f"tensorrt_llm.executor.ray.gpu_worker.{operation}") as memory_operation,
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize"),
        patch("tensorrt_llm.executor.ray.gpu_worker.gc.collect"),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.empty_cache"),
    ):
        _run_method(method, worker, ["model"])

    getattr(worker.engine, f"begin_{method}_transition").assert_called_once()
    getattr(worker.engine, f"complete_{method}_transition").assert_called_once_with()
    memory_operation.assert_called_once_with(
        ExecutorMemoryType.MODEL_ENGINE_MAIN, resource=worker.engine
    )
    if method == "sleep":
        worker.engine.validate_sleep_tags.assert_called_once()
        worker.engine.invalidate_v1_prefix_cache_for_sleep.assert_called_once()
    else:
        worker.engine.validate_sleep_tags.assert_not_called()
        worker.engine.invalidate_v1_prefix_cache_for_sleep.assert_not_called()


@pytest.mark.parametrize(
    "method, operation",
    [
        ("sleep", "release_with_tag"),
        ("wakeup", "materialize_with_tag"),
    ],
)
def test_ray_worker_fails_closed_after_mutation(method, operation):
    state = "running" if method == "sleep" else "parked"
    parked_tags = [] if method == "sleep" else ["model"]
    worker = _make_worker(state, parked_tags)

    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch(f"tensorrt_llm.executor.ray.gpu_worker.{operation}"),
        patch(
            "tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize",
            side_effect=[None, RuntimeError("post-mutation failure")],
        ),
        patch("tensorrt_llm.executor.ray.gpu_worker.gc.collect"),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.empty_cache"),
        pytest.raises(RuntimeError, match="post-mutation failure"),
    ):
        _run_method(method, worker, ["model"])

    worker.engine.fail_sleep_wakeup_transition.assert_called_once_with()
    getattr(worker.engine, f"abort_{method}_transition").assert_called_once()


@pytest.mark.parametrize(
    "method, operation",
    [
        ("sleep", "release_with_tag"),
        ("wakeup", "materialize_with_tag"),
    ],
)
def test_ray_worker_aborts_transition_before_mutation(method, operation):
    state = "running" if method == "sleep" else "parked"
    parked_tags = [] if method == "sleep" else ["model"]
    worker = _make_worker(state, parked_tags)

    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch(f"tensorrt_llm.executor.ray.gpu_worker.{operation}") as mutate_memory,
        patch(
            "tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize",
            side_effect=RuntimeError("pre-mutation failure"),
        ),
        patch("tensorrt_llm.executor.ray.gpu_worker.gc.collect"),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.empty_cache"),
        pytest.raises(RuntimeError, match="pre-mutation failure"),
    ):
        _run_method(method, worker, ["model"])

    getattr(worker.engine, f"begin_{method}_transition").assert_called_once()
    getattr(worker.engine, f"abort_{method}_transition").assert_called_once_with()
    getattr(worker.engine, f"complete_{method}_transition").assert_not_called()
    worker.engine.fail_sleep_wakeup_transition.assert_not_called()
    mutate_memory.assert_not_called()


def test_ray_sleep_rejects_concurrent_submission_before_draining() -> None:
    worker = _make_worker()
    request_queue = ExecutorRequestQueue(
        dist=MagicMock(),
        max_batch_size=8,
        enable_iter_perf_stats=False,
        batch_wait_timeout_ms=0,
    )
    for action in ("complete", "abort"):
        getattr(worker.engine, f"{action}_sleep_transition").side_effect = getattr(
            request_queue, f"{action}_sleep_transition"
        )
    worker.engine.begin_sleep_transition.side_effect = lambda tags: (
        request_queue.begin_sleep_transition(tag.value for tag in tags)
    )
    request = MagicMock(disagg_request_id=None)
    with patch.object(request_queue, "_generate_child_request_ids", return_value=None):
        admitted_id = request_queue.enqueue_request(request)

    @contextmanager
    def control_action() -> Iterator[None]:
        request_queue.enqueue_control_request()
        with ThreadPoolExecutor(max_workers=1) as submissions:
            result = submissions.submit(request_queue.enqueue_request, request)
            with pytest.raises(RuntimeError, match="admission is parking"):
                result.result(timeout=5)
        assert request_queue.request_queue.get_nowait().id == admitted_id
        assert request_queue.request_queue.get_nowait().is_control_request
        assert request_queue.request_queue.empty()
        yield

    worker.engine.control_action.side_effect = control_action
    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch("tensorrt_llm.executor.ray.gpu_worker.release_with_tag") as release,
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize"),
        patch("tensorrt_llm.executor.ray.gpu_worker.gc.collect"),
        patch("tensorrt_llm.executor.ray.gpu_worker.torch.cuda.empty_cache"),
    ):
        worker.sleep(["model"])

    release.assert_called_once()
    assert request_queue.get_admission_state() is RequestAdmissionState.PARKED
    with pytest.raises(RuntimeError, match="admission is parked"):
        request_queue.enqueue_request(request)


def test_ray_sleep_aborts_transition_when_drain_fails() -> None:
    worker = _make_worker()
    worker.engine.control_action.return_value.__enter__.side_effect = RuntimeError("drain failure")
    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch("tensorrt_llm.executor.ray.gpu_worker.release_with_tag") as release,
        pytest.raises(RuntimeError, match="drain failure"),
    ):
        worker.sleep(["model"])

    worker.engine.begin_sleep_transition.assert_called_once()
    worker.engine.abort_sleep_transition.assert_called_once_with()
    worker.engine.complete_sleep_transition.assert_not_called()
    worker.engine.fail_sleep_wakeup_transition.assert_not_called()
    release.assert_not_called()


def _make_executor() -> RayExecutor:
    executor = object.__new__(RayExecutor)
    executor._wait_for_pending_calls = MagicMock()
    executor._wait_for_pending_calls_async = AsyncMock()
    executor.doing_shutdown = True
    executor._fatal_error = None
    executor._results = {}
    executor._memory_control_lock = threading.Lock()
    executor._memory_control_failed = False
    executor._memory_operation_active = None
    executor._memory_parked_tags = set()
    executor.workers = [MagicMock(), MagicMock()]
    return executor


@pytest.mark.parametrize("workers_present", [False, True])
def test_failed_memory_group_fails_health_and_pending_results(workers_present: bool) -> None:
    """A stopped group cannot leave pending streams waiting for dead actors."""
    from queue import Queue

    from tensorrt_llm.executor.utils import EngineDeadError

    executor = _make_executor()
    executor.doing_shutdown = False
    result = SimpleNamespace(queue=Queue())
    executor._results[1] = result
    if not workers_present:
        executor.workers = None
    try:
        with patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill:
            executor._fail_memory_group()
            executor._fail_memory_group()
        assert not executor.check_health()
        assert isinstance(result.queue.get_nowait(), EngineDeadError)
        assert result.queue.empty()
        assert kill.call_count == (2 if workers_present else 0)
    finally:
        executor.doing_shutdown = True


@pytest.mark.parametrize("method", ["sleep", "wakeup"])
@pytest.mark.parametrize("async_control", [False, True])
def test_memory_control_requires_initialized_workers(method: str, async_control: bool) -> None:
    """An empty worker set cannot complete a memory operation."""
    executor = _make_executor()
    executor.workers = []
    with pytest.raises(RuntimeError, match="must be initialized"):
        if async_control:
            asyncio.run(executor._run_memory_operation_async(method, (["kv_cache"],)))
        else:
            executor._run_memory_operation(method, (["kv_cache"],))
    assert not executor._memory_parked_tags
    assert executor._memory_operation_active is None


def test_closed_result_queue_does_not_prevent_group_shutdown() -> None:
    """A client closing its event loop cannot prevent fail-closed actor termination."""
    executor = _make_executor()
    executor._results[1] = SimpleNamespace(queue=MagicMock())
    executor._results[1].queue.put.side_effect = RuntimeError("loop closed")
    with patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill:
        executor._fail_memory_group()
    assert executor._memory_control_failed
    assert kill.call_count == 2


def test_submit_publishes_result_before_rpc_dispatch() -> None:
    """An immediate worker rejection has a registered result to receive it."""
    executor = _make_executor()
    executor.rpc_client = MagicMock()
    executor._get_logprob_params = MagicMock()
    request = GenerationRequest([1], SamplingParams(max_tokens=1))
    request.set_id(17)

    def dispatch(*, need_response: bool) -> None:
        assert not need_response
        assert executor._results[17].request_id == 17

    executor.rpc_client.submit.return_value.remote.side_effect = dispatch
    result = executor.submit(request)
    assert executor._results[17] is result


def test_active_worker_wakeup_does_not_drain() -> None:
    """An idempotent wake must not wait for active generation."""
    worker = _make_worker()
    with patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args):
        worker.wakeup(["kv_cache"])
    worker.engine.control_action.assert_not_called()
    worker.engine.begin_wakeup_transition.assert_not_called()


def test_ray_memory_transition_uses_http_conflict_response() -> None:
    """The serving layer recognizes driver-level control conflicts."""
    from tensorrt_llm.serve.openai_server import OpenAIServer

    executor = _make_executor()
    executor._memory_operation_active = "parking"
    with pytest.raises(RuntimeError) as caught:
        executor.sleep(["kv_cache"])
    response = OpenAIServer._runtime_control_error_response(caught.value)
    assert response.status_code == 409


def test_rpc_submit_rejection_reaches_response_stream() -> None:
    """Fire-and-forget admission failures must complete the proxy's result."""
    from queue import Queue

    from tensorrt_llm.executor.rpc_worker_mixin import RpcWorkerMixin

    class RejectingWorker:
        def submit(self, request: GenerationRequest) -> None:
            """Model the worker admission boundary after it closes."""
            raise RuntimeError("Cannot enqueue while parking")

    class RpcWorker(RpcWorkerMixin, RejectingWorker):
        pass

    worker = RpcWorker()
    worker._response_queue = Queue()
    request = GenerationRequest(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=1))
    request.set_id(17)
    with pytest.raises(RuntimeError, match="Cannot enqueue"):
        worker.submit(request)
    response = worker._response_queue.get_nowait()[0]
    assert response.client_id == 17
    assert "Cannot enqueue" in response.error_msg


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_ray_pending_call_interruption_before_dispatch_keeps_group_alive(
    error: type[BaseException],
) -> None:
    executor = _make_executor()
    del executor._wait_for_pending_calls
    with (
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=error),
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
        pytest.raises(error),
    ):
        executor.sleep(["kv_cache"])
    assert not executor._memory_control_failed
    assert executor._memory_operation_active is None
    kill.assert_not_called()
    for worker in executor.workers:
        worker.validate_sleep.remote.assert_not_called()
        worker.sleep.remote.assert_not_called()


@pytest.mark.asyncio
async def test_ray_pending_call_cancellation_before_dispatch_keeps_group_alive():
    executor = _make_executor()
    del executor._wait_for_pending_calls_async
    loop = asyncio.get_running_loop()
    pending = [loop.create_future() for _ in executor.workers]
    for worker, ready in zip(executor.workers, pending):
        worker.wait_for_pending_calls.remote.return_value = ready
    with patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill:
        task = asyncio.create_task(executor.sleep_async(["kv_cache"]))
        await asyncio.sleep(0)
        assert executor._memory_operation_active == "parking"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not executor._memory_control_failed
        assert executor._memory_operation_active is None
        kill.assert_not_called()
    for worker in executor.workers:
        worker.validate_sleep.remote.assert_not_called()
        worker.sleep.remote.assert_not_called()


@pytest.mark.parametrize("failure_stage", [None, "validate", "sleep"])
def test_ray_explicit_sleep_validates_all_ranks_before_sleep(failure_stage: str | None) -> None:
    executor = _make_executor()
    events = []
    for rank, worker in enumerate(executor.workers):
        worker.validate_sleep.remote.side_effect = lambda *_, rank=rank: events.append(
            (rank, "validate")
        )
        worker.sleep.remote.side_effect = lambda *_, rank=rank: events.append((rank, "sleep"))

    def get(refs: list[Any], *, timeout: float | None) -> list[None]:
        if events[-1][1] == "sleep":
            assert timeout is None
        else:
            assert timeout is not None and timeout > 0
        if failure_stage is not None and events[-1][1] == failure_stage:
            raise ValueError(f"{failure_stage} failed")
        return [None] * len(refs)

    with (
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=get),
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
    ):
        if failure_stage is not None:
            with pytest.raises(ValueError, match=f"{failure_stage} failed"):
                executor.sleep(["kv_cache"])
        else:
            assert executor.sleep(["kv_cache"]) is None
    assert executor._memory_control_failed == (failure_stage == "sleep")
    assert kill.call_count == (2 if failure_stage == "sleep" else 0)
    assert events == (
        [(0, "validate"), (1, "validate")]
        + ([] if failure_stage == "validate" else [(0, "sleep"), (1, "sleep")])
    )
    for worker in executor.workers:
        worker.validate_sleep.remote.assert_called_once_with(["kv_cache"])
        if failure_stage != "validate":
            worker.sleep.remote.assert_called_once_with(["kv_cache"])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", [None, "validate", "sleep"])
async def test_ray_async_sleep_waits_for_all_preflight_results(failure_stage: str | None) -> None:
    executor = _make_executor()
    loop = asyncio.get_running_loop()
    validated = [loop.create_future() for _ in executor.workers]
    slept = [loop.create_future() for _ in executor.workers]
    sleep_started = asyncio.Event()
    for rank, worker in enumerate(executor.workers):
        worker.validate_sleep.remote.return_value = validated[rank]

        def start_sleep(tags, rank=rank):
            assert tags == ["kv_cache"]
            assert all(ref.done() and ref.exception() is None for ref in validated)
            sleep_started.set()
            return slept[rank]

        worker.sleep.remote.side_effect = start_sleep

    with (
        patch("tensorrt_llm.executor.ray.executor.ray.get") as blocking_get,
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
    ):
        task = asyncio.create_task(executor.sleep_async(["kv_cache"]))
        try:
            await asyncio.sleep(0)
            for worker in executor.workers:
                worker.validate_sleep.remote.assert_called_once_with(["kv_cache"])
                worker.sleep.remote.assert_not_called()
            validated[0].set_result(None)
            await asyncio.sleep(0)
            assert not task.done()
            assert not sleep_started.is_set()
            if failure_stage == "validate":
                validated[1].set_exception(ValueError("validate failed"))
            else:
                validated[1].set_result(None)
                await asyncio.wait_for(sleep_started.wait(), timeout=5)
                assert not task.done()
                slept[0].set_result(None)
                if failure_stage == "sleep":
                    slept[1].set_exception(ValueError("sleep failed"))
                else:
                    slept[1].set_result(None)
            if failure_stage is not None:
                with pytest.raises(ValueError, match=f"{failure_stage} failed"):
                    await asyncio.wait_for(task, timeout=5)
            else:
                assert await asyncio.wait_for(task, timeout=5) is None
            if failure_stage == "validate":
                for worker in executor.workers:
                    worker.sleep.remote.assert_not_called()
            blocking_get.assert_not_called()
            assert executor._memory_control_failed == (failure_stage == "sleep")
            assert kill.call_count == (2 if failure_stage == "sleep" else 0)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task


@pytest.mark.asyncio
@pytest.mark.parametrize("async_control", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "args, enabled, error_type",
    [
        ((), True, TypeError),
        ((["kv_cache"], ["model"]), True, TypeError),
        ((None,), True, TypeError),
        (("kv_cache",), True, ValueError),
        ((["bogus"],), True, ValueError),
        (([42],), True, ValueError),
        ((["kv_cache"],), False, ValueError),
    ],
    ids=["missing", "extra", "none", "string", "unknown-tag", "non-string-tag", "disabled"],
)
async def test_ray_wakeup_validation_does_not_fail_group(
    async_control: bool, args: tuple, enabled: bool, error_type: type[Exception]
) -> None:
    """Reject invalid direct RPC input before any worker changes memory."""
    executor = _make_executor()
    workers = [_make_worker() for _ in executor.workers]
    workers[-1].llm_args.sleep_config = object() if enabled else None
    loop = asyncio.get_running_loop()

    for remote, worker in zip(executor.workers, workers):

        def validate(*args: Any, worker: RayGPUWorker = worker) -> asyncio.Future[None]:
            reply = loop.create_future()
            try:
                worker.validate_wakeup(*args)
            except Exception as error:
                reply.set_exception(error)
            else:
                reply.set_result(None)
            return reply

        remote.validate_wakeup.remote.side_effect = validate

    def get(refs: list[asyncio.Future[None]], *, timeout: float) -> list[None]:
        assert timeout > 0
        errors = [ref.exception() for ref in refs]
        for error in errors:
            if error is not None:
                raise error
        return [ref.result() for ref in refs]

    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=get),
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
        pytest.raises(error_type),
    ):
        if async_control:
            await executor.collective_rpc_async("wakeup", args)
        else:
            executor.collective_rpc("wakeup", args)

    assert not executor._memory_control_failed
    assert not executor._memory_operation_active
    assert not executor._memory_parked_tags
    kill.assert_not_called()
    for remote, worker in zip(executor.workers, workers):
        remote.validate_wakeup.remote.assert_called_once_with(*args)
        remote.wakeup.remote.assert_not_called()
        worker.engine.begin_wakeup_transition.assert_not_called()
        worker.engine.control_action.assert_not_called()
    executor.collective_rpc("enqueue_request", (None,), non_block=True)
    for remote in executor.workers:
        remote.enqueue_request.remote.assert_called_once_with(None)


@pytest.mark.parametrize("state", ["running", "parked", "parking", "waking", "failed"])
def test_ray_worker_wakeup_preflight_is_read_only(state: str) -> None:
    """Validate admission state without entering the drained control barrier."""
    worker = _make_worker(state, ["kv_cache"] if state == "parked" else [])
    with patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args):
        if state in ("running", "parked"):
            worker.validate_wakeup(["kv_cache"])
        else:
            with pytest.raises(RuntimeError, match="Cannot resume runtime memory"):
                worker.validate_wakeup(["kv_cache"])
    worker.engine.get_memory_status.assert_called_once_with()
    worker.engine.begin_wakeup_transition.assert_not_called()
    worker.engine.control_action.assert_not_called()


@pytest.mark.parametrize("non_block", [False, True])
@pytest.mark.parametrize("rank_options", [{}, {"target_ranks": 1}, {"unique_reply_rank": 0}])
def test_ray_sleep_rpc_cannot_bypass_explicit_coordination(non_block, rank_options) -> None:
    executor = _make_executor()
    with (
        patch.object(executor, "_collective_rpc_refs") as dispatch,
        patch("tensorrt_llm.executor.ray.executor.ray.get") as get,
        pytest.raises(ValueError, match="Use LLM.release"),
    ):
        executor.collective_rpc("sleep", args=(["kv_cache"],), non_block=non_block, **rank_options)
    dispatch.assert_not_called()
    get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rank_options", [{}, {"target_ranks": [1]}, {"unique_reply_rank": 0}])
async def test_ray_async_sleep_rpc_cannot_bypass_explicit_coordination(rank_options) -> None:
    executor = _make_executor()
    with (
        patch.object(executor, "_collective_rpc_refs") as dispatch,
        pytest.raises(ValueError, match="Use LLM.release"),
    ):
        await executor.collective_rpc_async("sleep", args=(["kv_cache"],), **rank_options)
    dispatch.assert_not_called()


@pytest.mark.parametrize("non_block", [False, True])
def test_ray_generic_rpc_preserves_non_blocking_behavior(non_block: bool) -> None:
    executor = _make_executor()
    refs = [worker.get_memory_status.remote.return_value for worker in executor.workers]
    replies = [{"state": "running"}, {"state": "running"}]
    with patch("tensorrt_llm.executor.ray.executor.ray.get", return_value=replies) as get:
        result = executor.collective_rpc("get_memory_status", non_block=non_block)
    assert result == (refs if non_block else replies)
    if non_block:
        get.assert_not_called()
    else:
        get.assert_called_once_with(refs)
    for worker in executor.workers:
        worker.validate_sleep.remote.assert_not_called()


@pytest.mark.parametrize("method", ["sleep", "wakeup"])
@pytest.mark.parametrize("non_block", [False, True])
@pytest.mark.parametrize("rank_options", [{}, {"target_ranks": [1]}, {"unique_reply_rank": 0}])
def test_ray_status_polling_during_memory_transition(method, non_block, rank_options) -> None:
    executor = _make_executor()
    executor._memory_parked_tags = {"model"}
    with (
        executor._memory_operation(method),
        patch("tensorrt_llm.executor.ray.executor.ray.put", side_effect=lambda value: value),
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=lambda refs: refs),
    ):
        result = executor.collective_rpc("get_memory_status", non_block=non_block, **rank_options)
        assert result == [
            {"state": "parking" if method == "sleep" else "waking", "parked_tags": ["model"]}
        ] * (1 if rank_options else 2)
        with pytest.raises(RuntimeError, match="in progress"):
            executor.collective_rpc("enqueue_request", (None,))
    for worker in executor.workers:
        worker.get_memory_status.remote.assert_not_called()


def test_ray_sleep_completion_does_not_time_out_request_drain() -> None:
    executor = _make_executor()
    validation = [worker.validate_sleep.remote.return_value for worker in executor.workers]
    release = [worker.sleep.remote.return_value for worker in executor.workers]
    with (
        patch("tensorrt_llm.executor.ray.executor.ray.get", return_value=[None, None]) as get,
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
    ):
        executor.sleep(["kv_cache"])
    assert get.call_args_list == [call(validation, timeout=300.0), call(release, timeout=None)]
    kill.assert_not_called()
    assert executor._memory_parked_tags == {"kv_cache"}


@pytest.mark.asyncio
async def test_ray_generic_async_rpc_preserves_rank_selection() -> None:
    executor = _make_executor()
    reply = asyncio.get_running_loop().create_future()
    reply.set_result({"state": "running"})
    executor.workers[1].get_memory_status.remote.return_value = reply
    with patch("tensorrt_llm.executor.ray.executor.ray.get") as get:
        result = await executor.collective_rpc_async("get_memory_status", target_ranks=[1])
    assert result == [{"state": "running"}]
    get.assert_not_called()
    executor.workers[0].get_memory_status.remote.assert_not_called()
    for worker in executor.workers:
        worker.validate_sleep.remote.assert_not_called()


@pytest.mark.parametrize("method", ["sleep", "wakeup"])
@pytest.mark.parametrize("failure_rank", [0, 1])
def test_ray_partial_memory_failure_blocks_all_entrypoints(method: str, failure_rank: int) -> None:
    """Keep the executor closed even if one actor completed and kill fails."""
    executor = _make_executor()
    completed = []

    def get(refs: list[Any], *, timeout: float) -> list[None]:
        if refs == [
            getattr(worker, f"validate_{method}").remote.return_value for worker in executor.workers
        ]:
            return [None, None]
        completed.append(1 - failure_rank)
        raise RuntimeError(f"rank {failure_rank} capacity failure")

    request = MagicMock(prompt_adapter_request=None, id=1)
    executor.rpc_client = MagicMock()
    executor._get_logprob_params = MagicMock()
    with (
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=get),
        patch(
            "tensorrt_llm.executor.ray.executor.ray.kill", side_effect=RuntimeError("unreachable")
        ) as kill,
        pytest.raises(RuntimeError, match="capacity failure"),
    ):
        executor._run_memory_operation(method, (["kv_cache"],))
    assert completed == [1 - failure_rank]
    assert kill.call_args_list == [call(worker, no_restart=True) for worker in executor.workers]
    for operation in (
        lambda: executor.sleep(["kv_cache"]),
        lambda: executor.collective_rpc("wakeup", (["kv_cache"],)),
        lambda: executor.collective_rpc("enqueue_request", (request,), target_ranks=0),
        lambda: executor.submit(request),
    ):
        with pytest.raises(RuntimeError, match="restart the entire worker group"):
            operation()
    executor.rpc_client.submit.assert_not_called()
    with (
        patch("tensorrt_llm.executor.ray.executor.ray.put", side_effect=lambda value: value),
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=lambda refs: refs),
    ):
        assert executor.collective_rpc("get_memory_status") == [
            {"state": "failed", "parked_tags": []},
            {"state": "failed", "parked_tags": []},
        ]
        assert (
            len(executor.collective_rpc("get_memory_status", target_ranks=0, non_block=True)) == 1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,outcome",
    [
        ("sleep", "success"),
        ("sleep", "failure"),
        ("sleep", "cancel"),
        ("sleep", "long_drain"),
        ("wakeup", "success"),
        ("wakeup", "failure"),
        ("wakeup", "cancel"),
        ("wakeup", "timeout"),
    ],
)
async def test_ray_memory_completion_gates_dispatch(
    method: str, outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise a delayed rank while another rank has already completed."""
    executor = _make_executor()
    loop = asyncio.get_running_loop()
    pending = [loop.create_future() for _ in executor.workers]
    started = asyncio.Event()
    for rank, worker in enumerate(executor.workers):
        ready = loop.create_future()
        ready.set_result(None)
        getattr(worker, f"validate_{method}").remote.return_value = ready

        def dispatch(*args: Any, rank: int = rank) -> asyncio.Future[None]:
            started.set()
            return pending[rank]

        getattr(worker, method).remote.side_effect = dispatch
    monkeypatch.setattr("tensorrt_llm.executor.ray.executor._MEMORY_OPERATION_TIMEOUT_S", 0.2)
    with patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill:
        task = asyncio.create_task(executor._run_memory_operation_async(method, (["kv_cache"],)))
        await asyncio.wait_for(started.wait(), timeout=5)
        pending[0].set_result(None)
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(RuntimeError, match="in progress"):
            await executor.collective_rpc_async("enqueue_request", (None,))
        with pytest.raises(RuntimeError, match="in progress"):
            await executor.sleep_async(["kv_cache"])

        def put(value):
            reply = loop.create_future()
            reply.set_result(value)
            return reply

        with patch("tensorrt_llm.executor.ray.executor.ray.put", side_effect=put):
            statuses = await executor.collective_rpc_async("get_memory_status", target_ranks=[1])
        assert statuses == [
            {"state": "parking" if method == "sleep" else "waking", "parked_tags": []}
        ]
        if outcome == "long_drain":
            await asyncio.sleep(0.3)
            assert not task.done()
            assert not pending[1].cancelled()
            kill.assert_not_called()
        if outcome in ("success", "long_drain"):
            pending[1].set_result(None)
            assert await task == [None, None]
            assert not executor._memory_control_failed
            kill.assert_not_called()
        else:
            error = {
                "failure": RuntimeError,
                "cancel": asyncio.CancelledError,
                "timeout": asyncio.TimeoutError,
            }[outcome]
            if outcome == "failure":
                pending[1].set_exception(RuntimeError("restore failed"))
            elif outcome == "cancel":
                task.cancel()
            with pytest.raises(error):
                await task
            assert executor._memory_control_failed
            assert kill.call_count == 2
            with pytest.raises(RuntimeError, match="restart the entire worker group"):
                await executor.collective_rpc_async("wakeup", (["kv_cache"],))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["sleep", "wakeup"])
@pytest.mark.parametrize("async_control", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "status",
    [
        {"state": "running", "parked_tags": []},
        {"state": "parked", "parked_tags": ["model"]},
        {"state": "failed", "parked_tags": []},
        {"state": "draining", "parked_tags": []},
        {"state": "running", "parked_tags": ["model"]},
        {"state": "parked", "parked_tags": []},
        {},
        RuntimeError("status unavailable"),
        TimeoutError("status timed out"),
    ],
    ids=[
        "running",
        "parked",
        "failed",
        "draining",
        "running-with-tags",
        "empty-parked",
        "missing-state",
        "unreachable",
        "timeout",
    ],
)
async def test_ray_single_worker_failure_checks_admission(
    method: str, async_control: bool, status: dict[str, object] | Exception
) -> None:
    """Keep driver admission aligned with the worker after an operation error."""
    executor = _make_executor()
    executor.workers = executor.workers[:1]
    worker = executor.workers[0]
    request = GenerationRequest([1, 2], SamplingParams(max_tokens=1))
    request.set_id(1)
    executor.rpc_client = MagicMock()
    executor._get_logprob_params = MagicMock()
    executor._results = {}

    def status_reply() -> dict[str, object]:
        with pytest.raises(RuntimeError, match="in progress"):
            executor.submit(request)
        if isinstance(status, Exception):
            raise status
        return status

    def get(refs: list[Any], *, timeout: float | None) -> list[Any]:
        if refs == [getattr(worker, f"validate_{method}").remote.return_value]:
            assert timeout is not None and timeout > 0
            return [None]
        if refs == [worker.get_memory_status.remote.return_value]:
            assert timeout is not None and timeout > 0
            return [status_reply()]
        if method == "sleep":
            assert timeout is None
        else:
            assert timeout is not None and timeout > 0
        raise RuntimeError("memory operation failed")

    if async_control:
        loop = asyncio.get_running_loop()
        validated = loop.create_future()
        validated.set_result(None)
        getattr(worker, f"validate_{method}").remote.return_value = validated
        failed = loop.create_future()
        failed.set_exception(RuntimeError("memory operation failed"))
        getattr(worker, method).remote.return_value = failed

        def get_status() -> asyncio.Future[dict[str, object]]:
            reply = loop.create_future()
            try:
                reply.set_result(status_reply())
            except Exception as error:
                reply.set_exception(error)
            return reply

        worker.get_memory_status.remote.side_effect = get_status

    with (
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=get),
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
        pytest.raises(RuntimeError, match="memory operation failed"),
    ):
        if async_control:
            await executor._run_memory_operation_async(method, (["kv_cache"],))
        else:
            executor._run_memory_operation(method, (["kv_cache"],))

    worker.get_memory_status.remote.assert_called_once_with()
    recovered = isinstance(status, dict) and status in (
        {"state": "running", "parked_tags": []},
        {"state": "parked", "parked_tags": ["model"]},
    )
    assert executor._memory_control_failed is not recovered
    assert not executor._memory_operation_active
    if recovered:
        kill.assert_not_called()
        assert executor._memory_parked_tags == set(status["parked_tags"])
        if status["state"] == "running":
            executor.submit(request)
            executor.rpc_client.submit.assert_called_once_with(request)
        else:
            with pytest.raises(RuntimeError, match="memory is parked"):
                executor.submit(request)
    else:
        kill.assert_called_once_with(worker, no_restart=True)
        with pytest.raises(RuntimeError, match="restart the entire worker group"):
            executor.submit(request)
        with pytest.raises(RuntimeError, match="restart the entire worker group"):
            await executor.collective_rpc_async("wakeup", (["kv_cache"],))
        with pytest.raises(RuntimeError, match="restart the entire worker group"):
            executor.collective_rpc("enqueue_request", (request,))
    if not recovered or status["state"] != "running":
        executor.rpc_client.submit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel"])
async def test_ray_single_worker_recovery_query_stays_closed(
    outcome: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled or cancelled recovery query cannot reopen driver admission."""
    executor = _make_executor()
    executor.workers = executor.workers[:1]
    worker = executor.workers[0]
    loop = asyncio.get_running_loop()
    validated = loop.create_future()
    validated.set_result(None)
    worker.validate_wakeup.remote.return_value = validated
    failed = loop.create_future()
    failed.set_exception(RuntimeError("memory operation failed"))
    worker.wakeup.remote.return_value = failed
    status = loop.create_future()
    querying = asyncio.Event()

    def get_status() -> asyncio.Future[dict[str, object]]:
        querying.set()
        return status

    worker.get_memory_status.remote.side_effect = get_status
    monkeypatch.setattr("tensorrt_llm.executor.ray.executor._MEMORY_OPERATION_TIMEOUT_S", 0.2)
    with patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill:
        task = asyncio.create_task(executor._run_memory_operation_async("wakeup", (["kv_cache"],)))
        await asyncio.wait_for(querying.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="in progress"):
            await executor.collective_rpc_async("enqueue_request", (None,))
        if outcome == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if outcome == "cancel" else RuntimeError):
            await task
    assert executor._memory_control_failed
    assert not executor._memory_operation_active
    kill.assert_called_once_with(worker, no_restart=True)


class _LegacyAsyncioTimeoutError(Exception):
    """Model asyncio's distinct timeout exception on Python 3.10."""


@pytest.mark.asyncio
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("failure_stage", ["validate_sleep", "sleep", "validate_wakeup", "wakeup"])
@pytest.mark.parametrize(
    "timeout_type",
    [
        pytest.param(TimeoutError, id="builtin"),
        pytest.param(_LegacyAsyncioTimeoutError, id="asyncio-310"),
    ],
)
async def test_ray_async_memory_timeout_closes_group(
    world_size: int,
    failure_stage: str,
    timeout_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep admission closed for either timeout type at every async stage."""
    executor = _make_executor()
    executor.workers = executor.workers[:world_size]
    # Keep the two exception types distinct even on newer Python versions.
    monkeypatch.setattr(asyncio, "TimeoutError", _LegacyAsyncioTimeoutError)
    loop = asyncio.get_running_loop()
    for rank, worker in enumerate(executor.workers):
        for stage in ("validate_sleep", "sleep", "validate_wakeup", "wakeup"):
            reply = loop.create_future()
            if stage == failure_stage and rank == world_size - 1:
                reply.set_exception(timeout_type("memory operation timed out"))
            else:
                reply.set_result(None)
            getattr(worker, stage).remote.return_value = reply

    method = "wakeup" if failure_stage.endswith("wakeup") else "sleep"
    with (
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
        pytest.raises(timeout_type, match="memory operation timed out"),
    ):
        await executor._run_memory_operation_async(method, (["kv_cache"],))

    assert executor._memory_control_failed
    assert not executor._memory_operation_active
    assert kill.call_args_list == [call(worker, no_restart=True) for worker in executor.workers]
    if failure_stage.startswith("validate_"):
        for worker in executor.workers:
            getattr(worker, method).remote.assert_not_called()
    for operation, args in (
        ("enqueue_request", (None,)),
        ("wakeup", (["kv_cache"],)),
    ):
        with pytest.raises(RuntimeError, match="restart the entire worker group"):
            await executor.collective_rpc_async(operation, args)
    with pytest.raises(RuntimeError, match="restart the entire worker group"):
        await executor.sleep_async(["kv_cache"])


@pytest.mark.parametrize(
    "options", [{"non_block": True}, {"target_ranks": 0}, {"unique_reply_rank": 1}]
)
def test_ray_wakeup_cannot_skip_group_completion(options: dict) -> None:
    """Reject wakeup paths that would bypass the cohort completion gate."""
    executor = _make_executor()
    with pytest.raises(ValueError, match="requires all workers"):
        executor.collective_rpc("wakeup", (["kv_cache"],), **options)
    for worker in executor.workers:
        worker.wakeup.remote.assert_not_called()


@pytest.mark.parametrize("method", ["sleep", "wakeup"])
def test_ray_nested_worker_rpc_cannot_bypass_memory_control(method: str) -> None:
    """Reject the generic wrapper as an alternative memory-control entrypoint."""
    executor = _make_executor()
    with pytest.raises(ValueError, match="memory controls"):
        executor.collective_rpc("call_worker_method", (method, ["kv_cache"]))


def test_ray_partial_resume_keeps_submission_closed() -> None:
    """Only the final tag resume reopens driver-side submission."""
    executor = _make_executor()
    request = MagicMock(prompt_adapter_request=None, id=1)
    executor.rpc_client = MagicMock()
    executor._get_logprob_params = MagicMock()
    with patch("tensorrt_llm.executor.ray.executor.ray.get", return_value=[None, None]):
        executor.sleep(["kv_cache", "model"])
        for remaining in ({"kv_cache", "model"}, {"kv_cache"}):
            assert executor._memory_parked_tags == remaining
            with pytest.raises(RuntimeError, match="memory is parked"):
                executor.submit(request)
            with pytest.raises(RuntimeError, match="memory is parked"):
                executor.collective_rpc("enqueue_request", (request,), target_ranks=0)
            executor.collective_rpc("wakeup", (["model"],))
        executor.collective_rpc("wakeup", (["kv_cache"],))
        assert not executor._memory_parked_tags
        executor.collective_rpc("enqueue_request", (request,), target_ranks=0)
    executor.workers[0].enqueue_request.remote.assert_called_once_with(request)
    executor.rpc_client.submit.assert_not_called()


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_ray_sync_interruption_closes_group(error: type[BaseException]) -> None:
    """Interrupting an already dispatched operation must not reopen admission."""
    executor = _make_executor()
    with (
        patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=[[None, None], error]),
        patch("tensorrt_llm.executor.ray.executor.ray.kill") as kill,
        pytest.raises(error),
    ):
        executor.collective_rpc("wakeup", (["kv_cache"],))
    assert executor._memory_control_failed
    assert kill.call_count == 2
    with pytest.raises(RuntimeError, match="restart the entire worker group"):
        executor.collective_rpc("enqueue_request", (None,))


def test_v2_gpu_pool_report_uses_native_bytes_from_all_gpu_tiers() -> None:
    manager = MagicMock(spec=KVCacheManagerV2)
    manager.impl = MagicMock()
    manager.impl.init_config.cache_tiers = [
        GpuCacheTierConfig(quota=1024),
        GpuCacheTierConfig(quota=2048),
        HostCacheTierConfig(quota=4096),
    ]
    manager.impl.get_num_bytes.side_effect = [64, 128]
    worker = SimpleNamespace(global_rank=1, engine=SimpleNamespace(kv_cache_manager=manager))
    assert V2SleepWorkerExtension.get_v2_gpu_pool_bytes(worker) == (1, 192)
    assert manager.impl.get_num_bytes.call_args_list == [
        call(0, CacheTier.GPU_MEM),
        call(1, CacheTier.GPU_MEM),
    ]


@pytest.mark.parametrize("rank", [0, 1])
def test_v2_gpu_pool_bytes_detects_unreleased_rank(rank: int) -> None:
    replies = [(worker_rank, 1024 if worker_rank == rank else 0) for worker_rank in range(2)]
    with pytest.raises(AssertionError, match=f"Rank {rank}: expected 0"):
        assert_v2_gpu_pool_bytes(replies, {0: 0, 1: 0})


@pytest.mark.parametrize("rank", [0, 1])
def test_v2_gpu_pool_bytes_detects_incomplete_restore(rank: int) -> None:
    expected = {0: 1024, 1: 2048}
    replies = [
        (worker_rank, 0 if worker_rank == rank else size) for worker_rank, size in expected.items()
    ]
    with pytest.raises(AssertionError, match=f"Rank {rank}:"):
        assert_v2_gpu_pool_bytes(replies, expected)


@pytest.mark.parametrize(
    "replies", [[], [(0, 0)], [(0, 0), (0, 0)], [(0, 0), (2, 0)], [(0, 0), (1, 0), (1, 0)]]
)
def test_v2_gpu_pool_bytes_requires_one_reply_per_rank(replies: list[tuple[int, int]]) -> None:
    with pytest.raises(AssertionError, match="Expected one pool report per rank"):
        assert_v2_gpu_pool_bytes(replies, {0: 0, 1: 0})


def test_v2_gpu_pool_bytes_accepts_reordered_replies() -> None:
    assert_v2_gpu_pool_bytes([(1, 2048), (0, 1024)], {0: 1024, 1: 2048})
