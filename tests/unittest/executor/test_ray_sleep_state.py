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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestAdmissionState,
)
from tensorrt_llm.executor.ray.executor import RayExecutor
from tensorrt_llm.executor.ray.gpu_worker import RayGPUWorker
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType


class _Args:
    sleep_config = object()


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
    if method == "sleep":
        worker.sleep(tags)
    else:
        getattr(RayGPUWorker, method).__wrapped__(worker, tags)


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
    "method, operation, capacity_failure",
    [
        ("sleep", "release_with_tag", False),
        ("sleep", "release_with_tag", True),
        ("wakeup", "materialize_with_tag", False),
    ],
)
def test_ray_worker_aborts_transition_before_mutation(method, operation, capacity_failure):
    state = "running" if method == "sleep" else "parked"
    parked_tags = [] if method == "sleep" else ["model"]
    worker = _make_worker(state, parked_tags)
    if capacity_failure:
        worker.engine.validate_sleep_config.side_effect = RuntimeError("pre-mutation failure")

    with (
        patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args),
        patch("tensorrt_llm.executor.ray.gpu_worker.logger", MagicMock(), create=True),
        patch(f"tensorrt_llm.executor.ray.gpu_worker.{operation}") as mutate_memory,
        patch(
            "tensorrt_llm.executor.ray.gpu_worker.torch.cuda.synchronize",
            side_effect=None if capacity_failure else RuntimeError("pre-mutation failure"),
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


@pytest.mark.parametrize("parked", [False, True])
def test_ray_sleep_preflight_is_read_only(parked):
    worker = _make_worker("parked" if parked else "running", ["kv_cache"] if parked else [])
    worker.engine.validate_sleep_config.side_effect = ValueError("host quota is too small")
    with patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", _Args):
        if parked:
            worker.validate_sleep(["kv_cache"])
            worker.engine.validate_sleep_config.assert_not_called()
        else:
            with pytest.raises(ValueError, match="host quota is too small"):
                worker.validate_sleep(["kv_cache"])
    worker.engine.begin_sleep_transition.assert_not_called()
    worker.engine.control_action.assert_not_called()


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_ray_collective_validates_all_ranks_before_sleep(reject, asynchronous):
    executor = object.__new__(RayExecutor)
    executor.doing_shutdown = True
    executor.workers = [MagicMock(), MagicMock()]
    events = []

    def dispatch(rank, method, result):
        def remote(*args, **kwargs):
            assert args == (["kv_cache"],)
            assert kwargs == {}
            events.append((rank, method))
            return result

        return remote

    async def run_async():
        loop = asyncio.get_running_loop()
        preflight = [loop.create_future(), loop.create_future()]
        for rank, worker in enumerate(executor.workers):
            worker.validate_sleep.remote.side_effect = dispatch(rank, "validate", preflight[rank])
            done = loop.create_future()
            done.set_result(None)
            worker.sleep.remote.side_effect = dispatch(rank, "sleep", done)
        with patch("tensorrt_llm.executor.ray.executor.ray.get") as blocking_get:
            task = asyncio.create_task(executor.collective_rpc_async("sleep", args=(["kv_cache"],)))
            await asyncio.sleep(0)
            assert events == [(0, "validate"), (1, "validate")]
            assert not task.done()
            preflight[0].set_result(None)
            if reject:
                preflight[1].set_exception(ValueError("host quota is too small"))
                with pytest.raises(ValueError, match="host quota is too small"):
                    await task
            else:
                preflight[1].set_result(None)
                assert await task == [None, None]
            blocking_get.assert_not_called()

    if asynchronous:
        asyncio.run(run_async())
    else:
        for rank, worker in enumerate(executor.workers):
            worker.validate_sleep.remote.side_effect = dispatch(rank, "validate", rank)
            worker.sleep.remote.side_effect = dispatch(rank, "sleep", rank + 2)

        def get(refs):
            if refs == [0, 1]:
                assert events == [(0, "validate"), (1, "validate")]
                if reject:
                    raise ValueError("host quota is too small")
            return [None, None]

        with patch("tensorrt_llm.executor.ray.executor.ray.get", side_effect=get):
            if reject:
                with pytest.raises(ValueError, match="host quota is too small"):
                    executor.collective_rpc("sleep", args=(["kv_cache"],))
            else:
                assert executor.collective_rpc("sleep", args=(["kv_cache"],)) == [None, None]
    assert events == (
        [(0, "validate"), (1, "validate")] + ([] if reject else [(0, "sleep"), (1, "sleep")])
    )
