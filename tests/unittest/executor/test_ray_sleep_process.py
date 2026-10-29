# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-process memory-control failures with real Ray transport and admission.

CUDA allocation and scheduler draining are isolated at the worker boundary.
These tests do not qualify GPU reconstruction or NCCL failure recovery.
"""

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import ray

from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue
from tensorrt_llm.executor.ray.executor import RayExecutor
from tensorrt_llm.executor.ray.gpu_worker import RayGPUWorker

# Ray's native core-worker threads register Python DummyThreads during error reporting.
pytestmark = pytest.mark.threadleak(enabled=False)


@ray.remote(num_cpus=1)
class _MemoryActor:
    """Run production worker transitions with a controlled allocation boundary."""

    def __init__(self, rank: int, failure_rank: int, failure: str, directory: str) -> None:
        self.rank = rank
        self.failure_rank = failure_rank
        self.failure = failure
        self.directory = Path(directory)
        self.queue = ExecutorRequestQueue(
            dist=MagicMock(),
            max_batch_size=8,
            enable_iter_perf_stats=False,
            batch_wait_timeout_ms=0,
        )
        self.worker = object.__new__(RayGPUWorker)
        self.worker.doing_shutdown = True
        self.worker.llm_args = SimpleNamespace(
            sleep_config=object(), parallel_config=SimpleNamespace(world_size=2)
        )
        self.worker.engine = MagicMock()
        for operation in ("sleep", "wakeup"):
            for phase in ("complete", "abort"):
                name = f"{phase}_{operation}_transition"
                getattr(self.worker.engine, name).side_effect = getattr(self.queue, name)
            getattr(self.worker.engine, f"begin_{operation}_transition").side_effect = (
                lambda tags, operation=operation: getattr(
                    self.queue, f"begin_{operation}_transition"
                )(tag.value for tag in tags)
            )
        self.worker.engine.get_memory_status.side_effect = self.get_memory_status
        self.worker.engine.fail_sleep_wakeup_transition.side_effect = (
            self.queue.fail_sleep_wakeup_transition
        )
        self.worker.engine.validate_memory_release.side_effect = self._validate_capacity
        if failure == "drain":
            self.worker.engine.control_action.side_effect = self._drain

    @contextmanager
    def _drain(self, *, drain: bool = True) -> Iterator[None]:
        """Honor the control action's drain flag at the test barrier."""
        if drain:
            (self.directory / f"draining-{self.rank}").touch()
            deadline = time.monotonic() + 20
            while not (self.directory / "drained").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("test did not release the drain barrier")
                time.sleep(0.01)
        yield

    def get_memory_status(self) -> dict:
        """Return the real request queue's state."""
        state, tags = self.queue.get_memory_status()
        return {"state": state.value, "parked_tags": sorted(tags)}

    def wait_for_pending_calls(self) -> None:
        """Use the same serial-actor barrier as the production wrapper."""

    def update_weights(self) -> None:
        """Hold the actor queue until the test releases an earlier control call."""
        (self.directory / f"updating-{self.rank}").touch()
        deadline = time.monotonic() + 20
        while not (self.directory / "updated").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("test did not complete the weight update")
            time.sleep(0.01)

    def validate_sleep(self, tags: list[str]) -> None:
        """Validate configuration through the production worker method."""
        (self.directory / f"validating-{self.rank}").touch()
        with patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", SimpleNamespace):
            self.worker.validate_sleep(tags)

    def validate_wakeup(self, tags: list[str]) -> None:
        """Validate restoration through the production worker method."""
        with patch("tensorrt_llm.executor.ray.gpu_worker.TorchLlmArgs", SimpleNamespace):
            self.worker.validate_wakeup(tags)

    def _fail_after_peer(self, operation: str) -> None:
        """Wait until another process commits before injecting the failure."""
        peer_marker = self.directory / f"{operation}-{1 - self.rank}"
        deadline = time.monotonic() + 15
        while not peer_marker.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("peer did not complete")
            time.sleep(0.01)
        if self.failure == "crash":
            os._exit(17)
        if self.failure == "timeout":
            time.sleep(30)
        raise RuntimeError(f"injected {self.failure} on rank {self.rank}")

    def _validate_capacity(self, tags: list[str]) -> None:
        """Inject a post-drain capacity failure before this rank releases memory."""
        if self.failure == "capacity_all":
            raise ValueError("retained pages exceed host capacity")
        if self.rank == self.failure_rank and self.failure in ("capacity", "crash"):
            self._fail_after_peer("sleep")

    def _release(self, *args: Any, **kwargs: Any) -> None:
        """Record whether a rank reaches the physical memory release boundary."""
        (self.directory / f"released-{self.rank}").touch()

    def _restore(self, *args: Any, **kwargs: Any) -> None:
        """Inject an allocation failure during the selected rank's wakeup."""
        if self.rank == self.failure_rank and self.failure in ("restore", "timeout"):
            self._fail_after_peer("wakeup")

    def _operate(self, method: str, tags: list[str]) -> None:
        """Run the real worker method while isolating CUDA-only operations."""
        module = "tensorrt_llm.executor.ray.gpu_worker"
        with (
            patch(f"{module}.TorchLlmArgs", SimpleNamespace),
            patch(f"{module}.logger", MagicMock(), create=True),
            patch(f"{module}.torch.cuda.synchronize"),
            patch(f"{module}.torch.cuda.empty_cache"),
            patch(f"{module}.release_with_tag", side_effect=self._release),
            patch(f"{module}.materialize_with_tag", side_effect=self._restore),
        ):
            getattr(self.worker, method)(tags)
        (self.directory / f"{method}-{self.rank}").write_text("completed", encoding="ascii")

    def sleep(self, tags: list[str]) -> None:
        """Release through production worker and admission transitions."""
        self._operate("sleep", tags)

    def wakeup(self, tags: list[str]) -> None:
        """Restore through production worker and admission transitions."""
        self._operate("wakeup", tags)


@pytest.fixture(scope="module")
def memory_ray_runtime() -> Iterator[None]:
    """Start an isolated local Ray runtime unless the test runner owns one."""
    owned = not ray.is_initialized()
    if owned:
        ray.init(num_cpus=2, include_dashboard=False)
    try:
        yield
    finally:
        if owned:
            ray.shutdown()


@pytest.mark.parametrize(
    "failure, failure_rank",
    [(failure, rank) for failure in ("capacity", "restore", "crash", "timeout") for rank in (0, 1)]
    + [("capacity_all", 0)],
)
@pytest.mark.parametrize("async_control", [False, True], ids=["sync", "async"])
def test_ray_process_failure_requires_group_restart(
    memory_ray_runtime: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    failure_rank: int,
    async_control: bool,
) -> None:
    """Post-dispatch failures require restart, including capacity-only failures."""
    executor = object.__new__(RayExecutor)
    executor.doing_shutdown = True
    executor._fatal_error = None
    executor._results = {}
    executor._memory_control_lock = threading.Lock()
    executor._memory_control_failed = False
    executor._memory_operation_active = None
    executor._memory_parked_tags = set()
    executor.workers = [
        _MemoryActor.remote(rank, failure_rank, failure, str(tmp_path)) for rank in range(2)
    ]
    # Startup imports are outside the operation deadline under test.
    ray.get([worker.get_memory_status.remote() for worker in executor.workers], timeout=90)
    monkeypatch.setattr("tensorrt_llm.executor.ray.executor._MEMORY_OPERATION_TIMEOUT_S", 5.0)

    def operate(method: str) -> None:
        if async_control:
            if method == "sleep":
                asyncio.run(executor.sleep_async(["kv_cache"]))
            else:
                asyncio.run(executor.collective_rpc_async("wakeup", (["kv_cache"],)))
        elif method == "sleep":
            executor.sleep(["kv_cache"])
        else:
            executor.collective_rpc("wakeup", (["kv_cache"],))

    try:
        if failure in ("restore", "timeout"):
            operate("sleep")
            assert all(
                status["state"] == "parked"
                for status in executor.collective_rpc("get_memory_status")
            )
        method = "wakeup" if failure in ("restore", "timeout") else "sleep"
        expected_error = (
            RuntimeError,
            ValueError,
            ray.exceptions.RayActorError,
            ray.exceptions.GetTimeoutError,
            asyncio.TimeoutError,
            TimeoutError,
        )
        with pytest.raises(expected_error):
            operate(method)
        if failure == "capacity_all":
            assert not list(tmp_path.glob("released-*"))
            assert not list(tmp_path.glob("sleep-*"))
        else:
            assert (tmp_path / f"{method}-{1 - failure_rank}").read_text(
                encoding="ascii"
            ) == "completed"
        assert executor._memory_control_failed
        for retry in ("sleep", "wakeup"):
            with pytest.raises(RuntimeError, match="restart the entire worker group"):
                operate(retry)
        with pytest.raises(RuntimeError, match="restart the entire worker group"):
            executor.collective_rpc("enqueue_request", (None,), target_ranks=0)
        request = MagicMock(prompt_adapter_request=None, id=1)
        executor._get_logprob_params = MagicMock()
        executor.rpc_client = MagicMock()
        with pytest.raises(RuntimeError, match="restart the entire worker group"):
            executor.submit(request)
        executor.rpc_client.submit.assert_not_called()
        assert all(
            status["state"] == "failed" for status in executor.collective_rpc("get_memory_status")
        )
        for worker in executor.workers:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(worker.get_memory_status.remote(), timeout=10)
    finally:
        for worker in executor.workers:
            ray.kill(worker, no_restart=True)
        executor.workers = []


@pytest.mark.parametrize("async_control", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("blocked_operation", ["drain", "update"])
def test_ray_process_long_control_allows_status_polling(
    memory_ray_runtime: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    async_control: bool,
    blocked_operation: str,
) -> None:
    executor = object.__new__(RayExecutor)
    executor.doing_shutdown = True
    executor._fatal_error = None
    executor._results = {}
    executor._memory_control_lock = threading.Lock()
    executor._memory_control_failed = False
    executor._memory_operation_active = None
    executor._memory_parked_tags = set()
    executor.workers = [
        _MemoryActor.remote(rank, -1, blocked_operation, str(tmp_path)) for rank in range(2)
    ]
    ray.get([worker.get_memory_status.remote() for worker in executor.workers], timeout=90)
    monkeypatch.setattr("tensorrt_llm.executor.ray.executor._MEMORY_OPERATION_TIMEOUT_S", 1.0)

    def sleep() -> None:
        if async_control:
            asyncio.run(executor.sleep_async(["kv_cache"]))
        else:
            executor.sleep(["kv_cache"])

    try:
        pending = (
            executor.collective_rpc("update_weights", non_block=True)
            if blocked_operation == "update"
            else []
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(sleep)
            try:
                deadline = time.monotonic() + 10
                marker = "updating" if blocked_operation == "update" else "draining"
                while not all((tmp_path / f"{marker}-{rank}").exists() for rank in range(2)):
                    assert time.monotonic() < deadline, "workers did not start the blocking call"
                    time.sleep(0.01)
                time.sleep(1.2)
                assert not task.done()
                if blocked_operation == "update":
                    assert not any((tmp_path / f"validating-{rank}").exists() for rank in range(2))
                if async_control:
                    statuses = asyncio.run(executor.collective_rpc_async("get_memory_status"))
                else:
                    refs = executor.collective_rpc("get_memory_status", non_block=True)
                    statuses = ray.get(refs, timeout=1)
                assert statuses == [{"state": "parking", "parked_tags": []}] * 2
                assert not executor._memory_control_failed
                with pytest.raises(RuntimeError, match="in progress"):
                    executor.collective_rpc("enqueue_request", (None,))
            finally:
                (tmp_path / ("updated" if blocked_operation == "update" else "drained")).touch()
            task.result(timeout=10)
        ray.get(pending, timeout=5)
        assert all(
            status["state"] == "parked" for status in executor.collective_rpc("get_memory_status")
        )
        monkeypatch.setattr("tensorrt_llm.executor.ray.executor._MEMORY_OPERATION_TIMEOUT_S", 5.0)
        executor.collective_rpc("wakeup", (["kv_cache"],))
        assert all(
            status["state"] == "running" for status in executor.collective_rpc("get_memory_status")
        )
    finally:
        for worker in executor.workers:
            ray.kill(worker, no_restart=True)
        executor.workers = []
