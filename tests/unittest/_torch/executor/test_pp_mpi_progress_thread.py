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
"""CPU-only tests for the native pipeline parallelism MPI progress thread."""

import atexit
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mpi4py import MPI

from tensorrt_llm._torch.pyexecutor.pp_utils import MPI_PROGRESS_POLL_MS_ENV_VAR_NAME, PPCommTag
from tensorrt_llm._torch.pyexecutor.py_executor import (
    DEFAULT_PP_MPI_PROGRESS_INTERVAL_US,
    MAX_PP_MPI_PROGRESS_INTERVAL_US,
    PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME,
    PyExecutor,
    _resolve_pp_mpi_progress_interval_us,
)
from tensorrt_llm.bindings import MpiProgressThread

pytestmark = pytest.mark.cpu_only

needs_thread_multiple = pytest.mark.skipif(
    MPI.Query_thread() < MPI.THREAD_MULTIPLE,
    reason="the MPI progress thread needs MPI_THREAD_MULTIPLE",
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
        ("abc", DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
        ("2.5", DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
        ("0", 0),
        ("-1", -1),
        ("20000", 20000),
        (str(MAX_PP_MPI_PROGRESS_INTERVAL_US), MAX_PP_MPI_PROGRESS_INTERVAL_US),
        (str(MAX_PP_MPI_PROGRESS_INTERVAL_US + 1), DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
        (str(2**63), DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
    ],
)
def test_resolve_interval(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME, raw)
    assert _resolve_pp_mpi_progress_interval_us() == expected


def _wait_for_probes(thread, count, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while thread.num_probes < count and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread.num_probes


@needs_thread_multiple
def test_thread_probes_until_stopped():
    thread = MpiProgressThread(MPI.COMM_WORLD.py2f(), 1000, int(PPCommTag.MPI_PROGRESS_PROBE))
    try:
        assert _wait_for_probes(thread, 3) >= 3
        assert thread.is_running
    finally:
        thread.stop()
    assert not thread.is_running
    probes = thread.num_probes
    time.sleep(0.05)
    assert thread.num_probes == probes
    # stop() is idempotent.
    thread.stop()


@needs_thread_multiple
def test_concurrent_stop_calls_all_return():
    num_callers = 4
    for _ in range(20):
        thread = MpiProgressThread(MPI.COMM_WORLD.py2f(), 1000, int(PPCommTag.MPI_PROGRESS_PROBE))
        assert _wait_for_probes(thread, 1) >= 1
        barrier = threading.Barrier(num_callers)
        errors = []

        def stop(thread=thread, barrier=barrier, errors=errors):
            barrier.wait()
            try:
                thread.stop()
            except RuntimeError as error:
                errors.append(error)

        # Daemon callers and a bounded join, so that a deadlock in stop()
        # fails this test instead of hanging the test process.
        callers = [threading.Thread(target=stop, daemon=True) for _ in range(num_callers)]
        for caller in callers:
            caller.start()
        deadline = time.monotonic() + 30.0
        for caller in callers:
            caller.join(timeout=max(0.0, deadline - time.monotonic()))
        assert not any(caller.is_alive() for caller in callers), "stop() did not return"
        assert not errors
        assert not thread.is_running


@needs_thread_multiple
def test_thread_leaves_other_messages_to_the_receiver():
    """The probe never matches, so a message on another tag stays receivable."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tag = int(PPCommTag.SAMPLE_STATE)
    thread = MpiProgressThread(comm.py2f(), 1000, int(PPCommTag.MPI_PROGRESS_PROBE))
    try:
        request = comm.isend("payload", dest=rank, tag=tag)
        assert _wait_for_probes(thread, 3) >= 3
        # Poll the receive against a deadline, so that a thread that took the
        # message fails this test instead of hanging it.
        receive = comm.irecv(source=rank, tag=tag)
        deadline = time.monotonic() + 10.0
        received, payload = receive.test()
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
            received, payload = receive.test()
        assert received, "the message was not received"
        assert payload == "payload"
        request.wait()
    finally:
        thread.stop()


@pytest.mark.parametrize("interval_us", [0, -5])
def test_rejects_nonpositive_interval(interval_us):
    with pytest.raises(RuntimeError, match="interval must be positive"):
        MpiProgressThread(MPI.COMM_WORLD.py2f(), interval_us, int(PPCommTag.MPI_PROGRESS_PROBE))


def _record_atexit(monkeypatch) -> SimpleNamespace:
    """Record the callables passed to atexit.register and atexit.unregister.

    Both calls still reach atexit. atexit.unregister removes a callback that
    compares equal to its argument, which includes the registered object
    itself.
    """
    calls = SimpleNamespace(registered=[], unregistered=[])
    real_register, real_unregister = atexit.register, atexit.unregister

    def register(func, *args, **kwargs):
        calls.registered.append(func)
        return real_register(func, *args, **kwargs)

    def unregister(func):
        calls.unregistered.append(func)
        real_unregister(func)

    monkeypatch.setattr(atexit, "register", register)
    monkeypatch.setattr(atexit, "unregister", unregister)
    return calls


def _executor(monkeypatch, pp_size: int) -> PyExecutor:
    """An executor double that can run start_worker() and shutdown().

    The executor and relay loops are replaced with no-ops, and the relay
    thread's Python pump is disabled, so the native thread is the only MPI
    progress source under test.
    """
    monkeypatch.setenv("TLLM_NUMA_AWARE_WORKER_AFFINITY", "0")
    monkeypatch.setenv(MPI_PROGRESS_POLL_MS_ENV_VAR_NAME, "0")
    monkeypatch.delenv(PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME, raising=False)
    executor = object.__new__(PyExecutor)
    executor._closed = False
    executor._resources_released = False
    executor.kv_cache_transceiver = None
    executor.dist = SimpleNamespace(rank=0, world_size=1, pp_size=pp_size)
    executor.worker_lock = threading.Lock()
    executor.worker_started = False
    executor.num_micro_batches = 2
    executor._disable_mpi = False
    executor._mpi_progress_thread = None
    executor._mpi_progress_thread_exit_hook = None
    executor.sampler = None
    executor._event_loop_wrapper = lambda: None
    executor._broadcast_sample_state_loop = lambda: None
    executor.executor_request_queue = Mock()
    executor.shutdown_event = Mock()
    executor._profile_manager = Mock()
    executor.hang_detector = Mock()
    executor.hang_detector.detected.return_value = False
    executor.kv_connector_manager = None
    executor.enable_attention_dp = False
    executor._kv_connector_failed = False
    executor._shutdown_sleep_wakeup_listeners = Mock()
    executor.model_engine = None
    executor.draft_model_engine = None
    executor.resource_manager = SimpleNamespace(resource_managers={})
    executor.virtual_memory_pools = None
    executor.dwdp_manager = None
    return executor


@needs_thread_multiple
@pytest.mark.parametrize("hang_detected", [False, True], ids=["normal", "hang_detected"])
def test_executor_stops_its_progress_thread_on_shutdown(monkeypatch, hang_detected):
    executor = _executor(monkeypatch, pp_size=2)
    exit_hooks = _record_atexit(monkeypatch)
    executor.start_worker()
    thread = executor._mpi_progress_thread
    try:
        assert thread is not None
        assert thread.is_running
        # The relay thread's pump is disabled, so the native thread registers
        # the only atexit callback.
        assert len(exit_hooks.registered) == 1
        assert _wait_for_probes(thread, 3) >= 3

        executor.hang_detector.detected.return_value = hang_detected
        executor.shutdown()

        assert executor._mpi_progress_thread is None
        assert not thread.is_running
        # A freshly accessed `thread.stop` is a new object that atexit would
        # not match, so shutdown must pass the registered one back.
        assert len(exit_hooks.unregistered) == 1
        assert exit_hooks.unregistered[0] is exit_hooks.registered[0]
    finally:
        if thread is not None:
            thread.stop()
        # The hang-detected path returns without joining these threads.
        executor.worker_thread.join()
        executor.broadcast_sample_state_handler.join()
        executor._broadcast_mpi_comm.Free()


@pytest.mark.parametrize("cause", ["disabled", "no_thread_multiple"])
def test_executor_runs_without_progress_thread_when_unavailable(monkeypatch, cause):
    executor = _executor(monkeypatch, pp_size=2)
    if cause == "disabled":
        monkeypatch.setenv(PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME, "0")
    else:
        monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_SERIALIZED)
    exit_hooks = _record_atexit(monkeypatch)
    executor.start_worker()
    try:
        assert executor._mpi_progress_thread is None
        assert exit_hooks.registered == []
        executor.shutdown()
        assert exit_hooks.unregistered == []
    finally:
        executor.worker_thread.join()
        executor.broadcast_sample_state_handler.join()
        executor._broadcast_mpi_comm.Free()


def test_single_stage_executor_starts_no_progress_thread(monkeypatch):
    executor = _executor(monkeypatch, pp_size=1)
    executor.start_worker()
    assert executor._mpi_progress_thread is None
    executor.shutdown()
