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
import os
import signal
import sys
import threading
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from utils.spawn_process import SpawnProcessContext, spawn_process, wait_forever

from tensorrt_llm._torch.visual_gen import executor as executor_module
from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient

_THREADING_EVENT = threading.Event


def _pre_set_event() -> threading.Event:
    event = _THREADING_EVENT()
    event.set()
    return event


def _process_is_running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", maxsplit=1)[1].split()[0] != "Z"


def _wait_for_process_exit(pid: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while _process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_is_running(pid)


def _pause() -> None:
    # Keep the fixture process alive without polling or holding a Python lock.
    # SIGKILL terminates the process while blocked and never returns; the loop
    # only handles an unrelated caught signal that returns normally.
    while True:
        signal.pause()


def _parent_bound_worker(
    parent_pid: int,
    lifecycle_context: SpawnProcessContext,
    **_kwargs,
) -> None:
    blocked_signals = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    if signal.SIGINT in blocked_signals or signal.SIGTERM in blocked_signals:
        raise RuntimeError("worker inherited blocked termination signal")
    executor_module._set_worker_parent_death_signal(parent_pid)
    executor_module._start_parent_process_watchdog(parent_pid)
    lifecycle_context.send("worker", os.getpid())
    _pause()


def _parent_death_worker(
    parent_pid: int,
    lifecycle_context: SpawnProcessContext,
) -> None:
    executor_module._set_worker_parent_death_signal(parent_pid)
    lifecycle_context.send("worker", os.getpid())
    _pause()


def _lightweight_background(client: DiffusionRemoteClient) -> None:
    client.event_loop_ready.set()
    client.shutdown_event.wait()


def _run_owned_worker_coordinator(
    lifecycle_context: SpawnProcessContext,
    fail_owner_wait_once: bool,
) -> None:
    parent_pid = os.getpid()
    context = executor_module._get_mp_context("spawn")

    if not fail_owner_wait_once:
        args = SimpleNamespace(parallel_config=SimpleNamespace(n_workers=1))
        startup_error = []
        clients = []
        worker_target = partial(
            _parent_bound_worker,
            lifecycle_context=lifecycle_context,
        )

        def construct_client_from_temporary_thread() -> None:
            try:
                with (
                    patch.object(
                        executor_module,
                        "_detect_external_launch",
                        return_value=None,
                    ),
                    patch.object(
                        executor_module,
                        "find_free_port",
                        # DiffusionRemoteClient requests the distributed master,
                        # request ZMQ, and response ZMQ ports in this order. The
                        # patched networking paths below never bind these ports;
                        # fixed placeholders keep construction deterministic.
                        side_effect=[29500, 29501, 29502],
                    ),
                    patch.object(
                        executor_module,
                        "get_ip_address",
                        return_value="127.0.0.1",
                    ),
                    patch.object(
                        executor_module,
                        "_get_mp_context",
                        return_value=context,
                    ),
                    patch.object(
                        executor_module,
                        "run_diffusion_worker",
                        worker_target,
                    ),
                    patch.object(
                        DiffusionRemoteClient,
                        "_serve_forever_thread",
                        _lightweight_background,
                    ),
                    patch.object(DiffusionRemoteClient, "_wait_ready"),
                    patch.object(executor_module, "_register_atexit"),
                ):
                    clients.append(DiffusionRemoteClient(args=args))
            except BaseException as error:
                startup_error.append(error)

        constructor = threading.Thread(target=construct_client_from_temporary_thread)
        constructor.start()
        constructor.join(timeout=30.0)
        if constructor.is_alive() or startup_error or not clients:
            raise RuntimeError(f"client startup failed: {startup_error!r}")
        lifecycle_context.send("constructor")
        _pause()

    worker = context.Process(
        target=_parent_bound_worker,
        kwargs={
            "parent_pid": parent_pid,
            "lifecycle_context": lifecycle_context,
        },
    )
    initial_signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    owner = executor_module._WorkerProcessOwner([worker], initial_signal_mask)
    owner_wait_failed = threading.Event()
    wait_for_release = owner._wait_for_release

    def fail_once() -> None:
        if not owner_wait_failed.is_set():
            owner_wait_failed.set()
            raise RuntimeError("injected owner wait failure")
        wait_for_release()

    owner._wait_for_release = fail_once
    startup_error = []

    def construct_from_temporary_thread() -> None:
        try:
            owner.start()
            owner.wait_for_spawn(timeout=30.0)
        except BaseException as error:
            startup_error.append(error)

    constructor = threading.Thread(target=construct_from_temporary_thread)
    constructor.start()
    constructor.join(timeout=30.0)
    if constructor.is_alive() or startup_error:
        raise RuntimeError(f"owner startup failed: {startup_error!r}")
    if not owner_wait_failed.wait(timeout=30.0):
        raise RuntimeError("owner wait failure was not observed")
    lifecycle_context.send("constructor")
    _pause()


def _run_process_watchdog(
    lifecycle_context: SpawnProcessContext,
    watched_pid: int,
) -> None:
    watchdog = executor_module._start_parent_process_watchdog(watched_pid)
    if watchdog is None:
        raise RuntimeError("process watchdog did not start")
    lifecycle_context.send("ready")
    _pause()


def _run_parent_death_coordinator(
    lifecycle_context: SpawnProcessContext,
) -> None:
    context = executor_module._get_mp_context("spawn")
    worker = context.Process(
        target=_parent_death_worker,
        args=(os.getpid(), lifecycle_context),
    )
    worker.start()
    worker.join()


def _assert_owned_worker_lifecycle(fail_owner_wait_once: bool) -> None:
    worker_pid = None
    try:
        with spawn_process(
            _run_owned_worker_coordinator,
            fail_owner_wait_once,
        ) as coordinator:
            messages = coordinator.receive_many("constructor", "worker")
            worker_pid = messages["worker"]

            # The temporary constructor thread is gone. The dedicated process
            # owner must keep the thread-scoped PDEATHSIG from firing.
            time.sleep(1.0)
            assert coordinator.is_alive
            assert _process_is_running(worker_pid)

            coordinator.kill()
            assert coordinator.wait() == -signal.SIGKILL
            _wait_for_process_exit(worker_pid)
    finally:
        if worker_pid is not None and _process_is_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


def test_worker_parent_death_signal_uses_sigkill() -> None:
    libc = MagicMock()
    libc.prctl.return_value = 0

    with (
        patch.object(executor_module, "_load_libc", return_value=libc),
        patch.object(executor_module, "_get_parent_process_id", return_value=123),
        patch.object(executor_module, "_kill_process") as kill_process,
    ):
        executor_module._set_worker_parent_death_signal(123)

    libc.prctl.assert_called_once_with(
        executor_module._PR_SET_PDEATHSIG,
        signal.SIGKILL,
        0,
        0,
        0,
    )
    kill_process.assert_not_called()


def test_worker_kills_itself_if_coordinator_exits_before_registration() -> None:
    libc = MagicMock()
    libc.prctl.return_value = 0

    with (
        patch.object(executor_module, "_load_libc", return_value=libc),
        patch.object(executor_module, "_get_parent_process_id", return_value=456),
        patch.object(executor_module, "_get_process_id", return_value=789),
        patch.object(executor_module, "_kill_process") as kill_process,
        pytest.raises(RuntimeError, match="coordinator exited before worker startup"),
    ):
        executor_module._set_worker_parent_death_signal(123)

    kill_process.assert_called_once_with(789, signal.SIGKILL)


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"),
    reason="pidfd process monitoring is Linux-specific",
)
def test_process_watchdog_kills_worker_when_watched_process_exits() -> None:
    with (
        spawn_process(wait_forever) as watched,
        spawn_process(_run_process_watchdog, watched.pid) as worker,
    ):
        watched.receive("ready")
        worker.receive("ready")

        watched.kill()
        assert watched.wait() == -signal.SIGKILL
        assert worker.wait() == -signal.SIGKILL


def test_worker_arms_both_parent_death_guards_before_initialization() -> None:
    class StopWorker(BaseException):
        pass

    events = []

    def set_parent_death_signal(parent_pid):
        events.append(("parent_death", parent_pid))

    def start_parent_process_watchdog(parent_pid):
        events.append(("process_watchdog", parent_pid))

    def set_log_level(log_level):
        events.append(("log_level", log_level))
        raise StopWorker

    with (
        patch.object(
            executor_module,
            "_set_worker_parent_death_signal",
            side_effect=set_parent_death_signal,
        ),
        patch.object(
            executor_module,
            "_start_parent_process_watchdog",
            side_effect=start_parent_process_watchdog,
        ),
        patch.object(executor_module.logger, "set_level", side_effect=set_log_level),
        pytest.raises(StopWorker),
    ):
        executor_module.run_diffusion_worker(
            rank=0,
            world_size=1,
            master_addr="127.0.0.1",
            master_port=29500,
            request_queue_addr=None,
            response_queue_addr=None,
            visual_gen_args=MagicMock(),
            parent_pid=123,
        )

    assert events == [
        ("parent_death", 123),
        ("process_watchdog", 123),
        ("log_level", "info"),
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-specific")
def test_worker_is_killed_when_coordinator_is_sigkilled() -> None:
    worker_pid = None
    try:
        with spawn_process(_run_parent_death_coordinator) as coordinator:
            worker_pid = coordinator.receive("worker")

            coordinator.kill()
            assert coordinator.wait() == -signal.SIGKILL

            _wait_for_process_exit(worker_pid)
    finally:
        if worker_pid is not None and _process_is_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-specific")
def test_temporary_constructor_thread_does_not_kill_owned_worker() -> None:
    _assert_owned_worker_lifecycle(fail_owner_wait_once=False)


@pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-specific")
def test_owner_wait_exception_does_not_kill_owned_worker() -> None:
    _assert_owned_worker_lifecycle(fail_owner_wait_once=True)


def test_cleanup_is_registered_before_waiting_for_ready() -> None:
    events = []
    process = MagicMock()
    context = MagicMock()
    context.Process.return_value = process
    owner = MagicMock()
    args = MagicMock()
    args.parallel_config.n_workers = 1
    previous_signal_mask = {signal.SIGHUP}

    with (
        patch.object(executor_module, "_detect_external_launch", return_value=None),
        patch.object(executor_module, "find_free_port", side_effect=[29500, 29501, 29502]),
        patch.object(executor_module, "get_ip_address", return_value="127.0.0.1"),
        patch.object(executor_module, "_get_mp_context", return_value=context),
        patch.object(executor_module, "_Thread") as thread_class,
        patch.object(executor_module, "_Event", side_effect=_pre_set_event),
        patch.object(
            executor_module,
            "_pthread_sigmask",
            return_value=previous_signal_mask,
        ),
        patch.object(executor_module, "_WorkerProcessOwner", return_value=owner) as owner_class,
        patch.object(executor_module, "_register_atexit") as register,
        patch.object(DiffusionRemoteClient, "_wait_ready") as wait_ready,
    ):
        thread_class.return_value = MagicMock()
        owner.start.side_effect = lambda: events.append("spawn")
        register.side_effect = lambda *args: events.append("register")
        wait_ready.side_effect = lambda: events.append("wait_ready")

        DiffusionRemoteClient(args=args)

    assert events == ["register", "spawn", "wait_ready"]
    owner_class.assert_called_once_with([process], previous_signal_mask)


def test_owner_restores_signal_mask_before_spawning() -> None:
    events = []
    process = MagicMock()
    initial_signal_mask = {signal.SIGHUP}

    def restore_signal_mask(how, mask) -> None:
        events.append(("signal_mask", how, mask))

    process.start.side_effect = lambda: events.append(("process_start",))
    with patch.object(
        executor_module,
        "_pthread_sigmask",
        side_effect=restore_signal_mask,
    ):
        owner = executor_module._WorkerProcessOwner([process], initial_signal_mask)
        owner.start()
        try:
            assert owner.wait_for_spawn(timeout=10.0)
        finally:
            owner.release_after_reap({id(process)})

    assert events == [
        ("signal_mask", signal.SIG_SETMASK, initial_signal_mask),
        ("process_start",),
    ]


def test_shutdown_waits_for_spawn_batch_before_reaping() -> None:
    start_entered = threading.Event()
    allow_start_to_finish = threading.Event()
    first_process = MagicMock()
    first_process.pid = None
    first_process.is_alive.return_value = False
    second_process = MagicMock()
    second_process.pid = None

    def start_first_process() -> None:
        start_entered.set()
        assert allow_start_to_finish.wait(timeout=10.0)
        first_process.pid = 123

    first_process.start.side_effect = start_first_process
    owner = executor_module._WorkerProcessOwner([first_process, second_process], set())
    owner.start()
    assert start_entered.wait(timeout=10.0)

    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    client.worker_processes = [first_process, second_process]
    client._worker_owner = owner
    client._ext_worker_thread = None

    shutdown_thread = threading.Thread(target=client.shutdown)
    shutdown_thread.start()
    assert owner._spawn_cancelled.wait(timeout=10.0)
    assert shutdown_thread.is_alive()
    first_process.join.assert_not_called()

    allow_start_to_finish.set()
    shutdown_thread.join(timeout=10.0)
    assert not shutdown_thread.is_alive()

    first_process.start.assert_called_once_with()
    second_process.start.assert_not_called()
    first_process.join.assert_called_once_with(timeout=executor_module.WORKER_TIMEOUT)
    second_process.join.assert_not_called()


def test_shutdown_bounds_inflight_spawn_and_owner_reaps_late_worker() -> None:
    spawn_blocked = threading.Event()
    release_spawn = threading.Event()
    started_process = MagicMock()
    started_process.pid = 123
    started_process.is_alive.return_value = False
    in_flight_process = MagicMock()
    in_flight_process.pid = None
    in_flight_process.is_alive.return_value = False

    def block_in_process_start() -> None:
        spawn_blocked.set()
        assert release_spawn.wait(timeout=10.0)
        in_flight_process.pid = 456

    in_flight_process.start.side_effect = block_in_process_start
    owner = executor_module._WorkerProcessOwner([started_process, in_flight_process], set())
    owner.start()
    assert spawn_blocked.wait(timeout=10.0)

    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    client.worker_processes = [started_process, in_flight_process]
    client._worker_owner = owner
    client._ext_worker_thread = None

    try:
        with (
            patch.object(executor_module, "WORKER_SPAWN_SHUTDOWN_TIMEOUT", 0.05),
            patch.object(executor_module, "THREAD_TIMEOUT", 0.05),
            patch.object(executor_module.logger, "error") as log_error,
        ):
            start_time = time.monotonic()
            client.shutdown()
            elapsed = time.monotonic() - start_time

        assert elapsed < 1.0
        started_process.join.assert_called_once_with(timeout=executor_module.WORKER_TIMEOUT)
        in_flight_process.join.assert_not_called()
        assert any(
            "spawn batch did not complete" in call.args[0] for call in log_error.call_args_list
        )
    finally:
        release_spawn.set()
        owner._thread.join(timeout=10.0)

    assert not owner._thread.is_alive()
    in_flight_process.join.assert_called_once_with(timeout=executor_module.WORKER_TIMEOUT)


def test_shutdown_skips_registered_process_that_never_started() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    worker = MagicMock()
    worker.pid = None
    client.worker_processes = [worker]
    client._ext_worker_thread = None

    client.shutdown()

    worker.join.assert_not_called()


def test_wait_ready_times_out_while_workers_are_alive() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client.completed_responses = {}
    client.worker_processes = [MagicMock()]
    client.worker_processes[0].is_alive.return_value = True
    client._ext_worker_thread = None

    loop = asyncio.new_event_loop()
    try:
        client.lock = asyncio.Lock()
        client.response_event = asyncio.Event()
        client._worker_ready_timeout = 0.0

        with pytest.raises(TimeoutError, match="did not become ready within 0s"):
            loop.run_until_complete(client._wait_ready_async())
    finally:
        loop.close()


def test_worker_ready_timeout_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_VISUAL_GEN_WORKER_READY_TIMEOUT", "7200")

    assert executor_module._get_worker_ready_timeout() == 7200.0


@pytest.mark.parametrize("value", ["", "abc", "-1", "inf"])
def test_worker_ready_timeout_rejects_invalid_values_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("TLLM_VISUAL_GEN_WORKER_READY_TIMEOUT", value)
    args = MagicMock()

    with patch.object(executor_module, "_get_mp_context") as get_mp_context:
        with pytest.raises(ValueError, match="must be a positive number of seconds"):
            DiffusionRemoteClient(args=args)

    get_mp_context.assert_not_called()


def test_wait_ready_timeout_shuts_down_workers() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    event_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=event_loop.run_forever)
    loop_thread.start()

    async def time_out() -> None:
        raise TimeoutError("startup timed out")

    try:
        client._event_loop = event_loop
        client._wait_ready_async = time_out
        client.shutdown = MagicMock()

        with pytest.raises(TimeoutError, match="startup timed out"):
            client._wait_ready()

        client.shutdown.assert_called_once_with()
    finally:
        event_loop.call_soon_threadsafe(event_loop.stop)
        loop_thread.join()
        event_loop.close()


def test_shutdown_is_idempotent() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    worker = MagicMock()
    worker.pid = 123
    worker.is_alive.return_value = False
    client.worker_processes = [worker]
    client._ext_worker_thread = None

    client.shutdown()
    client.shutdown()

    client.pending_requests.put.assert_called_once_with(None)
    client.background_thread.join.assert_called_once_with(timeout=executor_module.THREAD_TIMEOUT)
    worker.join.assert_called_once_with(timeout=executor_module.WORKER_TIMEOUT)


def test_concurrent_shutdown_waits_for_active_reap() -> None:
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    wait_entered = threading.Event()

    class TrackedEvent:
        def __init__(self) -> None:
            self._event = threading.Event()

        def set(self) -> None:
            self._event.set()

        def wait(self) -> bool:
            wait_entered.set()
            return self._event.wait()

        def is_set(self) -> bool:
            return self._event.is_set()

    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = TrackedEvent()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False

    def block_active_cleanup(*args, **kwargs) -> None:
        del args, kwargs
        cleanup_entered.set()
        assert release_cleanup.wait(timeout=10.0)

    client.background_thread.join.side_effect = block_active_cleanup
    worker = MagicMock()
    worker.pid = 123
    worker.is_alive.return_value = False
    client.worker_processes = [worker]
    client._worker_owner = None
    client._ext_worker_thread = None

    first_shutdown = threading.Thread(target=client.shutdown)
    second_shutdown = threading.Thread(target=client.shutdown)
    first_shutdown.start()
    assert cleanup_entered.wait(timeout=10.0)
    second_shutdown.start()

    assert wait_entered.wait(timeout=10.0)
    assert second_shutdown.is_alive()

    release_cleanup.set()
    first_shutdown.join(timeout=10.0)
    second_shutdown.join(timeout=10.0)

    assert not first_shutdown.is_alive()
    assert not second_shutdown.is_alive()
    assert client._shutdown_complete.is_set()
    client.pending_requests.put.assert_called_once_with(None)
    worker.join.assert_called_once_with(timeout=executor_module.WORKER_TIMEOUT)


def test_shutdown_defers_signals_until_workers_are_reaped() -> None:
    class ShutdownTermination(BaseException):
        pass

    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    worker = MagicMock()
    worker.pid = 123
    worker.is_alive.side_effect = [True, True]
    client.worker_processes = [worker]
    client._ext_worker_thread = None
    previous_signal_mask = {signal.SIGHUP}

    def pthread_sigmask(how, mask):
        if how == signal.SIG_BLOCK:
            assert tuple(mask) == executor_module._SHUTDOWN_SIGNALS
            return previous_signal_mask
        assert how == signal.SIG_SETMASK
        assert mask == previous_signal_mask
        worker.kill.assert_called_once_with()
        assert worker.join.call_count == 3
        raise ShutdownTermination

    with patch.object(executor_module, "_pthread_sigmask", side_effect=pthread_sigmask):
        with pytest.raises(ShutdownTermination):
            client.shutdown()

    assert client._shutdown_complete.is_set()
    worker.terminate.assert_called_once_with()
    worker.kill.assert_called_once_with()
