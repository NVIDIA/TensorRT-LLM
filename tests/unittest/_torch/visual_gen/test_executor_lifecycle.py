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
import ctypes
import errno
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import zmq
from utils.spawn_process import SpawnProcessContext, spawn_process

from tensorrt_llm._torch.visual_gen import executor as executor_module
from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient
from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

pytestmark = pytest.mark.cpu_only

_THREADING_EVENT = threading.Event
_COLD_SPAWN_TIMEOUT = 120.0


def _pre_set_event() -> threading.Event:
    event = _THREADING_EVENT()
    event.set()
    return event


def _process_is_running(pid: int) -> bool:
    state = _process_state(pid)
    return state is not None and state != "Z"


def _process_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat.rsplit(")", maxsplit=1)[1].split()[0]


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


def _exit_immediately(exitcode: int) -> None:
    os._exit(exitcode)


def _gpu_bound_worker(
    rank: int,
    parent_pid: int,
    ready_queue,
) -> None:
    executor_module._start_coordinator_watchdog(parent_pid)
    torch.cuda.set_device(rank)
    # Keep a live CUDA allocation on each device while the parent injects a
    # process failure. NCCL is intentionally not initialized here: killing a
    # rank inside an active NCCL group can wedge NVIDIA UVM teardown and poison
    # the shared CI node, which tests driver recovery rather than this client's
    # worker-containment behavior.
    allocation = torch.empty(1024, device=f"cuda:{rank}")
    torch.cuda.synchronize(rank)
    ready_queue.put(rank)
    ready_queue.close()
    ready_queue.join_thread()
    assert allocation.is_cuda
    _pause()


def _supervised_pause_worker(parent_pid: int, ready_queue) -> None:
    executor_module._start_coordinator_watchdog(parent_pid)
    ready_queue.put(os.getpid())
    ready_queue.close()
    ready_queue.join_thread()
    _pause()


def _parent_bound_worker(
    parent_pid: int,
    lifecycle_context: SpawnProcessContext,
    **_kwargs,
) -> None:
    executor_module._start_coordinator_watchdog(parent_pid)
    lifecycle_context.send("worker", os.getpid())
    _pause()


def _gil_holding_parent_bound_worker(
    parent_pid: int,
    lifecycle_context: SpawnProcessContext,
) -> None:
    executor_module._start_coordinator_watchdog(parent_pid)
    lifecycle_context.send("worker", os.getpid())
    # Flush the message before blocking the Python interpreter in a native
    # call that keeps the GIL. Only the C++ watchdog can run after this point.
    lifecycle_context.close_sender()
    pause = ctypes.PyDLL(None).pause
    pause.argtypes = []
    pause.restype = ctypes.c_int
    pause()


def _delayed_watchdog_worker(
    parent_pid: int,
    lifecycle_context: SpawnProcessContext,
) -> None:
    lifecycle_context.send("worker", os.getpid())
    lifecycle_context.close_sender()
    time.sleep(1.0)
    try:
        executor_module._start_coordinator_watchdog(parent_pid)
    except RuntimeError:
        return
    _pause()


def _forced_polling_parent_bound_worker(
    parent_pid: int,
    lifecycle_context: SpawnProcessContext,
    pidfd_error_code: int,
    **_kwargs,
) -> None:
    from tensorrt_llm.bindings.internal.testing import start_coordinator_watchdog_with_pidfd_error

    warning = start_coordinator_watchdog_with_pidfd_error(parent_pid, pidfd_error_code)
    lifecycle_context.send(
        "watchdog",
        {
            "pid": os.getpid(),
            "warning": warning,
        },
    )
    _pause()


def _lightweight_background(client: DiffusionRemoteClient) -> None:
    client.event_loop_ready.set()
    client.shutdown_event.wait()


def _run_temporary_constructor_coordinator(
    lifecycle_context: SpawnProcessContext,
    worker_entrypoint: Callable[..., None] = _parent_bound_worker,
) -> None:
    context = executor_module._get_mp_context("spawn")
    args = SimpleNamespace(parallel_config=SimpleNamespace(n_workers=1))
    startup_error = []
    clients = []
    worker_target = partial(
        worker_entrypoint,
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


def _run_parent_death_coordinator(
    lifecycle_context: SpawnProcessContext,
    worker_target: Callable[..., None] = _parent_bound_worker,
) -> None:
    context = executor_module._get_mp_context("spawn")
    worker = context.Process(
        target=worker_target,
        args=(os.getpid(), lifecycle_context),
    )
    worker.start()
    worker.join()


def _run_worker_containment_coordinator(
    lifecycle_context: SpawnProcessContext,
    worker_count: int,
    begin_monitoring,
) -> None:
    context = executor_module._get_mp_context("spawn")
    ready_queue = context.Queue()
    workers = [
        context.Process(
            target=_supervised_pause_worker,
            args=(os.getpid(), ready_queue),
        )
        for _ in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    try:
        ready_deadline = time.monotonic() + _COLD_SPAWN_TIMEOUT
        worker_pids = [
            ready_queue.get(timeout=max(0.0, ready_deadline - time.monotonic())) for _ in workers
        ]
        lifecycle_context.send("workers", worker_pids)

        if not begin_monitoring.wait(timeout=30.0):
            raise TimeoutError("parent did not release worker monitoring")

        client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
        client.worker_processes = workers
        client._worker_spawner = executor_module._WorkerProcessSpawner(workers)
        client._ext_worker_thread = None
        client._monitor_worker_liveness = True
        client._worker_failure = None
        client._shutdown_started = False
        client.shutdown_event = threading.Event()
        client.response_event = threading.Event()

        containment_deadline = time.monotonic() + 10.0
        while client._worker_failure is None and time.monotonic() < containment_deadline:
            client._check_worker_liveness()
            time.sleep(0.001)
        if client._worker_failure is None:
            raise TimeoutError("coordinator did not detect the killed workers")

        lifecycle_context.send(
            "contained",
            {
                "failure": client._worker_failure,
                "exitcodes": [worker.exitcode for worker in workers],
            },
        )
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=10.0)
        ready_queue.close()
        ready_queue.join_thread()


def _run_worker_watchdog_coordinator(
    lifecycle_context: SpawnProcessContext,
    worker_count: int,
) -> None:
    """Spawn supervised workers, then remain alive until the test kills us."""
    context = executor_module._get_mp_context("spawn")
    ready_queue = context.Queue()
    workers = [
        context.Process(
            target=_supervised_pause_worker,
            args=(os.getpid(), ready_queue),
        )
        for _ in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    ready_deadline = time.monotonic() + _COLD_SPAWN_TIMEOUT
    worker_pids = [
        ready_queue.get(timeout=max(0.0, ready_deadline - time.monotonic())) for _ in workers
    ]
    ready_queue.close()
    ready_queue.join_thread()
    lifecycle_context.send("workers", worker_pids)
    lifecycle_context.close_sender()
    _pause()


def _run_sigint_during_shutdown(lifecycle_context: SpawnProcessContext) -> None:
    # Batch launchers may start the test process with SIGINT ignored or
    # blocked. Establish the ordinary interactive Python disposition that
    # this scenario is intended to exercise before creating helper threads.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})

    context = executor_module._get_mp_context("spawn")
    worker = context.Process(target=_pause)
    worker.start()

    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client._shutdown_error = None
    client._shutdown_thread = None
    client.pending_requests = executor_module.queue.Queue()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    client.shutdown_event = threading.Event()
    client.worker_processes = [worker]
    client._worker_spawner = None
    client._ext_worker_thread = None

    reap_started = threading.Event()
    real_reap_worker_process = executor_module._reap_worker_process

    def reap_worker_process(process) -> bool:
        reap_started.set()
        return real_reap_worker_process(process)

    def interrupt_shutdown() -> None:
        if not reap_started.wait(timeout=10.0):
            return
        os.kill(os.getpid(), signal.SIGINT)

    interrupt_thread = threading.Thread(target=interrupt_shutdown, daemon=True)
    try:
        with patch.object(
            executor_module,
            "_reap_worker_process",
            side_effect=reap_worker_process,
        ):
            interrupt_thread.start()
            try:
                client.shutdown()
            except KeyboardInterrupt:
                lifecycle_context.send(
                    "shutdown",
                    {
                        "complete": client._shutdown_complete.is_set(),
                        "worker_exitcode": worker.exitcode,
                    },
                )
            else:
                raise RuntimeError("SIGINT was not delivered after worker reap started")
    finally:
        interrupt_thread.join(timeout=10.0)
        if worker.is_alive():
            worker.kill()
        worker.join(timeout=10.0)


def _assert_temporary_constructor_worker_lifecycle() -> None:
    worker_pid = None
    try:
        with spawn_process(_run_temporary_constructor_coordinator) as coordinator:
            messages = coordinator.receive_many("constructor", "worker")
            worker_pid = messages["worker"]

            # The temporary constructor thread is gone. The process-scoped
            # watchdog must follow the coordinator rather than that thread.
            time.sleep(1.0)
            assert coordinator.is_alive
            assert _process_is_running(worker_pid)

            coordinator.kill()
            assert coordinator.wait() == -signal.SIGKILL
            _wait_for_process_exit(worker_pid)
    finally:
        if worker_pid is not None and _process_is_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


def test_worker_starts_native_watchdog_before_initialization() -> None:
    class StopWorker(BaseException):
        pass

    events = []

    def start_coordinator_watchdog(parent_pid):
        events.append(("coordinator_watchdog", parent_pid))
        return "using parent-PID polling fallback"

    def log_warning(message):
        events.append(("warning", message))

    def set_log_level(log_level):
        events.append(("log_level", log_level))
        raise StopWorker

    with (
        patch.object(
            executor_module,
            "_start_coordinator_watchdog",
            side_effect=start_coordinator_watchdog,
        ),
        patch.object(executor_module.logger, "warning", side_effect=log_warning),
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
        ("coordinator_watchdog", 123),
        (
            "warning",
            "VisualGen worker coordinator watchdog: using parent-PID polling fallback",
        ),
        ("log_level", "info"),
    ]


def test_worker_watchdog_failure_is_fatal_before_initialization() -> None:
    with (
        patch.object(
            executor_module,
            "_start_coordinator_watchdog",
            side_effect=RuntimeError("watchdog thread unavailable"),
        ),
        patch.object(executor_module.logger, "set_level") as set_log_level,
        pytest.raises(RuntimeError, match="watchdog thread unavailable"),
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

    set_log_level.assert_not_called()


def test_worker_failure_propagates_to_external_launcher() -> None:
    with (
        patch.object(
            executor_module.logger,
            "set_level",
            side_effect=RuntimeError("worker initialization failed"),
        ),
        patch.object(executor_module.logger, "error") as log_error,
        patch.object(executor_module.traceback, "print_exc") as print_exc,
        pytest.raises(RuntimeError, match="worker initialization failed"),
    ):
        executor_module.run_diffusion_worker(
            rank=1,
            world_size=2,
            master_addr="127.0.0.1",
            master_port=29500,
            request_queue_addr=None,
            response_queue_addr=None,
            visual_gen_args=MagicMock(),
        )

    log_error.assert_called_once_with("Worker failed: worker initialization failed")
    print_exc.assert_called_once_with()


@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
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


@pytest.mark.parametrize(
    ("pidfd_error_code", "error_name"),
    [
        (errno.ENOSYS, "ENOSYS"),
        (errno.EPERM, "EPERM"),
    ],
)
@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
def test_worker_uses_polling_watchdog_when_pidfd_is_unavailable(
    pidfd_error_code: int,
    error_name: str,
) -> None:
    worker_pid = None
    worker_target = partial(
        _forced_polling_parent_bound_worker,
        pidfd_error_code=pidfd_error_code,
    )
    try:
        with spawn_process(_run_temporary_constructor_coordinator, worker_target) as coordinator:
            messages = coordinator.receive_many("constructor", "watchdog")
            watchdog = messages["watchdog"]
            worker_pid = watchdog["pid"]
            assert error_name in watchdog["warning"]
            assert "native 1-second parent-PID polling fallback" in watchdog["warning"]

            # Both the finite worker-spawner thread and the temporary client
            # constructor thread have exited. Parent-PID polling follows the
            # coordinator process, so neither thread exit may kill the worker.
            # Wait through more than one polling interval before asserting.
            time.sleep(2.0)
            assert coordinator.is_alive
            assert _process_is_running(worker_pid)

            coordinator.kill()
            assert coordinator.wait() == -signal.SIGKILL

            _wait_for_process_exit(worker_pid)
    finally:
        if worker_pid is not None and _process_is_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
def test_worker_watchdog_does_not_hide_unexpected_pidfd_errors() -> None:
    from tensorrt_llm.bindings.internal.testing import start_coordinator_watchdog_with_pidfd_error

    with pytest.raises(RuntimeError, match="pidfd_open for VisualGen coordinator failed"):
        start_coordinator_watchdog_with_pidfd_error(os.getppid(), errno.EMFILE)


@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
def test_native_watchdog_does_not_require_python_gil() -> None:
    worker_pid = None
    try:
        with spawn_process(
            _run_parent_death_coordinator,
            _gil_holding_parent_bound_worker,
        ) as coordinator:
            worker_pid = coordinator.receive("worker")
            time.sleep(0.5)

            coordinator.kill()
            assert coordinator.wait() == -signal.SIGKILL

            _wait_for_process_exit(worker_pid)
    finally:
        if worker_pid is not None and _process_is_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
def test_worker_exits_if_coordinator_dies_before_watchdog_registration() -> None:
    worker_pid = None
    try:
        with spawn_process(
            _run_parent_death_coordinator,
            _delayed_watchdog_worker,
        ) as coordinator:
            worker_pid = coordinator.receive("worker")

            coordinator.kill()
            assert coordinator.wait() == -signal.SIGKILL

            _wait_for_process_exit(worker_pid)
    finally:
        if worker_pid is not None and _process_is_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.parametrize("killed_worker_count", [1, 2])
@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
def test_sigkill_workers_contains_remaining_group(killed_worker_count: int) -> None:
    context = executor_module._get_mp_context("spawn")
    begin_monitoring = context.Event()
    worker_pids = []
    try:
        with spawn_process(
            _run_worker_containment_coordinator,
            killed_worker_count + 1,
            begin_monitoring,
        ) as coordinator:
            worker_pids = coordinator.receive("workers", timeout=_COLD_SPAWN_TIMEOUT)
            killed_worker_pids = worker_pids[:killed_worker_count]
            for worker_pid in killed_worker_pids:
                os.kill(worker_pid, signal.SIGKILL)

            death_deadline = time.monotonic() + 10.0
            for worker_pid in killed_worker_pids:
                while _process_state(worker_pid) != "Z" and time.monotonic() < death_deadline:
                    time.sleep(0.01)
                assert _process_state(worker_pid) == "Z"

            begin_monitoring.set()
            result = coordinator.receive("contained", timeout=30.0)
            assert result["exitcodes"] == [-signal.SIGKILL] * len(worker_pids)
            # is_alive() harvests child status through waitpid(WNOHANG). Two
            # processes already visible as /proc zombies can become waitable
            # on adjacent checks, so the coordinator may latch the first death
            # and classify the second as a live rank to contain. The failure
            # must name an injected death, never the untouched survivor.
            assert any(
                f"pid={worker_pid}, exitcode=-9" in result["failure"]
                for worker_pid in killed_worker_pids
            )
            for worker_pid in worker_pids[killed_worker_count:]:
                assert f"pid={worker_pid}" not in result["failure"]
            assert coordinator.wait(timeout=30.0) == 0
            for worker_pid in worker_pids:
                _wait_for_process_exit(worker_pid)
    finally:
        begin_monitoring.set()
        for worker_pid in worker_pids:
            if _process_is_running(worker_pid):
                os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
def test_sigkill_worker_and_coordinator_kills_remaining_workers() -> None:
    worker_pids = []
    try:
        with spawn_process(
            _run_worker_watchdog_coordinator,
            2,
        ) as coordinator:
            worker_pids = coordinator.receive("workers", timeout=_COLD_SPAWN_TIMEOUT)
            os.kill(worker_pids[0], signal.SIGKILL)
            coordinator.kill()
            assert coordinator.wait(timeout=30.0) == -signal.SIGKILL
            for worker_pid in worker_pids:
                _wait_for_process_exit(worker_pid)
    finally:
        for worker_pid in worker_pids:
            if _process_is_running(worker_pid):
                os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="native parent monitoring is Linux-specific")
def test_temporary_constructor_thread_does_not_kill_worker() -> None:
    _assert_temporary_constructor_worker_lifecycle()


def test_cleanup_is_registered_before_waiting_for_ready() -> None:
    events = []
    process = MagicMock()
    context = MagicMock()
    context.Process.return_value = process
    spawner = MagicMock()
    args = MagicMock()
    args.parallel_config.n_workers = 1

    with (
        patch.object(executor_module, "_detect_external_launch", return_value=None),
        patch.object(executor_module, "find_free_port", side_effect=[29500, 29501, 29502]),
        patch.object(executor_module, "get_ip_address", return_value="127.0.0.1"),
        patch.object(executor_module, "_get_mp_context", return_value=context),
        patch.object(executor_module, "_Thread") as thread_class,
        patch.object(executor_module, "_Event", side_effect=_pre_set_event),
        patch.object(
            executor_module,
            "_WorkerProcessSpawner",
            return_value=spawner,
        ) as spawner_class,
        patch.object(executor_module, "_register_atexit") as register,
        patch.object(DiffusionRemoteClient, "_wait_ready") as wait_ready,
    ):
        thread_class.return_value = MagicMock()
        spawner.start.side_effect = lambda: events.append("spawn")
        register.side_effect = lambda *args: events.append("register")
        wait_ready.side_effect = lambda: events.append("wait_ready")

        DiffusionRemoteClient(args=args)

    assert events == ["register", "spawn", "wait_ready"]
    spawner_class.assert_called_once_with([process])
    spawner.wait_for_spawn.assert_called_once_with()


def test_worker_spawner_exits_after_spawn_batch() -> None:
    process = MagicMock()
    spawner = executor_module._WorkerProcessSpawner([process])

    spawner.start()
    assert spawner.wait_for_spawn(timeout=10.0)
    spawner._thread.join(timeout=10.0)

    assert not spawner._thread.is_alive()
    process.start.assert_called_once_with()


def test_worker_spawner_propagates_spawn_failure() -> None:
    process = MagicMock()
    process.start.side_effect = RuntimeError("spawn failed")
    spawner = executor_module._WorkerProcessSpawner([process])

    spawner.start()
    with pytest.raises(RuntimeError, match="spawn failed"):
        spawner.wait_for_spawn(timeout=10.0)
    spawner._thread.join(timeout=10.0)

    assert not spawner._thread.is_alive()


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
    spawner = executor_module._WorkerProcessSpawner([first_process, second_process])
    spawner.start()
    assert start_entered.wait(timeout=10.0)

    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    client.worker_processes = [first_process, second_process]
    client._worker_spawner = spawner
    client._ext_worker_thread = None

    shutdown_thread = threading.Thread(target=client.shutdown)
    shutdown_thread.start()
    assert spawner._spawn_cancelled.wait(timeout=10.0)
    assert shutdown_thread.is_alive()
    first_process.join.assert_not_called()

    allow_start_to_finish.set()
    shutdown_thread.join(timeout=10.0)
    assert not shutdown_thread.is_alive()
    spawner._thread.join(timeout=10.0)
    assert not spawner._thread.is_alive()

    first_process.start.assert_called_once_with()
    second_process.start.assert_not_called()
    first_process.join.assert_called_once_with(timeout=executor_module.WORKER_TIMEOUT)
    second_process.join.assert_not_called()


def test_shutdown_bounds_inflight_spawn_and_spawner_reaps_late_worker() -> None:
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
    spawner = executor_module._WorkerProcessSpawner([started_process, in_flight_process])
    spawner.start()
    assert spawn_blocked.wait(timeout=10.0)

    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client.pending_requests = MagicMock()
    client.background_thread = MagicMock()
    client.background_thread.is_alive.return_value = False
    client.worker_processes = [started_process, in_flight_process]
    client._worker_spawner = spawner
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
        spawner._thread.join(timeout=10.0)

    assert not spawner._thread.is_alive()
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


def test_worker_death_during_request_send_is_contained() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    dead_worker = MagicMock()
    dead_worker.pid = 123
    dead_worker.exitcode = -signal.SIGSEGV
    dead_worker.is_alive.return_value = False
    live_worker = MagicMock()
    live_worker.pid = 456
    live_worker.is_alive.return_value = True
    client.worker_processes = [dead_worker, live_worker]
    client._worker_spawner = MagicMock()
    client._ext_worker_thread = None
    client._monitor_worker_liveness = True
    client._worker_failure = None
    client._shutdown_started = False
    client.shutdown_event = threading.Event()
    client.response_event = MagicMock()
    client.pending_requests = executor_module.queue.Queue()
    request = SimpleNamespace(request_id=789)
    client.pending_requests.put(request)
    client._request_to_send = None
    client.requests_ipc = MagicMock()
    client.requests_ipc.put_nowait.side_effect = zmq.Again()
    client._iter_stats = MagicMock()

    client._process_requests()

    assert client._worker_failure == (
        "DiffusionClient: local worker processes exited: pid=123, exitcode=-11"
    )
    assert client._request_to_send is request
    dead_worker.kill.assert_not_called()
    live_worker.kill.assert_called_once_with()
    client._worker_spawner.reap_started_processes.assert_called_once_with()
    assert client.shutdown_event.is_set()
    client.response_event.set.assert_called_once_with()
    client._iter_stats.record_request_started.assert_not_called()


@pytest.mark.skipif(sys.platform != "linux", reason="zombie state is observed through /proc")
def test_worker_failure_reaps_dead_and_contained_processes() -> None:
    context = executor_module._get_mp_context("spawn")
    dead_worker = context.Process(target=_exit_immediately, args=(7,))
    live_worker = context.Process(target=_pause)
    dead_worker.start()
    live_worker.start()
    assert dead_worker.pid is not None

    try:
        # A spawn child imports the full test module before reaching its
        # target. On a cold network filesystem that can take substantially
        # longer than the lifecycle operation under test. Wait until the child
        # has reached os._exit() before invoking the coordinator's liveness
        # check, so cold import time is not charged to containment.
        deadline = time.monotonic() + _COLD_SPAWN_TIMEOUT
        while _process_state(dead_worker.pid) != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _process_state(dead_worker.pid) == "Z"

        client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
        client.worker_processes = [dead_worker, live_worker]
        client._worker_spawner = executor_module._WorkerProcessSpawner(client.worker_processes)
        client._ext_worker_thread = None
        client._monitor_worker_liveness = True
        client._worker_failure = None
        client._shutdown_started = False
        client.shutdown_event = threading.Event()
        client.response_event = threading.Event()

        # A process can become observable as a zombie just before
        # waitpid(WNOHANG), used by multiprocessing.Process.is_alive(), can
        # harvest it. Production checks every event-loop tick, so exercise
        # that same bounded retry behavior rather than assuming one tick.
        deadline = time.monotonic() + 10.0
        while client._worker_failure is None and time.monotonic() < deadline:
            client._check_worker_liveness()
            time.sleep(0.001)

        assert client._worker_failure == (
            f"DiffusionClient: local worker processes exited: pid={dead_worker.pid}, exitcode=7"
        )
        assert dead_worker.exitcode == 7
        assert live_worker.exitcode == -signal.SIGKILL
        assert _process_state(dead_worker.pid) is None
        assert _process_state(live_worker.pid) is None
    finally:
        for worker in (dead_worker, live_worker):
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=10.0)


def test_worker_failure_shutdown_does_not_wait_for_thread_timeout() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    dead_worker = MagicMock()
    dead_worker.pid = 123
    dead_worker.exitcode = -signal.SIGKILL
    dead_worker.is_alive.return_value = False
    client.worker_processes = [dead_worker]
    client._worker_spawner = MagicMock()
    client._ext_worker_thread = None
    client._monitor_worker_liveness = True
    client._worker_failure = None
    client._shutdown_lock = threading.Lock()
    client._shutdown_started = False
    client._shutdown_complete = threading.Event()
    client._shutdown_error = None
    client._shutdown_thread = None
    client.pending_requests = executor_module.queue.Queue()
    client.shutdown_event = threading.Event()
    client.response_event = threading.Event()
    client.event_loop_ready = threading.Event()
    client._init_ipc = MagicMock(return_value=True)
    client._cleanup_ipc = MagicMock()
    client.background_thread = threading.Thread(target=client._serve_forever_thread)
    client.background_thread.start()

    assert client.shutdown_event.wait(timeout=10.0)
    assert client._worker_failure == (
        "DiffusionClient: local worker processes exited: pid=123, exitcode=-9"
    )

    with patch.object(executor_module, "THREAD_TIMEOUT", 1.0):
        start_time = time.monotonic()
        client.shutdown()
        elapsed = time.monotonic() - start_time

    assert elapsed < 1.0
    assert not client.background_thread.is_alive()
    client._cleanup_ipc.assert_called_once_with()


@pytest.mark.gpu4
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="native parent monitoring and /proc are Linux-specific",
)
def test_sigkill_one_worker_contains_real_multi_gpu_group() -> None:
    world_size = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")

    context = executor_module._get_mp_context("spawn")
    ready_queue = context.Queue()
    parent_pid = os.getpid()
    workers = [
        context.Process(
            target=_gpu_bound_worker,
            args=(
                rank,
                parent_pid,
                ready_queue,
            ),
        )
        for rank in range(world_size)
    ]
    for worker in workers:
        worker.start()

    try:
        ready_deadline = time.monotonic() + _COLD_SPAWN_TIMEOUT
        ready_ranks = {
            ready_queue.get(timeout=max(0.0, ready_deadline - time.monotonic()))
            for _ in range(world_size)
        }
        assert ready_ranks == set(range(world_size))

        failed_worker = workers[0]
        assert failed_worker.pid is not None
        os.kill(failed_worker.pid, signal.SIGKILL)
        deadline = time.monotonic() + 10.0
        while _process_state(failed_worker.pid) != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _process_state(failed_worker.pid) == "Z"

        client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
        client.worker_processes = workers
        client._worker_spawner = executor_module._WorkerProcessSpawner(workers)
        client._ext_worker_thread = None
        client._monitor_worker_liveness = True
        client._worker_failure = None
        client._shutdown_started = False
        client.shutdown_event = threading.Event()
        client.response_event = threading.Event()

        deadline = time.monotonic() + 10.0
        while client._worker_failure is None and time.monotonic() < deadline:
            client._check_worker_liveness()
            time.sleep(0.001)

        assert client._worker_failure == (
            f"DiffusionClient: local worker processes exited: pid={failed_worker.pid}, exitcode=-9"
        )
        assert failed_worker.exitcode == -signal.SIGKILL
        for worker in workers[1:]:
            assert worker.exitcode == -signal.SIGKILL
        for worker in workers:
            assert worker.pid is not None
            assert _process_state(worker.pid) is None
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=10.0)
        ready_queue.close()
        ready_queue.join_thread()


def test_worker_failure_completes_pending_response_with_error() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client.completed_responses = {}
    client._worker_failure = "DiffusionClient: local worker process exited"

    loop = asyncio.new_event_loop()
    try:
        client.lock = asyncio.Lock()
        client.response_event = asyncio.Event()
        response = loop.run_until_complete(client.await_responses(123))
    finally:
        loop.close()

    assert response.request_id == 123
    assert response.error_msg == client._worker_failure


def test_worker_failure_result_remains_available_until_shutdown() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._worker_failure = "DiffusionClient: local worker processes exited: pid=123, exitcode=-11"
    client._shutdown_started = False
    client.shutdown_event = threading.Event()
    client.shutdown_event.set()
    client.event_loop_ready = threading.Event()
    client.completed_responses = {}
    client._init_ipc = MagicMock(return_value=True)
    client._cleanup_ipc = MagicMock()
    client.background_thread = threading.Thread(target=client._serve_forever_thread)
    client.background_thread.start()

    try:
        assert client.event_loop_ready.wait(timeout=10.0)
        result = VisualGenResult(request_id=123, executor=client)

        with pytest.raises(
            RuntimeError,
            match="Generation failed: DiffusionClient: local worker processes exited: "
            "pid=123, exitcode=-11",
        ):
            result.result(timeout=10.0)

        assert client.background_thread.is_alive()
    finally:
        client._shutdown_started = True
        client.background_thread.join(timeout=10.0)

    assert not client.background_thread.is_alive()
    client._cleanup_ipc.assert_called_once_with()


def test_worker_failure_sync_response_does_not_require_event_loop() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._worker_failure = "DiffusionClient: local worker processes exited: pid=123, exitcode=-11"
    client._event_loop = asyncio.new_event_loop()
    client._event_loop.close()

    response = client.await_responses_sync(123)

    assert response.request_id == 123
    assert response.error_msg == client._worker_failure


def test_wait_ready_reports_worker_monitor_failure() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._worker_failure = "DiffusionClient: local worker process exited"

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="local worker process exited"):
            loop.run_until_complete(client._wait_ready_async())
    finally:
        loop.close()


def test_wait_ready_failure_shuts_down_workers() -> None:
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    event_loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=event_loop.run_forever)
    loop_thread.start()

    async def fail() -> None:
        raise RuntimeError("startup failed")

    try:
        client._event_loop = event_loop
        client._wait_ready_async = fail
        client.shutdown = MagicMock()

        with pytest.raises(RuntimeError, match="startup failed"):
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
    client._worker_spawner = None
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


@pytest.mark.skipif(sys.platform != "linux", reason="signal delivery behavior is POSIX-specific")
def test_shutdown_defers_sigint_until_worker_is_reaped() -> None:
    with spawn_process(_run_sigint_during_shutdown) as scenario:
        result = scenario.receive("shutdown", timeout=_COLD_SPAWN_TIMEOUT)
        assert result == {
            "complete": True,
            "worker_exitcode": -signal.SIGTERM,
        }
        assert scenario.wait() == 0
