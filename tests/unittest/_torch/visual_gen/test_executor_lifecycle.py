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
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.visual_gen import executor as executor_module
from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient

_THREADING_EVENT = threading.Event
_LIFECYCLE_HELPER = Path(__file__).with_name("_executor_lifecycle_helper.py")
_SUBPROCESS_TIMEOUT = 180.0


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


def _start_lifecycle_helper(*args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(_LIFECYCLE_HELPER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _wait_for_helper_messages(
    process: subprocess.Popen,
    required_prefixes: tuple[str, ...],
) -> dict[str, str]:
    assert process.stdout is not None
    messages = {}
    output = []
    deadline = time.monotonic() + _SUBPROCESS_TIMEOUT
    while set(messages) != set(required_prefixes) and time.monotonic() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if not ready:
            if process.poll() is not None:
                break
            continue
        line = process.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="replace").rstrip()
        output.append(decoded_line)
        if decoded_line.startswith("error:"):
            pytest.fail(f"lifecycle helper failed: {decoded_line}\n" + "\n".join(output))
        for prefix in required_prefixes:
            if decoded_line.startswith(prefix):
                messages[prefix] = decoded_line.removeprefix(prefix)

    missing = set(required_prefixes) - set(messages)
    if missing:
        pytest.fail(
            f"lifecycle helper did not report {sorted(missing)}; "
            f"returncode={process.poll()}, output={output}"
        )
    return messages


def _kill_process_group(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)
    if process.stdout is not None:
        process.stdout.close()


def _assert_owned_worker_lifecycle(fail_owner_wait_once: bool) -> None:
    coordinator = _start_lifecycle_helper(
        "owned-worker",
        str(int(fail_owner_wait_once)),
    )
    worker_pid = None
    try:
        messages = _wait_for_helper_messages(
            coordinator,
            ("constructor:done", "worker:"),
        )
        worker_pid = int(messages["worker:"])

        # The temporary constructor thread is gone. The dedicated process
        # owner must keep the thread-scoped PDEATHSIG from firing.
        time.sleep(1.0)
        assert coordinator.poll() is None
        assert _process_is_running(worker_pid)

        coordinator.kill()
        assert coordinator.wait(timeout=_SUBPROCESS_TIMEOUT) == -signal.SIGKILL
        _wait_for_process_exit(worker_pid)
    finally:
        _kill_process_group(coordinator)
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
    watched = subprocess.Popen(
        [sys.executable, "-c", "import signal; signal.pause()"],
        start_new_session=True,
    )
    worker = _start_lifecycle_helper("process-watchdog", str(watched.pid))
    try:
        _wait_for_helper_messages(worker, ("ready",))

        watched.kill()
        assert watched.wait(timeout=_SUBPROCESS_TIMEOUT) == -signal.SIGKILL
        assert worker.wait(timeout=_SUBPROCESS_TIMEOUT) == -signal.SIGKILL
    finally:
        _kill_process_group(watched)
        _kill_process_group(worker)


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
    coordinator = _start_lifecycle_helper("parent-death")
    worker_pid = None
    try:
        messages = _wait_for_helper_messages(coordinator, ("worker:",))
        worker_pid = int(messages["worker:"])

        coordinator.kill()
        assert coordinator.wait(timeout=_SUBPROCESS_TIMEOUT) == -signal.SIGKILL

        _wait_for_process_exit(worker_pid)
    finally:
        _kill_process_group(coordinator)
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
