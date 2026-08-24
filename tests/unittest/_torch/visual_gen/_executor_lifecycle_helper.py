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

import os
import signal
import sys
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from tensorrt_llm._torch.visual_gen import executor as executor_module  # noqa: E402
from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient  # noqa: E402

_STARTUP_TIMEOUT = 180.0


def _emit(message: str) -> None:
    print(message, flush=True)


def _pause() -> None:
    while True:
        signal.pause()


def _parent_bound_worker(parent_pid: int, **_kwargs) -> None:
    blocked_signals = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    if signal.SIGINT in blocked_signals or signal.SIGTERM in blocked_signals:
        _emit("error:worker inherited blocked termination signal")
        os._exit(2)
    executor_module._set_worker_parent_death_signal(parent_pid)
    executor_module._start_parent_process_watchdog(parent_pid)
    _emit(f"worker:{os.getpid()}")
    _pause()


def _parent_death_worker(parent_pid: int) -> None:
    executor_module._set_worker_parent_death_signal(parent_pid)
    _emit(f"worker:{os.getpid()}")
    _pause()


def _lightweight_background(client: DiffusionRemoteClient) -> None:
    client.event_loop_ready.set()
    client.shutdown_event.wait()


def _run_owned_worker_coordinator(fail_owner_wait_once: bool) -> None:
    parent_pid = os.getpid()
    context = executor_module._get_mp_context("spawn")

    if not fail_owner_wait_once:
        args = SimpleNamespace(parallel_config=SimpleNamespace(n_workers=1))
        startup_error = []
        clients = []

        def construct_client_from_temporary_thread() -> None:
            try:
                with (
                    patch.object(executor_module, "_detect_external_launch", return_value=None),
                    patch.object(
                        executor_module,
                        "find_free_port",
                        side_effect=[29500, 29501, 29502],
                    ),
                    patch.object(executor_module, "get_ip_address", return_value="127.0.0.1"),
                    patch.object(executor_module, "_get_mp_context", return_value=context),
                    patch.object(
                        executor_module,
                        "run_diffusion_worker",
                        _parent_bound_worker,
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
            except BaseException as e:
                startup_error.append(e)

        constructor = threading.Thread(target=construct_client_from_temporary_thread)
        constructor.start()
        constructor.join(timeout=_STARTUP_TIMEOUT)
        if constructor.is_alive() or startup_error or not clients:
            _emit(f"error:client startup failed: {startup_error!r}")
            os._exit(1)
        _emit("constructor:done")
        _pause()

    worker = context.Process(target=_parent_bound_worker, args=(parent_pid,))
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
            owner.wait_for_spawn(timeout=_STARTUP_TIMEOUT)
        except BaseException as e:
            startup_error.append(e)

    constructor = threading.Thread(target=construct_from_temporary_thread)
    constructor.start()
    constructor.join(timeout=_STARTUP_TIMEOUT)
    if constructor.is_alive() or startup_error:
        _emit(f"error:owner startup failed: {startup_error!r}")
        os._exit(1)
    if not owner_wait_failed.wait(timeout=_STARTUP_TIMEOUT):
        _emit("error:owner wait failure was not observed")
        os._exit(1)
    _emit("constructor:done")
    _pause()


def _run_process_watchdog(watched_pid: int) -> None:
    watchdog = executor_module._start_parent_process_watchdog(watched_pid)
    if watchdog is None:
        raise RuntimeError("process watchdog did not start")
    _emit("ready")
    _pause()


def _run_parent_death_coordinator() -> None:
    context = executor_module._get_mp_context("spawn")
    worker = context.Process(target=_parent_death_worker, args=(os.getpid(),))
    worker.start()
    worker.join()


def main() -> None:
    try:
        scenario = sys.argv[1]
        if scenario == "owned-worker":
            _run_owned_worker_coordinator(bool(int(sys.argv[2])))
        elif scenario == "process-watchdog":
            _run_process_watchdog(int(sys.argv[2]))
        elif scenario == "parent-death":
            _run_parent_death_coordinator()
        else:
            raise ValueError(f"unknown lifecycle helper scenario: {scenario}")
    except BaseException as e:
        _emit(f"error:{type(e).__name__}:{e}")
        traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    main()
