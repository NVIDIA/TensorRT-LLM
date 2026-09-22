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

import multiprocessing
import os
import queue
import signal
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

_CHILD_ERROR = "__spawn_process_child_error__"


class SpawnProcessContext:
    """Message channel supplied to a callable running in a spawned process."""

    def __init__(self, message_queue) -> None:
        self._message_queue = message_queue

    def send(self, name: str, value: Any = None) -> None:
        """Send a named value to the parent test process."""
        self._message_queue.put((name, value))

    def close_sender(self) -> None:
        """Flush sent messages before the child intentionally stops running Python."""
        self._message_queue.close()
        self._message_queue.join_thread()


def _run_in_spawned_process(
    target: Callable[..., None],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    message_queue,
) -> None:
    if hasattr(os, "setsid"):
        os.setsid()
    context = SpawnProcessContext(message_queue)
    try:
        target(context, *args, **kwargs)
    except BaseException:
        context.send(_CHILD_ERROR, traceback.format_exc())
        raise


class SpawnedProcess:
    """Handle for a fresh-interpreter test process and its message channel."""

    def __init__(self, process: multiprocessing.Process, message_queue) -> None:
        self._process = process
        self._message_queue = message_queue
        self._buffered_messages = defaultdict(deque)

    @property
    def pid(self) -> int:
        assert self._process.pid is not None
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def receive(self, name: str, timeout: float = 300.0) -> Any:
        """Receive one named value, buffering messages that arrive out of order."""
        if self._buffered_messages[name]:
            return self._buffered_messages[name].popleft()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                received_name, value = self._message_queue.get(
                    timeout=min(0.1, max(0.0, deadline - time.monotonic()))
                )
            except queue.Empty:
                if not self._process.is_alive():
                    break
                continue
            if received_name == _CHILD_ERROR:
                raise AssertionError(f"spawned process failed:\n{value}")
            if received_name == name:
                return value
            self._buffered_messages[received_name].append(value)

        self._process.join(timeout=0)
        raise TimeoutError(
            f"spawned process {self.pid} did not send {name!r} within {timeout:.0f}s; "
            f"exitcode={self._process.exitcode}"
        )

    def receive_many(self, *names: str, timeout: float = 300.0) -> dict[str, Any]:
        """Receive several named values in any order within one shared timeout."""
        deadline = time.monotonic() + timeout
        values = {}
        for name in names:
            values[name] = self.receive(
                name,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        return values

    def kill(self) -> None:
        self._process.kill()

    def wait(self, timeout: float = 30.0) -> int:
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            raise TimeoutError(f"spawned process {self.pid} did not exit within {timeout:.0f}s")
        assert self._process.exitcode is not None
        return self._process.exitcode

    def close(self) -> None:
        """Kill the spawned process group if necessary and release its resources."""
        if self._process.is_alive():
            if hasattr(os, "killpg"):
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self._process.kill()
        self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=10.0)
        self._message_queue.close()
        self._message_queue.join_thread()
        self._process.close()

    def __enter__(self) -> "SpawnedProcess":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


def spawn_process(
    target: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> SpawnedProcess:
    """Run a module-level callable in a fresh, isolated Python interpreter.

    The callable receives a :class:`SpawnProcessContext` as its first argument.
    It must be importable by a spawned interpreter; nested functions and lambdas
    are therefore not supported.
    """
    context = multiprocessing.get_context("spawn")
    message_queue = context.Queue()
    process = context.Process(
        target=_run_in_spawned_process,
        args=(target, args, kwargs, message_queue),
        name=f"spawn-test-{target.__name__}",
    )
    process.start()
    return SpawnedProcess(process, message_queue)


def wait_forever(context: SpawnProcessContext) -> None:
    """Signal readiness, then keep a spawned test process alive until killed."""
    context.send("ready")
    while True:
        signal.pause()
