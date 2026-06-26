# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from contextlib import contextmanager
from typing import Callable, Optional

from tensorrt_llm._utils import ENABLE_MULTI_DEVICE, mpi_comm, mpi_disabled, print_all_stacks
from tensorrt_llm.logger import logger

# 137 == 128 + SIGKILL(9): the exit code a shell reports for a SIGKILL'd process.
_HARD_KILL_EXIT_CODE = 137


def _best_effort_flush_streams() -> None:
    """Flush stdout/stderr without ever raising; diagnostics must not block hard kill."""
    for stream in (sys.stderr, sys.stdout):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass


def _best_effort_log_error(message: str) -> None:
    """Log at error level without ever raising; diagnostics must not block hard kill."""
    try:
        logger.error(message)
    except Exception:  # noqa: BLE001 - diagnostics must not block hard kill
        pass


def propagate_hard_kill(exit_code: int = _HARD_KILL_EXIT_CODE) -> None:
    """Hard-kill this rank and propagate the kill to peer ranks.

    Cross-rank propagation is the load-bearing part: a peer blocked in an NCCL
    collective would otherwise hold its GPU until the job's wall-clock pod-kill.

    - Preferred (when safe): ``MPI_Abort`` aborts the whole MPI job in one call.
      Only safe from the detector's daemon thread when MPI was initialized with
      ``MPI_THREAD_MULTIPLE``; guarded by ``Query_thread``.
    - Fallback: self-``SIGKILL``. The launcher (``mpirun`` propagates by default;
      ``srun`` needs ``--kill-on-bad-exit``) then tears down peers.

    All flushing and logging is best-effort: a closed/broken stdout, stderr, or
    logger must never prevent reaching ``MPI_Abort`` or ``os.kill``.
    """
    _best_effort_flush_streams()
    try:
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            from mpi4py import MPI

            if MPI.Is_initialized() and MPI.Query_thread() == MPI.THREAD_MULTIPLE:
                _best_effort_log_error(
                    "HangDetector: propagating hard-kill to all ranks via MPI_Abort."
                )
                mpi_comm().Abort(exit_code)
                return  # not reached; Abort does not return
    except Exception as e:  # noqa: BLE001 - last-resort path must not raise
        _best_effort_log_error(
            f"HangDetector: MPI_Abort propagation failed ({e}); falling back to self-SIGKILL."
        )
    _best_effort_log_error(
        "HangDetector: self-SIGKILL; relying on the launcher to propagate to peer ranks."
    )
    os.kill(os.getpid(), signal.SIGKILL)


class HangDetector:
    """Watchdog that fires when the executor loop stops checkpointing.

    When ``timeout`` seconds pass without a ``checkpoint()``, all thread stacks
    are dumped for diagnosis and ``on_detected`` runs (the hard-kill +
    cross-rank propagation path).
    """

    def __init__(
        self, timeout: Optional[int] = None, on_detected: Optional[Callable[[], None]] = None
    ):
        self.timeout = timeout if timeout is not None else 300
        assert self.timeout > 0, "timeout must be greater than 0"
        self.on_detected = on_detected or (lambda: None)
        self.task = None
        self.loop = None
        self.loop_thread = None
        self.lock = threading.Lock()
        self.active = False
        self._detected = False

    def start(self):
        """Enable hang detection."""

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.active = True
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=run_loop, daemon=True, name="hang_detector_loop")
        self.loop_thread.start()

    async def _detect_hang(self):
        await asyncio.sleep(self.timeout)
        with self.lock:
            self._detected = True
            logger.error(f"Hang detected after {self.timeout} seconds.")
            print_all_stacks()
            self.on_detected()

    def detected(self):
        """Return True if hang is detected."""
        with self.lock:
            return self._detected

    def checkpoint(self):
        """Reset hang detection timer."""
        self.cancel_task()
        if self.active:
            self.task = asyncio.run_coroutine_threadsafe(self._detect_hang(), self.loop)

    def cancel_task(self):
        """Cancel the hang detection task."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
            self.task = None

    @contextmanager
    def pause(self):
        """Pause hang detection in scope."""
        try:
            self.cancel_task()
            yield
        finally:
            self.checkpoint()

    def stop(self):
        """Stop hang detection."""
        self.active = False
        self.cancel_task()
        if self.loop is not None:
            # Cancel all pending tasks before stopping the loop
            def cancel_all_tasks():
                for task in asyncio.all_tasks(self.loop):
                    if not task.done():
                        task.cancel()
                self.loop.call_soon(self.loop.stop)

            self.loop.call_soon_threadsafe(cancel_all_tasks)

            if self.loop_thread is not None and self.loop_thread.is_alive():
                self.loop_thread.join()

            self.loop = None
            self.loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
