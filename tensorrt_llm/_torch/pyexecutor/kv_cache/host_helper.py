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
"""One background thread that runs KV cache manager work beside the executor thread.

The executor thread opens two kinds of windows in which the helper may start an item:

* ``forward``: the executor is inside the forward call of a batch that captures no CUDA graph
  (a context-only batch outside warmup). Releases start at any time in it; claims and prefetches
  only during the first ``forward_start_fraction`` of the forward's running average length, so
  they do not contend with the scheduling pass that follows for the manager's lock.
* ``wait``: the executor waits for the previous step's sampler event. Releases only.

An item that started in a window may outlive it. :meth:`begin_forward` of a batch that may capture
a CUDA graph waits for it, so no helper call is ever made during a capture. The helper is idle
outside the windows. :meth:`drain` runs the queued releases on the calling thread.

KVCacheManagerV2 serializes its own state with an API lock and supports building a cache on one
thread and handing it to another (``cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md``,
"Using KVCM2 from more than one thread"). Every Python-side structure of the manager stays on the
executor thread: the helper runs only the closures it is given.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

from tensorrt_llm.logger import logger

# Item kinds, in the order the helper takes them.
RELEASE = 0
CLAIM = 1
PREFETCH = 2
_KIND_NAMES = ("release", "claim", "prefetch")


class KvHostHelper:
    """A FIFO per item kind, run by one daemon thread inside executor-defined windows."""

    def __init__(
        self,
        device: Optional[int],
        *,
        name: str = "kv-host-helper",
        forward_start_fraction: float = 0.7,
        start_thread: bool = True,
    ) -> None:
        self._device = device
        self._name = name
        self._forward_start_fraction = forward_start_fraction
        self._cond = threading.Condition()
        # Held while one item runs; begin_forward and drain take it to wait for that item.
        self._item_lock = threading.Lock()
        self._queues: tuple[deque, deque, deque] = (deque(), deque(), deque())
        self._in_forward = False
        self._in_wait = False
        self._forward_started = 0.0
        self._forward_ema: Optional[float] = None
        self._stopped = False
        self._error: Optional[BaseException] = None
        self.num_run = [0, 0, 0]
        self.num_forward_waits = 0
        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, name=name, daemon=True)
            self._thread.start()

    # ---- executor thread --------------------------------------------------------------------

    def submit(self, kind: int, item: Callable[[], None]) -> None:
        """Queue ``item``; it runs on the helper thread in a window that allows ``kind``."""
        with self._cond:
            if self._stopped:
                raise RuntimeError(f"{self._name} is stopped")
            self._queues[kind].append(item)
            self._cond.notify()

    def num_pending(self, kind: int) -> int:
        return len(self._queues[kind])

    def begin_forward(self, may_capture: bool) -> None:
        """The executor enters a forward call. With ``may_capture``, wait for a running item and
        start none until :meth:`end_forward`; otherwise open the ``forward`` window."""
        with self._cond:
            self._in_wait = False
            self._in_forward = not may_capture
            self._forward_started = time.monotonic()
            self._cond.notify()
        if may_capture:
            if not self._item_lock.acquire(blocking=False):
                self.num_forward_waits += 1
                self._item_lock.acquire()
            self._item_lock.release()
        self.raise_pending_error()

    def end_forward(self) -> None:
        with self._cond:
            if self._in_forward:
                elapsed = time.monotonic() - self._forward_started
                ema = self._forward_ema
                self._forward_ema = elapsed if ema is None else 0.8 * ema + 0.2 * elapsed
            self._in_forward = False

    def begin_wait(self) -> None:
        """The executor starts waiting for the previous step's GPU work."""
        with self._cond:
            self._in_wait = True
            self._cond.notify()

    def end_wait(self) -> None:
        with self._cond:
            self._in_wait = False

    def drain(self, kind: int, run_inline: Callable[[], None]) -> None:
        """Wait for a running item, drop the queued items of ``kind`` and call ``run_inline``
        on this thread; the caller's ``run_inline`` does what those items would have done."""
        with self._item_lock:
            with self._cond:
                self._queues[kind].clear()
            run_inline()
        self.raise_pending_error()

    def raise_pending_error(self) -> None:
        """Re-raise, on the executor thread, the first error an item raised on the helper."""
        error, self._error = self._error, None
        if error is not None:
            raise error

    def stop(self) -> None:
        """Stop the thread after its running item. Queued items are dropped."""
        with self._cond:
            self._stopped = True
            for queue in self._queues:
                queue.clear()
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    # ---- helper thread ----------------------------------------------------------------------

    def _allowed(self, kind: int) -> bool:
        if kind == RELEASE:
            return self._in_forward or self._in_wait
        if not self._in_forward:
            return False
        ema = self._forward_ema
        if ema is None:
            return True
        return time.monotonic() - self._forward_started < self._forward_start_fraction * ema

    def _next(self) -> Optional[tuple[int, Callable[[], None]]]:
        """The next item a window allows, or None. Called with the condition held."""
        for kind, queue in enumerate(self._queues):
            if queue and self._allowed(kind):
                return kind, queue.popleft()
        return None

    def run_one(self) -> bool:
        """Run one allowed item on the calling thread; False when there was none. The helper's
        loop step, also called by tests to drive the helper without a thread."""
        with self._item_lock:
            with self._cond:
                picked = self._next()
            if picked is None:
                return False
            kind, item = picked
            try:
                item()
            except BaseException as error:  # noqa: BLE001 - handed to the executor thread
                logger.warning(f"{self._name}: a {_KIND_NAMES[kind]} item failed: {error!r}")
                if self._error is None:
                    self._error = error
            self.num_run[kind] += 1
        return True

    def _run(self) -> None:
        if self._device is not None:
            import torch

            torch.cuda.set_device(self._device)
        while True:
            with self._cond:
                while not self._stopped and not any(
                    queue and self._allowed(kind) for kind, queue in enumerate(self._queues)
                ):
                    # Only a notification makes an item allowed: submit, begin_forward and
                    # begin_wait notify; the end of a window or of its start fraction only
                    # disallows.
                    self._cond.wait()
                if self._stopped:
                    return
            self.run_one()
