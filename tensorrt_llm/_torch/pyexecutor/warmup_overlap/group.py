# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``BuildGroup``: one kind of kernel build, run from a queue on background threads."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, List, Optional

import torch

from tensorrt_llm.logger import logger


class BuildGroup:
    """Builds one kind of kernel from a queue of items on ``threads`` threads, each on its
    own stream, and logs its time once the last thread is done.

    ``build(item, stop)`` returns the number of builds it ran, 0 when it skipped the item;
    it may return early once ``stop`` is set. An item whose build raises is counted as
    failed and left to the warmup. ``summary()``, if given, is appended to the final log.
    """

    def __init__(
        self,
        name: str,
        items: List[Any],
        build: Callable[[Any, threading.Event], int],
        threads: int,
        summary: Optional[Callable[[], str]] = None,
    ):
        self.name, self.build, self.summary = name, build, summary
        self.items: queue.Queue = queue.Queue()
        for item in items:
            self.items.put(item)
        self.count, self.done, self.launches, self.skipped, self.failed = len(items), 0, 0, 0, 0
        self.running = min(threads, len(items))
        self.mutex = threading.Lock()
        self.stop = threading.Event()
        self.threads: List[threading.Thread] = []
        self.t0 = time.perf_counter()

    def start(self, device: int) -> None:
        for i in range(self.running):
            thread = threading.Thread(
                target=self._run, args=(device,), name=f"tllm-overlap-{self.name}-{i}", daemon=True
            )
            thread.start()
            self.threads.append(thread)

    def alive(self) -> List[threading.Thread]:
        return [thread for thread in self.threads if thread.is_alive()]

    def progress(self) -> str:
        return f"{self.name} {self.done}/{self.count}"

    def _run(self, device: int) -> None:
        torch.cuda.set_device(device)
        # A side stream keeps the launches off the weight loader's stream.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            while not self.stop.is_set():
                try:
                    item = self.items.get_nowait()
                except queue.Empty:
                    break
                try:
                    launches = self.build(item, self.stop)
                    with self.mutex:
                        self.launches += launches
                        if launches == 0:
                            self.skipped += 1
                except Exception as e:  # noqa: BLE001 - the warmup builds what is missing
                    with self.mutex:
                        self.failed += 1
                    logger.warning(
                        f"Overlapped {self.name} build of {item} failed; the warmup builds it. "
                        f"{type(e).__name__}: {e}"
                    )
                with self.mutex:
                    self.done += 1
        stream.synchronize()
        with self.mutex:
            self.running -= 1
            last = self.running == 0
        if last:
            extra = f", {self.summary()}" if self.summary else ""
            logger.info(
                f"[warmup] overlap done: {self.name} {time.perf_counter() - self.t0:.1f}s "
                f"({self.count} items, {self.launches} builds, {self.skipped} skipped, "
                f"{self.failed} failed{extra})"
            )
