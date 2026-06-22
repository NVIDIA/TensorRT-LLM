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
"""Host-side watchdog for MoE AlltoAll completion flags.

The NVLinkOneSided kernels signal each collective by writing the current
``flag_val`` into the rank-local completion flag table.  A dead peer in the
silent-spin failure mode never writes its slot, so this watchdog polls the same
table from a CPU thread and reports peers whose flags do not reach the expected
generation before a bounded timeout.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Mapping, Optional, Protocol, Sequence

import torch

from tensorrt_llm.logger import logger as tllm_logger


class CompletionFlagReader(Protocol):
    """Reads one phase's rank-local completion flag row."""

    def read_completion_flags(self, phase: str) -> Sequence[int]:
        """Return ``ep_size`` flag values for ``phase``."""


class EPGroupHealthLike(Protocol):
    """Subset of EPGroupHealth used by the watchdog."""

    def get_mask(self) -> int:
        """Return the active-rank bitmask."""

    def mark_failed(self, rank: int) -> bool:
        """Mark ``rank`` failed and return whether state changed."""


@dataclass(frozen=True)
class AlltoAllWatchdogTimeout:
    """Details emitted when an AlltoAll phase times out."""

    phase: str
    expected_flag: int
    observed_flags: tuple[int, ...]
    missing_ranks: tuple[int, ...]
    marked_failed_ranks: tuple[int, ...]
    elapsed_s: float


@dataclass(frozen=True)
class _CollectiveWatch:
    phase: str
    expected_flag: int
    active_mask: int
    start_s: float


class _TorchCompletionFlagReader:
    """Completion-flag reader backed by the MoE AlltoAll workspace tensor."""

    def __init__(
        self,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        dispatch_completion_flags_offset: int,
        combine_completion_flags_offset: int,
    ) -> None:
        if workspace.dim() != 2:
            raise ValueError("workspace must be a 2D tensor [ep_size, size_per_rank]")
        if not 0 <= ep_rank < ep_size:
            raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")
        if workspace.size(0) != ep_size:
            raise ValueError(
                f"workspace first dimension must equal ep_size={ep_size}, got {workspace.size(0)}"
            )
        self._workspace = workspace
        self._ep_rank = ep_rank
        self._ep_size = ep_size
        self._offsets = {
            "dispatch": int(dispatch_completion_flags_offset),
            "combine": int(combine_completion_flags_offset),
        }

    def read_completion_flags(self, phase: str) -> tuple[int, ...]:
        offset = self._offsets[phase]
        end = offset + self._ep_size * 4
        flags = self._workspace[self._ep_rank, offset:end].view(torch.int32)
        if flags.device.type != "cpu":
            flags = flags.detach().cpu()
        return tuple(int(v) for v in flags.tolist())


class AlltoAllWatchdog:
    """Background host thread that watches AlltoAll completion flags.

    The watchdog is intentionally opt-in.  Callers queue phases with
    :meth:`watch`; the thread polls them in FIFO order so a queued combine cannot
    hide a still-spinning dispatch.
    """

    VALID_PHASES = frozenset({"dispatch", "combine"})

    def __init__(
        self,
        *,
        ep_size: int,
        ep_rank: int,
        completion_reader: CompletionFlagReader,
        timeout_s: float,
        poll_interval_s: float = 0.05,
        health: Optional[EPGroupHealthLike] = None,
        on_timeout: Optional[Callable[[AlltoAllWatchdogTimeout], None]] = None,
    ) -> None:
        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got {ep_size}")
        if not 0 <= ep_rank < ep_size:
            raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")

        self._ep_size = int(ep_size)
        self._ep_rank = int(ep_rank)
        self._completion_reader = completion_reader
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._health = health
        self._on_timeout = on_timeout

        self._cv = threading.Condition()
        self._queue: Deque[_CollectiveWatch] = deque()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._last_error: BaseException | None = None

    @classmethod
    def from_workspace(
        cls,
        *,
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        metainfo_index: Mapping[str, int],
        ep_rank: int,
        ep_size: int,
        timeout_s: float,
        poll_interval_s: float = 0.05,
        health: Optional[EPGroupHealthLike] = None,
        on_timeout: Optional[Callable[[AlltoAllWatchdogTimeout], None]] = None,
    ) -> "AlltoAllWatchdog":
        """Build a watchdog from the MoE AlltoAll workspace and metainfo."""
        dispatch_offset = int(
            metainfo[metainfo_index["DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX"]].item()
        )
        combine_offset = int(
            metainfo[metainfo_index["COMBINE_COMPLETION_FLAGS_OFFSET_INDEX"]].item()
        )
        reader = _TorchCompletionFlagReader(
            workspace=workspace,
            ep_rank=ep_rank,
            ep_size=ep_size,
            dispatch_completion_flags_offset=dispatch_offset,
            combine_completion_flags_offset=combine_offset,
        )
        return cls(
            ep_size=ep_size,
            ep_rank=ep_rank,
            completion_reader=reader,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            health=health,
            on_timeout=on_timeout,
        )

    @property
    def last_error(self) -> BaseException | None:
        """Return the last polling-thread error, if any."""
        with self._cv:
            return self._last_error

    def start(self) -> None:
        """Start the background polling thread. Idempotent."""
        with self._cv:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name=f"AlltoAllWatchdog-rank{self._ep_rank}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_s: float | None = None) -> None:
        """Stop the polling thread and wait for it to exit."""
        with self._cv:
            self._stopping = True
            self._queue.clear()
            self._cv.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)

    def watch(
        self,
        *,
        phase: str,
        expected_flag: int,
        active_mask: int | None = None,
    ) -> None:
        """Queue a just-launched AlltoAll phase for watchdog polling."""
        if phase not in self.VALID_PHASES:
            raise ValueError(f"phase must be one of {sorted(self.VALID_PHASES)}, got {phase!r}")
        if expected_flag < 0:
            raise ValueError(f"expected_flag must be non-negative, got {expected_flag}")
        if active_mask is None:
            if self._health is not None:
                active_mask = self._health.get_mask()
            else:
                active_mask = (1 << self._ep_size) - 1
        if not (active_mask >> self._ep_rank) & 1:
            raise ValueError("active_mask must include the local ep_rank")

        self.start()
        with self._cv:
            if self._stopping:
                raise RuntimeError("cannot queue a stopped AlltoAllWatchdog")
            self._queue.append(
                _CollectiveWatch(
                    phase=phase,
                    expected_flag=int(expected_flag),
                    active_mask=int(active_mask),
                    start_s=time.monotonic(),
                )
            )
            self._cv.notify_all()

    def wait_until_idle(self, timeout_s: float) -> bool:
        """Wait until all queued phases complete or timeout handling clears them."""
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True

    def __enter__(self) -> "AlltoAllWatchdog":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(timeout_s=1.0)

    def _active_ranks(self, active_mask: int) -> tuple[int, ...]:
        return tuple(rank for rank in range(self._ep_size) if (active_mask >> rank) & 1)

    def _phase_complete(self, watch: _CollectiveWatch, observed_flags: tuple[int, ...]) -> bool:
        return all(
            observed_flags[rank] == watch.expected_flag
            for rank in self._active_ranks(watch.active_mask)
        )

    def _missing_ranks(
        self, watch: _CollectiveWatch, observed_flags: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(
            rank
            for rank in self._active_ranks(watch.active_mask)
            if observed_flags[rank] != watch.expected_flag
        )

    def _handle_timeout(self, watch: _CollectiveWatch, observed_flags: tuple[int, ...]) -> None:
        elapsed_s = time.monotonic() - watch.start_s
        missing_ranks = self._missing_ranks(watch, observed_flags)
        marked_failed: list[int] = []
        if self._health is not None:
            for rank in missing_ranks:
                if rank == self._ep_rank:
                    continue
                if self._health.mark_failed(rank):
                    marked_failed.append(rank)

        event = AlltoAllWatchdogTimeout(
            phase=watch.phase,
            expected_flag=watch.expected_flag,
            observed_flags=observed_flags,
            missing_ranks=missing_ranks,
            marked_failed_ranks=tuple(marked_failed),
            elapsed_s=elapsed_s,
        )
        tllm_logger.warning(
            "AlltoAll watchdog timeout on rank %d during %s: expected flag %d, "
            "missing ranks %s, observed flags %s",
            self._ep_rank,
            watch.phase,
            watch.expected_flag,
            list(missing_ranks),
            list(observed_flags),
        )
        if self._on_timeout is not None:
            self._on_timeout(event)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stopping:
                    self._cv.wait()
                if self._stopping:
                    return
                watch = self._queue[0]

            try:
                observed_flags = tuple(
                    int(v) for v in self._completion_reader.read_completion_flags(watch.phase)
                )
                if len(observed_flags) != self._ep_size:
                    raise RuntimeError(
                        f"completion reader returned {len(observed_flags)} flags; "
                        f"expected ep_size={self._ep_size}"
                    )
            except BaseException as exc:  # noqa: BLE001 - keep watchdog failures visible.
                with self._cv:
                    self._last_error = exc
                    self._queue.clear()
                    self._cv.notify_all()
                tllm_logger.error("AlltoAll watchdog stopped after polling error: %s", exc)
                return

            if self._phase_complete(watch, observed_flags):
                with self._cv:
                    if self._queue and self._queue[0] is watch:
                        self._queue.popleft()
                    self._cv.notify_all()
                continue

            if time.monotonic() - watch.start_s >= self._timeout_s:
                self._handle_timeout(watch, observed_flags)
                with self._cv:
                    # The GPU stream is no longer trustworthy once a collective
                    # times out. Drop queued follow-on phases so they do not
                    # produce duplicate or misleading reports.
                    self._queue.clear()
                    self._cv.notify_all()
                continue

            with self._cv:
                self._cv.wait(timeout=self._poll_interval_s)
