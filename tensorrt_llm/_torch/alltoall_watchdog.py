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

Timeouts are detection events, not committed membership changes.  The optional
EP health object supplies the already-committed active peers to watch, and is
read-only here.  Higher-layer recovery coordination consumes ``on_timeout``
events and commits a new membership only after the data plane and expert
placement are ready for the same generation.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger as tllm_logger

DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S = 5.0
DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S = 0.1
UNKNOWN_COMPLETION_FLAG = -(2**63)
_COMPLETION_FLAG_MASK = (1 << 32) - 1
_COMPLETION_FLAG_HALF_RANGE = 1 << 31
_ACTIVE_RANK_MASK_WORD_BITS = 64
_ACTIVE_RANK_MASK_WORDS = 2
_ACTIVE_RANK_MASK_WORD_MASK = (1 << _ACTIVE_RANK_MASK_WORD_BITS) - 1
_WORKSPACE_WATCHDOG_STATE_KEY = "alltoall_watchdog_shared_state"
_WORKSPACE_WATCHDOG_STATE_INIT_LOCK = threading.Lock()


def _normalize_completion_flag(value: int) -> int:
    return int(value) & _COMPLETION_FLAG_MASK


def _completion_flag_reached(observed: int, expected: int) -> bool:
    if observed == UNKNOWN_COMPLETION_FLAG:
        return False
    # Counter values are ordered modulo uint32. A queued watch cannot lag by
    # half the counter space within the bounded watchdog timeout.
    distance = (_normalize_completion_flag(observed) - expected) & _COMPLETION_FLAG_MASK
    return distance < _COMPLETION_FLAG_HALF_RANGE


class CompletionFlagReader(Protocol):
    """Reads one phase's rank-local completion flag row."""

    def read_completion_flags(self, phase: str) -> Sequence[int]:
        """Return ``ep_size`` flag values for ``phase``."""


class EPGroupHealthSnapshotLike(Protocol):
    """Atomic fields required from a committed-membership snapshot."""

    @property
    def mask(self) -> int:
        """Return the committed active-rank bitmask."""

    @property
    def generation(self) -> int:
        """Return the matching committed-membership generation."""


class EPGroupHealthLike(Protocol):
    """Read-only committed EP membership used by AlltoAll frontends."""

    def get_mask(self) -> int:
        """Return the active-rank bitmask."""

    def get_mask_words(self) -> tuple[int, ...]:
        """Return the active-rank bitmask split into uint64 words."""

    def snapshot(self) -> EPGroupHealthSnapshotLike:
        """Return one atomic committed mask and generation snapshot."""


@dataclass(frozen=True)
class ActiveRankMaskSnapshot:
    """One dispatch epoch's rank mask and optional committed generation.

    ``committed_generation`` is populated only when ``active_rank_mask`` was
    read from the coordinator-owned EP health object. Explicit caller masks do
    not inherit a generation from unrelated health state.
    """

    active_rank_mask: torch.Tensor | None
    committed_generation: int | None


class CompletionFlagReadTimeout(TimeoutError):
    """Raised when the host watchdog cannot read completion flags in time."""


@dataclass(frozen=True)
class AlltoAllWatchdogTimeout:
    """Detection details emitted when an AlltoAll phase times out.

    ``missing_ranks`` contains suspects relative to the committed active mask
    captured for this collective.  Receiving this event does not publish a new
    active mask.
    """

    phase: str
    expected_flag: int
    observed_flags: tuple[int, ...]
    missing_ranks: tuple[int, ...]
    elapsed_s: float
    poll_timed_out: bool = False


@dataclass(frozen=True)
class _CollectiveWatch:
    phase: str
    expected_flag: int
    active_mask: int
    start_s: float


@dataclass(frozen=True)
class _SharedWatchdogConfig:
    ep_size: int
    ep_rank: int
    timeout_s: float
    poll_interval_s: float
    health: EPGroupHealthLike | None
    on_timeout: Callable[[AlltoAllWatchdogTimeout], None] | None


@dataclass
class _WorkspaceWatchdogState:
    lock: threading.Lock
    generation: int = 0
    watchdog: AlltoAllWatchdog | None = None
    config: _SharedWatchdogConfig | None = None
    ref_count: int = 0


class _TorchCompletionFlagReader:
    """Completion-flag reader backed by the MoE AlltoAll workspace tensor."""

    def __init__(
        self,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        dispatch_completion_flags_offset: int,
        combine_completion_flags_offset: int,
        device_copy_timeout_s: float = DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
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
        self._device_copy_timeout_s = float(device_copy_timeout_s)
        self._copy_stream: torch.cuda.Stream | None = None
        self._host_flags: torch.Tensor | None = None
        self._copy_event: torch.cuda.Event | None = None
        self._retired_copies: list[tuple[torch.Tensor, torch.cuda.Event]] = []
        if workspace.device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=workspace.device)

    def _prune_retired_copies(self) -> None:
        self._retired_copies = [
            (host_flags, event) for host_flags, event in self._retired_copies if not event.query()
        ]

    def _read_cuda_flags(self, flags: torch.Tensor) -> tuple[int, ...]:
        assert self._copy_stream is not None
        self._prune_retired_copies()

        if self._host_flags is None:
            self._host_flags = torch.empty(
                (self._ep_size,),
                dtype=torch.int32,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        if self._copy_event is None:
            self._copy_event = torch.cuda.Event(blocking=False)
        host_flags = self._host_flags
        event = self._copy_event
        with torch.cuda.device(flags.device), torch.cuda.stream(self._copy_stream):
            host_flags.copy_(flags.detach(), non_blocking=True)
            event.record(self._copy_stream)

        deadline_s = time.monotonic() + self._device_copy_timeout_s
        while not event.query():
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0:
                self._retired_copies.append((host_flags, event))
                self._host_flags = None
                self._copy_event = None
                raise CompletionFlagReadTimeout(
                    "timed out copying AlltoAll completion flags to host"
                )
            time.sleep(min(remaining_s, 0.001))

        return tuple(int(v) for v in host_flags.tolist())

    def read_completion_flags(self, phase: str) -> tuple[int, ...]:
        offset = self._offsets[phase]
        end = offset + self._ep_size * 4
        flags = self._workspace[self._ep_rank, offset:end].view(torch.int32)
        if flags.device.type == "cuda":
            return self._read_cuda_flags(flags)
        if flags.device.type != "cpu":
            flags = flags.detach().cpu()
        return tuple(int(v) for v in flags.tolist())


class AlltoAllWatchdogCoordinator:
    """Shared watchdog plumbing for MoE AlltoAll frontends.

    ``health`` is the committed membership snapshot source.  This class reads
    it when a dispatch starts but never publishes detected failures into it.
    """

    def __init__(
        self,
        *,
        workspace_state: MutableMapping[str, object],
        workspace: torch.Tensor,
        metainfo: torch.Tensor,
        metainfo_index: Mapping[str, int],
        ep_rank: int,
        health: EPGroupHealthLike | None = None,
    ) -> None:
        self._workspace_state = workspace_state
        self._workspace = workspace
        self._metainfo = metainfo
        self._metainfo_index = metainfo_index
        self._ep_rank = ep_rank
        self._health = health

    def _find_shared_state(self) -> _WorkspaceWatchdogState | None:
        state = self._workspace_state.get(_WORKSPACE_WATCHDOG_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, _WorkspaceWatchdogState):
            raise TypeError("invalid shared AlltoAll watchdog workspace state")
        return state

    def _get_shared_state(self) -> _WorkspaceWatchdogState:
        state = self._find_shared_state()
        if state is not None:
            return state
        with _WORKSPACE_WATCHDOG_STATE_INIT_LOCK:
            state = self._workspace_state.get(_WORKSPACE_WATCHDOG_STATE_KEY)
            if state is None:
                state = _WorkspaceWatchdogState(lock=threading.Lock())
                self._workspace_state[_WORKSPACE_WATCHDOG_STATE_KEY] = state
            if not isinstance(state, _WorkspaceWatchdogState):
                raise TypeError("invalid shared AlltoAll watchdog workspace state")
            return state

    def read_current_flag_val(self) -> int:
        flag_val_offset = int(self._metainfo[self._metainfo_index["FLAG_VAL_OFFSET_INDEX"]].item())
        flag_val = self._workspace[self._ep_rank, flag_val_offset : flag_val_offset + 4].view(
            torch.int32
        )
        if flag_val.device.type != "cpu":
            flag_val = flag_val.detach().cpu()
        return _normalize_completion_flag(int(flag_val.item()))

    def capture_active_rank_mask(
        self, active_rank_mask: torch.Tensor | None
    ) -> ActiveRankMaskSnapshot:
        """Capture a caller override or one coherent committed-mask epoch.

        The returned tensor does not alias a caller-owned override, so it can
        safely represent the mask used by one dispatch/combine pair. When the
        mask comes from ``health``, derive both fixed-width ABI words from one
        atomic mask/generation snapshot.
        """
        if active_rank_mask is not None:
            return ActiveRankMaskSnapshot(
                active_rank_mask=active_rank_mask.detach().clone(),
                committed_generation=None,
            )
        if self._health is None:
            return ActiveRankMaskSnapshot(
                active_rank_mask=None,
                committed_generation=None,
            )

        health_snapshot = self._health.snapshot()
        mask_words = tuple(
            (health_snapshot.mask >> (_ACTIVE_RANK_MASK_WORD_BITS * index))
            & _ACTIVE_RANK_MASK_WORD_MASK
            for index in range(_ACTIVE_RANK_MASK_WORDS)
        )
        return ActiveRankMaskSnapshot(
            active_rank_mask=torch.tensor(
                mask_words,
                dtype=torch.uint64,
                device="cpu",
            ),
            committed_generation=health_snapshot.generation,
        )

    def active_rank_mask_tensor(self, active_rank_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Return only the tensor from :meth:`capture_active_rank_mask`.

        Dispatch frontends must retain the full snapshot so combine can verify
        the committed generation before launching its kernel.
        """
        return self.capture_active_rank_mask(active_rank_mask).active_rank_mask

    def active_rank_mask_for_combine(
        self,
        dispatch_snapshot: ActiveRankMaskSnapshot,
        requested_active_rank_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Validate the dispatch epoch and return its captured rank mask.

        A coordinator-owned generation change between dispatch and combine is
        an invalid asynchronous commit, so fail closed before launching the
        combine kernel. Explicit caller masks are independent of health and
        therefore carry no committed generation.
        """
        dispatch_active_rank_mask = dispatch_snapshot.active_rank_mask
        committed_generation = dispatch_snapshot.committed_generation
        if committed_generation is not None:
            if self._health is None:
                raise RuntimeError("committed rank-mask snapshot has no EP health source")
            current_generation = self._health.snapshot().generation
            if current_generation != committed_generation:
                raise RuntimeError(
                    "committed EP membership changed between dispatch and combine "
                    f"(generation {committed_generation} -> {current_generation}); "
                    "aborting collective epoch"
                )

        if requested_active_rank_mask is None:
            return dispatch_active_rank_mask
        requested_mask = self.active_rank_mask_tensor(requested_active_rank_mask)
        assert requested_mask is not None
        if dispatch_active_rank_mask is None or not torch.equal(
            dispatch_active_rank_mask, requested_mask
        ):
            raise ValueError("active_rank_mask must match the mask captured at dispatch")
        return dispatch_active_rank_mask

    def active_mask_int(self, active_rank_mask: torch.Tensor | None) -> int | None:
        if active_rank_mask is not None:
            mask_cpu = active_rank_mask.detach().cpu()
            return sum(int(word) << (64 * idx) for idx, word in enumerate(mask_cpu.tolist()))
        if self._health is not None:
            return self._health.get_mask()
        return None

    def acquire_watchdog(
        self,
        *,
        ep_size: int,
        timeout_s: float,
        poll_interval_s: float,
        on_timeout: Callable[[AlltoAllWatchdogTimeout], None] | None = None,
    ) -> AlltoAllWatchdog:
        config = _SharedWatchdogConfig(
            ep_size=int(ep_size),
            ep_rank=self._ep_rank,
            timeout_s=float(timeout_s),
            poll_interval_s=float(poll_interval_s),
            health=self._health,
            on_timeout=on_timeout,
        )
        state = self._get_shared_state()
        with state.lock:
            if state.watchdog is None:
                state.generation = self.read_current_flag_val()
                state.watchdog = AlltoAllWatchdog.from_workspace(
                    workspace=self._workspace,
                    metainfo=self._metainfo,
                    metainfo_index=self._metainfo_index,
                    ep_rank=self._ep_rank,
                    ep_size=ep_size,
                    timeout_s=timeout_s,
                    poll_interval_s=poll_interval_s,
                    health=self._health,
                    on_timeout=on_timeout,
                )
                state.config = config
            elif not self._matching_config(state.config, config):
                raise ValueError(
                    "AlltoAll wrappers sharing a workspace must use the same "
                    "watchdog configuration and EP health object"
                )
            state.ref_count += 1
            watchdog = state.watchdog
            assert watchdog is not None
            return watchdog

    @staticmethod
    def _matching_config(
        existing: _SharedWatchdogConfig | None,
        requested: _SharedWatchdogConfig,
    ) -> bool:
        return (
            existing is not None
            and existing.ep_size == requested.ep_size
            and existing.ep_rank == requested.ep_rank
            and existing.timeout_s == requested.timeout_s
            and existing.poll_interval_s == requested.poll_interval_s
            and existing.health is requested.health
            and existing.on_timeout is requested.on_timeout
        )

    def release_watchdog(self, watchdog: AlltoAllWatchdog) -> None:
        state = self._get_shared_state()
        watchdog_to_stop: AlltoAllWatchdog | None = None
        with state.lock:
            if state.watchdog is not watchdog or state.ref_count <= 0:
                raise RuntimeError("attempted to release an unregistered AlltoAll watchdog")
            state.ref_count -= 1
            if state.ref_count == 0:
                watchdog_to_stop = state.watchdog
                state.watchdog = None
                state.config = None
        if watchdog_to_stop is not None:
            watchdog_to_stop.stop()

    def watch_collective(
        self,
        watchdog: AlltoAllWatchdog | None,
        phase: str,
        active_rank_mask: torch.Tensor | None,
    ) -> None:
        if watchdog is None:
            state = self._find_shared_state()
            if state is not None:
                with state.lock:
                    if state.watchdog is not None:
                        state.generation = (state.generation + 1) & _COMPLETION_FLAG_MASK
            return
        active_mask = self.active_mask_int(active_rank_mask)
        state = self._get_shared_state()
        with state.lock:
            if state.watchdog is not watchdog:
                raise RuntimeError("AlltoAll watchdog is not registered for this workspace")
            state.generation = (state.generation + 1) & _COMPLETION_FLAG_MASK
            watchdog.watch(
                phase=phase,
                expected_flag=state.generation,
                active_mask=active_mask,
            )


class AlltoAllWatchdog:
    """Background host thread that watches AlltoAll completion flags.

    The watchdog is intentionally opt-in.  Callers queue phases with
    :meth:`watch`; the thread polls them in FIFO order so a queued combine cannot
    hide a still-spinning dispatch.  A timeout is reported through
    ``on_timeout`` without mutating the committed EP membership.
    """

    VALID_PHASES = frozenset({"dispatch", "combine"})

    def __init__(
        self,
        *,
        ep_size: int,
        ep_rank: int,
        completion_reader: CompletionFlagReader,
        timeout_s: float = DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
        health: EPGroupHealthLike | None = None,
        on_timeout: Callable[[AlltoAllWatchdogTimeout], None] | None = None,
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
        self._queue: deque[_CollectiveWatch] = deque()
        self._closed = False
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
        timeout_s: float = DEFAULT_ALLTOALL_WATCHDOG_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_ALLTOALL_WATCHDOG_POLL_INTERVAL_S,
        health: EPGroupHealthLike | None = None,
        on_timeout: Callable[[AlltoAllWatchdogTimeout], None] | None = None,
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
            device_copy_timeout_s=poll_interval_s,
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
            if self._closed:
                raise RuntimeError("cannot start a stopped AlltoAllWatchdog")
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
            self._closed = True
            self._stopping = True
            self._queue.clear()
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
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
        if not 0 <= expected_flag <= _COMPLETION_FLAG_MASK:
            raise ValueError(
                f"expected_flag must be in [0, {_COMPLETION_FLAG_MASK}], got {expected_flag}"
            )
        if active_mask is None:
            if self._health is not None:
                active_mask = self._health.get_mask()
            else:
                active_mask = (1 << self._ep_size) - 1
        if not (active_mask >> self._ep_rank) & 1:
            raise ValueError("active_mask must include the local ep_rank")

        self.start()
        with self._cv:
            if self._closed:
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
            _completion_flag_reached(observed_flags[rank], watch.expected_flag)
            for rank in self._active_ranks(watch.active_mask)
        )

    def _missing_ranks(
        self, watch: _CollectiveWatch, observed_flags: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(
            rank
            for rank in self._active_ranks(watch.active_mask)
            if not _completion_flag_reached(observed_flags[rank], watch.expected_flag)
        )

    def _handle_timeout(
        self,
        watch: _CollectiveWatch,
        observed_flags: tuple[int, ...],
        *,
        poll_timed_out: bool = False,
    ) -> None:
        elapsed_s = time.monotonic() - watch.start_s
        missing_ranks = self._missing_ranks(watch, observed_flags)
        event = AlltoAllWatchdogTimeout(
            phase=watch.phase,
            expected_flag=watch.expected_flag,
            observed_flags=observed_flags,
            missing_ranks=missing_ranks,
            elapsed_s=elapsed_s,
            poll_timed_out=poll_timed_out,
        )
        if poll_timed_out:
            tllm_logger.error(
                "AlltoAll watchdog could not read completion flags on rank %d "
                "during %s before timeout %.3fs; expected flag %d, active "
                "ranks %s, observed flags %s; reporting detection event only",
                self._ep_rank,
                watch.phase,
                elapsed_s,
                watch.expected_flag,
                list(self._active_ranks(watch.active_mask)),
                list(observed_flags),
            )
        else:
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

    def _stop_after_error(self, exc: Exception) -> None:
        with self._cv:
            self._last_error = exc
            self._closed = True
            self._stopping = True
            self._queue.clear()
            self._cv.notify_all()

    def _run(self) -> None:
        last_observed_flags = tuple(UNKNOWN_COMPLETION_FLAG for _ in range(self._ep_size))
        poll_timed_out = False
        while True:
            with self._cv:
                while not self._queue and not self._stopping:
                    self._cv.wait()
                if self._stopping:
                    return
                watch = self._queue[0]

            try:
                observed_flags = tuple(
                    _normalize_completion_flag(int(v))
                    for v in self._completion_reader.read_completion_flags(watch.phase)
                )
                if len(observed_flags) != self._ep_size:
                    raise RuntimeError(
                        f"completion reader returned {len(observed_flags)} flags; "
                        f"expected ep_size={self._ep_size}"
                    )
                last_observed_flags = observed_flags
                poll_timed_out = False
            except CompletionFlagReadTimeout:
                observed_flags = last_observed_flags
                poll_timed_out = True
            except Exception as exc:  # noqa: BLE001 - keep watchdog failures visible.
                self._stop_after_error(exc)
                tllm_logger.error("AlltoAll watchdog stopped after polling error: %s", exc)
                return

            if self._phase_complete(watch, observed_flags):
                with self._cv:
                    if self._queue and self._queue[0] is watch:
                        self._queue.popleft()
                    self._cv.notify_all()
                last_observed_flags = tuple(UNKNOWN_COMPLETION_FLAG for _ in range(self._ep_size))
                poll_timed_out = False
                continue

            if time.monotonic() - watch.start_s >= self._timeout_s:
                try:
                    self._handle_timeout(watch, observed_flags, poll_timed_out=poll_timed_out)
                except Exception as exc:  # noqa: BLE001 - keep watchdog failures visible.
                    self._stop_after_error(exc)
                    tllm_logger.error(
                        "AlltoAll watchdog stopped after timeout handling error: %s", exc
                    )
                    return
                with self._cv:
                    # The GPU stream is no longer trustworthy once a collective
                    # times out. Drop queued follow-on phases so they do not
                    # produce duplicate or misleading reports.
                    self._queue.clear()
                    self._cv.notify_all()
                last_observed_flags = tuple(UNKNOWN_COMPLETION_FLAG for _ in range(self._ep_size))
                poll_timed_out = False
                continue

            with self._cv:
                self._cv.wait(timeout=self._poll_interval_s)
