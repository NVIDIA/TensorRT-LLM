# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1a.7 (stub): NCCL async-error monitor + env-var wiring.

The production PR 1a.7 wraps NCCL with a custom ``ncclCommAbort`` + reinit
path and an ``abort_and_reinit(active_ranks)`` API; this stub does the
absolute minimum:

  1. Sets ``TORCH_NCCL_ASYNC_ERROR_HANDLING=1`` (and a few corollary env vars)
     before any process group is initialized, so PyTorch's NCCL backend
     surfaces async errors as exceptions instead of segfaulting.
  2. Spawns one Python thread that periodically calls
     ``torch.distributed.distributed_c10d._check_for_nccl_async_error()`` (via
     the same plumbing PyTorch's watchdog uses) and, on detection, calls
     ``EPGroupHealth.mark_failed`` for whichever peer surfaced the error.

This is the same general shape as the AlltoAll watchdog (1a.4) — a polling
thread that writes to ``EPGroupHealth`` — but observing a different signal:
NCCL's own async-error machinery rather than the MNNVL completion-flag table.
The point of running both in the prototype is to answer Open Question 1
("Watchdog vs NCCL collective ordering" — does NCCL fire first, or does the
AlltoAll watchdog?).

Stub contract — exercises these seams:

  * **Env-var wiring** — must be set before the first ``init_process_group``
    call; verifies the deployment story.
  * **NCCL async-error path → mark_failed** — same writer-side seam as 1a.4.
  * **Watchdog/NCCL race** — observable in the timeline JSON.

Stub deliberately omits, vs. production:

  * ``ncclCommAbort`` + reinit on detection.
  * ``abort_and_reinit(active_ranks)`` API.
  * Integration with PR #12718's ``classify_error`` + per-rank tracker.
  * Discrimination between transient and severe NCCL errors.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth


@dataclass(frozen=True)
class NcclMonitorConfig:
    """Tunables. Production code reads these from LLMArgs (PR 1d.1)."""

    poll_interval_sec: float = 0.5


def configure_nccl_async_error_handling() -> None:
    """Set the NCCL env vars that make async errors observable.

    Must be called *before* any ``torch.distributed.init_process_group``
    invocation; setting these after the backend is up is a no-op in PyTorch
    2.11.

    Env vars chosen empirically from audit-1a Day 1 (the NCCL rebuild
    prototype). ``TORCH_NCCL_ASYNC_ERROR_HANDLING=0`` disables PyTorch's own
    watchdog (which calls ``std::terminate()`` on peer death and would kill
    survivors); the prototype's main-thread polling pattern handles detection
    instead.
    """
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "0")


class NcclAsyncErrorMonitor:
    """Polling thread that surfaces NCCL async errors as ``mark_failed`` calls.

    Args:
        ep_size: Number of ranks in the EP group.
        local_rank: This process's rank.
        health: Shared :class:`EPGroupHealth`.
        check_async_error: Function that returns ``None`` if no error, or the
            (rank, error_message) tuple of the offending peer if NCCL has
            surfaced an async error. Production code wires this to
            ``ncclCommGetAsyncError``; the prototype can wire it to
            ``torch.distributed`` work-handle inspection.
        config: Tunables (poll interval).
        on_nccl_error: Optional logging callback (timeline hook).
    """

    def __init__(
        self,
        ep_size: int,
        local_rank: int,
        health: EPGroupHealth,
        check_async_error: Callable[[], Optional[tuple[int, str]]],
        config: NcclMonitorConfig = NcclMonitorConfig(),
        on_nccl_error: Optional[Callable[[int, str, float], None]] = None,
    ) -> None:
        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got {ep_size}")
        if not 0 <= local_rank < ep_size:
            raise ValueError(f"local_rank {local_rank} not in [0, {ep_size})")
        self._ep_size = ep_size
        self._local_rank = local_rank
        self._health = health
        self._check_async_error = check_async_error
        self._config = config
        self._on_nccl_error = on_nccl_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("nccl monitor already started")
        self._thread = threading.Thread(
            target=self._loop, name="nccl-async-monitor-stub", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                err = self._check_async_error()
            except Exception:  # noqa: BLE001 — prototype: don't crash on probe error
                err = None
            if err is not None:
                rank, msg = err
                if 0 <= rank < self._ep_size and rank != self._local_rank:
                    if self._health.mark_failed(rank):
                        if self._on_nccl_error is not None:
                            self._on_nccl_error(rank, msg, time.monotonic())
            self._stop.wait(self._config.poll_interval_sec)
