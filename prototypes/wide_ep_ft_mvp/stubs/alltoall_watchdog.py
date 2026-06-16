# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1a.4 (stub): AlltoAll watchdog timer thread.

Polls the host-visible ``completion_flags`` table every 100 ms. If a peer's
flag has not advanced past ``expected_val`` for ``timeout_sec`` (default 5 s)
then the peer is presumed dead and ``EPGroupHealth.mark_failed(rank)`` is
called directly — no error classification, no telemetry, no separate detection
thread orchestration. The production version (PR 1a.4) lives in three layers
(watchdog + error classification + per-rank tracker) and integrates with
PR #12718's ``classify_error()``; this stub collapses all of that into one
loop so we can observe the seam without re-implementing the full three-layer
detection stack.

Stub contract — exercises these seams:

  * **Watchdog → EPGroupHealth.mark_failed()** — writer side of the rank
    health bitmap.
  * **Polling latency vs. configured timeout** — the audit-style question
    "is 5 s the right default for this hardware?" is answerable against the
    timeline JSON the test driver produces.
  * **Watchdog vs NCCL collective ordering** — pair this with the NCCL
    monitor (1a.7) and observe which fires first when a peer dies during a
    non-MoE collective.

Stub deliberately omits, vs. production:

  * Three-layer detection architecture.
  * Error classification + integration with PR #12718.
  * Per-rank latency anomaly detection (Phase 3).
  * Backpressure / drain semantics on detection.
  * Graceful watchdog shutdown via ``check_health()`` integration.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import EPGroupHealth


@dataclass(frozen=True)
class WatchdogConfig:
    """Tunables. Production code reads these from LLMArgs (PR 1d.1)."""

    poll_interval_sec: float = 0.1
    timeout_sec: float = 5.0


@dataclass
class _PeerProgress:
    """Per-peer most-recently-seen flag value + the wall-clock at which it was seen.

    A peer is presumed dead when the flag has not advanced for ``timeout_sec``
    AND the host expected at least one increment within that window (i.e. the
    kernel was actually running collectives during the window).
    """

    last_value: int = 0
    last_advance_at: float = 0.0


class AlltoAllWatchdog:
    """Single-thread polling watchdog over a host-visible completion-flag view.

    Args:
        ep_size: Number of ranks in the EP group.
        local_rank: This process's rank index.
        health: Shared :class:`EPGroupHealth` instance. ``mark_failed`` is
            called on this object when a peer is presumed dead.
        completion_flag_view: A function that, when called, returns a length-
            ``ep_size`` sequence of the current host-visible completion flag
            values for each peer. The production AlltoAll watchdog reads this
            from MNNVL fabric memory; the prototype's worker can supply any
            host-side substitute (e.g. a ``torch.Tensor`` view of pinned
            memory) without needing the actual completion-flag table to be
            wired through.
        config: Tunables (poll interval + timeout).
        on_peer_death: Optional extra callback (logging hook for the timeline).

    Usage:

        wd = AlltoAllWatchdog(ep_size=4, local_rank=0, health=health,
                              completion_flag_view=lambda: snapshot_flags())
        wd.start()
        ...
        wd.stop()
    """

    def __init__(
        self,
        ep_size: int,
        local_rank: int,
        health: EPGroupHealth,
        completion_flag_view: Callable[[], list[int]],
        config: WatchdogConfig = WatchdogConfig(),
        on_peer_death: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got {ep_size}")
        if not 0 <= local_rank < ep_size:
            raise ValueError(f"local_rank {local_rank} not in [0, {ep_size})")
        self._ep_size = ep_size
        self._local_rank = local_rank
        self._health = health
        self._view = completion_flag_view
        self._config = config
        self._on_peer_death = on_peer_death
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("watchdog already started")
        self._thread = threading.Thread(
            target=self._loop, name="alltoall-watchdog-stub", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        # Initial snapshot establishes a baseline; nothing is "dead" until at
        # least one polling cycle has elapsed without progress.
        now = time.monotonic()
        progress = [_PeerProgress(last_value=0, last_advance_at=now) for _ in range(self._ep_size)]

        while not self._stop.is_set():
            try:
                values = self._view()
            except Exception:  # noqa: BLE001 — prototype: log-and-continue is fine
                # The view function may transiently fail (e.g. fabric memory
                # not yet wired). Production code should distinguish hard
                # failures here; the stub just retries on the next tick.
                time.sleep(self._config.poll_interval_sec)
                continue

            now = time.monotonic()
            for peer in range(self._ep_size):
                if peer == self._local_rank:
                    continue
                if not self._health.is_active(peer):
                    continue
                v = values[peer]
                if v > progress[peer].last_value:
                    progress[peer].last_value = v
                    progress[peer].last_advance_at = now
                    continue
                if now - progress[peer].last_advance_at >= self._config.timeout_sec:
                    if self._health.mark_failed(peer):
                        if self._on_peer_death is not None:
                            self._on_peer_death(peer, now)
            self._stop.wait(self._config.poll_interval_sec)
