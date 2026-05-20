# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Disaggregated cancellation stress-test harness.

Five-thread architecture that drives a 3P3D disaggregated TRT-LLM
cluster under cancellation-heavy load for hours, with scheduled
failure injection (SIGSTOP / SIGCONT / SIGKILL+respawn), a parallel
canary client for use-after-free detection, log-pattern fail-fast,
and KV-cache utilization scraping for leak detection.

The class structure and lifecycle (``setup`` -> ``start`` ->
``wait_until_done`` -> ``stop`` -> ``collect_results``) is wired up
here; individual thread bodies are filled in incrementally as the
suite matures.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

# YAML -> Python type coercion for the typed ``stress_config`` fields.
# Keys must match the ``StressConfig`` dataclass field names. Defaults
# live on the dataclass declarations themselves; missing YAML keys
# simply aren't passed to the constructor, so the field defaults
# apply automatically and are not duplicated here.
_STRESS_CONFIG_COERCERS: dict[str, Callable[[Any], Any]] = {
    "duration_min": int,
    "kv_cache_manager": str,
    "transceiver": str,
    "base_concurrency": int,
    "client_cancel_rate": float,
    "output_length": int,
}


@dataclass
class StressConfig:
    """Parsed contents of a marathon YAML's ``stress_config:`` block.

    The schema is documented in the directory ``README.md``. The
    fields here are exposed as a flat dataclass so the harness can
    pass them around without re-parsing.
    """

    duration_min: int = 120
    kv_cache_manager: str = "v1"  # v1 | v2  (v2 + CPP is invalid)
    transceiver: str = "cpp"  # cpp | python
    base_concurrency: int = 64
    client_cancel_rate: float = 0.10
    output_length: int = 512
    # Full YAML subtree, for fields the harness consumes lazily
    # (bursts, injections, canary, log_scan, kv_cache_growth_max).
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml_path(cls, path: Path) -> "StressConfig":
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if "stress_config" not in doc:
            raise ValueError(
                f"YAML at {path} is missing the 'stress_config:' top-level block; "
                "see configs/README.md for the schema."
            )
        sc = doc["stress_config"]
        kwargs: dict[str, Any] = {"raw": sc}
        for name, coerce in _STRESS_CONFIG_COERCERS.items():
            if name in sc:
                kwargs[name] = coerce(sc[name])
        return cls(**kwargs)

    def validate(self) -> None:
        """Reject backend-knob combinations that are not supported."""
        if self.kv_cache_manager == "v2" and self.transceiver == "cpp":
            # The C++ transceiver (BindKvCacheTransceiver) only supports
            # the V1 KV cache manager. V2 must be paired with the Python
            # transceiver (KvCacheTransceiverV2).
            raise ValueError(
                "(kv_cache_manager=v2, transceiver=cpp) is an unsupported "
                "combination; pair V2 with the Python transceiver."
            )
        if self.kv_cache_manager not in ("v1", "v2"):
            raise ValueError(
                f"kv_cache_manager must be 'v1' or 'v2', got {self.kv_cache_manager!r}"
            )
        if self.transceiver not in ("cpp", "python"):
            raise ValueError(f"transceiver must be 'cpp' or 'python', got {self.transceiver!r}")


@dataclass
class WorkerLaunchSpec:
    """Shadow-tracked launch context for a single worker.

    Recorded at cluster-setup time so the injector thread can relaunch
    a SIGKILLed worker via ``_run_worker(*spec)`` without extending
    ``ProcessWrapper`` in shared disagg test infrastructure.
    """

    role: str  # "ctx" or "gen"
    index: int  # 0..(num_instances-1), e.g. gen_worker_0
    model_name: str
    worker_config: dict[str, Any]
    work_dir: str
    port: int  # the originally-allocated port; respawn may end up on a different one
    device: str  # CUDA_VISIBLE_DEVICES string, e.g. "0" or "0,1"
    env: dict[str, str]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class DisaggCancellationStressHarness:
    """Marathon harness coordinating cluster + load + canary + injector + log scan + metrics.

    Lifecycle: ``__init__`` → ``setup`` → ``start`` → ``wait_until_done``
    → ``stop`` → ``collect_results``. ``stop`` is idempotent; the
    pytest fixture should call it in ``finally`` to guarantee teardown.

    Threads are all daemon=True so they don't block process exit on
    catastrophic failure. Coordination via two ``threading.Event``s:

    - ``stop_event``: set by ``stop()`` (clean shutdown).
    - ``failed_event``: set by any thread on fail-fast condition (hard-zero
      log pattern, worker death). All other threads observe it and
      wind down promptly.

    Thread-based composition (rather than asyncio) keeps the
    subprocess-control injector, the file-tailing log scanner, and
    the HTTP/Prometheus metrics scraper failure-isolated and debugged
    independently. The load and canary threads each run their own
    asyncio event loops internally for HTTP I/O.
    """

    def __init__(self, yaml_path: Path) -> None:
        self.yaml_path: Path = yaml_path
        self.config: StressConfig = StressConfig.from_yaml_path(yaml_path)
        self.config.validate()

        # Coordination primitives.
        self.stop_event: threading.Event = threading.Event()
        self.failed_event: threading.Event = threading.Event()
        self._failure_reason: Optional[str] = None
        self._failure_lock = threading.Lock()

        # Cluster + worker tracking (populated by setup()).
        self._cluster: Any = None  # tuple returned by setup_disagg_cluster
        self._worker_specs: list[WorkerLaunchSpec] = []

        # Thread handles (populated by start()).
        self._load_thread: Optional[threading.Thread] = None
        self._canary_thread: Optional[threading.Thread] = None
        self._injector_thread: Optional[threading.Thread] = None
        self._log_scanner_thread: Optional[threading.Thread] = None
        self._metrics_thread: Optional[threading.Thread] = None

        # Result collection. Populated by thread bodies as they run.
        self._canary_records: list[dict[str, Any]] = []
        self._kv_utilization_samples: list[dict[str, Any]] = []
        self._injection_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Launch the disagg cluster from the YAML and record launch specs.

        Stub: real implementation delegates to ``setup_disagg_cluster``
        in ``tests/integration/defs/disaggregated/test_disaggregated.py``
        and shadow-tracks per-worker ``WorkerLaunchSpec`` so the
        injector thread can later relaunch a SIGKILLed worker without
        modifying shared infrastructure.
        """
        logger.info("[harness] setup() — stub: cluster not actually launched")

    def start(self) -> None:
        """Spawn the five worker threads. Returns immediately.

        Stub stage: each thread body is a no-op that returns
        immediately. The load-thread stub signals ``stop_event`` on
        exit so the lifecycle smoke ``start() -> wait_until_done() ->
        stop()`` completes cleanly without waiting out the
        ``wait_until_done`` timeout.
        """
        logger.info("[harness] start() — spawning 5 stub threads")
        self._load_thread = threading.Thread(
            target=self._load_thread_body, name="stress-load", daemon=True
        )
        self._canary_thread = threading.Thread(
            target=self._canary_thread_body, name="stress-canary", daemon=True
        )
        self._injector_thread = threading.Thread(
            target=self._injector_thread_body, name="stress-injector", daemon=True
        )
        self._log_scanner_thread = threading.Thread(
            target=self._log_scanner_thread_body, name="stress-log-scanner", daemon=True
        )
        self._metrics_thread = threading.Thread(
            target=self._metrics_thread_body, name="stress-metrics", daemon=True
        )
        for t in self._all_threads():
            t.start()

    def wait_until_done(self, timeout_s: Optional[float] = None) -> bool:
        """Block until ``stop_event`` or ``failed_event`` is set, or timeout.

        Returns True if stopped cleanly (stop_event), False if any
        thread tripped fail-fast (failed_event) or the timeout expired
        without either event.

        Stub: in this skeleton, ``start()``'s no-op threads exit
        immediately and set ``stop_event`` themselves, so this returns
        True almost instantly.
        """
        deadline = None
        if timeout_s is not None:
            deadline = threading.Event()
            threading.Timer(timeout_s, deadline.set).start()
        while not self.stop_event.is_set() and not self.failed_event.is_set():
            if deadline is not None and deadline.is_set():
                logger.warning("[harness] wait_until_done timed out after %ss", timeout_s)
                return False
            # Sleep with a small poll cadence so stop()/fail-fast wake us promptly.
            self.stop_event.wait(timeout=0.5)
        return not self.failed_event.is_set()

    def stop(self) -> None:
        """Signal threads to wind down, join them, then tear down the cluster.

        Idempotent. Safe to call from pytest's ``finally``.

        Threads that fail to join within the join timeout are logged
        as a warning but do not trip fail-fast on their own: a
        stale-thread leak during teardown is a hygiene issue, not a
        verdict on whether the marathon itself passed its pass
        criteria (failure_reason is reserved for actual harness
        observations like hard-zero log patterns, worker death, or
        respawn-timeout). The cluster is torn down regardless so its
        GPUs/ports do not leak alongside the thread.
        """
        if not self.stop_event.is_set():
            logger.info("[harness] stop() — setting stop_event")
            self.stop_event.set()
        for t in self._all_threads():
            if t is not None and t.is_alive():
                t.join(timeout=10.0)
                if t.is_alive():
                    logger.warning("[harness] thread %s did not join within 10s", t.name)
        self._teardown_cluster()

    # ------------------------------------------------------------------
    # Fail-fast support
    # ------------------------------------------------------------------

    def mark_failed(self, reason: str) -> None:
        """Trip the fail-fast event with a structured reason.

        Called by ``log_scanner_thread`` on hard-zero pattern hit, by
        ``injector_thread`` on respawn timeout, by ``wait_until_done``
        on worker death detection, etc.
        """
        with self._failure_lock:
            if self._failure_reason is None:
                self._failure_reason = reason
                logger.error("[harness] FAIL-FAST tripped: %s", reason)
                self.failed_event.set()

    @property
    def failure_reason(self) -> Optional[str]:
        with self._failure_lock:
            return self._failure_reason

    # ------------------------------------------------------------------
    # Result collection (called by pytest after stop())
    # ------------------------------------------------------------------

    def collect_results(self) -> dict[str, Any]:
        """Aggregate per-thread observations into a final pass/fail dict.

        Stub: returns empty buckets in the skeleton. The full
        implementation computes canary error rates, recovery times,
        and KV-cache growth from ``self._canary_records``, etc.
        """
        return {
            "canary_records": list(self._canary_records),
            "kv_utilization_samples": list(self._kv_utilization_samples),
            "injection_events": list(self._injection_events),
            "failure_reason": self.failure_reason,
        }

    # ------------------------------------------------------------------
    # Thread bodies (stubs — implemented incrementally)
    # ------------------------------------------------------------------

    def _load_thread_body(self) -> None:
        """Wrap ``run_cancel_stress_test`` in a duration-bounded loop.

        Stub: no-op that immediately signals end-of-marathon via
        ``stop_event``. The real implementation loops until either
        ``duration_min`` elapses or ``stop_event`` is set, calling
        ``run_cancel_stress_test`` repeatedly; at end-of-marathon it
        sets ``stop_event`` so the other four threads wind down.
        Setting ``stop_event`` here in the stub preserves that
        downstream contract and lets ``wait_until_done`` return
        cleanly from the lifecycle smoke.
        """
        logger.debug("[load_thread] stub — exiting and signalling stop_event")
        self.stop_event.set()

    def _canary_thread_body(self) -> None:
        """Send greedy-decode canaries, check token-equivalence.

        Stub: no-op. Real implementation loads
        ``stress_canary_prompts.json``, sends 5 reqs/min, asserts
        token IDs match the recorded reference.
        """
        logger.debug("[canary_thread] stub — exiting immediately")

    def _injector_thread_body(self) -> None:
        """Fire SIGSTOP / SIGCONT / SIGKILL+respawn on the configured schedule.

        Stub: no-op. Real implementation reads ``injections:`` schedule
        from config and uses ``os.kill`` + ``_run_worker`` to act on
        the shadow-tracked ``self._worker_specs``.
        """
        logger.debug("[injector_thread] stub — exiting immediately")

    def _log_scanner_thread_body(self) -> None:
        """Tail all worker logs; fail-fast on any hard-zero pattern hit.

        Stub: no-op. Real implementation opens each worker's
        ``log_path``, follows it line-by-line, and calls
        ``mark_failed`` when a hard-zero pattern matches.
        """
        logger.debug("[log_scanner_thread] stub — exiting immediately")

    def _metrics_thread_body(self) -> None:
        """Scrape ``/prometheus/metrics`` for KV-cache utilization at ~30 s cadence.

        Stub: no-op. Real implementation parses
        ``trtllm_kv_cache_utilization`` and appends timestamped
        samples to ``self._kv_utilization_samples``.
        """
        logger.debug("[metrics_thread] stub — exiting immediately")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_threads(self) -> list[threading.Thread]:
        return [
            t
            for t in (
                self._load_thread,
                self._canary_thread,
                self._injector_thread,
                self._log_scanner_thread,
                self._metrics_thread,
            )
            if t is not None
        ]

    def _teardown_cluster(self) -> None:
        """Best-effort cluster shutdown via ``terminate()``.

        Stub: no-op since ``setup()`` doesn't actually launch yet.
        """
        if self._cluster is None:
            return
        logger.info("[harness] _teardown_cluster — stub")
