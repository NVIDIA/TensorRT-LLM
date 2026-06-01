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
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Optional

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
        """Parse and validate a marathon YAML in one call.

        Callers that construct ``StressConfig`` directly via
        ``__init__`` (no current consumer, but the API is left open)
        remain responsible for invoking ``validate()`` themselves.

        Args:
            path: Path to a marathon YAML containing a top-level
                ``stress_config:`` block. See ``configs/README.md``
                for the schema.

        Returns:
            A validated ``StressConfig`` whose typed fields are
            coerced from the YAML and whose ``raw`` attribute
            carries the full ``stress_config:`` subtree.

        Raises:
            ValueError: If the YAML is missing the ``stress_config:``
                block, or if ``validate()`` rejects the resulting
                backend-knob combination.
            yaml.YAMLError: If the file is not valid YAML.
            OSError: If ``path`` cannot be opened.
        """
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
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Reject backend-knob combinations that are not supported.

        Raises:
            ValueError: If ``kv_cache_manager`` is not ``"v1"`` or
                ``"v2"``, if ``transceiver`` is not ``"cpp"`` or
                ``"python"``, or if the pair ``(v2, cpp)`` is
                supplied (the C++ transceiver only supports the V1
                KV cache manager).
        """
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
    # Worker stdout/stderr log path (from ``ProcessWrapper.log_path``).
    # ``None`` when launched with ``save_log=False`` (output inherits
    # pytest stdout); log_scanner skips ``None`` paths with a warning.
    log_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Log-scanner helpers
# ---------------------------------------------------------------------------


@dataclass
class _LogSource:
    r"""One tailed log file plus the per-file state the scanner needs.

    The scanner runs in a poll loop (single thread, single file
    descriptor per source) rather than a per-source thread, because
    file tailing is bursty and the number of sources is small (3 ctx
    + 3 gen + 1 disagg server = 7 fds in the common 3P3D shape).

    Why ``_carry`` exists: ``trtllm-serve``'s C++ stack is
    block-buffered when stdout is a file, so reads are chunked rather
    than line-bounded. A naive ``for line in fh:`` loop would either
    block waiting for newlines (defeating the whole point of polling)
    or silently swallow the partial trailing line. We read all
    available bytes per poll, prepend the previous tail-byte carry,
    split on ``\n``, and treat the final element as either a complete
    line (if the chunk ended on a newline) or carry for the next
    poll.
    """

    spec: WorkerLaunchSpec
    path: Path
    _fh: Optional[IO[str]] = None
    _carry: str = ""

    def poll(
        self,
        patterns: list[tuple[str, re.Pattern[str]]],
        mark_failed: Callable[[str], None],
    ) -> bool:
        """Read new content and scan; report any pattern hit via ``mark_failed``.

        Maintains a tail-byte carry (``_carry``) across calls so
        block-buffered C++ writes (chunked rather than line-bounded)
        reconstruct into whole lines before being matched against
        patterns.

        Args:
            patterns: Compiled hard-zero patterns as
                ``(source_str, regex)`` pairs. ``source_str`` is the
                original YAML string and is embedded in the
                ``mark_failed`` reason for debugging.
            mark_failed: Callback invoked on the first pattern hit.
                Treated as one-shot per ``poll()`` call (the first
                match returns immediately without scanning the
                remainder of the chunk).

        Returns:
            True if a pattern matched and ``mark_failed`` was
            called; False if no new content was available, no
            pattern matched, or the file did not yet exist (handled
            silently so the scanner can be started before every
            worker has flushed its first bytes).
        """
        if self._fh is None:
            if not self.path.exists():
                return False
            self._fh = self.path.open("r", encoding="utf-8", errors="replace")

        chunk = self._fh.read()
        if not chunk:
            return False

        buf = self._carry + chunk
        lines = buf.split("\n")
        # If buf ended on '\n', lines[-1] == "" (a complete line
        # terminator). Either way, lines[-1] is the partial-or-empty
        # tail carried into the next poll.
        self._carry = lines[-1]

        for line in lines[:-1]:
            for pat_str, pat in patterns:
                if pat.search(line):
                    role_idx = f"{self.spec.role}_{self.spec.index}"
                    mark_failed(
                        f"hard-zero log pattern hit in {role_idx} "
                        f"({self.path.name}): {pat_str!r} matched {line.strip()!r}"
                    )
                    return True
        return False

    def close(self) -> None:
        """Close the tailed file handle, if any.

        Idempotent. ``OSError`` from the underlying close is logged
        at DEBUG and swallowed — the scanner is exiting and a
        failed close is not worth tripping fail-fast.
        """
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                logger.debug("[log_scanner] closing %s raised; ignoring", self.path)
            self._fh = None


def _compile_patterns(raw_patterns: list[Any]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile the YAML's ``log_scan.hard_zero_patterns`` list to regex objects.

    Patterns that fail to compile are skipped with an ERROR log so
    config typos surface during the scanner's startup banner rather
    than as silent misses at marathon hour 1.5.

    Args:
        raw_patterns: Pattern entries as parsed from YAML; non-string
            entries are skipped with an ERROR log.

    Returns:
        List of ``(source_str, compiled_regex)`` pairs.
        ``source_str`` is the original YAML string, retained so
        failure reasons can name which pattern matched.
    """
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for entry in raw_patterns:
        if not isinstance(entry, str):
            logger.error(
                "[log_scanner] hard_zero_patterns entry %r is not a string; skipping",
                entry,
            )
            continue
        try:
            compiled.append((entry, re.compile(entry)))
        except re.error as exc:
            logger.error(
                "[log_scanner] failed to compile hard_zero pattern %r: %s; skipping",
                entry,
                exc,
            )
    return compiled


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

    def __init__(
        self,
        yaml_path: Path,
        *,
        log_scanner_poll_interval_s: float = 0.5,
    ) -> None:
        """Construct a marathon harness.

        Args:
            yaml_path: Path to a marathon YAML; loaded and validated
                eagerly via ``StressConfig.from_yaml_path``.
            log_scanner_poll_interval_s: Poll cadence (seconds) for
                the log-scanner thread. Default 0.5 s is reactive
                enough for human-scale debugging without becoming a
                measurable load source on its own; tests pass a
                smaller value (e.g. 0.02 s) to keep real-clock
                latency bounded.

        Raises:
            ValueError: If the YAML is malformed or its
                ``stress_config:`` block is missing / rejects
                validation.
        """
        self.yaml_path: Path = yaml_path
        self.config: StressConfig = StressConfig.from_yaml_path(yaml_path)

        # Coordination primitives.
        self.stop_event: threading.Event = threading.Event()
        self.failed_event: threading.Event = threading.Event()
        self._failure_reason: Optional[str] = None
        self._failure_lock = threading.Lock()

        # Per-thread tunables. The default cadence is reactive enough
        # for human-scale debugging (~0.5 s lag from log line to
        # fail-fast) without becoming a measurable load source on its
        # own; tests pass a smaller value via the constructor to keep
        # real-clock latency bounded.
        self._log_scanner_poll_interval_s: float = log_scanner_poll_interval_s

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

        Stub: in this skeleton, ``start()``'s no-op threads exit
        immediately and set ``stop_event`` themselves, so this returns
        ``True`` almost instantly.

        Deadline is a ``time.monotonic()`` reading computed once and
        checked on each 0.5 s poll wake-up (worst-case lateness ~0.5
        s, well below the marathon's ``timeout_s`` scale). The earlier
        ``threading.Timer`` approach leaked a non-daemon timer thread
        for the residual window on early-stop.

        Args:
            timeout_s: Optional ceiling on the wait, in seconds.
                ``None`` (the default) waits indefinitely until one
                of the events fires.

        Returns:
            True if stopped cleanly (``stop_event``); False if any
            thread tripped fail-fast (``failed_event``) or the
            timeout expired without either event being set.
        """
        deadline_at = None if timeout_s is None else (time.monotonic() + timeout_s)
        while not self.stop_event.is_set() and not self.failed_event.is_set():
            if deadline_at is not None and time.monotonic() >= deadline_at:
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

        First-reason-wins: subsequent calls (e.g. a second worker's
        log scan firing right after the first) are silently dropped
        so ``failure_reason`` reflects the original root cause rather
        than a later cascade.

        Called by ``log_scanner_thread`` on hard-zero pattern hit, by
        ``injector_thread`` on respawn timeout, by ``wait_until_done``
        on worker death detection, etc.

        Args:
            reason: Human-readable failure description. Logged at
                ERROR and exposed via the ``failure_reason``
                property.
        """
        with self._failure_lock:
            if self._failure_reason is None:
                self._failure_reason = reason
                logger.error("[harness] FAIL-FAST tripped: %s", reason)
                self.failed_event.set()

    @property
    def failure_reason(self) -> Optional[str]:
        """Reason of the first ``mark_failed`` call, or ``None`` if not tripped.

        Read under the failure lock so it stays consistent with
        ``failed_event``: if ``failed_event.is_set()`` is True, this
        is guaranteed to return a non-None string.
        """
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

        Returns:
            A dict with four keys (the list values are copies, safe
            for the caller to mutate without affecting the harness):

            - ``canary_records``: per-canary request outcomes.
            - ``kv_utilization_samples``: timestamped KV-cache
              utilization scrapes from the metrics thread.
            - ``injection_events``: SIGSTOP / SIGCONT / SIGKILL
              events fired by the injector thread.
            - ``failure_reason``: ``None`` if the marathon completed
              cleanly, otherwise the first reason passed to
              ``mark_failed``.
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

        Reads ``hard_zero_patterns`` from ``stress_config.log_scan``,
        compiles each as a regex, then poll-tails every worker's
        ``log_path`` at ``_log_scanner_poll_interval_s``. On the first
        match in any worker, ``mark_failed`` is called (idempotent —
        first-reason-wins in the existing skeleton plumbing) and the
        thread continues polling until ``stop_event`` /
        ``failed_event``: the first failure is enough for fail-fast,
        but tailing past it costs us nothing and keeps the file
        handles drained until teardown.

        Workers without a ``log_path`` (``save_log=False`` at launch)
        are skipped with a warning. The scanner is intentionally
        permissive about not-yet-created files: ``_LogSource.poll``
        retries lazily on each cycle, so the thread can be started
        before every worker has flushed its first bytes.
        """
        raw_log_scan = self.config.raw.get("log_scan") or {}
        patterns = _compile_patterns(raw_log_scan.get("hard_zero_patterns") or [])
        if not patterns:
            logger.warning(
                "[log_scanner] no usable hard_zero_patterns; exiting immediately. "
                "Check stress_config.log_scan.hard_zero_patterns in the YAML."
            )
            return

        sources: list[_LogSource] = []
        for spec in self._worker_specs:
            if spec.log_path is None:
                logger.warning(
                    "[log_scanner] worker %s_%d has no log_path; skipping "
                    "(launch with save_log=True to capture worker output)",
                    spec.role,
                    spec.index,
                )
                continue
            sources.append(_LogSource(spec=spec, path=Path(spec.log_path)))

        if not sources:
            logger.warning(
                "[log_scanner] no log sources to tail; exiting immediately. "
                "The marathon will run without log-pattern fail-fast coverage."
            )
            return

        logger.info(
            "[log_scanner] tailing %d worker log(s) against %d hard_zero pattern(s)",
            len(sources),
            len(patterns),
        )

        try:
            while not self.stop_event.is_set() and not self.failed_event.is_set():
                for source in sources:
                    if self.stop_event.is_set() or self.failed_event.is_set():
                        break
                    source.poll(patterns, self.mark_failed)
                self.stop_event.wait(timeout=self._log_scanner_poll_interval_s)
        finally:
            for source in sources:
                source.close()
            logger.debug("[log_scanner] exiting; closed %d source(s)", len(sources))

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
        """Return the subset of thread handles that have been instantiated.

        Used by ``start()`` to launch and by ``stop()`` to join.
        Filters out ``None`` so callers can iterate without guarding
        (e.g. when a thread handle was never populated because
        ``start()`` failed partway through).

        Returns:
            Thread handles in deterministic order (load, canary,
            injector, log_scanner, metrics), skipping any not yet
            instantiated.
        """
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
