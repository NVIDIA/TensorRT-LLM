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
import os
import random
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
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


_INJECTION_TARGET_RE = re.compile(r"^(ctx|gen)_worker_(\d+)$")


@dataclass
class _TrackedWorker:
    """Runtime binding between a shadow ``WorkerLaunchSpec`` and its subprocess."""

    spec: WorkerLaunchSpec
    wrapper: Any  # disagg_test_utils.ProcessWrapper


@dataclass(frozen=True)
class _InjectionSpec:
    """One entry from ``stress_config.injections`` after validation."""

    at_min: float
    type: str  # sigstop | sigkill
    target: str
    duration_s: Optional[float] = None
    respawn_within_s: Optional[float] = None


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
    # Host for HTTP scraping (Prometheus / health). Multi-host setups
    # set this from the YAML ``hostname:`` field at cluster-setup time.
    host: str = "localhost"


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


def _parse_injection_schedule(raw_injections: list[Any]) -> list[_InjectionSpec]:
    """Normalize and validate the YAML ``injections:`` schedule.

    Malformed entries are logged at ERROR and skipped so a typo in one
    slot does not abort the whole marathon.

    Args:
        raw_injections: Value of ``stress_config.injections`` from YAML.

    Returns:
        Sorted list of validated injection specs (by ``at_min``).
    """
    specs: list[_InjectionSpec] = []
    for idx, entry in enumerate(raw_injections):
        if not isinstance(entry, dict):
            logger.error("[injector] injections[%d] is not a mapping; skipping", idx)
            continue
        try:
            at_min = float(entry["at_min"])
            inj_type = str(entry["type"]).lower()
            target = str(entry["target"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("[injector] injections[%d] missing required fields: %s", idx, exc)
            continue
        if inj_type not in ("sigstop", "sigkill"):
            logger.error("[injector] injections[%d] unsupported type %r; skipping", idx, inj_type)
            continue
        duration_s = entry.get("duration_s")
        respawn_within_s = entry.get("respawn_within_s")
        if inj_type == "sigstop":
            if duration_s is None:
                logger.error("[injector] injections[%d] sigstop missing duration_s; skipping", idx)
                continue
            try:
                duration_s = float(duration_s)
            except (TypeError, ValueError):
                logger.error(
                    "[injector] injections[%d] invalid duration_s %r; skipping",
                    idx,
                    duration_s,
                )
                continue
        if inj_type == "sigkill" and respawn_within_s is not None:
            try:
                respawn_within_s = float(respawn_within_s)
            except (TypeError, ValueError):
                logger.error(
                    "[injector] injections[%d] invalid respawn_within_s %r; skipping",
                    idx,
                    respawn_within_s,
                )
                continue
        specs.append(
            _InjectionSpec(
                at_min=at_min,
                type=inj_type,
                target=target,
                duration_s=duration_s,
                respawn_within_s=respawn_within_s,
            )
        )
    specs.sort(key=lambda s: s.at_min)
    return specs


def _resolve_injection_target(target: str, tracked: list[_TrackedWorker]) -> _TrackedWorker:
    """Map a YAML target string to a tracked worker.

    Args:
        target: One of ``gen_worker_random``, ``ctx_worker_random``,
            or ``{ctx|gen}_worker_<index>``.
        tracked: Workers registered at cluster setup.

    Returns:
        The selected ``_TrackedWorker``.

    Raises:
        ValueError: If ``target`` is unknown or no worker matches.
    """
    if target == "gen_worker_random":
        pool = [t for t in tracked if t.spec.role == "gen"]
    elif target == "ctx_worker_random":
        pool = [t for t in tracked if t.spec.role == "ctx"]
    else:
        match = _INJECTION_TARGET_RE.match(target)
        if match is None:
            raise ValueError(f"unsupported injection target {target!r}")
        role, index = match.group(1), int(match.group(2))
        pool = [t for t in tracked if t.spec.role == role and t.spec.index == index]
    if not pool:
        raise ValueError(f"no worker matches injection target {target!r}")
    if target.endswith("_random"):
        return random.choice(pool)
    return pool[0]


def _worker_process_pid(wrapper: Any) -> Optional[int]:
    """Return the worker subprocess PID, or ``None`` if unavailable."""
    if wrapper is None or wrapper.process is None:
        return None
    return wrapper.process.pid


def _signal_process(pid: int, sig: signal.Signals, label: str) -> bool:
    """Send ``sig`` to ``pid``; log and return False on failure."""
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        logger.warning("[injector] %s: process %d already gone", label, pid)
        return False
    except OSError as exc:
        logger.warning("[injector] %s: os.kill(%d, %s) failed: %s", label, pid, sig.name, exc)
        return False


def _execute_sigstop_pause(
    wrapper: Any,
    duration_s: float,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """SIGSTOP a worker for ``duration_s``, then SIGCONT.

    Args:
        wrapper: ``ProcessWrapper`` for the victim worker.
        duration_s: Pause length in seconds.
        should_stop: Optional predicate polled during the pause; when
            it returns True the pause is cut short (and ``interrupted``
            is recorded), but SIGCONT is still sent so the worker is
            never left stopped.

    Returns:
        Event metadata dict for ``_injection_events``.
    """
    pid = _worker_process_pid(wrapper)
    outcome: dict[str, Any] = {"type": "sigstop", "duration_s": duration_s, "pid": pid}
    if pid is None:
        outcome["skipped"] = "no_process"
        return outcome
    stopped = _signal_process(pid, signal.SIGSTOP, "SIGSTOP")
    outcome["sigstop_sent"] = stopped
    cont_sent = False
    try:
        if stopped and duration_s > 0:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                if should_stop is not None and should_stop():
                    outcome["interrupted"] = True
                    break
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        if stopped:
            cont_sent = _signal_process(pid, signal.SIGCONT, "SIGCONT")
    outcome["sigcont_sent"] = cont_sent
    return outcome


def _execute_sigkill(wrapper: Any) -> dict[str, Any]:
    """SIGKILL the worker subprocess.

    Returns:
        Event metadata dict for ``_injection_events``.
    """
    pid = _worker_process_pid(wrapper)
    outcome: dict[str, Any] = {"type": "sigkill", "pid": pid}
    if pid is None:
        outcome["skipped"] = "no_process"
        return outcome
    outcome["sigkill_sent"] = _signal_process(pid, signal.SIGKILL, "SIGKILL")
    if wrapper.process is not None:
        try:
            wrapper.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            logger.warning("[injector] SIGKILL: pid %d did not exit within 10s", pid)
    return outcome


# ---------------------------------------------------------------------------
# Metrics-thread helpers
# ---------------------------------------------------------------------------


# Tolerates optional ``{labels}`` and trailing timestamp; captures value in group 1.
_KV_CACHE_UTIL_RE = re.compile(
    r"^trtllm_kv_cache_utilization(?:\{[^}]*\})?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def _parse_kv_cache_utilization(metrics_text: str) -> Optional[float]:
    """Extract ``trtllm_kv_cache_utilization`` from a Prometheus exposition.

    Skips ``# HELP`` / ``# TYPE`` lines. Returns the first matching
    sample's value as a float, or ``None`` if no sample is present.

    Args:
        metrics_text: Body of an HTTP response from
            ``/prometheus/metrics``.

    Returns:
        Float utilization in the range the worker exposes (typically
        ``[0.0, 1.0]``), or ``None`` if the metric is absent.
    """
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _KV_CACHE_UTIL_RE.match(line)
        if m is not None:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _fetch_kv_cache_utilization(
    host: str, port: int, timeout_s: float
) -> tuple[Optional[float], Optional[str]]:
    """HTTP GET ``/prometheus/metrics`` and parse the KV-cache utilization.

    All failures (connection refused, timeout, HTTP error, parse miss)
    are folded into ``(None, error_string)`` rather than raising. The
    metrics thread must not fail-fast on transient scrape misses — the
    worker may be mid-restart, the metrics endpoint may not be wired
    up on every backend, etc.

    Args:
        host: Worker hostname or IP.
        port: Worker HTTP port.
        timeout_s: Per-request timeout in seconds.

    Returns:
        Tuple ``(utilization, error)``. Exactly one is ``None``: on
        success ``utilization`` is the gauge value and ``error`` is
        ``None``; on failure ``utilization`` is ``None`` and ``error``
        is a short string explaining why (for the timeseries record).
    """
    url = f"http://{host}:{port}/prometheus/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return None, f"url_error: {exc.reason}"
    except (TimeoutError, OSError) as exc:
        return None, f"io_error: {exc}"
    util = _parse_kv_cache_utilization(body)
    if util is None:
        return None, "metric_absent"
    return util, None


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
        metrics_scrape_interval_s: float = 30.0,
        metrics_scrape_timeout_s: float = 5.0,
        injector_poll_interval_s: float = 1.0,
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
            metrics_scrape_interval_s: Cadence (seconds) between
                consecutive Prometheus scrapes of every worker.
                Default 30 s matches the spec's KV-cache utilization
                sampling rate; tests pass a smaller value.
            metrics_scrape_timeout_s: Per-request HTTP timeout for
                the metrics scrape. Short by design — a slow scrape
                is recorded as a miss rather than blocking the
                metrics thread past its next scheduled scrape.
            injector_poll_interval_s: Poll cadence (seconds) for the
                injector thread while waiting for the next scheduled
                event. Tests pass a smaller value to keep wall-clock
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

        self._log_scanner_poll_interval_s: float = log_scanner_poll_interval_s
        self._metrics_scrape_interval_s: float = metrics_scrape_interval_s
        self._metrics_scrape_timeout_s: float = metrics_scrape_timeout_s
        self._injector_poll_interval_s: float = injector_poll_interval_s

        # Cluster + worker tracking (populated by setup()).
        self._cluster: Any = None  # tuple returned by setup_disagg_cluster
        self._worker_specs: list[WorkerLaunchSpec] = []
        self._tracked_workers: list[_TrackedWorker] = []
        self._marathon_start_monotonic: float = 0.0

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

    def bind_tracked_workers(
        self,
        ctx_workers: list[Any],
        gen_workers: list[Any],
        ctx_specs: list[WorkerLaunchSpec],
        gen_specs: list[WorkerLaunchSpec],
    ) -> None:
        """Register live ``ProcessWrapper`` handles for the injector thread.

        Called by ``setup()`` once ``setup_disagg_cluster`` returns.
        Shadow ``WorkerLaunchSpec`` entries must align 1:1 with the
        wrapper lists (same ordering as ``setup_disagg_cluster``);
        ``zip(..., strict=True)`` enforces this and raises ``ValueError``
        on a length mismatch.
        """
        self._tracked_workers = [
            _TrackedWorker(spec=spec, wrapper=wrapper)
            for spec, wrapper in zip(ctx_specs, ctx_workers, strict=True)
        ] + [
            _TrackedWorker(spec=spec, wrapper=wrapper)
            for spec, wrapper in zip(gen_specs, gen_workers, strict=True)
        ]
        self._worker_specs = list(ctx_specs) + list(gen_specs)

    def start(self) -> None:
        """Spawn the five worker threads. Returns immediately.

        Stub stage: each thread body is a no-op that returns
        immediately. The load-thread stub signals ``stop_event`` on
        exit so the lifecycle smoke ``start() -> wait_until_done() ->
        stop()`` completes cleanly without waiting out the
        ``wait_until_done`` timeout.
        """
        self._marathon_start_monotonic = time.monotonic()
        logger.info("[harness] start() — spawning worker threads")
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

        Reads ``injections:`` from ``stress_config``, waits until each
        ``at_min`` offset from marathon start, then acts on
        ``self._tracked_workers`` via ``os.kill``. SIGKILL entries
        optionally relaunch the worker via ``disagg_test_utils._run_worker``
        and fail-fast if ``/health`` does not return 200 within
        ``respawn_within_s``.
        """
        schedule = _parse_injection_schedule(self.config.raw.get("injections") or [])
        if not schedule:
            logger.warning("[injector] no injections in stress_config; exiting immediately")
            return
        if not self._tracked_workers:
            logger.warning(
                "[injector] no tracked workers (setup() did not register any); "
                "exiting without running the injection schedule"
            )
            return

        pending = list(schedule)
        logger.info(
            "[injector] armed with %d injection(s) across %d tracked worker(s)",
            len(pending),
            len(self._tracked_workers),
        )

        try:
            while pending and not self.stop_event.is_set() and not self.failed_event.is_set():
                elapsed_s = time.monotonic() - self._marathon_start_monotonic
                next_inj = pending[0]
                fire_at_s = next_inj.at_min * 60.0
                if elapsed_s < fire_at_s:
                    wait_s = min(self._injector_poll_interval_s, fire_at_s - elapsed_s)
                    self.stop_event.wait(timeout=wait_s)
                    continue

                pending.pop(0)
                try:
                    tracked = _resolve_injection_target(next_inj.target, self._tracked_workers)
                except ValueError as exc:
                    logger.error(
                        "[injector] skipping injection at T+%.1f min: %s",
                        next_inj.at_min,
                        exc,
                    )
                    self._record_injection_event(
                        {
                            "at_min": next_inj.at_min,
                            "target": next_inj.target,
                            "skipped": str(exc),
                        }
                    )
                    continue

                role_idx = f"{tracked.spec.role}_{tracked.spec.index}"
                logger.info(
                    "[injector] T+%.1f min: %s on %s (pid=%s)",
                    next_inj.at_min,
                    next_inj.type,
                    role_idx,
                    _worker_process_pid(tracked.wrapper),
                )
                event: dict[str, Any] = {
                    "at_min": next_inj.at_min,
                    "elapsed_s": elapsed_s,
                    "target": next_inj.target,
                    "role": tracked.spec.role,
                    "index": tracked.spec.index,
                }
                if next_inj.type == "sigstop":
                    event.update(
                        _execute_sigstop_pause(
                            tracked.wrapper,
                            next_inj.duration_s or 0.0,
                            should_stop=lambda: self.stop_event.is_set()
                            or self.failed_event.is_set(),
                        )
                    )
                elif next_inj.type == "sigkill":
                    event.update(_execute_sigkill(tracked.wrapper))
                    if next_inj.respawn_within_s is not None:
                        respawned = self._respawn_tracked_worker(
                            tracked, timeout_s=next_inj.respawn_within_s
                        )
                        event["respawned"] = respawned
                        event["respawn_within_s"] = next_inj.respawn_within_s
                        if not respawned:
                            self.mark_failed(
                                f"injector: {role_idx} did not become healthy within "
                                f"{next_inj.respawn_within_s}s after SIGKILL"
                            )
                self._record_injection_event(event)
        finally:
            logger.debug("[injector] exiting; %d injection(s) remaining", len(pending))

    def _record_injection_event(self, event: dict[str, Any]) -> None:
        """Append one injector observation (thread-safe for single writer)."""
        self._injection_events.append(event)

    def _respawn_tracked_worker(self, tracked: _TrackedWorker, *, timeout_s: float) -> bool:
        """Relaunch a SIGKILLed worker and wait for ``/health``.

        Uses the shadow ``WorkerLaunchSpec`` recorded at setup time so
        shared ``ProcessWrapper`` infrastructure stays unchanged.

        Args:
            tracked: Worker to relaunch.
            timeout_s: Maximum seconds to wait for HTTP 200 on ``/health``.

        Returns:
            True if the respawned worker reports healthy within the
            deadline; False otherwise.
        """
        from disagg_test_utils import _run_worker, get_free_port

        spec = tracked.spec
        old = tracked.wrapper

        if old is not None and old.log_file is not None:
            try:
                old.log_file.close()
            except OSError:
                logger.debug("[injector] closing old log_file raised; ignoring")

        role_key = "ctx" if spec.role == "ctx" else "gen"
        save_log = spec.log_path is not None
        new_port = get_free_port()
        try:
            new_wrapper = _run_worker(
                spec.model_name,
                spec.worker_config,
                role_key,
                port=new_port,
                work_dir=spec.work_dir,
                device=spec.device,
                save_log=save_log,
                env=spec.env,
            )
        except Exception:
            logger.exception(
                "[injector] failed to respawn %s_%d on port %d",
                spec.role,
                spec.index,
                new_port,
            )
            return False
        tracked.wrapper = new_wrapper
        spec.port = new_wrapper.port
        if new_wrapper.log_path is not None:
            spec.log_path = new_wrapper.log_path

        logger.info(
            "[injector] respawned %s_%d on port %s; waiting up to %.0fs for /health",
            spec.role,
            spec.index,
            new_wrapper.port,
            timeout_s,
        )

        return self._wait_for_worker_health(new_wrapper.port, timeout_s=timeout_s)

    def _wait_for_worker_health(self, port: int, *, timeout_s: float) -> bool:
        """Poll a worker's ``/health`` endpoint until healthy or timed out."""
        import requests

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.stop_event.is_set() or self.failed_event.is_set():
                return False
            try:
                request_timeout = min(5.0, max(0.1, deadline - time.monotonic()))
                response = requests.get(f"http://localhost:{port}/health", timeout=request_timeout)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                logger.debug("[injector] worker %d /health not ready yet", port)
            self.stop_event.wait(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
        return False

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
        """Scrape Prometheus KV-cache utilization from every worker.

        Polls each ``WorkerLaunchSpec`` at
        ``_metrics_scrape_interval_s`` cadence, parses
        ``trtllm_kv_cache_utilization`` out of
        ``/prometheus/metrics``, and appends a timestamped sample to
        ``self._kv_utilization_samples``. Each sample records
        ``timestamp``, ``role``, ``index``, ``host``, ``port``,
        ``utilization`` (``float`` on success, ``None`` on scrape
        miss), and ``error`` (``None`` on success, short string on
        miss). Scrape misses are recorded — not fail-fast — because
        the worker may be mid-restart from a SIGKILL injection, mid-
        SIGSTOP pause, or the Prometheus endpoint may not be wired
        for the active backend.

        Exits when ``stop_event`` or ``failed_event`` is set. Honors
        either signal between scrapes via ``stop_event.wait`` rather
        than ``time.sleep`` to keep teardown latency bounded.
        """
        if not self._worker_specs:
            logger.warning("[metrics_thread] no worker specs; exiting")
            return

        logger.info(
            "[metrics_thread] scraping %d worker(s) every %.1fs",
            len(self._worker_specs),
            self._metrics_scrape_interval_s,
        )

        while not self.stop_event.is_set() and not self.failed_event.is_set():
            scrape_start = time.monotonic()
            for spec in self._worker_specs:
                if self.stop_event.is_set() or self.failed_event.is_set():
                    break
                util, err = _fetch_kv_cache_utilization(
                    spec.host, spec.port, self._metrics_scrape_timeout_s
                )
                self._kv_utilization_samples.append(
                    {
                        "timestamp": time.time(),
                        "role": spec.role,
                        "index": spec.index,
                        "host": spec.host,
                        "port": spec.port,
                        "utilization": util,
                        "error": err,
                    }
                )
            remaining = self._metrics_scrape_interval_s - (time.monotonic() - scrape_start)
            if remaining > 0.0:
                self.stop_event.wait(timeout=remaining)

        logger.debug("[metrics_thread] exiting; %d sample(s)", len(self._kv_utilization_samples))

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
