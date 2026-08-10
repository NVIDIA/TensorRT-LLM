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
"""Unit tests for ``DisaggCancellationStressHarness._log_scanner_thread_body``.

The log-scanner thread is the easiest component to test in isolation
because it has no dependency on a running disagg cluster: feed it a
list of ``WorkerLaunchSpec`` whose ``log_path`` points at a temp file
the test owns, write log lines into that file, and assert on
``harness.failure_reason``.

Buffering caveat the production scanner has to handle (chunked C++
``stdout`` writes from ``trtllm-serve``) is exercised by the
"partial / multi-line / no-trailing-newline" cases below.
"""

from __future__ import annotations

import re
import textwrap
import threading
import time
from pathlib import Path
from typing import Callable

import pytest

from ._testing import DUMMY_YAML, make_spec
from .harness import DisaggCancellationStressHarness, _LogSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness_with_two_workers(tmp_path: Path):
    """Harness wired to two temp-file log sources (ctx_0 + gen_0).

    Args:
        tmp_path: pytest-provided per-test temp dir.

    Returns:
        A 3-tuple ``(harness, ctx_log, gen_log)`` where the
        ``*_log`` entries are ``Path`` objects the test can write
        to. The harness is constructed with a 20 ms scanner cadence
        so test wall-clock times stay bounded.
    """
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(DUMMY_YAML)

    # Tighten the scanner cadence so the tests finish quickly. The
    # default 0.5 s makes tests slow; 20 ms keeps real-clock latency
    # bounded.
    h = DisaggCancellationStressHarness(yaml_path, log_scanner_poll_interval_s=0.02)

    ctx_log = tmp_path / "worker_ctx_18000.log"
    gen_log = tmp_path / "worker_gen_18001.log"
    ctx_log.touch()
    gen_log.touch()

    h._worker_specs = [
        make_spec("ctx", 0, log_path=ctx_log),
        make_spec("gen", 0, log_path=gen_log),
    ]
    return h, ctx_log, gen_log


def _run_scanner_until_failure_or_timeout(
    h: DisaggCancellationStressHarness,
    timeout_s: float = 2.0,
    settle_s: float = 0.05,
) -> bool:
    """Spawn the log-scanner thread; wait for ``failed_event`` or timeout.

    Args:
        h: Configured harness with ``_worker_specs`` already
            populated by the calling test.
        timeout_s: Maximum wall-clock seconds to wait for the
            scanner to trip ``failed_event``. Defaults to 2.0 s.
        settle_s: Pause after fail-fast fires (before
            ``stop_event.set()``) to let the scanner clean up file
            handles. Functionally optional; cosmetic for cleaner
            debug logs.

    Returns:
        True if fail-fast tripped within the timeout; False if the
        scanner exited cleanly without firing (e.g. no log sources,
        no patterns, or no matching content within the window).
    """
    t = threading.Thread(target=h._log_scanner_thread_body, name="test-log-scanner", daemon=True)
    t.start()
    fired = h.failed_event.wait(timeout=timeout_s)
    # Give the thread a moment to clean up file handles before we
    # signal stop_event (this keeps the test's debug logs more
    # readable; functionally optional).
    time.sleep(settle_s)
    h.stop_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "log_scanner thread did not exit after stop_event"
    return fired


# ---------------------------------------------------------------------------
# Tests: pattern matching
# ---------------------------------------------------------------------------


def test_hard_zero_pattern_in_ctx_log_trips_fail_fast(harness_with_two_workers):
    h, ctx_log, _ = harness_with_two_workers
    ctx_log.write_text("startup ok\nBroken promise: future destroyed\ntail\n")

    fired = _run_scanner_until_failure_or_timeout(h)

    assert fired is True
    assert h.failure_reason is not None
    assert "Broken promise" in h.failure_reason
    assert "ctx_0" in h.failure_reason
    assert "worker_ctx_18000.log" in h.failure_reason


def test_hard_zero_pattern_in_gen_log_trips_fail_fast(harness_with_two_workers):
    h, _, gen_log = harness_with_two_workers
    gen_log.write_text("warmup\nSegfault at 0xdeadbeef\n")

    fired = _run_scanner_until_failure_or_timeout(h)

    assert fired is True
    assert h.failure_reason is not None
    assert "Segfault" in h.failure_reason
    assert "gen_0" in h.failure_reason


def test_regex_metacharacters_match_correctly(harness_with_two_workers):
    """Pattern 'Poisoned .* cache transfer buffer' must match across a wildcard."""
    h, _, gen_log = harness_with_two_workers
    gen_log.write_text("Poisoned sender cache transfer buffer for request 42\n")

    fired = _run_scanner_until_failure_or_timeout(h)

    assert fired is True
    assert h.failure_reason is not None
    assert "Poisoned" in h.failure_reason


def test_benign_lines_do_not_trip_fail_fast(harness_with_two_workers):
    h, ctx_log, gen_log = harness_with_two_workers
    ctx_log.write_text(
        "INFO: trtllm-serve listening on 18000\nINFO: cache transceiver initialised\nINFO: ready\n"
    )
    gen_log.write_text("INFO: warmup complete\nINFO: serving requests\n")

    # No pattern hit ever, so we wait the full window then assert
    # nothing fired.
    fired = _run_scanner_until_failure_or_timeout(h, timeout_s=0.5)

    assert fired is False
    assert h.failure_reason is None


def test_first_match_wins_when_both_workers_hit(harness_with_two_workers):
    h, ctx_log, gen_log = harness_with_two_workers
    ctx_log.write_text("Broken promise: A\n")
    gen_log.write_text("Segfault at 0xbeef\n")

    fired = _run_scanner_until_failure_or_timeout(h)

    assert fired is True
    reason_before = h.failure_reason
    assert reason_before is not None
    # mark_failed is documented as first-reason-wins. Another call
    # should not overwrite the original reason. This is a guard
    # against future refactors weakening that contract.
    h.mark_failed("manual override that must not stick")
    assert h.failure_reason == reason_before


# ---------------------------------------------------------------------------
# Tests: I/O quirks (no trailing newline, chunked writes)
# ---------------------------------------------------------------------------


def test_partial_line_without_trailing_newline_is_carried(harness_with_two_workers):
    r"""A log write that doesn't end on '\n' must not lose the line.

    Simulates the C++ block-buffered-stdout pattern: two writes
    where the first write ends mid-line and the second completes it.
    The complete reconstructed line carries the hard-zero pattern.
    """
    h, ctx_log, _ = harness_with_two_workers
    # First write: partial line, no newline.
    ctx_log.write_text("Broken promi")

    t = threading.Thread(target=h._log_scanner_thread_body, daemon=True)
    t.start()
    time.sleep(0.1)  # let the scanner poll once and stash the carry

    assert not h.failed_event.is_set(), "scanner fired before the partial line was completed"

    # Second write: complete the line.
    with ctx_log.open("a", encoding="utf-8") as f:
        f.write("se: future destroyed\n")

    fired = h.failed_event.wait(timeout=2.0)
    h.stop_event.set()
    t.join(timeout=2.0)

    assert fired is True
    assert h.failure_reason is not None
    assert "Broken promise" in h.failure_reason


def test_chunked_multiline_write_finds_pattern_on_any_line(harness_with_two_workers):
    """A single fwrite of many lines must scan all of them."""
    h, _, gen_log = harness_with_two_workers
    lines = ["INFO: line 1\n", "INFO: line 2\n", "Segfault at 0x1\n", "INFO: line 4\n"]
    gen_log.write_text("".join(lines))

    fired = _run_scanner_until_failure_or_timeout(h)

    assert fired is True
    assert "Segfault" in (h.failure_reason or "")


# ---------------------------------------------------------------------------
# Tests: lifecycle / cancellation
# ---------------------------------------------------------------------------


def test_stop_event_exits_cleanly(harness_with_two_workers):
    h, ctx_log, gen_log = harness_with_two_workers
    ctx_log.write_text("INFO: nothing interesting\n")
    gen_log.write_text("INFO: also nothing\n")

    t = threading.Thread(target=h._log_scanner_thread_body, daemon=True)
    t.start()
    time.sleep(0.1)  # let the scanner reach its poll loop
    h.stop_event.set()
    t.join(timeout=2.0)

    assert not t.is_alive(), "log_scanner did not exit after stop_event"
    assert h.failure_reason is None


def test_failed_event_set_externally_exits_thread(harness_with_two_workers):
    """Externally-set ``failed_event`` must make the scanner wind down.

    Models the case where some other thread (canary / injector) trips
    fail-fast before the log scanner gets a chance to fire on its
    own. The scanner must observe the event and exit rather than
    continuing to poll.
    """
    h, _, _ = harness_with_two_workers
    h.mark_failed("set externally by some other thread")

    t = threading.Thread(target=h._log_scanner_thread_body, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert not t.is_alive(), "log_scanner did not observe failed_event set at start"


def test_log_file_created_after_scanner_starts(harness_with_two_workers):
    """A log file created after scanner startup must be picked up lazily.

    Models the race where the scanner spawns before the worker has
    flushed its first bytes (so the file doesn't exist yet at poll
    #0). The scanner must skip the missing source silently and retry
    on each cycle.
    """
    h, ctx_log, gen_log = harness_with_two_workers
    # Remove the touch()ed files so they don't exist when the scanner
    # spawns. ``_LogSource.poll`` should silently skip them.
    ctx_log.unlink()
    gen_log.unlink()

    t = threading.Thread(target=h._log_scanner_thread_body, daemon=True)
    t.start()
    time.sleep(0.1)  # scanner is polling; both files absent

    assert not h.failed_event.is_set()

    # Now create gen_log with a hit; scanner should pick it up.
    gen_log.write_text("Segfault at 0x0\n")

    fired = h.failed_event.wait(timeout=2.0)
    h.stop_event.set()
    t.join(timeout=2.0)

    assert fired is True
    assert "Segfault" in (h.failure_reason or "")


# ---------------------------------------------------------------------------
# Tests: empty / misconfigured input
# ---------------------------------------------------------------------------


def test_no_worker_specs_exits_immediately(tmp_path: Path):
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(DUMMY_YAML)
    h = DisaggCancellationStressHarness(yaml_path)
    # No specs at all.

    t = threading.Thread(target=h._log_scanner_thread_body, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert not t.is_alive(), "scanner did not exit with no log sources"
    assert h.failure_reason is None


def test_specs_with_none_log_path_skip_gracefully(tmp_path: Path):
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(DUMMY_YAML)
    h = DisaggCancellationStressHarness(yaml_path)
    h._worker_specs = [
        make_spec("ctx", 0, log_path=None),
        make_spec("gen", 0, log_path=None),
    ]

    t = threading.Thread(target=h._log_scanner_thread_body, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert h.failure_reason is None


def test_no_hard_zero_patterns_exits_immediately(tmp_path: Path):
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            hostname: localhost
            model: dummy
            backend: pytorch
            context_servers: {}
            generation_servers: {}
            stress_config:
              duration_min: 1
              kv_cache_manager: v1
              transceiver: cpp
              log_scan:
                hard_zero_patterns: []
            """
        )
    )
    h = DisaggCancellationStressHarness(yaml_path)
    ctx_log = tmp_path / "worker_ctx_18000.log"
    ctx_log.write_text("Broken promise: A\n")  # would match in normal config
    h._worker_specs = [make_spec("ctx", 0, log_path=ctx_log)]

    t = threading.Thread(target=h._log_scanner_thread_body, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert not t.is_alive()
    # No patterns -> scanner can't fire regardless of what's in the log.
    assert h.failure_reason is None


def test_invalid_regex_is_skipped_with_warning(tmp_path: Path, caplog):
    """A malformed regex must not crash the scanner.

    Bad entries in ``hard_zero_patterns`` should be reported as ERROR
    and skipped; the scanner continues with whatever other patterns
    compiled successfully.
    """
    yaml_path = tmp_path / "stress.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            hostname: localhost
            model: dummy
            backend: pytorch
            context_servers: {}
            generation_servers: {}
            stress_config:
              duration_min: 1
              kv_cache_manager: v1
              transceiver: cpp
              log_scan:
                hard_zero_patterns:
                  - "(unclosed"
                  - "Broken promise"
            """
        )
    )
    h = DisaggCancellationStressHarness(yaml_path, log_scanner_poll_interval_s=0.02)
    ctx_log = tmp_path / "worker_ctx_18000.log"
    ctx_log.write_text("Broken promise: A\n")
    h._worker_specs = [make_spec("ctx", 0, log_path=ctx_log)]

    with caplog.at_level("ERROR"):
        fired = _run_scanner_until_failure_or_timeout(h)

    # The bad pattern was rejected but the good one still tripped.
    assert fired is True
    assert "Broken promise" in (h.failure_reason or "")
    assert any("(unclosed" in record.getMessage() for record in caplog.records), (
        "expected an ERROR log naming the malformed regex"
    )


# ---------------------------------------------------------------------------
# Tests: _LogSource unit behaviour (without the harness wrapper)
# ---------------------------------------------------------------------------


def _consume_marks(seen: list[str]) -> Callable[[str], None]:
    """Build a ``mark_failed``-shaped callback that appends to a test list.

    Used by the standalone ``_LogSource`` tests (below the harness
    tests) to introspect what the source would have reported,
    without needing a full harness instance.

    Args:
        seen: Test-owned list that each invocation appends its
            ``reason`` argument to.

    Returns:
        A single-arg callable matching the
        ``Callable[[str], None]`` shape that ``_LogSource.poll``
        expects for its ``mark_failed`` parameter.
    """

    def _mark(reason: str) -> None:
        seen.append(reason)

    return _mark


def test_log_source_poll_returns_false_when_file_absent(tmp_path: Path):
    spec = make_spec("ctx", 0, log_path=tmp_path / "missing.log")
    src = _LogSource(spec=spec, path=Path(spec.log_path))  # type: ignore[arg-type]
    seen: list[str] = []
    assert src.poll([("X", re.compile("X"))], _consume_marks(seen)) is False
    assert seen == []


def test_log_source_poll_returns_false_on_empty_read(tmp_path: Path):
    log = tmp_path / "empty.log"
    log.touch()
    spec = make_spec("gen", 0, log_path=log)
    src = _LogSource(spec=spec, path=log)
    seen: list[str] = []
    assert src.poll([("X", re.compile("X"))], _consume_marks(seen)) is False
    assert seen == []
    src.close()


def test_log_source_close_is_idempotent(tmp_path: Path):
    log = tmp_path / "x.log"
    log.write_text("hello\n")
    spec = make_spec("ctx", 0, log_path=log)
    src = _LogSource(spec=spec, path=log)
    src.poll([("X", re.compile("X"))], lambda r: None)
    src.close()
    src.close()  # second close must not raise
