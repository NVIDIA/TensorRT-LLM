# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounds on the perf-sanity harness waits.

The perf-sanity stages could not fail on their own: the benchmark client
subprocess had no timeout, and every other harness wait was bounded at
DEFAULT_TIMEOUT (10800s), which sits above the pytest per-test marker. A stall
therefore surfaced only as "the client is still running" until Slurm or Jenkins
killed the stage hours later, with no results XML and no diagnostic.

These tests pin the two bounds that close that: a deadline on the client, and
SIGTERM->SIGKILL escalation on server teardown.
"""

import os
import subprocess
import sys
import time

import pytest

# Required to run in the CPU-Generic stage: tests/unittest/conftest.py's
# pytest_ignore_collect drops any file whose source lacks this literal when
# pytest runs with -m cpu_only. Nothing here needs a GPU.
pytestmark = pytest.mark.cpu_only

_INTEGRATION = os.path.join(os.path.dirname(__file__), "..", "..", "integration")
if _INTEGRATION not in sys.path:
    sys.path.insert(0, os.path.abspath(_INTEGRATION))

perf_sanity = pytest.importorskip("defs.perf.test_perf_sanity")


# ---------------------------------------------------------------------------
# The client deadline
# ---------------------------------------------------------------------------


def test_client_timeout_defaults_to_one_hour(monkeypatch):
    """The default must be finite -- an unset env var previously meant 'forever'."""
    # delenv, not os.environ.pop: pop would drop the variable for the rest of
    # the pytest process and silently change later tests in the same worker.
    monkeypatch.delenv(perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME, raising=False)
    assert perf_sanity._benchmark_client_timeout() == 3600


def test_client_timeout_honours_the_env_override(monkeypatch):
    monkeypatch.setenv(perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME, "120")
    assert perf_sanity._benchmark_client_timeout() == 120


def test_client_timeout_zero_disables_the_bound(monkeypatch):
    """0 must mean 'no deadline' (None), not 'expire immediately'."""
    monkeypatch.setenv(perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME, "0")
    assert perf_sanity._benchmark_client_timeout() is None


def test_client_timeout_falls_back_when_the_env_var_is_garbage(monkeypatch):
    """A typo in CI config must not silently restore the unbounded behaviour."""
    monkeypatch.setenv(perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME, "not-a-number")
    assert perf_sanity._benchmark_client_timeout() == 3600


# ---------------------------------------------------------------------------
# Teardown escalation
# ---------------------------------------------------------------------------


def test_stop_process_reaps_a_cooperative_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    perf_sanity.stop_process(proc, "cooperative", grace=10)
    assert proc.poll() is not None, "process should be reaped after SIGTERM"


def test_stop_process_escalates_to_sigkill_when_sigterm_is_ignored():
    """A rank wedged in native code never runs the SIGTERM handler.

    The bare terminate(); wait() this replaces would block teardown forever.
    """
    ignores_sigterm = (
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", ignores_sigterm])
    # Let the child install its handler before we signal it.
    time.sleep(1.0)

    started = time.monotonic()
    perf_sanity.stop_process(proc, "stubborn", grace=3)
    elapsed = time.monotonic() - started

    assert proc.poll() is not None, "SIGKILL should have reaped the process"
    assert elapsed < 30, f"stop_process took {elapsed:.1f}s; it must not wait out the full sleep"


def test_stop_process_is_a_noop_for_an_already_dead_process():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    perf_sanity.stop_process(proc, "already-dead", grace=5)
    assert proc.poll() is not None


# ---------------------------------------------------------------------------
# The client runner itself: the stage must be able to fail on its own.
# ---------------------------------------------------------------------------


def test_client_output_is_returned_on_success():
    out = perf_sanity.run_benchmark_client(
        [sys.executable, "-c", "print('benchmark done')"], dict(os.environ), []
    )
    assert "benchmark done" in out


def test_client_nonzero_exit_still_raises():
    """check_output semantics must survive: a failing client fails the test."""
    with pytest.raises(subprocess.CalledProcessError):
        perf_sanity.run_benchmark_client(
            [sys.executable, "-c", "import sys; sys.exit(3)"], dict(os.environ), []
        )


def test_client_hang_is_bounded_and_names_the_knob(monkeypatch, tmp_path):
    """The regression that mattered: a client that never returns.

    Before, this ran until Slurm killed the stage hours later. It must now
    raise promptly, name the env var, and carry the server-side errors.
    """
    monkeypatch.setenv(perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME, "2")
    server_log = tmp_path / "trtllm-serve.CTX_0.0.log"
    server_log.write_text("some line\n[TRT-LLM] [E] Error in event loop: boom\n")

    started = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        perf_sanity.run_benchmark_client(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            dict(os.environ),
            [str(server_log)],
        )
    elapsed = time.monotonic() - started

    msg = str(excinfo.value)
    assert perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME in msg
    assert "made no progress" in msg
    assert elapsed < 60, f"bound did not fire promptly ({elapsed:.1f}s)"


def test_client_hang_surfaces_server_side_errors(monkeypatch, tmp_path):
    monkeypatch.setenv(perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME, "2")
    server_log = tmp_path / "trtllm-serve.GEN_0.0.log"
    server_log.write_text("[TRT-LLM] [E] Error in event loop: kaboom\n")

    with pytest.raises(RuntimeError) as excinfo:
        perf_sanity.run_benchmark_client(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            dict(os.environ),
            [str(server_log)],
        )
    msg = str(excinfo.value)
    # A "[TRT-LLM] [E]" line matches no ERROR_KEYWORDS entry, so the keyword
    # scan finds nothing -- it must still reach the reader via the log tail.
    assert "kaboom" in msg, (
        "the server-side error is the whole point of failing here rather than "
        "letting the harness time out blind"
    )
    assert "tail of" in msg


def test_client_hang_surfaces_keyword_errors_too(monkeypatch, tmp_path):
    """The keyword scan still contributes when the log does match."""
    monkeypatch.setenv(perf_sanity.BENCHMARK_CLIENT_TIMEOUT_ENV_VAR_NAME, "2")
    server_log = tmp_path / "trtllm-serve.CTX_0.0.log"
    server_log.write_text("RuntimeError: engine exploded\n")

    with pytest.raises(RuntimeError) as excinfo:
        perf_sanity.run_benchmark_client(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            dict(os.environ),
            [str(server_log)],
        )
    msg = str(excinfo.value)
    assert "engine exploded" in msg
    assert "trtllm-serve.CTX_0.0.log:1:" in msg
