# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Fail-fast + log-tail-dump behavior of wait_for_endpoint_ready (no GPU, no server)."""

import subprocess

import pytest
from test_common.error_utils import check_error, report_error
from test_common.http_utils import fail_if_proc_died, wait_for_endpoint_ready

# Nothing listens here; connections are refused immediately.
DEAD_URL = "http://127.0.0.1:9/health"


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """A corporate HTTP proxy answering 200 would make DEAD_URL look ready."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)


def _server_log(tmp_path, content):
    p = tmp_path / "server.log"
    p.write_text(content)
    return str(p)


def test_dead_server_proc_fails_fast_and_dumps_log(tmp_path):
    log = _server_log(tmp_path, "loading weights...\nlast server words\n")
    proc = subprocess.Popen(["true"])
    proc.wait()
    with pytest.raises(RuntimeError) as e:
        wait_for_endpoint_ready(DEAD_URL, timeout=30, check_files=[log], server_proc=proc)
    msg = str(e.value)
    assert "exited unexpectedly with code" in msg
    # The server-side story must land in the failure message (CI log).
    assert "last server words" in msg


def test_error_keyword_in_server_log_fails_fast(tmp_path):
    log = _server_log(tmp_path, "starting\nRuntimeError: engine exploded\n")
    with pytest.raises(RuntimeError) as e:
        wait_for_endpoint_ready(DEAD_URL, timeout=30, check_files=[log], check_interval=0.0)
    msg = str(e.value)
    assert "Found error in server file" in msg
    assert "engine exploded" in msg


def test_ready_timeout_dumps_log_tail(tmp_path):
    log = _server_log(tmp_path, "clean init so far, no error keywords\n")
    with pytest.raises(RuntimeError) as e:
        # check_interval > timeout so the error-scan never trips; we want the
        # timeout path itself.
        wait_for_endpoint_ready(DEAD_URL, timeout=2, check_files=[log], check_interval=60.0)
    msg = str(e.value)
    assert "did not become ready within" in msg
    assert "clean init so far" in msg


def test_missing_check_file_does_not_crash_the_wait(tmp_path):
    missing = str(tmp_path / "not-written-yet.log")
    with pytest.raises(RuntimeError) as e:
        wait_for_endpoint_ready(DEAD_URL, timeout=2, check_files=[missing], check_interval=0.0)
    # Reaches the timeout path (not a FileNotFoundError) and reports the
    # missing log in the dump.
    msg = str(e.value)
    assert "did not become ready within" in msg
    assert "Path doesn't exist" in msg


def test_fail_if_proc_died_raises_with_log_tail(tmp_path):
    """Event-driven babysitter check: dead child -> immediate raise + log tail."""
    log = _server_log(tmp_path, "gen server init...\nfinal gen words\n")
    proc = subprocess.Popen(["true"])
    proc.wait()
    with pytest.raises(RuntimeError) as e:
        fail_if_proc_died(proc, "GEN_0 server", [log])
    msg = str(e.value)
    assert "GEN_0 server exited unexpectedly with code 0" in msg
    assert "final gen words" in msg


def test_fail_if_proc_died_noop_when_alive_or_none(tmp_path):
    fail_if_proc_died(None, "no server")  # no-op
    proc = subprocess.Popen(["sleep", "5"])
    try:
        fail_if_proc_died(proc, "alive server")  # no raise
    finally:
        proc.kill()
        proc.wait()


def test_report_error_keyword_scan_fires(tmp_path):
    """Regression: report_error scanned an exhausted handle; keywords never matched."""
    log = _server_log(
        tmp_path,
        "line one\nRuntimeError: boom\nline three\n",
    )
    with pytest.raises(RuntimeError) as e:
        report_error("wrapper message", [log])
    msg = str(e.value)
    assert "wrapper message" in msg
    assert "Error line 2" in msg
    assert "RuntimeError: boom" in msg


def test_check_error_skips_benign_autotuner_lines(tmp_path):
    """Autotuner warmup probe-OOMs are expected on healthy startups."""
    log = _server_log(
        tmp_path,
        "[Autotuner] Single-pair run failed: CUDA out of memory. Tried to allocate...\n"
        "normal progress line\n",
    )
    assert check_error(log) == []
    # A real error outside the benign marker still trips the scan.
    log2 = _server_log(tmp_path, "RuntimeError: engine exploded\n")
    assert len(check_error(log2)) == 1


def test_autotuner_marker_does_not_hide_real_errors(tmp_path):
    """Only the marker+OOM combination is benign; other autotuner errors are real."""
    log = _server_log(
        tmp_path,
        "[Autotuner] Single-pair run failed: CUDA out of memory. Tried to allocate...\n"
        "[Autotuner] RuntimeError: invalid tactic configuration\n",
    )
    hits = check_error(log)
    assert [(idx, "RuntimeError" in line) for idx, line in hits] == [(2, True)]
    # report_error's scanner applies the same boundary.
    with pytest.raises(RuntimeError) as e:
        report_error("wrapper", [log])
    assert "Error line 2" in str(e.value)


def test_report_error_always_appends_tail_even_on_keyword_hit(tmp_path):
    """The first keyword hit may be noise; the fatal error can sit at EOF."""
    lines = ["ValueError: early benign-looking hit\n"]
    lines += [f"filler {i}\n" for i in range(300)]
    lines += ["the actual fatal last words\n"]
    log = _server_log(tmp_path, "".join(lines))
    with pytest.raises(RuntimeError) as e:
        report_error("wrapper", [log])
    msg = str(e.value)
    assert "Error line 1" in msg  # keyword context present
    assert "the actual fatal last words" in msg  # AND the tail
