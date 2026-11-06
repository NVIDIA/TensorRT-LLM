# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that disaggregated benchmark output preserves the idle watchdog."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_DRAFT = (
    Path(__file__).resolve().parents[3] / "jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh"
)
_LOG_NAME = "trtllm-benchmark.0.log"


def _run_heartbeat(tmp_path: Path, tick_actions: dict[int, str] | None = None) -> list[str]:
    source = _DRAFT.read_text()
    block = source.split('export pytestCommand="$pytestCommandBenchmark"', 1)[1]
    block = block.split("heartbeatPid=$!", 1)[0]
    body = block.split("(\n", 1)[1].rsplit(") &", 1)[0]
    actions = "\n".join(f"{tick}) {action} ;;" for tick, action in (tick_actions or {}).items())
    script = f"""
set -Eeuo pipefail
trap 'echo unexpected-error >&2; exit 99' ERR
testOutputDir={shlex.quote(str(tmp_path))}
heartbeat_tick=0
sleep() {{
    [ "$1" = 600 ] || return 99
    heartbeat_tick=$((heartbeat_tick + 1))
    [ "$heartbeat_tick" -le 4 ] || return 1
    case "$heartbeat_tick" in
        {actions}
    esac
    return 0
}}
{body}
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_stalled_benchmark_stops_emitting_heartbeats(tmp_path):
    (tmp_path / _LOG_NAME).write_text("Benchmarking: 1/4\n")
    lines = _run_heartbeat(tmp_path)
    assert len(lines) == 1
    assert "Benchmarking: 1/4" in lines[0]


def test_progress_is_forwarded_after_each_append(tmp_path):
    (tmp_path / _LOG_NAME).write_text("Benchmarking: 1/4\n")
    lines = _run_heartbeat(
        tmp_path,
        {
            2: 'printf "Benchmarking: 2/4\\n" >> "$testOutputDir/trtllm-benchmark.0.log"',
            4: 'printf "Benchmarking: 3/4\\n" >> "$testOutputDir/trtllm-benchmark.0.log"',
        },
    )
    assert len(lines) == 3
    for progress, line in zip(("1/4", "2/4", "3/4"), lines):
        assert f"Benchmarking: {progress}" in line


def test_append_with_unchanged_tail_is_forwarded(tmp_path):
    (tmp_path / _LOG_NAME).write_text("x" * 200)
    lines = _run_heartbeat(
        tmp_path,
        {2: 'printf x >> "$testOutputDir/trtllm-benchmark.0.log"'},
    )
    assert len(lines) == 2
    assert all(line.endswith("x" * 200) for line in lines)


@pytest.mark.parametrize("create_empty_log", [False, True])
def test_missing_or_empty_log_does_not_emit_heartbeats(tmp_path, create_empty_log):
    if create_empty_log:
        (tmp_path / _LOG_NAME).touch()
    assert _run_heartbeat(tmp_path) == []


def test_log_created_after_monitor_starts_is_forwarded(tmp_path):
    lines = _run_heartbeat(
        tmp_path,
        {2: 'printf "Benchmarking: 1/4\\n" > "$testOutputDir/trtllm-benchmark.0.log"'},
    )
    assert len(lines) == 1
    assert "Benchmarking: 1/4" in lines[0]


def test_each_benchmark_log_is_tracked_independently(tmp_path):
    (tmp_path / _LOG_NAME).write_text("Benchmarking: 1/4\n")
    (tmp_path / "trtllm-benchmark.other.log").write_text("Benchmarking: 1/8\n")
    lines = _run_heartbeat(
        tmp_path,
        {2: 'printf "Benchmarking: 2/4\\n" >> "$testOutputDir/trtllm-benchmark.0.log"'},
    )
    assert len(lines) == 3
    assert any("trtllm-benchmark.other.log" in line and "1/8" in line for line in lines)
    assert any(_LOG_NAME in line and "2/4" in line for line in lines)


def test_touching_stalled_log_is_not_progress(tmp_path):
    (tmp_path / _LOG_NAME).write_text("Benchmarking: 1/4\n")
    lines = _run_heartbeat(tmp_path, {2: 'touch "$testOutputDir/trtllm-benchmark.0.log"'})
    assert len(lines) == 1


def test_rotated_log_with_same_size_is_forwarded(tmp_path):
    (tmp_path / _LOG_NAME).write_text("Benchmarking: 1/4\n")
    lines = _run_heartbeat(
        tmp_path,
        {
            2: 'mv "$testOutputDir/trtllm-benchmark.0.log" "$testOutputDir/previous.log"; '
            'printf "Benchmarking: 1/4\\n" > "$testOutputDir/trtllm-benchmark.0.log"'
        },
    )
    assert len(lines) == 2
    assert "Benchmarking: 1/4" in lines[-1]


def test_carriage_return_progress_uses_latest_line(tmp_path):
    (tmp_path / _LOG_NAME).write_bytes(b"Benchmarking: 1/4\rBenchmarking: 2/4\r")
    lines = _run_heartbeat(tmp_path)
    assert len(lines) == 1
    assert "Benchmarking: 2/4" in lines[0]


def test_same_size_log_rewrite_is_forwarded(tmp_path):
    (tmp_path / _LOG_NAME).write_text("Benchmarking: 1/4\n")
    lines = _run_heartbeat(
        tmp_path,
        {2: 'printf "Benchmarking: 2/4\\n" > "$testOutputDir/trtllm-benchmark.0.log"'},
    )
    assert len(lines) == 2
    assert "Benchmarking: 2/4" in lines[-1]


@pytest.mark.skipif(os.geteuid() == 0, reason="Root bypasses file permission checks")
def test_unreadable_log_does_not_emit_heartbeats(tmp_path):
    path = tmp_path / _LOG_NAME
    path.write_text("Benchmarking: 1/4\n")
    path.chmod(0)
    try:
        assert _run_heartbeat(tmp_path) == []
    finally:
        path.chmod(0o600)
