# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the real probe against a subprocess HTTP boundary, without GPU imports."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "snapshot_probe", _ROOT / "scripts/snapshot_probe.py"
)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)
_SERVER = Path(__file__).parent / "fixtures/snapshot_probe_server.py"
_PROFILE = {"model_revision": "test", "topology": {"tp": 2, "nodes": 1}}
_REQUESTS = [
    {
        "model": "test",
        "prompt": "The capital of Germany is",
        "max_tokens": 2,
        "temperature": 0,
        "stream": False,
    }
]


def _command(behavior: str = "ok") -> list[str]:
    """Build argv for the external HTTP boundary double.

    Args:
        behavior: Response or startup condition to simulate.

    Returns:
        Foreground argv understood by the real probe.
    """
    return [sys.executable, str(_SERVER), "{address}", behavior]


def test_cold_and_restore_compare_real_http(tmp_path: Path) -> None:
    """A matching generation passes without claiming any Snapshot gate passed."""
    cold = probe.run_probe(_command(), tmp_path / "cold", _PROFILE, _REQUESTS, 10, "cold")
    restored = probe.run_probe(
        _command(), tmp_path / "restored", _PROFILE, _REQUESTS, 10, "restore", cold
    )
    for name, report in (("cold", cold), ("restored", restored)):
        assert report["generation_probe_status"] == "PASS", report
        assert report["snapshot_qualification"] == "UNTESTED"
        assert set(report["gates"].values()) == {"UNTESTED"}
        assert report["outputs"] == [
            {"text": "Berlin", "finish_reason": "length", "completion_tokens": 2}
        ]
        timings = report["timings_seconds"]
        assert 0 < timings["http_health"] <= timings["first_completion"] <= timings["probe"]
        assert json.loads((tmp_path / name / "report.json").read_text()) == report
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o700
        assert report["cleanup_status"] == "PASS"
        with pytest.raises(ProcessLookupError):
            os.kill(report["candidate_pid"], 0)
    assert cold["command"][2] != restored["command"][2]


@pytest.mark.parametrize(
    "behavior,expected",
    [
        ("exit", "supervisor exited"),
        ("no-address", "TimeoutError"),
        ("unhealthy", "TimeoutError"),
        ("hang", "TimeoutError"),
        ("http-error", "HTTP 500"),
        ("malformed", "JSONDecodeError"),
        ("empty", "must contain text"),
        ("dribble", "TimeoutError"),
    ],
)
def test_failed_attempt_is_recorded(tmp_path: Path, behavior: str, expected: str) -> None:
    """Failures produce retained reports and never become successful baselines."""
    started = time.monotonic()
    report = probe.run_probe(
        _command(behavior), tmp_path / "attempt", _PROFILE, _REQUESTS, 1, "cold"
    )
    assert time.monotonic() - started < 5
    assert report["generation_probe_status"] == "FAIL"
    assert expected in report["error"]
    assert report["snapshot_qualification"] == "UNTESTED"
    assert json.loads((tmp_path / "attempt/report.json").read_text()) == report
    with pytest.raises(ValueError, match="successful cold baseline"):
        probe.run_probe(_command(), tmp_path / "unused", _PROFILE, _REQUESTS, 10, "restore", report)
    assert not (tmp_path / "unused").exists()


def test_mismatch_is_failure_not_startup_success(tmp_path: Path) -> None:
    """HTTP health and nonempty output cannot mask incorrect restored output."""
    cold = probe.run_probe(_command(), tmp_path / "cold", _PROFILE, _REQUESTS, 10, "cold")
    report = probe.run_probe(
        _command("mismatch"), tmp_path / "restore", _PROFILE, _REQUESTS, 10, "restore", cold
    )
    assert report["generation_probe_status"] == "FAIL"
    assert "do not exactly match" in report["error"]
    assert "first_completion" in report["timings_seconds"]


@pytest.mark.parametrize("changed", ["profile", "request", "mode", "schema", "cleanup"])
def test_incompatible_baseline_rejected_before_launch(tmp_path: Path, changed: str) -> None:
    """Compatibility checks run before candidate creation, including for topology."""
    cold = probe.run_probe(_command(), tmp_path / "cold", _PROFILE, _REQUESTS, 10, "cold")
    profile, requests = dict(_PROFILE), list(_REQUESTS)
    if changed == "profile":
        profile["topology"] = {"tp": 4, "nodes": 2}
    elif changed == "request":
        requests = [{**_REQUESTS[0], "prompt": "Other prompt"}]
    elif changed == "mode":
        cold["mode"] = "restore"
    elif changed == "cleanup":
        cold["cleanup_status"] = "FAIL"
    else:
        cold["schema_version"] = 100
    with pytest.raises(ValueError, match="matching profile and requests"):
        probe.run_probe(_command(), tmp_path / "unused", profile, requests, 10, "restore", cold)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "address",
    [
        "example.com:8000",
        "0.0.0.0:8000",
        "127.0.0.1:0",
        "127.0.0.1:8000/path",
        "user@127.0.0.1:8000",
    ],
)
def test_rejects_nonisolated_address(tmp_path: Path, address: str) -> None:
    """The runner cannot accidentally send prompts to a remote service."""
    path = tmp_path / "address"
    path.write_text(address)
    with pytest.raises(ValueError):
        probe._read_address(path)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected(tmp_path: Path, timeout: float) -> None:
    """Invalid budgets fail before creating an attempt directory."""
    with pytest.raises(ValueError, match="positive finite"):
        probe.run_probe(_command(), tmp_path / "unused", _PROFILE, _REQUESTS, timeout, "cold")


def test_existing_directory_is_not_reused(tmp_path: Path) -> None:
    """A stale report or address cannot be mistaken for this attempt's output."""
    sentinel = tmp_path / "address"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        probe.run_probe(_command(), tmp_path, _PROFILE, _REQUESTS, 10, "cold")
    assert sentinel.read_text() == "unchanged"


def test_launch_failure_has_report(tmp_path: Path) -> None:
    """A missing launcher is a failed attempt, not a missing observation."""
    report = probe.run_probe(
        [str(tmp_path / "nonexistent"), "{address}"],
        tmp_path / "attempt",
        _PROFILE,
        _REQUESTS,
        10,
        "cold",
    )
    assert report["generation_probe_status"] == "FAIL"
    assert "FileNotFoundError" in report["error"]


@pytest.mark.parametrize(
    "requests",
    [
        1,
        {},
        [],
        [None],
        [{**_REQUESTS[0], "stream": True}],
        [{**_REQUESTS[0], "max_tokens": True}],
        [{**_REQUESTS[0], "temperature": False}],
        [{**_REQUESTS[0], "n": True}],
    ],
)
def test_invalid_requests_rejected(tmp_path: Path, requests: object) -> None:
    """Malformed request files fail before launching a process."""
    with pytest.raises(ValueError):
        probe.run_probe(_command(), tmp_path / "unused", _PROFILE, requests, 10, "cold")
    assert not (tmp_path / "unused").exists()


def test_cli_writes_comparison_reports(tmp_path: Path) -> None:
    """Exercise CLI parsing, files and subprocess transport, not a copied runner."""
    profile = tmp_path / "profile.json"
    requests = tmp_path / "requests.json"
    profile.write_text(json.dumps(_PROFILE))
    requests.write_text(json.dumps(_REQUESTS))
    for mode in ("cold", "restore"):
        command = [
            sys.executable,
            str(_ROOT / "scripts/snapshot_probe.py"),
            "probe",
            "--mode",
            mode,
            "--profile",
            str(profile),
            "--requests",
            str(requests),
            "--run-dir",
            str(tmp_path / mode),
            "--timeout",
            "10",
        ]
        if mode == "restore":
            command += ["--baseline", str(tmp_path / "cold/report.json")]
        result = subprocess.run(
            command + ["--", *_command()], capture_output=True, text=True, timeout=20
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Snapshot qualification: UNTESTED" in result.stdout
        report = json.loads((tmp_path / mode / "report.json").read_text())
        assert report["generation_probe_status"] == "PASS"
        assert report["snapshot_qualification"] == "UNTESTED"


def test_summary_retains_failures_and_never_qualifies_restore(tmp_path: Path) -> None:
    """Summaries count failed attempts, and do not certify labeled restores."""
    cold = probe.run_probe(_command(), tmp_path / "cold", _PROFILE, _REQUESTS, 10, "cold")
    restored = probe.run_probe(
        _command(), tmp_path / "restored", _PROFILE, _REQUESTS, 10, "restore", cold
    )
    failed = probe.run_probe(
        _command("exit"), tmp_path / "failed", _PROFILE, _REQUESTS, 10, "restore", cold
    )
    summary = probe.summarize_reports([cold, restored, failed])
    assert summary["snapshot_qualification"] == "UNTESTED"
    assert summary["modes"]["restore"]["attempted"] == 2
    assert summary["modes"]["restore"]["passed"] == 1
    assert summary["modes"]["restore"]["failed"] == 1
    timing = summary["modes"]["restore"]["launch_to_first_completion_seconds"]
    assert timing["samples"] == 1
    assert timing["p50"] == timing["p95"] == restored["timings_seconds"]["first_completion"]


@pytest.mark.parametrize("change", ["profile", "timing", "output", "cleanup", "qualification"])
def test_summary_rejects_invalid_or_mixed_evidence(tmp_path: Path, change: str) -> None:
    """Incompatible or corrupt observations cannot enter a latency summary."""
    import copy

    cold = probe.run_probe(_command(), tmp_path / "cold", _PROFILE, _REQUESTS, 10, "cold")
    changed = copy.deepcopy(cold)
    changed["attempt_id"] = "another-attempt"
    if change == "profile":
        changed["profile"]["topology"]["tp"] = 4
    elif change == "timing":
        changed["timings_seconds"]["first_completion"] = float("nan")
    elif change == "output":
        changed["outputs"][0]["text"] = "Other"
    elif change == "cleanup":
        changed["cleanup_status"] = "FAIL"
    else:
        changed["snapshot_qualification"] = "PASS"
    with pytest.raises(ValueError):
        probe.summarize_reports([cold, changed])


def test_summary_rejects_duplicate_attempts(tmp_path: Path) -> None:
    """Copies of an attempt cannot inflate the sample or success count."""
    report = probe.run_probe(_command(), tmp_path / "cold", _PROFILE, _REQUESTS, 10, "cold")
    with pytest.raises(ValueError, match="Duplicate attempt"):
        probe.summarize_reports([report, dict(report)])


def test_summary_cli(tmp_path: Path) -> None:
    """The summary CLI writes a new report and rejects overwriting it."""
    probe.run_probe(_command(), tmp_path / "cold", _PROFILE, _REQUESTS, 10, "cold")
    output = tmp_path / "summary.json"
    command = [
        sys.executable,
        str(_ROOT / "scripts/snapshot_probe.py"),
        "summarize",
        "--output",
        str(output),
        str(tmp_path / "cold/report.json"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["modes"]["cold"]["attempted"] == 1
    assert subprocess.run(command, capture_output=True, timeout=10).returncode != 0


def test_summary_percentiles_and_empty_group(tmp_path: Path) -> None:
    """Interpolate multiple samples and leave an unobserved mode unmeasured."""
    reports = [
        probe.run_probe(_command(), tmp_path / str(index), _PROFILE, _REQUESTS, 10, "cold")
        for index in range(3)
    ]
    # Controlled timing observations exercise summary math, not performance.
    for report, elapsed in zip(reports, (1.0, 2.0, 3.0)):
        report["timings_seconds"]["first_completion"] = elapsed
    summary = probe.summarize_reports(reports)
    assert summary["modes"]["cold"]["launch_to_first_completion_seconds"] == {
        "samples": 3,
        "p50": 2.0,
        "p95": pytest.approx(2.9),
    }
    assert summary["modes"]["restore"]["launch_to_first_completion_seconds"] == {
        "samples": 0,
        "p50": None,
        "p95": None,
    }


def test_summary_all_failed_attempts(tmp_path: Path) -> None:
    """A failed campaign retains its denominator without producing latency."""
    report = probe.run_probe(_command("exit"), tmp_path / "failed", _PROFILE, _REQUESTS, 10, "cold")
    summary = probe.summarize_reports([report])
    assert summary["modes"]["cold"] == {
        "attempted": 1,
        "passed": 0,
        "failed": 1,
        "launch_to_first_completion_seconds": {"samples": 0, "p50": None, "p95": None},
    }
