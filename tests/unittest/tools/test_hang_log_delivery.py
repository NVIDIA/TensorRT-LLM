# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure reports must survive captured streams and abrupt worker termination."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from test_common.hang_diagnostics import (
    _MAX_REPORT_BYTES,
    collect_hang_diagnostics,
    hang_diagnostics_for_test,
    hang_diagnostics_root,
    prepare_hang_diagnostics,
    record_hang_diagnostic_test,
)
from test_common.s3_output import UploadLogPlugin

pytestmark = pytest.mark.cpu_only

_WORKER = """
import os
import signal
import sys
import threading
from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector
from tensorrt_llm._torch.pyexecutor.hang_diagnostics import ExecutorHangDiagnostics

def terminate():
    mode = sys.argv[1]
    if mode == 'exit':
        os._exit(7)
    os.kill(os.getpid(), getattr(signal, mode))

def blocked_worker():
    diagnostics = ExecutorHangDiagnostics(rank=2)
    diagnostics.record('sampling', iteration=17, details='active_request_ids=[42]')
    detector = HangDetector(timeout=0.1, on_detected=terminate)
    detector.register_status_provider(diagnostics.get_snapshot_dump)
    if sys.argv[1] == 'blocked-provider':
        def blocked_provider():
            threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGKILL)).start()
            threading.Event().wait(30)
        detector.register_status_provider(blocked_provider)
    else:
        detector.register_status_provider(diagnostics.get_status_dump)
    detector.start()
    detector.checkpoint()
    threading.Event().wait(30)
    raise RuntimeError('watchdog failed to terminate the worker')

# Model a capture destination that will never be reported to the runner.
sys.stdout.flush()
sys.stderr.flush()
with open(os.devnull, 'w') as sink:
    os.dup2(sink.fileno(), 1)
    os.dup2(sink.fileno(), 2)
blocked_worker()
"""


@pytest.mark.skipif(sys.platform != "linux", reason="tests Unix termination signals")
@pytest.mark.parametrize("termination", ["exit", "SIGTERM", "SIGKILL", "blocked-provider"])
def test_worker_report_survives_termination_in_retained_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], termination: str
) -> None:
    directory = prepare_hang_diagnostics(str(tmp_path))
    assert directory is not None
    record_hang_diagnostic_test(directory, "test_wrapper.py::test_seed")
    record_hang_diagnostic_test(directory, "test_wrapper.py::test_hung_worker")
    result = subprocess.run(
        [sys.executable, "-c", _WORKER, termination],
        env={**os.environ, "TLLM_HANG_DIAGNOSTICS_DIR": directory},
        capture_output=True,
        text=True,
        timeout=180,
    )
    expected_exit = {
        "exit": 7,
        "SIGTERM": -signal.SIGTERM,
        "SIGKILL": -signal.SIGKILL,
        "blocked-provider": -signal.SIGKILL,
    }[termination]
    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert "phase=sampling" not in result.stdout + result.stderr

    collect_hang_diagnostics(directory)
    collected = capsys.readouterr().out
    plugin = UploadLogPlugin(
        endpoint_url="https://unused.invalid",
        aws_access_key_id="unused",
        aws_secret_access_key=None,
        bucket="unused",
        upload_path="hang-test",
        output_path=str(tmp_path / "outer"),
        skip_upload=True,
        inline_output_max_bytes=0,
    )
    report = pytest.TestReport(
        nodeid="test_wrapper.py::test_hung_worker",
        location=("test_wrapper.py", 0, "test_hung_worker"),
        keywords={},
        outcome="failed",
        longrepr="inner worker terminated",
        when="call",
        sections=[("Captured stdout call", collected)],
    )
    manager = pytest.PytestPluginManager()
    manager.register(plugin, "log_output")
    manager.hook.pytest_runtest_logreport(report=report)
    manager.hook.pytest_runtest_logfinish(nodeid=report.nodeid, location=report.location)
    artifacts = list((tmp_path / ".s3-spool-outer").rglob("stdout.log"))
    assert len(artifacts) == 1
    content = artifacts[0].read_text()
    assert "collected hang diagnostics" in content
    assert "pid=" in content
    assert "test=test_wrapper.py::test_hung_worker" in content
    assert "test_seed" not in content
    assert "rank=2" in content
    assert "phase=sampling" in content
    assert "iteration=17" in content
    assert "active_request_ids=[42]" in content
    if termination != "blocked-provider":
        assert "forward_cuda_completion=not-recorded" in content
    assert "in blocked_worker" in content


def test_healthy_invocation_is_silent_and_retry_directory_is_separate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = hang_diagnostics_root(str(tmp_path))
    second = hang_diagnostics_root(str(tmp_path))
    assert first != second
    assert not Path(first).exists() and not Path(second).exists()
    prepare_hang_diagnostics(first)
    Path(first, "pytest-123.txt").write_text("")
    collect_hang_diagnostics(first)
    collect_hang_diagnostics(second)
    assert not Path(first).exists() and not Path(second).exists()
    assert capsys.readouterr().out == ""


def test_oversized_report_is_elided_around_its_ends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = prepare_hang_diagnostics(str(tmp_path))
    assert directory is not None
    body = "".join(f"line {index}\n" for index in range(200000))
    Path(directory, "worker-1-2.txt").write_text(f"HEAD-MARKER\n{body}TAIL-MARKER\n")
    collect_hang_diagnostics(directory)
    collected = capsys.readouterr().out
    assert "HEAD-MARKER" in collected
    assert "TAIL-MARKER" in collected
    assert "bytes of worker-1-2.txt" in collected
    assert len(collected) < 2 * _MAX_REPORT_BYTES


def test_unavailable_diagnostic_directory_does_not_mask_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    assert prepare_hang_diagnostics(str(blocker)) is None
    collect_hang_diagnostics(None)
    assert "Could not prepare hang diagnostics" in capsys.readouterr().out


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("worker", [None, "gw0"])
def test_diagnostic_marker_scopes_environment_and_creates_directories_lazily(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    enabled: bool,
    worker: str | None,
) -> None:
    root = hang_diagnostics_root(str(tmp_path))
    monkeypatch.setattr(request.config.option, "hang_diagnostics_root", root)
    monkeypatch.setenv("TLLM_HANG_DIAGNOSTICS", "0")
    monkeypatch.delenv("TLLM_HANG_DIAGNOSTICS_DIR", raising=False)
    if worker is None:
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    else:
        monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
    # Keep this synthetic item's context separate from the real runner's watchdog.
    get_plugin = request.config.pluginmanager.get_plugin
    monkeypatch.setattr(
        request.config.pluginmanager,
        "get_plugin",
        lambda name: None if name == "periodic_junit" else get_plugin(name),
    )
    item = pytest.Function.from_parent(
        request.node.parent, name="test_opt_in", callobj=lambda: None
    )
    if enabled:
        item.add_marker(pytest.mark.hang_diagnostics)
    with hang_diagnostics_for_test(item) as directory:
        assert os.environ["TLLM_HANG_DIAGNOSTICS"] == ("1" if enabled else "0")
        if enabled:
            expected = os.path.join(root, worker) if worker else root
            assert directory == expected
            assert os.environ["TLLM_HANG_DIAGNOSTICS_DIR"] == expected
            assert Path(expected, "current-test").read_text() == item.nodeid
        else:
            assert directory is None
            assert "TLLM_HANG_DIAGNOSTICS_DIR" not in os.environ
            assert not Path(root).exists()
    assert os.environ["TLLM_HANG_DIAGNOSTICS"] == "0"
    assert "TLLM_HANG_DIAGNOSTICS_DIR" not in os.environ
    collect_hang_diagnostics(root)
    assert not Path(root).exists()
    assert capsys.readouterr().out == ""


def test_marked_test_restores_environment_after_error(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request.config.option, "hang_diagnostics_root", None)
    monkeypatch.delenv("TLLM_HANG_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("TLLM_HANG_DIAGNOSTICS_DIR", raising=False)
    item = pytest.Function.from_parent(
        request.node.parent, name="test_opt_in_error", callobj=lambda: None
    )
    item.add_marker(pytest.mark.hang_diagnostics)
    with pytest.raises(RuntimeError, match="test failed"):
        with hang_diagnostics_for_test(item):
            assert os.environ["TLLM_HANG_DIAGNOSTICS"] == "1"
            raise RuntimeError("test failed")
    assert "TLLM_HANG_DIAGNOSTICS" not in os.environ
    assert "TLLM_HANG_DIAGNOSTICS_DIR" not in os.environ
