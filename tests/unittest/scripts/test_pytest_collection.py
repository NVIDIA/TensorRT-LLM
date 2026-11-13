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

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(os.name != "posix", reason="POSIX CI runner"),
]
REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "jenkins/scripts/collect_pytest.py"


def _prepare(tmp_path: Path, body: str) -> tuple[list[str], dict[str, str]]:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_ok.py").write_text(
        "def test_ok(): raise AssertionError('must only collect')\n"
    )
    (tmp_path / "conftest.py").write_text(
        "from pathlib import Path\nimport os, time\n"
        "counter = Path('attempts')\n"
        "attempt = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(attempt))\n" + body
    )
    command = [
        sys.executable,
        str(RUNNER),
        f"--log-dir={tmp_path / 'collection logs'}",
        "--timeout-seconds=2",
        "--retries=2",
        "--",
        "--collect-only",
        "-q",
    ]
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    return command, env


def _run(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    command, env = _prepare(tmp_path, body)
    return subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45
    )


@pytest.mark.parametrize("timeouts", [0, 1, 2, 3])
def test_only_timeouts_retry_and_only_success_publishes_output(
    tmp_path: Path, timeouts: int
) -> None:
    logs = tmp_path / "collection logs"
    logs.mkdir()
    (logs / "output.txt").write_text("STALE_SUCCESS")
    result = _run(tmp_path, f"if attempt <= {timeouts}: time.sleep(60)\n")
    assert (tmp_path / "attempts").read_text() == str(min(timeouts + 1, 3))
    assert result.returncode == (124 if timeouts == 3 else 0), result.stderr
    assert len(list(logs.glob("attempt-*.stdout.log"))) == min(timeouts + 1, 3)
    if timeouts == 3:
        assert not (logs / "output.txt").exists()
    else:
        output = (logs / "output.txt").read_text()
        assert "test_ok.py::test_ok" in output
        assert "STALE_SUCCESS" not in output
    for attempt in range(1, timeouts + 1):
        assert "conftest.py" in (logs / f"attempt-{attempt}.stderr.log").read_text()


@pytest.mark.parametrize(
    "failure,code", [("raise RuntimeError('collection failed')", 4), ("os._exit(124)", 124)]
)
def test_normal_failure_is_not_retried(tmp_path: Path, failure: str, code: int) -> None:
    # A child exit code equal to our timeout exit code is still not a timeout.
    result = _run(tmp_path, failure + "\n")
    assert result.returncode == code
    assert (tmp_path / "attempts").read_text() == "1"
    assert not (tmp_path / "collection logs/output.txt").exists()


def test_deadline_covers_session_finish(tmp_path: Path) -> None:
    result = _run(tmp_path, "def pytest_sessionfinish(session, exitstatus): time.sleep(60)\n")
    assert result.returncode == 124
    assert (tmp_path / "attempts").read_text() == "3"
    trace = (tmp_path / "collection logs/attempt-1.stderr.log").read_text()
    assert "pytest_sessionfinish" in trace


@pytest.mark.parametrize("exit_on_trace", [False, True])
def test_detached_child_is_dead_before_retry(tmp_path: Path, exit_on_trace: bool) -> None:
    result = _run(
        tmp_path,
        f"EXIT_ON_TRACE = {exit_on_trace}\n"
        + """import subprocess, sys, psutil, signal
if attempt == 1:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    Path('child.pid').write_text(str(child.pid))
    if EXIT_ON_TRACE:
        signal.signal(signal.SIGUSR1, lambda *_: sys.exit(0))
    time.sleep(60)
else:
    pid = int(Path('child.pid').read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "attempts").read_text() == "2"


def test_cancellation_cleans_child_without_retry(tmp_path: Path) -> None:
    command, env = _prepare(
        tmp_path,
        """import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
Path('child.pid').write_text(str(child.pid))
time.sleep(60)
""",
    )
    process = subprocess.Popen(
        command, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "child.pid").exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, stdout + stderr
        assert (tmp_path / "attempts").read_text() == "1"
        pid = int((tmp_path / "child.pid").read_text())
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


@pytest.mark.parametrize("with_output_dir", [False, True])
def test_invalid_selector_preserves_original_import_error(
    tmp_path: Path, with_output_dir: bool
) -> None:
    # Exercise the actual parser without importing the GPU integration conftest.
    package = tmp_path / "defs"
    package.mkdir()
    (package / "__init__.py").touch()
    source = REPO / "tests/integration/defs"
    for name in ("test_list_parser.py", "trt_test_alternative.py"):
        (package / name).write_bytes((source / name).read_bytes())
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_ok.py").write_text("def test_ok(): pass\n")
    (tmp_path / "test_broken.py").write_text("raise RuntimeError('ORIGINAL_IMPORT_FAILURE')\n")
    (tmp_path / "selection.txt").write_text("test_ok.py::test_ok\ntest_missing.py::test_missing\n")
    (tmp_path / "conftest.py").write_text("""from defs.test_list_parser import modify_by_test_list
def pytest_addoption(parser):
    parser.addoption('--test-list')
    parser.addoption('--test-prefix', default=None)
    parser.addoption('--output-dir', default=None)
    parser.addoption('--apply-test-list-correction', action='store_true')
def pytest_collection_modifyitems(config, items):
    modify_by_test_list(config.getoption('--test-list'), items, config)
""")
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--test-list=selection.txt"]
    if with_output_dir:
        command.append("--output-dir=reports")
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(tmp_path),
            "PYTEST_ADDOPTS": "",
        },
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ORIGINAL_IMPORT_FAILURE" in result.stdout
    assert "NoneType" not in result.stdout
    assert (tmp_path / "reports/invalid_tests.json").exists() == with_output_dir
