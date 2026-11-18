#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the wheel that tests/unittest/check_pip_install.py installs.

The script takes the wheel as a local file, which it uses where it is, or as a URL, which it
downloads. subprocess.check_call, subprocess.run, and requests.get are replaced and record what
they are asked, so nothing is installed or downloaded and no request leaves the test.
"""

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "tests" / "unittest" / "check_pip_install.py"
WHEEL_NAME = "tensorrt_llm-1.2.0-cp312-cp312-linux_x86_64.whl"
FOLDER_URL = "https://wheels.example/llm-artifacts/LLM/main/L0_PostMerge/88/x86_64"
WHEEL_URL = f"{FOLDER_URL}/{WHEEL_NAME}"
# A wheel folder's index page, one link per line: the wheel for cp312 is not the first wheel.
INDEX_PAGE = "\n".join(
    f'<a href="{name}">{name}</a>'
    for name in (
        "build_info.txt",
        "tensorrt_llm-1.2.0-cp310-cp310-linux_x86_64.whl",
        WHEEL_NAME,
    )
)
# The steps of the script that take the wheel.
ENTRY_POINTS = ("test_python_builds", "test_pip_install")


@pytest.fixture(scope="module")
def check_script() -> ModuleType:
    """The script as a module, loaded with the imports that `python3 check_pip_install.py` finds."""
    spec = importlib.util.spec_from_file_location("check_pip_install", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    # Where requests is not installed, the name only has to import: each test replaces
    # requests.get.
    stub_requests = importlib.util.find_spec("requests") is None
    if stub_requests:
        sys.modules["requests"] = ModuleType("requests")
    # A script's directory comes first on sys.path when it runs, and the script imports `utils`
    # from there.
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        if stub_requests:
            del sys.modules["requests"]
    return module


class _IndexResponse:
    """The answer to a GET of a wheel folder."""

    status_code = 200
    text = INDEX_PAGE


@pytest.fixture(autouse=True)
def requested_urls(check_script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The URLs the script asked requests.get for; each is answered with the index page."""
    urls: list[str] = []

    def get(url: str) -> _IndexResponse:
        urls.append(url)
        return _IndexResponse()

    monkeypatch.setattr(check_script.requests, "get", get, raising=False)
    return urls


class _Commands:
    """Stands in for subprocess.check_call and subprocess.run, and records each command."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | list[str], dict[str, object]]] = []

    def check_call(self, cmd: str | list[str], **kwargs: object) -> int:
        self.calls.append((cmd, kwargs))
        return 0

    def run(self, cmd: str | list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((cmd, kwargs))
        # No output: the script's query for an installed torch finds none, so it writes no
        # constraint file.
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    @property
    def lines(self) -> list[str]:
        """Each command as one line."""
        return [cmd if isinstance(cmd, str) else " ".join(cmd) for cmd, _ in self.calls]


@pytest.fixture
def commands(check_script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> _Commands:
    recorded = _Commands()
    monkeypatch.setattr(check_script.subprocess, "check_call", recorded.check_call)
    monkeypatch.setattr(check_script.subprocess, "run", recorded.run)
    # The steps that do not take the wheel: system packages, and checks of what was installed.
    for step in ("install_system_libs", "verify_openengine_distribution", "run_sanity_check"):
        monkeypatch.setattr(check_script, step, lambda *args, **kwargs: None)
    return recorded


@pytest.fixture
def local_wheel(tmp_path: Path) -> Path:
    """A wheel in a directory of its own, whose name needs quoting in a shell command."""
    wheel = tmp_path / "wheel read" / WHEEL_NAME
    wheel.parent.mkdir()
    wheel.write_bytes(b"wheel")
    return wheel


@pytest.fixture(autouse=True)
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory to run the script's steps in, apart from the wheel's."""
    directory = tmp_path / "working-directory"
    directory.mkdir()
    monkeypatch.chdir(directory)
    return directory


def _args(wheel_path: str) -> argparse.Namespace:
    return argparse.Namespace(wheel_path=wheel_path, version_local=None, cpython_version="cp312")


def _assert_nothing_fetches_or_removes_a_wheel(commands: _Commands) -> None:
    for line in commands.lines:
        assert "wget" not in line
        assert "http://" not in line
        assert "https://" not in line
        # Neither `rm *.whl` nor an install of whichever wheel the working directory holds.
        assert "*.whl" not in line


def test_get_wheel_url_returns_the_absolute_path_of_a_local_file(
    check_script: ModuleType, local_wheel: Path, requested_urls: list[str]
) -> None:
    assert check_script.get_wheel_url(str(local_wheel), None, "cp312") == str(local_wheel)
    assert requested_urls == []


def test_get_wheel_url_makes_a_relative_path_of_a_local_file_absolute(
    check_script: ModuleType,
    local_wheel: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_urls: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_path = str(local_wheel.relative_to(tmp_path))

    assert check_script.get_wheel_url(relative_path, None, "cp312") == str(local_wheel)
    assert requested_urls == []


@pytest.fixture(params=("absolute", "relative"))
def wheel_argument(
    request: pytest.FixtureRequest,
    local_wheel: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """The local wheel as ``--wheel_path`` names it: by its absolute path, or relative to the cwd."""
    if request.param == "absolute":
        return str(local_wheel)
    monkeypatch.chdir(tmp_path)
    return str(local_wheel.relative_to(tmp_path))


def test_pip_install_installs_a_local_file_by_its_absolute_path(
    check_script: ModuleType,
    commands: _Commands,
    local_wheel: Path,
    wheel_argument: str,
    requested_urls: list[str],
) -> None:
    check_script.test_pip_install(_args(wheel_argument))

    assert requested_urls == []
    _assert_nothing_fetches_or_removes_a_wheel(commands)
    [install] = [line for line in commands.lines if line.startswith("pip3 install")]
    # The path is one argument, though it holds a space.
    assert str(local_wheel) in shlex.split(install)
    assert local_wheel.read_bytes() == b"wheel"


def test_python_builds_passes_a_local_file_as_the_precompiled_location(
    check_script: ModuleType,
    commands: _Commands,
    local_wheel: Path,
    wheel_argument: str,
    requested_urls: list[str],
) -> None:
    check_script.test_python_builds(_args(wheel_argument))

    assert requested_urls == []
    _assert_nothing_fetches_or_removes_a_wheel(commands)
    # The editable install is the one command that gets an environment of its own.
    [(install, kwargs)] = [call for call in commands.calls if "env" in call[1]]
    assert install[:4] == ["pip3", "install", "-e", "."]
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["TRTLLM_PRECOMPILED_LOCATION"] == str(local_wheel)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
@pytest.mark.parametrize("is_directory", (False, True), ids=("missing", "directory"))
def test_an_absolute_path_that_is_not_a_file_exits_before_any_install(
    check_script: ModuleType,
    commands: _Commands,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    requested_urls: list[str],
    is_directory: bool,
    entry_point: str,
) -> None:
    not_a_file = tmp_path / "wheel" / WHEEL_NAME
    if is_directory:
        not_a_file.mkdir(parents=True)

    with pytest.raises(SystemExit) as exited:
        getattr(check_script, entry_point)(_args(str(not_a_file)))

    assert exited.value.code not in (0, None)
    captured = capfd.readouterr()
    assert str(not_a_file) in captured.out + captured.err
    assert commands.calls == []
    assert requested_urls == []


@pytest.mark.parametrize(
    ("wheel_path", "expected_requests"),
    (
        (WHEEL_URL, []),
        (WHEEL_URL.removeprefix("https://"), []),
        (f"{FOLDER_URL}/", [f"{FOLDER_URL}/"]),
    ),
    ids=("wheel-url", "wheel-url-without-scheme", "folder-url"),
)
def test_get_wheel_url_returns_the_url_of_a_wheel_given_by_url(
    check_script: ModuleType,
    requested_urls: list[str],
    wheel_path: str,
    expected_requests: list[str],
) -> None:
    assert check_script.get_wheel_url(wheel_path, None, "cp312") == WHEEL_URL
    assert requested_urls == expected_requests


@pytest.mark.parametrize("wheel_path", (WHEEL_URL, f"{FOLDER_URL}/"), ids=("wheel", "folder"))
def test_pip_install_downloads_a_wheel_given_by_url(
    check_script: ModuleType, commands: _Commands, wheel_path: str
) -> None:
    check_script.test_pip_install(_args(wheel_path))

    lines = commands.lines
    remove, download, install = (
        lines.index("rm *.whl || true"),
        lines.index(f"apt-get install -y wget && wget -q {WHEEL_URL}"),
        lines.index("pip3 install tensorrt_llm-*.whl"),
    )
    assert remove < download < install


def test_python_builds_passes_a_wheel_url_as_the_precompiled_location(
    check_script: ModuleType, commands: _Commands
) -> None:
    check_script.test_python_builds(_args(WHEEL_URL))

    [(_, kwargs)] = [call for call in commands.calls if "env" in call[1]]
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["TRTLLM_PRECOMPILED_LOCATION"] == WHEEL_URL
