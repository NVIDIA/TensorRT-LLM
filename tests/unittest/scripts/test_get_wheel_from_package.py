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
"""Tests for the packaged-build download in scripts/get_wheel_from_package.py.

The script polls jenkins/scripts/fetch_build_artifact.sh until a run's packaged build is
published. subprocess.run and time.sleep are replaced, so no helper, network, or wait is involved.
"""

import argparse
import importlib.util
import os
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "get_wheel_from_package.py"
HELPER_PATH = REPO_ROOT / "jenkins" / "scripts" / "fetch_build_artifact.sh"
# The legacy path the image build passes as --artifact_path; the helper maps it to each location.
ARTIFACT_PATH = "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/88"
TARBALL = "TensorRT-LLM.tar.gz"
NETRC = "/run/secrets/artifactory_netrc"
CREDENTIAL_ID = "aws-artifactory-credentials"
# Log phrases that make the CI shared library retry a step as a network fault.
RETRY_SIGNATURES = (
    "403 Forbidden",
    "URL returned error",
    "HTTP response code said error",
    "404 Not Found",
    "ERROR 404",
    "HTTP 50",
    "ERROR 50",
    "Server Error",
    "Error downloading",
    "Connection",
    "Read timed out",
    "Too Many Requests",
)


class _StopBeforeExtract(Exception):
    """Raised for the first command after the download, so a test ends before extraction."""


class _FakeRun:
    """Stands in for subprocess.run and answers each helper call with the next exit code."""

    def __init__(self, returncodes: list[int]) -> None:
        self._returncodes = list(returncodes)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(cmd), kwargs))
        if cmd[0] != "bash":
            raise _StopBeforeExtract(cmd)
        result: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(
            cmd, self._returncodes.pop(0)
        )
        if kwargs.get("check"):
            result.check_returncode()
        return result


@pytest.fixture(scope="module")
def wheel_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("get_wheel_from_package", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def read_helper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACTORY_NETRC", NETRC)
    monkeypatch.setenv("ARTIFACTORY_CREDENTIAL_ID", CREDENTIAL_ID)
    monkeypatch.delenv("LLM_ARTIFACTS_PRIMARY", raising=False)


@pytest.fixture(autouse=True)
def sleeps(wheel_script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(wheel_script.time, "sleep", recorded.append)
    return recorded


def _fake_run(
    wheel_script: ModuleType, monkeypatch: pytest.MonkeyPatch, returncodes: list[int]
) -> _FakeRun:
    fake = _FakeRun(returncodes)
    monkeypatch.setattr(wheel_script.subprocess, "run", fake)
    return fake


def _download(wheel_script: ModuleType, timeout: int = 60) -> None:
    wheel_script.download_package(ARTIFACT_PATH, TARBALL, timeout)


def test_download_runs_the_read_helper_with_the_environment(
    wheel_script: ModuleType, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    monkeypatch.setenv("LLM_ARTIFACTS_PRIMARY", "aws")
    run = _fake_run(wheel_script, monkeypatch, [0])

    _download(wheel_script)

    assert len(run.calls) == 1
    cmd, kwargs = run.calls[0]
    assert cmd == ["bash", str(HELPER_PATH), f"{ARTIFACT_PATH}/{TARBALL}", TARBALL]
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env == dict(os.environ)
    assert env["ARTIFACTORY_NETRC"] == NETRC
    assert env["ARTIFACTORY_CREDENTIAL_ID"] == CREDENTIAL_ID
    assert env["LLM_ARTIFACTS_PRIMARY"] == "aws"
    # The helper's served-by line and its error messages belong in the build log.
    assert not kwargs.get("capture_output")
    assert kwargs.get("stderr") is None
    assert sleeps == []


def test_download_polls_until_the_package_is_published(
    wheel_script: ModuleType, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    run = _fake_run(wheel_script, monkeypatch, [2, 4, 2, 0])

    _download(wheel_script)

    assert len(run.calls) == 4
    assert sleeps == [60, 60, 60]


@pytest.mark.parametrize("returncode", (2, 4))
def test_download_times_out_after_timeout_attempts(
    returncode: int,
    wheel_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    run = _fake_run(wheel_script, monkeypatch, [returncode] * 3)

    with pytest.raises(TimeoutError, match="after 3 attempts"):
        _download(wheel_script, timeout=3)

    assert len(run.calls) == 3
    assert sleeps == [60, 60]


@pytest.mark.parametrize("primary", (None, "", "aws"))
def test_download_stops_when_the_credentials_are_rejected(
    primary: str | None,
    wheel_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    if primary is not None:
        monkeypatch.setenv("LLM_ARTIFACTS_PRIMARY", primary)
    run = _fake_run(wheel_script, monkeypatch, [3])

    with pytest.raises(PermissionError) as excinfo:
        _download(wheel_script)

    message = str(excinfo.value)
    assert f"credential '{CREDENTIAL_ID}'" in message
    assert "needs renewal or lacks read permission on sw-tensorrt-llm-generic-local" in message
    assert "artifactory.nvidia.com" in message
    assert f"{ARTIFACT_PATH}/{TARBALL}" in message
    assert [signature for signature in RETRY_SIGNATURES if signature in message] == []
    assert len(run.calls) == 1
    assert sleeps == []


def test_rejected_credentials_name_a_placeholder_when_the_credential_id_is_unset(
    wheel_script: ModuleType, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    monkeypatch.delenv("ARTIFACTORY_CREDENTIAL_ID")
    run = _fake_run(wheel_script, monkeypatch, [3])

    with pytest.raises(
        PermissionError, match=re.escape("credential '<unset: ARTIFACTORY_CREDENTIAL_ID>'")
    ):
        _download(wheel_script)

    assert len(run.calls) == 1
    assert sleeps == []


def test_rejected_anonymous_request_when_urm_is_primary(
    wheel_script: ModuleType, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    monkeypatch.setenv("LLM_ARTIFACTS_PRIMARY", "urm")
    run = _fake_run(wheel_script, monkeypatch, [3])

    with pytest.raises(PermissionError) as excinfo:
        _download(wheel_script)

    message = str(excinfo.value)
    assert "rejected the anonymous request" in message
    assert f"{ARTIFACT_PATH}/{TARBALL}" in message
    assert CREDENTIAL_ID not in message
    assert [signature for signature in RETRY_SIGNATURES if signature in message] == []
    assert len(run.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("returncode", (64, 1, 127))
def test_download_stops_on_any_other_helper_failure(
    returncode: int,
    wheel_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    run = _fake_run(wheel_script, monkeypatch, [returncode])

    with pytest.raises(RuntimeError, match=rf"exit code {returncode}\b"):
        _download(wheel_script)

    assert len(run.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("arch", "tarball"),
    (("x86_64", "TensorRT-LLM.tar.gz"), ("sbsa", "TensorRT-LLM-GH200.tar.gz")),
)
def test_command_line_downloads_the_package_for_its_arch(
    arch: str,
    tarball: str,
    wheel_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Extraction writes under the project directory, so point it at tmp_path.
    monkeypatch.setattr(wheel_script, "get_project_dir", lambda: tmp_path)
    run = _fake_run(wheel_script, monkeypatch, [0])
    parser = argparse.ArgumentParser()
    wheel_script.add_arguments(parser)
    args = parser.parse_args(["--arch", arch, "--timeout", "60", "--artifact_path", ARTIFACT_PATH])

    with pytest.raises(_StopBeforeExtract):
        wheel_script.get_wheel_from_package(**vars(args))

    helper = tmp_path / "jenkins" / "scripts" / "fetch_build_artifact.sh"
    assert [cmd for cmd, _ in run.calls] == [
        ["bash", str(helper), f"{ARTIFACT_PATH}/{tarball}", tarball],
        ["tar", "-zxf", tarball, "-C", str(tmp_path / "tmp")],
    ]
