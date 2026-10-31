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
"""Tests for CTest report selection and retry merging."""

import importlib.util
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CPP_COMMON_PATH = _REPO_ROOT / "tests/integration/defs/cpp/cpp_common.py"


def _load_cpp_common(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    build_wheel = types.ModuleType("build_wheel")
    build_wheel.get_build_dir = lambda *args: Path(".")
    monkeypatch.setitem(sys.modules, "build_wheel", build_wheel)

    spec = importlib.util.spec_from_file_location("_cpp_common_under_test", _CPP_COMMON_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _output_path(command: list[str], cwd: Path) -> Path:
    path = Path(command[command.index("--output-junit") + 1])
    return path if path.is_absolute() else cwd / path


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_parallel_ctest_merges_retry_into_requested_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    cpp_common = _load_cpp_common(monkeypatch)
    stale_report = tmp_path / "results-stale.xml"
    stale_contents = '<testsuite failures="0" time="1"><testcase name="stale"/></testsuite>'
    stale_report.write_text(stale_contents, encoding="utf-8")

    report = tmp_path / "reports" / "results-current.xml"
    report.parent.mkdir()
    output = str(report) if absolute else str(report.relative_to(tmp_path))
    calls = []

    def run_ctest(command: list[str], cwd: Path, **_kwargs: object) -> None:
        calls.append(command)
        output_path = _output_path(command, cwd)
        if len(calls) == 1:
            output_path.write_text(
                '<testsuite failures="1" time="2"><testcase name="passed"/>'
                '<testcase name="flaky"><failure/></testcase></testsuite>',
                encoding="utf-8",
            )
            raise subprocess.CalledProcessError(1, command)
        output_path.write_text(
            '<testsuite failures="0" time="1"><testcase name="flaky"/></testsuite>',
            encoding="utf-8",
        )

    monkeypatch.setattr(cpp_common, "run_ctest", run_ctest)
    command = ["ctest", "--output-junit", output]

    cpp_common.parallel_run_ctest(command, cwd=tmp_path, env={}, parallel=2)

    result = ET.parse(report).getroot()
    assert [case.attrib["name"] for case in result] == ["passed", "flaky"]
    assert result.attrib["failures"] == "0"
    assert result.attrib["parallel"] == "2"
    assert stale_report.read_text(encoding="utf-8") == stale_contents


def test_parallel_ctest_does_not_retry_from_stale_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpp_common = _load_cpp_common(monkeypatch)
    stale_report = tmp_path / "results-stale.xml"
    stale_contents = '<testsuite failures="0" time="1"><testcase name="stale"/></testsuite>'
    stale_report.write_text(stale_contents, encoding="utf-8")
    missing_report = tmp_path / "reports" / "results-current.xml"
    missing_report.parent.mkdir()
    calls = []

    def fail_ctest(command: list[str], **_kwargs: object) -> None:
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cpp_common, "run_ctest", fail_ctest)
    command = ["ctest", "--output-junit", str(missing_report)]

    with pytest.raises(subprocess.CalledProcessError):
        cpp_common.parallel_run_ctest(command, cwd=tmp_path, env={}, parallel=2)

    assert len(calls) == 1
    assert stale_report.read_text(encoding="utf-8") == stale_contents
    assert not list(tmp_path.rglob("*.intermediate"))
