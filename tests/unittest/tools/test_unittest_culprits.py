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
"""Tests for culprit attribution in the unittest integration wrapper."""

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "tests/integration/defs/test_unittests.py"


def _load_test_unittests_module() -> types.ModuleType:
    """Load the wrapper without importing its heavyweight integration conftest."""
    defs_module = types.ModuleType("defs")
    defs_module.__path__ = []
    conftest_module = types.ModuleType("defs.conftest")

    def tests_path() -> str:
        return str(_REPO_ROOT / "tests")

    conftest_module.tests_path = tests_path

    previous_defs = sys.modules.get("defs")
    previous_conftest = sys.modules.get("defs.conftest")
    sys.modules["defs"] = defs_module
    sys.modules["defs.conftest"] = conftest_module
    try:
        spec = importlib.util.spec_from_file_location("_test_unittests_under_test", _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_defs is None:
            sys.modules.pop("defs", None)
        else:
            sys.modules["defs"] = previous_defs
        if previous_conftest is None:
            sys.modules.pop("defs.conftest", None)
        else:
            sys.modules["defs.conftest"] = previous_conftest
    return module


_TEST_UNITTESTS = _load_test_unittests_module()
_junit_culprits = _TEST_UNITTESTS._junit_culprits
_fail_unittests = _TEST_UNITTESTS._fail_unittests


def test_junit_culprits_reports_failures_and_errors(tmp_path: Path) -> None:
    report = tmp_path / "results.xml"
    report.write_text(
        """<testsuites>
  <testsuite>
    <testcase classname="pkg.test_alpha" name="test_failure"><failure /></testcase>
    <testcase classname="pkg.test_beta" name="test_error"><error /></testcase>
    <testcase classname="pkg.test_gamma" name="test_pass" />
    <testcase classname="pkg.test_delta"><failure /></testcase>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )

    assert _junit_culprits(str(report)) == [
        "FAILED pkg.test_alpha::test_failure",
        "ERROR pkg.test_beta::test_error",
    ]


@pytest.mark.parametrize("contents", [None, "<testsuites>"])
def test_junit_culprits_ignores_unreadable_or_malformed_xml(
    tmp_path: Path, contents: str | None
) -> None:
    report = tmp_path / "results.xml"
    if contents is not None:
        report.write_text(contents, encoding="utf-8")

    assert _junit_culprits(str(report)) == []


def _unfinished_failure(
    tmp_path: Path, case: str, nodeids: list[str], junit_contents: str | None = None
) -> str:
    (tmp_path / "unfinished_test.txt").write_text("\n".join(nodeids) + "\n", encoding="utf-8")
    report = tmp_path / "results.xml"
    if junit_contents is not None:
        report.write_text(junit_contents, encoding="utf-8")
    with pytest.raises(AssertionError) as error:
        _fail_unittests(
            "failure reported in unittests",
            str(report),
            str(tmp_path),
            case,
        )
    return str(error.value)


def test_fail_unittests_filters_unfinished_tests_by_path_boundary(tmp_path: Path) -> None:
    message = _unfinished_failure(
        tmp_path,
        "unittest/foo",
        [
            "unittest/foo::test_package",
            "unittest/foo/test_target.py::test_target",
            "unittest/foo_bar/test_stale.py::test_stale",
            "unittest/other/test_stale.py::test_stale",
        ],
    )

    assert "IN-FLIGHT unittest/foo::test_package" in message
    assert "IN-FLIGHT unittest/foo/test_target.py::test_target" in message
    assert "foo_bar" not in message
    assert "unittest/other" not in message


def test_fail_unittests_matches_stage_prefixed_unfinished_tests(tmp_path: Path) -> None:
    message = _unfinished_failure(
        tmp_path,
        "unittest/_torch/thop/parallel",
        [
            "DGX_H100-PyTorch-1/unittest/_torch/thop/parallel/test_target.py::test_target",
            "DGX_H100-PyTorch-1/unittest/_torch/thop/parallel_extra/test_stale.py::test_stale",
            "DGX_H100-PyTorch-1/unittest/_torch/other/test_stale.py::test_stale",
        ],
    )

    assert (
        "IN-FLIGHT DGX_H100-PyTorch-1/unittest/_torch/thop/parallel/test_target.py::test_target"
    ) in message
    assert "parallel_extra" not in message
    assert "unittest/_torch/other" not in message


def test_fail_unittests_normalizes_test_selector_to_its_case_path(tmp_path: Path) -> None:
    message = _unfinished_failure(
        tmp_path,
        "-m gpu unittest/foo/test_target.py::test_selected",
        [
            "DGX_H100-PyTorch-1/unittest/foo/test_target.py::test_selected",
            "DGX_H100-PyTorch-1/unittest/foo/test_target.py::test_selected[param]",
            "DGX_H100-PyTorch-1/unittest/foo/test_target.py::test_other",
            "DGX_H100-PyTorch-1/unittest/foo/test_target.pyx::test_stale",
            "DGX_H100-PyTorch-1/unittest/other/test_stale.py::test_stale",
        ],
    )

    assert "unittest/foo/test_target.py::test_selected" in message
    assert "unittest/foo/test_target.py::test_selected[param]" in message
    assert "test_target.py::test_other" not in message
    assert "test_target.pyx" not in message
    assert "unittest/other" not in message


def test_fail_unittests_preserves_parameter_selector(tmp_path: Path) -> None:
    message = _unfinished_failure(
        tmp_path,
        "unittest/foo/test_target.py::test_selected[param-a]",
        [
            "DGX_H100-PyTorch-1/unittest/foo/test_target.py::test_selected[param-a]",
            "DGX_H100-PyTorch-1/unittest/foo/test_target.py::test_selected[param-b]",
            "DGX_H100-PyTorch-1/unittest/foo/test_target.py::test_other[param-a]",
        ],
    )

    assert "test_selected[param-a]" in message
    assert "test_selected[param-b]" not in message
    assert "test_other[param-a]" not in message


def test_fail_unittests_caps_culprit_list(tmp_path: Path) -> None:
    nodeids = [f"unittest/foo/test_{index}.py::test_case" for index in range(22)]

    message = _unfinished_failure(tmp_path, "unittest/foo", nodeids)

    assert "IN-FLIGHT unittest/foo/test_19.py::test_case" in message
    assert "IN-FLIGHT unittest/foo/test_20.py::test_case" not in message
    assert message.endswith("(+2 more)")


def test_fail_unittests_prioritizes_and_deduplicates_in_flight_culprits(
    tmp_path: Path,
) -> None:
    testcases = "".join(
        f'<testcase classname="pkg" name="test_{index}"><failure /></testcase>'
        for index in range(22)
    )
    in_flight = "unittest/foo/test_fatal.py::test_fatal"

    message = _unfinished_failure(
        tmp_path,
        "unittest/foo",
        [in_flight, in_flight],
        f"<testsuites><testsuite>{testcases}</testsuite></testsuites>",
    )

    assert message.count(f"IN-FLIGHT {in_flight}") == 1
    assert message.index("IN-FLIGHT") < message.index("FAILED")
    assert "FAILED pkg::test_18" in message
    assert "FAILED pkg::test_19" not in message
    assert message.endswith("(+3 more)")


def _run_retry_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_once: bool, stale_cache: bool = False
) -> list[subprocess.CompletedProcess[str]]:
    import pandas as pd

    memory_config = pd.DataFrame(
        [{"gpu": "CPU", "unittest_case_name": "unittest/test_retry.py", "parallel_factor": 2}]
    )
    monkeypatch.setattr(pd, "read_csv", lambda path: memory_config)
    nvml = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    def nvml_init() -> None:
        raise NVMLError

    nvml.NVMLError = NVMLError
    nvml.nvmlInit = nvml_init
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setenv("LLM_TEST_PARALLEL_OVERRIDE", "2")
    monkeypatch.setattr(_TEST_UNITTESTS, "tests_path", lambda: str(tmp_path))

    test_dir = tmp_path / "unittest"
    test_dir.mkdir()
    test_file = test_dir / "test_retry.py"
    test_file.write_text(
        "from pathlib import Path\n"
        "def test_pass():\n"
        "    with Path('executions.txt').open('a') as stream:\n"
        "        stream.write('pass\\n')\n"
        + (
            "def test_flaky():\n"
            "    marker = Path('already_failed')\n"
            "    failed_before = marker.exists()\n"
            "    marker.touch()\n"
            "    assert failed_before\n"
            if fail_once
            else ""
        ),
        encoding="utf-8",
    )
    if stale_cache:
        cache = tmp_path / ".pytest_cache/v/cache"
        cache.mkdir(parents=True)
        (cache / "lastfailed").write_text(
            json.dumps({"unittest/test_other.py::test_stale": True}), encoding="utf-8"
        )

    results = []

    def run_cmd(command: list[str], cwd: str, env: dict[str, str]) -> None:
        xml_path = next(
            arg.split("=", 1)[1] for arg in command if arg.startswith("--periodic-junit-xmlpath=")
        )
        child_command = [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "--confcutdir",
            str(tmp_path),
            "--rootdir",
            str(tmp_path),
            f"--junitxml={xml_path}",
            str(test_file),
        ]
        for index, arg in enumerate(command):
            if arg == "-o":
                child_command += [arg, command[index + 1]]
            elif arg == "-n":
                child_command += ["-p", "xdist.plugin", arg, command[index + 1]]
            elif arg == "--lf" or arg.startswith("--lfnf="):
                child_command.append(arg)
        result = subprocess.run(
            child_command,
            cwd=cwd,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        results.append(result)
        result.check_returncode()
        if not fail_once and len(results) == 1:
            raise subprocess.CalledProcessError(1, command)

    request = Mock()
    request.config.getoption.return_value = None
    output = tmp_path / "output"
    output.mkdir()
    _TEST_UNITTESTS.test_unittests_v2(
        str(tmp_path),
        types.SimpleNamespace(run_cmd=run_cmd),
        "unittest/test_retry.py",
        str(output),
        request,
    )
    return results


@pytest.mark.parametrize("stale_cache", [False, True])
def test_retry_without_failed_tests_preserves_process_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_cache: bool
) -> None:
    with pytest.raises(AssertionError, match="failure reported in unittests"):
        _run_retry_wrapper(tmp_path, monkeypatch, fail_once=False, stale_cache=stale_cache)

    assert (tmp_path / "executions.txt").read_text(encoding="utf-8").splitlines() == ["pass"]


def test_retry_only_runs_current_failed_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _run_retry_wrapper(tmp_path, monkeypatch, fail_once=True, stale_cache=True)

    assert [result.returncode for result in results] == [1, 0]
    assert (tmp_path / "executions.txt").read_text(encoding="utf-8").splitlines() == ["pass"]
