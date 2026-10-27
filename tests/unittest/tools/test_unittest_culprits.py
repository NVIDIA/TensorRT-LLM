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
import sys
import types
from pathlib import Path

import pytest

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
