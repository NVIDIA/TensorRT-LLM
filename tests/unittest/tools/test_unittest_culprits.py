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
"""Tests for artifact names, culprit attribution and parallel-config lookup in the unittest wrapper."""

import importlib.util
import os
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

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
_parallel_config_key = _TEST_UNITTESTS._parallel_config_key
_unittest_case_filename = _TEST_UNITTESTS._unittest_case_filename


def test_junit_culprits_reports_failures_and_errors(tmp_path: Path) -> None:
    prefix = "unittest/_torch/modeling_v2/models/kimi_k3_vl/ops/norm/test_modeling_v2_attn_res"
    case = f"{prefix}_fwd.py"
    other_case = f"{prefix}_rmsnorm_fwd.py"
    report = tmp_path / f"results-sub-unittests-{_unittest_case_filename(case)}.xml"
    other_report = tmp_path / f"results-sub-unittests-{_unittest_case_filename(other_case)}.xml"
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

    # Writing a second long-prefix case must not overwrite the first case's failures.
    other_report.write_text(
        '<testsuites><testsuite><testcase name="pass" /></testsuite></testsuites>'
    )
    assert len(_unittest_case_filename(case)) <= 80
    assert _junit_culprits(str(other_report)) == []
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


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("unittest/_torch/attention", "unittest/_torch/attention"),
        (
            "unittest/_torch/attention --ignore=unittest/_torch/attention/sparse/test_a.py"
            " --ignore=unittest/_torch/attention/sparse/msa/test_b.py",
            "unittest/_torch/attention",
        ),
        (
            "unittest/_torch/modeling_v2 --ignore=unittest/_torch/modeling_v2/comm -k gpu",
            "unittest/_torch/modeling_v2 -k gpu",
        ),
        (
            'unittest/trt/attention/test_gpt_attention.py -k "partition0"',
            'unittest/trt/attention/test_gpt_attention.py -k "partition0"',
        ),
    ],
)
def test_parallel_config_key_strips_only_ignore_arguments(case: str, expected: str) -> None:
    assert _parallel_config_key(case) == expected


_GR100_M3_TARGETS = "unittest/_torch/modeling_v2/models/minimax_m3"
_GR100_M3_REPORT_CASES = (
    *(f"{_GR100_M3_TARGETS}/minimax_m3_nvfp4__sm_107__tp{tp}/ops/moe" for tp in (2, 4, 8)),
    *(f"{_GR100_M3_TARGETS}/minimax_m3_nvfp4__sm_107__tp{tp}/ops/m3_moe" for tp in (2, 4)),
)
_CaptureUnittest = Callable[[str, int, bool], list[list[str]]]


def _report_path(command: list[str]) -> Path:
    prefix = "--periodic-junit-xmlpath="
    return Path(next(arg.removeprefix(prefix) for arg in command if arg.startswith(prefix)))


@pytest.fixture
def capture_unittest_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _CaptureUnittest:
    dataframe = MagicMock()
    dataframe.__getitem__.return_value = dataframe
    dataframe.dropna.return_value = dataframe
    dataframe.unique.return_value = dataframe
    dataframe.tolist.return_value = ["CPU"]
    pandas_module = types.ModuleType("pandas")
    pandas_module.read_csv = MagicMock(return_value=dataframe)
    nvml_module = types.ModuleType("pynvml")
    nvml_module.NVMLError = RuntimeError
    nvml_module.nvmlInit = MagicMock()
    nvml_module.nvmlDeviceGetHandleByIndex = MagicMock(return_value=0)
    nvml_module.nvmlDeviceGetName = MagicMock(return_value="CPU")
    monkeypatch.setitem(sys.modules, "pandas", pandas_module)
    monkeypatch.setitem(sys.modules, "pynvml", nvml_module)

    def getoption(name: str, default: object | None = None) -> object | None:
        return default

    request = types.SimpleNamespace(config=types.SimpleNamespace(getoption=getoption))
    output_dir = tmp_path / "session-output"
    output_dir.mkdir()

    def capture(case: str, workers: int, retry: bool) -> list[list[str]]:
        commands: list[list[str]] = []
        dataframe.iterrows.return_value = [
            (0, {"gpu": "CPU", "unittest_case_name": case, "parallel_factor": workers})
        ]
        monkeypatch.setenv("LLM_TEST_PARALLEL_OVERRIDE", str(workers))

        def run_cmd(command: list[str], *, cwd: str, env: dict[str, str]) -> None:
            assert cwd == _TEST_UNITTESTS.tests_path()
            assert env["CBTS_PROCESS_ROLE"] == "inner_pytest"
            commands.append(list(command))
            if workers > 1:
                _report_path(command).write_text(
                    '<testsuites><testsuite tests="1" failures="0" errors="0" '
                    'skipped="0" time="0"><testcase classname="probe" '
                    'name="test_probe" /></testsuite></testsuites>',
                    encoding="utf-8",
                )
            if retry and len(commands) == 1:
                raise subprocess.CalledProcessError(1, command)

        _TEST_UNITTESTS.test_unittests_v2(
            str(tmp_path / "llm"),
            types.SimpleNamespace(run_cmd=run_cmd),
            case,
            str(output_dir),
            request,
        )
        return commands

    return capture


@pytest.mark.cpu_only
def test_unittest_reports_distinguish_gr100_selectors(
    capture_unittest_wrapper: _CaptureUnittest,
) -> None:
    paths = [
        _report_path(capture_unittest_wrapper(case, 1, False)[0]) for case in _GR100_M3_REPORT_CASES
    ]
    repeated = [
        _report_path(capture_unittest_wrapper(case, 1, False)[0]) for case in _GR100_M3_REPORT_CASES
    ]
    assert paths == repeated
    assert len(set(paths)) == len(_GR100_M3_REPORT_CASES) == 5
    assert len({path.parent for path in paths}) == 1
    assert all(
        path.name.startswith("results-sub-unittests-") and path.suffix == ".xml" for path in paths
    )
    assert all(len(os.fsencode(path.name)) <= 255 for path in paths)


@pytest.mark.cpu_only
@pytest.mark.parametrize("retry", [False, True])
def test_unittest_parallel_and_retry_reports_keep_distinct_paths(
    capture_unittest_wrapper: _CaptureUnittest, retry: bool
) -> None:
    calls = [capture_unittest_wrapper(case, 2, retry) for case in _GR100_M3_REPORT_CASES[1:3]]
    assert all(len(commands) == (2 if retry else 1) for commands in calls)
    paths = [_report_path(command) for commands in calls for command in commands]
    assert len(set(paths)) == len(paths)
    assert all(path.name.endswith(".xml.intermediate") for path in paths)
    assert all(
        _report_path(commands[0]).name.startswith("parallel-sub-results-") for commands in calls
    )
    if retry:
        assert all(
            _report_path(commands[1]).name.startswith("retry-sub-results-") for commands in calls
        )
        assert all("--lf" in commands[1] for commands in calls)
    final_reports = list(paths[0].parent.glob("results-sub-unittests-*.xml"))
    assert len(final_reports) == 2
    assert all(list(ET.parse(path).getroot().iter("testcase")) for path in final_reports)


@pytest.mark.cpu_only
def test_unittest_moe_reports_keep_reporting_exclusion_prefix(
    capture_unittest_wrapper: _CaptureUnittest,
) -> None:
    command = capture_unittest_wrapper("unittest/_torch/moe/test_moe_module.py", 1, False)[0]
    assert _report_path(command).name.startswith("moe-results-sub-unittests-")
    unfinished = next(arg for arg in command if arg.startswith("--periodic-unfinished-test-path="))
    label = (
        _report_path(command).name.removeprefix("moe-results-sub-unittests-").removesuffix(".xml")
    )
    assert Path(unfinished.split("=", 1)[1]).name == (f"moe-unfinished-sub-unittests-{label}.txt")


@pytest.mark.cpu_only
def test_unittest_report_names_and_long_boundary(
    capture_unittest_wrapper: _CaptureUnittest,
) -> None:
    cases = ("unittest/foo", "x" * 80, "x" * 81, "x" * 80 + "y")
    reports = [_report_path(capture_unittest_wrapper(case, 1, False)[0]) for case in cases]
    labels = [p.name.removeprefix("results-sub-unittests-").removesuffix(".xml") for p in reports]
    assert labels[0].startswith("unittest-foo-")
    assert len(labels[0]) == len("unittest-foo-") + 16
    assert all(len(label) == 80 for label in labels[1:])
    assert len(set(reports)) == len(reports)
