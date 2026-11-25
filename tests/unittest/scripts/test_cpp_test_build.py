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

"""Coverage and report ownership for incremental C++ test builds."""

import importlib.util
import subprocess
import sys
import types
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[3]


def _load(name):
    path = _ROOT / "tests/integration/defs/cpp" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(name, arch="90", **params):
    return SimpleNamespace(
        fixturenames=["build_google_tests"],
        originalname=name,
        callspec=SimpleNamespace(params={"build_google_tests": arch, **params}),
    )


def test_build_targets_match_collected_tests_for_one_architecture():
    items = [
        _item("test_unit_tests", test_group="batch_manager"),
        _item("test_unit_tests", test_group="executor_bounce"),
        _item("test_unit_tests", "80", test_group="runtime"),
        _item("test_cache_transceiver", nprocs=2),
        _item("test_cache_transceiver", nprocs=8),
        SimpleNamespace(fixturenames=[]),
    ]
    assert _load("build_targets").selected_build_targets(items, "90") == [
        "cacheTransceiverTest",
        "cpp-tests-batch_manager",
        "cpp-tests-executor-bounce",
    ]


@pytest.fixture
def cpp_common(monkeypatch):
    # These helper tests do not need a native build environment.
    build_wheel = types.ModuleType("build_wheel")
    build_wheel.get_build_dir = lambda *args: None
    monkeypatch.setitem(sys.modules, "build_wheel", build_wheel)
    return _load("cpp_common")


def test_bundle_preparation_uses_the_current_library_and_source_paths(
    cpp_common, monkeypatch, tmp_path
):
    bundle = tmp_path / "cpp/build"
    bundle.mkdir(parents=True)
    template = bundle / "CTestTestfile.cmake.in"
    template.write_text("__TRTLLM_CPP_BUNDLE_ROOT__\n__TRTLLM_CPP_SOURCE_ROOT__\n")
    monkeypatch.setattr(cpp_common, "find_test_bundle", lambda _: bundle)
    monkeypatch.setattr(cpp_common, "find_root_dir", lambda: tmp_path)

    for name in ("first", "second"):
        library_dir = tmp_path / name / "libs"
        library_dir.mkdir(parents=True)
        distribution = SimpleNamespace(locate_file=lambda _, path=library_dir: path)
        monkeypatch.setattr(
            cpp_common.importlib.metadata, "distribution", lambda _, package=distribution: package
        )
        assert cpp_common.prepare_test_bundle("Release")
        assert (bundle / "lib").resolve() == library_dir
        assert (bundle / "CTestTestfile.cmake").read_text() == f"{bundle}\n{tmp_path}\n"


def test_failed_ctest_without_new_report_cannot_retry_stale_results(
    cpp_common, monkeypatch, tmp_path
):
    current = tmp_path / "results-current.xml"
    current.write_text("stale current report")
    unrelated = tmp_path / "results-other.xml"
    unrelated.write_text("another group")
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        assert not current.exists()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cpp_common, "run_ctest", fail)
    with pytest.raises(subprocess.CalledProcessError):
        cpp_common.parallel_run_ctest(["ctest", "--output-junit", str(current)], cwd=tmp_path)
    assert len(calls) == 1
    assert unrelated.read_text() == "another group"


def test_retry_preserves_successes_and_replaces_failed_case(cpp_common, monkeypatch, tmp_path):
    current = tmp_path / "results-current.xml"
    unrelated = tmp_path / "results-other.xml"
    unrelated.write_text("another group")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            current.write_text(
                '<testsuite tests="2" failures="1" time="2">'
                '<testcase name="already_passed" time="1"/>'
                '<testcase name="retried" time="0.5"><failure message="parallel failure"/>'
                "</testcase></testsuite>"
            )
            raise subprocess.CalledProcessError(1, command)
        assert command[-1] == "--rerun-failed"
        assert "CTEST_PARALLEL_LEVEL" not in kwargs["env"]
        current.write_text(
            '<testsuite tests="1" failures="0" time="1">'
            '<testcase name="retried" time="0.25"/></testsuite>'
        )

    monkeypatch.setattr(cpp_common, "run_ctest", run)
    cpp_common.parallel_run_ctest(["ctest", "--output-junit", str(current)], cwd=tmp_path)
    assert len(calls) == 2
    report = ElementTree.parse(current).getroot()
    assert sorted(case.attrib["name"] for case in report) == ["already_passed", "retried"]
    assert report.find(".//failure") is None
    assert report.find("testcase[@name='retried']").attrib["time"] == "0.25"
    assert report.attrib["tests"] == "2"
    assert report.attrib["failures"] == "0"
    assert report.attrib["time"] == "3"
    assert report.attrib["parallel"] == "2"
    assert unrelated.read_text() == "another group"
