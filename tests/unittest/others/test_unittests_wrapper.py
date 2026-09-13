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

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_test_unittests_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    defs_module = types.ModuleType("defs")
    conftest_module = types.ModuleType("defs.conftest")
    conftest_module.tests_path = lambda: Path("tests")
    monkeypatch.setitem(sys.modules, "defs", defs_module)
    monkeypatch.setitem(sys.modules, "defs.conftest", conftest_module)

    module_path = Path(__file__).parents[2] / "integration" / "defs" / "test_unittests.py"
    spec = importlib.util.spec_from_file_location("_test_unittests_wrapper", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_loader_restores_stub_modules_after_context() -> None:
    original_defs = sys.modules.get("defs")
    original_conftest = sys.modules.get("defs.conftest")
    sentinel_defs = types.ModuleType("defs")
    sentinel_conftest = types.ModuleType("defs.conftest")
    sys.modules["defs"] = sentinel_defs
    sys.modules["defs.conftest"] = sentinel_conftest
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            _load_test_unittests_module(monkeypatch)
            assert sys.modules["defs"] is not sentinel_defs
            assert sys.modules["defs.conftest"] is not sentinel_conftest

        assert sys.modules["defs"] is sentinel_defs
        assert sys.modules["defs.conftest"] is sentinel_conftest
    finally:
        if original_defs is None:
            sys.modules.pop("defs", None)
        else:
            sys.modules["defs"] = original_defs
        if original_conftest is None:
            sys.modules.pop("defs.conftest", None)
        else:
            sys.modules["defs.conftest"] = original_conftest


def test_executor_unittest_group_disables_threadleak_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_test_unittests_module(monkeypatch)
    command = ["-m", "pytest"]

    module._append_case_specific_pytest_options(command, "unittest/_torch/executor")
    command += ["unittest/_torch/executor"]

    assert command == ["-m", "pytest", "-o", "threadleak=False", "unittest/_torch/executor"]


@pytest.mark.parametrize(
    "case",
    [
        "unittest/_torch/executor/test_example.py",
        "unittest/_torch/executor -k test_example",
        'unittest/_torch/executor "unterminated',
    ],
)
def test_executor_unittest_subsets_keep_threadleak_checker(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    module = _load_test_unittests_module(monkeypatch)
    command = ["-m", "pytest", case]

    module._append_case_specific_pytest_options(command, case)

    assert "threadleak=False" not in command


def test_non_executor_unittest_group_keeps_threadleak_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_test_unittests_module(monkeypatch)
    command = ["-m", "pytest", "unittest/_torch/attention"]

    module._append_case_specific_pytest_options(command, "unittest/_torch/attention")

    assert "threadleak=False" not in command
