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


def _load_test_unittests_module():
    defs_module = types.ModuleType("defs")
    conftest_module = types.ModuleType("defs.conftest")
    conftest_module.tests_path = lambda: Path("tests")
    sys.modules.setdefault("defs", defs_module)
    sys.modules.setdefault("defs.conftest", conftest_module)

    module_path = Path(__file__).parents[2] / "integration" / "defs" / "test_unittests.py"
    spec = importlib.util.spec_from_file_location("_test_unittests_wrapper", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_executor_unittest_group_disables_threadleak_checker():
    module = _load_test_unittests_module()
    command = ["-m", "pytest", "unittest/_torch/executor"]

    module._append_case_specific_pytest_options(command, "unittest/_torch/executor")

    assert "-o" in command
    assert "threadleak=False" in command


def test_non_executor_unittest_group_keeps_threadleak_checker():
    module = _load_test_unittests_module()
    command = ["-m", "pytest", "unittest/_torch/attention"]

    module._append_case_specific_pytest_options(command, "unittest/_torch/attention")

    assert "threadleak=False" not in command
