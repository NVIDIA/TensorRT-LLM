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

"""Shared pytest configuration for AutoDeploy unit tests.

AutoDeploy tests share lightweight helper modules under ``_utils_test``. Keeping
that directory on ``sys.path`` here lets tests import helpers by module name,
which mirrors the standalone llmc package's generated test conftest.

This avoids repeating path manipulation in individual test files and keeps the
same import style valid in both places:

* source tree: ``tests/unittest/auto_deploy/_utils_test``
* standalone package: ``tests/_utils_test``
"""

import sys
from pathlib import Path


def _add_utils_test_dir_to_path() -> None:
    """Make AutoDeploy test helpers importable by module name.

    The standalone package generator creates an equivalent conftest for copied
    tests. Keeping this source-tree behavior aligned prevents tests from
    depending on TensorRT-LLM-specific package names for test-only helpers.
    """
    utils_test_dir = str(Path(__file__).resolve().parent / "_utils_test")
    if utils_test_dir not in sys.path:
        sys.path.insert(0, utils_test_dir)


_add_utils_test_dir_to_path()
