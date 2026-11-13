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
"""Wiring only.

    framework.py    runs the plugin's command line
    cases/          the mock suite it runs against
    mocks.py        what `cases/` holds, named: one class per mock module
    test_*.py       the criteria: the options passed, and the ids expected back

Not `cases/conftest.py`, which is mock content loaded by the run under test.
"""

import pytest
from framework import SelectionHarness

pytest_plugins = ["pytester"]


@pytest.fixture
def selection(pytester: pytest.Pytester) -> SelectionHarness:
    """The plugin's command line, bound to a fresh temporary directory."""
    return SelectionHarness(pytester)
