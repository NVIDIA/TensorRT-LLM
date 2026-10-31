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
"""Path setup + marker registration for the prepare-inputs host-cost tests.

`tests/unittest` is added to `sys.path` so the shared `utils.util` helpers
(`skip_pre_blackwell`) resolve the same way they do for the unit tests, and
`tests/integration` so the optional clock-lock import in sibling harnesses keeps
working if these tests are ever run from this directory together.
"""

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[2]  # tests/

for p in (_TESTS / "unittest", _TESTS / "integration", _TESTS / "microbenchmarks"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "discrete: zero-threshold structural assert (dispatch counts), pre-merge gate",
    )
