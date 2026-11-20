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
"""VisualGen serving code reads environment variables only through ``serve/envs.py``.

CI narrows a VisualGen-only change to the tests under a ``visual_gen/`` path, so
the VisualGen part of the environment-read lint lives here; the rest of the
serve tree is checked by ``tests/unittest/serve/test_envs.py``.
"""

from pathlib import Path

import pytest
from test_common.serve.env_lint import find_direct_env_reads, scope_files

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_no_direct_environment_reads_in_visual_gen_serving():
    files = scope_files(REPO_ROOT, visual_gen=True)
    assert files, (
        "the VisualGen env-lint scanned no file; update scope_files(visual_gen=True) in "
        "tests/test_common/serve/env_lint.py to the current VisualGen serving paths."
    )
    findings = find_direct_env_reads(files, root=REPO_ROOT)
    assert not findings, "\n".join(finding.message() for finding in findings)
