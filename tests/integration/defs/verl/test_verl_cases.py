# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Wrapper tests that invoke verl repo tests via subprocess.

The verl repo is cloned by CI into the path specified by the VERL_ROOT
environment variable (see verl_config.yml for setup details).
"""

import os
import subprocess
import sys

import pytest

VERL_ROOT = os.environ.get("VERL_ROOT", "")


def _run_verl_test(test_path, timeout=600):
    """Run a test from the verl repo via subprocess."""
    full_path = os.path.join(VERL_ROOT, test_path)
    assert os.path.exists(full_path), f"Verl test not found: {full_path}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", full_path, "-v", "--tb=short"],
        env=os.environ.copy(),
        timeout=timeout,
    )
    assert result.returncode == 0, f"Verl test failed with return code {result.returncode}"


@pytest.mark.skipif(not VERL_ROOT, reason="VERL_ROOT not set")
def test_async_server():
    _run_verl_test("tests/workers/rollout/rollout_trtllm/test_async_server.py")
