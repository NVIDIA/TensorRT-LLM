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
"""``TRTLLM_SERVER_KEEP_ALIVE_TIMEOUT`` sets the aggregated server's HTTP keep-alive timeout.

The value is read once at import time, so each case imports the module in a
fresh interpreter.
"""

import os
import subprocess
import sys

import pytest

_ENV = "TRTLLM_SERVER_KEEP_ALIVE_TIMEOUT"
_PROBE = "from tensorrt_llm.serve import openai_server as m; print(m.TIMEOUT_KEEP_ALIVE)"


def _import_with(value):
    env = {k: v for k, v in os.environ.items() if k != _ENV}
    if value is not None:
        env[_ENV] = value
    return subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "5"), ("", "5"), ("3600", "3600"), ("0", "0")],
)
def test_keep_alive_timeout_env(value, expected):
    proc = _import_with(value)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == expected


def test_keep_alive_timeout_env_rejects_negative():
    proc = _import_with("-1")
    assert proc.returncode != 0
    assert f"{_ENV} must be a non-negative integer" in proc.stderr
