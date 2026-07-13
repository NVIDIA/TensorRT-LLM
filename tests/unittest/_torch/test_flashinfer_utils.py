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

import pytest

from tensorrt_llm._torch import flashinfer_utils


@pytest.mark.parametrize(
    ("env_value", "sm_version", "expected"),
    [
        (None, 89, False),
        ("1", 89, False),
        ("0", 89, False),
        (None, 90, True),
        ("1", 90, True),
        ("0", 90, False),
        (None, 100, True),
    ],
)
def test_get_env_enable_pdl_hardware_gate(monkeypatch, env_value, sm_version, expected):
    if env_value is None:
        monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_ENABLE_PDL", env_value)
    monkeypatch.setattr(flashinfer_utils, "get_sm_version", lambda: sm_version)
    monkeypatch.delattr(flashinfer_utils.get_env_enable_pdl, "_printed", raising=False)

    assert flashinfer_utils.get_env_enable_pdl() is expected
