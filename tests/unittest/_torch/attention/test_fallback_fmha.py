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

"""Dispatch contract of ``FallbackFmha.is_supported``.

The native op silently drops requests it cannot serve, so the fallback has to
refuse them itself rather than lower them and read back an untouched output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs


@pytest.mark.parametrize(
    ("is_cross", "update_kv_cache", "expected"),
    (
        (False, False, False),
        (False, True, True),
        (True, False, True),
    ),
)
def test_fallback_support_matches_thop_kv_update_contract(is_cross, update_kv_cache, expected):
    """Do not dispatch requests that the native attention op rejects."""
    fmha = object.__new__(FallbackFmha)
    metadata = SimpleNamespace(is_cross=is_cross)
    forward_args = AttentionForwardArgs(update_kv_cache=update_kv_cache)

    assert fmha.is_supported(None, None, None, metadata, forward_args) is expected


def test_fallback_rejects_raw_fp8_input():
    """Do not dispatch raw FP8 QKV to the native attention op."""
    fmha = object.__new__(FallbackFmha)
    metadata = SimpleNamespace(is_cross=False)
    forward_args = AttentionForwardArgs(update_kv_cache=True)
    q = torch.empty((1, 128), dtype=torch.float8_e4m3fn)

    assert not fmha.is_supported(q, None, None, metadata, forward_args)
