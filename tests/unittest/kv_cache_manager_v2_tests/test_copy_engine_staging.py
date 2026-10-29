# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The GPU<->Disk staging buffer size is configurable via env (default 64 MB)."""

from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2._copy_engine import staging_buffer_size_bytes
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2._copy_engine import staging_buffer_size_bytes

ENV = "TLLM_KV_CACHE_MANAGER_V2_STAGING_BUFFER_SIZE_MB"


def test_default_is_64_mib(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert staging_buffer_size_bytes() == 64 << 20


def test_empty_value_uses_default(monkeypatch):
    monkeypatch.setenv(ENV, "")
    assert staging_buffer_size_bytes() == 64 << 20


def test_override_in_mib(monkeypatch):
    monkeypatch.setenv(ENV, "1024")
    assert staging_buffer_size_bytes() == 1 << 30


@pytest.mark.parametrize("value", ["0", "-8"])
def test_non_positive_is_rejected(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        staging_buffer_size_bytes()
