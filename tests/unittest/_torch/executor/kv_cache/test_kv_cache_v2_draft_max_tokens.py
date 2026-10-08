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
"""Tests for KvCacheCreator._derive_v2_draft_max_tokens."""

import pytest

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

pytestmark = pytest.mark.cpu_only

GB = 1 << 30


def _make_creator(
    *,
    is_v2: bool = True,
    separate_draft: bool = True,
    user_max_tokens=None,
    slope: int = 100,
) -> KvCacheCreator:
    c = object.__new__(KvCacheCreator)
    c._kv_cache_config = KvCacheConfig(max_tokens=user_max_tokens)
    c._is_kv_cache_manager_v2 = is_v2
    c._max_kv_tokens_in = user_max_tokens
    c._should_create_separate_draft_kv_cache = lambda: separate_draft
    c._get_kv_size_per_token = lambda *args, **kwargs: CacheCost(slope=slope)
    return c


def test_separate_draft_derives_max_tokens_from_final_budget():
    c = _make_creator(slope=100)
    c._derive_v2_draft_max_tokens(GB)
    assert c._kv_cache_config.max_tokens == GB // 100


def test_target_only_leaves_max_tokens_unset():
    # KVCacheManagerV2 keeps max_tokens as _gpu_max_tokens; deriving it for a
    # target-only run would cap max_seq_len and warmup.
    c = _make_creator(separate_draft=False)
    c._derive_v2_draft_max_tokens(GB)
    assert c._kv_cache_config.max_tokens is None


def test_all_windowed_zero_slope_leaves_max_tokens_unset():
    # A zero slope makes tokens_for_budget return 0; V2 must keep its
    # unbounded default instead of a cap of zero tokens.
    c = _make_creator(slope=0)
    c._derive_v2_draft_max_tokens(GB)
    assert c._kv_cache_config.max_tokens is None


def test_user_max_tokens_is_not_overridden():
    c = _make_creator(user_max_tokens=1234)
    c._derive_v2_draft_max_tokens(GB)
    assert c._kv_cache_config.max_tokens == 1234


def test_v1_is_untouched():
    c = _make_creator(is_v2=False)
    c._derive_v2_draft_max_tokens(GB)
    assert c._kv_cache_config.max_tokens is None
