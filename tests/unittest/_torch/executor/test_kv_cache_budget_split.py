# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for _split_kv_cache_budget_for_draft in KvCacheCreator.

Verifies that both GPU (max_gpu_total_bytes) and host (host_cache_size)
budgets are split proportionally between target and draft KV cache managers.
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

GB = 1 << 30


def _make_creator(
    max_gpu_total_bytes: int,
    host_cache_size=None,
    total_kv_per_token: int = 100,
    target_kv_per_token: int = 80,
) -> KvCacheCreator:
    """Build a minimal KvCacheCreator wired for _split_kv_cache_budget_for_draft."""
    c = object.__new__(KvCacheCreator)

    c._kv_cache_config = KvCacheConfig(
        max_gpu_total_bytes=max_gpu_total_bytes,
        host_cache_size=host_cache_size,
    )
    c._tokens_per_block = 64
    c._mapping = Mock()
    c._model_engine = Mock()

    c._kv_cache_manager_cls = Mock()
    c._kv_cache_manager_cls.get_cache_size_per_token = Mock(return_value=target_kv_per_token)

    c._get_kv_size_per_token = Mock(return_value=total_kv_per_token)

    return c


class TestSplitKvCacheBudgetForDraft:
    def test_gpu_budget_split_proportionally(self):
        total_gpu = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu, total_kv_per_token=100, target_kv_per_token=80
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert draft_config is not None
        assert c._kv_cache_config.max_gpu_total_bytes == 8 * GB
        assert draft_config.max_gpu_total_bytes == 2 * GB

    def test_host_budget_split_proportionally(self):
        total_gpu = 10 * GB
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert draft_config is not None
        # GPU: 80% target, 20% draft
        assert c._kv_cache_config.max_gpu_total_bytes == 8 * GB
        assert draft_config.max_gpu_total_bytes == 2 * GB
        # Host: same ratio
        assert c._kv_cache_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB

    def test_host_budget_not_doubled(self):
        """Regression: before the fix, both target and draft managers each
        received the full host_cache_size, doubling total host memory."""
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        target_host = c._kv_cache_config.host_cache_size
        draft_host = draft_config.host_cache_size
        assert target_host + draft_host == total_host

    def test_budgets_sum_to_original(self):
        total_gpu = 15 * GB
        total_host = 30 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=1000,
            target_kv_per_token=700,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert (
            c._kv_cache_config.max_gpu_total_bytes + draft_config.max_gpu_total_bytes
        ) == total_gpu
        assert (c._kv_cache_config.host_cache_size + draft_config.host_cache_size) == total_host

    def test_no_host_cache_leaves_none(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=None,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert draft_config is not None
        assert c._kv_cache_config.host_cache_size is None
        assert draft_config.host_cache_size is None

    def test_zero_host_cache_unchanged(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=0,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert draft_config is not None
        # host_cache_size=0 should not be split (guard: host_budget > 0)
        assert draft_config.host_cache_size == 0

    def test_returns_none_when_no_gpu_budget(self):
        c = _make_creator(max_gpu_total_bytes=0)

        assert c._split_kv_cache_budget_for_draft() is None

    def test_returns_none_when_draft_kv_zero(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB, total_kv_per_token=100, target_kv_per_token=100
        )

        assert c._split_kv_cache_budget_for_draft() is None

    @pytest.mark.parametrize("target_frac", [0.5, 0.75, 0.9, 0.95])
    def test_various_ratios(self, target_frac):
        total_gpu = 10 * GB
        total_host = 20 * GB
        total_kv = 1000
        target_kv = int(total_kv * target_frac)

        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=total_kv,
            target_kv_per_token=target_kv,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert (
            c._kv_cache_config.max_gpu_total_bytes + draft_config.max_gpu_total_bytes
        ) == total_gpu
        assert (c._kv_cache_config.host_cache_size + draft_config.host_cache_size) == total_host
