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
"""Tests for KV cache budget splitting between target and draft managers."""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

GB = 1 << 30


def _make_creator(
    max_gpu_total_bytes: int,
    host_cache_size=None,
    total_kv_per_token: int = 100,
    target_kv_per_token: int = 80,
    total_kv_intercept: int = 0,
    target_kv_intercept: int = 0,
) -> KvCacheCreator:
    """Minimal KvCacheCreator for budget-split helpers.

    ``*_intercept`` model the affine fixed cost (e.g. mamba SSM state) that a
    manager pays per batch regardless of token count. The draft cost is derived
    as ``total - target`` for both slope and intercept (see
    ``_get_target_and_draft_cache_costs``).
    """
    c = object.__new__(KvCacheCreator)

    c._kv_cache_config = KvCacheConfig(
        max_gpu_total_bytes=max_gpu_total_bytes,
        host_cache_size=host_cache_size,
    )
    c._tokens_per_block = 64
    c._max_batch_size = 1
    c._mapping = Mock()
    c._model_engine = Mock()

    c._kv_cache_manager_cls = Mock()
    c._kv_cache_manager_cls.get_cache_size_per_token = Mock(
        return_value=(target_kv_per_token, target_kv_intercept)
    )

    c._get_kv_size_per_token = Mock(
        return_value=CacheCost(slope=total_kv_per_token, intercept=total_kv_intercept)
    )

    return c


class TestSplitGpuBudgetForDraft:
    def test_gpu_budget_split_proportionally(self):
        total_gpu = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu, total_kv_per_token=100, target_kv_per_token=80
        )

        draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        assert draft_config is not None
        assert c._kv_cache_config.max_gpu_total_bytes == 8 * GB
        assert draft_config.max_gpu_total_bytes == 2 * GB
        assert c._kv_cache_config.host_cache_size is None

    def test_returns_none_when_no_gpu_budget(self):
        c = _make_creator(max_gpu_total_bytes=0)

        assert c._split_kv_cache_budget_for_draft("max_gpu_total_bytes") is None

    def test_returns_none_when_draft_kv_zero(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB, total_kv_per_token=100, target_kv_per_token=100
        )

        assert c._split_kv_cache_budget_for_draft("max_gpu_total_bytes") is None


class TestSplitHostCacheBudgetForDraft:
    def test_host_budget_split_proportionally(self):
        total_gpu = 10 * GB
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert c._kv_cache_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB
        assert c._kv_cache_config.max_gpu_total_bytes == total_gpu

    def test_host_budget_not_doubled(self):
        """Regression: both managers used to receive the full host_cache_size."""
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert (c._kv_cache_config.host_cache_size + draft_config.host_cache_size) == total_host

    def test_host_split_without_gpu_budget_uses_slope_ratio(self):
        """V1 non-VSWA: host split must not depend on max_gpu_total_bytes."""
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert draft_config is not None
        assert c._kv_cache_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB

    def test_host_split_merges_into_existing_draft_config(self):
        total_gpu = 10 * GB
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size", draft_config)

        assert draft_config.max_gpu_total_bytes == 2 * GB
        assert c._kv_cache_config.max_gpu_total_bytes == 8 * GB
        assert draft_config.host_cache_size == 4 * GB
        assert c._kv_cache_config.host_cache_size == 16 * GB

    def test_host_split_after_gpu_split_is_unaffected_by_target_only_gpu_budget(self):
        """Regression: host split used to read max_gpu_total_bytes (already
        overridden to target's share by the prior GPU split) instead of the
        host budget, producing a skewed ratio. Now host split uses the host
        budget directly and stays proportional to the cache costs."""
        total_gpu = 10 * GB
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size", draft_config)

        assert c._kv_cache_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB

    def test_no_host_cache_leaves_none(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=None,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        assert c._split_kv_cache_budget_for_draft("host_cache_size") is None

    def test_zero_host_cache_unchanged(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=0,
            total_kv_per_token=100,
            target_kv_per_token=80,
        )

        assert c._split_kv_cache_budget_for_draft("host_cache_size") is None

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

        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert (c._kv_cache_config.host_cache_size + draft_config.host_cache_size) == total_host

    def test_budgets_sum_to_original_with_gpu_and_host(self):
        total_gpu = 15 * GB
        total_host = 30 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=total_host,
            total_kv_per_token=1000,
            target_kv_per_token=700,
        )

        draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")
        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size", draft_config)

        assert (
            c._kv_cache_config.max_gpu_total_bytes + draft_config.max_gpu_total_bytes
        ) == total_gpu
        assert (c._kv_cache_config.host_cache_size + draft_config.host_cache_size) == total_host


class TestHostSplitIgnoresGpuFixedCost:
    """The fixed (intercept) cost models GPU-resident state (e.g. mamba SSM
    state) and is not charged against host offload memory. The host split must
    therefore stay proportional to the per-token (slope) cost even when the
    GPU-resident fixed cost dwarfs the host budget."""

    def test_host_split_proportional_despite_large_intercept(self):
        total_host = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=50 * GB,  # huge GPU fixed cost, irrelevant to host
        )

        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        # Intercept ignored for host -> proportional on slope (draft 20/100).
        assert draft_config is not None
        assert c._kv_cache_config.host_cache_size == 8 * GB
        assert draft_config.host_cache_size == 2 * GB

    def test_host_split_sums_to_original_despite_large_intercept(self):
        total_host = 20 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=100 * GB,
        )

        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert (c._kv_cache_config.host_cache_size + draft_config.host_cache_size) == total_host


class TestGpuSplitChargesFixedCost:
    """``max_gpu_total_bytes`` is where the GPU-resident fixed cost lives, so it
    is charged the intercept and fails fast when the budget can't fit it."""

    def test_gpu_split_subtracts_intercept(self):
        total_gpu = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=5 * GB,  # draft intercept = 5 - 0 = 5 GB
            target_kv_intercept=0,
        )

        draft_config = c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

        # slope_budget = 10 - 5 = 5 GB; draft slope share = 5 * 20/100 = 1 GB;
        # draft_budget = draft_intercept (5) + 1 = 6 GB; target = 4 GB.
        assert draft_config.max_gpu_total_bytes == 6 * GB
        assert c._kv_cache_config.max_gpu_total_bytes == 4 * GB

    def test_gpu_split_infeasible_raises(self):
        """A GPU budget too small for the combined fixed cost is fatal (the run
        would OOM), so the split must fail fast instead of degrading."""
        total_gpu = 1 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=2 * GB,  # fixed cost exceeds the gpu budget
        )

        with pytest.raises(ValueError, match="GPU budget"):
            c._split_kv_cache_budget_for_draft("max_gpu_total_bytes")

    def test_gpu_raise_does_not_block_subsequent_host_split(self):
        """GPU split raises on infeasible budget, but a host-only split with the
        same large intercept still succeeds proportionally."""
        total_host = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=0,  # no gpu split attempted
            host_cache_size=total_host,
            total_kv_per_token=100,
            target_kv_per_token=80,
            total_kv_intercept=2 * GB,
        )

        draft_config = c._split_kv_cache_budget_for_draft("host_cache_size")

        assert c._kv_cache_config.host_cache_size == 8 * GB
        assert draft_config.host_cache_size == 2 * GB
