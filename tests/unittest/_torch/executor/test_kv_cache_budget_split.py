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

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             KVCacheManagerV2)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

GB = 1 << 30


def _make_creator(
    max_gpu_total_bytes: int,
    host_cache_size=None,
    total_kv_per_token: int = 100,
    target_kv_per_token: int = 80,
    kv_cache_manager_cls=KVCacheManager,
) -> KvCacheCreator:
    """Build a minimal KvCacheCreator wired for _split_kv_cache_budget_for_draft.

    Per-token costs are exposed via the new ``CacheCost`` shape; the manager
    mock returns a raw int so we also exercise ``_per_manager_cache_cost``'s
    ``CacheCost.from_raw`` wrapping.

    ``kv_cache_manager_cls`` must be a real class so the ``issubclass`` check
    inside ``_split_kv_cache_budget_for_draft`` (used to detect V2 and
    pre-split the auto-provisioned host quota) works.  Tests default to the
    V1 manager so they don't trip the V2 host auto-provision branch unless
    they explicitly opt in.
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

    c._kv_cache_manager_cls = kv_cache_manager_cls

    c._per_manager_cache_cost = Mock(
        return_value=CacheCost(slope=target_kv_per_token))
    c._get_kv_size_per_token = Mock(return_value=CacheCost(slope=total_kv_per_token))

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

    def test_v1_no_host_cache_leaves_none(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=None,
            total_kv_per_token=100,
            target_kv_per_token=80,
            kv_cache_manager_cls=KVCacheManager,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert draft_config is not None
        # V1 does not auto-provision a host tier, so an unset
        # host_cache_size must remain unset for both target and draft.
        assert c._kv_cache_config.host_cache_size is None
        assert draft_config.host_cache_size is None

    def test_v1_zero_host_cache_unchanged(self):
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=0,
            total_kv_per_token=100,
            target_kv_per_token=80,
            kv_cache_manager_cls=KVCacheManager,
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


class TestV2AutoProvisionHostSplit:
    """Tests for the V2-specific behaviour where the host tier is auto-
    provisioned by ``KVCacheManagerV2.__init__`` when the user did not set
    ``host_cache_size``.

    Before the fix, both target and draft managers each independently
    auto-provisioned a full host tier, doubling host memory.  The split
    method now materialises the would-be combined host quota up front
    (via ``compute_default_v2_host_quota``) and divides it explicitly.
    """

    def test_v2_unset_host_is_pre_split(self, monkeypatch):
        """V2 + host_cache_size unset must end up with both target and
        draft host budgets set (so the V2 constructor will not
        auto-provision again)."""
        from tensorrt_llm._torch.pyexecutor import _util

        # Pin the would-be auto-provision result so the test is deterministic
        # regardless of the runner's available memory.
        monkeypatch.setattr(
            _util, "compute_default_v2_host_quota", lambda gpu_quota: gpu_quota
        )

        total_gpu = 10 * GB
        c = _make_creator(
            max_gpu_total_bytes=total_gpu,
            host_cache_size=None,
            total_kv_per_token=100,
            target_kv_per_token=80,
            kv_cache_manager_cls=KVCacheManagerV2,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert draft_config is not None
        # Both budgets must now be explicitly set (and positive) so the V2
        # constructor's auto-provision branch will not fire.
        assert c._kv_cache_config.host_cache_size is not None
        assert c._kv_cache_config.host_cache_size > 0
        assert draft_config.host_cache_size is not None
        assert draft_config.host_cache_size > 0

    def test_v2_unset_host_sum_matches_auto_provision(self, monkeypatch):
        """The sum of target+draft host budgets must equal what V2 would
        have auto-provisioned for a single manager from the combined GPU
        quota -- otherwise the fix has either over- or under-allocated
        relative to the V2 baseline."""
        from tensorrt_llm._torch.pyexecutor import _util

        sentinel_host = 7 * GB

        def fake_default(gpu_quota):
            assert gpu_quota == 10 * GB  # must be called with combined budget
            return sentinel_host

        monkeypatch.setattr(_util, "compute_default_v2_host_quota", fake_default)

        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=None,
            total_kv_per_token=100,
            target_kv_per_token=80,
            kv_cache_manager_cls=KVCacheManagerV2,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        target_host = c._kv_cache_config.host_cache_size
        draft_host = draft_config.host_cache_size
        # Sum must exactly equal the pre-split auto-provision quota.
        assert target_host + draft_host == sentinel_host
        # And the split must follow the same 80/20 ratio as the GPU split.
        # Use the same draft-first int() truncation order as the implementation.
        expected_draft = int(sentinel_host * 0.2)
        expected_target = sentinel_host - expected_draft
        assert target_host == expected_target
        assert draft_host == expected_draft

    def test_v1_unset_host_left_alone(self):
        """V1 callers (no V2 auto-provision in the constructor) must not
        have a host budget conjured up.  This guards against accidentally
        adding host memory to spec dec runs that opted out."""
        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=None,
            total_kv_per_token=100,
            target_kv_per_token=80,
            kv_cache_manager_cls=KVCacheManager,
        )

        draft_config = c._split_kv_cache_budget_for_draft()

        assert draft_config is not None
        assert c._kv_cache_config.host_cache_size is None
        assert draft_config.host_cache_size is None

    def test_v2_explicit_host_still_split(self, monkeypatch):
        """When the user did set host_cache_size, that value (not the
        auto-provision default) must drive the split."""
        from tensorrt_llm._torch.pyexecutor import _util

        # If the auto-provision path is taken by mistake the test will see
        # a different total.
        monkeypatch.setattr(
            _util, "compute_default_v2_host_quota",
            lambda _: pytest.fail(
                "auto-provision must not run when host_cache_size is set"),
        )

        c = _make_creator(
            max_gpu_total_bytes=10 * GB,
            host_cache_size=20 * GB,
            total_kv_per_token=100,
            target_kv_per_token=80,
            kv_cache_manager_cls=KVCacheManagerV2,
        )

        draft_config = c._split_kv_cache_budget_for_draft()
        assert c._kv_cache_config.host_cache_size == 16 * GB
        assert draft_config.host_cache_size == 4 * GB
