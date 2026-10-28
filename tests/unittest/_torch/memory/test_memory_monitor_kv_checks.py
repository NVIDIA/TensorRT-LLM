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
"""KV cache monitor entries and the conservation checks over them.

The first thing these tests establish is that check C can *fail*. A check
that cannot be broken by any input is decoration, and the obvious way to
write this one -- against ``available()`` and ``unavailable()`` -- is exactly
that, since both are computed from the fields being compared.

CPU only: the statistics objects are fakes shaped like the real binding.
"""

import types

from tensorrt_llm._torch.memory.monitor.checks import check_kv_pool_conservation, check_kv_quota
from tensorrt_llm._torch.memory.monitor.providers.kv_cache import (
    KvCacheProvider,
    kv_pool_group_stats,
)
from tensorrt_llm._torch.memory.monitor.schema import Basis, Category

MB = 1 << 20


def _stats(total, free, evictable, slot_sizes=(MB,)):
    """Shaped like the nanobind StorageStatistics, derived props included."""
    obj = types.SimpleNamespace(
        total=total,
        free=free,
        evictable=evictable,
        slot_sizes=tuple(slot_sizes),
    )
    # Present so a test can prove the checks do not lean on them.
    obj.available = lambda: obj.free + obj.evictable
    obj.unavailable = lambda: obj.total - obj.available()
    return obj


class _Manager:
    """A real class, not a SimpleNamespace: the provider holds it weakly and
    SimpleNamespace forbids weak references."""

    def __init__(self, *stats):
        self._stats = list(stats)

    def _get_storage_statistics(self, level):
        return self._stats


def _manager(*stats):
    return _Manager(*stats)


class TestCheckCIsNotVacuous:
    """The reason this check exists at all."""

    def test_passes_on_a_consistent_pool(self):
        rows = kv_pool_group_stats(_manager(_stats(100, 30, 20)))
        assert all(r.passed for r in check_kv_pool_conservation(rows))

    def test_fails_when_free_and_evictable_exceed_total(self):
        # Double counting a block as both free and evictable produces exactly
        # this. If no input could make the check fail, it would be worthless.
        rows = kv_pool_group_stats(_manager(_stats(100, 70, 50)))
        results = check_kv_pool_conservation(rows)
        assert not results[0].passed
        assert "in_use=-20" in results[0].detail

    def test_boundary_is_allowed(self):
        rows = kv_pool_group_stats(_manager(_stats(100, 60, 40)))
        assert check_kv_pool_conservation(rows)[0].passed

    def test_does_not_read_the_derived_properties(self):
        # available()/unavailable() are computed from total/free/evictable,
        # so a check written against them passes by construction. Poison
        # them: a check that consults them would now disagree with itself.
        stats = _stats(100, 70, 50)
        stats.available = lambda: 0
        stats.unavailable = lambda: 100

        rows = kv_pool_group_stats(_manager(stats))
        assert not check_kv_pool_conservation(rows)[0].passed, (
            "the check must read the stored fields, not the derived ones"
        )

    def test_each_pool_group_checked_separately(self):
        rows = kv_pool_group_stats(_manager(_stats(100, 30, 20), _stats(100, 70, 50)))
        results = check_kv_pool_conservation(rows)
        assert [r.passed for r in results] == [True, False]


class TestCheckD:
    def test_rounding_above_quota_is_fine(self):
        rows = kv_pool_group_stats(_manager(_stats(102, 0, 0)))
        result = check_kv_quota(rows, quota_bytes=100 * MB)
        assert result.passed

    def test_capacity_below_quota_fails(self):
        # Pools round up, so landing under the quota means the sizing and
        # the allocation disagree.
        rows = kv_pool_group_stats(_manager(_stats(90, 0, 0)))
        assert not check_kv_quota(rows, quota_bytes=100 * MB).passed

    def test_slightly_below_quota_still_fails(self):
        # Within the tolerance in absolute terms, so a symmetric band would
        # wave it through. Rounding can only add, which makes any shortfall
        # a real disagreement however small.
        rows = kv_pool_group_stats(_manager(_stats(97, 0, 0)))
        result = check_kv_quota(rows, quota_bytes=100 * MB)
        assert not result.passed, (
            "the tolerance is one-sided: capacity under quota is never explained by rounding"
        )

    def test_far_above_quota_fails(self):
        rows = kv_pool_group_stats(_manager(_stats(200, 0, 0)))
        assert not check_kv_quota(rows, quota_bytes=100 * MB).passed

    def test_absent_quota_is_not_a_failure(self):
        rows = kv_pool_group_stats(_manager(_stats(100, 0, 0)))
        assert check_kv_quota(rows, quota_bytes=0).passed


class TestSlotsToBytes:
    def test_every_pool_in_the_group_contributes_a_slot(self):
        rows = kv_pool_group_stats(_manager(_stats(10, 2, 3, slot_sizes=(MB, 2 * MB))))
        assert rows[0]["slot_bytes"] == 3 * MB
        assert rows[0]["capacity_bytes"] == 30 * MB
        assert rows[0]["in_use_bytes"] == 15 * MB


class TestPhysicalVersusLogical:
    def test_capacity_is_physical_and_counts(self):
        manager = _manager(_stats(100, 60, 20))
        provider = KvCacheProvider(manager)
        physical = [e for e in provider.collect() if e.basis is Basis.PHYSICAL_BACKING]
        assert len(physical) == 1
        # The pool is resident in full; free blocks are not given back.
        assert physical[0].size_bytes == 100 * MB
        assert physical[0].counts_toward_total is True

    def test_in_use_is_logical_and_does_not_count(self):
        manager = _manager(_stats(100, 60, 20))
        provider = KvCacheProvider(manager)
        logical = [e for e in provider.collect() if e.basis is Basis.LOGICAL]
        assert len(logical) == 1
        assert logical[0].size_bytes == 20 * MB
        assert logical[0].counts_toward_total is False, (
            "in-use bytes live inside capacity_bytes; summing both would double count the pool"
        )

    def test_freeing_blocks_does_not_shrink_the_physical_figure(self):
        # The provider holds the manager weakly, so the caller keeps it
        # alive -- in production the ResourceManager owns it.
        busy_manager = _manager(_stats(100, 0, 0))
        idle_manager = _manager(_stats(100, 100, 0))
        busy = KvCacheProvider(busy_manager).collect()
        idle = KvCacheProvider(idle_manager).collect()

        def physical(entries):
            return next(e.size_bytes for e in entries if e.basis is Basis.PHYSICAL_BACKING)

        # The device has not taken anything back, so neither may the monitor.
        assert physical(busy) == physical(idle) == 100 * MB

    def test_category_and_owner(self):
        manager = _manager(_stats(8, 1, 1))
        entries = KvCacheProvider(manager).collect()
        assert all(e.category is Category.KV_CACHE_POOL for e in entries)
        assert "pool_group[0]" in entries[0].owner


class TestProviderAvailability:
    def test_unavailable_without_a_manager(self):
        provider = KvCacheProvider(None)
        assert provider.available() is False
        assert provider.collect() == []

    def test_unavailable_on_a_manager_without_the_accessor(self):
        # A v1 manager, say: reporting 0 would claim the cache is empty.
        provider = KvCacheProvider(types.SimpleNamespace())
        assert provider.available() is False
        assert provider.collect() == []

    def test_not_in_the_default_set(self):
        from tensorrt_llm._torch.memory.monitor.providers import default_providers

        assert not any(isinstance(p, KvCacheProvider) for p in default_providers()), (
            "the manager has no process-wide accessor, so it must be "
            "passed in rather than discovered"
        )
