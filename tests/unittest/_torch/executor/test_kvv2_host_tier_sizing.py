"""Tests for KVCacheManagerV2 rank-aware auto host tier sizing.

The auto-provisioned host tier is computed per rank but drawn from a
node-level memory budget, so it must be divided by the number of ranks
co-located on the same physical node to avoid host OOM.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import _compute_auto_host_tier_quota

GiB = 1 << 30


class TestComputeAutoHostTierQuota:
    def test_single_rank_uses_device_quota_when_memory_is_ample(self):
        # 1 rank, 440 GiB available: cap = 220 GiB > quota -> quota wins.
        assert (
            _compute_auto_host_tier_quota(
                quota=173 * GiB,
                local_ranks=1,
                mem_available=float(440 * GiB),
                memlock_limit=float("inf"),
            )
            == 173 * GiB
        )

    def test_colocated_ranks_divide_node_memory_budget(self):
        # 4 co-located ranks, 440 GiB available: each gets 440/4*0.5 = 55 GiB
        # instead of the full device quota (4 x 173 GiB would exceed node RAM).
        assert _compute_auto_host_tier_quota(
            quota=173 * GiB,
            local_ranks=4,
            mem_available=float(440 * GiB),
            memlock_limit=float("inf"),
        ) == int(440 * GiB / 4 * 0.5)

    def test_aggregate_across_ranks_stays_within_available_memory(self):
        local_ranks = 4
        mem_available = float(440 * GiB)
        per_rank = _compute_auto_host_tier_quota(
            quota=173 * GiB,
            local_ranks=local_ranks,
            mem_available=mem_available,
            memlock_limit=float("inf"),
        )
        assert per_rank * local_ranks <= mem_available

    def test_memlock_limit_caps_quota(self):
        assert _compute_auto_host_tier_quota(
            quota=173 * GiB,
            local_ranks=1,
            mem_available=float("inf"),
            memlock_limit=float(10 * GiB),
        ) == int(10 * GiB * 0.8)

    def test_unknown_limits_fall_back_to_device_quota(self):
        assert (
            _compute_auto_host_tier_quota(
                quota=173 * GiB,
                local_ranks=8,
                mem_available=float("inf"),
                memlock_limit=float("inf"),
            )
            == 173 * GiB
        )

    @pytest.mark.parametrize("memlock_limit", [0.0, float(1)])
    def test_non_positive_result_falls_back_to_device_quota(self, memlock_limit):
        # RLIMIT_MEMLOCK of 0 (common in restricted containers) would yield a
        # zero quota; a zero host tier would deadlock the MAX_UTILIZATION
        # scheduler's suspend/resume path, so fall back to the device quota.
        assert (
            _compute_auto_host_tier_quota(
                quota=173 * GiB,
                local_ranks=4,
                mem_available=float(440 * GiB),
                memlock_limit=memlock_limit,
            )
            == 173 * GiB
        )

    def test_result_is_always_positive(self):
        # Exhausted node memory reading must not produce a non-positive tier.
        assert (
            _compute_auto_host_tier_quota(
                quota=173 * GiB,
                local_ranks=4,
                mem_available=0.0,
                memlock_limit=float("inf"),
            )
            > 0
        )
