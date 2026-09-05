"""Tests for KVCacheManagerV2 rank-aware auto host tier sizing.

The auto-provisioned host tier is computed per rank but drawn from a
node-level memory budget, so it must be divided by the number of ranks
co-located on the same physical node to avoid host OOM.

The per-rank computation reads rank-local host state, so co-scheduled ranks
can arrive at divergent quotas; ``TestSyncHostTierQuota`` covers the
cross-rank ``allreduce(MIN)`` that reconciles them to the fleet minimum.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _compute_auto_host_tier_quota,
    _sync_host_tier_quota,
)
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE

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


def _host_tier_sync_worker(per_rank_mem_gib):
    """Run on every MPI rank by the MpiPoolSession harness below.

    Each rank simulates reading a different amount of available host memory
    (as ``os.sysconf("SC_AVPHYS_PAGES")`` would differ across co-scheduled
    ranks), computes its own auto host-tier quota, then runs the real
    cross-rank sync. Returns this rank's pre- and post-sync quota so the parent
    process can assert on convergence.
    """
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        _compute_auto_host_tier_quota,
        _sync_host_tier_quota,
    )
    from tensorrt_llm._utils import mpi_rank, mpi_world_size
    from tensorrt_llm.mapping import Mapping

    rank = mpi_rank()
    world_size = mpi_world_size()

    # A large device quota so the (rank-local) available-memory budget is the
    # binding constraint, producing a different local quota on each rank.
    local_quota = _compute_auto_host_tier_quota(
        quota=1024 * GiB,
        local_ranks=1,
        mem_available=float(per_rank_mem_gib[rank] * GiB),
        memlock_limit=float("inf"),
    )
    mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
    synced_quota = _sync_host_tier_quota(local_quota, mapping)
    return {"rank": rank, "local_quota": local_quota, "synced_quota": synced_quota}


class TestSyncHostTierQuota:
    def test_single_rank_is_a_noop(self):
        # world_size == 1 must not touch the collective layer (no MPI needed);
        # the local quota is returned unchanged.
        class _FakeMapping:
            world_size = 1

        quota = 173 * GiB
        assert _sync_host_tier_quota(quota, _FakeMapping()) == quota

    @pytest.mark.cpu_only
    @pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device (MPI) build required")
    def test_multi_rank_syncs_to_fleet_min(self):
        """Every rank must end up with the same host-tier quota after the sync.

        Regression guard for the hang fixed in PR #17380 (TRTLLM-15179): when
        co-scheduled ranks auto-compute divergent host quotas, per-rank
        MAX_UTILIZATION schedulers disagree about which suspended requests can
        resume and wedge collectives on non-attention-DP TP. The
        ``allreduce(MIN)`` in ``_sync_host_tier_quota`` makes the
        most-constrained rank set the fleet value.
        """
        from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

        world_size = 2
        # rank 0 sees ~440 GiB available (-> 220 GiB local quota), rank 1 sees
        # ~880 GiB (-> 440 GiB). rank 0 is the most-constrained rank.
        per_rank_mem_gib = [440, 880]

        session = MpiPoolSession(n_workers=world_size)
        try:
            results = session.submit_sync(_host_tier_sync_worker, per_rank_mem_gib)
        finally:
            session.shutdown()

        results = sorted(results, key=lambda r: r["rank"])
        local_quotas = [r["local_quota"] for r in results]
        synced_quotas = [r["synced_quota"] for r in results]

        # Pre-sync the ranks genuinely disagreed (else the test proves nothing).
        assert local_quotas[0] != local_quotas[1], local_quotas
        # Post-sync every rank agrees...
        assert len(set(synced_quotas)) == 1, synced_quotas
        # ...on the global MIN (the most-constrained rank sets the fleet value).
        assert synced_quotas[0] == min(local_quotas)
        assert synced_quotas[0] == local_quotas[0]
