# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool rebalancing keeps one max_seq_len request schedulable (KVCacheManager.raise_hot_min_slots).

The construction constraints only floor each pool group at one fresh warmup chunk. A tuner target
can then shrink a group below one long request, and the V2 scheduler deadlocks once a request
needs more than the group holds. ``raise_hot_min_slots`` raises the floors to one request at its last
prefill chunk -- max_seq_len blocks for full attention, window + chunk for a sliding window --
capped at the current slot counts. Checked here for VMM rebalancing (adjust()): a target that starves
a group stops at the raised floor, and without the raise it goes below it.

Run (the repo root must be first on PYTHONPATH, or a stale installed wheel is imported):
    PYTHONPATH=$REPO_ROOT python test_rebalance_min_floor.py -v
"""

import importlib
import math
import unittest

import torch

from tensorrt_llm.runtime import kv_cache_manager_v2 as kvm

intro = importlib.import_module(kvm.__name__ + "._introspection")

TPB = 32
WINDOW = 1024
CHUNK = 2048  # max_num_tokens
MAX_SEQ_LEN = 8192
MAX_UTIL = 0.95
KV_BYTES = (
    64 << 10
)  # per block per K or V per layer (VMM resizes in 2 MiB granules: keep them a few slots)
NUM_LAYERS = 6  # layer 5 full attention, layers 0-4 sliding window (Gemma-3 pattern)
FULL_LAYER, SWA_LAYER = 5, 0
QUOTA = 256 << 20

# One request at its last prefill chunk (what the executor passes after clamping max_seq_len).
FULL_NEED = MAX_SEQ_LEN // TPB  # 256 blocks
SWA_NEED = (WINDOW + CHUNK) // TPB  # 96 blocks (aligned chunk boundary)


def _floor_request(max_seq_len: int = MAX_SEQ_LEN) -> kvm.BatchDesc:
    return kvm.BatchDesc(
        [kvm.KVCacheDesc(capacity=max_seq_len, history_length=max(0, max_seq_len - CHUNK))]
    )


class _Manager:
    def __init__(self):
        torch.zeros(1, device="cuda")  # the manager needs a current CUDA context
        layers = [
            kvm.AttentionLayerConfig(
                layer_id=i,
                buffers=[kvm.BufferConfig(role=r, size=KV_BYTES) for r in ("key", "value")],
                sliding_window_size=None if i == FULL_LAYER else WINDOW,
            )
            for i in range(NUM_LAYERS)
        ]
        config = kvm.KVCacheManagerConfig(
            tokens_per_block=TPB,
            cache_tiers=[kvm.GpuCacheTierConfig(quota=QUOTA)],
            layers=layers,
            max_util_for_resume=MAX_UTIL,
            # The executor's warmup constraint: one fresh context chunk.
            constraints=[kvm.BatchDesc([kvm.KVCacheDesc(capacity=CHUNK, history_length=0)])],
            pool_rebalance=kvm.PoolRebalanceConfig(),
        )
        self.mgr = kvm.KVCacheManager(config)
        pg_of_lc = list(self.mgr.get_life_cycle_pool_group_indices())
        self.full_pg = pg_of_lc[self.mgr.get_layer_group_id(FULL_LAYER)]
        self.swa_pg = pg_of_lc[self.mgr.get_layer_group_id(SWA_LAYER)]
        assert self.full_pg != self.swa_pg

    def slots(self) -> dict[int, int]:
        return {d.pool_group_index: d.num_slots for d in self.mgr.pool_group_descs}

    def set_starving_target(self, pg: int):
        """Set a target that gives `pg` almost nothing, with the sample gate and cooldown passed."""
        target = [0.999] * len(self.slots())
        target[pg] = 0.001
        total = sum(target)
        intro.set_num_sampled_kv_caches(self.mgr, 1 << 20)
        intro.set_last_adjustment_time(self.mgr, 0.0)
        intro.set_target_ratio_list_gpu(self.mgr, [t / total for t in target])

    def starve(self, pg: int):
        """Force a rebalance to a target that gives `pg` almost nothing."""
        self.set_starving_target(pg)
        self.mgr.adjust()
        torch.cuda.synchronize()


class RebalanceMinFloorTest(unittest.TestCase):
    def test_floor_values(self):
        m = _Manager()
        before = m.slots()
        floors = m.mgr.raise_hot_min_slots(_floor_request())
        self.assertEqual(floors[m.full_pg], min(math.ceil(FULL_NEED / MAX_UTIL), before[m.full_pg]))
        # Window + chunk, plus at most one block for a chunk boundary inside a block.
        self.assertGreaterEqual(floors[m.swa_pg], math.ceil(SWA_NEED / MAX_UTIL))
        self.assertLessEqual(floors[m.swa_pg], math.ceil((SWA_NEED + 1) / MAX_UTIL))
        self.assertEqual(m.slots(), before, "raising the floors must not move anything")

    def test_capped_at_current_and_never_lowered(self):
        m = _Manager()
        before = m.slots()
        floors = m.mgr.raise_hot_min_slots(_floor_request(1 << 20))  # far beyond the pool
        self.assertEqual(floors[m.full_pg], before[m.full_pg])
        lower = m.mgr.raise_hot_min_slots(_floor_request(4 * CHUNK))
        self.assertEqual(lower, floors, "floors must never drop")

    def test_vmm_rebalance_respects_floor(self):
        for pg_name in ("full_pg", "swa_pg"):
            with self.subTest(starved=pg_name):
                # Control: without the raise, the starving target goes below one request.
                control = _Manager()
                pg = getattr(control, pg_name)
                need = FULL_NEED if pg_name == "full_pg" else SWA_NEED
                control.starve(pg)
                self.assertLess(
                    control.slots()[pg], need, "control should shrink below one request"
                )

                m = _Manager()
                floors = m.mgr.raise_hot_min_slots(_floor_request())
                m.starve(pg)
                self.assertGreaterEqual(m.slots()[pg], floors[pg])
                self.assertGreaterEqual(m.slots()[pg], need)

    def test_unreachable_target_is_attempted_once_per_recompute(self):
        """One adjust() per target, even when the pools cannot reach it.

        The raised floor stops the starving target here, so the split stays outside the tolerance
        after adjust(). need_adjustment must still be false until the target is recomputed or set
        again, with the sample gate and cooldown passed; otherwise the executor calls adjust() on
        every check (cooldown_secs 0) until the target moves.
        """
        m = _Manager()
        m.mgr.raise_hot_min_slots(_floor_request())
        m.set_starving_target(m.full_pg)
        self.assertTrue(m.mgr.need_adjustment)
        m.mgr.adjust()
        torch.cuda.synchronize()
        reached = m.slots()
        intro.set_last_adjustment_time(m.mgr, 0.0)
        self.assertFalse(m.mgr.need_adjustment, "an unreachable target must not be retried")
        # Setting the target again (as a recompute does) allows exactly one more attempt.
        m.set_starving_target(m.full_pg)
        self.assertTrue(m.mgr.need_adjustment)
        m.mgr.adjust()
        torch.cuda.synchronize()
        self.assertEqual(m.slots(), reached, "the floor still stops the target")
        intro.set_last_adjustment_time(m.mgr, 0.0)
        self.assertFalse(m.mgr.need_adjustment)


if __name__ == "__main__":
    unittest.main()
