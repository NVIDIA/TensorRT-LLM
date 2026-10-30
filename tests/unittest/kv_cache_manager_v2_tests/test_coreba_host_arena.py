# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy-based rebalancing (CoReba) with a host tier: host pools as windows of one host arena.

With copy_based_rebalancing the host quota becomes one arena, registered in fixed units and never
resized; a rebalance moves the windows of the host pool groups instead of releasing one pool's
memory and committing another's. The test fills the GPU so that cached blocks are offloaded to the
host, forces host-level rebalances in both directions, and then admits every cached prefix again,
which onboards its host pages through the cold-page codec and checks their data.

TLLM_KV_CACHE_MANAGER_V2_HOST_ARENA_UNIT_BYTES=2 MiB makes the arena's registration unit smaller than the lcm of the
cold page sizes (32 KiB and 80 KiB), so about 3% of the 80 KiB pages straddle two registrations;
copy-engine copies of those must be split (addHostCopySplitRange). The variable is read when the
manager is built.
"""

import importlib
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TLLM_KV_CACHE_MANAGER_V2_HOST_ARENA_UNIT_BYTES"] = str(2 << 20)

import test_coreba_arena as base  # noqa: E402

kvm = base.kvm
intro = importlib.import_module(kvm.__name__ + "._introspection")

HOST_LEVEL = 1
HOST_QUOTA = 96 << 20  # 48 units of 2 MiB; holds ~100 requests of 896 KiB
UNIT = 2 << 20


class _HostArena(base._Arena):
    def host_windows(self) -> list[tuple[int, int]]:
        return intro.slot_windows(self.mgr, HOST_LEVEL)

    def host_stats(self):
        return self.mgr.get_storage_statistics(kvm.CACHE_LEVEL1)

    def rebalance_host(self, target_ratio: list[float]):
        """Force a host-level rebalance to target_ratio (cold pool-group order); the GPU target is left
        at the current ratio, so only the host level moves."""
        if self.in_place:
            torch.cuda.synchronize()
        else:
            self.suspend_all()
        intro.set_num_sampled_kv_caches(self.mgr, 1 << 20)
        intro.set_last_adjustment_time(self.mgr, 0.0)
        intro.set_target_ratio_list_cold(self.mgr, target_ratio)
        self.mgr.adjust()
        torch.cuda.synchronize()
        if not self.in_place:
            self.resume_all()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestCopyBasedRebalancingHostArena(unittest.TestCase):
    def _check_host_layout(self, arena: _HostArena, cold_bytes: list[int]):
        spans = sorted((b * s, e * s) for (b, e), s in zip(arena.host_windows(), cold_bytes))
        self.assertGreaterEqual(spans[0][0], 0)
        self.assertLessEqual(spans[-1][1], HOST_QUOTA)
        for (_, end), (begin, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, begin, "host windows overlap")

    def _run(self, in_place: bool):
        arena = _HostArena(drop_unreferenced=False, ratio=[0.4, 0.6], in_place=in_place, host_quota=HOST_QUOTA)
        try:
            # Many more cached prefixes than the GPU holds (~140 requests): the oldest are offloaded.
            for seed in range(1000, 1300):
                arena.add_request(seed, commit=True).close()
                arena.cached.append(seed)
            for seed in range(3000, 3020):
                kv = arena.add_request(seed, commit=True)
                arena.live[seed] = (kv, base._tokens(seed))
            torch.cuda.synchronize()

            stats = arena.host_stats()
            cold_bytes = [s.slot_sizes[0] for s in stats]
            self.assertEqual(sorted(cold_bytes), [2 * base.FULL_BUF_BYTES, 2 * base.SWA_BUF_BYTES])
            used_before = sum(s.total - s.free for s in stats)
            self.assertGreater(used_before, 0, "nothing was offloaded to the host tier")
            self._check_host_layout(arena, cold_bytes)
            windows0 = arena.host_windows()

            total_moves = 0
            for target in ([0.8, 0.2], [0.15, 0.85]):
                arena.rebalance_host(target)
                windows = arena.host_windows()
                self.assertNotEqual(windows, windows0, f"host windows did not move for target {target}")
                self._check_host_layout(arena, cold_bytes)
                # The windows follow the target byte ratio (within the arena's rounding).
                shares = [(e - b) * s / HOST_QUOTA for (b, e), s in zip(windows, cold_bytes)]
                for share, want in zip(shares, target):
                    self.assertAlmostEqual(share, want, delta=0.05)
                total_moves += 1
                windows0 = windows
                self.assertEqual(arena.check_live(self), 0, "a GPU block moved in a host-only rebalance")

            # Admitting every cached prefix again onboards its host pages: their data must be intact.
            matched = arena.check_cached(self)
            self.assertGreater(matched, 0, "no cached prefix survived the host rebalances")
            self.assertEqual(total_moves, 2)
        finally:
            arena.close()

    def test_host_rebalance_suspend(self):
        self._run(in_place=False)

    def test_host_rebalance_in_place(self):
        self._run(in_place=True)


if __name__ == "__main__":
    unittest.main()
