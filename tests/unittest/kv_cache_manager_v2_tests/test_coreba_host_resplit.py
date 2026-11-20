# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy-based rebalancing (CoReba): a host-only re-split keeps the hottest pages and suspends nobody.

When only the host arena's split changes (the cold target follows the average reused prefix length), the
group that shrinks copies its hottest pages outside the new window into the window, host to host
(Gemma-3-27B: 14-16 GiB per re-split). The copy runs on threads pinned to the NUMA node holding the
pages, so it takes tens of ms instead of 1.5 s (doc 40); it must still keep exactly the hottest
prefixes, with their data. And since no ACTIVE cache holds a host page, such an adjust() runs with the
requests active (adjust_needs_suspend is False).
"""

import importlib
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_coreba_arena as base  # noqa: E402

kvm = base.kvm
intro = importlib.import_module(kvm.__name__ + "._introspection")

HOST_LEVEL = 1
HOST_QUOTA = 96 << 20


class _Host(base._Arena):
    def host_windows(self):
        return intro.slot_windows(self.mgr, HOST_LEVEL)

    def host_used(self):
        return [s.total - s.free for s in self.mgr.get_storage_statistics(kvm.CACHE_LEVEL1)]

    def set_cold_target(self, target):
        """Host target only: the GPU target is pinned to the current split (the tuner may have moved it)."""
        intro.set_num_sampled_kv_caches(self.mgr, 1 << 20)
        intro.set_last_adjustment_time(self.mgr, 0.0)
        intro.set_target_ratio_list_gpu(self.mgr, intro.current_gpu_ratio(self.mgr))
        intro.set_target_ratio_list_cold(self.mgr, target)

    def set_hot_target(self, target):
        intro.set_num_sampled_kv_caches(self.mgr, 1 << 20)
        intro.set_last_adjustment_time(self.mgr, 0.0)
        intro.set_target_ratio_list_gpu(self.mgr, target)

    def surviving(self):
        """Seeds of cached prefixes still matched in full."""
        full = base.BLOCKS_PER_REQUEST * base.TOKENS_PER_BLOCK
        return {
            s for s in self.cached if self.mgr.probe_reuse(input_tokens=base._tokens(s)) >= full
        }


def _build(in_place=False, host_quota=HOST_QUOTA):
    return _Host(
        drop_unreferenced=False, ratio=[0.4, 0.6], in_place=in_place, host_quota=host_quota
    )


N_CACHED = 1300


def _fill(arena, n_cached=N_CACHED, n_live=20):
    # Many more cached prefixes than the GPU and the host hold: the oldest are offloaded and fill the host.
    for seed in range(1000, 1000 + n_cached):
        arena.add_request(seed, commit=True).close()
        arena.cached.append(seed)
    for seed in range(3000, 3000 + n_live):
        kv = arena.add_request(seed, commit=True)
        arena.live[seed] = (kv, base._tokens(seed))
    torch.cuda.synchronize()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestHostResplit(unittest.TestCase):
    def _resplit_with_live_requests(self, arena, target):
        """Host-only re-split with every live request ACTIVE; returns whether the windows moved."""
        before_idx = {s: arena.indices(kv) for s, (kv, _) in arena.live.items()}
        windows0 = arena.host_windows()
        arena.set_cold_target(target)
        self.assertTrue(arena.mgr.need_adjustment)
        self.assertFalse(
            arena.mgr.adjust_needs_suspend, "a host-only re-split must not need a suspend"
        )
        torch.cuda.synchronize()
        arena.mgr.adjust()  # requests stay ACTIVE
        torch.cuda.synchronize()
        self.assertEqual(
            arena.check_live(self, before_idx), 0, "a GPU block moved in a host-only re-split"
        )
        return arena.host_windows() != windows0

    def test_host_only_resplit_runs_with_active_requests(self):
        arena = _build()
        try:
            _fill(arena)
            self.assertGreater(sum(arena.host_used()), 0, "nothing was offloaded to the host tier")
            for target in ([0.8, 0.2], [0.15, 0.85]):
                self.assertTrue(
                    self._resplit_with_live_requests(arena, target), f"no move for {target}"
                )
            # Surviving cached prefixes still hold their data.
            self.assertGreater(arena.check_cached(self), 0, "no cached prefix survived")
        finally:
            arena.close()

    def test_gpu_target_needs_suspend(self):
        arena = _build()
        try:
            _fill(arena)
            arena.set_hot_target([0.8, 0.2])
            self.assertTrue(arena.mgr.need_adjustment)
            self.assertTrue(arena.mgr.adjust_needs_suspend)
            with self.assertRaises(Exception):
                arena.mgr.adjust()  # ACTIVE caches and a GPU resize: refused
            arena.rebalance([0.8, 0.2])  # suspended: fine
            arena.check_live(self)
        finally:
            arena.close()

    def test_in_place_never_needs_suspend(self):
        arena = _build(in_place=True)
        try:
            _fill(arena)
            arena.set_hot_target([0.8, 0.2])
            self.assertFalse(arena.mgr.adjust_needs_suspend)
        finally:
            arena.close()

    def test_resplit_keeps_hottest_prefixes_intact(self):
        """A host tier big enough that the re-split copies more than one copy thread's share (32 MiB)."""
        arena = _build(host_quota=384 << 20)
        n_cached = 1300
        try:
            _fill(arena, n_cached=n_cached)
            used = arena.host_used()
            totals = [s.total for s in arena.mgr.get_storage_statistics(kvm.CACHE_LEVEL1)]
            # Pool group 1 shrinks (target [0.8, 0.2]): it must be full for pages to lie outside its new window.
            self.assertGreater(used[1] / totals[1], 0.9, "shrinking host group not full")
            before = arena.surviving()
            self._resplit_with_live_requests(arena, [0.8, 0.2])
            after = arena.surviving()
            self.assertLess(len(after), len(before), "the re-split evicted nothing")
            # Kept pages are the hottest: the prefixes lost are older on average than those kept (cached prefixes
            # are added in seed order, oldest first).
            lost = sorted(before - after)
            kept = sorted(after)
            self.assertLess(sum(lost) / len(lost), sum(kept) / len(kept), f"lost {lost}")
            # Every kept prefix still holds its data after the multi-threaded host copy.
            self.assertGreater(arena.check_cached(self), 0)
        finally:
            arena.close()


if __name__ == "__main__":
    unittest.main()
