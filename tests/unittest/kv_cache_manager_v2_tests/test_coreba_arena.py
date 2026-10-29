# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy-based rebalancing (CoReba): GPU pools as windows of one arena.

Drives the C++ KVCacheManagerV2 through its bindings. Every block written gets a
known pattern; after each rebalance the test checks that

* all pool groups share one base pointer (the arena start),
* the windows tile the arena without overlap and follow the target ratio,
* every live block's index lies inside its group's window, holds its data, and
  kept its index unless the window moved away from it,
* cached (reusable) prefixes are still matched with intact data, or, with the
  drop flag, were dropped rather than copied.

Run fast mode (no full tensorrt_llm import):
    PYTHONPATH=$REPO_ROOT/tensorrt_llm/runtime python test_coreba_arena.py -v
"""

import importlib
import random
import unittest

import torch

try:
    import kv_cache_manager_v2 as kvm  # fast mode
except ImportError:
    from tensorrt_llm.runtime import kv_cache_manager_v2 as kvm

intro = importlib.import_module(kvm.__name__ + "._introspection")

TOKENS_PER_BLOCK = 32
BLOCKS_PER_REQUEST = 8
FULL_LAYER, SWA_LAYER = 0, 1
# Different buffer sizes keep the two lifecycles in separate pool groups.
FULL_BUF_BYTES = 16 << 10  # slot: K+V = 32 KiB
SWA_BUF_BYTES = 40 << 10  # slot: K+V = 80 KiB
SWA_WINDOW = 2 * BLOCKS_PER_REQUEST * TOKENS_PER_BLOCK  # wider than any sequence: no stale blocks
QUOTA = 128 << 20


class _DevicePtr:
    """Zero-copy int32 view of raw device memory for torch.as_tensor."""

    def __init__(self, addr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes // 4,),
            "typestr": "<i4",
            "data": (addr, False),
            "version": 3,
            "strides": None,
        }


def _tokens(seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 30000) for _ in range(BLOCKS_PER_REQUEST * TOKENS_PER_BLOCK)]


def _pattern(seed: int, ordinal: int, lc: int) -> int:
    return (seed * 1000 + ordinal) * 4 + lc + 1


class _Arena:
    """A manager plus the bookkeeping needed to check its arena."""

    def __init__(self, drop_unreferenced: bool, ratio: list[float]):
        torch.zeros(1, device="cuda")  # the manager needs a current CUDA context
        layers = [
            kvm.AttentionLayerConfig(
                layer_id=FULL_LAYER,
                buffers=[
                    kvm.BufferConfig(role="key", size=FULL_BUF_BYTES),
                    kvm.BufferConfig(role="value", size=FULL_BUF_BYTES),
                ],
            ),
            kvm.AttentionLayerConfig(
                layer_id=SWA_LAYER,
                buffers=[
                    kvm.BufferConfig(role="key", size=SWA_BUF_BYTES),
                    kvm.BufferConfig(role="value", size=SWA_BUF_BYTES),
                ],
                sliding_window_size=SWA_WINDOW,
            ),
        ]
        config = kvm.KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[kvm.GpuCacheTierConfig(quota=QUOTA)],
            layers=layers,
            initial_pool_ratio=ratio,
            enable_partial_reuse=False,
            pool_rebalance=kvm.PoolRebalanceConfig(
                copy_based_rebalancing=True, drop_unreferenced_on_rebalance=drop_unreferenced
            ),
        )
        self.mgr = kvm.KVCacheManager(config)
        self.stream = torch.cuda.current_stream().cuda_stream
        self.lcs = {
            layer: self.mgr.get_layer_group_id(layer) for layer in (FULL_LAYER, SWA_LAYER)
        }
        self.pg_of_lc = list(self.mgr.get_life_cycle_pool_group_indices())
        self.slot_bytes = {
            self.lcs[FULL_LAYER]: 2 * FULL_BUF_BYTES,
            self.lcs[SWA_LAYER]: 2 * SWA_BUF_BYTES,
        }
        self.live: dict[int, tuple] = {}  # seed -> (kv_cache, tokens)
        self.cached: list[int] = []  # seeds of closed, committed requests
        self.unverified_cached_blocks = 0  # matched but not admitted, so data not read back

    # ---- memory access -------------------------------------------------------------------

    def base(self, layer: int) -> int:
        return self.mgr.get_mem_pool_base_address(layer, "key", kvm.PageIndexMode.PER_LAYER)

    def slot_view(self, lc: int, slot: int) -> torch.Tensor:
        layer = FULL_LAYER if lc == self.lcs[FULL_LAYER] else SWA_LAYER
        nbytes = self.slot_bytes[lc]
        return torch.as_tensor(_DevicePtr(self.base(layer) + slot * nbytes, nbytes), device="cuda")

    def indices(self, kv) -> dict[int, list[int]]:
        # The binding returns an int32 numpy view; plain ints keep address arithmetic 64-bit.
        return {lc: [int(i) for i in kv.get_base_page_indices(lc)] for lc in self.lcs.values()}

    # ---- workload -------------------------------------------------------------------------

    def add_request(self, seed: int, commit: bool) -> object:
        tokens = _tokens(seed)
        kv = self.mgr.create_kv_cache(input_tokens=tokens, id=seed)
        assert kv.resume(self.stream), f"resume failed for request {seed}"
        assert kv.resize(len(tokens)), f"resize failed for request {seed}"
        for lc, idx in self.indices(kv).items():
            for ordinal, slot in enumerate(idx):
                if slot != kvm.BAD_PAGE_INDEX:
                    self.slot_view(lc, slot).fill_(_pattern(seed, ordinal, lc))
        torch.cuda.synchronize()
        if commit:
            kv.commit(tokens)
        return kv

    def windows(self) -> list[tuple[int, int]]:
        return intro.gpu_slot_windows(self.mgr)

    def suspend_all(self):
        torch.cuda.synchronize()
        for kv, _ in self.live.values():
            kv.suspend()

    def resume_all(self):
        for seed, (kv, _) in self.live.items():
            assert kv.resume(self.stream), f"resume failed for request {seed}"

    def rebalance(self, target_ratio: list[float]):
        """Force a rebalance to target_ratio (pool-group order) with every request suspended."""
        self.suspend_all()
        intro.set_num_sampled_kv_caches(self.mgr, 1 << 20)
        intro.set_last_adjustment_time(self.mgr, 0.0)
        intro.set_target_ratio_list_gpu(self.mgr, target_ratio)
        self.mgr.adjust()
        torch.cuda.synchronize()
        self.resume_all()

    # ---- checks ---------------------------------------------------------------------------

    def check_layout(self, tc: unittest.TestCase):
        tc.assertEqual(self.base(FULL_LAYER), self.base(SWA_LAYER), "pools must share the arena base")
        spans = []
        for lc, pg in enumerate(self.pg_of_lc):
            b, e = self.windows()[pg]
            spans.append((b * self.slot_bytes[lc], e * self.slot_bytes[lc]))
        spans.sort()
        tc.assertEqual(spans[0][0], 0)
        tc.assertLessEqual(spans[-1][1], QUOTA)
        tc.assertLessEqual(spans[0][1], spans[1][0], "windows overlap")

    def check_live(self, tc: unittest.TestCase, before: dict | None = None) -> int:
        """Check every live block; return how many changed index since `before`."""
        moved = 0
        windows = self.windows()
        for seed, (kv, _) in self.live.items():
            for lc, idx in self.indices(kv).items():
                b, e = windows[self.pg_of_lc[lc]]
                for ordinal, slot in enumerate(idx):
                    tc.assertTrue(b <= slot < e, f"req {seed} lc {lc} block {ordinal}: {slot} not in [{b}, {e})")
                    expected = _pattern(seed, ordinal, lc)
                    view = self.slot_view(lc, slot)
                    tc.assertTrue(
                        bool((view == expected).all()),
                        f"req {seed} lc {lc} block {ordinal} at slot {slot}: data corrupted",
                    )
                    if before is not None:
                        old = before[seed][lc][ordinal]
                        if old != slot:
                            moved += 1
                            tc.assertFalse(b <= old < e, f"block at {old} moved although still inside its window")
        return moved

    def check_cached(self, tc: unittest.TestCase) -> int:
        """Count the cached blocks still matched; reuse them and check their data where the pools
        have room to admit a request (resume refuses above max_util_for_resume)."""
        matched_blocks = 0
        for seed in self.cached:
            tokens = _tokens(seed)
            n = self.mgr.probe_reuse(input_tokens=tokens) // TOKENS_PER_BLOCK
            matched_blocks += n
            if n == 0:
                continue
            kv = self.mgr.create_kv_cache(input_tokens=tokens, id=10_000_000 + seed)
            try:
                if not kv.resume(self.stream):
                    self.unverified_cached_blocks += n
                    continue
                assert kv.resize(len(tokens))
                for lc, idx in self.indices(kv).items():
                    for ordinal in range(n):
                        view = self.slot_view(lc, idx[ordinal])
                        tc.assertTrue(
                            bool((view == _pattern(seed, ordinal, lc)).all()),
                            f"cached req {seed} lc {lc} block {ordinal}: data corrupted",
                        )
                torch.cuda.synchronize()
            finally:
                kv.close()
        return matched_blocks

    def close(self):
        torch.cuda.synchronize()
        for kv, _ in self.live.values():
            kv.close()
        self.live.clear()
        self.mgr.shutdown()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestCopyBasedRebalancing(unittest.TestCase):
    def _populate(self, arena: _Arena):
        # Cached prefixes first: committed, then closed, so their blocks stay in the tree.
        for seed in range(100, 120):
            arena.add_request(seed, commit=True).close()
            arena.cached.append(seed)
        # Then requests that free their blocks again, leaving holes behind the cached blocks.
        for seed in range(200, 220):
            arena.add_request(seed, commit=False).close()
        # Live requests last: their blocks sit nearest each group's moving edge.
        for seed in range(300, 360):
            kv = arena.add_request(seed, commit=True)
            arena.live[seed] = (kv, _tokens(seed))
        torch.cuda.synchronize()

    def _run(self, drop_unreferenced: bool):
        arena = _Arena(drop_unreferenced, ratio=[0.4, 0.6])
        try:
            self._populate(arena)
            arena.check_layout(self)
            arena.check_live(self)
            cached_before = arena.check_cached(self)
            self.assertEqual(cached_before, 20 * BLOCKS_PER_REQUEST)

            # Ratios are in pool-group order. Squeeze the SWA group, then the full-attention
            # group, then go back.
            pg_full = arena.pg_of_lc[arena.lcs[FULL_LAYER]]
            history = []
            for full_share in (0.6, 0.75, 0.25, 0.4):
                target = [0.0, 0.0]
                target[pg_full] = full_share
                target[1 - pg_full] = 1.0 - full_share
                before_windows = arena.windows()
                before = {seed: arena.indices(kv) for seed, (kv, _) in arena.live.items()}
                arena.rebalance(target)
                after_windows = arena.windows()
                self.assertNotEqual(before_windows, after_windows, f"rebalance to {target} did nothing")
                arena.check_layout(self)
                moved = arena.check_live(self, before)
                cached = arena.check_cached(self)
                history.append((full_share, before_windows, after_windows, moved, cached))
            return history
        finally:
            arena.close()

    def test_rebalance_copies_live_blocks(self):
        history = self._run(drop_unreferenced=False)
        for full_share, wb, wa, moved, cached in history:
            print(f"  full={full_share:.2f} windows {wb} -> {wa}: moved {moved} live blocks, {cached} cached blocks matched")
        self.assertGreater(sum(h[3] for h in history), 0, "no live block was ever copied")

    def _drop_scenario(self, drop_unreferenced: bool) -> tuple[int, int, list, list]:
        """Cached blocks next to the SWA group's moving edge, with free holes to copy them into.

        SWA ids are issued from the top of its window down, so allocation order maps to distance
        from the moving edge (its begin): early cached, placeholders, live, then late cached nearest
        the edge. Closing the placeholders leaves holes inside the part that is kept.
        """
        arena = _Arena(drop_unreferenced, ratio=[0.4, 0.6])
        try:
            for seed in range(100, 120):
                arena.add_request(seed, commit=True).close()
                arena.cached.append(seed)
            placeholders = [arena.add_request(seed, commit=False) for seed in range(200, 220)]
            for seed in range(300, 330):
                arena.live[seed] = (arena.add_request(seed, commit=True), _tokens(seed))
            for seed in range(400, 420):
                arena.add_request(seed, commit=True).close()
                arena.cached.append(seed)
            for kv in placeholders:
                kv.close()
            torch.cuda.synchronize()
            cached_before = arena.check_cached(self)

            pg_full = arena.pg_of_lc[arena.lcs[FULL_LAYER]]
            target = [0.0, 0.0]
            target[pg_full], target[1 - pg_full] = 0.62, 0.38
            windows_before = arena.windows()
            before = {seed: arena.indices(kv) for seed, (kv, _) in arena.live.items()}
            arena.rebalance(target)
            arena.check_layout(self)
            arena.check_live(self, before)
            cached_after = arena.check_cached(self)
            self.assertEqual(arena.unverified_cached_blocks, 0)
            return cached_before, cached_after, windows_before, arena.windows()
        finally:
            arena.close()

    def test_drop_unreferenced(self):
        before_k, after_k, wb, wa = self._drop_scenario(drop_unreferenced=False)
        before_d, after_d, _, _ = self._drop_scenario(drop_unreferenced=True)
        print(f"  windows {wb} -> {wa}")
        print(f"  cached blocks matched: copy {before_k} -> {after_k}, drop {before_d} -> {after_d}")
        self.assertEqual(before_k, 40 * BLOCKS_PER_REQUEST)
        # There was room to copy every cached block out of the region given up ...
        self.assertEqual(after_k, before_k, "without the drop flag, cached blocks with room to move must be copied")
        # ... but with the flag they are dropped instead.
        self.assertLess(after_d, before_d, "the drop flag did not drop the cached blocks that had to move")


if __name__ == "__main__":
    unittest.main()
