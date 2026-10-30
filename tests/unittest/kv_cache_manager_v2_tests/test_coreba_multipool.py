# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy-based rebalancing (CoReba) with several pools in a pool group.

Each layer here has buffers of two sizes, so each pool group gets two pools. Pool 0 uses the
group's slot ids (the base page indices); the other pool has its own ids, bound to the group's
(``get_page_index_bindings``), so that each pool's window moves on its own. Every buffer of
every block written gets a known pattern, addressed the way a kernel would address it::

    base(layer, role) + converter(bindings[base_page_index]) * page_stride

After each rebalance the test checks that

* every pool's window lies inside the arena without overlapping another's, and the pools of a
  group have one slot count,
* every live block's base page index lies in its group's window, its bound-pool id in that
  pool's window, and every buffer still holds its data,
* cached (reusable) prefixes are still matched with intact data in every buffer.

With a host tier (test_host_tier_*), blocks are offloaded to the host and onboarded again through
the cold-page codec, which then gets one page-index array per pool: a bound pool's hot index is its
own id, not the group's.

Run (the repo root must be first on PYTHONPATH, or a stale installed wheel is imported):
    PYTHONPATH=$REPO_ROOT python test_coreba_multipool.py -v
"""

import importlib
import os
import random
import unittest

import numpy as np
import torch

from tensorrt_llm.runtime import kv_cache_manager_v2 as kvm

intro = importlib.import_module(kvm.__name__ + "._introspection")

TOKENS_PER_BLOCK = 32
BLOCKS_PER_REQUEST = 8
FULL_LAYER, SWA_LAYER = 0, 1
# K and V coalesce into one pool; the third buffer, of another size, gets a pool of its own.
BUFFERS = {
    FULL_LAYER: {"key": 16 << 10, "value": 16 << 10, "idx": 4 << 10},
    SWA_LAYER: {"key": 40 << 10, "value": 40 << 10, "state": 12 << 10},
}
SWA_WINDOW = 2 * BLOCKS_PER_REQUEST * TOKENS_PER_BLOCK  # wider than any sequence: no stale blocks
QUOTA = 128 << 20
PER_LAYER = kvm.PageIndexMode.PER_LAYER


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


def _pattern(seed: int, ordinal: int, layer: int, role: str) -> int:
    return ((seed * 1000 + ordinal) * 4 + layer) * 4 + sorted(BUFFERS[layer]).index(role) + 1


class _Arena:
    """A manager plus the bookkeeping needed to check its arena."""

    def __init__(self, ratio: list[float], in_place: bool = False, host_quota: int = 0):
        torch.zeros(1, device="cuda")  # the manager needs a current CUDA context
        self.in_place = in_place
        layers = [
            kvm.AttentionLayerConfig(
                layer_id=layer,
                buffers=[kvm.BufferConfig(role=role, size=size) for role, size in buffers.items()],
                sliding_window_size=SWA_WINDOW if layer == SWA_LAYER else None,
            )
            for layer, buffers in BUFFERS.items()
        ]
        config = kvm.KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[kvm.GpuCacheTierConfig(quota=QUOTA)]
            + ([kvm.HostCacheTierConfig(quota=host_quota)] if host_quota else []),
            layers=layers,
            initial_pool_ratio=ratio,
            enable_partial_reuse=False,
            pool_rebalance=kvm.PoolRebalanceConfig(
                copy_based_rebalancing=True, in_place_relocation=in_place
            ),
        )
        self.mgr = kvm.KVCacheManager(config)
        self.stream = torch.cuda.current_stream().cuda_stream
        self.lcs = {layer: self.mgr.get_layer_group_id(layer) for layer in BUFFERS}
        self.pg_of_lc = list(self.mgr.get_life_cycle_pool_group_indices())
        self.pool_bytes = [[p.slot_bytes for p in g.pools] for g in self.mgr.pool_group_descs]
        self.live: dict[int, tuple] = {}  # seed -> (kv_cache, tokens)
        self.cached: list[int] = []  # seeds of closed, committed requests
        self.unverified_cached_blocks = 0  # matched, but resume() refused (pools near full)
        self.verified_cached_blocks = 0
        # Caller-owned page-index buffers, bound as the PyTorch flow binds them. seed -> {lc: array}
        self.bufs: dict[int, dict[int, np.ndarray]] = {}
        self._bindings: dict[tuple[int, str], list[int] | None] = {}
        self._binding_version = None

    # ---- memory access -------------------------------------------------------------------

    def bindings(self, layer: int, role: str) -> list[int] | None:
        version = self.mgr.get_page_index_binding_version()
        if version != self._binding_version:
            self._bindings.clear()
            self._binding_version = version
        key = (layer, role)
        if key not in self._bindings:
            self._bindings[key] = self.mgr.get_page_index_bindings(layer, role)
        return self._bindings[key]

    def buffer_view(self, layer: int, role: str, base_index: int) -> torch.Tensor:
        """The bytes a kernel reads for this buffer of the block at base_index."""
        bindings = self.bindings(layer, role)
        index = bindings[base_index] if bindings is not None else base_index
        assert index >= 0, f"layer {layer} {role}: base index {base_index} is not bound"
        page = self.mgr.get_page_index_converter(layer, role)([index], PER_LAYER)[0]
        stride = self.mgr.get_page_stride(layer, role)
        addr = self.mgr.get_mem_pool_base_address(layer, role, PER_LAYER) + page * stride
        return torch.as_tensor(_DevicePtr(addr, stride), device="cuda")

    def indices(self, kv) -> dict[int, list[int]]:
        return {lc: [int(i) for i in kv.get_base_page_indices(lc)] for lc in self.lcs.values()}

    def write_block(self, seed: int, ordinal: int, layer: int, base_index: int):
        for role in BUFFERS[layer]:
            self.buffer_view(layer, role, base_index).fill_(_pattern(seed, ordinal, layer, role))

    def block_intact(self, seed: int, ordinal: int, layer: int, base_index: int) -> str | None:
        for role in BUFFERS[layer]:
            if not bool((self.buffer_view(layer, role, base_index) == _pattern(seed, ordinal, layer, role)).all()):
                return role
        return None

    # ---- workload -------------------------------------------------------------------------

    def add_request(self, seed: int, commit: bool) -> object:
        tokens = _tokens(seed)
        kv = self.mgr.create_kv_cache(input_tokens=tokens, id=seed)
        self.bufs[seed] = {lc: np.full(BLOCKS_PER_REQUEST, -2, dtype=np.int32) for lc in self.lcs.values()}
        self.bind_bufs(seed, kv)
        assert kv.resume(self.stream), f"resume failed for request {seed}"
        assert kv.resize(len(tokens)), f"resize failed for request {seed}"
        for layer, lc in self.lcs.items():
            for ordinal, slot in enumerate(self.indices(kv)[lc]):
                if slot != kvm.BAD_PAGE_INDEX:
                    self.write_block(seed, ordinal, layer, slot)
        torch.cuda.synchronize()
        if commit:
            kv.commit(tokens)
        return kv

    def bind_bufs(self, seed: int, kv):
        for lc, buf in self.bufs[seed].items():
            kv.set_base_page_index_buf(0, lc, memoryview(buf))

    def rebalance(self, target_ratio: list[float]):
        """Force a rebalance to target_ratio (pool-group order): live requests are suspended
        around adjust(), or stay active with in-place relocation."""
        torch.cuda.synchronize()  # the executor's quiescence point
        if not self.in_place:
            for kv, _ in self.live.values():
                kv.suspend()
        intro.set_num_sampled_kv_caches(self.mgr, 1 << 20)
        intro.set_last_adjustment_time(self.mgr, 0.0)
        intro.set_target_ratio_list_gpu(self.mgr, target_ratio)
        self.mgr.adjust()
        torch.cuda.synchronize()
        if not self.in_place:
            for seed, (kv, _) in self.live.items():
                assert kv.resume(self.stream), f"resume failed for request {seed}"
                self.bind_bufs(seed, kv)  # suspend() unbinds them, as in the PyTorch flow

    # ---- checks ---------------------------------------------------------------------------

    def check_layout(self, tc: unittest.TestCase):
        spans = []
        for pg, pools in enumerate(intro.gpu_pool_windows(self.mgr)):
            tc.assertEqual(len({e - b for b, e in pools}), 1, f"group {pg}: pools differ in slot count")
            for pool, (b, e) in enumerate(pools):
                s = self.pool_bytes[pg][pool]
                spans.append((b * s, e * s, pg, pool))
        spans.sort()
        tc.assertGreaterEqual(spans[0][0], 0)
        tc.assertLessEqual(spans[-1][1], QUOTA)
        for (_, end, *a), (begin, _, *b) in zip(spans, spans[1:]):
            tc.assertLessEqual(end, begin, f"pools {a} and {b} overlap")
        # Every slot id in a group's window is bound, to distinct ids inside its pool's window.
        windows = intro.gpu_slot_windows(self.mgr)
        pool_windows = intro.gpu_pool_windows(self.mgr)
        for layer, lc in self.lcs.items():
            pg = self.pg_of_lc[lc]
            for role in BUFFERS[layer]:
                bindings = self.bindings(layer, role)
                if bindings is None:
                    continue
                b, e = windows[pg]
                ids = [bindings[s] for s in range(b, e)]
                tc.assertEqual(len(set(ids)), len(ids), f"layer {layer} {role}: duplicate bindings")
                own = [w for w in pool_windows[pg] if w[0] <= ids[0] < w[1]]
                tc.assertTrue(own and all(own[0][0] <= i < own[0][1] for i in ids),
                              f"layer {layer} {role}: bound outside its pool's window")

    def page_range(self, layer: int, role: str) -> tuple[int, int]:
        """get_page_index_range, shifted for a PER_LAYER view as the DeepSeek-V4 cache manager does."""
        first, last = self.mgr.get_page_index_range(layer, role)
        converter = self.mgr.get_page_index_converter(layer, role)
        if converter.layer_offset is not None:
            first += converter.layer_offset * converter.expansion
            last += converter.layer_offset * converter.expansion
        return first, last

    def check_live(self, tc: unittest.TestCase) -> None:
        windows = intro.gpu_slot_windows(self.mgr)
        pool_windows = intro.gpu_pool_windows(self.mgr)
        # The page range a scan of the pool window covers holds every live page, and no more than the window.
        for layer, lc in self.lcs.items():
            pg = self.pg_of_lc[lc]
            for role in BUFFERS[layer]:
                first, last = self.page_range(layer, role)
                tc.assertLessEqual(last, self.mgr.get_page_index_upper_bound(layer, role))
                pool = 0 if self.bindings(layer, role) is None else 1  # two pools per group; the second is bound
                b, e = pool_windows[pg][pool]
                per_slot = self.pool_bytes[pg][pool] // self.mgr.get_page_stride(layer, role)
                tc.assertLessEqual(last - first, (e - b) * per_slot, f"layer {layer} {role}: range wider than the window")
                converter = self.mgr.get_page_index_converter(layer, role)
                for seed, (kv, _) in self.live.items():
                    for slot in self.indices(kv)[lc]:
                        bindings = self.bindings(layer, role)
                        index = bindings[slot] if bindings is not None else slot
                        page = converter([index], PER_LAYER)[0]
                        tc.assertTrue(first <= page < last,
                                      f"layer {layer} {role}: page {page} outside the scanned range [{first}, {last})")
        for seed, (kv, _) in self.live.items():
            for layer, lc in self.lcs.items():
                idx = self.indices(kv)[lc]
                tc.assertEqual(self.bufs[seed][lc].tolist(), idx[:BLOCKS_PER_REQUEST],
                               f"req {seed} lc {lc}: bound page-index buffer is stale")
                b, e = windows[self.pg_of_lc[lc]]
                for ordinal, slot in enumerate(idx):
                    tc.assertTrue(b <= slot < e, f"req {seed} lc {lc} block {ordinal}: {slot} not in [{b}, {e})")
                    bad = self.block_intact(seed, ordinal, layer, slot)
                    tc.assertIsNone(bad, f"req {seed} layer {layer} block {ordinal} at {slot}: {bad} corrupted")

    def check_cached(self, tc: unittest.TestCase) -> int:
        """Count the cached blocks still matched; reuse them and check every buffer."""
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
                for layer, lc in self.lcs.items():
                    idx = self.indices(kv)[lc]
                    for ordinal in range(n):
                        bad = self.block_intact(seed, ordinal, layer, idx[ordinal])
                        tc.assertIsNone(bad, f"cached req {seed} layer {layer} block {ordinal}: {bad} corrupted")
                self.verified_cached_blocks += n
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
class TestCopyBasedRebalancingMultiPool(unittest.TestCase):
    def _run(self, in_place: bool):
        arena = _Arena(ratio=[0.4, 0.6], in_place=in_place)
        try:
            self.assertTrue(all(len(pools) == 2 for pools in arena.pool_bytes), arena.pool_bytes)
            for seed in range(100, 120):
                arena.add_request(seed, commit=True).close()
                arena.cached.append(seed)
            for seed in range(200, 220):
                arena.add_request(seed, commit=False).close()
            for seed in range(300, 360):
                arena.live[seed] = (arena.add_request(seed, commit=True), _tokens(seed))
            torch.cuda.synchronize()
            arena.check_layout(self)
            arena.check_live(self)
            self.assertEqual(arena.check_cached(self), 20 * BLOCKS_PER_REQUEST)

            pg_full = arena.pg_of_lc[arena.lcs[FULL_LAYER]]
            history = []
            for full_share in (0.6, 0.75, 0.25, 0.4):
                target = [0.0, 0.0]
                target[pg_full] = full_share
                target[1 - pg_full] = 1.0 - full_share
                before = intro.gpu_pool_windows(arena.mgr)
                arena.rebalance(target)
                after = intro.gpu_pool_windows(arena.mgr)
                self.assertNotEqual(before, after, f"rebalance to {target} did nothing")
                arena.check_layout(self)
                arena.check_live(self)
                cached = arena.check_cached(self)
                history.append((full_share, after, cached))
                print(f"  full={full_share:.2f}: pool windows {after}, {cached} cached blocks matched")
            print(f"  cached blocks read back: {arena.verified_cached_blocks}, "
                  f"matched but not admitted: {arena.unverified_cached_blocks}")
            self.assertGreater(arena.verified_cached_blocks, 20 * BLOCKS_PER_REQUEST)
            return history
        finally:
            arena.close()

    def test_rebalance_suspended(self):
        self._run(in_place=False)

    def test_rebalance_in_place(self):
        self._run(in_place=True)

    def _run_host_tier(self, in_place: bool):
        # 2 MiB host arena units: the cold pages (36 KiB and 92 KiB) straddle unit boundaries too.
        os.environ["TLLM_KV_CACHE_MANAGER_V2_HOST_ARENA_UNIT_BYTES"] = str(2 << 20)
        arena = _Arena(ratio=[0.4, 0.6], in_place=in_place, host_quota=96 << 20)
        try:
            # Far more cached prefixes than the GPU holds: the oldest are offloaded to the host.
            for seed in range(1000, 1300):
                arena.add_request(seed, commit=True).close()
                arena.cached.append(seed)
            for seed in range(3000, 3020):
                arena.live[seed] = (arena.add_request(seed, commit=True), _tokens(seed))
            torch.cuda.synchronize()
            host_used = sum(s.total - s.free for s in arena.mgr.get_storage_statistics(kvm.CACHE_LEVEL1))
            self.assertGreater(host_used, 0, "nothing was offloaded to the host tier")

            # GPU rebalances move the bound pools (their bindings change), then one host rebalance.
            pg_full = arena.pg_of_lc[arena.lcs[FULL_LAYER]]
            for full_share in (0.7, 0.3):
                target = [0.0, 0.0]
                target[pg_full] = full_share
                target[1 - pg_full] = 1.0 - full_share
                before = intro.gpu_pool_windows(arena.mgr)
                arena.rebalance(target)
                self.assertNotEqual(before, intro.gpu_pool_windows(arena.mgr))
                arena.check_layout(self)
                arena.check_live(self)
            host_before = intro.slot_windows(arena.mgr, 1)
            torch.cuda.synchronize()
            if not in_place:
                for kv, _ in arena.live.values():
                    kv.suspend()
            intro.set_last_adjustment_time(arena.mgr, 0.0)
            intro.set_target_ratio_list_cold(arena.mgr, [0.8, 0.2])
            arena.mgr.adjust()
            torch.cuda.synchronize()
            if not in_place:
                for seed, (kv, _) in arena.live.items():
                    assert kv.resume(arena.stream)
                    arena.bind_bufs(seed, kv)
            self.assertNotEqual(host_before, intro.slot_windows(arena.mgr, 1), "host windows did not move")
            arena.check_live(self)

            # Onboarding the offloaded prefixes decodes into every pool, bound pools at their own ids.
            arena.mgr.get_and_reset_iteration_stats()
            matched = arena.check_cached(self)
            onboarded = sum(d.iter_onboard_blocks for d in arena.mgr.get_and_reset_iteration_stats().values())
            print(f"  host used {host_used}, matched {matched}, read back {arena.verified_cached_blocks}, "
                  f"onboarded {onboarded}, not admitted {arena.unverified_cached_blocks}")
            self.assertGreater(onboarded, 0, "no block was onboarded from the host tier")
            self.assertGreater(arena.verified_cached_blocks, 0)
        finally:
            arena.close()

    def test_host_tier_suspended(self):
        self._run_host_tier(in_place=False)

    def test_host_tier_in_place(self):
        self._run_host_tier(in_place=True)


if __name__ == "__main__":
    unittest.main()
