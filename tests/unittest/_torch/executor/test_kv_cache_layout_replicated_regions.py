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
"""Descriptor cost of storing a replicated KV role once (MiniMax-M3 index-K).

Every region of a page is a separate copy or RDMA descriptor, so the dedup of
index-K has to leave a page with about as many regions as it had before. These
tests build the real V2 storage configuration for M3's buffer set (61 layers: 3
dense FP8, 57 sparse NVFP4 with index-K, 1 shared draft layer) at TP=2 and TP=8,
aggregate it with V2's own ``get_aggregated_pages``, and count regions through
``build_kv_cache_layout_v2``. No GPU is needed: only V2's pure-Python layout
code runs, against a fake manager.

When torch or the compiled extensions are missing, the heavy imports are
replaced by stubs, so the file also runs on a login node.
"""

import importlib
import os
import sys
import types
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module
    return module


def _install_stubs_if_needed():
    try:
        import torch  # noqa: F401

        import tensorrt_llm  # noqa: F401
        import tensorrt_llm.runtime.kv_cache_manager_v2  # noqa: F401

        return
    except ImportError:
        pass
    for name in [m for m in sys.modules if m == "tensorrt_llm" or m.startswith("tensorrt_llm.")]:
        del sys.modules[name]
    src = os.path.join(_REPO_ROOT, "tensorrt_llm")
    _package("tensorrt_llm", src)
    _package("tensorrt_llm._torch", os.path.join(src, "_torch"))
    _package("tensorrt_llm._torch.pyexecutor", os.path.join(src, "_torch", "pyexecutor"))
    _package(
        "tensorrt_llm._torch.pyexecutor.connectors",
        os.path.join(src, "_torch", "pyexecutor", "connectors"),
    )
    _package(
        "tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
        os.path.join(src, "_torch", "pyexecutor", "connectors", "mooncake_store"),
    )
    _package("tensorrt_llm.runtime", os.path.join(src, "runtime"))
    for name in (
        "torch",
        "tensorrt_llm._utils",
        "tensorrt_llm.logger",
        "tensorrt_llm.bindings",
        "tensorrt_llm.bindings.internal",
        "tensorrt_llm.bindings.internal.batch_manager",
        "tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils",
        "tensorrt_llm.bindings.internal.runtime",
        "cuda",
        "cuda.bindings",
        "cuda.bindings.driver",
        "cuda.bindings.runtime",
        "cuda.bindings.nvrtc",
        "llist",
        "sortedcontainers",
        "pynvml",
        "psutil",
        "numpy",
        "cachetools",
    ):
        sys.modules.setdefault(name, mock.MagicMock())
    rawref = types.ModuleType("tensorrt_llm.runtime.kv_cache_manager_v2.rawref")
    rawref.NULL, rawref.ReferenceType, rawref.ref = None, weakref.ref, weakref.ref
    sys.modules[rawref.__name__] = rawref


_install_stubs_if_needed()

layout_mod = importlib.import_module("tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout")
addressing_mod = importlib.import_module(
    "tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing"
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (  # noqa: E402
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    KVCacheManagerConfig,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache_manager import (  # noqa: E402
    KVCacheManager,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._storage._config import (  # noqa: E402
    create_storage_config,
)

TOKENS_PER_BLOCK = 128
NUM_SPARSE, NUM_DENSE, NUM_DRAFT = 57, 3, 1
NUM_LAYERS = NUM_DENSE + NUM_SPARSE + NUM_DRAFT
TOTAL_KV_HEADS, HEAD_DIM = 4, 128
INDEX_K_BYTES = TOKENS_PER_BLOCK * 128  # fp8 index-K, one head of 128
GROUP_BYTES_PER_PAGE_TP2 = 3297280  # logged by the round-5 benchmark


def m3_layer_configs(tp: int, tail: bool = False):
    """Buffer set of the MiniMax-M3 hybrid NVFP4 cache, as V2 registers it.

    Per layer the order is K, V, K scale, V scale, index-K, matching
    ``KVCacheManagerV2`` plus ``_extra_buffers_per_layer``.
    """
    heads = max(1, TOTAL_KV_HEADS // tp)
    fp8 = heads * HEAD_DIM * TOKENS_PER_BLOCK
    layers = []
    for layer_id in range(NUM_LAYERS):
        sparse = NUM_DENSE <= layer_id < NUM_DENSE + NUM_SPARSE
        if sparse:
            data = fp8 // 2
            buffers = [
                BufferConfig("key", data),
                BufferConfig("value", data),
                BufferConfig("key_block_scale", fp8 // 16),
                BufferConfig("value_block_scale", fp8 // 16),
                BufferConfig("index_key", INDEX_K_BYTES, coalesce_last=tail),
            ]
        else:
            buffers = [BufferConfig("key", fp8), BufferConfig("value", fp8)]
        layers.append(AttentionLayerConfig(layer_id=layer_id, buffers=buffers))
    return layers


class _FakeStorage:
    """The slice of V2's storage that ``get_aggregated_pages`` reads."""

    def __init__(self, storage_config):
        self._buffer_attr = storage_config.buffer_attributes()
        self._life_cycle_grouping = storage_config.life_cycle_grouping()
        self._layer_to_life_cycle_ids = storage_config.layer_to_life_cycle_ids()
        self._slot_desc_list = storage_config.slot_desc_list
        self._levels = [SimpleNamespace(storage=self)]
        self.pool_size_granularity = 1 << 31

    def slot_size(self, pg_idx):
        return self._slot_desc_list[pg_idx].slot_size_list

    def slot_address(self, pg_idx, pool_idx, slot):
        return 0x7F0000000000 + (int(pg_idx) * 8 + int(pool_idx)) * (1 << 36)


class _FakeImpl:
    def __init__(self, layer_configs):
        config = KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[GpuCacheTierConfig(quota=1 << 40)],
            layers=layer_configs,
        )
        self.init_config = config
        self._storage = _FakeStorage(create_storage_config(config))

    # V2's own aggregation, run against the fake storage.
    get_aggregated_pages = KVCacheManager.get_aggregated_pages

    @property
    def layer_grouping(self):
        groups = {}
        for layer, life_cycle in self._storage._layer_to_life_cycle_ids.items():
            groups.setdefault(int(life_cycle), []).append(int(layer))
        return tuple(tuple(groups[i]) for i in sorted(groups))

    @property
    def all_buffer_ids(self):
        return iter(self._storage._buffer_attr.keys())

    @property
    def pool_group_descs(self):
        return [
            SimpleNamespace(
                num_slots=39606,
                slot_desc=pool_group,
            )
            for pool_group in self._storage._slot_desc_list
        ]


def build_layout(tp: int, tail: bool = False, replicated=("index_key",)):
    impl = _FakeImpl(m3_layer_configs(tp, tail))
    manager = SimpleNamespace(
        impl=impl,
        pp_layers=list(range(NUM_LAYERS)),
        tokens_per_block=TOKENS_PER_BLOCK,
        get_replicated_roles=lambda: frozenset(replicated),
    )
    return layout_mod.build_kv_cache_layout_v2(manager), impl


def region_count(group) -> int:
    return len(group.regions) + len(group.replicated_regions)


def baseline_region_count(tp: int) -> int:
    """Regions per page with no replicated role declared (the pre-dedup layout)."""
    layout, _ = build_layout(tp, replicated=())
    return len(layout.groups[0].regions)


class TestReplicatedRegionCost(unittest.TestCase):
    def test_baseline_matches_the_logged_tp2_layout(self):
        layout, _ = build_layout(2, replicated=())
        group = layout.groups[0]
        self.assertEqual(len(group.regions), 3)
        self.assertEqual(group.bytes_per_page, GROUP_BYTES_PER_PAGE_TP2)
        self.assertEqual(group.replicated_regions, ())

    def test_interleaved_tp2_falls_back_to_per_rank_pages(self):
        # Index-K sits between each layer's K/V, so splitting it out would
        # leave 59 + 57 regions. The guard must keep the baseline's 3.
        layout, _ = build_layout(2)
        group = layout.groups[0]
        self.assertEqual(group.replicated_regions, ())
        self.assertEqual(len(group.regions), 3)
        self.assertEqual(group.bytes_per_page, GROUP_BYTES_PER_PAGE_TP2)

    def test_unguarded_split_would_have_fragmented(self):
        # Pins the regression this guard prevents, using the layout helper
        # directly so the numbers stay the ones the benchmark logged.
        impl = _FakeImpl(m3_layer_configs(2))
        buffers = list(impl.all_buffer_ids)
        regions = layout_mod._regions_for
        sharded = regions(
            impl,
            [b for b in buffers if b.role != "index_key"],
            0,
            1,
            {i: i for i in range(NUM_LAYERS)},
        )
        replicated = regions(
            impl,
            [b for b in buffers if b.role == "index_key"],
            0,
            1,
            {i: i for i in range(NUM_LAYERS)},
        )
        self.assertEqual((len(sharded), len(replicated)), (59, 57))
        self.assertEqual({r.size for r in replicated}, {INDEX_K_BYTES})
        self.assertEqual(sum(r.size for r in replicated), 933888)

    def test_tail_layout_coalesces_replicated_into_one_region_tp2(self):
        layout, _ = build_layout(2, tail=True)
        group = layout.groups[0]
        self.assertEqual(len(group.replicated_regions), 1)
        self.assertEqual(group.replicated_regions[0].size, NUM_SPARSE * INDEX_K_BYTES)
        self.assertEqual(len(group.regions), 3)
        self.assertEqual(group.bytes_per_page + group.replicated_bytes_per_page, 3297280)
        self.assertEqual(group.replicated_bytes_per_page, 933888)

    def test_tail_layout_is_baseline_plus_one_region(self):
        for tp in (2, 8):
            with self.subTest(tp=tp):
                layout, _ = build_layout(tp, tail=True)
                self.assertEqual(region_count(layout.groups[0]), baseline_region_count(tp) + 1)

    def test_dedup_never_exceeds_the_allowed_extra_regions(self):
        limit = layout_mod.MAX_EXTRA_REGIONS_FOR_REPLICATION
        for tp in (1, 2, 4, 8):
            for tail in (False, True):
                with self.subTest(tp=tp, tail=tail):
                    layout, _ = build_layout(tp, tail=tail)
                    self.assertLessEqual(
                        region_count(layout.groups[0]), baseline_region_count(tp) + limit
                    )

    def test_tp8_keeps_dedup_in_the_existing_layout(self):
        # Index-K is one run between the dense K/V and the draft K/V, so the
        # split costs two extra regions and dedup stays on.
        layout, _ = build_layout(8)
        group = layout.groups[0]
        self.assertEqual(len(group.replicated_regions), 1)
        self.assertEqual(baseline_region_count(8), 3)
        self.assertEqual(region_count(group), 5)

    def test_split_preserves_total_bytes_and_buffers(self):
        for tp, tail in ((2, True), (8, False), (8, True), (2, False)):
            with self.subTest(tp=tp, tail=tail):
                layout, _ = build_layout(tp, tail=tail)
                base, _ = build_layout(tp, replicated=())
                group = layout.groups[0]
                self.assertEqual(
                    group.bytes_per_page + group.replicated_bytes_per_page,
                    base.groups[0].bytes_per_page,
                )
                refs = [
                    (b.layer_id, b.role)
                    for r in (*group.regions, *group.replicated_regions)
                    for b in r.buffers
                ]
                base_refs = [
                    (b.layer_id, b.role) for r in base.groups[0].regions for b in r.buffers
                ]
                self.assertEqual(sorted(refs), sorted(base_refs))

    def test_no_replicated_role_is_unchanged(self):
        layout, _ = build_layout(2, tail=True, replicated=())
        group = layout.groups[0]
        self.assertEqual(group.replicated_regions, ())
        self.assertEqual(group.bytes_per_page, GROUP_BYTES_PER_PAGE_TP2)


class TestPerPageTransfers(unittest.TestCase):
    """What the connector actually issues per page: keys and buffer descriptors."""

    @staticmethod
    def per_page(layout):
        addressing = addressing_mod.PageAddressing(layout)
        group = layout.groups[0].layer_group_id
        shard_buffers, _ = addressing.buffers(group, 7)
        replicated_buffers, _ = addressing.replicated_buffers(group, 7)
        keys = 1 + (1 if addressing.has_replicated(group) else 0)
        return keys, len(shard_buffers) + len(replicated_buffers)

    def test_descriptors_per_page_do_not_exceed_baseline_plus_one_region(self):
        for tp in (2, 8):
            baseline_layout, _ = build_layout(tp, replicated=())
            _, baseline = self.per_page(baseline_layout)
            for tail in (False, True):
                with self.subTest(tp=tp, tail=tail):
                    keys, descriptors = self.per_page(build_layout(tp, tail=tail)[0])
                    self.assertLessEqual(
                        descriptors, baseline + layout_mod.MAX_EXTRA_REGIONS_FOR_REPLICATION
                    )
                    self.assertLessEqual(keys, 2)
            _, descriptors = self.per_page(build_layout(tp, tail=True)[0])
            self.assertEqual(descriptors, baseline + 1)

    def test_tp2_default_layout_is_exactly_the_baseline(self):
        baseline = self.per_page(build_layout(2, replicated=())[0])
        self.assertEqual(self.per_page(build_layout(2)[0]), baseline)
        self.assertEqual(baseline, (1, 3))

    def test_the_unguarded_pr_layout_was_19x_the_descriptors(self):
        with mock.patch.object(layout_mod, "MAX_EXTRA_REGIONS_FOR_REPLICATION", 10**9):
            keys, descriptors = self.per_page(build_layout(2)[0])
        self.assertEqual((keys, descriptors), (2, 116))

    def test_replicated_and_shard_pages_do_not_overlap(self):
        layout, _ = build_layout(2, tail=True)
        addressing = addressing_mod.PageAddressing(layout)
        spans = []
        for fn in (addressing.buffers, addressing.replicated_buffers):
            addresses, sizes = fn(0, 11)
            spans += [(a, a + n) for a, n in zip(addresses, sizes)]
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start)
        self.assertEqual(sum(e - b for b, e in spans), GROUP_BYTES_PER_PAGE_TP2)


class TestCoalesceLast(unittest.TestCase):
    def test_default_order_is_unchanged(self):
        impl = _FakeImpl(m3_layer_configs(2))
        attrs = impl._storage._buffer_attr
        # Index-K sits right after the V of the same layer in the 16 KiB pool.
        self.assertEqual(attrs[(3, "index_key")].offset - attrs[(3, "value")].offset, INDEX_K_BYTES)
        self.assertEqual(attrs[(4, "key")].offset - attrs[(3, "key")].offset, 3 * INDEX_K_BYTES)

    def test_flagged_buffers_follow_all_others_and_keep_layer_order(self):
        impl = _FakeImpl(m3_layer_configs(2, tail=True))
        attrs = impl._storage._buffer_attr
        data = [
            a.offset
            for (lid, role), a in attrs.items()
            if role in ("key", "value") and a.pool_index == attrs[(3, "key")].pool_index
        ]
        index = [
            attrs[(lid, "index_key")].offset for lid in range(NUM_DENSE, NUM_DENSE + NUM_SPARSE)
        ]
        self.assertLess(max(data), min(index))
        self.assertEqual(index, sorted(index))
        self.assertEqual(index[1] - index[0], INDEX_K_BYTES)
        # K and V of a layer stay adjacent, which get_buffers() asserts.
        for lid in range(NUM_DENSE, NUM_DENSE + NUM_SPARSE):
            self.assertEqual(attrs[(lid, "value")].offset - attrs[(lid, "key")].offset, 8192 * 2)

    def test_pool_sizes_and_membership_are_unchanged(self):
        plain = _FakeImpl(m3_layer_configs(2))._storage
        tail = _FakeImpl(m3_layer_configs(2, tail=True))._storage
        self.assertEqual(
            {k: (a.pool_index, a.size) for k, a in plain._buffer_attr.items()},
            {k: (a.pool_index, a.size) for k, a in tail._buffer_attr.items()},
        )
        self.assertEqual(
            [list(s.slot_size_list) for s in plain._slot_desc_list],
            [list(s.slot_size_list) for s in tail._slot_desc_list],
        )

    def test_offsets_are_a_permutation_within_each_pool(self):
        plain = _FakeImpl(m3_layer_configs(2))._storage._buffer_attr
        tail = _FakeImpl(m3_layer_configs(2, tail=True))._storage._buffer_attr
        for pool in {a.pool_index for a in plain.values()}:
            self.assertEqual(
                sorted(a.offset for a in plain.values() if a.pool_index == pool),
                sorted(a.offset for a in tail.values() if a.pool_index == pool),
            )


class TestFoldReplicatedRegions(unittest.TestCase):
    """One attention shard has no second copy to save, so the split is undone."""

    def test_fold_restores_a_single_page_with_every_byte(self):
        layout, _ = build_layout(2, tail=True)
        folded = layout_mod.fold_replicated_regions(layout)
        (group,), (split,) = folded.groups, layout.groups
        self.assertEqual(group.replicated_regions, ())
        self.assertEqual(group.regions, (*split.regions, *split.replicated_regions))
        self.assertEqual(group.bytes_per_page, split.bytes_per_page + split.replicated_bytes_per_page)
        self.assertEqual(len(group.regions), 4)

    def test_fold_matches_the_unsplit_page_size(self):
        # The pre-PR page of the same pool: nothing declared replicated.
        unsplit, _ = build_layout(2, tail=True, replicated=())
        folded = layout_mod.fold_replicated_regions(build_layout(2, tail=True)[0])
        self.assertEqual(folded.groups[0].bytes_per_page, unsplit.groups[0].bytes_per_page)

    def test_fold_without_replicated_regions_is_the_same_layout(self):
        layout, _ = build_layout(2, tail=True, replicated=())
        self.assertIs(layout_mod.fold_replicated_regions(layout), layout)


class TestDescriptorTable(unittest.TestCase):
    """Prints the per-page descriptor table quoted in the README."""

    def test_print_table(self):
        rows = []
        for tp in (2, 8):
            base = baseline_region_count(tp)
            for label, tail, guard in (
                ("PR as shipped", False, 10**9),
                ("guard (default)", False, None),
                ("guard + pool tail", True, None),
            ):
                limit = guard or layout_mod.MAX_EXTRA_REGIONS_FOR_REPLICATION
                with mock.patch.object(layout_mod, "MAX_EXTRA_REGIONS_FOR_REPLICATION", limit):
                    layout, _ = build_layout(tp, tail=tail)
                group = layout.groups[0]
                rows.append(
                    f"TP={tp} {label:18s} baseline={base} shard={len(group.regions)} "
                    f"replicated={len(group.replicated_regions)} total={region_count(group)} "
                    f"keys/page={2 if group.replicated_regions else 1} "
                    f"bytes={group.bytes_per_page}+{group.replicated_bytes_per_page}"
                )
        print("\n" + "\n".join(rows))


if __name__ == "__main__":
    unittest.main()
