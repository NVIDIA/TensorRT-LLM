# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the KV cache layout description handed to a KV connector under
# KVCacheManagerV2 (``connectors/kv_cache_layout.py``).
#
# The layout replaces the single-pool-tensor registration used by the V1
# manager, whose memory V2 cannot express: V2 has one slot address space per
# pool and one page-index space per layer group.
#
# Tests in TestKvCacheRegionArithmetic need no GPU. The rest construct a real
# KVCacheManagerV2 and therefore allocate device memory pools.

import gc
import unittest

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
    build_kv_cache_layout_v2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
from tensorrt_llm.mapping import Mapping

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


def _make_kwargs(
    *,
    num_layers: int = 4,
    num_kv_heads=4,
    head_dim=128,
    tokens_per_block: int = 8,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_tokens: int = 2048,
    dtype=DataType.HALF,
    kv_cache_type=CacheType.SELF,
    vocab_size: int = 32000,
    kv_cache_config=None,
):
    return dict(
        kv_cache_config=kv_cache_config
        or KvCacheConfigV2(max_tokens=max_tokens, enable_block_reuse=False),
        kv_cache_type=kv_cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=dtype,
        vocab_size=vocab_size,
    )


class TestKvCacheRegionArithmetic(unittest.TestCase):
    """Address arithmetic and lookup helpers. No GPU required."""

    def _region(self, base=4096, size=256, stride=1024, num_slots=8):
        return KvCacheRegion(
            base=base,
            size=size,
            stride=stride,
            num_slots=num_slots,
            buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
        )

    def test_address_of_follows_stride(self):
        region = self._region()
        self.assertEqual(region.address_of(0), 4096)
        self.assertEqual(region.address_of(1), 4096 + 1024)
        self.assertEqual(region.address_of(7), 4096 + 7 * 1024)

    def test_address_of_rejects_out_of_range_slot(self):
        region = self._region(num_slots=8)
        for bad in (-1, 8, 99):
            with self.assertRaises(IndexError):
                region.address_of(bad)

    def test_as_tensor_rejects_dtype_not_dividing_extent(self):
        # size/stride are byte counts; a dtype whose itemsize does not divide
        # them cannot produce a correct view, so it must fail loudly rather
        # than silently truncate.
        region = self._region(size=6, stride=6)
        with self.assertRaises(ValueError):
            region.as_tensor(dtype=torch.float32)

    def test_bytes_per_page_sums_regions(self):
        group = KvCacheLayerGroupLayout(
            layer_group_id=0,
            layer_ids=(0, 1),
            window_size=None,
            regions=(self._region(size=256), self._region(base=8192, size=128)),
        )
        self.assertEqual(group.bytes_per_page, 384)

    def test_layout_lookup_by_group_and_layer(self):
        group_a = KvCacheLayerGroupLayout(0, (0, 2), None, ())
        group_b = KvCacheLayerGroupLayout(1, (1, 3), 128, ())
        layout = KvCacheLayout(tokens_per_block=8, groups=(group_a, group_b))

        self.assertIs(layout.group(1), group_b)
        self.assertIs(layout.group_of_layer(2), group_a)
        self.assertIs(layout.group_of_layer(3), group_b)
        with self.assertRaises(KeyError):
            layout.group(7)
        with self.assertRaises(KeyError):
            layout.group_of_layer(99)


class TestKvCacheRegionAliasing(unittest.TestCase):
    """as_tensor must alias the exact bytes address_of names."""

    def setUp(self):
        torch.cuda.init()

    def test_as_tensor_aliases_strided_slots(self):
        # Lay out 4 "slots" of 32 bytes each, and describe the middle 8 bytes
        # of every slot as a region. Writing through the view must land at
        # base + stride * i, and must not disturb neighbouring bytes.
        num_slots, stride, offset, size = 4, 32, 8, 8
        backing = torch.zeros(num_slots * stride, dtype=torch.uint8, device="cuda")

        region = KvCacheRegion(
            base=backing.data_ptr() + offset,
            size=size,
            stride=stride,
            num_slots=num_slots,
            buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
        )
        view = region.as_tensor()
        self.assertEqual(tuple(view.shape), (num_slots, size))

        for slot in range(num_slots):
            view[slot] = slot + 1

        flat = backing.cpu()
        for slot in range(num_slots):
            start = slot * stride
            self.assertTrue(
                bool((flat[start + offset : start + offset + size] == slot + 1).all()),
                f"slot {slot} payload not written at the address address_of() names",
            )
            # Bytes outside the described range must be untouched.
            self.assertTrue(bool((flat[start : start + offset] == 0).all()))
            self.assertTrue(bool((flat[start + offset + size : start + stride] == 0).all()))


class TestBuildKvCacheLayoutV2(unittest.TestCase):
    """The builder against a real KVCacheManagerV2."""

    def setUp(self):
        torch.cuda.init()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_layout_covers_every_layer_exactly_once(self):
        num_layers = 4
        mgr = KVCacheManagerV2(**_make_kwargs(num_layers=num_layers))
        try:
            layout = build_kv_cache_layout_v2(mgr)

            self.assertEqual(layout.tokens_per_block, mgr.tokens_per_block)
            self.assertTrue(layout.groups, "layout must describe at least one layer group")

            covered = [lid for group in layout.groups for lid in group.layer_ids]
            self.assertCountEqual(
                covered,
                list(mgr.pp_layers),
                "every local layer must appear in exactly one layer group",
            )

            # Every layer that owns storage must be reachable through a region.
            in_regions = [
                ref.layer_id
                for group in layout.groups
                for region in group.regions
                for ref in region.buffers
            ]
            self.assertCountEqual(set(in_regions), set(covered))
        finally:
            mgr.shutdown()
            del mgr

    def test_regions_are_disjoint_and_inside_the_slot(self):
        mgr = KVCacheManagerV2(**_make_kwargs())
        try:
            layout = build_kv_cache_layout_v2(mgr)
            pool_groups = list(mgr.impl.pool_group_descs)
            self.assertEqual(len(pool_groups), 1, "test config should yield one pool group")
            pool_group = pool_groups[0]
            self.assertEqual(len(pool_group.pools), 1, "test config should yield one pool")
            pool = pool_group.pools[0]

            for group in layout.groups:
                spans = []
                for region in group.regions:
                    self.assertEqual(region.num_slots, int(pool_group.num_slots))
                    self.assertEqual(region.stride, int(pool.slot_bytes))

                    offset = region.base - int(pool.base_address)
                    self.assertGreaterEqual(offset, 0)
                    self.assertLessEqual(
                        offset + region.size,
                        int(pool.slot_bytes),
                        "a region must lie inside one slot",
                    )
                    spans.append((offset, offset + region.size))

                spans.sort()
                for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
                    self.assertLessEqual(prev_end, next_start, "regions must not overlap")
        finally:
            mgr.shutdown()
            del mgr

    def test_region_addresses_agree_with_pool_descriptor(self):
        # Cross-check: region base/stride come from get_aggregated_pages, while
        # pool base_address/slot_bytes come from pool_group_descs. These are
        # independent public APIs and must agree on where slot i lives.
        mgr = KVCacheManagerV2(**_make_kwargs())
        try:
            layout = build_kv_cache_layout_v2(mgr)
            pool = list(mgr.impl.pool_group_descs)[0].pools[0]
            pool_base, slot_bytes = int(pool.base_address), int(pool.slot_bytes)

            for group in layout.groups:
                for region in group.regions:
                    offset = region.base - pool_base
                    for slot in (0, 1, region.num_slots - 1):
                        self.assertEqual(
                            region.address_of(slot),
                            pool_base + slot_bytes * slot + offset,
                        )
        finally:
            mgr.shutdown()
            del mgr

    def test_uniform_model_yields_one_full_slot_region(self):
        # With uniform K/V sizes every buffer in a layer group is adjacent, so
        # coalescing should collapse them into a single region spanning the
        # whole slot -- the efficient whole-page transfer, derived rather than
        # assumed.
        mgr = KVCacheManagerV2(**_make_kwargs(num_layers=4))
        try:
            layout = build_kv_cache_layout_v2(mgr)
            pool = list(mgr.impl.pool_group_descs)[0].pools[0]

            self.assertEqual(len(layout.groups), 1)
            group = layout.groups[0]
            self.assertEqual(len(group.regions), 1)
            region = group.regions[0]
            self.assertEqual(region.size, int(pool.slot_bytes))
            self.assertEqual(region.base, int(pool.base_address))
            # 4 layers * (K, V)
            self.assertEqual(len(region.buffers), 8)
            self.assertEqual(
                [ref.role for ref in region.buffers],
                ["key", "value"] * 4,
            )
        finally:
            mgr.shutdown()
            del mgr

    def test_full_attention_reports_no_window(self):
        mgr = KVCacheManagerV2(**_make_kwargs())
        try:
            layout = build_kv_cache_layout_v2(mgr)
            for group in layout.groups:
                self.assertIsNone(group.window_size)
        finally:
            mgr.shutdown()
            del mgr

    def test_mla_layout_has_no_value_buffers(self):
        # SELFKONLY carries a single compressed latent per token. Nothing in the
        # layout counts K against V, so this must simply come out as a layout
        # with no "value" role rather than needing a kv-factor special case.
        mgr = KVCacheManagerV2(**_make_kwargs(kv_cache_type=CacheType.SELFKONLY))
        try:
            layout = build_kv_cache_layout_v2(mgr)
            roles = {
                ref.role
                for group in layout.groups
                for region in group.regions
                for ref in region.buffers
            }
            self.assertIn("key", roles)
            self.assertNotIn("value", roles)
        finally:
            mgr.shutdown()
            del mgr


if __name__ == "__main__":
    unittest.main()
