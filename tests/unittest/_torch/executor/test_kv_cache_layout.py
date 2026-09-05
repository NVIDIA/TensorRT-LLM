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
import unittest.mock

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_layout import (
    KvCacheBufferRef,
    KvCacheLayerGroupLayout,
    KvCacheLayout,
    KvCacheRegion,
    build_kv_cache_layout_v2,
    valid_page_slots,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

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


def _make_request(request_id: int = 1, prompt_len: int = 120):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
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

    def test_slot_tensor_rejects_out_of_range_slot(self):
        # The check happens before any address is formed, so an invalid page
        # index cannot reach device memory. -1 is the one that matters:
        # BAD_PAGE_INDEX marks a block with no page, and as a subscript it
        # names the pool's last slot.
        region = self._region(num_slots=8)
        for bad in (BAD_PAGE_INDEX, -1, -2, 8, 99):
            with self.assertRaises(IndexError):
                region.slot_tensor(bad)

    def test_slot_tensor_rejects_dtype_not_dividing_extent(self):
        region = self._region(size=6, stride=6)
        with self.assertRaises(ValueError):
            region.slot_tensor(0, dtype=torch.float32)

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

    def _pool_layout(
        self,
        num_layers=2,
        kv_factor=2,
        block_bytes=64,
        num_slots=8,
        expansion=1,
        group_layer_ids=None,
    ):
        """A layout shaped the way a single-pool cache reports itself.

        Buffers are laid out layer-major and ascending, the way V2's storage
        config builds a coalesced buffer. ``group_layer_ids`` sets the group's
        membership list, which the manager reports in ``impl.layer_grouping``
        order rather than memory order.
        """
        layer_ids = tuple(range(num_layers))
        order = [lid for lid in layer_ids for _ in range(kv_factor)]
        buffers = tuple(
            KvCacheBufferRef(
                layer_id=lid, role="key" if i % kv_factor == 0 else "value", expansion=expansion
            )
            for i, lid in enumerate(order)
        )
        region = KvCacheRegion(
            base=4096,
            size=block_bytes * len(buffers),
            stride=4096,
            num_slots=num_slots,
            buffers=buffers,
        )
        group = KvCacheLayerGroupLayout(
            0, layer_ids if group_layer_ids is None else tuple(group_layer_ids), None, (region,)
        )
        return KvCacheLayout(tokens_per_block=8, groups=(group,), dtype=torch.float16)

    def test_single_pool_view_has_the_v1_pool_shape(self):
        # The shape `register_kv_caches` has always received, so a connector
        # written against it needs no changes when the same cache reports
        # itself as a layout instead.
        layout = self._pool_layout(num_layers=3, kv_factor=2, block_bytes=64, num_slots=5)
        backing = torch.zeros(5 * 4096 // 2, dtype=torch.float16)
        region = layout.groups[0].regions[0]
        flat = backing.as_strided((5, 3 * 2 * 32), (4096 // 2, 1))
        with unittest.mock.patch.object(KvCacheRegion, "as_tensor", return_value=flat):
            view = layout.as_single_pool_tensor()
        self.assertIsNotNone(view)
        # block_bytes=64 at float16 is 32 elements per (layer, role).
        self.assertEqual(tuple(view.shape), (5, 3, 2, 32))
        self.assertEqual(region.size, 6 * 64)

    def test_single_pool_view_ignores_the_layer_grouping_order(self):
        # `layer_ids` is a membership list carrying `impl.layer_grouping`'s
        # order, which comes off an unordered_map: Qwen3-0.6B reports layers
        # 27..13 then 0..12 for a cache whose buffers run 0..27. The view is
        # defined by the buffers alone, so consulting `layer_ids` for order must
        # not creep back in -- it once did, and sent every V2 connector run
        # without a `register_kv_cache_layout` override to NotImplementedError.
        layout = self._pool_layout(
            num_layers=3, kv_factor=2, block_bytes=64, num_slots=5, group_layer_ids=(2, 0, 1)
        )
        backing = torch.zeros(5 * 4096 // 2, dtype=torch.float16)
        flat = backing.as_strided((5, 3 * 2 * 32), (4096 // 2, 1))
        with unittest.mock.patch.object(KvCacheRegion, "as_tensor", return_value=flat):
            view = layout.as_single_pool_tensor()
        self.assertIsNotNone(view)
        self.assertEqual(tuple(view.shape), (5, 3, 2, 32))

    def test_single_pool_view_declines_what_it_cannot_describe(self):
        # Each of these is a real cache shape; the point is that the default
        # registration path refuses rather than mislabelling the bytes.
        two_groups = KvCacheLayout(
            tokens_per_block=8,
            groups=(
                KvCacheLayerGroupLayout(0, (0,), None, ()),
                KvCacheLayerGroupLayout(1, (1,), 128, ()),
            ),
        )
        self.assertIsNone(two_groups.as_single_pool_tensor())

        region = self._pool_layout().groups[0].regions[0]
        two_regions = KvCacheLayout(
            tokens_per_block=8, groups=(KvCacheLayerGroupLayout(0, (0, 1), None, (region, region)),)
        )
        self.assertIsNone(two_regions.as_single_pool_tensor())

        # Buffer order is deliberately not a decline case. V2 lays a coalesced
        # buffer out layer-major and ascending when the storage config walks
        # `config.layers`, so dimension 1 is layer-indexed the way V1's single
        # pool is and there is nothing here to re-derive.

        # A page expansion factor breaks the uniform grid.
        self.assertIsNone(self._pool_layout(expansion=2).as_single_pool_tensor())

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


class TestValidPageSlots(unittest.TestCase):
    """The filter a connector builds transfer targets through. No GPU required."""

    def test_preserves_the_original_ordinal(self):
        # The ordinal is what maps a page back to its token range, so the
        # filter must report the position in the input list, not a position
        # in its own output. Compacting here would silently re-point every
        # surviving block at an earlier token range.
        page_indices = [7, BAD_PAGE_INDEX, 9, BAD_PAGE_INDEX, BAD_PAGE_INDEX, 3]

        self.assertEqual(list(valid_page_slots(page_indices)), [(0, 7), (2, 9), (5, 3)])

    def test_drops_every_entry_without_a_page(self):
        page_indices = [BAD_PAGE_INDEX, 4, BAD_PAGE_INDEX, 0, BAD_PAGE_INDEX]

        yielded = list(valid_page_slots(page_indices))

        self.assertEqual([slot for _, slot in yielded], [4, 0])
        self.assertTrue(
            all(slot >= 0 for _, slot in yielded),
            "an entry that addresses no page survived the filter",
        )

    def test_slot_zero_is_a_page_and_survives(self):
        # `if slot:` instead of `if slot >= 0:` would drop page slot 0, which is
        # a real page and usually the first one a fresh pool hands out.
        self.assertEqual(list(valid_page_slots([0])), [(0, 0)])

    def test_a_list_with_no_pages_yields_nothing(self):
        self.assertEqual(list(valid_page_slots([BAD_PAGE_INDEX] * 4)), [])
        self.assertEqual(list(valid_page_slots([])), [])


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

    def test_slot_tensor_aliases_the_row_as_tensor_names(self):
        # Same bytes, not a copy: a write through one form is visible through
        # the other, and the pointer is the one address_of names.
        num_slots, stride, offset, size = 4, 32, 8, 8
        backing = torch.zeros(num_slots * stride, dtype=torch.uint8, device="cuda")
        region = KvCacheRegion(
            base=backing.data_ptr() + offset,
            size=size,
            stride=stride,
            num_slots=num_slots,
            buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
        )

        for slot in range(num_slots):
            region.slot_tensor(slot).fill_(slot + 1)

        view = region.as_tensor()
        for slot in range(num_slots):
            self.assertEqual(region.slot_tensor(slot).data_ptr(), region.address_of(slot))
            self.assertTrue(
                bool((view[slot] == slot + 1).all()),
                f"slot_tensor({slot}) did not write the bytes as_tensor()[{slot}] reads",
            )

    def test_guarded_transfer_never_reaches_the_page_bad_page_index_names(self):
        """The corruption the guarded path prevents, asserted from both sides.

        A page-index list reports ``BAD_PAGE_INDEX`` for a block with no page.
        Handed to a strided view that entry is a legal subscript naming the
        pool's *last* slot -- a live page holding another request's KV. This
        test first shows that hazard is live for the unguarded form, so the
        guarded assertions that follow are not vacuous.
        """
        num_slots, stride, size = 6, 16, 16
        backing = torch.zeros(num_slots * stride, dtype=torch.uint8, device="cuda")
        region = KvCacheRegion(
            base=backing.data_ptr(),
            size=size,
            stride=stride,
            num_slots=num_slots,
            buffers=(KvCacheBufferRef(layer_id=0, role="key"),),
        )

        # A distinct payload per page slot; the last one is what -1 names.
        view = region.as_tensor()
        for slot in range(num_slots):
            view[slot] = 10 + slot
        poison = 10 + num_slots - 1

        # What a connector is handed for a sequence whose window has moved on:
        # ordinals 0..2 hold no page, 3..5 hold pages 0..2.
        page_indices = [BAD_PAGE_INDEX, BAD_PAGE_INDEX, BAD_PAGE_INDEX, 0, 1, 2]

        # The hazard, live. Without this the guarded assertion proves nothing.
        unguarded = [int(view[slot][0].item()) for slot in page_indices]
        self.assertEqual(
            unguarded[:3],
            [poison] * 3,
            "indexing the raw list no longer reads the pool's last page, so the "
            "guarded assertions below would pass for the wrong reason",
        )

        # Addressing the same entry through the guard raises instead.
        with self.assertRaises(IndexError):
            region.slot_tensor(page_indices[0])

        # And the recommended shape never forms that address at all.
        saved = [
            (ordinal, int(region.slot_tensor(slot)[0].item()))
            for ordinal, slot in valid_page_slots(page_indices)
        ]
        self.assertEqual(
            saved,
            [(3, 10), (4, 11), (5, 12)],
            "the guarded loop must transfer exactly the blocks that hold a page, "
            "under their original ordinals",
        )
        self.assertTrue(
            all(payload != poison for _, payload in saved),
            f"a transfer read the page BAD_PAGE_INDEX names: {saved}",
        )


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


class TestBuildKvCacheLayoutV2Vswa(unittest.TestCase):
    """The builder against a real KVCacheManagerV2 with more than one window.

    This is the case the single-tensor registration cannot describe, and the
    case every per-layer-group callback exists for. The single-group tests
    above cannot catch a builder that mixes up which pool backs which group,
    because there is only one pool to get right.
    """

    #: Two distinct windows over 4 layers: the cyclic assignment in
    #: ``KVCacheManagerV2`` gives layers 0, 2 the sliding window and layers
    #: 1, 3 full attention (``max_attention_window`` entries equal to
    #: ``max_seq_len`` normalize to ``None``).
    WINDOW = 64
    MAX_SEQ_LEN = 256
    NUM_LAYERS = 4

    def setUp(self):
        torch.cuda.init()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _vswa_manager(self):
        return KVCacheManagerV2(
            **_make_kwargs(
                num_layers=self.NUM_LAYERS,
                max_seq_len=self.MAX_SEQ_LEN,
                kv_cache_config=KvCacheConfigV2(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=[self.WINDOW, self.MAX_SEQ_LEN],
                ),
            )
        )

    def _expected_window(self, layer_id: int):
        return None if layer_id % 2 else self.WINDOW

    def test_vswa_yields_one_group_per_window(self):
        mgr = self._vswa_manager()
        try:
            layout = build_kv_cache_layout_v2(mgr)

            self.assertEqual(
                len(layout.groups),
                2,
                "two distinct windows must produce two layer groups, or every "
                "assertion below is a single-group test in disguise",
            )
            self.assertEqual(
                {group.window_size for group in layout.groups},
                {self.WINDOW, None},
                "one group must be the sliding window and one full attention",
            )

            for group in layout.groups:
                for layer_id in group.layer_ids:
                    self.assertEqual(
                        group.window_size,
                        self._expected_window(layer_id),
                        f"layer {layer_id} landed in a group whose window is {group.window_size}",
                    )

            covered = [lid for group in layout.groups for lid in group.layer_ids]
            self.assertCountEqual(covered, list(mgr.pp_layers))
        finally:
            mgr.shutdown()
            del mgr

    def test_group_of_layer_resolves_to_that_layer_s_window(self):
        # The per-layer connector hooks receive a model layer index and have
        # nothing else to route on. If this mapping is wrong, a sliding layer
        # waits on the full-attention group's pages.
        mgr = self._vswa_manager()
        try:
            layout = build_kv_cache_layout_v2(mgr)
            for layer_id in mgr.pp_layers:
                group = layout.group_of_layer(int(layer_id))
                self.assertEqual(group.window_size, self._expected_window(int(layer_id)))
                self.assertIn(int(layer_id), group.layer_ids)
        finally:
            mgr.shutdown()
            del mgr

    def test_a_page_index_means_nothing_without_its_layer_group(self):
        """Why the flat block-id list cannot exist under VSWA.

        When two layer groups hold buffers of different sizes they are backed by
        different pool groups, each with its own slot address space starting at
        zero. Both groups then report page index 0, and the two index 0s are
        different pools. A connector handed one flat list could not tell them
        apart, and would write one group's KV over the other's.
        """
        # Heterogeneous KV head counts make the per-layer buffers differ in
        # size, which is what splits the two windows across two pool groups.
        mgr = KVCacheManagerV2(
            **_make_kwargs(
                num_layers=self.NUM_LAYERS,
                num_kv_heads=[4, 8, 4, 8],
                max_seq_len=self.MAX_SEQ_LEN,
                kv_cache_config=KvCacheConfigV2(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=[self.WINDOW, self.MAX_SEQ_LEN],
                ),
            )
        )
        try:
            layout = build_kv_cache_layout_v2(mgr)
            self.assertEqual(len(layout.groups), 2)

            pool_groups = list(mgr.impl.pool_group_descs)
            self.assertEqual(
                len(pool_groups),
                2,
                "layers of differing size must land in separate pool groups, or "
                "this test is the shared-slot-space case instead",
            )

            bases = {
                group.layer_group_id: {region.base for region in group.regions}
                for group in layout.groups
            }
            self.assertTrue(
                bases[0].isdisjoint(bases[1]),
                f"the two layer groups report the same pool base: {bases}",
            )

            # The same page index in both groups, addressing different memory,
            # is the whole reason the callbacks are per layer group.
            for slot in (0, 1):
                addresses = {
                    layout.group(group_id).regions[0].address_of(slot) for group_id in (0, 1)
                }
                self.assertEqual(
                    len(addresses),
                    2,
                    f"page index {slot} resolves to one address across two "
                    f"layer groups: {addresses}",
                )
        finally:
            mgr.shutdown()
            del mgr

    def test_filter_agrees_with_the_manager_valid_only_filter(self):
        """One definition of "holds a page", applied on both sides.

        The cache drops entries at the source with ``valid_only=True``. A
        connector cannot, because its list stays aligned to block ordinals, so
        it filters downstream with ``valid_page_slots`` instead. The two must
        select the same pages, or a connector is filtering against a different
        rule than the cache allocates by.
        """
        mgr = self._vswa_manager()
        try:
            prompt_len, tokens_per_block = 120, 8
            request = _make_request(prompt_len=prompt_len)
            self.assertTrue(mgr.prepare_context(request))
            mgr.resize_context(request, request.context_remaining_length)
            batch = ScheduledRequests()
            batch.append_context_request(request)
            mgr.prepare_resources(batch)

            # Advance the sequence past the sliding window, which is the state a
            # connector is reported at `request_finished`: the window has moved
            # on, so the early ordinals hold no readable KV. Without this the
            # lists below have nothing for the filter to remove.
            kv_cache = mgr.kv_cache_map[request.py_request_id]
            kv_cache.history_length = prompt_len
            stale_end = max(0, (prompt_len + 1 - self.WINDOW) // tokens_per_block)
            self.assertGreater(
                stale_end,
                0,
                "the sizes no longer carry the window past a whole block, so "
                "nothing here exercises a block without a page",
            )

            for group_id in range(len(mgr.impl.layer_grouping)):
                every = list(kv_cache.get_aggregated_page_indices(group_id, valid_only=False))
                only_valid = list(kv_cache.get_aggregated_page_indices(group_id, valid_only=True))
                self.assertEqual(
                    [slot for _, slot in valid_page_slots(every)],
                    only_valid,
                    f"group {group_id}: valid_page_slots and valid_only=True "
                    f"disagree on which blocks hold a page",
                )

            # The same filter over the list a connector is actually handed, which
            # carries the out-of-window mask on top of the missing-page entries.
            by_group = mgr.get_page_indices_by_layer_group(request)
            gaps = sum(1 for indices in by_group for slot in indices if slot < 0)
            self.assertGreaterEqual(
                gaps,
                stale_end,
                f"the sliding group must report at least {stale_end} ordinals "
                f"without a page, so the filter has something to remove: {by_group}",
            )
            for group_id, indices in enumerate(by_group):
                allocated = set(kv_cache.get_aggregated_page_indices(group_id, valid_only=True))
                for ordinal, slot in valid_page_slots(indices):
                    self.assertEqual(
                        indices[ordinal],
                        slot,
                        f"group {group_id}: ordinal {ordinal} does not index "
                        f"back to the slot reported for it",
                    )
                    self.assertIn(
                        slot,
                        allocated,
                        f"group {group_id}: page slot {slot} survived the filter "
                        f"but the cache does not hold it for this request",
                    )
        finally:
            mgr.shutdown()
            del mgr

    def test_live_pages_never_overlap_in_memory(self):
        """The invariant a connector transfers against, in the shared case.

        When both layer groups draw from one pool group they share a slot
        address space, so the same region base describes both and the page
        indices handed out are disjoint instead. Either way, no two live
        ``(layer group, page slot)`` pairs name the same bytes -- which is what
        makes a save keyed on that pair safe.
        """
        mgr = self._vswa_manager()
        try:
            layout = build_kv_cache_layout_v2(mgr)
            request = _make_request(prompt_len=120)
            self.assertTrue(mgr.prepare_context(request))
            mgr.resize_context(request, request.context_remaining_length)
            batch = ScheduledRequests()
            batch.append_context_request(request)
            mgr.prepare_resources(batch)

            by_group = mgr.get_page_indices_by_layer_group(request)
            self.assertEqual(len(by_group), 2)

            spans = []
            for layer_group_id, indices in enumerate(by_group):
                group = layout.group(layer_group_id)
                for slot in indices:
                    if slot == BAD_PAGE_INDEX:
                        continue
                    for region in group.regions:
                        start = region.address_of(slot)
                        spans.append((start, start + region.size, layer_group_id, slot))
            self.assertTrue(spans, "the request holds no live pages")

            spans.sort()
            for previous, current in zip(spans, spans[1:]):
                self.assertLessEqual(
                    previous[1],
                    current[0],
                    f"live page (group {previous[2]}, slot {previous[3]}) at "
                    f"[{previous[0]}, {previous[1]}) overlaps (group "
                    f"{current[2]}, slot {current[3]}) at [{current[0]}, "
                    f"{current[1]})",
                )
        finally:
            mgr.shutdown()
            del mgr

    def test_vswa_layout_declines_the_single_pool_view(self):
        # The back-compat shim must refuse here rather than reconstruct a
        # tensor that silently covers one group. `register_kv_cache_layout`'s
        # default turns this None into the startup error that names the method
        # a VSWA connector has to implement.
        mgr = self._vswa_manager()
        try:
            layout = build_kv_cache_layout_v2(mgr)
            self.assertIsNone(layout.as_single_pool_tensor())
        finally:
            mgr.shutdown()
            del mgr

    def test_each_group_addresses_the_pool_that_backs_it(self):
        # Cross-check against pool_group_descs, which names the layer groups a
        # pool group serves through slot_desc.variants. region base/stride come
        # from get_aggregated_pages; the two APIs must agree per group, not
        # just in aggregate.
        mgr = self._vswa_manager()
        try:
            layout = build_kv_cache_layout_v2(mgr)

            pools_by_group = {}
            slots_by_group = {}
            for pool_group in mgr.impl.pool_group_descs:
                for variant in pool_group.slot_desc.variants:
                    group_id = int(variant.layer_group_id)
                    pools_by_group.setdefault(group_id, []).extend(pool_group.pools)
                    slots_by_group[group_id] = int(pool_group.num_slots)

            self.assertEqual(
                set(pools_by_group),
                {group.layer_group_id for group in layout.groups},
                "every layer group in the layout must be backed by a pool group",
            )

            for group in layout.groups:
                bases = {int(pool.base_address) for pool in pools_by_group[group.layer_group_id]}
                slot_bytes = {int(pool.slot_bytes) for pool in pools_by_group[group.layer_group_id]}
                for region in group.regions:
                    self.assertEqual(region.num_slots, slots_by_group[group.layer_group_id])
                    self.assertIn(
                        region.stride,
                        slot_bytes,
                        f"layer group {group.layer_group_id} region stride "
                        f"{region.stride} matches no pool backing it",
                    )
                    offsets = [region.base - base for base in bases]
                    self.assertTrue(
                        any(0 <= offset < region.stride for offset in offsets),
                        f"layer group {group.layer_group_id} region base "
                        f"{region.base} lies in no pool backing it (pool bases "
                        f"{sorted(bases)})",
                    )
        finally:
            mgr.shutdown()
            del mgr


if __name__ == "__main__":
    unittest.main()
