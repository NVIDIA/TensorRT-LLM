# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Focused regression tests for the KVCacheManagerV2 per-layer extra-buffer
# registration hook introduced for MiniMax-M3 sparse index-K support
# (Stage 14 Goal 14.1 in workspace/hidden-trail).
#
# Goal 14.1 scope:
#   * Add ``Role.INDEX_KEY`` to the V2 role surface as an opaque DataRole.
#   * Let subclasses register additional per-layer ``BufferConfig`` entries
#     alongside the standard K/V/NVFP4 scale buffers via the
#     ``_extra_buffers_per_layer`` hook on ``KVCacheManagerV2``.
#   * Preserve the existing K/V/NVFP4 scale wiring and lifecycle.
#
# These tests run on CUDA/GPU because ``KVCacheManagerV2`` constructs real
# device memory pools and the goal explicitly requires GPU evidence
# (Stage 14 acceptance criteria item 1).

import gc
import unittest

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2, Role
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig

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
):
    return dict(
        kv_cache_config=KvCacheConfigV2(max_tokens=max_tokens, enable_block_reuse=False),
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


class _IndexKeyOnSparseLayersV2(KVCacheManagerV2):
    """Test-only V2 subclass that registers a per-layer INDEX_KEY buffer
    via the new ``_extra_buffers_per_layer`` hook for a fixed set of
    "sparse" layers, mirroring the MiniMax-M3 layer-3+ convention without
    pulling in the full M3 backend."""

    INDEX_BYTES_PER_TOKEN = 1 * 128 * 2  # 1 replicated head * sparse_index_dim=128 * BF16

    def __init__(self, *args, sparse_layer_ids, **kwargs):
        self._sparse_layer_ids = sorted(int(i) for i in sparse_layer_ids)
        super().__init__(*args, **kwargs)

    def _extra_buffers_per_layer(self, *, tokens_per_block):
        size_bytes_per_block = self.INDEX_BYTES_PER_TOKEN * tokens_per_block
        return {
            layer_id: [BufferConfig(role=Role.INDEX_KEY, size=size_bytes_per_block)]
            for layer_id in self._sparse_layer_ids
            if layer_id in self.layer_offsets
        }


class _DuplicateRoleV2(KVCacheManagerV2):
    """Negative-control subclass: register Role.KEY as an "extra" so the
    standard buffer + extra duplicate. Must raise."""

    def _extra_buffers_per_layer(self, *, tokens_per_block):
        return {
            0: [BufferConfig(role=Role.KEY, size=64 * tokens_per_block)],
        }


class TestRoleSurface(unittest.TestCase):
    def test_index_key_role_is_distinct(self):
        # Open DataRole is just a NewType('DataRole', str). Identity-by-value
        # check protects against accidental aliasing if someone retypes
        # Role.INDEX_KEY by hand.
        for other in (Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE, Role.ALL):
            self.assertNotEqual(Role.INDEX_KEY, other)
        self.assertEqual(str(Role.INDEX_KEY), "index_key")


class TestExtraBuffersCacheConfig(unittest.TestCase):
    """Tests that allocate GPU memory pools. Mirrors the cleanup pattern
    from ``test_per_layer_head_dim.py`` to keep CUDA virtual address state
    sane across sequential constructions."""

    def setUp(self):
        torch.cuda.init()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_default_hook_keeps_only_standard_buffers(self):
        # Default _extra_buffers_per_layer returns None; the cache config
        # must include exactly KEY + VALUE per layer with no index-K
        # buffer registered on any layer.
        mgr = KVCacheManagerV2(**_make_kwargs())
        try:
            cfg = mgr.kv_cache_manager_py_config
            self.assertEqual(len(cfg.layers), 4)
            for layer in cfg.layers:
                roles = [b.role for b in layer.buffers]
                self.assertEqual(roles, [Role.KEY, Role.VALUE])
                self.assertNotIn(Role.INDEX_KEY, roles)
        finally:
            mgr.shutdown()
            del mgr

    def test_default_hook_keeps_nvfp4_scale_buffers(self):
        # NVFP4 dtype must still register KEY + VALUE + KEY_BLOCK_SCALE +
        # VALUE_BLOCK_SCALE on every layer, and no INDEX_KEY anywhere.
        kwargs = _make_kwargs(dtype=DataType.NVFP4, num_kv_heads=4)
        kwargs["kv_cache_config"] = KvCacheConfigV2(
            max_tokens=1024,
            enable_block_reuse=False,
            dtype="nvfp4",
        )
        mgr = KVCacheManagerV2(**kwargs)
        try:
            cfg = mgr.kv_cache_manager_py_config
            for layer in cfg.layers:
                roles = [b.role for b in layer.buffers]
                self.assertEqual(
                    roles, [Role.KEY, Role.VALUE, Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE]
                )
                self.assertNotIn(Role.INDEX_KEY, roles)
        finally:
            mgr.shutdown()
            del mgr

    def test_subclass_registers_index_key_only_on_sparse_layers(self):
        # Sparse layer convention: layers 0-2 dense (no INDEX_KEY), 3+ sparse
        # (INDEX_KEY registered). Matches MiniMax-M3.
        sparse_layer_ids = [3]
        mgr = _IndexKeyOnSparseLayersV2(
            sparse_layer_ids=sparse_layer_ids, **_make_kwargs(num_layers=4)
        )
        try:
            cfg = mgr.kv_cache_manager_py_config
            self.assertEqual(len(cfg.layers), 4)
            for layer in cfg.layers:
                roles = [b.role for b in layer.buffers]
                if int(layer.layer_id) in sparse_layer_ids:
                    self.assertIn(Role.INDEX_KEY, roles)
                    self.assertEqual(
                        roles,
                        [Role.KEY, Role.VALUE, Role.INDEX_KEY],
                        "extras must be appended after the standard buffers",
                    )
                    # The registered size is bytes per block, not per token.
                    idx_buf = next(b for b in layer.buffers if b.role == Role.INDEX_KEY)
                    expected_bytes_per_block = (
                        _IndexKeyOnSparseLayersV2.INDEX_BYTES_PER_TOKEN * mgr.tokens_per_block
                    )
                    self.assertEqual(idx_buf.size, expected_bytes_per_block)
                else:
                    self.assertEqual(roles, [Role.KEY, Role.VALUE])
                    self.assertNotIn(Role.INDEX_KEY, roles)
        finally:
            mgr.shutdown()
            del mgr

    def test_index_key_layer_has_non_zero_page_upper_bound(self):
        # Once a subclass registers INDEX_KEY, the V2 storage must wire a
        # paged pool for that role with a non-zero page-index upper bound,
        # exercising the open-DataRole path through C++ without role
        # whitelisting.
        sparse_layer_ids = [3]
        mgr = _IndexKeyOnSparseLayersV2(
            sparse_layer_ids=sparse_layer_ids, **_make_kwargs(num_layers=4)
        )
        try:
            layer_offset = mgr.layer_offsets[3]
            page_upper = mgr.impl.get_page_index_upper_bound(layer_offset, Role.INDEX_KEY)
            self.assertGreater(
                page_upper, 0, "INDEX_KEY pool must be allocated for the sparse layer"
            )
            # Dense layer (0) must not have an INDEX_KEY pool.
            with self.assertRaises(Exception):
                mgr.impl.get_page_index_upper_bound(mgr.layer_offsets[0], Role.INDEX_KEY)
        finally:
            mgr.shutdown()
            del mgr

    def test_index_key_does_not_disturb_main_kv_buffers(self):
        # Negative control: registering an INDEX_KEY extra must not change
        # the K/V buffer shapes or bytes-per-token reported for the same
        # layer. Without this guard, an upstream wiring change that
        # accidentally swaps roles or mutates the buffer_type list could
        # silently break the main cache while the extra-buffer test passes.
        sparse_layer_ids = [3]
        mgr = _IndexKeyOnSparseLayersV2(
            sparse_layer_ids=sparse_layer_ids, **_make_kwargs(num_layers=4)
        )
        try:
            for layer_idx in range(4):
                bytes_key = mgr.get_layer_bytes_per_token(layer_idx, Role.KEY)
                bytes_value = mgr.get_layer_bytes_per_token(layer_idx, Role.VALUE)
                # head_dim=128, num_kv_heads=4, dtype=HALF -> 4*128*2=1024
                self.assertEqual(bytes_key, 4 * 128 * 2)
                self.assertEqual(bytes_value, 4 * 128 * 2)

                buf = mgr.get_buffers(layer_idx)
                # [num_blocks, kv_factor=2, tokens_per_block, num_kv_heads, head_dim]
                self.assertEqual(buf.shape[-1], 128)
                self.assertEqual(buf.shape[-2], 4)
        finally:
            mgr.shutdown()
            del mgr

    def test_duplicate_role_against_standard_buffer_asserts(self):
        # Negative control: registering Role.KEY as an extra collides with
        # the standard buffer and must fail loud, both for the assertion
        # this hook adds and for AttentionLayerConfig.__post_init__.
        with self.assertRaises(AssertionError):
            _DuplicateRoleV2(**_make_kwargs(num_layers=4))


class TestIndexKeyBufferAccessor(unittest.TestCase):
    """Focused CUDA/GPU regressions for the
    :meth:`KVCacheManagerV2.get_index_k_buffer` paged accessor introduced
    for MiniMax-M3 sparse index-K support (Stage 14 Goal 14.2 in
    workspace/hidden-trail).

    Goal 14.2 scope:
      * Add a V2 paged accessor that returns a torch view over the
        managed ``Role.INDEX_KEY`` pool for local sparse layers.
      * Shape the view as ``[num_pages, tokens_per_block, num_heads,
        head_dim]`` so sparse modeling code can index by page,
        token-within-block, head, and index dimension.
      * Return ``None`` when the layer has no INDEX_KEY buffer
        registered (e.g. dense layers).
      * Reject wiring mismatch between the caller's
        ``num_heads * head_dim * dtype_bytes * tokens_per_block`` and
        the V2-reported page stride.
      * Provide a zero-copy view (writes propagate; data_ptr stable
        across calls).
    """

    NUM_HEADS = 1
    HEAD_DIM = 128
    BF16_BYTES = 2
    INDEX_BYTES_PER_TOKEN = NUM_HEADS * HEAD_DIM * BF16_BYTES

    def setUp(self):
        torch.cuda.init()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _make_sparse_mgr(self, *, sparse_layer_ids=(3,), num_layers=4):
        return _IndexKeyOnSparseLayersV2(
            sparse_layer_ids=list(sparse_layer_ids),
            **_make_kwargs(num_layers=num_layers),
        )

    def test_accessor_returns_none_on_dense_layer(self):
        # A subclass that registers INDEX_KEY only on layer 3 must yield
        # None for the dense layers (0-2). Without this, sparse modeling
        # code that probes the accessor on dense layers would receive a
        # bogus view aliasing main K/V memory.
        mgr = self._make_sparse_mgr(sparse_layer_ids=(3,))
        try:
            for layer_idx in (0, 1, 2):
                buf = mgr.get_index_k_buffer(
                    layer_idx,
                    num_heads=self.NUM_HEADS,
                    head_dim=self.HEAD_DIM,
                    dtype=torch.bfloat16,
                )
                self.assertIsNone(
                    buf,
                    f"layer {layer_idx} is dense, get_index_k_buffer "
                    f"must return None (got {type(buf).__name__})",
                )
        finally:
            mgr.shutdown()
            del mgr

    def test_accessor_returns_view_on_sparse_layer(self):
        # On a sparse layer that registered INDEX_KEY, the accessor must
        # return a torch view with the documented 4-D shape and BF16
        # dtype.
        mgr = self._make_sparse_mgr(sparse_layer_ids=(3,))
        try:
            buf = mgr.get_index_k_buffer(
                3,
                num_heads=self.NUM_HEADS,
                head_dim=self.HEAD_DIM,
                dtype=torch.bfloat16,
            )
            self.assertIsNotNone(buf)
            self.assertEqual(buf.dtype, torch.bfloat16)
            self.assertEqual(buf.device.type, "cuda")
            self.assertEqual(buf.dim(), 4)
            # [num_pages, tokens_per_block, num_heads, head_dim]
            page_upper = mgr.impl.get_page_index_upper_bound(mgr.layer_offsets[3], Role.INDEX_KEY)
            self.assertEqual(buf.shape[0], page_upper)
            self.assertEqual(buf.shape[1], mgr.tokens_per_block)
            self.assertEqual(buf.shape[2], self.NUM_HEADS)
            self.assertEqual(buf.shape[3], self.HEAD_DIM)
        finally:
            mgr.shutdown()
            del mgr

    def test_accessor_view_is_zero_copy_over_pool(self):
        # The returned view must alias V2 pool memory: writes through one
        # view must be observable through a second call. Without this,
        # sparse decode would write into a private copy and the next
        # read would see stale data — the exact failure mode the plain
        # tensor side cache (Issue B in iter 13) exhibits.
        mgr = self._make_sparse_mgr(sparse_layer_ids=(3,))
        try:
            view1 = mgr.get_index_k_buffer(
                3,
                num_heads=self.NUM_HEADS,
                head_dim=self.HEAD_DIM,
                dtype=torch.bfloat16,
            )
            self.assertIsNotNone(view1)
            # Anchor a fixed BF16 value at a distinctive coordinate so
            # the test catches any silent reshape/copy-on-write.
            sentinel = torch.tensor(42.0, dtype=torch.bfloat16)
            # Pick a coordinate that exists on any nontrivial pool:
            # at least 1 page, 1 token-per-block-slot, head 0, dim 7.
            self.assertGreaterEqual(view1.shape[0], 1)
            self.assertGreaterEqual(view1.shape[1], 1)
            view1[0, 0, 0, 7] = sentinel
            torch.cuda.synchronize()

            view2 = mgr.get_index_k_buffer(
                3,
                num_heads=self.NUM_HEADS,
                head_dim=self.HEAD_DIM,
                dtype=torch.bfloat16,
            )
            self.assertIsNotNone(view2)
            self.assertEqual(view1.data_ptr(), view2.data_ptr())
            torch.cuda.synchronize()
            self.assertEqual(view2[0, 0, 0, 7].item(), float(sentinel))
        finally:
            mgr.shutdown()
            del mgr

    def test_accessor_pointer_matches_v2_pool_base(self):
        # Verify the wrapper points at the exact V2-managed pool base
        # for INDEX_KEY, so the view participates in the same lifecycle
        # as KEY/VALUE pools. Without this guard, a future refactor that
        # accidentally wraps a different base address (e.g. KEY's pool)
        # would alias the main K/V cache.
        mgr = self._make_sparse_mgr(sparse_layer_ids=(3,))
        try:
            buf = mgr.get_index_k_buffer(
                3,
                num_heads=self.NUM_HEADS,
                head_dim=self.HEAD_DIM,
                dtype=torch.bfloat16,
            )
            self.assertIsNotNone(buf)
            v2_addr = mgr.impl.get_mem_pool_base_address(mgr.layer_offsets[3], Role.INDEX_KEY)
            self.assertEqual(buf.data_ptr(), int(v2_addr))

            # And the address must differ from the main K/V pool base:
            # if these collide, the index-K cache would alias the main
            # K/V cache and corrupt both.
            key_addr = mgr.impl.get_mem_pool_base_address(mgr.layer_offsets[3], Role.KEY)
            self.assertNotEqual(int(v2_addr), int(key_addr))
        finally:
            mgr.shutdown()
            del mgr

    def test_accessor_rejects_wrong_head_dim(self):
        # Negative control: a wrong head_dim must fail loud rather than
        # silently producing a view with the wrong stride. The
        # registered BufferConfig.size implies a fixed
        # num_heads * head_dim * elem_bytes contract; any caller-side
        # divergence is a wiring bug.
        mgr = self._make_sparse_mgr(sparse_layer_ids=(3,))
        try:
            with self.assertRaises(AssertionError):
                mgr.get_index_k_buffer(
                    3,
                    num_heads=self.NUM_HEADS,
                    head_dim=self.HEAD_DIM + 1,  # wrong
                    dtype=torch.bfloat16,
                )
        finally:
            mgr.shutdown()
            del mgr

    def test_accessor_rejects_wrong_dtype(self):
        # Negative control: a wrong dtype (FP32 here vs BF16 registered)
        # changes elem_bytes and breaks the stride contract; must fail
        # loud.
        mgr = self._make_sparse_mgr(sparse_layer_ids=(3,))
        try:
            with self.assertRaises(AssertionError):
                mgr.get_index_k_buffer(
                    3,
                    num_heads=self.NUM_HEADS,
                    head_dim=self.HEAD_DIM,
                    dtype=torch.float32,  # 4 bytes vs registered 2
                )
        finally:
            mgr.shutdown()
            del mgr

    def test_accessor_returns_none_on_base_v2_default_manager(self):
        # The base KVCacheManagerV2 (no extra buffers) must report None
        # for every local layer when queried for INDEX_KEY, because the
        # role is not registered. This is the contract that lets generic
        # callers probe the accessor without breaking dense-only models.
        mgr = KVCacheManagerV2(**_make_kwargs())
        try:
            for layer_idx in range(4):
                buf = mgr.get_index_k_buffer(
                    layer_idx,
                    num_heads=self.NUM_HEADS,
                    head_dim=self.HEAD_DIM,
                    dtype=torch.bfloat16,
                )
                self.assertIsNone(buf)
        finally:
            mgr.shutdown()
            del mgr

    def test_accessor_accepts_bindings_data_type(self):
        # The accessor must accept a bindings DataType (HALF / BF16 /
        # FLOAT) alongside torch.dtype, matching ``get_buffers`` which
        # uses the bindings DataType internally. This test pins the
        # BF16 path; the wrong-dtype tests above cover the negative
        # case via torch.dtype.
        from tensorrt_llm._utils import binding_to_torch_dtype

        mgr = self._make_sparse_mgr(sparse_layer_ids=(3,))
        try:
            # Register was BF16 (INDEX_BYTES_PER_TOKEN = 1*128*2);
            # passing DataType.BF16 must succeed and return a BF16 view.
            buf = mgr.get_index_k_buffer(
                3,
                num_heads=self.NUM_HEADS,
                head_dim=self.HEAD_DIM,
                dtype=DataType.BF16,
            )
            self.assertIsNotNone(buf)
            self.assertEqual(buf.dtype, binding_to_torch_dtype(DataType.BF16))
        finally:
            mgr.shutdown()
            del mgr


if __name__ == "__main__":
    unittest.main()
