# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc
import unittest

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as KvCacheConfigV2
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


def _create_kv_cache_manager_v2(
    num_layers: int = 4,
    num_kv_heads=4,
    head_dim=128,
    tokens_per_block: int = 8,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_tokens: int = 2048,
    dtype=DataType.HALF,
    kv_cache_type=CacheType.SELF,
    mapping=None,
    vocab_size: int = 32000,
):
    if mapping is None:
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = KvCacheConfigV2(
        max_tokens=max_tokens,
        enable_block_reuse=False,
        dtype="nvfp4" if dtype == DataType.NVFP4 else "auto",
    )
    return KVCacheManagerV2(
        kv_cache_config,
        kv_cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype,
        vocab_size=vocab_size,
    )


class TestPerLayerHeadDimBasic(unittest.TestCase):
    """Tests that don't allocate GPU memory or use uniform buffer sizes."""

    def test_batch_cache_indices_skip_padded_entries(self):
        mgr = _create_kv_cache_manager_v2(
            num_layers=2,
            tokens_per_block=8,
            max_seq_len=64,
            max_tokens=64,
        )
        try:
            mgr.add_dummy_requests([7], [16])
            kv_cache = mgr.kv_cache_map[7]
            base_page_indices = kv_cache.get_base_page_indices(0)
            self.assertGreater(len(base_page_indices), kv_cache.num_blocks)

            result = mgr.get_batch_cache_indices([7])

            index_scale = int(mgr.index_scales[0])
            expected = [
                base_page_index * index_scale // mgr.kv_factor
                if base_page_index != BAD_PAGE_INDEX
                else BAD_PAGE_INDEX
                for base_page_index in base_page_indices[: kv_cache.num_blocks]
            ]
            self.assertEqual(result, [expected])
        finally:
            mgr.shutdown()

    def test_batch_cache_indices_honor_requested_blocks(self):
        mgr = _create_kv_cache_manager_v2(
            num_layers=2,
            tokens_per_block=8,
            max_seq_len=64,
            max_tokens=64,
        )
        try:
            mgr.add_dummy_requests([7], [16])

            all_indices = mgr.get_batch_cache_indices([7])
            requested_indices = mgr.get_batch_cache_indices([7], num_blocks_per_seq=[1])

            self.assertGreater(len(all_indices[0]), 1)
            self.assertEqual(requested_indices, [all_indices[0][:1]])
        finally:
            mgr.shutdown()

    def test_per_layer_head_dim_wrong_length(self):
        """Test that mismatched list length raises assertion."""
        with self.assertRaises(AssertionError):
            _create_kv_cache_manager_v2(
                num_layers=4,
                num_kv_heads=4,
                head_dim=[64, 128, 64],  # 3 != 4 layers
            )

    def test_uniform_head_dim_int(self):
        """Test backward compatibility: passing int head_dim works as before."""
        mgr = _create_kv_cache_manager_v2(
            num_layers=4,
            num_kv_heads=4,
            head_dim=128,
        )
        try:
            self.assertEqual(mgr.head_dim, 128)
            self.assertEqual(mgr.head_dim_per_layer, [128, 128, 128, 128])
            self.assertEqual(mgr.num_local_layers, 4)
        finally:
            mgr.shutdown()

    def test_nvfp4_per_layer_odd_head_dim_raises(self):
        """NVFP4 KV cache requires every per-layer head_dim to be divisible by
        2 (block-scale layout). A heterogeneous list with one odd entry must
        assert at construction rather than silently produce a misaligned
        scale-factor pool."""
        kv_cache_config = KvCacheConfigV2(
            max_tokens=1024,
            enable_block_reuse=False,
            dtype="nvfp4",
        )
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        with self.assertRaises(AssertionError):
            KVCacheManagerV2(
                kv_cache_config,
                CacheType.SELF,
                num_layers=2,
                num_kv_heads=4,
                head_dim=[64, 65],
                tokens_per_block=8,
                max_seq_len=256,
                max_batch_size=4,
                mapping=mapping,
                dtype=DataType.NVFP4,
                vocab_size=32000,
            )

    def test_uniform_head_dim_list(self):
        """Test passing a list with uniform values behaves like int."""
        mgr = _create_kv_cache_manager_v2(
            num_layers=4,
            num_kv_heads=4,
            head_dim=[128, 128, 128, 128],
        )
        try:
            self.assertEqual(mgr.head_dim_per_layer, [128, 128, 128, 128])
        finally:
            mgr.shutdown()

    def test_per_layer_head_dim_with_equal_buffer_sizes(self):
        """Test per-layer head_dim combined with per-layer kv_heads
        that produce equal buffer sizes per layer (avoids pool offset issues)."""
        # 4*64=256, 2*128=256, 4*64=256, 2*128=256 -> all equal
        head_dims = [64, 128, 64, 128]
        kv_heads = [4, 2, 4, 2]
        mgr = _create_kv_cache_manager_v2(
            num_layers=4,
            num_kv_heads=kv_heads,
            head_dim=head_dims,
        )
        try:
            self.assertEqual(mgr.head_dim_per_layer, head_dims)
            self.assertEqual(mgr.num_kv_heads_per_layer, kv_heads)

            # Verify bytes per token differ per layer
            bytes_0 = mgr.get_layer_bytes_per_token(0, Role.KEY)
            bytes_1 = mgr.get_layer_bytes_per_token(1, Role.KEY)
            # Both should be 256 * 2 = 512 bytes (HALF dtype)
            self.assertEqual(bytes_0, 4 * 64 * 2)
            self.assertEqual(bytes_1, 2 * 128 * 2)
            self.assertEqual(bytes_0, bytes_1)
            self.assertEqual(mgr.num_attention_op_pools, mgr.num_pools)
        finally:
            mgr.shutdown()


class TestPerLayerHeadDimHeterogeneous(unittest.TestCase):
    """Tests with different buffer sizes per layer."""

    def setUp(self):
        torch.cuda.init()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_per_layer_head_dim_heterogeneous(self):
        """Test per-layer head_dim with different buffer sizes across layers.

        Covers: construction, head_dim_per_layer, get_layer_bytes_per_token,
        get_cache_bytes_per_token, get_buffers shapes (NHD and HND layouts),
        SELFKONLY cache type, and quota computation.
        """
        num_kv_heads = 4

        # --- Part 1: Basic construction with different head_dims ---
        head_dims_4layer = [64, 128, 64, 128]
        mgr = _create_kv_cache_manager_v2(
            num_layers=4,
            num_kv_heads=num_kv_heads,
            head_dim=head_dims_4layer,
        )
        try:
            self.assertEqual(mgr.head_dim_per_layer, head_dims_4layer)
            self.assertEqual(mgr.head_dim, head_dims_4layer)
        finally:
            mgr.shutdown()

        # Force cleanup before creating next manager
        del mgr
        gc.collect()
        torch.cuda.empty_cache()

        # --- Part 2: Per-layer bytes and cache bytes ---
        head_dims_2layer = [64, 128]
        mgr = _create_kv_cache_manager_v2(
            num_layers=2,
            num_kv_heads=num_kv_heads,
            head_dim=head_dims_2layer,
            dtype=DataType.HALF,
        )
        try:
            # get_layer_bytes_per_token uses per-layer head_dim
            # HALF: each element is 2 bytes
            # Layer 0: 4 * 64 * 2 = 512
            bytes_layer_0 = mgr.get_layer_bytes_per_token(0, Role.KEY)
            # Layer 1: 4 * 128 * 2 = 1024
            bytes_layer_1 = mgr.get_layer_bytes_per_token(1, Role.KEY)
            self.assertEqual(bytes_layer_0, num_kv_heads * 64 * 2)
            self.assertEqual(bytes_layer_1, num_kv_heads * 128 * 2)
            self.assertNotEqual(bytes_layer_0, bytes_layer_1)

            # get_cache_bytes_per_token sums correctly
            total_bytes = mgr.get_cache_bytes_per_token()
            # CacheType.SELF: KEY + VALUE
            # Layer 0: 512*2=1024, Layer 1: 1024*2=2048. Total=3072.
            expected = (num_kv_heads * 64 * 2 + num_kv_heads * 128 * 2) * 2
            self.assertEqual(total_bytes, expected)

            # get_buffers returns correct shapes per layer - NHD layout
            buf_0_nhd = mgr.get_buffers(0, kv_layout="NHD")
            buf_1_nhd = mgr.get_buffers(1, kv_layout="NHD")
            # Shape: [num_blocks, kv_factor, tokens_per_block, num_kv_heads, head_dim]
            self.assertEqual(buf_0_nhd.shape[-1], 64)
            self.assertEqual(buf_1_nhd.shape[-1], 128)
            self.assertEqual(buf_0_nhd.shape[-2], num_kv_heads)
            self.assertEqual(buf_1_nhd.shape[-2], num_kv_heads)

            # get_buffers - HND layout
            buf_0_hnd = mgr.get_buffers(0, kv_layout="HND")
            buf_1_hnd = mgr.get_buffers(1, kv_layout="HND")
            # Shape: [num_blocks, kv_factor, num_kv_heads, tokens_per_block, head_dim]
            self.assertEqual(buf_0_hnd.shape[-1], 64)
            self.assertEqual(buf_1_hnd.shape[-1], 128)
            self.assertEqual(buf_0_hnd.shape[-3], num_kv_heads)
            self.assertEqual(buf_1_hnd.shape[-3], num_kv_heads)
        finally:
            mgr.shutdown()

        del mgr
        gc.collect()
        torch.cuda.empty_cache()

        # --- Part 3: SELFKONLY cache type ---
        mgr = _create_kv_cache_manager_v2(
            num_layers=2,
            num_kv_heads=1,
            head_dim=head_dims_2layer,
            kv_cache_type=CacheType.SELFKONLY,
        )
        try:
            self.assertEqual(mgr.head_dim_per_layer, head_dims_2layer)
            self.assertEqual(mgr.kv_factor, 1)
        finally:
            mgr.shutdown()

        del mgr
        gc.collect()
        torch.cuda.empty_cache()

        # --- Part 4: Quota computation ---
        mgr_uniform = _create_kv_cache_manager_v2(
            num_layers=2,
            num_kv_heads=num_kv_heads,
            head_dim=128,
            max_tokens=1024,
        )
        mgr_mixed = _create_kv_cache_manager_v2(
            num_layers=2,
            num_kv_heads=num_kv_heads,
            head_dim=[64, 192],
            max_tokens=1024,
        )
        try:
            bytes_uniform = mgr_uniform.get_cache_bytes_per_token()
            bytes_mixed = mgr_mixed.get_cache_bytes_per_token()
            # (64+192) == (128+128) = 256 total head_dim
            self.assertEqual(bytes_uniform, bytes_mixed)
        finally:
            mgr_uniform.shutdown()
            mgr_mixed.shutdown()


@pytest.mark.parametrize("dtype", [DataType.HALF, DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("is_gen", [False, True])
def test_heterogeneous_attention_page_addresses(dtype, is_gen):
    # Gemma's short-sequence configuration puts both head sizes in one
    # lifecycle group. Unequal layer counts also give them different page
    # index scales, so sharing a group's pointer or scale is incorrect.
    mgr = _create_kv_cache_manager_v2(
        num_layers=6,
        head_dim=[256] * 5 + [512],
        dtype=dtype,
    )
    try:
        request_ids = [11, 22, 33]
        token_nums = [1, 9, 17]
        requests = mgr.add_dummy_requests(request_ids, token_nums, is_gen=is_gen)
        assert requests is not None
        request_ids.reverse()
        token_nums.reverse()
        block_offsets = torch.empty(
            mgr.num_attention_op_pools,
            len(request_ids),
            2,
            mgr.max_blocks_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
        mgr.copy_batch_block_offsets(
            block_offsets,
            request_ids,
            beam_width=1,
            num_contexts=0 if is_gen else len(request_ids),
            num_seqs=len(request_ids),
        )
        torch.cuda.synchronize()
        block_offsets = block_offsets.cpu()
        for layer_id in range(mgr.num_local_layers):
            pool_id, layer_offset = mgr.kv_cache_pool_mapping[layer_id].tolist()
            lifecycle_id = mgr.impl.get_layer_group_id(layer_id)
            roles = [(Role.KEY, Role.VALUE)]
            if dtype == DataType.NVFP4:
                roles.append((Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE))
            for buffer_idx, role_pair in enumerate(roles):
                pool_pointer = mgr.kv_cache_pool_pointers[pool_id, 0]
                if dtype == DataType.NVFP4:
                    pool_pointer = pool_pointer[buffer_idx]
                for kv_idx, role in enumerate(role_pair):
                    stride = mgr.impl.get_page_stride(layer_id, role)
                    layer_base = mgr.impl.get_mem_pool_base_address(
                        layer_id, role, PageIndexMode.SHARED
                    )
                    scale = mgr.impl.get_page_index_scale(layer_id, role)
                    for seq_idx, (req_id, token_num) in enumerate(zip(request_ids, token_nums)):
                        base_indices = mgr.kv_cache_map[req_id].get_base_page_indices(lifecycle_id)
                        num_blocks = (token_num + mgr.tokens_per_block - 1) // mgr.tokens_per_block
                        for block_idx in range(num_blocks):
                            page_idx = int(block_offsets[pool_id, seq_idx, kv_idx, block_idx])
                            actual = (
                                int(pool_pointer)
                                + (layer_offset * mgr.kv_factor + page_idx) * stride
                            )
                            expected = layer_base + int(base_indices[block_idx]) * scale * stride
                            assert actual == expected, (
                                f"layer={layer_id}, role={role}, request={req_id}, "
                                f"block={block_idx}: attention address {actual:#x} "
                                f"does not match allocated address {expected:#x}"
                            )
    finally:
        mgr.shutdown()


if __name__ == "__main__":
    unittest.main()
