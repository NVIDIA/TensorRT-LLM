# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (VanillaAttention,
                                                   VanillaAttentionMetadata)
from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


class TestVanillaAttention(unittest.TestCase):

    @patch(
        "tensorrt_llm._torch.attention_backend.interface.AttentionMetadata.prepare"
    )
    def test_prepare_defers_cache_indices_for_layer_specific_pools(
            self, mock_prepare: MagicMock) -> None:
        for manager_attributes in ({
                "is_vswa": True
        }, {
                "is_linear_attention": True
        }, {
                "num_pools": 2
        }):
            with self.subTest(**manager_attributes):
                manager = MagicMock()
                manager.is_vswa = False
                manager.is_linear_attention = False
                manager.num_pools = 1
                for name, value in manager_attributes.items():
                    setattr(manager, name, value)
                metadata = object.__new__(VanillaAttentionMetadata)
                metadata.kv_cache_manager = manager
                metadata.request_ids = [11]

                metadata.prepare()

                self.assertIsNone(metadata.block_ids_per_seq)
                manager.get_batch_cache_indices.assert_not_called()
        self.assertEqual(mock_prepare.call_count, 3)

    @patch(
        "tensorrt_llm._torch.attention_backend.interface.AttentionMetadata.prepare"
    )
    def test_prepare_defers_cache_indices_for_layer_specific_scales(
            self, mock_prepare: MagicMock) -> None:
        manager = MagicMock()
        manager.is_vswa = False
        manager.is_linear_attention = False
        manager.num_pools = 1
        manager.get_layer_page_index_scale = MagicMock(return_value=2)
        metadata = object.__new__(VanillaAttentionMetadata)
        metadata.kv_cache_manager = manager
        metadata.request_ids = [11]

        metadata.prepare()

        self.assertIsNone(metadata.block_ids_per_seq)
        manager.get_batch_cache_indices.assert_not_called()
        mock_prepare.assert_called_once_with()

    @patch(
        "tensorrt_llm._torch.attention_backend.interface.AttentionMetadata.prepare"
    )
    def test_single_pool_cache_indices_remain_prepared_once(
            self, mock_prepare: MagicMock) -> None:
        manager = MagicMock()
        manager.is_vswa = False
        manager.is_linear_attention = False
        manager.num_pools = 1
        manager.get_layer_page_index_scale = None
        manager.get_batch_cache_indices.return_value = [[7]]
        metadata = object.__new__(VanillaAttentionMetadata)
        metadata.kv_cache_manager = manager
        metadata.request_ids = [11]

        metadata.prepare()
        attention = VanillaAttention(layer_idx=3,
                                     num_heads=1,
                                     head_dim=1,
                                     num_kv_heads=1)

        self.assertEqual(attention._get_block_ids_per_seq(metadata), [[7]])
        manager.get_batch_cache_indices.assert_called_once_with([11])
        mock_prepare.assert_called_once_with()

    def test_layer_specific_cache_indices_follow_attention_layer(self) -> None:
        events = []
        manager = MagicMock()

        def get_batch_cache_indices(request_ids: list[int],
                                    layer_idx: int) -> list[list[int]]:
            events.append(("indices", layer_idx))
            return [[10 + layer_idx]]

        def get_buffers(layer_idx: int, *, kv_layout: str) -> torch.Tensor:
            events.append(("buffer", layer_idx))
            return torch.empty(1)

        manager.get_batch_cache_indices.side_effect = get_batch_cache_indices
        manager.get_buffers.side_effect = get_buffers
        metadata = SimpleNamespace(
            block_ids_per_seq=None,
            kv_cache_manager=manager,
            request_ids=[11],
            multi_item_part_lens=None,
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
            seq_lens=torch.tensor([1]),
            seq_lens_kv=torch.tensor([1]),
            kv_layout="NHD",
        )
        layer_zero = VanillaAttention(layer_idx=0,
                                      num_heads=1,
                                      head_dim=1,
                                      num_kv_heads=1)
        layer_one = VanillaAttention(layer_idx=1,
                                     num_heads=1,
                                     head_dim=1,
                                     num_kv_heads=1)
        for attention in (layer_zero, layer_one):
            with patch.object(
                    attention,
                    "_single_request_forward",
                    return_value=torch.zeros(1, 1, 1),
            ):
                attention.forward(
                    torch.zeros(1, 1),
                    torch.zeros(1, 1),
                    torch.zeros(1, 1),
                    metadata,
                )

        self.assertEqual(manager.get_batch_cache_indices.call_args_list, [
            unittest.mock.call([11], layer_idx=0),
            unittest.mock.call([11], layer_idx=1),
        ])
        self.assertEqual(events, [
            ("indices", 0),
            ("buffer", 0),
            ("indices", 1),
            ("buffer", 1),
        ])

    def test_mla_generation_uses_layer_specific_cache_indices(self) -> None:
        layer_idx = 4
        mla_params = SimpleNamespace(
            kv_lora_rank=1,
            qk_rope_head_dim=1,
            qk_nope_head_dim=1,
            v_head_dim=1,
        )
        attention = VanillaAttention(
            layer_idx=layer_idx,
            num_heads=1,
            head_dim=2,
            num_kv_heads=1,
            mla_params=mla_params,
        )
        manager = MagicMock()
        manager.get_batch_cache_indices.return_value = [[0]]
        metadata = SimpleNamespace(
            block_ids_per_seq=None,
            kv_cache_manager=manager,
            request_ids=[11],
            seq_lens=torch.tensor([1]),
            kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[0]),
            kv_layout="NHD",
        )
        kv_cache = torch.zeros(1, 1, 1, 1, 2)

        with patch(
                "tensorrt_llm._torch.attention_backend.utils.append_mla_latent_cache",
                return_value=kv_cache,
        ):
            result = attention._mla_forward_generation(
                torch.zeros(1, 2),
                metadata,
                torch.zeros(1, 2),
            )

        self.assertEqual(result.shape, (1, 1))
        manager.get_batch_cache_indices.assert_called_once_with(
            [11], layer_idx=layer_idx)

    def test_sdpa_fallback_uses_metadata_cross_flag_for_causal_mask(self):
        vanilla_attn = VanillaAttention(layer_idx=0,
                                        num_heads=1,
                                        head_dim=1,
                                        num_kv_heads=1)
        q = torch.ones(2, 1, 1)
        k = torch.ones(2, 1, 1)
        v = torch.ones(2, 1, 1)
        seqlens = torch.tensor([2], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
        observed_is_causal = []

        def fake_sdpa(q_s, k_s, v_s, *, is_causal, **kwargs):
            del k_s, v_s, kwargs
            observed_is_causal.append(is_causal)
            return torch.zeros_like(q_s)

        with patch.object(F, "scaled_dot_product_attention", fake_sdpa):
            vanilla_attn._no_kv_cache_sdpa_fallback(
                q,
                k,
                v,
                num_heads=1,
                num_kv_heads=1,
                head_dim=1,
                seqlens_q=seqlens,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_q=2,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                seqlens_kv=seqlens.clone(),
                cu_seqlens_k=cu_seqlens.clone(),
                max_seqlen_k=2,
                is_cross=True,
            )
            vanilla_attn._no_kv_cache_sdpa_fallback(
                q,
                k,
                v,
                num_heads=1,
                num_kv_heads=1,
                head_dim=1,
                seqlens_q=seqlens,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_q=2,
                attention_mask=PredefinedAttentionMask.CAUSAL,
                is_cross=False,
            )

        self.assertEqual(observed_is_causal, [False, True])

    def test_vanilla_attention(self):
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        device = torch.device('cuda')
        dtype = torch.bfloat16

        context_sequence_lengths = [3, 2]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [30, 40, 62, 75]
        request_ids = [0, 1, 2, 3]
        token_nums = (torch.tensor(sequence_lengths) +
                      torch.tensor(past_seen_tokens)).tolist()

        context_1_q = torch.randn(context_sequence_lengths[0],
                                  num_heads * head_dim,
                                  dtype=dtype,
                                  device=device)
        context_1_k = torch.randn(context_sequence_lengths[0],
                                  num_kv_heads * head_dim,
                                  dtype=dtype,
                                  device=device)
        context_1_v = torch.randn(context_sequence_lengths[0],
                                  num_kv_heads * head_dim,
                                  dtype=dtype,
                                  device=device)

        context_2_q = torch.randn(context_sequence_lengths[1],
                                  num_heads * head_dim,
                                  dtype=dtype,
                                  device=device)
        context_2_k = torch.randn(context_sequence_lengths[1],
                                  num_kv_heads * head_dim,
                                  dtype=dtype,
                                  device=device)
        context_2_v = torch.randn(context_sequence_lengths[1],
                                  num_kv_heads * head_dim,
                                  dtype=dtype,
                                  device=device)

        gen_1_q = torch.randn(1,
                              num_heads * head_dim,
                              dtype=dtype,
                              device=device)
        gen_1_k = torch.randn(1,
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
        gen_1_v = torch.randn(1,
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)

        gen_2_q = torch.randn(1,
                              num_heads * head_dim,
                              dtype=dtype,
                              device=device)
        gen_2_k = torch.randn(1,
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)
        gen_2_v = torch.randn(1,
                              num_kv_heads * head_dim,
                              dtype=dtype,
                              device=device)

        num_blocks = 100
        tokens_per_block = 128
        num_layers = 1
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 4
        mapping = Mapping(world_size=1, tp_size=1, rank=0)

        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
        if dtype == torch.float16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype for unit test")

        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        for i in range(num_layers):
            buf = kv_cache_manager.get_buffers(i)
            if buf is not None:
                torch.nn.init.normal_(buf)
                del buf

        vanilla_attn = VanillaAttention(layer_idx=0,
                                        num_heads=num_heads,
                                        head_dim=head_dim,
                                        num_kv_heads=num_kv_heads)

        # [context_1, context_2, gen_1, gen_2]

        attn_metadata = VanillaAttentionMetadata(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
            num_contexts=len(context_sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
            max_num_requests=4,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )

        attn_metadata.prepare()
        result_1 = vanilla_attn.forward(
            torch.cat((context_1_q, context_2_q, gen_1_q, gen_2_q)),
            torch.cat((context_1_k, context_2_k, gen_1_k, gen_2_k)),
            torch.cat((context_1_v, context_2_v, gen_1_v, gen_2_v)),
            attn_metadata)
        self.assertEqual(result_1.size()[0], sum(context_sequence_lengths) + 2)

        # [context_1, gen_1]

        num_cached_tokens_per_seq = [
            j for j in [past_seen_tokens[0], past_seen_tokens[2]]
        ]

        attn_metadata = VanillaAttentionMetadata(
            seq_lens=torch.tensor([context_sequence_lengths[0], 1],
                                  dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=[0, 2],
            max_num_requests=4,
            max_num_tokens=8192,
        )
        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()
        result_2 = vanilla_attn.forward(torch.cat((context_1_q, gen_1_q)),
                                        torch.cat((context_1_k, gen_1_k)),
                                        torch.cat((context_1_v, gen_1_v)),
                                        attn_metadata)
        self.assertEqual(result_2.size()[0], context_sequence_lengths[0] + 1)

        # [context_2, gen_2]

        num_cached_tokens_per_seq = [
            j for j in [past_seen_tokens[1], past_seen_tokens[3]]
        ]

        attn_metadata = VanillaAttentionMetadata(
            seq_lens=torch.tensor([context_sequence_lengths[1], 1],
                                  dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=[1, 3],
            max_num_requests=4,
            max_num_tokens=8192,
        )
        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()
        result_3 = vanilla_attn.forward(torch.cat((context_2_q, gen_2_q)),
                                        torch.cat((context_2_k, gen_2_k)),
                                        torch.cat((context_2_v, gen_2_v)),
                                        attn_metadata)
        self.assertEqual(result_3.size()[0], context_sequence_lengths[1] + 1)

        # assert value

        torch.testing.assert_close(
            result_1,
            torch.cat((
                result_2[:context_sequence_lengths[0], :],
                result_3[:context_sequence_lengths[1], :],
                result_2[context_sequence_lengths[0]:, :],
                result_3[context_sequence_lengths[1]:, :],
            )))

        kv_cache_manager.shutdown()
