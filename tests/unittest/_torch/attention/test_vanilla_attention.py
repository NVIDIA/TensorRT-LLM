import unittest
from unittest.mock import patch

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
