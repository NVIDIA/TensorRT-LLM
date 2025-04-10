import unittest

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (VanillaAttention,
                                                   VanillaAttentionMetadata)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


class TestVanillaAttention(unittest.TestCase):

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
