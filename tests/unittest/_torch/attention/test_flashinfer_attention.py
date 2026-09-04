# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import unittest
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Union
from unittest import mock

import torch
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (FlashInferAttention,
                                                   FlashInferAttentionMetadata)
from tensorrt_llm._torch.attention_backend import \
    flashinfer as flashinfer_backend
from tensorrt_llm._torch.attention_backend.flashinfer import (
    FlashInferWrappers, PlanParams)
from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.mapping import Mapping


class TestingFlashInferAttentionMetadata(FlashInferAttentionMetadata):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_times_planned = defaultdict(int)

    def get_num_plans(self, plan_params) -> int:
        return self._num_times_planned[plan_params]

    def _plan_with_params(self, plan_params, flashinfer_backend: str = "fa2"):
        if self.needs_plan(plan_params):
            self._num_times_planned[plan_params] += 1
        return super()._plan_with_params(plan_params, flashinfer_backend)


@dataclass(repr=False)
class Scenario:
    num_layers: int
    num_heads: int
    num_kv_heads: Union[int, List[Optional[int]]]
    head_dim: int
    dtype: torch.dtype

    def __repr__(self) -> str:
        if isinstance(self.num_kv_heads, int):
            num_kv_heads_str = str(self.num_kv_heads)
        else:
            num_kv_heads_str = '_'.join(map(str, self.num_kv_heads))
        return f"num_layers:{self.num_layers}-num_heads:{self.num_heads}-num_kv_heads:{num_kv_heads_str}-head_dim:{self.head_dim}-dtype:{self.dtype}"


@dataclass
class CUDAGraphTestScenario:
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype


class TestFlashInferAttention(unittest.TestCase):

    def test_attention_layer_indices_ignore_zero_kv_layers(self) -> None:

        class FakeHybridManager:

            def __init__(self) -> None:
                self.layer_offsets = {10: 0, 20: 1, 30: 2}
                self.num_kv_heads_per_layer = [8, 0, 8]

            def is_attention_layer(self, layer_idx: int) -> bool:
                return layer_idx != 20

        self.assertEqual(
            flashinfer_backend._get_attention_layer_indices(
                FakeHybridManager()), [10, 30])

    def test_attention_layer_indices_support_v1_manager(self) -> None:

        class FakeV1Manager:

            def __init__(self) -> None:
                self.layer_offsets = {10: 0, 20: 1}

        self.assertEqual(
            flashinfer_backend._get_attention_layer_indices(FakeV1Manager()),
            [10, 20])

    def test_hybrid_v2_metadata_uses_first_attention_layer_for_prepare(
            self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        class FakeHybridManager:
            blocks_in_primary_pool = 4
            max_blocks_per_seq = 4
            tokens_per_block = 32
            is_vswa = False

            def __init__(self) -> None:
                self.layer_offsets = {10: 0, 20: 1}
                self.layer_to_pool_mapping_dict = {0: 0, 1: 0}

            def is_attention_layer(self, layer_idx: int) -> bool:
                return layer_idx == 20

            def get_layer_page_index_scale(self, layer_idx: int) -> int:
                return 1

        manager = FakeHybridManager()
        manager.get_buffers = mock.Mock(
            side_effect=lambda layer_idx: torch.empty(4) if layer_idx == 20 else
            self.fail("recurrent layer queried for KV buffers"))
        manager.get_batch_cache_indices_flat = mock.Mock(
            return_value=torch.tensor([2], dtype=torch.int32))

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.zeros(1, dtype=torch.int32),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[1],
            ),
            max_num_requests=1,
            max_num_tokens=1,
            kv_cache_manager=manager,
            request_ids=[7],
            workspace_buffer=torch.empty(1, dtype=torch.uint8, device="cuda"),
            mamba_metadata=False,
        )

        self.assertEqual(metadata._primary_kv_layer_idx, 20)
        manager.get_buffers.assert_called_once_with(20)

        metadata.prepare()

        manager.get_batch_cache_indices_flat.assert_called_once_with(
            [7], [1], layer_idx=20)

    def test_generation_page_table_uses_reserved_block_count(self):
        manager = SimpleNamespace(get_batch_cache_indices=mock.Mock(
            return_value=[list(range(325))]))

        self.assertEqual(
            flashinfer_backend._get_page_table_num_blocks(manager, [98, 99],
                                                          [3, 324],
                                                          num_contexts=1),
            [3, 325],
        )
        manager.get_batch_cache_indices.assert_called_once_with([99])

    def test_decode_plan_cache_key_reuses_single_token_batches(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.tensor([1, 1], dtype=torch.int32),
            num_contexts=0,
            kv_cache_manager=None,
            request_ids=[0, 1],
            max_num_requests=3,
            max_num_tokens=18,
        )

        def return_plan_params(plan_params, _flashinfer_backend):
            return plan_params

        with mock.patch.object(
                metadata,
                "_plan_with_params",
                side_effect=return_plan_params,
        ):
            single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata.seq_lens = torch.tensor([1, 1, 1], dtype=torch.int32)
            larger_single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata.seq_lens = torch.tensor([1, 1], dtype=torch.int32)
            metadata._uses_full_generation_page_table = True
            full_page_single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata._uses_full_generation_page_table = False
            metadata._is_shared_kv_draft_view = True
            shared_draft_single_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata._is_shared_kv_draft_view = False
            metadata.seq_lens = torch.tensor([6, 6], dtype=torch.int32)
            multi_token_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )
            metadata.seq_lens = torch.tensor([6, 6, 6], dtype=torch.int32)
            larger_batch_plan = metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )

        self.assertEqual(single_token_plan.q_len_per_req, 1)
        self.assertEqual(single_token_plan.num_generations, 0)
        self.assertEqual(larger_single_token_plan.q_len_per_req, 1)
        self.assertEqual(larger_single_token_plan.num_generations, 0)
        self.assertEqual(single_token_plan, larger_single_token_plan)
        self.assertEqual(full_page_single_token_plan.num_generations, 2)
        self.assertNotEqual(single_token_plan, full_page_single_token_plan)
        self.assertEqual(shared_draft_single_token_plan.num_generations, 2)
        self.assertNotEqual(single_token_plan, shared_draft_single_token_plan)
        self.assertEqual(multi_token_plan.q_len_per_req, 6)
        self.assertNotEqual(single_token_plan, multi_token_plan)
        self.assertEqual(multi_token_plan.num_generations, 2)
        self.assertEqual(larger_batch_plan.num_generations, 3)
        self.assertNotEqual(multi_token_plan, larger_batch_plan)

        metadata.seq_lens = torch.tensor([6, 5], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "uniform query length"):
            metadata.plan(
                num_heads=32,
                num_kv_heads=4,
                head_dim=512,
                q_dtype=torch.float8_e4m3fn,
                kv_dtype=torch.float8_e4m3fn,
                attention_mask_type=AttentionMaskType.causal.value,
                flashinfer_backend="trtllm-gen",
            )

    def test_generation_page_table_keeps_logical_positions(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=256),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=128,
            tokens_per_block=32,
            max_seq_len=64,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        try:
            kv_cache_manager.add_dummy_requests([0], [32],
                                                is_gen=True,
                                                max_num_draft_tokens=3)
            reserved_blocks = kv_cache_manager.get_batch_cache_indices([0])
            self.assertEqual(len(reserved_blocks[0]), 2)

            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.full((1, ), 4, dtype=torch.int32),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[28],
                    use_full_generation_page_table=True,
                ),
                max_num_requests=1,
                max_num_tokens=4,
                kv_cache_manager=kv_cache_manager,
                request_ids=[0],
            )
            metadata.prepare()

            self.assertEqual(metadata.num_blocks, [2])
            torch.testing.assert_close(
                metadata.paged_kv_indptr_decode[:2],
                torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._paged_kv_last_page_len[:1],
                torch.tensor([32], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._logical_kv_lens[:1],
                torch.tensor([32], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata.positions,
                torch.tensor([28, 29, 30, 31], dtype=torch.int32,
                             device="cuda"),
            )
        finally:
            kv_cache_manager.shutdown()

    def test_spec_decode_offsets_update_append_and_decode_lengths(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=256),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=128,
            tokens_per_block=32,
            max_seq_len=64,
            max_batch_size=2,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        try:
            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.full((2, ), 4, dtype=torch.int32),
                num_contexts=0,
                kv_cache_params=KVCacheParams(use_cache=True),
                max_num_requests=2,
                max_num_tokens=8,
                kv_cache_manager=kv_cache_manager,
            )
            cached_token_lens = torch.tensor([31, 62],
                                             dtype=torch.int32,
                                             device="cuda")
            positions = torch.tensor([31, 32, 33, 34, 62, 63, 64, 65],
                                     dtype=torch.int32,
                                     device="cuda")
            metadata._cached_token_lens[:2].copy_(cached_token_lens)
            metadata._logical_kv_lens[:2].copy_(cached_token_lens + 4)
            metadata._uses_full_generation_page_table = True
            metadata._positions[:8].copy_(positions)
            kv_lens_buffer = torch.tensor([35, 66],
                                          dtype=torch.int32,
                                          device="cuda")
            metadata._plan_params_to_wrappers = {
                object():
                FlashInferWrappers(
                    is_planned=True,
                    decode_wrapper=SimpleNamespace(
                        _kv_lens_buffer=kv_lens_buffer),
                )
            }
            offsets = torch.tensor([-3, -1], dtype=torch.int32, device="cuda")

            metadata.apply_spec_decode_kv_lens_offsets(
                offsets,
                num_generations=2,
                tokens_per_generation=4,
            )

            torch.testing.assert_close(
                metadata._cached_token_lens[:2],
                torch.tensor([28, 61], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._logical_kv_lens[:2],
                torch.tensor([32, 65], dtype=torch.int32, device="cuda"),
            )
            torch.testing.assert_close(
                metadata._positions[:8],
                torch.tensor([28, 29, 30, 31, 61, 62, 63, 64],
                             dtype=torch.int32,
                             device="cuda"),
            )
            torch.testing.assert_close(
                kv_lens_buffer,
                torch.tensor([32, 65], dtype=torch.int32, device="cuda"),
            )

            metadata.apply_spec_decode_kv_lens_offsets(
                offsets,
                num_generations=2,
                tokens_per_generation=4,
                restore=True,
            )
            torch.testing.assert_close(metadata._cached_token_lens[:2],
                                       cached_token_lens)
            torch.testing.assert_close(metadata._logical_kv_lens[:2],
                                       cached_token_lens + 4)
            torch.testing.assert_close(metadata._positions[:8], positions)
            torch.testing.assert_close(
                kv_lens_buffer,
                torch.tensor([35, 66], dtype=torch.int32, device="cuda"),
            )
        finally:
            kv_cache_manager.shutdown()

    def test_separate_kv_draft_metadata_uses_draft_manager(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("FlashInfer trtllm-gen requires SM100 or SM103")

        def create_manager():
            return KVCacheManager(
                KvCacheConfig(max_tokens=256),
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=1,
                num_kv_heads=1,
                head_dim=128,
                tokens_per_block=32,
                max_seq_len=64,
                max_batch_size=2,
                mapping=Mapping(world_size=1, tp_size=1, rank=0),
                dtype=tensorrt_llm.bindings.DataType.BF16,
            )

        target_manager = create_manager()
        draft_manager = create_manager()
        try:
            target_manager.add_dummy_requests([0, 1], [31, 45], is_gen=True)
            draft_manager.add_dummy_requests([0, 1], [31, 45],
                                             is_gen=True,
                                             max_num_draft_tokens=3)
            metadata = FlashInferAttentionMetadata(
                seq_lens=torch.ones(2, dtype=torch.int32),
                num_contexts=0,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[30, 44],
                ),
                max_num_requests=2,
                max_num_tokens=8,
                kv_cache_manager=target_manager,
                request_ids=[0, 1],
                is_cuda_graph=True,
            )
            metadata.prepare()
            draft_metadata = metadata.get_draft_metadata(draft_manager)

            self.assertIs(draft_metadata.kv_cache_manager, draft_manager)
            self.assertFalse(hasattr(metadata, "kv_lens_cuda"))
            torch.testing.assert_close(
                draft_metadata.kv_lens_cuda[:2],
                torch.tensor([31, 45], dtype=torch.int32, device="cuda"),
            )
            draft_blocks = draft_manager.get_batch_cache_indices([0, 1])
            self.assertEqual(draft_metadata.num_blocks,
                             list(map(len, draft_blocks)))

            layer = FlashInferAttention(
                layer_idx=0,
                num_heads=1,
                num_kv_heads=1,
                head_dim=128,
                flashinfer_backend="trtllm-gen",
            )
            q = torch.randn(2, 128, dtype=torch.bfloat16, device="cuda")
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            self.assertEqual(
                layer.forward(q, k, v, draft_metadata).shape, q.shape)
            for wrappers in draft_metadata._plan_params_to_wrappers.values():
                torch.testing.assert_close(
                    wrappers.decode_wrapper._kv_lens_buffer[:2],
                    draft_metadata.kv_lens_cuda[:2],
                )

            with mock.patch.object(
                    draft_metadata,
                    "_plan_with_params",
                    wraps=draft_metadata._plan_with_params,
            ) as replan, mock.patch.object(
                    draft_metadata,
                    "_build_decode_block_tables",
            ) as refresh_block_tables:
                metadata.prepare()
            replan.assert_not_called()
            self.assertEqual(refresh_block_tables.call_count,
                             len(draft_metadata._plan_params_to_wrappers))
        finally:
            target_manager.shutdown()
            draft_manager.shutdown()

    def test_ragged_no_kv_cuda_graph_uses_stable_indptr_aliases(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer metadata")

        class FakeRaggedWrapper:

            def plan(self, **kwargs):
                self.plan_kwargs = kwargs

        metadata = FlashInferAttentionMetadata(
            seq_lens=torch.tensor([3, 5], dtype=torch.int32),
            num_contexts=2,
            kv_cache_manager=None,
            request_ids=[0, 1],
            max_num_requests=2,
            max_num_tokens=8,
        )
        metadata.is_cuda_graph = True
        plan_params = PlanParams(
            num_heads=2,
            num_kv_heads=2,
            head_dim=4,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            attention_mask_type=AttentionMaskType.padding,
        )
        wrapper = FakeRaggedWrapper()

        with mock.patch.object(
                flashinfer_backend.flashinfer,
                "BatchPrefillWithRaggedKVCacheWrapper",
                FakeRaggedWrapper,
        ):
            metadata._plan_ragged_no_kv(plan_params, wrapper, "cudnn")

        self.assertIn("kv_indptr", wrapper.plan_kwargs)
        self.assertNotIn("v_indptr", wrapper.plan_kwargs)
        self.assertNotIn("o_indptr", wrapper.plan_kwargs)

    @parameterized.expand([
        Scenario(num_layers=1,
                 num_heads=32,
                 num_kv_heads=8,
                 head_dim=128,
                 dtype=torch.bfloat16),
        Scenario(num_layers=2,
                 num_heads=32,
                 num_kv_heads=8,
                 head_dim=64,
                 dtype=torch.float16),
        Scenario(num_layers=2,
                 num_heads=32,
                 num_kv_heads=[8, 16],
                 head_dim=128,
                 dtype=torch.bfloat16),
        Scenario(num_layers=3,
                 num_heads=32,
                 num_kv_heads=[8, None, 16],
                 head_dim=64,
                 dtype=torch.float16),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_flashinfer_attention(self, scenario: Scenario):
        num_layers = scenario.num_layers
        num_heads = scenario.num_heads
        num_kv_heads = scenario.num_kv_heads
        head_dim = scenario.head_dim
        dtype = scenario.dtype

        device = torch.device('cuda')

        # TODO: make these a part of the scenario?
        num_gens = 2
        context_sequence_lengths = [3, 2]
        sequence_lengths = context_sequence_lengths + [1] * num_gens
        past_seen_tokens = [30, 40, 62, 75]
        batch_size = num_gens + len(context_sequence_lengths)
        request_ids = list(range(batch_size))
        token_nums = (torch.tensor(sequence_lengths) +
                      torch.tensor(past_seen_tokens)).tolist()

        num_blocks = 16
        tokens_per_block = 128
        max_seq_len = tokens_per_block * num_blocks
        mapping = Mapping(world_size=1, tp_size=1, rank=0)

        if dtype == torch.float16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype for unit test")

        kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                        tokens_per_block)
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

        for i in range(kv_cache_manager.num_layers):
            buf = kv_cache_manager.get_buffers(i)
            if buf is not None:
                torch.nn.init.normal_(buf)
                del buf

        if isinstance(num_kv_heads, int):
            num_kv_heads = [num_kv_heads] * num_layers

        contexts_per_layer = []
        gens_per_layer = []

        for layer_idx in range(num_layers):
            kv_heads = num_kv_heads[layer_idx]
            if kv_heads is None:
                continue

            context_qs = [
                torch.randn(sequence_length,
                            num_heads * head_dim,
                            dtype=dtype,
                            device=device)
                for sequence_length in context_sequence_lengths
            ]

            context_ks = [
                torch.randn(sequence_length,
                            kv_heads * head_dim,
                            dtype=dtype,
                            device=device)
                for sequence_length in context_sequence_lengths
            ]
            context_vs = [
                torch.randn(sequence_length,
                            kv_heads * head_dim,
                            dtype=dtype,
                            device=device)
                for sequence_length in context_sequence_lengths
            ]

            contexts_per_layer.append((context_qs, context_ks, context_vs))

            gen_qs = [
                torch.randn(1, num_heads * head_dim, dtype=dtype, device=device)
                for _ in range(num_gens)
            ]

            gen_ks = [
                torch.randn(1, kv_heads * head_dim, dtype=dtype, device=device)
                for _ in range(num_gens)
            ]

            gen_vs = [
                torch.randn(1, kv_heads * head_dim, dtype=dtype, device=device)
                for _ in range(num_gens)
            ]

            gens_per_layer.append((gen_qs, gen_ks, gen_vs))

        layers = [
            FlashInferAttention(
                layer_idx=layer_idx,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=kv_heads,
            ) for layer_idx, kv_heads in enumerate(num_kv_heads)
            if kv_heads is not None
        ]

        # [context_1, context_2, gen_1, gen_2]
        results_1 = []

        seq_lens = torch.tensor(sequence_lengths).int()
        attn_metadata = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=len(context_sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
            max_num_requests=4,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )

        attn_metadata.prepare()
        for attn_layer_idx, flashinfer_attn in enumerate(layers):
            context_qs, context_ks, context_vs = contexts_per_layer[
                attn_layer_idx]
            gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

            q = torch.cat((*context_qs, *gen_qs))
            k = torch.cat((*context_ks, *gen_ks))
            v = torch.cat((*context_vs, *gen_vs))

            result_1 = flashinfer_attn.forward(q, k, v, attn_metadata)
            self.assertEqual(result_1.size()[0],
                             sum(context_sequence_lengths) + num_gens)

            # validate kv cache was updated expectedly
            cache_buf = kv_cache_manager.get_buffers(
                flashinfer_attn.layer_idx, kv_layout=attn_metadata.kv_layout)
            if attn_metadata.kv_layout == "HND":
                cache_buf = cache_buf.transpose(2, 3).contiguous()
            assert cache_buf is not None
            num_kv_heads = cache_buf.size(-2)

            # validate contexts
            block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(
                request_ids)
            for seq_id in range(len(context_sequence_lengths)):
                # get a contiguous copy of the cache for the sequence
                block_ids = block_ids_per_seq[seq_id]
                last_block_len = attn_metadata.paged_kv_last_page_len[seq_id]
                cached_kvs = torch.concat(cache_buf[block_ids, :].unbind(dim=0),
                                          dim=1)
                # only look at new tokens added
                cached_kvs = cached_kvs[:,
                                        past_seen_tokens[seq_id]:last_block_len]

                # compare to input kvs
                torch.testing.assert_close(
                    cached_kvs[0].to(context_ks[seq_id].dtype),
                    context_ks[seq_id].view(-1, num_kv_heads, head_dim))
                torch.testing.assert_close(
                    cached_kvs[1].to(context_vs[seq_id].dtype),
                    context_vs[seq_id].view(-1, num_kv_heads, head_dim))

            # validate generations (same way)
            for gen_seq_id in range(num_gens):
                seq_id = len(context_sequence_lengths) + gen_seq_id
                block_ids = block_ids_per_seq[seq_id]
                last_block_len = attn_metadata.paged_kv_last_page_len[seq_id]
                cached_kvs = torch.concat(
                    cache_buf[block_ids, :].unbind(dim=0),
                    dim=1)[:, past_seen_tokens[seq_id]:last_block_len]

                torch.testing.assert_close(
                    cached_kvs[0],
                    gen_ks[gen_seq_id].view(-1, num_kv_heads, head_dim))
                torch.testing.assert_close(
                    cached_kvs[1],
                    gen_vs[gen_seq_id].view(-1, num_kv_heads, head_dim))

            results_1.append(result_1)
            del cache_buf

        for plan_params in attn_metadata._plan_params_to_wrappers.keys():
            self.assertEqual(attn_metadata.get_num_plans(plan_params), 1)

        # prepare() defers re-planning to forward_impl only when multiple
        # wrappers share one workspace_buffer (hybrid attention); for the
        # single-wrapper case it re-plans eagerly so cuda-graph capture works.
        attn_metadata.prepare()
        defer_plan = len(attn_metadata._plan_params_to_wrappers) > 1
        for wrappers in attn_metadata._plan_params_to_wrappers.values():
            self.assertEqual(wrappers.is_planned, not defer_plan)

        # [context_1, gen_1]
        results_2 = []
        num_cached_tokens_per_seq = [
            j for j in [
                past_seen_tokens[0], past_seen_tokens[len(
                    context_sequence_lengths)]
            ]
        ]

        seq_lens = torch.tensor([context_sequence_lengths[0], 1],
                                dtype=torch.int)
        attn_metadata = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq),
            max_num_requests=2,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=[0, 2],
        )

        attn_metadata.prepare()

        for attn_layer_idx, flashinfer_attn in enumerate(layers):
            context_qs, context_ks, context_vs = contexts_per_layer[
                attn_layer_idx]
            gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

            result_2 = flashinfer_attn.forward(
                torch.cat((context_qs[0], gen_qs[0])),
                torch.cat((context_ks[0], gen_ks[0])),
                torch.cat((context_vs[0], gen_vs[0])), attn_metadata)
            self.assertEqual(result_2.size()[0],
                             context_sequence_lengths[0] + 1)
            results_2.append(result_2)

        for plan_params in attn_metadata._plan_params_to_wrappers.keys():
            self.assertEqual(attn_metadata.get_num_plans(plan_params), 1)

        # prepare() defers re-planning to forward_impl only when multiple
        # wrappers share one workspace_buffer (hybrid attention); for the
        # single-wrapper case it re-plans eagerly so cuda-graph capture works.
        attn_metadata.prepare()
        defer_plan = len(attn_metadata._plan_params_to_wrappers) > 1
        for wrappers in attn_metadata._plan_params_to_wrappers.values():
            self.assertEqual(wrappers.is_planned, not defer_plan)

        # [context_2, gen_2]
        results_3 = []
        num_cached_tokens_per_seq = [
            j for j in [
                past_seen_tokens[1], past_seen_tokens[
                    len(context_sequence_lengths) + 1]
            ]
        ]

        seq_lens = torch.tensor([context_sequence_lengths[1], 1],
                                dtype=torch.int)
        attn_metadata = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq),
            max_num_requests=2,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=[1, 3],
        )

        attn_metadata.prepare()
        for attn_layer_idx, flashinfer_attn in enumerate(layers):
            context_qs, context_ks, context_vs = contexts_per_layer[
                attn_layer_idx]
            gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

            result_3 = flashinfer_attn.forward(
                torch.cat((context_qs[1], gen_qs[1])),
                torch.cat((context_ks[1], gen_ks[1])),
                torch.cat((context_vs[1], gen_vs[1])), attn_metadata)
            self.assertEqual(result_3.size()[0],
                             context_sequence_lengths[1] + 1)
            results_3.append(result_3)

        for plan_params in attn_metadata._plan_params_to_wrappers.keys():
            self.assertEqual(attn_metadata.get_num_plans(plan_params), 1)

        # prepare() defers re-planning to forward_impl only when multiple
        # wrappers share one workspace_buffer (hybrid attention); for the
        # single-wrapper case it re-plans eagerly so cuda-graph capture works.
        attn_metadata.prepare()
        defer_plan = len(attn_metadata._plan_params_to_wrappers) > 1
        for wrappers in attn_metadata._plan_params_to_wrappers.values():
            self.assertEqual(wrappers.is_planned, not defer_plan)

        # assert value

        for result_1, result_2, result_3 in zip(results_1, results_2,
                                                results_3):
            torch.testing.assert_close(
                torch.cat((
                    result_1[:context_sequence_lengths[0] +
                             context_sequence_lengths[1], :],
                    result_1[sum(context_sequence_lengths
                                 ):sum(context_sequence_lengths) + 2],
                )),
                torch.cat((
                    result_2[:context_sequence_lengths[0], :],
                    result_3[:context_sequence_lengths[1], :],
                    result_2[context_sequence_lengths[0]:, :],
                    result_3[context_sequence_lengths[1]:, :],
                )))

        kv_cache_manager.shutdown()

    @parameterized.expand([
        CUDAGraphTestScenario(
            batch_size=1,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            dtype=torch.float16,
        ),
        CUDAGraphTestScenario(
            batch_size=16,
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            dtype=torch.bfloat16,
        ),
        CUDAGraphTestScenario(
            batch_size=16,
            num_heads=32,
            num_kv_heads=[32, 16],
            head_dim=128,
            dtype=torch.bfloat16,
        ),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    def test_attention_with_cuda_graphs(
            self, test_scenario: CUDAGraphTestScenario) -> None:
        # This test exercises our CUDAGraph metadata class and makes sure
        # that the flashinfer attention layer is compatible with graph capture/replay.
        # We compare the CUDA graph results to the results without CUDA graph.
        batch_size = test_scenario.batch_size
        num_heads = test_scenario.num_heads
        num_kv_heads = test_scenario.num_kv_heads
        head_dim = test_scenario.head_dim
        dtype = test_scenario.dtype
        device = 'cuda'

        # For simplicity, just use 1 page per request in this example.
        tokens_per_block = 128
        past_seen_tokens = [
            random.randint(1, tokens_per_block - 1) for _ in range(batch_size)
        ]
        request_ids = list(range(batch_size))
        token_nums = (torch.tensor(past_seen_tokens) + 1).tolist()

        num_blocks = 16
        max_seq_len = tokens_per_block * num_blocks
        num_layers = 1 if isinstance(num_kv_heads, int) else len(num_kv_heads)
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

        gen_qs = []
        gen_ks = []
        gen_vs = []

        for i in range(num_layers):
            gen_qs.append([
                torch.randn(1, num_heads * head_dim, dtype=dtype, device=device)
                for _ in range(batch_size)
            ])

            kv_heads = num_kv_heads if isinstance(num_kv_heads,
                                                  int) else num_kv_heads[i]
            gen_ks.append([
                torch.randn(1, kv_heads * head_dim, dtype=dtype, device=device)
                for _ in range(batch_size)
            ])

            gen_vs.append([
                torch.randn(1, kv_heads * head_dim, dtype=dtype, device=device)
                for _ in range(batch_size)
            ])

        layers = []
        for i in range(num_layers):
            kv_heads = num_kv_heads if isinstance(num_kv_heads,
                                                  int) else num_kv_heads[i]
            layers.append(
                FlashInferAttention(
                    layer_idx=i,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_kv_heads=kv_heads,
                ))

        seq_lens = torch.ones((batch_size, ), dtype=torch.int)
        attn_metadata_ref = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )

        attn_metadata_ref.kv_cache_manager = kv_cache_manager

        workspace = torch.empty(1024 * 1024 * 128,
                                dtype=torch.int,
                                device='cuda')
        attn_metadata_cuda_graph = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=0,
            is_cuda_graph=True,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
            workspace_buffer=workspace,
            max_num_requests=batch_size,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
        )

        attn_metadata_ref.prepare()
        attn_metadata_cuda_graph.prepare()

        results_ref = []

        for i in range(num_layers):
            q = torch.cat(gen_qs[i])
            k = torch.cat(gen_ks[i])
            v = torch.cat(gen_vs[i])
            layer = layers[i]
            results_ref.append(layer.forward(q, k, v, attn_metadata_ref))

        graph = torch.cuda.CUDAGraph()
        for i in range(num_layers):
            layer = layers[i]
            q = torch.cat(gen_qs[i])
            k = torch.cat(gen_ks[i])
            v = torch.cat(gen_vs[i])
            # Warmup run, required by PT
            for _ in range(2):
                layer.forward(q, k, v, attn_metadata_cuda_graph)

        results_actual = []
        with torch.cuda.graph(graph):
            for i in range(num_layers):
                layer = layers[i]
                q = torch.cat(gen_qs[i])
                k = torch.cat(gen_ks[i])
                v = torch.cat(gen_vs[i])
                results_actual.append(
                    layer.forward(q, k, v, attn_metadata_cuda_graph))

        graph.replay()

        for result_actual, result_ref in zip(results_actual, results_ref):
            torch.testing.assert_close(result_actual,
                                       result_ref,
                                       atol=1e-2,
                                       rtol=0)

        kv_cache_manager.shutdown()

    def test_ragged_prefill_no_kv_cache_uses_cudnn_plan(self) -> None:
        """Ragged QKV with ``kv_cache_manager=None`` plans via cuDNN (``_plan_ragged_cudnn_no_kv``)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = torch.device("cuda")
        num_heads = 8
        num_kv_heads = 8
        head_dim = 80
        hidden_size = num_heads * head_dim
        dtype = torch.bfloat16
        per_sequence_token_counts = [24, 56]
        num_context_sequences = len(per_sequence_token_counts)
        total_tokens = sum(per_sequence_token_counts)

        attn_metadata = TestingFlashInferAttentionMetadata(
            max_num_requests=max(num_context_sequences, 128),
            max_num_tokens=max(total_tokens * 2, 8192),
            kv_cache_manager=None,
        )
        attn_metadata.seq_lens = torch.tensor(
            per_sequence_token_counts,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        attn_metadata.num_contexts = num_context_sequences
        attn_metadata.request_ids = list(range(1, num_context_sequences + 1))
        attn_metadata.prompt_lens = list(per_sequence_token_counts)
        attn_metadata.prepare()

        layer = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        query_states = torch.randn(
            total_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        key_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        value_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        attention_output = layer.forward(
            query_states,
            key_states,
            value_states,
            attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
        )

        self.assertEqual(attention_output.shape, (total_tokens, hidden_size))
        self.assertGreaterEqual(len(attn_metadata._plan_params_to_wrappers), 1)
        # FlashInfer stores the chosen implementation on the wrapper (see
        # BatchPrefillWithRaggedKVCacheWrapper.__init__: self._backend = backend).
        # TRT-LLM no-cache ragged path passes backend="cudnn" in flashinfer.py.
        for flashinfer_wrappers in attn_metadata._plan_params_to_wrappers.values(
        ):
            ragged_wrapper = flashinfer_wrappers.ragged_prefill_wrapper
            self.assertIsNotNone(ragged_wrapper)
            self.assertEqual(
                getattr(ragged_wrapper, "_backend", None),
                "cudnn",
                msg="No-KV ragged prefill should use FlashInfer's cudnn backend",
            )

    def test_ragged_prefill_no_kv_cache_with_cuda_graphs(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

        device = torch.device("cuda")
        num_heads = 8
        num_kv_heads = 8
        head_dim = 80
        hidden_size = num_heads * head_dim
        dtype = torch.bfloat16
        per_sequence_token_counts = [24, 56]
        num_context_sequences = len(per_sequence_token_counts)
        total_tokens = sum(per_sequence_token_counts)
        max_num_requests = 8

        seq_lens = torch.tensor(
            per_sequence_token_counts,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        request_ids = list(range(1, num_context_sequences + 1))

        attn_metadata_ref = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=num_context_sequences,
            max_num_requests=max_num_requests,
            max_num_tokens=max(total_tokens * 2, 8192),
            kv_cache_manager=None,
            request_ids=request_ids,
        )
        attn_metadata_cuda_graph = TestingFlashInferAttentionMetadata(
            seq_lens=seq_lens,
            num_contexts=num_context_sequences,
            is_cuda_graph=True,
            max_num_requests=max_num_requests,
            max_num_tokens=max(total_tokens * 2, 8192),
            kv_cache_manager=None,
            request_ids=request_ids,
        )
        attn_metadata_ref.prompt_lens = list(per_sequence_token_counts)
        attn_metadata_cuda_graph.prompt_lens = list(per_sequence_token_counts)
        attn_metadata_ref.prepare()
        attn_metadata_cuda_graph.prepare()

        layer = FlashInferAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        query_states = torch.randn(
            total_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        key_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        value_states = torch.randn(
            total_tokens,
            num_kv_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        result_ref = layer.forward(
            query_states,
            key_states,
            value_states,
            attn_metadata_ref,
            attention_mask=PredefinedAttentionMask.FULL,
        )

        graph = torch.cuda.CUDAGraph()
        for _ in range(2):
            layer.forward(
                query_states,
                key_states,
                value_states,
                attn_metadata_cuda_graph,
                attention_mask=PredefinedAttentionMask.FULL,
            )

        with torch.cuda.graph(graph):
            result_actual = layer.forward(
                query_states,
                key_states,
                value_states,
                attn_metadata_cuda_graph,
                attention_mask=PredefinedAttentionMask.FULL,
            )

        graph.replay()
        torch.testing.assert_close(result_actual, result_ref, atol=1e-2, rtol=0)
