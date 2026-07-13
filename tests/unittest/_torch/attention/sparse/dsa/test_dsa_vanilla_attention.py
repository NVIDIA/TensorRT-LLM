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

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import AttentionInputType, MLAParams
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSAParams
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention, VanillaAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_metadata(
    manager: KVCacheManager,
    request_ids: list[int],
    seq_lens: list[int],
    cached_lens: list[int],
    num_contexts: int,
) -> VanillaAttentionMetadata:
    metadata = VanillaAttentionMetadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=request_ids,
        max_num_requests=len(request_ids),
        num_contexts=num_contexts,
        max_num_tokens=sum(seq_lens),
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_lens,
        ),
    )
    metadata.prepare()
    return metadata


def _make_singleton_topk(selections: list[int], device: torch.device) -> torch.Tensor:
    topk = torch.full(
        (len(selections), 4),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk[:, 0] = torch.tensor(selections, dtype=torch.int32, device=device)
    return topk


def _repeat_for_query_heads(values: torch.Tensor, num_heads: int) -> torch.Tensor:
    return values.unsqueeze(1).expand(-1, num_heads, -1).reshape(values.shape[0], -1)


def test_dsa_selected_mla_context_generation_and_mixed_phases() -> None:
    """Exercise ragged, paged DSA cache access with an analytic singleton oracle."""
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = 4
    kv_lora_rank = 8
    qk_nope_head_dim = 8
    qk_rope_head_dim = 4
    fused_head_dim = kv_lora_rank + qk_rope_head_dim
    context_lens = [5, 3]
    generation_len = 2
    request_ids = [0, 1]
    final_lens = [length + generation_len for length in context_lens]
    allocated_lens = [final_lens[0], final_lens[1] + 1]

    manager = KVCacheManager(
        KvCacheConfig(max_tokens=32, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=fused_head_dim,
        tokens_per_block=4,
        max_seq_len=max(allocated_lens),
        max_batch_size=len(request_ids),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.BF16,
    )
    manager.add_dummy_requests(request_ids, allocated_lens)

    attention = VanillaAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=fused_head_dim,
        num_kv_heads=1,
        q_scaling=1.25,
        mla_params=MLAParams(
            q_lora_rank=8,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=kv_lora_rank,
        ),
        sparse_params=DSAParams(),
    )

    try:
        num_context_tokens = sum(context_lens)
        context_q = torch.randn(
            num_context_tokens,
            num_heads * fused_head_dim,
            device=device,
            dtype=dtype,
        )
        context_latent = torch.randn(
            num_context_tokens,
            fused_head_dim,
            device=device,
            dtype=dtype,
        )
        context_metadata = _make_metadata(
            manager,
            request_ids,
            context_lens,
            cached_lens=[0, 0],
            num_contexts=len(request_ids),
        )
        context_output = attention.forward(
            context_q,
            None,
            None,
            context_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=context_latent,
            topk_indices=_make_singleton_topk(
                [0, 1, 2, 3, 4, 0, 1, 2],
                device,
            ),
        )
        context_expected = _repeat_for_query_heads(
            context_latent[:, :kv_lora_rank],
            num_heads,
        )
        torch.testing.assert_close(context_output, context_expected)

        num_generation_tokens = len(request_ids) * generation_len
        generation_q = torch.randn(
            num_generation_tokens,
            num_heads * fused_head_dim,
            device=device,
            dtype=dtype,
        )
        generation_latent = torch.randn(
            num_generation_tokens,
            fused_head_dim,
            device=device,
            dtype=dtype,
        )
        generation_metadata = _make_metadata(
            manager,
            request_ids,
            [generation_len] * len(request_ids),
            cached_lens=context_lens,
            num_contexts=0,
        )
        generation_output = attention.forward(
            generation_q,
            None,
            None,
            generation_metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=generation_latent,
            topk_indices=_make_singleton_topk([4, 6, 0, 4], device),
        )
        request_0_cache = torch.cat((context_latent[:5], generation_latent[:2]))
        request_1_cache = torch.cat((context_latent[5:], generation_latent[2:]))
        generation_selected_values = torch.stack(
            (
                request_0_cache[4, :kv_lora_rank],
                request_0_cache[6, :kv_lora_rank],
                request_1_cache[0, :kv_lora_rank],
                request_1_cache[4, :kv_lora_rank],
            )
        )
        generation_expected = _repeat_for_query_heads(
            generation_selected_values,
            num_heads,
        )
        torch.testing.assert_close(generation_output, generation_expected)

        mixed_metadata = _make_metadata(
            manager,
            request_ids,
            [2, 1],
            cached_lens=[0, final_lens[1]],
            num_contexts=1,
        )
        mixed_context_q = torch.randn(
            2,
            num_heads * fused_head_dim,
            device=device,
            dtype=dtype,
        )
        mixed_context_latent = torch.randn(
            2,
            fused_head_dim,
            device=device,
            dtype=dtype,
        )
        mixed_context_output = attention.forward(
            mixed_context_q,
            None,
            None,
            mixed_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=mixed_context_latent,
            topk_indices=_make_singleton_topk([0, 1], device),
        )
        mixed_context_expected = _repeat_for_query_heads(
            mixed_context_latent[:, :kv_lora_rank],
            num_heads,
        )
        torch.testing.assert_close(mixed_context_output, mixed_context_expected)

        mixed_generation_q = torch.randn(
            1,
            num_heads * fused_head_dim,
            device=device,
            dtype=dtype,
        )
        mixed_generation_latent = torch.randn(
            1,
            fused_head_dim,
            device=device,
            dtype=dtype,
        )
        mixed_generation_output = attention.forward(
            mixed_generation_q,
            None,
            None,
            mixed_metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=mixed_generation_latent,
            topk_indices=_make_singleton_topk([final_lens[1]], device),
        )
        mixed_generation_expected = _repeat_for_query_heads(
            mixed_generation_latent[:, :kv_lora_rank],
            num_heads,
        )
        torch.testing.assert_close(mixed_generation_output, mixed_generation_expected)
    finally:
        manager.shutdown()
