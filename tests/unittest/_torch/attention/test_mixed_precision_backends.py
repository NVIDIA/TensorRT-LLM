# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local GPU QA: exercise the optimized backends, not a reference fallback."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.mixed_precision_cache import (
    MixedPrecisionKVCacheManager,
)
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv


@pytest.mark.parametrize(
    "backend", (TrtllmAttention, FlashInferAttention), ids=("trtllm", "flashinfer")
)
@pytest.mark.parametrize(
    "history_dtype,high_dtype",
    ((DataType.FP8, DataType.BF16), (DataType.FP8, DataType.FP8), (DataType.BF16, DataType.BF16)),
)
@pytest.mark.parametrize("width", (64, 128))
def test_optimized_two_pool_prefill_and_decode(backend, history_dtype, high_dtype, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native KVCM")
    block, heads, length, batch = 128, 2, 640, 2
    request_ids = list(range(batch))
    high_torch_dtype = binding_to_torch_dtype(high_dtype)
    manager = MixedPrecisionKVCacheManager(
        KvCacheConfig(max_tokens=8192, enable_block_reuse=True, pool_ratio=[0.75, 0.25]),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=heads,
        head_dim=width,
        tokens_per_block=block,
        max_seq_len=1024,
        max_batch_size=batch,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=history_dtype,
        high_dtype=high_dtype,
        first_n=128,
        last_n=128,
    )
    generator = torch.Generator(device="cuda").manual_seed(123)
    query, key, value = [
        torch.randn(batch, length + 1, heads * width, generator=generator, device="cuda").to(
            torch.bfloat16
        )
        for _ in range(3)
    ]
    attention = backend(
        layer_idx=0,
        num_heads=heads,
        head_dim=width,
        num_kv_heads=heads,
        quant_config=QuantConfig(
            kv_cache_quant_algo="FP8" if history_dtype == DataType.FP8 else None
        ),
    )

    def forward(begin, end, context):
        metadata = backend.Metadata(
            seq_lens=torch.tensor([end - begin] * batch, dtype=torch.int32),
            num_contexts=batch if context else 0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[begin] * batch
            ),
            max_num_requests=batch,
            max_num_tokens=batch * (length + 1),
            kv_cache_manager=manager,
            request_ids=request_ids,
            prompt_lens=[end] * batch,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
        metadata.prepare()
        q, k, v = [
            tensor[:, begin:end].reshape(-1, heads * width) for tensor in (query, key, value)
        ]
        if backend is TrtllmAttention:
            qkv = torch.cat((q, k, v), dim=-1)
            return attention.forward(qkv, None, None, metadata)
        return attention.forward(q, k, v, metadata)

    try:
        manager.add_dummy_requests(request_ids, [length] * batch, is_gen=False)
        output = forward(0, length, True)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[:, :length].reshape(batch, length, heads, width).transpose(1, 2),
            key[:, :length].reshape(batch, length, heads, width).transpose(1, 2),
            value[:, :length].reshape(batch, length, heads, width).transpose(1, 2),
            is_causal=True,
        )
        torch.testing.assert_close(
            output, reference.transpose(1, 2).reshape_as(output), atol=0.02, rtol=0.02
        )
        arena = manager._staging_buffer.data_ptr()
        for request_id in request_ids:
            cache = manager.kv_cache_map[request_id]
            assert cache.resize(None, length)
            cache.suspend()
            assert cache.resume(torch.cuda.current_stream().cuda_stream, is_decoding=True)
            assert cache.resize(length + 1)
            manager._restore_page_index_bufs(request_id, cache)
        output = forward(length, length + 1, False)
        assert manager._staging_buffer.data_ptr() == arena
        history = manager.get_buffers(0, kv_layout="HND").transpose(2, 3)
        all_history_ids = manager.get_batch_cache_indices(request_ids, 0)
        high, all_high_ids = manager.get_high_precision_window(0, request_ids, "HND")
        high = high.transpose(2, 3)
        for row, (history_ids, high_ids) in enumerate(zip(all_history_ids, all_high_ids)):
            assert high_ids[0] >= 0 and high_ids[1:3] == [kv.BAD_PAGE_INDEX] * 2
            assert high_ids[-1] >= 0
            reference = []
            for role, tensor in enumerate((key, value)):
                original = tensor[row].view(-1, heads, width)
                low = original.to(history.dtype).to(torch.bfloat16)
                high_values = original.to(high_torch_dtype).to(torch.bfloat16)
                expected = low.clone()
                for position in range(length + 1):
                    ordinal, offset = divmod(position, block)
                    torch.testing.assert_close(
                        history[history_ids[ordinal], role, offset].to(torch.bfloat16),
                        low[position],
                        rtol=0,
                        atol=0,
                    )
                    if high_ids[ordinal] >= 0:
                        expected[position] = high_values[position]
                        torch.testing.assert_close(
                            high[high_ids[ordinal], role, offset].to(torch.bfloat16),
                            high_values[position],
                            rtol=0,
                            atol=0,
                        )
                reference.append(expected.transpose(0, 1)[None])
            expected = torch.nn.functional.scaled_dot_product_attention(
                query[row, -1].view(1, heads, 1, width), *reference
            ).reshape_as(output[row])
            torch.testing.assert_close(output[row], expected, atol=0.01, rtol=0.02)
    finally:
        manager.shutdown()
