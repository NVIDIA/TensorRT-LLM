# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare dense fallback context plans with regular execution on real KV pages."""

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("cached_tokens", [0, 17], ids=["fresh", "cached_prefix"])
def test_fallback_context_plan_matches_output_and_kv_pages(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, cached_tokens: int
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    num_heads, num_kv_heads, head_dim, page_size = 4, 2, 128, 32
    lengths = [31 + index for index in range(batch_size)]
    request_ids = list(range(batch_size))
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=batch_size * 64, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=64,
        max_batch_size=batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    try:
        manager.add_dummy_requests(request_ids, [cached_tokens + n for n in lengths])
        generator = torch.Generator(device="cuda").manual_seed(22218)
        layers = [
            TrtllmAttention(
                layer_idx=index,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                cache_context_plan=True,
            )
            for index in range(2)
        ]
        pools = [manager.get_buffers(index, kv_layout="HND") for index in range(2)]
        for pool in pools:
            pool.copy_(
                torch.randn(pool.shape, device="cuda", dtype=pool.dtype, generator=generator)
            )
        initial_pools = [pool.clone() for pool in pools]
        qkvs = [
            torch.randn(
                sum(lengths),
                (num_heads + 2 * num_kv_heads) * head_dim,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in layers
        ]
        gate = torch.ones(sum(lengths), num_heads * head_dim, device="cuda", dtype=torch.bfloat16)

        def run(planned: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            for pool, initial in zip(pools, initial_pools):
                pool.copy_(initial)
            metadata = TrtllmAttentionMetadata(
                max_num_requests=batch_size,
                max_num_tokens=sum(lengths),
                kv_cache_manager=manager,
                runtime_features=AttentionRuntimeFeatures(chunked_prefill=True),
                num_contexts=batch_size,
                seq_lens=torch.tensor(lengths, dtype=torch.int32),
                request_ids=request_ids,
                prompt_lens=lengths,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[cached_tokens] * batch_size
                ),
            )
            metadata.prepare()
            outputs = []
            shared_plan = None
            for layer, qkv in zip(layers, qkvs):
                layer.cache_context_plan = planned
                fwd = AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only, output_gate=gate
                )
                output = layer.forward(qkv.clone(), None, None, metadata, fwd)
                assert isinstance(output, torch.Tensor)
                assert torch.isfinite(output).all()
                assert not fwd.output_gate_applied
                outputs.append(output.clone())
                if planned:
                    assert metadata.context_plan.fmha_cls is FallbackFmha
                    if shared_plan is None:
                        shared_plan = metadata.context_plan
                    else:
                        assert metadata.context_plan is shared_plan
                else:
                    assert metadata.context_plan is None
            torch.cuda.synchronize()
            return outputs, [pool.clone() for pool in pools]

        with torch.no_grad():
            regular_outputs, regular_pools = run(False)
            planned_outputs, planned_pools = run(True)
        for regular, planned in zip(regular_outputs, planned_outputs):
            torch.testing.assert_close(planned, regular, rtol=1e-3, atol=1e-3)
        for regular, planned in zip(regular_pools, planned_pools):
            torch.testing.assert_close(planned, regular, rtol=0, atol=0)
        for regular, initial in zip(regular_pools, initial_pools):
            assert not torch.equal(regular, initial), "regular context must update the KV cache"
    finally:
        manager.shutdown()
