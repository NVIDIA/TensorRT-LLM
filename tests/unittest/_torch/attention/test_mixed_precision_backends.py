# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local GPU QA: exercise the optimized backends, not a reference fallback."""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.mixed_precision_kv_view import encode_high_precision_kv
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.memory_buffer_utils import Buffers
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import MixedPrecisionKVCacheManager
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
    "high_dtype,compute_dtype,width,query_heads,heads,kv_layout",
    [
        (DataType.HALF, torch.float16, 64, 2, 2, "HND"),
        (DataType.BF16, torch.bfloat16, 128, 4, 2, "NHD"),
        (DataType.FP8, torch.bfloat16, 64, 4, 1, "NHD"),
    ],
    ids=("mha_fp16", "gqa_bf16_nhd", "mqa_fp8_nhd"),
)
def test_optimized_two_pool_prefill_and_decode(
    backend, high_dtype, compute_dtype, width, query_heads, heads, kv_layout
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native KVCM")
    block, length, batch = 128, 640, 2
    request_ids = list(range(batch))
    high_torch_dtype = binding_to_torch_dtype(high_dtype)
    manager = MixedPrecisionKVCacheManager(
        KvCacheConfig(max_tokens=65536, enable_block_reuse=True),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=heads,
        head_dim=width,
        tokens_per_block=block,
        max_seq_len=1024,
        max_batch_size=batch,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=high_dtype,
        low_dtype=DataType.NVFP4,
        compute_dtype=compute_dtype,
        high_low_pool_ratio=1 / 3,
        first_n=128,
        last_n=128,
    )
    generator = torch.Generator(device="cuda").manual_seed(123)
    query, key, value = [
        torch.randn(batch, length + 1, count * width, generator=generator, device="cuda").to(
            compute_dtype
        )
        for count in (query_heads, heads, heads)
    ]
    attention = backend(
        layer_idx=0,
        num_heads=query_heads,
        head_dim=width,
        num_kv_heads=heads,
        quant_config=QuantConfig(
            kv_cache_quant_algo="FP8" if high_dtype == DataType.FP8 else "NVFP4"
        ),
    )
    original_fp8_kv_cache = attention.has_fp8_kv_cache
    if backend is TrtllmAttention:
        original_state = (
            attention.quant_mode,
            attention.has_fp8_kv_cache,
            attention.has_fp4_kv_cache,
            attention.local_layer_idx,
            attention._fmha_manager,
        )
    scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    decode_graph = None

    def forward(begin, end, context):
        nonlocal decode_graph
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
            kv_layout=kv_layout,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
        metadata.prepare()
        q, k, v = [
            tensor[:, begin:end].reshape(-1, tensor.shape[-1]) for tensor in (query, key, value)
        ]
        inputs = (
            (torch.cat((q, k, v), dim=-1), None, None)
            if backend is TrtllmAttention
            else (q.clone(), k.clone(), v.clone())
        )

        def run():
            return attention.forward(*inputs, metadata, kv_scale_quant_orig=scale)

        output = run()
        if not context and high_dtype == DataType.BF16:
            reference = output.clone()
            metadata = metadata.create_cuda_graph_metadata(batch, buffers=Buffers())
            metadata.prepare()
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            # Replay with different physical pages, without recapturing.
            metadata.request_ids = request_ids[::-1]
            metadata.prepare()
            for tensor in inputs:
                if tensor is not None:
                    tensor.copy_(tensor.flip(0))
            graph.replay()
            torch.testing.assert_close(output, reference.flip(0), atol=0.01, rtol=0.02)
            decode_graph = (metadata, inputs, graph, output)
            output = output.flip(0)
        assert metadata.kv_cache_manager is manager and metadata.kv_layout == kv_layout
        return output

    def token_major(pool):
        return pool.transpose(2, 3) if kv_layout == "HND" else pool

    try:
        manager.add_dummy_requests(request_ids, [length] * batch, is_gen=False)
        output = forward(0, length, True)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[:, :length].reshape(batch, length, query_heads, width).transpose(1, 2),
            key[:, :length].reshape(batch, length, heads, width).transpose(1, 2),
            value[:, :length].reshape(batch, length, heads, width).transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )
        torch.testing.assert_close(
            output, reference.transpose(1, 2).reshape_as(output), atol=0.02, rtol=0.02
        )
        arena = manager._staging_buffer.data_ptr()
        history_before = token_major(manager.get_buffers(0, kv_layout=kv_layout)).clone()
        scales_before = token_major(
            manager.get_mixed_precision_buffers(0, request_ids, kv_layout)[1][0]
        ).clone()
        if backend is TrtllmAttention:
            selection_cache = attention._unquantized_fmha_manager
            assert selection_cache is not None
        target, implementation = (
            (selection_cache, "select")
            if backend is TrtllmAttention
            else (attention, "forward_impl")
        )
        with patch.object(target, implementation, side_effect=RuntimeError("test failure")):
            with pytest.raises(RuntimeError, match="test failure"):
                forward(0, length, True)
        assert attention.has_fp8_kv_cache == original_fp8_kv_cache
        if backend is TrtllmAttention:
            assert (
                attention.quant_mode,
                attention.has_fp8_kv_cache,
                attention.has_fp4_kv_cache,
                attention.local_layer_idx,
                attention._fmha_manager,
            ) == original_state
        torch.testing.assert_close(
            token_major(manager.get_buffers(0, kv_layout=kv_layout)), history_before, rtol=0, atol=0
        )
        for request_id in request_ids:
            cache = manager.kv_cache_map[request_id]
            assert cache.resize(None, length)
            cache.suspend()
            assert cache.resume(torch.cuda.current_stream().cuda_stream, is_decoding=True)
            assert cache.resize(length + 1)
            manager._restore_page_index_bufs(request_id, cache)
        output = forward(length, length + 1, False)
        assert manager._staging_buffer.data_ptr() == arena
        assert attention.has_fp8_kv_cache == original_fp8_kv_cache
        if backend is TrtllmAttention:
            assert attention._unquantized_fmha_manager is selection_cache
            assert (
                attention.quant_mode,
                attention.has_fp8_kv_cache,
                attention.has_fp4_kv_cache,
                attention.local_layer_idx,
                attention._fmha_manager,
            ) == original_state
        history = token_major(manager.get_buffers(0, kv_layout=kv_layout))
        all_history_ids = manager.get_batch_cache_indices(request_ids, 0)
        (high, all_high_ids), (scales, all_scale_ids) = manager.get_mixed_precision_buffers(
            0, request_ids, kv_layout
        )
        scales = token_major(scales)
        high = token_major(high)
        for row, (history_ids, high_ids) in enumerate(zip(all_history_ids, all_high_ids)):
            assert high_ids[0] >= 0 and high_ids[1:3] == [kv.BAD_PAGE_INDEX] * 2
            assert high_ids[-1] >= 0
            reference = []
            for role, tensor in enumerate((key, value)):
                original = tensor[row].view(-1, heads, width)
                packed, scale_bytes = torch.ops.trtllm.fp4_quantize(
                    original.contiguous(), scale.reciprocal(), 16, False, False
                )
                scale_bytes = scale_bytes.view(length + 1, heads, width // 16)
                low = _reference_dequant(packed, scale_bytes, scale).to(compute_dtype)
                high_values = original.to(high_torch_dtype).to(compute_dtype)
                expected = low.clone()
                for position in range(length + 1):
                    ordinal, offset = divmod(position, block)
                    torch.testing.assert_close(
                        history[history_ids[ordinal], role, offset].view(torch.uint8),
                        packed[position],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        scales[all_scale_ids[row][ordinal], role, offset],
                        scale_bytes[position],
                        rtol=0,
                        atol=0,
                    )
                    if high_ids[ordinal] >= 0:
                        expected[position] = high_values[position]
                        torch.testing.assert_close(
                            high[high_ids[ordinal], role, offset].to(compute_dtype),
                            high_values[position],
                            rtol=0,
                            atol=0,
                        )
                # The current step reads the fresh K/V before storing either copy.
                expected[-1] = original[-1]
                reference.append(expected.transpose(0, 1)[None])
            for ordinal in range(length // block):
                torch.testing.assert_close(
                    history[history_ids[ordinal]],
                    history_before[history_ids[ordinal]],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    scales[all_scale_ids[row][ordinal]],
                    scales_before[all_scale_ids[row][ordinal]],
                    rtol=0,
                    atol=0,
                )
            expected = torch.nn.functional.scaled_dot_product_attention(
                query[row, -1].view(1, query_heads, 1, width), *reference, enable_gqa=True
            ).reshape_as(output[row])
            torch.testing.assert_close(output[row], expected, atol=0.01, rtol=0.02)
        if decode_graph is not None:
            metadata, inputs, graph, captured_output = decode_graph
            for request_id in request_ids:
                cache = manager.kv_cache_map[request_id]
                assert cache.resize(None, length + 1)
                assert cache.resize(length + 2)
                manager._restore_page_index_bufs(request_id, cache)
            metadata.kv_cache_params.num_cached_tokens_per_seq = [length + 1] * batch
            metadata.prepare()
            graph.replay()
            replayed = captured_output.clone()
            eager = attention.forward(*inputs, metadata, kv_scale_quant_orig=scale)
            torch.testing.assert_close(replayed, eager, atol=0.01, rtol=0.02)
    finally:
        manager.shutdown()


def _reference_dequant(packed, scales, scale):
    """Independent E2M1/linear-scale oracle, used only by these tests."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long().flatten(-2)
    block_scales = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return table[codes] * block_scales * scale


@pytest.mark.parametrize("dtype,width", [(torch.float16, 64), (torch.bfloat16, 80)])
def test_nvfp4_conversion_kernels(dtype, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from flashinfer.quantization.fp4_quantization import (
        get_fp4_kv_dequantization_module,
        get_fp4_kv_quantization_module,
    )

    values = torch.linspace(-6, 6, 17 * width, device="cuda").view(17, width).to(dtype)
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    packed = torch.empty((17, width // 2), device="cuda", dtype=torch.uint8)
    scales = torch.empty((17, width // 16), device="cuda", dtype=torch.uint8)
    get_fp4_kv_quantization_module().nvfp4_kv_quant(values, scale, packed, scales)
    native_packed, native_scales = torch.ops.trtllm.fp4_quantize(
        values, scale.reciprocal(), 16, False, False
    )
    torch.testing.assert_close(packed, native_packed, rtol=0, atol=0)
    torch.testing.assert_close(scales.flatten(), native_scales, rtol=0, atol=0)
    decoded = torch.empty_like(values)
    get_fp4_kv_dequantization_module().nvfp4_kv_dequant(packed, scales, scale, decoded)
    expected = _reference_dequant(packed, scales, scale).to(dtype)
    torch.testing.assert_close(decoded.float(), expected.float(), rtol=0, atol=0)


def test_high_fp8_uses_native_saturating_conversion():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    values = torch.linspace(-1000, 1000, 128, device="cuda").view(2, 64)
    expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        values, torch.ones(1, device="cuda", dtype=torch.float32)
    )
    actual = encode_high_precision_kv(values, torch.float8_e4m3fn)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_mixed_precision_public_options():
    from pydantic import ValidationError

    config = KvCacheConfig(enable_mix_precision="nvfp4", dtype="fp8", high_low_pool_ratio=0.25)
    assert config.first_n == config.last_n == 256
    assert config.dtype == "fp8"
    for invalid in ({"enable_mix_precision": "int4"}, {"first_n": -1}, {"high_low_pool_ratio": 0}):
        with pytest.raises(ValidationError):
            KvCacheConfig(**invalid)


@pytest.mark.parametrize(
    "backend", (TrtllmAttention, FlashInferAttention), ids=("trtllm", "flashinfer")
)
@torch.inference_mode()
def test_mla_uses_independent_rope_and_native_attention(backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.mla import MLA
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.functional import PositionEmbeddingType

    torch.manual_seed(123)
    manager = MixedPrecisionKVCacheManager(
        KvCacheConfig(max_tokens=65536),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=1,
        mapping=Mapping(),
        dtype=DataType.BF16,
        compute_dtype=torch.bfloat16,
        first_n=128,
        last_n=128,
        high_low_pool_ratio=1 / 3,
    )
    model = MLA(
        hidden_size=256,
        num_attention_heads=16,
        num_key_value_heads=16,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=128,
        kv_lora_rank=512,
        predicted_tokens_per_seq=1,
        max_position_embeddings=1024,
        bias=False,
        layer_idx=0,
        pos_embd_params=PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams(dim=64, max_positions=1024, duplicate_data=True),
            is_neox=False,
        ),
        config=ModelConfig(attn_backend="TRTLLM" if backend is TrtllmAttention else "FLASHINFER"),
        dtype=torch.bfloat16,
    ).cuda()
    for parameter in model.parameters():
        parameter.uniform_(-0.02, 0.02)
    hidden = torch.randn(513, 256, device="cuda", dtype=torch.bfloat16)
    try:
        manager.add_dummy_requests([0], [256], is_gen=False)
        cache = manager.kv_cache_map[0]
        for begin, end in ((0, 256), (256, 512), (512, 513)):
            if begin:
                assert cache.resize(None, begin)
                cache.suspend()
                assert cache.resume(torch.cuda.current_stream().cuda_stream, is_decoding=end == 513)
                assert cache.resize(end)
                manager._restore_page_index_bufs(0, cache)
            metadata = backend.Metadata(
                seq_lens=torch.tensor([end - begin], dtype=torch.int32),
                num_contexts=int(end != 513),
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[begin]),
                max_num_requests=1,
                max_num_tokens=512,
                kv_cache_manager=manager,
                request_ids=[0],
                prompt_lens=[512],
                kv_layout="HND",
                enable_context_mla_with_cached_kv=True,
                runtime_features=AttentionRuntimeFeatures(
                    chunked_prefill=True, cache_reuse=True, chunk_size=256
                ),
            )
            metadata.prepare()
            output = hidden.new_empty((end - begin, 16 * 128))
            # These fused writes would target the stored layout before conversion.
            with (
                patch.object(
                    model.mha,
                    "mla_rope_append_paged_kv_assign_q",
                    side_effect=AssertionError("fused write"),
                    create=True,
                ),
                patch.object(
                    model.mqa,
                    "mla_rope_generation",
                    side_effect=AssertionError("fused write"),
                    create=True,
                ),
            ):
                model.forward_impl(
                    torch.arange(begin, end, device="cuda"), hidden[begin:end], metadata, [output]
                )
            assert torch.isfinite(output).all()
            assert metadata.kv_cache_manager is manager
        (high, ids), _ = manager.get_mixed_precision_buffers(0, [0], "NHD")
        assert high.shape[1] == 1
        assert ids[0][0] >= 0 and ids[0][1] == kv.BAD_PAGE_INDEX and ids[0][-1] >= 0
    finally:
        manager.shutdown()
