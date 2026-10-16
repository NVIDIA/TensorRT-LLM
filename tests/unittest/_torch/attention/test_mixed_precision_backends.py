# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local GPU QA: exercise the optimized backends, not a reference fallback."""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention.kv_cache_conversion import (
    dequantize_nvfp4_kv,
    encode_high_precision_kv,
    quantize_nvfp4_kv,
)
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
@pytest.mark.parametrize("high_dtype", (DataType.HALF, DataType.BF16, DataType.FP8))
@pytest.mark.parametrize("compute_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("width", (64, 128))
def test_optimized_two_pool_prefill_and_decode(backend, high_dtype, compute_dtype, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native KVCM")
    block, heads, length, batch = 128, 2, 640, 2
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
        torch.randn(batch, length + 1, heads * width, generator=generator, device="cuda").to(
            compute_dtype
        )
        for _ in range(3)
    ]
    attention = backend(
        layer_idx=0,
        num_heads=heads,
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
            return attention.forward(qkv, None, None, metadata, kv_scale_quant_orig=scale)
        return attention.forward(q, k, v, metadata, kv_scale_quant_orig=scale)

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
        history_before = manager.get_buffers(0, kv_layout="HND").clone()
        scales_before = manager.get_mixed_precision_buffers(0, request_ids, "HND")[1][0].clone()
        if backend is TrtllmAttention:
            selection_cache = attention._unquantized_fmha_manager
            assert selection_cache is not None
        implementation = "_forward_impl" if backend is TrtllmAttention else "forward_impl"
        with patch.object(attention, implementation, side_effect=RuntimeError("test failure")):
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
            manager.get_buffers(0, kv_layout="HND"), history_before, rtol=0, atol=0
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
        history = manager.get_buffers(0, kv_layout="HND").transpose(2, 3)
        all_history_ids = manager.get_batch_cache_indices(request_ids, 0)
        (high, all_high_ids), (scales, all_scale_ids) = manager.get_mixed_precision_buffers(
            0, request_ids, "HND"
        )
        scales = scales.transpose(2, 3)
        high = high.transpose(2, 3)
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
                    history[history_ids[ordinal]].transpose(1, 2),
                    history_before[history_ids[ordinal]],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    scales[all_scale_ids[row][ordinal]].transpose(1, 2),
                    scales_before[all_scale_ids[row][ordinal]],
                    rtol=0,
                    atol=0,
                )
            expected = torch.nn.functional.scaled_dot_product_attention(
                query[row, -1].view(1, heads, 1, width), *reference
            ).reshape_as(output[row])
            torch.testing.assert_close(output[row], expected, atol=0.01, rtol=0.02)
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


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float8_e4m3fn))
@pytest.mark.parametrize("width", (64, 80, 128))
def test_nvfp4_conversion_kernels(dtype, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    values = torch.linspace(-6, 6, 17 * width, device="cuda").view(17, width).to(dtype)
    scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    packed, scales = quantize_nvfp4_kv(values, scale)
    native_packed, native_scales = torch.ops.trtllm.fp4_quantize(
        values, scale.reciprocal(), 16, False, False
    )
    torch.testing.assert_close(packed, native_packed, rtol=0, atol=0)
    torch.testing.assert_close(scales.flatten(), native_scales, rtol=0, atol=0)
    decoded = dequantize_nvfp4_kv(packed, scales, scale, dtype)
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
