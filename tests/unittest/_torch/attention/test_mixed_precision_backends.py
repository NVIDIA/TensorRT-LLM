# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise mixed-precision cache routing and optimized Attention backends."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
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


@pytest.mark.parametrize("scratch_reuse", [False, True])
@pytest.mark.parametrize(
    "backend,high_dtype,compute_dtype,width,query_heads,heads,kv_layout,fp8_input,block",
    [
        (TrtllmAttention, DataType.HALF, torch.float16, 64, 2, 2, "HND", False, 128),
        (TrtllmAttention, DataType.BF16, torch.bfloat16, 128, 4, 2, "NHD", False, 128),
        (TrtllmAttention, DataType.FP8, torch.bfloat16, 64, 4, 1, "NHD", False, 128),
        (FlashInferAttention, DataType.HALF, torch.float16, 64, 2, 2, "HND", False, 128),
        (FlashInferAttention, DataType.BF16, torch.bfloat16, 128, 4, 2, "NHD", False, 128),
        (FlashInferAttention, DataType.FP8, torch.bfloat16, 64, 4, 1, "NHD", False, 128),
        (FlashInferAttention, DataType.FP8, torch.bfloat16, 64, 4, 1, "NHD", True, 128),
        (TrtllmAttention, DataType.BF16, torch.bfloat16, 128, 4, 2, "HND", False, 32),
    ],
    ids=[
        "mha_fp16-trtllm",
        "gqa_bf16_nhd-trtllm",
        "mqa_fp8_nhd-trtllm",
        "mha_fp16-flashinfer",
        "gqa_bf16_nhd-flashinfer",
        "mqa_fp8_nhd-flashinfer",
        "mqa_fp8_nhd-flashinfer-fused_fp8",
        "gqa_bf16_native_p32-trtllm",
    ],
)
def test_optimized_two_pool_prefill_and_decode(
    backend,
    high_dtype,
    compute_dtype,
    width,
    query_heads,
    heads,
    kv_layout,
    fp8_input,
    block,
    scratch_reuse,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    length, batch = 640, 2
    native_interleave = backend is FlashInferAttention or block == 32
    request_ids = list(range(batch))
    high_torch_dtype = binding_to_torch_dtype(high_dtype)
    manager = MixedPrecisionKVCacheManager(
        KvCacheConfig(
            max_tokens=65536, enable_block_reuse=True, enable_swa_scratch_reuse=scratch_reuse
        ),
        CacheType.SELF,
        num_layers=4 if native_interleave else 2,
        num_kv_heads=heads,
        head_dim=width,
        tokens_per_block=block,
        max_seq_len=1024,
        max_batch_size=batch,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=high_dtype,
        low_dtype=DataType.NVFP4,
        high_low_pool_ratio=1 / 3,
        first_n=block,
        last_n=block,
        pretrained_config=SimpleNamespace(
            num_hidden_layers=4,
            num_kv_shared_layers=1,
            layer_types=["full_attention"] * 4,
        )
        if native_interleave
        else None,
    )
    generator = torch.Generator(device="cuda").manual_seed(123)
    query, key, value = [
        torch.randn(batch, length + 1, count * width, generator=generator, device="cuda").to(
            compute_dtype
        )
        for count in (query_heads, heads, heads)
    ]
    if high_dtype == DataType.FP8:
        query = query.to(torch.float8_e4m3fn).to(compute_dtype)
    if fp8_input:
        key, value = [tensor.to(torch.float8_e4m3fn).to(compute_dtype) for tensor in (key, value)]
    quant_config = QuantConfig(kv_cache_quant_algo="FP8" if high_dtype == DataType.FP8 else None)
    original_config = quant_config.model_dump()
    original_quant_mode = quant_config.quant_mode
    original_layer_quant_mode = quant_config.layer_quant_mode
    attentions = [
        backend(
            layer_idx=layer_idx,
            num_heads=query_heads,
            head_dim=width,
            num_kv_heads=heads,
            quant_config=quant_config,
            **({"flashinfer_backend": "trtllm-gen"} if fp8_input else {}),
        )
        for layer_idx in ((0, 2, 1) if native_interleave else (0, 1))
    ]
    attention = attentions[0]
    selection_cache = attention._fmha_manager if backend is TrtllmAttention else None
    history_scale = torch.full(
        (1,), 1 / 6 if high_dtype == DataType.FP8 else 1, dtype=torch.float32, device="cuda"
    )
    module = SimpleNamespace(
        attn=attention,
        mapping=manager.mapping,
        quant_config=quant_config,
        sparse_attn_hooks=None,
        register_to_config=False,
        _use_quantize_output=lambda: False,
    )
    module._attn_impl = partial(Attention._attn_impl, module)
    decode_graph = None
    staging_address = None
    native_output = None

    def forward(begin, end, context):
        nonlocal decode_graph, staging_address, native_output
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
        assert metadata.mixed_precision_kv_views == {}
        metadata.prepare()
        view = metadata.mixed_precision_kv_views[0]
        assert metadata.mixed_precision_kv_views[1] is view
        assert 2 not in metadata.mixed_precision_kv_views
        if backend is FlashInferAttention:
            assert metadata._vswa_layer_to_pool[0] == metadata._vswa_layer_to_pool[1]
            assert metadata._vswa_layer_to_pool[0] != metadata._vswa_layer_to_pool[2]
        assert view.get_buffers(0).dtype == high_torch_dtype
        if staging_address is None:
            staging_address = view.get_buffers(0).data_ptr()
        assert view.get_buffers(0).data_ptr() == staging_address
        assert not hasattr(view, "metadata")
        q, k, v = [
            tensor[:, begin:end].reshape(-1, tensor.shape[-1]) for tensor in (query, key, value)
        ]
        input_dtype = torch.float8_e4m3fn if fp8_input else compute_dtype
        inputs = (
            (torch.cat((q, k, v), dim=-1), None, None)
            if backend is TrtllmAttention
            else tuple(tensor.to(input_dtype, copy=True) for tensor in (q, k, v))
        )
        other_inputs = (
            (torch.cat((q, k, -v), dim=-1), None, None)
            if backend is TrtllmAttention
            else tuple(tensor.to(input_dtype, copy=True) for tensor in (q, k, -v))
        )
        layer_outputs = None

        def run():
            nonlocal layer_outputs
            layer_outputs = []
            seq_lens_cuda = metadata.seq_lens_cuda
            kv_cache_params = metadata.kv_cache_params
            layer_inputs_list = (
                (inputs, inputs, other_inputs) if native_interleave else (inputs, other_inputs)
            )
            try:
                for layer_attention, layer_inputs in zip(attentions, layer_inputs_list):
                    module.attn = layer_attention
                    with patch.object(
                        layer_attention, "forward", wraps=layer_attention.forward
                    ) as backend_forward:
                        layer_outputs.append(
                            Attention.forward_impl(
                                module,
                                *layer_inputs,
                                metadata,
                                PredefinedAttentionMask.CAUSAL,
                                None,
                                None,
                                None,
                            )
                        )
                    assert backend_forward.call_args.args[3] is metadata
                    assert metadata.seq_lens_cuda is seq_lens_cuda
                    assert metadata.kv_cache_params is kv_cache_params
                    assert metadata.kv_cache_manager is manager
                    assert metadata.kv_layout == kv_layout
                    if backend is FlashInferAttention:
                        assert (
                            metadata._vswa_active_pool_id
                            == (metadata._vswa_layer_to_pool[layer_attention.layer_idx])
                        )
            finally:
                module.attn = attention
            return layer_outputs[0]

        output = run()
        native_output = layer_outputs[1].clone() if native_interleave else None
        torch.testing.assert_close(layer_outputs[-1], -output, atol=0.01, rtol=0.02)
        if context and backend is TrtllmAttention:
            native_offsets = metadata.kv_cache_block_offsets
            paged_context = metadata.use_paged_context_fmha
            with (
                patch.object(attention, "forward", side_effect=RuntimeError("backend failure")),
                pytest.raises(RuntimeError, match="backend failure"),
            ):
                Attention.forward_impl(
                    module, *inputs, metadata, PredefinedAttentionMask.CAUSAL, None, None, None
                )
            assert metadata.kv_cache_manager is manager
            assert metadata.kv_layout == kv_layout
            assert metadata.kv_cache_block_offsets is native_offsets
            assert metadata.use_paged_context_fmha == paged_context
        if not context and high_dtype == DataType.BF16:
            reference = output.clone()
            eager_metadata = metadata
            metadata = metadata.create_cuda_graph_metadata(batch, buffers=Buffers())
            metadata.prepare()
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            # Replay with different physical pages, without recapturing.
            metadata.request_ids = request_ids[::-1]
            metadata.prepare()
            for tensor in (*inputs, *other_inputs):
                if tensor is not None:
                    tensor.copy_(tensor.flip(0))
            other_metadata = eager_metadata.create_cuda_graph_metadata(batch, buffers=Buffers())
            other_metadata.prepare()
            view = metadata.mixed_precision_kv_views[0]
            other_view = other_metadata.mixed_precision_kv_views[0]
            assert other_view is not view
            for tensor, other_tensor in (
                (view.page_tables[0], other_view.page_tables[0]),
                (view.page_tables[1], other_view.page_tables[1]),
                (view.lengths, other_view.lengths),
            ):
                assert tensor.untyped_storage().data_ptr() == (
                    other_tensor.untyped_storage().data_ptr()
                )
            # Another graph's preparation changed the shared tables; refresh before replay.
            metadata.prepare()
            graph.replay()
            torch.testing.assert_close(output, reference.flip(0), atol=0.01, rtol=0.02)
            if native_interleave:
                torch.testing.assert_close(
                    layer_outputs[1], native_output.flip(0), atol=0.01, rtol=0.02
                )
            torch.testing.assert_close(layer_outputs[-1], -output, atol=0.01, rtol=0.02)
            decode_graph = (metadata, graph, output, layer_outputs[-1], run)
            output = output.flip(0)
        assert metadata.kv_cache_manager is manager and metadata.kv_layout == kv_layout
        return output

    def token_major(pool):
        return pool.transpose(2, 3) if kv_layout == "HND" else pool

    def assert_native_config() -> None:
        assert quant_config.model_dump() == original_config
        assert quant_config.quant_mode is original_quant_mode
        assert quant_config.layer_quant_mode == original_layer_quant_mode
        assert attention.quant_config is quant_config
        assert attention.has_fp8_kv_cache == (high_dtype == DataType.FP8)
        if backend is TrtllmAttention:
            assert attention.quant_mode == int(original_layer_quant_mode)
            assert not attention.has_fp4_kv_cache
            assert attention._fmha_manager is selection_cache

    try:
        manager.add_dummy_requests(request_ids, [length] * batch, is_gen=False)
        assert all(manager.kv_cache_map[i].has_scratch_slots == scratch_reuse for i in request_ids)
        output = forward(0, length, True)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[:, :length].reshape(batch, length, query_heads, width).transpose(1, 2),
            key[:, :length]
            .to(high_torch_dtype)
            .to(compute_dtype)
            .reshape(batch, length, heads, width)
            .transpose(1, 2),
            value[:, :length]
            .to(high_torch_dtype)
            .to(compute_dtype)
            .reshape(batch, length, heads, width)
            .transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )
        # Match test_attention.py for native FP8 FMHA, which also quantizes P.
        torch.testing.assert_close(
            output,
            reference.transpose(1, 2).reshape_as(output),
            atol=0.05 if high_dtype == DataType.FP8 else 0.02,
            rtol=0.001 if high_dtype == DataType.FP8 else 0.02,
        )
        if native_interleave:
            torch.testing.assert_close(
                native_output,
                reference.transpose(1, 2).reshape_as(native_output),
                atol=0.05 if high_dtype == DataType.FP8 else 0.02,
                rtol=0.001 if high_dtype == DataType.FP8 else 0.02,
            )
        history_before = token_major(manager.get_buffers(0, kv_layout=kv_layout)).clone()
        assert_native_config()
        if backend is TrtllmAttention:
            assert selection_cache is not None
        target, implementation = (
            (selection_cache, "select")
            if backend is TrtllmAttention
            else (attention, "forward_impl")
        )
        with patch.object(target, implementation, side_effect=RuntimeError("test failure")):
            with pytest.raises(RuntimeError, match="test failure"):
                forward(0, length, True)
        assert_native_config()
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
        assert_native_config()
        if native_interleave:
            native_reference = torch.nn.functional.scaled_dot_product_attention(
                query[:, -1].reshape(batch, 1, query_heads, width).transpose(1, 2),
                key.to(high_torch_dtype)
                .to(compute_dtype)
                .reshape(batch, -1, heads, width)
                .transpose(1, 2),
                value.to(high_torch_dtype)
                .to(compute_dtype)
                .reshape(batch, -1, heads, width)
                .transpose(1, 2),
                enable_gqa=True,
            ).reshape_as(native_output)
            torch.testing.assert_close(native_output, native_reference, atol=0.01, rtol=0.02)
        history = token_major(manager.get_buffers(0, kv_layout=kv_layout))
        all_history_ids, all_high_ids = manager.get_mixed_precision_page_tables(request_ids, 0)
        high = manager.get_buffers(0, kv_layout, mixed_precision_pool="high")
        high = token_major(high)
        packed_bytes, scale_bytes_per_head = width // 2, width // 16
        payload_bytes = packed_bytes + scale_bytes_per_head
        assert history.shape[1:] == (2, block, heads, (payload_bytes + 7) // 8 * 8)
        for row, (history_ids, high_ids) in enumerate(zip(all_history_ids, all_high_ids)):
            assert high_ids[0] >= 0 and high_ids[1:3] == [kv.BAD_PAGE_INDEX] * 2
            assert high_ids[-1] >= 0
            reference = []
            for role, tensor in enumerate((key, value)):
                original = tensor[row].view(-1, heads, width)
                native_high_values = original.to(high_torch_dtype)
                high_values = native_high_values.to(compute_dtype)
                packed, scale_bytes = _reference_quant(
                    native_high_values, history_scale.reciprocal()
                )
                scale_bytes = scale_bytes.view(length + 1, heads, width // 16)
                low = (
                    _reference_dequant(packed, scale_bytes, history_scale)
                    .to(compute_dtype)
                    .to(high_torch_dtype)
                    .to(compute_dtype)
                )
                expected = low.clone()
                for position in range(length + 1):
                    ordinal, offset = divmod(position, block)
                    torch.testing.assert_close(
                        history[history_ids[ordinal], role, offset, :, :packed_bytes],
                        packed[position],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        history[history_ids[ordinal], role, offset, :, packed_bytes:payload_bytes],
                        scale_bytes[position],
                        rtol=0,
                        atol=0,
                    )
                    if high_ids[ordinal] >= 0:
                        expected[position] = high_values[position]
                        torch.testing.assert_close(
                            high[high_ids[ordinal], role, offset].view(torch.uint8),
                            native_high_values[position].view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                # New tokens use the native KV representation before NVFP4 storage.
                expected[-1] = high_values[-1]
                reference.append(expected.transpose(0, 1)[None])
            for ordinal in range(length // block):
                torch.testing.assert_close(
                    history[history_ids[ordinal]],
                    history_before[history_ids[ordinal]],
                    rtol=0,
                    atol=0,
                )
            expected = torch.nn.functional.scaled_dot_product_attention(
                query[row, -1].view(1, query_heads, 1, width), *reference, enable_gqa=True
            ).reshape_as(output[row])
            torch.testing.assert_close(output[row], expected, atol=0.01, rtol=0.02)
        if decode_graph is not None:
            metadata, graph, captured_output, captured_other_output, run = decode_graph
            for request_id in request_ids:
                cache = manager.kv_cache_map[request_id]
                assert cache.resize(None, length + 1)
                assert cache.resize(length + 2)
                manager._restore_page_index_bufs(request_id, cache)
            metadata.kv_cache_params.num_cached_tokens_per_seq = [length + 1] * batch
            metadata.prepare()
            graph.replay()
            replayed = captured_output.clone()
            torch.testing.assert_close(captured_other_output, -replayed, atol=0.01, rtol=0.02)
            ordinal, offset = divmod(length, block)
            for row, history_ids in enumerate(all_history_ids):
                # The next replay appends the same K/V at the updated sequence length.
                for pool, page_id in (
                    (history[..., :payload_bytes], history_ids[ordinal]),
                    (high, all_high_ids[row][ordinal]),
                ):
                    torch.testing.assert_close(
                        pool[page_id, :, offset + 1].view(torch.uint8),
                        pool[page_id, :, offset].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
            eager = run()
            torch.testing.assert_close(replayed, eager, atol=0.01, rtol=0.02)
    finally:
        manager.shutdown()


def _reference_quant(values, global_scale):
    """Use the contiguous codec to check the paged kernel's address mapping."""
    from flashinfer.quantization.fp4_quantization import SfLayout, nvfp4_quantize

    width = values.shape[-1]
    packed, scales = nvfp4_quantize(
        values.reshape(-1, width), global_scale, sfLayout=SfLayout.layout_linear, backend="cute-dsl"
    )
    return (
        packed.view(torch.uint8).reshape(*values.shape[:-1], width // 2),
        scales.view(torch.uint8).reshape(*values.shape[:-1], width // 16),
    )


def _reference_dequant(packed, scales, scale):
    """Independent E2M1/linear-scale oracle, used only by these tests."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long().flatten(-2)
    block_scales = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return table[codes] * block_scales * scale


@pytest.mark.parametrize("layout", ("HND", "NHD"))
@pytest.mark.parametrize(
    "dtype,width", [(torch.float16, 64), (torch.bfloat16, 80), (torch.float8_e4m3fn, 96)]
)
def test_mixed_precision_paged_conversion(dtype, width, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    from tensorrt_llm._torch.cute_dsl_kernels.mixed_precision_kv import convert_mixed_precision_kv

    values = torch.linspace(-6, 6, 192 * 4 * width, device="cuda").view(192, 2, 2, width).to(dtype)
    pool = values.view(3, 64, 2, 2, width).transpose(1, 2).contiguous()
    packed_bytes, scale_bytes = width // 2, width // 16
    payload_bytes = packed_bytes + scale_bytes
    row_bytes = (payload_bytes + 7) // 8 * 8
    low = torch.full((3, 2, 128, 2, row_bytes), 23, dtype=torch.uint8, device="cuda")
    high = torch.zeros((3, 2, 128, 2, width), dtype=dtype, device="cuda")
    tables = torch.tensor([[[2, 0]], [[1, -1]]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([[0], [130]], dtype=torch.int32, device="cuda")
    factor = 1 / 6 if dtype == torch.float8_e4m3fn else 1.0
    quant_scale = torch.tensor([1 / factor], dtype=torch.float32, device="cuda")
    packed, block_scales = _reference_quant(values[:130], quant_scale)
    expected_low, expected_high = low.clone(), high.clone()
    for page, count in enumerate((128, 2)):
        expected_low[(2, 0)[page], :, :count, :, :packed_bytes] = packed[
            128 * page : 128 * page + count
        ].transpose(0, 1)
        expected_low[(2, 0)[page], :, :count, :, packed_bytes:payload_bytes] = block_scales[
            128 * page : 128 * page + count
        ].transpose(0, 1)
    expected_high[1] = values[:128].transpose(0, 1)
    pools = tuple(t.transpose(2, 3) if layout == "HND" else t for t in (pool, low, high))
    convert_mixed_precision_kv(True, 130, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4)
    # Compare complete records so padding and untouched pages must retain their sentinel.
    for actual, expected in ((low, expected_low), (high, expected_high)):
        torch.testing.assert_close(
            actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )

    restored = _reference_dequant(packed, block_scales, factor)
    if dtype == torch.float8_e4m3fn:
        restored = restored.to(torch.bfloat16).float().clamp(-448, 448)
    expected = torch.zeros_like(values)
    expected[:128] = values[:128]
    expected[128:130] = restored[128:130].to(dtype)
    lengths[0].fill_(130)
    convert_mixed_precision_kv(
        False, 192, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4
    )
    actual = pool.transpose(1, 2).reshape_as(values)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("layout", ("NHD", "HND"))
def test_mixed_precision_conversion_reuses_dynamic_shapes(layout, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    from tensorrt_llm._torch.cute_dsl_kernels import mixed_precision_kv as kernels

    monkeypatch.setattr(kernels, "_compiled_kernels", {})
    block, heads, width = 32, 2, 64
    row_bytes = kernels.get_mixed_precision_kv_row_bytes(DataType.NVFP4, width)
    compiled = None
    for batch, pages, capacity, padding in ((1, 2, 33, 0), (1, 2, 35, 2), (2, 3, 65, 3)):
        tokens = pages * block
        # E2M1 values with power-of-two scales are exactly representable in NVFP4.
        values = torch.arange(batch * tokens * 2 * heads, device="cuda") % 9 - 4
        values = (2.0**values).to(torch.bfloat16).reshape(batch, tokens, 2, heads, 1)
        pattern = torch.tensor(
            [-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6, 0, 0, 0],
            device="cuda",
            dtype=values.dtype,
        ).repeat(width // 16)
        values = values * pattern
        pool = values.reshape(batch * pages, block, 2, heads, width).transpose(1, 2).contiguous()
        low = torch.zeros(
            (batch * pages + 2, 2, block, heads, row_bytes), device="cuda", dtype=torch.uint8
        )
        high = torch.zeros(
            (batch * pages + 3, 2, block, heads, width), device="cuda", dtype=values.dtype
        )
        pools = tuple(
            t.transpose(2, 3).contiguous() if layout == "HND" else t for t in (pool, low, high)
        )
        tables = torch.empty(
            (2, batch + padding, pages + padding), dtype=torch.int32, device="cuda"
        )[:, :batch, :pages]
        indices = torch.arange(batch * pages, 0, -1, device="cuda").reshape(batch, pages)
        tables[0].copy_(indices)
        tables[1].copy_(indices)
        tables[1, :, 1::2] = -1
        lengths = torch.zeros((2, batch + padding), dtype=torch.int32, device="cuda")[:, :batch]
        ends = capacity - torch.arange(batch, device="cuda")
        lengths[1].copy_(ends)
        kernels.convert_mixed_precision_kv(
            True, capacity, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4
        )
        native, low, high = tuple(t.transpose(2, 3) for t in pools) if layout == "HND" else pools
        records = low[indices.flatten()].transpose(1, 2).reshape(batch, tokens, 2, heads, row_bytes)
        decoded = _reference_dequant(
            records[..., : width // 2], records[..., width // 2 : width // 2 + width // 16], 1
        )
        valid = torch.arange(tokens, device="cuda")[None, :] < ends[:, None]
        torch.testing.assert_close(decoded[valid], values[valid].float(), rtol=0, atol=0)
        high_values = high[indices.flatten()].transpose(1, 2).reshape_as(values)
        high_valid = valid & ((torch.arange(tokens, device="cuda") // block % 2) == 0)
        torch.testing.assert_close(high_values[high_valid], values[high_valid], rtol=0, atol=0)

        lengths[0].copy_(ends)
        native.fill_(99)
        kernels.convert_mixed_precision_kv(
            False, tokens, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4
        )
        expected = values.masked_fill(~valid[..., None, None, None], 0)
        torch.testing.assert_close(
            native.transpose(1, 2).reshape_as(values), expected, rtol=0, atol=0
        )
        if compiled is None:
            assert len(kernels._compiled_kernels) == 2
            compiled = dict(kernels._compiled_kernels)
        assert kernels._compiled_kernels == compiled

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernels.convert_mixed_precision_kv(
            False, tokens, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4
        )
    lengths[0].sub_(1)
    native.fill_(99)
    graph.replay()
    valid = torch.arange(tokens, device="cuda")[None, :] < (ends - 1)[:, None]
    expected = values.masked_fill(~valid[..., None, None, None], 0)
    torch.testing.assert_close(native.transpose(1, 2).reshape_as(values), expected, rtol=0, atol=0)
    assert kernels._compiled_kernels == compiled


@pytest.mark.cpu_only
def test_mixed_precision_public_options():
    from pydantic import ValidationError

    assert KvCacheConfig().high_low_pool_ratio == 1 / 5
    config = KvCacheConfig(enable_mix_precision="nvfp4", dtype="fp8", high_low_pool_ratio=0.25)
    assert config.first_n == config.last_n == 256
    assert config.dtype == "fp8"
    for invalid in (
        {"enable_mix_precision": "int4"},
        {"first_n": -1},
        {"high_low_pool_ratio": 0},
        {"high_low_pool_ratio": None},
        {"enable_mix_precision": "nvfp4", "first_n": 0, "last_n": 0},
    ):
        with pytest.raises(ValidationError):
            KvCacheConfig(**invalid)
    for first_n, last_n in ((0, 1), (1, 0)):
        KvCacheConfig(enable_mix_precision="nvfp4", first_n=first_n, last_n=last_n)
    KvCacheConfig(first_n=0, last_n=0)


@pytest.mark.parametrize("backend", ("TRTLLM", "FLASHINFER"))
@pytest.mark.parametrize("mixed_precision,use_v2", ((None, False), (None, True), ("nvfp4", True)))
@pytest.mark.cpu_only
def test_mixed_precision_manager_routing(backend, mixed_precision, use_v2):
    from tensorrt_llm._torch.pyexecutor import _util

    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(), sparse_attention_config=None, attn_backend=backend
    )
    config = KvCacheConfig(enable_mix_precision=mixed_precision, use_kv_cache_manager_v2=use_v2)
    expected = (
        MixedPrecisionKVCacheManager
        if mixed_precision is not None
        else _util.KVCacheManagerV2
        if use_v2
        else _util.KVCacheManager
    )
    with patch.object(
        _util, "_non_hybrid_kv_cache_manager_cls", wraps=_util._non_hybrid_kv_cache_manager_cls
    ) as select_manager:
        assert _util.get_kv_cache_manager_cls(model_config, config) is expected
    select_manager.assert_called_once_with(model_config.pretrained_config, config)


@pytest.mark.parametrize(
    "model_overrides,use_v2,error,match",
    [
        ({}, False, ValueError, "requires use_kv_cache_manager_v2=True"),
        (
            {"pretrained_config": SimpleNamespace(architectures=["Qwen3NextForCausalLM"])},
            True,
            NotImplementedError,
            "sparse and recurrent",
        ),
        (
            {"sparse_attention_config": SimpleNamespace(algorithm="dsa")},
            True,
            NotImplementedError,
            "sparse and recurrent",
        ),
        ({"attn_backend": "FLASH_ATTN"}, True, NotImplementedError, "TRTLLM or FlashInfer"),
    ],
    ids=("v1", "mamba", "sparse", "backend"),
)
@pytest.mark.cpu_only
def test_mixed_precision_manager_rejects_unsupported_routes(model_overrides, use_v2, error, match):
    from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls

    model_config = SimpleNamespace(
        **{
            "pretrained_config": SimpleNamespace(),
            "sparse_attention_config": None,
            "attn_backend": "TRTLLM",
            **model_overrides,
        }
    )
    config = KvCacheConfig(enable_mix_precision="nvfp4", use_kv_cache_manager_v2=use_v2)
    with pytest.raises(error, match=match):
        get_kv_cache_manager_cls(model_config, config)


@pytest.mark.cpu_only
def test_mixed_precision_does_not_override_fp4_mla_routing():
    from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls

    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        sparse_attention_config=None,
        attn_backend="TRTLLM",
        quant_config=QuantConfig(kv_cache_quant_algo="NVFP4"),
    )
    assert (
        get_kv_cache_manager_cls(model_config, KvCacheConfig(use_kv_cache_manager_v2=True))
        is Fp4MlaKVCacheManagerV2
    )
    with pytest.raises(ValueError, match="requires FP16/BF16/FP8 high KV"):
        get_kv_cache_manager_cls(
            model_config, KvCacheConfig(enable_mix_precision="nvfp4", use_kv_cache_manager_v2=True)
        )


@pytest.mark.parametrize(
    "backend", (TrtllmAttention, FlashInferAttention), ids=("trtllm", "flashinfer")
)
@torch.inference_mode()
def test_mla_preserves_native_rope_and_attention(backend):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
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
        num_layers=2,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=128,
        max_seq_len=1024,
        max_batch_size=1,
        mapping=Mapping(),
        dtype=DataType.BF16,
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
        layer_idx=1,
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
                prompt_lens=[end - begin if begin < 512 else 512],
                kv_layout="HND",
                enable_context_mla_with_cached_kv=True,
                runtime_features=AttentionRuntimeFeatures(
                    chunked_prefill=True, cache_reuse=True, chunk_size=256
                ),
            )
            metadata.prepare()
            assert metadata.mixed_precision_kv_views[1].get_buffers(1).dtype == torch.bfloat16
            output = hidden.new_empty((end - begin, 16 * 128))
            seq_lens_cuda = metadata.seq_lens_cuda
            with patch.object(model, "_forward_impl", wraps=model._forward_impl) as model_forward:
                model.forward_impl(
                    torch.arange(begin, end, device="cuda"), hidden[begin:end], metadata, [output]
                )
            assert model_forward.call_args.args[2] is metadata
            assert metadata.seq_lens_cuda is seq_lens_cuda
            assert torch.isfinite(output).all()
            assert metadata.kv_cache_manager is manager
        high = manager.get_buffers(1, mixed_precision_pool="high")
        _, ids = manager.get_mixed_precision_page_tables([0], 1)
        assert high.shape[1] == 1
        assert ids[0][0] >= 0 and ids[0][1] == kv.BAD_PAGE_INDEX and ids[0][-1] >= 0
    finally:
        manager.shutdown()
