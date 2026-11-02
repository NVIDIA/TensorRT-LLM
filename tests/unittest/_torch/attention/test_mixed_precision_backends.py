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
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv


def _attention_reference(query, key, value, width):
    mask = torch.ones((query.shape[1], key.shape[1]), dtype=torch.bool, device=query.device).tril(
        key.shape[1] - query.shape[1]
    )
    tensors = [
        tensor.reshape(*tensor.shape[:2], -1, width).transpose(1, 2)
        for tensor in (query, key, value)
    ]
    output = torch.nn.functional.scaled_dot_product_attention(
        *tensors, attn_mask=mask, enable_gqa=True
    )
    return output.transpose(1, 2).reshape(-1, query.shape[-1])


@pytest.mark.parametrize(
    "backend,high_dtype,compute_dtype,width,query_heads,heads,kv_layout,fp8_input,block",
    [
        (TrtllmAttention, DataType.BF16, torch.bfloat16, 128, 4, 2, "NHD", False, 128),
        (TrtllmAttention, DataType.BF16, torch.bfloat16, 128, 4, 2, "HND", False, 32),
        (FlashInferAttention, DataType.BF16, torch.bfloat16, 128, 4, 2, "NHD", False, 128),
        (FlashInferAttention, DataType.FP8, torch.bfloat16, 64, 4, 1, "NHD", True, 128),
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
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    length, batch = 640, 2
    native_interleave = backend is FlashInferAttention or block == 32
    request_ids = list(range(batch))
    high_torch_dtype = binding_to_torch_dtype(high_dtype)
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=65536,
            enable_block_reuse=True,
            dtype="mixed_nvfp4_fp8" if high_dtype == DataType.FP8 else "mixed_nvfp4_bfloat16",
            mixed_precision_first_n=block,
            mixed_precision_last_n=block,
            # Full-history (low) group, then the retained-token (high) group.
            pool_ratio=[0.75, 0.25],
        ),
        CacheType.SELF,
        num_layers=4 if native_interleave else 2,
        num_kv_heads=heads,
        head_dim=width,
        tokens_per_block=block,
        max_seq_len=1024,
        # More request capacity than the batch: staging rows beyond the batch
        # must not shift the rows the kernels address.
        max_batch_size=batch + 1,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=high_dtype,
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
    history_scale = torch.full(
        (1,), 1 / 6 if high_dtype == DataType.FP8 else 1, dtype=torch.float32, device="cuda"
    )
    module = SimpleNamespace(
        attn=attention,
        mapping=manager.mapping,
        quant_config=quant_config,
        sparse_attn_hooks=None,
        register_to_config=False,
        use_gqa_decode_fmha=False,
        cache_context_plan=False,
        _output_quant_flags=None,
        _use_quantize_output=lambda: False,
    )
    module._attn_impl = partial(Attention._attn_impl, module)
    module._get_output_quant_flags = partial(Attention._get_output_quant_flags, module)
    module._use_custom_inplace_op = partial(Attention._use_custom_inplace_op, module)
    decode_graph = None
    native_output = None

    def forward(begin, end, context):
        nonlocal decode_graph, native_output
        metadata = backend.Metadata(
            seq_lens=torch.tensor([end - begin] * batch, dtype=torch.int32),
            num_contexts=batch if context else 0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[begin] * batch
            ),
            max_num_requests=batch + 1,
            max_num_tokens=batch * (length + 1),
            kv_cache_manager=manager,
            request_ids=request_ids,
            prompt_lens=[end - begin if context else length] * batch,
            kv_layout=kv_layout,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
        metadata.prepare()
        view = metadata.mixed_precision_staging
        assert view.get_buffers(0).dtype == high_torch_dtype
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
            finally:
                module.attn = attention
            return layer_outputs[0]

        output = run()
        native_output = layer_outputs[1].clone() if native_interleave else None
        torch.testing.assert_close(layer_outputs[-1], -output, atol=0.01, rtol=0.02)
        if context:
            history_before = manager.get_buffers(0).clone()
            native_offsets = getattr(metadata, "kv_cache_block_offsets", None)
            paged_context = getattr(metadata, "use_paged_context_fmha", None)
            with (
                patch.object(attention, "forward", side_effect=RuntimeError("backend failure")),
                pytest.raises(RuntimeError, match="backend failure"),
            ):
                Attention.forward_impl(
                    module, *inputs, metadata, PredefinedAttentionMask.CAUSAL, None, None, None
                )
            assert metadata.kv_cache_manager is manager
            assert metadata.kv_layout == kv_layout
            assert getattr(metadata, "kv_cache_block_offsets", None) is native_offsets
            assert getattr(metadata, "use_paged_context_fmha", None) == paged_context
            torch.testing.assert_close(manager.get_buffers(0), history_before, rtol=0, atol=0)
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
            for tensor in (*inputs, *other_inputs):
                if tensor is not None:
                    tensor.copy_(tensor.flip(0))
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

    try:
        prefix = 512
        manager.add_dummy_requests(request_ids, [prefix] * batch, is_gen=False)
        for begin, end, context in (
            (0, prefix, True),
            (prefix, length, True),
            (length, length + 1, False),
        ):
            if begin:
                for request_id in request_ids:
                    cache = manager.kv_cache_map[request_id]
                    assert cache.resize(None, begin)
                    cache.suspend()
                    assert cache.resume(
                        torch.cuda.current_stream().cuda_stream, is_decoding=not context
                    )
                    assert cache.resize(end)
                    manager._restore_page_index_bufs(request_id, cache)
            history_before = token_major(manager.get_buffers(0, kv_layout=kv_layout)).clone()
            all_history_ids, all_high_ids = manager.get_mixed_precision_page_tables(request_ids, 0)
            output = forward(begin, end, context).reshape(batch, end - begin, -1)
            # Native FP8 context FMHA also quantizes P; retain its existing tolerance.
            atol = (0.05 if high_dtype == DataType.FP8 else 0.02) if context else 0.01
            rtol = 0.001 if context and high_dtype == DataType.FP8 else 0.02
            if native_interleave:
                native_reference = _attention_reference(
                    query[:, begin:end],
                    key[:, :end].to(high_torch_dtype).to(compute_dtype),
                    value[:, :end].to(high_torch_dtype).to(compute_dtype),
                    width,
                )
                torch.testing.assert_close(native_output, native_reference, atol=atol, rtol=rtol)
            history = token_major(manager.get_buffers(0, kv_layout=kv_layout))
            for row, (history_ids, high_ids) in enumerate(zip(all_history_ids, all_high_ids)):
                if begin:
                    # Cached context and decode must both read low-only history.
                    assert high_ids[0] >= 0 and high_ids[1] == kv.BAD_PAGE_INDEX
                    assert high_ids[-1] >= 0
                high_pages = torch.tensor(high_ids, device="cuda")
                high_mask = high_pages[torch.arange(end, device="cuda") // block] >= 0
                reference = []
                for tensor in (key, value):
                    native_high_values = (
                        tensor[row, :end].view(-1, heads, width).to(high_torch_dtype)
                    )
                    high_values = native_high_values.to(compute_dtype)
                    packed, scale_bytes = _reference_quant(
                        native_high_values, history_scale.reciprocal()
                    )
                    expected = (
                        _reference_dequant(packed, scale_bytes, history_scale)
                        .to(compute_dtype)
                        .to(high_torch_dtype)
                        .to(compute_dtype)
                    )
                    expected[high_mask] = high_values[high_mask]
                    # Incoming tokens have not yet been compressed to the low pool.
                    expected[begin:] = high_values[begin:]
                    reference.append(expected.reshape(1, end, -1))
                pages = history_ids[: begin // block]
                torch.testing.assert_close(history[pages], history_before[pages], rtol=0, atol=0)
                expected = _attention_reference(query[row : row + 1, begin:end], *reference, width)
                torch.testing.assert_close(output[row], expected, atol=atol, rtol=rtol)
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
            eager = run()
            torch.testing.assert_close(replayed, eager, atol=0.01, rtol=0.02)
    finally:
        manager.shutdown()
    assert quant_config.model_dump() == original_config


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


@pytest.mark.parametrize("backend", (TrtllmAttention, FlashInferAttention))
@torch.no_grad()
def test_mla_cached_context_and_decode(backend):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    from test_attention_mla import apply_mla_rope

    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.mla import MLA
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.functional import PositionEmbeddingType

    torch.manual_seed(123)
    heads, rank, nope, rope_dim, block = 16, 512, 128, 64, 128
    length, prefix = 640, 512
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    rope = RopeParams(dim=rope_dim, max_positions=1024, duplicate_data=True)
    model = MLA(
        hidden_size=128,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope_dim,
        v_head_dim=nope,
        q_lora_rank=None,
        kv_lora_rank=rank,
        predicted_tokens_per_seq=1,
        max_position_embeddings=1024,
        bias=False,
        layer_idx=0,
        dtype=torch.bfloat16,
        pos_embd_params=PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj, rope=rope, is_neox=False
        ),
        config=ModelConfig(
            mapping=mapping,
            attn_backend="TRTLLM" if backend is TrtllmAttention else "FLASHINFER",
        ),
    ).cuda()
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=parameter.shape[-1] ** -0.5)
    key_weight, value_weight = model.kv_b_proj.weight.split(heads * nope)
    model.k_b_proj_trans.copy_(key_weight.view(heads, nope, rank).transpose(1, 2))
    model.v_b_proj.copy_(value_weight.view(heads, nope, rank))
    hidden = torch.randn(length + 1, 128, dtype=torch.bfloat16, device="cuda")

    # Independent projections, RoPE and contiguous NVFP4 codec form the oracle.
    query = torch.nn.functional.linear(hidden, model.q_b_proj.weight).view(
        -1, heads, nope + rope_dim
    )
    latent, key_rope = torch.nn.functional.linear(hidden, model.kv_a_proj_with_mqa.weight).split(
        (rank, rope_dim), dim=-1
    )
    latent = (
        latent.float() * torch.rsqrt(latent.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(hidden.dtype)
    rope_table = rope.create_rope_const_params()[1].view(1024, -1, 2).transpose(1, 2)
    positions = list(range(length + 1))
    query_rope = apply_mla_rope(query[..., nope:], positions, rope_table)
    key_rope = apply_mla_rope(key_rope, positions, rope_table)
    if backend is FlashInferAttention:
        # External GPT-J RoPE keeps pairs interleaved; native MLA uses split halves.
        query_rope = query_rope.unflatten(-1, (2, -1)).transpose(-2, -1).flatten(-2)
        key_rope = key_rope.unflatten(-1, (2, -1)).transpose(-2, -1).flatten(-2)
    query[..., nope:] = query_rope
    latent = torch.cat((latent, key_rope), dim=-1)
    scale = torch.ones(1, device="cuda")
    packed, scales = _reference_quant(latent, scale)
    low_latent = _reference_dequant(packed, scales, scale).to(hidden.dtype)

    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=65536,
            enable_block_reuse=True,
            dtype="mixed_nvfp4_bfloat16",
            mixed_precision_first_n=block,
            mixed_precision_last_n=block,
            pool_ratio=[0.75, 0.25],
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=rank + rope_dim,
        tokens_per_block=block,
        max_seq_len=1024,
        max_batch_size=1,
        mapping=mapping,
        dtype=DataType.BF16,
    )
    try:
        assert manager.kv_factor == 1
        manager.add_dummy_requests([0], [prefix], is_gen=False)
        for begin, end, context in (
            (0, prefix, True),
            (prefix, length, True),
            (length, length + 1, False),
        ):
            if begin:
                cache = manager.kv_cache_map[0]
                assert cache.resize(None, begin)
                cache.suspend()
                assert cache.resume(
                    torch.cuda.current_stream().cuda_stream, is_decoding=not context
                )
                assert cache.resize(end)
                manager._restore_page_index_bufs(0, cache)
            metadata = backend.Metadata(
                seq_lens=torch.tensor([end - begin], dtype=torch.int32),
                num_contexts=int(context),
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[begin]),
                max_num_requests=1,
                max_num_tokens=length + 1,
                kv_cache_manager=manager,
                request_ids=[0],
                prompt_lens=[end - begin if context else length],
                kv_layout="NHD",
                mapping=mapping,
                enable_context_mla_with_cached_kv=True,
                runtime_features=AttentionRuntimeFeatures(
                    chunked_prefill=True,
                    cache_reuse=True,
                    chunk_size=prefix,
                    chunked_prefill_buffer_batch_size=1,
                ),
            )
            metadata.prepare()
            if backend is TrtllmAttention:
                view = metadata.mixed_precision_staging
                paged_context_fmha = metadata.use_paged_context_fmha
                native_kv = metadata._swap_mixed_precision_kv(view)
                assert metadata.kv_cache_manager is view
                assert metadata.use_paged_context_fmha == paged_context_fmha
                metadata._restore_mixed_precision_kv(native_kv)
                assert metadata.kv_cache_manager is manager
                assert metadata.use_paged_context_fmha == paged_context_fmha
            _, high_pages = manager.get_mixed_precision_page_tables([0], 0)
            if begin:
                assert high_pages[0][0] >= 0 and high_pages[0][1] == kv.BAD_PAGE_INDEX
            high_mask = (
                torch.tensor(high_pages[0], device="cuda")[
                    torch.arange(end, device="cuda") // block
                ]
                >= 0
            )
            history = low_latent[:end].clone()
            history[high_mask] = latent[:end][high_mask]
            history[begin:] = latent[begin:end]
            keys = torch.nn.functional.linear(history[:, :rank], key_weight).view(end, heads, nope)
            keys = torch.cat((keys, history[:, None, rank:].expand(-1, heads, -1)), dim=-1)
            values = torch.nn.functional.linear(history[:, :rank], value_weight).view(
                end, heads, nope
            )
            mask = torch.ones(end - begin, end, dtype=torch.bool, device="cuda").tril(begin)
            expected = (
                torch.nn.functional.scaled_dot_product_attention(
                    query[begin:end].transpose(0, 1).float(),
                    keys.transpose(0, 1).float(),
                    values.transpose(0, 1).float(),
                    attn_mask=mask,
                    scale=model.softmax_scale,
                )
                .transpose(0, 1)
                .reshape(end - begin, -1)
            )
            output = torch.empty(end - begin, heads * nope, dtype=hidden.dtype, device="cuda")
            model.forward_impl(
                torch.arange(begin, end, device="cuda"), hidden[begin:end], metadata, [output]
            )
            assert metadata.kv_cache_manager is manager
            # BF16 MLA with a 512-wide latent accumulates more rounding than the
            # dense cases; the TRTLLM kernel lands within 0.03 of the reference.
            torch.testing.assert_close(output.float(), expected, atol=0.03, rtol=0.03)
    finally:
        manager.shutdown()


@pytest.mark.parametrize("layout", ("HND", "NHD"))
@pytest.mark.parametrize(
    "dtype,width", [(torch.float16, 64), (torch.bfloat16, 80), (torch.float8_e4m3fn, 96)]
)
def test_mixed_precision_paged_conversion(dtype, width, layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    from tensorrt_llm._torch.attention.kernels.blackwell import mixed_precision_kv as kernels

    convert_mixed_precision_kv = kernels.convert_mixed_precision_kv

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
    # Positions past the length stay untouched: the read never writes padding.
    expected = values.clone()
    expected[128:130] = restored[128:130].to(dtype)
    lengths[0].fill_(130)
    convert_mixed_precision_kv(
        False, 192, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4
    )
    actual = pool.transpose(1, 2).reshape_as(values)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    compiled = dict(kernels._compiled_kernels)
    for capacity in (129, 192):
        convert_mixed_precision_kv(
            False, capacity, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4
        )
    assert kernels._compiled_kernels == compiled
    # A small pass makes every thread loop over several tokens; the result must not change.
    pool.copy_(torch.zeros_like(pool))
    original_pass = kernels._PagedMixedPrecisionKV.tokens_per_pass
    kernels._PagedMixedPrecisionKV.tokens_per_pass = 8
    try:
        convert_mixed_precision_kv(
            False, 192, *pools, tables, lengths, layout, low_dtype=DataType.NVFP4
        )
    finally:
        kernels._PagedMixedPrecisionKV.tokens_per_pass = original_pass
    expected[130:] = 0
    actual = pool.transpose(1, 2).reshape_as(values)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.cpu_only
def test_conversion_kernel_imports_pinned_flashinfer_symbols():
    """The kernel relies on internal FlashInfer modules; a bump that drops them fails here."""
    import importlib

    module = importlib.import_module(
        "tensorrt_llm._torch.attention.kernels.blackwell.mixed_precision_kv"
    )
    assert callable(module.convert_mixed_precision_kv)
    assert module.get_mixed_precision_kv_row_bytes(DataType.NVFP4, 128) == 72


@pytest.mark.cpu_only
def test_mixed_precision_public_options():
    from pydantic import ValidationError

    assert KvCacheConfig().mixed_precision_kv is None
    config = KvCacheConfig(dtype="mixed_nvfp4_fp8")
    assert config.mixed_precision_kv == ("nvfp4", "fp8", 256, 256)
    assert config.mixed_precision_first_n == config.mixed_precision_last_n == 256
    assert config.pool_ratio is None
    for high in ("float16", "bfloat16"):
        assert KvCacheConfig(dtype=f"mixed_nvfp4_{high}").mixed_precision_kv.high == high
    for invalid in (
        {"dtype": "mixed_int4_fp8"},
        {"dtype": "mixed_nvfp4_nvfp4"},
        {"dtype": "mixed_nvfp4_auto"},
        {"dtype": "mixed_nvfp4_fp8", "mixed_precision_first_n": -1},
        {"dtype": "mixed_nvfp4_fp8", "mixed_precision_last_n": -1},
        {"dtype": "mixed_nvfp4_fp8", "mixed_precision_first_n": 0, "mixed_precision_last_n": 0},
        {"mixed_precision_config": {"mixed_precision_first_n": 1}},
    ):
        with pytest.raises(ValidationError):
            KvCacheConfig(**invalid)
    for first_n, last_n in ((0, 1), (1, 0)):
        KvCacheConfig(
            dtype="mixed_nvfp4_fp8", mixed_precision_first_n=first_n, mixed_precision_last_n=last_n
        )
    KvCacheConfig(mixed_precision_first_n=0, mixed_precision_last_n=0)


@pytest.mark.parametrize(
    "overrides,error,match",
    [
        (
            {
                "kv_cache_config": KvCacheConfig(
                    dtype="mixed_nvfp4_fp8", use_kv_cache_manager_v2=False
                )
            },
            ValueError,
            "use_kv_cache_manager_v2=True",
        ),
        (
            {
                "kv_cache_config": KvCacheConfig(
                    dtype="mixed_nvfp4_fp8", use_kv_cache_manager_v2="auto"
                )
            },
            ValueError,
            "use_kv_cache_manager_v2=True",
        ),
        ({"attn_backend": "vanilla"}, ValueError, "TRTLLM and FLASHINFER"),
        ({"sparse_attention_config": {"algorithm": "rocket"}}, ValueError, "sparse"),
        (
            {
                "kv_connector_config": {
                    "connector_module": "connector_module",
                    "connector_scheduler_class": "Scheduler",
                    "connector_worker_class": "Worker",
                }
            },
            ValueError,
            "KV connector",
        ),
        (
            {"speculative_config": {"decoding_type": "NGram", "max_draft_len": 2}},
            ValueError,
            "single-beam",
        ),
        (
            {"kv_cache_compression_config": {"algorithm": "quantization_for_cold_page"}},
            ValueError,
            "single-beam",
        ),
        ({"max_beam_width": 2}, ValueError, "single-beam"),
    ],
    ids=("v1", "auto_v2", "backend", "sparse", "connector", "speculative", "compression", "beam"),
)
@pytest.mark.cpu_only
def test_mixed_precision_llm_args_validation(overrides, error, match):
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    kwargs = dict(
        model="/tmp/mixed-precision-nonexistent-model",
        kv_cache_config=KvCacheConfig(dtype="mixed_nvfp4_fp8", use_kv_cache_manager_v2=True),
    )
    TorchLlmArgs(**kwargs)
    kwargs.update(overrides)
    with pytest.raises(error, match=match):
        TorchLlmArgs(**kwargs)


@pytest.mark.cpu_only
def test_mixed_precision_retained_dtype_must_match_model_dtype():
    from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_kv_cache_quant

    model_config = SimpleNamespace(
        quant_config=QuantConfig(),
        quant_config_dict=None,
        torch_dtype=torch.bfloat16,
        sparse_attention_config=None,
    )
    with pytest.raises(ValueError, match="model dtype"):
        validate_and_set_kv_cache_quant(model_config, "mixed_nvfp4_float16")
    validate_and_set_kv_cache_quant(model_config, "mixed_nvfp4_bfloat16")
    assert model_config.quant_config.kv_cache_quant_algo is None


@pytest.mark.cpu_only
def test_mixed_precision_llm_args_accept_lowercase_backend():
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    TorchLlmArgs(
        model="/tmp/mixed-precision-nonexistent-model",
        kv_cache_config=KvCacheConfig(dtype="mixed_nvfp4_fp8", use_kv_cache_manager_v2=True),
        attn_backend="flashinfer",
    )


@pytest.mark.parametrize(
    "tp_size,attention_dp,heads_per_rank",
    [(1, False, 4), (2, False, 2), (2, True, 4)],
    ids=("tp1", "tp2", "tp2_attention_dp"),
)
def test_staging_allocation_follows_rank_geometry(tp_size, attention_dp, heads_per_rank):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    from unittest.mock import Mock

    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module

    rows, block, width = 3, 128, 64
    mapping = Mapping(world_size=tp_size, rank=0, tp_size=tp_size, enable_attention_dp=attention_dp)
    distributed = Mock(local_world_size=1)
    distributed.allreduce.side_effect = lambda value, op: value
    with (
        patch.object(kv_cache_v2_module, "mpi_rank", return_value=0),
        patch.object(Distributed, "get", return_value=distributed),
    ):
        manager = KVCacheManagerV2(
            KvCacheConfig(
                max_tokens=65536,
                host_cache_size=0,
                enable_block_reuse=True,
                dtype="mixed_nvfp4_bfloat16",
                mixed_precision_first_n=block,
                mixed_precision_last_n=block,
                pool_ratio=[0.75, 0.25],
            ),
            CacheType.SELF,
            num_layers=2,
            num_kv_heads=4,
            head_dim=width,
            tokens_per_block=block,
            max_seq_len=1024,
            max_batch_size=rows,
            mapping=mapping,
            dtype=DataType.BF16,
        )
    try:
        # The manager shards KV heads across TP ranks; attention DP keeps them whole.
        assert manager.num_kv_heads_per_layer == [heads_per_rank, heads_per_rank]
        manager.add_dummy_requests([0], [block], is_gen=False)

        def make_metadata():
            metadata = TrtllmAttention.Metadata(
                seq_lens=torch.tensor([block], dtype=torch.int32),
                num_contexts=1,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
                max_num_requests=rows,
                max_num_tokens=block,
                max_num_sequences=rows,
                kv_cache_manager=manager,
                request_ids=[0],
                prompt_lens=[block],
                kv_layout="NHD",
                mapping=mapping,
                runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
            )
            metadata.prepare()
            return metadata

        view = make_metadata().mixed_precision_staging
        # TRTLLM stages 32-token HND pages; every row spans the full max_seq_len.
        page_bytes = manager.kv_factor * heads_per_rank * 32 * width * 2
        assert view.max_page_bytes == page_bytes
        assert view.staging_bytes == rows * (1024 // 32) * page_bytes
        staging = view.get_buffers(0)
        assert staging.shape[0] == rows * (1024 // 32)
        # Never-written positions are zero; attention may load them before masking.
        assert not staging.any()
        # Allocated once from the shared metadata buffer pool as a reserved block,
        # and reused by later metadata objects (the final engine after profiling).
        blocks = get_memory_buffers().buffers["mixed_kv_staging"]
        assert any(b.is_reserved and b.buffer.data_ptr() == staging.data_ptr() for b in blocks)
        # Both layers draw from the same pool pair, so they share one page table per batch.
        assert view.page_tables[1].data_ptr() == view.page_tables[0].data_ptr()
        again = make_metadata().mixed_precision_staging
        assert again.get_buffers(0).data_ptr() == staging.data_ptr()
        assert len(get_memory_buffers().buffers["mixed_kv_staging"]) == len(blocks)
        # Eager steps refresh the existing staging copy instead of rebuilding it.
        metadata = make_metadata()
        view = metadata.mixed_precision_staging
        metadata.prepare()
        assert metadata.mixed_precision_staging is view
        # The KV quota is not reduced for staging.
        assert manager.kv_cache_manager_py_config.cache_tiers[0].quota > 0
    finally:
        manager.shutdown()
