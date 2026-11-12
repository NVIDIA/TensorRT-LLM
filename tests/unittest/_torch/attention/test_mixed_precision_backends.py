# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise mixed-precision cache routing and optimized Attention backends."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from _torch.helpers import bare_extension, refresh_from_lengths

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    BatchInputs,
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
            # Full-history (low) group, then the mixed-precision high group.
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
                    assert cache.resume(torch.cuda.current_stream().cuda_stream)
                    assert cache.resize(end)
                    manager._restore_page_index_bufs(request_id, cache)
            history_before = token_major(manager.get_buffers(0, kv_layout=kv_layout)).clone()
            all_history_ids, all_high_ids = manager.get_mixed_precision_page_tables(request_ids, 0)
            forward(begin, end, context)
            history = token_major(manager.get_buffers(0, kv_layout=kv_layout))
            for history_ids, high_ids in zip(all_history_ids, all_high_ids):
                if begin:
                    # Cached context and decode must both read low-only history.
                    assert high_ids[0] >= 0 and high_ids[1] == kv.BAD_PAGE_INDEX
                    assert high_ids[-1] >= 0
                # Attention runs on the staging copy; the cached low history is untouched.
                pages = history_ids[: begin // block]
                torch.testing.assert_close(history[pages], history_before[pages], rtol=0, atol=0)
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
            {
                "speculative_config": {
                    "decoding_type": "MTP",
                    "num_nextn_predict_layers": 1,
                    "max_draft_len": 1,
                    "use_dynamic_tree": True,
                    "dynamic_tree_max_topK": 2,
                }
            },
            ValueError,
            "dynamic-tree",
        ),
        (
            {"kv_cache_compression_config": {"algorithm": "quantization_for_cold_page"}},
            ValueError,
            "single-beam",
        ),
        ({"max_beam_width": 2}, ValueError, "single-beam"),
        ({"sparse_attention_config": {"algorithm": "rocket"}}, ValueError, "sparse attention"),
        ({"sleep_config": {}}, ValueError, "sleep_config"),
        (
            {
                "speculative_config": {
                    "decoding_type": "Eagle3",
                    "max_draft_len": 2,
                    "speculative_model": "/tmp/mixed-precision-nonexistent-draft",
                    "use_dynamic_tree": True,
                    "dynamic_tree_max_topK": 2,
                }
            },
            ValueError,
            "dynamic-tree",
        ),
    ],
    ids=(
        "v1",
        "auto_v2",
        "backend",
        "connector",
        "dynamic_tree",
        "compression",
        "beam",
        "sparse",
        "sleep",
        "eagle3_dynamic_tree",
    ),
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


@pytest.mark.parametrize("is_mtp", [True, False])
@pytest.mark.cpu_only
def test_one_model_draft_kv_cache_dtype(is_mtp):
    """A separate MTP draft manager uses mixed-precision KV like the target; others do not."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    target = KvCacheConfig(
        dtype="mixed_nvfp4_fp8",
        mixed_precision_first_n=16,
        mixed_precision_last_n=64,
        pool_ratio=[0.9, 0.1],
    )
    creator = SimpleNamespace(
        _speculative_config=SimpleNamespace(
            spec_dec_mode=SimpleNamespace(is_mtp_one_model=lambda: is_mtp)
        )
    )
    draft = KvCacheCreator._get_one_model_draft_kv_cache_config(
        creator, target, max_seq_len=1024, estimating_kv_cache=True
    )
    assert draft.dtype == ("mixed_nvfp4_fp8" if is_mtp else "auto")
    assert (draft.mixed_precision_kv is not None) == is_mtp
    assert draft.max_attention_window is None
    assert target.mixed_precision_kv is not None


@pytest.mark.cpu_only
def test_staging_follows_this_metadatas_device_kv_lengths():
    """A draft's length changes reach its staging copy; another manager's staging is hidden."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmCoreMetadata

    core = bare_extension(TrtllmCoreMetadata, Mock())
    manager = Mock(name="draft_kv_cache_manager")
    core.kv_cache_manager = manager
    core._seq_lens_cuda = torch.tensor([1, 2], dtype=torch.int32)
    core.kv_lens_cuda = torch.tensor([10, 20, 0], dtype=torch.int32)
    view = SimpleNamespace(
        manager=manager, lengths=torch.zeros((2, 2), dtype=torch.int32), pools={0: None}
    )
    core.mixed_precision_staging = view
    assert core._mixed_precision_staging_name() == "mixed"
    # A draft metadata is created with a staging name of its own.
    core.mixed_precision_staging_name = "draft_mixed"
    assert core._mixed_precision_staging_name() == "draft_mixed"

    # A draft step advanced the device KV lengths, then the draft loop calls before_forward().
    core.kv_lens_cuda[:2] += 1
    core.resync_length_derived_state()
    assert view.lengths.tolist() == [[10, 19], [11, 21]]
    assert core.get_mixed_precision_kv_view(0) is view
    assert core.get_mixed_precision_kv_view(1) is None

    # A staging copy of another manager is neither used nor updated.
    view.manager = Mock(name="target_kv_cache_manager")
    core.kv_lens_cuda[:2] += 1
    core.resync_length_derived_state()
    assert view.lengths.tolist() == [[10, 19], [11, 21]]
    assert core.get_mixed_precision_kv_view(0) is None


@pytest.mark.cpu_only
def test_mixed_precision_high_dtype_must_match_model_dtype():
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
    # Sparse MLA models map the pool to NVFP4, but the mixed-precision high dtype must still match.
    sparse = SimpleNamespace(
        quant_config=QuantConfig(),
        quant_config_dict=None,
        torch_dtype=torch.bfloat16,
        sparse_attention_config=SimpleNamespace(algorithm="dsa"),
        pretrained_config=SimpleNamespace(),
        attn_backend="TRTLLM",
        extra_attrs={},
    )
    with pytest.raises(ValueError, match="model dtype"):
        validate_and_set_kv_cache_quant(sparse, "mixed_nvfp4_float16")
    validate_and_set_kv_cache_quant(sparse, "mixed_nvfp4_bfloat16")
    assert sparse.quant_config.kv_cache_quant_algo == "NVFP4"
    # SM107 has no NVFP4 sparse MLA: mixed KV fails instead of silently falling back to FP8.
    with (
        patch(f"{validate_and_set_kv_cache_quant.__module__}.get_sm_version", return_value=107),
        patch("torch.cuda.is_available", return_value=True),
        pytest.raises(ValueError, match="SM107"),
    ):
        validate_and_set_kv_cache_quant(sparse, "mixed_nvfp4_bfloat16")


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


@pytest.mark.parametrize(
    "backend", (TrtllmAttention, FlashInferAttention), ids=("trtllm", "flashinfer")
)
def test_staging_lengths_follow_corrected_kv_lens(backend):
    """The overlap scheduler corrects KV lengths on the device; staging follows."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    block, cached = 128, 32
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    manager, draft = (_small_manager(block, mapping) for _ in range(2))
    try:
        for kv_cache in (manager, draft):
            kv_cache.add_dummy_requests([0], [cached + block], is_gen=False)
        # A separate MTP draft manager holding the same tokens.
        metadata = _small_metadata(backend, manager, mapping, block, cached, draft=draft)
        metadata.prepare()
        view = metadata.mixed_precision_staging
        assert view.lengths.tolist() == [[cached], [cached + block]]
        offsets = torch.full((1,), -2, dtype=torch.int32, device="cuda")
        corrected = [[cached - 2], [cached + block - 2]]
        if backend is TrtllmAttention:
            # prepare applies the step's device corrections to the target and its draft.
            metadata.prepare(_small_inputs(manager, block, cached, kv_lens_offsets=offsets))
            view = metadata.mixed_precision_staging
            assert view.lengths.tolist() == corrected
            # The draft runs on its own staging copy of the draft manager.
            draft_view = metadata.draft.mixed_precision_staging
            assert draft_view is not view
            assert draft_view.manager is draft
            assert draft_view.name == "draft_mixed"
            assert draft_view.lengths.tolist() == corrected
            assert metadata.get_mixed_precision_kv_view(0) is view
            assert metadata.draft.get_mixed_precision_kv_view(0) is draft_view
            return

        def correct():
            metadata.core()._uses_full_generation_page_table = True
            metadata.apply_spec_decode_kv_lens_offsets(
                offsets, num_generations=0, tokens_per_generation=1, num_chunked_contexts=1
            )

        correct()
        assert view.lengths.tolist() == corrected
        # The separate draft view runs on its own staging copy of the draft
        # manager, which takes the corrected lengths each time the view is fetched.
        draft_view = metadata.get_draft_metadata(draft)
        draft_staging = draft_view.mixed_precision_staging
        assert draft_staging is not view and draft_staging.manager is draft
        assert draft_staging.name == "draft_mixed"
        assert draft_staging.lengths.tolist() == corrected
        metadata.prepare()
        correct()
        assert view.lengths.tolist() == corrected
        assert metadata.get_draft_metadata(draft) is draft_view
        assert draft_view.mixed_precision_staging is draft_staging
        assert draft_staging.lengths.tolist() == corrected
    finally:
        manager.shutdown()
        draft.shutdown()


def _small_manager(block: int, mapping: Mapping, dtype: str = "mixed_nvfp4_bfloat16"):
    """A one-layer BF16 cache; mixed precision keeps one page at each end in the high pool."""
    mixed = (
        dict(mixed_precision_first_n=block, mixed_precision_last_n=block, pool_ratio=[0.75, 0.25])
        if dtype.startswith("mixed_")
        else {}
    )
    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=65536, enable_block_reuse=True, dtype=dtype, **mixed),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=block,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=mapping,
        dtype=DataType.BF16,
    )


def _small_inputs(manager, block: int, cached: int, *, seq_lens=None, kv_lens_offsets=None):
    """The step _small_metadata describes, as BatchInputs (one context request)."""
    return BatchInputs(
        seq_lens=torch.tensor([block], dtype=torch.int32) if seq_lens is None else seq_lens,
        num_contexts=1 if seq_lens is None else 0,
        request_ids=[0],
        prompt_lens=[cached + block],
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached]),
        kv_lens_offsets=kv_lens_offsets,
    )


def _small_metadata(backend, manager, mapping: Mapping, block: int, cached: int, draft=None):
    if draft is not None and backend is TrtllmAttention:
        # One-model speculative decoding: the draft forwards get md.draft.
        return backend.Metadata(
            seq_lens=torch.tensor([block], dtype=torch.int32),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached]),
            max_num_requests=2,
            max_num_tokens=block,
            max_num_sequences=2,
            kv_cache_manager=manager,
            draft_kv_cache_manager=draft,
            one_model_draft=True,
            request_ids=[0],
            prompt_lens=[cached + block],
            kv_layout="NHD",
            mapping=mapping,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
    return backend.Metadata(
        seq_lens=torch.tensor([block], dtype=torch.int32),
        num_contexts=1,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached]),
        max_num_requests=2,
        max_num_tokens=block,
        max_num_sequences=2,
        kv_cache_manager=manager,
        draft_kv_cache_manager=draft,
        request_ids=[0],
        prompt_lens=[cached + block],
        kv_layout="NHD",
        mapping=mapping,
        runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"sparse_attention_config": {"algorithm": "dsa", "index_head_dim": 128}},
        {
            "sparse_attention_config": {
                "algorithm": "deepseek_v4",
                "index_head_dim": 128,
                "window_size": 128,
                "compress_ratios": [1, 4],
            }
        },
        {"speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1}},
        {
            "speculative_config": {
                "decoding_type": "Eagle3",
                "max_draft_len": 1,
                "speculative_model": "/tmp/mixed-precision-nonexistent-draft",
            }
        },
    ],
    ids=("dsa", "deepseek_v4", "mtp", "eagle3"),
)
@pytest.mark.cpu_only
def test_mixed_precision_llm_args_accept_supported_features(overrides):
    """The sparse algorithms and drafters mixed-precision KV supports pass validation."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    TorchLlmArgs(
        model="/tmp/mixed-precision-nonexistent-model",
        kv_cache_config=KvCacheConfig(dtype="mixed_nvfp4_fp8", use_kv_cache_manager_v2=True),
        **overrides,
    )


@pytest.mark.cpu_only
def test_gqa_decode_skips_the_kv_append_plan_of_a_mixed_layer():
    """A mixed-precision layer appends into its staging copy, not through a GQA-decode plan."""
    from tensorrt_llm._torch.attention.backends.fmha.gqa_decode import GqaDecodeFmha
    from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode

    attn = SimpleNamespace(
        head_dim=128,
        layer_idx=3,
        quant_mode=int(QuantMode.from_quant_algo(kv_cache_quant_algo=QuantAlgo.FP8)),
    )
    fmha = object.__new__(GqaDecodeFmha)
    fmha._attn_ref = lambda: attn
    metadata = Mock()
    metadata.get_mixed_precision_kv_view.side_effect = lambda layer_idx: (
        object() if layer_idx == 3 else None
    )
    assert fmha.kv_append_plan(metadata, num_tokens=4) is None
    metadata.get_mixed_precision_kv_view.assert_called_with(3)


def test_staging_page_tables_follow_each_layer_page_scale():
    """Mixed layers with different KV head counts get page tables in their own pools' page space."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    import numpy as np

    from tensorrt_llm._torch.pyexecutor.kv_cache.mixed_precision_staging import (
        MixedPrecisionKVStaging,
    )

    block = 32
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=4096,
            enable_block_reuse=False,
            dtype="mixed_nvfp4_bfloat16",
            mixed_precision_first_n=block,
            mixed_precision_last_n=block,
        ),
        CacheType.SELF,
        num_layers=3,
        num_kv_heads=[8, 8, 4],
        head_dim=64,
        tokens_per_block=block,
        max_seq_len=512,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        dtype=DataType.BF16,
    )
    try:
        manager.add_dummy_requests([0], [4 * block], is_gen=False)
        view = MixedPrecisionKVStaging(
            manager, Buffers(), rows=2, layout="HND", pool_layout="HND", tokens_per_block=block
        )
        view.update(request_ids=[0], cached_lengths=[0], seq_lens=[4 * block], capture_graph=False)
        for layer in range(3):
            expected = np.full(
                tuple(view.page_tables[layer].shape), kv.BAD_PAGE_INDEX, dtype=np.int32
            )
            manager.fill_mixed_precision_page_tables([0], layer, expected)
            assert view.page_tables[layer].cpu().numpy().tolist() == expected.tolist()
    finally:
        manager.shutdown()


def test_draft_staging_follows_corrected_kv_lens_in_cuda_graph():
    """Each prepare corrects both staging copies; a captured draft step keeps the draft's in step."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    block, cached = 128, 32
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    manager, draft = (_small_manager(block, mapping) for _ in range(2))
    try:
        for kv_cache in (manager, draft):
            kv_cache.add_dummy_requests([0], [cached + block], is_gen=False)
        # A separate MTP draft manager holds the same tokens; graph metadata is a one-token decode.
        metadata = _small_metadata(TrtllmAttention, manager, mapping, block, cached, draft=draft)
        metadata = metadata.create_cuda_graph_metadata(1, buffers=Buffers())
        offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
        decode = torch.ones(1, dtype=torch.int32)
        metadata.prepare(
            _small_inputs(manager, block, cached, seq_lens=decode, kv_lens_offsets=offsets)
        )
        draft_core = metadata.draft.core()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # A draft step advances the draft's KV lengths, then readies its next forward.
            draft_core.adjust_kv_lens(slice(0, 1), 1)
            refresh_from_lengths(metadata.draft)
        for offset in (-1, -3):
            offsets.fill_(offset)
            metadata.prepare(
                _small_inputs(manager, block, cached, seq_lens=decode, kv_lens_offsets=offsets)
            )
            corrected = [[cached + offset], [cached + 1 + offset]]
            assert metadata.mixed_precision_staging.lengths.tolist() == corrected
            assert metadata.draft.mixed_precision_staging.lengths.tolist() == corrected
            graph.replay()
            advanced = [[cached + 1 + offset], [cached + 2 + offset]]
            assert metadata.draft.mixed_precision_staging.lengths.tolist() == advanced
            assert metadata.mixed_precision_staging.lengths.tolist() == corrected
    finally:
        manager.shutdown()
        draft.shutdown()


@pytest.mark.parametrize(
    "backend", (TrtllmAttention, FlashInferAttention), ids=("trtllm", "flashinfer")
)
def test_staging_covers_allocated_capacity(backend):
    """Staging covers the allocated KV capacity, where MTP draft tokens go past the batch."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    block, cached = 128, 128
    # Three MTP draft tokens past the batch's 256 tokens start a new staging page:
    # TRTLLM staging pages hold 32 tokens and FlashInfer staging pages 128.
    capacity = cached + block + 3
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    manager = _small_manager(block, mapping)
    try:
        manager.add_dummy_requests([0], [cached + block], is_gen=False)
        assert manager.kv_cache_map[0].resize(capacity)
        metadata = _small_metadata(backend, manager, mapping, block, cached)
        metadata.prepare()
        view = metadata.mixed_precision_staging
        pages = -(-capacity // view.tokens_per_block)
        assert view.page_counts == [pages]
        assert view.read_capacity >= capacity
        if backend is TrtllmAttention:
            # Every staging page up to the capacity has key and value block offsets.
            assert -1 not in view.block_offsets[0, 0, :, :pages].flatten().tolist()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("dtype", ["auto", "mixed_nvfp4_bfloat16"])
def test_steady_decode_step_skips_mixed_precision(dtype):
    """The steady MTP decode shortcut stands in for prepare() only without mixed precision."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision NVFP4 KV requires Blackwell")
    block, cached = 128, 32
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    manager = _small_manager(block, mapping, dtype)
    try:
        manager.add_dummy_requests([0], [cached + block], is_gen=False)
        metadata = _small_metadata(TrtllmAttention, manager, mapping, block, cached)
        metadata.prepare()
        assert metadata.supports_steady_decode_step == (dtype == "auto")
    finally:
        manager.shutdown()


def _high_table_manager(high_layers):
    """A KV cache manager stub with mixed-precision high layers that records page-table fills."""
    manager = SimpleNamespace(
        mixed_precision_high_layers=high_layers, max_blocks_per_seq=4, filled=[]
    )
    manager._fill_page_table = lambda layer, request_ids, table: manager.filled.append(
        (layer, list(request_ids))
    )
    return manager


def _dsa_over(manager):
    """The DSA mixed-precision high-table state of a metadata over ``manager``."""
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        max_num_sequences=2,
        cuda_graph_buffers=None,
        num_seqs=2,
        request_ids=[3, 4],
        get_empty=lambda buffers, shape, cache_name, dtype, capture_graph: torch.full(
            shape, -1, dtype=dtype
        ),
    )
    return SimpleNamespace(metadata=metadata, mixed_precision_high_block_table=None)


@pytest.mark.cpu_only
def test_dsa_draft_over_a_mixed_draft_manager_fills_its_own_high_table():
    """A draft metadata over a separate mixed-precision draft manager fills its own
    mixed-precision high page table from that manager; the target's stays the target's."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAMetadata

    target_manager, draft_manager = _high_table_manager({0: 7}), _high_table_manager({0: 9})
    target, draft = _dsa_over(target_manager), _dsa_over(draft_manager)
    for dsa in (target, draft):
        DSAMetadata._allocate_mixed_precision_high_block_table(dsa, capture_graph=False)
        dsa._fill_mixed_precision_high_block_table = partial(
            DSAMetadata._fill_mixed_precision_high_block_table, dsa
        )
        DSAMetadata._fill_mixed_precision_high_block_tables(dsa)
    assert target.mixed_precision_high_block_table is not draft.mixed_precision_high_block_table
    assert target_manager.filled == [(7, [3, 4])]
    assert draft_manager.filled == [(9, [3, 4])]
    assert DSAMetadata.has_mixed_precision_high_rows.fget(draft) is True


@pytest.mark.parametrize("backend", ["dsa", "deepseek_v4"])
@pytest.mark.cpu_only
def test_draft_without_a_high_pool_has_no_mixed_precision_high_rows(backend):
    """A draft metadata whose manager has no mixed-precision high pool (DSA), or whose
    MTP layers have none (DeepSeek-V4, created with builds_mixed_precision_high_tables
    off), skips the high-pool writes and reads."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
        DeepseekV4Metadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAMetadata

    if backend == "dsa":
        draft = _dsa_over(_high_table_manager({}))
        DSAMetadata._allocate_mixed_precision_high_block_table(draft, capture_graph=False)
        assert draft.mixed_precision_high_block_table is None
        assert DSAMetadata.has_mixed_precision_high_rows.fget(draft) is False
        DSAMetadata._fill_mixed_precision_high_block_tables(draft)
        assert draft.metadata.kv_cache_manager.filled == []
    else:
        table = {4: torch.full((2, 4), -1, dtype=torch.int32)}
        draft = SimpleNamespace(
            mixed_precision_high_block_table=table, builds_mixed_precision_high_tables=False
        )
        assert DeepseekV4Metadata.has_mixed_precision_high_rows.fget(draft) is False
        target = SimpleNamespace(
            mixed_precision_high_block_table=table, builds_mixed_precision_high_tables=True
        )
        assert DeepseekV4Metadata.has_mixed_precision_high_rows.fget(target) is True
