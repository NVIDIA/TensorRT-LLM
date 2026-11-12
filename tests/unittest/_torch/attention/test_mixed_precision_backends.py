# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise mixed-precision cache routing and optimized Attention backends."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
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
        # Recomputing the last N tokens builds on reusing the prefix without them.
        {"dtype": "mixed_nvfp4_fp8", "mixed_precision_recompute_last_n": True},
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
            "kv_cache_compression_config",
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


def _draft_replay_metadata(cls):
    """A metadata stand-in with the real draft-swap bookkeeping and the class's draft hook, for
    prepare_attn_metadata_for_draft_replay with a separate draft manager."""
    metadata = Mock(spec=cls)
    metadata.enable_flash_mla = False
    metadata.kv_cache_manager = Mock(name="target_kv_cache_manager")
    metadata.draft_kv_cache_block_offsets = torch.zeros(1, dtype=torch.int32)
    metadata.prepare_for_draft_forward = lambda: cls.prepare_for_draft_forward(metadata)
    metadata.draft_replay_swapped_attrs = {}
    for helper in ("record_draft_swap", "swap_for_draft", "restore_draft_swaps"):
        setattr(metadata, helper, partial(getattr(TrtllmAttentionMetadata, helper), metadata))
    return metadata


_DRAFT_MANAGER = SimpleNamespace(host_kv_cache_block_offsets=torch.zeros(1, dtype=torch.int32))


@pytest.mark.cpu_only
def test_draft_replay_uses_the_draft_manager_mixed_precision_tables():
    """A draft forward on a separate draft manager reads that manager's mixed-precision high page
    table, block-offset copy and last-N restore plan (None when its layers take no mixed
    precision)."""
    from tensorrt_llm._torch.speculative.interface import (
        prepare_attn_metadata_for_draft_replay,
        restore_attn_metadata_after_draft_replay,
    )

    metadata = _draft_replay_metadata(TrtllmAttentionMetadata)
    target_table, draft_table = (torch.full((2, 4), -1, dtype=torch.int32) for _ in range(2))
    metadata.mixed_precision_high_block_table = target_table
    metadata.draft_mixed_precision_high_block_table = draft_table
    target_offsets = torch.zeros((1, 2, 2, 4), dtype=torch.int32)
    metadata.mixed_precision_low_block_offsets = target_offsets
    metadata.draft_mixed_precision_low_block_offsets = None
    target_restore, draft_restore = object(), object()
    metadata.mixed_precision_high_restore_spans = target_restore
    metadata.draft_mixed_precision_high_restore_spans = draft_restore

    saved = prepare_attn_metadata_for_draft_replay(metadata, _DRAFT_MANAGER)
    assert metadata.mixed_precision_high_block_table is draft_table
    assert metadata.mixed_precision_low_block_offsets is None
    assert metadata.mixed_precision_high_restore_spans is draft_restore
    restore_attn_metadata_after_draft_replay(metadata, saved)
    assert metadata.mixed_precision_high_block_table is target_table
    assert metadata.mixed_precision_low_block_offsets is target_offsets
    assert metadata.mixed_precision_high_restore_spans is target_restore


@pytest.mark.cpu_only
def test_draft_forward_swaps_in_draft_staging():
    """A mixed MTP draft manager's layers run on its own staging copy during the draft forward."""
    metadata = Mock()
    metadata.draft_replay_swapped_attrs = {}
    metadata.record_draft_swap = partial(TrtllmAttentionMetadata.record_draft_swap, metadata)
    metadata.swap_for_draft = partial(TrtllmAttentionMetadata.swap_for_draft, metadata)
    metadata.restore_draft_swaps = partial(TrtllmAttentionMetadata.restore_draft_swaps, metadata)
    metadata.sync_draft_mixed_precision_staging = partial(
        TrtllmAttentionMetadata.sync_draft_mixed_precision_staging, metadata
    )
    target, draft = Mock(name="target_staging"), Mock(name="draft_staging")
    metadata.mixed_precision_staging = target
    metadata.draft_mixed_precision_staging = draft

    TrtllmAttentionMetadata.prepare_for_draft_forward(metadata)
    assert metadata.mixed_precision_staging is draft
    metadata.restore_draft_swaps()
    assert metadata.mixed_precision_staging is target


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_precision_high_row_indices_point_at_the_high_pool():
    """Held positions point at the mixed-precision high pool's rows; released pages keep pool rows."""
    from tensorrt_llm._torch.attention.backends.sparse.mixed_precision_high_rows import (
        mixed_precision_high_row_indices,
    )

    manager = SimpleNamespace(
        layer_offsets={3: 0},
        mixed_precision_high_layer_offsets={0: 1},
        _mixed_precision_rows_per_page=lambda local: 4,
    )
    table = torch.tensor([[8, -1, 12]], dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(kv_cache_manager=manager, mixed_precision_high_block_table=table)
    attention = SimpleNamespace(layer_idx=3)
    topk = torch.tensor([[0, 5, 9, -1]], dtype=torch.int32, device="cuda")
    pool_rows = torch.tensor([[100, 101, 102, -1]], dtype=torch.int32, device="cuda")
    requests = torch.zeros(1, dtype=torch.int32, device="cuda")
    rows = mixed_precision_high_row_indices(metadata, attention, topk, pool_rows, requests)
    # High page 8 holds position 0 at row 8 * 4 + 1 * 4 + 0; the high pool released block 1.
    assert rows is pool_rows and rows.tolist() == [[36, 101, 53, -1]]


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
    """The overlap scheduler corrects KV lengths on the device after prepare(); staging follows."""
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

        def correct():
            if backend is TrtllmAttention:
                metadata.kv_lens_cuda[:1] += offsets
                metadata.on_update_kv_lens()
            else:
                metadata._uses_full_generation_page_table = True
                metadata.apply_spec_decode_kv_lens_offsets(
                    offsets, num_generations=0, tokens_per_generation=1, num_chunked_contexts=1
                )

        correct()
        corrected = [[cached - 2], [cached + block - 2]]
        assert view.lengths.tolist() == corrected
        # The draft forward runs on the draft staging copy with the corrected lengths.
        if backend is TrtllmAttention:
            metadata.prepare_for_draft_forward()
            assert metadata.mixed_precision_staging is metadata.draft_mixed_precision_staging
            assert metadata.mixed_precision_staging.lengths.tolist() == corrected
            metadata.restore_draft_swaps()
            assert metadata.mixed_precision_staging is view
        else:
            # The first draft view sync follows the correction; later syncs precede it.
            draft_view = metadata.get_draft_metadata(draft)
            assert draft_view.mixed_precision_staging is metadata.draft_mixed_precision_staging
            assert draft_view.mixed_precision_staging.lengths.tolist() == corrected
            metadata.prepare()
            correct()
            assert view.lengths.tolist() == corrected
            # Every draft forward starts here; the view reuses the same staging copy and
            # builds none of its own.
            assert metadata.get_draft_metadata(draft) is draft_view
            assert draft_view.mixed_precision_staging is metadata.draft_mixed_precision_staging
            assert draft_view.draft_mixed_precision_staging is None
            assert draft_view.mixed_precision_staging.lengths.tolist() == corrected
        # Draft layers run on another manager through this metadata: the view is
        # hidden and the staging lengths are left alone.
        metadata.kv_cache_manager = object()
        assert metadata.get_mixed_precision_kv_view(0) is None
        correct()
        assert view.lengths.tolist() == [[cached - 2], [cached + block - 2]]
        metadata.kv_cache_manager = manager
        assert metadata.get_mixed_precision_kv_view(0) is view
    finally:
        manager.shutdown()
        draft.shutdown()


class _ColdPageProvider:
    """Cold-page codec provider stand-in: the manager reads only its first/last-N window."""

    config = SimpleNamespace(algorithm="quantization_for_cold_page")

    def __init__(self, first_n: int, last_n: int):
        self.mixed_precision_high_window = (first_n, last_n)

    def create_cold_page_codec(self, cache_config, **kwargs):
        return None


def _small_manager(
    block: int,
    mapping: Mapping,
    dtype: str = "mixed_nvfp4_bfloat16",
    layout: str = "gqa",
    num_layers: int = 1,
):
    """A BF16 cache of `num_layers` layers; mixed precision ("mixed_*", or "cold" for the cold-page
    window) keeps one page at each end in the high pool. "mla" is a single latent KV head of 576
    (K only)."""
    mixed = (
        dict(mixed_precision_first_n=block, mixed_precision_last_n=block, pool_ratio=[0.75, 0.25])
        if dtype.startswith("mixed_")
        else {}
    )
    cold = dtype == "cold"
    return KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=65536,
            enable_block_reuse=True,
            dtype="auto" if cold else dtype,
            **(dict(pool_ratio=[0.75, 0.25]) if cold else mixed),
        ),
        CacheType.SELFKONLY if layout == "mla" else CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=1 if layout == "mla" else 2,
        head_dim=576 if layout == "mla" else 64,
        tokens_per_block=block,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=mapping,
        dtype=DataType.BF16,
        cold_page_codec_provider=_ColdPageProvider(block, block) if cold else None,
    )


def _small_metadata(backend, manager, mapping: Mapping, block: int, cached: int, draft=None):
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


@pytest.mark.parametrize("backend", ["dsa", "deepseek_v4"])
@pytest.mark.cpu_only
def test_draft_forward_hides_mixed_precision_high_rows(backend):
    """A draft manager without a mixed-precision high pool skips its writes, reads and restores."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.speculative.interface import (
        prepare_attn_metadata_for_draft_replay,
        restore_attn_metadata_after_draft_replay,
    )

    cls = {"dsa": DSAtrtllmAttentionMetadata, "deepseek_v4": DeepseekV4TrtllmAttentionMetadata}[
        backend
    ]
    metadata = _draft_replay_metadata(cls)
    # The property reads this class flag; a Mock attribute would be truthy.
    metadata.mixed_precision_block_offsets_redirect = cls.mixed_precision_block_offsets_redirect
    metadata.draft_sliding_block_tables = None
    metadata.draft_mixed_precision_high_block_table = None
    table = torch.full((2, 4), -1, dtype=torch.int32)
    metadata.mixed_precision_high_block_table = table
    # The target restores high rows in this forward; the draft manager restores none.
    plan = object()
    metadata.mixed_precision_high_restore = plan
    metadata.draft_mixed_precision_high_restore = None

    saved = prepare_attn_metadata_for_draft_replay(metadata, _DRAFT_MANAGER)
    assert cls.has_mixed_precision_high_rows.fget(metadata) is False
    assert metadata.mixed_precision_high_restore is None
    restore_attn_metadata_after_draft_replay(metadata, saved)
    assert metadata.mixed_precision_high_block_table is table
    assert metadata.mixed_precision_high_restore is plan
    assert cls.has_mixed_precision_high_rows.fget(metadata) is True


@pytest.mark.cpu_only
def test_high_rows_is_sparse_only_and_dense_reports_the_redirect():
    """The dense metadata has no high-rows property and reports the block-offset redirect; the sparse
    metadata reports high rows and no redirect. The MLA write path keys the sparse row store on the
    former, so it never runs for dense MLA models."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )

    table = torch.full((2, 4), -1, dtype=torch.int32)
    assert not hasattr(TrtllmAttentionMetadata, "has_mixed_precision_high_rows")
    dense = Mock(spec=TrtllmAttentionMetadata)
    dense.mixed_precision_block_offsets_redirect = True
    dense.mixed_precision_high_block_table = table
    assert TrtllmAttentionMetadata.redirects_mixed_precision_block_offsets.fget(dense) is True
    sparse = Mock(spec=DSAtrtllmAttentionMetadata)
    sparse.mixed_precision_block_offsets_redirect = False
    sparse.mixed_precision_high_block_table = table
    assert DSAtrtllmAttentionMetadata.has_mixed_precision_high_rows.fget(sparse) is True
    assert DSAtrtllmAttentionMetadata.redirects_mixed_precision_block_offsets.fget(sparse) is False
    sparse.mixed_precision_high_block_table = None
    assert DSAtrtllmAttentionMetadata.has_mixed_precision_high_rows.fget(sparse) is False


@pytest.mark.cpu_only
def test_cold_page_copy_back_runs_only_for_layers_with_a_high_layer():
    """Only a layer that has a high layer restores its fresh last-N rows and copies its rows back; a
    layer without one (a sliding-window layer of the same model) runs the staging context untouched."""
    metadata = Mock(spec=TrtllmAttentionMetadata)
    # A Mock attribute would be truthy on its own; set the property's value explicitly.
    metadata.redirects_mixed_precision_block_offsets = True
    metadata.kv_cache_manager = Mock(layer_offsets={0: 0, 1: 1}, mixed_precision_high_layers={0: 2})
    metadata._layer_takes_mixed_precision = partial(
        TrtllmAttentionMetadata._layer_takes_mixed_precision, metadata
    )
    backend = "tensorrt_llm._torch.attention.backends.trtllm"
    with (
        patch(f"{backend}.restore_mixed_precision_high_rows_from_low") as restore,
        patch(f"{backend}.store_mixed_precision_low_rows") as store,
    ):
        with TrtllmAttentionMetadata.mixed_precision_kv_staging(metadata, 0):
            pass
        restore.assert_called_once_with(metadata, 0)
        store.assert_called_once_with(metadata, 0)
        restore.reset_mock()
        store.reset_mock()
        with TrtllmAttentionMetadata.mixed_precision_kv_staging(metadata, 1):
            pass
        restore.assert_not_called()
        store.assert_not_called()


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
    """Inside a CUDA graph, each draft forward's staging copy follows the corrected KV lengths."""
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
        metadata.prepare()
        offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # The overlap scheduler corrects KV lengths on the device, then a draft forward runs.
            metadata.kv_lens_cuda[:1] += offsets
            metadata.on_update_kv_lens()
            metadata.prepare_for_draft_forward()
            metadata.restore_draft_swaps()
        for offset in (-1, -3):
            # Each step, prepare() writes uncorrected lengths; the replay must correct both copies.
            metadata.prepare()
            offsets.fill_(offset)
            graph.replay()
            corrected = [[cached + offset], [cached + 1 + offset]]
            assert metadata.mixed_precision_staging.lengths.tolist() == corrected
            assert metadata.draft_mixed_precision_staging.lengths.tolist() == corrected
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


@pytest.mark.parametrize("dtype", ["auto", "mixed_nvfp4_bfloat16", "cold"])
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mixed_precision_high_restore_covers_fresh_window_pages():
    """A context request that reused its prefix without its last-N high pages restores the cached
    tokens of its fresh pages after the sinks, once; a compressed row needs all of its tokens."""
    from tensorrt_llm._torch.attention.backends.sparse.mixed_precision_high_rows import (
        mixed_precision_high_restore_rows,
        take_mixed_precision_high_restore,
    )

    high_pages = {
        # Two sink pages, one page that left the window, then fresh window pages.
        11: [7, 6, -1, 8, 9],
        # The prefix ends inside the sinks, so no window page is fresh.
        12: [5, 6],
    }

    def fill(layer, request_ids, out):
        for row, request_id in enumerate(request_ids):
            pages = high_pages[request_id]
            out[row, : len(pages)] = pages

    manager = SimpleNamespace(
        mixed_precision_restore_requests={11, 12, 99},
        tokens_per_block=4,
        first_n=6,
        max_blocks_per_seq=6,
        mixed_precision_high_layers={0: 1},
        _fill_page_table=fill,
    )
    # Three context requests, then a generation request.
    metadata = SimpleNamespace(num_contexts=3, request_ids=[10, 11, 12, 13])
    spans = take_mixed_precision_high_restore(metadata, manager, [0, 14, 6, 40])
    assert spans == [(1, 12, 14), (2, 6, 6)]
    requests, positions = mixed_precision_high_restore_rows(spans)
    assert requests.tolist() == [1, 1] and positions.tolist() == [12, 13]
    # Two tokens per row: only row 6 (tokens 12-13) is complete.
    requests, positions = mixed_precision_high_restore_rows(spans, 2)
    assert requests.tolist() == [1] and positions.tolist() == [6]
    assert mixed_precision_high_restore_rows(spans, 4) is None
    # Taken once; a request outside this batch keeps waiting.
    assert manager.mixed_precision_restore_requests == {99}
    assert take_mixed_precision_high_restore(metadata, manager, [0, 14, 6, 40]) == []


def _nvfp4_pool(num_pool_tokens, row_width):
    """NVFP4 data and scale pools whose rows all decode differently."""
    codes = torch.arange(1, num_pool_tokens + 1, dtype=torch.uint8, device="cuda")
    data_pool = (codes | (codes << 4))[:, None].expand(-1, row_width // 2).contiguous()
    scale_pool = torch.full(
        (num_pool_tokens, row_width // 16), 2.0, dtype=torch.float8_e4m3fn, device="cuda"
    )
    return data_pool, scale_pool


def _nvfp4_read(data_pool, scale_pool, pool_rows, head_dim, residual_dim, scale):
    """The FP8 rows the NVFP4 MLA read returns for `pool_rows`."""
    rows = torch.empty((len(pool_rows), 1, head_dim), dtype=torch.float8_e4m3fn, device="cuda")
    indices = torch.tensor(pool_rows, dtype=torch.int32, device="cuda").view(-1, 1)
    torch.ops.trtllm.nvfp4_mla_kv_cache_gather_direct(
        data_pool, scale_pool, indices, rows, scale, residual_dim, data_pool.shape[0]
    )
    return rows


def _restore_case(rows_per_page, head_dim):
    """Request 1 restores rows 2-4: rows 2-3 sit in its block 1 (pool page 0), row 4 in its block 2
    (pool page 3); its high pages are 2 and 3 (page 0 holds its sink).

    Returns the restore plan, the high page table, the block table, and a high pool
    [pages, kv, rows, heads, width] filled with a sentinel no restored row equals.
    """
    plan = (
        torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda"),
        torch.tensor([2, 3, 4], dtype=torch.int32, device="cuda"),
    )
    high_table = torch.tensor([[1, -1, -1], [0, 2, 3]], dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[4, -1, -1], [2, 0, 3]], dtype=torch.int32, device="cuda")
    high = torch.full(
        (4, 1, rows_per_page, 1, head_dim), 448.0, dtype=torch.float8_e4m3fn, device="cuda"
    )
    return plan, high_table, block_table, high


def _expected_high(high, restored):
    """The sentinel high pool with `restored` at rows 2, 3 (high page 2) and row 4 (high page 3)."""
    expected = torch.full_like(high, 448.0).view(torch.uint8)
    expected[[2, 2, 3], 0, [0, 1, 0]] = restored.view(torch.uint8)
    return expected


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="NVFP4 MLA gather requires Blackwell",
)
def test_dsa_restore_fills_fresh_high_rows_from_the_pool():
    """Restored high rows hold what the NVFP4 read returns for their pool rows; other rows stay."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention

    tokens_per_block, num_pool_tokens, head_dim, residual_dim = 2, 10, 576, 64
    data_pool, scale_pool = _nvfp4_pool(num_pool_tokens, head_dim + residual_dim)
    host_pool_pointers = torch.zeros((1, 2, 2), dtype=torch.int64)
    host_pool_pointers[0, 0, 0] = data_pool.data_ptr()
    host_pool_pointers[0, 0, 1] = scale_pool.data_ptr()
    plan, high_table, block_table, high = _restore_case(tokens_per_block, head_dim)
    manager = SimpleNamespace(
        head_dim=head_dim,
        mla_kv_cache_residual_dim=residual_dim,
        layer_offsets={0: 0},
        mixed_precision_high_layers={0: 1},
        _get_mixed_precision_buffers=lambda *args: high,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        mixed_precision_high_restore=plan,
        mixed_precision_high_block_table=high_table,
        _cached_block_table_ctx=block_table,
        _cached_tokens_per_block=tokens_per_block,
        _cached_num_pool_tokens=num_pool_tokens,
        host_kv_cache_pool_pointers=host_pool_pointers,
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
    )
    scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")
    owner = SimpleNamespace(layer_idx=0, _nvfp4_mla_kv_scale_quant_orig=scale)
    DSATrtllmAttention._restore_mixed_precision_high_rows(owner, metadata, 0, 1, 0)

    # Positions 2, 3 -> pool rows 0, 1; position 4 -> pool row 6.
    read = _nvfp4_read(data_pool, scale_pool, [0, 1, 6], head_dim, residual_dim, scale)
    assert torch.equal(high.view(torch.uint8), _expected_high(high, read))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="NVFP4 MLA gather requires Blackwell",
)
@pytest.mark.parametrize("residual_dim", [0, 64])
def test_deepseek_v4_restore_fills_fresh_high_rows_from_the_pool(residual_dim):
    """Restored compressed high rows hold what the NVFP4 read returns for their pool rows."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend import (
        DeepseekV4TrtllmAttention,
    )

    ratio, tokens_per_block, head_dim, num_pool_tokens = 4, 8, 512, 12
    rows_per_page, row_bytes = tokens_per_block // ratio, (head_dim + residual_dim) // 2
    data_pool, scale_pool = _nvfp4_pool(num_pool_tokens, head_dim + residual_dim)
    plan, high_table, block_table, high = _restore_case(rows_per_page, head_dim)
    manager = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        nvfp4_residual_dim=residual_dim,
        layer_offsets={0: 0},
        mixed_precision_high_layers={0: 1},
        _get_mixed_precision_buffers=lambda *args: high,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=manager, mixed_precision_high_block_table={ratio: high_table}
    )
    scale = torch.tensor([0.25], dtype=torch.float32, device="cuda")
    owner = SimpleNamespace(
        compress_ratio=ratio, head_dim=head_dim, layer_idx=0, _nvfp4_compress_scale_quant_orig=scale
    )
    # This layer's rows start 2 rows into the pool.
    DeepseekV4TrtllmAttention._restore_mixed_precision_high_rows(
        owner, metadata, plan, data_pool, scale_pool, num_pool_tokens, 2 * row_bytes, block_table
    )

    # Rows 2, 3 -> pool rows 2, 3; row 4 -> pool row 2 + 3 * 2 = 8.
    read = _nvfp4_read(data_pool, scale_pool, [2, 3, 8], head_dim, residual_dim, scale)
    assert torch.equal(high.view(torch.uint8), _expected_high(high, read))


@pytest.mark.parametrize("num_layers", (1, 2))
@pytest.mark.parametrize("layout", ("gqa", "mla"))
def test_cold_page_window_redirects_dense_block_offsets(layout, num_layers):
    """TRTLLM block offsets of the held blocks point at the high layer's blocks of the same pages;
    after attention the step's rows come back to the layer's own blocks. GQA has K and V blocks,
    dense MLA one latent block per page; with two layers, each layer's own blocks of a page."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision KV requires Blackwell")
    from tensorrt_llm._torch.attention.backends.mixed_precision_rows import (
        store_mixed_precision_low_rows,
    )

    block, cached = 32, 4 * 32 + 5
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    manager = _small_manager(block, mapping, "cold", layout, num_layers)
    try:
        # A decode request with 133 cached tokens writes one: the high pool holds the first block
        # (first_n) and the last two (last_n + 1 tokens); the window released blocks 1 and 2.
        manager.add_dummy_requests([0], [cached + 1], is_gen=True)
        metadata = TrtllmAttentionMetadata(
            max_num_requests=2,
            max_num_tokens=2,
            max_num_sequences=2,
            kv_cache_manager=manager,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            num_contexts=0,
            request_ids=[0],
            prompt_lens=[cached],
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached]),
            kv_layout="HND",
            mapping=mapping,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
        metadata.prepare()
        blocks = -(-(cached + 1) // block)
        kv_factor = manager.kv_factor
        # The layers share the pool and its block-offset table; an offset counts blocks from
        # the layer's own base, so one table addresses every layer's blocks.
        delta, kv_offset = manager.mixed_precision_high_block_deltas[0], int(manager.kv_offset[0])
        offsets = metadata.kv_cache_block_offsets[0, 0, :, :blocks].tolist()
        kept = metadata.mixed_precision_low_block_offsets[0, 0, :, :blocks].tolist()
        # The MTP draft step corrects the query length (one new token) but not the metadata's
        # separate KV-length copy, which under CUDA graphs still says two; the copy back follows
        # the query length, so token 4 of the layer's block must stay untouched below.
        metadata.seq_lens_kv = torch.tensor([2], dtype=torch.int32)
        assert int(metadata.seq_lens_kv_cuda[0]) == 2 and int(metadata.seq_lens_cuda[0]) == 1
        torch.manual_seed(0)
        for layer_idx in range(num_layers):
            pages = torch.full((2, 1, blocks), -1, dtype=torch.int32)
            manager.fill_mixed_precision_page_tables([0], layer_idx, pages.numpy())
            low, high = pages[0, 0].tolist(), pages[1, 0].tolist()
            assert metadata.mixed_precision_high_block_table[0, :blocks].tolist() == high
            held = [i for i, page in enumerate(high) if page >= 0]
            assert held == [0, 3, 4]
            for kv in range(kv_factor):
                assert kept[kv] == [page * kv_factor + kv * kv_offset for page in low]
                assert offsets[kv] == [
                    (high[i] * kv_factor + delta if i in held else low[i] * kv_factor)
                    + kv * kv_offset
                    for i in range(blocks)
                ]
            # The redirected offsets address the high layer's blocks of those pages (HND: a block
            # is heads x tokens rows of one head's bytes), counted from the layer's own blocks.
            low_pool = manager.get_buffers(layer_idx, "HND", mixed_precision_pool="low")
            high_pool = manager.get_buffers(layer_idx, "HND", mixed_precision_pool="high")
            rows = low_pool.view(-1, low_pool.shape[4])
            heads = high_pool.shape[2]
            for i in held:
                for kv in range(kv_factor):
                    assert (
                        rows[offsets[kv][i] * heads * block].data_ptr()
                        == high_pool[high[i], kv].data_ptr()
                    )
            # Attention wrote this step's token (position 133: block 4, token 5) into the high
            # layer; the copy back fills that token of every head in the layer's block and nothing
            # else. Both views cover the shared pool, so only the layer's own pages take the
            # sentinel.
            for page in low:
                low_pool[page].fill_(7)
            written = torch.randn(kv_factor, heads, high_pool.shape[4], device="cuda").to(
                torch.bfloat16
            )
            high_pool[high[4], :, :, 5].copy_(written)
            high_pool[high[4], :, :, 4].fill_(1.0)
            store_mixed_precision_low_rows(metadata, layer_idx)
            torch.cuda.synchronize()
            assert torch.equal(low_pool[low[4], :, :, 5].view(torch.bfloat16), written)
            assert (low_pool[low[4], :, :, 4] == 7).all() and (low_pool[low[3]] == 7).all()
            assert (low_pool[low[0]] == 7).all()
    finally:
        manager.shutdown()


def test_cold_page_window_restores_fresh_high_blocks_from_the_layer():
    """A restoring context request's cached tokens in its high blocks after the sinks are copied
    from the layer's own blocks, once, before attention reads them; nothing else changes."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("Mixed-precision KV requires Blackwell")
    from tensorrt_llm._torch.attention.backends.mixed_precision_rows import (
        restore_mixed_precision_high_rows_from_low,
    )

    block, cached = 32, 4 * 32 + 5
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    manager = _small_manager(block, mapping, "cold")
    try:
        # A context request with 133 cached tokens adds 32. Its high blocks are all fresh, so the
        # cached tokens after the sink block, 32-132, are restored.
        manager.add_dummy_requests([0], [cached + block], is_gen=False)
        manager.mixed_precision_restore_requests.add(0)
        metadata = TrtllmAttentionMetadata(
            max_num_requests=2,
            max_num_tokens=block,
            max_num_sequences=2,
            kv_cache_manager=manager,
            seq_lens=torch.tensor([block], dtype=torch.int32),
            num_contexts=1,
            request_ids=[0],
            prompt_lens=[cached + block],
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached]),
            kv_layout="HND",
            mapping=mapping,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
        metadata.prepare()
        assert not manager.mixed_precision_restore_requests
        ends, counts, max_count = metadata.mixed_precision_high_restore_spans
        assert ends.tolist() == [cached] and counts.tolist() == [cached - block]
        assert max_count == cached - block

        blocks = -(-(cached + block) // block)
        pages = torch.full((2, 1, blocks), -1, dtype=torch.int32)
        manager.fill_mixed_precision_page_tables([0], 0, pages.numpy())
        low, high = pages[0, 0].tolist(), pages[1, 0].tolist()
        assert all(page >= 0 for page in high)
        # HND: [pages, K/V, heads, tokens, width]; the low view counts bytes.
        low_pool = manager.get_buffers(0, "HND", mixed_precision_pool="low")
        high_pool = manager.get_buffers(0, "HND", mixed_precision_pool="high")
        torch.manual_seed(0)
        for page in low:
            low_pool[page].copy_(torch.randint_like(low_pool[page], 0, 120))
        for page in high:
            high_pool[page].fill_(7.0)
        expected = [high_pool[page].clone() for page in high]
        for i in (1, 2, 3):
            expected[i] = low_pool[low[i]].view(torch.bfloat16).clone()
        expected[4][:, :, :5] = low_pool[low[4], :, :, :5].view(torch.bfloat16)
        restore_mixed_precision_high_rows_from_low(metadata, 0)
        torch.cuda.synchronize()
        for i, page in enumerate(high):
            assert torch.equal(high_pool[page], expected[i]), i
        # The request restored once: the next forward plans nothing.
        metadata.prepare()
        assert metadata.mixed_precision_high_restore_spans is None
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dsa_cold_page_window_restore_copies_the_layer_rows():
    """With the cold-page window, restored high rows are the layer's own pool rows, bit for bit;
    other rows stay."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention

    tokens_per_block, num_pool_rows, head_dim = 2, 10, 576
    pool = torch.randint(0, 120, (num_pool_rows, 1, head_dim), dtype=torch.uint8, device="cuda")
    plan, high_table, block_table, high = _restore_case(tokens_per_block, head_dim)
    manager = SimpleNamespace(
        layer_offsets={0: 0},
        mixed_precision_high_layers={0: 1},
        mixed_precision_high_dtype=DataType.FP8,
        get_primary_pool_page_index_params=lambda layer: (1, 0),
        _get_mixed_precision_buffers=lambda *args: high,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        mixed_precision_high_restore=plan,
        mixed_precision_high_block_table=high_table,
        _ensure_pool_view_cached=lambda: None,
        _cached_block_table_ctx=block_table,
        _cached_tokens_per_block=tokens_per_block,
        _cached_pool_view=pool,
    )
    DSATrtllmAttention._restore_mixed_precision_high_rows_from_pool(
        SimpleNamespace(layer_idx=0), metadata, 0
    )

    # Positions 2, 3 -> pool rows 0, 1; position 4 -> pool row 6.
    assert torch.equal(high.view(torch.uint8), _expected_high(high, pool[[0, 1, 6]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deepseek_v4_cold_page_window_restore_copies_the_layer_rows():
    """With the cold-page window, restored compressed high rows are the layer's own compressed rows,
    bit for bit; other rows stay."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend import (
        DeepseekV4TrtllmAttention,
    )

    ratio, tokens_per_block, head_dim = 4, 8, 512
    rows_per_page = tokens_per_block // ratio
    own = torch.randint(0, 120, (5, rows_per_page, head_dim), dtype=torch.uint8, device="cuda")
    plan, high_table, block_table, high = _restore_case(rows_per_page, head_dim)
    manager = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        layer_offsets={0: 0},
        mixed_precision_high_layers={0: 1},
        mixed_precision_high_dtype=DataType.FP8,
        get_buffers=lambda layer_idx, attn_type: own.view(torch.float8_e4m3fn),
        _get_mixed_precision_buffers=lambda *args: high,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=manager, mixed_precision_high_block_table={ratio: high_table}
    )
    DeepseekV4TrtllmAttention._restore_mixed_precision_high_rows_from_pool(
        SimpleNamespace(compress_ratio=ratio, layer_idx=0), metadata, plan, block_table
    )

    # Rows 2, 3 -> own page 0, rows 0 and 1; row 4 -> own page 3, row 0.
    restored = own[[0, 0, 3], [0, 1, 0]][:, None]
    assert torch.equal(high.view(torch.uint8), _expected_high(high, restored))
