# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native and Flash provider numerics, dispatch and metadata lifecycle."""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2FlashInfer, CSA2FlashMLA
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params


def _reference(q, swa, extra, swa_valid, extra_valid, sink):
    kv, valid = swa, swa_valid
    if extra is not None:
        kv = torch.cat((kv, extra), dim=1)
        valid = torch.cat((valid, extra_valid), dim=1)
    kv = torch.where(valid[..., None], kv, 0)
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * q.shape[-1] ** -0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    probs = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1).softmax(-1)[
        ..., :-1
    ]
    return torch.einsum("qhk,qkd->qhd", probs, kv.float()).to(q.dtype)


def _backend(
    heads,
    layer_idx=20,
    layout=None,
    compute_backend="auto",
    use_packed=False,
    kv_cache_dtype="auto",
    quant_config=None,
    fp8_staging=None,
):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import get_csa2_backend
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend
    from tensorrt_llm._utils import is_sm_100f

    if compute_backend in ("auto", "trtllm") and not is_sm_100f():
        pytest.skip("TRTLLM dynamic sparse MLA requires SM100-family")
    params = CSA2Params(
        layout=layout,
        compute_backend=compute_backend,
        use_packed_sparse_attention=use_packed,
        # ``None`` leaves staging at its default, so the backend resolves it from
        # its own support the way a served request does.
        use_fp8_staging=fp8_staging,
    )
    assert get_attention_backend("TRTLLM", params) is get_csa2_backend(params)
    attn = create_attention(
        "TRTLLM",
        layer_idx,
        heads,
        512,
        num_kv_heads=1,
        is_mla_enable=True,
        q_lora_rank=1280,
        kv_lora_rank=448,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        rope_append=False,
        sparse_params=params,
        kv_cache_dtype=kv_cache_dtype,
        quant_config=quant_config,
    )
    assert isinstance(attn, TrtllmAttention)
    return attn


def _helper_forward(attn, q, metadata, args):
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (
        CSA2FlashInfer,
        CSA2FlashMLA,
    )

    if attn.sparse_params.use_packed_sparse_attention:
        return attn.forward_packed(q, metadata, args)
    if not hasattr(attn, "_test_flash_helper"):
        helper = CSA2FlashMLA if attn.compute_backend == "flash_mla" else CSA2FlashInfer
        attn._test_flash_helper = helper(attn)
    compute = (
        attn._test_flash_helper.forward_context
        if args.attention_input_type == AttentionInputType.context_only
        else attn._test_flash_helper.forward_generation
    )
    return compute(q, metadata, args)


def _inputs(q, swa, extra, swa_valid, extra_valid, sink):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    swa_ids = torch.arange(swa.shape[0] * swa.shape[1], device=q.device).reshape_as(swa_valid)
    main_ids = (
        None
        if extra is None
        else torch.arange(extra.shape[0] * extra.shape[1], device=q.device).reshape_as(extra_valid)
    )
    inputs = CSA2BackendForwardArgs(
        swa_pool=pack_rows(swa.flatten(0, 1), "swa"),
        swa_indices=torch.where(swa_valid, swa_ids, -1),
        main_pool=None if extra is None else pack_rows(extra.flatten(0, 1), "main"),
        topk_indices=None if extra is None else torch.where(extra_valid, main_ids, -1),
    )
    return AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        attention_sinks=sink,
        sparse_backend_args=inputs,
    )


def _decoded_reference(q, args, swa_valid, extra_valid, sink):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    inputs = args.sparse_backend_args
    swa = unpack_rows(inputs.swa_pool, 512, "swa").reshape(q.shape[0], -1, 512)
    main = (
        None
        if inputs.main_pool is None
        else unpack_rows(inputs.main_pool, 512, "main").reshape(q.shape[0], -1, 512)
    )
    return _reference(q, swa, main, swa_valid, extra_valid, sink)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("extra_width", [0, 17, 512])
@torch.inference_mode()
def test_native_dual_pool(heads, extra_width, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

    calls = []
    original = FallbackFmha.forward

    def record(attn, *args, **kwargs):
        calls.append(attn.attn.sparse_params.algorithm)
        return original(attn, *args, **kwargs)

    monkeypatch.setattr(FallbackFmha, "forward", record)
    torch.manual_seed(451)
    attn = _backend(heads)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(3, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(3, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 2:] = False
    swa_valid[1] = False
    extra = (
        torch.randn(3, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(3, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[0, 1::2] = False  # holes before later valid entries
        extra_valid[1] = False
    sink = torch.randn(heads, device="cuda")
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    metadata = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    out = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    torch.cuda.synchronize()
    assert calls == ["csa2"]
    torch.testing.assert_close(
        out, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_trtllm_graph_replay_resets_sparse_state():
    attn = _backend(64)
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    extra_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    sink = torch.zeros(64, device="cuda")

    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    frame = CSA2TrtllmMetadata.for_query_tile(q, 17)

    def run():
        frame.is_cuda_graph = torch.cuda.is_current_stream_capturing()
        args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
        return attn.forward(q.flatten(1), None, None, frame, forward_args=args).view_as(q)

    for _ in range(3):
        run()
    pointers = frame.pool_pointers.clone()
    workspace_ptr = frame.workspace.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    # Larger eager queries use independent caller-owned metadata. No recursive
    # warmup or backend-private frame allocation is needed.
    large_q = q.repeat(8, 1, 1)
    large_frame = CSA2TrtllmMetadata.for_query_tile(large_q, 17)
    large_args = _inputs(
        large_q,
        swa.repeat(8, 1, 1),
        extra.repeat(8, 1, 1),
        swa_valid.repeat(8, 1),
        extra_valid.repeat(8, 1),
        sink,
    )
    attn.forward(large_q.flatten(1), None, None, large_frame, forward_args=large_args)
    for width in (17, 0, 3, 17):
        extra_valid.zero_()
        extra_valid[:, :width] = True
        swa_valid[0, 1:] = width != 0
        q.mul_(-1)
        extra.mul_(-1)
        graph.replay()
        # Quantize/dequantize independently of the prediction hook's gathering.
        from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

        swa_ref = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
        extra_ref = unpack_rows(pack_rows(extra, "main"), 512, "main")
        torch.testing.assert_close(
            output,
            _reference(q, swa_ref, extra_ref, swa_valid, extra_valid, sink),
            atol=0.03,
            rtol=0.03,
        )
        torch.testing.assert_close(frame.pool_pointers, pointers, atol=0, rtol=0)
        assert frame.workspace.data_ptr() == workspace_ptr
        assert large_frame.workspace.data_ptr() != workspace_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_standard_backend_multiple_tiles_and_reuse(monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
        CSA2CacheManager,
        CSA2CacheRole,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
        CSA2BackendForwardArgs,
        CSA2ForwardState,
        CSA2Layout,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    predictions = []
    original_predict = CSA2Indexer.sparse_attn_indexer

    def predict(indexer, metadata, hidden_states, *args, **kwargs):
        predictions.append((indexer.layer_idx, hidden_states.shape[0]))
        return original_predict(indexer, metadata, hidden_states, *args, **kwargs)

    monkeypatch.setattr(CSA2Indexer, "sparse_attn_indexer", predict)
    torch.manual_seed(419)
    layout = CSA2Layout((1, 1, 1), (0,), (0, 2), index_topk=4, window_size=4)
    count, heads = 19, 8
    manager = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, host_cache_size=0, dtype="fp8"),
        CacheType.SELFKONLY,
        num_layers=3,
        tokens_per_block=128,
        max_seq_len=64,
        max_batch_size=1,
        max_num_tokens=count,
        mapping=Mapping(),
        vocab_size=8192,
        layout=layout,
    )
    try:
        request = manager._create_kv_cache(100, None, [])
        assert manager._resume_and_restore(100, request)
        assert request.resize(count)
        runtime = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=count, kv_cache_manager=manager
        )
        positions = torch.arange(count, device="cuda")
        q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
        swa = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
        main = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
        index_k = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
        index_q = torch.randn(count, 2, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.ones(count, 2, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(heads, device="cuda")
        runtime.reset_routing()
        global_base = manager.get_cache_indices(100, 0, CSA2CacheRole.GLOBAL)[0] * 128
        global_slots = global_base + torch.arange(6, device="cuda")
        runtime.csa2_token_requests = torch.zeros(count, dtype=torch.int64, device="cuda")
        runtime.csa2_request_query_ranges = ((0, count),)
        runtime.csa2_request_start_positions = (0,)
        runtime.csa2_num_context_requests = 1
        runtime.csa2_global_page_tables = {
            0: torch.tensor([[global_base // 128]], dtype=torch.int32, device="cuda")
        }
        runtime.csa2_global_page_sizes = {0: 128}
        runtime.csa2_global_max_positions = {0: 6}
        runtime.csa2_main_write_slots = {0: global_slots}
        runtime.csa2_kv_sources = {0: 0, 1: 0, 2: 0}
        runtime.csa2_swa_indices = {}
        runtime.csa2_swa_write_slots = {}
        runtime.csa2_visible_lengths = {}
        for layer in range(3):
            swa_base = manager.get_cache_indices(100, layer, CSA2CacheRole.SWA)[0] * 128
            local = positions[:, None] - torch.arange(4, device="cuda")[None, :]
            swa_slots = torch.where(local >= 0, local + swa_base, -1)
            runtime.csa2_swa_indices[layer] = swa_slots
            runtime.csa2_swa_write_slots[layer] = positions + swa_base
            runtime.csa2_visible_lengths[layer] = positions.remainder(6) + 1
            args_dict = {}
            if layer != 1:
                args_dict.update(index_q=index_q * (-1 if layer else 1), index_weights=weights)
            if layer == 0:
                args_dict.update(main_kv=main, index_k=index_k)
            state = CSA2ForwardState(metadata=runtime, swa_kv=swa, **args_dict)
            backend = _backend(heads, layer, layout)
            outputs = []
            for start in range(0, count, 16):
                tile = q[start : start + 16]
                frame = runtime.get_query_tile_metadata(tile, layout.index_topk)
                args = AttentionForwardArgs(
                    attention_input_type=AttentionInputType.generation_only,
                    attention_sinks=sink,
                    sparse_backend_args=CSA2BackendForwardArgs(state=state, query_start=start),
                )
                outputs.append(
                    backend.forward(tile.flatten(1), None, None, frame, forward_args=args).view_as(
                        tile
                    )
                )
            actual = torch.cat(outputs)
            logical = runtime.csa2_indices[0 if layer == 1 else layer]
            slots = runtime.global_slot_tile(layer, 0, count, logical)
            swa_values = gather_rows(manager.get_swa_buffer(layer), swa_slots, 512, "swa")
            main_values = gather_rows(manager.get_main_buffer(0), slots, 512, "main")
            expected = _reference(q, swa_values, main_values, swa_slots >= 0, slots >= 0, sink)
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
            frame = runtime.get_query_tile_metadata(q[-3:], 4)
            assert frame.host_total_kv_lens.tolist() == [0, 3 * 256]
        assert predictions == [(0, count), (2, count)]
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shared_metadata_resets_global_to_swa_inputs():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    sink = torch.zeros(8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17)
    first, second = _backend(8, 20), _backend(8, 21)
    args = _inputs(q, swa, main, swa_valid, main_valid, sink)
    first.forward(q.flatten(1), None, None, metadata, forward_args=args)
    assert args.sparse_runtime_params.aux_kv_cache_pool_ptr is not None
    # Reuse the runtime carrier and metadata across distinct layers, but remove
    # main selection. No stale main pointer or sparse length may survive.
    args.sparse_backend_args = _inputs(q, -swa, None, swa_valid, None, sink).sparse_backend_args
    args.output = None
    output = second.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    assert args.sparse_runtime_params.aux_kv_cache_pool_ptr is None
    torch.testing.assert_close(metadata.prepared_lens, torch.full_like(metadata.prepared_lens, 4))
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, None, sink), atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cold_metadata_rejects_capture():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8)
    q = torch.zeros(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(2, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, 1, device="cuda", dtype=torch.bool)
    sink = torch.zeros(8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    args = _inputs(q, swa, None, valid, None, sink)
    metadata.is_cuda_graph = True
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="Warm up CSA2 metadata"):
        with torch.cuda.graph(graph):
            attn.forward(q.flatten(1), None, None, metadata, forward_args=args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["flash_mla", "flashinfer"])
@pytest.mark.parametrize("context", [False, True])
@torch.inference_mode()
def test_alternative_helper_dispatch(implementation, context, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Selected CSA2 implementation fell through to native fallback")

    monkeypatch.setattr(FallbackFmha, "forward", unexpected_fallback)
    attn = _backend(8, compute_backend=implementation)
    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    main_valid[0, ::2] = False
    swa_valid[1] = False
    main_valid[1] = False
    sink = torch.linspace(-2, 2, 8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17)
    args = _inputs(q, swa, main, swa_valid, main_valid, sink)
    if context:
        from tensorrt_llm._torch.attention.backends.interface import AttentionInputType

        metadata._bind_context_tile([q.shape[0]])
        args.attention_input_type = AttentionInputType.context_only
    args.output = torch.empty_like(q).flatten(1)
    actual = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    assert actual.data_ptr() == args.output.data_ptr()
    torch.testing.assert_close(
        actual, _decoded_reference(q, args, swa_valid, main_valid, sink), atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "option", ["custom_mask", "out_scale", "output_sf", "output_shape", "output_dtype"]
)
@torch.inference_mode()
def test_alternative_backend_rejects_unsupported_options(option):
    from tensorrt_llm._torch.attention.backends.interface import CustomAttentionMask
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8, compute_backend="flashinfer")
    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(8, device="cuda"))
    if option == "custom_mask":
        args.attention_mask = CustomAttentionMask.CUSTOM
    elif option == "out_scale":
        args.out_scale = torch.ones(1, device="cuda")
    elif option == "output_shape":
        args.output = torch.empty(2, q.shape[1] * 512, device=q.device, dtype=q.dtype)
    elif option == "output_dtype":
        args.output = torch.empty_like(q, dtype=torch.float32).flatten(1)
    else:
        args.output = torch.empty_like(q).flatten(1)
        args.output_sf = torch.empty(1, device="cuda", dtype=torch.uint8)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    with pytest.raises(ValueError, match="(do not support.*mask/output format|output must match)"):
        _helper_forward(attn, q.flatten(1), metadata, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_alternative_helper_revalidates_output_scale():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8, compute_backend="flashinfer")
    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(8, device="cuda"))
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    _helper_forward(attn, q.flatten(1), metadata, args)
    torch.cuda.synchronize()
    args.out_scale = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="do not support.*mask/output format"):
        _helper_forward(attn, q.flatten(1), metadata, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("extra_width", [0, 17, 512])
@pytest.mark.parametrize("prefix", [0, 1024])
@pytest.mark.parametrize("tile_start", [0, 2])
@torch.inference_mode()
def test_native_context_preserves_real_query_groups(extra_width, prefix, tile_start, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(459)
    heads, count = 64, 5
    attn = _backend(heads)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 1::2] = False
    swa_valid[2] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[1, ::3] = False
        extra_valid[2] = False
    sink = torch.randn(heads, device="cuda")
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count + tile_start)
    source._num_contexts = 2
    source._num_ctx_tokens = source._num_tokens = count + tile_start
    source.csa2_num_context_requests = 2
    source.csa2_request_query_ranges = ((0, 3 + tile_start), (3 + tile_start, count + tile_start))
    source.csa2_request_start_positions = (prefix, prefix * 2)
    source.csa2_request_lengths = (3 + tile_start, 2)
    metadata = source.get_query_tile_metadata(q, extra_width, query_start=tile_start)
    assert metadata.num_contexts == 2
    assert metadata.num_ctx_tokens == count
    assert metadata.num_generations == 0
    torch.testing.assert_close(
        metadata.cu_q_seqlens.cpu(), torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    # Native context's causal mask sees a virtual compacted K domain. Every
    # physical selected row was already filtered by source logical causality.
    topk = metadata.num_sparse_topk
    torch.testing.assert_close(
        metadata.kv_lens_runtime.cpu(), torch.tensor([topk + 2, topk + 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.cu_kv_seqlens.cpu(), torch.tensor([0, topk + 2, 2 * topk + 3], dtype=torch.int32)
    )
    calls = []
    original = FallbackFmha.forward

    def record(provider, query, key, value, meta, forward_args):
        calls.append((meta.num_contexts, meta.num_ctx_tokens, forward_args.attention_input_type))
        assert forward_args.latent_cache is None
        return original(provider, query, key, value, meta, forward_args)

    monkeypatch.setattr(FallbackFmha, "forward", record)
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    args.attention_input_type = AttentionInputType.context_only
    output = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    torch.cuda.synchronize()
    assert calls == [(2, count, AttentionInputType.context_only)]
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )

    monkeypatch.setattr(FallbackFmha, "forward", original)
    generation = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    generation_args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    reference_generation = attn.forward(
        q.flatten(1), None, None, generation, forward_args=generation_args
    ).view_as(q)
    torch.testing.assert_close(output, reference_generation, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("extra_width", [0, 17])
@torch.inference_mode()
def test_packed_helper_skips_bf16_staging(extra_width, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")

    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(460)
    heads, count = 32, 3
    attn = _backend(heads, use_packed=True)
    assert not any(isinstance(provider, FallbackFmha) for provider in attn._fmha_manager.fmha_libs)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 1::2] = False
    swa_valid[1] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[1] = False
    sink = torch.randn(heads, device="cuda")
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    if extra is not None:
        packed = args.sparse_backend_args.main_pool
        records = torch.full((packed.shape[0], 356), 73, dtype=torch.uint8, device="cuda")
        records[:, :288].copy_(packed)
        args.sparse_backend_args.main_pool = records[:, :288]
    metadata = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    monkeypatch.setattr(
        metadata, "stage_selected", lambda *a: pytest.fail("Packed FMHA staged BF16 rows")
    )
    output = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )
    if extra is not None:
        assert bool((records[:, 288:] == 73).all())
    # Reusing the selected library must revalidate unsupported output options.
    args.out_scale = torch.ones(1, device="cuda")
    with pytest.raises(
        ValueError, match="(Packed CSA2 does not support|do not support.*mask/output format)"
    ):
        _helper_forward(attn, q.flatten(1), metadata, args)

    args.out_scale = None
    args.sparse_backend_args.output_position_ids = torch.arange(
        count, device="cuda", dtype=torch.int32
    )
    with pytest.raises(
        ValueError, match="(Packed CSA2 does not support|do not support.*mask/output format)"
    ):
        _helper_forward(attn, q.flatten(1), metadata, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,packed", [("trtllm", False), ("flashinfer", False), ("trtllm", True)]
)
def test_csa2_quant_update_rebuilds_local_provider_policy(implementation, packed):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

    attn = _backend(32, compute_backend=implementation, use_packed=packed)
    native = implementation == "trtllm" and not packed
    first = attn._fmha_manager
    assert bool(first.fmha_libs) == native
    assert all(type(provider) is FallbackFmha for provider in first.fmha_libs)
    attn.update_quant_config(attn.quant_config)
    assert bool(attn._fmha_manager.fmha_libs) == native
    assert all(type(provider) is FallbackFmha for provider in attn._fmha_manager.fmha_libs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_backend_rejects_disabled_fallback(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    with pytest.raises(ValueError, match="requires the FallbackFmha"):
        _backend(32, compute_backend="trtllm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_context_graph_uses_fixed_generation_frame():
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(461)
    heads, count, extra_width = 64, 5, 17
    attn = _backend(heads)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    extra_valid = torch.ones(count, extra_width, device="cuda", dtype=torch.bool)
    sink = torch.randn(heads, device="cuda")
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count)
    source._num_contexts = 2
    source._num_ctx_tokens = source._num_tokens = count
    source.csa2_num_context_requests = 2
    source.csa2_request_query_ranges = ((0, 3), (3, 5))
    source.csa2_request_start_positions = (1024, 2048)
    source.csa2_request_lengths = (3, 2)
    source.is_cuda_graph = True
    metadata = source.get_query_tile_metadata(q, extra_width, query_start=0)
    assert metadata.num_contexts == 0
    assert metadata.num_generations == count
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    args.attention_input_type = (
        AttentionInputType.context_only
        if metadata.num_contexts
        else AttentionInputType.generation_only
    )
    for _ in range(3):
        attn.forward(q.flatten(1), None, None, metadata, forward_args=args)
    pointers = (
        metadata.swa_pool.data_ptr(),
        metadata.extra_pool.data_ptr(),
        metadata.workspace.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = source.get_query_tile_metadata(q, extra_width, query_start=0)
        assert captured is metadata
        output = attn.forward(q.flatten(1), None, None, captured, forward_args=args).view_as(q)
    swa_ids = torch.arange(count * 128, device="cuda").reshape(count, 128)
    main_ids = torch.arange(count * extra_width, device="cuda").reshape(count, extra_width)
    for active in (1, 0, 17):
        q.neg_()
        swa_valid.fill_(True)
        swa_valid[0, 1::2] = False
        swa_valid[1] = False
        extra_valid.copy_(torch.arange(extra_width, device="cuda")[None, :] < active)
        extra_valid[1] = False
        args.sparse_backend_args.swa_indices.copy_(torch.where(swa_valid, swa_ids, -1))
        args.sparse_backend_args.topk_indices.copy_(torch.where(extra_valid, main_ids, -1))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
        )
        assert pointers == (
            metadata.swa_pool.data_ptr(),
            metadata.extra_pool.data_ptr(),
            metadata.workspace.data_ptr(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation,packed", [("flashinfer", False), ("trtllm", True)])
@torch.inference_mode()
def test_direct_compute_rejects_native_forward_misuse(implementation, packed):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(32, compute_backend=implementation, use_packed=packed)
    q = torch.zeros(1, 32, 512, device="cuda", dtype=torch.bfloat16)
    values = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    args = _inputs(q, values, None, valid, None, torch.zeros(32, device="cuda"))
    with pytest.raises(ValueError, match="explicit module helper"):
        attn.forward(q.flatten(1), None, None, metadata, forward_args=args)


def test_segmented_mask_packing():
    mask = torch.tensor(
        [
            [True, False, True, False, False, False, False, True, True],
            [False, True, False, False, False, False, False, False, False],
        ]
    )
    assert CSA2FlashInfer.pack_query_masks(mask).tolist() == [133, 1, 2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads,width,shared", [(8, 17, False), (64, 640, False), (64, 640, True)])
@torch.inference_mode()
def test_flashinfer_bf16_and_graph(heads, width, shared):
    torch.manual_seed(544)
    attn = CSA2FlashInfer()
    q = torch.randn(2, heads, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(2, width, 512, dtype=torch.bfloat16, device="cuda")
    valid = torch.ones(2, width, dtype=torch.bool, device="cuda")
    sink = torch.linspace(-2, 2, heads, device="cuda")

    def reference():
        scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * 512**-0.5
        scores.masked_fill_(~valid[:, None, :], -torch.inf)
        probs = torch.cat((scores, sink[None, :, None].expand(2, -1, -1)), -1).softmax(-1)[..., :-1]
        return torch.einsum("qhk,qkd->qhd", probs, kv.float()).bfloat16()

    indices = torch.arange(2 * width, device="cuda", dtype=torch.int32).reshape(2, width)

    def run():
        if shared:
            return attn.run_shared(q, kv.flatten(0, 1), indices, sink, 512**-0.5)
        return attn(q, kv, valid, sink, 512**-0.5)

    for _ in range(3):
        output = run()
    torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for end in (0, 9, width):
        valid.zero_()
        valid[:, :end] = True
        valid[1, ::2] = False
        kv.mul_(-1)
        if shared:
            physical = torch.arange(2 * width, device="cuda", dtype=torch.int32).reshape(2, width)
            indices.copy_(torch.where(valid, physical, -1))
        graph.replay()
        torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_fixed_plan_ignores_dirty_workspace_padding(monkeypatch):
    """A one-query no-split plan must not launch uninitialized padded CTAs."""
    q = torch.zeros(1, 8, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(1, 128, 512, dtype=torch.bfloat16, device="cuda")
    valid = torch.zeros(1, 128, dtype=torch.bool, device="cuda")
    valid[:, 0] = True
    sink = torch.zeros(8, device="cuda")
    original_empty = torch.empty

    def dirty_workspace(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.is_cuda and tensor.dtype == torch.uint8 and tensor.numel() >= 1024 * 1024:
            tensor.fill_(127)
        return tensor

    monkeypatch.setattr(torch, "empty", dirty_workspace)
    attn = CSA2FlashInfer()
    result = attn(q, kv, valid, sink, 512**-0.5)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.zeros_like(q), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 64])
@torch.inference_mode()
def test_flash_mla_combined_pool_and_sink(heads):
    torch.manual_seed(345)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(3, 17, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(3, 17, device="cuda", dtype=torch.bool)
    valid[0, 9:] = False
    valid[1] = False
    sink = torch.linspace(-1, 2, heads, device="cuda")
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    weights = torch.cat((scores, sink[None, :, None].expand(3, -1, -1)), -1).softmax(-1)
    expected = torch.einsum("qhk,qkd->qhd", weights[..., :-1], kv.float()).bfloat16()
    actual = CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_flash_mla_cuda_graph_changes_selection():
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 128, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, 128, device="cuda", dtype=torch.bool)
    sink = torch.zeros(64, device="cuda")
    for _ in range(3):
        CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
    for end in (128, 3, 64):
        valid.zero_()
        valid[:, :end] = True
        kv.mul_(-1)
        graph.replay()
        expected = CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_global_slot_graph_mapping_folds_visibility(monkeypatch):
    """Graph-owned slot mapping folds per-query visibility and tracks live inputs."""
    from ._utils import _page_metadata

    table = torch.arange(32, dtype=torch.int32, device="cuda").reshape(8, 4)
    request_storage = torch.empty(16, dtype=torch.int64, device="cuda")
    requests = request_storage[::2]
    requests.copy_(torch.arange(8, device="cuda"))
    metadata = _page_metadata(table, requests, 128, 512)
    metadata.is_cuda_graph = False
    storage = torch.zeros((8, 1024), dtype=torch.int64, device="cuda")
    logical = storage[:, ::2]
    logical.copy_(torch.arange(512, device="cuda"))
    visible = torch.full((8,), 512, dtype=torch.int64, device="cuda")
    metadata.global_slot_tile(0, 0, 8, logical, visible_lengths=visible)
    metadata.is_cuda_graph = True
    metadata.global_slot_tile(0, 0, 8, logical)
    for _ in range(3):
        metadata.global_slot_tile(0, 0, 8, logical, visible_lengths=visible)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots = metadata.global_slot_tile(0, 0, 8, logical, visible_lengths=visible)
    for step in range(3):
        table[0, 1] = -1 if step == 0 else 1
        requests.copy_(
            torch.arange(7, -1, -1, device="cuda") if step == 2 else torch.arange(8, device="cuda")
        )
        logical.copy_(torch.arange(512, device="cuda").roll(step * 17))
        logical[:, 7::29] = 1 << 40
        visible.fill_(0 if step == 1 else 511)
        graph.replay()
        # Frozen CPU mapping expression supplies the oracle independently of the helper.
        ids, req, pages, limits = logical.cpu(), requests.cpu(), table.cpu(), visible.cpu()
        columns = ids.clamp_min(0) // 128
        valid = (ids >= 0) & (ids < 512) & (columns < pages.shape[1])
        valid &= ((req >= 0) & (req < pages.shape[0]))[:, None]
        physical = pages[req.clamp(0, 7)[:, None], columns.clamp(max=3)]
        expected = torch.where(valid & (physical >= 0), physical.long() * 128 + ids % 128, -1)
        expected = torch.where(ids < limits[:, None], expected, -1).cuda()
        torch.testing.assert_close(slots, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "phase,extra_width,heads",
    [
        ("generation", 0, 64),
        ("generation", 512, 64),
        ("context", 512, 64),
        ("captured_context", 512, 64),
    ],
)
@torch.inference_mode()
def test_native_fp8_quantized_attention(phase, extra_width, heads, monkeypatch, record_property):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    torch.manual_seed(482)
    count = 5
    attn = _backend(heads, kv_cache_dtype="fp8")
    assert attn.native_fp8 and attn.staging_dtype == torch.float8_e4m3fn
    assert attn.has_fp8_kv_cache
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    q[:, 0, :8] = torch.tensor(
        [0.0, -0.0, 1.0625, -1.0625, 1.1875, -1.1875, 0.001, -0.001],
        device="cuda",
        dtype=torch.bfloat16,
    )
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    sv = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    sv[0, 1::2] = False
    sv[2] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    ev = torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    if ev is not None:
        ev[1, ::3] = False
        ev[2] = False
    sink = torch.randn(heads, device="cuda")
    args = _inputs(q, swa, extra, sv, ev, sink)
    # Staging derives both scales from the packed pools; the oracle below reads
    # them back off the metadata and the forward args.
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count)
    if phase == "generation":
        metadata = CSA2TrtllmMetadata.for_query_tile(
            q, extra_width, staging_dtype=attn.staging_dtype
        )
    else:
        source._num_contexts = 2
        source._num_ctx_tokens = source._num_tokens = count
        source.csa2_num_context_requests = 2
        source.csa2_request_query_ranges = ((0, 3), (3, 5))
        source.csa2_request_start_positions = (1024, 2048)
        source.csa2_request_lengths = (3, 2)
        source.is_cuda_graph = phase == "captured_context"
        metadata = source.get_query_tile_metadata(
            q, extra_width, query_start=0, staging_dtype=attn.staging_dtype
        )
    args.attention_input_type = (
        AttentionInputType.context_only
        if metadata.num_contexts
        else AttentionInputType.generation_only
    )
    assert metadata.num_contexts == (2 if phase == "context" else 0)
    calls = []
    seen = []
    native = FallbackFmha.forward

    def observe(provider, query, k, v, meta, forward_args):
        calls.append(forward_args.attention_input_type)
        assert query.dtype == torch.bfloat16
        assert meta.swa_pool.dtype == meta.extra_pool.dtype == torch.float8_e4m3fn
        if meta.num_contexts:
            assert forward_args.quant_q_buffer is None
            assert forward_args.quant_scale_qkv is forward_args.kv_scale_orig_quant
            # Context runs on a copy of the caller's args, so the scales it was
            # handed are observable only from in here.
            seen.append((forward_args.kv_scale_orig_quant, forward_args.kv_scale_quant_orig))
        if not meta.num_contexts:
            assert forward_args.quant_q_buffer.dtype == torch.float8_e4m3fn
            assert forward_args.latent_cache.dtype == torch.bfloat16
        return native(provider, query, k, v, meta, forward_args)

    monkeypatch.setattr(FallbackFmha, "forward", observe)

    def run():
        if phase == "captured_context":
            captured = source.get_query_tile_metadata(
                q, extra_width, query_start=0, staging_dtype=attn.staging_dtype
            )
            assert captured is metadata
        result = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
        if phase == "context":
            # The native-only context multiplier must not leak into caller args:
            # the next inherited forward rejects an unpaired multiplier.
            assert args.quant_scale_qkv is None
            assert args.quant_q_buffer is None
        return result

    for _ in range(3):
        output = run()
    assert calls and all(p == args.attention_input_type for p in calls)
    graph = None
    if phase == "captured_context":
        pointers = (
            metadata.swa_pool.data_ptr(),
            metadata.extra_pool.data_ptr(),
            metadata.workspace.data_ptr(),
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
    for replay in range(2 if graph else 1):
        if graph:
            q.neg_()
            args.sparse_backend_args.swa_indices[0, : 3 * (replay + 1)] = -1
            sv[0, : 3 * (replay + 1)] = False
            graph.replay()
        torch.cuda.synchronize()
        kv_dq = metadata.stage_kv_scales[1]
        scale_kv = kv_dq.item()
        # A power of two costs no relative precision: it only moves the exponent.
        assert scale_kv > 0.0 and math.log2(scale_kv).is_integer()
        # These pools decode well inside the E4M3 range, so the derivation asks
        # for no extra room and every staged byte stays what the unit scale
        # produced before the range-aware scale existed.
        assert scale_kv == 1.0
        if metadata.num_contexts:
            # The C++ op points dequant_scale_q at dequant_scale_kv, so context
            # cannot give Q a scale of its own: it gets the staging pair as is.
            orig_quant, q_dq = seen[-1]
            assert (orig_quant, q_dq) == metadata.stage_kv_scales
        else:
            # Q takes the reciprocal, which keeps BMM1's dequant product at one.
            # The in-tree split-KV reduction no longer requires that, but the same
            # correction factors live inside the cubins serving the other two
            # MultiCtasKvMode values, whose selection is not auditable.
            q_dq = args.kv_scale_quant_orig
            torch.testing.assert_close(q_dq * kv_dq, torch.ones(1, device="cuda"), atol=0, rtol=0)
        packed_q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, q_dq)
        if not metadata.num_contexts:
            torch.testing.assert_close(
                args.quant_q_buffer.reshape_as(q).view(torch.uint8),
                packed_q.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            scale = q_dq.item() * scale_kv * 512**-0.5
            torch.testing.assert_close(
                metadata.mla_bmm1_scale,
                torch.tensor([scale, scale * 1.4426950408889634], device="cuda"),
                atol=0,
                rtol=1e-6,
            )
            torch.testing.assert_close(metadata.mla_bmm2_scale, kv_dq, atol=0, rtol=0)
        inp = args.sparse_backend_args
        decoded_swa = unpack_rows(inp.swa_pool, 512, "swa").reshape(count, 128, 512)
        qswa, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_swa, kv_dq)
        qextra = None
        if extra_width:
            decoded_extra = unpack_rows(inp.main_pool, 512, "main").reshape(count, extra_width, 512)
            qe, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_extra, kv_dq)
            qextra = qe.float() * scale_kv
        # BMM1 and BMM2 undo the staging scales, so the oracle runs on the
        # dequantized magnitudes that the kernel's own scale factors restore.
        reference = _reference(
            packed_q.float() * q_dq.item(), qswa.float() * scale_kv, qextra, sv, ev, sink
        ).to(torch.bfloat16)
        # The oracle accounts for input quantization. This is the existing
        # native BF16-output accumulation budget, not an extra quantization budget.
        torch.testing.assert_close(output, reference, atol=0.03, rtol=0.03)
        bf16 = _decoded_reference(q, args, sv, ev, sink)
        record_property(
            f"fp8_vs_bf16_max_abs_replay_{replay}",
            (output.float() - bf16.float()).abs().max().item(),
        )
        record_property(
            f"fp8_vs_quantized_reference_max_abs_replay_{replay}",
            (output.float() - reference.float()).abs().max().item(),
        )
        if graph:
            assert pointers == (
                metadata.swa_pool.data_ptr(),
                metadata.extra_pool.data_ptr(),
                metadata.workspace.data_ptr(),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "phase,extra_width",
    [("generation", 128), ("generation", 512), ("context", 512)],
)
@torch.inference_mode()
def test_native_fp8_staging_keeps_rows_beyond_the_e4m3_range(phase, extra_width):
    """Staged rows whose real magnitudes run past 448 must not be clamped to it.

    Both persistent encodings carry group scales in real magnitudes, so a unit
    per-tensor staging scale silently clamps every channel above the E4M3 maximum
    -- the saturation that kept native FP8 staging opt-in. The reference here
    keeps the pools' exactly decoded values, which makes the reported error the
    staging round trip alone. The 512-wide case also runs past the KV length where
    trtllm-gen splits its reduction across CTAs, whose correction factors have to
    be rescaled by the same BMM1 scale the main kernel used.
    """
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (
        _CONTEXT_STAGE_SCALE_MAX_EXPONENT,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    torch.manual_seed(482)
    count, heads = 5, 64
    attn = _backend(heads, kv_cache_dtype="fp8")
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    # Roughly the magnitude an image token reaches: inside what the packed pools
    # represent, far outside what a unit staging scale can hold.
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16) * 600.0
    sv = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    sv[0, 1::2] = False
    extra = torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16) * 600.0
    ev = torch.ones(count, extra_width, device="cuda", dtype=torch.bool)
    ev[1, ::3] = False
    sink = torch.randn(heads, device="cuda")
    args = _inputs(q, swa, extra, sv, ev, sink)
    if phase == "generation":
        metadata = CSA2TrtllmMetadata.for_query_tile(
            q, extra_width, staging_dtype=attn.staging_dtype
        )
    else:
        source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count)
        source._num_contexts = 2
        source._num_ctx_tokens = source._num_tokens = count
        source.csa2_num_context_requests = 2
        source.csa2_request_query_ranges = ((0, 3), (3, 5))
        source.csa2_request_start_positions = (1024, 2048)
        source.csa2_request_lengths = (3, 2)
        metadata = source.get_query_tile_metadata(
            q, extra_width, query_start=0, staging_dtype=attn.staging_dtype
        )
    args.attention_input_type = (
        AttentionInputType.context_only
        if metadata.num_contexts
        else AttentionInputType.generation_only
    )
    output = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    kv_dq = metadata.stage_kv_scales[1]
    scale_kv = kv_dq.item()
    inp = args.sparse_backend_args
    decoded_swa = unpack_rows(inp.swa_pool, 512, "swa").reshape(count, 128, 512).float()
    decoded_extra = unpack_rows(inp.main_pool, 512, "main").reshape(count, extra_width, 512).float()
    decoded_max = max(decoded_swa.abs().amax().item(), decoded_extra.abs().amax().item())
    assert decoded_max > 448.0
    if metadata.num_contexts:
        # Context shares one dequant tensor between Q and KV inside the C++ op, so Q
        # is divided by this scale instead of multiplying by its reciprocal. That
        # direction cannot saturate Q -- it spends Q's resolution below one -- so
        # context buys range too, as far as Q's own amax affords it. These pools ask
        # for more than either bound allows, so the tighter bound itself lands here,
        # well above the unit scale this path used to be pinned to; whatever still
        # runs past 448 * scale needs a decoupled dequant_scale_q in the C++ op.
        # Only the output comparison below stays generation-only, because nothing
        # here models a context reference.
        q_amax = q.abs().amax().item()
        affordable = 2.0 ** math.floor(math.log2(q_amax))
        assert 1.0 < scale_kv == min(affordable, 2.0**_CONTEXT_STAGE_SCALE_MAX_EXPONENT)
        assert q_amax / scale_kv >= 1.0  # Q gave up resolution, not magnitude.
        return
    # The derived scale must cover what the pools decode to, which is the whole
    # point: 448 * scale is the largest magnitude staging can still represent.
    assert scale_kv * 448.0 >= decoded_max
    q_dq = args.kv_scale_quant_orig
    packed_q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, q_dq)
    reference = _reference(
        packed_q.float() * q_dq.item(), decoded_swa, decoded_extra, sv, ev, sink
    ).to(torch.bfloat16)
    # Magnitudes this large make the softmax nearly a hard argmax, so a single
    # flipped winner dominates the maximum error; the mean is the stable measure.
    # A unit scale scores above 0.1 here, an order of magnitude worse.
    error = (output.float() - reference.float()).abs().mean().item()
    assert error / reference.float().abs().amax().item() < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kv_cache_dtype_selects_staging_without_changing_cache_formats():
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    fp8_kv = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    # With FP8 staging opted in, both FP8 spellings and a checkpoint FP8 KV algo
    # under "auto" select E4M3 staging on native trtllm-gen and report an FP8 KV
    # cache upstream. Without the opt-in every request keeps BF16 staging.
    for kv_dtype, quant_config in (("fp8", None), ("fp8_ds_mla", None), ("auto", fp8_kv)):
        attn = _backend(64, kv_cache_dtype=kv_dtype, quant_config=quant_config, fp8_staging=True)
        assert attn.native_fp8 and attn.staging_dtype == torch.float8_e4m3fn
        assert attn.has_fp8_kv_cache and attn.kv_cache_dtype == "fp8"
        attn = _backend(64, kv_cache_dtype=kv_dtype, quant_config=quant_config, fp8_staging=False)
        assert not attn.native_fp8 and attn.staging_dtype == torch.bfloat16
        assert not attn.has_fp8_kv_cache and attn.kv_cache_dtype == "auto"
    # Weight quantization in the same config is preserved for the parent.
    attn = _backend(
        64,
        kv_cache_dtype="auto",
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, kv_cache_quant_algo=QuantAlgo.FP8),
        fp8_staging=True,
    )
    assert attn.native_fp8 and attn.quant_config.quant_algo == QuantAlgo.MXFP8
    # BF16 staging keeps the inherited state free of an FP8 KV cache, and an
    # explicit bfloat16 request wins over a checkpoint FP8 KV algo.
    for kv_dtype, quant_config in (("auto", None), ("bfloat16", fp8_kv)):
        attn = _backend(64, kv_cache_dtype=kv_dtype, quant_config=quant_config, fp8_staging=True)
        assert not attn.native_fp8 and attn.staging_dtype == torch.bfloat16
        assert not attn.has_fp8_kv_cache
    # A later quantization config cannot change the staging chosen at
    # construction; FP8 KV is normalized, anything else is rejected.
    attn.update_quant_config(fp8_kv)
    assert not attn.has_fp8_kv_cache and attn.quant_config.kv_cache_quant_algo is None
    with pytest.raises(ValueError, match="only FP8"):
        attn.update_quant_config(QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4))
    with pytest.raises(ValueError, match="only FP8"):
        _backend(64, quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4))
    with pytest.raises(ValueError, match="dtype must be"):
        _backend(64, kv_cache_dtype="float16")
    # Packed attention reads persistent rows and has no staging: a checkpoint
    # FP8 KV algo just stays BF16, and opting into FP8 staging is contradictory.
    attn = _backend(64, use_packed=True, quant_config=fp8_kv)
    assert not attn.native_fp8 and not attn.has_fp8_kv_cache
    with pytest.raises(ValueError, match="FP8 staging requires"):
        _backend(64, use_packed=True, kv_cache_dtype="fp8", fp8_staging=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_library_backends_keep_bf16_staging_for_fp8_requests():
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    pytest.importorskip("flashinfer")
    attn = _backend(64, compute_backend="flashinfer", kv_cache_dtype="fp8", fp8_staging=False)
    assert not attn.native_fp8 and attn.staging_dtype == torch.bfloat16
    assert not attn.has_fp8_kv_cache and attn.kv_cache_dtype == "auto"
    with pytest.raises(ValueError, match="FP8 staging requires"):
        _backend(64, compute_backend="flashinfer", kv_cache_dtype="fp8", fp8_staging=True)
    attn.update_quant_config(QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8))
    assert not attn.has_fp8_kv_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
def test_default_staging_follows_the_backend_that_can_serve_it(kv_dtype):
    # Left at its default, FP8 staging follows the native path's own support
    # rather than the request: trtllm-gen stages E4M3 for an FP8 KV dtype, and
    # every configuration that cannot consume it keeps BF16 instead of failing.
    attn = _backend(64, kv_cache_dtype=kv_dtype)
    assert attn.native_fp8 and attn.staging_dtype == torch.float8_e4m3fn
    for kwargs in (
        {"use_packed": True},
        {"compute_backend": "flash_mla"},
        {"compute_backend": "flashinfer"},
    ):
        if kwargs.get("compute_backend") == "flashinfer":
            pytest.importorskip("flashinfer")
        attn = _backend(64, kv_cache_dtype=kv_dtype, **kwargs)
        assert not attn.native_fp8 and attn.staging_dtype == torch.bfloat16
        assert not attn.has_fp8_kv_cache and attn.kv_cache_dtype == "auto"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_fp8_rejects_incompatible_options():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    # Packed attention reads persistent rows directly and has no FP8 staging.
    with pytest.raises(ValueError):
        _backend(64, kv_cache_dtype="fp8", use_packed=True, fp8_staging=True)
    # A BF16-staged frame must never be consumed by an FP8 backend.
    attn = _backend(64, kv_cache_dtype="fp8")
    q = torch.zeros(1, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 128, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 128, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(64, device="cuda"))
    with pytest.raises(ValueError):
        attn.forward(
            q.flatten(1), None, None, CSA2TrtllmMetadata.for_query_tile(q, 0), forward_args=args
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,kv_dtype", [("trtllm", "auto"), ("trtllm", "fp8"), ("flashinfer", "auto")]
)
@torch.inference_mode()
def test_shared_native_context_matches_selected_rows(
    implementation, kv_dtype, monkeypatch, record_property
):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import (
        CSA2SharedKVPlan,
        CSA2TrtllmMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        gather_rows,
        pack_rows,
    )

    torch.manual_seed(489)
    count, heads, width = 5, 64, 512
    attn = _backend(heads, kv_cache_dtype=kv_dtype, compute_backend=implementation)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 1024, 1025], device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 1, 1], device="cuda", dtype=torch.int64)
    swa_pages = torch.full((2, 9), -1, device="cuda", dtype=torch.int32)
    swa_pages[0, 0] = 2
    swa_pages[1, 7:] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    main_pages = torch.full((2, 9), -1, device="cuda", dtype=torch.int32)
    main_pages[0, 0] = 9
    main_pages[1] = torch.arange(9, device="cuda", dtype=torch.int32)
    swa = pack_rows(torch.randn(384, 512, device="cuda", dtype=torch.bfloat16), "swa")
    main_storage = torch.empty(1280, 356, device="cuda", dtype=torch.uint8)
    main = main_storage[:, :288]
    main.copy_(pack_rows(torch.randn(1280, 512, device="cuda", dtype=torch.bfloat16), "main"))
    swa_logical = positions[:, None] - 127 + torch.arange(128, device="cuda")
    main_logical = torch.arange(width, device="cuda").expand(count, -1).clone()

    def physical(logical, pages):
        page = pages[requests[:, None], logical.clamp_min(0) // 128].long()
        return torch.where((logical >= 0) & (page >= 0), page * 128 + logical.remainder(128), -1)

    swa_slots = physical(swa_logical, swa_pages)
    main_logical.masked_fill_(main_logical > positions[:, None], -1)
    main_slots = physical(main_logical, main_pages)
    # Sparse holes cross the native128-row pool boundary; one query is empty.
    swa_slots[:, 1::11] = -1
    main_slots[:, 3::13] = -1
    swa_slots[2] = main_slots[2] = -1
    sink = torch.randn(heads, device="cuda")
    plan = CSA2SharedKVPlan(
        query_requests=requests,
        query_positions=positions,
        swa_starts=torch.tensor([0, 897], device="cuda", dtype=torch.int64),
        swa_offsets=torch.tensor([1, 131, 260], device="cuda", dtype=torch.int64),
        main_offsets=torch.tensor([260, 263, 1289], device="cuda", dtype=torch.int64),
        swa_pages=swa_pages,
        main_pages=main_pages,
        swa_page_size=128,
        main_page_size=128,
        num_requests=2,
        swa_rows=259,
        main_rows=1029,
    )
    calls = []
    native_stage = kernel.stage_shared_rows

    def observe(*args, **kwargs):
        calls.append(True)
        return native_stage(*args, **kwargs)

    monkeypatch.setattr(kernel, "stage_shared_rows", observe)
    outputs = []
    for shared in (False, True):
        metadata = CSA2TrtllmMetadata.for_query_tile(
            q,
            width,
            staging_dtype=attn.staging_dtype,
            context_lengths=[3, 2],
            shared_rows=1289 if shared else None,
        )
        if shared:
            metadata.shared_plan = plan
            assert metadata.swa_pool is metadata.extra_pool is metadata.shared_pool
            assert metadata.shared_pool.shape == (1289, 512)
            assert metadata.shared_pool.numel() < count * (128 + width) * 512
        args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.context_only,
            attention_sinks=sink,
            sparse_backend_args=CSA2BackendForwardArgs(
                swa_pool=swa,
                swa_indices=swa_slots,
                main_pool=main,
                topk_indices=main_slots,
                main_logical_indices=main_logical,
            ),
        )
        output = (
            attn.forward(q.flatten(1), None, None, metadata, forward_args=args)
            if implementation == "trtllm"
            else _helper_forward(attn, q.flatten(1), metadata, args)
        )
        outputs.append(output.view_as(q))
    torch.cuda.synchronize()
    assert calls == [True]
    if implementation == "trtllm":
        torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    else:
        # FlashInfer ragged and paged launches can use different reduction orders.
        torch.testing.assert_close(outputs[1], outputs[0], atol=0.03, rtol=0.03)
    decoded_swa = gather_rows(swa, swa_slots, 512, "swa")
    decoded_main = gather_rows(main, main_slots, 512, "main")
    ref_q = q
    if kv_dtype == "fp8":
        scale = torch.ones(1, device="cuda", dtype=torch.float32)
        ref_q = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, scale)[0].float()
        decoded_swa = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_swa, scale)[
            0
        ].float()
        decoded_main = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_main, scale)[
            0
        ].float()
    expected = _reference(
        ref_q, decoded_swa, decoded_main, swa_slots >= 0, main_slots >= 0, sink
    ).to(torch.bfloat16)
    for name, actual in zip(("selected", "shared"), outputs):
        if implementation == "trtllm" and kv_dtype == "fp8":
            # Native FP8 also quantizes P before BMM2; the FP32-P oracle does
            # not model that arithmetic. Exact native storage parity above is
            # the regression gate; retain independent FP32-P error as evidence.
            record_property(
                f"{name}_fp8_vs_fp32_p_max_abs",
                (actual.float() - expected.float()).abs().max().item(),
            )
        else:
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
        torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), atol=0, rtol=0)


@pytest.mark.cpu_only
def test_flashinfer_shared_plans_bound_eager_context_and_keep_protected(monkeypatch):
    """Eager context phases must not accumulate one paged plan per query count."""
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )

    heads, width = 8, 129
    bank = torch.zeros(7, 512, dtype=torch.bfloat16)
    sink = torch.zeros(heads)
    plans = []

    class Paged:
        def __init__(self, workspace, **kwargs):
            pass

        def plan(self, qo, indptr, page_indices, *args, **kwargs):
            plans.append(qo.numel() - 1)
            self._paged_kv_indices_buf = page_indices.clone()
            self._custom_mask_buf = torch.empty(
                (qo.numel() - 1) * ((width + 7) // 8), dtype=torch.uint8
            )

        def run(self, query, pools, *, return_lse):
            return torch.zeros_like(query), torch.zeros(query.shape[:2])

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.prefill",
        SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=Paged),
    )
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    helper = CSA2FlashInfer()
    selection = [None]
    helper.attention = SimpleNamespace(
        prepare_selected=lambda query, metadata, args: (
            query.view(-1, heads, 512),
            bank,
            selection[0],
            sink,
            512**-0.5,
        )
    )

    def phase(n, contexts):
        selection[0] = torch.arange(n * width, dtype=torch.int32).reshape(n, width) % 7
        query = torch.zeros(n, heads * 512, dtype=torch.bfloat16)
        metadata = SimpleNamespace(num_contexts=contexts, shared_plan=True)
        if contexts:
            args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)
            return helper.forward_context(query, metadata, args)
        args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
        return helper.forward_generation(query, metadata, args)

    key = lambda n: (n, heads, width, torch.device("cpu"), 512**-0.5)  # noqa: E731
    phase(3, contexts=1)
    capturing[0] = True
    phase(3, contexts=1)  # captured: must survive later eviction
    capturing[0] = False
    captured = helper._paged_plans[key(3)]
    phase(1, contexts=0)  # generation plan is protected as well
    generation = helper._paged_plans[key(1)]
    for n in (2, 4, 5):
        phase(n, contexts=1)
        assert set(helper._paged_plans) == {key(3), key(1), key(n)}
    assert helper._paged_plans[key(3)] is captured
    assert helper._paged_plans[key(1)] is generation
    assert plans == [3, 1, 2, 4, 5]
