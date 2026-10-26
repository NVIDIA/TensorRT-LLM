# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 module, configuration, checkpoint projections and native compressor contracts."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from pydantic import TypeAdapter
from torch import nn

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import _fused_context_supported
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2Layout,
    CSA2Mode,
    CSA2Params,
    select_csa2_backend,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
    read_index_rows,
    row_bytes,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.weights import load_attention_weights
from tensorrt_llm._torch.attention.backends.sparse.registry import get_sparse_attn_kv_cache_manager
from tensorrt_llm._torch.utils import MxFp8QuantizedTensor
from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig, SparseAttentionConfig

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)
skip_no_fused_context = pytest.mark.skipif(
    not torch.cuda.is_available() or not _fused_context_supported(torch.cuda.current_device()),
    reason="The fused CSA2 context kernel requires an SM100-family GPU",
)


def _rotate(x, positions, cos_sin, inverse=False):
    values = x.float().clone()
    rope_dim = cos_sin.shape[-1] * 2
    cos = cos_sin[positions.long(), 0].unsqueeze(1)
    sin = cos_sin[positions.long(), 1].unsqueeze(1)
    if inverse:
        sin = -sin
    even = values[..., -rope_dim::2].clone()
    odd = values[..., -rope_dim + 1 :: 2].clone()
    values[..., -rope_dim::2] = even * cos - odd * sin
    values[..., -rope_dim + 1 :: 2] = odd * cos + even * sin
    return values.to(x.dtype)


def _norm(x, weight, eps):
    return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


@pytest.mark.cpu_only
# flash_mla and flashinfer share the module-side helper dispatch; one Flash case suffices.
@pytest.mark.parametrize("implementation", ["trtllm", "flashinfer"])
@torch.inference_mode()
def test_module_sends_whole_context_and_generation_phases(monkeypatch, implementation):
    """Exercise module orchestration and real publication/slicing, without FMHA."""
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params

    contexts, generations = 263, 257
    count = contexts + generations
    hidden = (torch.arange(count * 512).reshape(count, 512) % 31 - 15).bfloat16()
    positions = torch.cat((torch.arange(contexts), torch.arange(generations) + 1024)).int()
    row_ids = torch.arange(count, dtype=torch.int64)
    layout = CSA2Layout((1,), (0,), (0,), index_topk=1)
    published, calls, index_calls, entered = [], [], [], []
    phase_outputs = []

    def write_swa(layer, slots, values, transform=None):
        torch.testing.assert_close(slots, row_ids, atol=0, rtol=0)
        torch.testing.assert_close(values, hidden, atol=0, rtol=0)
        published.append(("swa", layer))

    def write_global(layer, slots, main, index, main_transform=None, index_transform=None):
        torch.testing.assert_close(slots, row_ids, atol=0, rtol=0)
        torch.testing.assert_close(main, hidden, atol=0, rtol=0)
        torch.testing.assert_close(index, hidden[:, :128], atol=0, rtol=0)
        published.append(("global", layer))

    def write_layer_rows(layer, slots, values, transform=None, *, global_rows=None, index_q=None):
        write_swa(layer, slots, values, transform)
        if global_rows is not None:
            write_global(*global_rows)
        return None

    manager = SimpleNamespace(
        write_layer_rows=write_layer_rows,
        get_swa_buffer=lambda layer: torch.empty(count, 528, dtype=torch.uint8),
        get_main_buffer=lambda layer: torch.empty(count, 288, dtype=torch.uint8),
    )
    source = SimpleNamespace(
        num_ctx_tokens=contexts,
        csa2_precomputed_kv_layers=set(),
        kv_cache_manager=manager,
        csa2_swa_write_slots={0: row_ids},
        csa2_main_write_slots={0: row_ids},
        csa2_swa_indices={0: row_ids[:, None]},
        csa2_visible_lengths={0: row_ids + 1},
        csa2_indices={},
        get_compression_batch=lambda layer: None,
        get_compressed_positions=lambda layer: positions,
        select_global_source=lambda layer, local, global_hidden: local,
        enter_layer=lambda layer: entered.append(layer.layer_idx),
    )

    def phase_metadata(query, extra_width, query_start, *, staging_dtype, layer_idx=None):
        stop = query_start + query.shape[0]
        assert stop <= (contexts if query_start < contexts else count)
        torch.testing.assert_close(query.flatten(1), hidden[query_start:stop], atol=0, rtol=0)
        return SimpleNamespace(
            num_contexts=int(query_start < contexts),
            num_tokens=query.shape[0],
            num_query_heads=1,
            swa_pool=torch.empty(0, dtype=staging_dtype),
        )

    def map_indices(layer, start, end, logical, *, visible_lengths):
        torch.testing.assert_close(logical[:, 0], row_ids[start:end], atol=0, rtol=0)
        torch.testing.assert_close(visible_lengths, row_ids[start:end] + 1, atol=0, rtol=0)
        return logical

    source.get_query_tile_metadata = phase_metadata
    source.global_slot_tile = map_indices

    def indexer(state, start, end):
        index_calls.append((start, end))
        torch.testing.assert_close(state.index_q.flatten(), positions.float(), atol=0, rtol=0)
        source.csa2_indices[0] = row_ids[:, None]

    backend = SimpleNamespace(
        sparse_params=CSA2Params(layout=layout, compute_backend=implementation),
        staging_dtype=torch.bfloat16,
        native_fp8=False,
        layer_idx=0,
        num_heads=1,
        indexer=indexer,
    )
    backend._native_slot_mapping = lambda logical: CSA2TrtllmAttention._native_slot_mapping(
        backend, logical
    )

    def compute(query, k, v, metadata, *, forward_args):
        start = forward_args.sparse_backend_args.query_start
        calls.append((forward_args.attention_input_type, start, query.shape[0]))
        selected = CSA2TrtllmAttention.prepare_sparse_inputs(backend, query, metadata, forward_args)
        torch.testing.assert_close(
            selected.swa_indices[:, 0], row_ids[start : start + query.shape[0]], atol=0, rtol=0
        )
        # Providers write directly into one contiguous module-owned allocation.
        target = forward_args.output
        assert target.shape == query.shape and target.is_contiguous()
        assert target.device == query.device and target.dtype == query.dtype
        assert target.storage_offset() == start * query.shape[1]
        assert target.untyped_storage().nbytes() == hidden.numel() * hidden.element_size()
        if phase_outputs:
            assert (
                target.untyped_storage().data_ptr() == phase_outputs[0].untyped_storage().data_ptr()
            )
        phase_outputs.append(target)
        target.copy_(query + (selected.topk_indices.remainder(7) + 1).to(query.dtype))
        # Ignore provider return values: output ownership is carried by the args.
        return None

    def unexpected_native(*args, **kwargs):
        pytest.fail("Flash helper fell through to native forward")

    def flash_compute(phase, query, metadata, args):
        assert args.attention_input_type == phase
        return compute(query, None, None, metadata, forward_args=args)

    backend.forward = compute if implementation == "trtllm" else unexpected_native
    model = DeepseekV41Attention.__new__(DeepseekV41Attention)
    nn.Module.__init__(model)
    model.layout, model.layer, model.backend = layout, layout.layer(0), backend
    model.aux_stream = None
    model.num_heads_tp, model.qk_head_dim, model.eps = 1, 512, 1e-6
    model.wq_a = model.wq_b = model.wkv = nn.Identity()
    model.compressor = nn.Identity()
    model.q_norm = model.kv_norm = nn.Identity()
    # Neox RoPE keeps the explicit norm / RoPE calls this test patches out of the cache writes.
    model.rotary_emb = SimpleNamespace(is_neox=True)
    model.attn_sink = torch.zeros(1)
    model._flash_attention = (
        None
        if implementation == "trtllm"
        else SimpleNamespace(
            forward_context=lambda query, metadata, args: flash_compute(
                AttentionInputType.context_only, query, metadata, args
            ),
            forward_generation=lambda query, metadata, args: flash_compute(
                AttentionInputType.generation_only, query, metadata, args
            ),
        )
    )
    model._rope = lambda values, position, heads: values
    model._project_index_q = lambda qr, position: (position.float().reshape(-1, 1, 1), None)
    model._prepare_auxiliary = lambda local, global_hidden, batch, position, skip_global: (
        torch.ones(count, 1),
        global_hidden,
        global_hidden[:, :128],
    )
    model._project_output = lambda output, position, inverse_rope_applied: output.flatten(1)
    actual = model(hidden, positions, source)
    expected = hidden + (row_ids[:, None].remainder(7) + 1).to(hidden.dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.data_ptr() == phase_outputs[0].data_ptr()
    assert calls == [
        (AttentionInputType.context_only, 0, contexts),
        (AttentionInputType.generation_only, contexts, generations),
    ]
    assert published == [("swa", 0), ("global", 0)]
    assert index_calls == [(0, count)] and entered == [0]


@pytest.fixture
def module_cache(request):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    compress_ratio = getattr(request, "param", 2)
    # Ratio 0 makes the single layer SWA-only, without global KV or index sources.
    sources = () if compress_ratio == 0 else (0,)
    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=128 << 20,
            enable_swa_scratch_reuse=True,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        tokens_per_block=128,
        max_seq_len=128,
        max_batch_size=1,
        max_input_len=4,
        max_num_tokens=8,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=128,
        layout=CSA2Layout((compress_ratio,), sources, sources, index_topk=2, window_size=4),
    )
    request = LlmRequest(
        request_id=41,
        max_new_tokens=4,
        input_tokens=[1, 2, 3, 4],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.context_chunk_size)
    manager._stream.synchronize()
    yield manager, request
    manager.free_resources(request)
    manager.shutdown()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
@pytest.mark.parametrize("module_cache", [1, 2], indirect=True)
def test_ced_owner_is_written_once_before_narrowed_query(monkeypatch, module_cache):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm.functional import PositionEmbeddingType

    manager, request = module_cache
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([4], dtype=torch.int32)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.prepare()
    metadata.begin_model_forward()
    torch.manual_seed(4126)
    with torch.device("cuda"):
        model = DeepseekV41Attention(
            manager.layout,
            0,
            PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=RopeParams(dim=64, theta=160000, max_positions=8),
                is_neox=False,
            ),
            hidden_size=32,
            num_heads=8,
            head_dim=512,
            q_lora_rank=32,
            o_lora_rank=16,
            num_groups=2,
            index_heads=2,
        )
        for name, parameter in model.named_parameters():
            parameter.fill_(1) if name.endswith("norm.weight") else parameter.normal_(std=0.1)
        hidden = torch.randn(4, 32, dtype=torch.bfloat16)
        manager.get_swa_buffer(0).zero_()
    writes = []
    writer = manager.write_global

    def record_write(*args):
        writes.append(args[0])
        return writer(*args)

    monkeypatch.setattr(manager, "write_global", record_write)
    model.prepare_global_cache(hidden, metadata)
    assert writes == [0]
    expected_main = manager.get_main_buffer(0).clone()
    expected_index = manager.get_index_pages(0).clone()
    with pytest.raises(ValueError, match="already precomputed"):
        model.prepare_global_cache(hidden, metadata)

    def unexpected_projection(*args, **kwargs):
        raise AssertionError("Narrowed CED query projected the encoder global cache again")

    monkeypatch.setattr(model.compressor, "forward", unexpected_projection)
    monkeypatch.setattr(model.index_wk, "forward", unexpected_projection)
    metadata.seq_lens = torch.tensor([1], dtype=torch.int32)
    metadata.prompt_lens = [1]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[3])
    metadata.prepare()
    metadata._ensure_swa_slots()  # As entering the layer does.
    actual = model(hidden[-1:], metadata.csa2_positions, metadata)
    torch.cuda.synchronize()
    assert actual.shape == (1, 32) and torch.isfinite(actual).all()
    assert writes == [0]
    torch.testing.assert_close(manager.get_main_buffer(0), expected_main, rtol=0, atol=0)
    torch.testing.assert_close(manager.get_index_pages(0), expected_index, rtol=0, atol=0)


@torch.inference_mode()
def _check_ratio2_module_partial_groups(monkeypatch, module_cache, overlap, implementation):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm.functional import PositionEmbeddingType

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream", lambda: overlap
    )
    torch.manual_seed(818)
    manager, request = module_cache
    layout = manager.layout
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    with torch.device("cuda"):
        model = DeepseekV41Attention(
            layout,
            0,
            PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=RopeParams(dim=64, theta=160000, max_positions=8),
                is_neox=False,
            ),
            hidden_size=32,
            num_heads=8,
            head_dim=512,
            q_lora_rank=32,
            o_lora_rank=16,
            num_groups=2,
            index_heads=2,
            compute_backend=implementation,
            aux_stream=torch.cuda.Stream() if overlap else None,
        )
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.1)
        hidden = torch.randn(4, 32, dtype=torch.bfloat16)
        for pool in (
            manager.get_swa_buffer(0),
            manager.get_main_buffer(0),
            manager.get_index_pages(0),
        ):
            pool.zero_()

    preparation_calls = []
    original_prepare = model.backend.prepare_sparse_inputs

    def record_prepare(*args, **kwargs):
        preparation_calls.append(args[0].shape[0])
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(model.backend, "prepare_sparse_inputs", record_prepare)
    if implementation != "auto":
        from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

        def unexpected_native(*args, **kwargs):
            pytest.fail("Module Flash helper fell through to native attention")

        monkeypatch.setattr(FallbackFmha, "forward", unexpected_native)
        assert model._flash_attention is not None
    native_rope = torch.ops.trtllm.mla_rope_inplace
    rope_rows = []

    def record_rope_rows(x, *args):
        # Observe the dispatch contract without replacing any CUDA math.
        rope_rows.append(x.shape[0])
        return native_rope(x, *args)

    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", record_rope_rows)
    compressor = model.compressor
    cos_sin = model.rotary_emb.rotary_cos_sin
    for token in range(4):
        rope_rows.clear()
        completed = (token + 1) // 2
        new_rows = (token + 1) % 2 == 0
        metadata.seq_lens = torch.tensor([1], dtype=torch.int32, device="cpu")
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[token])
        metadata.prepare()
        positions = metadata.csa2_positions
        assert metadata.get_compression_batch(0).cu_compressed_lengths.tolist() == [
            0,
            int(new_rows),
        ]
        assert metadata.get_compression_batch(0).output_rows == 1
        old_main = manager.get_main_buffer(0).clone()
        old_index = manager.get_index_pages(0).clone()
        # Fixed-capacity runtime padding does not generate an empty latent,
        # so exercise the zero-grid guard explicitly while recording native calls.
        empty = hidden.new_empty((0, 1, 512))
        assert model._rope(empty, positions[:0], 1) is empty
        assert not rope_rows
        actual = model(hidden[token : token + 1], positions, metadata)
        torch.cuda.synchronize()
        # Numerics alone can miss an empty native dispatch if a runtime/kernel
        # handles it as a no-op or clears its launch error in a later call.
        assert all(rows > 0 for rows in rope_rows), "Empty rows reached native RoPE"
        assert torch.isfinite(actual).all()
        if not new_rows:
            torch.testing.assert_close(manager.get_main_buffer(0), old_main, atol=0, rtol=0)
            torch.testing.assert_close(manager.get_index_pages(0), old_index, atol=0, rtol=0)

        prefix_positions = torch.arange(token + 1, device="cuda", dtype=torch.int32)
        qr = _norm(
            F.linear(hidden[token : token + 1], model.wq_a.weight), model.q_norm.weight, model.eps
        )
        q = _rotate(F.linear(qr, model.wq_b.weight).reshape(1, 8, 512), positions, cos_sin)
        swa = _norm(
            F.linear(hidden[: token + 1], model.wkv.weight), model.kv_norm.weight, model.eps
        )
        swa = _rotate(swa.unsqueeze(1), prefix_positions, cos_sin).squeeze(1)
        selected = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
        if completed:
            source = hidden[: completed * 2]
            values, gates = (
                part.reshape(completed, 2, 512)
                for part in F.linear(source, compressor.wkv_gate.weight).float().split(512, dim=-1)
            )
            latent = (values * gates.softmax(1)).sum(1).bfloat16()
            latent = _norm(latent, compressor.norm.weight, model.eps)
            group_positions = torch.arange(completed, device="cuda", dtype=torch.int32) * 2
            main = _rotate(latent.unsqueeze(1), group_positions, cos_sin).squeeze(1)
            main = unpack_rows(pack_rows(main, "main"), 512, "main")
            selected = torch.cat((selected, main))
        # All visible global positions fit in top-k, so this reference needs
        # no indexer implementation. Quantization is covered independently.
        scores = torch.einsum("qhd,kd->qhk", q.float(), selected.float()) * 512**-0.5
        probs = torch.cat((scores, model.attn_sink[None, :, None]), -1).softmax(-1)[..., :-1]
        output = torch.einsum("qhk,kd->qhd", probs, selected.float()).bfloat16()
        output = _rotate(output, positions, cos_sin, inverse=True).reshape(1, 2, -1)
        projected = torch.einsum("qgd,grd->qgr", output, model.o_a_proj).flatten(1)
        expected = F.linear(projected, model.o_b_proj.weight)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        assert len(preparation_calls) == token + 1


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("overlap", [False, True])
def test_ratio2_module_partial_groups(monkeypatch, module_cache, overlap):
    _check_ratio2_module_partial_groups(monkeypatch, module_cache, overlap, "auto")


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["flash_mla", "flashinfer"])
def test_module_composed_flash_helper(monkeypatch, module_cache, implementation):
    _check_ratio2_module_partial_groups(monkeypatch, module_cache, False, implementation)


def _native_model(
    *,
    grouped_block=32,
    fused=False,
    aux_stream=None,
    layout=None,
    attention_heads=8,
    projection_quantization="mxfp8",
    sparse_options=None,
    mapping=None,
    num_groups=2,
    q_lora_rank=128,
    allreduce_strategy=None,
    kv_cache_dtype="auto",
    layer_idx=0,
    max_positions=32,
    hidden_size=128,
):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Params
    from tensorrt_llm.functional import PositionEmbeddingType

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Native MXFP8 projections require SM100/SM103/SM107")
    layout = layout or CSA2Layout((1,), (0,), (0,), index_topk=32)
    options = {"fuse_index_q": fused}
    options.update(sparse_options or {})
    with torch.device("cuda"):
        model = DeepseekV41Attention(
            layout,
            layer_idx,
            PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=RopeParams(dim=64, theta=160000, max_positions=max_positions),
                is_neox=False,
            ),
            hidden_size=hidden_size,
            num_heads=attention_heads,
            q_lora_rank=q_lora_rank,
            o_lora_rank=128,
            num_groups=num_groups,
            index_heads=32,
            mapping=mapping,
            sparse_params=CSA2Params(layout=layout, **options),
            projection_quantization=projection_quantization,
            kv_cache_dtype=kv_cache_dtype,
            aux_stream=aux_stream,
            **(
                {"allreduce_strategy": allreduce_strategy} if allreduce_strategy is not None else {}
            ),
        )
    aliases = {
        "o_a_proj": "wo_a.weight",
        "o_b_proj.weight": "wo_b.weight",
        "index_wq_b.weight": "indexer.wq_b.weight",
        "index_weights_proj.weight": "indexer.weights_proj.weight",
        "index_wk.weight": "indexer.wk.weight",
        "index_k_norm.weight": "indexer.k_norm.weight",
    }
    weights = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".weight_scale"):
            continue
        key = aliases.get(name, name)
        shape = (
            (model.num_groups * model.o_lora_rank, parameter.shape[-1])
            if name == "o_a_proj"
            else parameter.shape
        )
        if name in ("wq_b.weight", "attn_sink"):
            shape = (parameter.shape[0] * model.mapping.tp_size, *parameter.shape[1:])
        elif name == "o_b_proj.weight":
            shape = (parameter.shape[0], parameter.shape[1] * model.mapping.tp_size)
        if name in (
            "wq_a.weight",
            "wq_b.weight",
            "wkv.weight",
            "index_wq_b.weight",
            "o_b_proj.weight",
            "o_a_proj",
        ):
            weights[key] = torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)
            block = grouped_block if name == "o_a_proj" else 32
            weights[key.removesuffix("weight") + "weight_scale_inv"] = torch.full(
                ((shape[0] + block - 1) // block, (shape[1] + block - 1) // block),
                123,
                dtype=torch.uint8,
                device="cuda",
            )
        elif name.endswith("norm.weight"):
            weights[key] = torch.ones_like(parameter)
        else:
            weights[key] = torch.randn(shape, device=parameter.device, dtype=parameter.dtype) * 0.1
    model.load_hf_weights(weights)
    return model, weights


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_candidate_consumer_projects_bf16_index_weights():
    """A candidate consumer's BF16 index weights are bitwise the BF16 cast of the FP32 ones."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout

    layout = CSA2Layout(
        (1, 1), (0,), (0, 1), 0, candidate_topk_blocks=4, candidate_block_size=8, index_topk=32
    )
    torch.manual_seed(946)
    hidden = torch.randn(9, 128, device="cuda", dtype=torch.bfloat16)
    hidden *= torch.tensor([1e-30, 1e-3, 1.0, 1e4], device="cuda").repeat_interleave(32)
    for layer_idx, dtype in ((0, torch.float32), (1, torch.bfloat16)):
        model, _ = _native_model(layer_idx=layer_idx, layout=layout)
        assert model.layer.consumes_candidates == (layer_idx == 1)
        weights, _, _ = model._prepare_auxiliary(hidden, hidden, None, None, skip_global=True)
        assert weights.dtype == dtype
        fp32 = torch.mul(
            model.index_weights_proj(hidden),
            model._index_weight_scale,
            out=torch.empty(weights.shape, dtype=torch.float32, device="cuda"),
        )
        bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
        assert torch.equal(weights.view(bits), fp32.to(dtype).view(bits))


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
# Slicing is disabled under attention DP, so a nonzero rank proves full weights load;
# BF16 and MXFP8 take different weight-loading paths.
@pytest.mark.parametrize("rank,quantization", [(0, "bf16"), (7, "mxfp8")])
def test_attention_dp_loads_full_projection_weights(rank, quantization):
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=8, tp_size=8, rank=rank, enable_attention_dp=True)
    model, weights = _native_model(mapping=mapping, projection_quantization=quantization)
    reference, _ = _native_model(projection_quantization=quantization)
    reference.load_hf_weights(weights)
    assert mapping.tp_size == 8 and mapping.pp_size == 1
    assert model.mapping.tp_size == 1 and model.mapping.tp_rank == 0
    assert model.num_heads_tp == reference.num_heads_tp == 8
    assert model.n_local_groups == reference.n_local_groups == 2
    for name, expected in reference.named_parameters():
        actual = dict(model.named_parameters())[name]
        torch.testing.assert_close(actual.float(), expected.float(), atol=0, rtol=0)
    hidden = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(model.wq_b(hidden), reference.wq_b(hidden), atol=0, rtol=0)
    projected = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        model.o_b_proj(projected), reference.o_b_proj(projected), atol=0, rtol=0
    )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("q_lora_rank", [128, 1280])
@torch.inference_mode()
def test_native_mxfp8_projection_checkpoint_and_atomic_load(monkeypatch, q_lora_rank):
    from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight, quant_bf16_to_mxfp8

    torch.manual_seed(943)
    model, weights = _native_model(q_lora_rank=q_lora_rank)
    calls = []
    native = torch.ops.trtllm.mxfp8_mxfp8_gemm

    def gemm(*args, **kwargs):
        calls.append(tuple(args[2].shape))
        return native(*args, **kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "mxfp8_mxfp8_gemm", gemm)
    x = torch.randn(7, 128, dtype=torch.bfloat16, device="cuda")
    actual = model.wq_a(x)
    data, scale = quant_bf16_to_mxfp8(x)
    reference_input = dequant_mxfp8_weight(data, scale)
    reference_weight = weights["wq_a.weight"].float() / 16
    expected = F.linear(reference_input, reference_weight).bfloat16()
    assert calls == [(q_lora_rank, 128)]
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        model.wq_a.weight.view(torch.uint8),
        weights["wq_a.weight"].view(torch.uint8),
        atol=0,
        rtol=0,
    )
    old = {name: value.clone() for name, value in model.named_parameters()}
    invalid = dict(weights)
    invalid["indexer.weights_proj.weight"] = torch.empty(1, device="cuda")
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_hf_weights(invalid)
    for name, parameter in model.named_parameters():
        actual_bytes = (
            parameter.view(torch.uint8) if parameter.dtype == torch.float8_e4m3fn else parameter
        )
        expected_bytes = (
            old[name].view(torch.uint8) if parameter.dtype == torch.float8_e4m3fn else old[name]
        )
        torch.testing.assert_close(actual_bytes, expected_bytes, atol=0, rtol=0)

    from tensorrt_llm._torch.modules.linear import (
        flashinfer_mxfp8_autotune,
        flashinfer_mxfp8_decode_graph_capture,
    )

    for linear in (model.wq_a, model.wq_b, model.wkv):
        linear.quant_method.tune_mxfp8_backends = True
    with flashinfer_mxfp8_decode_graph_capture():
        with flashinfer_mxfp8_autotune():
            model._project_query_and_swa(x, decode_only=True)
            model._fuse_qkv_projection = False
            model._project_query_and_swa(x, decode_only=True)
            model.wkv(x)
            model._fuse_qkv_projection = True
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = model._project_query_and_swa(x, decode_only=True)
        for _ in range(2):
            x.normal_()
            graph.replay()
            model._fuse_qkv_projection = False
            qr, q, _ = model._project_query_and_swa(x, decode_only=True)
            swa = model.wkv(x)
            model._fuse_qkv_projection = True
            torch.testing.assert_close(outputs[0], qr, atol=0, rtol=1 / 128)
            # The small-M Q-B kernel reads BF16 activations, the plain path MXFP8 ones.
            torch.testing.assert_close(outputs[1], q, atol=5e-2 * q.abs().max().item(), rtol=0)
            torch.testing.assert_close(outputs[2], swa, atol=0, rtol=0)
        # A caller's MXFP8 copy of the input (the mHC boundary's) replaces the quantize launch.
        assert model.takes_mxfp8_input
        prequantized = MxFp8QuantizedTensor(
            *torch.ops.trtllm.mxfp8_quantize(x, True), is_sf_swizzled=True
        )
        given = model._project_query_and_swa(x, decode_only=True, prequantized=prequantized)
        for got, want in zip(given, model._project_query_and_swa(x, decode_only=True)):
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    # Reloading must refresh the packed projection as well as the original weights.
    weights["wkv.weight"] = torch.zeros_like(weights["wkv.weight"])
    model.load_hf_weights(weights)
    assert not bool(model._fused_qkv_weight[q_lora_rank:].view(torch.uint8).any())


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_decode_graph_query_projections_match_unfused(rows):
    """Decode-graph Q and index-Q projections (index Q with its RoPE) on the small-M kernel match
    the plain path at the model's shapes (hidden 5120, 8 heads, 32 index heads)."""
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.modules.linear import (
        flashinfer_mxfp8_autotune,
        flashinfer_mxfp8_decode_graph_capture,
    )

    torch.manual_seed(946)
    model, _ = _native_model(q_lora_rank=1280, hidden_size=5120)
    with flashinfer_mxfp8_decode_graph_capture():
        if not model._lowm_projects(model.wq_b, rows):
            pytest.skip("The small-M kernel runs on SM100/SM103")
        assert model._lowm_projects(model.index_wq_b, rows)
    x = torch.randn(rows, 5120, dtype=torch.bfloat16, device="cuda")
    positions = torch.randint(0, 32, (rows,), dtype=torch.int32, device="cuda")

    def plain():
        qr = model.q_norm(model.wq_a(x))
        index_q = model.index_wq_b(qr).reshape(rows, model.index_heads, -1)
        return model.wq_b(qr), model._rope(index_q, positions, model.index_heads)

    for linear in (model.wq_a, model.wq_b, model.wkv, model.index_wq_b):
        linear.quant_method.tune_mxfp8_backends = True
    AutoTuner.get().clear_cache()
    with flashinfer_mxfp8_decode_graph_capture():
        with flashinfer_mxfp8_autotune():
            model._project_query_and_swa(x, decode_only=True)
            plain()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            qr, q, _ = model._project_query_and_swa(x, decode_only=True)
            index_q, _ = model._project_index_q(qr, positions)
    x.normal_()
    graph.replay()
    expected_q, expected_index_q = plain()
    # BF16 versus MXFP8 activations.
    for actual, expected in ((q, expected_q), (index_q, expected_index_q)):
        torch.testing.assert_close(
            actual.reshape(expected.shape).float(),
            expected.float(),
            rtol=0,
            atol=5e-2 * expected.abs().max().item(),
        )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [6, 32])
@torch.inference_mode()
def test_decode_query_norm_writes_only_what_the_q_b_kernel_reads(monkeypatch, rows):
    """A small step's q_norm writes only the BF16 rows the small-M Q-B kernel reads; qr and Q,
    eager and replayed, are bitwise the quantizing norm's rows and that kernel's Q. Larger steps
    keep the quantizing norm and GEMM."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import projection
    from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture
    from tensorrt_llm._torch.modules.low_m_gemm import mxfp8_lowm_linear

    torch.manual_seed(950)
    model, _ = _native_model(q_lora_rank=1280)
    for linear in (model.wq_a, model.wq_b, model.wkv):
        linear.quant_method.tune_mxfp8_backends = True
    with flashinfer_mxfp8_decode_graph_capture():
        small_m = model._lowm_projects(model.wq_b, rows)
    x = torch.randn(rows, 128, dtype=torch.bfloat16, device="cuda")
    norm = projection.rms_norm_mxfp8
    calls = []

    def spy(latent, *args, quantize=True, **kwargs):
        calls.append((latent, quantize))
        return norm(latent, *args, quantize=quantize, **kwargs)

    def reference():
        qr, quantized, scales = norm(
            calls[-1][0], model.q_norm.weight, model.q_norm.variance_epsilon
        )
        if small_m:
            return qr, mxfp8_lowm_linear(
                qr, model.wq_b.weight, model.wq_b.weight_scale, scale_swizzled=True
            )
        return qr, projection.mxfp8_query_gemm(
            quantized,
            scales,
            model.wq_b.weight,
            model.wq_b.weight_scale,
            allowed_backends=model.wq_b.mxfp8_allowed_backends,
        )

    monkeypatch.setattr(projection, "rms_norm_mxfp8", spy)
    with flashinfer_mxfp8_decode_graph_capture():
        qr, q, _ = model._project_query_and_swa(x, decode_only=True)
        assert [quantize for _, quantize in calls] == [not small_m]
        for actual, expected in zip((qr, q), reference()):
            assert torch.equal(actual, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            qr, q, _ = model._project_query_and_swa(x, decode_only=True)
        for _ in range(2):
            x.normal_()
            graph.replay()
            for actual, expected in zip((qr, q), reference()):
                assert torch.equal(actual, expected)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "block,path",
    [(32, "native"), (128, "fp8_bmm"), (128, "small_m"), (128, "mxfp8"), (128, "mxfp8_latent")],
)
@torch.inference_mode()
def test_native_grouped_output_paths_and_replay(monkeypatch, block, path):
    torch.manual_seed(944)
    model, weights = _native_model(grouped_block=block)
    assert (model._o_a_mxfp8_scale is not None) == (block == 128)
    model._small_m_projection = path == "small_m"
    if path == "fp8_bmm":
        model._o_a_mxfp8_scale = None
    # Without a TP reduction O-B takes the latent quantized by the O-A epilogue.
    model.o_b_proj.reduce_output = path != "mxfp8_latent"
    names = ("mxfp8_mxfp8_gemm", "fused_inv_rope_fp8_quant_vllm_port", "cute_dsl_fp8_bmm_blackwell")
    calls = []
    for name in names:
        original = getattr(torch.ops.trtllm, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch.ops.trtllm, name, wrapped)
    values = torch.randn(3, 8, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([1, 5, 11], dtype=torch.int32, device="cuda")
    for _ in range(3):
        model._project_output(values.clone(), positions)
    if path in ("small_m", "mxfp8", "mxfp8_latent"):
        assert "fused_inv_rope_fp8_quant_vllm_port" not in calls
        assert "cute_dsl_fp8_bmm_blackwell" not in calls
    elif block == 128:
        assert "fused_inv_rope_fp8_quant_vllm_port" in calls
        assert "cute_dsl_fp8_bmm_blackwell" in calls
        # The fused context kernel's FP8 projection input replaces the separate quantizer exactly.
        prequantized = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
            values,
            positions,
            model.inverse_rotary_emb.rotary_cos_sin,
            2,
            4,
            448,
            64,
            128,
            model.inverse_rotary_emb.is_neox,
        )
        torch.testing.assert_close(
            model._project_output(values.clone(), positions, prequantized=prequantized),
            model._project_output(values.clone(), positions),
            atol=0,
            rtol=0,
        )
    else:
        assert "fused_inv_rope_fp8_quant_vllm_port" not in calls
        assert len(calls) == 9  # two grouped GEMMs plus O-B each time
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model._project_output(values.clone(), positions)
    for sign in (1.0, -1.0):
        values.mul_(sign)
        positions.add_(1)
        graph.replay()
        eager = model._project_output(values.clone(), positions)
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)
        # A coarse independent BF16 oracle catches wrong groups/RoPE; exact
        # matched-precision native replay is checked above.
        rotated = _rotate(values, positions, model.rotary_emb.rotary_cos_sin, inverse=True)
        o_a = (weights["wo_a.weight"].float() / 16).reshape(2, 128, 2048)
        latent = torch.einsum("qgd,grd->qgr", rotated.float().reshape(3, 2, 2048), o_a).flatten(1)
        reference = F.linear(latent, weights["wo_b.weight"].float() / 16)
        error = (captured.float() - reference).norm() / reference.norm().clamp_min(1e-6)
        assert error < 0.12


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_fused_native_index_q_and_graph(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    torch.manual_seed(945)
    model, _ = _native_model(fused=True)
    qr = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    qr *= torch.tensor([1e-7, 1e-4, 0.1, 1.0], device="cuda").repeat_interleave(32)
    # Diagonal projection removes reduction-rounding ambiguity while unequal
    # 32-channel blocks make a wrong 128-channel activation quantum observable.
    model.index_wq_b.weight.copy_(
        torch.eye(128, device="cuda").repeat(32, 1).to(torch.float8_e4m3fn)
    )
    positions = torch.zeros(7, device="cuda", dtype=torch.int32)
    unfused = model._rope(model.index_wq_b(qr).reshape(7, 32, 128), positions, 32)
    reference = unpack_rows(pack_rows(unfused, "index"), 128, "index")
    monkeypatch.setattr(
        model.index_wq_b,
        "forward",
        lambda *args: pytest.fail("Fused index Q must not call unfused Linear.forward"),
    )
    for _ in range(3):
        data, scales = model._project_index_q(qr, positions)
    assert data.shape == (7, 32, 64) and scales.shape == (7, 32)
    rows = torch.cat(
        (data.view(torch.uint8), scales.contiguous().view(torch.uint8).reshape(7, 32, 4)), dim=-1
    )
    actual = unpack_rows(rows, 128, "index")
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    native = torch.ops.trtllm.cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell
    arguments = (
        qr,
        model.index_wq_b.weight,
        model.index_wq_b.weight_scale,
        positions,
        model.rotary_emb.rotary_cos_sin.view(-1, 64),
        model._fp8_alpha,
    )
    default_data, default_scales = native(*arguments)
    assert "activation_quantization_block_size" not in str(native.default._schema)
    # The nonuniform input makes default V4 128-block quantization observably
    # different from the explicit CSA2 32-block contract.
    assert not torch.equal(default_data.view(torch.uint8).reshape_as(data), data.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        data, scales = model._project_index_q(qr, positions)
    qr.neg_()
    positions.add_(1)
    graph.replay()
    expected_data, expected_scales = model._project_index_q(qr, positions)
    torch.testing.assert_close(
        data.view(torch.uint8), expected_data.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_native_module_auxiliary_stream_and_graph(monkeypatch, module_cache):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import module as csa2_module
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    torch.manual_seed(946)
    manager, request = module_cache
    stream = torch.cuda.Stream()
    model, _ = _native_model(aux_stream=stream, layout=manager.layout)
    enabled = [True]
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream",
        lambda: enabled[0],
    )
    side_streams = []
    side_stream = csa2_module.csa2_side_stream
    monkeypatch.setattr(
        csa2_module,
        "csa2_side_stream",
        lambda device, index=0: side_streams.append(index) or side_stream(device, index),
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([1], dtype=torch.int32, device="cpu")
    hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    # Persist the first half of the compressor group, then repeatedly replace
    # its second half while comparing identical cache/history state.
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.prepare()
    model(hidden, metadata.csa2_positions, metadata)
    # A forward with context tokens keeps its index weights on the auxiliary stream.
    assert not side_streams
    enabled[0] = False
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[1])
    metadata.num_contexts = 0
    metadata.is_cuda_graph = True
    metadata.prepare()
    positions = metadata.csa2_positions
    hidden.normal_()

    def run():
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    serial = run()
    enabled[0] = True
    for _ in range(3):
        parallel = run()
    torch.testing.assert_close(parallel, serial, atol=0, rtol=0)
    # Index weights and paged descriptors each take a side stream in forwards without context.
    assert set(side_streams) == {0, 1}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for sign in (-1.0, 1.0):
        hidden.mul_(sign)
        graph.replay()
        actual = captured.clone()
        enabled[0] = False
        expected = run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        enabled[0] = True
    torch.cuda.synchronize()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("module_cache", [1, 2], indirect=True)
@pytest.mark.parametrize("overlap", [False, True])
@torch.inference_mode()
def test_small_step_query_amax_word_decode_and_graph(monkeypatch, module_cache, overlap):
    """A small step's query RoPE raises Q's amax word; staging waits for it only to finalize."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.modules.linear import (
        flashinfer_mxfp8_autotune,
        flashinfer_mxfp8_decode_graph_capture,
    )

    # FP8 staging path: keep the direct decode, which takes <= 16-row generation tiles, out.
    monkeypatch.setattr(CSA2TrtllmAttention, "direct_decode_serves", lambda *args: False)
    torch.manual_seed(949)
    manager, request = module_cache
    model, _ = _native_model(
        aux_stream=torch.cuda.Stream() if overlap else None,
        layout=manager.layout,
        kv_cache_dtype="fp8",
    )
    assert model.backend.native_fp8
    for linear in (model.wq_a, model.wq_b, model.wkv):
        linear.quant_method.tune_mxfp8_backends = True
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream", lambda: overlap
    )
    words = []
    original = kernel.derive_stage_kv_scales

    def spy(*args, **kwargs):
        words.append(kwargs.get("q_word") is not None)
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "derive_stage_kv_scales", spy)
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([1], dtype=torch.int32, device="cpu")
    hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.prepare()
    model(hidden, metadata.csa2_positions, metadata)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[1])
    metadata.num_contexts = 0
    metadata.is_cuda_graph = True
    metadata.prepare()
    positions = metadata.csa2_positions

    def run(word):
        monkeypatch.setattr(model, "_stages_native_fp8_generation", lambda attn_metadata: word)
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    with flashinfer_mxfp8_decode_graph_capture():
        with flashinfer_mxfp8_autotune():
            run(False)
        for _ in range(2):
            hidden.normal_()
            words.clear()
            expected = run(False)
            assert words == [False]
            words.clear()
            actual = run(True)
            assert words == [True]
            assert model._fp8_q_word.item() == 0
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(True)
        for sign in (-1.0, 1.0):
            hidden.mul_(sign)
            graph.replay()
            actual = captured.clone()
            expected = run(False)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("module_cache", [0, 1, 2], indirect=True)
@pytest.mark.parametrize("overlap", [False, True])
@torch.inference_mode()
def test_fp8_query_projection_decode_and_graph(monkeypatch, module_cache, overlap):
    """Decode Q projected straight to E4M3 matches staging's own BF16 Q quantization.

    An overlapped SWA step reduces the selected rows before waiting for Q-B and waits only
    to finalize the staging scale.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel, query_projection
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.modules.linear import (
        flashinfer_mxfp8_autotune,
        flashinfer_mxfp8_decode_graph_capture,
    )

    # FP8 staging path: keep the direct decode, which takes <= 16-row generation tiles, out.
    monkeypatch.setattr(CSA2TrtllmAttention, "direct_decode_serves", lambda *args: False)
    torch.manual_seed(948)
    manager, request = module_cache
    model, _ = _native_model(
        aux_stream=torch.cuda.Stream() if overlap else None,
        layout=manager.layout,
        kv_cache_dtype="fp8",
    )
    assert model.backend.native_fp8
    model._fp8_q_min_tokens = 1
    # The BF16 comparison Q comes from the same MXFP8 GEMM, not the small-M kernel.
    monkeypatch.setattr(model, "_lowm_projects", lambda linear, rows: False)
    for linear in (model.wq_a, model.wq_b, model.wkv):
        linear.quant_method.tune_mxfp8_backends = True
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream", lambda: overlap
    )
    calls = []
    for owner, name in ((query_projection, "fp8_query_gemm"), (kernel, "quantize_stage_query")):

        def spy(*args, _original=getattr(owner, name), _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, spy)
    late = []
    derive = kernel.derive_stage_kv_scales

    def derive_spy(*args, **kwargs):
        late.append(kwargs.get("q_ready") is not None)
        return derive(*args, **kwargs)

    monkeypatch.setattr(kernel, "derive_stage_kv_scales", derive_spy)
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([1], dtype=torch.int32, device="cpu")
    hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.prepare()
    model(hidden, metadata.csa2_positions, metadata)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[1])
    metadata.num_contexts = 0
    metadata.is_cuda_graph = True
    metadata.prepare()
    positions = metadata.csa2_positions

    def run(fp8_query):
        model._fp8_q_projection = fp8_query
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    with flashinfer_mxfp8_decode_graph_capture():
        with flashinfer_mxfp8_autotune():
            run(False)
        for _ in range(2):
            hidden.normal_()
            calls.clear()
            expected = run(False)
            assert calls == ["quantize_stage_query"]
            calls.clear()
            late.clear()
            actual = run(True)
            assert calls == ["fp8_query_gemm"]
            assert late == [overlap and model.layer.mode == CSA2Mode.SWA]
            assert model._fp8_q_word.item() == 0
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(True)
        for sign in (-1.0, 1.0):
            hidden.mul_(sign)
            graph.replay()
            actual = captured.clone()
            expected = run(False)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("overlap", [False, True])
@torch.inference_mode()
def test_native_module_fixed_context_graph(monkeypatch, module_cache, overlap):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    torch.manual_seed(947)
    manager, request = module_cache
    model, _ = _native_model(
        aux_stream=torch.cuda.Stream() if overlap else None, layout=manager.layout
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream", lambda: overlap
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([4], dtype=torch.int32, device="cpu")
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.prepare()
    positions = metadata.csa2_positions
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    phases = []
    original = model.backend.forward

    def observe(*args, **kwargs):
        phases.append(args[3].num_contexts)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backend, "forward", observe)

    def run():
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    baseline = run()
    assert phases[-1] == 1
    metadata.is_cuda_graph = True
    for _ in range(3):
        warmup = run()
    assert phases[-1] == 0
    torch.testing.assert_close(warmup, baseline, atol=0.03, rtol=0.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for sign in (-1.0, 1.0):
        hidden.mul_(sign)
        graph.replay()
        actual = captured.clone()
        expected = run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [17, 64])
@torch.inference_mode()
def test_large_index_q_uses_unfused_native_path(monkeypatch, tokens):
    torch.manual_seed(948)
    model, _ = _native_model(fused=True)
    qr = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32) % 16
    expected = model._rope(model.index_wq_b(qr).reshape(tokens, 32, 128), positions, 32)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.kernel.csa2_indexer_q_gemm_rope_fp4",
        lambda *args: pytest.fail("Large prefill must not force many small-M fused launches"),
    )
    actual, scales = model._project_index_q(qr, positions)
    assert scales is None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [128, 512, 1280])
@torch.inference_mode()
def test_rms_norm_native_single_rounding(monkeypatch, width):
    from tensorrt_llm._torch import custom_ops
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    calls = []
    original = custom_ops.flashinfer_rmsnorm

    def native(*args):
        calls.append(tuple(args[0].shape))
        return original(*args)

    monkeypatch.setattr(custom_ops, "flashinfer_rmsnorm", native)
    x = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.bfloat16).repeat(3, width // 2)
    weight = torch.linspace(0.3, 2.1, width, device="cuda").bfloat16()
    norm = RMSNorm(hidden_size=width, eps=1e-20, dtype=torch.bfloat16).cuda()
    norm.weight.data.copy_(weight)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
    expected = (normalized * weight.float()).bfloat16()
    extra_rounding = normalized.bfloat16() * weight
    assert bool((extra_rounding != expected).any())
    actual = norm(x)
    assert calls == [(3, width)]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.manual_seed(949)
    x.normal_()
    reference = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20) * weight.float()
    ).bfloat16()
    # Parallel FP32 reduction/rsqrt can move a value across one BF16 boundary.
    torch.testing.assert_close(norm(x), reference, atol=0, rtol=1 / 128)
    if width == 1280 and torch.cuda.get_device_capability() in ((10, 0), (10, 3)):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8

        for values in (x, torch.zeros_like(x), x.repeat(1, 2)[:, :width]):
            normalized, quantized, scales = rms_norm_mxfp8(values, weight, 1e-20)
            torch.testing.assert_close(normalized, norm(values), atol=0, rtol=1 / 128)
            ref_quantized, ref_scales = torch.ops.trtllm.mxfp8_quantize(normalized, True)
            torch.testing.assert_close(
                quantized.view(torch.uint8), ref_quantized.view(torch.uint8), atol=0, rtol=0
            )
            rows = torch.arange(3, device="cuda")[:, None]
            blocks = torch.arange(width // 32, device="cuda")[None, :]
            offsets = blocks // 4 * 512 + rows * 16 + blocks % 4
            torch.testing.assert_close(scales[offsets], ref_scales[offsets], atol=0, rtol=0)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantization,block", [("bf16", 32), ("mxfp8", 32), ("mxfp8", 128)])
@torch.inference_mode()
def test_packed_output_inverse_rope_module_and_graph(
    monkeypatch, module_cache, quantization, block
):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")

    from dataclasses import replace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    torch.manual_seed(950)
    manager, request = module_cache
    model, _ = _native_model(
        layout=manager.layout,
        attention_heads=16,
        projection_quantization=quantization,
        grouped_block=block,
        sparse_options={"use_packed_sparse_attention": True, "fuse_packed_output_rope": True},
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([4], dtype=torch.int32, device="cpu")
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.is_cuda_graph = True
    metadata.prepare()
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    positions = metadata.csa2_positions
    inverse_calls = []
    native_rope = torch.ops.trtllm.mla_rope_inplace

    def observe_rope(x, *args):
        if args[5]:
            inverse_calls.append(x.shape)
        return native_rope(x, *args)

    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", observe_rope)
    projected = []
    original_projection = model._project_output

    def observe_projection(*args, **kwargs):
        projected.append(kwargs.get("inverse_rope_applied", False))
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(model, "_project_output", observe_projection)

    def run():
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    enabled = model.backend.sparse_params
    model.backend.sparse_params = replace(enabled, fuse_packed_output_rope=False)
    baseline = run()
    assert projected[-1] is False
    model.backend.sparse_params = enabled
    inverse_calls.clear()
    for _ in range(3):
        actual = run()
    expected_fusion = not (quantization == "mxfp8" and block == 128)
    assert projected[-1] == expected_fusion
    assert not inverse_calls  # reducer rotates, or grouped128 fused quantizer owns rotation
    torch.testing.assert_close(actual, baseline, atol=0.02, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for sign in (-1.0, 1.0):
        hidden.mul_(sign)
        graph.replay()
        actual = captured.clone()
        expected = run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantization", ["bf16", "mxfp8"])
@torch.inference_mode()
def test_two_rank_module_projection_parity(monkeypatch, quantization):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._utils import mpi_rank, mpi_world_size
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    if mpi_world_size() != 2:
        pytest.skip("Requires exactly two MPI ranks, each bound to its own GPU")
    torch.manual_seed(951)
    layout = CSA2Layout((1,), (0,), (0,), index_topk=32, window_size=4)
    tp_mapping = Mapping(world_size=2, rank=mpi_rank(), tp_size=2)
    model, checkpoint = _native_model(
        layout=layout,
        attention_heads=16,
        num_groups=4,
        mapping=tp_mapping,
        projection_quantization=quantization,
    )
    reference, _ = _native_model(
        layout=layout,
        attention_heads=16,
        num_groups=4,
        mapping=Mapping(),
        projection_quantization=quantization,
    )
    reference.load_hf_weights(checkpoint)
    assert model.wq_b.out_features * 2 == reference.wq_b.out_features
    assert model.o_b_proj.in_features * 2 == reference.o_b_proj.in_features
    assert model.o_b_proj.all_reduce is not None
    reductions = []
    native_reduce = model.o_b_proj.all_reduce.forward

    def observe_reduction(*args, **kwargs):
        reductions.append(args[0].shape)
        return native_reduce(*args, **kwargs)

    monkeypatch.setattr(model.o_b_proj.all_reduce, "forward", observe_reduction)
    generator = torch.Generator(device="cuda").manual_seed(952)
    hidden = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    resources = []

    def fresh_metadata(mapping):
        manager = CSA2CacheManager(
            KvCacheConfig(
                enable_block_reuse=False,
                max_gpu_total_bytes=128 << 20,
                enable_swa_scratch_reuse=True,
                dtype="fp8",
            ),
            CacheType.SELFKONLY,
            num_layers=1,
            tokens_per_block=128,
            max_seq_len=128,
            max_batch_size=1,
            max_input_len=3,
            max_num_tokens=3,
            mapping=mapping,
            dtype=DataType.BF16,
            vocab_size=128,
            layout=layout,
        )
        assert not manager._encoder_replay_enabled
        request = LlmRequest(
            request_id=953,
            max_new_tokens=1,
            input_tokens=[1, 2, 3],
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        resources.append((manager, request))
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=3, kv_cache_manager=manager
        )
        metadata.request_ids = [request.py_request_id]
        metadata.seq_lens = torch.tensor([3], dtype=torch.int32, device="cpu")
        metadata.num_contexts = 1
        metadata.prompt_lens = [3]
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
        metadata.prepare()
        return metadata

    try:
        tp_metadata = fresh_metadata(tp_mapping)
        reference_metadata = fresh_metadata(Mapping())
        # Both ranks execute both calls, including any shared startup/tuning
        # collectives. The reference has replicated heads and no TP reduction.
        actual = model(hidden, tp_metadata.csa2_positions, tp_metadata)
        expected = reference(hidden, reference_metadata.csa2_positions, reference_metadata)
        torch.cuda.synchronize()
        assert reductions == [torch.Size([3, 128])]
        # Each TP shard rounds its O-B accumulation to BF16 before summation;
        # the reference rounds once after a full-width accumulation.
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
    finally:
        for manager, request in resources:
            manager.free_resources(request)
            manager.shutdown()


@pytest.fixture
def replay_module_factory():
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.functional import PositionEmbeddingType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    resources = []

    def create(ratio, input_length=8):
        layout = CSA2Layout((ratio,), (0,), (0,), index_topk=16, window_size=4)
        manager = CSA2CacheManager(
            KvCacheConfig(
                enable_block_reuse=False,
                max_gpu_total_bytes=128 << 20,
                enable_swa_scratch_reuse=True,
                dtype="fp8",
            ),
            CacheType.SELFKONLY,
            num_layers=1,
            tokens_per_block=128,
            max_seq_len=128,
            max_batch_size=1,
            max_input_len=16,
            max_num_tokens=16,
            mapping=Mapping(),
            dtype=DataType.BF16,
            vocab_size=128,
            layout=layout,
        )
        request = LlmRequest(
            request_id=954,
            max_new_tokens=8,
            input_tokens=list(range(input_length)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        resources.append((manager, request))
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=16, kv_cache_manager=manager
        )
        metadata.request_ids = [request.py_request_id]
        metadata.num_contexts = 1
        metadata.prompt_lens = [input_length]
        with torch.device("cuda"):
            model = DeepseekV41Attention(
                layout,
                0,
                PositionalEmbeddingParams(
                    type=PositionEmbeddingType.rope_gptj,
                    rope=RopeParams(dim=64, theta=160000, max_positions=128),
                    is_neox=False,
                ),
                hidden_size=32,
                num_heads=8,
                q_lora_rank=32,
                o_lora_rank=16,
                num_groups=2,
                index_heads=8,
            )
            for name, parameter in model.named_parameters():
                parameter.fill_(1) if name.endswith("norm.weight") else parameter.normal_(std=0.1)
            if ratio == 2:
                # Equal gates isolate replay boundary/state restoration from
                # softmax rounding (nonuniform gates have separate coverage).
                model.compressor.wkv_gate.weight[512:].zero_()
            hidden = torch.randn(input_length + 1, 32, dtype=torch.bfloat16)
        return model, manager, metadata, hidden

    yield create
    for manager, request in resources:
        manager.free_resources(request)
        manager.shutdown()


def _prepare_replay_step(metadata, start, end, cached=None, decoder=False):
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata.seq_lens = torch.tensor([end - start], dtype=torch.int32, device="cpu")
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[start])
    if cached is not None:
        metadata.set_swa_bounded_replay([cached], decoder=decoder)
    metadata.prepare()


def _reference_global_rows(model, source, first_position):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    ratio = model.layer.compress_ratio
    if ratio == 2:
        complete = source.shape[0] // 2
        values, gates = (
            part[: complete * 2].reshape(complete, 2, 512)
            for part in model.compressor.wkv_gate(source).float().split(512, dim=-1)
        )
        projected = (values * gates.softmax(1)).sum(1).bfloat16()
    else:
        projected = model.compressor.wkv(source)
    latent = _norm(projected, model.compressor.norm.weight, model.eps)
    positions = (
        torch.arange(latent.shape[0], device="cuda", dtype=torch.int32) * ratio + first_position
    )
    rotated = _rotate(latent.unsqueeze(1), positions, model.rotary_emb.rotary_cos_sin).squeeze(1)
    return unpack_rows(pack_rows(rotated, "main"), 512, "main")


def _reference_bounded_output(model, hidden, positions, global_rows):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    qrank = _norm(F.linear(hidden, model.wq_a.weight), model.q_norm.weight, model.eps)
    query = _rotate(
        F.linear(qrank, model.wq_b.weight).reshape(-1, 8, 512),
        positions,
        model.rotary_emb.rotary_cos_sin,
    )
    swa = _norm(F.linear(hidden, model.wkv.weight), model.kv_norm.weight, model.eps)
    swa = _rotate(swa.unsqueeze(1), positions, model.rotary_emb.rotary_cos_sin).squeeze(1)
    swa = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
    outputs = []
    for row, position in enumerate(positions.tolist()):
        visible = (position + 1) // model.layer.compress_ratio
        selected = torch.cat((swa[max(0, row - 3) : row + 1], global_rows[:visible]))
        scores = torch.einsum("hd,kd->hk", query[row].float(), selected.float()) * 512**-0.5
        probs = torch.cat((scores, model.attn_sink[:, None]), -1).softmax(-1)[..., :-1]
        outputs.append(torch.einsum("hk,kd->hd", probs, selected.float()).bfloat16())
    output = _rotate(torch.stack(outputs), positions, model.rotary_emb.rotary_cos_sin, inverse=True)
    latent = torch.einsum("qgd,grd->qgr", output.reshape(-1, 2, 2048), model.o_a_proj).flatten(1)
    return F.linear(latent, model.o_b_proj.weight)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_encoder_bounded_replay_preserves_global_and_truncates_swa(
    monkeypatch, replay_module_factory
):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    torch.manual_seed(955)
    model, manager, metadata, hidden = replay_module_factory(2)
    _prepare_replay_step(metadata, 0, 5)
    model(hidden[:5], metadata.csa2_positions, metadata)
    old_slots = metadata.csa2_main_write_slots[0]
    old_slots = old_slots[old_slots >= 0].clone()
    old_main = manager.get_main_buffer(0)[old_slots].clone()
    old_index = read_index_rows(manager.get_index_pages(0), old_slots).clone()
    manager.get_swa_buffer(0).fill_(255)
    for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        manager.get_buffers(0, role).fill_(float("nan"))
    replay = hidden[1:8].clone()
    replay[:4].add_(0.125)  # model-layer replay inputs may be approximate
    seen = []
    compressor = model.compressor.forward

    def observe(x, batch=None):
        seen.append(x.clone())
        return compressor(x, batch)

    monkeypatch.setattr(model.compressor, "forward", observe)
    _prepare_replay_step(metadata, 1, 8, cached=5)
    actual = model(replay, metadata.csa2_positions, metadata)
    assert len(seen) == 1 and seen[0].shape[0] == 4
    torch.testing.assert_close(seen[0], replay[3:], atol=0, rtol=0)
    torch.testing.assert_close(manager.get_main_buffer(0)[old_slots], old_main, atol=0, rtol=0)
    torch.testing.assert_close(
        read_index_rows(manager.get_index_pages(0), old_slots), old_index, atol=0, rtol=0
    )
    assert int((metadata.csa2_swa_indices[0][0] >= 0).sum()) == 1
    expected_global = torch.cat(
        (unpack_rows(old_main, 512, "main"), _reference_global_rows(model, replay[3:], 4))
    )
    expected = _reference_bounded_output(model, replay, metadata.csa2_positions, expected_global)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "ratio,cached,decoder", [(1, 5, False), (2, 4, False), (2, 5, False), (1, 5, True)]
)
@torch.inference_mode()
def test_pure_replay_skips_global_projection_and_restores_odd_state(
    monkeypatch, replay_module_factory, ratio, cached, decoder
):
    from dataclasses import replace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    torch.manual_seed(956)
    model, manager, metadata, hidden = replay_module_factory(ratio)
    _prepare_replay_step(metadata, 0, cached)
    model(hidden[:cached], metadata.csa2_positions, metadata)
    slots = metadata.csa2_main_write_slots[0]
    slots = slots[slots >= 0].clone()
    old_main = manager.get_main_buffer(0)[slots].clone()
    old_index = read_index_rows(manager.get_index_pages(0), slots).clone()
    manager.get_swa_buffer(0).fill_(255)
    if ratio == 2:
        for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
            manager.get_buffers(0, role).fill_(float("nan"))
    start = max(cached - 4, 0)
    replay = hidden[start:cached].clone().add_(0.125)
    counts = []
    projection = model.compressor.wkv_gate if ratio == 2 else model.compressor.wkv
    forward = projection.forward

    def observe(x):
        counts.append(x.shape[0])
        return forward(x)

    monkeypatch.setattr(projection, "forward", observe)
    _prepare_replay_step(metadata, start, cached, cached=cached, decoder=decoder)
    actual = model(replay, metadata.csa2_positions, metadata)
    assert counts == ([1] if ratio == 2 and cached % 2 and not decoder else [])
    torch.testing.assert_close(manager.get_main_buffer(0)[slots], old_main, atol=0, rtol=0)
    torch.testing.assert_close(
        read_index_rows(manager.get_index_pages(0), slots), old_index, atol=0, rtol=0
    )
    assert bool((metadata.csa2_main_write_slots[0] < 0).all())
    expected = _reference_bounded_output(
        model, replay, metadata.csa2_positions, unpack_rows(old_main, 512, "main")
    )
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    if ratio == 2 and cached % 2:
        _prepare_replay_step(metadata, cached, cached + 1)
        batch = metadata.get_compression_batch(0)
        page, offset = divmod(cached - 1, batch.page_size)
        tail = replay[-1:]
        expected_kv, expected_score = (
            part[0]
            for part in F.linear(tail, model.compressor.wkv_gate.weight).float().split(512, -1)
        )
        torch.testing.assert_close(
            batch.kv_state[batch.kv_page_table[0, page], offset], expected_kv, atol=0, rtol=0
        )
        torch.testing.assert_close(
            batch.score_state[batch.score_page_table[0, page], offset],
            expected_score,
            atol=0,
            rtol=0,
        )

        # Seed an independent continuation with only the expected raw tail.
        # Matching the one-token arithmetic avoids FP4 threshold flips from
        # independently quantizing a two-token GEMM reference.
        reference_batch = replace(
            batch,
            kv_state=torch.full_like(batch.kv_state[:1], float("nan")),
            score_state=torch.full_like(batch.score_state[:1], float("nan")),
            kv_page_table=torch.zeros_like(batch.kv_page_table),
            score_page_table=torch.zeros_like(batch.score_page_table),
        )
        reference_batch.kv_state[0, offset].copy_(expected_kv)
        reference_batch.score_state[0, offset].copy_(expected_score)
        next_hidden = hidden[cached : cached + 1]
        reference_latent = model.compressor(next_hidden, reference_batch)
        reference_position = torch.tensor([cached - 1], dtype=torch.int32, device=tail.device)
        expected_row = pack_rows(
            model._rope(reference_latent.unsqueeze(1), reference_position, 1).squeeze(1), "main"
        )
        model(next_hidden, metadata.csa2_positions, metadata)
        new_slots = metadata.csa2_main_write_slots[0]
        new_slots = new_slots[new_slots >= 0]
        torch.testing.assert_close(
            manager.get_main_buffer(0)[new_slots], expected_row, atol=0, rtol=0
        )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_bounded_replay_full_module_graph_changes_hit_position(replay_module_factory):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(957)
    model, manager, seed_metadata, hidden = replay_module_factory(2, input_length=12)

    def new_metadata(graph=False):
        result = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=16, kv_cache_manager=manager)
        result.request_ids = list(seed_metadata.request_ids)
        result.num_contexts = 1
        result.prompt_lens = [12]
        result.is_cuda_graph = graph
        return result

    eager_metadata = new_metadata()
    graph_metadata = new_metadata(graph=True)
    replay_input = torch.empty(7, 32, device="cuda", dtype=torch.bfloat16)

    def seed(cached):
        _prepare_replay_step(seed_metadata, 0, cached)
        model(hidden[:cached], seed_metadata.csa2_positions, seed_metadata)
        slots = seed_metadata.csa2_main_write_slots[0]
        return slots[slots >= 0].clone()

    def drop_local_state():
        manager.get_swa_buffer(0).fill_(255)
        for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
            manager.get_buffers(0, role).fill_(float("nan"))

    def run_graph_body():
        graph_metadata.reset_routing()
        return model(replay_input, graph_metadata.csa2_positions, graph_metadata)

    graph = None
    pointers = None
    for cached in (5, 7):
        start, end = cached - 4, cached + 3
        replay_input.copy_(hidden[start:end])
        replay_input[:4].add_(0.125)
        seed(cached)
        drop_local_state()
        _prepare_replay_step(eager_metadata, start, end, cached=cached)
        expected = model(replay_input, eager_metadata.csa2_positions, eager_metadata).clone()
        # Reset real GLOBAL contents to this hit position's seeded prefix,
        # then remove local state again before executing the captured path.
        slots = seed(cached)
        old_main = manager.get_main_buffer(0)[slots].clone()
        old_index = read_index_rows(manager.get_index_pages(0), slots).clone()
        _prepare_replay_step(graph_metadata, start, end, cached=cached)
        current_pointers = (
            graph_metadata.csa2_positions.data_ptr(),
            graph_metadata.get_compressed_positions(0).data_ptr(),
            graph_metadata.get_compression_batch(0).start_positions.data_ptr(),
        )
        if graph is None:
            for _ in range(3):
                drop_local_state()
                run_graph_body()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run_graph_body()
            pointers = current_pointers
        else:
            assert current_pointers == pointers
        drop_local_state()
        graph.replay()
        torch.cuda.synchronize()
        # The oracle uses the same approximate/truncated replay policy,
        # not a full-prefix reconstruction. Context/gen kernels can differ
        # by ordinary BF16 reduction rounding.
        torch.testing.assert_close(captured, expected, atol=0.03, rtol=0.03)
        torch.testing.assert_close(manager.get_main_buffer(0)[slots], old_main, atol=0, rtol=0)
        torch.testing.assert_close(
            read_index_rows(manager.get_index_pages(0), slots), old_index, atol=0, rtol=0
        )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", [None, "NCCL"], ids=["default-auto", "explicit-nccl"])
def test_tp_output_projection_uses_requested_allreduce_strategy(strategy):
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    from tensorrt_llm._utils import mpi_rank, mpi_world_size
    from tensorrt_llm.mapping import Mapping

    if mpi_world_size() != 2:
        pytest.skip("Requires two MPI ranks")
    torch.cuda.set_device(mpi_rank())
    requested = None if strategy is None else AllReduceStrategy.NCCL
    model, _ = _native_model(
        fused=False,
        projection_quantization="bf16",
        mapping=Mapping(world_size=2, rank=mpi_rank(), tp_size=2),
        allreduce_strategy=requested,
    )
    assert model.o_b_proj.all_reduce is not None
    assert model.o_b_proj.all_reduce.strategy == (
        AllReduceStrategy.AUTO if requested is None else requested
    )
    assert not model.o_b_proj.use_fused_gemm_allreduce


def test_checkpoint_owned_config_and_cache_factory():
    text = dict(
        compress_ratios=[0, 2, 2, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 2, 3],
        candidate_source_layer_id=3,
        candidate_topk_blocks=4,
        candidate_block_size=8,
        index_topk=16,
        sliding_window=128,
    )
    config = TypeAdapter(SparseAttentionConfig).validate_python({"algorithm": "csa2"})
    assert isinstance(config, CSA2SparseAttentionConfig)
    # The config carries no geometry: only the selector plus runtime knobs, and
    # staging stays unset so the backend decides it from its own support.
    expected = {
        "algorithm": "csa2",
        "use_fp8_staging": None,
        "indexer_max_chunk_size": None,
        "q_split_threshold": 8192,
        "skip_indexer_for_short_seqs": True,
        "use_cute_dsl_topk": True,
        "use_cute_dsl_paged_mqa_logits": False,
        "enable_heuristic_topk": False,
        "use_self_sampling_topk": True,
        "use_gvr_emission": False,
    }
    assert {key: config.model_dump()[key] for key in expected} == expected
    assert not config.needs_separate_short_long_cuda_graphs()
    lowered = config.to_sparse_params(pretrained_config={"text_config": text})
    metadata = config.to_sparse_metadata_params(pretrained_config=SimpleNamespace(**text))
    assert lowered.layout == metadata.layout
    assert lowered.use_fp8_staging is None
    tuned = CSA2SparseAttentionConfig(
        indexer_max_chunk_size=4096, q_split_threshold=-1, use_cute_dsl_topk=False
    )
    assert not tuned.to_sparse_params(pretrained_config={"text_config": text}).use_cute_dsl_topk
    tuned_metadata = tuned.to_sparse_metadata_params(pretrained_config=SimpleNamespace(**text))
    assert (tuned_metadata.indexer_max_chunk_size, tuned_metadata.q_split_threshold) == (4096, -1)
    assert metadata.layout.compress_ratios == (0, 2, 2, 1)
    assert get_sparse_attn_kv_cache_manager(config) is CSA2CacheManager
    assert config.supports_backend("pytorch")
    assert not config.supports_backend("autodeploy")


def test_release_layout():
    layout = CSA2Layout(
        (0, 0) + (2,) * 18 + (1,) * 20 + (0,) * 3,
        (2, 8, 14, 20),
        (2, 8, 14, 20, 24, 28, 32, 36),
        20,
    )
    assert Counter(layout.layer(i).mode for i in range(40)) == {
        CSA2Mode.SWA: 2,
        CSA2Mode.FULL: 4,
        CSA2Mode.REINDEX: 4,
        CSA2Mode.REUSE: 30,
    }
    assert (
        sum(
            (row_bytes(512, "main") + row_bytes(128, "index")) // layout.compress_ratios[i]
            for i in layout.kv_source_layer_ids
        )
        == 890
    )
    assert layout.layer(39).kv_source == 20
    assert layout.layer(39).index_source == 36
    assert layout.layer(40).kv_source is None


@pytest.mark.parametrize(
    "ratios,kv,index,candidate",
    [
        ((2,), (), (), None),
        ((2, 1), (0,), (0,), None),
        ((2,), (0,), (), None),
        ((0,), (0,), (0,), None),
        ((2,), (0,), (0,), 0),
        ((1, 1), (0, 1), (0, 1), 0),
    ],
)
def test_invalid_layout(ratios, kv, index, candidate):
    with pytest.raises(ValueError):
        CSA2Layout(ratios, kv, index, candidate)


@pytest.mark.parametrize(
    "sm,backend",
    [
        (90, "flash_mla"),
        (100, "trtllm"),
        (103, "trtllm"),
        (107, "trtllm"),
        (120, "flashinfer"),
        (121, "flashinfer"),
    ],
)
def test_hardware_policy(sm, backend):
    assert select_csa2_backend(sm) == backend


@pytest.mark.parametrize("sm", [80, 89, 110, 130])
def test_unsupported_hardware(sm):
    with pytest.raises(ValueError, match="unsupported"):
        select_csa2_backend(sm)


def _reference(kv_score, ratio, dim):
    overlap = ratio == 4
    state_dim = dim * (2 if overlap else 1)
    values, gates = kv_score.split(state_dim, dim=-1)
    outputs = []
    for end in range(ratio, kv_score.shape[0] + 1, ratio):
        if overlap:
            current_values = values[end - ratio : end, dim:]
            current_gates = gates[end - ratio : end, dim:]
            if end > ratio:
                current_values = torch.cat(
                    (values[end - 2 * ratio : end - ratio, :dim], current_values)
                )
                current_gates = torch.cat(
                    (gates[end - 2 * ratio : end - ratio, :dim], current_gates)
                )
        else:
            current_values = values[end - ratio : end]
            current_gates = gates[end - ratio : end]
        outputs.append((current_values * current_gates.softmax(0)).sum(0).bfloat16())
    return torch.stack(outputs)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [2, 4, 128])
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("decode_tail", [False, True])
def test_native_chunked_compression(ratio, dim, decode_tail):
    torch.manual_seed(732)
    length = 2 * ratio + 3
    state_dim = dim * (2 if ratio == 4 else 1)
    data = torch.randn(length, state_dim * 2, dtype=torch.float32, device="cuda")
    page_size = 32
    pages = (length + page_size - 1) // page_size
    kv = torch.full((pages, page_size, state_dim), float("nan"), device="cuda")
    score = torch.full_like(kv, float("nan"))
    kv_table = torch.randperm(pages, device="cuda").int().unsqueeze(0)
    score_table = torch.randperm(pages, device="cuda").int().unsqueeze(0)
    ape = torch.zeros(ratio, state_dim, device="cuda")
    outputs = []
    start = 0
    for chunk_idx, count in enumerate((ratio - 1, 2, ratio + 2)):
        end = start + count
        num_outputs = end // ratio - start // ratio
        output = torch.empty(num_outputs, dim, dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor([end], dtype=torch.int32, device="cuda")
        cu_seq = torch.tensor([0, count], dtype=torch.int32, device="cuda")
        cu_comp = torch.tensor([0, num_outputs], dtype=torch.int32, device="cuda")
        chunk = data[start:end].contiguous()
        if decode_tail and chunk_idx > 0:
            torch.ops.trtllm.compressor_paged_kv_compress(
                chunk,
                ape,
                kv,
                score,
                kv_table,
                score_table,
                output,
                lengths,
                cu_seq,
                cu_comp,
                1,
                page_size,
                dim,
                ratio,
                count,
            )
        else:
            starts = torch.tensor([start], dtype=torch.int32, device="cuda")
            torch.ops.trtllm.compressor_prefill_reduction(
                chunk,
                ape,
                kv,
                score,
                kv_table,
                score_table,
                output,
                lengths,
                starts,
                cu_seq,
                cu_comp,
                1,
                page_size,
                dim,
                ratio,
                max(1, num_outputs),
            )
        outputs.append(output)
        start = end
    torch.testing.assert_close(
        torch.cat(outputs), _reference(data, ratio, dim), atol=0.016, rtol=0.008
    )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ratio2_native_cuda_graph_changed_partial_state():
    dim, page_size = 512, 32
    kv = torch.zeros(1, page_size, dim, device="cuda")
    score = torch.zeros_like(kv)
    table = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    ape = torch.zeros(2, dim, device="cuda")
    data = torch.zeros(1, dim * 2, device="cuda")
    output = torch.empty(1, dim, dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor([2], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    def run():
        torch.ops.trtllm.compressor_paged_kv_compress(
            data, ape, kv, score, table, table, output, lengths, cu, cu, 1, page_size, dim, 2, 1
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for previous, current in ((2.0, 4.0), (-2.0, 6.0)):
        kv[:, 0].fill_(previous)
        score.zero_()
        data[:, :dim].fill_(current)
        data[:, dim:].zero_()
        graph.replay()
        torch.testing.assert_close(
            output, torch.full_like(output, (previous + current) / 2), atol=0, rtol=0
        )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_module_pre_rope_latent(ratio):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import (
        CSA2CompressionBatch,
        CSA2Compressor,
    )

    torch.manual_seed(927)
    model = CSA2Compressor(16, 128, ratio, 1e-20).cuda()
    (model.wkv_gate if ratio == 2 else model.wkv).weight.normal_(std=0.1)
    x = torch.randn(6, 16, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        if ratio == 2:
            values, gates = model.wkv_gate(x).float().reshape(3, 2, 256).split(128, dim=-1)
            projected = (values * gates.softmax(1)).sum(1).bfloat16()
        else:
            projected = model.wkv(x)
        v = projected.float()
        expected = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-20)).bfloat16()
        if ratio == 1:
            actual = model(x)
        else:
            state = torch.zeros(1, 32, 128, device="cuda")
            score = torch.zeros_like(state)
            table = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
            chunks = []
            start = 0
            for count in (1, 3, 2):
                end = start + count
                outputs = end // 2 - start // 2
                batch = CSA2CompressionBatch(
                    state,
                    score,
                    table,
                    table,
                    torch.tensor([end], dtype=torch.int32, device="cuda"),
                    torch.tensor([start], dtype=torch.int32, device="cuda"),
                    torch.tensor([0, count], dtype=torch.int32, device="cuda"),
                    torch.tensor([0, outputs], dtype=torch.int32, device="cuda"),
                    outputs,
                    32,
                    max(outputs, 1),
                )
                chunks.append(model(x[start:end], batch))
                start = end
            actual = torch.cat(chunks)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
    if ratio == 2:
        assert "zero_ape" not in dict(model.named_parameters())
        assert model.wkv_gate.weight.dtype == torch.bfloat16


def _parameters():
    module = nn.Module()
    module.wq_b = nn.Linear(32, 64, bias=False)
    module.o_a_proj = nn.Parameter(torch.zeros(1, 128, 128))
    module.o_b_proj = nn.Linear(128, 16, bias=False)
    module.attn_sink = nn.Parameter(torch.zeros(4))
    return module


def _weights():
    return {
        "wq_b.weight": torch.arange(128 * 32).reshape(128, 32).float(),
        "wo_a.weight": torch.arange(256 * 128).reshape(256, 128).float(),
        "wo_b.weight": torch.arange(16 * 256).reshape(16, 256).float(),
        "attn_sink": torch.arange(8).float(),
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_weight_sharding_preserves_output_groups(rank):
    module, weights = _parameters(), _weights()
    load_attention_weights(module, weights, "", 2, 128, 2, rank)
    torch.testing.assert_close(
        module.wq_b.weight, weights["wq_b.weight"][rank * 64 : (rank + 1) * 64]
    )
    torch.testing.assert_close(
        module.o_a_proj[0], weights["wo_a.weight"][rank * 128 : (rank + 1) * 128]
    )
    torch.testing.assert_close(
        module.o_b_proj.weight, weights["wo_b.weight"][:, rank * 128 : (rank + 1) * 128]
    )
    torch.testing.assert_close(module.attn_sink, weights["attn_sink"][rank * 4 : (rank + 1) * 4])


@pytest.mark.parametrize("name,dim", [("wq_b.weight", 0), ("attn_sink", 0), ("wo_b.weight", 1)])
def test_oversized_checkpoint_rejected_before_copy(name, dim):
    module, weights = _parameters(), _weights()
    original = {k: p.clone() for k, p in module.named_parameters()}
    weights[name] = torch.cat((weights[name], weights[name]), dim=dim)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_attention_weights(module, weights, "", 2, 128, 2, 0)
    for k, p in module.named_parameters():
        torch.testing.assert_close(p, original[k], atol=0, rtol=0)


@pytest.mark.parametrize("block", [32, 128])
def test_wo_a_fp8_scales(block):
    module, weights = _parameters(), _weights()
    weights["wo_a.weight"] = torch.ones(256, 128).to(torch.float8_e4m3fn)
    scales = (
        torch.arange(256 // block * 128 // block).reshape(256 // block, 128 // block).float() + 1
    )
    weights["wo_a.weight_scale_inv"] = scales
    load_attention_weights(module, weights, "", 2, 128, 2, 1)
    expanded = scales.repeat_interleave(block, 0).repeat_interleave(block, 1)
    torch.testing.assert_close(module.o_a_proj[0], expanded[128:])


def test_invalid_fp8_scale_shape_rejected():
    module, weights = _parameters(), _weights()
    weights["wo_a.weight"] = torch.ones(256, 128).to(torch.float8_e4m3fn)
    weights["wo_a.weight_scale_inv"] = torch.ones(3, 2)
    with pytest.raises(ValueError, match="scale geometry"):
        load_attention_weights(module, weights, "", 2, 128, 2, 0)


@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("storage", ["bytes", "float"])
def test_native_scale_expansion_preserves_checkpoint_values(block, storage):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.weights import expand_mxfp8_scales

    exponents = (
        torch.arange((256 // block) * (256 // block)).reshape(256 // block, 256 // block) % 7 - 3
    )
    scales = (
        (exponents + 127).to(torch.uint8) if storage == "bytes" else torch.exp2(exponents.float())
    )
    expanded = expand_mxfp8_scales(scales, (256, 256), block)
    assert expanded.shape == (256, 8)
    actual = torch.exp2(expanded.float() - 127).repeat_interleave(32, 1)
    expected = torch.exp2(exponents.float()).repeat_interleave(block, 0).repeat_interleave(block, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), 1.5])
def test_native_scale_expansion_rejects_lossy_conversion(value):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.weights import expand_mxfp8_scales

    with pytest.raises(ValueError, match="scales"):
        expand_mxfp8_scales(torch.full((1, 1), value), (32, 32), 32)


@pytest.mark.parametrize("block", [32, 128])
@pytest.mark.parametrize("name", ["wq_b.weight", "wo_b.weight"])
def test_dense_fp8_requantized_scale_blocks_preserve_tp_slices(block, name):
    module, weights = _parameters(), _weights()
    shape = weights[name].shape
    weights[name] = torch.ones(shape).to(torch.float8_e4m3fn)
    scale_shape = tuple((dim + block - 1) // block for dim in shape)
    scales = torch.arange(scale_shape[0] * scale_shape[1]).reshape(scale_shape).float() + 1
    weights[name.removesuffix("weight") + "weight_scale_inv"] = scales
    load_attention_weights(module, weights, "", 2, 128, 2, 1)
    expanded = scales.repeat_interleave(block, 0).repeat_interleave(block, 1)[
        : shape[0], : shape[1]
    ]
    if name == "wq_b.weight":
        torch.testing.assert_close(module.wq_b.weight, expanded[64:])
    else:
        torch.testing.assert_close(module.o_b_proj.weight, expanded[:, 128:])


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_hf_factory_selects_native_fp8_for_both_fp8_cache_spellings():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        pytest.skip("Native FP8 sparse MLA requires SM100-family")
    config = SimpleNamespace(
        model_type="deepseek_v41",
        compress_ratios=(0,),
        kv_source_layer_ids=(),
        index_source_layer_ids=(),
        candidate_source_layer_id=None,
        candidate_topk_blocks=1,
        candidate_block_size=16,
        index_topk=2,
        sliding_window=4,
        hidden_size=32,
        num_attention_heads=8,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=160000,
        max_position_embeddings=8,
    )
    with torch.device("cuda"):
        reference = DeepseekV41Attention.from_hf_config(config, 0, kv_cache_dtype="auto")
    # Either FP8 spelling reaches native E4M3 staging on its own, because this
    # factory builds the trtllm-gen path that serves it. ``auto`` requests no FP8
    # at all, and an explicit ``False`` still pins BF16 against an FP8 KV dtype.
    for requested, staging, fp8 in (
        ("fp8", None, True),
        ("fp8_ds_mla", None, True),
        ("auto", None, False),
        ("fp8", True, True),
        ("fp8", False, False),
    ):
        params = (
            None
            if staging is None
            else CSA2Params(layout=reference.layout, use_fp8_staging=staging)
        )
        with torch.device("cuda"):
            model = DeepseekV41Attention.from_hf_config(
                config, 0, kv_cache_dtype=requested, sparse_params=params
            )
        assert model.backend.compute_backend == "trtllm"
        assert model.backend.native_fp8 is fp8
        assert model.backend.staging_dtype == (torch.float8_e4m3fn if fp8 else torch.bfloat16)
        assert model.backend.has_fp8_kv_cache == fp8


@pytest.fixture
def fused_context_cache(request):
    """Manager with SWA-only, Full and Reuse layers (index_topk 512) and four requests to prefill."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    ratio = request.param
    layout = CSA2Layout((0, ratio, ratio), (1,), (1,), index_topk=512, window_size=128)
    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=16384,
            enable_swa_scratch_reuse=True,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        layout=layout,
        num_layers=3,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=4096,
        max_batch_size=4,
        max_input_len=4096,
        max_num_tokens=8192,
        dtype=DataType.BF16,
        vocab_size=1024,
    )
    requests = []
    for request_id, length in zip((21, 22, 23, 24), (2300, 1, 640, 129)):
        llm_request = LlmRequest(
            request_id=request_id,
            max_new_tokens=4,
            input_tokens=list(range(length)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert manager.prepare_context(llm_request)
        assert manager.resize_context(llm_request, llm_request.context_chunk_size)
        requests.append(llm_request)
    manager._stream.synchronize()
    yield manager, requests
    for llm_request in requests:
        manager.free_resources(llm_request)
    manager.shutdown()


def _fused_context_layers(layout):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params
    from tensorrt_llm.functional import PositionEmbeddingType

    torch.manual_seed(77)
    layers = []
    with torch.device("cuda"):
        for layer_idx in range(len(layout.compress_ratios)):
            model = DeepseekV41Attention(
                layout,
                layer_idx,
                PositionalEmbeddingParams(
                    type=PositionEmbeddingType.rope_gptj,
                    rope=RopeParams(dim=64, theta=160000, max_positions=4096),
                    is_neox=False,
                ),
                hidden_size=128,
                num_heads=64,
                q_lora_rank=128,
                o_lora_rank=128,
                num_groups=8,
                index_heads=32,
                sparse_params=CSA2Params(layout=layout),
                projection_quantization="bf16",
                kv_cache_dtype="fp8",
            )
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.data.fill_(1.0)
                elif parameter.dtype.is_floating_point:
                    parameter.data.copy_(torch.randn_like(parameter, dtype=torch.float32) * 0.05)
            model.attn_sink.data.normal_()
            layers.append(model)
    return layers


def _prefill_attention(layers, manager, requests, hidden):
    """One prefill forward through every layer on fresh metadata; returns each layer's attention output."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata = CSA2TrtllmMetadata(max_num_requests=4, max_num_tokens=8192, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.num_contexts = len(requests)
    metadata.prompt_lens = [request.prompt_len for request in requests]
    metadata.seq_lens = torch.tensor(metadata.prompt_lens, dtype=torch.int32)
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[0] * len(requests)
    )
    metadata.prepare()
    attention = []
    metadata.reset_routing()
    for model in layers:
        project = model._project_output

        def capture(output, positions, **kwargs):
            attention.append(output.clone())
            return project(output, positions, **kwargs)

        model._project_output = capture
        try:
            model(hidden, metadata.csa2_positions, metadata)
        finally:
            del model._project_output
    return attention


@skip_no_fused_context
@pytest.mark.parametrize("fused_context_cache", [1, 2], indirect=True)
@torch.inference_mode()
def test_context_prefill_runs_fused_kernel(monkeypatch, fused_context_cache):
    """A prefill of SWA-only, Full and Reuse layers runs every layer's context tile on the fused
    kernel and matches the trtllm-gen context FMHA."""
    manager, requests = fused_context_cache
    layers = _fused_context_layers(manager.layout)
    hidden = torch.randn(
        sum(request.prompt_len for request in requests), 128, device="cuda", dtype=torch.bfloat16
    )
    op = torch.ops.trtllm.csa2_sparse_context
    calls = []

    def record(*args):
        calls.append(args[1].shape[0])
        return op(*args)

    monkeypatch.setattr(torch.ops.trtllm, "csa2_sparse_context", record)
    fused = _prefill_attention(layers, manager, requests, hidden)
    assert calls == [hidden.shape[0]] * len(layers)
    for model in layers:
        model.backend.fused_context = False
    reference = _prefill_attention(layers, manager, requests, hidden)
    assert len(calls) == len(layers)
    for actual, expected in zip(fused, reference, strict=True):
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error <= 1e-3


def _native_context_layers(layout, kv_cache_dtype="fp8", weights=None):
    """Native MXFP8 layers with 64 heads and grouped 128-block O-A, one per layout layer.

    Returns the layers and their checkpoint weights; ``weights`` loads given ones instead.
    """
    layers, checkpoints = [], []
    for layer_idx in range(len(layout.compress_ratios)):
        model, checkpoint = _native_model(
            layout=layout,
            layer_idx=layer_idx,
            attention_heads=64,
            num_groups=8,
            grouped_block=128,
            kv_cache_dtype=kv_cache_dtype,
            max_positions=4096,
            hidden_size=256,
        )
        if weights is not None:
            checkpoint = weights[layer_idx]
            model.load_hf_weights(checkpoint)
        layers.append(model)
        checkpoints.append(checkpoint)
    return layers, checkpoints


def _prefill_outputs(layers, manager, requests, hidden):
    """One prefill forward through every layer on fresh metadata; returns each layer's output."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata = CSA2TrtllmMetadata(max_num_requests=4, max_num_tokens=8192, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.num_contexts = len(requests)
    metadata.prompt_lens = [request.prompt_len for request in requests]
    metadata.seq_lens = torch.tensor(metadata.prompt_lens, dtype=torch.int32)
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[0] * len(requests)
    )
    metadata.prepare()
    metadata.reset_routing()
    return [model(hidden, metadata.csa2_positions, metadata) for model in layers]


@skip_no_fused_context
@pytest.mark.parametrize("fused_context_cache", [2], indirect=True)
@pytest.mark.parametrize("kv_magnitude", [1.0, 1024.0])
@torch.inference_mode()
def test_context_query_stays_unit_scale_fp8(monkeypatch, fused_context_cache, kv_magnitude):
    """A context-only prefill on the fused kernel projects Q to unit-scale E4M3 only.

    BMM1 then takes the staged KV scale alone and the KV range is no longer capped by
    Q. With unit staging scales this is byte-identical to the BF16 Q plus fixup path;
    with KV beyond the E4M3 range it is closer to BF16 staging than that path.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import query_projection

    manager, requests = fused_context_cache
    torch.manual_seed(79)
    layers, weights = _native_context_layers(manager.layout)
    # BF16 staging of the same weights is the higher-precision reference.
    reference_layers, _ = _native_context_layers(manager.layout, "auto", weights)
    for model, reference in zip(layers, reference_layers, strict=True):
        assert model.backend.fused_context and not reference.backend.native_fp8
        for norm in (model.kv_norm, reference.kv_norm):
            norm.weight.data.fill_(kv_magnitude)
    hidden = torch.randn(
        sum(request.prompt_len for request in requests), 256, device="cuda", dtype=torch.bfloat16
    )
    calls = []
    for name in ("fp8_query_gemm", "query_gemm_rope", "fixup_fp8_query"):

        def spy(*args, _original=getattr(query_projection, name), _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(query_projection, name, spy)
    unit = _prefill_outputs(layers, manager, requests, hidden)
    assert calls == ["fp8_query_gemm"] * len(layers)
    calls.clear()
    for model in layers:
        model._unit_scale_context_query = False
    scaled = _prefill_outputs(layers, manager, requests, hidden)
    assert calls == ["query_gemm_rope", "fixup_fp8_query"] * len(layers)
    reference = _prefill_outputs(reference_layers, manager, requests, hidden)
    for actual, previous, expected in zip(unit, scaled, reference, strict=True):
        assert torch.isfinite(actual).all()
        if kv_magnitude == 1.0:
            torch.testing.assert_close(actual, previous, atol=0, rtol=0)
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        previous_error = (previous.float() - expected.float()).norm() / expected.float().norm()
        assert error <= max(previous_error * 1.01, 0.02), (error, previous_error)


@skip_no_fused_context
@pytest.mark.parametrize("fused_context_cache", [2], indirect=True)
@torch.inference_mode()
def test_context_output_projection_quantizes_o_b_input(monkeypatch, fused_context_cache):
    """The fused context kernel's E4M3 output reaches O-B through one block-128 O-A GEMM whose
    epilogue writes O-B's MXFP8 input, and O-B runs on the persistent CuTe DSL GEMM; the
    outputs equal BF16 O-A, quantization and the native O-B GEMM."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import output_projection

    manager, requests = fused_context_cache
    torch.manual_seed(83)
    layers, _ = _native_context_layers(manager.layout)
    hidden = torch.randn(
        sum(request.prompt_len for request in requests), 256, device="cuda", dtype=torch.bfloat16
    )
    calls = []
    for name in ("grouped_output_block128_mxfp8", "dense_mxfp8_gemm"):

        def spy(*args, _original=getattr(output_projection, name), _name=name, **kwargs):
            calls.append((_name, args[0].shape))
            return _original(*args, **kwargs)

        monkeypatch.setattr(output_projection, name, spy)
    fused = _prefill_outputs(layers, manager, requests, hidden)
    assert calls == [
        ("grouped_output_block128_mxfp8", (8, hidden.shape[0], 8 * 512)),
        ("dense_mxfp8_gemm", (hidden.shape[0], 8 * 128)),
    ] * len(layers)
    calls.clear()
    for model in layers:
        model._fuse_o_a_mxfp8 = False
    separate = _prefill_outputs(layers, manager, requests, hidden)
    assert not calls
    for actual, expected in zip(fused, separate, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@skip_no_fused_context
@pytest.mark.parametrize("fused_context_cache", [1, 2], indirect=True)
@torch.inference_mode()
def test_context_index_q_reads_mxfp8_latent(monkeypatch, fused_context_cache):
    """The fused context Q projection hands the index-Q projection its MXFP8 latent instead of
    the BF16 one; the index queries are unchanged, and so are the outputs up to the order in
    which the top-k selection breaks ties between equal index scores."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import projection

    manager, requests = fused_context_cache
    torch.manual_seed(85)
    layers, _ = _native_context_layers(manager.layout)
    hidden = torch.randn(
        sum(request.prompt_len for request in requests), 256, device="cuda", dtype=torch.bfloat16
    )
    stored = []
    original = projection.rms_norm_mxfp8

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        stored.append(result[0] is not None)
        return result

    monkeypatch.setattr(projection, "rms_norm_mxfp8", spy)
    index_queries = []
    for model in layers:
        project = model._project_index_q

        def capture(qr, positions, _project=project):
            result = _project(qr, positions)
            index_queries.append(None if result[0] is None else result[0].clone())
            return result

        model._project_index_q = capture
    latent = _prefill_outputs(layers, manager, requests, hidden)
    assert stored == [False] * len(layers)
    stored.clear()
    for model in layers:
        model._mxfp8_query_latent = False
    normalized = _prefill_outputs(layers, manager, requests, hidden)
    assert stored == [True] * len(layers)
    half = len(index_queries) // 2
    assert any(query is not None for query in index_queries[:half])
    for actual, expected in zip(index_queries[:half], index_queries[half:], strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # A tie broken the other way changes one query row's output (about 1e-3 relative).
    for actual, expected in zip(latent, normalized, strict=True):
        assert torch.isfinite(actual).all()
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error <= 1e-2, error


def _inverse_rope_reference(output, positions, cos_sin):
    """FP64 GPT-J inverse RoPE of each 512-channel head's last 64 channels."""
    rows = output.shape[0]
    x = output.double().view(rows, -1, 512).clone()
    tail = x[..., 448:].reshape(rows, -1, 32, 2)
    trig = cos_sin.double().view(-1, 2, 32)[positions.long()]
    cosine, sine = trig[:, None, 0, :], trig[:, None, 1, :]
    even, odd = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0], tail[..., 1] = even * cosine + odd * sine, odd * cosine - even * sine
    x[..., 448:] = tail.view(rows, -1, 64)
    return x


def _inverse_rope_o_a_reference(output, positions, cos_sin, weight, scale):
    """FP64 GPT-J inverse RoPE of each head's last 64 channels, then the block-128 O-A."""
    rows = output.shape[0]
    groups, rank, depth = weight.shape
    x = _inverse_rope_reference(output, positions, cos_sin)
    dequant = weight.double() * scale.double().repeat_interleave(128, 1).repeat_interleave(128, 2)
    return torch.einsum("mgd,grd->mgr", x.view(rows, groups, depth), dequant).reshape(rows, -1)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,groups,heads,split_k", [(1, 1, 8, None), (6, 1, 8, None), (6, 1, 8, 2), (16, 2, 4, None)]
)
@torch.inference_mode()
def test_small_m_inverse_rope_o_a_matches_reference(rows, groups, heads, split_k):
    """Fused inverse RoPE + block-128 split-K O-A projection against an FP64 reference."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import (
        inverse_rope_block128_small_m,
    )

    gen = torch.Generator(device="cuda").manual_seed(rows * 7 + groups)
    rank, depth = 1024, heads * 512
    weight = (torch.randn(groups, rank, depth, device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(groups, rank // 128, depth // 128, device="cuda", generator=gen) * 0.02
    cos_sin = (torch.rand(512, 2, 32, device="cuda", generator=gen) * 2 - 1).contiguous()
    output = torch.randn(rows, groups * depth, device="cuda", dtype=torch.bfloat16, generator=gen)
    positions = torch.randint(0, 512, (rows,), device="cuda", dtype=torch.int32, generator=gen)
    reference = _inverse_rope_o_a_reference(output, positions, cos_sin, weight, scale)
    kwargs = {} if split_k is None else {"split_k": split_k}
    actual = inverse_rope_block128_small_m(
        output, positions, cos_sin, weight, scale, name="test", **kwargs
    )
    # One BF16 rounding of the rotated input plus the BF16 output rounding.
    bound = reference.abs() * 2**-7 + 2e-3 * reference.abs().max()
    assert ((actual.double() - reference).abs() <= bound).all()
    again = inverse_rope_block128_small_m(
        output, positions, cos_sin, weight, scale, name="test", **kwargs
    )
    torch.testing.assert_close(again, actual, atol=0, rtol=0)
    # The quantized output is mxfp8_quantize of that BF16 output, scale rows past M zero.
    quantized, scales = inverse_rope_block128_small_m(
        output, positions, cos_sin, weight, scale, name="test_q", quantize_output=True, **kwargs
    )
    expected, expected_scales = torch.ops.trtllm.mxfp8_quantize(actual, True)
    assert torch.equal(quantized.view(torch.uint8), expected.view(torch.uint8))
    blocks = groups * rank // 32
    linear = torch.ops.trtllm.block_scale_interleave_reverse(scales.view(128, blocks))
    reference_scales = torch.ops.trtllm.block_scale_interleave_reverse(
        expected_scales.view(-1, blocks)
    )
    assert torch.equal(linear[:rows], reference_scales[:rows])
    assert not linear[rows:].any()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,heads,depth", [(1, 8, 512), (6, 64, 512), (16, 3, 512), (5, 32, 128)]
)
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_small_m_rope_matches_cuda_rope(rows, heads, depth, positions_dtype):
    """The small-step query RoPE is bit-identical to mla_rope_inplace, graph capture included."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import gptj_rope_small_m

    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + heads)
    q = (torch.randn(rows, heads, depth, device="cuda", generator=gen) * 4).to(torch.bfloat16)
    positions = torch.randint(0, 512, (rows,), device="cuda", generator=gen).to(positions_dtype)
    angle = torch.rand(512, 32, device="cuda", generator=gen) * 6.3
    cos_sin = torch.stack((angle.cos(), angle.sin()), dim=1).contiguous()
    expected = q.clone()
    torch.ops.trtllm.mla_rope_inplace(
        expected, positions.int(), cos_sin, heads, depth - 64, 64, False, False
    )
    actual = q.clone()
    gptj_rope_small_m(actual, positions, cos_sin)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    graph = torch.cuda.CUDAGraph()
    static = q.clone()
    with torch.cuda.graph(graph):
        gptj_rope_small_m(static, positions, cos_sin)
    static.copy_(q)
    graph.replay()
    assert torch.equal(static.view(torch.int16), expected.view(torch.int16))
    # The word rises to the rotated Q's amax bits and keeps a larger earlier value.
    halves = expected.view(torch.int16).to(torch.int32) & 0x7FFF
    bits = int(halves.max()) << 16
    for start, result in ((0, bits), (bits + (1 << 16), bits + (1 << 16))):
        word = torch.full((1,), start, device="cuda", dtype=torch.int32)
        raised = q.clone()
        gptj_rope_small_m(raised, positions, cos_sin, word)
        assert torch.equal(raised.view(torch.int16), expected.view(torch.int16))
        assert word.item() == result


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,width", [(1, 5120), (6, 5120), (16, 1280), (130, 256)])
@pytest.mark.parametrize("pdl", [False, True])
@torch.inference_mode()
def test_mxfp8_quantize_rows_and_prefetch(monkeypatch, rows, width, pdl):
    """Row quantization is mxfp8_quantize byte for byte; L2 prefetches change no output."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import projection

    monkeypatch.setattr(projection, "_pdl", lambda: pdl)
    gen = torch.Generator(device="cuda").manual_seed(rows * 13 + width)
    x = torch.randn(rows, width, device="cuda", generator=gen) * 4
    x[0, :64] = 0.0
    x[-1, 32:64] *= 2.0**40
    x = x.to(torch.bfloat16)
    weight = (torch.randn(3000, 701, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.randint(0, 255, (5, 99), device="cuda", dtype=torch.uint8, generator=gen)
    expected, expected_scales = torch.ops.trtllm.mxfp8_quantize(x, True)
    for prefetch in ((), (weight,), (weight, scale)):
        quantized, scales = projection.mxfp8_quantize_rows(x, prefetch=prefetch)
        assert torch.equal(quantized.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(scales.view(torch.uint8), expected_scales.view(torch.uint8))
    gamma = (torch.rand(width, device="cuda", generator=gen) + 0.5).to(torch.bfloat16)
    reference = projection.rms_norm_mxfp8(x, gamma, 1e-6)
    prefetched = projection.rms_norm_mxfp8(x, gamma, 1e-6, prefetch=(scale, weight))
    for a, b in zip(reference, prefetched):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    # Without its BF16 output the kernel quantizes the same values.
    unstored = projection.rms_norm_mxfp8(x, gamma, 1e-6, normalized=False)
    assert unstored[0] is None
    for a, b in zip(reference[1:], unstored[1:]):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    with pytest.raises(ValueError):
        projection.mxfp8_quantize_rows(x, prefetch=(weight.t(),))
    graph = torch.cuda.CUDAGraph()
    static = x.clone()
    with torch.cuda.graph(graph):
        captured = projection.mxfp8_quantize_rows(static, prefetch=(weight, scale))
    static.copy_(-x)
    graph.replay()
    expected, expected_scales = torch.ops.trtllm.mxfp8_quantize(-x, True)
    assert torch.equal(captured[0].view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(captured[1].view(torch.uint8), expected_scales.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_l2_touch_reads_without_writing():
    """The L2 touch only reads its ranges, under graph capture too, and rejects bad ranges."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import l2_touch

    gen = torch.Generator(device="cuda").manual_seed(5)
    weight = (torch.randn(1024, 4100, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    scale = torch.randint(0, 255, (163840,), device="cuda", dtype=torch.uint8, generator=gen)
    copies = [weight.clone(), scale.clone()]
    sink = torch.zeros(64, dtype=torch.int32, device="cuda")
    l2_touch((weight, scale), sink)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        l2_touch((weight, scale, weight[:3]), sink)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(weight.view(torch.uint8), copies[0].view(torch.uint8))
    assert torch.equal(scale, copies[1])
    with pytest.raises(ValueError):
        l2_touch((weight.t(),), sink)
    with pytest.raises(ValueError):
        l2_touch((scale[:3],), sink)
    with pytest.raises(ValueError):
        l2_touch((weight, weight, weight, weight), sink)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 6, 17])
@torch.inference_mode()
def test_native_small_m_output_projection(rows):
    """The module's small-step O-A matches FP64 and the FP8-activation block-128 path."""
    torch.manual_seed(1017 + rows)
    model, _ = _native_model(grouped_block=128)
    assert model._o_a_fp8_block128
    latents = []
    model.o_b_proj.register_forward_pre_hook(lambda _, args: latents.append(args[0].clone()))
    output = torch.randn(rows, model.num_heads_tp, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(0, 32, (rows,), device="cuda", dtype=torch.int32)

    def project(small_m):
        model._small_m_projection = small_m
        return model._project_output(output.clone(), positions)

    small, large = project(True), project(False)
    assert small.shape == large.shape
    reference = _inverse_rope_o_a_reference(
        output.view(rows, -1),
        positions,
        model.inverse_rotary_emb.rotary_cos_sin,
        model.o_a_proj,
        model.o_a_proj_scale,
    )
    scale = reference.abs().max()
    large_latent = latents[-1]
    if rows <= 16:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import (
            inverse_rope_block128_small_m,
        )

        # The small step quantizes its latent for O-B inside the O-A reduction.
        assert len(latents) == 1
        latent = inverse_rope_block128_small_m(
            output.view(rows, -1),
            positions,
            model.inverse_rotary_emb.rotary_cos_sin,
            model.o_a_proj,
            model.o_a_proj_scale,
            name="test_latent",
        )
        assert ((latent.double() - reference).abs() <= reference.abs() * 2**-7 + 2e-3 * scale).all()
        torch.testing.assert_close(small, model.o_b_proj(latent), atol=0, rtol=0)
    else:
        torch.testing.assert_close(latents[0], latents[1], atol=0, rtol=0)
    # Without small steps the rotated activation is quantized to MXFP8 first.
    assert (large_latent.double() - reference).abs().max() <= 0.08 * scale


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,groups,heads", [(1, 1, 8), (6, 1, 8), (12, 1, 8), (16, 2, 8)])
@torch.inference_mode()
def test_small_m_warp_mma_o_a_matches_reference(rows, groups, heads):
    """The small-M kernel's grouped O-A (inverse RoPE on load, FP32 power-of-two block
    multipliers) against an FP64 reference, eager and replayed."""
    from tensorrt_llm._torch.modules.low_m_gemm import fp8_grouped_lowm, lowm_fp8_supported

    if not lowm_fp8_supported():
        pytest.skip("The small-M kernel runs on SM100/SM103")
    gen = torch.Generator(device="cuda").manual_seed(rows * 11 + groups)
    rank, depth = 1024, heads * 512
    weight = (torch.randn(groups, rank, depth, device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    exponent = torch.randint(
        -9, -4, (groups, rank // 128, depth // 128), generator=gen, device="cuda"
    )
    scale = torch.exp2(exponent.float())
    cos_sin = (torch.rand(512, 2, 32, device="cuda", generator=gen) * 2 - 1).contiguous()
    output = torch.randn(rows, groups * depth, device="cuda", dtype=torch.bfloat16, generator=gen)
    positions = torch.randint(0, 512, (rows,), device="cuda", dtype=torch.int32, generator=gen)

    def project():
        return fp8_grouped_lowm(
            output,
            weight.view(groups * rank, depth),
            scale.view(groups * rank // 128, depth // 128),
            rank,
            inverse_rope=(positions, cos_sin, 512),
        )

    def check(actual):
        reference = _inverse_rope_o_a_reference(output, positions, cos_sin, weight, scale)
        # One BF16 rounding of the rotated input plus the BF16 output rounding.
        bound = reference.abs() * 2**-7 + 2e-3 * reference.abs().max()
        assert ((actual.double() - reference).abs() <= bound).all()

    check(project())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = project()
    output.normal_(generator=gen)
    graph.replay()
    check(captured)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_decode_graph_output_projection_matches_plain(rows):
    """Decode-graph O-A -> O-B (the small-M kernel's O-A, its BF16 latent into the small-M O-B)
    matches an FP32 O-A -> O-B, eager and replayed."""
    from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture

    torch.manual_seed(1019 + rows)
    model, weights = _native_model(
        grouped_block=128, hidden_size=5120, num_groups=1, attention_heads=8, q_lora_rank=1280
    )
    output = torch.randn(rows, model.num_heads_tp, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(0, 32, (rows,), device="cuda", dtype=torch.int32)
    with flashinfer_mxfp8_decode_graph_capture():
        if not (model._lowm_o_a(rows) and model._lowm_projects(model.o_b_proj, rows)):
            pytest.skip("The small-M kernel runs on SM100/SM103")

    def plain():
        reference = _inverse_rope_o_a_reference(
            output.view(rows, -1),
            positions,
            model.inverse_rotary_emb.rotary_cos_sin,
            model.o_a_proj,
            model.o_a_proj_scale,
        )
        return F.linear(reference.float(), weights["wo_b.weight"].float() / 16)

    with flashinfer_mxfp8_decode_graph_capture():
        eager = model._project_output(output.clone(), positions)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = model._project_output(output, positions)
    expected = plain()
    torch.testing.assert_close(
        eager.float(), expected, rtol=0, atol=2e-2 * expected.abs().max().item()
    )
    output.normal_()
    graph.replay()
    expected = plain()
    torch.testing.assert_close(
        captured.float(), expected, rtol=0, atol=2e-2 * expected.abs().max().item()
    )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,columns,depth", [(1, 256, 128), (1500, 512, 1024), (3000, 5120, 8192)]
)
@torch.inference_mode()
def test_dense_mxfp8_gemm_matches_native(rows, columns, depth):
    """The persistent CuTe DSL MXFP8 GEMM equals the native CUTLASS GEMM byte for byte."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.output_projection import (
        dense_mxfp8_gemm,
    )

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("MXFP8 CuTe DSL GEMMs require SM100/SM103")
    gen = torch.Generator(device="cuda").manual_seed(rows + columns)
    x = (torch.randn(rows, depth, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    activation, activation_scale = torch.ops.trtllm.mxfp8_quantize(x, True)
    weight = (torch.randn(columns, depth, device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.ops.trtllm.block_scale_interleave(
        torch.randint(
            118, 124, (columns, depth // 32), device="cuda", dtype=torch.uint8, generator=gen
        )
    ).reshape(-1)
    alpha = torch.ones(1, device="cuda")
    actual = dense_mxfp8_gemm(activation, activation_scale.view(-1), weight, weight_scale, alpha)
    expected = torch.ops.trtllm.mxfp8_mxfp8_gemm(
        activation, activation_scale, weight, weight_scale, alpha, torch.bfloat16
    )
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,groups,heads,rank",
    [(1, 1, 1, 128), (130, 2, 2, 256), (1000, 8, 8, 1024), (517, 1, 4, 384)],
)
@torch.inference_mode()
def test_grouped_output_block128_mxfp8_matches_bmm_and_reference(rows, groups, heads, rank):
    """Block-128 O-A with an MXFP8 epilogue: bytes of the blockwise BMM's BF16 output quantized
    by ``mxfp8_quantize``, and within E4M3 rounding of an FP64 reference."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.output_projection import (
        grouped_output_block128_mxfp8,
    )
    from tensorrt_llm._torch.attention.mla import _cute_dsl_fp8_bmm_out

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Blockwise CuTe DSL GEMMs require SM100/SM103")
    gen = torch.Generator(device="cuda").manual_seed(rows * 7 + rank)
    depth = heads * 512
    padded = (rows + 3) // 4 * 4
    data = (torch.randn(groups, rows, depth, device="cuda", generator=gen) * 100).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(groups, depth // 128, padded, device="cuda", generator=gen) * 0.01 + 1e-4
    weight = (torch.randn(groups, rank, depth, device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.exp2(
        torch.randint(-10, -5, (groups, rank // 128, depth // 128), device="cuda", generator=gen)
    ).float()
    quantized, quantized_scale = grouped_output_block128_mxfp8(data, scale, weight, weight_scale)
    assert quantized.shape == (rows, groups * rank)
    # A 2x2 cluster also runs tiles past a group's columns when its tile count is odd.
    paired = grouped_output_block128_mxfp8(data, scale, weight, weight_scale, cluster=(2, 2))
    for actual, expected in zip(paired, (quantized, quantized_scale)):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    latent = torch.empty(rows, groups, rank, device="cuda", dtype=torch.bfloat16)
    _cute_dsl_fp8_bmm_out(data, weight, scale, weight_scale, latent.transpose(0, 1))
    expected, expected_scale = torch.ops.trtllm.mxfp8_quantize(latent.view(rows, -1), True)
    torch.testing.assert_close(
        quantized.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )
    sf_rows = (rows + 127) // 128 * 128
    unswizzle = torch.ops.trtllm.block_scale_interleave_reverse
    torch.testing.assert_close(
        unswizzle(quantized_scale.view(sf_rows, -1)),
        unswizzle(expected_scale.view(sf_rows, -1)),
        atol=0,
        rtol=0,
    )

    activation = data.double() * scale[..., :rows].double().transpose(1, 2).repeat_interleave(
        128, 2
    )
    dequant = weight.double() * weight_scale.double().repeat_interleave(128, 1).repeat_interleave(
        128, 2
    )
    reference = torch.einsum("gmd,grd->mgr", activation, dequant).reshape(rows, -1)
    exponent = unswizzle(quantized_scale.view(sf_rows, -1))[:rows].double() - 127
    actual = quantized.double() * torch.exp2(exponent).repeat_interleave(32, 1)
    # BF16 rounding, then E4M3 rounding under a power-of-two block scale.
    bound = (
        reference.abs() * (2**-8 + 2**-4) + torch.exp2(exponent).repeat_interleave(32, 1) * 2**-9
    )
    assert ((actual - reference).abs() <= bound + 1e-5 * reference.abs().max()).all()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,groups,heads,rank",
    [(1, 1, 8, 1024), (6, 2, 4, 256), (130, 8, 8, 1024), (300, 2, 8, 1024), (513, 1, 2, 128)],
)
@torch.inference_mode()
def test_grouped_output_mxfp8_matches_reference(rows, groups, heads, rank):
    """Inverse-RoPE MXFP8 quantization and the grouped MXFP8 O-A GEMM against FP64."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.output_projection import (
        grouped_output_mxfp8,
        inverse_rope_mxfp8,
        mxfp8_weight_scale,
    )

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("MXFP8 CuTe DSL GEMMs require SM100/SM103")
    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + groups)
    depth = heads * 512
    weight = (torch.randn(groups, rank, depth, device="cuda", generator=gen) * 0.5).to(
        torch.float8_e4m3fn
    )
    exponents = torch.randint(
        -9, -4, (groups, rank // 128, depth // 128), device="cuda", generator=gen
    )
    scale = torch.exp2(exponents.float())
    assert mxfp8_weight_scale(scale * 1.5) is None
    weight_scale = mxfp8_weight_scale(scale)
    cos_sin = (torch.rand(512, 2, 32, device="cuda", generator=gen) * 2 - 1).contiguous()
    output = torch.randn(rows, groups * depth, device="cuda", dtype=torch.bfloat16, generator=gen)
    positions = torch.randint(0, 512, (rows,), device="cuda", dtype=torch.int32, generator=gen)
    reference = _inverse_rope_o_a_reference(output, positions, cos_sin, weight, scale)

    # Each 1x32 block of the rotated output is E4M3 under its power-of-two scale.
    quantized, scales = inverse_rope_mxfp8(output, positions, cos_sin, groups)
    sf_rows = (rows + 127) // 128 * 128
    exponent = torch.stack(
        [
            torch.ops.trtllm.block_scale_interleave_reverse(group.view(sf_rows, depth // 32))[:rows]
            for group in scales.view(groups, -1)
        ]
    )
    block_scale = torch.exp2(exponent.double() - 127).repeat_interleave(32, 2)
    dequantized = quantized.double() * block_scale  # [groups, rows, depth]
    rotated = _inverse_rope_reference(output, positions, cos_sin).view(rows, groups, depth)
    rotated = rotated.transpose(0, 1)
    bound = rotated.abs() * (2**-4 + 2**-8) + block_scale * 2**-10
    assert ((dequantized - rotated).abs() <= bound).all()

    alpha = torch.ones(1, device="cuda")
    actual = grouped_output_mxfp8(output, positions, cos_sin, weight, weight_scale, alpha)
    emulated = torch.einsum(
        "gmd,grd->mgr",
        dequantized,
        weight.double() * scale.double().repeat_interleave(128, 1).repeat_interleave(128, 2),
    ).reshape(rows, -1)
    # FP32 accumulation of the quantized operands, rounded to BF16.
    tight = emulated.abs() * 2**-7 + 1e-3 * emulated.abs().max()
    assert ((actual.double() - emulated).abs() <= tight).all()
    assert (actual.double() - reference).abs().max() <= 0.06 * reference.abs().max()

    # The MXFP8 epilogue matches quantizing the BF16 projection afterwards.
    latent, latent_scale = grouped_output_mxfp8(
        output, positions, cos_sin, weight, weight_scale, alpha, quantize_output=True
    )
    expected, expected_scale = torch.ops.trtllm.mxfp8_quantize(actual, True)
    torch.testing.assert_close(latent.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)
    unswizzle = torch.ops.trtllm.block_scale_interleave_reverse
    torch.testing.assert_close(
        unswizzle(latent_scale.view(sf_rows, -1))[:rows],
        unswizzle(expected_scale.view(sf_rows, -1))[:rows],
        atol=0,
        rtol=0,
    )
