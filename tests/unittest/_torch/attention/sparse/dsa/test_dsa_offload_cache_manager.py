# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSACacheManagerV2 with sparse KV cache offload, on the native KV cache manager."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.dsa import custom_ops as dsa_custom_ops
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import forward_sparse_attn_custom_op
from tensorrt_llm._torch.attention.backends.sparse.offload import (
    SPARSE_OFFLOAD_STATE_COPIES,
    SparseOffloadState,
    allocate_sparse_offload_state,
)
from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import merge_sparse_arena_indices
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_sm_version
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="DSA KV cache offload needs an SM100+ GPU",
)

NUM_LAYERS = 7
FULL_INDEXER_LAYERS = (0, 1, 2, 6)
TOPK = 64
TOKENS_PER_BLOCK = 64
MAX_BATCH_SIZE = 4
HEAD_DIM = 576
INDEXER_BYTES_PER_TOKEN = 132
# An NVFP4 latent token: 576 values and a 64-value RoPE residual, two values per
# byte and one E4M3 block scale per 16 values.
NVFP4_RESIDUAL_DIM = 64
NVFP4_DATA_BYTES = (HEAD_DIM + NVFP4_RESIDUAL_DIM) // 2
NVFP4_SCALE_BYTES = (HEAD_DIM + NVFP4_RESIDUAL_DIM) // 16


def _pretrained_config(num_layers: int = NUM_LAYERS) -> SimpleNamespace:
    # GLM-5.2's indexer pattern: a full layer every fourth from layer 3 on.
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk=TOPK,
        index_head_dim=128,
        index_n_heads=32,
        qk_rope_head_dim=64,
    )


def _pretrained_config_all_full(num_layers: int = NUM_LAYERS) -> SimpleNamespace:
    """Every layer owns an indexer, so no layer reuses another's top-k."""
    config = _pretrained_config(num_layers)
    config.index_topk_freq = 1
    config.index_skip_topk_offset = 0
    return config


def _manager(
    offload: bool = True,
    num_layers: int = NUM_LAYERS,
    dtype: DataType = DataType.FP8,
    pretrained_config: SimpleNamespace | None = None,
) -> DSACacheManagerV2:
    sparse_config = DeepSeekSparseAttentionConfig(
        index_head_dim=128,
        index_n_heads=32,
        index_topk=TOPK,
        enable_kv_cache_offload=offload,
        kv_cache_offload_config=(
            {"gpu_cache_factor": 2.0, "host_history_size": 1 << 30} if offload else {}
        ),
    )
    return DSACacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, max_tokens=16384, host_cache_size=0
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=1024,
        max_batch_size=MAX_BATCH_SIZE,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config or _pretrained_config(num_layers),
    )


def _request(request_id: int, prompt_len: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=32,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


@pytest.fixture
def manager():
    manager = _manager()
    yield manager
    manager.shutdown()


def test_offload_puts_every_latent_in_host_memory_and_keeps_the_indexer(manager):
    layers = manager.kv_cache_manager_py_config.layers
    assert len(layers) == NUM_LAYERS
    for layer in layers:
        roles = {str(buffer.role): buffer for buffer in layer.buffers}
        assert roles[str(Role.KEY)].is_sparse
        assert layer.sparse_cache.num_selected_blocks == TOPK
        assert layer.sparse_cache.unit_tokens == 1
        has_indexer = str(Role.INDEX_KEY) in roles
        assert has_indexer == (int(layer.layer_id) in FULL_INDEXER_LAYERS)
        if has_indexer:
            assert not roles[str(Role.INDEX_KEY)].is_sparse
    for local_layer in range(NUM_LAYERS):
        expected = INDEXER_BYTES_PER_TOKEN if local_layer in FULL_INDEXER_LAYERS else 0
        assert manager.get_layer_bytes_per_token(local_layer, Role.ALL) == expected
        assert manager.get_layer_bytes_per_token(local_layer, Role.KEY) == HEAD_DIM
    descriptors = manager.get_sparse_offload_descriptors()
    assert sorted(descriptors) == list(range(NUM_LAYERS))
    for layer, descriptor in descriptors.items():
        assert descriptor.unit_tokens == 1
        assert descriptor.scale_buffer_index is None
        assert len(descriptor.buffer_ids) == 1
        pool_pointers, pool_mapping = manager.get_sparse_latent_write_pool(layer)
        assert pool_pointers.tolist() == [[descriptor.arena_base_addresses[0], 0]]
        assert pool_mapping.tolist() == [[0, 0]]
    assert manager._sparse_offload_nvfp4_head_dim() is None


def test_nvfp4_offload_puts_block_scales_in_host_memory_beside_the_latent():
    manager = _manager(dtype=DataType.NVFP4)
    try:
        for layer in manager.kv_cache_manager_py_config.layers:
            roles = {str(buffer.role): buffer for buffer in layer.buffers}
            assert roles[str(Role.KEY)].is_sparse
            assert roles[str(Role.KEY_BLOCK_SCALE)].is_sparse
        for local_layer in range(NUM_LAYERS):
            expected = INDEXER_BYTES_PER_TOKEN if local_layer in FULL_INDEXER_LAYERS else 0
            assert manager.get_layer_bytes_per_token(local_layer, Role.ALL) == expected
            assert manager.get_layer_bytes_per_token(local_layer, Role.KEY) == NVFP4_DATA_BYTES
            assert (
                manager.get_layer_bytes_per_token(local_layer, Role.KEY_BLOCK_SCALE)
                == NVFP4_SCALE_BYTES
            )
        for layer, descriptor in manager.get_sparse_offload_descriptors().items():
            data, scales = descriptor.data_buffer_index, descriptor.scale_buffer_index
            assert len(descriptor.buffer_ids) == 2 and {data, scales} == {0, 1}
            # The writer and the gather address both arenas with the data's rows.
            assert descriptor.arena_row_counts[data] == descriptor.arena_row_counts[scales]
            data_base = descriptor.arena_base_addresses[data]
            scale_base = descriptor.arena_base_addresses[scales]
            pool_pointers, pool_mapping = manager.get_sparse_latent_write_pool(layer)
            assert pool_pointers.tolist() == [[[data_base, scale_base], [0, 0]]]
            assert pool_mapping.tolist() == [[0, 0]]
            data_rows, scale_rows = manager.get_sparse_latent_nvfp4_arenas(layer)
            assert (data_rows.data_ptr(), scale_rows.data_ptr()) == (data_base, scale_base)
            assert (data_rows.dtype, scale_rows.dtype) == (torch.uint8, torch.float8_e4m3fn)
            rows = descriptor.arena_row_counts[data]
            assert data_rows.shape == (rows, 1, NVFP4_DATA_BYTES)
            assert scale_rows.shape == (rows, 1, NVFP4_SCALE_BYTES)
        # Attention reads the dequantized FP8 rows of each selection from offload scratch.
        assert manager._sparse_offload_nvfp4_head_dim() == HEAD_DIM
        state_bytes = super(DSACacheManagerV2, manager).get_sparse_offload_scratch_bytes(
            manager.kv_cache_manager_py_config
        )
        assert state_bytes == SPARSE_OFFLOAD_STATE_COPIES * SparseOffloadState.device_bytes(
            max_batch_size=MAX_BATCH_SIZE, topk=TOPK, buffer_counts={2}, nvfp4_head_dim=HEAD_DIM
        )
    finally:
        manager.shutdown()


def test_shared_indexer_layers_name_one_selection(manager):
    """A run of layers reusing one indexer resolves once and shares one resident set.

    The groups here are GLM-5.2's: full indexers at 0, 1, 2 and 6, so layers 3-5
    reuse layer 2's top-k and nothing else is shared. A layer that selects for
    itself names no selection, which is what keeps the fused fetch path for every
    model without indexer sharing.
    """
    expected_owner = {0: None, 1: None, 2: 2, 3: 2, 4: 2, 5: 2, 6: None}
    assert manager._selection_leaders() == expected_owner

    layers = manager.kv_cache_manager_py_config.layers
    ids = {int(layer.layer_id): layer.sparse_cache.selection_id for layer in layers}
    # Members agree; the layers selecting for themselves name nothing. The value is
    # negative so it cannot be read as a layer id, which is a space an unset id
    # already occupies.
    assert ids == {0: None, 1: None, 2: -3, 3: -3, 4: -3, 5: -3, 6: None}

    descriptors = manager.get_sparse_offload_descriptors()
    assert {layer: d.selection_owner for layer, d in descriptors.items()} == expected_owner


def test_a_model_without_indexer_sharing_shares_no_selection():
    """Every layer owning an indexer leaves every selection a layer's own."""
    manager = _manager(pretrained_config=_pretrained_config_all_full())
    try:
        assert set(manager._selection_leaders().values()) == {None}
        for layer in manager.kv_cache_manager_py_config.layers:
            assert layer.sparse_cache.selection_id is None
        for descriptor in manager.get_sparse_offload_descriptors().values():
            assert descriptor.selection_owner is None
    finally:
        manager.shutdown()


def test_offload_refuses_the_unified_pool_view(manager):
    with pytest.raises(RuntimeError, match="host memory"):
        manager.get_unique_primary_pool()


def test_decode_step_writes_fetches_and_advances_history(manager):
    prompt_len = 3 * TOKENS_PER_BLOCK + 5
    request = _request(7, prompt_len)
    assert manager.prepare_disagg_gen_init(request)
    assert manager.try_allocate_generation(request)
    cache = manager.kv_cache_map[request.py_request_id]
    assert (cache.history_length, cache.capacity) == (prompt_len, prompt_len + 1)

    descriptors = manager.get_sparse_offload_descriptors()
    state = allocate_sparse_offload_state(
        lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        descriptors,
        max_batch_size=MAX_BATCH_SIZE,
        topk=TOPK,
    )
    manager.prepare_sparse_batch([request.py_request_id])
    manager.prepare_sparse_offload(state, [request.py_request_id], beam_width=1)
    assert state.history_lengths[0].item() == prompt_len

    # The last history token and the token written this step, then older history.
    topk = torch.full((1, TOPK), -1, dtype=torch.int32, device="cuda")
    topk[0, : TOPK - 2] = torch.arange(TOPK - 2, dtype=torch.int32)
    topk[0, TOPK - 2] = prompt_len - 1
    topk[0, TOPK - 1] = prompt_len
    request_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    kv_lengths = torch.tensor([prompt_len + 1], dtype=torch.int32, device="cuda")
    out = torch.empty((1, TOPK), dtype=torch.int32, device="cuda")
    assert state.batch_rows.tolist() == [0] + [-1] * (MAX_BATCH_SIZE - 1)
    for layer in range(NUM_LAYERS):
        writes = manager.prepare_sparse_write_indices(state, layer)
        assert writes[0, 0].item() >= 0
        assert (writes[1:, 0] == -1).all()
        fetched = manager.fetch_sparse_indices(state, layer, topk, request_indices, kv_lengths)
        merge_sparse_arena_indices(
            topk,
            request_indices,
            state.batch_rows,
            state.active_request_count,
            kv_lengths,
            state.history_lengths,
            fetched[:, 0, :],
            writes[:, 0],
            out,
            unit_tokens=1,
        )
        merged = out[0].tolist()
        assert all(index >= 0 for index in merged)
        # Every selected token has its own arena row, the written one included.
        assert len(set(merged)) == TOPK
        assert merged[-1] == writes[0, 0].item()
        assert fetched[0, 0, -1].item() == -1
        rows_in_arena = descriptors[layer].arena_row_counts[0]
        assert max(merged) < rows_in_arena

    scheduled = SimpleNamespace(context_requests=[], generation_requests=[request])
    manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
    torch.cuda.synchronize()
    assert cache.history_length == prompt_len + 1
    manager.free_resources(request)


def _arena_rows(descriptor, index: int, row_bytes: int) -> torch.Tensor:
    """One buffer's arena as int32 words per row."""
    return convert_to_torch_tensor(
        TensorWrapper(
            descriptor.arena_base_addresses[index],
            torch.int32,
            (descriptor.arena_row_counts[index], row_bytes // 4),
        )
    )


@pytest.mark.parametrize(
    ("num_layers", "num_requests", "dtype"),
    [
        (1, 1, DataType.FP8),
        (2, 1, DataType.FP8),
        (NUM_LAYERS, 1, DataType.FP8),
        (NUM_LAYERS, 3, DataType.FP8),
        (NUM_LAYERS, 1, DataType.NVFP4),
        (NUM_LAYERS, 3, DataType.NVFP4),
    ],
    ids=["1layer", "2layers", "7layers", "7layers-3requests", "nvfp4", "nvfp4-3requests"],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_fetched_rows_hold_written_history_across_evictions(seed, num_layers, num_requests, dtype):
    """Every step writes each request's token row; later fetches must return it.

    The GPU cache holds a few selections, so a long decode evicts rows and fetches
    them back from the host pages the write-back filled. Requests join at different
    steps, so the Batch's rows carry different histories. Input runs newest first,
    as on a disaggregated decode worker, while each request keeps the Batch row it
    joined in, so input positions and Batch rows disagree. An NVFP4 latent's block
    scales travel in their own buffer and must come back in the data's rows.
    """
    generator = torch.Generator().manual_seed(seed)
    manager = _manager(num_layers=num_layers, dtype=dtype)
    requests = [_request(9 + index, 1) for index in range(num_requests)]
    join_step = {request.py_request_id: 40 * index for index, request in enumerate(requests)}
    descriptors = manager.get_sparse_offload_descriptors()
    nvfp4 = dtype == DataType.NVFP4
    arenas = {
        layer: [_arena_rows(descriptor, descriptor.data_buffer_index, HEAD_DIM)]
        for layer, descriptor in descriptors.items()
    }
    if nvfp4:
        arenas = {
            layer: [
                _arena_rows(descriptor, descriptor.data_buffer_index, NVFP4_DATA_BYTES),
                _arena_rows(descriptor, descriptor.scale_buffer_index, NVFP4_SCALE_BYTES),
            ]
            for layer, descriptor in descriptors.items()
        }
    state = allocate_sparse_offload_state(
        lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        descriptors,
        max_batch_size=MAX_BATCH_SIZE,
        topk=TOPK,
    )
    active: list[LlmRequest] = []
    permuted = False
    for step in range(6 * TOKENS_PER_BLOCK):
        for request in requests:
            if join_step[request.py_request_id] == step:
                assert manager.prepare_disagg_gen_init(request)
                active.append(request)
        for request in active:
            assert manager.try_allocate_generation(request)
        scheduled = SimpleNamespace(context_requests=[], generation_requests=active[::-1])
        manager.prepare_resources(scheduled)
        request_ids = [request.py_request_id for request in scheduled.generation_requests]
        manager.prepare_sparse_offload(state, request_ids, beam_width=1)
        # The rows the manager recorded at add_kv_cache are still the native ones.
        for member in manager._sparse_batch_members.values():
            assert manager._sparse_batch.row_of(member.cache) == member.row
        batch_rows = state.batch_rows.tolist()[: len(active)]
        permuted |= batch_rows != list(range(len(active)))
        histories = [manager.kv_cache_map[rid].history_length for rid in request_ids]
        topk = torch.full((len(active), TOPK), -1, dtype=torch.int32)
        for position, history in enumerate(histories):
            # Unit 0 came with the prompt and was never written; select from the rest,
            # always including the token written this step.
            candidates = torch.randperm(history, generator=generator)[: TOPK - 1]
            candidates = candidates[candidates > 0][: TOPK - 1].to(torch.int32)
            topk[position, : candidates.numel()] = candidates
            topk[position, -1] = history
        topk = topk.cuda()
        request_indices = torch.arange(len(active), dtype=torch.int32, device="cuda")
        kv_lengths = torch.tensor([h + 1 for h in histories], dtype=torch.int32, device="cuda")
        out = torch.empty((len(active), TOPK), dtype=torch.int32, device="cuda")
        for layer in range(num_layers):
            descriptor = descriptors[layer]
            data = descriptor.data_buffer_index
            writes = manager.prepare_sparse_write_indices(state, layer)
            # At the buffer's slice, not its position: buffers whose arena indices
            # coincide share one, and the slices no buffer maps to are never written.
            data = state.out_slots[layer][data]
            for row, rid, history in zip(batch_rows, request_ids, histories):
                write_row = writes[row, data].item()
                assert write_row >= 0, (rid, history, layer)
                # The token's row: its position, its layer, its request.
                for arena in arenas[layer]:
                    arena[write_row, :3] = torch.tensor(
                        [history, layer, rid], dtype=torch.int32, device="cuda"
                    )
            free_rows = [row for row in range(MAX_BATCH_SIZE) if row not in batch_rows]
            assert (writes[free_rows, data] == -1).all()
            fetched = manager.fetch_sparse_indices(state, layer, topk, request_indices, kv_lengths)
            if nvfp4:
                # The scale mirrors the data pool, so it resolves to the same index and
                # reads the data's slice rather than one of its own. One slice written
                # once is what replaces the two equal rows this used to compare.
                assert state.out_slots[layer][descriptor.scale_buffer_index] == data
            merge_sparse_arena_indices(
                topk,
                request_indices,
                state.batch_rows,
                state.active_request_count,
                kv_lengths,
                state.history_lengths,
                fetched[:, data, :],
                writes[:, data],
                out,
                unit_tokens=1,
            )
            for position, rid in enumerate(request_ids):
                valid = topk[position] >= 0
                units = topk[position][valid]
                expected = torch.stack(
                    (units, torch.full_like(units, layer), torch.full_like(units, rid)), dim=1
                )
                for arena in arenas[layer]:
                    got = arena[out[position][valid].long(), :3]
                    wrong = (got != expected).any(dim=1)
                    assert not wrong.any(), (
                        f"request {rid}, history {histories[position]}, layer {layer}: units "
                        f"{units[wrong].tolist()} read {got[wrong].tolist()}"
                    )
        manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
        for rid, history in zip(request_ids, histories):
            assert manager.kv_cache_map[rid].history_length == history + 1
    assert permuted == (num_requests > 1)
    for request in active:
        manager.free_resources(request)
    manager.shutdown()


def test_nvfp4_attention_reads_the_selected_rows_dequantized():
    """Attention reads FP8 scratch rows dequantized from the arena rows the merge picked.

    Arena rows hold random NVFP4 bytes and scales. The reference dequantizes the
    merged rows after copying them out of the arena, so it checks which rows the
    gather read, from which views, and that each index became its scratch row.
    """
    manager = _manager(dtype=DataType.NVFP4)
    generator = torch.Generator(device="cuda").manual_seed(0)
    layer = 2
    try:
        prompt_len = 3 * TOKENS_PER_BLOCK + 5
        request = _request(7, prompt_len)
        assert manager.prepare_disagg_gen_init(request)
        assert manager.try_allocate_generation(request)
        state = allocate_sparse_offload_state(
            lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
            manager.get_sparse_offload_descriptors(),
            max_batch_size=MAX_BATCH_SIZE,
            topk=TOPK,
            nvfp4_head_dim=manager._sparse_offload_nvfp4_head_dim(),
        )
        manager.prepare_sparse_batch([request.py_request_id])
        manager.prepare_sparse_offload(state, [request.py_request_id], beam_width=1)
        data_rows, scale_rows = manager.get_sparse_latent_nvfp4_arenas(layer)
        data_rows.copy_(
            torch.randint(
                0, 256, data_rows.shape, dtype=torch.uint8, device="cuda", generator=generator
            )
        )
        scale_rows.copy_(
            torch.empty(scale_rows.shape, device="cuda")
            .uniform_(0.25, 4.0, generator=generator)
            .to(torch.float8_e4m3fn)
        )
        writes = manager.prepare_sparse_write_indices(state, layer)

        # Older history, the last history token and the token written this step,
        # with unselected slots between them.
        topk = torch.full((1, TOPK), -1, dtype=torch.int32, device="cuda")
        topk[0, : TOPK - 8] = torch.arange(TOPK - 8, dtype=torch.int32) * 3
        topk[0, TOPK - 2] = prompt_len - 1
        topk[0, TOPK - 1] = prompt_len
        metadata = SimpleNamespace(
            kv_cache_manager=manager,
            sparse_offload_state=state,
            num_contexts=0,
            num_seqs=1,
            sparse_mla_topk=TOPK,
            req_idx_per_token=torch.zeros(1, dtype=torch.int32, device="cuda"),
            kv_lens_cuda=torch.tensor([prompt_len + 1], dtype=torch.int32, device="cuda"),
            sparse_offload_attn_indices=torch.empty(
                (MAX_BATCH_SIZE, TOPK), dtype=torch.int32, device="cuda"
            ),
        )
        backend = SimpleNamespace(
            layer_idx=layer,
            _nvfp4_mla_kv_scale_quant_orig=torch.tensor([0.5], device="cuda"),
        )
        args = SimpleNamespace(sparse_runtime_params=SimpleNamespace(aux_kv_cache_pool_ptr=0))
        indices = DSATrtllmAttention._sparse_offload_arena_indices(backend, topk, metadata, args)

        descriptor = state.layers[layer]
        data = descriptor.data_buffer_index
        rows = torch.empty_like(topk)
        merge_sparse_arena_indices(
            topk,
            metadata.req_idx_per_token,
            state.batch_rows,
            state.active_request_count,
            metadata.kv_lens_cuda,
            state.history_lengths,
            state.fetched_indices[2][:, data, :],
            writes[:, data],
            rows,
            unit_tokens=1,
        )
        valid = rows[0] >= 0
        assert valid.sum().item() == TOPK - 6
        assert rows[0, -1].item() == writes[0, data].item()
        flat = rows.view(-1).clamp(min=0).long()
        reference = torch.empty((1, TOPK, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")
        compact = torch.where(valid, torch.arange(TOPK, device="cuda"), -1).to(torch.int32)[None]
        torch.ops.trtllm.nvfp4_mla_kv_cache_gather_direct(
            data_rows[flat].contiguous(),
            scale_rows.view(torch.uint8)[flat].contiguous().view(torch.float8_e4m3fn),
            compact,
            reference,
            backend._nvfp4_mla_kv_scale_quant_orig,
            NVFP4_RESIDUAL_DIM,
            TOPK,
        )
        assert args.sparse_runtime_params.aux_kv_cache_pool_ptr == state.nvfp4_scratch.data_ptr()
        torch.testing.assert_close(indices, compact)
        scratch = state.nvfp4_scratch[0].view(torch.uint8)
        assert torch.equal(scratch[valid], reference[0].view(torch.uint8)[valid])
        manager.free_resources(request)
    finally:
        manager.shutdown()


def test_offload_scratch_counts_the_latent_writer(manager):
    with_writer = manager.get_sparse_offload_scratch_bytes(manager.kv_cache_manager_py_config)
    state_only = super(DSACacheManagerV2, manager).get_sparse_offload_scratch_bytes(
        manager.kv_cache_manager_py_config
    )
    assert with_writer - state_only == 2 * 3 * MAX_BATCH_SIZE * 4


@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "cuda-graph"])
@pytest.mark.parametrize("custom_op", [False, True], ids=["direct", "registered"])
def test_overlapped_write_mapping_tracks_reordered_requests(monkeypatch, dtype, capture, custom_op):
    """The writer joins reservation and reads current Batch rows on every replay."""
    manager = _manager(num_layers=2, dtype=dtype)
    requests = [_request(17, 1), _request(18, 1)]
    descriptors = manager.get_sparse_offload_descriptors()
    state = allocate_sparse_offload_state(
        lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        descriptors,
        max_batch_size=MAX_BATCH_SIZE,
        topk=TOPK,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=manager,
        sparse_offload_state=state,
        sparse_offload_stream=torch.cuda.Stream(),
        sparse_offload_start_event=torch.cuda.Event(),
        sparse_offload_write_event=torch.cuda.Event(),
        sparse_offload_write_pending=None,
        sparse_latent_block_offsets=torch.full(
            (1, MAX_BATCH_SIZE, 2, 1), -1, dtype=torch.int32, device="cuda"
        ),
        sparse_latent_write_slots=torch.full(
            (MAX_BATCH_SIZE,), -1, dtype=torch.int32, device="cuda"
        ),
    )
    observed = torch.empty((2, MAX_BATCH_SIZE), dtype=torch.int32, device="cuda")
    observed_slots = torch.empty_like(observed)
    latent = torch.empty((len(requests), HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    backends = []
    for layer in range(2):
        backend = object.__new__(DSATrtllmAttention)
        backend.layer_idx = layer
        backend.sparse_params = SimpleNamespace(enable_kv_cache_offload=True)
        backends.append(backend)

    def consume_write_mapping(self, fused_q, q_pe, latent_cache, metadata, **kwargs):
        target = kwargs["latent_write_target"]
        observed[self.layer_idx].copy_(target.block_offsets[0, :, 0, 0])
        observed_slots[self.layer_idx].copy_(metadata.sparse_latent_write_slots)

    monkeypatch.setattr(TrtllmAttention, "mla_rope_generation", consume_write_mapping)

    def project(hidden_states, position_ids, layer_idx):
        assert metadata.sparse_offload_write_pending == int(layer_idx)
        return [latent] * 4

    def attend(q, compressed_kv, k_pe, latent_cache, intermediates, positions, layer_idx, output):
        backend = backends[int(layer_idx)]
        backend.prepare_sparse_write(metadata)
        backend.mla_rope_generation(None, None, latent, metadata)

    monkeypatch.setattr(dsa_custom_ops, "_extract_extra_attrs", lambda layer: (metadata, None))
    monkeypatch.setattr(torch.ops.trtllm, "mla_dsa_proj", project)
    monkeypatch.setattr(dsa_custom_ops, "maybe_bcg_mla_dsa_attn_inplace", attend)

    def forward():
        for backend in backends:
            if custom_op:
                mla = SimpleNamespace(mqa=backend, layer_idx_str=str(backend.layer_idx))
                forward_sparse_attn_custom_op(mla, latent, None, [latent], None)
                continue
            backend.prepare_sparse_write(metadata)
            # Entry through the attention custom op must reuse the pending mapping.
            backend.prepare_sparse_write(metadata)
            backend.mla_rope_generation(None, None, latent, metadata)

    graph = None
    try:
        for request in requests:
            assert manager.prepare_disagg_gen_init(request)
        for step in range(6):
            for request in requests:
                assert manager.try_allocate_generation(request)
            ordered = requests if step % 2 == 0 else requests[::-1]
            scheduled = SimpleNamespace(context_requests=[], generation_requests=ordered)
            manager.prepare_resources(scheduled)
            manager.prepare_sparse_offload(
                state, [request.py_request_id for request in ordered], beam_width=1
            )
            with with_multi_stream(True):
                if capture and graph is None:
                    forward()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        forward()
                if graph is None:
                    forward()
                else:
                    graph.replay()
            assert metadata.sparse_offload_write_pending is None
            # The serial native API is idempotent for the same in-flight token.
            # It independently checks the destinations consumed by the writer.
            for layer in range(2):
                writes = manager.prepare_sparse_write_indices(state, layer)
                data = descriptors[layer].data_buffer_index
                expected = torch.full_like(observed[layer], -1)
                expected[: len(requests)] = writes[state.batch_rows[: len(requests)], data]
                torch.testing.assert_close(observed[layer], expected)
                torch.testing.assert_close(
                    observed_slots[layer], (expected >= 0).to(torch.int32) - 1
                )
            manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
        for request in requests:
            manager.free_resources(request)
    finally:
        torch.cuda.synchronize()
        manager.shutdown()


@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "cuda-graph"])
@pytest.mark.parametrize("explicit_pattern", [False, True], ids=["glm52", "glm53"])
def test_shared_layer_prefetch_preserves_history(monkeypatch, dtype, capture, explicit_pattern):
    """Lookahead fetch preserves rows through eviction, request reordering and replay."""
    config = _pretrained_config()
    if explicit_pattern:
        config.index_topk_pattern = ["F", "S", "S", "F", "S", "F", "S"]
    manager = _manager(dtype=dtype, pretrained_config=config)
    requests = [_request(17, 1), _request(18, 1)]
    metadata = object.__new__(DSAtrtllmAttentionMetadata)
    metadata.kv_cache_manager = manager
    metadata.sparse_metadata_params = SimpleNamespace(enable_kv_cache_offload=True)
    metadata.beam_width = 1
    metadata.max_draft_tokens = 0
    metadata.draft_kv_cache_manager = None
    metadata.sparse_mla_topk = TOPK
    metadata.is_cuda_graph = False
    metadata.cuda_graph_buffers = {}
    metadata.get_empty = lambda buffers, shape, dtype, **kwargs: torch.empty(
        shape, dtype=dtype, device="cuda"
    )
    metadata._init_dsa_sparse_offload_state()
    expected_pairs = {0: 1, 1: 2, 3: 4, 5: 6} if explicit_pattern else {2: 3, 3: 4, 4: 5}
    assert metadata.sparse_offload_next_shared_layer == expected_pairs
    state = metadata.sparse_offload_state
    # Inspect raw data and scale bytes separately; dequantization has its own test.
    state.nvfp4_scratch = None
    num_requests = len(requests)
    metadata._num_tokens = num_requests
    metadata._seq_lens = torch.ones(num_requests, dtype=torch.int32)
    metadata._num_contexts = 0
    metadata.shared_topk_indices = torch.full(
        (num_requests, TOPK), -1, dtype=torch.int32, device="cuda"
    )
    metadata.req_idx_per_token = torch.arange(num_requests, dtype=torch.int32, device="cuda")
    metadata.kv_lens_cuda = torch.empty(num_requests, dtype=torch.int32, device="cuda")
    latent = torch.empty((num_requests, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    token_values = torch.empty((num_requests, 3), dtype=torch.int32, device="cuda")
    arenas = {}
    backends = []
    for layer, descriptor in state.layers.items():
        if dtype == DataType.NVFP4:
            arenas[layer] = [
                _arena_rows(descriptor, descriptor.data_buffer_index, NVFP4_DATA_BYTES),
                _arena_rows(descriptor, descriptor.scale_buffer_index, NVFP4_SCALE_BYTES),
            ]
        else:
            arenas[layer] = [_arena_rows(descriptor, descriptor.data_buffer_index, HEAD_DIM)]
        backend = object.__new__(DSATrtllmAttention)
        backend.layer_idx = layer
        backend.sparse_params = SimpleNamespace(enable_kv_cache_offload=True)
        backends.append(backend)
    observed = torch.empty(
        (NUM_LAYERS, len(arenas[0]), num_requests, TOPK, 3), dtype=torch.int32, device="cuda"
    )
    args = SimpleNamespace(sparse_runtime_params=SimpleNamespace(aux_kv_cache_pool_ptr=0))

    def write_token(self, fused_q, q_pe, latent_cache, metadata, **kwargs):
        rows = kwargs["latent_write_target"].block_offsets[0, :num_requests, 0, 0].long()
        token_values[:, 1] = self.layer_idx
        for arena in arenas[self.layer_idx]:
            arena[rows, :3] = token_values

    monkeypatch.setattr(TrtllmAttention, "mla_rope_generation", write_token)

    def forward():
        for backend in backends:
            backend.prepare_sparse_write(metadata)
            backend.mla_rope_generation(None, None, latent, metadata)
            indices = backend._sparse_offload_arena_indices(
                metadata.shared_topk_indices, metadata, args
            )
            for buffer, arena in enumerate(arenas[backend.layer_idx]):
                observed[backend.layer_idx, buffer].copy_(arena[indices.clamp_min(0).long(), :3])
            backend.prefetch_next_sparse_history(metadata)

    graph = None
    generator = torch.Generator().manual_seed(17)
    try:
        for request in requests:
            assert manager.prepare_disagg_gen_init(request)
        for step in range(3 * TOKENS_PER_BLOCK):
            for request in requests:
                assert manager.try_allocate_generation(request)
            ordered = requests if step % 2 == 0 else requests[::-1]
            scheduled = SimpleNamespace(context_requests=[], generation_requests=ordered)
            manager.prepare_resources(scheduled)
            ids = [request.py_request_id for request in ordered]
            manager.prepare_sparse_offload(state, ids, beam_width=1)
            history = step + 1
            selected = torch.full((num_requests, TOPK), -1, dtype=torch.int32)
            for row in range(num_requests):
                candidates = torch.randperm(history - 1, generator=generator)[: TOPK - 1] + 1
                selected[row, : candidates.numel()] = candidates
                selected[row, -1] = history
            metadata.shared_topk_indices.copy_(selected)
            metadata.kv_lens_cuda.fill_(history + 1)
            token_values.copy_(torch.tensor([[history, 0, rid] for rid in ids], dtype=torch.int32))
            with with_multi_stream(True):
                if capture and graph is None:
                    forward()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        forward()
                if graph is None:
                    forward()
                else:
                    graph.replay()
            assert metadata.sparse_offload_write_pending is None
            assert metadata.sparse_offload_fetch_pending is None
            actual = observed.cpu()
            for layer in range(NUM_LAYERS):
                for row, rid in enumerate(ids):
                    units = selected[row][selected[row] >= 0]
                    expected = torch.stack(
                        (units, torch.full_like(units, layer), torch.full_like(units, rid)), dim=1
                    )
                    for buffer in range(len(arenas[layer])):
                        torch.testing.assert_close(
                            actual[layer, buffer, row][selected[row] >= 0], expected
                        )
            manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
        for request in requests:
            manager.free_resources(request)
    finally:
        torch.cuda.synchronize()
        manager.shutdown()
