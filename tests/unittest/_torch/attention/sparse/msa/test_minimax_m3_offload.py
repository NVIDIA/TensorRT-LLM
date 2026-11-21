# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M3 offload against resident, identically quantized K/V through native cache transitions."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
    MiniMaxM3SparseMetadataParams,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
    fused_write_layer_caches,
    fused_write_layer_caches_nvfp4,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    minimax_m3_sparse_attn_decode,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttention,
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import create_cache_reuse_adapter
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_pool_view_global_layer_ids
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import (
    TensorWrapper,
    binding_to_torch_dtype,
    convert_to_torch_tensor,
    is_sm_100f,
)
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="M3 offload needs an SM100-family GPU"
)

PAGE = 128
HEAD_DIM = 128
TOPK = 16
BATCH = 4
SPARSE_LAYERS = (1, 3)


def _manager(
    dtype: DataType,
    heads: int,
    max_seq_len: int,
    *,
    offload: bool = True,
    block_reuse: bool = False,
) -> MiniMaxM3KVCacheManagerV2:
    return MiniMaxM3KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=block_reuse, max_tokens=BATCH * max_seq_len, host_cache_size=0
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=4,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=max_seq_len,
        max_batch_size=BATCH,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=dtype,
        sparse_layer_ids=SPARSE_LAYERS,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa",
            indexer_kv_dtype="fp8",
            enable_kv_cache_offload=offload,
            kv_cache_offload_config={"gpu_cache_factor": 1.0, "host_history_size": 512 << 20}
            if offload
            else {},
        ),
    )


@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.HALF, DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_m3_resident_and_offloaded_transfer_views_match(dtype, heads, monkeypatch):
    """Native transfer maps every dense, sparse and index pool in both directions."""
    monkeypatch.setenv("TRTLLM_KV_GUARD_PAGE", "zero")
    resident = _manager(dtype, heads, 40 * PAGE, offload=False)
    offloaded = _manager(dtype, heads, 40 * PAGE)
    try:
        assert not set(offloaded._guard_page_by_layer) & set(SPARSE_LAYERS)
        assert offloaded.get_cache_bytes_per_token() == sum(
            offloaded.get_layer_bytes_per_token(layer, Role.ALL) for layer in range(4)
        )
        for layer in offloaded.impl.init_config.layers:
            if int(layer.layer_id) not in SPARSE_LAYERS:
                assert all(
                    buffer.role not in (Role.KEY_BLOCK_SCALE, Role.VALUE_BLOCK_SCALE)
                    for buffer in layer.buffers
                )
        ranks = [
            RankInfo.from_kv_cache_manager(name, manager, device_id=0)
            for name, manager in (("context", resident), ("generation", offloaded))
        ]
        for local, peer in (ranks, ranks[::-1]):
            registrar = PeerRegistrar(local, KVRegionExtractorV1(local.page_table))
            registrar.register(peer.instance_name, peer.instance_rank, peer)
            mapping = registrar.get_pool_mapping(peer)
            assert set(mapping) == {
                (group, pool)
                for group, layer_group in enumerate(local.page_table.layer_groups)
                for pool in range(len(layer_group.pool_views))
            }
            main_layer_sets = set()
            for local_key, peer_key in mapping.items():
                local_group = local.page_table.layer_groups[local_key[0]]
                peer_group = peer.page_table.layer_groups[peer_key[0]]
                local_view = local_group.pool_views[local_key[1]]
                peer_view = peer_group.pool_views[peer_key[1]]
                local_layers = frozenset(get_pool_view_global_layer_ids(local_view, local_group))
                assert local_layers == frozenset(
                    get_pool_view_global_layer_ids(peer_view, peer_group)
                )
                if local_view.pool_role == frozenset({"key", "value"}):
                    main_layer_sets.add(local_layers)
                registrar.get_kv_map(peer, local_key, peer_key)
            assert main_layer_sets == {frozenset({0, 2}), frozenset(SPARSE_LAYERS)}
    finally:
        offloaded.shutdown()
        resident.shutdown()


def _host_pages(manager, layer: int, request_id: int, *, scales: bool = False):
    """Views of the same host pages a context worker transfers into."""
    geometry = manager._kv_scale_slot_geometry if scales else manager._kv_slot_geometry
    address, dtype, _, slot_stride, shape = geometry(layer, "HND")
    page_bytes = torch.empty((), dtype=dtype).element_size()
    for dimension in shape:
        page_bytes *= dimension
    (slots,) = manager.get_batch_cache_indices([request_id], layer_idx=layer)
    return [
        convert_to_torch_tensor(
            TensorWrapper(address + int(slot) * slot_stride * page_bytes, dtype, (2, *shape))
        )
        for slot in slots
        if int(slot) >= 0
    ]


def _write(data, scales, slots, k, v, quant):
    k, v = k.flatten(1), v.flatten(1)
    if scales is None:
        assert fused_write_layer_caches(data[:, 0], data[:, 1], None, slots, k, v, None)
    else:
        assert fused_write_layer_caches_nvfp4(
            data[:, 0], data[:, 1], scales[:, 0], scales[:, 1], None, slots, k, v, None, quant
        )


def _attention(q, data, scales, topk, table, lengths, out, quant):
    extra = (
        {}
        if scales is None
        else dict(
            k_block_scale=scales[:, 0],
            v_block_scale=scales[:, 1],
            k_global_scale=quant[1:2],
            v_global_scale=quant[2:3],
        )
    )
    minimax_m3_sparse_attn_decode(
        q,
        data[:, 0],
        data[:, 1],
        topk.transpose(0, 1),
        table,
        lengths,
        sm_scale=HEAD_DIM**-0.5,
        output=out,
        decode_query_len=1,
        **extra,
    )


@pytest.mark.parametrize("dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("shorter_prefix", [False, True], ids=["same_prompt", "shorter_prefix"])
def test_m3_offload_reuses_received_prompt(dtype, heads, shorter_prefix):
    """Reuse a received prompt or its shorter prefix, then decode across a partial page."""
    shared_blocks = 5
    prompt_len = (shared_blocks + 1) * PAGE - 1
    max_blocks = shared_blocks + 3
    manager = _manager(dtype, heads, max_blocks * PAGE, block_reuse=True)
    generator = torch.Generator(device="cuda").manual_seed(283)
    quant = torch.ones(3, dtype=torch.float32, device="cuda")
    nvfp4 = dtype == DataType.NVFP4
    try:
        metadata = MiniMaxM3MsaSparseAttentionMetadata(
            max_num_requests=BATCH,
            max_num_tokens=BATCH,
            kv_cache_manager=manager,
            mapping=manager.mapping,
            sparse_metadata_params=MiniMaxM3SparseMetadataParams(
                global_num_q_heads=16 * heads,
                global_num_kv_heads=heads,
                num_index_heads=4,
                topk=TOPK,
            ),
        )
        reference = {
            layer: (
                torch.zeros(
                    (max_blocks, 2, heads, PAGE, HEAD_DIM // (2 if nvfp4 else 1)),
                    dtype=torch.int8 if nvfp4 else torch.float8_e4m3fn,
                    device="cuda",
                ),
                torch.zeros(
                    (max_blocks, 2, heads, PAGE, HEAD_DIM // 16),
                    dtype=torch.uint8,
                    device="cuda",
                )
                if nvfp4
                else None,
            )
            for layer in SPARSE_LAYERS
        }
        first_pages = {}
        for rid in (1, 2):
            if rid == 2 and shorter_prefix:
                shared_blocks -= 1
                prompt_len -= PAGE
            request = LlmRequest(
                request_id=rid,
                max_new_tokens=8,
                input_tokens=list(range(prompt_len)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            assert manager.prepare_disagg_gen_init(request)
            cache = manager.kv_cache_map[rid]
            start = 0 if rid == 1 else shared_blocks * PAGE
            if rid == 2:
                assert request.context_current_position == cache.num_committed_tokens >= start
                assert create_cache_reuse_adapter(manager).get_cached_token_count_per_layer_group(
                    request, [None] * len(cache.reuse_status)
                ) == [start] * len(cache.reuse_status)
            for layer in range(4):
                pages = manager.get_batch_cache_indices([rid], layer_idx=layer)[0]
                if rid == 2:
                    assert pages[:shared_blocks] == first_pages[layer][:shared_blocks]
                else:
                    first_pages[layer] = pages
                if layer in SPARSE_LAYERS:
                    index = manager.get_index_k_buffer(layer).view(torch.uint8)
                    for page in range(start // PAGE, shared_blocks + 1):
                        index[pages[page]].fill_(layer + page + rid)
                    if rid == 2:
                        for page in range(shared_blocks):
                            assert (index[pages[page]] == layer + page + 1).all()
                    data, scales = reference[layer]
                    k, v = (
                        torch.randn(
                            (prompt_len - start, heads, HEAD_DIM),
                            device="cuda",
                            generator=generator,
                        ).to(torch.bfloat16)
                        for _ in range(2)
                    )
                    _write(
                        data,
                        scales,
                        torch.arange(start, prompt_len, device="cuda", dtype=torch.int32),
                        k,
                        v,
                        quant,
                    )
                    for page, host in enumerate(_host_pages(manager, layer, rid)):
                        if start // PAGE <= page <= shared_blocks:
                            host.copy_(data[page])
                    if nvfp4:
                        for page, host in enumerate(_host_pages(manager, layer, rid, scales=True)):
                            if start // PAGE <= page <= shared_blocks:
                                host.copy_(scales[page])
                else:
                    dense = manager.get_buffers(layer, "HND").view(torch.uint8)
                    for page in range(start // PAGE, shared_blocks + 1):
                        dense[pages[page]].fill_(layer + page + rid)
                    if rid == 2:
                        for page in range(shared_blocks):
                            assert (dense[pages[page]] == layer + page + 1).all()
            torch.cuda.synchronize()
            request.context_current_position = prompt_len
            manager.try_commit_blocks(request)
            assert not _introspection.is_commit_allowed(cache)

            for step in range(3):
                assert manager.try_allocate_generation(request)
                scheduled = SimpleNamespace(context_requests=[], generation_requests=[request])
                manager.prepare_resources(scheduled)
                history = cache.history_length
                assert history == prompt_len + step
                metadata.request_ids = [rid]
                metadata.seq_lens = torch.ones(1, dtype=torch.int32)
                metadata.prompt_lens = [prompt_len]
                metadata.kv_cache_params = KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[history]
                )
                metadata.prepare()
                topk = torch.full((1, heads, TOPK), -1, dtype=torch.int32, device="cuda")
                count = history // PAGE + 1
                topk[:, :, :count] = torch.arange(count, device="cuda", dtype=torch.int32)
                q = torch.randn((1, 16 * heads, HEAD_DIM), device="cuda", generator=generator).to(
                    torch.bfloat16
                )
                for layer in SPARSE_LAYERS:
                    k, v = (
                        torch.randn((1, heads, HEAD_DIM), device="cuda", generator=generator).to(
                            torch.bfloat16
                        )
                        for _ in range(2)
                    )
                    data, scales = reference[layer]
                    _write(
                        data,
                        scales,
                        torch.tensor([history], device="cuda", dtype=torch.int32),
                        k,
                        v,
                        quant,
                    )
                    MiniMaxM3MsaSparseAttention.write_layer_caches(
                        SimpleNamespace(layer_idx=layer),
                        k.flatten(1),
                        v.flatten(1),
                        None,
                        metadata,
                        quant,
                    )
                    table = metadata.fetch_sparse_blocks(layer, topk)
                    actual, expected = torch.empty_like(q), torch.empty_like(q)
                    _attention(
                        q,
                        manager.get_buffers(layer, "HND"),
                        manager.get_block_scale_buffers(layer, "HND") if nvfp4 else None,
                        topk,
                        table,
                        metadata.msa_seq_lens_cuda[:1],
                        actual,
                        quant,
                    )
                    _attention(
                        q,
                        data,
                        scales,
                        topk,
                        torch.arange(max_blocks, device="cuda", dtype=torch.int32).unsqueeze(0),
                        metadata.msa_seq_lens_cuda[:1],
                        expected,
                        quant,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
            manager.free_resources(request)
            assert _introspection.all_tree_pages_droppable(manager.impl)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "dtype,heads,graph",
    [
        (DataType.BF16, 2, False),
        (DataType.HALF, 1, False),
        (DataType.FP8, 1, False),
        (DataType.FP8, 4, True),
        (DataType.NVFP4, 1, False),
        (DataType.NVFP4, 2, True),
    ],
)
def test_m3_offload_matches_resident_across_partial_blocks_eviction_and_reseating(
    dtype: DataType, heads: int, graph: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bitwise data/scale and attention parity, including replay with changing batch rows.

    Two histories exceed two GPU cache capacities; another has one page.
    Per-head selections rotate, a request leaves and rejoins inside a growing block, and
    requests cross page boundaries at different steps. Padded graph queries
    must never write into another request's arena or become native Batch members.
    """
    history_blocks = 2 * TOPK * heads + 7
    max_blocks = history_blocks + 4
    max_seq_len = max_blocks * PAGE
    manager = _manager(dtype, heads, max_seq_len)
    generator = torch.Generator(device="cuda").manual_seed(437)
    nvfp4 = dtype == DataType.NVFP4
    requests = [
        LlmRequest(
            request_id=11 * (index + 1),
            max_new_tokens=32,
            input_tokens=list(range(length)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for index, length in enumerate(
            [
                history_blocks * PAGE + 125,
                (history_blocks + 1) * PAGE - 1,
                PAGE,
            ]
        )
    ]
    try:
        for layer in manager.kv_cache_manager_py_config.layers:
            sparse = int(layer.layer_id) in SPARSE_LAYERS
            for buffer in layer.buffers:
                assert buffer.is_sparse == (sparse and buffer.role != Role.INDEX_KEY)
            if sparse:
                assert layer.sparse_cache.include_partial_units
                assert layer.sparse_cache.num_selected_blocks == TOPK * heads
                assert layer.sparse_cache.max_cached_units == TOPK * heads + 1
        get_subpage_pool = manager.get_kv_subpage_pool

        def resident_subpage_pool(
            layer_idx: int, kv_layout: str = "HND"
        ) -> tuple[torch.Tensor, int]:
            assert not manager.offloads_layer(layer_idx), "Host history is not a CUDA pool"
            return get_subpage_pool(layer_idx, kv_layout)

        monkeypatch.setattr(manager, "get_kv_subpage_pool", resident_subpage_pool)
        metadata = MiniMaxM3MsaSparseAttentionMetadata(
            max_num_requests=BATCH,
            max_num_tokens=BATCH,
            kv_cache_manager=manager,
            mapping=manager.mapping,
            sparse_metadata_params=MiniMaxM3SparseMetadataParams(
                global_num_q_heads=16 * heads,
                global_num_kv_heads=heads,
                num_index_heads=4,
                topk=TOPK,
            ),
        )
        state = metadata.sparse_offload_state
        assert state is not None
        metadata.msa_out_cache_loc.fill_(-1)
        quant = torch.ones(3, dtype=torch.float32, device="cuda")
        q = torch.randn((BATCH, 16 * heads, HEAD_DIM), device="cuda", generator=generator).to(
            torch.bfloat16
        )
        topk = torch.full((BATCH, heads, TOPK), -1, dtype=torch.int32, device="cuda")
        reference = {}
        source = {}
        resident_data = {
            layer: torch.zeros(
                (len(requests) * max_blocks, 2, heads, PAGE, HEAD_DIM // (2 if nvfp4 else 1)),
                dtype=torch.int8 if nvfp4 else binding_to_torch_dtype(dtype),
                device="cuda",
            )
            for layer in SPARSE_LAYERS
        }
        resident_scales = {
            layer: torch.zeros(
                (len(requests) * max_blocks, 2, heads, PAGE, HEAD_DIM // 16),
                dtype=torch.uint8,
                device="cuda",
            )
            if nvfp4
            else None
            for layer in SPARSE_LAYERS
        }
        resident_table = torch.full((BATCH, max_blocks), -1, dtype=torch.int32, device="cuda")
        for request_number, request in enumerate(requests):
            assert manager.prepare_disagg_gen_init(request)
            rid, prompt = request.py_request_id, request.prompt_len
            for layer in SPARSE_LAYERS:
                k, v = (
                    torch.randn(
                        (max_seq_len, heads, HEAD_DIM), device="cuda", generator=generator
                    ).to(torch.bfloat16)
                    for _ in range(2)
                )
                pages = slice(request_number * max_blocks, (request_number + 1) * max_blocks)
                data = resident_data[layer][pages]
                scales = resident_scales[layer][pages] if nvfp4 else None
                _write(
                    data,
                    scales,
                    torch.arange(prompt, device="cuda", dtype=torch.int32),
                    k[:prompt],
                    v[:prompt],
                    quant,
                )
                for page, host in enumerate(_host_pages(manager, layer, rid)):
                    host.copy_(data[page])
                if nvfp4:
                    for page, host in enumerate(_host_pages(manager, layer, rid, scales=True)):
                        host.copy_(scales[page])
                reference[rid, layer] = (data, scales)
                source[rid, layer] = (k, v)
        torch.cuda.synchronize()

        inputs = {
            layer: tuple(
                torch.zeros((BATCH, heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            )
            for layer in SPARSE_LAYERS
        }
        outputs = {layer: torch.empty_like(q) for layer in SPARSE_LAYERS}

        def decode():
            for layer in SPARSE_LAYERS:
                k, v = inputs[layer]
                MiniMaxM3MsaSparseAttention.write_layer_caches(
                    SimpleNamespace(layer_idx=layer),
                    k.flatten(1),
                    v.flatten(1),
                    None,
                    metadata,
                    quant,
                )
                table = metadata.fetch_sparse_blocks(layer, topk)
                if not graph:
                    arena = manager.get_buffers(layer, "HND").view(torch.uint8)
                    for position, (rid, history) in enumerate(zip(ids, histories)):
                        units = topk[position].flatten().unique()
                        units = units[units >= 0].long()
                        physical = table[position, units].long()
                        assert (physical >= 0).all(), (step, layer, rid)
                        valid = units[:, None] * PAGE + torch.arange(PAGE, device="cuda") <= history
                        mask = ~valid[:, None, None, :, None]
                        torch.testing.assert_close(
                            arena[physical].masked_fill(mask, 0),
                            reference[rid, layer][0].view(torch.uint8)[units].masked_fill(mask, 0),
                            rtol=0,
                            atol=0,
                            msg=(
                                f"step={step}, layer={layer}, request={rid}, "
                                f"selected={units.tolist()}, pages={physical.tolist()}"
                            ),
                        )
                _attention(
                    q,
                    manager.get_buffers(layer, "HND"),
                    manager.get_block_scale_buffers(layer, "HND") if nvfp4 else None,
                    topk,
                    table,
                    metadata.msa_seq_lens_cuda,
                    outputs[layer],
                    quant,
                )

        captured = None
        permuted = False
        for step in range(9):
            # Request 11 loses its lease while its first partial block is still growing.
            active = requests[1:] if step in (1, 2) else requests
            active = list(reversed(active)) if step % 2 else list(active)
            for request in active:
                assert manager.try_allocate_generation(request)
            scheduled = SimpleNamespace(context_requests=[], generation_requests=active)
            manager.prepare_resources(scheduled)
            ids = [request.py_request_id for request in active]
            histories = [manager.kv_cache_map[rid].history_length for rid in ids]
            metadata.request_ids = ids
            metadata.seq_lens = torch.ones(len(ids), dtype=torch.int32)
            metadata.prompt_lens = [request.prompt_len for request in active]
            metadata.kv_cache_params = KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=histories
            )
            metadata.prepare()
            # The fixed-width kernel test executes padded queries beyond metadata.num_seqs.
            metadata.msa_seq_lens_cuda[len(ids) :].zero_()
            permuted |= state.batch_rows[: len(active)].tolist() != list(range(len(active)))
            resident_table.fill_(-1)
            for position, request in enumerate(active):
                resident_table[position].copy_(
                    torch.arange(max_blocks, device="cuda", dtype=torch.int32)
                    + requests.index(request) * max_blocks
                )
            torch.testing.assert_close(
                metadata.msa_seq_lens_cuda[: len(ids)],
                torch.tensor([h + 1 for h in histories], device="cuda", dtype=torch.int32),
            )
            topk.fill_(-1)
            for position, history in enumerate(histories):
                for head in range(heads):
                    units = [
                        1
                        + (step * TOPK * heads + head * (TOPK - 2) + i)
                        % max(history // PAGE - 1, 1)
                        for i in range(TOPK - 2)
                    ]
                    # Shared first/tail blocks exercise union deduplication across heads.
                    topk[position, head] = torch.tensor(
                        sorted({0, history // PAGE, *units})
                        + [-1] * (TOPK - len({0, history // PAGE, *units})),
                        device="cuda",
                        dtype=torch.int32,
                    )
                    assert (topk[position, head] >= 0).sum() == min(TOPK, history // PAGE + 1)
            for layer in SPARSE_LAYERS:
                k, v = inputs[layer]
                k.zero_()
                v.zero_()
                for position, (rid, history) in enumerate(zip(ids, histories)):
                    source_k, source_v = source[rid, layer]
                    k[position].copy_(source_k[history])
                    v[position].copy_(source_v[history])
                    data, scales = reference[rid, layer]
                    _write(
                        data,
                        scales,
                        torch.tensor([history], dtype=torch.int32, device="cuda"),
                        source_k[history : history + 1],
                        source_v[history : history + 1],
                        quant,
                    )
            if graph and captured is None:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    decode()
                torch.cuda.current_stream().wait_stream(stream)
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured):
                    decode()
            if captured is None:
                decode()
            else:
                captured.replay()
            torch.cuda.synchronize()
            assert (metadata.msa_sparse_write_slots[len(active) :] == -1).all()
            for layer in SPARSE_LAYERS:
                expected = torch.empty_like(q)
                _attention(
                    q,
                    resident_data[layer],
                    resident_scales[layer],
                    topk,
                    resident_table,
                    metadata.msa_seq_lens_cuda,
                    expected,
                    quant,
                )
                torch.testing.assert_close(
                    outputs[layer],
                    expected,
                    rtol=0,
                    atol=0,
                    msg=lambda error: f"step={step}, layer={layer}, requests={ids}: {error}",
                )
            manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
            torch.cuda.synchronize()
            for rid, history in zip(ids, histories):
                assert manager.kv_cache_map[rid].history_length == history + 1
                for layer in SPARSE_LAYERS:
                    data, scales = reference[rid, layer]
                    page = history // PAGE
                    valid = history % PAGE + 1
                    torch.testing.assert_close(
                        _host_pages(manager, layer, rid)[page][:, :, :valid].view(torch.uint8),
                        data[page, :, :, :valid].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    if nvfp4:
                        host_scales = _host_pages(manager, layer, rid, scales=True)[page]
                        torch.testing.assert_close(
                            host_scales[0, :, :valid],
                            scales[page, 0, :, :valid],
                            rtol=0,
                            atol=0,
                        )

                        def unpack_v_scale(values):
                            return (
                                values.reshape(heads, PAGE // 4, HEAD_DIM // 16, 4)
                                .transpose(2, 3)
                                .reshape(heads, PAGE, HEAD_DIM // 16)
                            )

                        torch.testing.assert_close(
                            unpack_v_scale(host_scales[1])[:, :valid],
                            unpack_v_scale(scales[page, 1])[:, :valid],
                            rtol=0,
                            atol=0,
                        )
        assert permuted
        for request in requests:
            manager.free_resources(request)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "seq_lens,seq_lens_kv,contexts,device,error",
    [
        ([1], [0], 0, "cpu", NotImplementedError),
        ([2], [2], 0, "cpu", NotImplementedError),
        ([1], [1], 1, "cpu", NotImplementedError),
        ([1], [1], 0, "cuda", ValueError),
    ],
)
def test_m3_offload_rejects_unsupported_rows(seq_lens, seq_lens_kv, contexts, device, error):
    metadata = SimpleNamespace(
        sparse_offload_state=SimpleNamespace(prepared=True),
        num_contexts=contexts,
        seq_lens=torch.tensor(seq_lens, device=device),
        seq_lens_kv=torch.tensor(seq_lens_kv, device=device),
    )
    with pytest.raises(error):
        MiniMaxM3MsaSparseAttentionMetadata.prepare(metadata)
    assert not metadata.sparse_offload_state.prepared


def test_m3_offload_rejects_metadata_topk_mismatch():
    manager = _manager(DataType.FP8, 2, 40 * PAGE)
    try:
        metadata = SimpleNamespace(
            kv_cache_manager=manager,
            _msa_params=MiniMaxM3SparseMetadataParams(global_num_kv_heads=1, topk=TOPK),
            mapping=manager.mapping,
            beam_width=1,
            max_total_draft_tokens=0,
            draft_kv_cache_manager=None,
        )
        with pytest.raises(ValueError, match="top-k"):
            MiniMaxM3MsaSparseAttentionMetadata._init_sparse_offload_state(metadata)
    finally:
        manager.shutdown()


def test_m3_selection_reuses_compilation_across_batch_sizes():
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.offload import (
        _selection_kernel,
        prepare_selection,
    )

    selected = torch.empty((16, 2 * TOPK), dtype=torch.int32, device="cuda")
    units = torch.empty(16, dtype=torch.int32, device="cuda")
    lengths = torch.full((16,), 3 * PAGE + 1, dtype=torch.int32, device="cuda")
    compiled = []
    for queries in (1, 2, 3, 7, 16):
        topk = (
            (torch.arange(2 * queries * TOPK, device="cuda", dtype=torch.int32) % 4)
            .reshape(2, queries, TOPK)
            .transpose(0, 1)
        )
        prepare_selection(topk, lengths, selected, units)
        expected = torch.full((queries, 2 * TOPK), -1, dtype=torch.int32, device="cuda")
        expected[:, :4] = torch.arange(4, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(selected[:queries], expected)
        assert (units[:queries] == 4).all()
        compiled.append(len(_selection_kernel.device_caches[torch.cuda.current_device()][0]))
    assert len(set(compiled)) == 1, compiled


def test_m3_sparse_block_table_checks_coverage(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.offload import (
        build_block_table,
    )

    def device(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    fetched = device([[[-1, -1]]])
    state = SimpleNamespace(
        layers={1: SimpleNamespace(buffer_ids=[1], unit_tokens=PAGE, data_buffer_index=0)},
        fetched_indices={1: fetched},
        write_indices={1: device([[7]])},
        batch_rows=device([0]),
        active_request_count=device([1]),
        history_lengths=device([PAGE]),
        out_slots={1: [0]},
        coverage=device([1]),
    )
    checks = []
    # Observe the device flag without poisoning this process's CUDA context.
    monkeypatch.setattr(
        torch, "_assert_async", lambda flag, message: checks.append(int(flag.item()))
    )
    out = device([[-1, -1]])
    selected = device([[0, 1]])
    build_block_table(state, 1, selected, out)
    assert checks == [0]
    torch.testing.assert_close(out, device([[-1, 7]]))
    fetched[0, 0, 0] = 3
    build_block_table(state, 1, selected, out)
    assert checks == [0, 1]
    torch.testing.assert_close(out, device([[3, 7]]))
