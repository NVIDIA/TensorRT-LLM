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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(), reason="M3 offload needs an SM100-family GPU"
)

PAGE = 128
HEAD_DIM = 128
TOPK = 16
BATCH = 4
SPARSE_LAYERS = (1, 3)


def _manager(
    dtype: DataType, heads: int, max_seq_len: int, *, offload: bool = True
) -> MiniMaxM3KVCacheManagerV2:
    return MiniMaxM3KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, max_tokens=BATCH * max_seq_len, host_cache_size=0
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
def test_m3_resident_and_offloaded_transfer_views_match(dtype, heads):
    """Native transfer maps every dense, sparse and index pool in both directions."""
    resident = _manager(dtype, heads, 40 * PAGE, offload=False)
    offloaded = _manager(dtype, heads, 40 * PAGE)
    try:
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
    dtype, heads, graph
):
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
