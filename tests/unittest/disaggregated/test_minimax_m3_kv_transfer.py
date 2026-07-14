# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniMax M3 heterogeneous-topology KV transfer tests.

This drives the shared threaded NIXL harness (``kv_transfer_harness``) with
MiniMax M3 cache managers and validates both ordinary K/V and the
replicated INDEX_KEY cache. V2 storage coalesces buffers purely by (life
cycle, size), so at TP degrees where K == V == INDEX_KEY bytes per block the
index-K cache shares the K/V pool and the slot interleaves per layer; at
other degrees it gets its own pool. The disagg page-table builder splits each
pool into per-mapper-kind views, so both layouts (and transfers between them)
are exercised by the topology matrix below.
"""

from collections.abc import Sequence

import kv_transfer_harness as transfer_harness
import pytest
import torch

from tensorrt_llm import Mapping
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3KVCacheManagerV2
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool, get_pool_bytes
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128
INDEX_DIM = 128
TOKENS_PER_BLOCK = transfer_harness.TOKENS_PER_BLOCK
SPARSE_LAYERS = [3]


class _FakeKVCache:
    num_blocks = 3

    @staticmethod
    def get_base_page_indices(_pool_id):
        return [4, 5, 6, -1]


class _ShortFakeKVCache:
    num_blocks = 2

    @staticmethod
    def get_base_page_indices(_pool_id):
        return [7, 8, -1, -1]


def test_minimax_cache_indices_support_block_count_limit() -> None:
    manager = object.__new__(MiniMaxM3KVCacheManagerV2)
    manager.kv_cache_map = {7: _FakeKVCache(), 8: _ShortFakeKVCache()}

    assert manager._get_batch_cache_indices_by_pool_id([7]) == [[4, 5, 6, -1]]
    assert manager._get_batch_cache_indices_by_pool_id([7], num_blocks_per_seq=[2]) == [[4, 5]]
    assert manager._get_batch_cache_indices_by_pool_id([7], num_blocks_per_seq=[99]) == [[4, 5, 6]]
    assert manager._get_batch_cache_indices_by_pool_id([7, 8]) == [
        [4, 5, 6, -1],
        [7, 8, -1, -1],
    ]
    assert manager.get_block_ids_per_seq([7]).tolist() == [[4, 5, 6, 0]]
    assert manager.get_block_ids_per_seq([7, 8]).tolist() == [
        [4, 5, 6, 0],
        [7, 8, 0, 0],
    ]


def test_v2_disagg_role_mapper_kind_defaults() -> None:
    manager = object.__new__(KVCacheManagerV2)

    assert manager.get_disagg_role_mapper_kinds() == {
        Role.ALL: MapperKind.INDEXED,
        Role.INDEX_KEY: MapperKind.REPLICATED,
    }


def test_minimax_disagg_role_mapper_kinds() -> None:
    manager = object.__new__(MiniMaxM3KVCacheManagerV2)

    role_mapper_kinds = manager.get_disagg_role_mapper_kinds()

    assert role_mapper_kinds == {
        Role.ALL: MapperKind.NHD,
        Role.INDEX_KEY: MapperKind.REPLICATED,
    }


def test_minimax_disagg_rejects_unmanaged_index_value(monkeypatch) -> None:
    def fake_base_init(self, *args, **kwargs):
        self.is_disagg = kwargs.get("is_disagg", False)
        self.layer_offsets = {layer_id: layer_id for layer_id in range(kwargs["num_layers"])}

    monkeypatch.setattr(KVCacheManagerV2, "__init__", fake_base_init)

    with pytest.raises(ValueError, match="requires disable_index_value=True"):
        MiniMaxM3KVCacheManagerV2(
            num_layers=4,
            sparse_layer_ids=[3],
            disable_index_value_layer_ids=[],
            sparse_index_dim=INDEX_DIM,
            is_disagg=True,
        )


def _create_manager(
    mapping: Mapping, dtype: DataType, sparse_layers: list[int] | None = None
) -> MiniMaxM3KVCacheManagerV2:
    max_num_tokens = 2048
    kv_cache_dtype = {
        DataType.FP8: "fp8",
        DataType.NVFP4: "nvfp4",
    }.get(dtype, "auto")
    return MiniMaxM3KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_num_tokens,
            event_buffer_max_size=0,
            dtype=kv_cache_dtype,
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=transfer_harness.MAX_SEQ_LEN,
        max_batch_size=transfer_harness.MAX_BATCH_SIZE,
        mapping=mapping,
        dtype=dtype,
        vocab_size=transfer_harness.VOCAB_SIZE,
        max_num_tokens=max_num_tokens,
        sparse_layer_ids=sparse_layers if sparse_layers is not None else SPARSE_LAYERS,
        disable_index_value_layer_ids=sparse_layers if sparse_layers is not None else SPARSE_LAYERS,
        sparse_index_dim=INDEX_DIM,
    )


def test_minimax_m3_pool_view_scheme_coalesced_vs_separate() -> None:
    """Pool composition varies with TP; the view scheme must not.

    TP=2 makes kv_heads_per_rank == 1, so K == V == INDEX_KEY bytes per
    block and V2 coalesces all three into one pool whose slot interleaves
    the sparse layer's index-K between K/V regions (non-uniform layer
    stride). TP=1 doubles K/V, so index-K gets its own pool and strides are
    uniform. Both layouts must yield the same two per-class views.
    """
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )
    from tensorrt_llm._torch.disaggregation.resource.utils import get_layer_byte_ranges

    # --- TP=2: coalesced, interleaved slot ---
    manager = _create_manager(
        Mapping(world_size=2, rank=0, tp_size=2, pp_size=1), DataType.BF16, sparse_layers=[1]
    )
    try:
        lg = build_page_table_from_manager(manager).layer_groups[0]
        assert len(lg.pool_views) == 2
        kv_view = next(pv for pv in lg.pool_views if pv.mapper_kind == MapperKind.NHD)
        idx_view = next(pv for pv in lg.pool_views if pv.mapper_kind == MapperKind.REPLICATED)
        assert kv_view.pool_role == frozenset({str(Role.KEY), str(Role.VALUE)})
        assert idx_view.pool_role == frozenset({str(Role.INDEX_KEY)})
        # Coalesced: both views address the same physical pool.
        assert kv_view.pool_idx == idx_view.pool_idx
        starts, bytes_per_layer = get_layer_byte_ranges(kv_view)
        unit = bytes_per_layer // 2  # one K or V buffer per block
        # Slot: L0K L0V L1K L1V L1IDX L2K L2V L3K L3V — the sparse layer's
        # index-K makes the K/V layer stride non-uniform.
        assert starts == {0: 0, 1: 2 * unit, 2: 5 * unit, 3: 7 * unit}
        idx_starts, idx_bytes_per_layer = get_layer_byte_ranges(idx_view)
        assert idx_starts == {1: 4 * unit}
        assert idx_bytes_per_layer == unit
    finally:
        manager.shutdown()

    # --- TP=1: separate pools, uniform strides ---
    manager = _create_manager(
        Mapping(world_size=1, rank=0, tp_size=1, pp_size=1), DataType.BF16, sparse_layers=[1]
    )
    try:
        lg = build_page_table_from_manager(manager).layer_groups[0]
        assert len(lg.pool_views) == 2
        kv_view = next(pv for pv in lg.pool_views if pv.mapper_kind == MapperKind.NHD)
        idx_view = next(pv for pv in lg.pool_views if pv.mapper_kind == MapperKind.REPLICATED)
        assert kv_view.pool_idx != idx_view.pool_idx
        starts, bytes_per_layer = get_layer_byte_ranges(kv_view)
        assert starts == {i: i * bytes_per_layer for i in range(NUM_LAYERS)}
    finally:
        manager.shutdown()


def _create_managers(
    tp: int,
    pp: int,
    enable_dp: bool,
    dtype: DataType = DataType.BF16,
    sparse_layers: list[int] | None = None,
) -> list[MiniMaxM3KVCacheManagerV2]:
    return [
        _create_manager(
            Mapping(
                world_size=tp * pp,
                rank=rank,
                tp_size=tp,
                pp_size=pp,
                enable_attention_dp=enable_dp,
            ),
            dtype,
            sparse_layers,
        )
        for rank in range(tp * pp)
    ]


def _zero_physical_pools(manager: MiniMaxM3KVCacheManagerV2) -> None:
    page_table = KVRegionExtractorV1(manager).page_table
    unique_pools = {}
    for pool_group in page_table.pool_groups:
        for pool in pool_group.pools:
            unique_pools[pool.base_address] = max(
                unique_pools.get(pool.base_address, 0), get_pool_bytes(pool)
            )
    for base_address, size in unique_pools.items():
        tensor = convert_to_torch_tensor(TensorWrapper(base_address, DataType.INT8, [size]))
        tensor.zero_()


def _get_nvfp4_scale_view(
    manager: MiniMaxM3KVCacheManagerV2, layer_idx: int
) -> torch.Tensor | None:
    if manager.dtype != DataType.NVFP4:
        return None
    page_table = KVRegionExtractorV1(manager).page_table
    local_layer_id = manager.layer_offsets[layer_idx]
    scale_roles = frozenset({"key_block_scale", "value_block_scale"})
    for layer_group_id, layer_group in enumerate(page_table.layer_groups):
        for pool_view in getattr(layer_group, "pool_views", ()):
            # Scale buffers land in their own pool (smaller size class), so
            # selecting the pool by role set suffices; every entry for the
            # layer inside that pool is a scale buffer.
            if not (pool_view.pool_role & scale_roles):
                continue
            matching_entries = [
                entry
                for entry in pool_view.buffer_entries
                if int(entry["local_layer_id"]) == local_layer_id
            ]
            if not matching_entries:
                continue
            pool = get_physical_pool(page_table, layer_group_id, pool_view.pool_idx)
            raw_slots = convert_to_torch_tensor(
                TensorWrapper(
                    pool.base_address,
                    DataType.INT8,
                    [pool.num_slots, pool.slot_bytes],
                )
            )
            start = min(int(entry["offset"]) for entry in matching_entries)
            end = max(int(entry["offset"] + entry["size"]) for entry in matching_entries)
            return raw_slots[:, start:end]
    raise AssertionError(f"missing NVFP4 scale view for layer {layer_idx}")


def _fill_position_dependent(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    first_global_head: int,
) -> None:
    """Fill ``[block, role, token, head, dim]`` with exact small integers."""
    block = torch.arange(tensor.shape[0], device=tensor.device)[:, None, None, None, None]
    role = torch.arange(tensor.shape[1], device=tensor.device)[None, :, None, None, None]
    token = torch.arange(tensor.shape[2], device=tensor.device)[None, None, :, None, None]
    head = (first_global_head + torch.arange(tensor.shape[3], device=tensor.device))[
        None, None, None, :, None
    ]
    dim = torch.arange(tensor.shape[4], device=tensor.device)[None, None, None, None, :]
    values = (layer_idx * 17 + block * 11 + role * 13 + token * 3 + head * 19 + dim) % 97
    tensor.copy_(values.to(tensor.dtype))


def _as_nvfp4_scale_tensor(
    manager: MiniMaxM3KVCacheManagerV2,
    layer_idx: int,
) -> torch.Tensor | None:
    scale_view = _get_nvfp4_scale_view(manager, layer_idx)
    if scale_view is None:
        return None
    local_layer_id = manager.layer_offsets[layer_idx]
    local_heads = manager.num_kv_heads_per_layer[local_layer_id]
    bytes_per_token_head = HEAD_DIM // 16
    return scale_view.view(
        scale_view.shape[0],
        2,
        TOKENS_PER_BLOCK,
        local_heads,
        bytes_per_token_head,
    )


def _valid_indices(
    manager: MiniMaxM3KVCacheManagerV2,
    request_id: int,
    layer_idx: int,
) -> list[int]:
    return [
        index for index in manager.get_batch_cache_indices([request_id], layer_idx)[0] if index >= 0
    ]


def _first_global_head(manager: MiniMaxM3KVCacheManagerV2) -> int:
    """Map a TP rank to its first logical head, including duplicated heads."""
    if manager.mapping.enable_attention_dp:
        return 0
    return manager.mapping.tp_rank * NUM_KV_HEADS // manager.mapping.tp_size


def _find_ctx_source(
    ctx_managers: list[MiniMaxM3KVCacheManagerV2],
    *,
    ctx_tp: int,
    ctx_enable_dp: bool,
    request_idx: int,
    layer_idx: int,
    global_head: int,
) -> tuple[MiniMaxM3KVCacheManagerV2, int]:
    """Return the context manager/local head that owns one logical KV head."""
    for manager in ctx_managers:
        if layer_idx not in manager.pp_layers:
            continue
        tp_rank = manager.mapping.tp_rank
        local_layer_id = manager.layer_offsets[layer_idx]
        local_heads = manager.num_kv_heads_per_layer[local_layer_id]
        if ctx_enable_dp:
            if tp_rank == request_idx % ctx_tp:
                return manager, global_head
            continue
        first_global_head = _first_global_head(manager)
        if first_global_head <= global_head < first_global_head + local_heads:
            return manager, global_head - first_global_head
    raise AssertionError(
        f"missing context source for request={request_idx}, layer={layer_idx}, "
        f"global_head={global_head}"
    )


def _initialize_cache(
    managers: Sequence[MiniMaxM3KVCacheManagerV2],
    _tp: int,
    seed_base: int = 0,
    fill_random: bool = True,
) -> None:
    del seed_base
    for manager in managers:
        _zero_physical_pools(manager)
        if not fill_random:
            continue

        for layer_idx in manager.pp_layers:
            kv = manager.get_buffers(layer_idx, kv_layout="NHD")
            first_global_head = _first_global_head(manager)
            _fill_position_dependent(
                kv,
                layer_idx=layer_idx,
                first_global_head=first_global_head,
            )

            index_key = manager.get_index_k_buffer(layer_idx)
            if index_key is not None:
                index_tensor = index_key.unsqueeze(1)
                _fill_position_dependent(
                    index_tensor,
                    layer_idx=layer_idx,
                    first_global_head=0,
                )

            scale_tensor = _as_nvfp4_scale_tensor(manager, layer_idx)
            if scale_tensor is not None:
                _fill_position_dependent(
                    scale_tensor,
                    layer_idx=layer_idx,
                    first_global_head=first_global_head,
                )


def _verify_cache(
    request_lengths: list[int],
    ctx_managers: Sequence[MiniMaxM3KVCacheManagerV2],
    gen_managers: Sequence[MiniMaxM3KVCacheManagerV2],
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    ctx_request_ids: list[int],
    gen_request_ids: list[int],
) -> None:
    del ctx_pp

    for req_idx, _request_length in enumerate(request_lengths):
        gen_request_id = gen_request_ids[req_idx]
        for gen_rank, manager in enumerate(gen_managers):
            tp_rank = gen_rank % gen_tp
            if gen_enable_dp and req_idx % gen_tp != tp_rank:
                continue

            for layer_idx in manager.pp_layers:
                gen_indices = _valid_indices(manager, gen_request_id, layer_idx)
                assert gen_indices

                gen_kv = manager.get_buffers(layer_idx, kv_layout="NHD")[gen_indices]
                local_heads = gen_kv.shape[3]
                first_global_head = _first_global_head(manager)
                for kv_idx in range(2):
                    for local_head in range(local_heads):
                        global_head = first_global_head + local_head
                        ctx_manager, ctx_local_head = _find_ctx_source(
                            ctx_managers,
                            ctx_tp=ctx_tp,
                            ctx_enable_dp=ctx_enable_dp,
                            request_idx=req_idx,
                            layer_idx=layer_idx,
                            global_head=global_head,
                        )
                        ctx_indices = _valid_indices(
                            ctx_manager, ctx_request_ids[req_idx], layer_idx
                        )
                        ctx_kv = ctx_manager.get_buffers(layer_idx, kv_layout="NHD")
                        torch.testing.assert_close(
                            gen_kv[:, kv_idx, :, local_head, :],
                            ctx_kv[ctx_indices, kv_idx, :, ctx_local_head, :],
                            rtol=0,
                            atol=0,
                        )

                index_key = manager.get_index_k_buffer(layer_idx)
                if index_key is not None:
                    ctx_manager, _ = _find_ctx_source(
                        ctx_managers,
                        ctx_tp=ctx_tp,
                        ctx_enable_dp=ctx_enable_dp,
                        request_idx=req_idx,
                        layer_idx=layer_idx,
                        global_head=0,
                    )
                    ctx_indices = _valid_indices(ctx_manager, ctx_request_ids[req_idx], layer_idx)
                    ctx_index_key = ctx_manager.get_index_k_buffer(layer_idx)
                    assert ctx_index_key is not None
                    torch.testing.assert_close(
                        index_key[gen_indices],
                        ctx_index_key[ctx_indices],
                        rtol=0,
                        atol=0,
                    )

                gen_scales = _as_nvfp4_scale_tensor(manager, layer_idx)
                if gen_scales is not None:
                    for local_head in range(local_heads):
                        global_head = first_global_head + local_head
                        ctx_manager, ctx_local_head = _find_ctx_source(
                            ctx_managers,
                            ctx_tp=ctx_tp,
                            ctx_enable_dp=ctx_enable_dp,
                            request_idx=req_idx,
                            layer_idx=layer_idx,
                            global_head=global_head,
                        )
                        ctx_indices = _valid_indices(
                            ctx_manager, ctx_request_ids[req_idx], layer_idx
                        )
                        ctx_scales = _as_nvfp4_scale_tensor(ctx_manager, layer_idx)
                        assert ctx_scales is not None
                        torch.testing.assert_close(
                            gen_scales[gen_indices, :, :, local_head, :],
                            ctx_scales[ctx_indices, :, :, ctx_local_head, :],
                            rtol=0,
                            atol=0,
                        )


# Production is expected to use TEP/DEP context and DEP generation. Bias the
# committed matrix toward head-matched layouts while retaining representative
# head-mismatch, degree-8 duplication, fan-in/fan-out, and PP2 coverage.
HEAD_MATCHED_BF16_TOPOLOGIES = [
    (1, 1, True, 1, 1, True, "dep1_to_dep1"),
    (2, 1, True, 2, 1, True, "dep2_to_dep2"),
    (4, 1, True, 4, 1, True, "dep4_to_dep4"),
    (8, 1, True, 8, 1, True, "dep8_to_dep8"),
    (1, 1, True, 8, 1, True, "dep1_to_dep8"),
    (8, 1, True, 1, 1, True, "dep8_to_dep1"),
    (1, 1, False, 4, 1, True, "tep1_to_dep4"),
    (1, 1, False, 8, 1, True, "tep1_to_dep8"),
    (1, 2, False, 2, 1, True, "tp1pp2_to_dep2"),
]

HEAD_MISMATCHED_BF16_TOPOLOGIES = [
    (2, 1, False, 2, 1, True, "tep2_to_dep2"),
    (4, 1, False, 4, 1, True, "tep4_to_dep4"),
    (8, 1, False, 8, 1, True, "tep8_to_dep8"),
    (8, 1, False, 1, 1, True, "tep8_to_dep1"),
]

BF16_TOPOLOGIES = HEAD_MATCHED_BF16_TOPOLOGIES + HEAD_MISMATCHED_BF16_TOPOLOGIES

# Smaller quantized-cache sets cover packed K/V geometry across the same
# production directions without repeating the BF16 topology matrix. NVFP4
# additionally exercises its block-scale pools.
QUANTIZED_TOPOLOGIES = [
    (4, 1, True, 4, 1, True, "dep4_to_dep4"),
    (8, 1, True, 1, 1, True, "dep8_to_dep1"),
    (1, 1, False, 8, 1, True, "tep1_to_dep8"),
    (4, 1, False, 4, 1, True, "tep4_to_dep4"),
    (8, 1, False, 1, 1, True, "tep8_to_dep1"),
]

CACHE_CASES = [(*topology[:6], DataType.BF16, topology[6]) for topology in BF16_TOPOLOGIES] + [
    (*topology[:6], dtype, f"{prefix}_{topology[6]}")
    for dtype, prefix in ((DataType.FP8, "fp8"), (DataType.NVFP4, "nvfp4"))
    for topology in QUANTIZED_TOPOLOGIES
]


@pytest.mark.cuda
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp,cache_dtype",
    [case[:7] for case in CACHE_CASES],
    ids=[case[7] for case in CACHE_CASES],
)
@pytest.mark.parametrize(
    "update_before_transfer",
    [True, False],
    ids=["update_before", "update_after"],
)
def test_minimax_m3_kv_transfer(
    ctx_tp,
    ctx_pp,
    ctx_enable_dp,
    gen_tp,
    gen_pp,
    gen_enable_dp,
    cache_dtype,
    update_before_transfer,
) -> None:
    transfer_harness.run_kv_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        update_before_transfer=update_before_transfer,
        manager_factory=lambda tp, pp, enable_dp: _create_managers(tp, pp, enable_dp, cache_dtype),
        init_fn=_initialize_cache,
        verify_fn=_verify_cache,
    )


# Multiple sparse layers spread across PP stages: the replicated index-key
# pool overlaps only partially between peers, exercising the layer-strided
# ReplicatedMapper offsets (a single-sparse-layer model always fully
# overlaps and would degenerate to a whole-slot copy).
MULTI_SPARSE_LAYERS = [1, 2, 3]


@pytest.mark.cuda
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp",
    [
        (1, 1, True, 1, 2, False),  # ctx full index pool -> per-PP-stage subsets
        (1, 2, False, 1, 1, True),  # per-PP-stage subsets -> full index pool
        (2, 1, False, 1, 2, False),  # TEP fan-in + PP subset on the same transfer
    ],
    ids=["dep1_to_tp1pp2", "tp1pp2_to_dep1", "tep2_to_tp1pp2"],
)
def test_minimax_m3_multi_sparse_layer_pp_transfer(
    ctx_tp,
    ctx_pp,
    ctx_enable_dp,
    gen_tp,
    gen_pp,
    gen_enable_dp,
) -> None:
    transfer_harness.run_kv_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        update_before_transfer=True,
        manager_factory=lambda tp, pp, enable_dp: _create_managers(
            tp, pp, enable_dp, DataType.BF16, sparse_layers=MULTI_SPARSE_LAYERS
        ),
        init_fn=_initialize_cache,
        verify_fn=_verify_cache,
    )
