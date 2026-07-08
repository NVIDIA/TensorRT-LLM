# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax M3 heterogeneous-topology KV transfer tests.

This reuses the threaded NIXL harness from the DeepSeek-V4 V2 transfer test,
but supplies MiniMax M3 cache managers and validates both ordinary K/V and the
replicated INDEX_KEY cache. The TP=2 TEP layout coalesces K/V/INDEX_KEY in one
physical pool, while attention-DP keeps INDEX_KEY in a separate pool.
"""

from collections.abc import Sequence

import pytest
import test_deepseek_v4_kv_transfer as transfer_harness
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

    assert manager.get_disagg_role_mapper_kinds() == {Role.ALL: MapperKind.INDEXED}


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


def _create_manager(mapping: Mapping, dtype: DataType) -> MiniMaxM3KVCacheManagerV2:
    max_num_tokens = 2048
    return MiniMaxM3KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_num_tokens,
            event_buffer_max_size=0,
            dtype="nvfp4" if dtype == DataType.NVFP4 else "auto",
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
        sparse_layer_ids=SPARSE_LAYERS,
        disable_index_value_layer_ids=SPARSE_LAYERS,
        sparse_index_dim=INDEX_DIM,
    )


def _create_managers(
    tp: int,
    pp: int,
    enable_dp: bool,
    _unused_layout: list[int],
    dtype: DataType = DataType.BF16,
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
            if pool_view.pool_role != scale_roles:
                continue
            if local_layer_id not in pool_view.buffer_entries["local_layer_id"]:
                continue
            pool = get_physical_pool(page_table, layer_group_id, pool_view.pool_idx)
            raw_slots = convert_to_torch_tensor(
                TensorWrapper(
                    pool.base_address,
                    DataType.INT8,
                    [pool.num_slots, pool.slot_bytes],
                )
            )
            end = pool_view.byte_offset + pool_view.bytes_per_region
            return raw_slots[:, pool_view.byte_offset : end]
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
    compress_ratios: list[int],
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
    del compress_ratios, ctx_pp

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

# A smaller NVFP4 set covers packed K/V and scale-view geometry across the
# same production directions without repeating the BF16 topology matrix.
NVFP4_TOPOLOGIES = [
    (4, 1, True, 4, 1, True, "nvfp4_dep4_to_dep4"),
    (8, 1, True, 1, 1, True, "nvfp4_dep8_to_dep1"),
    (1, 1, False, 8, 1, True, "nvfp4_tep1_to_dep8"),
    (4, 1, False, 4, 1, True, "nvfp4_tep4_to_dep4"),
    (8, 1, False, 1, 1, True, "nvfp4_tep8_to_dep1"),
]

CACHE_CASES = [(*topology[:6], DataType.BF16, topology[6]) for topology in BF16_TOPOLOGIES] + [
    (*topology[:6], DataType.NVFP4, topology[6]) for topology in NVFP4_TOPOLOGIES
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
    transfer_harness.run_deepseek_v4_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        compress_ratios=[1] * NUM_LAYERS,
        update_before_transfer=update_before_transfer,
        manager_factory=lambda tp, pp, enable_dp, layout: _create_managers(
            tp, pp, enable_dp, layout, cache_dtype
        ),
        init_fn=_initialize_cache,
        verify_fn=_verify_cache,
    )
