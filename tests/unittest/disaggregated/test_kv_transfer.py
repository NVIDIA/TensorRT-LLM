"""Test KV Transfer with KVCacheManager (V1) and KVCacheManagerV2 (V2)."""

import random
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
import tensorrt_llm.tensorrt_llm_transfer_agent_binding  # TODO: remove it.  # noqa: F401
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    LayerRange,
    SessionStatus,
    TokenRange,
)
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker, TransferWorkerConfig
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, KVCacheManagerV2
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings import LayerType as LayerTypeCpp
from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger


@dataclass
class KvCacheConfigV2:
    """KvCacheConfig wrapper with max_util_for_resume for KVCacheManagerV2."""

    max_tokens: Optional[int] = None
    enable_block_reuse: bool = False
    max_attention_window: Optional[List[int]] = None
    sink_token_length: Optional[int] = None
    free_gpu_memory_fraction: Optional[float] = None
    host_cache_size: Optional[int] = None
    onboard_blocks: bool = True
    cross_kv_cache_fraction: Optional[float] = None
    secondary_offload_min_priority: Optional[int] = None
    event_buffer_max_size: int = 0

    max_gpu_total_bytes: Optional[int] = None
    enable_partial_reuse: bool = False
    copy_on_partial_reuse: bool = False
    dtype: str = "auto"
    # V2 specific field
    max_util_for_resume: float = 0.95


def test_token_range_valid():
    tr = TokenRange(start=0, end=10)
    assert tr.start == 0
    assert tr.end == 10


def test_token_range_invalid_negative():
    with pytest.raises(ValueError, match="non-negative"):
        TokenRange(start=-1, end=5)
    with pytest.raises(ValueError, match="non-negative"):
        TokenRange(start=0, end=-1)


def test_token_range_invalid_start_ge_end():
    with pytest.raises(ValueError, match="Invalid range"):
        TokenRange(start=5, end=5)
    with pytest.raises(ValueError, match="Invalid range"):
        TokenRange(start=10, end=3)


def test_layer_range_valid():
    lr = LayerRange(start=0, end=32)
    assert lr.start == 0
    assert lr.end == 32


def test_layer_range_invalid_negative():
    with pytest.raises(ValueError, match="non-negative"):
        LayerRange(start=-1, end=5)
    with pytest.raises(ValueError, match="non-negative"):
        LayerRange(start=0, end=-1)


def test_layer_range_invalid_start_ge_end():
    with pytest.raises(ValueError, match="Invalid range"):
        LayerRange(start=5, end=5)
    with pytest.raises(ValueError, match="Invalid range"):
        LayerRange(start=10, end=3)


def test_kv_slice_construction():
    tr = TokenRange(0, 128)
    lr = LayerRange(0, 32)
    s = KVSlice(
        token_range=tr,
        layer_range=lr,
        block_ids_per_layer_groups=[[1, 2, 3]],
        is_last_slice=True,
    )
    assert s.token_range == tr
    assert s.layer_range == lr
    assert s.block_ids_per_layer_groups == [[1, 2, 3]]
    assert s.is_last_slice is True

    # Test defaults
    s2 = KVSlice()
    assert s2.token_range is None
    assert s2.layer_range is None
    assert s2.block_ids_per_layer_groups == []
    assert s2.is_last_slice is False


def test_session_status_enum():
    expected = [
        "INIT",
        "READY",
        "TRANSFERRING",
        "KV_TRANSFERRED",
        "FULLY_TRANSFERRED",
        "ERROR",
    ]
    for name in expected:
        assert hasattr(SessionStatus, name)
        assert SessionStatus[name].value == name
    assert len(SessionStatus) == 6


# ---------------------------------------------------------------------------
# Chunked KV slice creation tests
# ---------------------------------------------------------------------------


def _chunk_block_ids(all_block_ids, chunk_size_blocks):
    """Standalone helper that replicates _create_kv_slices chunking logic."""
    import math

    if chunk_size_blocks is None:
        return [KVSlice(is_last_slice=True, block_ids_per_layer_groups=all_block_ids)]

    max_blocks = max((len(ids) for ids in all_block_ids), default=0)
    if max_blocks == 0:
        return [KVSlice(is_last_slice=True, block_ids_per_layer_groups=all_block_ids)]

    num_chunks = math.ceil(max_blocks / chunk_size_blocks)
    slices = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size_blocks
        end = start + chunk_size_blocks
        is_last = chunk_idx == num_chunks - 1
        chunk_block_ids = [ids[start:end] for ids in all_block_ids]
        slices.append(KVSlice(is_last_slice=is_last, block_ids_per_layer_groups=chunk_block_ids))
    return slices


@pytest.mark.parametrize(
    "all_block_ids,chunk_size,expected_num_slices",
    [
        ([[0, 1, 2, 3, 4, 5, 6, 7]], None, 1),
        ([[0, 1, 2, 3, 4, 5, 6, 7]], 4, 2),
        ([list(range(10))], 4, 3),
        ([[], []], 4, 1),
        ([[0, 1, 2]], 64, 1),
    ],
    ids=["no_chunking", "even_split", "uneven_split", "empty_blocks", "chunk_larger_than_total"],
)
def test_create_kv_slices_basic(all_block_ids, chunk_size, expected_num_slices):
    """Chunking produces the expected number of slices."""
    slices = _chunk_block_ids(all_block_ids, chunk_size_blocks=chunk_size)
    assert len(slices) == expected_num_slices
    assert slices[-1].is_last_slice is True
    if expected_num_slices > 1:
        for s in slices[:-1]:
            assert s.is_last_slice is False


def test_create_kv_slices_integrity_check():
    """Reassembled block IDs from all slices must match the original."""
    all_block_ids = [list(range(17)), list(range(5))]
    slices = _chunk_block_ids(all_block_ids, chunk_size_blocks=4)
    for lg_idx, original in enumerate(all_block_ids):
        reassembled = []
        for s in slices:
            reassembled.extend(s.block_ids_per_layer_groups[lg_idx])
        assert reassembled == original


def test_create_kv_slices_multiple_layer_groups():
    """Different layer groups with different block counts produce correct chunking."""
    all_block_ids = [list(range(8)), list(range(3))]
    slices = _chunk_block_ids(all_block_ids, chunk_size_blocks=4)
    assert len(slices) == 2
    assert slices[0].block_ids_per_layer_groups[0] == [0, 1, 2, 3]
    assert slices[1].block_ids_per_layer_groups[0] == [4, 5, 6, 7]
    assert slices[0].block_ids_per_layer_groups[1] == [0, 1, 2]
    assert slices[1].block_ids_per_layer_groups[1] == []


def create_transfer_worker_setup(
    ctx_tp: int,
    ctx_pp: int,
    ctx_enable_dp: bool,
    gen_tp: int,
    gen_pp: int,
    gen_enable_dp: bool,
    is_mla: bool = False,
    use_v2: bool = False,
    max_attention_window_vec: Optional[List[int]] = None,
):
    """Helper function to set up transfer workers for testing.

    Args:
        use_v2: If True, use KVCacheManagerV2. Otherwise use KVCacheManager (C++ bindings).
        max_attention_window_vec: List of window sizes, e.g. [max_seq_len] or [max_seq_len, small_window].
    """
    ctx_mappings = []
    for i in range(ctx_pp):
        for j in range(ctx_tp):
            ctx_mappings.append(
                Mapping(
                    world_size=ctx_tp * ctx_pp,
                    rank=i * ctx_tp + j,
                    tp_size=ctx_tp,
                    pp_size=ctx_pp,
                    enable_attention_dp=ctx_enable_dp,
                )
            )
    gen_mappings = []
    for i in range(gen_pp):
        for j in range(gen_tp):
            gen_mappings.append(
                Mapping(
                    world_size=gen_tp * gen_pp,
                    rank=i * gen_tp + j,
                    tp_size=gen_tp,
                    pp_size=gen_pp,
                    enable_attention_dp=gen_enable_dp,
                )
            )

    ctx_instance_num = ctx_tp * ctx_pp
    gen_instance_num = gen_tp * gen_pp
    num_layers = 4
    head_dim = 128
    num_kv_heads = 4 if not is_mla else 1
    tokens_per_block = 8
    max_seq_len = 1024
    max_batch_size = 4
    dtype = DataType.FLOAT
    vocab_size = 32000  # V2 requires vocab_size

    # Default max_attention_window_vec to [max_seq_len] if not specified
    if max_attention_window_vec is None:
        max_attention_window_vec = [max_seq_len]
    ctx_transfer_workers = []
    ctx_kv_cache_managers = []
    device_id = 0
    ctx_instance_name = "ctx_instance"
    gen_instance_name = "gen_instance"

    request_len = 16

    for i in range(ctx_instance_num):
        cache_type = (
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
            if not is_mla
            else tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY
        )

        if use_v2:
            ctx_kv_cache_manager = KVCacheManagerV2(
                KvCacheConfigV2(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=max_attention_window_vec,
                ),
                cache_type,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=ctx_mappings[i],
                dtype=dtype,
                vocab_size=vocab_size,
            )
            # V2: Initialize pools using page_table
            random_seed = 0 if is_mla else None
            page_table = KVRegionExtractorV1(ctx_kv_cache_manager).page_table

            # Collect unique pools (deduplicate by base_address)
            unique_pools = {}  # base_address -> pool_bytes
            for pg in page_table.pool_groups:
                for pool_desc in pg.pools:
                    key = pool_desc.base_address
                    pool_bytes = pool_desc.slot_bytes * pool_desc.num_slots
                    if key not in unique_pools or pool_bytes > unique_pools[key]:
                        unique_pools[key] = pool_bytes

            # Initialize each unique pool with random data
            element_bytes = get_size_in_bytes(1, ctx_kv_cache_manager.dtype)
            for pool_base_ptr, pool_size in unique_pools.items():
                pool_size_elements = pool_size // element_bytes
                pool_tensor = convert_to_torch_tensor(
                    TensorWrapper(pool_base_ptr, ctx_kv_cache_manager.dtype, [pool_size_elements])
                )
                if random_seed is not None:
                    generator = torch.Generator(device=pool_tensor.device).manual_seed(random_seed)
                else:
                    generator = None
                random_values = torch.rand(
                    pool_tensor.shape,
                    dtype=pool_tensor.dtype,
                    device=pool_tensor.device,
                    generator=generator,
                )
                pool_tensor.copy_(random_values)
        else:
            # Construct model_config for VSWA (Variable Sliding Window Attention)
            is_vswa = max_attention_window_vec and len(set(max_attention_window_vec)) > 1
            model_config = None
            if is_vswa:
                model_config = ModelConfigCpp(
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                    num_attention_layers=num_layers,
                    num_rnn_layers=0,
                    num_heads=num_kv_heads,
                    hidden_size=num_kv_heads * head_dim,
                    data_type=dtype,
                )
                model_config.layer_types = [LayerTypeCpp.ATTENTION] * num_layers
                model_config.set_num_kv_heads(num_kv_heads)
                model_config.size_per_head = head_dim
                model_config.tokens_per_block = tokens_per_block

            if is_vswa:
                kv_cache_cfg = KvCacheConfig(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=max_attention_window_vec,
                )
            else:
                kv_cache_cfg = trtllm.KvCacheConfig(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=max_attention_window_vec,
                )
            ctx_kv_cache_manager = KVCacheManager(
                kv_cache_cfg,
                cache_type,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=ctx_mappings[i],
                dtype=dtype,
                model_config=model_config,
            )
            # V1: Initialize pool using get_unique_primary_pool
            random_seed = 0 if is_mla else None
            pool_tensor = ctx_kv_cache_manager.get_unique_primary_pool()
            if random_seed is not None:
                generator = torch.Generator(device=pool_tensor.device).manual_seed(random_seed)
            else:
                generator = None
            random_values = torch.rand(
                pool_tensor.shape,
                dtype=pool_tensor.dtype,
                device=pool_tensor.device,
                generator=generator,
            )
            pool_tensor.copy_(random_values)

        ctx_kv_cache_managers.append(ctx_kv_cache_manager)
        ctx_transfer_workers.append(
            TransferWorker(
                TransferWorkerConfig(
                    kv_cache_manager=ctx_kv_cache_manager,
                    device_id=device_id,
                    instance_name=ctx_instance_name,
                    max_concurrent_sessions=max_batch_size * 2,
                    max_draft_len=4,
                )
            )
        )

    ctx_info_endpoint = ctx_transfer_workers[0]._rank_info_server.endpoint
    ctx_endpoints = [
        ctx_transfer_worker._sender.endpoint for ctx_transfer_worker in ctx_transfer_workers
    ]
    ctx_layer_num_per_pp = []
    for pp_rank in range(ctx_pp):
        ctx_layer_num_per_pp.append(len(ctx_kv_cache_managers[pp_rank * ctx_tp].pp_layers))

    for ctx_transfer_worker in ctx_transfer_workers:
        ctx_transfer_worker.populate_instance_and_rank_info(
            endpoints=ctx_endpoints, layer_num_per_pp=ctx_layer_num_per_pp
        )

    gen_transfer_workers = []
    gen_kv_cache_managers = []
    for i in range(gen_instance_num):
        cache_type = (
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
            if not is_mla
            else tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY
        )

        if use_v2:
            gen_kv_cache_manager = KVCacheManagerV2(
                KvCacheConfigV2(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=max_attention_window_vec,
                ),
                cache_type,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=gen_mappings[i],
                dtype=dtype,
                vocab_size=vocab_size,
            )
            # Initialize gen pool to zeros to verify transfer correctness
            gen_page_table = KVRegionExtractorV1(gen_kv_cache_manager).page_table
            gen_unique_pools = {}
            for pg in gen_page_table.pool_groups:
                for pool_desc in pg.pools:
                    key = pool_desc.base_address
                    pool_bytes = pool_desc.slot_bytes * pool_desc.num_slots
                    if key not in gen_unique_pools or pool_bytes > gen_unique_pools[key]:
                        gen_unique_pools[key] = pool_bytes
            gen_element_bytes = get_size_in_bytes(1, gen_kv_cache_manager.dtype)
            for pool_base_ptr, pool_size in gen_unique_pools.items():
                pool_size_elements = pool_size // gen_element_bytes
                pool_tensor = convert_to_torch_tensor(
                    TensorWrapper(pool_base_ptr, gen_kv_cache_manager.dtype, [pool_size_elements])
                )
                pool_tensor.zero_()
        else:
            # Construct model_config for VSWA
            is_vswa = max_attention_window_vec and len(set(max_attention_window_vec)) > 1
            model_config = None
            if is_vswa:
                model_config = ModelConfigCpp(
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                    num_attention_layers=num_layers,
                    num_rnn_layers=0,
                    num_heads=num_kv_heads,
                    hidden_size=num_kv_heads * head_dim,
                    data_type=dtype,
                )
                model_config.layer_types = [LayerTypeCpp.ATTENTION] * num_layers
                model_config.set_num_kv_heads(num_kv_heads)
                model_config.size_per_head = head_dim
                model_config.tokens_per_block = tokens_per_block

            if is_vswa:
                kv_cache_cfg = KvCacheConfig(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=max_attention_window_vec,
                )
            else:
                kv_cache_cfg = trtllm.KvCacheConfig(
                    max_tokens=2048,
                    enable_block_reuse=False,
                    max_attention_window=max_attention_window_vec,
                )
            gen_kv_cache_manager = KVCacheManager(
                kv_cache_cfg,
                cache_type,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=gen_mappings[i],
                dtype=dtype,
                model_config=model_config,
            )
            # Initialize gen pool to zeros
            gen_pool_tensor = gen_kv_cache_manager.get_unique_primary_pool()
            gen_pool_tensor.zero_()
        gen_kv_cache_managers.append(gen_kv_cache_manager)
        gen_transfer_workers.append(
            TransferWorker(
                TransferWorkerConfig(
                    kv_cache_manager=gen_kv_cache_manager,
                    device_id=device_id,
                    instance_name=gen_instance_name,
                    max_concurrent_sessions=max_batch_size * 2,
                    max_draft_len=4,
                )
            )
        )
    _ = gen_transfer_workers[0]._rank_info_server.endpoint  # noqa: F841
    gen_endpoints = [
        gen_transfer_worker._sender.endpoint for gen_transfer_worker in gen_transfer_workers
    ]
    gen_layer_num_per_pp = []
    for pp_rank in range(gen_pp):
        gen_layer_num_per_pp.append(len(gen_kv_cache_managers[pp_rank * gen_tp].pp_layers))
    for gen_transfer_worker in gen_transfer_workers:
        gen_transfer_worker.populate_instance_and_rank_info(
            endpoints=gen_endpoints, layer_num_per_pp=gen_layer_num_per_pp
        )

    return {
        "ctx_transfer_workers": ctx_transfer_workers,
        "ctx_kv_cache_managers": ctx_kv_cache_managers,
        "gen_transfer_workers": gen_transfer_workers,
        "gen_kv_cache_managers": gen_kv_cache_managers,
        "ctx_info_endpoint": ctx_info_endpoint,
        "ctx_layer_num_per_pp": ctx_layer_num_per_pp,
        "gen_layer_num_per_pp": gen_layer_num_per_pp,
        "ctx_tp": ctx_tp,
        "ctx_pp": ctx_pp,
        "ctx_enable_dp": ctx_enable_dp,
        "gen_tp": gen_tp,
        "gen_pp": gen_pp,
        "gen_enable_dp": gen_enable_dp,
        "is_mla": is_mla,
        "use_v2": use_v2,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "tokens_per_block": tokens_per_block,
        "request_len": request_len,
        "max_attention_window_vec": max_attention_window_vec,
    }


def get_block_data(
    kv_cache_manager,
    block_ids: List[int],
    layer_group_id: int,
    use_v2: bool,
    request_id: Optional[int] = None,
) -> torch.Tensor:
    """Unified block data retrieval for both V1 and V2 KVCacheManager."""
    if use_v2:
        layer_grouping = kv_cache_manager.impl.layer_grouping
        local_layer_indices = layer_grouping[layer_group_id]

        all_layer_data = []
        for local_layer_idx in local_layer_indices:
            global_layer_idx = kv_cache_manager.pp_layers[local_layer_idx]
            all_buffer_indices = kv_cache_manager.get_batch_cache_indices(
                [request_id], global_layer_idx
            )[0]
            valid_buffer_indices = [idx for idx in all_buffer_indices if idx >= 0]
            num_valid = len(valid_buffer_indices)
            num_requested = len(block_ids)
            if num_requested < num_valid:
                selected_indices = valid_buffer_indices[-num_requested:]
            else:
                selected_indices = valid_buffer_indices

            layer_buffer = kv_cache_manager.get_buffers(layer_idx=global_layer_idx, kv_layout="HND")
            layer_data = layer_buffer[selected_indices]
            layer_data = layer_data.reshape(layer_data.shape[0], layer_data.shape[1], -1)
            all_layer_data.append(layer_data)

        result = torch.stack(all_layer_data, dim=0).permute(1, 0, 2, 3)
        return result
    else:
        pool_tensor = kv_cache_manager.get_unique_primary_pool()
        block_datas = pool_tensor[block_ids]
        return block_datas


def get_block_ids_per_layer_groups(
    kv_cache_manager, transfer_worker, request_id: int, use_v2: bool, tokens_per_block: int
) -> List[List[int]]:
    """Get block_ids for each layer group with window_size filtering."""
    page_table = transfer_worker._rank_info.page_table
    block_ids_per_layer_groups: List[List[int]] = []

    for group_id, group_meta in enumerate(page_table.layer_groups):
        if use_v2:
            block_ids = list(
                kv_cache_manager.kv_cache_map[request_id].get_aggregated_page_indices(
                    group_id, valid_only=True
                )
            )
        else:
            first_global_layer_id = group_meta.local_layers[0].global_layer_id
            block_ids = kv_cache_manager.get_batch_cache_indices(
                [request_id], first_global_layer_id
            )[0]

        # Filter by window_size if request_len > window_size
        window_size = group_meta.sliding_window_size
        if window_size is not None:
            max_blocks_in_window = window_size // tokens_per_block + 1
            if len(block_ids) > max_blocks_in_window:
                block_ids = block_ids[-max_blocks_in_window:]

        block_ids_per_layer_groups.append(np.asarray(block_ids, dtype=np.int64))

    return block_ids_per_layer_groups


def add_and_verify_request(
    setup, ctx_request_id, gen_request_id, request_len, send_first: bool = True
):
    """Helper function to add and verify a request transfer."""
    ctx_transfer_workers = setup["ctx_transfer_workers"]
    ctx_kv_cache_managers = setup["ctx_kv_cache_managers"]
    gen_transfer_workers = setup["gen_transfer_workers"]
    gen_kv_cache_managers = setup["gen_kv_cache_managers"]
    ctx_info_endpoint = setup["ctx_info_endpoint"]
    ctx_tp = setup["ctx_tp"]
    ctx_pp = setup["ctx_pp"]
    ctx_enable_dp = setup["ctx_enable_dp"]
    gen_tp = setup["gen_tp"]
    gen_pp = setup["gen_pp"]
    gen_enable_dp = setup["gen_enable_dp"]
    is_mla = setup["is_mla"]
    use_v2 = setup["use_v2"]
    num_kv_heads = setup["num_kv_heads"]
    head_dim = setup["head_dim"]
    tokens_per_block = setup["tokens_per_block"]

    sampling_params = SamplingParams()

    ctx_dp_rank = 0
    if ctx_enable_dp:
        ctx_dp_rank = ctx_request_id % ctx_tp
        valid_ctx_kv_cache_managers = []
        valid_ctx_transfer_workers = []
        for i in range(ctx_pp):
            valid_ctx_kv_cache_managers.append(ctx_kv_cache_managers[ctx_dp_rank + i * ctx_tp])
            valid_ctx_transfer_workers.append(ctx_transfer_workers[ctx_dp_rank + i * ctx_tp])
    else:
        valid_ctx_kv_cache_managers = ctx_kv_cache_managers
        valid_ctx_transfer_workers = ctx_transfer_workers
    gen_dp_rank = 0
    if gen_enable_dp:
        gen_dp_rank = gen_request_id % gen_tp
        valid_gen_kv_cache_managers = []
        valid_gen_transfer_workers = []
        for i in range(gen_pp):
            valid_gen_kv_cache_managers.append(gen_kv_cache_managers[gen_dp_rank + i * gen_tp])
            valid_gen_transfer_workers.append(gen_transfer_workers[gen_dp_rank + i * gen_tp])
    else:
        valid_gen_kv_cache_managers = gen_kv_cache_managers
        valid_gen_transfer_workers = gen_transfer_workers

    unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
    ctx_request = LlmRequest(
        request_id=ctx_request_id,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    ctx_request.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)

    ctx_request.add_new_token(8 + ctx_request_id, 0)
    ctx_request.py_draft_tokens = [
        9 + ctx_request_id,
        10 + ctx_request_id,
        11 + ctx_request_id,
        12 + ctx_request_id,
    ]

    # Add sequence to ctx KV cache managers
    ctx_kv_caches = []  # V2: Store kv_cache objects for cleanup
    if use_v2:
        for ctx_kv_cache_manager in valid_ctx_kv_cache_managers:
            kv_cache = ctx_kv_cache_manager._create_kv_cache(ctx_request.py_request_id, None, None)
            success = kv_cache.resume(torch.cuda.current_stream().cuda_stream)
            assert success, "Failed to resume kv_cache for ctx request"
            kv_cache.resize(ctx_request.prompt_len)
            ctx_kv_caches.append(kv_cache)
    else:
        for ctx_kv_cache_manager in valid_ctx_kv_cache_managers:
            ctx_kv_cache_manager.impl.add_sequence(
                ctx_request.py_request_id, ctx_request.prompt_len, 1, ctx_request
            )

    gen_request = LlmRequest(
        request_id=gen_request_id,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    gen_request.py_disaggregated_params = DisaggregatedParams(
        ctx_request_id=ctx_request.py_request_id,
        ctx_dp_rank=ctx_dp_rank,
        ctx_info_endpoint=ctx_info_endpoint,
        disagg_request_id=unique_rid,
    )
    # Add sequence to gen KV cache managers
    gen_kv_caches = []
    if use_v2:
        for gen_kv_cache_manager in valid_gen_kv_cache_managers:
            kv_cache = gen_kv_cache_manager._create_kv_cache(gen_request.py_request_id, None, None)
            success = kv_cache.resume(torch.cuda.current_stream().cuda_stream)
            assert success, "Failed to resume kv_cache for gen request"
            kv_cache.resize(gen_request.prompt_len)
            gen_kv_caches.append(kv_cache)
    else:
        for gen_kv_cache_manager in valid_gen_kv_cache_managers:
            gen_kv_cache_manager.impl.add_sequence(
                gen_request.py_request_id, gen_request.prompt_len, 1, gen_request
            )

    # Get block_ids per layer_group with window_size filtering
    ctx_block_ids_per_groups = [
        get_block_ids_per_layer_groups(
            ctx_kv_cache_manager,
            ctx_transfer_worker,
            ctx_request.py_request_id,
            use_v2,
            tokens_per_block,
        )
        for ctx_kv_cache_manager, ctx_transfer_worker in zip(
            valid_ctx_kv_cache_managers, valid_ctx_transfer_workers
        )
    ]

    gen_block_ids_per_groups = [
        get_block_ids_per_layer_groups(
            gen_kv_cache_manager,
            gen_transfer_worker,
            gen_request.py_request_id,
            use_v2,
            tokens_per_block,
        )
        for gen_kv_cache_manager, gen_transfer_worker in zip(
            valid_gen_kv_cache_managers, valid_gen_transfer_workers
        )
    ]

    # Determine number of layer_groups
    num_layer_groups = len(ctx_block_ids_per_groups[0]) if ctx_block_ids_per_groups else 1

    if send_first:
        sender_sessions = [
            ctx_transfer_worker.create_tx_session(ctx_request)
            for ctx_transfer_worker in valid_ctx_transfer_workers
        ]

        send_kv_slices = [
            KVSlice(is_last_slice=True, block_ids_per_layer_groups=ctx_block_ids_per_group)
            for ctx_block_ids_per_group in ctx_block_ids_per_groups
        ]
        send_slice_futures = [
            sender_session.send(send_kv_slice)
            for sender_session, send_kv_slice in zip(sender_sessions, send_kv_slices)
        ]

        for sender_session in sender_sessions:
            assert sender_session.status == SessionStatus.INIT

        receiver_sessions = [
            gen_transfer_worker.create_rx_session(gen_request)
            for gen_transfer_worker in valid_gen_transfer_workers
        ]
        recv_kv_slices = [
            KVSlice(is_last_slice=True, block_ids_per_layer_groups=gen_block_ids_per_group)
            for gen_block_ids_per_group in gen_block_ids_per_groups
        ]
        recv_slice_futures = [
            receiver_session.receive(recv_kv_slice)
            for receiver_session, recv_kv_slice in zip(receiver_sessions, recv_kv_slices)
        ]

    else:
        receiver_sessions = [
            gen_transfer_worker.create_rx_session(gen_request)
            for gen_transfer_worker in valid_gen_transfer_workers
        ]
        recv_kv_slices = [
            KVSlice(is_last_slice=True, block_ids_per_layer_groups=gen_block_ids_per_group)
            for gen_block_ids_per_group in gen_block_ids_per_groups
        ]
        recv_slice_futures = [
            receiver_session.receive(recv_kv_slice)
            for receiver_session, recv_kv_slice in zip(receiver_sessions, recv_kv_slices)
        ]

        random_sleep_time = random.uniform(0.000001, 0.001)
        time.sleep(random_sleep_time)
        sender_sessions = [
            ctx_transfer_worker.create_tx_session(ctx_request)
            for ctx_transfer_worker in valid_ctx_transfer_workers
        ]

        time.sleep(0.1)

        for sender_session in sender_sessions:
            assert sender_session.status != SessionStatus.INIT

        send_kv_slices = [
            KVSlice(is_last_slice=True, block_ids_per_layer_groups=ctx_block_ids_per_group)
            for ctx_block_ids_per_group in ctx_block_ids_per_groups
        ]
        send_slice_futures = [
            sender_session.send(send_kv_slice)
            for sender_session, send_kv_slice in zip(sender_sessions, send_kv_slices)
        ]
        send_aux_tasks = []
        for sender_session in sender_sessions:
            sender_session.pack_aux(ctx_request)
            send_aux_tasks.append(sender_session.send_aux())

    for future in send_slice_futures:
        future.result()
    for future in recv_slice_futures:
        future.result()
    if not send_first:
        for send_aux_task in send_aux_tasks:
            send_aux_task.future.result()

    sync_session_status = (
        SessionStatus.KV_TRANSFERRED if send_first else SessionStatus.FULLY_TRANSFERRED
    )
    for sender_session in sender_sessions:
        assert sender_session.status == sync_session_status
    if not send_first:
        time.sleep(0.1)
    for receiver_session in receiver_sessions:
        assert receiver_session.status == sync_session_status, (
            f"receiver_session.status={receiver_session.status}, "
            f"sync_session_status={sync_session_status} send_first={send_first}"
        )

    # Get block data for verification
    valid_ctx_tp = 1 if ctx_enable_dp else ctx_tp
    valid_gen_tp = 1 if gen_enable_dp else gen_tp
    if is_mla:
        valid_ctx_tp = 1
        valid_gen_tp = 1

    # Unified per-layer-group verification for both V1 and V2
    def get_layers_in_group_per_pp(kv_cache_managers, pp_size, tp_size, group_id, is_v2):
        """Get the number of layers in a group for each PP rank."""
        layers_per_pp = []
        for pp_rank in range(pp_size):
            mgr = kv_cache_managers[pp_rank * tp_size]
            if is_v2:
                layers_per_pp.append(len(mgr.impl.layer_grouping[group_id]))
            else:
                window_to_layers = mgr._get_window_size_to_layers()
                sorted_windows = sorted(window_to_layers.keys(), key=lambda x: (x is None, x))
                window_size = sorted_windows[group_id]
                layers_per_pp.append(len(window_to_layers[window_size]))
        return layers_per_pp

    for layer_group_id in range(num_layer_groups):
        # Get block_ids for this layer_group
        ctx_group_block_ids = [groups[layer_group_id] for groups in ctx_block_ids_per_groups]
        gen_group_block_ids = [groups[layer_group_id] for groups in gen_block_ids_per_groups]

        # Get data using unified get_block_data function
        ctx_block_datas = [
            get_block_data(mgr, bids, layer_group_id, use_v2, ctx_request.py_request_id)
            for mgr, bids in zip(valid_ctx_kv_cache_managers, ctx_group_block_ids)
        ]
        gen_block_datas = [
            get_block_data(mgr, bids, layer_group_id, use_v2, gen_request.py_request_id)
            for mgr, bids in zip(valid_gen_kv_cache_managers, gen_group_block_ids)
        ]

        # Get layers per PP rank for this group
        ctx_layers_per_pp = get_layers_in_group_per_pp(
            valid_ctx_kv_cache_managers, ctx_pp, valid_ctx_tp, layer_group_id, use_v2
        )
        gen_layers_per_pp = get_layers_in_group_per_pp(
            valid_gen_kv_cache_managers, gen_pp, valid_gen_tp, layer_group_id, use_v2
        )

        ctx_layers_in_group = sum(ctx_layers_per_pp)
        gen_layers_in_group = sum(gen_layers_per_pp)

        assert ctx_layers_in_group == gen_layers_in_group, (
            f"Layer group {layer_group_id}: ctx has {ctx_layers_in_group} layers, "
            f"gen has {gen_layers_in_group}"
        )
        num_layers_in_group = ctx_layers_in_group

        # Create merge tensors for ctx
        ctx_block_data_merge = torch.zeros(
            size=(
                ctx_block_datas[0].shape[0],
                num_layers_in_group,
                ctx_block_datas[0].shape[2],
                ctx_block_datas[0].shape[3] * valid_ctx_tp,
            )
        )
        ctx_layer_offset = 0
        for pp_rank in range(ctx_pp):
            pp_layers_in_group = ctx_layers_per_pp[pp_rank]
            for tp_rank in range(valid_ctx_tp):
                head_dim_per_rank = num_kv_heads // valid_ctx_tp * head_dim * tokens_per_block
                start_head_offset = tp_rank * head_dim_per_rank
                end_head_offset = start_head_offset + head_dim_per_rank
                block_idx = pp_rank * valid_ctx_tp + tp_rank
                ctx_block_data_merge[
                    :,
                    ctx_layer_offset : ctx_layer_offset + pp_layers_in_group,
                    :,
                    start_head_offset:end_head_offset,
                ] = ctx_block_datas[block_idx]
            ctx_layer_offset += pp_layers_in_group

        # Create merge tensors for gen
        gen_block_data_merge = torch.zeros(
            size=(
                gen_block_datas[0].shape[0],
                num_layers_in_group,
                gen_block_datas[0].shape[2],
                gen_block_datas[0].shape[3] * valid_gen_tp,
            )
        )
        gen_layer_offset = 0
        for pp_rank in range(gen_pp):
            pp_layers_in_group = gen_layers_per_pp[pp_rank]
            for tp_rank in range(valid_gen_tp):
                head_dim_per_rank = num_kv_heads // valid_gen_tp * head_dim * tokens_per_block
                start_head_offset = tp_rank * head_dim_per_rank
                end_head_offset = start_head_offset + head_dim_per_rank
                block_idx = pp_rank * valid_gen_tp + tp_rank
                gen_block_data_merge[
                    :,
                    gen_layer_offset : gen_layer_offset + pp_layers_in_group,
                    :,
                    start_head_offset:end_head_offset,
                ] = gen_block_datas[block_idx]
            gen_layer_offset += pp_layers_in_group

        assert ctx_block_data_merge.equal(gen_block_data_merge), (
            f"Layer group {layer_group_id} data mismatch"
        )

    if not send_first:
        for pp_rank in range(gen_pp):
            for tp_rank in range(valid_gen_tp):
                recv_session = receiver_sessions[pp_rank * valid_gen_tp + tp_rank]
                recv_session.unpack_aux(gen_request)

                assert gen_request.py_first_gen_tokens == [8 + ctx_request_id]
                assert gen_request.py_draft_tokens == [
                    9 + ctx_request_id,
                    10 + ctx_request_id,
                    11 + ctx_request_id,
                    12 + ctx_request_id,
                ]
    for receiver_session in receiver_sessions:
        receiver_session.close()
    for sender_session in sender_sessions:
        sender_session.close()

    # V2: Close kv_caches to release slots
    if use_v2:
        torch.cuda.current_stream().synchronize()
        for kv_cache in ctx_kv_caches:
            kv_cache.close()
        for kv_cache in gen_kv_caches:
            kv_cache.close()


# Test configurations as pytest parameters
PARALLEL_TEST_CONFIGS = [
    (1, 1, False, 1, 1, False, False, "tp1_pp1_to_tp1_pp1"),
    (1, 1, False, 1, 2, False, False, "tp1_pp1_to_tp1_pp2"),
    (1, 2, False, 1, 1, False, False, "tp1_pp2_to_tp1_pp1"),
    (1, 2, False, 1, 2, False, False, "tp1_pp2_to_tp1_pp2"),
    (1, 2, False, 2, 1, False, False, "tp1_pp2_to_tp2_pp1"),
    (2, 1, False, 1, 2, False, False, "tp2_pp1_to_tp1_pp2"),
    (4, 1, False, 2, 2, False, False, "tp4_pp1_to_tp2_pp2"),
    (1, 4, False, 2, 2, False, False, "tp1_pp4_to_tp2_pp2"),
    (2, 1, True, 2, 1, True, False, "tp2_pp1_dp_to_tp2_pp1_dp"),
    (2, 1, True, 1, 2, False, False, "tp2_pp1_dp_to_tp1_pp2"),
    (1, 4, False, 2, 2, True, False, "tp1_pp4_to_tp2_pp2_dp"),
    (2, 1, False, 2, 1, True, True, "tp2_pp1_to_tp2_pp1_dp_mla"),
    (2, 1, True, 2, 1, False, True, "tp2_pp1_dp_to_tp2_pp1_mla"),
]

WINDOW_SIZE_TEST_CONFIGS = [
    (1, 1, False, 1, 1, False, False, [1024], "no_window"),
    (1, 1, False, 1, 1, False, False, [1024, 24], "with_window"),
    (1, 2, False, 1, 1, False, False, [1024, 24], "pp2_to_pp1_window"),
    (1, 1, False, 1, 2, False, False, [1024, 24], "pp1_to_pp2_window"),
    (1, 2, False, 2, 1, False, False, [1024, 24], "pp2_to_tp2_window"),
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp,is_mla",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in PARALLEL_TEST_CONFIGS],
    ids=[c[7] for c in PARALLEL_TEST_CONFIGS],
)
def test_transfer_worker_v1(ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla):
    """Test transfer worker with KVCacheManager (V1)."""
    tensorrt_llm.logger.set_level("info")
    logger.info("Test transfer worker V1 with parallel configurations")

    setup = create_transfer_worker_setup(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=False,
    )

    request_len = setup["request_len"]
    try:
        add_and_verify_request(setup, 0, 1, request_len, send_first=True)
        add_and_verify_request(setup, 2, 3, request_len, send_first=True)
        add_and_verify_request(setup, 4, 5, request_len * 2, send_first=False)
    finally:
        for worker in setup["ctx_transfer_workers"]:
            worker.shutdown()
        for worker in setup["gen_transfer_workers"]:
            worker.shutdown()


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp,is_mla",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in PARALLEL_TEST_CONFIGS],
    ids=[c[7] for c in PARALLEL_TEST_CONFIGS],
)
def test_transfer_worker_v2(ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla):
    """Test transfer worker with KVCacheManagerV2 (V2)."""
    tensorrt_llm.logger.set_level("info")
    logger.info("Test transfer worker V2 with parallel configurations")

    setup = create_transfer_worker_setup(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=True,
    )

    request_len = setup["request_len"]
    try:
        add_and_verify_request(setup, 0, 1, request_len, send_first=True)
        add_and_verify_request(setup, 2, 3, request_len, send_first=True)
        add_and_verify_request(setup, 4, 5, request_len * 2, send_first=False)
    finally:
        for worker in setup["ctx_transfer_workers"]:
            worker.shutdown()
        for worker in setup["gen_transfer_workers"]:
            worker.shutdown()


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp,is_mla,max_attention_window_vec",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in WINDOW_SIZE_TEST_CONFIGS],
    ids=[c[8] for c in WINDOW_SIZE_TEST_CONFIGS],
)
def test_transfer_worker_v2_with_window(
    ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla, max_attention_window_vec
):
    """Test V2 transfer worker with sliding window attention."""
    tensorrt_llm.logger.set_level("info")
    logger.info("Test transfer worker V2 with sliding window attention")

    setup = create_transfer_worker_setup(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=True,
        max_attention_window_vec=max_attention_window_vec,
    )

    try:
        add_and_verify_request(setup, 0, 1, request_len=16, send_first=True)
        add_and_verify_request(setup, 2, 3, request_len=32, send_first=True)
        add_and_verify_request(setup, 4, 5, request_len=64, send_first=False)
    finally:
        for worker in setup["ctx_transfer_workers"]:
            worker.shutdown()
        for worker in setup["gen_transfer_workers"]:
            worker.shutdown()


def _setup_chunked_request(setup, ctx_request_id, gen_request_id, request_len):
    """Shared setup for chunked transfer tests: create requests, allocate KV, get block IDs."""
    ctx_transfer_workers = setup["ctx_transfer_workers"]
    ctx_kv_cache_managers = setup["ctx_kv_cache_managers"]
    gen_transfer_workers = setup["gen_transfer_workers"]
    gen_kv_cache_managers = setup["gen_kv_cache_managers"]
    ctx_info_endpoint = setup["ctx_info_endpoint"]
    use_v2 = setup["use_v2"]
    tokens_per_block = setup["tokens_per_block"]

    sampling_params = SamplingParams()
    unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF

    ctx_request = LlmRequest(
        request_id=ctx_request_id,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    )
    ctx_request.py_disaggregated_params = DisaggregatedParams(disagg_request_id=unique_rid)

    gen_request = LlmRequest(
        request_id=gen_request_id,
        max_new_tokens=1,
        input_tokens=list(range(request_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
    )
    gen_request.py_disaggregated_params = DisaggregatedParams(
        ctx_request_id=ctx_request.py_request_id,
        ctx_dp_rank=0,
        ctx_info_endpoint=ctx_info_endpoint,
        disagg_request_id=unique_rid,
    )

    ctx_kv_caches, gen_kv_caches = [], []
    for mgr in ctx_kv_cache_managers:
        if use_v2:
            kv = mgr._create_kv_cache(ctx_request.py_request_id, None, None)
            assert kv.resume(torch.cuda.current_stream().cuda_stream)
            assert kv.resize(request_len)
            ctx_kv_caches.append(kv)
        else:
            mgr.impl.add_sequence(ctx_request.py_request_id, request_len, 1, ctx_request)

    for mgr in gen_kv_cache_managers:
        if use_v2:
            kv = mgr._create_kv_cache(gen_request.py_request_id, None, None)
            assert kv.resume(torch.cuda.current_stream().cuda_stream)
            assert kv.resize(request_len)
            gen_kv_caches.append(kv)
        else:
            mgr.impl.add_sequence(gen_request.py_request_id, request_len, 1, gen_request)

    ctx_block_ids = [
        get_block_ids_per_layer_groups(mgr, tw, ctx_request.py_request_id, use_v2, tokens_per_block)
        for mgr, tw in zip(ctx_kv_cache_managers, ctx_transfer_workers)
    ]
    gen_block_ids = [
        get_block_ids_per_layer_groups(mgr, tw, gen_request.py_request_id, use_v2, tokens_per_block)
        for mgr, tw in zip(gen_kv_cache_managers, gen_transfer_workers)
    ]

    return {
        "ctx_request": ctx_request,
        "gen_request": gen_request,
        "ctx_kv_caches": ctx_kv_caches,
        "gen_kv_caches": gen_kv_caches,
        "ctx_block_ids": ctx_block_ids,
        "gen_block_ids": gen_block_ids,
    }


def _verify_and_cleanup_chunked(setup, ctx_info, sender_sessions, receiver_sessions):
    """Shared verification and cleanup for chunked transfer tests."""
    ctx_kv_cache_managers = setup["ctx_kv_cache_managers"]
    gen_kv_cache_managers = setup["gen_kv_cache_managers"]
    ctx_transfer_workers = setup["ctx_transfer_workers"]
    gen_transfer_workers = setup["gen_transfer_workers"]
    use_v2 = setup["use_v2"]

    ctx_block_ids = ctx_info["ctx_block_ids"]
    gen_block_ids = ctx_info["gen_block_ids"]

    for session in sender_sessions:
        assert session.status == SessionStatus.KV_TRANSFERRED
    for session in receiver_sessions:
        assert session.status == SessionStatus.KV_TRANSFERRED

    num_layer_groups = len(ctx_block_ids[0])
    for lg_id in range(num_layer_groups):
        ctx_data = [
            get_block_data(mgr, bids[lg_id], lg_id, use_v2, ctx_info["ctx_request"].py_request_id)
            for mgr, bids in zip(ctx_kv_cache_managers, ctx_block_ids)
        ]
        gen_data = [
            get_block_data(mgr, bids[lg_id], lg_id, use_v2, ctx_info["gen_request"].py_request_id)
            for mgr, bids in zip(gen_kv_cache_managers, gen_block_ids)
        ]
        for c, g in zip(ctx_data, gen_data):
            assert c.equal(g), f"Layer group {lg_id}: data mismatch with chunked transfer"

    for tw, s in zip(gen_transfer_workers, receiver_sessions):
        tw.clear_session(s)
    for tw, s in zip(ctx_transfer_workers, sender_sessions):
        tw.clear_session(s)
    if use_v2:
        torch.cuda.current_stream().synchronize()
        for kv in ctx_info["ctx_kv_caches"]:
            kv.close()
        for kv in ctx_info["gen_kv_caches"]:
            kv.close()


def add_and_verify_chunked_request(
    setup,
    ctx_request_id,
    gen_request_id,
    request_len,
    chunk_size_blocks,
):
    """Chunked transfer variant: sender sends N slices, receiver sends 1."""
    import math

    ctx_transfer_workers = setup["ctx_transfer_workers"]
    gen_transfer_workers = setup["gen_transfer_workers"]

    ctx_info = _setup_chunked_request(setup, ctx_request_id, gen_request_id, request_len)
    ctx_block_ids = ctx_info["ctx_block_ids"]
    gen_block_ids = ctx_info["gen_block_ids"]

    sender_sessions = [tw.create_tx_session(ctx_info["ctx_request"]) for tw in ctx_transfer_workers]
    send_futures = []
    for sender_session, block_ids_per_groups in zip(sender_sessions, ctx_block_ids):
        max_blocks = max(len(ids) for ids in block_ids_per_groups)
        num_chunks = math.ceil(max_blocks / chunk_size_blocks)
        chunk_offset = 0
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size_blocks
            end = start + chunk_size_blocks
            is_last = chunk_idx == num_chunks - 1
            chunk_block_ids = [ids[start:end] for ids in block_ids_per_groups]
            kv_slice = KVSlice(
                is_last_slice=is_last,
                block_ids_per_layer_groups=chunk_block_ids,
            )
            send_futures.append(sender_session.send(kv_slice, chunk_block_offset=chunk_offset))
            chunk_offset += max(len(ids) for ids in chunk_block_ids)

    receiver_sessions = [
        tw.create_rx_session(ctx_info["gen_request"]) for tw in gen_transfer_workers
    ]
    recv_futures = []
    for recv_session, block_ids_per_groups in zip(receiver_sessions, gen_block_ids):
        full_slice = KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=block_ids_per_groups,
        )
        recv_futures.append(recv_session.receive(full_slice))

    for f in send_futures:
        f.result()
    for f in recv_futures:
        f.result()

    _verify_and_cleanup_chunked(setup, ctx_info, sender_sessions, receiver_sessions)


CHUNKED_TEST_CONFIGS = [
    (1, 1, False, 1, 1, False, False, True, "v2_tp1_pp1_chunked"),
    (1, 1, False, 1, 1, False, False, False, "v1_tp1_pp1_chunked"),
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp,is_mla,use_v2",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in CHUNKED_TEST_CONFIGS],
    ids=[c[8] for c in CHUNKED_TEST_CONFIGS],
)
def test_transfer_worker_chunked(
    ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla, use_v2
):
    """Test transfer worker with sender-side chunking for V1 and V2."""
    tensorrt_llm.logger.set_level("info")
    logger.info(f"Test transfer worker {'V2' if use_v2 else 'V1'} with chunked transfer")

    setup = create_transfer_worker_setup(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=use_v2,
    )

    request_len = setup["request_len"]
    tokens_per_block = setup["tokens_per_block"]
    total_blocks = (request_len + tokens_per_block - 1) // tokens_per_block
    chunk_size = max(1, total_blocks // 2)

    try:
        add_and_verify_chunked_request(setup, 0, 1, request_len, chunk_size_blocks=chunk_size)
        add_and_verify_chunked_request(setup, 2, 3, request_len * 2, chunk_size_blocks=chunk_size)
    finally:
        for worker in setup["ctx_transfer_workers"]:
            worker.shutdown()
        for worker in setup["gen_transfer_workers"]:
            worker.shutdown()


if __name__ == "__main__":
    test_transfer_worker_v1(1, 1, False, 1, 1, False, False)
