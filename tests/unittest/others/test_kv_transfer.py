"""Test KV Transfer with KVCacheManager (V1) and KVCacheManagerV2 (V2)."""

import random
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
import tensorrt_llm.tensorrt_llm_transfer_agent_binding  # TODO: remove it.  # noqa: F401
from tensorrt_llm import DisaggregatedParams, Mapping, SamplingParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.region.auxiliary import AuxBuffer
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker
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
    # V2 specific field
    max_util_for_resume: float = 0.95


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
        use_v2: If True, use KVCacheManagerV2 (Python-side). Otherwise use KVCacheManager (C++ bindings).
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

    meta_max_batch_size = 32
    beam_width = 1
    max_draft_len = 4

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
        ctx_aux_buffer = AuxBuffer(meta_max_batch_size, beam_width, max_draft_len)
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
            # V2: Initialize pools using kv_pool_attrs
            random_seed = 0 if is_mla else None
            kv_pool_attrs = KVRegionExtractorV1.create_kv_pool_attrs_from_manager(
                ctx_kv_cache_manager
            )

            # Collect unique pools (deduplicate by pool_base_ptr, keep max size)
            unique_pools = {}  # pool_base_ptr -> pool_size
            for layer_group_attrs in kv_pool_attrs.layer_group_attrs_list:
                for pool_idx, pool_ptr in enumerate(layer_group_attrs.pool_base_ptrs):
                    pool_size = layer_group_attrs.pool_sizes[pool_idx]
                    if pool_ptr not in unique_pools or pool_size > unique_pools[pool_ptr]:
                        unique_pools[pool_ptr] = pool_size

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

            # Use KvCacheConfig from llmapi for VSWA, trtllm.KvCacheConfig otherwise
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
                kv_cache_manager=ctx_kv_cache_manager,
                mapping=ctx_mappings[i],
                device_id=device_id,
                instance_name=ctx_instance_name,
                aux_buffer=ctx_aux_buffer,
            )
        )

    ctx_info_endpoint = ctx_transfer_workers[0]._instance_info_server.endpoint
    ctx_endpoints = [
        ctx_transfer_worker._sender.endpoint for ctx_transfer_worker in ctx_transfer_workers
    ]
    ctx_layer_num_per_pp = []
    for pp_rank in range(ctx_pp):
        ctx_layer_num_per_pp.append(
            len(ctx_transfer_workers[pp_rank * ctx_tp]._kv_cache_manager.pp_layers)
        )

    for ctx_transfer_worker in ctx_transfer_workers:
        ctx_transfer_worker.populate_instance_and_rank_info(
            endpoints=ctx_endpoints, layer_num_per_pp=ctx_layer_num_per_pp
        )

    gen_transfer_workers = []
    gen_kv_cache_managers = []
    for i in range(gen_instance_num):
        gen_aux_buffer = AuxBuffer(meta_max_batch_size, beam_width, max_draft_len)
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
            gen_pool_attrs = KVRegionExtractorV1.create_kv_pool_attrs_from_manager(
                gen_kv_cache_manager
            )
            gen_unique_pools = {}
            for layer_group_attrs in gen_pool_attrs.layer_group_attrs_list:
                for pool_idx, pool_ptr in enumerate(layer_group_attrs.pool_base_ptrs):
                    pool_size = layer_group_attrs.pool_sizes[pool_idx]
                    if pool_ptr not in gen_unique_pools or pool_size > gen_unique_pools[pool_ptr]:
                        gen_unique_pools[pool_ptr] = pool_size
            gen_element_bytes = get_size_in_bytes(1, gen_kv_cache_manager.dtype)
            for pool_base_ptr, pool_size in gen_unique_pools.items():
                pool_size_elements = pool_size // gen_element_bytes
                pool_tensor = convert_to_torch_tensor(
                    TensorWrapper(pool_base_ptr, gen_kv_cache_manager.dtype, [pool_size_elements])
                )
                pool_tensor.zero_()
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

            # Use KvCacheConfig from llmapi for VSWA, trtllm.KvCacheConfig otherwise
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
            # Initialize gen pool to zeros to verify transfer correctness
            gen_pool_tensor = gen_kv_cache_manager.get_unique_primary_pool()
            gen_pool_tensor.zero_()
        gen_kv_cache_managers.append(gen_kv_cache_manager)
        gen_transfer_workers.append(
            TransferWorker(
                kv_cache_manager=gen_kv_cache_manager,
                mapping=gen_mappings[i],
                device_id=device_id,
                instance_name=gen_instance_name,
                aux_buffer=gen_aux_buffer,
            )
        )
    _ = gen_transfer_workers[0]._instance_info_server.endpoint  # noqa: F841
    gen_endpoints = [
        gen_transfer_worker._sender.endpoint for gen_transfer_worker in gen_transfer_workers
    ]
    gen_layer_num_per_pp = []
    for pp_rank in range(gen_pp):
        gen_layer_num_per_pp.append(
            len(gen_transfer_workers[pp_rank * gen_tp]._kv_cache_manager.pp_layers)
        )
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
    request_id: int = None,
) -> torch.Tensor:
    """Unified block data retrieval for both V1 and V2 KVCacheManager.

    Returns tensor with shape [num_slots, num_layers_in_group, kv_factor, flat_dim]

    Args:
        kv_cache_manager: KVCacheManager (V1) or KVCacheManagerV2 instance
        block_ids: Block IDs to retrieve (used for V1 only)
        layer_group_id: Layer group ID (required for both V1 and V2)
        use_v2: Whether using KVCacheManagerV2
        request_id: Request ID (required for V2, ignored for V1)
    """
    if use_v2:
        # V2: Use get_buffers with block_ids directly
        # For V2, block_ids from get_aggregated_page_indices are slot indices
        # We need to map them to buffer indices via get_batch_cache_indices
        layer_grouping = kv_cache_manager.impl.layer_grouping
        local_layer_indices = layer_grouping[layer_group_id]

        all_layer_data = []
        for local_layer_idx in local_layer_indices:
            global_layer_idx = kv_cache_manager.pp_layers[local_layer_idx]
            # Get all buffer indices for this layer
            all_buffer_indices = kv_cache_manager.get_batch_cache_indices(
                [request_id], layer_id=global_layer_idx
            )[0]
            # Filter to valid indices only
            valid_buffer_indices = [idx for idx in all_buffer_indices if idx >= 0]
            # Apply window filtering: block_ids specifies which valid slots to use
            # block_ids are indices into the valid slots (0, 1, 2, ... N-1)
            # For window filtering, we take the last N slots
            num_valid = len(valid_buffer_indices)
            num_requested = len(block_ids)
            if num_requested < num_valid:
                # Window filtering: take last num_requested slots
                selected_indices = valid_buffer_indices[-num_requested:]
            else:
                selected_indices = valid_buffer_indices

            layer_buffer = kv_cache_manager.get_buffers(layer_idx=global_layer_idx, kv_layout="HND")
            # Get selected slots: shape [num_slots, kv_factor, ...]
            layer_data = layer_buffer[selected_indices]
            # Reshape to [num_slots, kv_factor, flat_dim]
            layer_data = layer_data.reshape(layer_data.shape[0], layer_data.shape[1], -1)
            all_layer_data.append(layer_data)

        # Stack layers: [num_layers, num_slots, kv_factor, flat_dim]
        # Then transpose to [num_slots, num_layers, kv_factor, flat_dim]
        result = torch.stack(all_layer_data, dim=0).permute(1, 0, 2, 3)
        return result
    else:
        # V1: Use get_unique_primary_pool (single window only, VSWA tests are skipped)
        pool_tensor = kv_cache_manager.get_unique_primary_pool()
        # V1 pool shape: [num_blocks, num_layers, kv_factor, blockSize]
        # where blockSize = tokens_per_block * num_kv_heads * head_dim
        block_datas = pool_tensor[block_ids]
        # Shape: [num_slots, num_layers, kv_factor, blockSize]
        return block_datas


def get_block_ids_per_layer_groups(
    kv_cache_manager, transfer_worker, request_id: int, use_v2: bool, tokens_per_block: int
) -> List[List[int]]:
    """Get block_ids for each layer group with window_size filtering.

    Args:
        kv_cache_manager: KVCacheManager or KVCacheManagerV2 instance
        transfer_worker: TransferWorker instance (to access kv_pool_attrs)
        request_id: Request ID
        use_v2: Whether using KVCacheManagerV2
        tokens_per_block: Tokens per block

    Returns:
        List of block_ids for each layer_group
    """
    kv_pool_attrs = transfer_worker._rank_info.kv_pool_attrs
    block_ids_per_layer_groups: List[List[int]] = []

    for group_attrs in kv_pool_attrs.layer_group_attrs_list:
        if use_v2:
            # V2: Use get_aggregated_page_indices for efficient slot indices
            block_ids = list(
                kv_cache_manager.kv_cache_map[request_id].get_aggregated_page_indices(
                    group_attrs.group_id, valid_only=True
                )
            )
        else:
            # V1: Use get_batch_cache_indices
            first_global_layer_id = group_attrs.global_layer_ids[0]
            block_ids = kv_cache_manager.get_batch_cache_indices(
                [request_id], layer_id=first_global_layer_id
            )[0]

        # Filter by window_size if request_len > window_size
        window_size = group_attrs.sliding_window_size
        if window_size is not None:
            max_blocks_in_window = window_size // tokens_per_block + 1
            if len(block_ids) > max_blocks_in_window:
                block_ids = block_ids[-max_blocks_in_window:]

        block_ids_per_layer_groups.append(list(block_ids))

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

    unique_rid = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF  # Generate positive int64 random number
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
    gen_kv_caches = []  # V2: Store kv_cache objects for cleanup
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
        send_slice_tasks = [
            sender_session._kv_tasks[sender_session.send(send_kv_slice)]
            for sender_session, send_kv_slice in zip(sender_sessions, send_kv_slices)
        ]

        for sender_session in sender_sessions:
            assert sender_session.state.status == SessionStatus.INIT

        receiver_sessions = [
            gen_transfer_worker.create_rx_session(gen_request)
            for gen_transfer_worker in valid_gen_transfer_workers
        ]
        recv_kv_slices = [
            KVSlice(is_last_slice=True, block_ids_per_layer_groups=gen_block_ids_per_group)
            for gen_block_ids_per_group in gen_block_ids_per_groups
        ]
        recv_slice_tasks = [
            receiver_session._kv_tasks[receiver_session.receive(recv_kv_slice)]
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
        recv_slice_tasks = [
            receiver_session._kv_tasks[receiver_session.receive(recv_kv_slice)]
            for receiver_session, recv_kv_slice in zip(receiver_sessions, recv_kv_slices)
        ]

        random_sleep_time = random.uniform(0.000001, 0.001)
        time.sleep(random_sleep_time)  # here , recv session may send or not send req_info
        sender_sessions = [
            ctx_transfer_worker.create_tx_session(ctx_request)
            for ctx_transfer_worker in valid_ctx_transfer_workers
        ]

        time.sleep(0.1)  # wait for recv session to send req_info  , wait send session to be ready

        for sender_session in sender_sessions:
            assert sender_session.state.status != SessionStatus.INIT

        send_kv_slices = [
            KVSlice(is_last_slice=True, block_ids_per_layer_groups=ctx_block_ids_per_group)
            for ctx_block_ids_per_group in ctx_block_ids_per_groups
        ]
        send_slice_tasks = [
            sender_session._kv_tasks[sender_session.send(send_kv_slice)]
            for sender_session, send_kv_slice in zip(sender_sessions, send_kv_slices)
        ]
        send_aux_tasks = []
        for ctx_transfer_worker, sender_session in zip(valid_ctx_transfer_workers, sender_sessions):
            ctx_transfer_worker.pack_aux(sender_session, ctx_request)
            send_aux_tasks.append(sender_session.send_aux())

    for send_slice_task in send_slice_tasks:
        send_slice_task.future.result()
    for recv_slice_task in recv_slice_tasks:
        recv_slice_task.future.result()
    if not send_first:
        for send_aux_task in send_aux_tasks:
            send_aux_task.future.result()

    sync_session_status = SessionStatus.TRANSFERRED if send_first else SessionStatus.AUX_TRANSFERRED
    for sender_session in sender_sessions:
        assert sender_session.state.status == sync_session_status
    if not send_first:
        time.sleep(0.1)
    for receiver_session in receiver_sessions:
        assert receiver_session.state.status == sync_session_status, (
            f"receiver_session.state.status={receiver_session.state.status}, "
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
                ctx_block_datas[0].shape[0],  # num_slots
                num_layers_in_group,
                ctx_block_datas[0].shape[2],  # kv_factor
                ctx_block_datas[0].shape[3] * valid_ctx_tp,  # heads
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
                gen_block_datas[0].shape[0],  # num_slots
                num_layers_in_group,
                gen_block_datas[0].shape[2],  # kv_factor
                gen_block_datas[0].shape[3] * valid_gen_tp,  # heads
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
                transfer_worker = valid_gen_transfer_workers[pp_rank * valid_gen_tp + tp_rank]
                recv_session = receiver_sessions[pp_rank * valid_gen_tp + tp_rank]
                transfer_worker.unpack_aux(recv_session, gen_request)

                assert gen_request.py_first_gen_tokens == [8 + ctx_request_id]
                assert gen_request.py_draft_tokens == [
                    9 + ctx_request_id,
                    10 + ctx_request_id,
                    11 + ctx_request_id,
                    12 + ctx_request_id,
                ]
    for transfer_worker, receiver_session in zip(valid_gen_transfer_workers, receiver_sessions):
        transfer_worker.clear_session(receiver_session)
    for transfer_worker, sender_session in zip(valid_ctx_transfer_workers, sender_sessions):
        transfer_worker.clear_session(sender_session)

    # V2: Close kv_caches to release slots (required for KVCacheManagerV2)
    if use_v2:
        torch.cuda.current_stream().synchronize()
        for kv_cache in ctx_kv_caches:
            kv_cache.close()
        for kv_cache in gen_kv_caches:
            kv_cache.close()


# Test configurations as pytest parameters
PARALLEL_TEST_CONFIGS = [
    # (ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla, test_id)
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

# Window size test configurations
# max_seq_len=1024 (hardcoded), tokens_per_block=8
# small_window=24 (3 blocks)
# Test with request_len: 16 (< 24), 32 (> 24), 64 (>> 24)
WINDOW_SIZE_TEST_CONFIGS = [
    # (ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla,
    #  max_attention_window_vec, test_id)
    # No window (full attention only)
    (1, 1, False, 1, 1, False, False, [1024], "no_window"),
    # With window: [max_seq_len, small_window]
    (1, 1, False, 1, 1, False, False, [1024, 24], "with_window"),
    # Different PP + window
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
    add_and_verify_request(setup, 0, 1, request_len, send_first=True)
    add_and_verify_request(setup, 2, 3, request_len, send_first=True)
    add_and_verify_request(setup, 4, 5, request_len * 2, send_first=False)


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
    add_and_verify_request(setup, 0, 1, request_len, send_first=True)
    add_and_verify_request(setup, 2, 3, request_len, send_first=True)
    add_and_verify_request(setup, 4, 5, request_len * 2, send_first=False)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,ctx_enable_dp,gen_tp,gen_pp,gen_enable_dp,is_mla,max_attention_window_vec",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]) for c in WINDOW_SIZE_TEST_CONFIGS],
    ids=[c[8] for c in WINDOW_SIZE_TEST_CONFIGS],
)
def test_transfer_worker_v1_with_window(
    ctx_tp, ctx_pp, ctx_enable_dp, gen_tp, gen_pp, gen_enable_dp, is_mla, max_attention_window_vec
):
    """Test V1 transfer worker with sliding window attention."""
    pytest.skip("V1 with VSWA (Variable Sliding Window Attention) not supported in unit test")

    tensorrt_llm.logger.set_level("info")
    logger.info("Test transfer worker V1 with sliding window attention")

    setup = create_transfer_worker_setup(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        gen_enable_dp=gen_enable_dp,
        is_mla=is_mla,
        use_v2=False,
        max_attention_window_vec=max_attention_window_vec,
    )

    # Test multiple requests with different lengths
    # small_window=24, tokens_per_block=8
    # 16 tokens = 2 blocks (< small_window)
    # 32 tokens = 4 blocks (> small_window, need last 4 blocks for sliding window layers)
    # 64 tokens = 8 blocks (>> small_window, need last 4 blocks for sliding window layers)
    add_and_verify_request(setup, 0, 1, request_len=16, send_first=True)
    add_and_verify_request(setup, 2, 3, request_len=32, send_first=True)
    add_and_verify_request(setup, 4, 5, request_len=64, send_first=False)


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

    # Test multiple requests with different lengths
    add_and_verify_request(setup, 0, 1, request_len=16, send_first=True)
    add_and_verify_request(setup, 2, 3, request_len=32, send_first=True)
    add_and_verify_request(setup, 4, 5, request_len=64, send_first=False)


if __name__ == "__main__":
    test_transfer_worker_v1(1, 1, False, 1, 1, False, False)
