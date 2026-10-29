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

"""Test KV Transfer for DeepseekV4CacheManager with KvCacheTransceiverV2.

Drives the shared threaded single-process harness (``kv_transfer_harness``)
with DeepSeek-V4 cache managers. Validates all DeepseekV4AttentionType cache
transfers across different TP/PP/DP configurations.
"""

import ctypes
import functools
import uuid
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from kv_transfer_harness import (
    MAX_BATCH_SIZE,
    MAX_SEQ_LEN,
    TOKENS_PER_BLOCK,
    VOCAB_SIZE,
    get_layers_per_pp,
    run_kv_transfer_test,
)

from tensorrt_llm import Mapping
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool, get_pool_bytes
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig

# ---------------------------------------------------------------------------
# Constants matching DeepseekV4CacheManager defaults
# ---------------------------------------------------------------------------
HEAD_DIM = 512
INDEX_HEAD_DIM = 128
WINDOW_SIZE = 128
NUM_KV_HEADS = 1

# DeepSeek-V4 specific ratios (mirrors module constants)
SPARSE_RATIO = DEEPSEEK_V4_SPARSE_RATIO
OVERLAP_COMPRESSOR_RATIO = DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO


# ---------------------------------------------------------------------------
# DeepseekV4CacheManager creation helpers
# ---------------------------------------------------------------------------
def _create_deepseek_v4_manager(
    mapping: Mapping,
    compress_ratios: List[int],
    dtype: DataType = DataType.BF16,
    compressor_dtype: DataType = DataType.FLOAT,
) -> DeepseekV4CacheManager:
    """Create a DeepseekV4CacheManager for the given mapping."""
    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        index_head_dim=INDEX_HEAD_DIM,
        window_size=WINDOW_SIZE,
        compress_ratios=compress_ratios,
    )
    max_num_tokens = MAX_SEQ_LEN * MAX_BATCH_SIZE
    kv_cache_config = KvCacheConfig(
        dtype="fp8_ds_mla" if get_sm_version() == 90 else "auto",
        enable_block_reuse=False,
        max_tokens=max_num_tokens,
        event_buffer_max_size=0,
    )
    return DeepseekV4CacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=len(compress_ratios),
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        mapping=mapping,
        dtype=dtype,
        compressor_dtype=compressor_dtype,
        vocab_size=VOCAB_SIZE,
        max_num_tokens=max_num_tokens,
        sparse_attn_config=sparse_attn_config,
    )


def _create_managers_for_instance(
    tp: int,
    pp: int,
    enable_dp: bool,
    compress_ratios: List[int],
) -> List[DeepseekV4CacheManager]:
    """Create DeepseekV4CacheManagers for all ranks in an instance."""
    world_size = tp * pp
    managers = []
    for rank in range(world_size):
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=tp,
            pp_size=pp,
            enable_attention_dp=enable_dp,
        )
        managers.append(_create_deepseek_v4_manager(mapping, compress_ratios))
    return managers


def _init_pool_data(
    managers: Sequence[DeepseekV4CacheManager],
    tp: int,
    seed_base: int = 0,
    fill_random: bool = True,
) -> None:
    """Initialize pool data for all managers.

    Uses half-precision view of pool memory for initialization.
    Since pools have mixed dtypes (BF16, FP32, FP8), we use HALF (2 bytes) as a
    common element type that evenly divides all pool sizes.

    Args:
        managers: List of DeepseekV4CacheManagers.
        tp: TP size.
        seed_base: Base seed offset (different for ctx vs gen to avoid collisions).
        fill_random: If True fill with random data (ctx), if False fill with zeros (gen).
    """
    initialized_devices = set()
    for rank, mgr in enumerate(managers):
        pp_rank = rank // tp
        page_table = KVRegionExtractorV1(mgr).page_table

        # Collect unique pools (deduplicate by base_address)
        unique_pools: Dict[int, int] = {}
        for lg_idx, lg in enumerate(page_table.layer_groups):
            for pv in lg.pool_views:
                pool = get_physical_pool(page_table, lg_idx, pv.pool_idx)
                key = pool.base_address
                if key not in unique_pools or get_pool_bytes(pool) > unique_pools[key]:
                    unique_pools[key] = get_pool_bytes(pool)

        # Use HALF (2 bytes) as element type - divides all pool sizes evenly
        elem_bytes = 2  # sizeof(half)
        for pool_base_ptr, pool_size in unique_pools.items():
            pool_size_elements = pool_size // elem_bytes
            pool_tensor = convert_to_torch_tensor(
                TensorWrapper(pool_base_ptr, DataType.HALF, [pool_size_elements])
            )
            initialized_devices.add(pool_tensor.device)
            if fill_random:
                # Same seed for same pp_rank across TP ranks (kv_heads=1)
                seed = seed_base + pp_rank
                generator = torch.Generator(device=pool_tensor.device).manual_seed(seed)
                random_values = torch.rand(
                    pool_tensor.shape,
                    dtype=pool_tensor.dtype,
                    device=pool_tensor.device,
                    generator=generator,
                )
                pool_tensor.copy_(random_values)
            else:
                pool_tensor.zero_()

    # NIXL's worker streams do not inherit this torch stream's dependencies.
    # Finish synthetic source writes and destination clearing before transfer.
    for device in initialized_devices:
        torch.cuda.current_stream(device).synchronize()


# ---------------------------------------------------------------------------
# Attention type helpers
# ---------------------------------------------------------------------------
def _get_attn_types_for_layer(
    layer_idx: int, compress_ratios: List[int]
) -> List[DeepseekV4AttentionType]:
    """Get the attention types for a given layer based on its compress ratio."""
    ratio = compress_ratios[layer_idx]
    is_compress = ratio > 1
    is_sparse = ratio == SPARSE_RATIO
    types = [DeepseekV4AttentionType.SWA]
    if is_compress:
        types.extend(
            [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.COMPRESSOR_KV,
                DeepseekV4AttentionType.COMPRESSOR_SCORE,
            ]
        )
    if is_sparse:
        types.extend(
            [
                DeepseekV4AttentionType.INDEXER_COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
                DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
            ]
        )
    return types


# Mirror transceiver.py windowed-block trim: only in-window blocks are transferred.
_WINDOWED_ATTN_TYPES = {
    DeepseekV4AttentionType.SWA,
    DeepseekV4AttentionType.COMPRESSOR_KV,
    DeepseekV4AttentionType.COMPRESSOR_SCORE,
    DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
    DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
}


def _expected_valid_blocks(
    attn_type: DeepseekV4AttentionType, compress_ratio: int, prompt_len: int
) -> Optional[int]:
    if attn_type == DeepseekV4AttentionType.SWA:
        window = WINDOW_SIZE
    elif attn_type in _WINDOWED_ATTN_TYPES:
        state_factor = 2 if compress_ratio == OVERLAP_COMPRESSOR_RATIO else 1
        window = state_factor * compress_ratio
    else:
        return None
    total = (prompt_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    stale = max(0, (prompt_len + 1 - window) // TOKENS_PER_BLOCK)
    return total - stale


def _split_blockwise_buffer(
    buffer: torch.Tensor,
    data_size: int,
    scale_size: int,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a blockwise quantized buffer into value and scale buffers.

    Args:
        buffer: shape [num_blocks, tokens_per_block, bytes_per_token]

    Returns:
        (values_buffer, scales_buffer), preserving the manager's scale dtype.
    """
    num_blocks, tokens_per_block, bytes_per_token = buffer.shape
    bytes_per_block = bytes_per_token * tokens_per_block

    # Value buffer
    value_shape = (num_blocks, tokens_per_block, data_size)
    value_stride = (bytes_per_block, data_size, 1)
    value_buffer = buffer.as_strided(value_shape, value_stride, 0).view(torch.uint8)

    # Scale buffer
    scale_shape = (num_blocks, tokens_per_block, scale_size)
    scale_stride = (bytes_per_block, scale_size, 1)
    scale_offset = data_size * tokens_per_block
    scale_buffer = buffer.as_strided(scale_shape, scale_stride, scale_offset).view(scale_dtype)

    return value_buffer, scale_buffer


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _read_cache_data(
    mgr: DeepseekV4CacheManager,
    layer_idx: int,
    attn_type: DeepseekV4AttentionType,
    request_id: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Read cache data for a layer/attn_type from a DeepseekV4CacheManager.

    Returns all non-BAD block data. For INDEXER_COMPRESS returns (values, scales).
    """
    buffer = mgr.get_buffers(layer_idx, attn_type)
    # get_cache_indices may contain BAD_PAGE_INDEX (-1) for evicted blocks; filter them out.
    indices = [i for i in mgr.get_cache_indices(request_id, layer_idx, attn_type) if i >= 0]

    if not indices:
        return torch.tensor([]), None

    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        scale_dtype = torch.float32 if mgr._indexer_k_dtype == "fp8" else torch.uint8
        values_buf, scales_buf = _split_blockwise_buffer(
            buffer,
            data_size=mgr._indexer_data_size,
            scale_size=mgr._indexer_scale_size,
            scale_dtype=scale_dtype,
        )
        return values_buf[indices], scales_buf[indices]

    return buffer[indices], None


def _find_ctx_rank_for_layer(
    layer_idx: int,
    ctx_managers: Sequence[DeepseekV4CacheManager],
    ctx_tp: int,
    ctx_enable_dp: bool,
    req_idx: int,
) -> int:
    """Find the ctx rank that owns a given model layer.

    Returns a rank with the correct PP rank and correct TP rank for DP.
    """
    for rank, mgr in enumerate(ctx_managers):
        tp_rank = rank % ctx_tp
        if ctx_enable_dp:
            if req_idx % ctx_tp != tp_rank:
                continue
        elif tp_rank != 0:
            # Without DP, all TP ranks have same data; use tp_rank=0
            continue
        if layer_idx in mgr.pp_layers:
            return rank
    raise ValueError(f"No ctx rank found for layer {layer_idx}")


def verify_all_requests(
    request_lengths: List[int],
    compress_ratios: List[int],
    ctx_managers: Sequence[DeepseekV4CacheManager],
    gen_managers: Sequence[DeepseekV4CacheManager],
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    ctx_request_ids: List[int],
    gen_request_ids: List[int],
):
    """Verify transferred cache data for all requests across all gen ranks."""
    gen_world = gen_tp * gen_pp

    for req_idx, req_len in enumerate(request_lengths):
        ctx_rid = ctx_request_ids[req_idx]
        gen_rid = gen_request_ids[req_idx]

        for gen_rank in range(gen_world):
            tp_rank = gen_rank % gen_tp
            # Skip ranks that didn't handle this request (DP mode)
            if gen_enable_dp and req_idx % gen_tp != tp_rank:
                continue

            gen_mgr = gen_managers[gen_rank]
            gen_pp_layers = gen_mgr.pp_layers

            for layer_idx in gen_pp_layers:
                # Find the ctx rank owning this layer
                ctx_rank = _find_ctx_rank_for_layer(
                    layer_idx,
                    ctx_managers,
                    ctx_tp,
                    ctx_enable_dp,
                    req_idx,
                )
                ctx_mgr = ctx_managers[ctx_rank]

                for attn_type in _get_attn_types_for_layer(layer_idx, compress_ratios):
                    ctx_data, ctx_scales = _read_cache_data(ctx_mgr, layer_idx, attn_type, ctx_rid)
                    gen_data, gen_scales = _read_cache_data(gen_mgr, layer_idx, attn_type, gen_rid)

                    expected_valid = _expected_valid_blocks(
                        attn_type, compress_ratios[layer_idx], req_len
                    )
                    if expected_valid is not None:
                        if expected_valid <= 0:
                            ctx_data = ctx_data[:0]
                            gen_data = gen_data[:0]
                        else:
                            ctx_data = ctx_data[-expected_valid:]
                            gen_data = gen_data[-expected_valid:]

                    assert ctx_data.shape == gen_data.shape, (
                        f"Shape mismatch at req={req_idx} layer={layer_idx} "
                        f"attn={attn_type.name}: ctx={ctx_data.shape} gen={gen_data.shape}"
                    )

                    torch.testing.assert_close(
                        gen_data,
                        ctx_data,
                        rtol=0,
                        atol=0,
                        msg=lambda m: (
                            f"Data mismatch at req={req_idx} layer={layer_idx} "
                            f"attn={attn_type.name} gen_rank={gen_rank}: {m}"
                        ),
                    )

                    if ctx_scales is not None:
                        assert gen_scales is not None, (
                            f"Expected scales at req={req_idx} layer={layer_idx} attn={attn_type.name}"
                        )
                        torch.testing.assert_close(
                            gen_scales,
                            ctx_scales,
                            rtol=0,
                            atol=0,
                            msg=lambda m: (
                                f"Scale mismatch at req={req_idx} layer={layer_idx} "
                                f"attn={attn_type.name} gen_rank={gen_rank}: {m}"
                            ),
                        )
                    else:
                        assert gen_scales is None


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------
def run_deepseek_v4_transfer_test(
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    compress_ratios: List[int],
    update_before_transfer: bool = True,
) -> None:
    """Run the shared transfer harness with DeepSeek-V4 cache hooks."""
    run_kv_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        update_before_transfer=update_before_transfer,
        manager_factory=lambda tp, pp, enable_dp: _create_managers_for_instance(
            tp, pp, enable_dp, compress_ratios
        ),
        init_fn=_init_pool_data,
        verify_fn=functools.partial(verify_all_requests, compress_ratios=compress_ratios),
    )


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
TEST_CONFIGS = [
    # (ctx_tp, ctx_pp, gen_tp, gen_pp, ctx_enable_dp, gen_enable_dp, test_id)
    # Basic
    (1, 1, 1, 1, False, False, "tp1_pp1"),
    (2, 1, 2, 1, False, False, "tp2_pp1"),
    (1, 2, 1, 2, False, False, "pp2_symmetric"),
    (1, 2, 1, 1, False, False, "pp2_to_pp1"),
    (2, 2, 2, 2, False, False, "tp2_pp2"),
    # DP
    (2, 1, 2, 1, True, True, "tp2_dp_both"),
    (2, 1, 1, 2, True, False, "ctx_dp_gen_pp2"),
    (2, 2, 2, 2, True, True, "tp2_pp2_dp_both"),
]


# @pytest.mark.threadleak(enabled=False)
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,gen_tp,gen_pp,ctx_enable_dp,gen_enable_dp",
    [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in TEST_CONFIGS],
    ids=[c[6] for c in TEST_CONFIGS],
)
@pytest.mark.parametrize(
    "compress_ratios",
    [[1, 4, 128], [128, 1, 4, 128]],
    ids=["cr_1_4_128", "cr_128_1_4_128"],
)
@pytest.mark.parametrize(
    "update_before_transfer",
    [True, False],
    ids=["update_before", "update_after"],
)
def test_deepseek_v4_kv_transfer(
    ctx_tp,
    ctx_pp,
    gen_tp,
    gen_pp,
    ctx_enable_dp,
    gen_enable_dp,
    compress_ratios,
    update_before_transfer,
):
    """Test KvCacheTransceiverV2 with DeepseekV4CacheManager."""
    mode = "update_before" if update_before_transfer else "update_after"
    print(
        f"\nRunning DeepSeek-V4 transfer test [{mode}]: "
        f"ctx_tp={ctx_tp} ctx_pp={ctx_pp} gen_tp={gen_tp} gen_pp={gen_pp} "
        f"ctx_dp={ctx_enable_dp} gen_dp={gen_enable_dp} "
        f"compress_ratios={compress_ratios}"
    )

    run_deepseek_v4_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        compress_ratios=compress_ratios,
        update_before_transfer=update_before_transfer,
    )

    print("PASSED")


def _native_transfer_manager(
    offload: bool, dtype: DataType, tokens_per_block: int
) -> DeepseekV4CacheManager:
    """Build real V4 pools without enabling the guarded model-inference path."""
    from tensorrt_llm.llmapi.llm_args import SparseKvCacheOffloadConfig
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
    )

    adapter = object.__new__(DeepseekV4CacheManager)
    adapter._enable_kv_cache_offload = offload
    adapter._sparse_offload_topk = 8
    adapter._sparse_kv_offload = SparseKvCacheOffloadConfig() if offload else None
    adapter.pp_layers = [0, 1, 2, 3]
    adapter._compress_ratios = [1, 4, 128, 4]
    adapter.compressed_block_sizes = [tokens_per_block // r for r in adapter._compress_ratios]
    adapter.tokens_per_block = tokens_per_block
    adapter.head_dim = HEAD_DIM
    adapter.index_head_dim = INDEX_HEAD_DIM
    adapter.dtype = DataType.FP8 if dtype == DataType.NVFP4 else dtype
    adapter._indexer_k_dtype = "fp4"
    adapter._use_nvfp4_compress = dtype == DataType.NVFP4
    adapter._nvfp4_residual_dim = 64
    adapter.use_fp8_ds_mla = False
    adapter._swa_window_size = WINDOW_SIZE
    adapter.max_seq_len = 4 * tokens_per_block
    adapter.max_batch_size = 2
    adapter.is_estimating_kv_cache = False
    adapter.max_cuda_graph_batch_size = None
    adapter.max_attention_window_vec = [None] * 4
    adapter.max_draft_len = adapter._max_draft_len = 0
    adapter.num_extra_kv_tokens = 0
    adapter.num_kv_heads_per_layer = [1] * 4
    config = adapter._build_cache_config(
        KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=[GpuCacheTierConfig(quota=32 << 20)],
            layers=[],
        )
    )
    adapter.impl = KVCacheManager(config)
    return adapter


@pytest.mark.timeout(180)
@pytest.mark.parametrize("dtype", [DataType.BF16, DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_deepseek_v4_native_sparse_host_transfer(dtype: DataType, tokens_per_block: int) -> None:
    """NIXL writes V4 compressed KV/scales to host and all other roles to GPU.

    Uses the production V4 layout builder, native KVCM allocations, and the
    sender's registration, mapping, grouping, and submission. Raw byte checks
    include partial pages and repeated requests reusing the same pools.
    """
    from cuda.bindings import runtime as cudart
    from test_sparse_kv_transfer import _rank_info, _sender_and_meta

    from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CudaStream

    device_id = torch.cuda.current_device()
    stream = CudaStream(torch.cuda.current_stream().cuda_stream)
    managers, workers, agents = [], [], []
    suffix = uuid.uuid4().hex

    def copy_bytes(dst: int, src: int, size: int, kind) -> None:
        (error,) = cudart.cudaMemcpy(dst, src, size, kind)
        assert int(error) == 0

    try:
        for offload, name in ((False, "ctx"), (True, "gen")):
            manager = _native_transfer_manager(offload, dtype, tokens_per_block)
            managers.append(manager)
            table = KVRegionExtractorV1(manager).page_table
            agent = BindingsNixlTransferAgent(f"{name}-{suffix}", num_threads=1)
            agents.append(agent)
            worker = SimpleNamespace()
            worker._rank_info = _rank_info(name, table)
            worker._rank_info.device_id = device_id
            worker._agent = agent
            worker._registered_mem = []
            TransferWorker._register_kv_cache(worker)
            workers.append(worker)
        src, dst = managers
        assert {d.type for d in workers[0]._registered_mem} == {"VRAM"}
        assert {d.type for d in workers[1]._registered_mem} == {"VRAM", "DRAM"}
        agents[0].load_remote_agent(agents[1].name, agents[1].get_local_agent_desc())

        for request_idx, prompt_len in enumerate(
            (tokens_per_block - 1, tokens_per_block, tokens_per_block + 1, 3 * tokens_per_block - 1)
        ):
            caches = []
            transfer_active = False
            try:
                blocks = []
                for manager in managers:
                    cache = manager.impl.create_kv_cache()
                    caches.append(cache)
                    assert cache.resume(stream)
                    assert cache.resize(prompt_len, prompt_len)
                    blocks.append(
                        [
                            np.array(
                                list(cache.get_aggregated_page_indices(lg, valid_only=False)),
                                dtype=np.int64,
                            )
                            for lg in range(len(manager.impl.layer_grouping))
                        ]
                    )
                torch.cuda.synchronize()
                expected = []
                host_scale_buffers = 0
                for layer in dst.impl.init_config.layers:
                    lg = dst.impl.get_layer_group_id(layer.layer_id)
                    src_lg = src.impl.get_layer_group_id(layer.layer_id)
                    src_indices = list(caches[0].get_base_page_indices(src_lg))
                    dst_indices = list(caches[1].get_base_page_indices(lg))
                    for buffer_idx, buffer in enumerate(layer.buffers):
                        if buffer.role in dst.get_disagg_ignored_roles():
                            continue
                        src_base = src.impl.get_mem_pool_base_address(layer.layer_id, buffer.role)
                        dst_base = dst.impl.get_mem_pool_base_address(layer.layer_id, buffer.role)
                        src_stride = src.impl.get_page_stride(layer.layer_id, buffer.role)
                        dst_stride = dst.impl.get_page_stride(layer.layer_id, buffer.role)
                        src_converter = src.impl.get_page_index_converter(
                            layer.layer_id, buffer.role
                        )
                        dst_converter = dst.impl.get_page_index_converter(
                            layer.layer_id, buffer.role
                        )
                        assert src_converter.expansion == dst_converter.expansion == 1
                        src_role_indices = src_converter(src_indices)
                        dst_role_indices = dst_converter(dst_indices)
                        host_scale_buffers += int(buffer.is_sparse and "scale" in str(buffer.role))
                        for ordinal, (src_index, dst_index) in enumerate(
                            zip(src_role_indices, dst_role_indices)
                        ):
                            if src_index < 0 or dst_index < 0:
                                continue
                            data = (
                                (
                                    np.arange(buffer.size, dtype=np.int64) * (2 * buffer_idx + 1)
                                    + int(layer.layer_id) * 23
                                    + ordinal * 17
                                    + request_idx * 11
                                )
                                % 251
                            ).astype(np.uint8)
                            src_ptr = int(src_base) + int(src_index) * src_stride
                            dst_ptr = int(dst_base) + int(dst_index) * dst_stride
                            copy_bytes(
                                src_ptr,
                                data.ctypes.data,
                                buffer.size,
                                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                            )
                            if buffer.is_sparse:
                                ctypes.memset(dst_ptr, 0, buffer.size)
                            else:
                                zeros = np.zeros(buffer.size, dtype=np.uint8)
                                copy_bytes(
                                    dst_ptr,
                                    zeros.ctypes.data,
                                    buffer.size,
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                )
                            expected.append((dst_ptr, data, buffer.is_sparse, str(buffer.role)))
                if dtype == DataType.NVFP4:
                    assert host_scale_buffers == 2
                sender, meta = _sender_and_meta(
                    workers[0]._rank_info.page_table,
                    workers[1]._rank_info.page_table,
                    blocks[0],
                    blocks[1],
                    device_id=device_id,
                    prompt_len=prompt_len,
                )
                sender._agent = agents[0]
                sender._enforce_physical_ownership = False
                meta.peer_name = agents[1].name
                assert {(g.src_memory_type, g.dst_memory_type) for g in meta.transfer_groups} == {
                    ("VRAM", "VRAM"),
                    ("VRAM", "DRAM"),
                }
                request = sender._make_agent_request(meta, device_id)
                transfer_active = True
                assert sender._submit_transfer(meta.task, 0, request) == (True, None)
                transfer_active = False
                assert meta.task.resources_drained
                for address, data, on_host, role in expected:
                    if on_host:
                        actual = np.frombuffer(ctypes.string_at(address, data.size), dtype=np.uint8)
                    else:
                        actual = np.empty_like(data)
                        copy_bytes(
                            actual.ctypes.data,
                            address,
                            data.size,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        )
                    np.testing.assert_array_equal(actual, data, err_msg=role)
            finally:
                if transfer_active:
                    # Quiesce any failed submission before releasing its pools.
                    for agent in agents:
                        agent.shutdown()
                    agents.clear()
                    workers.clear()
                for cache in caches:
                    cache.close()
    finally:
        for worker in workers:
            for registration in reversed(worker._registered_mem):
                worker._agent.deregister_memory(registration)
            worker._registered_mem.clear()
        for agent in agents:
            agent.shutdown()
        for manager in managers:
            manager.impl.shutdown()


# ---------------------------------------------------------------------------
# Uneven PP layer test configurations
# ---------------------------------------------------------------------------
# compress_ratios templates for different num_layers (covering ratios 1, 4, 128)
UNEVEN_PP_COMPRESS_RATIOS = {
    5: [1, 4, 128, 1, 4],
    7: [1, 4, 128, 1, 4, 128, 1],
}

UNEVEN_PP_CONFIGS = [
    # (ctx_tp, ctx_pp, gen_tp, gen_pp, ctx_dp, gen_dp, num_layers, test_id)
    # 5 layers, pp=2 → [3, 2]
    (1, 2, 1, 2, False, False, 5, "5L_tp1_pp2"),
    (2, 2, 2, 2, False, False, 5, "5L_tp2_pp2"),
    # 5 layers, pp=3 → [2, 2, 1]
    (1, 3, 1, 3, False, False, 5, "5L_tp1_pp3"),
    # 7 layers, pp=2 → [4, 3]
    (1, 2, 1, 2, False, False, 7, "7L_tp1_pp2"),
    (2, 2, 2, 2, False, False, 7, "7L_tp2_pp2"),
    # 7 layers, pp=3 → [3, 2, 2]
    (1, 3, 1, 3, False, False, 7, "7L_tp1_pp3"),
    # 7 layers, pp=4 → [2, 2, 2, 1]
    (1, 4, 1, 4, False, False, 7, "7L_tp1_pp4"),
    # Asymmetric TP/PP with uneven layers
    (2, 1, 1, 2, False, False, 5, "5L_tp2_to_pp2"),
    (1, 2, 2, 1, False, False, 5, "5L_pp2_to_tp2"),
    (4, 1, 1, 4, False, False, 7, "7L_tp4_to_pp4"),
    (1, 4, 4, 1, False, False, 7, "7L_pp4_to_tp4"),
    (2, 2, 1, 4, False, False, 7, "7L_tp2pp2_to_pp4"),
    # Uneven layers + DP
    (2, 2, 2, 2, True, True, 5, "5L_tp2_pp2_dp_both"),
]


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "ctx_tp,ctx_pp,gen_tp,gen_pp,ctx_enable_dp,gen_enable_dp,num_layers",
    [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in UNEVEN_PP_CONFIGS],
    ids=[c[7] for c in UNEVEN_PP_CONFIGS],
)
@pytest.mark.parametrize(
    "update_before_transfer",
    [True, False],
    ids=["update_before", "update_after"],
)
def test_deepseek_v4_kv_transfer_uneven_pp(
    ctx_tp,
    ctx_pp,
    gen_tp,
    gen_pp,
    ctx_enable_dp,
    gen_enable_dp,
    num_layers,
    update_before_transfer,
):
    """Test KvCacheTransceiverV2 with DeepseekV4CacheManager and uneven layers-per-PP-rank.

    When num_layers is not evenly divisible by pp_size, the first
    (num_layers % pp_size) PP ranks get one extra layer.
    """
    compress_ratios = UNEVEN_PP_COMPRESS_RATIOS[num_layers]
    mode = "update_before" if update_before_transfer else "update_after"

    print(
        f"\nRunning DeepSeek-V4 uneven PP transfer test [{mode}]: "
        f"ctx_tp={ctx_tp} ctx_pp={ctx_pp} gen_tp={gen_tp} gen_pp={gen_pp} "
        f"ctx_dp={ctx_enable_dp} gen_dp={gen_enable_dp} "
        f"num_layers={num_layers} compress_ratios={compress_ratios} "
        f"layers_per_pp(ctx)={get_layers_per_pp(num_layers, ctx_pp)} "
        f"layers_per_pp(gen)={get_layers_per_pp(num_layers, gen_pp)}"
    )

    run_deepseek_v4_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=ctx_pp,
        gen_tp=gen_tp,
        gen_pp=gen_pp,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        compress_ratios=compress_ratios,
        update_before_transfer=update_before_transfer,
    )

    print("PASSED")
