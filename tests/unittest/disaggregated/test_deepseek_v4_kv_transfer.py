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

import functools
from typing import Dict, List, Optional, Sequence, Tuple

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
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool, get_pool_bytes
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig

# ---------------------------------------------------------------------------
# Constants matching DeepseekV4CacheManager defaults
# ---------------------------------------------------------------------------
HEAD_DIM = 256  # Reduced from 512: test validates transfer, not attention correctness
INDEX_HEAD_DIM = 128
WINDOW_SIZE = 128
NUM_KV_HEADS = 1
INDEXER_QUANT_BLOCK_SIZE = 128

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
    index_head_dim: int = INDEX_HEAD_DIM,
    quant_block_size: int = INDEXER_QUANT_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a blockwise FP8 quantized buffer into value and scale buffers.

    Args:
        buffer: shape [num_blocks, tokens_per_block, bytes_per_token]

    Returns:
        (values_buffer, scales_buffer) where values are uint8 and scales are float32
    """
    num_blocks, tokens_per_block, bytes_per_token = buffer.shape
    bytes_per_block = bytes_per_token * tokens_per_block

    # Value buffer
    value_shape = (num_blocks, tokens_per_block, index_head_dim)
    value_stride = (bytes_per_block, index_head_dim, 1)
    value_buffer = buffer.as_strided(value_shape, value_stride, 0).view(torch.uint8)

    # Scale buffer
    scale_dim = index_head_dim // quant_block_size
    scale_bytes = scale_dim * 4  # float32 = 4 bytes
    scale_shape = (num_blocks, tokens_per_block, scale_bytes)
    scale_stride = (bytes_per_block, scale_bytes, 1)
    scale_offset = index_head_dim * tokens_per_block
    scale_buffer = buffer.as_strided(scale_shape, scale_stride, scale_offset).view(torch.float32)

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
        values_buf, scales_buf = _split_blockwise_buffer(buffer)
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
