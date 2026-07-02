# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Variable Sliding Window Attention (VSWA) calculation utilities.

These are initialization-time pure computations with no per-step runtime
state.  Extracted from KVCacheManager to keep the main class focused on
runtime resource management.
"""

import copy
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import tensorrt_llm.bindings
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger

DataType = tensorrt_llm.bindings.DataType
ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
BlocksPerWindow = Dict[int, Tuple[int, int]]


def get_window_size_to_layers(
    max_attention_window_vec: List[int],
    num_local_layers: int,
) -> Dict[int, List[int]]:
    """Get the window size to layers mapping.

    The returned map has window sizes as keys and lists of layer indices as values.
    max_attention_window_vec is treated as a repeating pattern.
    """
    window_size_to_layers_map = defaultdict(list)

    if not max_attention_window_vec:
        if num_local_layers > 0:
            raise Exception("max_attention_window_vec cannot be empty if there are local layers.")
        return {}

    pattern_len = len(max_attention_window_vec)
    if pattern_len == 1:
        return {max_attention_window_vec[0]: list(range(num_local_layers))}
    for local_layer_idx in range(num_local_layers):
        window_size = max_attention_window_vec[local_layer_idx % pattern_len]
        window_size_to_layers_map[window_size].append(local_layer_idx)
    return window_size_to_layers_map


def adjust_window_sizes_for_vswa(
    window_size_to_layers: Dict[int, List[int]],
    max_attention_window_vec: List[int],
    kv_cache_config: KvCacheConfig,
    num_kv_heads_per_layer: List[int],
    kv_factor: int,
    head_size: int,
    dtype: DataType,
    pool_memory_bytes: int,
    calculate_scaling_factor_size_bytes_fn,
    is_cross_attention: bool = False,
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Adjust window sizes to fit available memory for VSWA.

    If even a single sequence cannot fit, the larger windows are shrunk
    until the total per-sequence memory fits.
    """
    assert is_cross_attention is False, "Cross attention is not supported"

    max_tokens_from_config = kv_cache_config.max_tokens

    def calculate_cache_size_per_token(layers: Set[int]) -> int:
        total_kv_heads = sum(num_kv_heads_per_layer[i] for i in layers)
        return total_kv_heads * kv_factor * head_size

    # Calculate the required memory bytes per sequence.
    required_mem_bytes_per_seq = 0
    for window_size in sorted(window_size_to_layers):
        layers = window_size_to_layers[window_size]
        cache_size_per_token = calculate_cache_size_per_token(layers)
        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, dtype)
        if dtype == DataType.NVFP4:
            cache_size_bytes_per_token += calculate_scaling_factor_size_bytes_fn(
                cache_size_per_token, quant_vector_size=16, scaling_factor_dtype=DataType.FP8
            )
        required_mem_bytes_per_seq += window_size * cache_size_bytes_per_token
    logger.info(f"Required memory per sequence: {required_mem_bytes_per_seq} bytes")
    logger.info(f"Memory bytes in pool: {pool_memory_bytes}")

    if required_mem_bytes_per_seq < pool_memory_bytes:
        logger.info("No need to adjust the window sizes, returning")
        return (copy.deepcopy(window_size_to_layers), max_attention_window_vec)

    logger.info(
        f"Adjusting the window sizes {list(window_size_to_layers)} to fit "
        f"the memory {pool_memory_bytes} bytes."
    )
    adjusted_window_size_to_layers = {}

    remaining_mem_bytes = pool_memory_bytes
    remaining_layers = set(i for layers in window_size_to_layers.values() for i in layers)

    accum_max_tokens = 0
    prev_window_size = 0
    adjusted_dict = {}
    adjusted_max_attention_window_vec = max_attention_window_vec.copy()

    for window_size in sorted(window_size_to_layers):
        layers = window_size_to_layers[window_size]
        if remaining_mem_bytes > 0 and remaining_layers:
            cache_size_per_token = calculate_cache_size_per_token(remaining_layers)
            cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, dtype)
            if dtype == DataType.NVFP4:
                cache_size_bytes_per_token += calculate_scaling_factor_size_bytes_fn(
                    cache_size_per_token, quant_vector_size=16, scaling_factor_dtype=DataType.FP8
                )
            logger.debug(
                f"Cache size per token for {len(remaining_layers)} layers: "
                f"{cache_size_bytes_per_token} bytes"
            )
            max_tokens_in_window = min(
                remaining_mem_bytes // cache_size_bytes_per_token, window_size - prev_window_size
            )
            remaining_mem_bytes -= max_tokens_in_window * cache_size_bytes_per_token
            accum_max_tokens += max_tokens_in_window
            logger.debug(f"Remaining memory: {remaining_mem_bytes} bytes")
            logger.debug(f"Max token of window {window_size}: {accum_max_tokens}")

            if accum_max_tokens < window_size:
                logger.debug(
                    f"Max tokens ({accum_max_tokens}) cannot fill the current window ({window_size}). "
                    f"The larger windows will have the same max tokens."
                )
                remaining_mem_bytes = 0

            if max_tokens_from_config is not None:
                accum_max_tokens = min(max_tokens_from_config, accum_max_tokens)
                if accum_max_tokens == max_tokens_from_config:
                    remaining_mem_bytes = 0

        if accum_max_tokens not in adjusted_window_size_to_layers:
            adjusted_window_size_to_layers[accum_max_tokens] = layers.copy()
        else:
            adjusted_window_size_to_layers[accum_max_tokens].extend(layers)
        adjusted_dict[window_size] = accum_max_tokens
        adjusted_max_attention_window_vec = [
            adjusted_dict.get(v, v) for v in adjusted_max_attention_window_vec
        ]

        remaining_layers -= set(layers)
        prev_window_size = window_size

    return (adjusted_window_size_to_layers, adjusted_max_attention_window_vec)


def calculate_max_num_blocks_for_vswa(
    *,
    kv_cache_config: KvCacheConfig,
    max_attention_window_vec: List[int],
    num_kv_heads_per_layer: List[int],
    num_local_layers: int,
    kv_factor: int,
    dtype: DataType,
    tokens_per_block: int,
    is_vswa: bool,
    mapping,
    model_config: ModelConfigCpp,
    calculate_scaling_factor_size_bytes_fn,
) -> Tuple[Dict[int, Tuple[int, int]], List[int], int, int]:
    """Calculate blocks per window for VSWA.

    Returns:
        (blocks_per_window, adjusted_max_attention_window_vec,
         primary_pool_memory_bytes, secondary_pool_memory_bytes)
    """
    is_cross_attention = False
    assert model_config.layer_types is not None, "layer_types have to be set correctly for VSWA"

    window_size_to_layers = get_window_size_to_layers(max_attention_window_vec, num_local_layers)
    logger.debug(f"window_size_to_layers: {window_size_to_layers}")

    import torch

    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gpu_memory_fraction = (
        kv_cache_config.free_gpu_memory_fraction
        if kv_cache_config.free_gpu_memory_fraction
        else 0.9
    )
    primary_pool_memory_bytes = (
        kv_cache_config.max_gpu_total_bytes
        if kv_cache_config.max_gpu_total_bytes > 0
        else int(free_mem * free_gpu_memory_fraction)
    )
    secondary_pool_memory_bytes = (
        kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
    )
    logger.debug(
        f"primary_pool_memory_bytes is set to {primary_pool_memory_bytes / 1024**3}GB, \n"
        f"secondary_pool_memory_bytes is set to {secondary_pool_memory_bytes / 1024**3}GB"
    )

    window_size_to_layers, adjusted_max_attention_window_vec = adjust_window_sizes_for_vswa(
        window_size_to_layers=window_size_to_layers,
        max_attention_window_vec=max_attention_window_vec,
        kv_cache_config=kv_cache_config,
        num_kv_heads_per_layer=num_kv_heads_per_layer,
        kv_factor=kv_factor,
        head_size=model_config.head_size,
        dtype=dtype,
        pool_memory_bytes=primary_pool_memory_bytes,
        calculate_scaling_factor_size_bytes_fn=calculate_scaling_factor_size_bytes_fn,
        is_cross_attention=is_cross_attention,
    )

    def calculate_cache_size_per_token(layers: Set[int]) -> int:
        total_kv_heads = sum(num_kv_heads_per_layer[i] for i in layers)
        return total_kv_heads * kv_factor * model_config.head_size

    logger.info(f"Primary pool memory bytes: {primary_pool_memory_bytes}")
    logger.info(f"Secondary pool memory bytes: {secondary_pool_memory_bytes}")

    if os.getenv("TRTLLM_WINDOW_SIZE_SHARES") is not None:
        logger.info("Environment variable TRTLLM_WINDOW_SIZE_SHARES is set")
        window_size_shares = os.getenv("TRTLLM_WINDOW_SIZE_SHARES").split(",")
        window_size_shares = [float(share) for share in window_size_shares]
        assert len(window_size_shares) == len(window_size_to_layers), (
            "Number of shares in TRTLLM_WINDOW_SIZE_SHARES must match number of window sizes"
        )
        assert sum(window_size_shares) == 1.0, (
            "Sum of shares in TRTLLM_WINDOW_SIZE_SHARES must be 1.0"
        )
    else:
        logger.info("Using default allocation of equal proportion of memory to each window size")
        window_size_shares = [1.0 / len(window_size_to_layers) for _ in window_size_to_layers]

    logger.info(f"Derived window_size_shares: {window_size_shares}")

    blocks_per_window = {}
    for window_idx, (window_size, layers) in enumerate(sorted(window_size_to_layers.items())):
        cache_size_per_token = calculate_cache_size_per_token(layers)
        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token, dtype)

        primary_tokens = (
            primary_pool_memory_bytes * window_size_shares[window_idx] / cache_size_bytes_per_token
        )
        secondary_tokens = (
            secondary_pool_memory_bytes
            * window_size_shares[window_idx]
            / cache_size_bytes_per_token
        )

        if kv_cache_config.max_tokens is not None:
            if is_vswa:
                logger.info(
                    f"kv_cache_config.max_tokens is not None "
                    f"({kv_cache_config.max_tokens}) but we are operating "
                    f"on VSWA scheme. Ignoring the configuration."
                )
            if not is_vswa:
                logger.info(f"kv_cache_config.max_tokens is {kv_cache_config.max_tokens}")
                if kv_cache_config.max_tokens < primary_tokens:
                    logger.info(
                        f"kv_cache_config.max_tokens "
                        f"{kv_cache_config.max_tokens} is less than "
                        f"primary_tokens {primary_tokens}. Reducing "
                        f"primary_tokens to {kv_cache_config.max_tokens}"
                    )
                    primary_tokens = kv_cache_config.max_tokens

        primary_blocks = int(primary_tokens // tokens_per_block)
        secondary_blocks = int(secondary_tokens // tokens_per_block)
        logger.info(
            f"Window size = {window_size}, primary_blocks: {primary_blocks}, secondary_blocks: {secondary_blocks}"
        )
        blocks_per_window[window_size] = (primary_blocks, secondary_blocks)

    return (
        blocks_per_window,
        adjusted_max_attention_window_vec,
        primary_pool_memory_bytes,
        secondary_pool_memory_bytes,
    )


def validate_and_adjust_attention_windows(
    max_attention_window_vec: List[int],
    blocks_per_window: BlocksPerWindow,
    tokens_per_block: int,
    sink_token_length: int,
    max_seq_len: int,
    max_beam_width: int,
    get_max_atten_window_upper_bound_fn,
) -> Tuple[BlocksPerWindow, int, List[int]]:
    """Validate and adjust attention windows against their upper bounds.

    Returns:
        Tuple of (adjusted_blocks_per_window, adjusted_max_seq_len,
                  adjusted_max_attention_window_vec)
    """
    window_adjustments = {}
    for window_size, (blocks_in_primary_pool, _) in blocks_per_window.items():
        upper_bound = get_max_atten_window_upper_bound_fn(
            blocks_in_primary_pool=blocks_in_primary_pool,
            tokens_per_block=tokens_per_block,
            max_beam_width=max_beam_width,
            sink_token_len=sink_token_length,
            max_seq_len=max_seq_len,
        )
        if window_size > upper_bound:
            logger.warning(
                f"Attention window size {window_size} exceeds upper bound {upper_bound} "
                f"for available blocks. Reducing to {upper_bound}."
            )
            window_adjustments[window_size] = upper_bound
    if window_adjustments:
        adjusted_window_vec = [
            window_adjustments.get(window, window) for window in max_attention_window_vec
        ]
        logger.warning(f"Adjusted max_attention_window_vec to {adjusted_window_vec}")
        adjusted_blocks_per_window = {}
        for window_size, memory_pools in blocks_per_window.items():
            if window_size in window_adjustments:
                adjusted_window_size = window_adjustments[window_size]
                adjusted_blocks_per_window[adjusted_window_size] = memory_pools
                logger.warning(
                    f"Adjusted window size {window_size} to {adjusted_window_size} in blocks_per_window"
                )
            else:
                adjusted_blocks_per_window[window_size] = memory_pools
        adjusted_max_seq_len = max(adjusted_window_vec)
        logger.warning(f"Adjusted max_seq_len to {adjusted_max_seq_len}")

        return adjusted_blocks_per_window, adjusted_max_seq_len, adjusted_window_vec
    else:
        return blocks_per_window, max_seq_len, max_attention_window_vec
