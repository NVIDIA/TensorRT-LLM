# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVCacheManagerV2 fixtures for multiprocess transport tests."""

import torch

from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_size_in_bytes


def initialize_cache(manager: KVCacheManagerV2, seed: int | None = None) -> None:
    """Fill each physical pool once, using matching seeds for replicated ranks."""
    page_table = build_page_table_from_manager(manager)
    pools = get_unique_pool_memory_descs(page_table, torch.cuda.current_device())
    element_bytes = get_size_in_bytes(1, manager.dtype)
    for pool_idx, (address, size_bytes, _, _) in enumerate(pools):
        buffer = convert_to_torch_tensor(
            TensorWrapper(address, manager.dtype, [size_bytes // element_bytes])
        )
        generator = (
            torch.Generator(device=buffer.device).manual_seed(seed + pool_idx)
            if seed is not None
            else None
        )
        buffer.copy_(
            torch.rand(buffer.shape, dtype=torch.float32, device=buffer.device, generator=generator)
        )


def allocate_request(manager: KVCacheManagerV2, request: LlmRequest) -> None:
    """Allocate complete prompt KV pages without running the scheduler."""
    cache = manager._create_kv_cache(request.py_request_id, None, None)
    assert cache is not None
    assert cache.resume(torch.cuda.current_stream().cuda_stream)
    assert cache.resize(request.prompt_len, history_length=request.prompt_len)


def request_block_data(manager: KVCacheManagerV2, request: LlmRequest) -> torch.Tensor:
    """Return request KV data as [blocks, layers, K/V, flattened page]."""
    layers = []
    for layer_idx in manager.pp_layers:
        blocks = manager.get_batch_cache_indices([request.py_request_id], layer_idx)[0]
        data = manager.get_buffers(layer_idx, kv_layout="HND")[blocks]
        layers.append(data.reshape(data.shape[0], data.shape[1], -1))
    return torch.stack(layers, dim=1)
