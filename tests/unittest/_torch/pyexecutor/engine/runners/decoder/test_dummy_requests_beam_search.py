# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import Mock, patch

import torch
from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config
from _torch.pyexecutor.engine_test_utils import DummyModel

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

_BEAM_WIDTH = 4
_BEAM_BATCH_SIZE = 4
_BEAM_MAX_SEQ_LEN = 32
_BEAM_TOKENS_PER_BLOCK = 8


def _make_factory(max_tokens: int) -> DummyRequestFactory:
    config = decoder_config(
        max_batch_size=_BEAM_BATCH_SIZE,
        max_num_tokens=_BEAM_BATCH_SIZE * _BEAM_MAX_SEQ_LEN,
        max_seq_len=_BEAM_MAX_SEQ_LEN,
        max_beam_width=_BEAM_WIDTH,
        cuda_graph_config=None,
        cuda_graph_batch_sizes=[1, 2, 4],
        max_cuda_graph_batch_size=4,
        enable_autotuner=False,
    )
    model = DummyModel(torch.half)
    return DummyRequestFactory(
        config=config,
        model=model,
        get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
        release_request_state=None,
    )


def _make_kv_cache_manager(max_tokens: int) -> KVCacheManager:
    model = DummyModel(torch.half)
    return KVCacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        tokens_per_block=_BEAM_TOKENS_PER_BLOCK,
        max_seq_len=_BEAM_MAX_SEQ_LEN,
        max_batch_size=_BEAM_BATCH_SIZE,
        max_beam_width=_BEAM_WIDTH,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )


def test_cuda_graph_warmup_request_fits_beam_search_kv_blocks() -> None:
    """The max-length warmup request must fit the blocks that beam search
    allocates per beam instead of failing inside the block manager."""
    factory = _make_factory(max_tokens=128)
    kv_cache_manager = _make_kv_cache_manager(max_tokens=128)
    resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
    try:
        total_free = kv_cache_manager.get_num_free_blocks()
        assert total_free == 16
        warmup_request = factory.create_cuda_graph_warmup_request(
            resource_manager, _BEAM_BATCH_SIZE, 0
        )
        with factory.release_batch_context(warmup_request, resource_manager) as batch:
            assert batch is not None
            assert len(batch.generation_requests) == _BEAM_BATCH_SIZE
            assert kv_cache_manager.get_num_free_blocks() == 0
        assert kv_cache_manager.get_num_free_blocks() == total_free
    finally:
        kv_cache_manager.shutdown()


def test_cuda_graph_warmup_request_passes_beam_width_to_kv_cache_managers() -> None:
    """Both the target and the draft KV cache capacity must be computed for
    the configured beam width."""
    factory = _make_factory(max_tokens=1024)
    target = _make_kv_cache_manager(max_tokens=1024)
    draft = _make_kv_cache_manager(max_tokens=1024)
    target.get_num_available_tokens = Mock(wraps=target.get_num_available_tokens)
    draft.get_num_available_tokens = Mock(wraps=draft.get_num_available_tokens)
    resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: target})
    try:
        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests."
            "get_draft_kv_cache_manager",
            return_value=draft,
        ):
            managers = (target, draft)
            total_free = [manager.get_num_free_blocks() for manager in managers]
            warmup_request = factory.create_cuda_graph_warmup_request(
                resource_manager, _BEAM_BATCH_SIZE, 0
            )
            with factory.release_batch_context(warmup_request, resource_manager) as batch:
                assert batch is not None
            for manager, free_blocks in zip(managers, total_free):
                manager.get_num_available_tokens.assert_called_once()
                kwargs = manager.get_num_available_tokens.call_args.kwargs
                assert kwargs["max_beam_width"] == _BEAM_WIDTH
                assert kwargs["batch_size"] == _BEAM_BATCH_SIZE
                assert manager.get_num_free_blocks() == free_blocks
    finally:
        target.shutdown()
        draft.shutdown()


def _make_vswa_kv_cache_manager(blocks_per_window: int) -> KVCacheManager:
    # One sliding-window layer and one full-attention layer. VSWA pools are
    # sized from max_gpu_total_bytes, split evenly between the windows.
    config = DummyModel(torch.half).config
    bytes_per_block = 2 * config.num_key_value_heads * config.head_dim * 2 * _BEAM_TOKENS_PER_BLOCK
    return KVCacheManager(
        KvCacheConfig(
            enable_block_reuse=False,
            max_attention_window=[_BEAM_MAX_SEQ_LEN // 2, _BEAM_MAX_SEQ_LEN],
            max_gpu_total_bytes=2 * blocks_per_window * bytes_per_block,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        tokens_per_block=_BEAM_TOKENS_PER_BLOCK,
        max_seq_len=_BEAM_MAX_SEQ_LEN,
        max_batch_size=_BEAM_BATCH_SIZE,
        max_beam_width=_BEAM_WIDTH,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )


def test_cuda_graph_warmup_request_skips_beam_search_batch_below_one_token() -> None:
    """add_dummy_requests does not check beam-search capacity for VSWA pools,
    so a batch size whose capacity is below one token must be skipped and the
    warmup requests already added must be freed."""
    blocks_per_window = 28
    factory = _make_factory(max_tokens=1024)
    factory._config = replace(factory._config, max_draft_loop_tokens=_BEAM_TOKENS_PER_BLOCK)
    kv_cache_manager = _make_vswa_kv_cache_manager(blocks_per_window)
    kv_cache_manager.add_dummy_requests = Mock(wraps=kv_cache_manager.add_dummy_requests)
    resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

    def free_blocks_per_window() -> dict[int, int]:
        return dict(kv_cache_manager.impl.get_kv_cache_stats().num_free_blocks_per_window_size)

    try:
        assert kv_cache_manager.is_vswa
        total_free = free_blocks_per_window()
        assert sorted(total_free.values()) == [blocks_per_window] * 2
        assert (
            factory.create_cuda_graph_warmup_request(resource_manager, _BEAM_BATCH_SIZE, 0) is None
        )
        # Only the filler requests were added; the longest one was skipped.
        kv_cache_manager.add_dummy_requests.assert_called_once()
        assert free_blocks_per_window() == total_free
    finally:
        kv_cache_manager.shutdown()
