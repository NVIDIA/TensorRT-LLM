# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime Indexer maps must address the manager's physical payload and scales."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import (
    DSACacheManagerV2,
    DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import _fused_dsa_meta_enabled
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    DeepSeekV4SparseAttentionConfig,
    KvCacheConfig,
)
from tensorrt_llm.mapping import Mapping


@contextmanager
def _prepared_metadata(
    indexer_dtype: str,
    cached_tokens: list[int],
    seq_lens: list[int],
    *,
    context: bool,
    compressed: bool,
) -> Iterator[DSAtrtllmAttentionMetadata]:
    config_cls = DeepSeekV4SparseAttentionConfig if compressed else DeepSeekSparseAttentionConfig
    sparse_config = config_cls(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        indexer_k_dtype=indexer_dtype,
        skip_indexer_for_short_seqs=False,
        **({"compress_ratios": [4]} if compressed else {}),
    )
    manager_cls = DeepseekV4CacheManager if compressed else DSACacheManagerV2
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    manager = manager_cls(
        kv_cache_config=KvCacheConfig(
            max_tokens=1024,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=128,
        max_seq_len=512,
        max_batch_size=2,
        max_input_len=512,
        mapping=mapping,
        dtype=DataType.BF16,
        vocab_size=128,
        max_num_tokens=1024,
        sparse_attention_config=sparse_config,
    )
    try:
        kv_lens = [cached + length for cached, length in zip(cached_tokens, seq_lens)]
        assert manager.add_dummy_requests([0, 1], kv_lens, is_gen=not context) is not None
        metadata_cls = (
            DeepseekV4TrtllmAttentionMetadata if compressed else DSAtrtllmAttentionMetadata
        )
        metadata = metadata_cls(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            request_ids=[0, 1],
            max_num_requests=2,
            num_contexts=2 if context else 0,
            prompt_lens=kv_lens if context else cached_tokens,
            max_num_tokens=sum(seq_lens),
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_tokens),
            enable_context_mla_with_cached_kv=True,
            mapping=mapping,
            sparse_metadata_params=sparse_config.to_sparse_metadata_params(),
        )
        metadata.prepare()
        assert manager.quant_block_size == (32 if compressed and indexer_dtype == "fp4" else 128)
        assert metadata.indexer_quant_block_size == 128
        yield metadata
    finally:
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("indexer_dtype", ["fp8", "fp4"])
@pytest.mark.parametrize("phase", ["context", "decode-eager", "decode-fused"])
def test_v4_runtime_maps_gather_physical_cache_bytes(
    indexer_dtype: str, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, fused = phase == "context", phase == "decode-fused"
    monkeypatch.setenv("TRTLLM_FUSED_DSA_METADATA", "1" if fused else "0")
    _fused_dsa_meta_enabled.cache_clear()
    cached, lengths = [12, 20], [5, 6] if context else [1, 1]
    try:
        with _prepared_metadata(
            indexer_dtype, cached, lengths, context=context, compressed=True
        ) as metadata:
            cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(0)
            data_width = 64 if indexer_dtype == "fp4" else 128
            assert cache.shape[1:] == (32, 1, data_width + 4)
            generator = torch.Generator(device=cache.device).manual_seed(41)
            cache.copy_(
                torch.randint(
                    256, cache.shape, device=cache.device, dtype=torch.uint8, generator=generator
                )
            )
            pages = cache.flatten(1)
            payload = pages[:, : 32 * data_width].reshape(-1, 32, data_width)
            scales = pages[:, 32 * data_width :].reshape(-1, 32, 4)
            requests = torch.repeat_interleave(
                torch.arange(2, device=cache.device), torch.tensor(lengths, device=cache.device)
            )
            positions = torch.cat(
                [
                    torch.arange(start, start + length, device=cache.device)
                    for start, length in zip(cached, lengths)
                ]
            )
            blocks = metadata.indexer_k_cache_block_offsets[requests, positions // 32].long()
            assert blocks.gt(0).any()
            expected_data = payload[blocks, positions % 32].clone()
            expected_scales = scales[blocks, positions % 32].clone()
            gather_maps = (
                (
                    metadata.slot_mapping_fp8_fullkv.clone(),
                    metadata.slot_mapping_scale_fullkv.clone(),
                )
                if context
                else None
            )
            metadata.slot_mapping_fp8.fill_(-1)
            metadata.slot_mapping_scale.fill_(-1)

            metadata.on_update_kv_lens()

            assert getattr(metadata, "_fused_dsa_meta_armed", False) is fused
            for offsets, width in (
                (metadata.slot_mapping_fp8, data_width),
                (metadata.slot_mapping_scale, 4),
            ):
                assert offsets.ge(0).all() and (offsets + width).le(cache.numel()).all()
            data, scale = torch.ops.trtllm.indexer_k_cache_gather_op(
                cache,
                metadata.slot_mapping_fp8,
                metadata.slot_mapping_scale,
                0,
                sum(lengths),
                data_width,
            )
            torch.testing.assert_close(data.view(torch.uint8), expected_data, rtol=0, atol=0)
            torch.testing.assert_close(
                scale.view(torch.uint8).reshape(-1, 4), expected_scales, rtol=0, atol=0
            )
            if gather_maps is not None:
                # Compressed context gathers have independent maps; scatter positions are raw.
                torch.testing.assert_close(
                    metadata.slot_mapping_fp8_fullkv, gather_maps[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    metadata.slot_mapping_scale_fullkv, gather_maps[1], rtol=0, atol=0
                )
    finally:
        _fused_dsa_meta_enabled.cache_clear()


@skip_pre_blackwell
@pytest.mark.parametrize("indexer_dtype", ["fp8", "fp4"])
@pytest.mark.parametrize("cached", [False, True], ids=["fresh", "cached"])
def test_dsa_context_gather_alias_survives_runtime_refresh(
    indexer_dtype: str, cached: bool
) -> None:
    with _prepared_metadata(
        indexer_dtype,
        [127, 129] if cached else [0, 0],
        [130, 131],
        context=True,
        compressed=False,
    ) as metadata:
        assert (metadata.slot_mapping_fp8_fullkv is metadata.slot_mapping_fp8) is (not cached)
        assert (metadata.slot_mapping_scale_fullkv is metadata.slot_mapping_scale) is (not cached)
        names = (
            "slot_mapping_fp8",
            "slot_mapping_scale",
            "slot_mapping_fp8_fullkv",
            "slot_mapping_scale_fullkv",
        )
        expected = [getattr(metadata, name).clone() for name in names]

        metadata.on_update_kv_lens()

        assert (metadata.slot_mapping_fp8_fullkv is metadata.slot_mapping_fp8) is (not cached)
        assert (metadata.slot_mapping_scale_fullkv is metadata.slot_mapping_scale) is (not cached)
        for name, prepared in zip(names, expected):
            torch.testing.assert_close(getattr(metadata, name), prepared, rtol=0, atol=0)
