# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral Large 4 shared latent and independent indexer pages over native NIXL."""

from collections.abc import Sequence

import pytest
import torch
from kv_transfer_harness import run_kv_transfer_test

from tensorrt_llm._torch.attention.backends.sparse.mistral_large4 import MistralLarge4CacheManager
from tensorrt_llm._torch.configs.mistral_large4 import MistralLarge4Config
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

SOURCES = [35 if layer >= 39 and (layer - 39) % 4 == 0 else layer for layer in range(72)]
INDEX_LAYERS = [layer for layer in range(72) if layer < 17 or (layer - 17) % 4 >= 2]


def _managers(tp: int, pp: int, enable_dp: bool) -> list[MistralLarge4CacheManager]:
    assert pp == 1 and not enable_dp
    config = MistralLarge4Config(
        num_hidden_layers=72,
        kv_source_layers=SOURCES,
        index_sharing=["full" if layer in INDEX_LAYERS else "shared" for layer in range(72)],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
    )
    return [
        MistralLarge4CacheManager(
            kv_cache_config=KvCacheConfig(
                max_tokens=1024, host_cache_size=32 << 20, enable_block_reuse=False
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=72,
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=32,
            max_seq_len=512,
            max_batch_size=4,
            max_num_tokens=512,
            mapping=Mapping(world_size=tp, rank=rank, tp_size=tp),
            dtype=DataType.FP8,
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                index_n_heads=32, index_head_dim=128, index_topk=2048
            ),
            pretrained_config=config,
        )
        for rank in range(tp)
    ]


def _initialize(
    managers: Sequence[MistralLarge4CacheManager],
    tp: int,
    *,
    seed_base: int = 0,
    fill_random: bool = True,
) -> None:
    for manager in managers:
        latent = manager.get_latent_pool()
        for block in range(latent.shape[0]):
            for owner, layer in enumerate(manager.kv_owner_layers):
                # Nonzero, distinct page/layer patterns make omissions and wrong offsets visible.
                value = 1 + (seed_base + 17 * layer + block) % 251 if fill_random else 0
                latent[block, owner].fill_(value)
        for layer in INDEX_LAYERS:
            pool = manager.get_indexer_k_cache_buffers(layer)
            for block in range(manager.blocks_in_primary_pool):
                value = 1 + (seed_base + 31 * layer + block) % 251 if fill_random else 0
                pool[block * len(INDEX_LAYERS)].fill_(value)
        torch.cuda.current_stream(latent.device).synchronize()


def _verify(
    *,
    request_lengths: list[int],
    ctx_managers: Sequence[MistralLarge4CacheManager],
    gen_managers: Sequence[MistralLarge4CacheManager],
    ctx_tp: int,
    ctx_pp: int,
    gen_tp: int,
    gen_pp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    ctx_request_ids: list[int],
    gen_request_ids: list[int],
) -> None:
    source = ctx_managers[0]
    for target in gen_managers:
        assert len(target.kv_owner_layers) == 63 and len(INDEX_LAYERS) == 44
        used = set()
        for length, ctx_id, gen_id in zip(
            request_lengths, ctx_request_ids, gen_request_ids, strict=True
        ):
            count = (length + 31) // 32
            source_blocks = source.get_batch_cache_indices([ctx_id], raw_indices=True)[0][:count]
            target_blocks = target.get_batch_cache_indices([gen_id], raw_indices=True)[0][:count]
            assert len(source_blocks) == len(target_blocks) == count
            assert min(source_blocks + target_blocks) >= 0
            used.update(target_blocks)
            for src, dst in zip(source_blocks, target_blocks, strict=True):
                torch.testing.assert_close(
                    target.get_latent_pool()[dst], source.get_latent_pool()[src], rtol=0, atol=0
                )
                for layer in INDEX_LAYERS:
                    torch.testing.assert_close(
                        target.get_indexer_k_cache_buffers(layer)[dst * 44],
                        source.get_indexer_k_cache_buffers(layer)[src * 44],
                        rtol=0,
                        atol=0,
                    )
                for layer in range(39, 72, 4):
                    assert not target.has_buffer(layer, Role.KEY)
                    assert target.latent_pool_row(layer) == target.latent_pool_row(35)
        for block in set(range(target.blocks_in_primary_pool)) - used:
            assert not target.get_latent_pool()[block].any()
            for layer in INDEX_LAYERS:
                assert not target.get_indexer_k_cache_buffers(layer)[block * 44].any()


@pytest.mark.timeout(300)
def test_mistral_large4_shared_kv_transfer() -> None:
    """One TEP4 handoff covers exact/partial blocks and all latent/indexer bytes."""
    run_kv_transfer_test(
        ctx_tp=4,
        ctx_pp=1,
        gen_tp=4,
        gen_pp=1,
        ctx_enable_dp=False,
        gen_enable_dp=False,
        manager_factory=_managers,
        init_fn=_initialize,
        verify_fn=_verify,
    )
