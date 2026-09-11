# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor.engine.metadata import (
    build_attention_metadata,
    update_spec_metadata,
)

pytestmark = pytest.mark.cpu_only


class _AttentionMetadata:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _AttentionBackend:
    Metadata = _AttentionMetadata


def test_build_attention_metadata_resolves_model_derived_values() -> None:
    sparse_metadata_params = object()
    sparse_attention_config = SimpleNamespace(
        to_sparse_metadata_params=Mock(return_value=sparse_metadata_params)
    )
    pretrained_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=[4, 2],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    model_config = SimpleNamespace(
        pretrained_config=pretrained_config,
        sparse_attention_config=sparse_attention_config,
        enable_flash_mla=True,
    )
    runtime_features = AttentionRuntimeFeatures(cache_reuse=True)

    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=_AttentionBackend,
        attention_runtime_features=runtime_features,
        mapping=object(),
        cache_indirection=None,
    )

    assert metadata.num_heads_per_kv == 4
    assert metadata.enable_flash_mla
    assert metadata.enable_context_mla_with_cached_kv
    assert metadata.sparse_metadata_params is sparse_metadata_params
    sparse_attention_config.to_sparse_metadata_params.assert_called_once_with(
        pretrained_config=pretrained_config
    )


def test_build_attention_metadata_forwards_shared_and_cache_inputs() -> None:
    mapping = object()
    cache_indirection = object()
    kv_cache_manager = object()
    draft_kv_cache_manager = object()
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        sparse_attention_config=None,
        enable_flash_mla=True,
    )

    metadata = build_attention_metadata(
        model_config,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=2,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(),
        mapping=mapping,
        cache_indirection=cache_indirection,
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=draft_kv_cache_manager,
        enable_context_mla_with_cached_kv=True,
        num_heads_per_kv=4,
    )

    assert metadata.max_num_requests == 4
    assert metadata.max_num_tokens == 16
    assert metadata.max_num_sequences == 8
    assert metadata.mapping is mapping
    assert metadata.cache_indirection is cache_indirection
    assert metadata.kv_cache_manager is kv_cache_manager
    assert metadata.draft_kv_cache_manager is draft_kv_cache_manager
    assert metadata.enable_context_mla_with_cached_kv
    assert metadata.num_heads_per_kv == 4


def test_update_spec_metadata_handles_parallel_draft_and_dynamic_tree() -> None:
    spec_mode = SimpleNamespace(
        attention_need_spec_dec_mode=Mock(return_value=True),
        is_parallel_draft=Mock(return_value=True),
    )
    spec_metadata = SimpleNamespace(
        spec_dec_mode=spec_mode,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=True,
    )
    scheduled_requests = SimpleNamespace(
        batch_size=2,
        num_context_requests=1,
        context_requests=[object()],
        generation_requests=[object()],
    )
    attn_metadata = SimpleNamespace(update_spec_dec_param=Mock())
    spec_tree_manager = SimpleNamespace(
        use_dynamic_tree=True,
        slot_storage=SimpleNamespace(fill_all_slot_ids=Mock()),
    )

    update_spec_metadata(
        spec_metadata,
        scheduled_requests,
        attn_metadata,
        spec_tree_manager,
        runtime_draft_len=3,
        runtime_tokens_per_gen_step=4,
        is_draft_model=False,
        attention_backend=_AttentionBackend,
        original_max_draft_len=2,
        original_max_total_draft_tokens=6,
        spec_dec_max_total_draft_tokens=5,
    )

    assert spec_metadata.runtime_draft_len == 3
    assert spec_metadata.runtime_tokens_per_gen_step == 4
    spec_tree_manager.slot_storage.fill_all_slot_ids.assert_called_once_with(
        scheduled_requests.context_requests,
        scheduled_requests.generation_requests,
    )
    attn_metadata.update_spec_dec_param.assert_called_once_with(
        batch_size=2,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=True,
        max_draft_len=6,
        max_total_draft_tokens=6,
        spec_metadata=spec_metadata,
        spec_tree_manager=spec_tree_manager,
        num_contexts=1,
    )


def test_update_spec_metadata_uses_non_parallel_limits_for_draft_model() -> None:
    spec_mode = SimpleNamespace(
        attention_need_spec_dec_mode=Mock(return_value=False),
        is_parallel_draft=Mock(return_value=False),
    )
    spec_metadata = SimpleNamespace(
        spec_dec_mode=spec_mode,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
    )
    scheduled_requests = SimpleNamespace(
        batch_size=1,
        num_context_requests=0,
        context_requests=[],
        generation_requests=[object()],
    )
    attn_metadata = SimpleNamespace(update_spec_dec_param=Mock())
    spec_tree_manager = SimpleNamespace(
        use_dynamic_tree=True,
        slot_storage=SimpleNamespace(fill_all_slot_ids=Mock()),
    )

    update_spec_metadata(
        spec_metadata,
        scheduled_requests,
        attn_metadata,
        spec_tree_manager,
        runtime_draft_len=2,
        runtime_tokens_per_gen_step=3,
        is_draft_model=True,
        attention_backend=_AttentionBackend,
        original_max_draft_len=4,
        original_max_total_draft_tokens=8,
        spec_dec_max_total_draft_tokens=6,
    )

    spec_tree_manager.slot_storage.fill_all_slot_ids.assert_not_called()
    attn_metadata.update_spec_dec_param.assert_called_once_with(
        batch_size=1,
        is_spec_decoding_enabled=False,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        max_draft_len=4,
        max_total_draft_tokens=6,
        spec_metadata=spec_metadata,
        spec_tree_manager=spec_tree_manager,
        num_contexts=0,
    )
