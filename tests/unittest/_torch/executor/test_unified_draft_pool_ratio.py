# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit pool ratios of the unified DSpark draft cache.

Storage orders pool groups by cache domain first, and standalone-draft groups sort before target
groups, so the draft share has to lead the ratio list that the manager hands to storage.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv_runtime

CacheDomain = getattr(kv_runtime, "CacheDomain", None)
if CacheDomain is None:
    pytest.skip("cache domains need the C++ KV cache manager backend", allow_module_level=True)

AttentionLayerConfig = kv_runtime.AttentionLayerConfig
BufferConfig = kv_runtime.BufferConfig
DataRole = kv_runtime.DataRole
GpuCacheTierConfig = kv_runtime.GpuCacheTierConfig
KVCacheManagerConfig = kv_runtime.KVCacheManagerConfig
LayerId = kv_runtime.LayerId
SsmLayerConfig = kv_runtime.SsmLayerConfig

_QUOTA = 1 << 30
_TOKENS_PER_BLOCK = 64
_MAX_TOKENS = 4096
_ROLE = DataRole("kv")


def _attention(layer_id, *, window=None, domain=None):
    kwargs = {} if domain is None else {"cache_domain": domain}
    return AttentionLayerConfig(
        layer_id=LayerId(layer_id),
        buffers=[BufferConfig(role=_ROLE, size=4096)],
        sliding_window_size=window,
        **kwargs,
    )


def _ssm(layer_id):
    return SsmLayerConfig(layer_id=LayerId(layer_id), buffers=[BufferConfig(role=_ROLE, size=4096)])


def _draft_layers(first_layer_id=10, count=2):
    return [
        _attention(first_layer_id + i, domain=CacheDomain.STANDALONE_DRAFT) for i in range(count)
    ]


def _config(layers, pool_ratio):
    return KVCacheManagerConfig(
        tokens_per_block=_TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=_QUOTA)],
        layers=layers,
        max_util_for_resume=1.0,
        initial_pool_ratio=pool_ratio,
        # The native config refuses recurrent layers without it.
        commit_min_snapshot=any(isinstance(layer, SsmLayerConfig) for layer in layers),
    )


def _layer_key(layer):
    return (
        type(layer).__name__,
        int(layer.layer_id),
        getattr(layer, "cache_domain", None),
        getattr(layer, "sliding_window_size", None),
        getattr(layer, "num_sink_tokens", None),
        [(buffer.role, buffer.size, buffer.tokens_per_block_override) for buffer in layer.buffers],
    )


def _tier_key(tier):
    return (tier.tier, tier.quota)


def _manager(*, draft_layer_ids=(10, 11), draft_quota=_QUOTA // 4, total_quota=_QUOTA):
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.draft_layer_ids = list(draft_layer_ids)
    manager.max_num_tokens = None
    manager.max_seq_len = _MAX_TOKENS
    manager._get_max_tokens_from_quota = lambda quota: _MAX_TOKENS
    manager._get_quota_from_max_tokens = lambda tokens: total_quota
    manager._get_draft_quota_from_max_tokens = lambda tokens: draft_quota
    return manager


# Two target layer groups each: full + sliding-window attention, and recurrent + full attention.
_TARGET_LAYOUTS = {
    "attention_full_and_window": [_attention(0), _attention(1, window=128)],
    "recurrent_and_attention": [_ssm(0), _attention(1)],
}


def test_standalone_draft_sorts_before_target_in_storage():
    assert CacheDomain.STANDALONE_DRAFT.value < CacheDomain.TARGET.value


@pytest.mark.parametrize("layout", sorted(_TARGET_LAYOUTS))
def test_draft_share_leads_the_target_ratios(layout):
    config = _config(_TARGET_LAYOUTS[layout] + _draft_layers(), [0.75, 0.25])

    result = _manager()._configure_unified_draft_pool_ratio(config)

    assert result.initial_pool_ratio == pytest.approx([0.25, 0.75 * 0.75, 0.25 * 0.75])


def test_target_ratios_keep_their_proportions_and_the_total_is_one():
    config = _config(_TARGET_LAYOUTS["recurrent_and_attention"] + _draft_layers(), [0.9, 0.1])

    result = _manager(draft_quota=_QUOTA // 8)._configure_unified_draft_pool_ratio(config)

    draft, first, second = result.initial_pool_ratio
    assert draft == pytest.approx(0.125)
    assert first / second == pytest.approx(9.0)
    assert draft + first + second == pytest.approx(1.0)


def test_only_the_ratio_list_changes():
    config = _config(_TARGET_LAYOUTS["attention_full_and_window"] + _draft_layers(), [0.5, 0.5])

    result = _manager()._configure_unified_draft_pool_ratio(config)

    # The native layer and tier configs compare by identity, and a config hands out new objects
    # when it is copied, so compare what they hold.
    assert [_layer_key(layer) for layer in result.layers] == [
        _layer_key(layer) for layer in config.layers
    ]
    assert result.tokens_per_block == config.tokens_per_block
    assert [_tier_key(tier) for tier in result.cache_tiers] == [
        _tier_key(tier) for tier in config.cache_tiers
    ]
    assert config.initial_pool_ratio == [0.5, 0.5]


def test_without_draft_layers_the_config_is_untouched():
    config = _config(_TARGET_LAYOUTS["attention_full_and_window"], [0.5, 0.5])

    assert _manager(draft_layer_ids=())._configure_unified_draft_pool_ratio(config) is config


def test_without_an_explicit_ratio_the_config_is_untouched():
    config = _config(_TARGET_LAYOUTS["attention_full_and_window"] + _draft_layers(), None)

    assert _manager()._configure_unified_draft_pool_ratio(config) is config


def test_ratio_must_cover_exactly_the_target_layer_groups():
    config = _config(_TARGET_LAYOUTS["attention_full_and_window"] + _draft_layers(), [1.0])

    with pytest.raises(ValueError, match="one entry per target layer group"):
        _manager()._configure_unified_draft_pool_ratio(config)


def test_quota_too_small_for_the_reserves_is_rejected():
    config = _config(_TARGET_LAYOUTS["attention_full_and_window"] + _draft_layers(), [0.5, 0.5])

    with pytest.raises(ValueError, match="cannot hold the target and draft reserves"):
        _manager(total_quota=_QUOTA + 1)._configure_unified_draft_pool_ratio(config)


@pytest.mark.parametrize("draft_quota", [0, _QUOTA])
def test_draft_share_must_leave_room_for_both_pools(draft_quota):
    config = _config(_TARGET_LAYOUTS["attention_full_and_window"] + _draft_layers(), [0.5, 0.5])

    with pytest.raises(ValueError, match="must leave room"):
        _manager(draft_quota=draft_quota)._configure_unified_draft_pool_ratio(config)
