# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix tokens lost to missing recurrent-state snapshots, in the V2 iteration stats."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import kv_cache_stats
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig


def _fake_kv_cache(matched: int, reused: int) -> SimpleNamespace:
    return SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: matched,
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: matched,
        num_committed_tokens=reused,
    )


def _fake_request(
    is_dummy: bool = False, is_disagg_generation_init_state: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=1,
        prompt_len=1 << 21,
        context_current_position=0,
        # Like an attention DP or CUDA graph padding dummy, which is not a dummy_request.
        is_dummy=is_dummy,
        is_dummy_request=False,
        is_disagg_generation_init_state=is_disagg_generation_init_state,
        expect_snapshot_points=[],
    )


def _manager(rank: int = 0, enable_branch_snapshot: bool = False) -> MambaHybridCacheManagerV2:
    """Bare manager carrying only what the loss accounting reads; no cache pools."""
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.local_num_mamba_layers = 1
    manager.is_estimating_kv_cache = False
    manager.mapping = SimpleNamespace(rank=rank)
    manager.kv_cache_config = KvCacheConfig(
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=enable_branch_snapshot)
    )
    manager._snapshot_pruned_tokens_total = 0
    manager._page_pruned_tokens_total = 0
    manager._prefix_matched_tokens_total = 0
    manager._iter_prefix_matched_tokens = 0
    manager._iter_prefix_snapshot_lost_tokens = 0
    manager._snapshot_loss_warned = False
    return manager


def test_counts_loss_without_branch_snapshots():
    manager = _manager()
    # Attention matched 20k tokens, the deepest snapshot was at 4k
    manager._record_branch_snapshot_point(_fake_request(), _fake_kv_cache(20_000, 4_000), 20_000)
    manager._record_branch_snapshot_point(_fake_request(), _fake_kv_cache(8_000, 8_000), 8_000)
    assert manager._iter_prefix_matched_tokens == 28_000
    assert manager._iter_prefix_snapshot_lost_tokens == 16_000
    assert manager._snapshot_pruned_tokens_total == 16_000
    # The attention page diagnostics stay off with the branch snapshot flag.
    assert manager._page_pruned_tokens_total == 0


def test_skips_resumed_dummy_disagg_and_estimation_lookups():
    manager = _manager()
    manager._record_branch_snapshot_point(_fake_request(), _fake_kv_cache(100, 0), None)
    manager._record_branch_snapshot_point(_fake_request(is_dummy=True), _fake_kv_cache(100, 0), 100)
    manager._record_branch_snapshot_point(
        _fake_request(is_disagg_generation_init_state=True), _fake_kv_cache(100, 0), 100
    )
    manager.is_estimating_kv_cache = True
    manager._record_branch_snapshot_point(_fake_request(), _fake_kv_cache(100, 0), 100)
    assert manager._prefix_matched_tokens_total == 0
    assert manager._snapshot_pruned_tokens_total == 0
    assert manager._iter_prefix_snapshot_lost_tokens == 0


@pytest.mark.parametrize(
    ("enable_branch_snapshot", "hint"),
    [(False, "enable_branch_snapshot"), (True, "periodic_snapshot_interval")],
)
def test_warns_once_above_threshold_without_iteration_stats(
    monkeypatch, enable_branch_snapshot, hint
):
    warnings = []
    monkeypatch.setattr(mamba_cache_manager_v2.logger, "warning", warnings.append)
    manager = _manager(enable_branch_snapshot=enable_branch_snapshot)
    matched = mamba_cache_manager_v2._SNAPSHOT_LOSS_MIN_MATCHED_TOKENS
    # No get_iteration_stats() call: the warning must not depend on stats being enabled.
    manager._record_prefix_loss(_fake_request(), _fake_kv_cache(matched, matched // 2))
    manager._record_prefix_loss(_fake_request(), _fake_kv_cache(matched, 0))
    assert len(warnings) == 1
    assert "50%" in warnings[0] and hint in warnings[0] and "pool_ratio" in warnings[0]


def test_no_warning_below_threshold_or_on_other_ranks(monkeypatch):
    warnings = []
    monkeypatch.setattr(mamba_cache_manager_v2.logger, "warning", warnings.append)
    matched = mamba_cache_manager_v2._SNAPSHOT_LOSS_MIN_MATCHED_TOKENS
    small_loss = _manager()
    small_loss._record_prefix_loss(_fake_request(), _fake_kv_cache(matched, matched - 1))
    other_rank = _manager(rank=1)
    other_rank._record_prefix_loss(_fake_request(), _fake_kv_cache(matched, 0))
    assert warnings == []


def test_serialized_only_for_recurrent_models():
    report = kv_cache_stats.KVCacheV2IterationStatsReport(by_window_size={}, by_pool_group={})
    stats = {}
    kv_cache_stats.append_kv_cache_iteration_stats(stats, report)
    assert "iterPrefixSnapshotLostTokens" not in stats
    report.prefix_matched_tokens = 28_000
    report.prefix_snapshot_lost_tokens = 16_000
    kv_cache_stats.append_kv_cache_iteration_stats(stats, report)
    assert stats["iterPrefixMatchedTokens"] == 28_000
    assert stats["iterPrefixSnapshotLostTokens"] == 16_000
