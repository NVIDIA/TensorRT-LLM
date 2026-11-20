# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix tokens lost to missing recurrent-state snapshots, in the V2 iteration stats."""

from types import SimpleNamespace
from typing import Optional

import pytest

from tensorrt_llm._torch.pyexecutor import kv_cache_stats
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BlockReusePolicy
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig, MambaStateConfig

_MIN_MATCHED = mamba_cache_manager_v2._SNAPSHOT_LOSS_MIN_MATCHED_TOKENS


def _fake_kv_cache(matched: int, reused: int, divergence: Optional[int] = None) -> SimpleNamespace:
    return SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: (
            matched if divergence is None else divergence
        ),
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: matched,
        num_committed_tokens=reused,
    )


def _fake_request(
    request_id: int = 1, is_dummy: bool = False, is_disagg_generation_init_state: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        prompt_len=1 << 21,
        context_current_position=0,
        # Like an attention DP or CUDA graph padding dummy, which is not a dummy_request.
        is_dummy=is_dummy,
        is_dummy_request=False,
        is_disagg_generation_init_state=is_disagg_generation_init_state,
        expect_snapshot_points=[],
    )


def _manager(
    rank: int = 0,
    enable_branch_snapshot: bool = False,
    periodic_snapshot_interval: int = 0,
    policy: BlockReusePolicy = BlockReusePolicy.PER_REQUEST,
) -> MambaHybridCacheManagerV2:
    """Bare manager carrying only what the loss accounting reads; no cache pools."""
    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.is_draft = False
    manager.enable_stats = False
    manager._request_stats_enabled_ids = set()
    manager.local_num_mamba_layers = 1
    manager.is_estimating_kv_cache = False
    manager.mapping = SimpleNamespace(rank=rank)
    manager.kv_cache_config = KvCacheConfig(
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=enable_branch_snapshot),
        mamba_state_config=MambaStateConfig(periodic_snapshot_interval=periodic_snapshot_interval),
    )
    manager.block_reuse_policy = policy
    manager._snapshot_pruned_tokens_total = 0
    manager._page_pruned_tokens_total = 0
    manager._context_fork_hints = {}
    manager._pending_prefix_loss = {}
    manager._prefix_matched_tokens_total = 0
    manager._iter_prefix_matched_tokens = 0
    manager._iter_prefix_snapshot_lost_tokens = 0
    manager._snapshot_loss_warned = False
    return manager


def _commit(manager: MambaHybridCacheManagerV2, *requests: SimpleNamespace) -> None:
    """Commit the stats of a batch holding these requests' first chunks."""
    manager.commit_scheduled_kv_cache_stats(SimpleNamespace(context_requests=list(requests)))


def _admit(
    manager: MambaHybridCacheManagerV2,
    matched: int,
    reused: int,
    request: Optional[SimpleNamespace] = None,
) -> None:
    """Look a request up and schedule its first chunk."""
    request = request or _fake_request()
    manager._record_prefix_loss(request, _fake_kv_cache(matched, reused))
    _commit(manager, request)


def test_counts_loss_without_branch_snapshots() -> None:
    manager = _manager()
    # Attention matched 20k tokens, the deepest snapshot was at 4k, and 4k more
    # shared tokens had their attention pages evicted.
    first, second = _fake_request(1), _fake_request(2)
    manager._record_branch_snapshot_point(first, _fake_kv_cache(20_000, 4_000, 24_000), 20_000)
    manager._record_branch_snapshot_point(second, _fake_kv_cache(8_000, 8_000), 8_000)
    assert manager._prefix_matched_tokens_total == 0
    _commit(manager, first, second)
    assert manager._iter_prefix_matched_tokens == 28_000
    assert manager._iter_prefix_snapshot_lost_tokens == 16_000
    assert manager._snapshot_pruned_tokens_total == 16_000
    assert manager._page_pruned_tokens_total == 4_000


def test_counts_a_refused_admission_once() -> None:
    manager = _manager()
    request = _fake_request()
    # Refused for budget or pages twice: each retry is a fresh lookup.
    for _ in range(3):
        manager._record_prefix_loss(request, _fake_kv_cache(24_000, 4_000, 30_000))
    assert manager._prefix_matched_tokens_total == 0
    assert manager._page_pruned_tokens_total == 0
    _commit(manager, request)
    # Later chunks of the same request add nothing.
    _commit(manager, request)
    assert manager._prefix_matched_tokens_total == 24_000
    assert manager._iter_prefix_matched_tokens == 24_000
    assert manager._iter_prefix_snapshot_lost_tokens == 20_000
    assert manager._page_pruned_tokens_total == 6_000


@pytest.mark.parametrize(
    ("draft_match", "target_match", "matched", "lost"),
    [
        # Attention matched through 192, the target's last snapshot is at 64.
        (192, 64, 192, 128),
        # The draft matched only 64; the target's snapshot at 192 is not lost.
        (64, 192, 192, 0),
    ],
)
def test_measures_a_paired_claim_against_the_uncapped_probes(
    draft_match: int, target_match: int, matched: int, lost: int
) -> None:
    manager = _manager()
    request = _fake_request()
    common = min(draft_match, target_match)
    manager.note_context_fork(request, draft_match, target_match)
    # The claim stops the lookup at the common depth, which it fully reuses.
    manager._record_branch_snapshot_point(request, _fake_kv_cache(common, common), common)
    _commit(manager, request)
    assert manager._context_fork_hints == {}
    assert manager._iter_prefix_matched_tokens == matched
    assert manager._iter_prefix_snapshot_lost_tokens == lost


def test_skips_resumed_dummy_disagg_and_estimation_lookups() -> None:
    manager = _manager()
    lookups = [
        (_fake_request(1), None),
        (_fake_request(2, is_dummy=True), 100),
        (_fake_request(3, is_disagg_generation_init_state=True), 100),
    ]
    for request, num_lookup_tokens in lookups:
        manager._record_branch_snapshot_point(request, _fake_kv_cache(100, 0), num_lookup_tokens)
    manager.is_estimating_kv_cache = True
    manager._record_branch_snapshot_point(_fake_request(4), _fake_kv_cache(100, 0), 100)
    _commit(manager, *(_fake_request(request_id) for request_id in (1, 2, 3, 4)))
    assert manager._prefix_matched_tokens_total == 0
    assert manager._snapshot_pruned_tokens_total == 0
    assert manager._iter_prefix_matched_tokens == 0
    assert manager._iter_prefix_snapshot_lost_tokens == 0


@pytest.mark.parametrize(
    ("enable_branch_snapshot", "periodic_snapshot_interval", "policy", "hint"),
    [
        (False, 0, BlockReusePolicy.PER_REQUEST, "try 'kv_cache_config.block_reuse_config"),
        (True, 0, BlockReusePolicy.PER_REQUEST, "a positive 'kv_cache_config.mamba_state_config"),
        (True, 64, BlockReusePolicy.PER_REQUEST, "a smaller 'kv_cache_config.mamba_state_config"),
        (True, 0, BlockReusePolicy.PER_CONVERSATION, None),
    ],
)
def test_warns_once_above_threshold_without_iteration_stats(
    monkeypatch: pytest.MonkeyPatch,
    enable_branch_snapshot: bool,
    periodic_snapshot_interval: int,
    policy: BlockReusePolicy,
    hint: Optional[str],
) -> None:
    warnings = []
    monkeypatch.setattr(mamba_cache_manager_v2.logger, "warning", warnings.append)
    manager = _manager(
        enable_branch_snapshot=enable_branch_snapshot,
        periodic_snapshot_interval=periodic_snapshot_interval,
        policy=policy,
    )
    # No get_iteration_stats() call: the warning must not depend on stats being enabled.
    _admit(manager, _MIN_MATCHED, _MIN_MATCHED // 2, _fake_request(1))
    _admit(manager, _MIN_MATCHED, 0, _fake_request(2))
    assert len(warnings) == 1
    assert "50%" in warnings[0] and "pool_ratio" in warnings[0]
    if hint is None:
        assert "diverge" not in warnings[0]
    else:
        assert hint in warnings[0]


def test_no_warning_below_threshold_or_on_other_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings = []
    monkeypatch.setattr(mamba_cache_manager_v2.logger, "warning", warnings.append)
    _admit(_manager(), _MIN_MATCHED, _MIN_MATCHED - 1)
    # Every matched token lost, but too few matched to judge.
    _admit(_manager(), _MIN_MATCHED - 1, 0)
    _admit(_manager(rank=1), _MIN_MATCHED, 0)
    # Looked up but never scheduled.
    _manager()._record_prefix_loss(_fake_request(), _fake_kv_cache(_MIN_MATCHED, 0))
    assert warnings == []


def test_serialized_only_for_recurrent_models() -> None:
    report = kv_cache_stats.KVCacheV2IterationStatsReport(by_window_size={}, by_pool_group={})
    stats = {}
    kv_cache_stats.append_kv_cache_iteration_stats(stats, report)
    assert "iterPrefixSnapshotLostTokens" not in stats
    report.prefix_matched_tokens = 28_000
    report.prefix_snapshot_lost_tokens = 16_000
    kv_cache_stats.append_kv_cache_iteration_stats(stats, report)
    assert stats["iterPrefixMatchedTokens"] == 28_000
    assert stats["iterPrefixSnapshotLostTokens"] == 16_000
