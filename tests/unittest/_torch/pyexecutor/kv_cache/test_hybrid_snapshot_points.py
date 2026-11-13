# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent-state snapshot points of a hybrid model's context requests: chunking and counters.

The scheduler ends a context chunk at the next snapshot point, unless the cache manager marked
the request as able to take that snapshot inside the step: ``py_midchunk_snapshot_alignment``
lets the last snapshot point fall inside the final chunk, and ``py_midchunk_pass_point`` lets a
step pass one other point, ending at the following point at the latest. A model's cache manager
sets those marks (the Kimi K3 SM107 targets' manager, tested with the family). How a token
budget caps such a chunk is tested with the scheduler (``test_kv_cache_v2_scheduler.py``,
``TestMidchunkSnapshotBudget``).

``MambaHybridCacheManagerV2`` logs its recurrent snapshot and reuse counters at INFO, at most
once a minute.

Every test here runs on stubs, without a GPU.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BlockReusePolicy
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    mamba_cache_manager_v2 as mamba_cache_manager_v2_module,
)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    _get_forced_context_chunk_size,
    _is_forced_context_chunk_boundary,
)
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig

pytestmark = pytest.mark.cpu_only


def _sched_request(cur, prompt_len, points, alignment=None, branch_point=None):
    request = SimpleNamespace(
        context_current_position=cur,
        prompt_len=prompt_len,
        context_remaining_length=prompt_len - cur,
        expect_snapshot_points=points,
    )
    if alignment is not None:
        request.py_midchunk_snapshot_alignment = alignment
    if branch_point is not None:
        request.py_midchunk_pass_point = branch_point
    return request


@pytest.mark.parametrize(
    "cur,prompt_len,points,alignment,expected",
    [
        # Without the manager's opt-in the last point splits the chunk.
        (0, 300, [256], None, 256),
        (128, 300, [256], None, 128),
        # The last point inside the step no longer splits it.
        (0, 300, [256], 64, 300),
        (128, 300, [256], 64, 172),
        # Earlier points still split; the last one merges once it is next.
        (0, 700, [256, 640], 64, 256),
        (256, 700, [256, 640], 64, 444),
        # A point the manager cannot snapshot (misaligned to the step) splits.
        (0, 300, [200], 64, 200),
        (32, 300, [256], 64, 224),
        # A point at the prompt end is the end of the step anyway.
        (0, 256, [256], 64, 256),
        # No point left.
        (256, 300, [256], 64, 44),
    ],
)
def test_forced_chunk_size(cur, prompt_len, points, alignment, expected):
    request = _sched_request(cur, prompt_len, points, alignment)
    chunk = _get_forced_context_chunk_size(request)
    assert chunk == expected
    assert _is_forced_context_chunk_boundary(request, chunk)


@pytest.mark.parametrize(
    "cur,prompt_len,points,alignment,branch_point,expected",
    [
        # The step passes the branch point and ends at the following point.
        (0, 700, [256, 640], 64, 256, 640),
        (128, 700, [256, 640], 64, 256, 512),
        (0, 700, [256, 700], 64, 256, 700),
        (256, 700, [256, 384, 640], 64, 384, 384),
        # The branch point splits when it is not next, misaligned, or the
        # manager does not take it inside the step.
        (0, 700, [256, 384, 640], 64, 384, 256),
        (32, 700, [256, 640], 64, 256, 224),
        (0, 700, [256, 640], None, 256, 256),
        (0, 700, [256, 640], 64, None, 256),
        # Once passed, the last point merges as before.
        (256, 700, [256, 640], 64, 256, 444),
    ],
)
def test_forced_chunk_size_with_branch_point(
    cur, prompt_len, points, alignment, branch_point, expected
):
    request = _sched_request(cur, prompt_len, points, alignment, branch_point)
    chunk = _get_forced_context_chunk_size(request)
    assert chunk == expected
    assert _is_forced_context_chunk_boundary(request, chunk)


def _counters_stub():
    """A block-reuse ``MambaHybridCacheManagerV2`` with branch snapshots, minus its pools."""
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
    mgr.local_num_mamba_layers = 1
    mgr.block_reuse_policy = BlockReusePolicy.PER_REQUEST
    mgr.conversation_manager = None
    mgr._branch_snapshot_points = {}
    mgr._waiter_snapshot_points = None
    mgr._branch_snapshots_taken_total = 0
    mgr.kv_cache_map = {}
    # Logged just now, so the rate-limited INFO counters stay quiet.
    mgr._branch_snapshot_counters_logged_at = time.monotonic()
    mgr.tokens_per_block = 128
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=True),
    )
    mgr._snapshot_pruned_tokens_total = 0
    mgr._page_pruned_tokens_total = 0
    mgr._branch_snapshots_skipped_total = {}
    return mgr


def _context_request(request_id, cur, chunk, prompt_len, points):
    request = SimpleNamespace(
        py_request_id=request_id,
        is_dummy_request=False,
        is_dummy=False,
        is_disagg_generation_init_state=False,
        context_current_position=cur,
        context_chunk_size=chunk,
        prompt_len=prompt_len,
        expect_snapshot_points=points,
        tokens=list(range(1000, 1000 + prompt_len)),
        is_last_context_chunk=True,
        py_conversation_params=None,
    )
    request.context_remaining_length = prompt_len - cur
    return request


def test_inflight_prefix_counters():
    """Distinct waiter points are counted when requested and when committed;
    prefix waits are counted as the scheduler reports them."""
    mgr = _counters_stub()
    mgr.num_extra_kv_tokens = 0
    owner = _context_request(1, 0, 256, 1000, [1000])
    assert mgr.request_prefix_snapshot(owner, 520)
    assert mgr.request_prefix_snapshot(owner, 600)  # rounds down to the same point
    assert not mgr.request_prefix_snapshot(owner, 1100)  # past the prompt
    mgr.count_prefix_waits(2)
    mgr._count_snapshot_commit(owner, 256)
    mgr._count_snapshot_commit(owner, 512)
    assert mgr._format_inflight_prefix_counters() == (
        "inflight_prefix_deferrals=2 inflight_prefix_points_requested=1 "
        "inflight_prefix_points_taken=1"
    )
    assert "inflight_prefix_deferrals=2" in mgr._format_branch_snapshot_counters()


def test_snapshot_counters_are_logged_at_info_once_a_minute(monkeypatch):
    mgr = _counters_stub()
    mgr._branch_snapshot_counters_logged_at = float("-inf")
    clock = iter([1000.0, 1030.0, 1061.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    info = MagicMock()
    monkeypatch.setattr(mamba_cache_manager_v2_module.logger, "info", info)
    request = _context_request(1, 0, 1000, 1000, [1000])
    kv_cache = SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: 0,
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: 0,
        num_committed_tokens=0,
    )
    for _ in range(3):
        mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=999)
    # At most one INFO line a minute: the first two recordings are one minute apart.
    assert info.call_count == 2
