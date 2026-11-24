# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""KDA mid-chunk recurrent-state snapshots: scheduler and cache-manager side.

With ``TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1`` the last snapshot point of a context
request no longer splits its context chunk, and neither does its branch point
(``block_reuse_config.enable_branch_snapshot``, where it leaves the cached
prefix). ``KimiK3HybridCacheManager`` reserves a pending SSM page for the step,
the KDA layers of the K3 Rubin package fetch the plan
(``get_kda_midchunk_snapshots``) and write the snapshot into that page, and the
context update commits the prompt up to the point with it before the usual
end-of-step update. A planned snapshot whose plan was not fetched is released,
never committed.

The stub tests need no GPU. Tests on a real manager need the pending SSM
snapshot pages of the KV cache manager V2 C++ backend; the KDA layer tests also
need the K3 Rubin CuTe DSL prefill.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.modules.kimi_kda import cache_manager as kda_cache_manager
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KimiK3HybridCacheManager,
    _KDAMidchunkSnapshot,
)
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BlockReusePolicy
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    _get_forced_context_chunk_size,
    _is_forced_context_chunk_boundary,
)
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig, MambaStateConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_pending_ssm_pages = pytest.mark.skipif(
    not hasattr(_KVCache, "acquire_ssm_snapshot_slot"),
    reason="rebuild needed: the KV cache manager V2 C++ backend of this build has no "
    "pending SSM snapshot pages",
)


def _private_kda_module(name):
    """A module of the K3 Rubin KDA package, whose prefill stores mid-chunk snapshots."""
    return pytest.importorskip(f"{kda_cache_manager._MIDCHUNK_SNAPSHOT_KDA_PACKAGE}.{name}")


def _enable_midchunk_snapshots(monkeypatch):
    """TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 with a snapshot-capable prefill; set before construction."""
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    kernels = _private_kda_module("_kda_kernels")
    monkeypatch.setattr(kernels, "kda_prefill_supports_state_snapshots", lambda: True)


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
        # A point the KDA kernel cannot snapshot (misaligned to the step) splits.
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


def test_budget_capped_chunk_never_passes_the_point():
    """Capped chunks are rounded down to the unit, so they end at or before the point."""
    unit = 128
    for prompt_len in (300, 1000, 4099):
        point = prompt_len // unit * unit
        for cur in range(0, point, unit):
            request = _sched_request(cur, prompt_len, [point], 64)
            chunk = _get_forced_context_chunk_size(request)
            assert chunk == prompt_len - cur
            for budget in range(unit, chunk, unit):
                assert cur + budget // unit * unit <= point


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


def test_budget_capped_chunk_with_branch_point_never_passes_the_next_point():
    unit = 128
    branch, following, prompt_len = 1024, 2048, 2100
    for cur in range(0, branch, unit):
        request = _sched_request(cur, prompt_len, [branch, following], 64, branch)
        chunk = _get_forced_context_chunk_size(request)
        assert cur + chunk == following
        for budget in range(unit, chunk, unit):
            end = cur + budget // unit * unit
            assert end <= following
            # Inside a capped step the branch point is the only snapshot point.
            assert len([p for p in (branch, following) if cur < p < end]) <= 1


def _manager_stub(local_num_mamba_layers=1, alignment=64):
    mgr = object.__new__(KimiK3HybridCacheManager)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
    mgr.local_num_mamba_layers = local_num_mamba_layers
    mgr.block_reuse_policy = BlockReusePolicy.PER_REQUEST
    mgr.conversation_manager = None
    mgr._branch_snapshot_points = {}
    mgr._requested_snapshot_points = {}
    mgr._inflight_prefix = None
    mgr._branch_snapshots_taken_total = 0
    mgr._kda_replay = None
    mgr._kda_midchunk_snapshot_alignment = alignment
    mgr._kda_midchunk_snapshots = {}
    mgr._kda_midchunk_handed_out = set()
    mgr._kda_midchunk_snapshots_taken_total = 0
    mgr._kda_midchunk_snapshots_skipped_total = 0
    mgr._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]
    mgr.kv_cache_map = {}
    # No cache levels: the INFO counters line reports pool occupancy through impl.
    mgr.impl = SimpleNamespace(cache_tier_list=[], get_storage_statistics=lambda level: [])
    return mgr


def _context_request(request_id, cur, chunk, prompt_len, points, alignment=64):
    tokens = list(range(1000, 1000 + prompt_len))
    request = SimpleNamespace(
        py_request_id=request_id,
        is_dummy_request=False,
        is_dummy=False,
        context_current_position=cur,
        context_chunk_size=chunk,
        prompt_len=prompt_len,
        expect_snapshot_points=points,
        py_midchunk_snapshot_alignment=alignment,
        get_tokens=lambda beam_idx: tokens,
        is_last_context_chunk=True,
        py_conversation_params=None,
    )
    request.context_remaining_length = prompt_len - cur
    return request, tokens


def _mock_cache(events, slot=40, num_committed=0, history=0):
    kv_cache = SimpleNamespace(
        is_active=True,
        num_committed_tokens=num_committed,
        history_length=history,
        capacity=10_000,
        enable_swa_scratch_reuse=True,
        pending_ssm_snapshot_slot=None,
    )

    def acquire():
        events.append("acquire")
        kv_cache.pending_ssm_snapshot_slot = slot
        return slot

    def release():
        events.append("release")
        kv_cache.pending_ssm_snapshot_slot = None

    def resize(capacity, history_length=None):
        events.append(("resize", capacity, history_length))
        if history_length is not None:
            kv_cache.history_length = history_length
        return True

    def commit(tokens, ssm_from_pending=False):
        events.append(("commit", len(tokens), kv_cache.num_committed_tokens, ssm_from_pending))
        kv_cache.num_committed_tokens += len(tokens)
        if ssm_from_pending:
            kv_cache.pending_ssm_snapshot_slot = None

    kv_cache.acquire_ssm_snapshot_slot = MagicMock(side_effect=acquire)
    kv_cache.release_ssm_snapshot_slot = MagicMock(side_effect=release)
    kv_cache.resize = MagicMock(side_effect=resize)
    kv_cache.commit = MagicMock(side_effect=commit)
    kv_cache.stop_committing = MagicMock(side_effect=lambda: events.append("stop"))
    return kv_cache


def _committed_from_pending(events):
    return any(isinstance(e, tuple) and e[0] == "commit" and e[3] for e in events)


def _planned_one_step(mgr, events, request_id=7):
    """Plan the snapshot at 256 inside one step over a 300-token prompt."""
    request, tokens = _context_request(request_id, 0, 300, 300, [256])
    kv_cache = _mock_cache(events)
    mgr.kv_cache_map[request_id] = kv_cache
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr._prepare_kda_midchunk_snapshots(batch.context_requests)
    return request, tokens, kv_cache, batch


def _finish_step(request):
    request.context_current_position = request.prompt_len
    request.context_remaining_length = 0


@pytest.mark.parametrize(
    "cur,chunk,prompt_len,points,alignment,expected",
    [
        (0, 300, 300, [256], 64, (256, 256)),
        (128, 172, 300, [256], 64, (256, 128)),
        (0, 256, 300, [256], 64, None),  # the step ends at the point
        (0, 300, 300, [200], 64, None),  # misaligned
        # Two points inside the step (one requested after scheduling): the deepest.
        (0, 700, 700, [256, 640], 64, (640, 640)),
        (0, 700, 700, [256, 600], 64, (256, 256)),  # the deepest is misaligned
        (0, 640, 700, [256, 640], 64, (256, 256)),  # branch point, step ends at the next
        (0, 300, 300, [256], None, None),  # request not opted in
    ],
)
def test_plan_midchunk_snapshot(cur, chunk, prompt_len, points, alignment, expected):
    mgr = _manager_stub()
    request, _ = _context_request(1, cur, chunk, prompt_len, points, alignment)
    assert mgr._plan_kda_midchunk_snapshot(request) == expected


class _KVCacheWithPendingPages:
    acquire_ssm_snapshot_slot = None


def test_alignment_is_off_by_default_and_for_draft_or_no_reuse(monkeypatch):
    monkeypatch.setattr(kda_cache_manager, "_KVCache", _KVCacheWithPendingPages)
    mgr = _manager_stub(alignment=None)
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", False)
    assert mgr._get_kda_midchunk_snapshot_alignment() is None
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    mgr.is_draft = True
    assert mgr._get_kda_midchunk_snapshot_alignment() is None
    mgr.is_draft = False
    mgr.enable_block_reuse = False
    assert mgr._get_kda_midchunk_snapshot_alignment() is None


def test_alignment_requires_pending_ssm_pages(monkeypatch):
    """A C++ backend without pending SSM pages keeps it off (checked before any import)."""
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    monkeypatch.setattr(kda_cache_manager, "_KVCache", object)
    monkeypatch.setattr(
        kda_cache_manager, "_MIDCHUNK_SNAPSHOT_KDA_PACKAGE", "tensorrt_llm._no_such_kda_package"
    )
    assert _manager_stub(alignment=None)._get_kda_midchunk_snapshot_alignment() is None


def test_alignment_requires_the_snapshot_prefill_package(monkeypatch):
    """Main's own KDA layers never fetch a plan: without the K3 Rubin package it stays off."""
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    monkeypatch.setattr(kda_cache_manager, "_KVCache", _KVCacheWithPendingPages)
    monkeypatch.setattr(
        kda_cache_manager, "_MIDCHUNK_SNAPSHOT_KDA_PACKAGE", "tensorrt_llm._no_such_kda_package"
    )
    assert _manager_stub(alignment=None)._get_kda_midchunk_snapshot_alignment() is None


def test_alignment_follows_the_snapshot_prefill_kernel(monkeypatch):
    kernels = _private_kda_module("_kda_kernels")
    metadata = _private_kda_module("kimi_k3_mamba_metadata")
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    monkeypatch.setattr(kda_cache_manager, "_KVCache", _KVCacheWithPendingPages)
    mgr = _manager_stub(alignment=None)
    monkeypatch.setattr(kernels, "kda_prefill_supports_state_snapshots", lambda: True)
    assert mgr._get_kda_midchunk_snapshot_alignment() == metadata.KDA_PREFILL_CHUNK_SIZE == 64
    monkeypatch.setattr(kernels, "kda_prefill_supports_state_snapshots", lambda: False)
    assert mgr._get_kda_midchunk_snapshot_alignment() is None


def test_prepare_expect_snapshot_points_sets_alignment(monkeypatch):
    mgr = _manager_stub()
    monkeypatch.setattr(
        MambaHybridCacheManagerV2, "prepare_expect_snapshot_points", lambda self, requests: None
    )
    real, _ = _context_request(1, 0, 300, 300, [256])
    dummy, _ = _context_request(2, 0, 300, 300, [256])
    dummy.is_dummy_request = True
    mgr.prepare_expect_snapshot_points([real, dummy])
    assert real.py_midchunk_snapshot_alignment == 64
    assert dummy.py_midchunk_snapshot_alignment is None

    mgr._kda_midchunk_snapshot_alignment = None
    mgr.prepare_expect_snapshot_points([real])
    assert real.py_midchunk_snapshot_alignment is None


def test_one_step_context_commits_pending_page_before_end_update():
    mgr = _manager_stub()
    events = []
    request, tokens = _context_request(7, 128, 172, 300, [256])
    kv_cache = _mock_cache(events, num_committed=128, history=128)
    mgr.kv_cache_map[7] = kv_cache
    batch = ScheduledRequests()
    batch.append_context_request(request)

    mgr._prepare_kda_midchunk_snapshots(batch.context_requests)
    assert mgr._kda_midchunk_snapshots[7] == _KDAMidchunkSnapshot(256, 128, 40)
    assert mgr.get_kda_midchunk_snapshots([3, 7]) == [(1, 40, 128)]
    assert mgr.get_kda_midchunk_snapshots([3]) == []

    # The step ran: the context reached the prompt end.
    _finish_step(request)
    mgr.update_context_resources(batch)

    assert events == [
        "acquire",
        ("resize", None, 256),
        ("commit", 128, 128, True),
        ("resize", None, 300),
        "stop",
    ]
    kv_cache.commit.assert_called_once_with(tokens[128:256], ssm_from_pending=True)
    assert mgr._kda_midchunk_snapshots == {}
    assert mgr._kda_midchunk_handed_out == set()
    assert mgr._kda_midchunk_snapshots_taken_total == 1
    assert mgr._kda_midchunk_snapshots_skipped_total == 0


def test_branch_point_snapshot_is_counted():
    mgr = _manager_stub()
    events = []
    mgr._branch_snapshot_points[7] = 256
    request, _, _, batch = _planned_one_step(mgr, events)
    assert mgr.get_kda_midchunk_snapshots([7]) == [(0, 40, 256)]
    _finish_step(request)
    mgr.update_context_resources(batch)
    assert mgr._kda_midchunk_snapshots_taken_total == 1
    assert mgr._branch_snapshots_taken_total == 1


@pytest.mark.parametrize(
    "cur,alignment,branch_enabled,dummy,expected",
    [
        (0, 64, True, False, 256),
        (256, 64, True, False, None),  # already passed
        (0, None, True, False, None),  # no mid-chunk snapshots
        (0, 64, False, False, None),  # TRTLLM_KDA_MIDCHUNK_BRANCH_SNAPSHOT=0
        (0, 64, True, True, None),
    ],
)
def test_apply_branch_snapshot_point_marks_midchunk_pass_point(
    monkeypatch, cur, alignment, branch_enabled, dummy, expected
):
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_BRANCH_SNAPSHOT", branch_enabled)
    mgr = _manager_stub(alignment=alignment)
    mgr.tokens_per_block = 128
    mgr._branch_snapshot_points[7] = 256
    request, _ = _context_request(7, cur, 700 - cur, 700, [640])
    request.is_dummy_request = dummy
    mgr._apply_branch_snapshot_point(request)
    assert request.py_midchunk_pass_point == expected
    assert request.expect_snapshot_points == ([256, 640] if cur < 256 else [640])


def test_prepare_expect_snapshot_points_marks_the_next_point_to_pass(monkeypatch):
    """Offset points before the prompt end (e.g. offsets_from_end {0, 2K, 8K})
    are taken inside a step too: one per step, the last one merges."""
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_BRANCH_SNAPSHOT", True)
    mgr = _manager_stub()

    def offset_points(self, requests):
        for request in requests:
            request.expect_snapshot_points = [20480, 26624, 28544]

    monkeypatch.setattr(MambaHybridCacheManagerV2, "prepare_expect_snapshot_points", offset_points)
    request, _ = _context_request(1, 16384, 0, 28600, [])
    mgr.prepare_expect_snapshot_points([request])
    assert request.py_midchunk_pass_point == 20480
    assert _get_forced_context_chunk_size(request) == 26624 - 16384

    request.context_current_position = 26624
    request.context_remaining_length = 28600 - 26624
    mgr.prepare_expect_snapshot_points([request])
    assert request.py_midchunk_pass_point is None
    assert _get_forced_context_chunk_size(request) == 28600 - 26624


def test_branch_point_snapshot_inside_step_then_boundary_commit():
    """The step passes the branch point: the pending page commits the prompt up
    to it, then the boundary update commits the rest of the step."""
    mgr = _manager_stub()
    events = []
    request, tokens = _context_request(7, 0, 640, 700, [256, 640])
    request.is_last_context_chunk = False
    mgr._branch_snapshot_points[7] = 256
    kv_cache = _mock_cache(events)
    mgr.kv_cache_map[7] = kv_cache
    batch = ScheduledRequests()
    batch.append_context_request(request)

    mgr._prepare_kda_midchunk_snapshots(batch.context_requests)
    assert mgr.get_kda_midchunk_snapshots([7]) == [(0, 40, 256)]
    request.context_current_position = 640
    request.context_remaining_length = 60
    mgr.update_context_resources(batch)

    assert events == [
        "acquire",
        ("resize", None, 256),
        ("commit", 256, 0, True),
        ("resize", None, 640),
        ("commit", 384, 256, False),
    ]
    assert kv_cache.commit.call_args_list[1].args[0] == tokens[256:640]
    assert mgr._kda_midchunk_snapshots_taken_total == 1
    assert mgr._branch_snapshots_taken_total == 1


def test_record_branch_snapshot_point_marks_the_pass_point(monkeypatch):
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_BRANCH_SNAPSHOT", True)
    mgr = _manager_stub()
    mgr.tokens_per_block = 128
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=True),
    )
    mgr._snapshot_pruned_tokens_total = 0
    mgr._page_pruned_tokens_total = 0
    mgr._branch_snapshots_skipped_total = {}
    request, _ = _context_request(1, 0, 1000, 1000, [1000])
    kv_cache = SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: 700,
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: 0,
        num_committed_tokens=0,
    )
    mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=999)
    assert mgr._branch_snapshot_points == {1: 640}
    assert request.expect_snapshot_points == [640, 1000]
    assert request.py_midchunk_pass_point == 640
    assert mgr._page_pruned_tokens_total == 700
    assert mgr._format_branch_snapshot_counters().endswith(
        " midchunk_snapshots_taken=0 midchunk_snapshots_skipped=0"
    )


def test_reuse_breakdown_classes_fresh_lookups():
    """A request's first fresh lookup is new / known / none by its conversation;
    a later fresh lookup of the same request (a released claim) is rep."""
    mgr = _manager_stub()
    mgr.tokens_per_block = 128
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=True),
    )
    mgr._snapshot_pruned_tokens_total = 0
    mgr._page_pruned_tokens_total = 0
    mgr._branch_snapshots_skipped_total = {}
    kv_cache = SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: 700,
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: 640,
        num_committed_tokens=512,
    )

    def lookup(request_id, conversation_id, matched=True):
        request, _ = _context_request(request_id, 0, 1000, 1000, [1000])
        if conversation_id is not None:
            request.py_conversation_params = SimpleNamespace(conversation_id=conversation_id)
        cache = kv_cache if matched else SimpleNamespace(
            _get_num_reusable_tokens_before_pruning=lambda: 0,
            _get_num_reusable_tokens_before_hybrid_pruning=lambda: 0,
            num_committed_tokens=0,
        )
        mgr._record_branch_snapshot_point(request, cache, num_lookup_tokens=999)

    lookup(1, "a", matched=False)
    lookup(2, "a")
    lookup(2, "a")
    lookup(3, " ")
    # A resumed cache is not a fresh lookup.
    mgr._record_branch_snapshot_point(
        _context_request(4, 0, 1000, 1000, [1000])[0], kv_cache, num_lookup_tokens=None
    )
    assert mgr._reuse_breakdown.format() == (
        "reuse_new=n:1,hit:0,lookup:999,matched:0,attn:0,reused:0 "
        "reuse_known=n:1,hit:1,lookup:999,matched:700,attn:640,reused:512 "
        "reuse_none=n:1,hit:1,lookup:999,matched:700,attn:640,reused:512 "
        "reuse_rep=n:1,hit:1,lookup:999,matched:700,attn:640,reused:512"
    )


@pytest.mark.parametrize("case", ["not_run", "inactive", "lost_page", "already_committed"])
def test_snapshot_is_skipped_and_page_released(case):
    mgr = _manager_stub()
    events = []
    request, _, kv_cache, batch = _planned_one_step(mgr, events)
    # The KDA layers fetched the plan; the snapshot is skipped for another reason.
    assert mgr.get_kda_midchunk_snapshots([7]) == [(0, 40, 256)]
    _finish_step(request)
    if case == "not_run":
        request.context_current_position = 0
        request.context_remaining_length = 300
    elif case == "inactive":
        kv_cache.is_active = False
        kv_cache.pending_ssm_snapshot_slot = None  # suspension released it
    elif case == "lost_page":
        kv_cache.pending_ssm_snapshot_slot = None
    else:
        kv_cache.num_committed_tokens = 256
    events.clear()

    mgr.update_context_resources(batch)

    assert not _committed_from_pending(events)
    if case != "inactive":
        assert events[0] == "release"
    assert mgr._kda_midchunk_snapshots == {}
    assert mgr._kda_midchunk_snapshots_skipped_total == 1
    assert mgr._kda_midchunk_snapshots_taken_total == 0
    # A fetched plan keeps mid-chunk snapshots on.
    assert mgr._kda_midchunk_snapshot_alignment == 64


def test_unfetched_snapshot_of_a_step_that_ran_turns_midchunk_snapshots_off(monkeypatch):
    """The KDA layers never fetched the plan, so the pending page holds no state.

    It is released, not committed, and the manager stops planning snapshots:
    the running model's prefill does not store them.
    """
    mgr = _manager_stub()
    events = []
    request, _, _, batch = _planned_one_step(mgr, events)
    _finish_step(request)
    mgr.update_context_resources(batch)

    # Released before the end update, which leaves the prompt uncommitted at the point.
    assert events == ["acquire", "release", ("resize", None, 300), "stop"]
    assert mgr._kda_midchunk_snapshots == {}
    assert mgr._kda_midchunk_snapshots_skipped_total == 1
    assert mgr._kda_midchunk_snapshots_taken_total == 0
    assert mgr._kda_midchunk_snapshot_alignment is None

    # Nothing is planned any more, and the point splits the next request's chunk again.
    assert mgr._plan_kda_midchunk_snapshot(request) is None
    monkeypatch.setattr(
        MambaHybridCacheManagerV2, "prepare_expect_snapshot_points", lambda self, requests: None
    )
    later, _ = _context_request(8, 0, 300, 300, [256])
    mgr.prepare_expect_snapshot_points([later])
    assert later.py_midchunk_snapshot_alignment is None
    assert _get_forced_context_chunk_size(later) == 256


def test_unfetched_snapshot_of_a_step_that_did_not_run_keeps_midchunk_snapshots_on():
    """A step that did not run does not show the model ignoring the plan."""
    mgr = _manager_stub()
    events = []
    _, _, _, batch = _planned_one_step(mgr, events)
    # The context position is still before the point.
    mgr.update_context_resources(batch)

    assert events == ["acquire", "release"]
    assert mgr._kda_midchunk_snapshots == {}
    assert mgr._kda_midchunk_snapshots_skipped_total == 1
    assert mgr._kda_midchunk_snapshots_taken_total == 0
    assert mgr._kda_midchunk_snapshot_alignment == 64


def test_a_fetch_counts_for_its_own_step_only():
    """A step planned again after a fetch needs a fetch of its own."""
    mgr = _manager_stub()
    events = []
    request, _, _, batch = _planned_one_step(mgr, events)
    assert mgr.get_kda_midchunk_snapshots([7]) == [(0, 40, 256)]
    # That step never reached its update; the request is scheduled again and
    # this time the plan is not fetched.
    mgr._prepare_kda_midchunk_snapshots(batch.context_requests)
    _finish_step(request)
    mgr.update_context_resources(batch)

    assert events[:4] == ["acquire", "release", "acquire", "release"]
    assert not _committed_from_pending(events)
    assert mgr._kda_midchunk_snapshots_taken_total == 0
    assert mgr._kda_midchunk_snapshot_alignment is None


def test_no_free_page_skips_snapshot():
    mgr = _manager_stub()
    events = []
    request, _ = _context_request(7, 0, 300, 300, [256])
    kv_cache = _mock_cache(events)
    kv_cache.acquire_ssm_snapshot_slot = MagicMock(return_value=None)
    mgr.kv_cache_map[7] = kv_cache
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr._prepare_kda_midchunk_snapshots(batch.context_requests)
    assert mgr.get_kda_midchunk_snapshots([7]) == []
    assert mgr._kda_midchunk_snapshots_skipped_total == 1
    _finish_step(request)
    mgr.update_context_resources(batch)
    # No pending page: the prompt is not committed at the point, the live state
    # holds the end state.
    assert not any(isinstance(event, tuple) and event[0] == "commit" for event in events)
    # Not a sign of a model that ignores the plan.
    assert mgr._kda_midchunk_snapshot_alignment == 64


def test_rank_without_kda_layers_commits_attention_at_the_point():
    mgr = _manager_stub(local_num_mamba_layers=0)
    events = []
    request, tokens = _context_request(7, 0, 300, 300, [256])
    kv_cache = _mock_cache(events)
    mgr.kv_cache_map[7] = kv_cache
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr._prepare_kda_midchunk_snapshots(batch.context_requests)
    kv_cache.acquire_ssm_snapshot_slot.assert_not_called()
    assert mgr.get_kda_midchunk_snapshots([7]) == []
    _finish_step(request)
    mgr.update_context_resources(batch)
    kv_cache.commit.assert_called_once_with(tokens[:256], ssm_from_pending=False)


def test_stale_pending_page_is_released_on_next_prepare_and_free(monkeypatch):
    mgr = _manager_stub()
    events = []
    request, _, _, batch = _planned_one_step(mgr, events)
    # The step never reached its update; the request is scheduled again.
    mgr._prepare_kda_midchunk_snapshots(batch.context_requests)
    assert events == ["acquire", "release", "acquire"]

    events.clear()
    base_free = MagicMock()
    monkeypatch.setattr(MambaHybridCacheManagerV2, "free_resources", base_free)
    mgr.free_resources(request)
    assert events == ["release"]
    base_free.assert_called_once()
    assert mgr._kda_midchunk_snapshots == {}


def test_inflight_prefix_index_finds_the_deepest_prefilling_owner():
    index = kv_cache_manager_v2._InflightPrefixIndex(unit_tokens=4)
    owner = SimpleNamespace(py_request_id=1, context_remaining_length=20)
    waiter = SimpleNamespace(py_request_id=2, context_remaining_length=17)
    other_scope = SimpleNamespace(py_request_id=3, context_remaining_length=17)
    shared = list(range(1, 13))
    owner_fps = index.fingerprints(owner, shared + [20, 21, 22, 23, 24, 25, 26, 27], b"s")
    waiter_fps = index.fingerprints(waiter, shared + [99, 98, 97, 96, 95], b"s")
    assert len(owner_fps) == 4 and len(waiter_fps) == 4
    assert index.fingerprints(waiter, [], b"s") is waiter_fps  # cached per request
    other_fps = index.fingerprints(other_scope, shared + [99, 98, 97, 96, 95], b"t")

    assert index.deepest_owner(waiter_fps, 2) == (None, 0)
    index.add(owner)
    assert index.deepest_owner(waiter_fps, 2) == (owner, 3)
    assert index.deepest_owner(owner_fps, 1) == (None, 0)  # not its own owner
    assert index.deepest_owner(other_fps, 3) == (None, 0)  # other reuse scope
    owner.context_remaining_length = 0
    assert index.deepest_owner(waiter_fps, 2) == (None, 0)  # context ended
    owner.context_remaining_length = 20
    index.remove(1)
    assert index.deepest_owner(waiter_fps, 2) == (None, 0)


# ---------------------------------------------------------------------------
# Real KimiK3HybridCacheManager (KV cache manager V2 C++ backend)
# ---------------------------------------------------------------------------


def _build_kda_v2_manager(prompt_end_offset, enable_branch_snapshot=False, policy="per_request"):
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        enable_block_reuse=True,
        enable_partial_reuse=False,
        block_reuse_config=BlockReuseConfig(
            policy=policy, max_num_turns=1, enable_branch_snapshot=enable_branch_snapshot
        ),
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=0,
            additional_snapshot_offsets_from_end=[prompt_end_offset],
        ),
    )
    return KimiK3HybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.bfloat16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=4,
        mapping=mapping,
        layer_mask=[False, True],
        vocab_size=1024,
        dtype=DataType.HALF,
        conv_state_layout="q_k_v",
    )


def _llm_request(request_id, tokens):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


@skip_no_cuda
@requires_pending_ssm_pages
def test_real_manager_one_step_snapshot_is_reused(monkeypatch):
    _enable_midchunk_snapshots(monkeypatch)
    prompt = list(range(100, 200))
    point = 64
    mgr = _build_kda_v2_manager(prompt_end_offset=len(prompt) - point)
    try:
        assert mgr._kda_midchunk_snapshot_alignment == 64

        request = _llm_request(1, prompt)
        mgr.prepare_expect_snapshot_points([request])
        assert request.expect_snapshot_points == [point]
        assert _get_forced_context_chunk_size(request) == len(prompt)
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, num_tokens=len(prompt))
        request.context_chunk_size = len(prompt)
        batch = ScheduledRequests()
        batch.append_context_request(request)
        mgr.prepare_resources(batch)

        ((index, slot, offset),) = mgr.get_kda_midchunk_snapshots([1])
        assert (index, offset) == (0, point)
        live = mgr.get_state_indices([1], [False])[0]
        assert slot != live
        ssm, conv = mgr.all_ssm_states[0], mgr.all_conv_states[0]
        snapshot_ssm = torch.randn_like(ssm[slot], dtype=torch.float32).to(ssm.dtype)
        snapshot_conv = torch.randn_like(conv[slot], dtype=torch.float32).to(conv.dtype)
        end_ssm = torch.randn_like(snapshot_ssm, dtype=torch.float32).to(ssm.dtype)
        end_conv = torch.randn_like(snapshot_conv, dtype=torch.float32).to(conv.dtype)
        # What the KDA layers write in the step.
        ssm[slot], conv[slot] = snapshot_ssm, snapshot_conv
        ssm[live], conv[live] = end_ssm, end_conv
        torch.cuda.synchronize()

        request.context_current_position = len(prompt)
        assert request.context_remaining_length == 0
        mgr.update_context_resources(batch)
        kv_cache = mgr.kv_cache_map[1]
        assert kv_cache.num_committed_tokens == point
        assert kv_cache.pending_ssm_snapshot_slot is None
        assert mgr._kda_midchunk_snapshots_taken_total == 1
        torch.cuda.synchronize()
        # The live slot keeps the end state for generation and disagg transfer.
        assert mgr.get_state_indices([1], [False])[0] == live
        assert torch.equal(ssm[live], end_ssm) and torch.equal(conv[live], end_conv)
        mgr.free_resources(request)

        follow_up = _llm_request(2, prompt + list(range(500, 510)))
        mgr.prepare_expect_snapshot_points([follow_up])
        assert mgr.prepare_context(follow_up)
        assert follow_up.context_current_position == point
        assert mgr.resize_context(follow_up, num_tokens=follow_up.context_remaining_length)
        follow_up.context_chunk_size = follow_up.context_remaining_length
        batch = ScheduledRequests()
        batch.append_context_request(follow_up)
        mgr.prepare_resources(batch)
        torch.cuda.synchronize()
        reused = mgr.get_state_indices([2], [False])[0]
        assert torch.equal(ssm[reused], snapshot_ssm)
        assert torch.equal(conv[reused], snapshot_conv)
        mgr.free_resources(follow_up)
    finally:
        mgr.shutdown()


def _run_context_step(mgr, request, write_states=None):
    """Schedule one context step of ``request`` as the V2 scheduler does and run it.

    ``write_states(mgr, live_slot, snapshots)`` stands in for the KDA layers.
    """
    mgr.prepare_expect_snapshot_points([request])
    assert mgr.prepare_context(request)
    chunk = _get_forced_context_chunk_size(request)
    assert mgr.resize_context(request, num_tokens=chunk)
    request.context_chunk_size = chunk
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr.prepare_resources(batch)
    snapshots = mgr.get_kda_midchunk_snapshots([request.py_request_id])
    live = mgr.get_state_indices([request.py_request_id], [False])[0]
    if write_states is not None:
        write_states(mgr, live, snapshots)
    torch.cuda.synchronize()
    request.context_current_position += chunk
    mgr.update_context_resources(batch)
    return chunk, snapshots


@skip_no_cuda
@requires_pending_ssm_pages
@pytest.mark.parametrize("policy", ["per_request", "per_conversation"])
def test_real_manager_branch_snapshot_inside_step_is_reused_by_sibling(monkeypatch, policy):
    """Sibling B leaves A's cached prefix at 64: it is prefilled in one step with
    a mid-chunk snapshot at 64, which sibling C then reuses."""
    _enable_midchunk_snapshots(monkeypatch)
    shared = list(range(100, 164))  # two blocks of 32
    mgr = _build_kda_v2_manager(prompt_end_offset=0, enable_branch_snapshot=True, policy=policy)
    try:
        ssm, conv = mgr.all_ssm_states[0], mgr.all_conv_states[0]
        snapshot_ssm = torch.randn_like(ssm[0], dtype=torch.float32).to(ssm.dtype)
        snapshot_conv = torch.randn_like(conv[0], dtype=torch.float32).to(conv.dtype)

        sibling_a = _llm_request(1, shared + list(range(300, 336)))
        chunk, snapshots = _run_context_step(mgr, sibling_a)
        assert chunk == 100 and snapshots == []
        assert mgr._branch_snapshot_points == {}
        mgr.free_resources(sibling_a)

        def write_b(mgr, live, snapshots):
            ((_, slot, offset),) = snapshots
            assert offset == 64 and slot != live
            ssm[slot], conv[slot] = snapshot_ssm, snapshot_conv

        sibling_b = _llm_request(2, shared + list(range(400, 440)))
        chunk, _ = _run_context_step(mgr, sibling_b, write_b)
        # One step over the whole prompt although it passes the branch point.
        assert chunk == 104
        assert sibling_b.expect_snapshot_points == [64, 104]
        assert mgr._branch_snapshots_taken_total == 1
        assert mgr._kda_midchunk_snapshots_taken_total == 1
        mgr.free_resources(sibling_b)

        sibling_c = _llm_request(3, shared + list(range(500, 530)))
        mgr.prepare_expect_snapshot_points([sibling_c])
        assert mgr.prepare_context(sibling_c)
        assert sibling_c.context_current_position == 64
        assert mgr.resize_context(sibling_c, num_tokens=sibling_c.context_remaining_length)
        sibling_c.context_chunk_size = sibling_c.context_remaining_length
        batch = ScheduledRequests()
        batch.append_context_request(sibling_c)
        mgr.prepare_resources(batch)
        torch.cuda.synchronize()
        reused = mgr.get_state_indices([3], [False])[0]
        assert torch.equal(ssm[reused], snapshot_ssm)
        assert torch.equal(conv[reused], snapshot_conv)
        mgr.free_resources(sibling_c)
    finally:
        mgr.shutdown()


def _wait_for_inflight_prefix(monkeypatch, max_wait_passes=None):
    """TRTLLM_KV_V2_WAIT_FOR_INFLIGHT_PREFIX=1 with one-block units; set before construction."""
    monkeypatch.setattr(kv_cache_manager_v2, "_WAIT_FOR_INFLIGHT_PREFIX", True)
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_UNIT_BLOCKS", 1)
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_MIN_GAIN_BLOCKS", 1)
    if max_wait_passes is not None:
        monkeypatch.setattr(
            kv_cache_manager_v2, "_INFLIGHT_PREFIX_MAX_WAIT_PASSES", max_wait_passes
        )


@skip_no_cuda
@requires_pending_ssm_pages
def test_real_manager_sibling_waits_for_inflight_prefix(monkeypatch):
    """Y arrives while X is prefilling their shared 64 tokens: Y waits, X takes
    the requested snapshot at 64 inside its step, then Y reuses it."""
    _wait_for_inflight_prefix(monkeypatch, max_wait_passes=2)
    _enable_midchunk_snapshots(monkeypatch)
    shared = list(range(100, 164))  # two blocks of 32
    mgr = _build_kda_v2_manager(
        prompt_end_offset=0, enable_branch_snapshot=True, policy="per_conversation"
    )
    try:
        ssm, conv = mgr.all_ssm_states[0], mgr.all_conv_states[0]
        snapshot_ssm = torch.randn_like(ssm[0], dtype=torch.float32).to(ssm.dtype)
        snapshot_conv = torch.randn_like(conv[0], dtype=torch.float32).to(conv.dtype)
        sibling_x = _llm_request(1, shared + list(range(300, 340)))
        sibling_y = _llm_request(2, shared + list(range(400, 430)))

        # X claimed its cache in an earlier pass; Y waits for it.
        mgr.prepare_expect_snapshot_points([sibling_x, sibling_y])
        assert mgr.prepare_context(sibling_x)
        assert mgr.plan_context_waits([sibling_x, sibling_y]) == {2}
        assert 2 not in mgr.kv_cache_map
        assert sibling_x.expect_snapshot_points == [64, 104]
        assert sibling_x.py_midchunk_pass_point == 64

        # X's step passes 64 and takes the snapshot inside it.
        chunk = _get_forced_context_chunk_size(sibling_x)
        assert chunk == 104
        assert mgr.resize_context(sibling_x, num_tokens=chunk)
        sibling_x.context_chunk_size = chunk
        batch = ScheduledRequests()
        batch.append_context_request(sibling_x)
        mgr.prepare_resources(batch)
        ((_, slot, offset),) = mgr.get_kda_midchunk_snapshots([1])
        assert offset == 64
        ssm[slot], conv[slot] = snapshot_ssm, snapshot_conv
        torch.cuda.synchronize()
        sibling_x.context_current_position += chunk
        mgr.update_context_resources(batch)
        assert mgr._inflight_prefix.points_taken_total == 1

        # X's context ended: Y no longer waits and resumes at 64.
        mgr.prepare_expect_snapshot_points([sibling_y])
        assert mgr.plan_context_waits([sibling_y]) == set()
        assert mgr.prepare_context(sibling_y)
        assert sibling_y.context_current_position == 64
        assert mgr.resize_context(sibling_y, num_tokens=sibling_y.context_remaining_length)
        sibling_y.context_chunk_size = sibling_y.context_remaining_length
        batch = ScheduledRequests()
        batch.append_context_request(sibling_y)
        mgr.prepare_resources(batch)
        torch.cuda.synchronize()
        reused = mgr.get_state_indices([2], [False])[0]
        assert torch.equal(ssm[reused], snapshot_ssm)
        assert torch.equal(conv[reused], snapshot_conv)
        assert mgr._inflight_prefix.deferrals_total == 1
        assert "inflight_prefix_points_taken=1" in mgr._format_branch_snapshot_counters()
        mgr.free_resources(sibling_x)
        mgr.free_resources(sibling_y)
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_real_manager_waiter_gives_up(monkeypatch):
    """Needs no mid-chunk snapshots: the owner is asked for a branch-like point."""
    _wait_for_inflight_prefix(monkeypatch, max_wait_passes=2)
    shared = list(range(100, 164))
    mgr = _build_kda_v2_manager(prompt_end_offset=0, enable_branch_snapshot=True)
    try:
        sibling_x = _llm_request(1, shared + list(range(300, 340)))
        sibling_y = _llm_request(2, shared + list(range(400, 430)))
        mgr.prepare_expect_snapshot_points([sibling_x, sibling_y])
        assert mgr.prepare_context(sibling_x)
        # At most two passes of waiting.
        assert mgr.plan_context_waits([sibling_x, sibling_y]) == {2}
        assert mgr.plan_context_waits([sibling_x, sibling_y]) == {2}
        assert mgr.plan_context_waits([sibling_x, sibling_y]) == set()
        # The owner passed the point without a snapshot: no waiting.
        sibling_z = _llm_request(3, shared + list(range(500, 520)))
        sibling_x.context_current_position = 96
        assert mgr.plan_context_waits([sibling_x, sibling_z]) == set()
        sibling_x.context_current_position = 0
        mgr.free_resources(sibling_x)
    finally:
        mgr.shutdown()


@skip_no_cuda
@requires_pending_ssm_pages
def test_real_manager_same_pass_siblings_plan_the_point_before_scheduling(monkeypatch):
    """Both siblings are new in the same pass: Y waits, and X gets the point
    before its first step is sized, so the step passes it and snapshots it."""
    _wait_for_inflight_prefix(monkeypatch)
    _enable_midchunk_snapshots(monkeypatch)
    shared = list(range(100, 164))
    mgr = _build_kda_v2_manager(prompt_end_offset=0, enable_branch_snapshot=True)
    try:
        sibling_x = _llm_request(1, shared + list(range(300, 340)))
        sibling_y = _llm_request(2, shared + list(range(400, 430)))
        mgr.prepare_expect_snapshot_points([sibling_x, sibling_y])
        assert mgr.plan_context_waits([sibling_x, sibling_y]) == {2}
        assert mgr.prepare_context(sibling_x)
        assert sibling_x.expect_snapshot_points == [64, 104]
        assert _get_forced_context_chunk_size(sibling_x) == 104
        assert mgr.resize_context(sibling_x, num_tokens=104)
        sibling_x.context_chunk_size = 104
        batch = ScheduledRequests()
        batch.append_context_request(sibling_x)
        mgr.prepare_resources(batch)
        assert [offset for _, _, offset in mgr.get_kda_midchunk_snapshots([1])] == [64]
        mgr.free_resources(sibling_x)
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# One K3 Rubin KDA layer (CuTe DSL prefill) on a real manager
# ---------------------------------------------------------------------------


class _KdaLayerConfig:
    hidden_size = 256
    rms_norm_eps = 1e-6
    linear_attn_config = {
        "num_heads": 4,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }


def _build_kda_layer_manager(prompt_end_offset, ssm_dtype, enable_branch_snapshot=False):
    """One KDA layer (4 heads of 128) + one attention layer, KDA pool shapes."""
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    kv_cache_config = KvCacheConfig(
        max_tokens=4096,
        enable_block_reuse=True,
        enable_partial_reuse=False,
        block_reuse_config=BlockReuseConfig(
            policy="per_request", max_num_turns=1, enable_branch_snapshot=enable_branch_snapshot
        ),
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=0,
            additional_snapshot_offsets_from_end=[prompt_end_offset],
        ),
    )
    num_heads, head_dim = 4, 128
    return KimiK3HybridCacheManager(
        mamba_d_state=head_dim,
        mamba_d_conv=4,
        mamba_num_heads=num_heads,
        mamba_n_groups=num_heads,
        mamba_head_dim=head_dim,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=ssm_dtype,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=64,
        max_seq_len=1024,
        max_batch_size=4,
        mapping=mapping,
        layer_mask=[False, True],
        vocab_size=1024,
        dtype=DataType.HALF,
        conv_state_layout="q_k_v",
    )


def _snapshot_kda_layer(monkeypatch):
    """A K3 Rubin KDA layer whose prefill stores mid-chunk snapshots, or skip."""
    pytest.importorskip("fla")
    kernels = _private_kda_module("_kda_kernels")
    mixer = _private_kda_module("kimi_kda_mixer")
    if not kernels.kda_prefill_supports_state_snapshots():
        pytest.skip("the K3 Rubin CuTe DSL KDA prefill is unavailable")
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", True)
    torch.manual_seed(0)
    layer = mixer.KimiKDALinearAttention(_KdaLayerConfig(), layer_idx=0).to("cuda")
    torch.nn.init.normal_(layer.dt_bias, std=0.1)
    layer.finalize_decode_weights()
    return layer


def _build_kda_snapshot_plan(snapshots, context_lengths):
    metadata = _private_kda_module("kimi_k3_mamba_metadata")
    return metadata.build_kda_snapshot_plan(snapshots, context_lengths, torch.device("cuda"))


def _kda_prefill(layer, hidden, lengths, has_init, slots, conv_pool, ssm_pool, snapshot=None):
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.long
    )
    metadata = SimpleNamespace(
        use_initial_states=any(has_init),
        has_initial_states=torch.tensor(has_init, device="cuda", dtype=torch.bool),
        kda_chunk_indices=torch.tensor(
            [[b, c] for b, n in enumerate(lengths) for c in range((n + 63) // 64)],
            device="cuda",
            dtype=torch.long,
        ),
        kda_varlen_is_aligned=all(n % 64 == 0 for n in lengths),
        kda_single_sequence_length=lengths[0] if len(lengths) == 1 else None,
        query_start_loc=cu_seqlens.to(torch.int32),
        state_indices=slots,
        _arange_buffer=torch.arange(len(lengths) + 1, dtype=torch.int32, device="cuda"),
        kda_snapshot=snapshot,
    )
    return layer.forward_prefill(
        hidden, cu_seqlens, metadata, len(lengths), conv_pool, ssm_pool, slots
    )


@skip_no_cuda
@requires_pending_ssm_pages
@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_two_turns_one_step_snapshot_matches_two_step_prefill(monkeypatch, ssm_dtype):
    """Turn 1 prefills in one step; turn 2 reuses exactly the two-step snapshot state."""
    layer = _snapshot_kda_layer(monkeypatch)

    prompt_len, point, turn2_len = 300, 256, 350
    hidden_size = _KdaLayerConfig.hidden_size
    hidden = torch.randn(turn2_len, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.05
    prompt = list(range(1000, 1000 + prompt_len))
    turn2_prompt = prompt + list(range(5000, 5000 + turn2_len - prompt_len))

    mgr = _build_kda_layer_manager(prompt_end_offset=prompt_len - point, ssm_dtype=ssm_dtype)
    try:
        ssm_pool, conv_pool = mgr.get_ssm_states(0), mgr.get_conv_states(0)
        # Turn 1: one step over the whole prompt with the snapshot at the point.
        request = _llm_request(1, prompt)
        mgr.prepare_expect_snapshot_points([request])
        assert request.expect_snapshot_points == [point]
        assert _get_forced_context_chunk_size(request) == prompt_len
        assert mgr.prepare_context(request)
        assert mgr.resize_context(request, num_tokens=prompt_len)
        request.context_chunk_size = prompt_len
        batch = ScheduledRequests()
        batch.append_context_request(request)
        mgr.prepare_resources(batch)
        live = torch.tensor(mgr.get_state_indices([1], [False]), dtype=torch.int32, device="cuda")
        plan = _build_kda_snapshot_plan(mgr.get_kda_midchunk_snapshots([1]), [prompt_len])
        assert plan is not None and plan.num_snapshots == 1
        out_one = _kda_prefill(
            layer, hidden[:prompt_len], [prompt_len], [False], live, conv_pool, ssm_pool, plan
        )
        end_ssm_one = ssm_pool[live.long()].clone()
        torch.cuda.synchronize()
        request.context_current_position = prompt_len
        mgr.update_context_resources(batch)
        assert mgr.kv_cache_map[1].num_committed_tokens == point
        assert mgr._kda_midchunk_snapshots_taken_total == 1
        mgr.free_resources(request)

        # Reference: two steps on private pools, split at the point.
        ref_ssm = torch.zeros((1, *ssm_pool.shape[1:]), dtype=ssm_dtype, device="cuda")
        ref_conv = torch.zeros((1, *conv_pool.shape[1:]), dtype=conv_pool.dtype, device="cuda")
        ref_slot = torch.zeros(1, dtype=torch.int32, device="cuda")
        out_head = _kda_prefill(
            layer, hidden[:point], [point], [False], ref_slot, ref_conv, ref_ssm
        )
        snap_ssm, snap_conv = ref_ssm[0].clone(), ref_conv[0].clone()
        out_tail = _kda_prefill(
            layer,
            hidden[point:prompt_len],
            [prompt_len - point],
            [True],
            ref_slot,
            ref_conv,
            ref_ssm,
        )
        assert torch.equal(out_one[:point], out_head)
        if ssm_dtype == torch.float32:
            assert torch.equal(out_one[point:], out_tail)
            assert torch.equal(end_ssm_one[0], ref_ssm[0])
        else:
            # The one step continues from the fp32 state, the second step from
            # the rounded one.
            rel = (out_one[point:].float() - out_tail.float()).norm() / out_tail.float().norm()
            assert rel.item() < 2e-2

        # Turn 2 reuses the snapshot: its live state is the two-step state at the point.
        follow_up = _llm_request(2, turn2_prompt)
        mgr.prepare_expect_snapshot_points([follow_up])
        assert mgr.prepare_context(follow_up)
        assert follow_up.context_current_position == point
        remaining = follow_up.context_remaining_length
        assert mgr.resize_context(follow_up, num_tokens=remaining)
        follow_up.context_chunk_size = remaining
        batch = ScheduledRequests()
        batch.append_context_request(follow_up)
        mgr.prepare_resources(batch)
        torch.cuda.synchronize()
        live2 = torch.tensor(mgr.get_state_indices([2], [False]), dtype=torch.int32, device="cuda")
        assert torch.equal(ssm_pool[live2.long()][0], snap_ssm)
        assert torch.equal(conv_pool[live2.long()][0], snap_conv)
        out_turn2 = _kda_prefill(
            layer, hidden[point:turn2_len], [remaining], [True], live2, conv_pool, ssm_pool
        )
        ref_ssm[0], ref_conv[0] = snap_ssm, snap_conv
        ref_turn2 = _kda_prefill(
            layer, hidden[point:turn2_len], [remaining], [True], ref_slot, ref_conv, ref_ssm
        )
        assert torch.equal(out_turn2, ref_turn2)
        mgr.free_resources(follow_up)
    finally:
        mgr.shutdown()


def _kda_context_step(mgr, request, layer, hidden):
    """Schedule and run one context step of ``request`` through a KDA layer."""
    mgr.prepare_expect_snapshot_points([request])
    assert mgr.prepare_context(request)
    start = request.context_current_position
    chunk = _get_forced_context_chunk_size(request)
    assert mgr.resize_context(request, num_tokens=chunk)
    request.context_chunk_size = chunk
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr.prepare_resources(batch)
    live = torch.tensor(
        mgr.get_state_indices([request.py_request_id], [False]), dtype=torch.int32, device="cuda"
    )
    plan = _build_kda_snapshot_plan(
        mgr.get_kda_midchunk_snapshots([request.py_request_id]), [chunk]
    )
    out = _kda_prefill(
        layer,
        hidden[start : start + chunk],
        [chunk],
        [start > 0],
        live,
        mgr.get_conv_states(0),
        mgr.get_ssm_states(0),
        plan,
    )
    torch.cuda.synchronize()
    request.context_current_position += chunk
    mgr.update_context_resources(batch)
    return start, chunk, plan, out


@skip_no_cuda
@requires_pending_ssm_pages
@pytest.mark.parametrize("ssm_dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_sibling_reuses_branch_snapshot_and_matches_full_prefill(monkeypatch, ssm_dtype):
    """Siblings A, B, C share a 256-token prefix. B is prefilled in one step with
    a mid-chunk branch snapshot at 256; C resumes from it, and its prefill matches
    a from-scratch prefill of its whole prompt."""
    layer = _snapshot_kda_layer(monkeypatch)

    hidden_size = _KdaLayerConfig.hidden_size
    point = 256
    shared_tokens = list(range(1000, 1000 + point))
    shared_hidden = torch.randn(point, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.05

    def sibling(request_id, suffix_start, suffix_len):
        tokens = shared_tokens + list(range(suffix_start, suffix_start + suffix_len))
        suffix = torch.randn(suffix_len, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.05
        return _llm_request(request_id, tokens), torch.cat([shared_hidden, suffix])

    def private_prefill(hidden, lengths, has_init, ssm=None, conv=None):
        ssm_pool, conv_pool = mgr.get_ssm_states(0), mgr.get_conv_states(0)
        ssm = (
            torch.zeros((1, *ssm_pool.shape[1:]), dtype=ssm_dtype, device="cuda")
            if ssm is None
            else ssm
        )
        conv = (
            torch.zeros((1, *conv_pool.shape[1:]), dtype=conv_pool.dtype, device="cuda")
            if conv is None
            else conv
        )
        slot = torch.zeros(1, dtype=torch.int32, device="cuda")
        return _kda_prefill(layer, hidden, lengths, has_init, slot, conv, ssm), ssm, conv

    mgr = _build_kda_layer_manager(
        prompt_end_offset=0, ssm_dtype=ssm_dtype, enable_branch_snapshot=True
    )
    try:
        request_a, hidden_a = sibling(1, 5000, 44)
        start, chunk, plan, _ = _kda_context_step(mgr, request_a, layer, hidden_a)
        assert (start, chunk, plan) == (0, 300, None)
        mgr.free_resources(request_a)

        # B leaves A's prefix at 256: one step over its whole prompt with the
        # branch snapshot inside it, outputs bitwise those of a plain prefill.
        request_b, hidden_b = sibling(2, 6000, 60)
        start, chunk, plan, out_b = _kda_context_step(mgr, request_b, layer, hidden_b)
        assert (start, chunk) == (0, 316)
        assert plan is not None and plan.num_snapshots == 1
        assert request_b.expect_snapshot_points == [point, 316]
        assert mgr._branch_snapshots_taken_total == 1
        assert mgr._kda_midchunk_snapshots_taken_total == 1
        ref_b, _, _ = private_prefill(hidden_b, [316], [False])
        assert torch.equal(out_b, ref_b)
        mgr.free_resources(request_b)

        # C resumes at the branch snapshot, which is bitwise the state of a
        # prefill stopped at the point.
        request_c, hidden_c = sibling(3, 7000, 100)
        _, snap_ssm, snap_conv = private_prefill(hidden_c[:point], [point], [False])
        start, chunk, plan, out_c = _kda_context_step(mgr, request_c, layer, hidden_c)
        assert (start, chunk, plan) == (point, 100, None)
        live_c = mgr.get_state_indices([3], [False])[0]
        # The step advanced the live state; check the resumed state by
        # rerunning C's step from the reference snapshot.
        ref_tail, ref_ssm, _ = private_prefill(
            hidden_c[point:], [100], [True], snap_ssm.clone(), snap_conv.clone()
        )
        assert torch.equal(out_c, ref_tail)
        assert torch.equal(mgr.get_ssm_states(0)[live_c], ref_ssm[0])

        # Reuse matches a from-scratch prefill of C's whole prompt.
        full_c, full_ssm, _ = private_prefill(hidden_c, [356], [False])
        if ssm_dtype == torch.float32:
            assert torch.equal(out_c, full_c[point:])
            assert torch.equal(ref_ssm, full_ssm)
        else:
            # Resuming starts from the bf16-rounded state at the point.
            rel = (out_c.float() - full_c[point:].float()).norm() / full_c[point:].float().norm()
            assert rel.item() < 2e-2
        mgr.free_resources(request_c)
    finally:
        mgr.shutdown()
