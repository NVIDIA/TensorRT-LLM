# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA mid-chunk recurrent-state snapshots of the Kimi K3 SM107 targets' cache manager.

The SM107 targets declare a family subclass of ``KimiK3HybridCacheManager`` as
``RuntimeHooks.kv_cache_manager_cls``: ``KimiK3Sm107MidchunkCacheManager`` (tp1), whose KDA prefill
stores the recurrent state at an aligned point inside a context step, or
``KimiK3Sm107CacheManager`` (dep16, helix8), with mid-chunk snapshots off. With them on, the last
snapshot point of a context request no longer splits its context chunk, and a step may pass one
other snapshot point (a branch point, a point requested by a waiting request, an additional
offset). The manager reserves a pending SSM page for the step, the model's KDA metadata fetches the
plan (``get_kda_midchunk_snapshots``) and its prefill writes the snapshot into that page, and the
context update commits the prompt up to the point with it before the usual end-of-step update. A
planned snapshot whose plan was not fetched is released, never committed.

The stub tests need no GPU. The real-manager tests need a GPU and the pending SSM snapshot pages of
the KV cache manager V2 C++ backend; a test writes the states the KDA layers would. The scheduler's
chunking at the marks this manager sets is tested with the scheduler
(``unittest/_torch/pyexecutor/kv_cache/test_hybrid_snapshot_points.py``).
"""

import time
from types import FunctionType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.common.kda_metadata import (
    KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.cache.k3_sm107_cache_manager import (  # noqa: E501
    KimiK3Sm107CacheManager,
    KimiK3Sm107MidchunkCacheManager,
    _KDAMidchunkSnapshot,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
from tensorrt_llm._torch.pyexecutor._util import model_declared_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BlockReusePolicy
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    mamba_cache_manager_v2 as mamba_cache_manager_v2_module,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import _get_forced_context_chunk_size
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig, MambaStateConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_pending_ssm_pages = pytest.mark.skipif(
    not hasattr(_KVCache, "acquire_ssm_snapshot_slot"),
    reason="rebuild needed: the KV cache manager V2 C++ backend of this build has no "
    "pending SSM snapshot pages",
)


def _manager_stub(local_num_mamba_layers=1, alignment=64, cls=KimiK3Sm107MidchunkCacheManager):
    mgr = object.__new__(cls)
    mgr.enable_block_reuse = True
    mgr.is_draft = False
    mgr.local_num_mamba_layers = local_num_mamba_layers
    mgr.block_reuse_policy = BlockReusePolicy.PER_REQUEST
    mgr.conversation_manager = None
    mgr._branch_snapshot_points = {}
    mgr._waiter_snapshot_points = None
    mgr._branch_snapshots_taken_total = 0
    mgr._kda_replay = None
    mgr._kda_midchunk_snapshot_alignment = alignment
    mgr._kda_midchunk_snapshots = {}
    mgr._kda_midchunk_handed_out = set()
    mgr._kda_midchunk_snapshots_taken_total = 0
    mgr._kda_midchunk_snapshots_skipped_total = 0
    mgr._reuse_token_source = lambda request: request.tokens
    mgr._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]
    mgr.kv_cache_map = {}
    # Logged just now, so the rate-limited INFO counters stay quiet.
    mgr._branch_snapshot_counters_logged_at = time.monotonic()
    return mgr


def _context_request(request_id, cur, chunk, prompt_len, points, alignment=64):
    tokens = list(range(1000, 1000 + prompt_len))
    request = SimpleNamespace(
        py_request_id=request_id,
        is_dummy_request=False,
        is_dummy=False,
        is_disagg_generation_init_state=False,
        context_current_position=cur,
        context_chunk_size=chunk,
        prompt_len=prompt_len,
        expect_snapshot_points=points,
        py_midchunk_snapshot_alignment=alignment,
        tokens=tokens,
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cls, expected",
    [
        (KimiK3Sm107MidchunkCacheManager, KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT),
        (KimiK3Sm107CacheManager, None),
    ],
)
def test_alignment_follows_the_declared_class(cls, expected):
    mgr = _manager_stub(alignment=None, cls=cls)
    assert mgr._get_kda_midchunk_snapshot_alignment() == expected


@pytest.mark.cpu_only
def test_the_midchunk_alignment_is_the_kda_prefill_chunk():
    assert KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT == 64
    assert KimiK3Sm107MidchunkCacheManager.KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT == 64
    assert KimiK3Sm107CacheManager.KDA_MIDCHUNK_SNAPSHOT_ALIGNMENT == 0


@pytest.mark.cpu_only
def test_alignment_is_off_for_draft_or_without_block_reuse():
    mgr = _manager_stub(alignment=None)
    mgr.is_draft = True
    assert mgr._get_kda_midchunk_snapshot_alignment() is None
    mgr.is_draft = False
    mgr.enable_block_reuse = False
    assert mgr._get_kda_midchunk_snapshot_alignment() is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("cls", [KimiK3Sm107CacheManager, KimiK3Sm107MidchunkCacheManager])
def test_the_engine_builds_the_declared_class(cls):
    """A subclass of the class the engine resolves for Kimi K3, declared on the model class."""

    class _Declares:
        runtime_hooks = RuntimeHooks(kv_cache_manager_cls=cls)

    assert issubclass(cls, KimiK3HybridCacheManager)
    assert model_declared_kv_cache_manager_cls(_Declares(), KimiK3HybridCacheManager) is cls
    # The DSpark worker's fused accept path requires the engine's KDA state update.
    assert cls.update_mamba_states is KimiK3HybridCacheManager.update_mamba_states
    assert cls.accepts_appended_draft_layers()


@pytest.mark.cpu_only
@pytest.mark.parametrize("cls", [KimiK3Sm107CacheManager, KimiK3Sm107MidchunkCacheManager])
def test_inherited_method_overrides_are_annotated(cls):
    for name, attribute in vars(cls).items():
        function = attribute.__func__ if isinstance(attribute, classmethod) else attribute
        if not isinstance(function, FunctionType):
            continue
        inherited = any(name in vars(parent) for parent in cls.__mro__[1:] if parent is not object)
        assert getattr(function, "__override__", False) == inherited, f"{cls.__name__}.{name}"


@pytest.mark.cpu_only
def test_shutdown_drops_the_planned_snapshots_before_the_pools(monkeypatch):
    mgr = _manager_stub()
    mgr._kda_midchunk_snapshots = {42: object()}
    mgr._kda_midchunk_handed_out = {42}

    def check_before_base_shutdown():
        assert not mgr._kda_midchunk_snapshots and not mgr._kda_midchunk_handed_out

    base_shutdown = Mock(side_effect=check_before_base_shutdown)
    monkeypatch.setattr(KimiK3HybridCacheManager, "shutdown", base_shutdown)
    mgr.shutdown()
    base_shutdown.assert_called_once_with()


@pytest.mark.cpu_only
def test_waiter_point_after_scheduling_marks_the_pass_point():
    """A point requested by a waiting request is passed inside a step like a branch point."""
    mgr = _manager_stub()
    mgr.tokens_per_block = 128
    request, _ = _context_request(7, 0, 0, 1000, [1000])
    mgr._waiter_snapshot_points = {7: {512}}
    mgr._merge_waiter_snapshot_points(request)
    assert request.expect_snapshot_points == [512, 1000]
    assert request.py_midchunk_pass_point == 512
    assert _get_forced_context_chunk_size(request) == 1000


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cur,alignment,dummy,expected",
    [
        (0, 64, False, 256),
        (256, 64, False, None),  # already passed
        (0, None, False, None),  # no mid-chunk snapshots
        (0, 64, True, None),
    ],
)
def test_apply_branch_snapshot_point_marks_midchunk_pass_point(cur, alignment, dummy, expected):
    mgr = _manager_stub(alignment=alignment)
    mgr.tokens_per_block = 128
    mgr._branch_snapshot_points[7] = 256
    request, _ = _context_request(7, cur, 700 - cur, 700, [640])
    request.is_dummy_request = dummy
    mgr._apply_branch_snapshot_point(request)
    assert request.py_midchunk_pass_point == expected
    assert request.expect_snapshot_points == ([256, 640] if cur < 256 else [640])


@pytest.mark.cpu_only
def test_prepare_expect_snapshot_points_marks_the_next_point_to_pass(monkeypatch):
    """Offset points before the prompt end (e.g. offsets_from_end {0, 2K, 8K})
    are taken inside a step too: one per step, the last one merges."""
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_record_branch_snapshot_point_marks_the_pass_point():
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


def _counters_stub():
    mgr = _manager_stub()
    mgr.tokens_per_block = 128
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        block_reuse_config=BlockReuseConfig(enable_branch_snapshot=True),
    )
    mgr._snapshot_pruned_tokens_total = 0
    mgr._page_pruned_tokens_total = 0
    mgr._branch_snapshots_skipped_total = {}
    return mgr


@pytest.mark.cpu_only
def test_snapshot_counters_are_logged_at_info_once_a_minute(monkeypatch):
    mgr = _counters_stub()
    mgr._branch_snapshot_counters_logged_at = float("-inf")
    clock = iter([1000.0, 1030.0, 1061.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    info = MagicMock()
    monkeypatch.setattr(mamba_cache_manager_v2_module.logger, "info", info)
    request, _ = _context_request(1, 0, 1000, 1000, [1000])
    kv_cache = SimpleNamespace(
        _get_num_reusable_tokens_before_pruning=lambda: 0,
        _get_num_reusable_tokens_before_hybrid_pruning=lambda: 0,
        num_committed_tokens=0,
    )
    for _ in range(3):
        mgr._record_branch_snapshot_point(request, kv_cache, num_lookup_tokens=999)
    assert info.call_count == 2
    line = info.call_args.args[0]
    assert line.startswith("[MambaHybridCacheManagerV2] recurrent snapshot counters ")
    assert "inflight_prefix_deferrals=0" in line
    assert "midchunk_snapshots_taken=0" in line


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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
    return KimiK3Sm107MidchunkCacheManager(
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
def test_real_manager_one_step_snapshot_is_reused():
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
def test_real_manager_branch_snapshot_inside_step_is_reused_by_sibling(policy):
    """Sibling B leaves A's cached prefix at 64: it is prefilled in one step with
    a mid-chunk snapshot at 64, which sibling C then reuses."""
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
