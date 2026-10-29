# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-flight prefix wait of KVCacheManagerV2.

Agent fan-outs send
many requests at once that share a long prompt prefix. Each one looks up the reuse tree
before any sibling has committed the prefix, so all of them compute it. With
``TRTLLM_KV_V2_WAIT_FOR_INFLIGHT_PREFIX=1`` the manager keeps such a request queued while
another request prefills the shared prefix, and asks that request (the owner) to take a
recurrent-state snapshot at the shared depth, which is a forced context chunk boundary.

Layout: the prefix index and the wait planner run on the CPU with stub managers, the
scheduler hook with mocks, and the end-to-end flow with a real hybrid V2 manager (CUDA).
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _InflightPrefixIndex,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MambaHybridCacheManagerV2,
    _inflight_prefix_wait_blocker,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import _get_forced_context_chunk_size
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import (
    BlockReuseConfig,
    CapacitySchedulerPolicy,
    KvCacheConfig,
    MambaStateConfig,
)
from tensorrt_llm.mapping import Mapping

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS_PER_BLOCK = 32


# ---------------------------------------------------------------------------
# _InflightPrefixIndex
# ---------------------------------------------------------------------------
def _index_request(request_id: int, remaining: int = 10) -> SimpleNamespace:
    return SimpleNamespace(py_request_id=request_id, context_remaining_length=remaining)


def test_fingerprints_cover_full_units_and_leave_out_the_last_prompt_token():
    index = _InflightPrefixIndex(unit_tokens=4)

    # The last prompt token is recomputed, so 13 tokens hold (13 - 1) // 4 = 3 units.
    assert len(index.fingerprints(_index_request(1), list(range(13)), b"s")) == 3
    assert len(index.fingerprints(_index_request(2), list(range(12)), b"s")) == 2
    assert index.fingerprints(_index_request(3), [1, 2, 3], b"s") == []


def test_fingerprints_depend_on_the_prefix_and_the_reuse_scope_only():
    index = _InflightPrefixIndex(unit_tokens=4)
    shared = list(range(8))

    first = index.fingerprints(_index_request(1), shared + [100, 101, 102, 103, 104], b"s")
    sibling = index.fingerprints(_index_request(2), shared + [200, 201, 202, 203, 204], b"s")
    other_scope = index.fingerprints(_index_request(3), shared + [100, 101, 102, 103, 104], b"t")

    assert first[:2] == sibling[:2]
    assert first[2] != sibling[2]
    assert all(a != b for a, b in zip(first, other_scope))
    # Deterministic across instances: every rank takes the same decisions.
    again = _InflightPrefixIndex(unit_tokens=4).fingerprints(
        _index_request(1), shared + [100, 101, 102, 103, 104], b"s"
    )
    assert again == first


def test_inflight_prefix_index_finds_the_deepest_prefilling_owner():
    index = _InflightPrefixIndex(unit_tokens=4)
    owner = _index_request(1, remaining=20)
    waiter = _index_request(2, remaining=17)
    other_scope = _index_request(3, remaining=17)
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


def test_index_remove_forgets_the_owner_the_fingerprints_and_the_wait_budget():
    index = _InflightPrefixIndex(unit_tokens=4)
    owner, waiter = _index_request(1, 20), _index_request(2, 20)
    index.fingerprints(owner, list(range(21)), b"s")
    waiter_fps = index.fingerprints(waiter, list(range(21)), b"s")
    index.add(owner)
    index.add(owner)  # idempotent
    index.wait_passes[2] = 3
    assert index.deepest_owner(waiter_fps, 2) == (owner, 5)

    index.remove(1)
    assert index.deepest_owner(waiter_fps, 2) == (None, 0)
    index.remove(2)
    assert index.wait_passes == {}
    assert index.fingerprints(waiter, list(range(8)), b"s") != waiter_fps  # recomputed


# ---------------------------------------------------------------------------
# KVCacheManagerV2.plan_context_waits on a stub manager
# ---------------------------------------------------------------------------
class _PlannerManager(KVCacheManagerV2):
    """A KVCacheManagerV2 reduced to what plan_context_waits reads (no pools, no GPU)."""

    def __init__(self, reusable: int = 0, owner_takes_snapshot: bool = True) -> None:
        self.tokens_per_block = 4
        self.is_draft = False
        self.enable_block_reuse = True
        self.kv_cache_map = {}
        self._inflight_prefix = _InflightPrefixIndex(unit_tokens=4)  # one block per unit
        self.reusable = reusable
        self.owner_takes_snapshot = owner_takes_snapshot
        self.probes = 0
        self.requested = []

    def _reuse_token_source(self, req):
        return req.tokens

    def probe_context_reuse(self, req):
        self.probes += 1
        return self.reusable

    def _request_snapshot_point(self, owner, point):
        self.requested.append((owner.py_request_id, point))
        return self.owner_takes_snapshot


def _ctx_request(request_id: int, tokens: list, position: int = 0, **overrides) -> SimpleNamespace:
    fields = dict(
        py_request_id=request_id,
        tokens=tokens,
        prompt_len=len(tokens),
        context_current_position=position,
        context_remaining_length=len(tokens) - position,
        is_first_context_chunk=position == 0,
        lora_task_id=None,
        cache_salt=None,
        is_dummy_request=False,
        is_disagg_generation_init_state=False,
        py_mm_run_metadata=None,  # memoized "text only", see _resolve_multimodal_run_metadata
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


SHARED = list(range(100, 116))  # four units of four tokens


def _sibling(request_id: int, tail_start: int, **overrides) -> SimpleNamespace:
    return _ctx_request(request_id, SHARED + list(range(tail_start, tail_start + 8)), **overrides)


@pytest.fixture
def small_wait_constants(monkeypatch):
    """Two blocks (8 tokens) are enough to wait; three passes at most."""
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_MIN_GAIN_BLOCKS", 2)
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_MAX_WAIT_PASSES", 3)


def test_sibling_waits_for_the_request_ahead_of_it_in_the_pass(small_wait_constants):
    mgr = _PlannerManager()
    owner, sibling = _sibling(1, 300), _sibling(2, 400)

    assert mgr.plan_context_waits([owner, sibling]) == {2}

    # The owner is asked for a snapshot at the shared depth: four units of four tokens.
    assert mgr.requested == [(1, 16)]
    assert mgr._inflight_prefix.deferrals_total == 1
    assert mgr._inflight_prefix.wait_passes == {2: 1}


def test_every_sibling_of_a_pass_waits_for_the_same_owner(small_wait_constants):
    """A waiting request never becomes the owner of the requests behind it."""
    mgr = _PlannerManager()
    owner, second, third = _sibling(1, 300), _sibling(2, 400), _sibling(3, 500)

    assert mgr.plan_context_waits([owner, second, third]) == {2, 3}
    assert mgr.requested == [(1, 16), (1, 16)]


def test_sibling_waits_for_a_prefilling_request_that_holds_a_cache(small_wait_constants):
    mgr = _PlannerManager()
    owner, sibling = _sibling(1, 300, position=8), _sibling(2, 400)
    mgr.kv_cache_map[1] = object()
    mgr._inflight_prefix.fingerprints(owner, owner.tokens, b"(None, None)")
    mgr._inflight_prefix.add(owner)

    assert mgr.plan_context_waits([owner, sibling]) == {2}
    assert mgr.requested == [(1, 16)]


def test_unrelated_prompts_do_not_wait(small_wait_constants):
    mgr = _PlannerManager()
    first = _ctx_request(1, list(range(0, 24)))
    second = _ctx_request(2, list(range(500, 524)))

    assert mgr.plan_context_waits([first, second]) == set()
    assert mgr.requested == []
    assert mgr.probes == 0


def test_no_wait_when_the_reuse_tree_already_offers_the_prefix(small_wait_constants):
    mgr = _PlannerManager(reusable=16)

    assert mgr.plan_context_waits([_sibling(1, 300), _sibling(2, 400)]) == set()
    assert mgr.requested == []


def test_no_wait_below_the_minimum_gain_and_no_probe_either(small_wait_constants, monkeypatch):
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_MIN_GAIN_BLOCKS", 5)
    mgr = _PlannerManager()  # the shared 16 tokens are less than 5 blocks of 4 tokens

    assert mgr.plan_context_waits([_sibling(1, 300), _sibling(2, 400)]) == set()
    assert mgr.probes == 0


def test_the_gain_is_measured_against_what_the_tree_offers(small_wait_constants):
    mgr = _PlannerManager(reusable=12)  # 16 shared - 12 reusable = 4 < 8 tokens

    assert mgr.plan_context_waits([_sibling(1, 300), _sibling(2, 400)]) == set()
    assert mgr.probes == 1


def test_a_request_waits_a_bounded_number_of_passes(small_wait_constants):
    mgr = _PlannerManager()
    owner, sibling = _sibling(1, 300), _sibling(2, 400)

    assert [mgr.plan_context_waits([owner, sibling]) for _ in range(5)] == [
        {2},
        {2},
        {2},
        set(),
        set(),
    ]
    assert mgr._inflight_prefix.deferrals_total == 3


def test_no_wait_when_the_owner_cannot_take_the_snapshot(small_wait_constants):
    mgr = _PlannerManager(owner_takes_snapshot=False)

    assert mgr.plan_context_waits([_sibling(1, 300), _sibling(2, 400)]) == set()
    assert mgr._inflight_prefix.deferrals_total == 0
    assert mgr._inflight_prefix.wait_passes == {}


def test_requests_that_hold_a_cache_or_resume_a_chunk_are_never_deferred(small_wait_constants):
    mgr = _PlannerManager()
    owner = _sibling(1, 300)
    claimed = _sibling(2, 400)
    mgr.kv_cache_map[2] = object()
    later_chunk = _sibling(3, 500, position=8, is_first_context_chunk=False)

    assert mgr.plan_context_waits([owner, claimed, later_chunk]) == set()
    assert mgr.probes == 0


@pytest.mark.parametrize(
    "override",
    [
        dict(is_dummy_request=True),
        dict(is_disagg_generation_init_state=True),
        dict(py_mm_run_metadata=object()),  # multimodal: placeholder ids do not identify content
    ],
    ids=["dummy", "disagg_generation_init", "multimodal"],
)
def test_requests_the_fingerprints_cannot_describe_never_wait(small_wait_constants, override):
    mgr = _PlannerManager()

    assert mgr.plan_context_waits([_sibling(1, 300), _sibling(2, 400, **override)]) == set()


def test_draft_managers_and_disabled_waits_plan_nothing(small_wait_constants):
    mgr = _PlannerManager()
    mgr.is_draft = True
    assert mgr.plan_context_waits([_sibling(1, 300), _sibling(2, 400)]) == set()

    mgr = _PlannerManager()
    mgr._inflight_prefix = None
    assert mgr.plan_context_waits([_sibling(1, 300), _sibling(2, 400)]) == set()


def test_attention_only_managers_always_accept_a_wait():
    """Attention blocks are committed when the owner's context ends at the latest."""
    mgr = _PlannerManager()
    assert KVCacheManagerV2._request_snapshot_point(mgr, _sibling(1, 300), 16) is True


def test_scopes_do_not_share_a_prefix(small_wait_constants):
    mgr = _PlannerManager()
    other_salt = _sibling(2, 400, cache_salt="tenant-b")

    assert mgr.plan_context_waits([_sibling(1, 300), other_salt]) == set()


# ---------------------------------------------------------------------------
# MambaHybridCacheManagerV2: requested snapshot points
# ---------------------------------------------------------------------------
def _bare_hybrid_manager(enable_branch_snapshot: bool = True, tokens_per_block: int = 64):
    """A hybrid V2 manager carrying only what the snapshot-point bookkeeping reads."""
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.enable_block_reuse = True
    mgr.tokens_per_block = tokens_per_block
    mgr.local_num_mamba_layers = 1
    mgr.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_state_config=MambaStateConfig(
            periodic_snapshot_interval=0,
            additional_snapshot_offsets_from_end=[0],
            enable_branch_snapshot=enable_branch_snapshot,
        ),
    )
    mgr._branch_snapshot_points = {}
    mgr._requested_snapshot_points = {}
    mgr._branch_snapshots_taken_total = 0
    mgr._inflight_prefix = _InflightPrefixIndex(unit_tokens=tokens_per_block)
    return mgr


def _owner(request_id: int = 1, prompt_len: int = 512, position: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        prompt_len=prompt_len,
        context_current_position=position,
        expect_snapshot_points=[],
    )


def test_requested_point_is_aligned_down_and_joins_the_expected_points():
    mgr = _bare_hybrid_manager()
    owner = _owner()

    assert mgr._request_snapshot_point(owner, 200)

    assert owner.expect_snapshot_points == [192, 512]
    assert mgr._inflight_prefix.points_total == 1
    # The same point again is accepted without counting it twice.
    assert mgr._request_snapshot_point(owner, 223)
    assert mgr._inflight_prefix.points_total == 1


@pytest.mark.parametrize(
    "point, position, prompt_len",
    [
        (192, 192, 512),  # the owner already is at the point
        (192, 256, 512),  # the owner passed it
        (512, 0, 512),  # the prompt end is snapshotted anyway
        (640, 0, 512),  # beyond the prompt
        (40, 0, 512),  # aligns down to zero, never ahead of the cursor
    ],
)
def test_snapshot_point_is_refused_when_the_owner_cannot_take_it(point, position, prompt_len):
    mgr = _bare_hybrid_manager()
    owner = _owner(prompt_len=prompt_len, position=position)

    assert not mgr._request_snapshot_point(owner, point)
    assert owner.expect_snapshot_points == []
    assert mgr._requested_snapshot_points == {}


def test_snapshot_point_is_refused_without_branch_snapshots():
    mgr = _bare_hybrid_manager(enable_branch_snapshot=False)

    assert not mgr._request_snapshot_point(_owner(), 192)


def test_requested_points_survive_the_next_scheduler_pass_until_the_owner_passes_them():
    mgr = _bare_hybrid_manager()
    owner = _owner()
    assert mgr._request_snapshot_point(owner, 192)

    # prepare_expect_snapshot_points rebuilds the list on every pass.
    mgr.prepare_expect_snapshot_points([owner])
    assert owner.expect_snapshot_points == [192, 512]

    owner.context_current_position = 192
    mgr.prepare_expect_snapshot_points([owner])
    assert owner.expect_snapshot_points == [512]


def test_requested_and_branch_points_merge_sorted_and_deduplicated():
    mgr = _bare_hybrid_manager()
    owner = _owner()
    mgr._branch_snapshot_points[1] = 320
    assert mgr._request_snapshot_point(owner, 128)
    assert mgr._request_snapshot_point(owner, 320)  # coincides with the branch point

    mgr.prepare_expect_snapshot_points([owner])

    assert owner.expect_snapshot_points == [128, 320, 512]
    assert mgr._branch_like_points(owner, 0) == [128, 320]
    assert mgr._branch_like_points(owner, 128) == [320]


def test_snapshot_commits_are_counted_per_kind():
    mgr = _bare_hybrid_manager()
    owner = _owner()
    mgr._branch_snapshot_points[1] = 320
    assert mgr._request_snapshot_point(owner, 128)

    mgr._count_snapshot_commit(owner, 128)  # requested only
    assert (mgr._branch_snapshots_taken_total, mgr._inflight_prefix.points_taken_total) == (0, 1)
    mgr._count_snapshot_commit(owner, 320)  # branch only
    assert (mgr._branch_snapshots_taken_total, mgr._inflight_prefix.points_taken_total) == (1, 1)
    mgr._count_snapshot_commit(owner, 512)  # prompt end: neither
    assert (mgr._branch_snapshots_taken_total, mgr._inflight_prefix.points_taken_total) == (1, 1)


def test_snapshot_bookkeeping_tolerates_a_manager_without_the_wait():
    """Bare managers (and the pre-wait tests) never create the requested-point table."""
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr._branch_snapshot_points = {1: 192}
    mgr._branch_snapshots_taken_total = 0
    mgr._inflight_prefix = None
    owner = _owner()

    assert mgr._branch_like_points(owner, 0) == [192]
    mgr._count_snapshot_commit(owner, 192)
    assert mgr._branch_snapshots_taken_total == 1


def test_counters_line_reports_the_wait_only_when_it_is_enabled():
    mgr = _bare_hybrid_manager()
    mgr._snapshot_pruned_tokens_total = 0
    mgr._page_pruned_tokens_total = 0
    mgr._branch_snapshots_skipped_total = {}
    mgr._inflight_prefix.deferrals_total = 4
    mgr._inflight_prefix.points_total = 2
    mgr._inflight_prefix.points_taken_total = 1

    line = mgr._format_branch_snapshot_counters()
    assert "inflight_prefix_deferrals=4 inflight_prefix_points_requested=2" in line
    assert line.endswith("inflight_prefix_points_taken=1")

    mgr._inflight_prefix = None
    assert "inflight_prefix" not in mgr._format_branch_snapshot_counters()


# ---------------------------------------------------------------------------
# KVCacheV2Scheduler: deferred requests are skipped before they claim a cache
# ---------------------------------------------------------------------------
class _AutoKVCacheMap(dict):
    def __missing__(self, key):
        entry = Mock()
        entry.is_active = True
        self[key] = entry
        return entry


def _mock_ctx_request(request_id: int, length: int) -> Mock:
    req = Mock()
    req.request_id = req.py_request_id = request_id
    req.state_value = LlmRequestState.CONTEXT_INIT.value
    req.context_remaining_length = req.prompt_len = length
    req.context_current_position = 0
    req.context_chunk_size = 0
    req.expect_snapshot_points = []
    req.num_draft_tokens = 0
    req.has_draft_tokens = False
    req.py_draft_tokens = []
    req.is_first_context_chunk = req.is_last_context_chunk = True
    req.lora_task_id = None
    req.is_context_init_state = True
    req.is_generation_in_progress_state = False
    req.encoder_output_len = None
    req.py_encoder_output_ready_event = None
    req.py_skip_cross_kv_projection = False
    return req


@pytest.mark.parametrize("chunk_config", [(None, 64), None], ids=["chunked", "unchunked"])
@pytest.mark.parametrize("enabled", [True, False], ids=["wait_on", "wait_off"])
def test_scheduler_skips_deferred_context_without_preparing(monkeypatch, chunk_config, enabled):
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    monkeypatch.setattr(kv_cache_manager_v2, "_WAIT_FOR_INFLIGHT_PREFIX", enabled)
    mgr = Mock()
    mgr.tokens_per_block = 64
    mgr.can_evict = False
    mgr._has_cp_helix = False
    mgr.kv_cache_map = _AutoKVCacheMap()
    mgr.prepare_context.side_effect = lambda req: True
    mgr.resize_context.side_effect = lambda req, n: True
    mgr.plan_context_waits.side_effect = lambda reqs: {1}
    with patch(
        "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.KVCacheManagerV2", new=type(mgr)
    ):
        sched = KVCacheV2Scheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            kv_cache_manager=mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            ctx_chunk_config=chunk_config,
        )
    reqs = [_mock_ctx_request(i, 500) for i in range(3)]

    out = sched.schedule_request(reqs, set())

    scheduled = [req.py_request_id for req in out.context_requests]
    prepared = [call.args[0].py_request_id for call in mgr.prepare_context.call_args_list]
    if enabled:
        assert scheduled == [0, 2] and 1 not in prepared
    else:
        assert scheduled == [0, 1, 2]
        mgr.plan_context_waits.assert_not_called()


# ---------------------------------------------------------------------------
# End to end on a real hybrid V2 manager (CUDA)
# ---------------------------------------------------------------------------
def _build_hybrid_manager(
    enable_branch_snapshot: bool = True, mapping: Mapping = None
) -> MambaHybridCacheManagerV2:
    return MambaHybridCacheManagerV2(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            host_cache_size=0,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="per_conversation", max_num_turns=1),
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0,
                additional_snapshot_offsets_from_end=[0],
                enable_branch_snapshot=enable_branch_snapshot,
            ),
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=128,
        max_batch_size=4,
        mapping=mapping or Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        layer_mask=[False, True],
        vocab_size=1024,
        dtype=DataType.HALF,
        conv_state_layout="x_b_c",
    )


def _conversation_request(request_id: int, tokens: list, conversation_id: str) -> LlmRequest:
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    request.py_conversation_params = ConversationParams(conversation_id=conversation_id)
    return request


@pytest.fixture
def wait_for_inflight_prefix(monkeypatch):
    """Wait on one-block units as soon as one block can be gained; two passes at most."""
    monkeypatch.setattr(kv_cache_manager_v2, "_WAIT_FOR_INFLIGHT_PREFIX", True)
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_UNIT_BLOCKS", 1)
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_MIN_GAIN_BLOCKS", 1)
    monkeypatch.setattr(kv_cache_manager_v2, "_INFLIGHT_PREFIX_MAX_WAIT_PASSES", 2)


@skip_no_cuda
def test_real_manager_sibling_waits_for_the_prefix_then_reuses_the_requested_snapshot(
    wait_for_inflight_prefix,
):
    """Y arrives while X prefills their shared 64 tokens. Y waits, X ends a chunk at the
    requested point and snapshots there, and Y then resumes from X's snapshot."""
    shared = list(range(100, 164))  # two blocks
    mgr = _build_hybrid_manager()
    try:
        ssm, conv = mgr.all_ssm_states[0], mgr.all_conv_states[0]
        snapshot_ssm = torch.randn_like(ssm[0], dtype=torch.float32).to(ssm.dtype)
        snapshot_conv = torch.randn_like(conv[0], dtype=torch.float32).to(conv.dtype)
        x = _conversation_request(1, shared + list(range(300, 340)), "conv-x")
        y = _conversation_request(2, shared + list(range(400, 430)), "conv-y")

        # X claimed its cache in an earlier pass; Y waits for it.
        mgr.prepare_expect_snapshot_points([x, y])
        assert x.expect_snapshot_points == [104]
        assert mgr.prepare_context(x)
        assert mgr.plan_context_waits([x, y]) == {2}
        assert 2 not in mgr.kv_cache_map
        # The shared depth became a forced chunk boundary of X.
        assert x.expect_snapshot_points == [64, 104]
        chunk = _get_forced_context_chunk_size(x)
        assert chunk == 64

        # X's first step ends at the point; its recurrent state is committed there.
        assert mgr.resize_context(x, num_tokens=chunk)
        batch = ScheduledRequests()
        batch.append_context_request(x)
        mgr.prepare_resources(batch)
        slot = mgr.get_state_indices([1], [False])[0]
        ssm[slot].copy_(snapshot_ssm)
        conv[slot].copy_(snapshot_conv)
        torch.cuda.synchronize()
        x.context_chunk_size = chunk
        x.context_current_position = chunk
        mgr.update_context_resources(batch)
        assert mgr._inflight_prefix.points_taken_total == 1
        assert "inflight_prefix_points_taken=1" in mgr._format_branch_snapshot_counters()

        # X still prefills its tail, but the tree now offers the shared depth: no more waiting.
        assert x.context_remaining_length > 0
        mgr.prepare_expect_snapshot_points([x, y])
        assert mgr.plan_context_waits([x, y]) == set()
        assert mgr.prepare_context(y)
        assert y.context_current_position == 64
        assert mgr.resize_context(y, num_tokens=y.context_remaining_length)
        batch = ScheduledRequests()
        batch.append_context_request(y)
        mgr.prepare_resources(batch)
        torch.cuda.synchronize()
        reused = mgr.get_state_indices([2], [False])[0]
        assert torch.equal(ssm[reused], snapshot_ssm)
        assert torch.equal(conv[reused], snapshot_conv)
        assert mgr._inflight_prefix.deferrals_total == 1

        # Nothing is left behind once the requests are gone.
        mgr.free_resources(x)
        mgr.free_resources(y)
        assert mgr._requested_snapshot_points == {}
        assert mgr._inflight_prefix._prefilling == {}
        assert mgr._inflight_prefix.wait_passes == {}
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_real_manager_waiter_stops_waiting_when_the_owner_is_gone(wait_for_inflight_prefix):
    shared = list(range(100, 164))
    mgr = _build_hybrid_manager()
    try:
        x = _conversation_request(1, shared + list(range(300, 340)), "conv-x")
        y = _conversation_request(2, shared + list(range(400, 430)), "conv-y")
        mgr.prepare_expect_snapshot_points([x, y])
        assert mgr.prepare_context(x)
        assert mgr.plan_context_waits([x, y]) == {2}
        mgr.free_resources(x)

        # The owner is gone: the wait ends although the budget is not spent.
        mgr.prepare_expect_snapshot_points([y])
        assert mgr.plan_context_waits([y]) == set()
        # The waiter stays free of a cache claim until the scheduler prepares it.
        assert 2 not in mgr.kv_cache_map
    finally:
        mgr.shutdown()


@skip_no_cuda
def test_real_manager_disables_the_wait_without_branch_snapshots(wait_for_inflight_prefix):
    mgr = _build_hybrid_manager(enable_branch_snapshot=False)
    try:
        assert mgr._inflight_prefix is None
        x = _conversation_request(1, list(range(100, 204)), "conv-x")
        y = _conversation_request(2, list(range(100, 194)), "conv-y")
        assert mgr.plan_context_waits([x, y]) == set()
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Which ranks may run the wait: the lock-step ones (pipeline ranks, tensor-parallel ranks without attention DP) may not
# ---------------------------------------------------------------------------
def _wait_blocker(world_size, tp_size, pp_size, attention_dp, branch_snapshot=True):
    mapping = Mapping(
        world_size=world_size,
        rank=0,
        tp_size=tp_size,
        pp_size=pp_size,
        enable_attention_dp=attention_dp,
    )
    config = KvCacheConfig(
        mamba_state_config=MambaStateConfig(enable_branch_snapshot=branch_snapshot)
    )
    return _inflight_prefix_wait_blocker(config, mapping)


@pytest.mark.parametrize(
    "world_size,tp_size,pp_size,attention_dp,branch_snapshot,reason",
    [
        (1, 1, 1, False, True, None),  # one rank (a DEP16 context server is one GPU)
        (2, 2, 1, False, True, "tp_size is 2 without attention DP"),
        (8, 8, 1, False, True, "tp_size is 8 without attention DP"),  # the TP8 recipes
        (2, 2, 1, True, True, None),  # attention DP: every rank schedules its own requests
        (16, 16, 1, True, True, None),
        (2, 1, 2, False, True, "pp_size is 2"),
        (4, 2, 2, True, True, "pp_size is 2"),
        (1, 1, 1, False, False, "enable_branch_snapshot is off"),
        (8, 8, 1, True, False, "enable_branch_snapshot is off"),
    ],
)
def test_wait_blocker_names_why_a_rank_group_cannot_run_the_wait(
    world_size, tp_size, pp_size, attention_dp, branch_snapshot, reason
):
    blocker = _wait_blocker(world_size, tp_size, pp_size, attention_dp, branch_snapshot)
    if reason is None:
        assert blocker is None
    else:
        assert blocker is not None and reason in blocker


@skip_no_cuda
@pytest.mark.parametrize(
    "attention_dp,wait_runs",
    [(False, False), (True, True)],
    ids=["tp2_lock_step", "tp2_attention_dp"],
)
def test_real_manager_disables_the_wait_for_lock_step_tp_ranks(
    wait_for_inflight_prefix, attention_dp, wait_runs
):
    mapping = Mapping(world_size=2, rank=0, tp_size=2, pp_size=1, enable_attention_dp=attention_dp)
    mgr = _build_hybrid_manager(mapping=mapping)
    try:
        assert (mgr._inflight_prefix is not None) is wait_runs
        x = _conversation_request(1, list(range(100, 204)), "conv-x")
        y = _conversation_request(2, list(range(100, 194)), "conv-y")
        if not wait_runs:
            assert mgr.plan_context_waits([x, y]) == set()
    finally:
        mgr.shutdown()
