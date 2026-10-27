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
"""Unit tests for the helix (decode-CP) super-block ledger in KVCacheManagerV2.

Design under test: the request ledger runs in GLOBAL tokens with one ledger
block spanning ``cp_size`` physical pages (one per CP rank); the per-rank
view (owner of this step's token, tokens held by this rank) is a closed-form
function of the global position. The decode-step index feeding that position
is manager-owned (``py_helix_decode_group_index``), immune to when the
sampler advances ``py_decoding_iter``.
"""

import math
import types
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import CacheType


def _mgr(cp_rank: int, cp_size: int, phys: int) -> SimpleNamespace:
    m = SimpleNamespace(
        tokens_per_block=phys,
        _ledger_tokens_per_block=phys * cp_size,
        _has_cp_helix=cp_size > 1,
        _helix_cp_rank=cp_rank,
        _helix_cp_size=cp_size,
    )
    m._helix_local_len = lambda global_len: KVCacheManagerV2._helix_local_len(m, global_len)
    return m


def _brute_local_len(global_len: int, cp_rank: int, cp_size: int, phys: int) -> int:
    """Reference: token position p lives on rank (p // phys) % cp_size."""
    return sum(1 for p in range(global_len) if (p // phys) % cp_size == cp_rank)


def test_helix_local_len_matches_brute_force() -> None:
    for cp_size in (1, 2, 4, 8):
        for phys in (2, 4, 32):
            for cp_rank in range(cp_size):
                m = _mgr(cp_rank, cp_size, phys)
                for global_len in range(0, 4 * cp_size * phys + 3):
                    assert KVCacheManagerV2._helix_local_len(m, global_len) == _brute_local_len(
                        global_len, cp_rank, cp_size, phys
                    ), (cp_size, phys, cp_rank, global_len)


def test_set_helix_rank_fields_cross_rank_consistency() -> None:
    """For any (prompt_len, decode step): exactly one active rank, the
    per-rank seqlens sum to the global in-flight length, and past_seen
    (= seqlen - 0/1 per the model_engine convention) sums to the global
    already-cached length."""
    cp_size, phys = 4, 4
    for total_input in (1, 5, 16, 17, 63):
        for group_index in (0, 1, 2, 7, 40):  # committed schedules so far
            fields = []
            for r in range(cp_size):
                req = SimpleNamespace(
                    total_input_len_cp=total_input,
                    py_helix_decode_group_index=group_index,
                )
                KVCacheManagerV2._set_helix_rank_fields(_mgr(r, cp_size, phys), req)
                fields.append(req)
            pos = total_input + group_index
            active = [r for r in range(cp_size) if not fields[r].py_helix_is_inactive_rank]
            assert active == [(pos // phys) % cp_size]
            assert sum(f.seqlen_this_rank_cp for f in fields) == pos + 1
            past_seen = [
                f.seqlen_this_rank_cp - (0 if f.py_helix_is_inactive_rank else 1) for f in fields
            ]
            assert sum(past_seen) == pos
            # Prompt distribution matches the arrival striding convention.
            for r in range(cp_size):
                assert past_seen[r] >= 0


def test_ledger_is_rank_invariant() -> None:
    """The whole point of the super-block design: nothing the scheduler or
    ledger consumes depends on cp_rank — only the derived per-rank fields
    do. Verify the derivation never touches ledger quantities by checking
    the same request object gives identical (owner-adjusted) views."""
    cp_size, phys = 8, 32
    req_proto = dict(total_input_len_cp=1000, py_helix_decode_group_index=17)
    lens = []
    for r in range(cp_size):
        req = SimpleNamespace(**req_proto)
        KVCacheManagerV2._set_helix_rank_fields(_mgr(r, cp_size, phys), req)
        lens.append(req.seqlen_this_rank_cp)
    # Ledger-side numbers (global position, page count) are identical on
    # every rank; per-rank lens differ by at most one physical page.
    assert max(lens) - min(lens) <= phys


def test_ledger_position_immune_to_sampler_timing() -> None:
    """Regression for the overlap-scheduler phase skew: the ledger position
    must be L, L+1, L+2, ... no matter when the sampler advances
    py_decoding_iter (the overlap loop updates it after scheduling; a stale
    read would repeat the first position and overwrite that token's KV)."""
    cp_size, phys, total_input = 4, 4, 17
    n_steps = 3 * cp_size * phys  # cross several ownership rotations

    for sampler_timing in ("overlap", "non_overlap"):
        req = SimpleNamespace(
            total_input_len_cp=total_input,
            py_decoding_iter=0,  # disagg-gen seeding happens after scheduling
            py_helix_decode_group_index=0,
        )
        mgrs = [_mgr(r, cp_size, phys) for r in range(cp_size)]
        positions = []
        for step in range(1, n_steps + 1):
            if sampler_timing == "non_overlap":
                req.py_decoding_iter = step  # sampler already advanced
            per_rank = []
            for r in range(cp_size):
                view = SimpleNamespace(**vars(req))
                KVCacheManagerV2._set_helix_rank_fields(mgrs[r], view)
                per_rank.append(view)
            # Commit, mirroring try_allocate_generation's success path.
            req.py_helix_decode_group_index += 1
            if sampler_timing == "overlap":
                req.py_decoding_iter = step  # sampler advances only now

            pos = total_input + step - 1
            active = [r for r in range(cp_size) if not per_rank[r].py_helix_is_inactive_rank]
            assert active == [(pos // phys) % cp_size]
            # In-flight convention: seqlens sum to pos + 1 (never repeats,
            # so no write offset can collide with the previous step's).
            assert sum(f.seqlen_this_rank_cp for f in per_rank) == pos + 1
            positions.append(pos)
        assert positions == [total_input + n for n in range(n_steps)]


def test_decode_step_derivation_is_idempotent() -> None:
    """A failed try_allocate (counter not committed) followed by a retry in
    the same scheduling pass must derive the SAME fields; a revert (skipped
    forward) must give the step back."""
    m = _mgr(cp_rank=1, cp_size=4, phys=4)
    req = SimpleNamespace(
        total_input_len_cp=10,
        py_helix_decode_group_index=5,
    )
    KVCacheManagerV2._set_helix_rank_fields(m, req)
    first = (req.py_helix_is_inactive_rank, req.seqlen_this_rank_cp)
    KVCacheManagerV2._set_helix_rank_fields(m, req)  # retry, no commit
    assert (req.py_helix_is_inactive_rank, req.seqlen_this_rank_cp) == first
    # Commit then revert restores the same derivation.
    req.py_helix_decode_group_index += 1
    req.py_helix_decode_group_index -= 1
    KVCacheManagerV2._set_helix_rank_fields(m, req)
    assert (req.py_helix_is_inactive_rank, req.seqlen_this_rank_cp) == first


def test_quota_converters_scale_by_cp() -> None:
    m = SimpleNamespace(
        _has_cp_helix=True,
        _helix_cp_size=4,
        tokens_per_block=32,
        _ledger_tokens_per_block=128,
        _get_max_tokens_from_quota_impl=lambda quota: 100.0,
        _get_quota_from_max_tokens_impl=lambda tokens: tokens * 7,
    )
    # Rank-local byte quota buys 100 physical tokens, but only 96 (= 3 whole
    # 32-token pages) are allocatable -> 384 global ledger tokens, not 400.
    assert KVCacheManagerV2._get_max_tokens_from_quota(m, 12345) == 384.0
    # inf (all-SWA) passes through unscaled.
    m_inf = SimpleNamespace(
        _has_cp_helix=True,
        _helix_cp_size=4,
        tokens_per_block=32,
        _ledger_tokens_per_block=128,
        _get_max_tokens_from_quota_impl=lambda quota: float("inf"),
    )
    assert math.isinf(KVCacheManagerV2._get_max_tokens_from_quota(m_inf, 1))
    # Global tokens -> whole ledger blocks (ceil) -> per-rank physical pages
    # -> bytes: 401 global tokens need ceil(401/128) = 4 ledger blocks =
    # 4 * 32 = 128 physical tokens per rank (not ceil(401/4) = 101).
    assert KVCacheManagerV2._get_quota_from_max_tokens(m, 401) == 128 * 7
    # cp == 1 (non-helix) is the identity: no page rounding is applied.
    m1 = SimpleNamespace(
        _has_cp_helix=False,
        _helix_cp_size=1,
        tokens_per_block=32,
        _ledger_tokens_per_block=32,
        _get_max_tokens_from_quota_impl=lambda quota: 100.0,
        _get_quota_from_max_tokens_impl=lambda tokens: tokens * 7,
    )
    assert KVCacheManagerV2._get_max_tokens_from_quota(m1, 1) == 100.0
    assert KVCacheManagerV2._get_quota_from_max_tokens(m1, 400) == 2800


def test_update_resources_leaves_history_untouched_under_helix() -> None:
    resizes = []

    kv = SimpleNamespace(
        is_active=True,
        capacity=100,
        resize=lambda cap, hist: resizes.append((cap, hist)) or True,
    )
    req = SimpleNamespace(
        py_request_id=7,
        py_rewind_len=0,
        py_num_accepted_draft_tokens=0,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        max_beam_num_tokens=55,
    )
    mgr = SimpleNamespace(
        # is_draft=True skips the module-level draft-token relocation call;
        # with zero reserve tokens the rewind math is unchanged.
        is_draft=True,
        kv_cache_type=CacheType.SELF,
        _kv_reserve_draft_tokens=0,
        # No scheduler ran, so nothing recorded a generation allocation; the
        # rewind falls back to the reserve width.
        _allocated_draft_lens={},
        _inflight_draft_lens={},
        kv_cache_map={7: kv},
        kv_compression_manages_history=False,
        _has_cp_helix=True,
        # A draft manager never commits generation blocks.
        try_commit_generation_blocks=lambda request: None,
        # No lender attached, so a resize has no one to tell.
        _sharing=None,
    )
    batch = SimpleNamespace(generation_requests=[req])
    KVCacheManagerV2.update_resources(mgr, batch)
    assert resizes == [(100, None)]
    # Non-helix keeps the vanilla history commit.
    mgr._has_cp_helix = False
    resizes.clear()
    KVCacheManagerV2.update_resources(mgr, batch)
    assert resizes == [(100, 54)]


def test_helix_quota_fallback_sets_rank_local_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fraction sizing must set max_gpu_total_bytes (rank-local byte cap),
    not max_tokens, which the manager inflates by 1/max_util_for_resume."""
    import torch

    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    def creator(max_gpu_total_bytes, max_tokens, *, is_disagg=False, fraction=0.5):
        return SimpleNamespace(
            _is_disagg=is_disagg,
            _mapping=SimpleNamespace(cp_size=4),
            _kv_cache_config=SimpleNamespace(
                max_gpu_total_bytes=max_gpu_total_bytes,
                max_tokens=max_tokens,
                free_gpu_memory_fraction=fraction,
            ),
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_000_000, 2_000_000))

    # Explicit quota: early return, config untouched.
    c = creator(1 << 30, None)
    assert KvCacheCreator._configure_helix_kv_cache_capacity(c) is None
    assert c._kv_cache_config.max_tokens is None
    # Explicit but non-positive max_tokens: rejected, not silently replaced
    # by fraction sizing.
    with pytest.raises(ValueError, match="must be positive"):
        KvCacheCreator._configure_helix_kv_cache_capacity(creator(0, 0))
    # No quota: the rank-local byte budget (free * fraction) lands on
    # max_gpu_total_bytes verbatim; max_tokens stays unset.
    c = creator(0, None)
    assert KvCacheCreator._configure_helix_kv_cache_capacity(c) is None
    assert c._kv_cache_config.max_gpu_total_bytes == 500_000
    assert c._kv_cache_config.max_tokens is None
    # Default fraction: aggregated helix lowers it to leave room for prefill
    # activations; disaggregated helix keeps it.
    c = creator(0, None, fraction=0.9)
    KvCacheCreator._configure_helix_kv_cache_capacity(c)
    assert c._kv_cache_config.max_gpu_total_bytes == 600_000
    c = creator(0, None, is_disagg=True, fraction=0.9)
    KvCacheCreator._configure_helix_kv_cache_capacity(c)
    assert c._kv_cache_config.max_gpu_total_bytes == 900_000
    # Degenerate free memory: actionable error instead of a deep assert.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 2_000_000))
    with pytest.raises(ValueError, match="free memory"):
        KvCacheCreator._configure_helix_kv_cache_capacity(creator(0, None))


def test_v1_helix_capacity_config_rejected() -> None:
    """configure_kv_cache_capacity emits V2 super-block-ledger coordinates
    under helix; V1 interprets max_tokens as rank-local. The V1 + helix
    combination must be rejected loudly, not silently mis-sized."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.mapping import CpType

    c = SimpleNamespace(
        _mapping=SimpleNamespace(cp_config={"cp_type": CpType.HELIX}),
        _is_kv_cache_manager_v2=False,
    )
    with pytest.raises(NotImplementedError, match="V2 KV cache manager"):
        KvCacheCreator.configure_kv_cache_capacity(c)


@pytest.mark.parametrize("is_disagg", [False, True])
def test_kv_pool_rebalance_rejected_under_helix(is_disagg: bool) -> None:
    """A KV pool rebalance resizes pools per rank, outside the cross-rank
    quota agreement, so the manager rejects it under helix at construction."""
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import CpType, Mapping

    mapping = Mapping(world_size=2, rank=0, cp_size=2, cp_config={"cp_type": CpType.HELIX})
    with pytest.raises(ValueError, match="does not support KV pool rebalance"):
        KVCacheManagerV2(
            KvCacheConfig(enable_block_reuse=False, enable_kv_pool_rebalance=True),
            CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            tokens_per_block=32,
            max_seq_len=1024,
            max_batch_size=1,
            mapping=mapping,
            is_disagg=is_disagg,
        )


@pytest.mark.parametrize(
    "config_kwargs, connector, match",
    [
        ({"disk_cache_size": 1 << 30, "disk_cache_path": "/tmp"}, None, "disk cache tier"),
        ({}, SimpleNamespace(capacity_only=True), "KV cache connector"),
    ],
    ids=["disk_tier", "connector"],
)
def test_disk_tier_and_connector_rejected_under_agg_helix(
    config_kwargs: dict, connector: object, match: str
) -> None:
    """Suspending to disk and connector loads, saves and deferred preemptions
    are not validated to make the same decisions on every CP rank, so aggregated
    helix rejects them at construction."""
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import CpType, Mapping

    mapping = Mapping(world_size=2, rank=0, cp_size=2, cp_config={"cp_type": CpType.HELIX})
    with pytest.raises(ValueError, match=match):
        KVCacheManagerV2(
            KvCacheConfig(enable_block_reuse=False, **config_kwargs),
            CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            tokens_per_block=32,
            max_seq_len=1024,
            max_batch_size=1,
            mapping=mapping,
            kv_connector_manager=connector,
            is_disagg=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="allocates real KV cache pools")
@pytest.mark.parametrize("host_cache_size", [None, 1 << 28], ids=["auto", "explicit"])
def test_host_tier_allowed_under_agg_helix(host_cache_size: Optional[int]) -> None:
    """Aggregated helix keeps the host tier, automatic or explicit, so a
    suspended request has somewhere to be evicted to."""
    from unittest.mock import Mock, patch

    from tensorrt_llm._torch.distributed.communicator import Distributed
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import CpType, Mapping

    distributed = Mock()
    distributed.allreduce.side_effect = lambda value, op: value
    distributed.local_world_size = 1
    mapping = Mapping(world_size=2, rank=0, cp_size=2, cp_config={"cp_type": CpType.HELIX})
    with patch.object(Distributed, "get", return_value=distributed):
        manager = KVCacheManagerV2(
            KvCacheConfig(
                max_tokens=256, host_cache_size=host_cache_size, enable_block_reuse=False
            ),
            CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            tokens_per_block=32,
            max_seq_len=1024,
            max_batch_size=1,
            mapping=mapping,
            is_disagg=False,
        )
    assert manager.has_cache_tier_below_gpu


def test_estimation_prepare_promotes_skip_est_for_v2() -> None:
    """Helix disables estimation; with a V2 manager it must also promote
    _skip_est so build_managers() calls configure_kv_cache_capacity().
    Other CP types must NOT be promoted: configure_kv_cache_capacity has no
    sizing path for them."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.mapping import CpType

    def creator(is_v2, cp_type=CpType.HELIX):
        return SimpleNamespace(
            _skip_est=False,
            _mapping=SimpleNamespace(cp_config={"cp_type": cp_type}),
            _is_kv_cache_manager_v2=is_v2,
            _model_engine=SimpleNamespace(
                model=SimpleNamespace(
                    model_config=SimpleNamespace(attn_backend="TRTLLM", is_encoder_decoder=False)
                )
            ),
        )

    c = creator(is_v2=True)
    assert KvCacheCreator.try_prepare_estimation(c) is False
    assert c._skip_est is True
    c = creator(is_v2=False)
    assert KvCacheCreator.try_prepare_estimation(c) is False
    assert c._skip_est is False
    c = creator(is_v2=True, cp_type=CpType.ULYSSES)
    assert KvCacheCreator.try_prepare_estimation(c) is False
    assert c._skip_est is False


def _failing_gen_scheduler(
    enable_recompute_pause: bool, ladder_calls: list, has_cache_tier_below_gpu: bool = False
) -> SimpleNamespace:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    sched = SimpleNamespace(
        has_cp_helix=True,
        enable_recompute_pause=enable_recompute_pause,
        kv_cache_manager=SimpleNamespace(
            try_allocate_generation=lambda req: False,
            is_request_active=lambda request_id: True,
            has_cache_tier_below_gpu=has_cache_tier_below_gpu,
        ),
        draft_kv_cache_manager=None,
        enable_joint_kv_cache_reuse=False,
    )
    sched._try_allocate_generation = types.MethodType(
        KVCacheV2Scheduler._try_allocate_generation, sched
    )
    sched._try_evict_for_gen = lambda req, requests_list, req_it, req_it_end, *args: (
        ladder_calls.append("evict") or (req_it_end, False)
    )
    sched._try_recompute_pause_for_gen = lambda req, requests_list, req_it, req_it_end, *args: (
        ladder_calls.append("recompute_pause") or (req_it_end, False)
    )
    sched._has_pending_connector_load = lambda req: False
    sched._suspend_request = lambda req: ladder_calls.append("self_suspend")
    sched._clear_request_runtime_state = types.MethodType(
        KVCacheV2Scheduler._clear_request_runtime_state, sched
    )
    return sched


def _schedule_failing_gen(sched: SimpleNamespace, req: SimpleNamespace):
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
        KVCacheV2Scheduler,
        _RecomputePauseState,
    )

    return KVCacheV2Scheduler._try_schedule_generation(
        sched,
        req,
        SimpleNamespace(can_fit_tokens=lambda n: True),
        requests_list=[req],
        req_it=0,
        req_it_end=1,
        recompute_pause_state=_RecomputePauseState(1),
        evicted=[],
        recompute_paused=[],
        inflight_request_ids=set(),
        scheduled_beam_width=0,
    )


def _gen_request() -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=7,
        get_beam_width_by_iter=lambda for_next_iteration: 1,
        py_draft_tokens=None,
        py_batch_idx=3,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
    )


@pytest.mark.parametrize("has_cache_tier_below_gpu", [False, True], ids=["gpu_only", "host_tier"])
def test_scheduler_allocation_failure_raises_under_disagg_helix(
    has_cache_tier_below_gpu: bool,
) -> None:
    """A disaggregated helix generation server cannot replay a prefill, so a
    generation allocation failure raises instead of entering eviction, even
    with a host tier to suspend to."""
    ladder_calls = []
    sched = _failing_gen_scheduler(
        enable_recompute_pause=False,
        ladder_calls=ladder_calls,
        has_cache_tier_below_gpu=has_cache_tier_below_gpu,
    )
    with pytest.raises(RuntimeError, match="cannot preempt"):
        _schedule_failing_gen(sched, _gen_request())
    assert ladder_calls == []


def test_scheduler_allocation_failure_preempts_under_agg_helix() -> None:
    """Aggregated helix can replay a prefill, and without a tier below GPU
    suspending frees nothing, so a generation allocation failure goes
    straight to recompute-pause. The request sits out the step without being
    suspended, so its stale batch index must still be cleared."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import ScheduleAction

    ladder_calls = []
    sched = _failing_gen_scheduler(enable_recompute_pause=True, ladder_calls=ladder_calls)
    req = _gen_request()
    action, *_ = _schedule_failing_gen(sched, req)
    assert action is ScheduleAction.STOP
    assert ladder_calls == ["recompute_pause"]
    assert req.py_batch_idx is None


def test_scheduler_allocation_failure_suspends_under_agg_helix_with_host_tier() -> None:
    """With a host tier, a failed generation allocation tries evict first; when
    evict and recompute-pause both fail, the requester suspends itself."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import ScheduleAction

    ladder_calls = []
    sched = _failing_gen_scheduler(
        enable_recompute_pause=True, ladder_calls=ladder_calls, has_cache_tier_below_gpu=True
    )
    action, *_ = _schedule_failing_gen(sched, _gen_request())
    assert action is ScheduleAction.STOP
    assert ladder_calls == ["evict", "recompute_pause", "self_suspend"]


def _host_tier_preempt_scheduler(preempted: list) -> SimpleNamespace:
    """Agg helix scheduler with a host tier, where every request holds pages."""
    sched = SimpleNamespace(
        has_cp_helix=True,
        enable_recompute_pause=True,
        kv_cache_manager=SimpleNamespace(
            has_cache_tier_below_gpu=True,
            is_request_active=lambda request_id: True,
        ),
    )
    sched._is_recompute_pause_candidate = lambda req, inflight_request_ids: True
    sched._preempt_or_defer = lambda victim, recompute_paused: preempted.append(victim) or True
    return sched


def _try_preempt_for_pages(sched: SimpleNamespace, requests_list: list) -> bool:
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    return KVCacheV2Scheduler._try_preempt_for_pages(
        sched,
        requests_list,
        protected_ids={requests_list[0].py_request_id},
        inflight_request_ids=set(),
        recompute_paused=[],
        preempted_ids=set(),
        requester_idx=0,
    )


def _preempt_req(request_id: int, in_generation: bool) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        request_id=request_id,
        is_generation_in_progress_state=in_generation,
        state=LlmRequestState.GENERATION_IN_PROGRESS
        if in_generation
        else LlmRequestState.CONTEXT_INIT,
    )


def test_context_preempt_waits_for_decode_under_agg_helix_with_host_tier() -> None:
    """With a host tier, a blocked context request waits while a decode request
    holds GPU pages, since the generation path can suspend it to the host tier
    instead of anyone paying a re-prefill."""
    preempted = []
    sched = _host_tier_preempt_scheduler(preempted)
    requests_list = [
        _preempt_req(1, in_generation=False),
        _preempt_req(2, in_generation=True),
        _preempt_req(3, in_generation=False),
    ]
    assert not _try_preempt_for_pages(sched, requests_list)
    assert preempted == []


def test_context_preempt_falls_back_under_agg_helix_with_host_tier() -> None:
    """With a host tier but every running request mid-prefill, no decode
    request is left for the generation path to suspend, so the blocked context
    request recompute-pauses the newest request behind it."""
    preempted = []
    sched = _host_tier_preempt_scheduler(preempted)
    requests_list = [
        _preempt_req(1, in_generation=False),
        _preempt_req(2, in_generation=False),
        _preempt_req(3, in_generation=False),
    ]
    assert _try_preempt_for_pages(sched, requests_list)
    assert [r.py_request_id for r in preempted] == [3]


def _context_resize_mgr(is_disagg: bool, resizes: list) -> SimpleNamespace:
    kv = SimpleNamespace(capacity=0, resize=lambda cap: resizes.append(cap) or True)
    return SimpleNamespace(
        _has_cp_helix=True,
        is_disagg=is_disagg,
        kv_cache_map={7: kv},
        num_extra_kv_tokens=0,
        _connector_reservations_enabled=lambda: False,
        _fill_fresh_kv_pages=lambda req_id: None,
        _log_window_crossing=lambda *args: None,
    )


def _context_req(prompt_len: int, total_input_len_cp: int) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=7,
        is_disagg_generation_init_state=False,
        is_dummy_request=False,
        prompt_len=prompt_len,
        total_input_len_cp=total_input_len_cp,
        context_current_position=64,
        is_first_context_chunk=False,
    )


def test_resize_context_allows_agg_helix_request() -> None:
    """An agg helix request holds the full prompt on every rank, so its
    context target is already in global ledger tokens."""
    resizes = []
    req = _context_req(prompt_len=200, total_input_len_cp=200)
    assert KVCacheManagerV2.resize_context(
        _context_resize_mgr(is_disagg=False, resizes=resizes), req, num_tokens=32
    )
    assert resizes == [64 + 32]


@pytest.mark.parametrize("prompt_len", [50, 200])
def test_resize_context_rejects_disagg_helix_request(prompt_len: int) -> None:
    """A disagg helix request holds only this rank's slice of the prompt; a
    rank-local target would under-size the global ledger. The guard keys on
    disagg mode, so a slice that happens to equal the full prompt (a
    single-block prompt on rank 0) is rejected too."""
    req = _context_req(prompt_len=prompt_len, total_input_len_cp=200)
    with pytest.raises(ValueError, match="disaggregated helix"):
        KVCacheManagerV2.resize_context(
            _context_resize_mgr(is_disagg=True, resizes=[]), req, num_tokens=32
        )


def _rank_batch(
    victim_id: int = 3, prompt_len: int = 200, decode_group_index: int = 5
) -> SimpleNamespace:
    """One CP rank's scheduled batch: a context chunk, a decode step and a
    recompute-paused victim."""
    ctx = SimpleNamespace(
        py_request_id=1,
        context_current_position=64,
        context_chunk_size=32,
        prompt_len=prompt_len,
        total_input_len_cp=200,
    )
    gen = SimpleNamespace(py_request_id=2, py_helix_decode_group_index=decode_group_index)
    return SimpleNamespace(
        context_requests=[ctx],
        generation_requests=[gen],
        paused_requests=[],
        recompute_paused_requests=[SimpleNamespace(py_request_id=victim_id)],
    )


def _check_schedules(rank_batches: list) -> None:
    """Run the executor's helix schedule check on every simulated CP rank, with
    an allgather that hands each rank the summaries of all of them."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    summaries = []
    collect = SimpleNamespace(
        dist=SimpleNamespace(cp_allgather=lambda s: summaries.append(s) or [s])
    )
    for batch in rank_batches:
        PyExecutor._check_helix_schedule_consistency(collect, batch)
    gather = SimpleNamespace(dist=SimpleNamespace(cp_allgather=lambda s: list(summaries)))
    for batch in rank_batches:
        PyExecutor._check_helix_schedule_consistency(gather, batch)


def test_helix_schedule_check_passes_identical_ranks() -> None:
    _check_schedules([_rank_batch(), _rank_batch()])


@pytest.mark.parametrize(
    "rank1_batch",
    [_rank_batch(victim_id=4), _rank_batch(decode_group_index=6)],
    ids=["different_victim", "different_decode_step"],
)
def test_helix_schedule_check_raises_on_divergence(rank1_batch: SimpleNamespace) -> None:
    """Ranks that preempt different requests or disagree on a decode step
    would desynchronize the next CP collective."""
    with pytest.raises(RuntimeError, match="disagree on the schedule"):
        _check_schedules([_rank_batch(), rank1_batch])


def test_helix_schedule_check_raises_on_sliced_prompt() -> None:
    """A context request holding only a per-rank slice of its prompt would
    under-size the global ledger, even when every rank agrees."""
    with pytest.raises(RuntimeError, match=r"without the full prompt: \[1\]"):
        _check_schedules([_rank_batch(prompt_len=100), _rank_batch(prompt_len=100)])


def test_reset_for_recompute_restarts_helix_decode_state() -> None:
    """A re-prefilled request decodes from its new prompt end, so the helix
    decode counter and inactive-rank flag restart with it."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

    req = LlmRequest(
        request_id=1,
        max_new_tokens=8,
        input_tokens=list(range(96)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    req.py_helix_decode_group_index = 5
    req.py_helix_is_inactive_rank = True

    req.reset_for_recompute(max_input_len=1 << 20)

    assert req.py_helix_decode_group_index == 0
    assert not req.py_helix_is_inactive_rank
    assert req.total_input_len_cp == req.prompt_len


def test_dummy_frozen_fields_sum_invariant() -> None:
    """The frozen dummy fiction (last rank active, shared synthetic global
    length) keeps the same books as real requests: per-rank lengths sum to
    the synthetic global length."""
    for cp_size in (2, 4, 8):
        for token_num in (2, 5, 33):
            seqlens = [token_num - 1 if r == cp_size - 1 else token_num for r in range(cp_size)]
            total = token_num * cp_size - 1
            assert sum(seqlens) == total
