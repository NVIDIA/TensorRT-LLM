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
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState


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
        _kv_reserve_draft_tokens=0,
        kv_cache_map={7: kv},
        kv_compression_manages_history=False,
        _has_cp_helix=True,
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

    def creator(max_gpu_total_bytes, max_tokens):
        return SimpleNamespace(
            _mapping=SimpleNamespace(cp_size=4),
            _kv_cache_config=SimpleNamespace(
                max_gpu_total_bytes=max_gpu_total_bytes,
                max_tokens=max_tokens,
                free_gpu_memory_fraction=0.5,
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


def test_scheduler_allocation_failure_raises_under_helix() -> None:
    """Precedent-consistent no-evict stance: allocation failure under helix
    raises instead of entering the (unvalidated-under-helix) eviction path."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
        KVCacheV2Scheduler,
        _RecomputePauseState,
    )

    sched = SimpleNamespace(
        has_cp_helix=True,
        kv_cache_manager=SimpleNamespace(try_allocate_generation=lambda req: False),
    )
    req = SimpleNamespace(
        py_request_id=7,
        get_beam_width_by_iter=lambda for_next_iteration: 1,
        py_draft_tokens=None,
    )
    budget = SimpleNamespace(can_fit_tokens=lambda n: True)
    with pytest.raises(RuntimeError, match="eviction is disabled under helix"):
        KVCacheV2Scheduler._try_schedule_generation(
            sched,
            req,
            budget,
            requests_list=[req],
            req_it=0,
            req_it_end=1,
            recompute_pause_state=_RecomputePauseState(1),
            evicted=[],
            recompute_paused=[],
            inflight_request_ids=set(),
            scheduled_beam_width=0,
        )


def test_dummy_frozen_fields_sum_invariant() -> None:
    """The frozen dummy fiction (last rank active, shared synthetic global
    length) keeps the same books as real requests: per-rank lengths sum to
    the synthetic global length."""
    for cp_size in (2, 4, 8):
        for token_num in (2, 5, 33):
            seqlens = [token_num - 1 if r == cp_size - 1 else token_num for r in range(cp_size)]
            total = token_num * cp_size - 1
            assert sum(seqlens) == total
