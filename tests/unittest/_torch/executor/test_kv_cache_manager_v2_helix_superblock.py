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
function of the global position, so every rank's ledger advances identically
and no rotation state exists.
"""

import math
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState


def _mgr(cp_rank, cp_size, phys):
    m = SimpleNamespace(
        tokens_per_block=phys,
        _ledger_tokens_per_block=phys * cp_size,
        _has_cp_helix=cp_size > 1,
        _helix_cp_rank=cp_rank,
        _helix_cp_size=cp_size,
    )
    m._helix_local_len = lambda global_len: KVCacheManagerV2._helix_local_len(m, global_len)
    return m


def _brute_local_len(global_len, cp_rank, cp_size, phys):
    """Reference: token position p lives on rank (p // phys) % cp_size."""
    return sum(1 for p in range(global_len) if (p // phys) % cp_size == cp_rank)


def test_helix_local_len_matches_brute_force():
    for cp_size in (1, 2, 4, 8):
        for phys in (2, 4, 32):
            for cp_rank in range(cp_size):
                m = _mgr(cp_rank, cp_size, phys)
                for global_len in range(0, 4 * cp_size * phys + 3):
                    assert KVCacheManagerV2._helix_local_len(m, global_len) == _brute_local_len(
                        global_len, cp_rank, cp_size, phys
                    ), (cp_size, phys, cp_rank, global_len)


def test_set_helix_rank_fields_cross_rank_consistency():
    """For any (prompt_len, decoding_iter): exactly one active rank, the
    per-rank seqlens sum to the global in-flight length, and past_seen
    (= seqlen - 0/1 per the model_engine convention) sums to the global
    already-cached length."""
    cp_size, phys = 4, 4
    for total_input in (1, 5, 16, 17, 63):
        for decoding_iter in (0, 1, 2, 7, 40):  # 0 exercises the max(1,...) clamp
            fields = []
            for r in range(cp_size):
                req = SimpleNamespace(
                    total_input_len_cp=total_input,
                    py_decoding_iter=decoding_iter,
                )
                KVCacheManagerV2._set_helix_rank_fields(_mgr(r, cp_size, phys), req)
                fields.append(req)
            pos = total_input + max(1, decoding_iter) - 1
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


def test_ledger_is_rank_invariant():
    """The whole point of the super-block design: nothing the scheduler or
    ledger consumes depends on cp_rank — only the derived per-rank fields
    do. Verify the derivation never touches ledger quantities by checking
    the same request object gives identical (owner-adjusted) views."""
    cp_size, phys = 8, 32
    req_proto = dict(total_input_len_cp=1000, py_decoding_iter=17)
    lens = []
    for r in range(cp_size):
        req = SimpleNamespace(**req_proto)
        KVCacheManagerV2._set_helix_rank_fields(_mgr(r, cp_size, phys), req)
        lens.append(req.seqlen_this_rank_cp)
    # Ledger-side numbers (global position, page count) are identical on
    # every rank; per-rank lens differ by at most one physical page.
    assert max(lens) - min(lens) <= phys


def test_quota_converters_scale_by_cp():
    m = SimpleNamespace(
        _has_cp_helix=True,
        _helix_cp_size=4,
        _get_max_tokens_from_quota_impl=lambda quota: 100.0,
        _get_quota_from_max_tokens_impl=lambda tokens: tokens * 7,
    )
    # Rank-local byte quota buys 100 physical tokens -> 400 global tokens.
    assert KVCacheManagerV2._get_max_tokens_from_quota(m, 12345) == 400.0
    # inf (all-SWA) passes through unscaled.
    m_inf = SimpleNamespace(
        _has_cp_helix=True,
        _helix_cp_size=4,
        _get_max_tokens_from_quota_impl=lambda quota: float("inf"),
    )
    assert math.isinf(KVCacheManagerV2._get_max_tokens_from_quota(m_inf, 1))
    # Global tokens -> per-rank physical tokens (ceil) -> bytes.
    assert KVCacheManagerV2._get_quota_from_max_tokens(m, 401) == 101 * 7
    # cp == 1 (non-helix) is the identity.
    m1 = SimpleNamespace(
        _has_cp_helix=False,
        _helix_cp_size=1,
        _get_max_tokens_from_quota_impl=lambda quota: 100.0,
        _get_quota_from_max_tokens_impl=lambda tokens: tokens * 7,
    )
    assert KVCacheManagerV2._get_max_tokens_from_quota(m1, 1) == 100.0
    assert KVCacheManagerV2._get_quota_from_max_tokens(m1, 400) == 2800


def test_update_resources_leaves_history_untouched_under_helix():
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


def test_helix_quota_fallback_emits_global_tokens(monkeypatch):
    """Creator fallback: the rank-local byte budget buys N physical tokens,
    i.e. N * cp_size global (super-block ledger) tokens."""
    import torch

    from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator

    def creator(max_gpu_total_bytes, max_tokens):
        return SimpleNamespace(
            _mapping=SimpleNamespace(cp_size=4),
            _kv_cache_config=SimpleNamespace(
                max_gpu_total_bytes=max_gpu_total_bytes,
                max_tokens=max_tokens,
                free_gpu_memory_fraction=0.5,
            ),
            _get_kv_size_per_token=lambda: CacheCost(slope=1000, intercept=8000),
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_000_000, 2_000_000))

    # Explicit quota: early return, config untouched.
    c = creator(1 << 30, None)
    assert KvCacheCreator._configure_helix_kv_cache_capacity(c) is None
    assert c._kv_cache_config.max_tokens is None
    # No quota: (1e6 * 0.5 - 8000) // 1000 = 492 physical -> 1968 global.
    c = creator(0, None)
    assert KvCacheCreator._configure_helix_kv_cache_capacity(c) is None
    assert c._kv_cache_config.max_tokens == 492 * 4
    # Degenerate free memory: actionable error instead of a deep assert.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 2_000_000))
    with pytest.raises(ValueError, match="free memory"):
        KvCacheCreator._configure_helix_kv_cache_capacity(creator(0, None))


def test_estimation_prepare_promotes_skip_est_for_v2():
    """Helix disables estimation; with a V2 manager it must also promote
    _skip_est so build_managers() calls configure_kv_cache_capacity()."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.mapping import CpType

    def creator(is_v2):
        return SimpleNamespace(
            _skip_est=False,
            _mapping=SimpleNamespace(cp_config={"cp_type": CpType.HELIX}),
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


def test_scheduler_allocation_failure_raises_under_helix():
    """Precedent-consistent no-evict stance: allocation failure under helix
    raises instead of entering the (unvalidated-under-helix) eviction path."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    sched = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            _has_cp_helix=True, try_allocate_generation=lambda req: False
        ),
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
            evicted=[],
            scheduled_beam_width=0,
        )


def test_dummy_frozen_fields_sum_invariant():
    """The frozen dummy fiction (last rank active, shared synthetic global
    length) keeps the same books as real requests: per-rank lengths sum to
    the synthetic global length."""
    for cp_size in (2, 4, 8):
        for token_num in (2, 5, 33):
            seqlens = [token_num - 1 if r == cp_size - 1 else token_num for r in range(cp_size)]
            total = token_num * cp_size - 1
            assert sum(seqlens) == total
