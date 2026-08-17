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
"""Bookkeeping-only tests of the KVCacheManagerV2 helix round-robin gate.

Binds the real KVCacheManagerV2 methods onto a stub object so the gate
arithmetic (try_allocate_generation / revert_allocate_generation) is
exercised without pools, GPUs or a model.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only

TPB = 4  # tokens_per_block: small so ownership rotates quickly
CP = 2
INITIAL_CAPACITY = 10


class _FakeKvCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.is_active = True
        self.history_length = 0
        self.resize_calls = []

    def resume(self, _stream):
        self.is_active = True
        return True

    def resize(self, capacity, history_length=None):
        self.resize_calls.append((capacity, history_length))
        if capacity is not None:
            self.capacity = capacity
        return True


def _make_mgr(cp_rank):
    mgr = SimpleNamespace()
    mgr._has_cp_helix = True
    mgr._helix_cp_rank = cp_rank
    mgr._helix_cp_size = CP
    mgr.tokens_per_block = TPB
    mgr.kv_cache_map = {}
    mgr._allocated_draft_lens = {}
    mgr._stream = SimpleNamespace(cuda_stream=None)
    mgr._effective_draft_len = lambda req: 0
    mgr._required_gen_capacity = lambda req, cap: cap + 1
    mgr._restore_page_index_bufs = lambda rid, kv: None
    return mgr


def _make_req(rid, iter0, seqlen):
    return SimpleNamespace(
        py_request_id=rid,
        py_decoding_iter=iter0,
        seqlen_this_rank_cp=seqlen,
        py_helix_is_inactive_rank=False,
        is_dummy_request=False,
    )


def _make_rank(cp_rank):
    mgr = _make_mgr(cp_rank)
    kv = _FakeKvCache(capacity=INITIAL_CAPACITY)
    mgr.kv_cache_map[1] = kv
    req = _make_req(1, 1, INITIAL_CAPACITY)
    return mgr, kv, req


def _run_rank(cp_rank, steps):
    """Simulate `steps` decode iterations on one rank; return per-step record."""
    mgr, kv, req = _make_rank(cp_rank)
    trace = []
    for it in range(1, steps + 1):
        req.py_decoding_iter = it
        ok = KVCacheManagerV2.try_allocate_generation(mgr, req)
        trace.append((it, ok, req.py_helix_is_inactive_rank, req.seqlen_this_rank_cp, kv.capacity))
    return trace, mgr, kv, req


@pytest.mark.parametrize("rank", range(CP))
def test_ownership_rotation(rank):
    """Decode block b is owned by rank b % CP; other ranks stay schedulable."""
    trace, _, kv, _ = _run_rank(rank, steps=TPB * 2)
    for it, ok, inactive, _seq, _cap in trace:
        owner = ((it - 1) // TPB) % CP
        assert ok, f"iter {it} must stay schedulable on every rank"
        assert inactive == (owner != rank), f"iter {it}: owner={owner} inactive={inactive}"
    active_steps = sum(1 for _, _, inactive, _, _ in trace if not inactive)
    assert kv.capacity == INITIAL_CAPACITY + active_steps


def test_seqlen_advances_only_on_active_steps():
    _, _, _, req = _run_rank(0, steps=TPB)
    assert req.seqlen_this_rank_cp == INITIAL_CAPACITY + TPB
    _, _, _, req = _run_rank(1, steps=TPB)
    assert req.seqlen_this_rank_cp == INITIAL_CAPACITY


def test_first_schedule_before_decoding_iter_seeded():
    """First schedule reads py_decoding_iter == 0 (before the disagg
    transmission-complete handler seeds it to 1) and must be treated as
    decode step 1: owner = block 0 = rank 0, never rank cp_size - 1 via
    Python's negative modulo."""
    mgr, kv, req = _make_rank(0)
    req.py_decoding_iter = 0
    ok = KVCacheManagerV2.try_allocate_generation(mgr, req)
    assert ok and not req.py_helix_is_inactive_rank
    assert kv.capacity == INITIAL_CAPACITY + 1

    mgr, kv, req = _make_rank(1)
    req.py_decoding_iter = 0
    ok = KVCacheManagerV2.try_allocate_generation(mgr, req)
    assert ok and req.py_helix_is_inactive_rank
    assert kv.capacity == INITIAL_CAPACITY


def test_revert_symmetry_active_rank():
    mgr, kv, req = _make_rank(0)  # rank 0 owns block 0 -> active
    KVCacheManagerV2.try_allocate_generation(mgr, req)
    KVCacheManagerV2.revert_allocate_generation(mgr, req)
    assert kv.capacity == INITIAL_CAPACITY
    assert req.seqlen_this_rank_cp == INITIAL_CAPACITY


def test_revert_is_noop_on_inactive_rank():
    mgr, kv, req = _make_rank(1)  # rank 1 inactive during block 0
    KVCacheManagerV2.try_allocate_generation(mgr, req)
    n_resizes = len(kv.resize_calls)
    KVCacheManagerV2.revert_allocate_generation(mgr, req)
    assert len(kv.resize_calls) == n_resizes
    assert kv.capacity == INITIAL_CAPACITY
    assert req.seqlen_this_rank_cp == INITIAL_CAPACITY


def test_helix_quota_fallback_uses_fraction_sizing(monkeypatch):
    """Helix skips profiling (_configure_helix_kv_cache_capacity): an
    explicit quota returns early untouched; without one it falls back to
    V1-style fraction sizing of free memory (min-synced later by
    KVCacheManagerV2)."""
    import torch

    from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
    from tensorrt_llm.mapping import CpType

    def creator(max_gpu_total_bytes, max_tokens):
        return SimpleNamespace(
            _mapping=SimpleNamespace(cp_config={"cp_type": CpType.HELIX}),
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
    # No quota: fraction fallback -> (1e6 * 0.5 - 8000) // 1000 = 492.
    c = creator(0, None)
    assert KvCacheCreator._configure_helix_kv_cache_capacity(c) is None
    assert c._kv_cache_config.max_tokens == 492
    # Degenerate free memory: actionable error instead of a deep assert.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 2_000_000))
    with pytest.raises(ValueError, match="free memory"):
        KvCacheCreator._configure_helix_kv_cache_capacity(creator(0, None))


def test_helix_estimation_prepare_promotes_skip_est_for_v2():
    """Helix disables estimation; with a V2 manager it must also promote
    _skip_est so build_managers() calls configure_kv_cache_capacity() —
    otherwise KVCacheManagerV2 asserts "Quota not set" at construction.
    V1 keeps sizing itself from the fraction, so it stays unpromoted."""
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
    """V2 scheduler: a failed generation allocation must raise under helix
    instead of falling into rank-local eviction (which would desynchronize
    the CP group)."""
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
