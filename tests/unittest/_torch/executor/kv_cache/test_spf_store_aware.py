# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store-aware shortest-prefill-first ranking and the per-request context chunk cap (CPU only)."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
    KVCacheV2Scheduler,
    ShortestPrefillFirstOrder,
)

CTX = 7  # context-init state value used by these fakes


def _req(rid, prompt_len, started=False):
    return SimpleNamespace(
        py_request_id=rid,
        prompt_len=prompt_len,
        context_current_position=0,
        context_remaining_length=prompt_len,
        is_first_context_chunk=not started,
        state_value=CTX,
        py_connector_allocation_reported=False,
    )


class _Connector:
    def __init__(self, offers):
        self.offers = offers
        self.calls = []

    def probe_num_matched_tokens(self, req, num_computed_tokens):
        self.calls.append((req.py_request_id, num_computed_tokens))
        return self.offers.get(req.py_request_id)


def _manager(local, offers, may_serve=True):
    return SimpleNamespace(
        probe_context_reuse=lambda req: local.get(req.py_request_id),
        kv_connector_manager=_Connector(offers),
        _connector_may_serve=lambda req: may_serve,
    )


def _order(spf, reqs, manager):
    return [r.py_request_id for r in spf.order(reqs, reqs, CTX, manager, reorder=True)]


def test_store_hit_ranks_ahead_of_cold_prompt():
    # Request 1 is long but its prefix sits in the store; request 2 is short and cold.
    reqs = [_req(1, 10_000), _req(2, 3_000)]
    manager = _manager(local={1: 0}, offers={1: 9_000})
    assert _order(ShortestPrefillFirstOrder(128, store_aware=True), reqs, manager) == [1, 2]
    manager = _manager(local={1: 0}, offers={1: 9_000})
    assert _order(ShortestPrefillFirstOrder(128, store_aware=False), reqs, manager) == [2, 1]
    assert manager.kv_connector_manager.calls == []


def test_store_probe_starts_from_local_match_and_runs_once():
    reqs = [_req(1, 10_000), _req(2, 3_000)]
    manager = _manager(local={1: 2_048}, offers={1: 4_096})
    spf = ShortestPrefillFirstOrder(128)
    _order(spf, reqs, manager)
    _order(spf, reqs, manager)
    assert manager.kv_connector_manager.calls.count((1, 2_048)) == 1
    # 10_000 - (2_048 + 4_096)
    assert spf._uncached_estimate[1] == 3_856


def test_no_probe_when_connector_cannot_serve():
    reqs = [_req(1, 10_000), _req(2, 3_000)]
    manager = _manager(local={}, offers={1: 9_000}, may_serve=False)
    assert _order(ShortestPrefillFirstOrder(128), reqs, manager) == [2, 1]
    assert manager.kv_connector_manager.calls == []


def test_store_end_leaves_the_last_prompt_token():
    reqs = [_req(1, 4_000), _req(2, 3_000)]
    manager = _manager(local={}, offers={1: 10_000})
    spf = ShortestPrefillFirstOrder(128)
    _order(spf, reqs, manager)
    assert spf._uncached_estimate[1] == 1


@pytest.mark.parametrize(
    "remaining, eligible, expected",
    [(8192, 1, None), (8192, 2, 4096), (8192, 4, 2048), (8192, 16, 1024), (None, 3, 1024)],
)
def test_chunk_cap_share(remaining, eligible, expected):
    sched = object.__new__(KVCacheV2Scheduler)
    sched._ctx_request_chunk_cap = 1024
    sched.chunk_unit_size = 256
    assert sched._ctx_chunk_cap_for_share(remaining, eligible) == expected


def test_chunk_cap_unset_is_uncapped():
    sched = object.__new__(KVCacheV2Scheduler)
    sched.chunk_unit_size = 256
    assert sched._ctx_chunk_cap_this_pass is None
    assert sched._ctx_chunk_cap_for_share(8192, 4) is None
