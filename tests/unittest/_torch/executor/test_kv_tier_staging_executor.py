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
"""Executor and manager glue of the step-ahead KV tier staging (TRTLLM_M3_ASYNC_OFFLOAD,
TRTLLM_KV_ONBOARD_PREFETCH). The copies themselves are covered against the real core in
tests/unittest/kv_cache_manager_v2_tests/test_kv_tier_staging.py."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only

TPB = 4


class _FakeCache:
    def __init__(self, log, tokens, levels):
        self._log, self._tokens = log, tokens
        self.cached_tokens_by_level = levels

    def close(self):
        self._log.append(("close", tuple(self._tokens)))


class _FakeImpl:
    """Reused tokens all sit on the host until a prefetch of the same tokens moves them."""

    def __init__(self, tiers=2):
        self.log = []
        self.cache_tier_list = list(range(tiers))
        self.on_gpu = set()

    def create_kv_cache(self, scope, tokens, **kwargs):
        self.log.append(("create", scope, tuple(tokens)))
        key = tuple(tokens)
        levels = [len(tokens), 0] if key in self.on_gpu else [0, len(tokens)]
        cache = _FakeCache(self.log, tokens, levels)
        cache.prefetch = lambda level: (
            self.log.append(("prefetch", key, level)) or (self.on_gpu.add(key) or True)
        )
        return cache


def _manager(tiers=2, reuse=True, estimating=False) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = _FakeImpl(tiers)
    manager.enable_block_reuse = reuse
    manager.is_estimating_kv_cache = estimating
    manager.tokens_per_block = TPB
    manager.kv_cache_map = {}
    manager._context_reuse_tokens = lambda req, limit=None: req.tokens[:limit]
    return manager


def _req(rid, tokens=8, **kwargs):
    fields = dict(
        py_request_id=rid,
        tokens=list(range(rid * 100, rid * 100 + tokens)),
        is_dummy=False,
        lora_task_id=None,
        cache_salt=None,
        is_context_init_state=True,
        is_first_context_chunk=True,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_prefetch_stages_unadmitted_requests_with_a_reusable_block():
    manager = _manager()
    manager.kv_cache_map = {2: object()}
    reqs = [_req(1), _req(2), _req(3, tokens=TPB - 1), _req(4, is_dummy=True), _req(5)]
    assert manager.prefetch_context_to_gpu(reqs) == {1: 0, 5: 0}
    created = [entry[2] for entry in manager.impl.log if entry[0] == "create"]
    assert created == [tuple(reqs[0].tokens), tuple(reqs[4].tokens)]
    prefetched = [entry for entry in manager.impl.log if entry[0] == "prefetch"]
    assert [level for *_, level in prefetched] == [kvm.GPU_LEVEL] * 2
    assert sum(entry[0] == "close" for entry in manager.impl.log) == 2


def test_prefetch_measures_what_it_brought_to_the_gpu_and_skips_resident_prefixes():
    manager = _manager()
    req = _req(1, tokens=12)
    assert manager.prefetch_context_to_gpu([req], measure=True) == {1: 12}
    manager.impl.log.clear()
    assert manager.prefetch_context_to_gpu([req], measure=True) == {1: 0}
    assert not any(entry[0] == "prefetch" for entry in manager.impl.log)


def test_prefetch_matches_with_the_admission_claim_cap():
    manager = _manager()
    manager._spec_recompute_tail = 3
    manager.is_draft = False
    req = _req(1, tokens=13, prompt_len=13)
    assert manager.prefetch_context_to_gpu([req]) == {1: 0}
    # (13 - 3) // 4 * 4 = 8 tokens claimed, as prepare_context would.
    assert manager.impl.log[0][2] == tuple(req.tokens[:8])


@pytest.mark.parametrize(
    "kwargs", [dict(tiers=1), dict(reuse=False), dict(estimating=True)], ids=str
)
def test_prefetch_is_off_without_a_lower_tier_reuse_or_outside_serving(kwargs):
    manager = _manager(**kwargs)
    assert manager.prefetch_context_to_gpu([_req(1)]) == {}
    assert manager.impl.log == []


@pytest.mark.parametrize("kwargs", [dict(tiers=1), dict(estimating=True)], ids=str)
def test_headroom_arms_only_with_a_lower_tier_outside_estimation(kwargs):
    manager = _manager(**kwargs)
    assert not manager.enable_offload_headroom(64)
    assert not manager.reserve_offload_headroom()
    manager.release_offload_headroom()


def test_headroom_disarmed_by_default():
    manager = _manager()
    assert manager._offload_headroom is None
    assert not manager.reserve_offload_headroom()
    assert not manager.enable_offload_headroom(0)


class _Manager:
    def __init__(self, log, name, onboarded=None):
        self.log, self.name = log, name
        self.kv_cache_map = {}
        self.onboarded = onboarded or {}

    def prefetch_context_to_gpu(self, reqs, measure=False):
        ids = [r.py_request_id for r in reqs]
        self.log.append((self.name, "prefetch", ids) + (("measure",) if measure else ()))
        return {rid: self.onboarded.get(rid, 0) if measure else 0 for rid in ids}

    def reserve_offload_headroom(self):
        self.log.append((self.name, "reserve"))

    def release_offload_headroom(self):
        self.log.append((self.name, "release"))


def _executor(active, num_reqs=2, early_fetch=False, fetched=()):
    executor = object.__new__(PyExecutor)
    log = []
    target, draft = _Manager(log, "target"), _Manager(log, "draft")
    executor.kv_cache_manager = target
    executor._kv_headroom_managers = [target, draft]
    executor._kv_prefetch_managers = [target, draft]
    executor._kv_onboard_prefetch_reqs = num_reqs
    executor._kv_onboard_early_fetch = early_fetch
    executor._kv_onboard_prefetched_ids = set()
    executor._kv_onboard_defer_tokens = 0
    executor._kv_onboard_deferred_ids = set()
    executor._early_activated_requests = []
    executor.active_requests = list(active)
    executor.is_shutdown = False
    executor.control_requests = []

    def fetch():
        log.append(("fetch",))
        executor.active_requests.extend(fetched)
        return list(fetched)

    executor._fetch_and_activate_new_requests = fetch
    return executor, log


def test_staging_prefetches_then_reserves_and_stages_each_request_once():
    reqs = [_req(1), _req(2), _req(3)]
    executor, log = _executor(reqs)
    executor._stage_kv_tiers_after_launch()
    assert log == [
        ("target", "prefetch", [1, 2]),
        ("draft", "prefetch", [1, 2]),
        ("target", "reserve"),
        ("draft", "reserve"),
    ]
    log.clear()
    executor._stage_kv_tiers_after_launch()
    assert log[:2] == [("target", "prefetch", [3]), ("draft", "prefetch", [3])]
    log.clear()
    executor._stage_kv_tiers_after_launch()
    assert log == [("target", "reserve"), ("draft", "reserve")]

    executor._release_kv_offload_headroom()
    assert log[-2:] == [("target", "release"), ("draft", "release")]


def test_candidates_skip_admitted_started_and_dummy_requests():
    reqs = [
        _req(1),
        _req(2, is_first_context_chunk=False),
        _req(3, is_context_init_state=False),
        _req(4, is_dummy=True),
        _req(5),
    ]
    executor, _ = _executor(reqs, num_reqs=8)
    executor.kv_cache_manager.kv_cache_map = {5: object()}
    assert [r.py_request_id for r in executor._onboard_prefetch_candidates()] == [1]


def test_staged_ids_are_forgotten_once_the_request_is_admitted():
    reqs = [_req(1)]
    executor, _ = _executor(reqs)
    executor._onboard_prefetch_candidates()
    assert executor._kv_onboard_prefetched_ids == {1}
    executor.kv_cache_manager.kv_cache_map = {1: object()}
    executor._onboard_prefetch_candidates()
    assert executor._kv_onboard_prefetched_ids == set()


def test_early_fetch_activates_arrivals_and_stages_them():
    arrival = _req(7)
    executor, log = _executor([_req(1)], num_reqs=4, early_fetch=True, fetched=[arrival])
    executor._stage_kv_tiers_after_launch()
    assert log[0] == ("fetch",)
    assert log[1] == ("target", "prefetch", [1, 7])
    assert executor._early_activated_requests == [arrival]


@pytest.mark.parametrize("state", ["idle", "shutdown", "control"])
def test_early_fetch_waits_for_the_regular_fetch_when_it_could_block_or_reorder(state):
    executor, log = _executor([_req(1)], early_fetch=True, fetched=[_req(7)])
    if state == "idle":
        executor.active_requests = []
    elif state == "shutdown":
        executor.is_shutdown = True
    else:
        executor.control_requests = [object()]
    executor._stage_kv_tiers_after_launch()
    assert ("fetch",) not in log
    assert executor._early_activated_requests == []


def test_env_arms_each_feature_on_v2_pools_with_a_host_tier(monkeypatch):
    executor = object.__new__(PyExecutor)
    target = _manager()
    draft = _manager(tiers=1)
    armed = []
    for manager in (target, draft):
        manager._stream = SimpleNamespace(device=None)
    monkeypatch.setattr(
        kvm.KVCacheManagerV2,
        "enable_offload_headroom",
        lambda self, tokens: armed.append((self, tokens)) or self.has_lower_cache_tier(),
    )
    executor._is_kv_manager_v2 = True
    executor.kv_cache_manager = target
    executor.draft_kv_cache_manager = draft
    executor.max_num_tokens = 16384
    executor.enable_attention_dp = False

    monkeypatch.delenv("TRTLLM_M3_ASYNC_OFFLOAD", raising=False)
    monkeypatch.delenv("TRTLLM_KV_ONBOARD_PREFETCH", raising=False)
    executor._init_kv_tier_staging()
    assert executor._kv_headroom_managers == []
    assert executor._kv_onboard_prefetch_reqs == 0
    assert executor._early_activated_requests == []

    monkeypatch.setenv("TRTLLM_M3_ASYNC_OFFLOAD", "1")
    monkeypatch.setenv("TRTLLM_KV_ONBOARD_PREFETCH", "6")
    executor._init_kv_tier_staging()
    assert [tokens for _, tokens in armed] == [16384, 16384]
    assert executor._kv_headroom_managers == [target]
    assert executor._kv_prefetch_managers == [target]
    assert executor._kv_onboard_prefetch_reqs == 6
    assert executor._kv_onboard_early_fetch
    assert executor._kv_onboard_defer_tokens == 0

    armed.clear()
    monkeypatch.setenv("TRTLLM_M3_ASYNC_OFFLOAD_TOKENS", "4096")
    monkeypatch.setenv("TRTLLM_KV_ONBOARD_PREFETCH_EARLY_FETCH", "0")
    monkeypatch.setenv("TRTLLM_KV_ONBOARD_PREFETCH_DEFER_TOKENS", "65536")
    executor._init_kv_tier_staging()
    assert executor._kv_onboard_defer_tokens == 65536
    assert [tokens for _, tokens in armed] == [4096, 4096]
    assert not executor._kv_onboard_early_fetch

    executor.enable_attention_dp = True
    monkeypatch.delenv("TRTLLM_KV_ONBOARD_PREFETCH_EARLY_FETCH")
    executor._init_kv_tier_staging()
    assert not executor._kv_onboard_early_fetch


def test_arrivals_with_much_to_onboard_are_staged_and_held_out_of_one_schedule():
    reqs = [_req(1), _req(2), _req(3), _req(4)]
    executor, log = _executor(reqs, num_reqs=8)
    executor.kv_cache_manager.onboarded = {2: 100, 3: 5000, 4: 4096}
    executor._kv_onboard_defer_tokens = 4096

    executor._stage_kv_for_new_arrivals(reqs[1:])
    assert log == [("target", "prefetch", [2, 3, 4], "measure"), ("draft", "prefetch", [2, 3, 4])]
    assert executor._kv_onboard_deferred_ids == {3, 4}

    class Scheduled(Exception):
        pass

    def schedule_request(active, inflight):
        raise Scheduled([r.py_request_id for r in active])

    executor.scheduler = SimpleNamespace(schedule_request=schedule_request)
    executor.inflight_req_ids = set()
    executor._maybe_record_hang_diagnostic_phase = lambda *args, **kwargs: None
    with pytest.raises(Scheduled) as first:
        executor._schedule()
    assert first.value.args[0] == [1, 2]
    with pytest.raises(Scheduled) as second:
        executor._schedule()
    assert second.value.args[0] == [1, 2, 3, 4]

    # Staged once: the next arrival pass neither re-stages nor defers them.
    log.clear()
    executor._stage_kv_for_new_arrivals(reqs[1:])
    assert log == [] and executor._kv_onboard_deferred_ids == set()


def test_arrival_staging_is_off_without_a_threshold():
    executor, log = _executor([_req(1)])
    executor._kv_onboard_defer_tokens = 0
    executor._stage_kv_for_new_arrivals([_req(1)])
    assert log == [] and executor._kv_onboard_deferred_ids == set()
