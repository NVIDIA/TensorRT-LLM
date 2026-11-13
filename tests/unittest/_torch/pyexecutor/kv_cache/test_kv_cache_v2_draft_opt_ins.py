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
"""Model opt-ins of a standalone DFlash / DSpark draft KV cache manager.

``draft_kv_reserve_guard`` (RuntimeHooks.draft_kv_reserve_guard) widens the draft manager's
generation reserve by the drafter block width; ``draft_pool_fixes``
(RuntimeHooks.draft_pool_fixes) resizes a windowed mirror in two steps so its stale window pages
are released before the next chunk is allocated. Both are keyword arguments of the draft manager
and its sizing classmethods that the KV cache creator passes for an opted-in model only, so one
model's opt-in never reaches another manager in the same process.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only


def _spec_config(dflash: bool):
    mode = SimpleNamespace(is_dflash=lambda: dflash, is_dspark=lambda: False)
    return SimpleNamespace(
        max_total_draft_tokens=7,
        max_draft_len=7,
        tokens_per_gen_step=8,
        spec_dec_mode=mode,
        use_dynamic_tree=False,
    )


def test_draft_reserve_widens_only_when_opted_in(monkeypatch):
    monkeypatch.setattr(kvm, "_get_kv_alloc_extra_tokens", lambda *args: 1)
    spec = _spec_config(dflash=True)
    base = kvm._get_generation_kv_capacity(spec, is_draft=True)
    assert base == (7, 1 + 8)

    reserve, headroom = kvm._get_generation_kv_capacity(
        spec, is_draft=True, draft_kv_reserve_guard=True
    )
    assert (reserve, headroom) == (base[0] + 7, base[1] + 7)
    # A later call without the opt-in (another model's manager) is unchanged.
    assert kvm._get_generation_kv_capacity(spec, is_draft=True) == base
    # The target manager and other drafters keep their reserve.
    assert kvm._get_generation_kv_capacity(spec, is_draft=False, draft_kv_reserve_guard=True) == (
        7,
        1 + 8,
    )
    other = _spec_config(dflash=False)
    assert (
        kvm._get_generation_kv_capacity(other, is_draft=True, draft_kv_reserve_guard=True) == base
    )


class _Cache:
    def __init__(self, capacity, history_length, fail_at=None):
        self.capacity = capacity
        self.history_length = history_length
        self.fail_at = fail_at
        self.calls = []

    def resize(self, capacity, history_length=None):
        self.calls.append((capacity, history_length))
        if capacity == self.fail_at:
            return False
        self.capacity = capacity
        if history_length is not None:
            self.history_length = history_length
        return True


def test_resize_draft_context_commits_history_before_growing():
    cache = _Cache(capacity=64, history_length=0)
    assert kvm._resize_draft_context(cache, 96, 64)
    assert cache.calls == [(64, 64), (96, 64)]


@pytest.mark.parametrize(
    "capacity, history_length, target, history",
    [
        (64, 64, 96, 64),  # history already committed
        (32, 0, 96, 64),  # history beyond the current capacity
        (64, 0, 64, 32),  # no growth
    ],
)
def test_resize_draft_context_is_one_resize_otherwise(capacity, history_length, target, history):
    cache = _Cache(capacity=capacity, history_length=history_length)
    assert kvm._resize_draft_context(cache, target, history)
    assert cache.calls == [(target, history)]


def test_resize_draft_context_stops_when_the_history_commit_fails():
    cache = _Cache(capacity=64, history_length=0, fail_at=64)
    assert not kvm._resize_draft_context(cache, 96, 64)
    assert cache.calls == [(64, 64)]


class _Req:
    py_request_id = 3
    context_current_position = 64


def _draft_manager(draft_pool_fixes, cache):
    manager = object.__new__(kvm.KVCacheManagerV2)
    manager.is_draft = True
    manager.kv_cache_map = {_Req.py_request_id: cache}
    manager._resume_and_restore = lambda request_id, kv_cache: True
    manager.num_extra_kv_tokens = 0
    manager._draft_pool_fixes = draft_pool_fixes
    return manager


@pytest.mark.parametrize("draft_pool_fixes", [False, True])
def test_draft_context_resize_follows_the_manager_opt_in(draft_pool_fixes):
    cache = _Cache(capacity=64, history_length=0)
    manager = _draft_manager(draft_pool_fixes, cache)
    assert manager.try_allocate_draft_context(_Req(), 32)
    if draft_pool_fixes:
        # History committed first, then the growth.
        assert cache.calls == [(64, 64), (96, 64)]
    else:
        # The previous single resize, no history hint.
        assert cache.calls == [(96, None)]


def test_one_managers_opt_in_does_not_reach_another():
    opted_in = _draft_manager(True, _Cache(capacity=64, history_length=0))
    plain_cache = _Cache(capacity=64, history_length=0)
    plain = _draft_manager(False, plain_cache)
    assert opted_in.try_allocate_draft_context(_Req(), 32)
    assert plain.try_allocate_draft_context(_Req(), 32)
    assert plain_cache.calls == [(96, None)]


def test_creator_passes_only_the_declared_opt_ins_to_a_v2_draft_manager():
    assert _util._draft_kv_opt_ins(RuntimeHooks()) == {}
    hooks = RuntimeHooks(draft_pool_fixes=True, draft_kv_reserve_guard=True)
    opt_ins = _util._draft_kv_opt_ins(hooks)
    assert opt_ins == {"draft_pool_fixes": True, "draft_kv_reserve_guard": True}
    base = {"is_draft": True}
    assert _util._with_draft_kv_opt_ins(kvm.KVCacheManagerV2, {}, base) is base
    assert _util._with_draft_kv_opt_ins(kvm.KVCacheManagerV2, opt_ins, base) == {
        **base,
        **opt_ins,
    }
    # A manager that does not take them runs without.
    assert _util._with_draft_kv_opt_ins(object, opt_ins, base) is base
    assert base == {"is_draft": True}
