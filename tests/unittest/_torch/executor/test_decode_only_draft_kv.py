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
"""Decode-only draft KV pool (TLLM_DSPARK_DECODE_ONLY_DRAFT_KV).

On a disaggregated generation worker the legacy DSpark/DFlash drafter stores only tokens generated
there, so its separate draft pool may leave the transferred prompt out. The replays below drive the
draft mirror of KVCacheManagerV2 through the scheduler / prepare / update sequence of the executor,
with and without the flag, and check that

* the decode-only capacity is the full one minus a per-request constant at every step, and
* every drafter step still finds the columns it stores and the block latents it writes inside the
  allocation, so it stores and attends exactly what it would with the full reservation.

They also pin the budget arithmetic that hands the freed bytes and max_tokens to the target pool.
"""

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import CacheCost, KvCacheCreator
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    DECODE_ONLY_DRAFT_KV_ENV,
    DECODE_ONLY_DRAFT_KV_TOKENS_ENV,
    KVCacheManagerV2,
    compute_decode_only_draft_kv_slack,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy, KvCacheConfig

pytestmark = pytest.mark.cpu_only

GB = 1 << 30
K = 3  # max_draft_len of the K3 DSpark GEN configs
BLOCK = 3  # speculative_config.block_size there
PAGE = 32  # draft pool tokens_per_block
RID = 7


class _Cache:
    """KV cache stand-in that enforces the capacity / history rules of ``_KVCache.resize``."""

    def __init__(self):
        self.capacity = 0
        self.history_length = 0
        self.is_active = True

    def resize(self, capacity=None, history_length=None):
        capacity = self.capacity if capacity is None else capacity
        history_length = self.history_length if history_length is None else history_length
        if history_length < self.history_length:
            raise ValueError("History length cannot be decreased")
        if capacity < history_length:
            raise ValueError("History length cannot be greater than capacity")
        self.capacity, self.history_length = capacity, history_length
        return True


def _draft_manager(decode_only, page=PAGE, slack=None):
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_joint_kv_cache_reuse = False
    manager._has_cp_helix = False
    manager.num_extra_kv_tokens = 0
    manager.tokens_per_block = page
    manager.kv_cache_map = {}
    manager._allocated_draft_lens = {}
    manager._pending_overlap_slack = {}
    manager._kv_reserve_draft_tokens = K
    manager.max_total_draft_tokens = K
    manager.kv_compression_manages_history = False
    manager.decode_only_draft_kv = decode_only
    manager._decode_only_offsets = {}
    if decode_only:
        manager.decode_only_draft_kv_slack = (
            compute_decode_only_draft_kv_slack(
                SimpleNamespace(max_total_draft_tokens=K, block_size=BLOCK), page
            )
            if slack is None
            else slack
        )
    manager._resume_and_restore = lambda request_id, cache: True
    manager._mirror_draft_kv_cache = lambda req: manager.kv_cache_map.setdefault(
        req.py_request_id, _Cache()
    )
    return manager


def _gen_init_request(prompt_len):
    # The draft view of a disagg generation-init request: position 0, chunk = prompt.
    return SimpleNamespace(
        py_request_id=RID,
        prompt_len=prompt_len,
        max_beam_num_tokens=prompt_len,
        py_draft_tokens=[],
        context_phase_params=SimpleNamespace(draft_tokens=[0] * K),
        context_current_position=0,
        context_chunk_size=prompt_len,
        is_last_context_chunk=True,
        is_disagg_generation_init_state=True,
        is_disagg_generation_transmission_complete=False,
        py_disable_speculative_decoding=False,
        is_dummy_request=False,
        py_num_accepted_draft_tokens=0,
        py_rewind_len=0,
        state=LlmRequestState.DISAGG_GENERATION_INIT,
    )


def _batch(context=(), generation=()):
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = list(context)
    batch.generation_requests = list(generation)
    return batch


def _replay(decode_only, prompt_len, accepts, overlap=True, page=PAGE, slack=None, block=BLOCK):
    """Drive one disagg request; return the mirror capacity at each forward and the drafter faults.

    A fault is a step whose drafter stores (columns ``ctx_len .. ctx_len + K``) or block latents
    (``new_len .. new_len + block - 1``) would leave the request's allocated pages, i.e. a step
    the drafter would clamp or truncate.
    """
    manager = _draft_manager(decode_only, page, slack)
    req = _gen_init_request(prompt_len)
    # Admission: a decode-only pool is paired with the target in the scheduler; the executor then
    # mirrors the generation-init request through prepare_resources (_prepare_disagg_gen_init).
    if decode_only:
        assert manager.try_allocate_draft_disagg_gen_init(req)
    manager.prepare_resources(_batch(context=[req]))
    req.is_disagg_generation_init_state = False
    req.is_disagg_generation_transmission_complete = True
    req.state = LlmRequestState.GENERATION_IN_PROGRESS

    capacities, faults, pending, ctx_len = [], [], [], 0
    for step, accepted in enumerate(accepts):
        batch = _batch(generation=[req])
        if decode_only:
            assert manager.try_allocate_generation(req)
        if step == 0:
            # _prepare_disagg_gen_transmission_complete: first token and context drafts.
            req.max_beam_num_tokens = prompt_len + 1
            req.py_draft_tokens = [0] * K
            req.is_disagg_generation_transmission_complete = False
        manager.prepare_resources(batch)
        capacity = manager.kv_cache_map[RID].capacity
        pages = -(-capacity // page) * page
        new_len = ctx_len + accepted
        if ctx_len + K + 1 > pages or new_len + block > pages:
            faults.append(step)
        capacities.append(capacity)
        ctx_len = new_len
        pending.append((batch, accepted))
        # The overlap executor updates step i - 1 after launching step i.
        while len(pending) > (1 if overlap else 0):
            done, done_accepted = pending.pop(0)
            req.max_beam_num_tokens += done_accepted
            req.py_num_accepted_draft_tokens = done_accepted - 1
            req.py_rewind_len = K - (done_accepted - 1)
            manager.update_resources(done)
    return manager, capacities, faults


def _acceptance(seed, steps=200):
    rng = random.Random(seed)
    return [rng.randint(1, K + 1) for _ in range(steps)]


SEQUENCES = {
    "all_accepted": [K + 1] * 200,
    "none_accepted": [1] * 200,
    "alternating": [K + 1, 1] * 100,
    "random0": _acceptance(0),
    "random1": _acceptance(1),
}


@pytest.mark.parametrize("overlap", [True, False], ids=["overlap", "sync"])
@pytest.mark.parametrize("prompt_len", [40000, 3000, 40])
@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_decode_only_is_full_reservation_minus_offset(name, prompt_len, overlap):
    accepts = SEQUENCES[name]
    _, full, full_faults = _replay(False, prompt_len, accepts, overlap)
    manager, decode_only, faults = _replay(True, prompt_len, accepts, overlap)
    slack = manager.decode_only_draft_kv_slack
    offset = max(0, prompt_len - slack)
    assert manager._decode_only_offsets == {RID: offset}
    assert [f - d for f, d in zip(full, decode_only)] == [offset] * len(accepts)
    assert full_faults == [] and faults == []
    # The pool holds only what was generated, plus slack and one round of growth.
    assert decode_only[-1] <= sum(accepts) + slack + 2 * (1 + 2 * K)


@pytest.mark.parametrize("overlap", [True, False], ids=["overlap", "sync"])
@pytest.mark.parametrize("name", sorted(SEQUENCES))
@pytest.mark.parametrize("block", [BLOCK, 16])
def test_block_size_plus_one_slack_suffices_without_page_rounding(block, name, overlap):
    # Each step's 1 + 2K growth keeps a full round ahead of the stored tokens, so only the
    # block latents past it lean on the slack. dflash.py requires block_size + 1.
    _, _, faults = _replay(
        True, 40000, SEQUENCES[name], overlap, page=1, slack=block + 1, block=block
    )
    assert faults == []


@pytest.mark.parametrize("overlap", [True, False], ids=["overlap", "sync"])
def test_large_block_without_slack_is_truncated(overlap):
    # Negative control: the growth alone cannot hold a 16-token block past the stored tokens.
    _, _, faults = _replay(
        True, 40000, SEQUENCES["all_accepted"], overlap, page=1, slack=0, block=16
    )
    assert faults


def test_free_drops_the_offset():
    manager, _, _ = _replay(True, 40000, [2] * 5)
    manager.kv_cache_map[RID].close = Mock()
    manager.kv_cache_map[RID].discard_pending_stats = Mock()
    manager.conversation_manager = None
    manager.impl = Mock()
    manager._early_freed_index_requests = {RID}
    manager.free_resources(SimpleNamespace(py_request_id=RID))
    assert manager._decode_only_offsets == {}


def test_local_prefill_keeps_full_reservation():
    manager = _draft_manager(True)
    req = _gen_init_request(5000)
    req.is_disagg_generation_init_state = False
    manager.prepare_resources(_batch(context=[req]))
    assert manager.kv_cache_map[RID].capacity == 5000
    assert manager._decode_only_offsets == {}


def test_first_seen_generation_request_offsets_its_tokens():
    manager = _draft_manager(True)
    req = _gen_init_request(5000)
    req.is_disagg_generation_init_state = False
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.max_beam_num_tokens = 5001
    req.py_draft_tokens = [0] * K
    manager.prepare_resources(_batch(generation=[req]))
    assert manager._decode_only_offsets == {RID: 5000 - manager.decode_only_draft_kv_slack}
    assert manager.kv_cache_map[RID].capacity == 1 + 2 * K


def test_flag_off_mirror_is_unchanged():
    manager = _draft_manager(False)
    manager.register_decode_only_history(RID, 40000)
    assert manager.decode_only_capacity(RID, 40003) == 40003
    req = _gen_init_request(40000)
    assert manager.try_allocate_draft_disagg_gen_init(req)
    assert manager.kv_cache_map[RID].capacity == 40000


def test_available_tokens_ignore_the_prompt():
    manager = _draft_manager(True)
    manager._standalone_draft_reserve = 0
    manager._gpu_max_tokens = 400000
    manager.impl = Mock()
    manager.impl.clamp_max_seq_len_for_mem.side_effect = lambda batch, length: min(length, 3000)
    assert manager.get_num_available_tokens(token_num_upper_bound=262144, batch_size=96) == 262144
    manager.impl.clamp_max_seq_len_for_mem.side_effect = lambda batch, length: 10
    assert manager.get_num_available_tokens(token_num_upper_bound=262144, batch_size=96) < 262144


class TestSchedulerPairing:
    def _scheduler(self, draft):
        sched = KVCacheV2Scheduler.__new__(KVCacheV2Scheduler)
        target = Mock()
        target.try_allocate_generation.return_value = True
        sched.kv_cache_manager = target
        sched.draft_kv_cache_manager = draft
        sched._gen_alloc_draft_manager = draft
        return sched, target

    @pytest.mark.parametrize(
        "draft, paired",
        [
            (SimpleNamespace(decode_only_draft_kv=True), True),
            (SimpleNamespace(decode_only_draft_kv=False), False),
            (Mock(), False),
            (None, False),
        ],
        ids=["decode_only", "unpaired", "mock", "none"],
    )
    def test_decode_only_draft_pool_is_admission_paired(self, draft, paired, monkeypatch):
        target = Mock(tokens_per_block=64, _has_cp_helix=False, enable_joint_kv_cache_reuse=False)
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2.KVCacheManagerV2", Mock
        )
        sched = KVCacheV2Scheduler(
            max_batch_size=8,
            max_num_tokens=256,
            kv_cache_manager=target,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            draft_kv_cache_manager=draft,
        )
        assert sched._gen_alloc_draft_manager is (draft if paired else None)
        assert sched._joint_draft_manager is None

    def test_draft_overflow_rolls_back_the_target_step(self):
        draft = Mock()
        draft.try_allocate_generation.return_value = False
        sched, target = self._scheduler(draft)
        req = SimpleNamespace(py_request_id=RID)
        assert not sched._try_allocate_generation(req)
        target.revert_allocate_generation.assert_called_once_with(req)

    def test_draft_overflow_skips_generation_init(self):
        draft = Mock()
        draft.try_allocate_draft_disagg_gen_init.return_value = False
        sched, target = self._scheduler(draft)
        target.prepare_disagg_gen_init.return_value = True
        req = SimpleNamespace(py_request_id=RID)
        action, _ = sched._try_schedule_disagg_gen_init(req, None)
        assert action.value == "skip"
        target.revert_allocate_context.assert_called_once_with(req)
        draft.free_resources.assert_called_once_with(req)


@pytest.mark.parametrize("paired", [True, False])
def test_skipped_batch_reverts_a_paired_draft_step(paired):
    stub = SimpleNamespace(
        _is_kv_manager_v2=True,
        kv_cache_manager=Mock(),
        draft_kv_cache_manager=Mock(),
        _draft_gen_alloc_paired=paired,
    )
    req = SimpleNamespace(py_skip_gen_alloc_revert=False)
    PyExecutor._revert_gen_alloc(stub, Mock(generation_requests=[req]))
    stub.kv_cache_manager.revert_allocate_generation.assert_called_once_with(req)
    assert stub.draft_kv_cache_manager.revert_allocate_generation.called is paired


# Per-rank KV budgets of two DEP16 generation configurations:
# total = max_tokens * (target + draft bytes/token) + the target's fixed KDA state.
TARGET_SLOPE, DRAFT_SLOPE = 13824, 5760
CENSUS = {
    "config_a": (2_959_269, 27_756_442_784, 96),
    "config_b": (4_236_645, 18_693_114_528, 64),
}


def _creator(max_batch_size, fixed, max_tokens_in=None):
    creator = object.__new__(KvCacheCreator)
    creator._max_batch_size = max_batch_size
    creator._max_kv_tokens_in = max_tokens_in
    creator._speculative_config = SimpleNamespace(max_total_draft_tokens=K, block_size=BLOCK)
    creator._get_draft_tokens_per_block = Mock(return_value=PAGE)
    creator._get_target_and_draft_cache_costs = Mock(
        return_value=(CacheCost(TARGET_SLOPE, fixed), CacheCost(DRAFT_SLOPE))
    )
    return creator


def _split(name, max_tokens_in=None):
    max_tokens, fixed, max_batch = CENSUS[name]
    total = max_tokens * (TARGET_SLOPE + DRAFT_SLOPE) + fixed
    creator = _creator(max_batch, fixed, max_tokens_in)
    target_budget, draft_budget = creator._compute_draft_budget_shares(
        total, CacheCost(TARGET_SLOPE, fixed), CacheCost(DRAFT_SLOPE)
    )
    shared = KvCacheConfig(max_tokens=max_tokens)
    target = shared.model_copy(update={"max_gpu_total_bytes": target_budget})
    draft = shared.model_copy(update={"max_gpu_total_bytes": draft_budget})
    return total, creator._size_decode_only_draft_budget(target, draft, 262144)


@pytest.mark.parametrize(
    "name, target_tokens",
    [("config_a", 4_020_103), ("config_b", 5_885_946)],
)
def test_freed_draft_budget_raises_target_max_tokens(name, target_tokens, monkeypatch):
    monkeypatch.delenv(DECODE_ONLY_DRAFT_KV_TOKENS_ENV, raising=False)
    total, (target, draft) = _split(name)
    max_tokens, fixed, max_batch = CENSUS[name]
    per_request = 4096 + (2 * PAGE + BLOCK + 2 * (K + 1)) + 2 * (1 + 2 * K) + PAGE
    assert draft.max_gpu_total_bytes == DRAFT_SLOPE * (max_batch + 2) * per_request
    assert target.max_gpu_total_bytes + draft.max_gpu_total_bytes == total
    assert target.max_tokens == target_tokens
    assert draft.max_tokens == (max_batch + 2) * per_request
    assert target.max_tokens > 1.35 * max_tokens


def test_explicit_max_tokens_is_kept(monkeypatch):
    monkeypatch.delenv(DECODE_ONLY_DRAFT_KV_TOKENS_ENV, raising=False)
    _, (target, draft) = _split("config_a", max_tokens_in=2_959_269)
    assert target.max_tokens == draft.max_tokens == 2_959_269


def test_pool_holds_one_max_length_request(monkeypatch):
    monkeypatch.setenv(DECODE_ONLY_DRAFT_KV_TOKENS_ENV, "16")
    _, (_, draft) = _split("config_b")
    assert draft.max_tokens >= 262144


class TestResolve:
    def _creator(self, **overrides):
        creator = object.__new__(KvCacheCreator)
        mode = Mock()
        mode.is_dflash.return_value = False
        mode.is_dspark.return_value = True
        creator._speculative_config = SimpleNamespace(spec_dec_mode=mode)
        creator._is_kv_cache_manager_v2 = True
        creator._joint_kv_cache_reuse = False
        creator._is_disagg = True
        for key, value in overrides.items():
            setattr(creator, key, value)
        return creator

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(DECODE_ONLY_DRAFT_KV_ENV, raising=False)
        assert not self._creator()._resolve_decode_only_draft_kv(True)

    def test_generation_worker(self, monkeypatch):
        monkeypatch.setenv(DECODE_ONLY_DRAFT_KV_ENV, "1")
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
        assert self._creator()._resolve_decode_only_draft_kv(True)

    @pytest.mark.parametrize(
        "overrides, separate, role",
        [
            ({}, True, "context"),
            ({}, False, ""),
            ({"_joint_kv_cache_reuse": True}, True, ""),
            ({"_is_disagg": False}, True, ""),
            ({"_is_kv_cache_manager_v2": False}, True, ""),
        ],
        ids=["context_worker", "no_draft_pool", "joint_reuse", "aggregated", "v1_manager"],
    )
    def test_ignored(self, overrides, separate, role, monkeypatch):
        monkeypatch.setenv(DECODE_ONLY_DRAFT_KV_ENV, "1")
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
        assert not self._creator(**overrides)._resolve_decode_only_draft_kv(separate)
