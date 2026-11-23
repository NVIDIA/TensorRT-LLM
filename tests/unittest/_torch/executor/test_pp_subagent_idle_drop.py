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
"""PP glue of the idle child-chain drop (``TLLM_KV_SUBAGENT_IDLE_DROP_S``).

The first rank decides at each pass of the real ``PyExecutor._executor_loop_pp``,
after the request fetch and before the stage-agreement exchange, and its keys ride
that pass's schedule message; a follower demotes them after receiving the message
and before applying the released reservations. Stand-ins replace the scheduler,
the KV cache manager and the PP transport; the loop and the message are real.
"""

import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.subagent_idle_drop import (
    DECIDE_EVERY_PASSES,
    DEFAULT_PARENT_RESUME_FALLBACK_S,
    IDLE_S_ENV,
    PARENT_RESUME_S_ENV,
    SubagentIdleDropTracker,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SerializableSchedulerOutput
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    LayerId,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only


class _Stop(Exception):
    """Ends the loop at the step under test."""


class _Hang:
    def checkpoint(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class _Manager:
    """The KV cache manager's idle-drop surface."""

    def __init__(self, decided: List[str], events: List[tuple]) -> None:
        self.subagent_idle_drop_enabled = True
        self.decided = decided
        self.events = events
        self.busy: List[str] = []
        self.disabled = False

    def subagent_idle_drop_decide(self, pass_idx, busy_fn):
        self.busy = list(busy_fn())
        self.events.append(("decide", pass_idx))
        return list(self.decided) or None

    def apply_subagent_idle_drops(self, keys, pass_idx=None) -> None:
        self.events.append(("apply", list(keys), pass_idx))

    def disable_subagent_idle_drop(self) -> None:
        self.disabled = True


class _ParentManager(_Manager):
    """The same surface with the parent-resume trigger on."""

    subagent_parent_resume_enabled = True

    def subagent_idle_drop_decide(self, pass_idx, busy_fn, parent_busy_fn):
        self.parents = list(parent_busy_fn())
        return super().subagent_idle_drop_decide(pass_idx, busy_fn)


def _child(conv: str, affinity: str = "parent"):
    return SimpleNamespace(
        is_dummy_request=False,
        multimodal_hashes=None,
        py_subagent_affinity_id=affinity,
        py_conversation_params=SimpleNamespace(conversation_id=conv),
    )


def _queued(conv: str, affinity):
    return SimpleNamespace(
        request=SimpleNamespace(
            py_scheduling_params=SimpleNamespace(subagent_affinity_id=affinity),
            py_conversation_params=SimpleNamespace(conversation_id=conv),
        )
    )


@pytest.fixture
def cpu_loop(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(py_executor_module, "CUASSERT", lambda result: result)
    monkeypatch.setattr(py_executor_module, "cudart", MagicMock())


def _executor(rank: int, manager: _Manager, events: List[tuple]) -> PyExecutor:
    exe = object.__new__(PyExecutor)
    exe.dist = SimpleNamespace(
        rank=rank,
        pp_rank=rank,
        pp_size=2,
        tp_size=1,
        cp_size=1,
        is_first_pp_rank=rank == 0,
        is_last_pp_rank=rank == 1,
        next_pp_rank=rank + 1,
        prev_pp_rank=rank - 1,
    )
    exe.device_id = 0
    exe._profiler = lambda: nullcontext(lambda: None)
    exe.hang_detector = _Hang()
    exe.enable_iter_perf_stats = False
    exe._is_kv_manager_v2 = False
    exe.is_shutdown = False
    exe.active_requests = []
    exe.waiting_queue = []
    exe.enable_attention_dp = False
    # ``disagg`` is a read-only property that returns ``_disagg_coordinator`` once set.
    exe._disagg_coordinator = MagicMock()
    exe._disagg_coordinator.admit.return_value = ([], False)
    exe.kv_cache_manager = manager
    exe._pp_disagg_lag = None
    exe._pp_stage_agree = SimpleNamespace(take_proposals=lambda: None)
    exe._subagent_idle_drop_on = True

    def fetch():
        events.append(("fetch",))
        return []

    exe._fetch_and_activate_new_requests = fetch
    exe._handle_control_request = lambda: None
    exe._init_iter_stats_if_sampled = lambda num_new: None
    exe._pad_attention_dp_dummy_request = lambda: None
    exe._pp_stage_agree_exchange = lambda: events.append(("exchange", exe._subagent_idle_pending))
    return exe


class TestScheduleMessage:
    def test_round_trip(self) -> None:
        empty = ScheduledRequests()
        assert empty.leader_subagent_idle_drops is None
        message = SerializableSchedulerOutput.from_scheduler_result(
            empty, [], 0, subagent_idle_drops=["child-a", "child-b"]
        )
        received = pickle.loads(pickle.dumps(message))
        batch, _, _ = received.to_scheduler_result([])
        assert batch.leader_subagent_idle_drops == ["child-a", "child-b"]

    def test_none_when_off(self) -> None:
        message = SerializableSchedulerOutput.from_scheduler_result(ScheduledRequests(), [], 0)
        assert message.subagent_idle_drops is None
        batch, _, _ = pickle.loads(pickle.dumps(message)).to_scheduler_result([])
        assert batch.leader_subagent_idle_drops is None


class TestFirstRank:
    def test_decides_after_fetch_before_exchange_and_stamps_the_message(self, cpu_loop) -> None:
        events: List[tuple] = []
        manager = _Manager(["child-a"], events)
        exe = _executor(0, manager, events)
        exe.active_requests = [_child("busy-active"), _child("main", affinity=None)]
        exe.waiting_queue = [_queued("busy-queued", "parent"), _queued("flat-main", None)]
        exe._schedule = lambda: (ScheduledRequests(), [], 0)
        exe._pp_stage_agree_check_admitted = lambda batch: None
        exe.send_schedule_handles = [None]
        exe.wait_on_pp_send_handles = lambda handles, microbatch_id: None
        sent = []

        def isend_object(obj, dest, tag):
            sent.append(obj)
            raise _Stop

        exe.dist.isend_object = isend_object
        with pytest.raises(_Stop):
            exe._executor_loop_pp()

        assert events == [("fetch",), ("decide", 1), ("exchange", ["child-a"])]
        assert manager.busy == ["busy-active", "busy-queued"]
        assert len(sent) == 1 and sent[0].subagent_idle_drops == ["child-a"]
        # Taken once: the next pass sends its own decision.
        assert exe._subagent_idle_pending is None

    def test_parent_resume_gets_every_conversation_id(self, cpu_loop) -> None:
        events: List[tuple] = []
        manager = _ParentManager(["child-a"], events)
        exe = _executor(0, manager, events)
        exe.active_requests = [_child("busy-active"), _child("main", affinity=None)]
        exe.waiting_queue = [_queued("busy-queued", "parent"), _queued("flat-main", None)]

        def stop():
            raise _Stop

        exe._pp_stage_agree_exchange = stop
        with pytest.raises(_Stop):
            exe._executor_loop_pp()
        assert events == [("fetch",), ("decide", 1)]
        # Child keys need a parent-session id; parent ids are every conversation id.
        assert manager.busy == ["busy-active", "busy-queued"]
        assert manager.parents == ["busy-active", "main", "busy-queued", "flat-main"]
        assert exe._subagent_idle_pending == ["child-a"]

    def test_off_leaves_the_loop_unchanged(self, cpu_loop) -> None:
        events: List[tuple] = []
        manager = _Manager(["child-a"], events)
        exe = _executor(0, manager, events)
        exe._subagent_idle_drop_on = False

        def stop():
            raise _Stop

        exe._pp_stage_agree_exchange = stop
        with pytest.raises(_Stop):
            exe._executor_loop_pp()
        assert events == [("fetch",)]
        assert exe._subagent_idle_pass == 0
        assert exe._take_subagent_idle_pending() is None


class TestFollower:
    @pytest.mark.parametrize("keys", [["child-a"], None, []])
    def test_applies_before_the_early_release(self, cpu_loop, keys) -> None:
        events: List[tuple] = []
        manager = _Manager([], events)
        exe = _executor(1, manager, events)
        batch = ScheduledRequests()
        batch.leader_subagent_idle_drops = keys

        def propagate(microbatch_id):
            events.append(("propagate",))
            return batch, [], 0, False

        def early_release(scheduled_batch):
            events.append(("early_release",))
            raise _Stop

        exe._pp_schedule_and_propagate = propagate
        exe._pp_follower_early_release = True
        exe._pp_stage_agree_early_release = early_release
        with pytest.raises(_Stop):
            exe._executor_loop_pp()

        expected = [("fetch",), ("exchange", None), ("propagate",)]
        if keys:
            # Tagged with the follower's own pass count, the first rank's pass index.
            expected.append(("apply", keys, 1))
        expected.append(("early_release",))
        assert events == expected
        # A follower never decides.
        assert not any(event[0] == "decide" for event in events)


class TestExecutorGate:
    @pytest.fixture(autouse=True)
    def knob(self, monkeypatch):
        monkeypatch.setenv(IDLE_S_ENV, "240")
        return monkeypatch

    def _bare(self, manager, agreement, pp_size: int, others=(True,)) -> PyExecutor:
        exe = object.__new__(PyExecutor)
        exe.kv_cache_manager = manager
        exe._pp_stage_agree = agreement
        exe.gathered = []

        def pp_allgather(local):
            exe.gathered.append(local)
            return [local, *others]

        exe.dist = SimpleNamespace(pp_size=pp_size, pp_allgather=pp_allgather)
        return exe

    def test_on_with_stage_agreement_and_pp(self) -> None:
        manager = _Manager([], [])
        exe = self._bare(manager, object(), 2)
        assert exe._init_subagent_idle_drop() is True
        assert exe.gathered == [True]
        assert not manager.disabled

    @pytest.mark.parametrize("agreement, pp_size", [(None, 2), (object(), 1)])
    def test_refuses_and_disables_without_stage_agreement(self, agreement, pp_size) -> None:
        manager = _Manager([], [])
        exe = self._bare(manager, agreement, pp_size)
        assert exe._init_subagent_idle_drop() is False
        assert manager.disabled
        # The condition is the same on every rank: nothing is exchanged.
        assert exe.gathered == []

    def test_another_stage_refused(self) -> None:
        manager = _Manager([], [])
        exe = self._bare(manager, object(), 4, others=(True, False, True))
        assert exe._init_subagent_idle_drop() is False
        assert manager.disabled

    def test_manager_without_the_tracker(self) -> None:
        for manager in (SimpleNamespace(), None):
            exe = self._bare(manager, object(), 2)
            # It still takes part in the exchange, so the other stages refuse too.
            assert exe._init_subagent_idle_drop() is False
            assert exe.gathered == [False]

    def test_parent_resume_alone_turns_it_on(self, knob) -> None:
        knob.delenv(IDLE_S_ENV, raising=False)
        knob.setenv(PARENT_RESUME_S_ENV, "60")
        manager = _Manager([], [])
        exe = self._bare(manager, object(), 2)
        assert exe._init_subagent_idle_drop() is True
        assert exe.gathered == [True]

    def test_off_exchanges_nothing(self, knob) -> None:
        for value in (None, "0"):
            if value is None:
                knob.delenv(IDLE_S_ENV, raising=False)
            else:
                knob.setenv(IDLE_S_ENV, value)
            manager = _Manager([], [])
            exe = self._bare(manager, object(), 2)
            assert exe._init_subagent_idle_drop() is False
            assert exe.gathered == []
            assert not manager.disabled


class _Handle:
    def __init__(self, log: List[str], name: str) -> None:
        self.log = log
        self.name = name
        self.valid = True

    @property
    def num_refs(self) -> int:
        return 3 if self.valid else 0

    def absorb(self, other) -> None:
        other.valid = False

    def demote(self):
        self.valid = False
        self.log.append(f"demote {self.name}")
        return (3, 3, 3, 0, 0, 0)

    def release(self) -> None:
        self.valid = False
        self.log.append(f"release {self.name}")


class TestManagerGlue:
    def _manager(self) -> KVCacheManagerV2:
        manager = object.__new__(KVCacheManagerV2)
        manager._subagent_idle_drop = SubagentIdleDropTracker(
            idle_s=1.0, tokens_per_block=32, rank=0
        )
        return manager

    def test_off_by_default(self) -> None:
        manager = object.__new__(KVCacheManagerV2)
        assert manager.subagent_idle_drop_enabled is False
        assert manager.subagent_parent_resume_enabled is False
        assert manager.subagent_idle_drop_decide(DECIDE_EVERY_PASSES, list) is None
        manager.apply_subagent_idle_drops(["a"])
        manager.disable_subagent_idle_drop()

    def test_decide_apply_and_disable(self) -> None:
        log: List[str] = []
        manager = self._manager()
        tracker = manager._subagent_idle_drop
        assert manager.subagent_idle_drop_enabled
        req = _child("c")
        req.prepopulated_prompt_len = 0
        req.context_current_position = 64
        req.context_remaining_length = 0
        tracker.commit(req, lambda from_block: _Handle(log, "c"), now=0.0)
        # Nothing idle yet at a decision pass: None, not an empty list.
        assert manager.subagent_idle_drop_decide(DECIDE_EVERY_PASSES, list) is None
        manager.apply_subagent_idle_drops(["missing"], 7)
        assert tracker.stats["missing"] == 1

        other = _child("d")
        other.prepopulated_prompt_len = 0
        other.context_current_position = 64
        other.context_remaining_length = 0
        tracker.commit(other, lambda from_block: _Handle(log, "d"), now=0.0)
        manager.apply_subagent_idle_drops(["c"])
        assert log == ["demote c"]
        manager.disable_subagent_idle_drop()
        assert log == ["demote c", "release d"]
        assert manager.subagent_idle_drop_enabled is False

    def test_parent_resume_decide(self) -> None:
        log: List[str] = []
        manager = object.__new__(KVCacheManagerV2)
        manager._subagent_idle_drop = SubagentIdleDropTracker(
            idle_s=1e6, tokens_per_block=32, rank=0, parent_resume_s=1e-3
        )
        assert manager.subagent_parent_resume_enabled
        req = _child("c", affinity="P")
        req.prepopulated_prompt_len = 0
        req.context_current_position = 64
        req.context_remaining_length = 0
        manager._subagent_idle_drop.commit(req, lambda from_block: _Handle(log, "c"), now=0.0)
        later = 4 * DECIDE_EVERY_PASSES * DECIDE_EVERY_PASSES
        assert manager.subagent_idle_drop_decide(later, list, lambda: ["P"]) == ["c"]
        assert log == ["demote c"]


def _maker_manager(**overrides) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    attrs = dict(
        mapping=SimpleNamespace(enable_attention_dp=False, tp_size=1, cp_size=1, pp_size=4),
        max_attention_window_vec=[None, 128],
        max_seq_len=4096,
        enable_block_reuse=True,
        block_reuse_policy=BlockReusePolicy.ALL_REUSABLE,
        _can_publish_block_reuse=True,
        is_draft=False,
        # A disaggregated server, where the PP stage agreement runs.
        is_disagg=True,
        _has_cp_helix=False,
        tokens_per_block=32,
        impl=SimpleNamespace(chain_demote_stats=lambda: (0, 0)),
        kv_cache_manager_py_config=SimpleNamespace(
            layers=[AttentionLayerConfig(LayerId(0), [BufferConfig("key", 4096)])]
        ),
    )
    attrs.update(overrides)
    for name, value in attrs.items():
        setattr(manager, name, value)
    return manager


class TestMaker:
    @pytest.fixture
    def env(self, monkeypatch):
        monkeypatch.setenv(IDLE_S_ENV, "240")
        return monkeypatch

    def test_off_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv(IDLE_S_ENV, raising=False)
        assert _maker_manager()._make_subagent_idle_drop_tracker() is None
        monkeypatch.setenv(IDLE_S_ENV, "0")
        assert _maker_manager()._make_subagent_idle_drop_tracker() is None

    def test_bad_value_raises(self, monkeypatch) -> None:
        monkeypatch.setenv(IDLE_S_ENV, "-5")
        with pytest.raises(ValueError, match=IDLE_S_ENV):
            _maker_manager()._make_subagent_idle_drop_tracker()

    def test_enabled(self, env) -> None:
        tracker = _maker_manager()._make_subagent_idle_drop_tracker()
        assert isinstance(tracker, SubagentIdleDropTracker)
        assert tracker.idle_s == 240.0 and tracker.tokens_per_block == 32

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(enable_block_reuse=False),
            dict(block_reuse_policy=BlockReusePolicy.PER_CONVERSATION),
            dict(_can_publish_block_reuse=False),
            dict(is_draft=True),
            dict(_has_cp_helix=True),
            dict(max_attention_window_vec=[128, 256]),
            dict(
                mapping=SimpleNamespace(enable_attention_dp=True, tp_size=1, cp_size=1, pp_size=4)
            ),
            dict(
                mapping=SimpleNamespace(enable_attention_dp=False, tp_size=2, cp_size=1, pp_size=4)
            ),
            dict(
                mapping=SimpleNamespace(enable_attention_dp=False, tp_size=1, cp_size=2, pp_size=4)
            ),
            dict(
                mapping=SimpleNamespace(enable_attention_dp=False, tp_size=1, cp_size=1, pp_size=1)
            ),
        ],
    )
    def test_ignored(self, env, overrides) -> None:
        assert _maker_manager(**overrides)._make_subagent_idle_drop_tracker() is None

    @pytest.mark.parametrize(
        "layer",
        [
            SsmLayerConfig(
                layer_id=LayerId(1), buffers=[BufferConfig(role=DataRole("ssm_state"), size=8192)]
            ),
            AttentionLayerConfig(LayerId(1), [BufferConfig("key", 4096, is_sparse=True)]),
        ],
    )
    def test_ignored_with_a_life_cycle_the_planner_does_not_cover(self, env, layer) -> None:
        dense = AttentionLayerConfig(LayerId(0), [BufferConfig("key", 4096)])
        manager = _maker_manager(kv_cache_manager_py_config=SimpleNamespace(layers=[dense, layer]))
        assert manager._make_subagent_idle_drop_tracker() is None

    def test_ignored_without_stage_agreement(self, env) -> None:
        """An aggregated server never runs the PP stage agreement."""
        assert _maker_manager(is_disagg=False)._make_subagent_idle_drop_tracker() is None

    def test_parent_resume_alone_builds_with_the_fallback(self, env) -> None:
        env.delenv(IDLE_S_ENV, raising=False)
        env.setenv(PARENT_RESUME_S_ENV, "60")
        tracker = _maker_manager()._make_subagent_idle_drop_tracker()
        assert isinstance(tracker, SubagentIdleDropTracker)
        assert tracker.parent_resume_s == 60.0
        assert tracker.idle_s == DEFAULT_PARENT_RESUME_FALLBACK_S

    def test_parent_resume_bad_value_raises(self, monkeypatch) -> None:
        monkeypatch.setenv(PARENT_RESUME_S_ENV, "-1")
        with pytest.raises(ValueError, match=PARENT_RESUME_S_ENV):
            _maker_manager()._make_subagent_idle_drop_tracker()

    def test_window_at_max_seq_len_is_full_attention(self, env) -> None:
        manager = _maker_manager(max_attention_window_vec=[4096, 128])
        assert manager._make_subagent_idle_drop_tracker() is not None
