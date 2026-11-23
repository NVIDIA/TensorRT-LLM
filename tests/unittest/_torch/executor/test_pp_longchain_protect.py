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
"""PP and manager glue of the long-chain protection (``TLLM_KV_LONGCHAIN_PROTECT_ISL``).

Every rank checks the host floor at each pass of the real ``PyExecutor._executor_loop_pp``, after
the request fetch and before the stage-agreement exchange; the first rank then decides, and its
operations ride that pass's schedule message. A follower applies them after receiving the message,
after the idle child-chain drop and before applying the released reservations. Stand-ins replace
the scheduler, the KV cache manager and the PP transport; the loop and the message are real. The
manager tests run the real hooks on a bare ``KVCacheManagerV2`` with a real protector on fakes.
"""

import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvcm_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BlockReusePolicy,
    KVCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.longchain_protect import (
    DECIDE_EVERY_PASSES,
    ISL_ENV,
    OP_PIN,
    OP_RELEASE,
    LongChainProtectConfig,
    LongChainProtector,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.subagent_idle_drop import (
    IDLE_S_ENV as SUBAGENT_IDLE_S_ENV,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import SerializableSchedulerOutput
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    CacheTier,
    DataRole,
    LayerId,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only

OPS = [(OP_PIN, "conv-a", 3), (OP_RELEASE, "conv-b", 1)]


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
    """The KV cache manager's long-chain and idle-drop surface."""

    def __init__(self, decided, events: List[tuple]) -> None:
        self.longchain_protect_enabled = True
        self.subagent_idle_drop_enabled = True
        self.decided = decided
        self.events = events
        self.busy: List[str] = []
        self.protected = {"protected-a"}
        self.disabled = False

    def longchain_protect_floor(self) -> int:
        self.events.append(("floor",))
        return 0

    def longchain_protect_decide(self, pass_idx, busy_fn):
        self.busy = list(busy_fn())
        self.events.append(("lc_decide", pass_idx))
        return list(self.decided) or None

    def apply_longchain_protect_ops(self, ops, pass_idx=None) -> None:
        self.events.append(("lc_apply", list(ops), pass_idx))

    def longchain_protected_keys(self):
        return set(self.protected)

    def disable_longchain_protect(self) -> None:
        self.disabled = True

    def subagent_idle_drop_decide(self, pass_idx, busy_fn):
        self.events.append(("idle_decide", sorted(busy_fn())))
        return None

    def apply_subagent_idle_drops(self, keys, pass_idx=None) -> None:
        self.events.append(("idle_apply", list(keys)))


def _req(conv, affinity=None, dummy=False):
    return SimpleNamespace(
        is_dummy_request=dummy,
        multimodal_hashes=None,
        py_subagent_affinity_id=affinity,
        py_conversation_params=None if conv is None else SimpleNamespace(conversation_id=conv),
    )


def _queued(conv, affinity=None):
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


def _executor(rank: int, manager: _Manager, events: List[tuple], idle_on=False) -> PyExecutor:
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
    exe._subagent_idle_drop_on = idle_on
    exe._longchain_protect_on = True

    def fetch():
        events.append(("fetch",))
        return []

    exe._fetch_and_activate_new_requests = fetch
    exe._handle_control_request = lambda: None
    exe._init_iter_stats_if_sampled = lambda num_new: None
    exe._pad_attention_dp_dummy_request = lambda: None
    exe._pp_stage_agree_exchange = lambda: events.append(("exchange", exe._longchain_pending_ops))
    return exe


class TestScheduleMessage:
    def test_round_trip(self) -> None:
        empty = ScheduledRequests()
        assert empty.leader_longchain_ops is None
        message = SerializableSchedulerOutput.from_scheduler_result(
            empty, [], 0, longchain_ops=list(OPS)
        )
        received = pickle.loads(pickle.dumps(message))
        batch, _, _ = received.to_scheduler_result([])
        assert batch.leader_longchain_ops == OPS

    def test_none_when_off(self) -> None:
        message = SerializableSchedulerOutput.from_scheduler_result(ScheduledRequests(), [], 0)
        assert message.longchain_ops is None
        batch, _, _ = pickle.loads(pickle.dumps(message)).to_scheduler_result([])
        assert batch.leader_longchain_ops is None


def _first_rank_loop(exe):
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
    return sent


class TestFirstRank:
    def test_floor_then_decide_after_fetch_before_exchange(self, cpu_loop) -> None:
        events: List[tuple] = []
        manager = _Manager(OPS, events)
        exe = _executor(0, manager, events)
        exe.active_requests = [_req("busy-active"), _req(None), _req("dummy", dummy=True)]
        exe.waiting_queue = [_queued("busy-queued"), _queued("child-queued", "parent")]
        sent = _first_rank_loop(exe)

        assert events == [("fetch",), ("floor",), ("lc_decide", 1), ("exchange", OPS)]
        # Every conversation counts, with or without a parent-session id.
        assert manager.busy == ["busy-active", "busy-queued", "child-queued"]
        assert len(sent) == 1 and sent[0].longchain_ops == OPS
        assert sent[0].subagent_idle_drops is None
        # Taken once: the next pass sends its own decision.
        assert exe._longchain_pending_ops is None

    def test_after_the_idle_drop_decision(self, cpu_loop) -> None:
        events: List[tuple] = []
        manager = _Manager([], events)
        exe = _executor(0, manager, events, idle_on=True)
        exe.active_requests = [_req("child", "parent"), _req("main")]
        sent = _first_rank_loop(exe)
        # The idle drop sees the protected keys as busy.
        assert events == [
            ("fetch",),
            ("idle_decide", ["child", "protected-a"]),
            ("floor",),
            ("lc_decide", 1),
            ("exchange", None),
        ]
        assert sent[0].longchain_ops is None

    def test_off_leaves_the_loop_unchanged(self, cpu_loop) -> None:
        events: List[tuple] = []
        manager = _Manager(OPS, events)
        exe = _executor(0, manager, events)
        exe._longchain_protect_on = False

        def stop():
            raise _Stop

        exe._pp_stage_agree_exchange = stop
        with pytest.raises(_Stop):
            exe._executor_loop_pp()
        assert events == [("fetch",)]
        assert exe._longchain_pass == 0
        assert exe._take_longchain_pending_ops() is None


class TestFollower:
    @pytest.mark.parametrize("ops", [OPS, None, []])
    def test_floor_then_apply_after_idle_apply_before_early_release(self, cpu_loop, ops) -> None:
        events: List[tuple] = []
        manager = _Manager([], events)
        exe = _executor(1, manager, events, idle_on=True)
        batch = ScheduledRequests()
        batch.leader_longchain_ops = ops
        batch.leader_subagent_idle_drops = ["child"]

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

        expected = [("fetch",), ("floor",), ("exchange", None), ("propagate",)]
        expected.append(("idle_apply", ["child"]))
        if ops:
            # Tagged with the follower's own pass count, the first rank's pass index.
            expected.append(("lc_apply", ops, 1))
        expected.append(("early_release",))
        assert events == expected
        # A follower never decides.
        assert not any(event[0] in ("lc_decide", "idle_decide") for event in events)


class TestBusyKeys:
    def test_idle_drop_busy_keys_unchanged_when_off(self) -> None:
        exe = object.__new__(PyExecutor)
        exe.active_requests = [_req("child", "parent"), _req("main")]
        exe.waiting_queue = [_queued("queued-child", "p"), _queued("queued-main")]
        exe.kv_cache_manager = _Manager([], [])
        assert exe._subagent_idle_busy_keys() == ["child", "queued-child"]
        exe._longchain_protect_on = True
        assert sorted(exe._subagent_idle_busy_keys()) == ["child", "protected-a", "queued-child"]
        assert exe._longchain_busy_keys() == ["child", "main", "queued-child", "queued-main"]


class TestExecutorGate:
    @pytest.fixture(autouse=True)
    def knob(self, monkeypatch):
        monkeypatch.setenv(ISL_ENV, "256000")
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
        assert exe._init_longchain_protect() is True
        assert exe.gathered == [True]
        assert not manager.disabled

    @pytest.mark.parametrize("agreement, pp_size", [(None, 2), (object(), 1)])
    def test_refuses_and_disables_without_stage_agreement(self, agreement, pp_size) -> None:
        manager = _Manager([], [])
        exe = self._bare(manager, agreement, pp_size)
        assert exe._init_longchain_protect() is False
        assert manager.disabled
        assert exe.gathered == []

    def test_another_stage_refused(self) -> None:
        manager = _Manager([], [])
        exe = self._bare(manager, object(), 4, others=(True, False, True))
        assert exe._init_longchain_protect() is False
        assert manager.disabled

    def test_manager_without_the_protector(self) -> None:
        for manager in (SimpleNamespace(), None):
            exe = self._bare(manager, object(), 2)
            # It still takes part in the exchange, so the other stages refuse too.
            assert exe._init_longchain_protect() is False
            assert exe.gathered == [False]

    def test_off_exchanges_nothing(self, knob) -> None:
        for value in (None, "0"):
            if value is None:
                knob.delenv(ISL_ENV, raising=False)
            else:
                knob.setenv(ISL_ENV, value)
            manager = _Manager([], [])
            exe = self._bare(manager, object(), 2)
            assert exe._init_longchain_protect() is False
            assert exe.gathered == []
            assert not manager.disabled


# ---- manager glue -------------------------------------------------------------------------------


class _Pin:
    def __init__(self, log, pin_id, length) -> None:
        self.log = log
        self.pin_id = pin_id
        self.num_committed_tokens = length

    def stop_committing(self) -> None:
        pass

    def plan_committed_chain_demote(self, from_block):
        return None

    def close(self) -> None:
        self.log.append(("close", self.pin_id))


def _protector(log, isl=64) -> LongChainProtector:
    return LongChainProtector(
        LongChainProtectConfig(isl=isl, pin_after_s=0.0, idle_s=1e9),
        tokens_per_block=32,
        rewind_tokens=0,
        rank=0,
        create_pin=lambda scope, tokens, pin_id: _Pin(log, pin_id, len(tokens)),
        stats_fn=lambda: [(1, 1)],
        mark_excluded=lambda pin_id: None,
        clear_excluded=lambda pin_id: None,
    )


class TestManagerGlue:
    def test_off_by_default(self) -> None:
        manager = object.__new__(KVCacheManagerV2)
        assert manager.longchain_protect_enabled is False
        assert manager.longchain_protect_decide(DECIDE_EVERY_PASSES, list) is None
        manager.apply_longchain_protect_ops(OPS)
        assert manager.longchain_protect_floor() == 0
        assert manager.longchain_protected_keys() == set()
        manager.disable_longchain_protect()

    def test_decide_apply_floor_and_disable(self) -> None:
        log: List[tuple] = []
        manager = object.__new__(KVCacheManagerV2)
        manager._longchain_protect = _protector(log)
        protector = manager._longchain_protect
        req = _req("c")
        req.context_current_position, req.context_remaining_length = 64, 0
        protector.commit(
            req,
            SimpleNamespace(num_committed_tokens=64, reuse_scope=None),
            list(range(64)),
            now=0.0,
        )
        assert manager.longchain_protected_keys() == {"c"}
        # Not a decision pass: None, not an empty list.
        assert manager.longchain_protect_decide(DECIDE_EVERY_PASSES + 1, list) is None
        assert manager.longchain_protect_decide(DECIDE_EVERY_PASSES, list) == [(OP_PIN, "c", 1)]
        assert protector.is_pinned("c")
        manager.apply_longchain_protect_ops([(OP_RELEASE, "missing", 1)], 7)
        assert protector.stats["missing"] == 1
        assert manager.longchain_protect_floor() == 0
        manager.disable_longchain_protect()
        assert log == [("close", 1 << 63)]
        assert manager.longchain_protect_enabled is False

    def _context_manager(self, events) -> KVCacheManagerV2:
        manager = object.__new__(KVCacheManagerV2)
        kv_cache = SimpleNamespace(is_active=True, enable_swa_scratch_reuse=True)
        manager.kv_cache_map = {7: kv_cache}
        manager.enable_block_reuse = True
        manager._can_publish_block_reuse = True
        manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
        manager.conversation_manager = None
        manager.try_commit_blocks = lambda req, commit_end=None: events.append("commit_blocks")
        manager._resize_context_history = lambda req, kv_cache, end: events.append("resize")
        manager._reuse_token_source = lambda req: "tokens"
        manager._swa_stale_endpoints = object()
        manager._commit_swa_stale_endpoint = lambda req, kv: events.append("stale_endpoint")
        manager._longchain_protect = SimpleNamespace(
            commit=lambda req, kv, tokens, now: events.append(("lc", tokens)),
            clear=lambda: events.append("lc_clear"),
        )
        return manager

    def test_hook_runs_before_the_stale_endpoint_commit_at_context_end(self) -> None:
        events: List = []
        manager = self._context_manager(events)
        req = _req("c")
        req.py_request_id = 7
        req.context_current_position, req.context_remaining_length = 64, 0
        batch = SimpleNamespace(context_requests=[req])
        manager.update_context_resources(batch)
        assert events == ["commit_blocks", ("lc", "tokens"), "stale_endpoint"]

        # Not at context end, or a dummy: no commit.
        events.clear()
        req.context_remaining_length = 32
        manager.update_context_resources(batch)
        req.context_remaining_length = 0
        req.is_dummy_request = True
        manager.update_context_resources(batch)
        assert ("lc", "tokens") not in events

    def test_reset_clears_pins_before_clearing_reusable_blocks(self) -> None:
        events: List = []
        manager = self._context_manager(events)
        manager.impl = SimpleNamespace(clear_reusable_blocks=lambda: events.append("clear_blocks"))
        manager._swa_stale_endpoints = None
        manager.reset_reuse_state()
        assert events == ["lc_clear", "clear_blocks"]

    def test_shutdown_clears_pins_before_the_manager_shuts_down(self) -> None:
        events: List = []
        manager = self._context_manager(events)
        manager.flush_deferred_closes = lambda: None
        manager.sparse_metadata_batch = None
        manager.kv_cache_map = {7: SimpleNamespace(close=lambda: events.append("close_request"))}
        manager._disagg_receive_ready = set()
        manager._request_stats_enabled_ids = set()
        manager._fresh_pages_filled = set()
        manager._swa_stale_endpoints = None
        manager._disagg_gen_fit_gate = None
        manager.event_manager = None
        manager.impl = SimpleNamespace(shutdown=lambda: events.append("impl_shutdown"))
        manager.shutdown()
        assert events == ["close_request", "lc_clear", "impl_shutdown"]


def _maker_manager(**overrides) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    impl = SimpleNamespace(
        chain_demote_stats=lambda: (0, 0),
        cache_tier_list=[CacheTier.GPU_MEM, CacheTier.HOST_MEM],
        get_storage_statistics=lambda level: [SimpleNamespace(available=5, total=10)],
        create_kv_cache=lambda scope, tokens, id: SimpleNamespace(id=id),
        mark_stats_excluded=lambda pin_id: None,
        clear_stats_excluded=lambda pin_id: None,
    )
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
        _swa_endpoint_rewind=512,
        kv_cache_map={},
        impl=impl,
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
        monkeypatch.setenv(ISL_ENV, "256000")
        monkeypatch.delenv(SUBAGENT_IDLE_S_ENV, raising=False)
        return monkeypatch

    def test_off_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv(ISL_ENV, raising=False)
        assert _maker_manager()._make_longchain_protector() is None
        monkeypatch.setenv(ISL_ENV, "0")
        assert _maker_manager()._make_longchain_protector() is None

    def test_bad_value_raises(self, monkeypatch) -> None:
        monkeypatch.setenv(ISL_ENV, "-5")
        with pytest.raises(ValueError, match=ISL_ENV):
            _maker_manager()._make_longchain_protector()

    def test_enabled_and_wired(self, env) -> None:
        manager = _maker_manager()
        protector = manager._make_longchain_protector()
        assert isinstance(protector, LongChainProtector)
        assert protector.config.isl == 256000
        assert protector.pin_lengths(4096) == [4096, 3584]
        assert protector._min_avail() == 0.5
        manager.kv_cache_map[1 << 63] = object()
        assert protector._new_pin_id() == (1 << 63) + 1

    def test_rewind_zero_warns_and_uses_one_pin(self, env, monkeypatch) -> None:
        warnings = []
        monkeypatch.setattr(kvcm_module.logger, "warning", warnings.append)
        protector = _maker_manager(_swa_endpoint_rewind=0)._make_longchain_protector()
        assert protector.pin_lengths(4096) == [4096]
        assert len(warnings) == 1 and "one pin" in warnings[0]

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
        assert _maker_manager(**overrides)._make_longchain_protector() is None

    @pytest.mark.parametrize(
        "tiers",
        [[CacheTier.GPU_MEM], [CacheTier.GPU_MEM, CacheTier.HOST_MEM, CacheTier.DISK]],
    )
    def test_ignored_without_a_host_last_tier(self, env, tiers) -> None:
        manager = _maker_manager()
        manager.impl.cache_tier_list = tiers
        assert manager._make_longchain_protector() is None

    def test_ignored_without_the_chain_demote_binding(self, env) -> None:
        manager = _maker_manager()
        del manager.impl.chain_demote_stats
        assert manager._make_longchain_protector() is None

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
        assert manager._make_longchain_protector() is None

    def test_ignored_without_stage_agreement(self, env) -> None:
        """An aggregated server never runs the PP stage agreement."""
        assert _maker_manager(is_disagg=False)._make_longchain_protector() is None
