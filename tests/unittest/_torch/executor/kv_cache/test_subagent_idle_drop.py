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
"""CPU tests of the opt-in idle child-chain drop tracker (TLLM_KV_SUBAGENT_IDLE_DROP_S).

The tracker runs here with fake chain handles. The C++ handle itself is covered by the KV cache
manager v2 binding tests (TestChainDemote) and the kvCacheManagerV2ChainDemoteTest gtest.
"""

import math
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import subagent_idle_drop as sid
from tensorrt_llm._torch.pyexecutor.kv_cache.subagent_idle_drop import (
    DECIDE_EVERY_PASSES,
    DEFAULT_MAX_REFS,
    IDLE_S_ENV,
    LOG_PREFIX,
    MAX_CHAINS_PER_DECISION,
    MAX_REFS_ENV,
    MAX_REFS_PER_DECISION,
    MIN_IDLE_PASSES,
    PARENT_RESUME_S_ENV,
    SubagentIdleDropTracker,
    chain_key,
    chain_parent,
    parse_idle_s,
    parse_max_refs,
    parse_parent_resume_s,
    parse_triggers,
)

TPB = 32
IDLE_S = 240.0


class FakeError(RuntimeError):
    pass


class FakeHandle:
    """Mirrors ChainDemoteHandle: absorb, demote and release consume or invalidate it."""

    def __init__(self, name: str, refs: int, log: List[Tuple[str, str]], gpu: int = 0) -> None:
        self.name = name
        self.refs = refs
        self.gpu = gpu
        self.valid = True
        self.fail: Optional[str] = None
        self._log = log

    @property
    def num_refs(self) -> int:
        return self.refs if self.valid else 0

    def absorb(self, other: "FakeHandle") -> None:
        if self.fail == "absorb":
            raise FakeError("absorb")
        if other is self or not (self.valid and other.valid):
            raise FakeError("invalid absorb")
        self.refs += other.refs
        self.gpu += other.gpu
        other.valid = False
        self._log.append(("absorb", f"{self.name}<-{other.name}"))

    def demote(self) -> Tuple[int, int, int, int, int, int]:
        if self.fail == "demote":
            raise FakeError("demote")
        if not self.valid:
            raise FakeError("invalid demote")
        self.valid = False
        self._log.append(("demote", self.name))
        return (self.refs, self.refs - 1, self.refs - 1, self.gpu, 1, 0)

    def release(self) -> None:
        if not self.valid:
            raise FakeError("invalid release")
        self.valid = False
        self._log.append(("release", self.name))


class Driver:
    """Feeds context ends to a tracker the way the manager does, with fake plans."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("errors", (FakeError,))
        self.tracker = SubagentIdleDropTracker(
            idle_s=kwargs.pop("idle_s", IDLE_S), tokens_per_block=TPB, rank=0, **kwargs
        )
        self.log: List[Tuple[str, str]] = []
        self.plans: List[Tuple[str, int]] = []
        self.handles: List[FakeHandle] = []
        self.next_refs = 4
        self.plan_error = False
        self.plan_none = False

    def commit(self, conv: str, reused: int, prompt_end: int, now: float, **req_kwargs) -> None:
        req = make_req(conv, reused=reused, prompt_end=prompt_end, **req_kwargs)

        def plan(from_block: int) -> Optional[FakeHandle]:
            self.plans.append((conv, from_block))
            if self.plan_error:
                raise FakeError("plan")
            if self.plan_none:
                return None
            handle = FakeHandle(f"{conv}#{len(self.handles)}", self.next_refs, self.log)
            self.handles.append(handle)
            return handle

        self.tracker.commit(req, plan, now)


def make_req(
    conv: Optional[str],
    reused: int = 0,
    prompt_end: int = 1024,
    affinity: Optional[str] = "parent",
    dummy: bool = False,
    multimodal: bool = False,
):
    params = None if conv is None else SimpleNamespace(conversation_id=conv)
    return SimpleNamespace(
        is_dummy_request=dummy,
        multimodal_hashes=[[1, 2]] if multimodal else None,
        py_subagent_affinity_id=affinity,
        py_conversation_params=params,
        prepopulated_prompt_len=reused,
        context_current_position=prompt_end,
        context_remaining_length=0,
    )


def idle_pass(after: int) -> int:
    """The first decision pass at least MIN_IDLE_PASSES after ``after``."""
    target = after + MIN_IDLE_PASSES
    return -(-target // DECIDE_EVERY_PASSES) * DECIDE_EVERY_PASSES


class TestParse:
    def test_idle_s(self) -> None:
        assert parse_idle_s({}) == 0.0
        assert parse_idle_s({IDLE_S_ENV: ""}) == 0.0
        assert parse_idle_s({IDLE_S_ENV: " 0 "}) == 0.0
        assert parse_idle_s({IDLE_S_ENV: "240"}) == 240.0
        assert parse_idle_s({IDLE_S_ENV: "12.5"}) == 12.5

    @pytest.mark.parametrize("raw", ["-1", "nan", "inf", "-inf", "abc", "1e400"])
    def test_idle_s_rejects(self, raw: str) -> None:
        with pytest.raises(ValueError, match=IDLE_S_ENV):
            parse_idle_s({IDLE_S_ENV: raw})

    def test_max_refs(self) -> None:
        assert parse_max_refs({}) == DEFAULT_MAX_REFS == 4194304
        assert parse_max_refs({MAX_REFS_ENV: "5"}) == 5
        for raw in ("0", "-3", "x", "1.5"):
            with pytest.raises(ValueError, match=MAX_REFS_ENV):
                parse_max_refs({MAX_REFS_ENV: raw})

    def test_tracker_rejects_bad_geometry(self) -> None:
        for kwargs in (
            dict(idle_s=0.0, tokens_per_block=TPB),
            dict(idle_s=math.nan, tokens_per_block=TPB),
            dict(idle_s=IDLE_S, tokens_per_block=0),
            dict(idle_s=IDLE_S, tokens_per_block=TPB, max_refs=0),
        ):
            with pytest.raises(ValueError):
                SubagentIdleDropTracker(rank=0, **kwargs)


class TestKey:
    def test_child_key_is_its_conversation_id(self) -> None:
        assert chain_key(make_req("child-1")) == "child-1"
        assert chain_key(make_req("  child-1 ")) == "child-1"
        # The id is never parsed: a subagent-looking suffix is just part of the key.
        assert chain_key(make_req("root::sa:3")) == "root::sa:3"
        assert SubagentIdleDropTracker.key(make_req("c")) == "c"

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(affinity=None),
            dict(affinity=""),
            dict(affinity="   "),
            dict(dummy=True),
            dict(multimodal=True),
        ],
    )
    def test_not_a_child_chain(self, kwargs) -> None:
        assert chain_key(make_req("c", **kwargs)) is None

    def test_missing_conversation_id(self) -> None:
        assert chain_key(make_req(None)) is None
        assert chain_key(make_req("   ")) is None
        req = make_req("c")
        req.py_conversation_params = SimpleNamespace()
        assert chain_key(req) is None

    def test_untracked_requests_never_plan(self) -> None:
        d = Driver()
        d.commit("c", 0, 100, 0.0, affinity=None)
        d.commit("c", 0, 100, 0.0, dummy=True)
        assert d.plans == []
        assert d.tracker.stats["plans"] == 0


class TestCommit:
    def test_anchor_persists_and_bounds_from_block(self) -> None:
        d = Driver()
        d.commit("c", reused=100, prompt_end=1000, now=0.0)
        assert d.tracker.anchor("c") == 100 // TPB
        assert d.plans[-1] == ("c", 3)
        d.commit("c", reused=600, prompt_end=2000, now=1.0)
        assert d.plans[-1] == ("c", 600 // TPB)
        # The next turn lost its prefix to eviction: the plan still starts at the anchor.
        d.tracker.apply(["c"])
        d.commit("c", reused=40, prompt_end=2500, now=2.0)
        assert d.tracker.anchor("c") == 3
        assert d.plans[-1] == ("c", 3)

    def test_turns_absorb_into_one_handle(self) -> None:
        d = Driver()
        d.commit("c", 0, 100, 0.0)
        d.next_refs = 6
        d.commit("c", 96, 300, 1.0)
        assert ("absorb", "c#0<-c#1") in d.log
        assert not d.handles[1].valid
        assert d.tracker.num_chains == 1
        assert d.tracker.total_refs == 10
        d.tracker.apply(["c"])
        assert ("demote", "c#0") in d.log
        assert d.tracker.total_refs == 0

    def test_plan_none_and_errors_are_counted(self) -> None:
        d = Driver()
        d.plan_none = True
        d.commit("a", 0, 100, 0.0)
        assert d.tracker.stats["plan_none"] == 1
        assert d.tracker.num_chains == 0
        d.plan_none = False
        d.plan_error = True
        d.commit("a", 0, 100, 0.0)
        assert d.tracker.stats["plan_errors"] == 1
        d.plan_error = False
        d.commit("a", 0, 100, 0.0)
        d.tracker._chains["a"].handle.fail = "absorb"
        d.commit("a", 96, 200, 1.0)
        assert d.tracker.stats["plan_errors"] == 2
        # The handle that could not be absorbed is released, never demoted.
        assert ("release", "a#1") in d.log
        assert d.tracker.stats["plans"] == 4

    def test_errors_outside_the_tuple_propagate(self) -> None:
        d = Driver()

        def plan(_: int):
            raise ValueError("not a KV cache error")

        with pytest.raises(ValueError):
            d.tracker.commit(make_req("c"), plan, 0.0)

    def test_ref_cap_releases_oldest_and_never_demotes(self) -> None:
        d = Driver(max_refs=10)
        d.commit("a", 0, 100, 0.0)
        d.commit("b", 0, 100, 1.0)
        assert d.tracker.stats["capped"] == 0
        d.commit("c", 0, 100, 2.0)
        assert d.tracker.stats["capped"] == 1
        assert not d.tracker.is_tracked("a")
        assert ("release", "a#0") in d.log
        assert not any(kind == "demote" for kind, _ in d.log)
        assert d.tracker.total_refs == 8

    def test_chain_cap(self, monkeypatch) -> None:
        monkeypatch.setattr(sid, "MAX_CHAINS", 2)
        d = Driver()
        for i, conv in enumerate("abc"):
            d.commit(conv, 0, 100, float(i))
        assert d.tracker.num_chains == 2
        assert d.tracker.stats["capped"] == 1
        assert ("release", "a#0") in d.log

    def test_returning_chain_replans_from_its_anchor(self) -> None:
        # The first turn reuses 320 tokens (anchor 10) and commits 3200 (blocks 10-99).
        d = Driver()
        d.commit("c", reused=320, prompt_end=3200, now=0.0)
        assert d.plans == [("c", 10)]
        d.tracker.apply(["c"])
        # It returns before eviction and rescues blocks 10-99: the plan covers them again.
        d.commit("c", reused=3200, prompt_end=3500, now=310.0)
        assert d.plans[-1] == ("c", 10)
        assert d.tracker.is_tracked("c")
        # While tracked, a turn plans only past its reuse.
        d.commit("c", reused=3500, prompt_end=3900, now=320.0)
        assert d.plans[-1] == ("c", 3500 // TPB)
        assert d.tracker.stats["absorbs"] == 1
        # The next idle period demotes the whole chain through one handle.
        d.tracker.apply(["c"])
        assert [name for kind, name in d.log if kind == "demote"] == ["c#0", "c#1"]

    def test_capped_or_cleared_chain_replans_from_its_anchor(self) -> None:
        d = Driver(max_refs=8)
        d.commit("a", reused=64, prompt_end=640, now=0.0)
        d.commit("b", reused=0, prompt_end=640, now=1.0)
        d.commit("c", reused=0, prompt_end=640, now=2.0)
        assert not d.tracker.is_tracked("a")
        d.commit("a", reused=640, prompt_end=700, now=3.0)
        assert d.plans[-1] == ("a", 2)
        d.tracker.clear()
        d.commit("b", reused=640, prompt_end=700, now=4.0)
        assert d.plans[-1] == ("b", 0)

    def test_absorbs_and_anchor0_counters(self) -> None:
        d = Driver()
        d.commit("a", reused=0, prompt_end=100, now=0.0)
        d.commit("b", reused=64, prompt_end=100, now=0.0)
        assert d.tracker.stats["anchor0"] == 1
        d.commit("a", reused=96, prompt_end=200, now=1.0)
        d.commit("b", reused=96, prompt_end=200, now=1.0)
        assert d.tracker.stats["absorbs"] == 2
        # The anchor is counted once per chain, also after a demotion.
        d.tracker.apply(["a"])
        d.commit("a", reused=0, prompt_end=300, now=2.0)
        assert d.tracker.stats["anchor0"] == 1

    def test_returned_and_lost_tokens(self) -> None:
        d = Driver()
        d.commit("c", 0, 1000, 0.0)
        d.tracker.apply(["c"])
        d.commit("c", reused=300, prompt_end=1500, now=500.0)
        assert d.tracker.stats["returned"] == 1
        assert d.tracker.stats["returned_lost_tokens"] == 700
        # The tombstone is consumed.
        d.commit("c", reused=1500, prompt_end=1600, now=501.0)
        assert d.tracker.stats["returned"] == 1


class TestDecide:
    def test_seconds_threshold_idle_passes_and_cadence(self) -> None:
        d = Driver()
        d.tracker.decide(0, 0.0, list)
        d.commit("c", 0, 100, now=0.0)
        busy = list
        # Not a decision pass.
        assert d.tracker.decide(DECIDE_EVERY_PASSES * 4 + 1, 1000.0, busy) == []
        # A decision pass, enough seconds, but not enough passes since the commit.
        assert d.tracker.decide(DECIDE_EVERY_PASSES, 1000.0, busy) == []
        # Enough passes, not enough seconds.
        assert d.tracker.decide(idle_pass(0), IDLE_S - 1.0, busy) == []
        assert d.tracker.is_tracked("c")
        # Both.
        assert d.tracker.decide(idle_pass(0), IDLE_S, busy) == ["c"]
        assert ("demote", "c#0") in d.log
        assert not d.tracker.is_tracked("c")
        assert d.tracker.stats["drops"] == 1

    def test_commit_stamps_the_current_pass(self) -> None:
        d = Driver()
        d.tracker.decide(160, 0.0, list)
        d.commit("c", 0, 100, now=0.0)
        assert d.tracker.decide(160 + MIN_IDLE_PASSES - DECIDE_EVERY_PASSES, 1e6, list) == []
        assert d.tracker.decide(idle_pass(160), 1e6, list) == ["c"]

    def test_busy_keys_are_touched(self) -> None:
        d = Driver()
        d.commit("a", 0, 100, 0.0)
        d.commit("b", 0, 100, 1.0)
        p = idle_pass(0)
        assert d.tracker.decide(p, 1000.0, lambda: ["a", "unknown"]) == ["b"]
        # "a" was touched at pass p, time 1000: idle again only later.
        assert d.tracker.decide(p + DECIDE_EVERY_PASSES, 2000.0, list) == []
        assert d.tracker.decide(idle_pass(p), 1000.0 + IDLE_S, list) == ["a"]

    def test_busy_touch_keeps_commit_order_on_every_stage(self) -> None:
        # Rank 0 touches "a" as busy, the follower cannot: both then cap "a" first.
        leader, follower = Driver(max_refs=8), Driver(max_refs=8)
        for d in (leader, follower):
            d.commit("a", 0, 100, 0.0)
            d.commit("b", 0, 100, 1.0)
        assert leader.tracker.decide(idle_pass(0), 1000.0, lambda: ["a"]) == ["b"]
        follower.tracker.apply(["b"])
        for d in (leader, follower):
            d.commit("c", 0, 100, 1001.0)
            d.commit("e", 0, 100, 1002.0)
            assert not d.tracker.is_tracked("a")
            assert d.tracker.stats["capped"] == 1
        assert follower.tracker.stats["missing"] == 0

    def test_busy_chain_is_skipped_not_a_stop(self) -> None:
        d = Driver()
        d.commit("a", 0, 100, 0.0)
        d.commit("b", 0, 100, 1.0)
        d.commit("c", 0, 100, 2.0)
        p = idle_pass(0)
        assert d.tracker.decide(p, 1000.0, lambda: ["a", "b"]) == ["c"]
        assert list(d.tracker._chains) == ["a", "b"]

    def test_fifo_and_chain_cap_per_decision(self) -> None:
        d = Driver()
        convs = [f"c{i}" for i in range(MAX_CHAINS_PER_DECISION + 2)]
        for i, conv in enumerate(convs):
            d.commit(conv, 0, 100, float(i))
        p = idle_pass(0)
        assert d.tracker.decide(p, 1e6, list) == convs[:MAX_CHAINS_PER_DECISION]
        assert (
            d.tracker.decide(p + DECIDE_EVERY_PASSES, 1e6, list) == convs[MAX_CHAINS_PER_DECISION:]
        )
        assert [name for kind, name in d.log if kind == "demote"] == [
            f"{c}#{i}" for i, c in enumerate(convs)
        ]

    def test_stops_at_the_first_recent_chain(self) -> None:
        d = Driver()
        d.commit("old", 0, 100, 0.0)
        d.commit("new", 0, 100, 500.0)
        d.commit("recent", 0, 100, 600.0)
        assert d.tracker.decide(idle_pass(0), 600.0 + IDLE_S - 1, list) == ["old", "new"]

    def test_ref_cap_per_decision_keeps_at_least_one(self) -> None:
        d = Driver()
        d.next_refs = MAX_REFS_PER_DECISION + 1
        d.commit("big", 0, 100, 0.0)
        d.next_refs = MAX_REFS_PER_DECISION // 2 + 1
        d.commit("half1", 0, 100, 1.0)
        d.commit("half2", 0, 100, 2.0)
        p = idle_pass(0)
        assert d.tracker.decide(p, 1e6, list) == ["big"]
        assert d.tracker.decide(p + DECIDE_EVERY_PASSES, 1e6, list) == ["half1"]
        assert d.tracker.decide(p + 2 * DECIDE_EVERY_PASSES, 1e6, list) == ["half2"]

    def test_empty_tracker_does_not_call_busy_fn(self) -> None:
        d = Driver()

        def busy():
            raise AssertionError("busy_fn called without chains")

        assert d.tracker.decide(DECIDE_EVERY_PASSES, 1e6, busy) == []


class TestApply:
    def test_follower_counts_missing_keys(self) -> None:
        d = Driver()
        d.commit("c", 0, 100, 0.0)
        d.tracker.apply(["nope", "c"], pass_idx=48)
        assert d.tracker.stats["missing"] == 1
        assert d.tracker.stats["drops"] == 1
        assert "pass=48" in d.tracker.format_stats()

    def test_result_fields_are_summed(self) -> None:
        d = Driver()
        d.next_refs = 5
        d.commit("a", 0, 100, 0.0)
        d.commit("b", 0, 100, 0.0)
        d.handles[0].gpu = 2
        d.tracker.apply(["a", "b"])
        s = d.tracker.stats
        assert (s["plan_blocks"], s["blocks"], s["pages"], s["gpu_pages"]) == (10, 8, 8, 2)
        assert (s["kept_shared"], s["kept_held"], s["drops"]) == (2, 0, 2)

    def test_demote_error_is_counted_and_released(self) -> None:
        d = Driver()
        d.commit("c", 0, 100, 0.0)
        d.handles[0].fail = "demote"
        d.tracker.apply(["c"])
        assert d.tracker.stats["drop_errors"] == 1
        assert d.tracker.stats["drops"] == 0
        assert ("release", "c#0") in d.log
        assert not d.tracker.is_tracked("c")


class TestLogAndClear:
    @pytest.fixture
    def lines(self, monkeypatch) -> List[str]:
        captured: List[str] = []
        fake = SimpleNamespace(info=captured.append, warning=captured.append)
        monkeypatch.setattr(sid, "logger", fake)
        return captured

    def test_log_line_fields(self, lines: List[str]) -> None:
        d = Driver(stats_fn=lambda: (7, 3))
        d.commit("c", 0, 100, 0.0)
        d.tracker.log_stats(force=True)
        line = lines[-1]
        assert line.startswith(f"{LOG_PREFIX} rank=0 pass=0 ")
        fields = [token.split("=")[0] for token in line[len(LOG_PREFIX) + 1 :].split()]
        assert fields == [
            "rank",
            "pass",
            "chains",
            "refs",
            "plans",
            "plan_none",
            "drops",
            "blocks",
            "pages",
            "gpu_pages",
            "kept_shared",
            "kept_held",
            "missing",
            "capped",
            "errors",
            "plan_errors",
            "drop_errors",
            "evicted_pages",
            "rescued_pages",
            "returned",
            "returned_lost_tokens",
            "plan_blocks",
            "absorbs",
            "anchor0",
            "rss_mb",
        ]
        assert "evicted_pages=7 rescued_pages=3" in line
        assert "chains=1 refs=4 plans=1" in line
        assert "plan_blocks=0 absorbs=0 anchor0=1 " in line
        assert int(line.rsplit("rss_mb=", 1)[1]) >= -1

    def test_errors_sum_and_stats_fn_failure(self, lines: List[str]) -> None:
        def broken():
            raise FakeError("stats")

        d = Driver(stats_fn=broken)
        d.plan_error = True
        d.commit("a", 0, 100, 0.0)
        d.plan_error = False
        d.commit("b", 0, 100, 0.0)
        d.handles[0].fail = "demote"
        d.tracker.apply(["b"])
        line = d.tracker.format_stats()
        assert "errors=2 plan_errors=1 drop_errors=1" in line
        assert "evicted_pages=-1 rescued_pages=-1" in line
        # One warning per kind.
        assert sum(LOG_PREFIX in entry and "failed" in entry for entry in lines) == 2

    def test_periodic_logging(self, lines: List[str], monkeypatch) -> None:
        monkeypatch.setattr(sid, "LOG_EVERY_DROPS", 2)
        d = Driver()
        for conv in "abc":
            d.commit(conv, 0, 100, 0.0)
        assert lines == []
        d.tracker.apply(["a"])
        assert lines == []
        d.tracker.apply(["b"])
        assert len(lines) == 1 and "drops=2" in lines[0]

    def test_clear_releases_every_handle(self, lines: List[str]) -> None:
        d = Driver()
        d.commit("a", 0, 100, 0.0)
        d.commit("b", 0, 100, 0.0)
        d.tracker.apply(["a"])
        d.tracker.clear()
        assert ("release", "b#1") in d.log
        assert not any(kind == "demote" and name == "b#1" for kind, name in d.log)
        assert d.tracker.num_chains == 0 and d.tracker.total_refs == 0
        assert len(lines) == 1 and "chains=1" in lines[0]
        # Anchors survive a clear; tombstones do not.
        assert d.tracker.anchor("a") == 0
        d.commit("a", 0, 100, 1.0)
        assert d.tracker.stats["returned"] == 0


@pytest.mark.parametrize(
    "raw, resume_raw, tracked",
    [
        (None, None, False),
        ("0", None, False),
        ("0", "0", False),
        ("30", None, True),
        (None, "60", True),
        ("0", "60", True),
    ],
)
def test_request_conversion_tracks_the_affinity_id_only_with_the_knob(
    monkeypatch, raw, resume_raw, tracked
):
    """Without either trigger a converted request carries no parent-session id
    attribute, as before; with the idle trigger or the parent-resume trigger
    (which alone keeps the tracker on) the id is copied from the scheduling params."""
    from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
    from tensorrt_llm.bindings import executor as trtllm
    from tensorrt_llm.scheduling_params import SchedulingParams

    for env, value in ((sid.IDLE_S_ENV, raw), (sid.PARENT_RESUME_S_ENV, resume_raw)):
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, value)
    assert sid.subagent_idle_drop_requested() is tracked
    executor_request = trtllm.Request(input_token_ids=[1, 2, 3, 4], max_tokens=8)
    executor_request.py_scheduling_params = SchedulingParams(subagent_affinity_id="parent")
    req = executor_request_to_llm_request(
        7, executor_request, child_req_ids=[], exclude_last_generation_logits=False
    )
    if tracked:
        assert req.py_subagent_affinity_id == "parent"
    else:
        assert not hasattr(req, "py_subagent_affinity_id")


GRACE = 60.0
FALLBACK = sid.DEFAULT_PARENT_RESUME_FALLBACK_S


def parent_driver(**kwargs) -> Driver:
    kwargs.setdefault("idle_s", FALLBACK)
    kwargs.setdefault("parent_resume_s", GRACE)
    return Driver(**kwargs)


class TestParentResumeParse:
    def test_parent_resume_s(self) -> None:
        assert parse_parent_resume_s({}) == 0.0
        assert parse_parent_resume_s({PARENT_RESUME_S_ENV: " "}) == 0.0
        assert parse_parent_resume_s({PARENT_RESUME_S_ENV: "60"}) == 60.0
        for raw in ("-1", "nan", "inf", "x"):
            with pytest.raises(ValueError, match=PARENT_RESUME_S_ENV):
                parse_parent_resume_s({PARENT_RESUME_S_ENV: raw})

    def test_triggers(self) -> None:
        # Off by default; the idle trigger alone is unchanged.
        assert parse_triggers({}) == (0.0, 0.0)
        assert parse_triggers({IDLE_S_ENV: "240"}) == (240.0, 0.0)
        # The parent-resume trigger alone keeps an idle fallback, so tracking stays bounded.
        assert parse_triggers({PARENT_RESUME_S_ENV: "60"}) == (FALLBACK, 60.0)
        assert parse_triggers({PARENT_RESUME_S_ENV: "60", IDLE_S_ENV: "240"}) == (240.0, 60.0)
        assert parse_triggers({PARENT_RESUME_S_ENV: "0", IDLE_S_ENV: "0"}) == (0.0, 0.0)

    def test_requested_by_either_trigger(self) -> None:
        # The request conversion copies the parent-session id exactly when this is True,
        # so the parent-resume trigger alone must turn it on.
        assert not sid.subagent_idle_drop_requested({})
        assert not sid.subagent_idle_drop_requested({IDLE_S_ENV: "0", PARENT_RESUME_S_ENV: "0"})
        assert sid.subagent_idle_drop_requested({IDLE_S_ENV: "240"})
        assert sid.subagent_idle_drop_requested({PARENT_RESUME_S_ENV: "60"})
        with pytest.raises(ValueError, match=PARENT_RESUME_S_ENV):
            sid.subagent_idle_drop_requested({PARENT_RESUME_S_ENV: "-1"})

    def test_tracker_rejects_bad_grace(self) -> None:
        for grace in (-1.0, math.nan, math.inf):
            with pytest.raises(ValueError):
                SubagentIdleDropTracker(
                    idle_s=IDLE_S, tokens_per_block=TPB, rank=0, parent_resume_s=grace
                )

    def test_chain_parent(self) -> None:
        assert chain_parent(make_req("c", affinity=" P ")) == "P"
        assert chain_parent(make_req("c", affinity=None)) is None
        assert chain_parent(make_req("c", affinity="  ")) is None


class TestParentResume:
    def test_parent_request_then_grace_demotes(self) -> None:
        d = parent_driver()
        d.commit("c", 0, 100, now=0.0, affinity="P")
        p = idle_pass(0)
        # The parent's next request is seen: the child is marked, but has idled only 10 s.
        assert d.tracker.decide(p, 10.0, list, lambda: ["P", "other"]) == []
        assert d.tracker.is_parent_resumed("c")
        assert d.tracker.stats["parent_marks"] == 1
        # The parent request has left; the mark stays and the grace has passed.
        assert d.tracker.decide(p + DECIDE_EVERY_PASSES, GRACE - 1.0, list, list) == []
        assert d.tracker.decide(p + 2 * DECIDE_EVERY_PASSES, GRACE, list, list) == ["c"]
        assert ("demote", "c#0") in d.log
        assert d.tracker.stats["parent_drops"] == 1
        assert not d.tracker.is_tracked("c")
        assert d.tracker._children == {} and len(d.tracker._resumed) == 0

    def test_without_parent_only_the_fallback_demotes(self) -> None:
        d = parent_driver()
        d.commit("c", 0, 100, now=0.0, affinity="P")
        p = idle_pass(0)
        assert d.tracker.decide(p, FALLBACK - 1.0, list, lambda: ["Q"]) == []
        assert d.tracker.decide(p + DECIDE_EVERY_PASSES, FALLBACK, list, list) == ["c"]
        assert d.tracker.stats["parent_drops"] == 0
        assert d.tracker.stats["drops"] == 1

    def test_parent_request_in_flight_at_commit_does_not_mark(self) -> None:
        # A child running in the background: its parent's request is already queued.
        d = parent_driver()
        assert d.tracker.decide(DECIDE_EVERY_PASSES, 0.0, list, lambda: ["P"]) == []
        d.tracker.decide(DECIDE_EVERY_PASSES + 4, 0.0, list, lambda: ["P"])
        d.commit("c", 0, 100, now=0.0, affinity="P")
        p = 2 * DECIDE_EVERY_PASSES
        assert d.tracker.decide(p, 100.0, list, lambda: ["P"]) == []
        assert not d.tracker.is_parent_resumed("c")
        # The parent request leaves; its next one is a resume.
        assert d.tracker.decide(p + DECIDE_EVERY_PASSES, 110.0, list, list) == []
        q = idle_pass(DECIDE_EVERY_PASSES + 4)
        assert d.tracker.decide(q, 120.0, list, lambda: ["P"]) == ["c"]
        assert d.tracker.stats["parent_drops"] == 1

    def test_new_commit_clears_the_mark(self) -> None:
        d = parent_driver()
        d.commit("c", 0, 100, now=0.0, affinity="P")
        p = idle_pass(0)
        d.tracker.decide(p, 10.0, list, lambda: ["P"])
        assert d.tracker.is_parent_resumed("c")
        # The child returns before the grace ends.
        d.commit("c", 96, 300, now=20.0, affinity="P")
        assert not d.tracker.is_parent_resumed("c")
        q = idle_pass(p)
        assert d.tracker.decide(q, 500.0, list, list) == []
        assert d.tracker.is_tracked("c")
        # The parent comes back again: a new mark, demoted after the grace.
        assert d.tracker.decide(q + DECIDE_EVERY_PASSES, 500.0, list, lambda: ["P"]) == ["c"]

    def test_busy_child_waits_for_the_grace(self) -> None:
        d = parent_driver()
        d.commit("c", 0, 100, now=0.0, affinity="P")
        p = idle_pass(0)
        assert d.tracker.decide(p, 100.0, lambda: ["c"], lambda: ["P"]) == []
        assert d.tracker.is_parent_resumed("c")
        # Touched busy at 100 s and pass p: idle again only after the grace and the passes.
        assert d.tracker.decide(p + DECIDE_EVERY_PASSES, 100.0 + GRACE, list, list) == []
        assert d.tracker.decide(idle_pass(p), 100.0 + GRACE, list, list) == ["c"]

    def test_siblings_and_other_parents(self) -> None:
        d = parent_driver()
        d.commit("a", 0, 100, now=0.0, affinity="P")
        d.commit("b", 0, 100, now=0.0, affinity="P")
        d.commit("x", 0, 100, now=0.0, affinity="Q")
        p = idle_pass(0)
        assert d.tracker.decide(p, GRACE, list, lambda: ["P"]) == ["a", "b"]
        assert d.tracker.is_tracked("x") and not d.tracker.is_parent_resumed("x")
        assert set(d.tracker._children) == {"Q"}

    def test_followers_apply_the_same_keys(self) -> None:
        leader, follower = parent_driver(max_refs=12), parent_driver(max_refs=12)
        for d in (leader, follower):
            d.commit("a", 0, 100, now=0.0, affinity="P")
            d.commit("b", 0, 100, now=1.0, affinity="Q")
        assert leader.tracker.decide(idle_pass(0), GRACE + 1.0, list, lambda: ["Q"]) == ["b"]
        follower.tracker.apply(["b"])
        for d in (leader, follower):
            d.commit("c", 0, 100, now=2.0, affinity="P")
            d.commit("e", 0, 100, now=3.0, affinity="P")
            d.commit("f", 0, 100, now=4.0, affinity="P")
            # Same chains on both stages: the cap releases "a" on both.
            assert list(d.tracker._chains) == ["c", "e", "f"]
            assert "a" not in d.tracker._children.get("P", {})
        assert follower.tracker.stats["missing"] == 0

    def test_per_decision_cap(self) -> None:
        d = parent_driver()
        convs = [f"c{i}" for i in range(MAX_CHAINS_PER_DECISION + 1)]
        for conv in convs:
            d.commit(conv, 0, 100, now=0.0, affinity="P")
        p = idle_pass(0)
        assert d.tracker.decide(p, GRACE, list, lambda: ["P"]) == convs[:MAX_CHAINS_PER_DECISION]
        assert d.tracker.decide(p + DECIDE_EVERY_PASSES, GRACE, list, list) == convs[-1:]

    def test_off_ignores_parent_ids(self) -> None:
        d = Driver()
        d.commit("c", 0, 100, now=0.0, affinity="P")
        assert not d.tracker.parent_resume_on
        assert d.tracker.decide(idle_pass(0), IDLE_S - 1.0, list, lambda: ["P"]) == []
        assert d.tracker.stats["parent_marks"] == 0
        assert d.tracker._children == {}
        assert "parent_resume_s" not in d.tracker.format_stats()

    def test_stats_line(self) -> None:
        d = parent_driver()
        d.commit("c", 0, 100, now=0.0, affinity="P")
        d.tracker.decide(idle_pass(0), 10.0, list, lambda: ["P"])
        line = d.tracker.format_stats()
        assert line.endswith("parent_resume_s=60 parent_marks=1 parent_drops=0 resumed=1")

    def test_clear_drops_the_indexes(self) -> None:
        d = parent_driver()
        d.commit("c", 0, 100, now=0.0, affinity="P")
        d.tracker.decide(idle_pass(0), 10.0, list, lambda: ["P"])
        d.tracker.clear()
        assert d.tracker._children == {} and len(d.tracker._resumed) == 0
        assert d.tracker._parent_seen == {}
