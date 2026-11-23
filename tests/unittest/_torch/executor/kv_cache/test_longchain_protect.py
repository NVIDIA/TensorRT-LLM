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
"""CPU tests of the opt-in long-chain protector (TLLM_KV_LONGCHAIN_PROTECT_ISL).

The protector runs here with fake pins, fake demote handles and fake storage statistics. The pin
semantics of the real KV cache are covered by the KV cache manager v2 binding tests
(TestLongChainPin).
"""

from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import longchain_protect as lcp
from tensorrt_llm._torch.pyexecutor.kv_cache.longchain_protect import (
    DECIDE_EVERY_PASSES,
    FLOOR_ENV,
    IDLE_S_ENV,
    ISL_ENV,
    LOG_PREFIX,
    MAX_FLOOR_RELEASES,
    MAX_PINS_PER_DECISION,
    MAX_RELEASES_PER_DECISION,
    MIN_HOST_AVAIL_ENV,
    MIN_IDLE_PASSES,
    OP_DROP,
    OP_PIN,
    OP_RELEASE,
    PIN_AFTER_S_ENV,
    PIN_ID_BASE,
    LongChainProtectConfig,
    LongChainProtector,
    conversation_key,
    copy_tokens,
    parse_config,
)

TPB = 32
REWIND = 512
THR = 1024
SCOPE = ("lora", "salt")


class FakeError(RuntimeError):
    pass


class FakeHandle:
    def __init__(self, pin: "FakePin", log: List[Tuple[str, Any]]) -> None:
        self.pin = pin
        self.log = log
        self.fail = False
        self.valid = True

    def demote(self) -> Tuple[int, int, int, int, int, int]:
        if self.fail:
            raise FakeError("demote")
        self.valid = False
        blocks = self.pin.matched // TPB
        self.log.append(("demote", self.pin.pin_id))
        return (blocks, blocks - 1, blocks - 1, 0, 1, 0)

    def release(self) -> None:
        self.valid = False
        self.log.append(("handle_release", self.pin.pin_id))


class FakePin:
    def __init__(self, pin_id: int, tokens: Sequence[int], matched: int, log) -> None:
        self.pin_id = pin_id
        self.tokens = list(tokens)
        self.matched = matched
        self.log = log
        self.stopped = False
        self.closed = False
        self.fail_plan = False
        self.fail_close = False
        self.fail_stop = False
        self.handle: Optional[FakeHandle] = None

    @property
    def num_committed_tokens(self) -> int:
        return self.matched

    def stop_committing(self) -> None:
        if self.fail_stop:
            raise FakeError("stop")
        self.stopped = True

    def plan_committed_chain_demote(self, from_block: int) -> Optional[FakeHandle]:
        assert from_block == 0
        assert self.stopped and not self.closed
        if self.fail_plan:
            raise FakeError("plan")
        self.log.append(("plan", self.pin_id))
        self.handle = FakeHandle(self, self.log)
        return self.handle

    def close(self) -> None:
        if self.fail_close:
            raise FakeError("close")
        assert not self.closed
        self.closed = True
        self.log.append(("close", self.pin_id))


class Binding:
    """Fake KV cache manager surface: pin creation, stats exclusion and last-tier statistics."""

    def __init__(self) -> None:
        self.log: List[Tuple[str, Any]] = []
        self.pins: List[FakePin] = []
        self.excluded: set = set()
        self.avail: List[Tuple[int, int]] = [(900, 1000), (90, 100)]
        self.match_short = 0
        self.fail_create = False
        self.fail_stats = False
        self.in_use: set = set()

    def create_pin(self, scope, tokens, pin_id) -> FakePin:
        assert scope == SCOPE
        assert pin_id in self.excluded, "stats exclusion must precede the create"
        if self.fail_create:
            raise FakeError("create")
        pin = FakePin(pin_id, tokens, max(0, len(tokens) - self.match_short), self.log)
        self.log.append(("create", pin_id, len(tokens)))
        self.pins.append(pin)
        return pin

    def stats(self):
        if self.fail_stats:
            raise FakeError("stats")
        self.log.append(("stats",))
        return list(self.avail)

    def mark(self, pin_id: int) -> None:
        self.excluded.add(pin_id)

    def clear(self, pin_id: int) -> None:
        self.excluded.discard(pin_id)

    def open_pins(self) -> List[FakePin]:
        return [pin for pin in self.pins if not pin.closed]


def make_req(conv: Optional[str], prompt_end: int, **kwargs):
    return SimpleNamespace(
        is_dummy_request=kwargs.get("dummy", False),
        multimodal_hashes=kwargs.get("mm"),
        py_conversation_params=(None if conv is None else SimpleNamespace(conversation_id=conv)),
        context_current_position=prompt_end,
        context_remaining_length=0,
    )


class Driver:
    def __init__(self, rewind: int = REWIND, rank: int = 0, **config) -> None:
        config.setdefault("isl", THR)
        self.binding = Binding()
        self.clock = [0.0]

        def clock() -> float:
            self.clock[0] += 0.001
            return self.clock[0]

        self.protector = LongChainProtector(
            LongChainProtectConfig(**config),
            tokens_per_block=TPB,
            rewind_tokens=rewind,
            rank=rank,
            create_pin=self.binding.create_pin,
            stats_fn=self.binding.stats,
            mark_excluded=self.binding.mark,
            clear_excluded=self.binding.clear,
            id_in_use=lambda pin_id: pin_id in self.binding.in_use,
            errors=(FakeError,),
            clock=clock,
        )

    def commit(self, conv: Optional[str], prompt_end: int, now: float, committed=None, **kw):
        req = make_req(conv, prompt_end, **kw)
        committed = prompt_end if committed is None else committed
        kv_cache = SimpleNamespace(num_committed_tokens=committed, reuse_scope=SCOPE)
        tokens = list(range(prompt_end))
        self.protector.commit(req, kv_cache, tokens, now)

    def decide(self, pass_idx: int, now: float, busy=()) -> List[Tuple[str, str, int]]:
        return self.protector.decide(pass_idx, now, lambda: list(busy))


def p(k: int) -> int:
    """The k-th decision pass, far enough apart that MIN_IDLE_PASSES never binds by itself."""
    return k * DECIDE_EVERY_PASSES * (MIN_IDLE_PASSES // DECIDE_EVERY_PASSES)


# ---- configuration ------------------------------------------------------------------------------


class TestParseConfig:
    def test_off(self) -> None:
        assert parse_config({}) is None
        assert parse_config({ISL_ENV: ""}) is None
        assert parse_config({ISL_ENV: "  "}) is None
        assert parse_config({ISL_ENV: "0"}) is None

    def test_defaults(self) -> None:
        config = parse_config({ISL_ENV: "256000"})
        assert config == LongChainProtectConfig(
            isl=256000, pin_after_s=180.0, idle_s=400.0, min_host_avail=0.20, floor=0.05
        )

    def test_overrides(self) -> None:
        config = parse_config(
            {
                ISL_ENV: "1000",
                PIN_AFTER_S_ENV: "60",
                IDLE_S_ENV: "120",
                MIN_HOST_AVAIL_ENV: "0.5",
                FLOOR_ENV: "0.1",
            }
        )
        assert (config.isl, config.pin_after_s, config.idle_s) == (1000, 60.0, 120.0)
        assert (config.min_host_avail, config.floor) == (0.5, 0.1)

    @pytest.mark.parametrize(
        "env",
        [
            {ISL_ENV: "-1"},
            {ISL_ENV: "1.5"},
            {ISL_ENV: "abc"},
            {ISL_ENV: "100", PIN_AFTER_S_ENV: "x"},
            {ISL_ENV: "100", PIN_AFTER_S_ENV: "-1"},
            {ISL_ENV: "100", PIN_AFTER_S_ENV: "inf"},
            {ISL_ENV: "100", IDLE_S_ENV: "180"},
            {ISL_ENV: "100", IDLE_S_ENV: "nan"},
            {ISL_ENV: "100", MIN_HOST_AVAIL_ENV: "1.5"},
            {ISL_ENV: "100", FLOOR_ENV: "-0.1"},
            {ISL_ENV: "100", FLOOR_ENV: "0.3"},
        ],
    )
    def test_bad(self, env) -> None:
        with pytest.raises(ValueError, match="TLLM_KV_LONGCHAIN_PROTECT"):
            parse_config(env)

    def test_bad_knob_ignored_when_off(self) -> None:
        assert parse_config({PIN_AFTER_S_ENV: "x"}) is None

    def test_os_environ(self, monkeypatch) -> None:
        monkeypatch.delenv(ISL_ENV, raising=False)
        assert parse_config() is None
        monkeypatch.setenv(ISL_ENV, "77")
        assert parse_config().isl == 77

    def test_bad_protector(self) -> None:
        binding = Binding()
        for tpb, rewind in ((0, 0), (32, -1)):
            with pytest.raises(ValueError):
                LongChainProtector(
                    LongChainProtectConfig(isl=1),
                    tokens_per_block=tpb,
                    rewind_tokens=rewind,
                    rank=0,
                    create_pin=binding.create_pin,
                    stats_fn=binding.stats,
                    mark_excluded=binding.mark,
                    clear_excluded=binding.clear,
                )


class TestKeyAndTokens:
    def test_key_rules(self) -> None:
        assert conversation_key(make_req("conv", 1)) == "conv"
        assert conversation_key(make_req("  conv  ", 1)) == "conv"
        assert conversation_key(make_req(None, 1)) is None
        assert conversation_key(make_req("", 1)) is None
        assert conversation_key(make_req("conv", 1, dummy=True)) is None
        assert conversation_key(make_req("conv", 1, mm=[[1]])) is None
        assert conversation_key(SimpleNamespace()) is None
        # An executor request in the waiting queue has the same attribute.
        assert (
            conversation_key(
                SimpleNamespace(py_conversation_params=SimpleNamespace(conversation_id=7))
            )
            == "7"
        )

    def test_copy_tokens(self) -> None:
        source = [1, 2, 3, 4]
        copied = copy_tokens(source, 3)
        assert copied == [1, 2, 3]
        source[0] = 9
        assert copied == [1, 2, 3]

        class View(list):
            copied = False

            def __getitem__(self, item):
                return View(super().__getitem__(item))

            def copy(self):
                View.copied = True
                return list(self)

        assert copy_tokens(View([5, 6, 7]), 2) == [5, 6]
        assert View.copied

    @pytest.mark.parametrize(
        "rewind, n, expected",
        [
            (512, 4096, [4096, 3584]),
            (500, 4096, [4096, 3584]),  # rounded up to whole blocks
            (512, 512 + TPB, [544, 32]),
            (512, 512, [512]),  # nothing left before the rewind span
            (0, 4096, [4096]),
        ],
    )
    def test_pin_lengths(self, rewind, n, expected) -> None:
        driver = Driver(rewind=rewind)
        assert driver.protector.pin_lengths(n) == expected


# ---- commit ----------------------------------------------------------------------------------


class TestCommit:
    def test_giant_becomes_pending_small_does_not(self) -> None:
        d = Driver()
        d.commit("small", THR - 1, now=0.0)
        d.commit("giant", THR + 40, now=0.0)
        protector = d.protector
        assert not protector.is_pending("small")
        assert protector.is_pending("giant")
        # Block-aligned, from the committed count.
        assert protector.pending_tokens == THR + 32
        assert protector.stats["giants"] == 1 and protector.stats["commits"] == 2
        assert protector.gen("small") == 1 and protector.gen("giant") == 1
        assert protector.protected_keys() == {"giant"}

    def test_pending_holds_an_owned_copy(self) -> None:
        d = Driver()
        tokens = list(range(THR))
        req = make_req("a", THR)
        d.protector.commit(
            req, SimpleNamespace(num_committed_tokens=THR, reuse_scope=SCOPE), tokens, 0.0
        )
        tokens[0] = 12345
        d.decide(p(1), 200.0)
        assert d.binding.pins[0].tokens[0] == 0

    def test_committed_count_bounds_n(self) -> None:
        d = Driver()
        d.commit("a", THR + 100, now=0.0, committed=THR + 10)
        assert d.protector.pending_tokens == THR

    def test_committed_below_one_block_is_short(self) -> None:
        d = Driver(isl=1)
        d.commit("a", 64, now=0.0, committed=TPB - 1)
        assert not d.protector.is_pending("a")
        assert d.protector.stats["short"] == 1

    @pytest.mark.parametrize("kw", [dict(conv=None), dict(dummy=True), dict(mm=[[1]])])
    def test_nokey(self, kw) -> None:
        d = Driver()
        kw = dict(kw)
        conv = kw.pop("conv", "x")
        d.commit(conv, THR, now=0.0, **kw)
        d.commit(conv, THR - 1, now=0.0, **kw)
        assert d.protector.stats["nokey"] == 1
        assert d.protector.num_pending == 0
        assert d.protector.gen("x") == 0

    def test_gen_counts_every_commit(self) -> None:
        d = Driver()
        for i in range(3):
            d.commit("a", THR if i % 2 else 10, now=float(i))
        assert d.protector.gen("a") == 3

    def test_recommit_below_threshold_closes_pins_and_drops_pending(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.commit("b", THR, now=0.0)
        d.decide(p(1), 200.0)
        assert d.protector.is_pinned("a") and d.protector.is_pinned("b")
        d.commit("c", THR, now=200.0)
        d.commit("a", 10, now=201.0)
        d.commit("c", 10, now=201.0)
        assert not d.protector.is_pinned("a") and not d.protector.is_pending("c")
        assert d.protector.stats["replaced"] == 1
        # Plain close, no demote: the returning request holds the live chain.
        a_pins = [pin for pin in d.binding.pins if pin.closed]
        assert len(a_pins) == 2
        assert not any(event[0] in ("plan", "demote") for event in d.binding.log)
        assert all(pin.pin_id not in d.binding.excluded for pin in a_pins)
        assert d.protector.pending_tokens == 0
        assert d.protector.protected_keys() == {"b"}

    def test_recommit_above_threshold_replaces(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        d.commit("a", THR + 64, now=250.0)
        assert d.protector.is_pending("a") and not d.protector.is_pinned("a")
        assert d.protector.pending_tokens == THR + 64
        assert d.binding.open_pins() == []

    def test_pending_token_cap_drops_the_oldest(self, monkeypatch) -> None:
        monkeypatch.setattr(lcp, "MAX_PENDING_TOKENS", 2 * THR)
        d = Driver()
        for conv in "abc":
            d.commit(conv, THR, now=0.0)
        assert [d.protector.is_pending(c) for c in "abc"] == [False, True, True]
        assert d.protector.stats["pending_dropped"] == 1
        assert d.protector.pending_tokens == 2 * THR

    def test_gen_table_is_bounded(self, monkeypatch) -> None:
        monkeypatch.setattr(lcp, "MAX_GENS", 2)
        d = Driver()
        for conv in ("a", "b", "a", "c"):
            d.commit(conv, 10, now=0.0)
        assert (d.protector.gen("a"), d.protector.gen("b"), d.protector.gen("c")) == (2, 0, 1)


# ---- decide ----------------------------------------------------------------------------------


class TestDecide:
    def test_only_on_decision_passes_and_with_entries(self) -> None:
        d = Driver()
        calls = []
        assert d.protector.decide(p(1), 0.0, lambda: calls.append(1) or []) == []
        assert calls == []  # nothing tracked: busy keys are not even read
        d.commit("a", THR, now=0.0)
        assert d.decide(p(1) + 1, 500.0) == []

    def test_pin_exactly_at_pin_after(self) -> None:
        d = Driver()
        d.commit("a", THR, now=10.0)
        assert d.decide(p(1), 189.9) == []
        ops = d.decide(p(2), 190.0)
        assert ops == [(OP_PIN, "a", 1)]
        assert d.protector.is_pinned("a") and not d.protector.is_pending("a")
        assert [event[0] for event in d.binding.log] == ["stats", "create", "create"]
        assert [(pin.pin_id - PIN_ID_BASE, len(pin.tokens)) for pin in d.binding.pins] == [
            (0, THR),
            (1, THR - REWIND),
        ]
        assert all(pin.stopped for pin in d.binding.pins)
        assert d.binding.excluded == {PIN_ID_BASE, PIN_ID_BASE + 1}
        assert d.protector.stats["pins"] == 1 and d.protector.stats["pins_rewind"] == 1
        assert d.protector.pinned_tokens == THR and d.protector.pending_tokens == 0
        assert d.protector.pin_ms > 0

    def test_one_pin_without_rewind(self) -> None:
        d = Driver(rewind=0)
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        assert len(d.binding.pins) == 1
        assert d.protector.stats["pins_rewind"] == 0

    def test_busy_delays_pin_and_release(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        assert d.decide(p(1), 200.0, busy=["a"]) == []
        assert d.decide(p(2), 379.0) == []
        assert d.decide(p(3), 380.0) == [(OP_PIN, "a", 1)]
        assert d.decide(p(4), 700.0, busy=["a"]) == []
        assert d.decide(p(5), 1100.0) == []
        assert d.decide(p(6), 1100.1) == [(OP_RELEASE, "a", 1)]

    def test_release_after_idle_in_order_with_cap(self) -> None:
        d = Driver()
        keys = [f"k{i}" for i in range(MAX_RELEASES_PER_DECISION + 2)]
        for i, key in enumerate(keys):
            d.commit(key, THR, now=float(i))
        now = 200.0
        for k in range(len(keys)):
            d.decide(p(k + 1), now)
        assert d.protector.num_pinned == len(keys)
        # Keys k0..k8 are idle for more than 400 s at 408.5 s; the cap takes the first eight.
        assert d.decide(p(20), 408.5) == [
            (OP_RELEASE, key, 1) for key in keys[:MAX_RELEASES_PER_DECISION]
        ]
        assert d.protector.stats["releases"] == MAX_RELEASES_PER_DECISION
        assert d.decide(p(21), 1000.0) == [(OP_RELEASE, key, 1) for key in keys[-2:]]
        assert d.binding.open_pins() == []
        assert d.binding.excluded == set()
        assert d.protector.pinned_tokens == 0

    def test_release_needs_min_idle_passes(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        start = 10 * DECIDE_EVERY_PASSES
        d.decide(start, 200.0)
        # The pin keeps the commit's pass, 0: long enough.
        assert d.decide(start + DECIDE_EVERY_PASSES, 500.0) == [(OP_RELEASE, "a", 1)]
        d.commit("b", THR, now=500.0)
        d.decide(start + 2 * DECIDE_EVERY_PASSES, 700.0, busy=["b"])
        d.decide(start + 3 * DECIDE_EVERY_PASSES, 880.0)
        assert d.protector.is_pinned("b")
        # Busy at pass start + 2 * 16: idle for 400 s but not for MIN_IDLE_PASSES passes yet.
        late = start + 2 * DECIDE_EVERY_PASSES + MIN_IDLE_PASSES
        assert d.decide(late - DECIDE_EVERY_PASSES, 1200.0) == []
        assert d.decide(late, 1200.0) == [(OP_RELEASE, "b", 1)]

    def test_release_demotes_after_closing(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        n_pin, rewind_pin = d.binding.pins
        d.binding.log.clear()
        d.decide(p(2), 401.0)
        assert d.binding.log == [
            ("plan", n_pin.pin_id),
            ("close", n_pin.pin_id),
            ("close", rewind_pin.pin_id),
            ("demote", n_pin.pin_id),
        ]
        s = d.protector.stats
        assert (s["demoted_blocks"], s["demoted_pages"], s["kept_shared"]) == (31, 31, 1)

    def test_pin_cap_per_decision(self) -> None:
        d = Driver()
        for key in "abc":
            d.commit(key, THR, now=0.0)
        ops = d.decide(p(1), 200.0)
        assert ops == [(OP_PIN, key, 1) for key in "ab"][:MAX_PINS_PER_DECISION]
        assert d.decide(p(2), 200.0) == [(OP_PIN, "c", 1)]

    def test_admission_refusal_keeps_pending_and_ages_out(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.binding.avail = [(900, 1000), (19, 100)]
        assert d.decide(p(1), 200.0) == []
        assert d.protector.stats["refused"] == 1
        assert d.protector.is_pending("a")
        assert d.decide(p(2), 400.0) == []
        assert d.decide(p(3), 400.1) == [(OP_DROP, "a", 1)]
        assert d.protector.num_pending == 0 and d.protector.pending_tokens == 0
        assert d.protector.stats["aged_out"] == 1
        assert d.binding.pins == []

    def test_admission_reads_stats_once_and_only_when_due(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.commit("b", THR, now=0.0)
        d.decide(p(1), 100.0)
        assert ("stats",) not in d.binding.log
        d.decide(p(2), 200.0)
        assert d.binding.log.count(("stats",)) == 1

    def test_admission_ignores_empty_pool_groups_and_stats_errors(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.binding.avail = [(0, 0), (50, 100)]
        assert d.decide(p(1), 200.0) == [(OP_PIN, "a", 1)]
        d.commit("b", THR, now=0.0)
        d.binding.fail_stats = True
        assert d.decide(p(2), 200.0) == []
        assert d.protector.stats["refused"] == 1

    @pytest.mark.parametrize("cap", ["chains", "tokens"])
    def test_caps(self, monkeypatch, cap) -> None:
        if cap == "chains":
            monkeypatch.setattr(lcp, "MAX_PINNED_CHAINS", 1)
        else:
            monkeypatch.setattr(lcp, "MAX_PINNED_TOKENS", THR + 1)
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.commit("b", THR, now=0.0)
        assert d.decide(p(1), 200.0) == [(OP_PIN, "a", 1)]
        assert d.protector.stats["capped"] == 1
        assert d.protector.is_pending("b")
        # "b" ages out while "a" holds the cap; room again once "a" is released.
        assert d.decide(p(2), 380.0) == []
        assert d.decide(p(3), 401.0) == [(OP_RELEASE, "a", 1), (OP_DROP, "b", 1)]
        d.commit("b", THR, now=401.0)
        assert d.decide(p(4), 600.0) == [(OP_PIN, "b", 2)]

    def test_short_match_is_counted_and_kept(self) -> None:
        d = Driver()
        d.binding.match_short = TPB + 1
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        assert d.protector.stats["short"] == 1 and d.protector.stats["short_rewind"] == 1
        assert d.protector.is_pinned("a")

    def test_pin_ids_skip_ids_in_use(self) -> None:
        d = Driver()
        d.binding.in_use = {PIN_ID_BASE, PIN_ID_BASE + 2}
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        assert [pin.pin_id - PIN_ID_BASE for pin in d.binding.pins] == [1, 3]


# ---- followers ---------------------------------------------------------------------------------


class TestFollower:
    def test_follows_rank0(self) -> None:
        leader, follower = Driver(), Driver(rank=1)
        for driver in (leader, follower):
            driver.commit("a", THR, now=0.0)
            driver.commit("b", THR, now=0.0)
        follower.commit("a", THR, now=0.0)  # a follower-only commit: gen 2
        ops = leader.decide(p(1), 200.0)
        follower.protector.apply(ops, p(1))
        assert follower.protector.stats["gen_mismatch"] == 1
        assert follower.protector.is_pinned("b")
        follower.protector.apply([(OP_PIN, "zzz", 1), (OP_RELEASE, "zzz", 1), (OP_DROP, "zzz", 1)])
        assert follower.protector.stats["missing"] == 3
        ops = leader.decide(p(2), 600.0)
        assert ops == [(OP_RELEASE, "a", 1), (OP_RELEASE, "b", 1)]
        follower.protector.apply(ops)
        # "a" is pending here (gen 2): rank 0 no longer tracks it, so the stale copy goes too.
        assert follower.protector.stats["releases"] == 1
        assert follower.protector.stats["gen_mismatch"] == 2
        assert follower.protector.stats["missing"] == 3
        assert follower.protector.num_pending == 0 and follower.protector.pending_tokens == 0
        assert follower.binding.open_pins() == []

    def test_drop_op_frees_follower_copy(self) -> None:
        leader, follower = Driver(), Driver(rank=1)
        for driver in (leader, follower):
            driver.commit("a", THR, now=0.0)
        leader.binding.avail = [(0, 100)]
        leader.decide(p(1), 200.0)
        ops = leader.decide(p(2), 401.0)
        assert ops == [(OP_DROP, "a", 1)]
        follower.protector.apply(ops)
        assert follower.protector.num_pending == 0 and follower.protector.pending_tokens == 0

    def test_pin_gen_mismatch_is_skipped(self) -> None:
        follower = Driver(rank=1)
        follower.commit("a", THR, now=0.0)
        follower.protector.apply([(OP_PIN, "a", 2)])
        assert follower.protector.stats["gen_mismatch"] == 1
        assert follower.protector.is_pending("a") and not follower.protector.is_pinned("a")
        assert follower.binding.open_pins() == []

    def test_drop_gen_mismatch_still_drops(self) -> None:
        follower = Driver(rank=1)
        follower.commit("a", THR, now=0.0)
        follower.protector.apply([(OP_DROP, "a", 2), (OP_RELEASE, "a", 1)])
        assert follower.protector.stats["gen_mismatch"] == 1
        assert follower.protector.stats["aged_out"] == 1
        assert follower.protector.stats["missing"] == 1
        assert not follower.protector.is_pending("a") and follower.protector.pending_tokens == 0

    def test_release_gen_mismatch_still_releases(self) -> None:
        follower = Driver(rank=1)
        follower.commit("a", THR, now=0.0)
        follower.protector.apply([(OP_PIN, "a", 1)])
        assert follower.protector.is_pinned("a")
        follower.protector.apply([(OP_RELEASE, "a", 2)])
        assert follower.protector.stats["gen_mismatch"] == 1
        assert follower.protector.stats["releases"] == 1
        assert not follower.protector.is_pinned("a") and follower.protector.pinned_tokens == 0
        assert follower.binding.open_pins() == []

    def test_drop_of_a_pinned_chain_releases_it(self) -> None:
        # A stage that missed a context end still holds the older pins when rank 0, which
        # holds a newer pending copy, ages the key out.
        follower = Driver(rank=1)
        follower.commit("a", THR, now=0.0)
        follower.protector.apply([(OP_PIN, "a", 1)])
        follower.protector.apply([(OP_DROP, "a", 2)])
        assert follower.protector.stats["gen_mismatch"] == 1
        assert follower.protector.stats["releases"] == 1
        assert not follower.protector.is_pinned("a")
        assert follower.binding.open_pins() == []

    def test_unknown_op(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            Driver().protector.apply([("x", "a", 1)])


# ---- floor -------------------------------------------------------------------------------------


class TestFloor:
    def _pinned(self, keys="abc") -> Driver:
        d = Driver()
        for i, key in enumerate(keys):
            d.commit(key, THR, now=float(i))
            d.decide(p(i + 1), 200.0 + i)
        assert d.protector.num_pinned == len(keys)
        return d

    def test_no_pins_no_stats_read(self) -> None:
        d = Driver()
        d.binding.avail = [(0, 100)]
        assert d.protector.floor() == 0
        assert ("stats",) not in d.binding.log

    def test_above_floor_does_nothing(self) -> None:
        d = self._pinned()
        d.binding.avail = [(5, 100)]
        assert d.protector.floor() == 0

    def test_releases_oldest_first_and_rereads(self) -> None:
        d = self._pinned()
        reads = iter([[(1, 100)], [(4, 100)], [(5, 100)]])
        d.binding.stats = lambda: next(reads)
        d.protector._stats_fn = d.binding.stats
        assert d.protector.floor() == 2
        assert [d.protector.is_pinned(k) for k in "abc"] == [False, False, True]
        assert d.protector.stats["rel_floor"] == 2
        assert d.protector.stats["releases"] == 0
        # Released locally with a demote, but still listed: rank 0's release finds it.
        assert ("demote", PIN_ID_BASE) in d.binding.log
        d.protector.apply([(OP_RELEASE, "a", 1)])
        assert d.protector.stats["missing"] == 0 and d.protector.stats["releases"] == 1

    def test_at_most_max_releases(self, monkeypatch) -> None:
        keys = [f"k{i}" for i in range(MAX_FLOOR_RELEASES + 2)]
        d = Driver()
        for i, key in enumerate(keys):
            d.commit(key, THR, now=float(i))
        for k in range(len(keys)):
            d.decide(p(k + 1), 200.0)
        d.binding.avail = [(0, 100)]
        assert d.protector.floor() == MAX_FLOOR_RELEASES
        assert d.protector.num_pinned == 2
        assert d.protector.floor() == 2
        assert d.protector.floor() == 0

    def test_rank0_releases_floored_chain_later(self) -> None:
        d = self._pinned("a")
        d.binding.avail = [(0, 100)]
        d.protector.floor()
        d.binding.avail = [(90, 100)]
        assert d.decide(p(5), 600.0) == [(OP_RELEASE, "a", 1)]
        assert d.protector.stats["releases"] == 1
        assert d.protector.protected_keys() == set()


# ---- errors ------------------------------------------------------------------------------------


class TestErrors:
    def test_create_error(self) -> None:
        d = Driver()
        d.binding.fail_create = True
        d.commit("a", THR, now=0.0)
        assert d.decide(p(1), 200.0) == [(OP_PIN, "a", 1)]
        assert d.protector.stats["pin_errors"] == 2
        assert d.binding.excluded == set()
        assert not d.protector.is_pinned("a")
        # Still listed: rank 0 releases it later, so the followers drop theirs.
        assert d.decide(p(2), 401.0) == [(OP_RELEASE, "a", 1)]

    def test_stop_error_closes_the_pin(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        original = d.binding.create_pin

        def create(scope, tokens, pin_id):
            pin = original(scope, tokens, pin_id)
            pin.fail_stop = len(tokens) == THR
            return pin

        d.binding.create_pin = create
        d.protector._create_pin = create
        d.decide(p(1), 200.0)
        n_pin, rewind_pin = d.binding.pins
        assert n_pin.closed and not rewind_pin.closed
        assert d.protector.stats["pin_errors"] == 1
        assert d.binding.excluded == {rewind_pin.pin_id}
        assert d.protector.is_pinned("a")

    def test_plan_error_still_closes(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        d.binding.pins[0].fail_plan = True
        d.decide(p(2), 401.0)
        assert d.protector.stats["plan_errors"] == 1
        assert d.binding.open_pins() == [] and d.binding.excluded == set()

    def test_demote_error_releases_the_handle(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        original = d.binding.pins[0].plan_committed_chain_demote

        def plan(from_block):
            handle = original(from_block)
            handle.fail = True
            return handle

        d.binding.pins[0].plan_committed_chain_demote = plan
        d.decide(p(2), 401.0)
        assert d.protector.stats["demote_errors"] == 1
        assert ("handle_release", PIN_ID_BASE) in d.binding.log
        assert d.binding.open_pins() == []

    def test_close_error_is_counted(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        d.binding.pins[1].fail_close = True
        d.commit("a", 10, now=300.0)
        assert d.protector.stats["close_errors"] == 1
        assert d.binding.excluded == set()
        assert d.protector.pinned_tokens == 0

    def test_plan_none(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        d.binding.pins[0].plan_committed_chain_demote = lambda from_block: None
        d.decide(p(2), 401.0)
        assert d.protector.stats["plan_none"] == 1
        assert d.binding.open_pins() == []

    def test_errors_warn_once(self, monkeypatch) -> None:
        warnings = []
        monkeypatch.setattr(lcp.logger, "warning", warnings.append)
        d = Driver()
        d.binding.fail_create = True
        d.commit("a", THR, now=0.0)
        d.decide(p(1), 200.0)
        assert len(warnings) == 1 and LOG_PREFIX in warnings[0]

    def test_other_errors_propagate(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)

        def boom(scope, tokens, pin_id):
            raise KeyError("not a binding error")

        d.protector._create_pin = boom
        with pytest.raises(KeyError):
            d.decide(p(1), 200.0)


# ---- clear, logging, protected keys -------------------------------------------------------------


class TestClearAndLog:
    def test_clear_closes_without_demote(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.commit("b", THR, now=0.0)
        d.decide(p(1), 200.0)
        d.commit("c", THR, now=200.0)
        d.protector.clear()
        assert d.binding.open_pins() == [] and d.binding.excluded == set()
        assert not any(event[0] in ("plan", "demote") for event in d.binding.log)
        assert d.protector.protected_keys() == set()
        assert (d.protector.pinned_tokens, d.protector.pending_tokens) == (0, 0)
        # Generations survive, so every stage still numbers alike.
        assert d.protector.gen("a") == 1
        # Rank 0's later ops for cleared chains are only counted.
        d.protector.apply([(OP_RELEASE, "a", 1)])
        assert d.protector.stats["missing"] == 1

    def test_protected_keys(self) -> None:
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.commit("b", THR, now=100.0)
        d.decide(p(1), 200.0)
        assert d.protector.is_pinned("a") and d.protector.is_pending("b")
        assert d.protector.protected_keys() == {"a", "b"}

    def test_log_fields(self, monkeypatch) -> None:
        lines = []
        monkeypatch.setattr(lcp.logger, "info", lines.append)
        d = Driver()
        d.protector.log_stats()
        assert lines == []
        d.protector.log_stats(force=True)
        line = lines[-1]
        assert line.startswith(f"{LOG_PREFIX} rank=0 ")
        for field in (
            "pending=",
            "pending_tokens=",
            "pinned=",
            "pinned_tokens=",
            "commits=",
            "giants=",
            "nokey=",
            "pins=",
            "pins_rewind=",
            "short=",
            "short_rewind=",
            "releases=",
            "rel_floor=",
            "aged_out=",
            "refused=",
            "capped=",
            "missing=",
            "gen_mismatch=",
            "demoted_blocks=",
            "kept_shared=",
            "kept_held=",
            "errors=",
            "pin_ms=",
            "host_avail=",
            "rss_mb=",
        ):
            assert f" {field}" in line, field

    def test_logs_every_events(self, monkeypatch) -> None:
        lines = []
        monkeypatch.setattr(lcp.logger, "info", lines.append)
        monkeypatch.setattr(lcp, "LOG_EVERY_EVENTS", 2)
        d = Driver()
        d.commit("a", THR, now=0.0)
        d.commit("b", THR, now=0.0)
        d.decide(p(1), 200.0)
        assert len(lines) == 1 and " pins=2 " in lines[0]
