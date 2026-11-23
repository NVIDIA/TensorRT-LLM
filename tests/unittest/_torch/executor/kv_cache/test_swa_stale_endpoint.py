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
"""Tests for the opt-in stale SWA endpoint drop (TLLM_KV_SWA_STALE_ENDPOINT_KEEP).

The tracker tests run on CPU with fake drop handles. The manager-glue tests need the KV cache
manager module (triton), and the end-to-end test needs the C++ KV cache manager on a GPU.
"""

import random
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.swa_stale_endpoint import (
    DEFAULT_MAX_NODES,
    KEEP_ENV,
    MAX_LINK_BACKOFF_TOKENS,
    MAX_NODES_ENV,
    SwaStaleEndpointTracker,
    parse_keep,
    parse_max_nodes,
    scope_seed,
)

TPB = 32
WINDOW = 2048
REWIND = 512


class FakeHandle:
    """Mirrors PlannedDropHandle: drop() applies once, a second call raises."""

    def __init__(self, name: str, log: List[str]) -> None:
        self.name = name
        self.dropped = False
        self._log = log

    def drop(self) -> None:
        if self.dropped:
            raise RuntimeError("Planned drop handle has already been dropped")
        self.dropped = True
        self._log.append(self.name)


class Driver:
    """Feeds prompts to a tracker the way the manager does at context end."""

    def __init__(self, keep: int, rewind: int = REWIND, window: int = WINDOW, **kwargs) -> None:
        self.tracker = SwaStaleEndpointTracker(
            keep=keep,
            tokens_per_block=TPB,
            window_size=window,
            rewind_tokens=rewind,
            log_every=0,
            **kwargs,
        )
        self.drops: List[str] = []
        self.handles: Dict[str, FakeHandle] = {}

    def commit(
        self,
        name: str,
        tokens: np.ndarray,
        seed: int = 0,
        plan: bool = True,
        committed: Optional[int] = None,
        reused: Optional[int] = None,
    ) -> FakeHandle:
        handle = FakeHandle(name, self.drops)
        self.handles[name] = handle
        num_committed = len(tokens) if committed is None else committed
        self.tracker.commit(
            tokens=tokens,
            num_tokens=len(tokens),
            seed=seed,
            num_committed_tokens=num_committed,
            plan_drop=(lambda: handle) if plan else (lambda: None),
            reused_tokens=reused,
        )
        return handle

    def dropped(self) -> List[str]:
        return list(self.drops)


_rng = np.random.default_rng(1234)


def fresh(n: int) -> np.ndarray:
    return _rng.integers(0, 150_000, size=n, dtype=np.int32)


def chain(base: np.ndarray, *increments: int) -> List[np.ndarray]:
    """Prompts of one conversation: each turn extends the previous prompt exactly."""
    prompts = [base]
    for inc in increments:
        prompts.append(np.concatenate([prompts[-1], fresh(inc)]))
    return prompts


# --------------------------------------------------------------------------- knob parsing


def test_parse_keep_unset_and_zero_is_off():
    assert parse_keep({}) == 0
    assert parse_keep({KEEP_ENV: ""}) == 0
    assert parse_keep({KEEP_ENV: "0"}) == 0


def test_parse_keep_values():
    assert parse_keep({KEEP_ENV: "3"}) == 3
    assert parse_keep({KEEP_ENV: " 2 "}) == 2


def test_parse_keep_one_is_allowed_with_warning(monkeypatch):
    warnings = []
    from tensorrt_llm._torch.pyexecutor.kv_cache import swa_stale_endpoint

    monkeypatch.setattr(swa_stale_endpoint.logger, "warning", lambda msg: warnings.append(msg))
    assert parse_keep({KEEP_ENV: "1"}) == 1
    assert len(warnings) == 1 and KEEP_ENV in warnings[0]


@pytest.mark.parametrize("raw", ["-1", "abc", "2.5"])
def test_parse_keep_rejects_invalid(raw):
    with pytest.raises(ValueError, match=KEEP_ENV):
        parse_keep({KEEP_ENV: raw})


def test_parse_max_nodes():
    assert parse_max_nodes({}) == DEFAULT_MAX_NODES
    assert parse_max_nodes({MAX_NODES_ENV: "100"}) == 100
    for raw in ["0", "-3", "x"]:
        with pytest.raises(ValueError, match=MAX_NODES_ENV):
            parse_max_nodes({MAX_NODES_ENV: raw})


def test_scope_seed_is_deterministic_and_scoped():
    assert scope_seed(None, None) == scope_seed(None, None)
    assert scope_seed(1, None) != scope_seed(None, None)
    assert scope_seed(None, 7) != scope_seed(None, 8)


def test_tracker_rejects_bad_geometry():
    with pytest.raises(ValueError):
        SwaStaleEndpointTracker(keep=0, tokens_per_block=32, window_size=2048, rewind_tokens=0)
    with pytest.raises(ValueError):
        SwaStaleEndpointTracker(keep=2, tokens_per_block=0, window_size=2048, rewind_tokens=0)


# --------------------------------------------------------------------------- selection logic


@pytest.mark.parametrize("keep", [2, 3, 4])
def test_linear_chain_keeps_newest_n(keep):
    d = Driver(keep)
    prompts = chain(fresh(20_000), *[4_000] * 7)
    for i, p in enumerate(prompts):
        d.commit(f"t{i}", p)
        expected = [f"t{j}" for j in range(0, max(0, i - keep + 1))]
        assert d.dropped() == expected, f"after t{i}"
    assert d.tracker.stats["linked"] == len(prompts) - 1
    assert d.tracker.stats["deferred"] == 0


def test_short_turns_defer_until_window_clears_kept_regions():
    keep = 2
    d = Driver(keep)
    prompts = chain(fresh(20_000), *[300] * 20)
    ends = [len(p) for p in prompts]
    geo = d.tracker.geometry
    for i, p in enumerate(prompts):
        before = set(d.dropped())
        d.commit(f"t{i}", p)
        newly = set(d.dropped()) - before
        kept = range(max(0, i - keep + 1), i + 1)
        kept_min_prot = min(geo(ends[j])[0] for j in kept)
        for name in newly:
            j = int(name[1:])
            assert j <= i - keep
            # A dropped window never reaches the protected region of a kept endpoint.
            assert geo(ends[j])[2] < kept_min_prot
        for j in kept:
            assert not d.handles[f"t{j}"].dropped
    assert d.tracker.stats["deferred"] > 0
    # The line has moved far past the first turns, so they are dropped in the end.
    assert d.handles["t0"].dropped and d.handles["t5"].dropped


@pytest.mark.parametrize("backoff", [1, 40, MAX_LINK_BACKOFF_TOKENS - TPB])
def test_tail_rerender_within_backoff_links(backoff):
    d = Driver(2)
    t0 = fresh(30_000)
    # The next turn re-renders the last tokens of t0 (e.g. the generation prompt).
    t1 = np.concatenate([t0[:-backoff], fresh(5_000)])
    t2 = np.concatenate([t1, fresh(5_000)])
    for name, p in [("t0", t0), ("t1", t1), ("t2", t2)]:
        d.commit(name, p)
    assert d.tracker.stats["linked"] == 2
    assert d.dropped() == ["t0"]


@pytest.mark.parametrize("backoff", [MAX_LINK_BACKOFF_TOKENS + TPB, 200, REWIND, 2_000])
def test_deeper_divergence_does_not_link(backoff):
    """Diverging more than MAX_LINK_BACKOFF_TOKENS before an endpoint is what two conversations
    with one system prompt look like; such a prompt starts a new line."""
    d = Driver(2)
    t0 = fresh(30_000)
    t1 = np.concatenate([t0[:-backoff], fresh(5_000)])
    for name, p in [("t0", t0)] + [(f"t{i + 1}", q) for i, q in enumerate(chain(t1, 5_000, 5_000))]:
        d.commit(name, p)
    assert d.tracker.stats["linked"] == 2  # t2 -> t1 and t3 -> t2 only
    assert d.dropped() == ["t1"]
    assert not d.handles["t0"].dropped


@pytest.mark.parametrize(
    "backoff,links", [(200, True), (REWIND - TPB, True), (REWIND + TPB, False), (3_000, False)]
)
def test_later_endpoints_of_a_line_take_divergence_up_to_the_rewind(backoff, links):
    d = Driver(2)
    t0 = fresh(30_000)
    t1 = np.concatenate([t0, fresh(5_000)])
    # t2 rewrites the end of t1 (a rollback inside t1's rewind region); t1 already has a parent.
    t2 = np.concatenate([t1[:-backoff], fresh(5_000)])
    t3 = np.concatenate([t2, fresh(5_000)])
    for name, p in [("t0", t0), ("t1", t1), ("t2", t2), ("t3", t3)]:
        d.commit(name, p)
    assert d.tracker.stats["linked"] == 3
    if links:
        assert d.dropped() == ["t0", "t1"]
    else:
        # t2 extends t0 instead, a second child: t0 becomes a fork and t1 the abandoned tip.
        assert d.dropped() == []
        assert d.tracker.stats["fork_stops"] > 0


def test_deep_divergence_starts_a_new_line_and_keeps_the_old_tip():
    d = Driver(2)
    t0 = fresh(30_000)
    t1 = np.concatenate([t0, fresh(5_000)])
    # Diverges 4,000 tokens before t1's end: deeper than rewind + window.
    b0 = np.concatenate([t1[:-4_000], fresh(6_000)])
    line = chain(b0, 5_000, 5_000, 5_000)
    d.commit("t0", t0)
    d.commit("t1", t1)
    for i, p in enumerate(line):
        d.commit(f"b{i}", p)
    # b0 links to t0 (the deepest endpoint whose link prefix it shares), making t0 a fork.
    assert not d.handles["t1"].dropped
    assert not d.handles["t0"].dropped
    assert d.dropped() == ["b0", "b1"]


def test_regenerate_fork_pins_the_fork_point():
    d = Driver(2)
    a = chain(fresh(20_000), 5_000, 5_000)  # a0, a1, a2
    for i, p in enumerate(a):
        d.commit(f"a{i}", p)
    assert d.dropped() == ["a0"]
    # Two different continuations of a2 (a regenerate), each followed by more turns.
    x = chain(np.concatenate([a[2], fresh(5_000)]), 5_000, 5_000)
    y = chain(np.concatenate([a[2], fresh(5_000)]), 5_000, 5_000)
    for i in range(3):
        d.commit(f"x{i}", x[i])
        d.commit(f"y{i}", y[i])
    assert d.tracker.stats["fork_stops"] > 0
    # x0 makes a1 stale before y0 turns a2 into a fork; a2 is never dropped afterwards.
    assert not d.handles["a2"].dropped
    assert sorted(d.dropped()) == ["a0", "a1", "x0", "y0"]


def test_subagents_forked_from_one_endpoint_only_drop_their_own_lines():
    d = Driver(3)
    parent = chain(fresh(40_000), 5_000)  # p0, p1
    for i, p in enumerate(parent):
        d.commit(f"p{i}", p)
    lines = {
        s: chain(np.concatenate([parent[1], fresh(3_000 + 100 * s)]), *[4_000] * 5)
        for s in range(4)
    }
    # Interleave the subagents the way concurrent requests commit.
    for turn in range(6):
        for s, line in lines.items():
            d.commit(f"s{s}.{turn}", line[turn])
    assert not d.handles["p1"].dropped and not d.handles["p0"].dropped
    for s in lines:
        own = sorted(n for n in d.dropped() if n.startswith(f"s{s}."))
        assert own == [f"s{s}.{t}" for t in range(3)]
    assert len(d.dropped()) == 12


def test_conversations_sharing_a_system_prompt_do_not_drop_each_other():
    d = Driver(2)
    system = fresh(20_000)
    convs = {c: chain(np.concatenate([system, fresh(6_000)]), *[5_000] * 4) for c in "xyz"}
    order = [("x", 0), ("y", 0), ("x", 1), ("x", 2), ("x", 3), ("z", 0), ("y", 1), ("x", 4)]
    for c, t in order:
        d.commit(f"{c}{t}", convs[c][t])
    assert sorted(d.dropped()) == ["x0", "x1", "x2"]
    assert not d.handles["y0"].dropped and not d.handles["z0"].dropped


@pytest.mark.parametrize("keep", [2, 3])
@pytest.mark.parametrize(
    "first_x,first_y",
    [
        (300, 500),
        (500, 800),
        (800, 500),
        (1_500, 2_000),
        (300, 2_400),
        (2_000, 2_500),
        (6_000, 6_500),
    ],
)
def test_idle_conversation_survives_a_sibling_with_the_same_system_prompt(keep, first_x, first_y):
    """X and Y share a system prompt; Y runs while X waits (a long tool call).

    With first messages shorter than rewind + window, Y used to link to X's first turn and its
    walk dropped X's only endpoint."""
    d = Driver(keep)
    system = fresh(20_000)
    x0 = np.concatenate([system, fresh(first_x)])
    d.commit("x0", x0)
    for i, p in enumerate(chain(np.concatenate([system, fresh(first_y)]), *[3_000] * (keep + 2))):
        d.commit(f"y{i}", p)
    assert not d.handles["x0"].dropped
    # y0's window may share S's tail with x0's rewind region; then y0 waits for x0.
    y0_waits = first_x - REWIND - WINDOW < first_y and first_y - WINDOW < first_x - WINDOW
    assert sorted(d.dropped()) == [f"y{i}" for i in range(int(y0_waits), 3)]
    assert (d.tracker.stats["guard_deferred"] > 0) == y0_waits
    # X resumes from its endpoint and links to it; nothing it needed was dropped.
    d.commit("x1", np.concatenate([x0, fresh(3_000)]), reused=len(x0) - 1)
    assert d.tracker.stats["resume_dropped"] == 0
    assert d.tracker.stats["parent_miss"] == 0


@pytest.mark.parametrize("keep", [2, 3])
@pytest.mark.parametrize("task_b", [1_500, 4_000])
def test_shared_system_prompt_sibling_does_not_drop_a_live_endpoint(keep, task_b):
    """Sa1 = S + 800 and Sb1 = S + task_b share an 18k S; Sb
    commits keep + 1 turns before Sa2 arrives."""
    d = Driver(keep)
    system = fresh(18_000)
    sa1 = np.concatenate([system, fresh(800)])
    d.commit("sa1", sa1)
    for k, p in enumerate(chain(np.concatenate([system, fresh(task_b)]), *[3_000] * keep), 1):
        d.commit(f"sb{k}", p)
    assert d.dropped() == ["sb1"]
    d.commit("sa2", np.concatenate([sa1, fresh(3_000)]), reused=len(sa1) - 1)
    assert d.tracker.stats["linked"] == keep + 1
    assert d.tracker.stats["resume_dropped"] == 0


def test_a_link_to_another_first_turn_does_not_widen_the_next_link():
    """a0's first message is shorter than MAX_LINK_BACKOFF_TOKENS, so sb1 (another
    conversation) links to it. sb1 must still take only small re-renders: sc0 (a third
    conversation, 450 tokens after the system prompt) must not link to sb1, or sc's walk
    would drop sb1, b's only endpoint."""
    d = Driver(2)
    system = fresh(18_000)
    d.commit("a0", np.concatenate([system, fresh(MAX_LINK_BACKOFF_TOKENS - 28)]))
    d.commit("sb1", np.concatenate([system, fresh(400)]))
    assert d.tracker.stats["linked"] == 1
    for i, p in enumerate(chain(np.concatenate([system, fresh(450)]), 3_000, 3_000, 3_000)):
        d.commit(f"sc{i}", p)
    assert not d.handles["sb1"].dropped and not d.handles["a0"].dropped
    assert sorted(d.dropped()) == ["sc0", "sc1"]


def test_sibling_rewind_region_defers_an_overlapping_window():
    """Sb0's protected region before its planned window shares S's tail pages with Sa0's
    window; dropping Sa0's plan alone would free them, so Sa0 waits until Sb0 is dropped."""
    d = Driver(2)
    system = fresh(18_000)
    sa = chain(np.concatenate([system, fresh(800)]), 3_000, 3_000, 3_000, 3_000)
    sb = chain(np.concatenate([system, fresh(1_500)]), 3_000, 3_000)
    d.commit("sa0", sa[0])
    d.commit("sb0", sb[0])
    for i in (1, 2, 3):
        d.commit(f"sa{i}", sa[i])
    assert d.dropped() == ["sa1"]
    assert d.tracker.stats["guard_deferred"] > 0
    d.commit("sb1", sb[1])
    d.commit("sb2", sb[2])
    assert "sb0" in d.dropped() and not d.handles["sa0"].dropped
    d.commit("sa4", sa[4])
    assert d.handles["sa0"].dropped and d.handles["sa2"].dropped


def test_damage_counters():
    d = Driver(2)
    t = chain(fresh(20_000), 5_000, 5_000, 5_000)
    d.commit("t0", t[0])
    # Continues t0 but reused nothing: the kept window was missing (pool pressure).
    d.commit("t1", t[1], reused=0)
    assert d.tracker.stats["parent_miss"] == 1
    assert d.tracker.stats["parent_miss_tokens"] == len(t[0])
    d.commit("t2", t[2], reused=len(t[1]) - 1)
    d.commit("t3", t[3], reused=len(t[2]) - 1)
    assert d.dropped() == ["t0", "t1"]
    assert d.tracker.stats["parent_miss"] == 1
    # A prompt that resumes from the dropped t1 and reused less than its rewind region: damage.
    d.commit("r1", np.concatenate([t[1], fresh(2_000)]), reused=len(t[0]))
    assert d.tracker.stats["resume_dropped"] == 1
    assert d.tracker.stats["resume_dropped_tokens"] == len(t[1]) - len(t[0])
    # The same with the reuse inside t1's rewind region (another endpoint covered it): no damage.
    d.commit("r2", np.concatenate([t[1], fresh(2_500)]), reused=len(t[1]) - 100)
    assert d.tracker.stats["resume_dropped"] == 1
    # Without a reuse figure only the tombstone match is counted.
    d.commit("r3", np.concatenate([t[1], fresh(3_000)]))
    assert d.tracker.stats["resume_dropped"] == 2


def _reference_pages(hasher, tokens: np.ndarray):
    """Page identities of a prompt: one per full block (its prefix key) and the partial tail."""
    full = len(tokens) // TPB
    keys = [int(k) for k in hasher.block_keys(tokens, full, 0)]
    tail = None
    if len(tokens) > full * TPB:
        tail = ("tail", keys[-1] if keys else 0, tuple(tokens[full * TPB :].tolist()))
    return keys, tail


@pytest.mark.parametrize("keep", [2, 3])
@pytest.mark.parametrize("seed", [11, 12])
def test_drops_only_continued_endpoints_and_free_no_protected_page(keep, seed):
    """Random multi-conversation streams against a page-level reference model.

    Pages: one per (prefix key, block) and one per partial tail; a plan holds a count on the
    window pages it saw (by generation); a drop frees a page when its count reaches 0. Checks
    after every commit: a walk drops only an endpoint that its own conversation continued; no
    freed page is in the protected region of an endpoint that is not stale on a fork-free line;
    ``resume_dropped`` counts only rollbacks that go back past a dropped (stale) endpoint.
    """
    rng = random.Random(seed)
    d = Driver(keep)
    tr = d.tracker
    hasher = tr._hasher
    systems = [fresh(rng.randint(3_000, 20_000)) for _ in range(3)]
    present: Dict[object, int] = {}  # page identity -> generation
    counts: Dict[tuple, int] = {}
    plans: Dict[str, List[tuple]] = {}
    gen = [0]
    owner: Dict[str, int] = {}  # endpoint name -> conversation
    continued: Dict[str, set] = {}  # endpoint name -> conversations that continued it
    node_name: Dict[int, str] = {}
    convs: List[dict] = []
    freed: List[object] = []

    def protected_pages(tokens):
        keys, tail = _reference_pages(hasher, tokens)
        prot = max(0, len(tokens) - REWIND - WINDOW) // TPB
        return set(keys[prot:]) | ({tail} if tail else set())

    def window_pages(tokens):
        keys, tail = _reference_pages(hasher, tokens)
        start = max(0, len(tokens) + 1 - WINDOW) // TPB
        return keys[start:] + ([tail] if tail else [])

    class PageHandle:
        def __init__(self, name):
            self.name = name

        def drop(self):
            d.drops.append(self.name)
            for ident, g in plans.pop(self.name):
                counts[(ident, g)] -= 1
                if counts[(ident, g)] == 0 and present.get(ident) == g:
                    del present[ident]
                    freed.append(ident)

    for step in range(500):
        name = f"n{step}"
        r = rng.random()
        deep = False
        if not convs or r < 0.12:
            sys_tokens = systems[rng.randrange(len(systems))]
            p = np.concatenate(
                [sys_tokens, fresh(rng.randint(MAX_LINK_BACKOFF_TOKENS + 72, 6_000))]
            )
            conv = {"id": len(convs), "tip": None, "prev": None}
            convs.append(conv)
        else:
            conv = convs[rng.randrange(len(convs))]
            base_name, base = conv["tip"]
            if r < 0.17 and conv["prev"] is not None:
                base_name, base = conv[
                    "prev"
                ]  # regenerate: a second child of the previous endpoint
            q = rng.random()
            backoff = (
                0 if q < 0.8 else (rng.randint(1, 100) if q < 0.95 else rng.randint(300, 3_000))
            )
            p = np.concatenate([base[: len(base) - backoff], fresh(rng.randint(50, 5_000))])
            if backoff <= REWIND:
                continued.setdefault(base_name, set()).add(conv["id"])
            deep = backoff > 100
        owner[name] = conv["id"]
        # Commit: create the pages this prompt does not find, then plan its window.
        keys, tail = _reference_pages(hasher, p)
        for ident in keys + ([tail] if tail else []):
            if ident not in present:
                gen[0] += 1
                present[ident] = gen[0]
        refs = [(ident, present[ident]) for ident in window_pages(p)]
        for ref in refs:
            counts[ref] = counts.get(ref, 0) + 1
        plans[name] = refs
        handle = PageHandle(name)
        before = len(d.drops)
        damage_before = tr.stats["resume_dropped"]
        freed.clear()
        tr.commit(
            tokens=p,
            num_tokens=len(p),
            seed=0,
            num_committed_tokens=len(p),
            plan_drop=lambda h=handle: h,
            reused_tokens=None if conv["tip"] is None else 0,
        )
        for node in tr._lru.values():
            node_name.setdefault(node.uid, None)
        newest = next(reversed(tr._lru.values()))
        node_name[newest.uid] = node_name[newest.uid] or name
        if conv["tip"] is not None:
            conv["prev"] = conv["tip"]
        conv["tip"] = (name, p)
        # 0. Damage only from a rollback past the previous endpoint (keep-N's known cost).
        assert deep or tr.stats["resume_dropped"] == damage_before, step
        # 1. A walk drops only endpoints their own conversation continued.
        for dropped in d.drops[before:]:
            assert owner[dropped] in continued.get(dropped, set()), (step, dropped)
        # 2. No freed page is protected by an endpoint that is not stale.
        children: Dict[int, List[object]] = {}
        nodes = list(tr._lru.values()) + list(tr._tombs.values())
        for node in nodes:
            if node.parent is not None:
                children.setdefault(node.parent.uid, []).append(node)

        def depth_below(node):
            kids = children.get(node.uid, [])
            if node.children != 1 or len(kids) != 1:
                return 0 if node.children == 0 else None
            below = depth_below(kids[0])
            return None if below is None else below + 1

        names = {n.uid: nm for nm, n in ((node_name.get(n.uid), n) for n in nodes) if nm}
        freed_set = set(freed)
        if freed_set:
            for node in tr._lru.values():
                below = depth_below(node)
                if node.children == 1 and below is not None and below >= keep - 1:
                    continue  # stale on its line
                nm = names.get(node.uid)
                if nm is None:
                    continue
                toks = None
                for c in convs:
                    for cand in (c["tip"], c["prev"]):
                        if cand is not None and cand[0] == nm:
                            toks = cand[1]
                if toks is None:
                    continue
                assert not (protected_pages(toks) & freed_set), (step, nm)
    assert tr.stats["dropped"] > 50
    assert tr.stats["guard_deferred"] + tr.stats["deferred"] > 0


def test_new_and_parent_endpoints_are_never_dropped_on_random_forests():
    rng = random.Random(7)
    for keep in (2, 3):
        d = Driver(keep)
        system = fresh(8_000)
        tips: List[tuple] = []
        for step in range(400):
            name = f"n{step}"
            prev_name = None
            if not tips or rng.random() < 0.15:
                p = np.concatenate([system, fresh(rng.randint(100, 9_000))])
                tips.append((name, p))
            else:
                k = rng.randrange(len(tips))
                prev_name, prev = tips[k]
                cut = len(prev)
                if rng.random() < 0.1:
                    cut -= rng.randint(1, 3_000)  # re-rendered tail or a deeper divergence
                    prev_name = None
                p = np.concatenate([prev[:cut], fresh(rng.randint(50, 6_000))])
                if rng.random() < 0.8:
                    tips[k] = (name, p)
                else:
                    tips.append((name, p))  # a branch: the old tip gets a second child later
            before = len(d.drops)
            h = d.commit(name, p)
            assert not h.dropped
            if prev_name is not None:
                # An exactly extended endpoint is the new endpoint's parent, which this commit
                # keeps. (Another line may have dropped it earlier: a late second child.)
                assert prev_name not in d.drops[before:]
        assert d.tracker.stats["drop_error"] == 0
        assert d.tracker.stats["dropped"] > 0
        assert d.tracker.stats["fork_stops"] > 0


def test_identical_prompt_refreshes_the_endpoint():
    d = Driver(2)
    p = fresh(25_000)
    first = d.commit("a", p)
    second = d.commit("a-again", p)
    assert d.tracker.num_nodes == 1
    assert d.tracker.stats["refreshed"] == 1
    # The old handle is released; the endpoint now carries the new plan.
    assert first.dropped and not second.dropped
    nxt = np.concatenate([p, fresh(5_000)])
    d.commit("b", nxt)
    d.commit("c", np.concatenate([nxt, fresh(5_000)]))
    assert second.dropped


def test_reuse_scopes_do_not_link():
    d = Driver(2)
    prompts = chain(fresh(20_000), 5_000, 5_000)
    for i, p in enumerate(prompts):
        d.commit(f"t{i}", p, seed=scope_seed(None, i))  # a different salt every turn
    assert d.tracker.stats["linked"] == 0
    assert d.dropped() == []


def test_short_prompt_is_not_planned():
    d = Driver(2)
    calls = []
    d.tracker.commit(
        tokens=fresh(TPB - 1),
        num_tokens=TPB - 1,
        seed=0,
        num_committed_tokens=TPB - 1,
        plan_drop=lambda: calls.append(1),
    )
    assert calls == []
    assert d.tracker.stats["skipped_short"] == 1
    assert d.tracker.num_nodes == 0


def test_plan_outcomes_do_not_change_decisions():
    """PP stages may differ in which plans exist; the drop decisions must not."""
    rng = random.Random(3)
    prompts = []
    for c in range(6):
        prompts.append([(f"c{c}.{i}", p) for i, p in enumerate(chain(fresh(15_000), *[4_000] * 6))])
    order = []
    while any(prompts):
        c = rng.choice([i for i, q in enumerate(prompts) if q])
        order.append(prompts[c].pop(0))

    def run(mode):
        d = Driver(2)
        decided = []
        for name, p in order:
            if mode == "none" and rng.random() < 0.3:
                d.commit(name, p, plan=False)
            elif mode == "skip" and rng.random() < 0.3:
                d.commit(name, p, committed=len(p) - 1)
            else:
                d.commit(name, p)
            decided.append(
                (int(d.tracker.stats["dropped"]) + int(d.tracker.stats["dropped_no_handle"]))
            )
        return decided, d

    full, d_full = run("all")
    none, d_none = run("none")
    skip, d_skip = run("skip")
    assert full == none == skip
    assert d_none.tracker.stats["plan_none"] > 0
    assert d_skip.tracker.stats["plan_skipped"] > 0
    assert d_none.tracker.stats["dropped_no_handle"] > 0


def test_plan_and_drop_errors_are_counted_not_raised():
    d = Driver(2)

    def bad_plan():
        raise RuntimeError("plan_committed_block_drop() requires stop_committing()")

    d.tracker.commit(
        tokens=fresh(10_000),
        num_tokens=10_000,
        seed=0,
        num_committed_tokens=10_000,
        plan_drop=bad_plan,
    )
    assert d.tracker.stats["plan_error"] == 1
    assert d.tracker.num_nodes == 1

    class BadHandle:
        def drop(self):
            raise RuntimeError("boom")

    d2 = Driver(2)
    prompts = chain(fresh(20_000), 5_000, 5_000)
    d2.tracker.commit(
        tokens=prompts[0],
        num_tokens=len(prompts[0]),
        seed=0,
        num_committed_tokens=len(prompts[0]),
        plan_drop=BadHandle,
    )
    d2.commit("t1", prompts[1])
    d2.commit("t2", prompts[2])
    assert d2.tracker.stats["drop_error"] == 1


def test_node_cap_drops_least_recently_touched():
    d = Driver(3, max_nodes=4)
    for c in range(6):
        d.commit(f"c{c}", fresh(10_000))
    assert d.tracker.num_nodes == 4
    assert d.dropped() == ["c0", "c1"]
    assert d.tracker.stats["cap_dropped"] == 2


def test_clear_applies_every_outstanding_plan():
    d = Driver(3)
    for i, p in enumerate(chain(fresh(20_000), 5_000, 5_000)):
        d.commit(f"t{i}", p)
    assert d.dropped() == []
    d.tracker.clear()
    assert sorted(d.dropped()) == ["t0", "t1", "t2"]
    assert d.tracker.num_nodes == 0
    d.tracker.clear()  # idempotent


def test_list_tokens_and_int32_views_hash_alike():
    tokens = fresh(10_000)
    a = Driver(2)
    b = Driver(2)
    nxt = np.concatenate([tokens, fresh(5_000)])
    a.commit("x", tokens)
    b.tracker.commit(
        tokens=tokens.tolist(),
        num_tokens=len(tokens),
        seed=0,
        num_committed_tokens=len(tokens),
        plan_drop=lambda: FakeHandle("x", b.drops),
    )
    a.commit("y", nxt)
    b.commit("y", nxt)
    assert a.tracker.stats["linked"] == b.tracker.stats["linked"] == 1


# --------------------------------------------------------------------------- manager glue


def _manager_module():
    pytest.importorskip("triton")
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2

    return kv_cache_manager_v2


def _fake_manager(m, **overrides):
    fields = dict(
        max_attention_window_vec=[WINDOW, None, WINDOW],
        max_seq_len=1 << 20,
        enable_block_reuse=True,
        block_reuse_policy=m.BlockReusePolicy.ALL_REUSABLE,
        _swa_endpoint_rewind=REWIND,
        is_draft=False,
        _can_publish_block_reuse=True,
        _has_cp_helix=False,
        tokens_per_block=TPB,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_manager_default_off_builds_nothing(monkeypatch):
    m = _manager_module()
    monkeypatch.delenv(KEEP_ENV, raising=False)
    assert m.KVCacheManagerV2._make_swa_stale_endpoint_tracker(_fake_manager(m)) is None
    monkeypatch.setenv(KEEP_ENV, "0")
    assert m.KVCacheManagerV2._make_swa_stale_endpoint_tracker(_fake_manager(m)) is None


def test_manager_builds_tracker_only_when_eligible(monkeypatch):
    m = _manager_module()
    monkeypatch.setenv(KEEP_ENV, "3")
    tracker = m.KVCacheManagerV2._make_swa_stale_endpoint_tracker(_fake_manager(m))
    assert isinstance(tracker, SwaStaleEndpointTracker)
    assert (tracker.keep, tracker.window_size, tracker.rewind_tokens) == (3, WINDOW, REWIND)
    for bad in [
        dict(enable_block_reuse=False),
        dict(block_reuse_policy=m.BlockReusePolicy.PER_CONVERSATION),
        dict(_swa_endpoint_rewind=0),
        dict(is_draft=True),
        dict(_can_publish_block_reuse=False),
        dict(max_attention_window_vec=[None, None]),
    ]:
        assert m.KVCacheManagerV2._make_swa_stale_endpoint_tracker(_fake_manager(m, **bad)) is None


# --------------------------------------------------------------------------- C++ end to end


def test_cpp_manager_drops_only_stale_windows():
    """Chain five prompts through the C++ manager; the stale endpoint's window is dropped."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU and the C++ KV cache manager")
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
        TokenId,
    )

    tpb, window, rewind, step = 16, 64, 32, 128
    layers = [
        AttentionLayerConfig(
            layer_id=LayerId(i),
            buffers=[
                BufferConfig(role=DataRole("key"), size=8192),
                BufferConfig(role=DataRole("value"), size=8192),
            ],
            sliding_window_size=window if i == 0 else None,
        )
        for i in range(2)
    ]
    manager = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=tpb, cache_tiers=[GpuCacheTierConfig(quota=64 << 20)], layers=layers
        )
    )
    tracker = SwaStaleEndpointTracker(
        keep=2, tokens_per_block=tpb, window_size=window, rewind_tokens=rewind, log_every=0
    )
    stream = torch.cuda.Stream()
    ids = list(range(1, 5 * step + 1))
    try:
        for k in range(1, 6):
            tokens = [TokenId(t) for t in ids[: k * step]]
            kv_cache = manager.create_kv_cache(None, tokens)
            assert kv_cache.resume(stream.cuda_stream)
            assert kv_cache.resize(len(tokens))
            uncommitted = tokens[kv_cache.num_committed_tokens :]
            if uncommitted:
                kv_cache.commit(uncommitted)
            kv_cache.stop_committing()
            tracker.commit(
                tokens=ids[: k * step],
                num_tokens=len(tokens),
                seed=0,
                num_committed_tokens=kv_cache.num_committed_tokens,
                plan_drop=kv_cache.plan_committed_block_drop,
            )
            stream.synchronize()
            kv_cache.close()
            ends = [j * step for j in range(1, k + 1)]
            for j, e in enumerate(ends, start=1):
                probe = manager.probe_reuse(None, [TokenId(t) for t in ids[:e]])
                if j <= k - 2:
                    assert probe < e, f"endpoint {j} should be dropped after turn {k}"
                else:
                    assert probe == e, f"endpoint {j} should be intact after turn {k}"
        assert tracker.stats["plan_none"] == 0
        assert tracker.stats["plan_skipped"] == 0
        assert tracker.stats["dropped"] == 3
    finally:
        tracker.clear()
        manager.shutdown()


class _RecordingHandle:
    """Wraps a real PlannedDropHandle and records which endpoint it belonged to."""

    def __init__(self, handle, turn: int, log: List[int]) -> None:
        self._handle = handle
        self._turn = turn
        self._log = log

    def drop(self) -> None:
        self._log.append(self._turn)
        self._handle.drop()


@pytest.mark.parametrize(
    "case",
    [
        # Served shape: non-aligned prompts (205 + 120 k, tokens_per_block 16) committed in
        # non-aligned chunks of 40 through the endpoint priority callback, with a host tier and
        # pool_ratio [0.15, 0.85]. Windows do not overlap: every stale endpoint is dropped.
        dict(step=120, turns=6, chunk=40, gpu=64 << 20, churn=0, ratio=[0.15, 0.85]),
        # Turns shorter than the window: plans overlap and stale endpoints wait for the
        # protected regions of the kept ones (the deferral, with the real page counts).
        dict(step=40, turns=10, chunk=24, gpu=64 << 20, churn=0, ratio=[0.15, 0.85]),
        # A small GPU tier and unrelated traffic between turns: planned pages migrate to host.
        dict(step=120, turns=6, chunk=40, gpu=8 << 20, churn=4, ratio=None),
    ],
    ids=["nonaligned-chunked", "overlapping-windows", "host-churn"],
)
def test_cpp_manager_served_shape(case):
    """The served CTX sequence on the C++ manager, at the end of every context:
    create (lookup = prompt minus its last token, priority callback, expected prompt length),
    resume, per chunk resize + commit, stop_committing, then the tracker plans and drops.

    Checks: every plan exists (plan_none = plan_skipped = 0); each turn reuses its parent's
    endpoint; the kept endpoints and their rewind regions stay fully reusable; a dropped
    endpoint whose window no later plan covers loses its window.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU and the C++ KV cache manager")
    m = _manager_module()
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        HostCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
    )

    tpb, window, rewind, keep, base = 16, 64, 32, 2, 205
    step, turns, chunk = case["step"], case["turns"], case["chunk"]
    layers = [
        AttentionLayerConfig(
            layer_id=LayerId(i),
            buffers=[
                BufferConfig(role=DataRole("key"), size=8192),
                BufferConfig(role=DataRole("value"), size=8192),
            ],
            sliding_window_size=window if i == 0 else None,
        )
        for i in range(2)
    ]
    config_kwargs = dict(
        tokens_per_block=tpb,
        cache_tiers=[
            GpuCacheTierConfig(quota=case["gpu"]),
            HostCacheTierConfig(quota=64 << 20),
        ],
        layers=layers,
        enable_partial_reuse=True,
        enable_partial_commit=True,
    )
    if case["ratio"] is not None:
        config_kwargs["initial_pool_ratio"] = case["ratio"]
    manager = KVCacheManager(KVCacheManagerConfig(**config_kwargs))
    tracker = SwaStaleEndpointTracker(
        keep=keep, tokens_per_block=tpb, window_size=window, rewind_tokens=rewind, log_every=0
    )
    stream = torch.cuda.Stream()
    ids = list(range(1, base + turns * step + 1))
    ends = [base + k * step for k in range(1, turns + 1)]
    unique = iter(range(10_000_000, 20_000_000))
    dropped: List[int] = []

    def continuation(end: int) -> List[int]:
        """A prompt that shares ids[:end] and then diverges from every committed prompt."""
        return ids[:end] + [next(unique) for _ in range(tpb)]

    def probe(end: int) -> int:
        return manager.probe_reuse(None, continuation(end))

    def run_request(tokens: List[int], plan: bool) -> int:
        e = len(tokens)
        kwargs = {}
        if plan:
            kwargs["custom_priority_callback"] = m._swa_endpoint_priority(e, tpb, rewind)
        kv_cache = manager.create_kv_cache(
            None, tokens[: e - 1], expected_prompt_length=e, **kwargs
        )
        assert kv_cache.resume(stream.cuda_stream)
        reused = kv_cache.num_committed_tokens
        pos = reused
        while pos < e:
            end = min(pos + chunk, e)
            assert kv_cache.resize(max(kv_cache.capacity, end))
            kv_cache.commit(tokens[kv_cache.num_committed_tokens : end])
            pos = end
        kv_cache.stop_committing()
        if plan:
            turn = len(probes)
            tracker.commit(
                tokens=tokens,
                num_tokens=e,
                seed=0,
                num_committed_tokens=kv_cache.num_committed_tokens,
                plan_drop=lambda: (
                    None
                    if (h := kv_cache.plan_committed_block_drop()) is None
                    else _RecordingHandle(h, turn, dropped)
                ),
                reused_tokens=reused,
            )
        stream.synchronize()
        kv_cache.close()
        return reused

    probes: List[tuple] = []  # per committed endpoint: (endpoint probe, rewind-region probe)
    try:
        for k, e in enumerate(ends):
            reused = run_request(ids[:e], plan=True)
            if k > 0:
                # The served property: the turn resumed at its parent's endpoint.
                assert reused >= ends[k - 1] - tpb, (k, reused)
            probes.append((probe(e), probe(e - rewind // 2)))
            assert probes[-1][0] >= e - tpb and probes[-1][1] >= e - rewind // 2 - tpb, probes[-1]
            for c in range(case["churn"]):
                filler = [next(unique) for _ in range(60 * tpb + 7)]
                run_request(filler, plan=False)
            for j in range(k + 1):
                now = (probe(ends[j]), probe(ends[j] - rewind // 2))
                if j > k - keep:
                    assert now == probes[j], f"kept endpoint {j} degraded after turn {k}"
                elif j in dropped and step > window + rewind:
                    assert now[0] < probes[j][0], f"endpoint {j} still matched after its drop"
        s = tracker.stats
        assert s["plan_none"] == 0 and s["plan_skipped"] == 0
        assert s["plan_error"] == 0 and s["drop_error"] == 0
        assert s["parent_miss"] == 0 and s["resume_dropped"] == 0
        assert s["linked"] == turns - 1
        assert all(j <= turns - 1 - keep for j in dropped)
        if step > window + rewind:
            assert sorted(dropped) == list(range(turns - keep))
        else:
            assert dropped and s["deferred"] > 0
    finally:
        tracker.clear()
        manager.shutdown()
