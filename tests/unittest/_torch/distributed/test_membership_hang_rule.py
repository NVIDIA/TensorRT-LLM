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
"""The authority's hang rule over every small deployment's failures (no GPU, no I/O).

A rank blocked in a collective waits on a peer, so only the end of a wait-for chain is hung. Each
case gives every rank one role and derives the waits that its reports carry:

- waiting: running and blocked, waits on the blockers, or on the waiting ranks that wait on them;
- slow: running and making progress, waits on nobody;
- idle: running in a call with its hang detector paused, as rank 0 waiting for requests;
- crashed: its connection closes and it reports nothing more;
- hung: its main thread stops outside any wait, so its fresh reports wait on nobody;
- stopped: its reports stop, and the last one waits on nobody or on the ranks that now wait on it;
- paused: its reports stop, and its node supervisor reports its hang detector paused.

A rank whose node supervisor reports its hang detector paused is never hung outside a change, also
once its reports stop, so an idle or paused rank that others wait on is kept. A change waits on a
paused rank, which never replies, as on any other failed rank. The blockers are any nonempty set of
non-waiting ranks.
Ranks 1 and above are interchangeable to the rule, so their roles, the blockers and the chained
ranks are enumerated up to swapping ranks of one role. The waited rounds (``waited_flags``,
``waited_epochs``) and the install steps stay empty, so a rank's progress is its iteration.
"""

import itertools
import random

import pytest

from tensorrt_llm._torch.distributed.membership import (
    Authority,
    MembershipState,
    NodeReport,
    RankReport,
    Reply,
    authority,
)

pytestmark = pytest.mark.cpu_only

HANG_TIMEOUT_S = 1.0
STEP_S = HANG_TIMEOUT_S / 4
# Every case settles within four hang timeouts. The cases run one hang timeout longer.
SETTLE_S = 4 * HANG_TIMEOUT_S
TICKS = round((SETTLE_S + HANG_TIMEOUT_S) / STEP_S)

WAITING, SLOW, IDLE = "waiting", "slow", "idle"
CRASHED, HUNG, STOPPED, PAUSED = "crashed", "hung", "stopped", "paused"
SILENT = (STOPPED, PAUSED)
FAULTY = (CRASHED, HUNG, *SILENT)
RANK_ROLES = (WAITING, SLOW, IDLE, *FAULTY)

_reports = {}


def _report(**fields):
    """Return a cached report with ``fields``, since the cases send the same reports many times."""
    key = tuple(sorted(fields.items()))
    if key not in _reports:
        _reports[key] = RankReport(**fields)
    return _reports[key]


def _mask(ranks):
    return sum(1 << r for r in ranks)


def _groups(roles):
    """Return the ranks grouped by role, rank 0 on its own, since the other ranks are alike."""
    groups = [[0]]
    for r in range(1, len(roles)):
        if r > 1 and roles[r] == roles[r - 1]:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def _subsets(groups):
    """Yield one mask per subset of the ranks in ``groups``, up to swapping alike ranks."""
    for sizes in itertools.product(*(range(len(group) + 1) for group in groups)):
        yield _mask(r for group, size in zip(groups, sizes) for r in group[:size])


def _cases(world):
    """Yield ``(roles, waits, stale)`` for every role assignment and every wait they imply."""
    for rank0_role in RANK_ROLES:
        for rest in itertools.combinations_with_replacement(RANK_ROLES, world - 1):
            roles = (rank0_role,) + rest
            groups = _groups(roles)
            waiting = [g for g in groups if roles[g[0]] == WAITING]
            others = [g for g in groups if roles[g[0]] != WAITING]
            if not waiting:
                yield from _variants(roles, {})
                continue
            # Running ranks never wait on each other in a cycle, because collectives send before
            # they wait.
            for blockers in filter(None, _subsets(others)):
                # The chained ranks wait on the other waiting ranks, not on the blockers.
                for chained in _subsets(waiting):
                    first_level = _mask(r for g in waiting for r in g) & ~chained
                    if first_level:
                        waits = {
                            r: first_level if chained >> r & 1 else blockers
                            for g in waiting
                            for r in g
                        }
                        yield from _variants(roles, waits)


def _variants(roles, waits):
    stopped = [r for r, role in enumerate(roles) if role == STOPPED]
    yield roles, waits, {}
    if stopped:
        # A stopped rank's last report waits on the ranks that now wait on it.
        yield roles, waits, {s: _mask(r for r in waits if waits[r] >> s & 1) for s in stopped}


def _run(world, roles, waits, stale):
    """Drive the authority through the case and return ``(time, record)`` for each record."""
    auth = Authority(world, hang_timeout_s=HANG_TIMEOUT_S)
    everyone = (1 << world) - 1
    started = {r: RankReport(started=True) for r in range(world)}
    assert auth.tick(0.0, started, everyone).state == MembershipState.COMMITTED
    connected = everyone & ~_mask(r for r in range(world) if roles[r] == CRASHED)
    paused_mask = _mask(r for r, role in enumerate(roles) if role in (IDLE, PAUSED))
    published = []
    for tick in range(1, TICKS + 1):
        record = auth.record
        reports = {}
        for r, role in enumerate(roles):
            if role == CRASHED:
                continue
            if role in SILENT:
                reports[r] = _report(received_s=STEP_S, waiting_for=stale.get(r, 0))
            elif role == HUNG:
                reports[r] = _report(received_s=tick * STEP_S)
            else:
                reports[r] = _running_report(record, role, waits.get(r, 0), tick)
        now_s = tick * STEP_S
        node_report = NodeReport(paused_mask=paused_mask, received_s=now_s)
        new = auth.tick(now_s, reports, connected, [node_report])
        if new is not None:
            published.append((now_s, new))
        if new is not None and new.state == MembershipState.ABORTED:
            break
        # Once the running ranks commit without the failed ranks, they only make progress.
        settled = record.generation and record.state == MembershipState.COMMITTED
        if settled and now_s - published[-1][0] > HANG_TIMEOUT_S:
            break
    return published


def _running_report(record, role, waiting_for, tick):
    """Return a running rank's report, which replies to each change and resumes after the commit."""
    if record.generation:
        resumed = record.state == MembershipState.COMMITTED
        reply = {
            MembershipState.PROPOSE: Reply.READY,
            MembershipState.INSTALL: Reply.INSTALLED,
        }.get(record.state, Reply.NONE)
        return _report(
            received_s=tick * STEP_S,
            progress=tick if resumed else 0,
            reply=reply,
            reply_generation=record.generation,
        )
    if role == SLOW:
        return _report(received_s=tick * STEP_S, progress=tick)
    return _report(received_s=tick * STEP_S, waiting_for=waiting_for)


def _mismatch(world, roles, waits, stale, published):
    """Return why ``published`` breaks the hang rule for the case, or None."""
    everyone = (1 << world) - 1
    running = _mask(r for r in range(world) if roles[r] not in FAULTY)
    crashed = _mask(r for r in range(world) if roles[r] == CRASHED)
    paused = _mask(r for r in range(world) if roles[r] == PAUSED)
    waited_on = 0
    for mask in waits.values():
        waited_on |= mask
    # The ends of the wait-for chains: the failed ranks that a waiting rank waits on, except the
    # paused ones.
    roots = waited_on & everyone & ~running & ~crashed & ~paused
    if any(record.active_mask & running != running for _, record in published):
        return "removed a running rank"
    if not crashed and not roots:
        return "changed membership without a failure" if published else None
    if not published:
        return "missed the failure"
    first_s, first = published[0]
    removed = everyone & ~first.active_mask
    expected = crashed or roots
    if removed != expected:
        return f"first removed {removed:#b}, expected {expected:#b}"
    # The failures start with the first reports, one step in.
    if crashed:
        in_time = first_s == STEP_S
    else:
        in_time = HANG_TIMEOUT_S < first_s - STEP_S <= HANG_TIMEOUT_S + 2 * STEP_S
    if not in_time:
        return f"first change at {first_s} s"
    last_s, last = published[-1]
    if last_s > SETTLE_S:
        return f"still changing at {last_s} s"
    if running & 1:
        if last.state != MembershipState.COMMITTED or last.active_mask != running:
            return f"settled on {last}"
    elif last.state != MembershipState.ABORTED:
        return f"did not abort without rank 0: {last}"
    return None


@pytest.mark.parametrize("world", [2, 3, 4, 5])
def test_only_chain_ends_are_hung(world):
    """Remove exactly the crashed ranks or the ends of the wait-for chains, then every failed rank.

    A crash starts a change at once. Without one, the authority removes the hung or stopped ranks
    that waiting ranks wait on, one hang timeout after they stop. The change then waits on every
    other failed rank, which never replies, so the deployment settles on the running ranks, or
    aborts without rank 0. A failed rank that no running rank waits on is slow, not hung. An idle
    or paused rank is never hung outside a change, so it is kept while other ranks wait on it.
    """
    mismatches = []
    count = 0
    for roles, waits, stale in _cases(world):
        count += 1
        problem = _mismatch(world, roles, waits, stale, _run(world, roles, waits, stale))
        if problem is not None:
            mismatches.append(f"roles {roles}, waits {waits}, stale {stale}: {problem}")
    assert count > 0
    assert not mismatches, f"{len(mismatches)} of {count} cases: " + "\n".join(mismatches[:20])


def _reachability_chain_ends(world, waits):
    """Return the mask of the ranks that every rank they reach through the waits reaches back."""
    reach = dict(waits)
    changed = True
    while changed:
        changed = False
        for r in range(world):
            beyond = reach[r]
            for q in range(world):
                if reach[r] >> q & 1:
                    beyond |= reach[q]
            changed |= beyond != reach[r]
            reach[r] = beyond
    ends = 0
    for r in range(world):
        if all(reach[q] >> r & 1 for q in range(world) if reach[r] >> q & 1):
            ends |= 1 << r
    return ends


@pytest.mark.parametrize("world", [2, 3, 4])
def test_chain_end_matches_reachability(world):
    """Find the chain ends of every wait-for graph, cycles included.

    A rank is a chain end when every rank it reaches through the waits reaches it back.
    """
    pairs = [(r, q) for r in range(world) for q in range(world) if q != r]
    for edges in itertools.product((False, True), repeat=len(pairs)):
        waits = {r: 0 for r in range(world)}
        for (r, q), edge in zip(pairs, edges):
            waits[r] |= edge << q
        expected = _reachability_chain_ends(world, waits)
        assert authority._chain_ends(range(world), waits) == expected, waits


@pytest.mark.parametrize("world", [8, 32, 64])
@pytest.mark.parametrize("density", [0.02, 0.1, 0.5])
def test_chain_ends_match_reachability_on_random_graphs(world, density):
    """Find the chain ends of random wait-for graphs of many ranks, with chains and cycles."""
    rng = random.Random(world * 1000 + round(density * 100))
    for _ in range(20):
        waits = {
            r: sum(1 << q for q in range(world) if q != r and rng.random() < density)
            for r in range(world)
        }
        expected = _reachability_chain_ends(world, waits)
        assert authority._chain_ends(range(world), waits) == expected, waits


def test_hang_timer_restarts_after_stall_ends():
    """Time a hang from the start of the latest stall, not from an earlier stall that ended."""
    world, hung = 4, 2
    auth = Authority(world, hang_timeout_s=HANG_TIMEOUT_S)
    everyone = (1 << world) - 1
    started = {r: RankReport(started=True) for r in range(world)}
    assert auth.tick(0.0, started, everyone).state == MembershipState.COMMITTED
    # Rank 2 makes no progress, and the others wait on it twice, first for less than the timeout.
    first_stall, second_stall = range(1, 4), 6
    for tick in range(1, TICKS + 1):
        waiting_for = 1 << hung if tick in first_stall or tick >= second_stall else 0
        reports = {
            r: _report(received_s=tick * STEP_S, waiting_for=waiting_for) for r in range(world)
        }
        reports[hung] = _report(received_s=tick * STEP_S)
        record = auth.tick(tick * STEP_S, reports, everyone)
        if record is not None:
            break
    assert record is not None
    assert record.active_mask == everyone & ~(1 << hung)
    stalled_s = tick * STEP_S - second_stall * STEP_S
    assert HANG_TIMEOUT_S < stalled_s <= HANG_TIMEOUT_S + 2 * STEP_S


@pytest.mark.parametrize("node_report", ["fresh", "stale"])
def test_paused_rank_kept_only_while_node_report_fresh(node_report):
    """Keep a paused rank that others wait on only while its node supervisor's report is fresh.

    Rank 2 stops reporting in a paused call. Its node supervisor's paused mask names it, but a
    supervisor that stopped reporting no longer vouches for the pause after half the hang timeout.
    The rank is then hung like a stopped rank, one hang timeout after its last report.
    """
    world, paused = 4, 2
    auth = Authority(world, hang_timeout_s=HANG_TIMEOUT_S)
    everyone = (1 << world) - 1
    started = {r: RankReport(started=True) for r in range(world)}
    assert auth.tick(0.0, started, everyone).state == MembershipState.COMMITTED
    record = None
    for tick in range(1, TICKS + 1):
        now_s = tick * STEP_S
        reports = {r: _report(received_s=now_s, waiting_for=1 << paused) for r in range(world)}
        reports[paused] = _report(received_s=STEP_S)
        received_s = now_s if node_report == "fresh" else STEP_S
        node = NodeReport(paused_mask=1 << paused, received_s=received_s)
        record = auth.tick(now_s, reports, everyone, [node])
        if record is not None:
            break
    if node_report == "fresh":
        assert record is None
    else:
        assert record.active_mask == everyone & ~(1 << paused)
        assert HANG_TIMEOUT_S < now_s - STEP_S <= HANG_TIMEOUT_S + 2 * STEP_S


@pytest.mark.parametrize("after_pause", ["resumes", "stuck"])
def test_hang_timer_starts_when_pause_ends(after_pause):
    """Time a hang from the end of a pause, not from the last report sent before it.

    Rank 2 stops reporting in a paused call for two hang timeouts while the others wait on it. Its
    node supervisor reports the pause ended one tick before the rank's own reports resume. A rank
    that then resumes is kept, and one that stays stuck is hung one hang timeout after the pause.
    """
    world, paused = 4, 2
    auth = Authority(world, hang_timeout_s=HANG_TIMEOUT_S)
    everyone = (1 << world) - 1
    started = {r: RankReport(started=True) for r in range(world)}
    assert auth.tick(0.0, started, everyone).state == MembershipState.COMMITTED
    last_paused_tick = round(2 * HANG_TIMEOUT_S / STEP_S)
    record = None
    for tick in range(1, TICKS + 1):
        now_s = tick * STEP_S
        resumed = after_pause == "resumes" and tick > last_paused_tick + 1
        waiting_for = 0 if resumed else 1 << paused
        reports = {r: _report(received_s=now_s, waiting_for=waiting_for) for r in range(world)}
        reports[paused] = (
            _report(received_s=now_s, progress=tick) if resumed else _report(received_s=STEP_S)
        )
        paused_mask = 1 << paused if tick <= last_paused_tick else 0
        node = NodeReport(paused_mask=paused_mask, received_s=now_s)
        record = auth.tick(now_s, reports, everyone, [node])
        if record is not None:
            break
    if after_pause == "resumes":
        assert record is None
    else:
        assert record.active_mask == everyone & ~(1 << paused)
        stalled_s = now_s - last_paused_tick * STEP_S
        assert HANG_TIMEOUT_S < stalled_s <= HANG_TIMEOUT_S + 2 * STEP_S
