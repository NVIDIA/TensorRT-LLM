# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Home-wait admission of ConversationAwareADPRouter (attention_dp_config.kv_cache_routing_home_wait_rounds).

Without it, a returning conversation whose home rank is at max_num_active_requests overflows round-robin to a rank
without its cached prefix. With home-wait on, pop_requests admits it only into a free slot on its home rank (others
may pass it), and it overflows only after it was held home_wait_rounds rounds while another rank had room.
"""

import random
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue, PriorityWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ADPRouter,
    ConversationAwareADPRouter,
)
from tensorrt_llm._torch.pyexecutor.scheduler.rank_state import RankState
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.llmapi.llm_args import AttentionDpConfig

pytestmark = pytest.mark.cpu_only


class _Req(MagicMock):
    @property
    def num_input_tokens(self):
        return len(self.input_token_ids)


def _dist(tp_size, tp_rank=0):
    dist = MagicMock()
    dist.tp_rank = tp_rank
    dist.tp_size = tp_size
    dist.has_cp_helix = False
    dist.mapping.has_pp.return_value = False
    return dist


def _item(req_id, conv, num_tokens=100, target=None, relax=True, children=0, priority=0.5):
    item = MagicMock()
    item.id = req_id
    item.child_req_ids = (
        list(range(10_000 + req_id * 10, 10_000 + req_id * 10 + children)) if children else None
    )
    sp = MagicMock()
    sp.attention_dp_rank = target
    sp.attention_dp_relax = relax
    sp.subagent_affinity_id = None
    item.request = _Req()
    item.request.py_scheduling_params = sp
    item.request.py_conversation_params = ConversationParams(conversation_id=conv) if conv else None
    item.request.input_token_ids = list(range(num_tokens))
    item.request.priority = priority
    return item


def _states(active, tokens=None):
    tokens = tokens or [a * 100 for a in active]
    return [
        RankState(rank=r, num_active_requests=active[r], num_active_tokens=tokens[r])
        for r in range(len(active))
    ]


def _router(tp_size=4, rounds=4, placement="round_robin", tp_rank=0):
    return ConversationAwareADPRouter(
        dist=_dist(tp_size, tp_rank), new_conv_placement=placement, home_wait_rounds=rounds
    )


def _round(router, queue, active, cap, max_new=None):
    """One admission round: pop + route. Returns {request id: rank}."""
    states = _states(active)
    free = sum(max(0, cap - a) for a in active)
    chosen, placement = router.pop_requests(
        queue, states, free if max_new is None else max_new, cap
    )
    assign, expected = router.route_requests(states, chosen, cap, placement=placement)
    pos = {it.id: rank for rank, lst in assign.items() for it in lst}
    for r in range(len(active)):
        assert active[r] + len(assign[r]) <= max(cap, active[r]), "a rank went over the hard cap"
        assert expected >= active[r] + len(assign[r]), "padding invariant"
    return pos


def _bind(router, conv_ranks, tp_size=4, cap=8):
    """Bind conversations to ranks by routing their first turns one rank at a time."""
    rid = 1_000
    for conv, rank in conv_ranks.items():
        active = [cap] * tp_size
        active[rank] = 0
        queue = FCFSWaitingQueue()
        queue.add_request(_item(rid, conv))
        pos = _round(router, queue, active, cap)
        assert pos == {rid: rank}
        rid += 1


@pytest.mark.parametrize("rounds", [None, 64])
def test_config_field_reaches_the_router(rounds):
    config = AttentionDpConfig(
        kv_cache_routing_conversation_affinity=True, kv_cache_routing_home_wait_rounds=rounds
    )
    router = ADPRouter.create(
        dist=_dist(4), has_seq_slot_headroom=False, attention_dp_config=config
    )
    assert isinstance(router, ConversationAwareADPRouter)
    assert router._home_wait_rounds == rounds


def test_config_field_rejects_zero():
    with pytest.raises(ValueError):
        AttentionDpConfig(
            kv_cache_routing_conversation_affinity=True, kv_cache_routing_home_wait_rounds=0
        )


def test_config_field_requires_conversation_affinity():
    with pytest.raises(ValueError, match="kv_cache_routing_conversation_affinity"):
        AttentionDpConfig(kv_cache_routing_home_wait_rounds=4)


def test_off_is_the_base_fifo_pop():
    """Without home-wait, pop_requests is the base class's FIFO pop (no plan)."""
    router = ConversationAwareADPRouter(dist=_dist(4))
    ref = ConversationAwareADPRouter(dist=_dist(4))
    q1, q2 = FCFSWaitingQueue(), FCFSWaitingQueue()
    for i in range(6):
        q1.add_request(_item(i, f"c{i}"))
        q2.add_request(_item(i, f"c{i}"))
    got, plan = router.pop_requests(q1, _states([0, 0, 0, 0]), 4, 8)
    want, _ = ADPRouter.pop_requests(ref, q2, _states([0, 0, 0, 0]), 4, 8)
    assert [x.id for x in got] == [x.id for x in want] == [0, 1, 2, 3]
    assert plan == {}
    assert [x.id for x in q1] == [4, 5]


def test_bound_request_waits_for_home_while_others_pass():
    router = _router(rounds=10)
    _bind(router, {"A": 0, "B": 1})
    queue = FCFSWaitingQueue()
    queue.add_requests([_item(1, "A"), _item(2, "C"), _item(3, "B")])
    cap = 2
    pos = _round(router, queue, [2, 0, 0, 0], cap)  # rank 0 (A's home) is full
    assert 1 not in pos, "A must wait for its home rank"
    assert pos[3] == 1, "B goes home"
    assert pos[2] != 0, "the new conversation C goes where there is room"
    assert [x.id for x in queue] == [1], "A stays at the front"
    pos = _round(router, queue, [1, 1, 1, 1], cap)  # a slot frees on rank 0
    assert pos == {1: 0}
    assert len(queue) == 0


def test_overflow_after_rounds_keeps_binding():
    router = _router(rounds=2)
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_request(_item(1, "A"))
    cap = 1
    assert _round(router, queue, [1, 0, 0, 0], cap) == {}  # held, age 1
    assert _round(router, queue, [1, 0, 0, 0], cap) == {}  # held, age 2
    pos = _round(router, queue, [1, 0, 0, 0], cap)  # overflow
    assert list(pos) == [1] and pos[1] != 0
    assert router._conv_to_rank["A"] == 0, "the binding stays home, as without home-wait"


def test_no_aging_when_no_rank_has_room():
    router = _router(rounds=1)
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_request(_item(1, "A"))
    for _ in range(5):
        assert _round(router, queue, [1, 1, 1, 1], 1, max_new=0) == {}
        assert _round(router, queue, [1, 1, 1, 1], 1, max_new=1) == {}
    assert router._home_wait_age.get(1, 0) == 0
    assert _round(router, queue, [0, 1, 1, 1], 1) == {1: 0}


def test_a_request_the_round_does_not_reach_keeps_its_age():
    """The bound on waiting holds even across rounds that stop before the held request."""
    router = _router(rounds=10)
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_request(_item(1, "A"))
    assert _round(router, queue, [1, 0, 0, 0], 1) == {}  # A held while other ranks had room
    assert router._home_wait_age[1] == 1
    # A new conversation ahead of A takes the round's whole budget, so the round stops before A (still in the window).
    queue.prepend_requests([_item(2, "C")])
    assert list(_round(router, queue, [1, 0, 0, 0], 1, max_new=1)) == [2]
    assert router._home_wait_age == {1: 1}


def test_window_extends_past_held_requests():
    """A round looks past a window of held requests while a rank has room."""
    router = _router(rounds=10)
    router.HOME_WAIT_WINDOW = 1
    _bind(router, {"A": 0, "B": 1})
    queue = FCFSWaitingQueue()
    queue.add_requests([_item(1, "A"), _item(2, "B"), _item(3, "C"), _item(4, "D")])
    pos = _round(router, queue, [1, 1, 0, 0], 1, max_new=1)
    assert list(pos) == [3] and pos[3] in (2, 3)
    assert router._home_wait_age == {1: 1, 2: 1}
    # Only the requests up to the last admitted one were popped and put back.
    assert [x.id for x in queue] == [1, 2, 4]


def test_a_request_out_of_the_window_loses_its_age():
    router = _router(rounds=10)
    router.HOME_WAIT_WINDOW = 1
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_request(_item(1, "A"))
    assert _round(router, queue, [1, 0, 0, 0], 1) == {}  # A held while other ranks had room
    assert router._home_wait_age == {1: 1}
    # A new request goes ahead of A and uses the round's whole budget, so the window never reaches A.
    queue.prepend_requests([_item(2, "C")])
    assert list(_round(router, queue, [1, 0, 0, 0], 1, max_new=1)) == [2]
    assert router._home_wait_age == {}
    assert [x.id for x in queue] == [1]


def test_unbound_requests_are_placed_as_without_home_wait():
    """Home-wait plans only returning conversations; route_requests places the rest by its own rules."""
    routers = [_router(rounds=4), ConversationAwareADPRouter(dist=_dist(4))]
    results = []
    for router in routers:
        queue = FCFSWaitingQueue()
        queue.add_requests([_item(i, f"n{i}", children=i % 2) for i in range(6)])
        results.append(_round(router, queue, [2, 0, 1, 3], 4))
        assert router._conv_to_rank == {f"n{i}": results[-1][i] for i in range(6)}
    assert results[0] == results[1]


def test_a_removed_request_drops_its_age():
    router = _router(rounds=10)
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_requests([_item(1, "A"), _item(2, "A")])
    assert _round(router, queue, [1, 0, 0, 0], 1) == {}
    assert router._home_wait_age == {1: 1, 2: 1}
    queue.remove_by_ids({1})  # cancelled while held
    assert _round(router, queue, [1, 0, 0, 0], 1) == {}
    assert router._home_wait_age == {2: 2}


def test_unplanned_requests_are_placed_by_the_real_token_load():
    """route_requests places a request pop_requests did not plan as without home-wait, on shared token load."""
    router = _router(tp_size=2, placement="least_tokens")
    states = _states([0, 0], tokens=[1000, 0])
    assign, _ = router.route_requests(states, [_item(1, "X")], 8, placement={999: 0})
    assert [it.id for it in assign[1]] == [1]


def test_new_conversation_bound_in_the_same_round():
    router = _router(rounds=4)
    queue = FCFSWaitingQueue()
    queue.add_requests([_item(1, "X"), _item(2, "Y"), _item(3, "X")])
    pos = _round(router, queue, [0, 0, 0, 0], 8)
    assert pos[1] == pos[3] == router._conv_to_rank["X"]


def test_admission_groups_are_respected():
    """A later admission group passes a held request of an earlier group, but not into its home rank."""
    router = _router(rounds=10)
    _bind(router, {"A": 0, "B": 1, "E": 0})
    queue = FCFSWaitingQueue()
    # A needs 2 slots and its home rank 0 has 1. Its own group's F may still take that slot; group 1's E may not.
    queue.add_requests([_item(1, "A", children=1), _item(4, "F"), _item(2, "B"), _item(3, "E")])
    queue.admission_groups = lambda reqs: [0, 0, 1, 1][: len(reqs)]
    pos = _round(router, queue, [1, 0, 0, 0], 2)
    assert 1 not in pos and 4 in pos and pos[2] == 1 and 3 not in pos
    assert [x.id for x in queue] == [1, 3]
    assert router._home_wait_age == {1: 1, 3: 1}


def test_continuous_priorities_do_not_block_other_ranks():
    """Priority queue with one group per request (output-length predictor): a request held for a full home rank does
    not stop lower-priority requests bound to other ranks, and a lower-priority request does not pass it at home."""
    router = _router(rounds=10)
    _bind(router, {"A": 0, "B": 1, "C": 2, "D": 0})
    queue = PriorityWaitingQueue()
    queue.add_requests(
        [
            _item(1, "A", priority=0.9),
            _item(2, "B", priority=0.8),
            _item(3, "C", priority=0.7),
            _item(4, "D", priority=0.6),
        ]
    )
    assert queue.admission_groups(queue.peek_requests(4)) == [0, 1, 2, 3]
    pos = _round(router, queue, [1, 0, 0, 0], 1)
    assert pos == {2: 1, 3: 2}
    assert [x.id for x in queue] == [1, 4]
    assert router._home_wait_age == {1: 1, 4: 1}
    # Rank 0 frees a slot: the higher-priority A takes it, D keeps waiting.
    assert _round(router, queue, [0, 1, 1, 0], 1) == {1: 0}
    assert [x.id for x in queue] == [4]


def test_an_aged_lower_priority_request_does_not_take_a_held_requests_home():
    """Once aged, a lower-priority request is admitted unplanned; route_requests must still keep it off the home rank
    of a higher-priority request held in the same round, so each rank admits in priority order."""
    router = _router(tp_size=2, rounds=2)
    _bind(router, {"A": 0, "E": 0}, tp_size=2, cap=2)
    queue = PriorityWaitingQueue()
    queue.add_request(_item(2, "E", priority=0.5))
    for _ in range(2):
        assert _round(router, queue, [2, 1], 2) == {}  # E held while rank 1 has room
    queue.add_request(_item(1, "A", priority=0.9, children=1))  # needs both slots of rank 0
    pos = _round(router, queue, [1, 1], 2)
    assert pos == {2: 1}, pos
    assert [x.id for x in queue] == [1]
    assert router._conv_to_rank["E"] == 0


def test_round_passes_over_a_bounded_number_of_requests():
    """A deep queue of held requests costs a round at most HOME_WAIT_MAX_PASSED requests, not the whole queue."""
    router = _router(rounds=10_000)
    router.HOME_WAIT_WINDOW = 4
    router.HOME_WAIT_MAX_PASSED = 8
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_requests([_item(i, "A") for i in range(1, 1001)])
    queue.add_request(_item(2000, "C"))
    peeked = []
    peek = queue.peek_requests
    queue.peek_requests = lambda k: peeked.append(k) or peek(k)
    for _ in range(3):
        assert _round(router, queue, [1, 0, 0, 0], 1) == {}
    assert max(peeked) <= 2 * router.HOME_WAIT_MAX_PASSED
    assert router._home_wait_age == {i: 3 for i in range(1, 9)}
    assert len(queue) == 1001


def test_children_need_their_slots_on_one_rank():
    router = _router(rounds=10)
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_request(_item(1, "A", children=2))
    assert _round(router, queue, [6, 0, 0, 0], 8) == {}  # home has 2 free, needs 3
    assert _round(router, queue, [5, 0, 0, 0], 8) == {1: 0}


def test_a_held_request_with_children_ages_while_another_rank_has_a_slot():
    """An n > 1 turn ages whenever another rank has a free slot (where route_requests overflows it), not only when one
    rank has all of its slots free, so a steady single-slot stream cannot hold it past home_wait_rounds."""
    router = _router(tp_size=2, rounds=4)
    _bind(router, {"A": 0}, tp_size=2, cap=4)
    queue = FCFSWaitingQueue()
    queue.add_request(_item(1, "A", children=1))  # needs 2 slots
    admitted_at = None
    for i in range(20):
        queue.add_request(_item(100 + i, f"s{i}"))
        pos = _round(router, queue, [3, 3], 4, max_new=2)
        if 1 in pos:
            admitted_at = i
            break
    assert admitted_at == 4, (admitted_at, router._home_wait_age)


def test_unbound_strict_requests_without_room_do_not_stop_the_round():
    """Strict-rank requests whose rank is full are set aside as by the base pop, not counted against the round's
    limit on held requests, so a relaxed request behind any number of them is still admitted."""
    router = _router(rounds=4)
    queue = FCFSWaitingQueue()
    queue.add_requests(
        [_item(i, None, target=0, relax=False) for i in range(1, router.HOME_WAIT_MAX_PASSED + 45)]
    )
    queue.add_request(_item(5000, None))
    pos = _round(router, queue, [8, 2, 2, 2], 8)
    assert list(pos) == [5000] and pos[5000] != 0
    assert len(queue) == router.HOME_WAIT_MAX_PASSED + 44
    assert router._home_wait_age == {}


def test_a_request_no_rank_can_take_whole_is_admitted_at_once():
    """1 + children over the per-rank cap fits no home, so it is not held (it would never age): as without home-wait."""
    router = _router(rounds=10)
    _bind(router, {"A": 0})
    queue = FCFSWaitingQueue()
    queue.add_request(_item(1, "A", children=2))  # needs 3 slots, the cap is 2
    pos = _round(router, queue, [0, 0, 0, 0], 2)
    assert list(pos) == [1]
    assert len(queue) == 0
    assert router._conv_to_rank["A"] == 0


def test_no_kv_balance_admits_in_queue_order():
    """Warmup and the benchmark fill phase pass kv_balance=False: FIFO, no request held for its home rank."""
    router = _router(rounds=10)
    ref = ConversationAwareADPRouter(dist=_dist(4))
    _bind(router, {"A": 0})
    _bind(ref, {"A": 0})
    q1, q2 = FCFSWaitingQueue(), FCFSWaitingQueue()
    for queue in (q1, q2):
        queue.add_requests([_item(1, "A"), _item(2, "C")])
    states = _states([1, 0, 0, 0])
    got, plan = router.pop_requests(q1, states, 3, 1, kv_balance=False)
    want, _ = ref.pop_requests(q2, states, 3, 1, kv_balance=False)
    assert [x.id for x in got] == [x.id for x in want] == [1, 2]
    assert plan == {}
    assert router._home_wait_age == {}


def test_strict_explicit_rank_waits_relaxed_one_moves():
    router = _router(rounds=10)
    queue = FCFSWaitingQueue()
    queue.add_requests(
        [_item(1, None, target=2, relax=False), _item(2, None, target=2, relax=True)]
    )
    pos = _round(router, queue, [0, 0, 1, 0], 1)
    assert 1 not in pos and pos[2] != 2
    assert [x.id for x in queue] == [1]


def test_all_ranks_decide_alike():
    """Every rank runs the router on identical inputs; their decisions must match."""
    rng = random.Random(7)
    routers = [_router(tp_size=4, rounds=3, tp_rank=r) for r in range(4)]
    queues = [FCFSWaitingQueue() for _ in routers]
    rid = 0
    for _ in range(60):
        for _ in range(rng.randint(0, 5)):
            conv = f"c{rng.randint(0, 20)}"
            ntok = rng.randint(10, 500)
            for q in queues:
                q.add_request(_item(rid, conv, num_tokens=ntok))
            rid += 1
        active = [rng.randint(0, 4) for _ in range(4)]
        results = [_round(rt, q, active, 4) for rt, q in zip(routers, queues)]
        assert all(res == results[0] for res in results)
        assert all([x.id for x in q] == [x.id for x in queues[0]] for q in queues)


def _closed_loop(home_wait_rounds, seed=3, rounds=400, tp_size=4, cap=3, convs=24, p_done=0.3):
    """Each conversation has one request at a time; a rank finishes each of its requests with prob p_done per round.
    Saturated: convs > tp_size x cap. Returns (off-home placements, placements)."""
    rng = random.Random(seed)
    router = ConversationAwareADPRouter(dist=_dist(tp_size), home_wait_rounds=home_wait_rounds)
    queue = FCFSWaitingQueue()
    rid = 0
    for c in range(convs):
        queue.add_request(_item(rid, f"k{c}"))
        rid += 1
    running = [[] for _ in range(tp_size)]  # conversation ids active per rank
    home, off, placed = {}, 0, 0
    for _ in range(rounds):
        for r in range(tp_size):  # completions: the conversation's next turn re-enters the queue
            keep = []
            for conv in running[r]:
                if rng.random() < p_done:
                    queue.add_request(_item(rid, conv))
                    rid += 1
                else:
                    keep.append(conv)
            running[r] = keep
        active = [len(x) for x in running]
        states = _states(active)
        free = sum(max(0, cap - a) for a in active)
        chosen, plan = router.pop_requests(queue, states, free, cap)
        assign, _ = router.route_requests(states, chosen, cap, placement=plan)
        for r, lst in assign.items():
            for it in lst:
                conv = it.request.py_conversation_params.conversation_id
                if conv in home:
                    placed += 1
                    off += home[conv] != r
                else:
                    home[conv] = r
                running[r].append(conv)
        assert all(len(x) <= cap for x in running)
    return off, placed


def test_closed_loop_without_home_wait_misplaces_with_it_does_not():
    off_base, placed_base = _closed_loop(None)
    off_hw, placed_hw = _closed_loop(10_000)
    assert placed_base > 100 and placed_hw > 100
    assert off_base / placed_base > 0.3, (
        f"without home-wait only {off_base}/{placed_base} turns went off-home"
    )
    assert off_hw == 0, f"home-wait placed {off_hw}/{placed_hw} turns off-home"
