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
"""Pooled generation dispatch: serve/seat_pool.py and router.PooledLoadBalancingRouter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import msgpack
import pytest
from fastapi import HTTPException

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    RouterConfig,
    ServerRole,
    extract_disagg_cfg,
)
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import (
    POOLED_RESERVATION_TIMEOUT_S,
    DisaggCoordinatorService,
)
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams
from tensorrt_llm.serve.router import (
    POOLED_SELECT_TIMEOUT_S,
    CoordinatorDelegatingRouter,
    LoadBalancingRouter,
    PooledLoadBalancingRouter,
    Router,
    build_disagg_routers,
    create_router,
)
from tensorrt_llm.serve.seat_pool import NoGenerationSeatError, SeatPool, priority_or_default

pytestmark = pytest.mark.cpu_only


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ---------------------------------------------------------------------------------- SeatPool


def test_priority_or_default():
    assert priority_or_default(0.25) == 0.25 and priority_or_default(1) == 1.0
    for bad in (None, True, "0.3", float("nan"), -0.1, 1.5):
        assert priority_or_default(bad) == DEFAULT_REQUEST_PRIORITY


@pytest.mark.parametrize("seats, cap", [(0, None), (True, None), (1.5, None), (1, 0.0), (1, -1.0)])
def test_pool_rejects_bad_options(seats, cap):
    with pytest.raises(ValueError):
        SeatPool(seats, cap)


def test_pool_order_highest_priority_then_arrival():
    pool = SeatPool(1, clock=_Clock())
    for rid, priority in [(1, 0.3), (2, 0.9), (3, 0.9), (4, None)]:
        pool.add(rid, priority, None, None)
    order = []
    while len(pool):
        waiter, servers = pool.pick(["s0"])
        assert servers == ["s0"]
        pool.record_grant(waiter)
        order.append(waiter.rid)
    assert order == [2, 3, 4, 1]
    assert pool.placed == 4 and pool.reordered == 3


def test_pool_cap_puts_aged_waiters_first_oldest_first():
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=60.0, clock=clock)
    pool.add(1, 0.1, None, None)
    clock.now = 10.0
    pool.add(2, 0.2, None, None)
    clock.now = 50.0
    pool.add(3, 0.9, None, None)
    clock.now = 69.0
    assert pool.pick(["s"])[0].rid == 1  # only request 1 has waited 60 s
    clock.now = 70.0
    first = pool.pick(["s"])[0]
    pool.record_grant(first)
    assert first.rid == 1 and pool.aged == 1
    assert pool.pick(["s"])[0].rid == 2


def test_pool_ages_from_the_orchestrator_arrival():
    """A request past its context phase has waited since it reached the orchestrator."""
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=60.0, clock=clock)
    pool.add(1, 0.9, None, None)
    pool.add(2, 0.1, None, None, orig_elapsed_s=55.0)
    assert pool.pick(["s"])[0].rid == 1
    clock.now = 5.0
    assert pool.pick(["s"])[0].rid == 2  # 60 s since its arrival, 5 s in the pool


def test_pool_expires_at_the_kv_transfer_timeout():
    """Every waiter gets the deadline, counted from its entry here (generation_first ones too).

    It comes KV_DEADLINE_MARGIN early: the C++ transceiver counts the timeout from the KV send.
    """
    clock = _Clock()
    pool = SeatPool(1, clock=clock, kv_transfer_timeout_s=60.0)
    pool.add(1, 0.5, None, None, orig_elapsed_s=10.0)
    clock.now = 10.0
    pool.add(2, 0.5, None, None)  # no context phase yet: generation_first
    clock.now = 56.9
    assert pool.expire() == [] and pool.remaining_s(pool.waiter(1)) == pytest.approx(0.1)
    clock.now = 57.0
    expired = pool.expire()
    assert [w.rid for w in expired] == [1] and pool.expired == 1
    assert "KV transfer timeout (60 s less a 5% margin" in pool.expiry_message(expired[0])
    assert len(pool) == 1 and pool.remaining_s(pool.waiter(2)) == pytest.approx(10.0)
    clock.now = 67.0
    assert [w.rid for w in pool.expire()] == [2] and len(pool) == 0
    # Without the timeout nothing expires on its own.
    plain = SeatPool(1, clock=clock)
    plain.add(3, 0.5, None, None, orig_elapsed_s=10.0)
    clock.now = 1e6
    assert plain.expire() == []


def test_pool_ages_by_arrival_then_seq():
    """Aged waiters go by arrival at the orchestrator, like AgingPriorityWaitingQueue."""
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=60.0, clock=clock)
    pool.add(1, 0.5, None, None)  # arrival 0
    clock.now = 1.0
    pool.add(2, 0.5, None, None, orig_elapsed_s=30.0)  # arrival -29, joined later
    clock.now = 60.0
    assert pool.pick(["s"])[0].rid == 2
    assert pool.sort_key(pool.waiter(2), 60.0) < pool.sort_key(pool.waiter(1), 60.0)


def test_pool_pick_does_not_scan_every_waiter(monkeypatch):
    """A placement looks at one waiter per excluded server, not at the whole queue."""
    clock = _Clock()
    pool = SeatPool(1, max_wait_s=600.0, clock=clock, kv_transfer_timeout_s=60.0)
    for rid in range(2000):
        pool.add(rid, (rid % 97) / 97.0, "ctx0" if rid % 2 else None, None)
    calls = []
    real = pool.sort_key
    monkeypatch.setattr(pool, "sort_key", lambda w, now: calls.append(w) or real(w, now))
    order = []
    while len(pool) > 1990:
        waiter, _ = pool.pick(["s"])
        pool.record_grant(waiter)
        order.append(waiter.priority)
    assert len(calls) <= 2 * 10
    assert order == sorted(order, reverse=True) and order[0] == 96 / 97.0
    # Removed and placed waiters are never picked again, and expiry pops only the expired.
    for rid in range(0, 2000, 3):
        pool.remove(rid)
    live = len(pool)
    assert 0 < live == len(pool.waiters()) < 1990
    clock.now = 56.0  # the deadline is KV_DEADLINE_MARGIN before the 60 s timeout
    assert pool.expire() == []
    clock.now = 57.0
    assert len(pool.expire()) == live and len(pool) == 0 and pool.pick(["s"]) is None


@pytest.mark.parametrize("cap", [None, 600.0])
def test_pool_heaps_stay_bounded_under_a_queue_that_never_drains(cap):
    """Granted waiters leave both heaps, with or without a wait cap: no growth per placement."""
    pool = SeatPool(1, max_wait_s=cap, clock=_Clock())
    live = 9
    for rid in range(live):
        pool.add(rid, 0.5, None, None)
    for rid in range(live, live + 20000):
        pool.add(rid, 0.5, None, None)
        waiter, _ = pool.pick(["s"])
        pool.record_grant(waiter)
        bucket = pool._buckets[None]
        assert len(bucket.by_arrival) <= 2 * bucket.live + 64
        assert len(bucket.by_priority) <= 2 * bucket.live + 64
    assert len(pool) == live


def test_pool_skips_excluded_server():
    pool = SeatPool(1, clock=_Clock())
    pool.add(1, 0.9, "s0", None)
    pool.add(2, 0.1, None, None)
    assert pool.pick(["s0"])[0].rid == 2
    waiter, servers = pool.pick(["s0", "s1"])
    assert waiter.rid == 1 and servers == ["s1"]
    assert pool.pick([]) is None


# ---------------------------------------------------------------------------------- router


def _router(servers=("s0", "s1"), seats=1, cap=None, kv_ms=None) -> PooledLoadBalancingRouter:
    args = {"max_requests_per_server": seats, "max_queue_wait_s": cap}
    if kv_ms is not None:
        args["kv_transfer_timeout_ms"] = kv_ms
    config = RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION, args=args)
    router = create_router(config, list(servers))
    assert isinstance(router, PooledLoadBalancingRouter)
    return router


def test_create_router():
    plain = create_router(
        RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION), ["s"]
    )
    assert type(plain) is LoadBalancingRouter and not getattr(plain, "pooled", False)
    with pytest.raises(ValueError, match="load_balancing"):
        create_router(
            RouterConfig(
                type="round_robin",
                server_role=ServerRole.GENERATION,
                args={"max_requests_per_server": 1},
            ),
            ["s"],
        )
    with pytest.raises(ValueError, match="generation"):
        create_router(
            RouterConfig(
                type="load_balancing",
                server_role=ServerRole.CONTEXT,
                args={"max_requests_per_server": 1},
            ),
            ["s"],
        )
    with pytest.raises(ValueError, match="use_tokens"):
        create_router(
            RouterConfig(
                type="load_balancing",
                server_role=ServerRole.GENERATION,
                args={"max_requests_per_server": 1, "use_tokens": True},
            ),
            ["s"],
        )


def test_routing_key_carries_priority():
    router = _router()
    assert router.routing_key(CompletionRequest(model="m", prompt="x", priority=0.25)) == {
        "num_tokens": 0,
        "priority": 0.25,
    }
    assert router.routing_key(CompletionRequest(model="m", prompt="x"))["priority"] == 0.5
    after_ctx = CompletionRequest(
        model="m",
        prompt="x",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", orig_elapsed_s=2.5
        ),
    )
    assert router.routing_key(after_ctx)["orig_elapsed_s"] == 2.5


def test_extract_disagg_cfg_gives_the_pooled_router_the_ctx_kv_timeout():
    pooled = {"router": {"type": "load_balancing", "max_requests_per_server": 1}}
    default = extract_disagg_cfg(generation_servers=dict(pooled, router=dict(pooled["router"])))
    assert default.gen_router_config.args["kv_transfer_timeout_ms"] == 60000
    config = extract_disagg_cfg(
        context_servers={"cache_transceiver_config": {"kv_transfer_timeout_ms": 30000}},
        generation_servers=dict(pooled, router=dict(pooled["router"])),
    )
    assert config.gen_router_config.args["kv_transfer_timeout_ms"] == 30000
    router = create_router(config.gen_router_config, ["s"])
    assert router._pool.kv_transfer_timeout_s == 30.0
    plain = extract_disagg_cfg(generation_servers={"router": {"type": "load_balancing"}})
    assert "kv_transfer_timeout_ms" not in plain.gen_router_config.args


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_free_seats_place_at_once_like_load_balancing():
    router = _router(seats=2)
    stock = LoadBalancingRouter(ServerRole.GENERATION, ["s0", "s1"])
    placed, expected = [], []
    for rid in range(4):
        placed.append((await router.get_next_server_by_key({}, req_id=rid))[0])
        expected.append((await stock.get_next_server_by_key({}, req_id=rid))[0])
    assert placed == expected
    assert router._pool.placed_at_once == 4


@pytest.mark.asyncio
async def test_freed_seat_goes_to_the_best_waiter():
    router = _router(servers=("s0",), seats=1)
    assert (await router.get_next_server_by_key({}, req_id=0))[0] == "s0"
    waits = {
        rid: asyncio.create_task(router.get_next_server_by_key({"priority": p}, req_id=rid))
        for rid, p in [(1, 0.2), (2, 0.8), (3, 0.8)]
    }
    await _settle()
    assert not any(t.done() for t in waits.values()) and len(router._pool) == 3
    order = []
    for done_rid in (0, 2, 3):
        await router.finish_request_by_id(done_rid)
        await _settle()
        ready = [rid for rid, t in waits.items() if t.done() and rid not in order]
        assert len(ready) == 1
        order.extend(ready)
    assert order == [2, 3, 1]
    assert (await waits[1])[0] == "s0"


@pytest.mark.asyncio
async def test_repeated_select_keeps_its_seat_and_place():
    router = _router(servers=("s0",), seats=1)
    first = await router.get_next_server_by_key({}, req_id=1)
    assert await router.get_next_server_by_key({}, req_id=1) == first
    a = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    b = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    assert len(router._pool) == 1 and router._pool.waiter(2).holders == 2
    await router.finish_request_by_id(1)
    assert (await a)[0] == (await b)[0] == "s0"
    assert router._get_server_load("s0") == 1


@pytest.mark.asyncio
async def test_finish_while_waiting_drops_the_waiter():
    router = _router(servers=("s0",), seats=1)
    await router.get_next_server_by_key({}, req_id=1)
    waiting = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    await router.finish_request_by_id(2)
    with pytest.raises(ValueError, match="finished while waiting"):
        await waiting
    assert len(router._pool) == 0 and router._get_server_load("s0") == 1


@pytest.mark.asyncio
async def test_cancellation_releases_queue_place_and_seat():
    router = _router(servers=("s0",), seats=1)
    await router.get_next_server_by_key({}, req_id=1)
    waiting = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert len(router._pool) == 0
    # A placement whose only caller was cancelled gives its seat back.
    late = asyncio.create_task(router.get_next_server_by_key({}, req_id=3))
    await _settle()
    await router.finish_request_by_id(1)
    late.cancel()
    with pytest.raises(asyncio.CancelledError):
        await late
    assert router._get_server_load("s0") == 0 and router._req_routing_table == {}


@pytest.mark.asyncio
async def test_waiter_fails_at_the_kv_transfer_timeout():
    """A request that finds no seat within the timeout fails, generation_first ones too."""
    router = _router(servers=("s0",), seats=1, kv_ms=50)
    await router.get_next_server_by_key({}, req_id=0)
    doomed = asyncio.create_task(
        router.get_next_server_by_key({"priority": 0.9, "orig_elapsed_s": 3.0}, req_id=1)
    )
    gen_first = asyncio.create_task(router.get_next_server_by_key({"priority": 0.1}, req_id=2))
    with pytest.raises(NoGenerationSeatError, match="KV transfer timeout"):
        await asyncio.wait_for(doomed, timeout=5)
    with pytest.raises(NoGenerationSeatError, match="KV transfer timeout"):
        await asyncio.wait_for(gen_first, timeout=5)
    assert len(router._pool) == 0 and router._pool.expired == 2
    # The in-process path reads the elapsed time from the request.
    request = CompletionRequest(
        model="m",
        prompt="x",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", orig_elapsed_s=1.0
        ),
    )
    with pytest.raises(ValueError, match="KV transfer timeout"):
        await asyncio.wait_for(router.get_next_server(request), timeout=5)
    assert router._get_server_load("s0") == 1 and len(router._pool) == 0


@pytest.mark.asyncio
async def test_in_process_path_keys_by_request():
    router = _router(servers=("s0",), seats=1)
    first = CompletionRequest(model="m", prompt="x", priority=0.1)
    second = CompletionRequest(model="m", prompt="x", priority=0.9)
    assert (await router.get_next_server(first))[0] == "s0"
    waiting = asyncio.create_task(router.get_next_server(second))
    await _settle()
    assert not waiting.done()
    await router.finish_request(first)
    assert (await waiting)[0] == "s0"


# ---------------------------------------------------------------------------------- coordinator


def _client_factory(*_args, **_kwargs):
    return AsyncMock()


def _coordinator(seats=1):
    config = DisaggServerConfig(
        server_configs=[CtxGenServerConfig(type="gen", hostname="g", port=1)],
        ctx_router_config=RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(
            type="load_balancing",
            server_role=ServerRole.GENERATION,
            args={"max_requests_per_server": seats},
        ),
    )
    return DisaggCoordinatorService(config, _client_factory, reservation_timeout_secs=60)


@pytest.mark.asyncio
async def test_pooled_select_waits_outside_the_reservation_lock():
    """A waiting select must not block the finish (on the same lock stripe) that frees its seat."""
    coordinator = _coordinator()
    assert isinstance(coordinator.gen_router, PooledLoadBalancingRouter)
    stripe = coordinator._reservation_lock(("generation", 1))
    other = next(
        rid
        for rid in range(2, 100000)
        if coordinator._reservation_lock(("generation", rid)) is stripe
    )
    assert (await coordinator.select("generation", {}, 1, None))[0] == "g:1"
    waiting = asyncio.create_task(coordinator.select("generation", {"priority": 0.5}, other, None))
    await _settle()
    assert not waiting.done()
    await asyncio.wait_for(coordinator.finish("generation", 1), timeout=5)
    server, _, req_id = await asyncio.wait_for(waiting, timeout=5)
    assert (server, req_id) == ("g:1", other)
    # A repeated select renews the reservation and keeps the seat.
    await coordinator.select("generation", {}, other, None)
    assert coordinator.gen_router._get_server_load("g:1") == 1
    await coordinator.finish("generation", other)
    assert coordinator.gen_router._get_server_load("g:1") == 0
    assert coordinator._reservation_tasks == {}


def test_delegating_router_waits_longer_for_a_pooled_select():
    pooled = CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0)
    plain = CoordinatorDelegatingRouter(
        "http://c:1", LoadBalancingRouter(ServerRole.GENERATION, ["s"]), "generation", 5.0
    )
    assert pooled._select_timeout_s == POOLED_SELECT_TIMEOUT_S
    assert plain._select_timeout_s == 5.0


# ------------------------------------------------ server churn, seat lifecycle and /finish delivery


async def _set_servers(router, servers):
    """What add_server / remove_server / the cluster watch do to the server list."""
    async with router._lock:
        old = router._servers.copy()
        router._servers = list(servers)
        router._on_servers_updated(old, router._servers)


@pytest.mark.asyncio
async def test_seat_count_survives_a_server_leaving_and_rejoining():
    """A re-registered server keeps the seats its running requests hold (no -k drift)."""
    router = _router(servers=("s0",), seats=1)
    await router.get_next_server_by_key({}, req_id=1)
    await _set_servers(router, [])
    await _set_servers(router, ["s0"])
    assert router._get_server_load("s0") == 1  # request 1 still holds its seat
    waiting = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    assert not waiting.done()
    await router.finish_request_by_id(1)
    assert (await asyncio.wait_for(waiting, 5))[0] == "s0"
    await router.finish_request_by_id(2)
    assert router._get_server_load("s0") == 0 and router._departed_state == {}
    # A server that leaves for good drops its state once its last seat is released.
    await router.get_next_server_by_key({}, req_id=3)
    await _set_servers(router, [])
    await router.finish_request_by_id(3)
    assert router._departed_state == {} and router._seat_state == {}


def _ctx_router_with(infos):
    ctx = create_router(
        RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT), list(infos)
    )
    ctx._server_info.update(infos)
    return ctx


@pytest.mark.asyncio
async def test_seat_wait_bound_comes_from_the_context_servers():
    """The context servers' own kv_transfer_timeout_ms (/server_info) wins over the server YAML."""
    gen_config = RouterConfig(
        type="load_balancing",
        server_role=ServerRole.GENERATION,
        args={"max_requests_per_server": 1, "kv_transfer_timeout_ms": 60000},
    )
    ctx_config = RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT)
    ctx, gen = build_disagg_routers(ctx_config, gen_config, ["c0", "c1"], ["s0"])
    assert gen._context_router is ctx
    ctx._server_info.update(
        {"c0": {"kv_transfer_timeout_ms": 600000}, "c1": {"kv_transfer_timeout_ms": 300000}}
    )
    await gen.get_next_server_by_key({}, req_id=1)
    assert gen._pool.kv_transfer_timeout_s == 300.0
    # Without a reported value the configured one applies.
    gen.set_context_router(_ctx_router_with({"c0": {}}))
    waiting = asyncio.create_task(gen.get_next_server_by_key({}, req_id=2))
    await _settle()
    assert gen._pool.kv_transfer_timeout_s == 60.0
    assert gen._pool.waiter(2).deadline - gen._pool.waiter(2).entered == pytest.approx(57.0)
    waiting.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("infos", "expected_s"),
    [
        ({"c0": {"kv_transfer_timeout_ms": None}, "c1": {"kv_transfer_timeout_ms": None}}, None),
        ({"c0": {"kv_transfer_timeout_ms": None}, "c1": {}}, None),
        ({"c0": {"kv_transfer_timeout_ms": None}, "c1": {"kv_transfer_timeout_ms": 30000}}, 30.0),
        ({"c0": {}}, 60.0),
    ],
    ids=["all_null", "null_and_unreported", "null_and_bounded", "unreported"],
)
async def test_a_null_kv_transfer_timeout_leaves_the_seat_wait_unbounded(infos, expected_s):
    """Context servers without a KV transfer timeout never drop the transfer: no 503 at 60 s."""
    gen = _router(servers=("s0",), seats=1, kv_ms=60000)
    gen.set_context_router(_ctx_router_with(infos))
    await gen.get_next_server_by_key({}, req_id=1)
    waiting = asyncio.create_task(gen.get_next_server_by_key({}, req_id=2))
    await _settle()
    assert gen._pool.kv_transfer_timeout_s == expected_s
    assert (gen._pool.waiter(2).deadline is None) == (expected_s is None)
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)


def test_max_queue_wait_without_pooling_is_rejected():
    with pytest.raises(ValueError, match="max_requests_per_server"):
        extract_disagg_cfg(
            generation_servers={"router": {"type": "load_balancing", "max_queue_wait_s": 30}}
        )
    for context_router in (
        {"max_queue_wait_s": 30},
        {"max_queue_wait_s": 30, "max_requests_per_server": 4},
    ):
        with pytest.raises(ValueError, match="context_servers.router.* generation router option"):
            extract_disagg_cfg(context_servers={"router": context_router})


def test_router_declares_pooled():
    assert Router.pooled is False and PooledLoadBalancingRouter.pooled is True
    assert not create_router(
        RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION), ["s"]
    ).pooled
    assert CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0).pooled


@pytest.mark.asyncio
async def test_coordinator_stop_fails_seat_waiters():
    coordinator = _coordinator()
    await coordinator.select("generation", {}, 1, None)
    waiting = asyncio.create_task(coordinator.select("generation", {}, 2, None))
    await _settle()
    await asyncio.wait_for(coordinator.stop(), timeout=5)
    with pytest.raises(NoGenerationSeatError, match="stopping"):
        await asyncio.wait_for(waiting, timeout=5)
    # A request arriving after that is refused at once rather than queued.
    with pytest.raises(NoGenerationSeatError, match="stopping"):
        await asyncio.wait_for(coordinator.select("generation", {}, 3, None), timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("uds", [False, True], ids=["lifespan", "uds_and_tcp"])
async def test_coordinator_shutdown_answers_parked_selects_at_once(tmp_path, monkeypatch, uds):
    """Uvicorn's graceful shutdown waits for in-flight requests: a parked /select gets 503 first."""
    import socket

    import aiohttp

    from tensorrt_llm.serve import coordinator_server

    created = []
    real_create = coordinator_server.create_uvicorn_server

    def create(config):
        created.append(real_create(config))
        return created[-1]

    monkeypatch.setattr(coordinator_server, "create_uvicorn_server", create)
    coordinator = _coordinator()
    coordinator.start = AsyncMock()  # no servers to prepare or wait for
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    monkeypatch.chdir(tmp_path)  # a relative path keeps the socket under AF_UNIX's 108-byte limit
    uds_path = "coordinator.sock" if uds else None
    serving = asyncio.create_task(
        CoordinatorServer(coordinator)("127.0.0.1", port, uds=uds_path, sockets=[sock])
    )
    connector = aiohttp.UnixConnector(path=uds_path) if uds else aiohttp.TCPConnector()
    base = "http://localhost" if uds else f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession(connector=connector) as session:

            async def select(rid):
                body = msgpack.packb({"role": "generation", "routing_key": {}, "req_id": rid})
                async with session.post(f"{base}/select", data=body) as resp:
                    return resp.status, msgpack.unpackb(await resp.read(), raw=False)

            for _ in range(200):
                try:
                    status, _ = await select(1)  # takes the only seat
                    break
                except aiohttp.ClientConnectionError:
                    await asyncio.sleep(0.05)
            assert status == 200
            parked = asyncio.create_task(select(2))
            await asyncio.sleep(0.3)
            assert not parked.done()
            for server in created:
                server.should_exit = True  # what SIGINT / SIGTERM do
            status, body = await asyncio.wait_for(parked, timeout=3)
            assert status == 503 and body["no_generation_seat"] is True
        await asyncio.wait_for(asyncio.shield(serving), timeout=3)
    finally:
        await coordinator.stop()
        for server in created:
            server.force_exit = True
        await asyncio.wait_for(serving, timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("uds", [False, True], ids=["tcp", "uds"])
async def test_parked_select_of_a_disconnected_client_takes_no_seat(tmp_path, monkeypatch, uds):
    """A worker that dies while its /select waits for a seat leaves no waiter and no seat behind."""
    import socket

    coordinator = _coordinator()
    coordinator.start = AsyncMock()  # no servers to prepare or wait for
    router = coordinator.gen_router
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    monkeypatch.chdir(tmp_path)  # a relative path keeps the socket under AF_UNIX's 108-byte limit
    uds_path = "coordinator.sock" if uds else None
    server = CoordinatorServer(coordinator)
    serving = asyncio.create_task(server("127.0.0.1", port, uds=uds_path, sockets=[sock]))

    async def post_select(rid):
        """Send a /select over a raw socket; return its reader and writer."""
        for _ in range(200):
            try:
                if uds:
                    reader, writer = await asyncio.open_unix_connection(uds_path)
                else:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except OSError:
                await asyncio.sleep(0.05)
        body = msgpack.packb({"role": "generation", "routing_key": {}, "req_id": rid})
        writer.write(
            b"POST /select HTTP/1.1\r\nHost: c\r\nContent-Type: application/msgpack\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        return reader, writer

    try:
        reader, writer = await post_select(1)  # takes the only seat
        assert b" 200 " in await asyncio.wait_for(reader.readline(), timeout=10)
        writer.close()
        _, parked = await post_select(2)
        for _ in range(100):
            if router._pool.waiter(2) is not None:
                break
            await asyncio.sleep(0.02)
        assert router._pool.waiter(2) is not None
        parked.close()  # the worker dies: its socket closes
        for _ in range(100):
            if len(router._pool) == 0:
                break
            await asyncio.sleep(0.02)
        assert len(router._pool) == 0
        await coordinator.finish("generation", 1)
        await _settle()
        assert router._get_server_load("g:1") == 0 and router._req_routing_table == {}
        assert ("generation", 2) not in coordinator._reservation_tasks
    finally:
        for uvicorn_server in server.servers:
            uvicorn_server.should_exit = True
        await asyncio.wait_for(serving, timeout=10)
        await coordinator.stop()


@pytest.mark.asyncio
async def test_finish_before_select_refuses_the_late_select(monkeypatch):
    """A cancelled request's /finish can overtake its /select; that select must not queue."""
    coordinator = _coordinator()
    assert (await coordinator.select("generation", {}, 1, None))[0] == "g:1"
    await coordinator.finish("generation", 2)
    with pytest.raises(ValueError, match="finished before its placement"):
        await asyncio.wait_for(coordinator.select("generation", {}, 2, None), timeout=5)
    router = coordinator.gen_router
    assert len(router._pool) == 0 and ("generation", 2) not in coordinator._reservation_tasks
    await coordinator.finish("generation", 1)
    assert (await coordinator.select("generation", {}, 3, None))[0] == "g:1"
    await coordinator.finish("generation", 3)
    # The memory is bounded in count and in time.
    monkeypatch.setattr("tensorrt_llm.serve.router.POOLED_FINISHED_MAX", 4)
    for rid in range(10, 20):
        await coordinator.finish("generation", rid)
    assert list(router._finished_ids) == [16, 17, 18, 19]
    monkeypatch.setattr("tensorrt_llm.serve.router.POOLED_FINISHED_TTL_S", 0.0)
    assert (await coordinator.select("generation", {}, 19, None))[0] == "g:1"
    await coordinator.finish("generation", 19)


@pytest.mark.asyncio
async def test_pooled_seat_is_not_reclaimed_at_the_reservation_ttl():
    """A granted seat outlives the coordinator's 180 s TTL: only /finish (or the 6 h net) frees it."""
    coordinator = _coordinator()
    coordinator._reservation_timeout_secs = 0.05
    await coordinator.select("generation", {}, 1, None)
    await asyncio.sleep(0.3)
    assert coordinator.gen_router._get_server_load("g:1") == 1
    assert coordinator._reservation_timeout_for(coordinator.gen_router) == (
        POOLED_RESERVATION_TIMEOUT_S
    )
    assert coordinator._reservation_timeout_for(coordinator.ctx_router) == 0.05
    await coordinator.finish("generation", 1)
    assert coordinator.gen_router._get_server_load("g:1") == 0


def test_seat_timeout_reaches_the_client_as_503(monkeypatch):
    from fastapi.testclient import TestClient

    coordinator = _coordinator()

    async def no_seat(*_args):
        raise NoGenerationSeatError("no seat within 60 s")

    monkeypatch.setattr(coordinator, "select", no_seat)
    http = TestClient(CoordinatorServer(coordinator).app)
    raw = http.post(
        "/select",
        content=msgpack.packb({"role": "generation", "routing_key": {}, "req_id": 1}),
    )
    assert raw.status_code == 503
    assert msgpack.unpackb(raw.content, raw=False)["no_generation_seat"] is True
    server = SimpleNamespace(_perf_metrics_collector=Mock())
    with pytest.raises(HTTPException) as error:
        OpenAIDisaggServer._handle_exception(server, NoGenerationSeatError("no seat"))
    assert error.value.status_code == 503


class _FakeCoordinatorHttp:
    """A coordinator whose pooled /select waits until a /finish for its req_id arrives."""

    def __init__(self, select_delay_s=None):
        self.finished = []
        self.select_delay_s = select_delay_s
        self._finish_events = {}

    async def select(self, request):
        from aiohttp import web

        body = msgpack.unpackb(await request.read(), raw=False)
        if self.select_delay_s is not None:
            await asyncio.sleep(self.select_delay_s)
        else:
            event = self._finish_events.setdefault(body["req_id"], asyncio.Event())
            await event.wait()
            return web.Response(
                status=503,
                body=msgpack.packb({"error": "finished while waiting"}),
                content_type="application/msgpack",
            )
        return web.Response(
            body=msgpack.packb({"server": "s0", "info": {}, "req_id": body["req_id"]}),
            content_type="application/msgpack",
        )

    async def finish(self, request):
        from aiohttp import web

        body = msgpack.unpackb(await request.read(), raw=False)
        self.finished.append(body["req_id"])
        self._finish_events.setdefault(body["req_id"], asyncio.Event()).set()
        return web.Response(body=msgpack.packb({}), content_type="application/msgpack")


async def _with_fake_coordinator(fake, body):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    app = web.Application()
    app.router.add_post("/select", fake.select)
    app.router.add_post("/finish", fake.finish)
    async with TestServer(app) as server:
        router = CoordinatorDelegatingRouter(
            f"http://{server.host}:{server.port}", _router(), "generation", 5.0
        )
        try:
            await body(router)
        finally:
            await router.close()


def _gen_request(rid):
    return CompletionRequest(
        model="m",
        prompt="x",
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only", ctx_request_id=rid, disagg_request_id=rid
        ),
    )


@pytest.mark.asyncio
async def test_delegating_router_finishes_a_cancelled_pooled_select_at_once():
    """A cancelled caller sends /finish at once; it does not wait for the seat first."""
    fake = _FakeCoordinatorHttp()

    async def body(router):
        selecting = asyncio.create_task(router.get_next_server(_gen_request(7)))
        await asyncio.sleep(0.2)
        selecting.cancel()
        done, _ = await asyncio.wait({selecting}, timeout=5)
        assert selecting in done and selecting.cancelled()
        await asyncio.wait_for(router._finish_queue.join(), timeout=5)
        assert 7 in fake.finished

    await _with_fake_coordinator(fake, body)


@pytest.mark.asyncio
async def test_delegating_router_finishes_a_timed_out_select():
    fake = _FakeCoordinatorHttp(select_delay_s=2.0)

    async def body(router):
        router._select_timeout_s = 0.1
        with pytest.raises(asyncio.TimeoutError):
            await router.get_next_server(_gen_request(9))
        await asyncio.wait_for(router._finish_queue.join(), timeout=5)
        assert fake.finished == [9]

    await _with_fake_coordinator(fake, body)


class _LossyCoordinatorHttp(_FakeCoordinatorHttp):
    """A coordinator that takes the first /select but whose reply never arrives.

    ``disconnect``: attempt 1 parks (its connection drops) and every retry's connection drops too.
    ``refused``: attempt 1 parks (its connection drops) and attempt 2 gets an error status.
    ``truncated``: attempt 1 is granted, but its 200 reply is cut short.
    """

    def __init__(self, failure):
        super().__init__()
        self.failure = failure
        self.selects = 0

    async def select(self, request):
        from aiohttp import web

        body = msgpack.unpackb(await request.read(), raw=False)
        self.selects += 1
        if self.failure == "truncated":
            response = web.StreamResponse(
                headers={"Content-Type": "application/msgpack", "Content-Length": "1000"}
            )
            await response.prepare(request)
            await response.write(b"\x83")
            request.transport.close()
            return response
        if self.failure == "refused" and self.selects > 1:
            return web.Response(
                status=500,
                body=msgpack.packb({"error": "refused"}),
                content_type="application/msgpack",
            )
        request.transport.close()
        if self.selects == 1:
            # Parked: the coordinator still waits for a seat for this request.
            await self._finish_events.setdefault(body["req_id"], asyncio.Event()).wait()
        return web.Response(body=msgpack.packb({}), content_type="application/msgpack")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnect", "refused", "truncated"])
async def test_delegating_router_finishes_a_select_whose_reply_is_lost(monkeypatch, failure):
    """A select the coordinator took is ended by /finish when the worker gets no reply."""
    import aiohttp

    monkeypatch.setattr("tensorrt_llm.serve.router.COORDINATOR_SELECT_RETRY_DELAY_S", 0.001)
    fake = _LossyCoordinatorHttp(failure)

    async def body(router):
        with pytest.raises((aiohttp.ClientError, ValueError)):
            await asyncio.wait_for(router.get_next_server(_gen_request(11)), timeout=10)
        await asyncio.wait_for(router._finish_queue.join(), timeout=5)
        assert fake.finished == [11]
        assert fake.selects == {"disconnect": 4, "refused": 2, "truncated": 1}[failure]

    await _with_fake_coordinator(fake, body)


class _FlakyCoordinatorFinish:
    """A coordinator /finish that fails its first ``failures`` calls, then reaches ``coordinator``."""

    def __init__(self, coordinator, failures, status=503):
        self.coordinator = coordinator
        self.failures = failures
        self.status = status
        self.calls = 0
        self.finished = []

    async def finish(self, request):
        from aiohttp import web

        body = msgpack.unpackb(await request.read(), raw=False)
        self.calls += 1
        if self.calls <= self.failures:
            return web.Response(
                status=self.status,
                body=msgpack.packb({"error": "unavailable"}),
                content_type="application/msgpack",
            )
        await self.coordinator.finish(body["role"], body["req_id"], body["success"])
        self.finished.append(body["req_id"])
        return web.Response(body=msgpack.packb({}), content_type="application/msgpack")


async def _finish_through_flaky_coordinator(local_router, fake, rids, monkeypatch):
    """Finish ``rids`` through a delegating router whose finish queue holds one entry."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    monkeypatch.setattr("tensorrt_llm.serve.router.COORDINATOR_FINISH_RETRY_DELAY_S", 0.001)
    app = web.Application()
    app.router.add_post("/finish", fake.finish)
    async with TestServer(app) as server:
        router = CoordinatorDelegatingRouter(
            f"http://{server.host}:{server.port}", local_router, "generation", 5.0
        )
        if not router.pooled:
            router._finish_queue = asyncio.Queue(maxsize=1)
        try:
            # No await between the puts: the workers have not run, so a bounded queue is full.
            for rid in rids:
                await router.finish_request(_gen_request(rid), req_id=rid)
            await asyncio.wait_for(router._finish_queue.join(), timeout=30)
        finally:
            await router.close()
        return router


@pytest.mark.asyncio
async def test_pooled_finish_is_parked_and_retried_until_the_seat_is_released(monkeypatch):
    """A pooled /finish is never dropped (its queue is unbounded) nor abandoned after 3 failures."""
    coordinator = _coordinator(seats=2)
    for rid in (1, 2):
        assert (await coordinator.select("generation", {}, rid, None))[0] == "g:1"
    assert coordinator.gen_router._get_server_load("g:1") == 2
    fake = _FlakyCoordinatorFinish(coordinator, failures=5)
    router = await _finish_through_flaky_coordinator(_router(), fake, (1, 2), monkeypatch)
    assert router._dropped_finishes == 0 and router._finish_queue.maxsize == 0
    assert sorted(fake.finished) == [1, 2]
    assert coordinator.gen_router._get_server_load("g:1") == 0
    assert coordinator._reservation_tasks == {}


@pytest.mark.asyncio
async def test_non_pooled_finish_still_drops_on_a_full_queue_and_gives_up(monkeypatch):
    """The non-pooled path is unchanged: overflow is dropped and a finish stops after 3 attempts."""
    local_router = LoadBalancingRouter(server_role=None, servers=["s0"])
    assert not local_router.pooled
    fake = _FlakyCoordinatorFinish(AsyncMock(), failures=5)
    router = await _finish_through_flaky_coordinator(local_router, fake, (1, 2), monkeypatch)
    assert router._dropped_finishes == 1
    assert fake.calls == 3 and fake.finished == []


@pytest.mark.asyncio
async def test_pooled_finish_stops_on_a_malformed_reply_or_at_the_expiry(monkeypatch):
    fake = _FlakyCoordinatorFinish(AsyncMock(), failures=100, status=400)
    router = await _finish_through_flaky_coordinator(_router(), fake, (1,), monkeypatch)
    assert fake.calls == 1
    monkeypatch.setattr(
        CoordinatorDelegatingRouter,
        "__init__",
        _with_finish_window(CoordinatorDelegatingRouter.__init__, 0.05),
    )
    fake = _FlakyCoordinatorFinish(AsyncMock(), failures=10**6)
    router = await _finish_through_flaky_coordinator(_router(), fake, (1,), monkeypatch)
    assert router._pooled_finish_window_s == 0.05
    assert 3 < fake.calls < 10**6 and fake.finished == []


def _with_finish_window(init, window_s):
    def patched(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._pooled_finish_window_s = window_s

    return patched


def test_pooled_finish_window_follows_the_coordinator_expiry(monkeypatch):
    """The env timeout can lengthen the pooled expiry (and the finish retry window), not shorten it."""
    monkeypatch.setenv("TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT", "60")
    router = CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0)
    assert router._pooled_finish_window_s == POOLED_RESERVATION_TIMEOUT_S
    coordinator = DisaggCoordinatorService(_coordinator()._config, _client_factory)
    assert coordinator._reservation_timeout_for(coordinator.gen_router) == (
        POOLED_RESERVATION_TIMEOUT_S
    )
    longer = 2 * POOLED_RESERVATION_TIMEOUT_S
    monkeypatch.setenv("TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT", str(longer))
    router = CoordinatorDelegatingRouter("http://c:1", _router(), "generation", 5.0)
    assert router._pooled_finish_window_s == longer
    # A request timeout above 6 h lengthens both sides alike (one helper computes them).
    monkeypatch.delenv("TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT")
    worker = CoordinatorDelegatingRouter("http://c:1", _router(), "generation", longer + 1)
    coordinator = DisaggCoordinatorService(
        _coordinator()._config, _client_factory, request_timeout_secs=longer + 1
    )
    assert worker._pooled_finish_window_s == longer + 1
    assert coordinator._reservation_timeout_for(coordinator.gen_router) == longer + 1
