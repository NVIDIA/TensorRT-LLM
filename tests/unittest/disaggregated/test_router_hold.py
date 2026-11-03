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
"""Held placement in LoadBalancingRouter (``max_inflight_per_server``).

Covers the in-process router, the coordinator-owned router reached through
``DisaggCoordinatorService.select`` / ``finish``, and the fleet-worker side
``CoordinatorDelegatingRouter``.
"""

import asyncio
from unittest import mock

import pytest

from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    RouterConfig,
    ServerRole,
    extract_router_config,
)
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams
from tensorrt_llm.serve.router import (
    PLACEMENT_HOLD_TIMEOUT_DEFAULT_S,
    CoordinatorDelegatingRouter,
    HeldPlacementCancelled,
    LoadBalancingRouter,
    create_router,
)

pytestmark = pytest.mark.cpu_only


def _request(i: int = 0) -> CompletionRequest:
    return CompletionRequest(model="m", prompt=f"prompt {i}")


def _router(servers, cap=None, **kwargs) -> LoadBalancingRouter:
    return LoadBalancingRouter(
        server_role=ServerRole.GENERATION,
        servers=list(servers),
        max_inflight_per_server=cap,
        **kwargs,
    )


def _inflight(router: LoadBalancingRouter) -> dict:
    return {server: state._num_active_requests for server, state in router._server_state.items()}


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


# ---- gate off ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_never_holds():
    router = _router(["g0", "g1"])
    assert router.placement_hold_timeout_s is None
    requests = [_request(i) for i in range(5)]
    placed = [(await router.get_next_server(r))[0] for r in requests]
    # JSQ with round-robin ties: alternates and never waits.
    assert placed == ["g0", "g1", "g0", "g1", "g0"]
    assert _inflight(router) == {"g0": 3, "g1": 2}
    for r in requests:
        await router.finish_request(r)
    assert _inflight(router) == {"g0": 0, "g1": 0}


def test_router_config_reaches_the_router():
    server_cfg = {
        "router": {"type": "load_balancing", "max_inflight_per_server": 2, "hold_timeout_s": 30},
        "max_batch_size": 1,
    }
    config = extract_router_config(server_cfg)
    config.server_role = ServerRole.GENERATION
    router = create_router(config, ["g0", "g1"])
    assert isinstance(router, LoadBalancingRouter)
    assert router._max_inflight_per_server == 2
    assert router.placement_hold_timeout_s == 30.0

    plain = create_router(
        RouterConfig(type="load_balancing", server_role=ServerRole.GENERATION, args={}), ["g0"]
    )
    assert plain.placement_hold_timeout_s is None
    assert PLACEMENT_HOLD_TIMEOUT_DEFAULT_S > 0


@pytest.mark.parametrize("cap", [0, -1, 1.5, True, "2"])
def test_invalid_cap_is_rejected(cap):
    with pytest.raises(ValueError):
        _router(["g0"], cap=cap)


def test_invalid_hold_timeout_is_rejected():
    with pytest.raises(ValueError):
        _router(["g0"], cap=1, hold_timeout_s=0)


# ---- in-process hold -----------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_and_release_are_fifo():
    router = _router(["g0", "g1"], cap=1)
    r = [_request(i) for i in range(5)]
    s0, _ = await router.get_next_server(r[0])
    s1, _ = await router.get_next_server(r[1])
    assert {s0, s1} == {"g0", "g1"}

    tasks = []
    for i in (2, 3, 4):
        tasks.append(asyncio.create_task(router.get_next_server(r[i])))
        await _settle()
    assert router.num_held_requests == 3
    assert not any(t.done() for t in tasks)
    assert _inflight(router) == {"g0": 1, "g1": 1}

    await router.finish_request(r[1])
    await _settle()
    assert tasks[0].done() and not tasks[1].done()
    assert tasks[0].result()[0] == s1

    await router.finish_request(r[0])
    await _settle()
    assert tasks[1].done() and not tasks[2].done()
    assert tasks[1].result()[0] == s0
    assert router.num_held_requests == 1

    await router.finish_request(r[2])
    await _settle()
    assert tasks[2].result()[0] == s1
    assert _inflight(router) == {"g0": 1, "g1": 1}
    for i in (3, 4):
        await router.finish_request(r[i])
    assert _inflight(router) == {"g0": 0, "g1": 0}
    assert router._req_routing_table == {}


@pytest.mark.asyncio
async def test_prefers_idle_then_youngest_running():
    router = _router(["g0", "g1", "g2"], cap=2)
    r = [_request(i) for i in range(4)]
    first = []
    for x in r[:3]:
        first.append((await router.get_next_server(x))[0])
        await asyncio.sleep(0.002)
    # Idle servers first: one request per server before any gets a second.
    assert sorted(first) == ["g0", "g1", "g2"]
    # All have one in flight: the server whose request started last wins.
    s3, _ = await router.get_next_server(r[3])
    assert s3 == first[2]

    # A new request never waits while a server is below the cap.
    await router.finish_request(r[0])
    s4, _ = await router.get_next_server(_request(4))
    assert s4 == first[0]


@pytest.mark.asyncio
async def test_new_arrival_waits_behind_held_requests():
    router = _router(["g0"], cap=1)
    r = [_request(i) for i in range(3)]
    await router.get_next_server(r[0])
    t1 = asyncio.create_task(router.get_next_server(r[1]))
    await _settle()
    await router.finish_request(r[0])
    # The slot freed by r0 goes to r1, which was waiting, not to r2.
    t2 = asyncio.create_task(router.get_next_server(r[2]))
    await _settle()
    assert t1.done() and not t2.done()
    await router.finish_request(r[1])
    await _settle()
    assert t2.done()
    await router.finish_request(r[2])
    assert _inflight(router) == {"g0": 0}


@pytest.mark.asyncio
async def test_cancel_while_held_releases_cleanly():
    router = _router(["g0"], cap=1)
    r = [_request(i) for i in range(3)]
    await router.get_next_server(r[0])
    t1 = asyncio.create_task(router.get_next_server(r[1]))
    t2 = asyncio.create_task(router.get_next_server(r[2]))
    await _settle()
    assert router.num_held_requests == 2

    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1
    assert router.num_held_requests == 1

    await router.finish_request(r[0])
    await _settle()
    assert t2.result()[0] == "g0"
    assert _inflight(router) == {"g0": 1}
    await router.finish_request(r[2])
    assert _inflight(router) == {"g0": 0}
    assert router._req_routing_table == {}
    assert router._hold_started == {}


@pytest.mark.asyncio
async def test_cancel_after_assignment_gives_the_slot_back():
    router = _router(["g0"], cap=1)
    r = [_request(i) for i in range(3)]
    await router.get_next_server(r[0])
    t1 = asyncio.create_task(router.get_next_server(r[1]))
    t2 = asyncio.create_task(router.get_next_server(r[2]))
    await _settle()
    # r1 is assigned g0 and cancelled in the same loop turn.
    await router.finish_request(r[0])
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1
    await _settle()
    # The slot r1 never used went to r2.
    assert t2.result()[0] == "g0"
    assert _inflight(router) == {"g0": 1}
    await router.finish_request(r[2])
    assert _inflight(router) == {"g0": 0}


@pytest.mark.asyncio
async def test_hold_timeout_falls_back_to_least_loaded():
    router = _router(["g0", "g1"], cap=1, hold_timeout_s=0.05)
    r = [_request(i) for i in range(3)]
    await router.get_next_server(r[0])
    await router.get_next_server(r[1])
    server, _ = await asyncio.wait_for(router.get_next_server(r[2]), 5)
    assert server in ("g0", "g1")
    assert router._hold_fallbacks == 1
    assert router.num_held_requests == 0
    assert sum(_inflight(router).values()) == 3
    for x in r:
        await router.finish_request(x)
    assert _inflight(router) == {"g0": 0, "g1": 0}


@pytest.mark.asyncio
async def test_new_server_releases_held_requests():
    router = _router(["g0"], cap=1)
    r = [_request(i) for i in range(2)]
    await router.get_next_server(r[0])
    t1 = asyncio.create_task(router.get_next_server(r[1]))
    await _settle()
    old = router._servers.copy()
    router._servers = old + ["g1"]
    router._on_servers_updated(old, router._servers)
    await _settle()
    assert t1.result()[0] == "g1"


# ---- coordinator-owned router (by request ID) ------------------------------------


@pytest.mark.asyncio
async def test_by_id_finish_cancels_a_held_placement():
    router = _router(["g0"], cap=1)
    await router.get_next_server_by_key({}, req_id=1)
    held = asyncio.create_task(router.get_next_server_by_key({}, req_id=2))
    await _settle()
    await router.finish_request_by_id(2, False)
    with pytest.raises(HeldPlacementCancelled):
        await held
    assert router.num_held_requests == 0
    await router.finish_request_by_id(1)
    assert _inflight(router) == {"g0": 0}


@pytest.mark.asyncio
async def test_by_id_release_before_placement_refuses_the_late_select():
    router = _router(["g0"], cap=1)
    await router.finish_request_by_id(7, False)
    with pytest.raises(HeldPlacementCancelled):
        await router.get_next_server_by_key({}, req_id=7)
    assert _inflight(router) == {"g0": 0}
    # The marker is consumed: a later placement for the ID goes through.
    server, _, rid = await router.get_next_server_by_key({}, req_id=7)
    assert (server, rid) == ("g0", 7)


@pytest.mark.asyncio
async def test_by_id_repeated_placement_does_not_leak_load():
    router = _router(["g0", "g1"], cap=2, use_tokens=True)
    await router.get_next_server_by_key({"num_tokens": 10}, req_id=5)
    await router.get_next_server_by_key({"num_tokens": 10}, req_id=5)
    assert sum(_inflight(router).values()) == 1
    await router.finish_request_by_id(5)
    assert _inflight(router) == {"g0": 0, "g1": 0}
    assert all(state._num_active_tokens == 0 for state in router._server_state.values())


def _coordinator(gen_urls, gen_router_args):
    config = DisaggServerConfig(
        server_configs=[
            CtxGenServerConfig(type="gen", hostname=u.split(":")[0], port=int(u.split(":")[1]))
            for u in gen_urls
        ],
        ctx_router_config=RouterConfig(type="round_robin", server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(
            type="load_balancing", server_role=ServerRole.GENERATION, args=dict(gen_router_args)
        ),
    )
    return DisaggCoordinatorService(config, lambda *a, **k: None, reservation_timeout_secs=60)


def _same_stripe_id(coordinator, req_id):
    stripe = coordinator._reservation_lock(("generation", req_id))
    for other in range(req_id + 1, req_id + 100000):
        if coordinator._reservation_lock(("generation", other)) is stripe:
            return other
    raise AssertionError("no colliding request ID found")


@pytest.mark.asyncio
async def test_coordinator_hold_releases_on_finish_sharing_the_lock_stripe():
    coordinator = _coordinator(["gen0:8000"], {"max_inflight_per_server": 1})
    first = 11
    held_id = _same_stripe_id(coordinator, first)
    await coordinator.select("generation", {}, first, None)
    held = asyncio.create_task(coordinator.select("generation", {}, held_id, None))
    await _settle()
    assert coordinator.gen_router.num_held_requests == 1
    # The finish takes the same reservation-lock stripe as the held select.
    await asyncio.wait_for(coordinator.finish("generation", first), 5)
    server, _, rid = await asyncio.wait_for(held, 5)
    assert (server, rid) == ("gen0:8000", held_id)
    assert ("generation", held_id) in coordinator._reservation_tasks
    await coordinator.finish("generation", held_id)
    assert coordinator._reservation_tasks == {}
    assert _inflight(coordinator.gen_router) == {"gen0:8000": 0}


@pytest.mark.asyncio
async def test_coordinator_finish_of_a_held_select_fails_it():
    coordinator = _coordinator(["gen0:8000"], {"max_inflight_per_server": 1})
    await coordinator.select("generation", {}, 1, None)
    held = asyncio.create_task(coordinator.select("generation", {}, 2, None))
    await _settle()
    await coordinator.finish("generation", 2, False)
    with pytest.raises(ValueError):
        await held
    assert ("generation", 2) not in coordinator._reservation_tasks
    await coordinator.finish("generation", 1)
    assert _inflight(coordinator.gen_router) == {"gen0:8000": 0}


@pytest.mark.asyncio
async def test_coordinator_concurrent_duplicate_select_with_hold():
    coordinator = _coordinator(["gen0:8000", "gen1:8000"], {"max_inflight_per_server": 1})
    await asyncio.gather(
        coordinator.select("generation", {}, 123, None),
        coordinator.select("generation", {}, 123, None),
    )
    assert sum(_inflight(coordinator.gen_router).values()) == 1
    await coordinator.finish("generation", 123)
    assert sum(_inflight(coordinator.gen_router).values()) == 0
    assert coordinator._reservation_tasks == {}


# ---- fleet-worker side -----------------------------------------------------------


def _delegating(local):
    return CoordinatorDelegatingRouter(
        "http://coordinator", local, "generation", request_timeout_s=5.0
    )


def _gen_request():
    request = _request()
    request.disaggregated_params = DisaggregatedParams(
        request_type="generation_only", ctx_request_id=42, disagg_request_id=42
    )
    return request


@pytest.mark.asyncio
async def test_worker_select_timeout_default_is_unchanged():
    router = _delegating(_router(["g0"]))
    router._post_select = mock.AsyncMock(return_value={"server": "g0", "info": {}})
    server, _ = await router.get_next_server(_gen_request())
    assert server == "g0"
    assert router._post_select.await_args.kwargs == {}


@pytest.mark.asyncio
async def test_worker_select_waits_for_the_hold_and_releases_on_cancel():
    router = _delegating(_router(["g0"], cap=1, hold_timeout_s=120))
    router._post_select = mock.AsyncMock(return_value={"server": "g0", "info": {}})
    await router.get_next_server(_gen_request())
    assert router._post_select.await_args.kwargs == {"timeout": 125.0}

    started = asyncio.Event()

    async def _hang(payload, timeout=None):
        started.set()
        await asyncio.sleep(3600)

    router._post_select = _hang
    router._ensure_finish_workers = mock.Mock()
    task = asyncio.create_task(router.get_next_server(_gen_request()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert router._finish_queue.get_nowait() == (42, False)
