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
"""/health must stop promising readiness once a worker is gone.

Without a cluster manager ``is_ready()`` used to be an unconditional ``True``,
so a ctx or gen worker could die after startup and the coordinator kept
answering 200 for it.

Two layers here, deliberately:

* ``TestReadinessPredicate`` drives ``is_ready`` against stub routers. Fast,
  but it can only confirm the predicate -- it says nothing about whether a
  real ``Router`` ever reaches the states it simulates.
* ``TestMonitorDrivesReadiness`` drives a real ``Router`` through
  ``_monitor_servers()`` with a stubbed metadata server. This is the layer
  that matters: the original change keyed readiness off an empty server list
  that the monitor could never produce, because ``_filter_servers_by_role``
  raised and the monitor task died with ``self._servers`` left stale.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Optional, TypeVar

import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.disagg_coordinator import GEN_ONLY_BENCHMARK_ENV, DisaggCoordinatorService
from tensorrt_llm.serve.router import RoundRobinRouter

# Stub routers only: no engine, no GPU. The marker is load-bearing --
# l0_cpu reaches this file solely through its `unittest/others` directory
# entry, and tests/unittest/conftest.py's pytest_ignore_collect drops any
# file whose source lacks this literal when pytest runs with -m cpu_only.
# Without it these tests are collected nowhere at all.
pytestmark = pytest.mark.cpu_only


# Truthy stand-in for a metadata server. Module-level rather than a default
# argument: evaluating `object()` in the signature would bind one instance for
# the life of the module anyway, so naming it says that out loud.
_DEFAULT_METADATA_SERVER = object()

# How long a test will wait for the monitor to reach the state it asserts on.
# Generous on purpose -- it is a deadlock guard, not a timing assertion, and a
# loaded CI worker must not be able to turn a passing test red.
_PROGRESS_TIMEOUT_SECS = 10.0

_T = TypeVar("_T")


class _StubRouter:
    """Just the surface ``is_ready`` touches."""

    def __init__(self, servers: Iterable[str], stale: bool = False) -> None:
        self._servers = list(servers)
        self._stale = stale

    @property
    def servers(self) -> list[str]:
        return self._servers

    @property
    def num_prepared_servers(self) -> int:
        return len(self._servers)

    def monitoring_is_stale(self, max_age_secs: float) -> bool:
        return self._stale


def _coordinator(
    ctx_servers: Iterable[str],
    gen_servers: Iterable[str],
    cluster_manager: Optional[Any] = None,
    metadata_server: Optional[Any] = _DEFAULT_METADATA_SERVER,
    ctx_stale: bool = False,
    gen_stale: bool = False,
) -> DisaggCoordinatorService:
    """A DisaggCoordinatorService with only the readiness surface populated.

    __init__ builds sessions and config we do not need, so the object is
    created without running it -- the method under test reads a handful of
    attributes. ``metadata_server`` defaults to a truthy sentinel because the
    interesting behaviour only exists for metadata-driven deployments; pass
    ``None`` for the static-list shape.
    """
    c = object.__new__(DisaggCoordinatorService)
    c._ctx_router = _StubRouter(ctx_servers, stale=ctx_stale)
    c._gen_router = _StubRouter(gen_servers, stale=gen_stale)
    c._disagg_cluster_manager = cluster_manager
    c._metadata_server = metadata_server
    c._monitor_staleness_secs = 30.0
    return c


def _ready(c: DisaggCoordinatorService) -> bool:
    return asyncio.run(c.is_ready())


async def _await_progress(event: asyncio.Event, what: str) -> None:
    """Wait for the monitor to actually reach a state, instead of sleeping.

    A fixed sleep asserts a timing guess: on a loaded CI worker the monitor
    may not have completed the poll the assertions depend on, which either
    fails the test for the wrong reason or -- worse -- lets it pass without
    the code under test having run at all.
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=_PROGRESS_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        pytest.fail(f"monitor did not {what} within {_PROGRESS_TIMEOUT_SECS}s")


async def _run_monitored(router: RoundRobinRouter, body: Callable[[], Awaitable[_T]]) -> _T:
    """Run ``body`` with ``router``'s monitor live, then stop it cleanly."""
    try:
        return await body()
    finally:
        await router.stop_server_monitoring()


class TestReadinessPredicate:
    """``is_ready`` itself, against stub routers."""

    def test_ready_while_both_roles_have_servers(self):
        assert _ready(_coordinator(["ctx0"], ["gen0"])) is True

    @pytest.mark.parametrize(
        "ctx,gen,gone",
        [
            ([], ["gen0"], "context"),
            (["ctx0"], [], "generation"),
            ([], [], "both"),
        ],
    )
    def test_not_ready_once_a_role_has_no_servers(self, ctx, gen, gone):
        """The regression: this returned True for every one of these."""
        assert _ready(_coordinator(ctx, gen)) is False, (
            f"coordinator reported ready with no {gone} server"
        )

    def test_recovers_when_the_worker_comes_back(self):
        """Deliberately not sticky.

        A metadata-driven deployment adds and removes workers as a matter of
        course. Latching 'dead' on the first removal would turn a routine
        topology change into a permanently unhealthy coordinator.
        """
        c = _coordinator(["ctx0"], ["gen0"])
        assert _ready(c) is True
        c._gen_router._servers.clear()  # worker dies
        assert _ready(c) is False
        c._gen_router._servers.append("gen1")  # replacement arrives
        assert _ready(c) is True

    def test_cluster_manager_path_is_untouched(self):
        """With a cluster manager, readiness is delegated to it verbatim."""
        seen = {}

        class _CM:
            async def is_ready_with_router(self, n_ctx, n_gen):
                seen["args"] = (n_ctx, n_gen)
                return False  # distinct from what the fallback would say

        c = _coordinator(["ctx0", "ctx1"], ["gen0"], cluster_manager=_CM())
        assert _ready(c) is False, "the cluster manager's verdict must win"
        assert seen["args"] == (2, 1), "router counts are forwarded unchanged"

    def test_static_deployment_is_unchanged(self):
        """No metadata server means no monitor, so the lists never shrink.

        That shape keeps its old behaviour: readiness cannot be derived from a
        list nothing maintains, so reporting anything but ready would be a
        regression for static deployments.
        """
        c = _coordinator([], [], metadata_server=None)
        for _ in range(5):
            assert _ready(c) is True

    def test_generation_only_benchmark_needs_no_context_server(self, monkeypatch):
        """TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 configures no ctx servers.

        Requiring one there would make /health permanently 503 for that mode.
        """
        monkeypatch.setenv(GEN_ONLY_BENCHMARK_ENV, "1")
        assert _ready(_coordinator([], ["gen0"])) is True
        # A generation worker is still mandatory even in this mode.
        assert _ready(_coordinator([], [])) is False

    @pytest.mark.parametrize("ctx_stale,gen_stale", [(True, False), (False, True)])
    def test_stale_monitoring_fails_closed(self, ctx_stale, gen_stale):
        """A monitor that stopped leaves `servers` frozen and healthy-looking.

        Readiness must not trust a list nothing is refreshing, even though
        both roles still appear populated.
        """
        c = _coordinator(["ctx0"], ["gen0"], ctx_stale=ctx_stale, gen_stale=gen_stale)
        assert _ready(c) is False, "stale monitoring must report not-ready"


class _StubMetadataServer:
    """Minimal JsonDictionary surface used by ``fetch_live_servers``."""

    def __init__(self, entries: dict[str, dict[str, str]]) -> None:
        # {key: {"url": ...}} for keys under 'trtllm/'
        self._entries = dict(entries)

    def keys(self) -> list[str]:
        return list(self._entries)

    def get(self, key: str) -> Optional[dict[str, str]]:
        return self._entries.get(key)

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)


class TestMonitorDrivesReadiness:
    """A real Router through ``_monitor_servers()``.

    The point brnguyen2 raised: the stub-router tests above can only confirm
    the predicate. These assert the state it depends on is actually reachable.
    """

    @staticmethod
    def _router(
        servers: Iterable[str], healthy: set[str], role: ServerRole = ServerRole.CONTEXT
    ) -> RoundRobinRouter:
        """A real RoundRobinRouter whose health check answers from `healthy`."""
        servers = list(servers)
        router = RoundRobinRouter(
            server_role=role,
            servers=servers,
            metadata_server_cfg=None,
            metadata_server=_StubMetadataServer({f"trtllm/{s}": {"url": s} for s in servers}),
        )

        async def _check(server_url: str) -> bool:
            return server_url in healthy

        router._check_server_health = _check
        return router

    def test_monitor_publishes_empty_list_when_role_dies(self):
        """The state the predicate depends on must be reachable.

        Before the fix, ``_filter_servers_by_role`` raised on an empty live
        list, the monitor's ``except`` re-raised, the task died, and
        ``servers`` kept its stale entries -- so this assertion failed.
        """

        async def scenario():
            router = self._router(["ctx0"], healthy=set())  # worker is dead
            published = asyncio.Event()
            inner_updated = router._on_servers_updated

            def updated_then_signal(old_servers: list[str], new_servers: list[str]) -> None:
                inner_updated(old_servers, new_servers)
                # `_on_servers_updated` runs under the monitor's lock, after
                # `self._servers` has been reassigned. Signalling here -- and
                # not around the fetch -- is what makes the assertion below
                # read a published list rather than race the publish.
                published.set()

            router._on_servers_updated = updated_then_signal
            await router.start_server_monitoring(poll_interval=0.01)

            async def body():
                await _await_progress(published, "publish a server-list change")
                return list(router.servers), not router._monitor_task.done()

            return await _run_monitored(router, body)

        servers, still_running = asyncio.run(scenario())
        assert servers == [], (
            "monitor left a stale server list; readiness can never see the role as gone"
        )
        assert still_running, (
            "monitor task died on the empty-role poll, so nothing would be updated from then on"
        )

    def test_monitor_survives_a_failing_poll(self):
        """A transient metadata error must not end monitoring.

        A dead monitor freezes ``servers``, and readiness then reports a
        long-gone cluster as healthy.
        """

        async def scenario():
            router = self._router(["ctx0"], healthy={"ctx0"})
            calls = {"n": 0}
            polled_again = asyncio.Event()

            async def flaky_fetch() -> dict[str, str]:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("metadata server unreachable")
                # The retry is the whole point of the test, so wait for it
                # rather than for a duration that merely usually contains it.
                polled_again.set()
                return {"trtllm/ctx0": "ctx0"}

            router.fetch_live_servers = flaky_fetch
            await router.start_server_monitoring(poll_interval=0.01)

            async def body():
                await _await_progress(polled_again, "poll again after the error")
                return not router._monitor_task.done(), calls["n"]

            return await _run_monitored(router, body)

        alive, n_calls = asyncio.run(scenario())
        assert alive, "monitor task died on a transient poll error"
        assert n_calls > 1, "monitor did not poll again after the error"

    def test_dead_monitor_is_reported_stale(self):
        """``monitoring_is_stale`` is what makes readiness fail closed."""

        async def scenario():
            router = self._router(["ctx0"], healthy={"ctx0"})

            async def boom():
                raise RuntimeError("fatal")

            task = asyncio.create_task(boom())
            router._monitor_task = task
            try:
                await task
            except RuntimeError:
                pass
            return router.monitoring_is_stale(30.0)

        assert asyncio.run(scenario()) is True

    def test_static_router_is_never_stale(self):
        """No monitor task means a static list, which cannot go stale."""
        router = self._router(["ctx0"], healthy={"ctx0"})
        assert router._monitor_task is None
        assert router.monitoring_is_stale(0.0) is False

    def test_monitoring_that_never_succeeds_ages_into_staleness(self):
        """Failing from the very first poll must still fail closed.

        Routers are built with the static list from the disagg config even
        when a metadata server is configured, so ``servers`` starts non-empty.
        If metadata is unreachable from startup onwards, no poll ever
        succeeds. Keying staleness only off the last *successful* poll would
        leave the reference point unset forever, the monitor would never look
        stale, and readiness would keep trusting that never-updated list --
        the same fail-open this PR closes, entered from startup instead of
        from a worker dying later.
        """

        async def scenario():
            router = self._router(["gen0"], healthy={"gen0"}, role=ServerRole.GENERATION)
            calls = {"n": 0}
            failed_twice = asyncio.Event()

            async def always_fails():
                calls["n"] += 1
                # Two failures, not one: it proves the loop retried rather
                # than merely entered once. Waiting on this is also what stops
                # the assertions below from passing vacuously -- with a fixed
                # sleep, a monitor that never got scheduled at all would leave
                # `_last_successful_poll` unset and look identical to one that
                # tried and failed.
                if calls["n"] >= 2:
                    failed_twice.set()
                raise RuntimeError("metadata server unreachable")

            router.fetch_live_servers = always_fails
            # The real entry point: it is what records the monitor's start.
            await router.start_server_monitoring(poll_interval=0.01)

            async def body():
                await _await_progress(failed_twice, "retry after a failed poll")

                # The precondition for the fail-open: alive, never succeeded,
                # and still holding the list it was constructed with.
                alive = not router._monitor_task.done()
                never_succeeded = router._last_successful_poll is None
                servers = list(router.servers)

                # Age the monitor deterministically instead of waiting out a
                # real bound: the assertion is about the staleness arithmetic,
                # and tying it to wall-clock would put CI load back in the
                # loop. 60s of failing polls against a 30s bound.
                router._monitor_started_at -= 60.0

                c = _coordinator(["ctx0"], [])
                c._gen_router = router
                c._monitor_staleness_secs = 30.0
                ready = await c.is_ready()

                # A window the monitor has not yet outlived must NOT report
                # stale, or a merely slow first poll would flap /health during
                # startup.
                c._monitor_staleness_secs = 300.0
                generous = await c.is_ready()
                return alive, never_succeeded, servers, ready, generous, calls["n"]

            return await _run_monitored(router, body)

        alive, never_succeeded, servers, ready, generous, n_calls = asyncio.run(scenario())
        assert alive, "monitor died; this test is meant to cover the still-running case"
        assert n_calls >= 2, "the monitor never actually retried a failing poll"
        assert never_succeeded, "no poll should have succeeded"
        assert servers == ["gen0"], "the un-refreshed initial list is what readiness must not trust"
        assert ready is False, (
            "monitoring never succeeded past the staleness bound, yet /health "
            "still promised readiness"
        )
        assert generous is True, (
            "within the staleness bound a not-yet-landed first poll must not report not-ready"
        )
