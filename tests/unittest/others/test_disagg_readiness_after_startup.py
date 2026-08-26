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


class _StubRouter:
    """Just the surface ``is_ready`` touches."""

    def __init__(self, servers, stale=False):
        self._servers = list(servers)
        self._stale = stale

    @property
    def servers(self):
        return self._servers

    @property
    def num_prepared_servers(self):
        return len(self._servers)

    def monitoring_is_stale(self, max_age_secs):
        return self._stale


def _coordinator(
    ctx_servers,
    gen_servers,
    cluster_manager=None,
    metadata_server=object(),
    ctx_stale=False,
    gen_stale=False,
):
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


def _ready(c):
    return asyncio.run(c.is_ready())


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

    def __init__(self, entries):
        # {key: {"url": ...}} for keys under 'trtllm/'
        self._entries = dict(entries)

    def keys(self):
        return list(self._entries)

    def get(self, key):
        return self._entries.get(key)

    def remove(self, key):
        self._entries.pop(key, None)


class TestMonitorDrivesReadiness:
    """A real Router through ``_monitor_servers()``.

    The point brnguyen2 raised: the stub-router tests above can only confirm
    the predicate. These assert the state it depends on is actually reachable.
    """

    @staticmethod
    def _router(servers, healthy):
        """A real RoundRobinRouter whose health check answers from `healthy`."""
        router = RoundRobinRouter(
            server_role=ServerRole.CONTEXT,
            servers=list(servers),
            metadata_server_cfg=None,
            metadata_server=_StubMetadataServer({f"trtllm/{s}": {"url": s} for s in servers}),
        )

        async def _check(server_url):
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
            # One poll, then stop: _monitor_servers loops forever by design.
            task = asyncio.create_task(router._monitor_servers(poll_interval=0.01))
            router._monitor_task = task
            await asyncio.sleep(0.1)
            still_running = not task.done()
            servers = list(router.servers)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return servers, still_running

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

            async def flaky_fetch():
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("metadata server unreachable")
                return {"trtllm/ctx0": "ctx0"}

            router.fetch_live_servers = flaky_fetch
            task = asyncio.create_task(router._monitor_servers(poll_interval=0.01))
            router._monitor_task = task
            await asyncio.sleep(0.15)
            alive = not task.done()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return alive, calls["n"]

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
