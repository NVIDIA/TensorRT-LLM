# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The num_workers > 1 disaggregated coordinator disables GC like the other disagg servers."""

import gc
from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm.commands import serve

pytestmark = pytest.mark.cpu_only

_DISABLE_GC_ENV = "TRTLLM_DISAGG_SERVER_DISABLE_GC"


@pytest.fixture
def gc_enabled():
    """Start from an enabled collector and restore the caller's state afterwards."""
    was_enabled = gc.isenabled()
    gc.enable()
    yield
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


def _gc_enabled_when_coordinator_serves(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Run the coordinator path without sockets, fleet or server; return its GC state at serve."""
    monkeypatch.delenv(serve.DisaggWorkerEnvs.TLLM_DISAGG_COORDINATOR_PORT, raising=False)
    disagg_config = SimpleNamespace(
        hostname="127.0.0.1",
        port=8000,
        server_keep_alive_timeout=10,
        internal_request_auth_key=None,
    )
    observed = []

    def _run(coroutine):
        observed.append(gc.isenabled())
        coroutine.close()

    with (
        mock.patch.object(serve.socket, "socket"),
        mock.patch.object(serve, "_launch_disagg_fleet", return_value=[]) as launch_fleet,
        mock.patch("tensorrt_llm.serve.disagg_coordinator.DisaggCoordinatorService"),
        mock.patch("tensorrt_llm.serve.coordinator_server.CoordinatorServer"),
        mock.patch.object(serve.asyncio, "run", side_effect=_run),
    ):
        serve._serve_coordinator_and_fleet(disagg_config, "disagg.yaml", None, None, 180, 180, 2)

    launch_fleet.assert_called_once()
    assert len(observed) == 1
    return observed[0]


@pytest.mark.parametrize("value", [None, "1"])
def test_coordinator_disables_gc_by_default(monkeypatch, gc_enabled, value) -> None:
    """Every fleet worker waits on the coordinator, so it must not pause for full collections."""
    if value is None:
        monkeypatch.delenv(_DISABLE_GC_ENV, raising=False)
    else:
        monkeypatch.setenv(_DISABLE_GC_ENV, value)
    assert _gc_enabled_when_coordinator_serves(monkeypatch) is False


def test_coordinator_keeps_gc_when_opted_out(monkeypatch, gc_enabled) -> None:
    """TRTLLM_DISAGG_SERVER_DISABLE_GC=0 keeps the collector on, as on the other disagg servers."""
    monkeypatch.setenv(_DISABLE_GC_ENV, "0")
    assert _gc_enabled_when_coordinator_serves(monkeypatch) is True
