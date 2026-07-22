# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tensorrt_llm.openengine._schema_pin import OPENENGINE_COMMIT
from tensorrt_llm.openengine.server import OpenEngineServer, validate_schema_release
from tensorrt_llm.openengine.servicer import schema_release


def test_schema_release_accepts_exact_pinned_identity() -> None:
    assert validate_schema_release(OPENENGINE_COMMIT) == OPENENGINE_COMMIT


def test_packaged_schema_pin_matches_repository_pin() -> None:
    repository_pin = (
        (Path(__file__).resolve().parents[3] / "OPENENGINE_COMMIT")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert OPENENGINE_COMMIT == repository_pin


@pytest.mark.parametrize(
    "release",
    [
        "",
        "unreleased",
        "main",
        "cea19cb",
        "latest",
        "v0.2.0",
        "signed-tag:1.2.3",
        "a" * 40,
        "a" * 64,
    ],
)
def test_schema_release_rejects_mutable_or_unknown_identity(release: str) -> None:
    with pytest.raises(RuntimeError, match="exactly match"):
        validate_schema_release(release)


def test_schema_release_reads_launch_environment(monkeypatch) -> None:
    commit = "b0cf2a4826d246192dc65b055dab6d2b38d2d67e"
    monkeypatch.setenv("OPENENGINE_SCHEMA_RELEASE", commit)
    assert schema_release() == commit


@pytest.mark.asyncio
async def test_server_stop_cleans_protocol_state_and_stats_on_grpc_failure() -> None:
    calls = []

    class _GrpcServer:
        async def stop(self, grace: float) -> None:
            calls.append(("grpc", grace))
            raise RuntimeError("stop failed")

    class _Servicer:
        class _Tracker:
            async def close_reapers(self, timeout: float) -> bool:
                calls.append(("reapers", timeout))
                return True

        tracker = _Tracker()

        def close(self) -> None:
            calls.append(("servicer", None))

    class _Stats:
        async def stop(self) -> None:
            calls.append(("stats", None))

    class _KvEvents:
        async def stop(self) -> None:
            calls.append(("kv_events", None))

    server = object.__new__(OpenEngineServer)
    server._server = _GrpcServer()
    server.servicer = _Servicer()
    server._kv_event_fanout = _KvEvents()
    server._stats_fanout = _Stats()

    with pytest.raises(RuntimeError, match="stop failed"):
        await server.stop(grace=1.5)

    assert calls == [
        ("grpc", 1.5),
        ("reapers", 1.5),
        ("servicer", None),
        ("kv_events", None),
        ("stats", None),
    ]
