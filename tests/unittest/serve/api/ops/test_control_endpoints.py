# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The control routers per configuration: resource governor, profiling, runtime control, RL control.

Pins that each router attaches only when its condition holds, that the
constructor check keeps RL and runtime control exclusive, that the signed
dependencies reject unsigned requests and cover the declared route path, and
the constructor contract of the three serving classes.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.routes import effective_routes

from tensorrt_llm.serve.api.ops.control import api_router as control_router
from tensorrt_llm.serve.api.ops.profile import api_router as profile_router
from tensorrt_llm.serve.api.ops.profile.serving import ProfileServing
from tensorrt_llm.serve.api.ops.rl import api_router as rl_router
from tensorrt_llm.serve.api.ops.rl.serving import RLServing
from tensorrt_llm.serve.api.ops.runtime_control import api_router as runtime_control_router
from tensorrt_llm.serve.api.ops.runtime_control.serving import RuntimeControlServing
from tensorrt_llm.serve.rl_control_auth import build_rl_control_auth_headers
from tensorrt_llm.serve.runtime_control_auth import (
    RuntimeControlReplayCache,
    build_runtime_control_auth_headers,
)

# threadleak is off: the tests create TestClients inside the test body and the
# fake server's lazy worker-pool threads are alive when the leak check runs; the
# serve_env fixture closes them in its teardown, which runs after that check
# (precedent: tests/unittest/serve/api/ops/system/test_system_endpoints.py).
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

CONTROL_PATHS = {"/release_memory", "/resume_memory", "/update_weights", "/memory_status"}
GOVERNOR_PATHS = {
    "/_resource_governor/truncate",
    "/_control/kv_cache/truncate",
    "/_control/kv_cache/truncate_tokens",
}


def _paths(app):
    return {path for path, _ in effective_routes(app)}


def _assert_requires_each_dependency(cls, dependencies):
    # The control: with every dependency present, construction succeeds.
    cls(**dependencies)
    for name in dependencies:
        partial = {key: value for key, value in dependencies.items() if key != name}
        try:
            cls(**partial)
        except TypeError:
            continue
        pytest.fail(
            f"{cls.__name__} must require {name!r}: serving dependencies are keyword-only "
            "constructor arguments without defaults (serve/AGENTS.md)"
        )


def test_constructors_require_each_dependency():
    _assert_requires_each_dependency(ProfileServing, {"model": "m", "generator": SimpleNamespace()})
    _assert_requires_each_dependency(
        RuntimeControlServing,
        {
            "model": "m",
            "generator": SimpleNamespace(),
            "runtime_control_api_key": "k",
            "runtime_control_replay_cache": RuntimeControlReplayCache(),
        },
    )
    _assert_requires_each_dependency(
        RLServing, {"model": "m", "generator": SimpleNamespace(), "rl_control_api_key": "k"}
    )


@pytest.mark.parametrize(
    "router",
    [control_router, profile_router, runtime_control_router, rl_router],
    ids=["control", "profile", "runtime_control", "rl"],
)
def test_attach_router_requires_the_serving_object(router):
    with pytest.raises(RuntimeError, match=router.APP_STATE_KEY):
        router.attach_router(FastAPI())


def test_rl_and_runtime_control_mount_only_when_enabled(serve_env):
    neither = serve_env.build_openai_server(AggConfig())
    assert not (_paths(neither.app) & CONTROL_PATHS)

    runtime = serve_env.build_openai_server(AggConfig(runtime_control=True))
    assert _paths(runtime.app) & CONTROL_PATHS == {
        "/release_memory",
        "/resume_memory",
        "/memory_status",
    }
    serving = getattr(runtime.app.state, runtime_control_router.APP_STATE_KEY)
    # The replay cache is the root's own: one nonce ledger per frontend.
    assert serving._runtime_control_replay_cache is runtime._runtime_control_replay_cache
    client = serve_env.client(runtime.app, lifespan=False)
    assert client.get("/memory_status").status_code == 401  # unsigned

    rl = serve_env.build_openai_server(AggConfig(rl_control=True))
    assert _paths(rl.app) & CONTROL_PATHS == {
        "/release_memory",
        "/resume_memory",
        "/update_weights",
    }
    client = serve_env.client(rl.app, lifespan=False)
    assert client.post("/update_weights", json={"weights": None}).status_code == 401  # unsigned

    with pytest.raises(ValueError, match="cannot both be enabled"):
        serve_env.build_openai_server(AggConfig(rl_control=True, runtime_control=True))


def test_runtime_control_signature_covers_the_declared_route_path(serve_env):
    server = serve_env.build_openai_server(AggConfig(runtime_control=True))
    server.generator.release = Mock()
    client = serve_env.client(server.app, lifespan=False)
    body = b""

    signed = {"Content-Type": "application/json"}
    signed.update(build_runtime_control_auth_headers("test-key", "POST", "/release_memory", body))
    assert client.post("/release_memory", content=body, headers=signed).status_code == 200
    server.generator.release.assert_called_once_with(None)

    # A signature over another route's path is rejected: the dependency reads
    # the declared path off the matched route, with no include prefix in the way.
    other = {"Content-Type": "application/json"}
    other.update(build_runtime_control_auth_headers("test-key", "POST", "/resume_memory", body))
    assert client.post("/release_memory", content=body, headers=other).status_code == 401
    server.generator.release.assert_called_once_with(None)


def test_rl_control_accepts_a_signed_request(serve_env):
    server = serve_env.build_openai_server(AggConfig(rl_control=True))
    server.generator.collective_rpc = AsyncMock()
    client = serve_env.client(server.app, lifespan=False)
    body = json.dumps({"tags": ["kv_cache"]}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(build_rl_control_auth_headers("test-key", body))

    assert client.post("/release_memory", content=body, headers=headers).status_code == 200

    server.generator.collective_rpc.assert_awaited_once_with("sleep", args=(["kv_cache"],))


def test_governor_routes_only_with_a_queue(serve_env):
    without = serve_env.build_openai_server(AggConfig())
    assert without.resource_governor is None
    assert not (_paths(without.app) & GOVERNOR_PATHS)

    with_queue = serve_env.build_openai_server(AggConfig(governor_queue=True))
    assert with_queue.resource_governor is not None
    assert (
        getattr(with_queue.app.state, control_router.APP_STATE_KEY) is with_queue.resource_governor
    )
    assert GOVERNOR_PATHS <= _paths(with_queue.app)


def test_profile_routes_drive_the_generator(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.generator.start_profile = Mock()
    server.generator.stop_profile = Mock()
    client = serve_env.client(server.app, lifespan=False)

    response = client.post("/start_profile", json={"num_steps": 3})

    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Profiling started"}
    server.generator.start_profile.assert_called_once_with(
        output_dir=None, num_steps=3, start_step=0, activities=["CPU", "GPU"]
    )
    assert client.post("/stop_profile").status_code == 200
    server.generator.stop_profile.assert_called_once_with()

    # An empty body means the defaults, as before the move.
    server.generator.start_profile.reset_mock()
    assert client.post("/start_profile").status_code == 200
    server.generator.start_profile.assert_called_once_with(
        output_dir=None, num_steps=None, start_step=0, activities=["CPU", "GPU"]
    )
