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
"""``make_router()``, ``attach_routers()`` and the role mapping of ``serve/_base/routing.py``."""

import ast
import importlib
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from test_common.serve.app_factory import AggConfig
from test_common.serve.routes import effective_routes

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute
from tensorrt_llm.serve._base.routing import (
    ROUTERS,
    RoleGroup,
    RouterSpec,
    ServerCapabilities,
    attach_routers,
    make_router,
    role_group_of,
)
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _msgpack_encoder

pytestmark = pytest.mark.cpu_only


def test_make_router_uses_msgspec_route():
    router = make_router()
    assert router.route_class is _MsgspecRoute

    @router.get("/x")
    async def x():
        return {}

    assert len(router.routes) == 1
    assert isinstance(router.routes[0], _MsgspecRoute)


def test_make_router_rejects_route_class():
    with pytest.raises(TypeError, match="_MsgspecRoute"):
        make_router(route_class=APIRoute)


def test_role_group_of_every_role():
    expected = {
        None: RoleGroup.LLM,
        ServerRole.CONTEXT: RoleGroup.LLM,
        ServerRole.GENERATION: RoleGroup.LLM,
        ServerRole.MM_ENCODER: RoleGroup.MM_ENCODER,
        ServerRole.EMBEDDING: RoleGroup.EMBEDDING,
        ServerRole.VISUAL_GEN: RoleGroup.VISUAL_GEN,
    }
    # Every ServerRole member must have a row here: a new role must decide its
    # role group (and its routes) explicitly.
    assert set(expected) - {None} == set(ServerRole)
    for role, group in expected.items():
        assert role_group_of(role) is group
        caps = ServerCapabilities.for_role(role)
        assert caps.role is role
        assert caps.role_group is group


_ROUTER_MODULE_TEMPLATE = """
from fastapi import FastAPI

from tensorrt_llm.serve._base.routing import make_router

router = make_router()


@router.get("{path}")
async def handler():
    return {{}}


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
"""


def _write_router_package(tmp_path, package: str, modules: dict) -> None:
    pkg = tmp_path / package
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name, path in modules.items():
        (pkg / f"{name}.py").write_text(_ROUTER_MODULE_TEMPLATE.format(path=path))


def test_attach_routers_imports_only_matching_specs(tmp_path, monkeypatch):
    _write_router_package(
        tmp_path,
        "attachpkg",
        {"llm_router": "/llm-only", "mm_router": "/mm-only", "gated_router": "/gated"},
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module_names = ("attachpkg.llm_router", "attachpkg.mm_router", "attachpkg.gated_router")
    specs = (
        RouterSpec("attachpkg.llm_router", roles=frozenset({RoleGroup.LLM})),
        RouterSpec("attachpkg.mm_router", roles=frozenset({RoleGroup.MM_ENCODER})),
        RouterSpec(
            "attachpkg.gated_router",
            roles=frozenset({RoleGroup.LLM}),
            when=lambda caps: False,
        ),
    )
    try:
        app = FastAPI()
        attach_routers(app, ServerCapabilities.for_role(None), routers=specs)
        assert "attachpkg.llm_router" in sys.modules
        assert "attachpkg.mm_router" not in sys.modules, "role mismatch must not be imported"
        assert "attachpkg.gated_router" not in sys.modules, "a false `when` must not be imported"
        paths = [path for path, _ in effective_routes(app)]
        assert "/llm-only" in paths
        assert "/mm-only" not in paths
        assert "/gated" not in paths
    finally:
        for name in ("attachpkg", *module_names):
            sys.modules.pop(name, None)


def test_router_routes_decode_msgpack():
    router = make_router()

    @router.post("/echo")
    async def echo(raw_request: Request):
        return {"body": await raw_request.json()}

    app = FastAPI()
    app.include_router(router)
    body = _msgpack_encoder.encode({"prompt": [1, 2, 3]})
    with TestClient(app) as client:
        response = client.post("/echo", content=body, headers=MSGPACK_HEADERS)
    assert response.status_code == 200
    assert response.json() == {"body": {"prompt": [1, 2, 3]}}


def _route_path(raw_request: Request) -> str:
    return raw_request.scope["route"].path


def test_include_keeps_route_path():
    # A path prefix goes to make_router(prefix=...), never include_router(prefix=...):
    # FastAPI 0.139 drops include prefixes from scope["route"].path.
    router = make_router(prefix="/p")

    @router.get("/x")
    async def x(route_path: str = Depends(_route_path)):
        return {"route_path": route_path}

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/p/x")
    assert response.status_code == 200
    assert response.json() == {"route_path": "/p/x"}


def test_every_router_spec_module_exposes_attach_router():
    # ROUTERS holds module paths as strings, so a renamed router module would
    # break attachment only at server startup; import each one here instead.
    assert ROUTERS, "ROUTERS must list at least the system router"
    for spec in ROUTERS:
        module = importlib.import_module(spec.module)
        assert callable(getattr(module, "attach_router", None)), (
            f"{spec.module} must define attach_router(app)"
        )
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            is_include_call = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "include_router"
            )
            if is_include_call:
                assert all(keyword.arg != "prefix" for keyword in node.keywords), (
                    f"{spec.module} passes prefix= to include_router; give the prefix to "
                    "make_router(prefix=...) instead: FastAPI 0.139 drops include prefixes "
                    "from scope['route'].path."
                )


_ECHO_ROUTER_MODULE = textwrap.dedent(
    '''
    """A new endpoint is a router plus a serving class: the router reads its
    serving object from app.state; the composition root sets it and attaches."""

    from fastapi import FastAPI, Request

    from tensorrt_llm.serve._base.routing import make_router

    router = make_router()


    @router.post("/v1/test-echo")
    async def echo(raw_request: Request):
        serving = raw_request.app.state.test_serving_echo
        return serving.render(await raw_request.json())


    def attach_router(app: FastAPI) -> None:
        if not hasattr(app.state, "test_serving_echo"):
            raise RuntimeError("app.state.test_serving_echo must be set before attaching")
        app.include_router(router)
    '''
)


class _EchoServing:
    def render(self, body):
        return {"echo": body}


# The TestClient portal thread and the fake server's lazy worker-pool threads
# are alive when the leak check runs; the serve_env fixture closes them in its
# teardown (precedent: tests/unittest/serve/api/ops/system/test_system_endpoints.py).
@pytest.mark.threadleak(enabled=False)
def test_new_endpoint_lands_as_router_and_serving_class(tmp_path, monkeypatch, serve_env):
    pkg = tmp_path / "newendpointpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "api_router.py").write_text(_ECHO_ROUTER_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = RouterSpec("newendpointpkg.api_router", roles=frozenset({RoleGroup.LLM}))
    body = _msgpack_encoder.encode({"prompt": "hi"})
    try:
        server = serve_env.build_openai_server(AggConfig())
        server.app.state.test_serving_echo = _EchoServing()
        attach_routers(server.app, ServerCapabilities.for_role(None), routers=(spec,))
        client = serve_env.client(server.app)
        response = client.post("/v1/test-echo", content=body, headers=MSGPACK_HEADERS)
        assert response.status_code == 200
        assert response.json() == {"echo": {"prompt": "hi"}}

        # The same spec on a role outside its role set: not imported again, not
        # attached, and the route is absent.
        mm_server = serve_env.build_openai_server(AggConfig(role=ServerRole.MM_ENCODER))
        attach_routers(
            mm_server.app, ServerCapabilities.for_role(ServerRole.MM_ENCODER), routers=(spec,)
        )
        mm_client = serve_env.client(mm_server.app)
        mm_response = mm_client.post("/v1/test-echo", content=body, headers=MSGPACK_HEADERS)
        assert mm_response.status_code == 404
    finally:
        for name in ("newendpointpkg", "newendpointpkg.api_router"):
            sys.modules.pop(name, None)
