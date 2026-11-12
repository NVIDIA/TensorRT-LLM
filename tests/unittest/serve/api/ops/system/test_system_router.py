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
"""The ops.system router: attachment contract and route provenance per role."""

import pytest
from fastapi import FastAPI
from test_common.serve.app_factory import AggConfig
from test_common.serve.routes import effective_routes

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute
from tensorrt_llm.serve.api.ops.system.api_router import attach_router
from tensorrt_llm.serve.api.ops.system.serving import SystemServing

pytestmark = pytest.mark.cpu_only

SYSTEM_PATHS = ("/health", "/version", "/v1/models")
SYSTEM_ROUTER_MODULE = "tensorrt_llm.serve.api.ops.system.api_router"


def test_attach_requires_serving_state():
    # The composition root must set the serving object before attaching, so a
    # wiring bug fails at startup, not with an AttributeError per request.
    with pytest.raises(RuntimeError, match="ops_serving_system"):
        attach_router(FastAPI())


def test_system_serving_requires_arguments():
    # No defaults: a serving class built without its collaborators must fail at
    # construction, not fall back silently.
    with pytest.raises(TypeError):
        SystemServing()


@pytest.mark.parametrize(
    "role",
    [None, ServerRole.MM_ENCODER, ServerRole.EMBEDDING],
    ids=["llm", "mm_encoder", "embedding"],
)
def test_routes_come_from_ops_system(role, serve_env):
    server = serve_env.build_openai_server(AggConfig(role=role))
    by_path = {path: [] for path in SYSTEM_PATHS}
    for path, route in effective_routes(server.app):
        if path in by_path:
            by_path[path].append(route)
    for path, routes in by_path.items():
        assert len(routes) == 1, f"{path} must be registered exactly once, found {len(routes)}"
        route = routes[0]
        assert isinstance(route, _MsgspecRoute), (
            f"{path} must be a _MsgspecRoute; a plain APIRoute breaks msgpack request "
            "bodies from the disaggregated frontend"
        )
        assert route.endpoint.__module__ == SYSTEM_ROUTER_MODULE
