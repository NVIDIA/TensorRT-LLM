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
"""Route provenance of the system endpoints for the VisualGen serving role.

Lives under a ``visual_gen/`` path so that VisualGen-only pull requests, whose
CI keeps only test entries containing ``visual_gen/``, still run it.
"""

import pytest
from test_common.serve.app_factory import isolated_serve_env
from test_common.serve.routes import VISUAL_GEN_CONFIGS, effective_routes

from tensorrt_llm.serve._base.msgspec_route import _MsgspecRoute

pytestmark = pytest.mark.cpu_only

SYSTEM_PATHS = ("/health", "/version", "/v1/models")
SYSTEM_ROUTER_MODULE = "tensorrt_llm.serve.api.ops.system.api_router"


@pytest.mark.parametrize("config", VISUAL_GEN_CONFIGS, ids=lambda config: config.name)
def test_routes_come_from_ops_system_visual_gen(config, tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(config.build)
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
