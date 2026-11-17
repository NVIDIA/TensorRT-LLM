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
"""Route snapshot of the VisualGen serving configuration.

Lives under a ``visual_gen/`` path so that VisualGen-only pull requests, whose
CI keeps only test entries containing ``visual_gen/``, still run it.
"""

import pytest
from test_common.serve.routes import (
    VISUAL_GEN_CONFIGS,
    compare,
    format_failure,
    read_reference,
    snapshot_config,
)

pytestmark = pytest.mark.cpu_only

_DOC_ROUTES = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


@pytest.mark.parametrize("config", VISUAL_GEN_CONFIGS, ids=lambda config: config.name)
def test_visual_gen_routes_match_reference(config, tmp_path):
    reference = read_reference()
    recorded = reference["configs"][config.name]
    live = snapshot_config(config, tmp_path)
    assert recorded.get("error") == live.get("error")
    differences = compare(recorded.get("routes", []), live.get("routes", []))
    assert not differences, format_failure(config.name, differences)


@pytest.mark.parametrize("config", VISUAL_GEN_CONFIGS, ids=lambda config: config.name)
def test_visual_gen_routes_keep_msgpack_route_class(config, tmp_path):
    live = snapshot_config(config, tmp_path)
    for record in live["routes"]:
        methods, path, cls = record.split(" ")[:3]
        if path in _DOC_ROUTES:
            assert cls == "Route", record
        else:
            assert cls == "_MsgspecRoute", (
                f"{record}: every VisualGen API route must keep the msgpack route class, "
                "like every other aggregated-server route."
            )
