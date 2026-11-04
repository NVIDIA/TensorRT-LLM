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
"""/health, /version and /v1/models of the VisualGen serving role."""

import pytest
from test_common.serve.app_factory import AggConfig, isolated_serve_env

import tensorrt_llm.version
from tensorrt_llm.llmapi.disagg_utils import ServerRole

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def visual_gen_client(tmp_path):
    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(
            AggConfig(role=ServerRole.VISUAL_GEN, model_name="fake-visual-gen")
        )
        yield env.client(server.app)


def test_health_always_succeeds_today(visual_gen_client):
    # The VisualGen generator exposes no health probe, so /health answers 200
    # unconditionally today; a change to that is a behavior change.
    assert visual_gen_client.get("/health").status_code == 200


def test_version(visual_gen_client):
    response = visual_gen_client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": tensorrt_llm.version.__version__}


def test_models_lists_the_served_model(visual_gen_client):
    response = visual_gen_client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [entry["id"] for entry in body["data"]] == ["fake-visual-gen"]
