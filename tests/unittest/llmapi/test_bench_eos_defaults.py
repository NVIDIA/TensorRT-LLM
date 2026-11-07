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
"""EOS defaults of trtllm-bench and benchmark_serving under speculative decoding."""

import asyncio
import socket
from types import SimpleNamespace

import pytest
from aiohttp import web

from tensorrt_llm.bench.benchmark.utils.general import resolve_end_id
from tensorrt_llm.metrics.collector import speculative_config_info
from tensorrt_llm.serve.openai_protocol import ModelCard, ModelList
from tensorrt_llm.serve.scripts.benchmark_serving import (
    server_uses_speculative_decoding,
    speculative_decoding_from_models,
)

pytestmark = pytest.mark.cpu_only

EAGLE3_ARGS = SimpleNamespace(
    speculative_config=SimpleNamespace(
        decoding_type="Eagle3", max_draft_len=3, speculative_model="draft"
    )
)
NO_SPEC_ARGS = SimpleNamespace(speculative_config=None, decoding_config=None)


@pytest.mark.parametrize(
    "llm_args, end_id, expected",
    [
        (EAGLE3_ARGS, None, 2),
        (NO_SPEC_ARGS, None, -1),
        (EAGLE3_ARGS, -1, -1),
        (NO_SPEC_ARGS, 7, 7),
    ],
)
def test_resolve_end_id(llm_args, end_id, expected):
    assert resolve_end_id(llm_args, 2, end_id) == expected


def test_speculative_config_info():
    assert speculative_config_info(NO_SPEC_ARGS) is None
    assert speculative_config_info(EAGLE3_ARGS) == {
        "spec_enabled": "true",
        "spec_method": "Eagle3",
        "spec_num_tokens": "3",
        "spec_draft_model": "draft",
    }


def _models_payload(args):
    card = ModelCard(id="m", metadata={"speculative_config": speculative_config_info(args)})
    return ModelList(data=[card]).model_dump()


@pytest.mark.parametrize(
    "payload, expected",
    [
        (_models_payload(EAGLE3_ARGS), True),
        (_models_payload(NO_SPEC_ARGS), False),
        (ModelList(data=[ModelCard(id="m")]).model_dump(), None),
        ({"object": "list", "data": []}, None),
        ({"data": [{"id": "m", "metadata": {}}]}, None),
    ],
)
def test_speculative_decoding_from_models(payload, expected):
    assert speculative_decoding_from_models(payload) is expected


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("args, expected", [(EAGLE3_ARGS, True), (NO_SPEC_ARGS, False)])
def test_server_uses_speculative_decoding(args, expected):
    async def run():
        app = web.Application()

        async def models(_):
            return web.json_response(_models_payload(args))

        app.router.add_get("/v1/models", models)
        runner = web.AppRunner(app)
        await runner.setup()
        port = _free_port()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        try:
            return await server_uses_speculative_decoding(f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) is expected


def test_server_uses_speculative_decoding_unreachable():
    url = f"http://127.0.0.1:{_free_port()}"
    assert asyncio.run(server_uses_speculative_decoding(url)) is None
