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
"""The multimodal handoff on an OpenAI context-to-generation round trip.

A context server, the disaggregated router and a generation server run in
process and talk over their real wire format; only the engines are stubbed.
The context engine returns what a context worker returns for one image. The
generation engine runs the real generation-only checks of
``BaseLLM._preprocess`` on what arrives.
"""

import asyncio
import base64
import dataclasses
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import torch
from fastapi import FastAPI
from PIL import Image

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs.multimodal import MultimodalInput
from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_client import MSGPACK_HEADERS, _encode_request_body
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import _MsgspecRoute
from tensorrt_llm.serve.router import Router

from .openai_server_stub import make_stub_openai_server

# No GPU, engine or sockets: this runs in the CPU-Generic CI stage, which
# selects with `-m cpu_only`.
pytestmark = pytest.mark.cpu_only

MODEL = "test-vlm"
IMAGE_TOKEN = 91
# The client sends one image placeholder; the context worker expands it.
CLIENT_TOKEN_IDS = [1, IMAGE_TOKEN, 2]
EXPANDED_TOKEN_IDS = [1, IMAGE_TOKEN, IMAGE_TOKEN, 2]
IDENTITY = MultimodalInput.from_components([[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-0"])
MROPE_POSITION_IDS = torch.arange(12).reshape(3, 1, 4)
MROPE_POSITION_DELTAS = torch.tensor([-2], dtype=torch.int32)


class _StubModelConfig:
    """Enough of an HF config to pick the multimodal placeholder strategy."""

    model_type = "qwen3_vl"
    vocab_size = 1024


class _GenerationProcessor:
    """What the generation-only checks read: the media token and the rotary type."""

    def __init__(self, uses_mrope: bool):
        rope = {"mrope_section": [1, 1, 1]} if uses_mrope else {"rope_type": "default"}
        self.config = SimpleNamespace(text_config=SimpleNamespace(rope_scaling=rope))

    def get_mm_token_ids(self):
        return torch.tensor([IMAGE_TOKEN])

    def __call__(self, *args, **kwargs):
        raise AssertionError("the generation worker must not process inputs")


def _image_url_part() -> dict:
    with io.BytesIO() as buf:
        Image.new("RGB", (8, 8), (0, 128, 255)).save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _worker_app(server) -> FastAPI:
    app = FastAPI()
    # Decode the router's msgpack bodies, as a trtllm-serve worker does.
    app.router.route_class = _MsgspecRoute
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    return app


def _context_app(model_dir, uses_mrope: bool, sends_handoff: bool, tamper=None) -> FastAPI:
    def generate_async(*, inputs, disaggregated_params, **kwargs):
        assert inputs["multi_modal_data"], "the context worker gets the image"
        # A context-only result: BaseLLM._preprocess adds the mRoPE copy and
        # GenerationResult the cache identity. An older context server sends
        # neither.
        handoff = {}
        if sends_handoff:
            handoff["multimodal_input"] = dataclasses.asdict(IDENTITY)
        if sends_handoff and uses_mrope:
            handoff |= {
                "mrope_position_ids_handle": SharedTensorContainer.from_tensor_bytes(
                    MROPE_POSITION_IDS
                ).dump_to_dict(),
                "mrope_position_deltas_handle": SharedTensorContainer.from_tensor_bytes(
                    MROPE_POSITION_DELTAS
                ).dump_to_dict(),
            }
        if tamper is not None:
            tamper(handoff)
        params = dataclasses.replace(
            disaggregated_params,
            first_gen_tokens=[17],
            ctx_request_id=disaggregated_params.disagg_request_id,
            multimodal_positions=IDENTITY.multimodal_positions,
            multimodal_lengths=IDENTITY.multimodal_lengths,
            **handoff,
        )
        output = SimpleNamespace(
            index=0,
            text="",
            token_ids=[17],
            finish_reason="length",
            stop_reason=None,
            logprobs=None,
            disaggregated_params=params,
        )
        return SimpleNamespace(
            request_id=1,
            prompt_token_ids=EXPANDED_TOKEN_IDS,
            finished=True,
            error=None,
            aresult=AsyncMock(),
            outputs=[output],
            cached_tokens=0,
        )

    server = make_stub_openai_server(model_dir, MODEL, _StubModelConfig(), generate_async)
    server.await_disconnected = AsyncMock()
    return _worker_app(server)


def _generation_app(model_dir, uses_mrope: bool, reuse: bool, seen: dict) -> FastAPI:
    llm = BaseLLM.__new__(BaseLLM)
    llm.input_processor = _GenerationProcessor(uses_mrope)
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=reuse))
    llm._is_encoder_decoder_model = lambda: False

    def generate_async(*, inputs, sampling_params, disaggregated_params, **kwargs):
        prompt_token_ids, _, multimodal_params, _ = llm._preprocess(
            inputs, sampling_params, disaggregated_params
        )
        seen.update(prompt_token_ids=prompt_token_ids, multimodal_params=multimodal_params)
        return SimpleNamespace(request_id=2, prompt_token_ids=prompt_token_ids, finished=True)

    server = make_stub_openai_server(model_dir, MODEL, _StubModelConfig(), generate_async)
    # A generation server has a cache transceiver; the handoff fields need one.
    server.generator.args.cache_transceiver_config = {"backend": "DEFAULT"}
    server.await_disconnected = AsyncMock()
    server._create_chat_response = AsyncMock(
        return_value=ChatCompletionResponse(
            model=MODEL,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=len(EXPANDED_TOKEN_IDS), completion_tokens=1),
        )
    )
    return _worker_app(server)


async def _post(app: FastAPI, request) -> httpx.Response:
    """Send a request the way OpenAIHttpClient does: a msgpack body."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        return await client.post(
            "/v1/chat/completions", content=_encode_request_body(request), headers=MSGPACK_HEADERS
        )


def _router(ctx_app: FastAPI, gen_app: FastAPI):
    config = DisaggServerConfig(server_configs=[], schedule_style="context_first")
    coordinator = DisaggCoordinatorService(config, client_factory=lambda *_, **__: AsyncMock())
    service = OpenAIDisaggregatedService(
        config, coordinator, client_factory=lambda *_, **__: AsyncMock()
    )
    for name, server in (("_ctx_router", "ctx:8001"), ("_gen_router", "gen:8002")):
        router = AsyncMock(spec=Router)
        router.get_next_server.return_value = (server, {})
        setattr(coordinator, name, router)
        setattr(service, name, router)

    async def send_ctx(request, **_):
        reply = await _post(ctx_app, request)
        assert reply.status_code == 200, reply.text
        return ChatCompletionResponse(**reply.json())

    async def send_gen(request, **_):
        return await _post(gen_app, request)

    service._ctx_client = SimpleNamespace(send_request=send_ctx)
    service._gen_client = SimpleNamespace(send_request=send_gen)
    return service


def _image_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe the image."}, _image_url_part()],
            }
        ],
        prompt_token_ids=CLIENT_TOKEN_IDS,
        max_tokens=4,
    )


@pytest.mark.parametrize(
    ("ctx_sends_handoff", "uses_mrope", "reuse", "error"),
    [
        pytest.param(True, True, True, None, id="handoff-mrope-reuse"),
        pytest.param(True, False, True, None, id="handoff-plain-rope-reuse"),
        pytest.param(False, True, False, "requires portable mRoPE", id="no-handoff-mrope"),
        pytest.param(False, True, True, "requires portable mRoPE", id="no-handoff-mrope-reuse"),
        pytest.param(
            False, False, True, "KV reuse requires cache identity", id="no-handoff-plain-rope-reuse"
        ),
        pytest.param(False, False, False, None, id="no-handoff-plain-rope-control"),
    ],
)
def test_context_to_generation_round_trip(tmp_path, ctx_sends_handoff, uses_mrope, reuse, error):
    seen = {}
    (tmp_path / "ctx").mkdir()
    (tmp_path / "gen").mkdir()
    service = _router(
        _context_app(tmp_path / "ctx", uses_mrope, ctx_sends_handoff),
        _generation_app(tmp_path / "gen", uses_mrope, reuse, seen),
    )

    reply = asyncio.run(service._send_disagg_request(_image_request()))

    if error is not None:
        assert reply.status_code == 400, reply.text
        assert error in reply.json()["message"]
        return
    assert reply.status_code == 200, reply.text
    assert seen["prompt_token_ids"] == EXPANDED_TOKEN_IDS
    multimodal_params = seen["multimodal_params"]
    if not ctx_sends_handoff:
        assert multimodal_params is None
        return
    assert dataclasses.asdict(multimodal_params.multimodal_input) == dataclasses.asdict(IDENTITY)
    multimodal_params.to_tensor("multimodal_data")
    if uses_mrope:
        mrope = multimodal_params.multimodal_data["mrope_config"]
        assert torch.equal(mrope["mrope_position_ids"], MROPE_POSITION_IDS)
        assert torch.equal(mrope["mrope_position_deltas"], MROPE_POSITION_DELTAS)
    else:
        assert "mrope_config" not in multimodal_params.multimodal_data


def _update(field: str, **values):
    return lambda handoff: handoff[field].update(values)


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        pytest.param(
            _update("mrope_position_ids_handle", tensor_size=[3, 1, 3]),
            "mrope_position_ids_handle shape",
            id="mrope-shape",
        ),
        pytest.param(
            _update("mrope_position_ids_handle", dtype="torch.int32"),
            "mrope_position_ids_handle data must be base64 of 48 bytes",
            id="mrope-size",
        ),
        pytest.param(
            _update("mrope_position_deltas_handle", data="*v///w=="),
            "mrope_position_deltas_handle data must be base64 of 4 bytes",
            id="mrope-encoding",
        ),
        pytest.param(
            _update("multimodal_input", multimodal_hashes=[[1] * 7]), "eight int32", id="hash-width"
        ),
        pytest.param(
            _update("multimodal_input", multimodal_hashes=[[2**31] + [1] * 7]),
            "eight int32",
            id="hash-int32",
        ),
        pytest.param(
            _update("multimodal_input", multimodal_positions=[3]), "exceed the prompt", id="span"
        ),
    ],
)
def test_malformed_handoff_gets_a_client_error(tmp_path, tamper, error):
    """The generation server checks the handoff before it decodes or submits it."""
    seen = {}
    (tmp_path / "ctx").mkdir()
    (tmp_path / "gen").mkdir()
    service = _router(
        _context_app(tmp_path / "ctx", True, True, tamper),
        _generation_app(tmp_path / "gen", True, True, seen),
    )

    reply = asyncio.run(service._send_disagg_request(_image_request()))

    assert reply.status_code == 400, reply.text
    assert error in reply.json()["message"]
    assert not seen
