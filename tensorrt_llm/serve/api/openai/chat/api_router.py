# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``ChatServing``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route functions only map HTTP to it. There are two
routers with one route each, and ``attach_router`` includes the one the
serving object's ``use_harmony`` selects, so ``/v1/chat/completions`` keeps
the endpoint name the server gave it: ``openai_chat``, or ``chat_harmony`` on
a gpt-oss server.
"""

from fastapi import FastAPI, Request
from fastapi.responses import Response

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.openai.chat.serving import ChatServing
from tensorrt_llm.serve.openai_protocol.chat import ChatCompletionRequest

APP_STATE_KEY = "openai_serving_chat"

# Module-level routers, shared by every app in the process; each app includes
# exactly one of them.
router = make_router()
harmony_router = make_router()


def _serving(raw_request: Request) -> ChatServing:
    # No app.state default: wiring bugs fail loudly.
    serving: ChatServing = raw_request.app.state.openai_serving_chat
    return serving


@router.post("/v1/chat/completions")
async def openai_chat(request: ChatCompletionRequest, raw_request: Request) -> Response:
    return await _serving(raw_request).openai_chat(request, raw_request)


@harmony_router.post("/v1/chat/completions")
async def chat_harmony(request: ChatCompletionRequest, raw_request: Request) -> Response:
    return await _serving(raw_request).chat_harmony(request, raw_request)


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a ChatServing) must be set before attaching "
            "tensorrt_llm.serve.api.openai.chat: the /v1/chat/completions endpoint is chosen "
            "from ChatServing.use_harmony; the composition root sets it in the LLM branch."
        )
    serving: ChatServing = app.state.openai_serving_chat
    app.include_router(harmony_router if serving.use_harmony else router)
