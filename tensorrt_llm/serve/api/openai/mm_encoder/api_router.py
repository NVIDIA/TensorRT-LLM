# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``MMEncoderServing``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route functions only map HTTP to it.
"""

from fastapi import FastAPI, Request
from fastapi.responses import Response

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.openai.mm_encoder.serving import MMEncoderServing
from tensorrt_llm.serve.openai_protocol.chat import ChatCompletionRequest

APP_STATE_KEY = "openai_serving_mm_encoder"

router = make_router()


def _serving(raw_request: Request) -> MMEncoderServing:
    # No app.state default: wiring bugs fail loudly.
    serving: MMEncoderServing = raw_request.app.state.openai_serving_mm_encoder
    return serving


@router.post("/v1/chat/completions")
async def openai_mm_encoder(request: ChatCompletionRequest, raw_request: Request) -> Response:
    return await _serving(raw_request).openai_mm_encoder(request, raw_request)


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (an MMEncoderServing) must be set before attaching "
            "tensorrt_llm.serve.api.openai.mm_encoder; the composition root sets it in the "
            "MM_ENCODER branch."
        )
    app.include_router(router)
