# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``CompletionServing``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route functions only map HTTP to it.
"""

from fastapi import FastAPI, Request
from fastapi.responses import Response

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.openai.completions.serving import CompletionServing
from tensorrt_llm.serve.openai_protocol.completions import CompletionRequest

APP_STATE_KEY = "openai_serving_completions"

router = make_router()


def _serving(raw_request: Request) -> CompletionServing:
    # No app.state default: wiring bugs fail loudly.
    serving: CompletionServing = raw_request.app.state.openai_serving_completions
    return serving


@router.post("/v1/completions")
async def openai_completion(request: CompletionRequest, raw_request: Request) -> Response:
    return await _serving(raw_request).openai_completion(request, raw_request)


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a CompletionServing) must be set before attaching "
            "tensorrt_llm.serve.api.openai.completions; the composition root sets it in the LLM "
            "branch."
        )
    app.include_router(router)
