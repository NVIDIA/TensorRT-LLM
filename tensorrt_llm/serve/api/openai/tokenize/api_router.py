# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``TokenizeServing``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route functions only map HTTP to it.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.openai.tokenize.serving import TokenizeServing
from tensorrt_llm.serve.openai_protocol.tokenize import TokenizeRequest

APP_STATE_KEY = "openai_serving_tokenize"

router = make_router()


def _serving(raw_request: Request) -> TokenizeServing:
    # No app.state default: wiring bugs fail loudly.
    serving: TokenizeServing = raw_request.app.state.openai_serving_tokenize
    return serving


@router.post("/_internal/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request) -> JSONResponse:
    # raw_request only locates the serving object; FastAPI leaves Request
    # parameters out of the OpenAPI document, so the body schema is unchanged.
    return await _serving(raw_request).tokenize(request)


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a TokenizeServing) must be set before attaching "
            "tensorrt_llm.serve.api.openai.tokenize; the composition root sets it in the LLM "
            "branch."
        )
    app.include_router(router)
