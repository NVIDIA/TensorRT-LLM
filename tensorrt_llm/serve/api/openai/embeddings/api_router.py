# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``EmbeddingServing``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route functions only map HTTP to it.
"""

from fastapi import FastAPI, Request
from fastapi.responses import Response

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.openai.embeddings.serving import EmbeddingServing
from tensorrt_llm.serve.openai_protocol.embeddings import EmbeddingRequest

APP_STATE_KEY = "openai_serving_embeddings"

router = make_router()


def _serving(raw_request: Request) -> EmbeddingServing:
    # No app.state default: wiring bugs fail loudly.
    serving: EmbeddingServing = raw_request.app.state.openai_serving_embeddings
    return serving


@router.post("/v1/embeddings")
async def openai_embedding(request: EmbeddingRequest, raw_request: Request) -> Response:
    return await _serving(raw_request).openai_embedding(request, raw_request)


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (an EmbeddingServing) must be set before attaching "
            "tensorrt_llm.serve.api.openai.embeddings; the composition root sets it in the "
            "EMBEDDING branch."
        )
    app.include_router(router)
