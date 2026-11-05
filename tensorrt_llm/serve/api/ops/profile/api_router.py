# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``ProfileServing``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route functions only map HTTP to it.
"""

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.ops.profile.serving import ProfileServing
from tensorrt_llm.serve.openai_protocol.control import StartProfileRequest

APP_STATE_KEY = "ops_serving_profile"

router = make_router()


def _serving(raw_request: Request) -> ProfileServing:
    # No app.state default: wiring bugs fail loudly.
    serving: ProfileServing = raw_request.app.state.ops_serving_profile
    return serving


# Profiling endpoints (PyTorch backend only).
@router.post("/start_profile")
async def start_profile(
    raw_request: Request, request: Optional[StartProfileRequest] = None
) -> JSONResponse:
    # raw_request only locates the serving object; FastAPI leaves Request
    # parameters out of the OpenAPI document, so the body schema is unchanged.
    return await _serving(raw_request).start_profile(request)


@router.post("/stop_profile")
async def stop_profile(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).stop_profile()


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a ProfileServing) must be set before attaching "
            "tensorrt_llm.serve.api.ops.profile; the composition root sets it in the LLM branch."
        )
    app.include_router(router)
