# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``RuntimeControlServing``.

Every route of this router carries the signed, replay-protected dependency.
No include prefix and no router prefix: each route declares its full path, so
``scope["route"].path``, which the signature covers, is the same on every
FastAPI version in the pinned range.
"""

from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.ops.runtime_control.serving import RuntimeControlServing
from tensorrt_llm.serve.openai_protocol.control import RuntimeMemoryUpdateRequest

APP_STATE_KEY = "ops_serving_runtime_control"


def _serving(raw_request: Request) -> RuntimeControlServing:
    # No app.state default: wiring bugs fail loudly.
    serving: RuntimeControlServing = raw_request.app.state.ops_serving_runtime_control
    return serving


async def _runtime_control_auth(raw_request: Request) -> None:
    await _serving(raw_request)._require_runtime_control_auth(raw_request)


router = make_router(dependencies=[Depends(_runtime_control_auth)])


@router.post("/release_memory")
async def runtime_release_memory(
    raw_request: Request, request: Optional[RuntimeMemoryUpdateRequest] = None
) -> JSONResponse:
    # raw_request only locates the serving object; FastAPI leaves Request
    # parameters out of the OpenAPI document, so the body schema is unchanged.
    return await _serving(raw_request).runtime_release_memory(request)


@router.post("/resume_memory")
async def runtime_resume_memory(
    raw_request: Request, request: Optional[RuntimeMemoryUpdateRequest] = None
) -> JSONResponse:
    return await _serving(raw_request).runtime_resume_memory(request)


@router.get("/memory_status")
async def runtime_memory_status(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).runtime_memory_status()


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a RuntimeControlServing) must be set before attaching "
            "tensorrt_llm.serve.api.ops.runtime_control; the composition root sets it in the LLM "
            "branch when runtime control endpoints are enabled."
        )
    app.include_router(router)
