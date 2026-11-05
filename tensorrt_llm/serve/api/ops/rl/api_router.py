# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``RLServing`` (the deprecated RL control endpoints).

Every route of this router carries the HMAC dependency. This router and the
runtime-control one register the same two paths and are never attached
together: the composition root's constructor check keeps them exclusive.
"""

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.ops.rl.serving import RLServing
from tensorrt_llm.serve.openai_protocol.control import MemoryUpdateRequest, UpdateWeightsRequest

APP_STATE_KEY = "ops_serving_rl"


def _serving(raw_request: Request) -> RLServing:
    # No app.state default: wiring bugs fail loudly.
    serving: RLServing = raw_request.app.state.ops_serving_rl
    return serving


async def _rl_control_auth(raw_request: Request) -> None:
    await _serving(raw_request)._require_rl_control_auth(raw_request)


router = make_router(dependencies=[Depends(_rl_control_auth)])


@router.post("/release_memory")
async def release_memory(request: MemoryUpdateRequest, raw_request: Request) -> JSONResponse:
    # raw_request only locates the serving object; FastAPI leaves Request
    # parameters out of the OpenAPI document, so the body schema is unchanged.
    return await _serving(raw_request).release_memory(request)


@router.post("/resume_memory")
async def resume_memory(request: MemoryUpdateRequest, raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).resume_memory(request)


@router.post("/update_weights")
async def update_weights(request: UpdateWeightsRequest, raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).update_weights(request)


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (an RLServing) must be set before attaching "
            "tensorrt_llm.serve.api.ops.rl; the composition root sets it in the LLM branch when "
            "RL control endpoints are enabled."
        )
    app.include_router(router)
