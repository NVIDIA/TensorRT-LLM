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
"""The system router: /health, /version and /v1/models for every server role."""

from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.ops.system.serving import SystemServing

APP_STATE_KEY = "ops_serving_system"
_UNAVAILABLE = "LLM is unavailable. Please check the server logs for more details."

router = make_router()


def _serving(raw_request: Request) -> SystemServing:
    # No app.state default: wiring bugs fail loudly.
    serving: SystemServing = raw_request.app.state.ops_serving_system
    return serving


@router.get("/health")
async def health(raw_request: Request) -> Response:
    if _serving(raw_request).is_healthy():
        return Response(status_code=200)
    return Response(status_code=503, content=_UNAVAILABLE)


@router.get("/version")
async def version(raw_request: Request) -> JSONResponse:
    return JSONResponse(content=_serving(raw_request).version())


@router.get("/v1/models")
async def get_model(raw_request: Request) -> JSONResponse:
    return JSONResponse(content=_serving(raw_request).models().model_dump())


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a SystemServing) must be set before attaching "
            "tensorrt_llm.serve.api.ops.system; the composition root sets it."
        )
    app.include_router(router)
