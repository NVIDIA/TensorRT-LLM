# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP routes of ``WorkerServing``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route functions only map HTTP to it.
"""

from typing import Annotated

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.ops.worker.serving import WorkerServing

APP_STATE_KEY = "ops_serving_worker"

router = make_router()


def _serving(raw_request: Request) -> WorkerServing:
    # No app.state default: wiring bugs fail loudly.
    serving: WorkerServing = raw_request.app.state.ops_serving_worker
    return serving


@router.get("/health_generate")
async def health_generate(raw_request: Request) -> Response:
    return await _serving(raw_request).health_generate(raw_request)


@router.get("/v1/data_transceiver_state")
async def data_transceiver_state(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).data_transceiver_state()


@router.get("/energy_metrics")
async def get_energy_metrics(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).get_energy_metrics()


@router.get("/steady_clock_offset")
async def get_steady_clock_offset(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).get_steady_clock_offset()


# Called by the disaggregated server to update the worker reference clock.
@router.post("/steady_clock_offset")
async def set_steady_clock_offset(
    offset: Annotated[float, Body(embed=True)], raw_request: Request
) -> Response:
    return await _serving(raw_request).set_steady_clock_offset(offset)


@router.post("/kv_cache_events")
async def get_kv_cache_events(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).get_kv_cache_events()


@router.get("/server_info")
async def get_server_info(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).get_server_info()


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a WorkerServing) must be set before attaching "
            "tensorrt_llm.serve.api.ops.worker; the composition root sets it in the LLM branch."
        )
    app.include_router(router)
