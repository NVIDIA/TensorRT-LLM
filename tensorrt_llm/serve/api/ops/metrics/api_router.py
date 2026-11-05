# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP route of ``MetricsServing``: the JSON ``GET /metrics``.

The composition root puts the serving object on ``app.state`` before the
router is attached; the route function only maps HTTP to it. The Prometheus
``GET /metrics`` lives in ``prometheus.py`` and is attached last, so this route
keeps winning (FastAPI matches in registration order).
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.ops.metrics.serving import MetricsServing

APP_STATE_KEY = "ops_serving_metrics"

router = make_router()


def _serving(raw_request: Request) -> MetricsServing:
    # No app.state default: wiring bugs fail loudly.
    serving: MetricsServing = raw_request.app.state.ops_serving_metrics
    return serving


# The metrics endpoint only reports iteration stats, not the runtime stats for now.
@router.get("/metrics")
async def get_iteration_stats(raw_request: Request) -> JSONResponse:
    return await _serving(raw_request).get_iteration_stats()


def attach_router(app: FastAPI) -> None:
    if not hasattr(app.state, APP_STATE_KEY):
        raise RuntimeError(
            f"app.state.{APP_STATE_KEY} (a MetricsServing) must be set before attaching "
            "tensorrt_llm.serve.api.ops.metrics; the composition root sets it for every role."
        )
    app.include_router(router)
