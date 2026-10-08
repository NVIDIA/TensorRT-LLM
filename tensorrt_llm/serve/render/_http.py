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
"""The standalone renderer's HTTP app: render routes plus health and server info."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from .api_router import attach_render_routes
from .prompt_types import GENERATE_REQUEST_SCHEMA_VERSION
from .resources import RenderResources
from .serving import ServingRender


def build_render_app(
    resources: RenderResources, *, served_model_name: Optional[str] = None
) -> FastAPI:
    """Build the FastAPI app of a CPU-only renderer.

    Only the render routes, ``/health`` and ``/server_info`` are mounted: there is
    no executor here, so no inference routes and no ``/generate``.
    """
    app = FastAPI(title="TensorRT-LLM renderer")
    serving = ServingRender(lambda: resources)
    # Computed up front so an unusable tokenizer fails the start, not a request.
    serving.fingerprint()
    attach_render_routes(app, serving)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/server_info")
    async def server_info() -> dict:
        return {
            "role": "renderer",
            "model": served_model_name,
            "schema_version": GENERATE_REQUEST_SCHEMA_VERSION,
            "render_fingerprint": serving.fingerprint(),
        }

    return app
