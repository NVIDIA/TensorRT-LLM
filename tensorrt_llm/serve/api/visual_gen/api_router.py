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
"""The VisualGen router: the image and video endpoints of the VISUAL_GEN role.

The handlers live on :class:`ImageServing` and :class:`VideoServing`. The
composition root builds both with :func:`init_serving_state` inside its
VISUAL_GEN branch and attaches this router after its role switch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response

from tensorrt_llm.serve._base.routing import make_router
from tensorrt_llm.serve.api.visual_gen.errors import render_visual_gen_validation_error
from tensorrt_llm.serve.api.visual_gen.media import _resolve_media_dir
from tensorrt_llm.serve.api.visual_gen.protocol import ImageGenerationRequest
from tensorrt_llm.serve.api.visual_gen.serving_images import ImageServing
from tensorrt_llm.serve.api.visual_gen.serving_videos import VideoServing

if TYPE_CHECKING:
    from starlette.datastructures import State

    from tensorrt_llm._utils import AdjustedSteadyClock
    from tensorrt_llm.visual_gen.visual_gen import VisualGen

__all__ = [
    "APP_STATE_KEYS",
    "attach_router",
    "init_serving_state",
    "render_visual_gen_validation_error",
    "router",
]

# Route class _MsgspecRoute: every route decodes msgpack request bodies.
router = make_router()

# The serving objects init_serving_state puts on app.state; attach_router checks them.
APP_STATE_KEYS = ("visual_gen_serving_images", "visual_gen_serving_videos")


def _images(raw_request: Request) -> ImageServing:
    # No app.state default: a missing serving object is a wiring bug and fails loudly.
    serving: ImageServing = raw_request.app.state.visual_gen_serving_images
    return serving


def _videos(raw_request: Request) -> VideoServing:
    serving: VideoServing = raw_request.app.state.visual_gen_serving_videos
    return serving


async def openai_image_generation(
    request: ImageGenerationRequest, raw_request: Request
) -> Response:
    return await _images(raw_request).openai_image_generation(request, raw_request)


async def openai_image_edit(raw_request: Request) -> Response:
    return await _images(raw_request).openai_image_edit(raw_request)


async def get_image_content(image_id: str, raw_request: Request, i: int = 0) -> Response:
    return await _images(raw_request).get_image_content(image_id, raw_request, i)


async def openai_video_generation_async(raw_request: Request) -> Response:
    return await _videos(raw_request).openai_video_generation_async(raw_request)


async def openai_video_generation_sync(raw_request: Request) -> Response:
    return await _videos(raw_request).openai_video_generation_sync(raw_request)


async def list_videos(raw_request: Request) -> Response:
    return await _videos(raw_request).list_videos(raw_request)


async def get_video_metadata(video_id: str, raw_request: Request) -> Response:
    return await _videos(raw_request).get_video_metadata(video_id, raw_request)


async def get_video_content(video_id: str, raw_request: Request) -> Response:
    return await _videos(raw_request).get_video_content(video_id, raw_request)


async def delete_video(video_id: str, raw_request: Request) -> Response:
    return await _videos(raw_request).delete_video(video_id, raw_request)


# Image generation endpoints (OpenAI compatible), in registration order.
router.add_api_route("/v1/images/generations", openai_image_generation, methods=["POST"])
router.add_api_route("/v1/images/edits", openai_image_edit, methods=["POST"])
router.add_api_route("/v1/images/{image_id}/content", get_image_content, methods=["GET"])
# Video generation endpoints (extended OpenAI API).
# Asynchronous video generation (returns immediately with job metadata, OpenAI API).
router.add_api_route("/v1/videos", openai_video_generation_async, methods=["POST"])
# Synchronous video generation (waits for completion, extended API).
router.add_api_route("/v1/videos/sync", openai_video_generation_sync, methods=["POST"])
# Deprecated alias of /v1/videos/sync, retained for upstream back-compat after
# the rename; both hit the same sync handler.
router.add_api_route(
    "/v1/videos/generations", openai_video_generation_sync, methods=["POST"], deprecated=True
)
# Video management endpoints.
router.add_api_route("/v1/videos", list_videos, methods=["GET"])
router.add_api_route("/v1/videos/{video_id}", get_video_metadata, methods=["GET"])
router.add_api_route("/v1/videos/{video_id}/content", get_video_content, methods=["GET"])
router.add_api_route("/v1/videos/{video_id}", delete_video, methods=["DELETE"])


def attach_router(app: FastAPI) -> None:
    for key in APP_STATE_KEYS:
        if not hasattr(app.state, key):
            raise RuntimeError(
                f"app.state.{key} is not set: call init_serving_state(app.state, ...) for the "
                "VISUAL_GEN role before attaching tensorrt_llm.serve.api.visual_gen.api_router "
                "(the composition root attaches routers after its role switch)."
            )
    app.include_router(router)


def init_serving_state(
    state: State, *, generator: VisualGen, model: str, adjusted_clock: AdjustedSteadyClock
) -> Path:
    """Build the VisualGen serving objects of one server and put them on ``state``.

    Resolves the media directory once and gives the same directory to the
    image and the video serving object. Returns that directory.
    """
    media_storage_path = _resolve_media_dir()
    state.visual_gen_serving_images = ImageServing(
        generator=generator,
        model=model,
        media_storage_path=media_storage_path,
        adjusted_clock=adjusted_clock,
    )
    state.visual_gen_serving_videos = VideoServing(
        generator=generator,
        model=model,
        media_storage_path=media_storage_path,
        adjusted_clock=adjusted_clock,
    )
    return media_storage_path
