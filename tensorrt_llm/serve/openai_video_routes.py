# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-generation route handlers for the OpenAI-compatible server.

Extracted from ``openai_server.py`` as a small mixin to keep that file
under the project's per-file line budget. ``OpenAIServer`` consumes this
mixin via subclassing; the handlers reach back into the server via
``self`` (e.g. ``self.generator``, ``self.media_storage_path``,
``self.create_error_response``, ``self.video_gen_tasks``), so behavior
is strictly unchanged from the inlined version.
"""

import asyncio
import os
import time
import traceback
import uuid
from http import HTTPStatus
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response

from tensorrt_llm.logger import logger
from tensorrt_llm.media.encoding import resolve_video_format
from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest, VideoJob, VideoJobList
from tensorrt_llm.serve.visual_gen_utils import VIDEO_STORE, parse_visual_gen_params
from tensorrt_llm.visual_gen.params import VisualGenParams


class _VideoRoutesMixin:
    """Mixin providing the eight video-generation endpoints.

    Concrete subclasses (``OpenAIServer``) supply ``self.generator``,
    ``self.media_storage_path``, ``self.video_gen_tasks``, ``self.model``,
    and ``self.create_error_response`` from their own initializer.
    """

    async def openai_video_generation_sync(self, raw_request: Request) -> Response:
        """Synchronous video generation endpoint.

        Waits for video generation to complete before returning.
        Compatible with simple use cases where waiting is acceptable.

        Supports both JSON and multipart/form-data requests:
        - JSON: Send VideoGenerationRequest as application/json
        - Multipart: Send form fields + optional input_reference file
        """
        try:
            # Parse request based on content-type
            request = await self._parse_video_generation_request(raw_request)

            # Resolve the video encode format (mp4/avi/auto)
            resolved_fmt, _ = resolve_video_format(request.output_format)

            video_id = f"video_{uuid.uuid4().hex}"
            params = parse_visual_gen_params(
                request, video_id, self.generator, media_storage_path=str(self.media_storage_path)
            )
            logger.info(
                f"Generating video: {video_id} with params: {params} and prompt: {request.prompt}"
            )

            sync_video_start = time.perf_counter()
            output = self.generator.generate(inputs=request.prompt, params=params)
            if output.video is None:
                return self.create_error_response(
                    message="Video generation failed",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            # Save all generated videos (batch-aware).
            batch_size = output.video.shape[0] if output.video.dim() == 5 else 1
            paths_in = [self.media_storage_path / f"{video_id}_{i}" for i in range(batch_size)]
            saved_paths = output.save(
                paths_in,
                format=resolved_fmt,
                frame_rate=output.frame_rate or request.fps or params.frame_rate,
            )
            latency = time.perf_counter() - sync_video_start  # seconds
            logger.info(
                f"Video {video_id} generated and encoded: "
                f"latency={latency:.3f}s generation={getattr(output.metrics, 'generation', 0.0):.3f}s "
                f"denoise={getattr(output.metrics, 'denoise', 0.0):.3f}s"
            )

            # TODO(TRTLLM-11579): the OpenAI Videos API does not yet define a
            # multi-file response, so we return only the first video as a file
            # download while persisting all of them to disk.
            actual_path = saved_paths[0]
            actual_output_path = str(actual_path)
            media_type = "video/mp4" if actual_path.suffix == ".mp4" else "video/x-msvideo"

            return FileResponse(
                actual_output_path,
                media_type=media_type,
                filename=actual_path.name,
            )

        except ValueError as e:
            logger.error(f"Request parsing error: {e}")
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    async def _parse_video_generation_request(
        self,
        raw_request: Request,
    ) -> VideoGenerationRequest:
        """Parse video generation request from either JSON or multipart/form-data.

        Supports both:
        - application/json: Standard JSON request with VideoGenerationRequest model
        - multipart/form-data: Form fields + file upload for input_reference
        """
        content_type = raw_request.headers.get("content-type", "")

        if "application/json" in content_type:
            # Parse as JSON using Pydantic model
            body = await raw_request.json()
            return VideoGenerationRequest(**body)

        if "multipart/form-data" in content_type:
            # Parse multipart/form-data manually
            form = await raw_request.form()

            # Extract all fields and convert to proper types
            data = {}

            # Required field
            if "prompt" in form:
                data["prompt"] = form["prompt"]
            else:
                raise ValueError("'prompt' is required")

            # Optional string fields
            for field in ["model", "size", "negative_prompt", "output_format"]:
                if field in form and form[field]:
                    data[field] = form[field]

            # Optional numeric fields
            if "seconds" in form and form["seconds"]:
                data["seconds"] = float(form["seconds"])
            if "fps" in form and form["fps"]:
                data["fps"] = int(form["fps"])
            if "n" in form and form["n"]:
                data["n"] = int(form["n"])
            if "num_inference_steps" in form and form["num_inference_steps"]:
                data["num_inference_steps"] = int(form["num_inference_steps"])
            if "guidance_scale" in form and form["guidance_scale"]:
                data["guidance_scale"] = float(form["guidance_scale"])
            if "guidance_rescale" in form and form["guidance_rescale"]:
                data["guidance_rescale"] = float(form["guidance_rescale"])
            if "seed" in form and form["seed"]:
                data["seed"] = int(form["seed"])

            # Handle file upload for input_reference
            if "input_reference" in form:
                input_ref = form["input_reference"]
                if hasattr(input_ref, "file"):  # It's an UploadFile
                    data["input_reference"] = input_ref

            return VideoGenerationRequest(**data)

        else:
            raise ValueError(
                f"Unsupported content-type: {content_type}. Use 'application/json' or 'multipart/form-data'"
            )

    async def openai_video_generation_async(
        self,
        raw_request: Request,
    ) -> Response:
        """Asynchronous video generation endpoint (OpenAI Videos API compatible).

        Creates a video generation job and returns immediately with job metadata.
        The video is generated in the background and stored in media storage.
        Client can poll GET /v1/videos/{video_id} to check status and retrieve the video.

        Supports both JSON and multipart/form-data requests:
        - JSON: Send VideoGenerationRequest as application/json
        - Multipart: Send form fields + optional input_reference file
        """
        try:
            # Parse request based on content-type
            request = await self._parse_video_generation_request(raw_request)

            video_id = f"video_{uuid.uuid4().hex}"
            params = parse_visual_gen_params(
                request, video_id, self.generator, media_storage_path=str(self.media_storage_path)
            )
            logger.info(
                f"Generating video: {video_id} with params: {params} and prompt: {request.prompt}"
            )

            # Persist the queued job before scheduling the background task so
            # that a fast-completing task can always look it up in VIDEO_STORE.
            video_job = VideoJob(
                created_at=int(time.time()),
                id=video_id,
                model=request.model or self.model,
                prompt=request.prompt,
                status="queued",
                duration=request.seconds,
                fps=request.fps,
                size=f"{params.width}x{params.height}",
            )
            await VIDEO_STORE.upsert(video_id, video_job)

            # Start background generation task
            task = asyncio.create_task(
                self._generate_video_background(
                    video_id=video_id,
                    request=request,
                    params=params,
                )
            )
            self.video_gen_tasks[video_id] = task
            task.add_done_callback(lambda t, vid=video_id: self._on_video_task_done(vid, t))

            return JSONResponse(content=video_job.model_dump(), status_code=202)

        except ValueError as e:
            logger.error(f"Request parsing error: {e}")
            return self.create_error_response(str(e))
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    async def _generate_video_background(
        self,
        video_id: str,
        request: VideoGenerationRequest,
        params: VisualGenParams,
    ):
        """Background task to generate video and save to storage."""
        try:
            # Resolve the video encode format (mp4/avi/auto)
            resolved_fmt, _ = resolve_video_format(request.output_format)

            background_start = time.perf_counter()
            future = self.generator.generate_async(inputs=request.prompt, params=params)
            output = await future

            if output.video is None:
                # Update job status to failed since we're in a background task
                job = await VIDEO_STORE.get(video_id)
                if job:
                    job.status = "failed"
                    job.completed_at = int(time.time())
                    job.error = "Video generation failed: output.video is None"
                    await VIDEO_STORE.upsert(video_id, job)
                return

            # Save all generated videos (batch-aware).
            batch_size = output.video.shape[0] if output.video.dim() == 5 else 1
            paths_in = [self.media_storage_path / f"{video_id}_{i}" for i in range(batch_size)]
            saved_paths = output.save(
                paths_in,
                format=resolved_fmt,
                frame_rate=output.frame_rate or request.fps or params.frame_rate,
            )
            latency = time.perf_counter() - background_start  # seconds
            logger.info(
                f"Video {video_id} async-generated and encoded: "
                f"latency={latency:.3f}s generation={getattr(output.metrics, 'generation', 0.0):.3f}s "
                f"denoise={getattr(output.metrics, 'denoise', 0.0):.3f}s"
            )
            job = await VIDEO_STORE.get(video_id)
            if job:
                job.status = "completed"
                job.completed_at = int(time.time())
                # Store the first path on output_path for single-video
                # compatibility, and the full list on output_paths.
                job.output_path = str(saved_paths[0])
                job.output_paths = [str(p) for p in saved_paths]
                await VIDEO_STORE.upsert(video_id, job)

        except Exception as e:
            logger.error(traceback.format_exc())
            job = await VIDEO_STORE.get(video_id)
            if job:
                job.status = "failed"
                job.completed_at = int(time.time())
                job.error = str(e)
                await VIDEO_STORE.upsert(video_id, job)

    async def list_videos(self, raw_request: Request) -> Response:
        """List all generated videos.

        GET /v1/videos
        Returns a list of generated video metadata (job details).
        """
        try:
            # List videos from storage
            video_jobs = await VIDEO_STORE.list_values()

            # Convert to API format
            response = VideoJobList(
                data=video_jobs,
            )
            return JSONResponse(content=response.model_dump())

        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    async def get_video_metadata(self, video_id: str, raw_request: Request) -> Response:
        """Get video metadata by ID.

        GET /v1/videos/{video_id}
        Retrieves the metadata (job status and details) for a specific generated video.
        """
        try:
            logger.info(f"Getting video metadata: {video_id}")
            # Get metadata from storage
            job = await VIDEO_STORE.get(video_id)
            if not job:
                return self.create_error_response(
                    f"Video {video_id} not found",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )

            # Ensure it's a video
            if job.object != "video":
                return self.create_error_response(
                    f"Resource {video_id} is not a video",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            return JSONResponse(content=job.model_dump())

        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    async def get_video_content(self, video_id: str, raw_request: Request) -> Response:
        """Download video file by ID.

        GET /v1/videos/{video_id}/content
        Downloads the generated video file.
        """
        try:
            # Get metadata first to check status
            job = await VIDEO_STORE.get(video_id)
            if not job:
                return self.create_error_response(
                    f"Video {video_id} not found",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )

            # Ensure it's a video and completed
            if job.object != "video":
                return self.create_error_response(
                    f"Resource {video_id} is not a video",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            if job.status != "completed":
                return self.create_error_response(
                    f"Video {video_id} is not ready (status: {job.status})",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            # Try to use stored output path, otherwise check for both .mp4 and .avi
            video_path = None
            if job.output_path and os.path.exists(job.output_path):
                video_path = Path(job.output_path)
            else:
                # Fall back to checking common extensions
                for ext in [".mp4", ".avi"]:
                    candidate = self.media_storage_path / f"{video_id}{ext}"
                    if os.path.exists(candidate):
                        video_path = candidate
                        break

            if video_path and os.path.exists(video_path):
                media_type = "video/mp4" if video_path.suffix == ".mp4" else "video/x-msvideo"
                return FileResponse(
                    video_path,
                    media_type=media_type,
                    filename=video_path.name,
                )
            else:
                return self.create_error_response(
                    f"Video {video_id} not found",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )

        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    async def delete_video(self, video_id: str, raw_request: Request) -> Response:
        """Delete a video by ID.

        DELETE /v1/videos/{video_id}
        Deletes a generated video by its ID.
        """
        try:
            # Check if video exists
            job = await VIDEO_STORE.get(video_id)
            if not job:
                return self.create_error_response(
                    f"Video {video_id} not found",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )

            # Ensure it's a video
            if job.object != "video":
                return self.create_error_response(
                    f"Resource {video_id} is not a video",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            # Cancel any in-flight generation task so it cannot recreate the
            # output file or pin memory after the delete returns.
            task = self.video_gen_tasks.pop(video_id, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            # Delete all generated video files (batch-aware).
            paths_to_delete: list[str] = []
            if job.output_paths:
                paths_to_delete.extend(job.output_paths)
            elif job.output_path:
                paths_to_delete.append(job.output_path)
            else:
                # Fall back to checking common extensions for either the
                # single-file name or the batch-indexed name.
                for ext in [".mp4", ".avi"]:
                    for name in (f"{video_id}{ext}", f"{video_id}_0{ext}"):
                        candidate = self.media_storage_path / name
                        if os.path.exists(candidate):
                            paths_to_delete.append(str(candidate))
                            break
                    if paths_to_delete:
                        break

            for video_path in paths_to_delete:
                if os.path.exists(video_path):
                    os.remove(video_path)

            # Delete from store
            success = await VIDEO_STORE.pop(video_id)

            return JSONResponse(content={"deleted": success is not None})

        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _on_video_task_done(self, video_id: str, task: "asyncio.Task") -> None:
        """Drop a finished background task and surface its outcome.

        Pops ``video_id`` from ``video_gen_tasks`` to bound memory growth
        and logs any unexpected exception the task raised.
        """
        # Pop only if the slot still points at this task — delete_video may
        # have replaced or removed it already.
        if self.video_gen_tasks.get(video_id) is task:
            self.video_gen_tasks.pop(video_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Background video generation task for {video_id} failed: {exc!r}")
