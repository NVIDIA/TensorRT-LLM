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
import base64
import json
import os
import time
import traceback
import uuid
from http import HTTPStatus
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError

from tensorrt_llm.logger import logger
from tensorrt_llm.media.encoding import resolve_video_format
from tensorrt_llm.media.tensor_payload import is_tensor_format
from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest, VideoJob, VideoJobList
from tensorrt_llm.serve.visual_gen_metrics import build_visual_gen_timing_headers
from tensorrt_llm.serve.visual_gen_utils import VIDEO_STORE, parse_visual_gen_params
from tensorrt_llm.visual_gen.params import VisualGenParams


def _video_content_type(suffix: str) -> str:
    """Map a video file suffix to its HTTP ``Content-Type``."""
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".avi":
        return "video/x-msvideo"
    return "application/octet-stream"


# File suffixes the GET /v1/videos/{id}/content and DELETE
# /v1/videos/{id} routes try when the stored output_path is missing.
_KNOWN_VIDEO_OUTPUT_SUFFIXES = (".mp4", ".avi", ".safetensors", ".pt")


def _preflight_encoder_format(fmt):
    """Pre-flight an encoder format string before any GPU work.

    Returns the resolved encoder format token, or ``None`` for tensor
    formats (which carry no encoder dependency). Raises ``ValueError``
    for both unsupported format strings and the missing-ffmpeg case on
    ``format='mp4'`` so the route's existing 400 handler renders the
    message; without this normalization the missing-ffmpeg
    ``RuntimeError`` would fall through to the generic 500 handler.
    """
    if is_tensor_format(fmt):
        return None
    try:
        return resolve_video_format(fmt)[0]
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def _b64_json_video_response(video_id: str, fmt: str, path: Path) -> JSONResponse:
    """Build the OpenAI-style ``{id, format, b64_json}`` envelope.

    Reads bytes from a saved video file on disk and base64-inlines them.
    """
    return JSONResponse(
        content={
            "id": video_id,
            "format": fmt,
            "b64_json": base64.b64encode(path.read_bytes()).decode("utf-8"),
        }
    )


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
            # Client-side ValueErrors from content-type parsing, request
            # translation, encoder-format preflight, parameter validation,
            # and the synchronous engine call return 400. Serialization /
            # encoder failures further down (server-side) fall through to
            # the outer ``except Exception`` → 500.
            try:
                # Parse request based on content-type
                request = await self._parse_video_generation_request(raw_request)
                video_id = f"video_{uuid.uuid4().hex}"
                params = parse_visual_gen_params(
                    request,
                    video_id,
                    self.generator,
                    media_storage_path=str(self.media_storage_path),
                )
                resolved_encoder_fmt = _preflight_encoder_format(request.format)
                logger.info(
                    f"Generating video: {video_id} with params: {params} and prompt: {request.prompt}"
                )
                sync_video_start = time.perf_counter()
                output = self.generator.generate(inputs=request.prompt, params=params)
            except ValidationError as exc:
                return self._render_pydantic_validation_error(exc)
            except ValueError as exc:
                logger.error(f"Video request error: {exc}")
                return self.create_error_response(str(exc), status_code=HTTPStatus.BAD_REQUEST)

            if output.video is None:
                return self.create_error_response(
                    message="Video generation failed",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            if is_tensor_format(request.format):
                ext = f".{request.format}"
                media_type = "application/octet-stream"
                # Match the encoder-format path: persist one file per batch
                # item, ship the first as the route's primary download
                # (OpenAI sync video API does not define a multi-file
                # response yet — TRTLLM-11579).
                batch_size = output.video.shape[0] if output.video.dim() == 5 else 1
                tensor_paths = [
                    self.media_storage_path / f"{video_id}_{i}{ext}" for i in range(batch_size)
                ]
                saved_paths = output.save(tensor_paths, format=request.format)
                target = saved_paths[0]
                latency = time.perf_counter() - sync_video_start
                logger.info(
                    f"Video {video_id} serialized as tensor: latency={latency:.3f}s "
                    f"generation={getattr(output.metrics, 'generation', 0.0):.3f}s"
                )
                if request.response_format == "b64_json":
                    return _b64_json_video_response(video_id, request.format, target)
                return FileResponse(str(target), media_type=media_type, filename=target.name)

            # Encoder formats: one file per item; ship the first item as
            # the route's primary download (OpenAI sync video API does
            # not define a multi-file response yet — TRTLLM-11579).
            resolved_fmt = resolved_encoder_fmt
            batch_size = output.video.shape[0] if output.video.dim() == 5 else 1
            paths_in = [self.media_storage_path / f"{video_id}_{i}" for i in range(batch_size)]
            _save_kwargs = dict(
                format=resolved_fmt,
                frame_rate=output.frame_rate or request.frame_rate or params.frame_rate,
            )
            if os.environ.get("TRTLLM_VIDEO_ASYNC_ENCODE", "0") == "1":
                # Offload the blocking ffmpeg encode to a thread-pool executor so
                # the event loop can start the next request's diffusion while this
                # video encodes. Only overlaps when >=2 requests are in flight per
                # server (i.e. client num_workers > server count).
                saved_paths = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: output.save(paths_in, **_save_kwargs)
                )
            else:
                saved_paths = output.save(paths_in, **_save_kwargs)
            latency = time.perf_counter() - sync_video_start  # seconds
            metrics = output.metrics
            generation = metrics.generation if metrics is not None else 0.0
            denoise = metrics.denoise if metrics is not None else 0.0
            logger.info(
                f"Video {video_id} generated and encoded: "
                f"latency={latency:.3f}s generation={generation:.3f}s "
                f"denoise={denoise:.3f}s"
            )
            headers = build_visual_gen_timing_headers(metrics)

            # TODO(TRTLLM-11579): the OpenAI Videos API does not yet define a
            # multi-file response, so we return only the first video as a file
            # download while persisting all of them to disk.
            actual_path = saved_paths[0]
            if request.response_format == "b64_json":
                return _b64_json_video_response(
                    video_id, actual_path.suffix.lstrip("."), actual_path
                )
            return FileResponse(
                str(actual_path),
                media_type=_video_content_type(actual_path.suffix),
                filename=actual_path.name,
                headers=headers,
            )

        except ValidationError as exc:
            return self._render_pydantic_validation_error(exc)
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
        """Parse a video generation request from JSON or multipart form data.

        Both content types funnel through ``VideoGenerationRequest`` for
        final validation so the wire contract is identical on either
        path: unknown top-level fields are rejected by
        ``extra="forbid"``, the paired ``width``/``height`` validator
        runs, and the ``fps`` alias is honored via the model's
        ``populate_by_name=True`` config.

        Multipart payloads come in as strings; Pydantic coerces them to
        the declared field types. ``extra_params`` accepts a
        JSON-encoded object as its string form so multipart callers can
        pass model-specific knobs.
        """
        content_type = raw_request.headers.get("content-type", "")

        if "application/json" in content_type:
            body = await raw_request.json()
            return VideoGenerationRequest(**body)

        if "multipart/form-data" in content_type:
            form = await raw_request.form()
            data = {}
            for key in form:
                value = form[key]
                if hasattr(value, "file"):
                    # Uploaded file (``input_reference``) — pass through
                    # so the conversion layer reads ``.file``.
                    data[key] = value
                    continue
                if key == "extra_params":
                    if value == "":
                        continue
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"'extra_params' must be a JSON object string; {exc}"
                        ) from exc
                    continue
                if value == "":
                    continue
                data[key] = value
            return VideoGenerationRequest(**data)

        raise ValueError(
            f"Unsupported content-type: {content_type}. Use 'application/json' or 'multipart/form-data'"
        )

    def _render_pydantic_validation_error(self, exc: ValidationError) -> Response:
        """Render a multipart Pydantic ``ValidationError`` as the LLM envelope.

        The visual-gen-scoped 422 envelope is the same shape JSON requests
        get from the FastAPI ``RequestValidationError`` handler, so JSON
        and multipart clients see indistinguishable bodies on bad
        payloads.
        """
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(seg) for seg in err.get("loc", ()) if seg != "body")
            etype = err.get("type", "")
            msg = err.get("msg", "")
            if etype == "extra_forbidden":
                parts.append(
                    f"Unknown request field {loc!r}. Pass model-specific "
                    "parameters via 'extra_params' instead."
                )
            elif loc:
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(msg)
        message = "; ".join(parts) if parts else str(exc)
        return self.create_error_response(
            message=message,
            err_type="BadRequestError",
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
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
            # Synchronously validate the resolved params against the
            # loaded pipeline's extra-param specs / declared defaults
            # so unknown ``extra_params`` keys and similar engine-side
            # rejections surface as HTTP 400 here instead of becoming
            # a queued job whose background task later fails.
            from tensorrt_llm.visual_gen.params import validate_visual_gen_params

            validate_visual_gen_params(
                params,
                declared_defaults=self.generator.executor.default_generation_params,
                extra_param_specs=self.generator.executor.extra_param_specs,
            )
            _preflight_encoder_format(request.format)
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
                fps=params.frame_rate,
                size=f"{params.width}x{params.height}",
                response_format=request.response_format,
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

        except ValidationError as exc:
            return self._render_pydantic_validation_error(exc)
        except ValueError as e:
            logger.error(f"Async video request error: {e}")
            return self.create_error_response(str(e), status_code=HTTPStatus.BAD_REQUEST)
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

            if is_tensor_format(request.format):
                # One tensor file per batch item, mirroring the encoder
                # path; the async job records all paths on
                # ``output_paths`` so subsequent GETs can find each item.
                batch_size = output.video.shape[0] if output.video.dim() == 5 else 1
                tensor_paths = [
                    self.media_storage_path / f"{video_id}_{i}.{request.format}"
                    for i in range(batch_size)
                ]
                saved_paths = output.save(tensor_paths, format=request.format)
            else:
                resolved_fmt, _ = resolve_video_format(request.format)
                batch_size = output.video.shape[0] if output.video.dim() == 5 else 1
                paths_in = [self.media_storage_path / f"{video_id}_{i}" for i in range(batch_size)]
                saved_paths = output.save(
                    paths_in,
                    format=resolved_fmt,
                    frame_rate=output.frame_rate or request.frame_rate or params.frame_rate,
                )
            latency = time.perf_counter() - background_start  # seconds
            metrics = output.metrics
            generation = metrics.generation if metrics is not None else 0.0
            denoise = metrics.denoise if metrics is not None else 0.0
            logger.info(
                f"Video {video_id} async-generated and encoded: "
                f"latency={latency:.3f}s generation={generation:.3f}s "
                f"denoise={denoise:.3f}s"
            )
            job = await VIDEO_STORE.get(video_id)
            if job:
                job.status = "completed"
                job.completed_at = int(time.time())
                # TODO: Expose VisualGen timing metrics for async jobs once the
                # OpenAI video job metadata contract includes server timings.
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

            # Use the stored output path when present, otherwise probe the
            # well-known output suffixes for this video_id — try both the
            # bare ``{vid}{ext}`` and the batch-indexed ``{vid}_0{ext}``
            # names, matching the convention ``delete_video`` uses.
            video_path = None
            if job.output_path and os.path.exists(job.output_path):
                video_path = Path(job.output_path)
            else:
                for ext in _KNOWN_VIDEO_OUTPUT_SUFFIXES:
                    for name in (f"{video_id}{ext}", f"{video_id}_0{ext}"):
                        candidate = self.media_storage_path / name
                        if os.path.exists(candidate):
                            video_path = candidate
                            break
                    if video_path is not None:
                        break

            if video_path and os.path.exists(video_path):
                suffix = video_path.suffix.lstrip(".")
                # When the original ``POST /v1/videos`` requested
                # ``response_format="b64_json"``, return the bytes
                # as a base64 envelope so the async transport
                # matches what the sync route does for the same
                # ``response_format``.
                if job.response_format == "b64_json":
                    return _b64_json_video_response(video_id, suffix, video_path)
                if is_tensor_format(suffix):
                    media_type = "application/octet-stream"
                else:
                    media_type = _video_content_type(video_path.suffix)
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
                for ext in _KNOWN_VIDEO_OUTPUT_SUFFIXES:
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
