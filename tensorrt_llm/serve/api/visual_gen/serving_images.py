# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VisualGen image endpoints: ``ImageServing`` and its module helpers."""

import base64
import json
import time
import traceback
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Optional, Union

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError

from tensorrt_llm.logger import logger
from tensorrt_llm.media.encoding import image_to_bytes
from tensorrt_llm.media.tensor_payload import is_tensor_format
from tensorrt_llm.serve.api.visual_gen.errors import \
    render_visual_gen_validation_error
from tensorrt_llm.serve.api.visual_gen.metrics import (
    build_visual_gen_server_timings, build_visual_gen_timing_headers)
from tensorrt_llm.serve.api.visual_gen.protocol import (ImageEditRequest,
                                                        ImageGenerationRequest,
                                                        ImageGenerationResponse,
                                                        ImageObject)
from tensorrt_llm.serve.api.visual_gen.serving_base import VisualGenBaseServing
from tensorrt_llm.serve.api.visual_gen.utils import parse_visual_gen_params


def _normalize_image_output(image) -> list:
    """Normalize image output to a list of individual images.

    Handles single tensors, batched 4D tensors, and lists.
    """
    if isinstance(image, list):
        return image
    if hasattr(image, "dim") and image.dim() == 4:
        return [image[i] for i in range(image.shape[0])]
    return [image]


def _image_output_size(image) -> Optional[str]:
    pil_size = getattr(image, "size", None)
    if isinstance(pil_size, tuple) and len(pil_size) >= 2:
        width, height = pil_size[:2]
        return f"{int(width)}x{int(height)}"

    shape = getattr(image, "shape", None)
    if shape is None:
        return None

    dims = tuple(int(dim) for dim in shape)
    if len(dims) == 2:
        height, width = dims
    elif len(dims) == 3:
        if dims[0] in (1, 3, 4) and dims[1] > 4 and dims[2] > 4:
            height, width = dims[1], dims[2]
        else:
            height, width = dims[0], dims[1]
    else:
        return None

    return f"{width}x{height}"


class ImageServing(VisualGenBaseServing):
    """Image generation and editing endpoints (generation, edit, content)."""

    def _supports_image_edit(self) -> bool:
        executor = getattr(self.generator, "executor", None)
        if executor is None:
            return False

        pipeline = getattr(executor, "pipeline", None)
        if pipeline is not None:
            return bool(getattr(pipeline, "supports_image_edit", False))

        supports_image_edit = getattr(executor, "supports_image_edit", None)
        if supports_image_edit is not None:
            return bool(supports_image_edit)

        return False

    def _create_not_supported_error(self, message: str) -> Response:
        return self.create_error_response(
            err_type="NotImplementedError",
            message=message,
            status_code=HTTPStatus.NOT_IMPLEMENTED,
        )

    async def openai_image_generation(self, request: ImageGenerationRequest,
                                      raw_request: Request) -> Response:
        """OpenAI-compatible image generation endpoint.

        Follows the OpenAI Images API specification for image generation,
        with ``request.format`` extended to accept tensor payloads
        (``"safetensors"``/``"pt"``) alongside the PNG/WebP/JPEG encoders.
        """
        request_received = raw_request.state.server_arrival_time
        try:
            image_id = f"image_{uuid.uuid4().hex}"

            path_error = self._reject_disabled_path(request.response_format)
            if path_error is not None:
                return path_error

            # Client-side ValueErrors from request translation and
            # parameter validation are 400. Serialization failures below
            # (server-side: missing media, inconsistent batch) fall
            # through to the outer ``except Exception`` → 500 so the
            # client doesn't get blamed for a server-internal failure.
            try:
                params = parse_visual_gen_params(request, self.generator)
                logger.info(
                    f"Generating image: {image_id} with params: {params} and prompt: {request.prompt}"
                )
                image_gen_start = time.perf_counter()
                # Offload the blocking resolve/enqueue off the event loop but
                # await it (bad params → 400 here); then await generation.
                handle = await self._generate(request.prompt, params)
                output = await handle.aresult()
            except ValueError as exc:
                logger.error(f"Image request error: {exc}")
                return self.create_error_response(
                    message=str(exc),
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            if output.image is None:
                return self.create_error_response(
                    message="Image generation failed",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            if is_tensor_format(request.format):
                # Tensor payloads carry every populated modality in a
                # single file. Match the image-encoder fan-out by
                # emitting one ``ImageObject`` per batch item.
                from tensorrt_llm.media.tensor_payload import infer_batch_size

                ext = f".{request.format}"
                batch_size = infer_batch_size(output)
                if request.response_format == "b64_json":
                    data = [
                        ImageObject(
                            b64_json=base64.b64encode(
                                output._save_bytes(
                                    request.format,
                                    batch_index=i)).decode("utf-8"),
                            revised_prompt=request.prompt,
                        ) for i in range(batch_size)
                    ]
                else:
                    paths_in = [
                        self.media_storage_path / f"{image_id}_{i}{ext}"
                        for i in range(batch_size)
                    ]
                    # Report the paths save() actually wrote, not paths_in --
                    # save() may normalize (e.g. fill a missing extension), so
                    # its return is the on-disk location the client will open().
                    saved = output.save(paths_in, format=request.format)
                    data = [
                        self._image_object(request, raw_request, image_id, i,
                                           saved[i]) for i in range(batch_size)
                    ]
                response = ImageGenerationResponse(
                    created=int(time.time()),
                    data=data,
                    output_format=request.format,
                    size=f"{params.width}x{params.height}",
                )
            else:
                output_images = _normalize_image_output(output.image)
                # Pillow's format name is the upper-case form of our
                # request token. The on-disk extension matches the
                # request token directly; ``.jpeg`` is interchangeable
                # with ``.jpg`` for Pillow, the OS, and the OpenAI API.
                pil_format = request.format.upper()
                ext = f".{request.format}"
                if request.response_format == "b64_json":
                    data = [
                        ImageObject(
                            b64_json=base64.b64encode(
                                image_to_bytes(
                                    image, format=pil_format)).decode("utf-8"),
                            revised_prompt=request.prompt,
                        ) for image in output_images
                    ]
                else:
                    data = []
                    for i, image in enumerate(output_images):
                        path = self.media_storage_path / f"{image_id}_{i}{ext}"
                        path.write_bytes(
                            image_to_bytes(image, format=pil_format))
                        data.append(
                            self._image_object(request, raw_request, image_id,
                                               i, path))
                response = ImageGenerationResponse(
                    created=int(time.time()),
                    data=data,
                    output_format=request.format,
                    size=f"{params.width}x{params.height}",
                )

            latency = time.perf_counter() - image_gen_start  # seconds
            metrics = output.metrics
            generation = metrics.generation if metrics is not None else 0.0
            denoise = metrics.denoise if metrics is not None else 0.0
            logger.info(f"Image {image_id} generated and encoded: "
                        f"latency={latency:.3f}s generation={generation:.3f}s "
                        f"denoise={denoise:.3f}s")
            total = self._adjusted_steady_clock.now() - request_received
            headers = build_visual_gen_timing_headers(
                build_visual_gen_server_timings(metrics, total=total))

            return JSONResponse(content=response.model_dump(), headers=headers)

        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                message=str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    async def get_image_content(
        self,
        image_id: str,
        raw_request: Request,
        i: int = 0,
    ) -> Response:
        """Serve a generated image by ID and batch index.

        ``GET /v1/images/{image_id}/content?i=<batch_index>`` returns
        the image file the corresponding ``POST /v1/images/generations``
        wrote into ``media_storage_path``. ``i`` defaults to ``0`` for
        single-image requests; URL responses for ``n > 1`` requests
        carry ``?i=N`` per item.
        """
        for ext in (".png", ".webp", ".jpg", ".jpeg", ".safetensors", ".pt"):
            candidate = self.media_storage_path / f"{image_id}_{i}{ext}"
            if candidate.exists():
                media_type = ("application/octet-stream"
                              if ext in (".safetensors", ".pt") else
                              f"image/{ext.lstrip('.').replace('jpg', 'jpeg')}")
                return FileResponse(
                    str(candidate),
                    media_type=media_type,
                    filename=candidate.name,
                )
        return self.create_error_response(
            message=f"Image {image_id!r} (batch index {i}) not found",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    @staticmethod
    def _build_image_content_url(raw_request: Request, image_id: str,
                                 i: int) -> str:
        """Return a fetchable HTTP URL for a generated image item."""
        base = str(raw_request.base_url).rstrip("/")
        return f"{base}/v1/images/{image_id}/content?i={i}"

    def _image_object(self, request: Union[ImageGenerationRequest,
                                           ImageEditRequest],
                      raw_request: Request, image_id: str, i: int,
                      path: Path) -> ImageObject:
        """Build the per-item ``ImageObject`` for the ``path``/``url`` transports.

        ``b64_json`` is handled separately. Shared by the generation
        (tensor + encoder) and edit routes so they cannot drift when a
        transport changes.
        """
        if request.response_format == "path":
            return ImageObject(path=str(path), revised_prompt=request.prompt)
        return ImageObject(url=self._build_image_content_url(
            raw_request, image_id, i),
                           revised_prompt=request.prompt)

    async def _parse_image_edit_request(
        self,
        raw_request: Request,
    ) -> ImageEditRequest:
        """Parse an image edit request from JSON or multipart form data."""
        content_type = raw_request.headers.get("content-type", "")
        normalized_content_type = content_type.lower()

        if "application/json" in normalized_content_type:
            body = await raw_request.json()
            if not isinstance(body, dict):
                raise ValueError(
                    "JSON image edit request body must be an object")
            return ImageEditRequest(**body)

        if "multipart/form-data" in normalized_content_type:
            form = await raw_request.form()
            data = {}
            for key in form:
                values = form.getlist(key)
                if key == "image":
                    values = [
                        value for value in values
                        if not (isinstance(value, str) and value == "")
                    ]
                    if not values:
                        continue
                    data[key] = values if len(values) > 1 else values[-1]
                    continue

                value = values[-1]
                if key == "extra_params":
                    if value == "":
                        continue
                    if not isinstance(value, str):
                        raise ValueError(
                            "'extra_params' must be a JSON object string")
                    try:
                        data[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise ValueError(
                            f"'extra_params' must be a JSON object string; {exc}"
                        ) from exc
                    continue
                if value == "":
                    continue
                data[key] = value
            return ImageEditRequest(**data)

        raise ValueError(
            f"Unsupported content-type: {content_type}. Use 'application/json' or 'multipart/form-data'"
        )

    async def openai_image_edit(self, raw_request: Request) -> Response:
        """OpenAI-compatible image editing endpoint."""
        request_received = raw_request.state.server_arrival_time
        if not self._supports_image_edit():
            return self._create_not_supported_error(
                "Image editing is not supported by the loaded visual generation model."
            )

        try:
            image_id = f"image_{uuid.uuid4().hex}"

            try:
                request = await self._parse_image_edit_request(raw_request)
                path_error = self._reject_disabled_path(request.response_format)
                if path_error is not None:
                    return path_error
                params = parse_visual_gen_params(request, self.generator)
                logger.info(
                    f"Editing image: {image_id} with params: {params} and prompt: {request.prompt}"
                )
                image_edit_start = time.perf_counter()
                output = self.generator.generate(inputs=request.prompt,
                                                 params=params)
            except ValidationError as exc:
                return render_visual_gen_validation_error(
                    exc, self.create_error_response)
            except ValueError as exc:
                logger.error(f"Image edit request error: {exc}")
                return self.create_error_response(
                    message=str(exc),
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            if output.image is None:
                return self.create_error_response(
                    message="Image editing failed",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            # Model-specific ``extra_params`` stay pipeline-owned. Layered
            # image-edit models such as Qwen-Image-Layered use
            # ``save_layers_to_grid`` to pack all layers into one image here.
            output_images = _normalize_image_output(output.image)
            output_size = _image_output_size(
                output_images[0]) if output_images else None
            pil_format = request.output_format.upper()
            ext = f".{request.output_format}"
            if request.response_format == "b64_json":
                data = [
                    ImageObject(
                        b64_json=base64.b64encode(
                            image_to_bytes(image,
                                           format=pil_format)).decode("utf-8"),
                        revised_prompt=request.prompt,
                    ) for image in output_images
                ]
            else:
                data = []
                for i, image in enumerate(output_images):
                    path = self.media_storage_path / f"{image_id}_{i}{ext}"
                    path.write_bytes(image_to_bytes(image, format=pil_format))
                    data.append(
                        self._image_object(request, raw_request, image_id, i,
                                           path))

            response = ImageGenerationResponse(
                created=int(time.time()),
                data=data,
                output_format=request.output_format,
                size=output_size,
            )

            latency = time.perf_counter() - image_edit_start
            metrics = output.metrics
            generation = metrics.generation if metrics is not None else 0.0
            denoise = metrics.denoise if metrics is not None else 0.0
            logger.info(f"Image {image_id} edited and encoded: "
                        f"latency={latency:.3f}s generation={generation:.3f}s "
                        f"denoise={denoise:.3f}s")
            total = self._adjusted_steady_clock.now() - request_received
            headers = build_visual_gen_timing_headers(
                build_visual_gen_server_timings(metrics, total=total))

            return JSONResponse(content=response.model_dump(), headers=headers)

        except ValidationError as exc:
            return render_visual_gen_validation_error(
                exc, self.create_error_response)
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.create_error_response(
                message=str(e),
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
