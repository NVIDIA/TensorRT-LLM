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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py
"""Request and response models of the VisualGen endpoints.

Image generation, image edit and video. ``tensorrt_llm.serve.openai_protocol``
re-exports them lazily under their old names.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import UploadFile
from pydantic import AliasChoices, Field, field_validator, model_validator

from tensorrt_llm.serve.openai_protocol import OpenAIBaseModel
from tensorrt_llm.visual_gen.params import MediaRole


class ImageGenerationRequest(OpenAIBaseModel):
    """OpenAI-compatible image generation request.

    Universal per-request fields map 1:1 to :class:`VisualGenParams`.
    Model-specific knobs (``stg_scale``, ``guidance_rescale``, …)
    travel through ``extra_params``; the executor validates each
    key against the loaded pipeline's
    ``extra_param_specs``. Unknown top-level fields are rejected
    with HTTP 422 via the inherited ``extra="forbid"`` policy.
    """

    # Prompt + transport (OpenAI-standard, always honored)
    prompt: str
    response_format: Literal["url", "b64_json", "path"] = "url"
    format: Literal["png", "webp", "jpeg", "safetensors", "pt"] = Field(
        default="png",
        description=(
            "Generation content encoding format. Image encoders write "
            "``png``/``webp``/``jpeg``; tensor encoders write "
            "``safetensors``/``pt`` for programmatic post-processing."),
    )
    seed: Optional[int] = Field(default=None,
                                ge=0,
                                description="Random seed for reproducibility.")

    # Resolution. ``size`` is OpenAI-shaped; ``width`` + ``height`` are an
    # equivalent structured alternative. Exactly one of width/height is
    # rejected by the paired validator below. Numeric fields use
    # ``gt=0`` as a safety net so zero / negative inputs are rejected
    # with HTTP 422 before reaching the pipeline.
    size: Optional[str] = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)

    # TRT-LLM-supported per-request params (1:1 with VisualGenParams fields)
    num_inference_steps: Optional[int] = Field(default=None, gt=0)
    guidance_scale: Optional[float] = Field(default=None, gt=0)
    max_sequence_length: Optional[int] = Field(default=None, gt=0)
    negative_prompt: Optional[str] = None
    n: Optional[int] = Field(
        default=None,
        gt=0,
        le=10,
        description=("Number of images to generate. Capped at 10 to match the "
                     "OpenAI images API and to bound GPU memory / disk usage."),
    )

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys."),
    )

    # Accepted-but-ignored OpenAI-shaped fields. The conversion no-ops; the
    # server logs WARNING when a client sets ``quality`` or ``style``, and
    # WARNING-on-mismatch for ``model``. Kept in the schema so OpenAI-SDK
    # clients don't trip ``extra="forbid"``.
    model: Optional[str] = None
    quality: Optional[Literal["standard", "hd"]] = None
    style: Optional[Literal["vivid", "natural"]] = None
    user: Optional[str] = None

    @model_validator(mode="after")
    def _check_paired_dimensions(self):
        """Reject sending exactly one of ``width`` / ``height``.

        Either both are sent (structured resolution wins over ``size``)
        or neither is sent (``size`` or pipeline default applies).
        """
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "width and height must be sent together; got width="
                f"{self.width!r}, height={self.height!r}")
        return self


class ImageEditRequest(OpenAIBaseModel):
    """OpenAI-compatible image editing request.

    The server accepts the OpenAI multipart shape and a JSON/base64
    shape for tests and non-SDK clients. Model-specific knobs travel
    through ``extra_params`` and are validated by the loaded visual
    generation pipeline.
    """

    prompt: str
    image: Union[str, UploadFile, List[Union[str, UploadFile]]] = Field(
        description="Input image or images to edit.")
    mask: Optional[Union[str, UploadFile]] = Field(
        default=None,
        description=
        "Optional edit mask. Currently accepted for compatibility but unsupported.",
    )
    response_format: Literal["url", "b64_json", "path"] = "url"
    output_format: Literal["png", "webp", "jpeg"] = Field(
        default="png",
        validation_alias=AliasChoices("output_format", "format"),
        description="Edited image content encoding format.",
    )
    seed: Optional[int] = Field(default=None,
                                ge=0,
                                description="Random seed for reproducibility.")

    size: Optional[str] = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)

    num_inference_steps: Optional[int] = Field(default=None, gt=0)
    guidance_scale: Optional[float] = Field(default=None, gt=0)
    max_sequence_length: Optional[int] = Field(default=None, gt=0)
    negative_prompt: Optional[str] = None
    n: Optional[int] = Field(
        default=None,
        gt=0,
        le=10,
        description=("Number of edited images to generate. Capped at 10 to "
                     "match the OpenAI images API."),
    )

    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys."),
    )

    model: Optional[str] = None
    quality: Optional[Literal["standard", "hd"]] = None
    user: Optional[str] = None

    @model_validator(mode="after")
    def _check_paired_dimensions(self):
        if isinstance(self.image, list) and not self.image:
            raise ValueError("image must contain at least one input image")
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "width and height must be sent together; got width="
                f"{self.width!r}, height={self.height!r}")
        return self


class ImageObject(OpenAIBaseModel):
    """Generated image object in the response."""
    b64_json: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(OpenAIBaseModel):
    """Response from image generation endpoint.

    ``output_format`` reports the encoding actually applied to the
    returned bytes / files so clients can decode or label the payload
    correctly. Image encoders are ``"png"``/``"webp"``/``"jpeg"``;
    tensor formats are ``"safetensors"``/``"pt"``.
    """

    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageObject]
    output_format: Literal["png", "webp", "jpeg", "safetensors", "pt"] = "png"
    quality: Literal["low", "medium", "high"] = "medium"
    size: Optional[str] = None


class MediaReferenceItem(OpenAIBaseModel):
    """One media reference (image / video / audio) for conditioning (mirrors ``MediaRef``).

    ``format`` declares how to read ``content``; it is required, so no wire form
    is ever guessed. The request field it sits in (``image_reference`` /
    ``video_reference`` / ``audio_reference``) fixes the modality. ``role`` is
    required only when it is ambiguous — a model with more than one required role
    for that modality; a single role, or a single required role (e.g. i2v
    first_frame), is inferred.
    """

    content: str = Field(
        description="The reference payload, in the form declared by ``format``."
    )
    format: Literal["path", "url", "base64"] = Field(description=(
        "Wire form of ``content``: ``path`` (a file readable by the server; a "
        "``file://`` URI is also accepted), ``url`` (``http(s)``, fetched "
        "through the SSRF-guarded loader), or ``base64`` (a ``data:`` URI is "
        "also accepted). Raw bytes reach the server as a multipart upload, "
        "which carries no ``format`` of its own. Distinct from the top-level "
        "``format``, which selects the *output* encoding."))
    role: Optional[MediaRole] = Field(
        default=None,
        description="Which conditioning slot this reference fills. Required only "
        "when the target model accepts this modality in more than one slot; omit "
        "it when the model leaves no ambiguity.",
    )


class VideoGenerationRequest(OpenAIBaseModel):
    """Video generation request (extended API).

    Universal per-request fields map 1:1 to :class:`VisualGenParams`.
    Model-specific knobs travel through ``extra_params``. Unknown
    top-level fields are rejected with HTTP 422 via the inherited
    ``extra="forbid"`` policy.
    """

    # Prompt + transport
    prompt: str
    response_format: Literal["file", "path"] = "file"
    format: Literal["mp4", "avi", "auto", "safetensors", "pt"] = Field(
        default="auto",
        description=(
            "Generation content encoding format. Video encoders write "
            "``mp4``/``avi``/``auto``; tensor encoders write "
            "``safetensors``/``pt`` and carry video, audio, and scalar "
            "metadata (frame rate, audio sample rate) in one payload."),
    )
    seed: Optional[int] = Field(default=None,
                                ge=0,
                                description="Random seed for reproducibility.")
    image_reference: Optional[Union[
        UploadFile, MediaReferenceItem, List[MediaReferenceItem]]] = Field(
            default=None,
            description=
            ("Image reference(s) conditioning generation (e.g. image-to-video "
             "first frame). Send a ``{content, format, role}`` object or a list "
             "of them, where ``format`` is ``path`` / ``url`` / ``base64``; or "
             "upload a single image file via multipart, whose form is implied. "
             "PNG or JPEG only — HEIF/AVIF are not supported."),
        )
    video_reference: Optional[Union[
        UploadFile, MediaReferenceItem, List[MediaReferenceItem]]] = Field(
            default=None,
            description=
            ("Video reference(s) conditioning generation (video-to-video). Send "
             "a ``{content, format}`` object or a list of them, where ``format`` "
             "is ``path`` / ``url`` / ``base64``; or upload a single video file "
             "via multipart, whose form is implied. MP4 or AVI, with H.264 the "
             "tested codec and others best-effort."),
        )
    audio_reference: Optional[Union[
        UploadFile, MediaReferenceItem, List[MediaReferenceItem]]] = Field(
            default=None,
            description=
            ("Audio reference(s) conditioning generation. Send a "
             "``{content, format}`` object or a list of them, where ``format`` "
             "is ``path`` / ``url`` / ``base64``; or upload a single audio "
             "file via multipart, whose form is implied. Accepted only by "
             "models that declare an audio reference slot."),
        )
    input_reference: Optional[Union[str, UploadFile]] = Field(
        default=None,
        description=
        ("Deprecated. A single image or video reference, routed by content "
         "signature to image-to-video or video-to-video. A JSON request carries "
         "base64 bytes; a multipart request uploads the file. Kept for backward "
         "compatibility; prefer the typed ``image_reference`` / "
         "``video_reference`` fields, which take precedence — this field is "
         "ignored whenever a typed ``image_reference`` or ``video_reference`` "
         "is provided."),
    )

    # Resolution
    size: Optional[str] = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)

    # Frame budget. ``num_frames`` is preferred; if absent the engine
    # derives it from ``seconds * frame_rate``. ``frame_rate`` is the
    # canonical name (matches the Python field); ``fps`` is an alias for
    # OpenAI-shape clients via ``populate_by_name=True``.
    # All three constrain to strictly positive values so a zero
    # ``frame_rate`` (division-by-zero in the AVI fallback) or a
    # negative ``num_frames`` are rejected with HTTP 422 before
    # reaching the encoder.
    # Upper bounds keep request-boundary protection against requests
    # that can exhaust GPU memory or pin the server on unbounded work.
    # The numbers are generous (a minute of video at 120 fps) so common
    # workloads pass; clients that need larger budgets can lift the cap
    # at deployment time.
    num_frames: Optional[int] = Field(default=None, gt=0, le=7200)
    seconds: Optional[float] = Field(default=None, gt=0, le=60.0)
    frame_rate: Optional[float] = Field(default=None,
                                        alias="fps",
                                        gt=0,
                                        le=120.0)

    # TRT-LLM-supported per-request params (1:1 with VisualGenParams)
    num_inference_steps: Optional[int] = Field(default=None, gt=0)
    guidance_scale: Optional[float] = Field(default=None, gt=0)
    max_sequence_length: Optional[int] = Field(default=None, gt=0)
    negative_prompt: Optional[str] = None

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys."),
    )

    # Accepted-but-ignored OpenAI-shaped field
    model: Optional[str] = None

    @model_validator(mode="after")
    def _check_paired_dimensions(self):
        """Reject sending exactly one of ``width`` / ``height``."""
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "width and height must be sent together; got width="
                f"{self.width!r}, height={self.height!r}")
        return self

    @field_validator("response_format", mode="before")
    @classmethod
    def _reject_removed_response_format(cls, value):
        """Give migrating callers an actionable error for removed values.

        ``url``/``b64_json`` were valid before the transport rewrite; run
        before the ``Literal`` check so the error names the replacement
        instead of the generic "Input should be 'file' or 'path'".
        """
        removed = {
            "url":
            ("'url' was removed for video; use 'file' (raw bytes -- the "
             "old 'url' behavior, renamed) or 'path' (server-side path)."),
            "b64_json": ("'b64_json' was removed for video; use 'file' (raw "
                         "bytes) or 'path' (server-side path)."),
        }
        if isinstance(value, str) and value in removed:
            raise ValueError(removed[value])
        return value


class VideoJob(OpenAIBaseModel):
    """Metadata for an asynchronous video generation job.

    Follows the OpenAI Videos API specification:
    https://platform.openai.com/docs/api-reference/videos
    """
    completed_at: Optional[int] = Field(
        default=None, description="Unix timestamp of completion")
    created_at: int = Field(description="Unix timestamp of creation")
    error: Optional[str] = Field(default=None,
                                 description="Error message if failed")
    expires_at: Optional[int] = Field(
        default=None, description="Unix timestamp of expiration")
    id: str = Field(description="Unique identifier for the video")
    model: str = Field(description="The model used for generation")
    object: str = Field(default="video", description="Object type")
    progress: Optional[int] = Field(
        default=None,
        description="Progress of the video generation job (0-100)")
    prompt: str = Field(description="The prompt used to generate the video")
    status: Literal["queued", "generating", "postprocessing", "completed",
                    "failed"] = Field(description=(
                        "Current status of the video generation job. "
                        "``generating`` (model inference) becomes "
                        "``postprocessing`` (encode and/or write the output "
                        "file) when inference finishes, then ``completed`` "
                        "once downloadable via ``/content``."))

    # Video properties
    duration: Optional[float] = Field(default=None,
                                      description="Video duration in seconds")
    fps: Optional[float] = Field(
        default=None,
        description=(
            "Frames per second. Float to preserve cinematic rates such "
            "as 23.976 / 29.97 that some encoders / pipelines use."),
    )
    size: Optional[str] = Field(default=None,
                                description="Video dimensions in 'WxH' format")
    # exclude=True: internal file-location for /content resolution + delete;
    # never on the wire (the path payload is the hand-built {id, output_path}
    # envelope in /content), so status/list model_dump() stays status-only.
    output_path: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Server-side saved path (internal; excluded from the wire)."
    )
    output_paths: Optional[List[str]] = Field(
        default=None,
        exclude=True,
        description=
        "Server-side paths for n>1 (internal; excluded from the wire).")
    # exclude=True internal timings, never on the wire (status/list
    # model_dump() stays status-only). ``request_started`` carries the
    # steady-clock ``server_arrival_time``; the background task computes
    # ``total`` from it and stores the header timings
    # (``generation``/``denoise``/``total``) in ``timing_metrics`` so
    # ``/content`` emits the same Server-Timing header as the sync route.
    request_started: Optional[float] = Field(default=None, exclude=True)
    timing_metrics: Optional[Dict[str, float]] = Field(default=None,
                                                       exclude=True)
    response_format: Optional[Literal["file", "path"]] = Field(
        default=None,
        description=(
            "Transport the client requested. ``GET /v1/videos/{id}/content`` "
            "honors this: ``path`` returns the server-side output path(s) in a "
            "JSON envelope; ``file`` (or unset) returns the file as a "
            "``FileResponse`` download."),
    )


class VideoJobList(OpenAIBaseModel):
    """Response from listing video jobs endpoint."""
    data: List[VideoJob] = Field(description="List of video jobs")
    object: str = Field(default="list", description="Object type")
