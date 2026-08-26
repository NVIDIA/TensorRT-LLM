from __future__ import annotations

import asyncio
import base64
import binascii
import os
from collections.abc import Mapping
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from tensorrt_llm.inputs.media_io import sniff_media_kind
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ImageEditRequest,
    ImageGenerationRequest,
    VideoGenerationRequest,
)

if TYPE_CHECKING:
    from fastapi import UploadFile

    # Type-only: importing tensorrt_llm.visual_gen at runtime would pull the
    # whole visual_gen tree into every LLM serving process.
    from tensorrt_llm.visual_gen import VisualGen, VisualGenParams

IMAGE_EDIT_MAX_IMAGES = 16
IMAGE_EDIT_MAX_IMAGE_BYTES = 50 * 1024 * 1024
IMAGE_EDIT_MAX_TOTAL_IMAGE_BYTES = 256 * 1024 * 1024
IMAGE_EDIT_MAX_OUTPUT_IMAGES = 64
_IMAGE_EDIT_INPUT_FORMATS = {"PNG", "JPEG"}
_INVALID_IMAGE_EDIT_INPUT_MESSAGE = "image edit input is not a PNG/JPEG image"

# Per-field warnings for OpenAI-shaped knobs that the engine has no
# semantic for. Each entry maps the request attribute to the message
# logged when the client sends a non-None value.
_NO_SEMANTIC_FIELD_WARNINGS: Dict[str, str] = {
    "quality": (
        "Request field 'quality' accepted for OpenAI-SDK compatibility but "
        "ignored; pass 'num_inference_steps' for explicit step control."
    ),
    "style": (
        "Request field 'style' accepted for OpenAI-SDK compatibility but "
        "ignored; the engine has no equivalent semantic."
    ),
}


def _warn_if_set_with_no_semantic(
    request: ImageGenerationRequest | VideoGenerationRequest,
    loaded_model_id: Optional[str] = None,
) -> None:
    """Log WARNING for OpenAI-shape fields the engine cannot honor.

    ``model`` is warn-on-mismatch (trtllm-serve is single-model per
    process). ``quality`` and ``style`` are warn-on-set. ``user`` is
    accepted silently — it's an OpenAI trace field with no engine
    semantic and keeps request logs clean.
    """
    for field, message in _NO_SEMANTIC_FIELD_WARNINGS.items():
        if getattr(request, field, None) is not None:
            logger.warning(message)
    model_value = getattr(request, "model", None)
    if model_value is not None and loaded_model_id is not None and model_value != loaded_model_id:
        logger.warning(
            "Request field 'model'=%r does not match the loaded model "
            "%r; the model field is logged but ignored.",
            model_value,
            loaded_model_id,
        )


def _merge_extra_params(
    params: VisualGenParams,
    request_extras: Optional[Dict[str, Any]],
    extra_param_specs: Dict[str, Any],
) -> None:
    """Shallow-merge request ``extra_params`` into ``params.extra_params``.

    Pipeline defaults are already populated in ``params.extra_params``
    by ``generator.default_params``. Per-key behavior:

    - Known key + non-null value: override the default.
    - Known key + ``null`` value: keep the pipeline default. The
      pre-seeded default already encodes the right state; do not pop
      so pipelines that genuinely distinguish ``None`` from "absent"
      see the same value they would for a client that omitted the key.
    - Unknown key + any value (including ``null``): pass through to
      ``params.extra_params`` so the executor's strict-key validation
      raises ``unknown_extra_param``. This is the key guarantee
      against silent typos — schema-blind null stripping would let
      ``{"stg_sclae": null}`` produce a 200 with retained defaults.

    When the request supplies no extras and the pipeline declared
    none either, the params dict is normalized to ``None`` to match
    the convention that "no extras" is the absence of the dict.
    """
    if request_extras:
        if params.extra_params is None:
            params.extra_params = {}
        for key, value in request_extras.items():
            if key in extra_param_specs and value is None:
                continue
            params.extra_params[key] = value

    if not params.extra_params:
        params.extra_params = None


def local_media_path_is_disallowed() -> bool:
    """Whether server-side filesystem paths are refused at the HTTP boundary.

    One switch for both directions, because both are the same trust question:
    a ``path`` reference has the server read its own disk, and
    ``response_format='path'`` has it disclose where it wrote. Either is what a
    co-located client wants and what a remote one has no business doing, and
    the code cannot tell the two deployments apart, so both are allowed by
    default and turned off together with ``TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1``.
    The local Python API is unaffected: this gate is the HTTP boundary's.
    """
    raw = os.environ.get("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "0")
    if raw not in ("0", "1"):
        logger.warning(
            "Unrecognized value for TRTLLM_DISALLOW_LOCAL_MEDIA_PATH: "
            f"{raw!r}. Expected '0' or '1'. Treating as '0' "
            "(server-side paths allowed)."
        )
    return raw == "1"


def _reference_transport(ref: Any) -> tuple[Any, str, Optional[str]]:
    """Extract ``(content, format, role)`` from one raw HTTP reference.

    ``ref`` is a multipart ``UploadFile`` (has ``.file``) or a
    ``MediaReferenceItem`` exposing ``content`` / ``format`` / ``role``. An
    upload is read to ``bytes`` here — the only decode the boundary owns — and
    its format is implied by the transport rather than declared by the client;
    everything else passes through for the engine to resolve.
    """
    if hasattr(ref, "file"):  # multipart UploadFile
        return ref.file.read(), "bytes", getattr(ref, "role", None)
    content = getattr(ref, "content", None)
    if not isinstance(content, str):
        raise ValueError("reference item must carry a 'content' string.")
    if ref.format == "path" and local_media_path_is_disallowed():
        raise ValueError(
            "reference format='path' is disallowed on this server "
            "(TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1); it reads server-side files "
            "and is only meaningful for co-located clients. Send the file as "
            "base64 or upload it via multipart/form-data."
        )
    return content, ref.format, getattr(ref, "role", None)


def _build_reference_list(value: Any) -> Optional[list]:
    """Normalize one HTTP reference field into a list of ``MediaRef`` objects.

    ``value`` is None, a multipart ``UploadFile``, a ``MediaReferenceItem``, or
    a list of those. Each entry becomes a ``MediaRef`` carrying its transport
    content plus the declared (or, for an upload, implied) wire format.
    Resolution to bytes happens later, at the engine choke point.
    """
    if value is None:
        return None
    # Local import: the visual_gen tree is already loaded in a VisualGen serving
    # process, and this keeps it out of every plain-LLM process (see TYPE_CHECKING).
    from tensorrt_llm.visual_gen.params import MediaRef

    raw_items = value if isinstance(value, list) else [value]
    refs = []
    for item in raw_items:
        content, content_format, role = _reference_transport(item)
        refs.append(MediaRef(content=content, format=content_format, role=role))
    return refs


def _read_image_edit_upload(value: Any) -> bytes:
    """Read a multipart image-edit upload, capping it while it streams."""
    total = 0
    if hasattr(value.file, "seek"):
        value.file.seek(0)
    chunks = []
    while True:
        chunk = value.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > IMAGE_EDIT_MAX_IMAGE_BYTES:
            raise ValueError(
                "Image edit input exceeds the per-image byte limit "
                f"({total} > {IMAGE_EDIT_MAX_IMAGE_BYTES})."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_image_edit_string(value: str) -> bytes:
    """Decode one base64 image-edit input, refusing paths and URLs.

    OpenAI's ``image`` field is base64 or an upload. A path or URL there would
    ask the server to fetch on the client's behalf, which this endpoint has
    never offered.
    """
    decoded = _decode_base64_media(value)
    if decoded is not None:
        return decoded
    if urlparse(value).scheme in ("file", "http", "https"):
        raise ValueError(
            "Image edit inputs must be uploaded files or base64-encoded images; "
            "local paths and URLs are not supported."
        )
    raise ValueError("String image edit inputs must be base64-encoded image data.")


def _build_image_edit_reference_list(value: Any) -> Optional[list]:
    """Build references from an image-edit request's OpenAI-shaped ``image``.

    That field follows OpenAI's schema, which has no place to declare a wire
    form, so each entry is decoded here — the same read a multipart upload
    already gets — and carried on as ``bytes``. The size and PNG/JPEG checks
    stay at the boundary because only this endpoint imposes them.
    """
    if value is None:
        return None
    from tensorrt_llm.visual_gen.params import MediaRef

    refs = []
    total_bytes = 0
    for item in value if isinstance(value, list) else [value]:
        if hasattr(item, "file"):  # multipart UploadFile
            payload = _read_image_edit_upload(item)
        elif isinstance(item, str):
            payload = _decode_image_edit_string(item)
        else:
            raise ValueError(
                "Image edit inputs must be base64-encoded images or uploaded files, "
                f"got {type(item).__name__}."
            )
        if len(payload) > IMAGE_EDIT_MAX_IMAGE_BYTES:
            raise ValueError(
                "Image edit input exceeds the per-image byte limit "
                f"({len(payload)} > {IMAGE_EDIT_MAX_IMAGE_BYTES})."
            )
        _validate_png_jpeg_image(payload)
        total_bytes += len(payload)
        if total_bytes > IMAGE_EDIT_MAX_TOTAL_IMAGE_BYTES:
            raise ValueError(
                "Image edit inputs exceed the total byte limit "
                f"({total_bytes} > {IMAGE_EDIT_MAX_TOTAL_IMAGE_BYTES})."
            )
        refs.append(MediaRef(content=payload, format="bytes"))
    return refs


def _decode_inline_media(extra_params: dict | None, specs) -> None:
    """Turn base64 strings into bytes for extra params declared as media.

    JSON has no byte type, so a client can only inline binary as base64. Any
    extra param whose spec accepts ``bytes`` is decoded here, at the HTTP
    boundary, so pipelines keep a bytes-only contract and never parse
    transport encodings. Values that already arrived as bytes (multipart)
    pass through.
    """
    if not extra_params:
        return
    for key, value in list(extra_params.items()):
        spec = specs.get(key) if specs else None
        if spec is None or "bytes" not in getattr(spec, "type", ""):
            continue
        if isinstance(value, Mapping):
            inner = value.get("control")
            if isinstance(inner, str):
                extra_params[key] = {**value, "control": _b64(key, inner)}
        elif isinstance(value, str):
            extra_params[key] = _b64(key, value)


def _b64(key: str, value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:  # binascii.Error subclasses ValueError
        raise ValueError(
            f"extra_params['{key}'] must be base64-encoded media bytes; "
            "it is not valid base64 data."
        ) from exc


def _decode_base64_media(value: str) -> Optional[bytes]:
    payload = value
    if value.startswith("data:"):
        _, sep, payload = value.partition(",")
        if not sep:
            return None
    if len(payload) > ((IMAGE_EDIT_MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError(
            "Image edit input exceeds the per-image byte limit "
            f"before decoding ({len(payload)} encoded bytes)."
        )
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None


def _validate_png_jpeg_image(value: bytes) -> None:
    try:
        with Image.open(BytesIO(value)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(_INVALID_IMAGE_EDIT_INPUT_MESSAGE) from exc
    if image_format not in _IMAGE_EDIT_INPUT_FORMATS:
        raise ValueError(_INVALID_IMAGE_EDIT_INPUT_MESSAGE)


def _resolve_image_edit_layer_multiplier(
    request: ImageEditRequest,
    generator: VisualGen,
) -> int:
    extra = request.extra_params or {}
    save_layers_to_grid = extra.get("save_layers_to_grid", False)
    if save_layers_to_grid is True:
        return 1
    if save_layers_to_grid not in (False, None):
        raise ValueError(
            "extra_params.save_layers_to_grid must be a bool when estimating image edit output count."
        )

    layer_spec = generator.extra_param_specs.get("layers")
    if layer_spec is None:
        return 1

    layers = extra.get("layers", getattr(layer_spec, "default", 1))
    if layers is None:
        return 1
    if isinstance(layers, bool) or not isinstance(layers, int):
        raise ValueError(
            "extra_params.layers must be an int when estimating image edit output count."
        )
    return layers


def _validate_image_edit_request_limits(
    request: ImageEditRequest,
    generator: VisualGen,
) -> int:
    image_count = len(request.image) if isinstance(request.image, list) else 1
    if image_count > IMAGE_EDIT_MAX_IMAGES:
        raise ValueError(
            f"Image edit accepts at most {IMAGE_EDIT_MAX_IMAGES} input images, got {image_count}."
        )

    output_count = (request.n or 1) * _resolve_image_edit_layer_multiplier(request, generator)
    if output_count > IMAGE_EDIT_MAX_OUTPUT_IMAGES:
        raise ValueError(
            "Image edit request can produce at most "
            f"{IMAGE_EDIT_MAX_OUTPUT_IMAGES} output images, got {output_count}."
        )
    return image_count


def _apply_deprecated_input_reference(
    input_reference: str | UploadFile | None,
    params: VisualGenParams,
    input_reference_format: Optional[str] = None,
) -> None:
    """Back-compat for the deprecated single ``input_reference``.

    Sniff-routes the payload to ``image_reference`` (image) or ``video_reference``
    (video), preserving the pre-typed-fields behavior. Ignored when a typed
    image/video reference is already set — the typed fields take precedence.
    Routing needs the bytes, so the payload is resolved here using the wire form
    from the sibling ``input_reference_format`` (implied for an upload); the
    resolved bytes are then handed to the engine like any other reference.
    """
    if input_reference is None:
        return
    logger.warning("'input_reference' is deprecated; use 'image_reference' / 'video_reference'.")
    if params.image_reference or params.video_reference:
        return
    from tensorrt_llm.visual_gen.params import MediaRef

    if hasattr(input_reference, "file"):  # multipart upload — form implied
        payload = input_reference.file.read()
    else:
        if input_reference_format == "path" and local_media_path_is_disallowed():
            raise ValueError(
                "reference format='path' is disallowed on this server "
                "(TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1); it reads server-side "
                "files and is only meaningful for co-located clients. Send the "
                "file as base64 or upload it via multipart/form-data."
            )
        # Local import, for the reason given at the first one.
        from tensorrt_llm.visual_gen.params import _resolve_reference

        payload = _resolve_reference(input_reference, input_reference_format)
    kind = sniff_media_kind(payload)
    if kind == "image":
        params.image_reference = [MediaRef(content=payload, format="bytes")]
    elif kind == "video":
        params.video_reference = [MediaRef(content=payload, format="bytes")]
    else:
        raise ValueError(
            "input_reference is not a recognized media container; supported "
            "inputs are PNG/JPEG images and MP4/AVI video."
        )


def parse_visual_gen_params(
    request: ImageGenerationRequest | ImageEditRequest | VideoGenerationRequest,
    generator: VisualGen,
) -> VisualGenParams:
    """Translate an HTTP request into :class:`VisualGenParams`.

    Starts from ``generator.default_params`` (already populated with
    pipeline-level defaults plus per-key ``extra_params`` defaults) and
    overlays only the fields the client sent with a non-``None`` value.
    The HTTP layer never invents a default. Validation lives elsewhere:
    Pydantic at the request boundary (422), this helper for translation
    errors (400 via ``ValueError``), and the executor's
    ``validate_visual_gen_params`` for ``extra_params``
    strict-key/type/range checks (400 via ``ValueError``).
    """
    params = generator.default_params

    # Resolution: structured (width + height) wins over the OpenAI-shaped
    # ``size`` string. Sending exactly one of {width, height} is rejected
    # at the Pydantic boundary by the request's model_validator.
    if request.width is not None and request.height is not None:
        params.width, params.height = request.width, request.height
    elif request.size is not None and request.size != "auto":
        params.width, params.height = map(int, request.size.split("x"))
    elif isinstance(request, ImageEditRequest):
        if request.width is None and request.height is None and request.size in (None, "auto"):
            params.width = None
            params.height = None

    # Universal per-request overlays — each guard is the "do not
    # override with None" rule in action.
    if request.negative_prompt is not None:
        params.negative_prompt = request.negative_prompt
    if request.num_inference_steps is not None:
        params.num_inference_steps = request.num_inference_steps
    if request.guidance_scale is not None:
        params.guidance_scale = request.guidance_scale
    if request.max_sequence_length is not None:
        params.max_sequence_length = request.max_sequence_length
    if request.seed is not None:
        params.seed = int(request.seed)

    if isinstance(request, ImageGenerationRequest):
        if request.n is not None:
            params.num_images_per_prompt = request.n

    elif isinstance(request, ImageEditRequest):
        if request.mask is not None:
            raise ValueError("Image edit mask input is not supported yet.")
        if request.n is not None:
            params.num_images_per_prompt = request.n
        _validate_image_edit_request_limits(request, generator)
        params.image_reference = _build_image_edit_reference_list(request.image)

    elif isinstance(request, VideoGenerationRequest):
        if request.frame_rate is not None:
            params.frame_rate = request.frame_rate
        # num_frames wins; otherwise derive from seconds * frame_rate
        # (using whichever frame_rate is now in effect on params).
        if request.num_frames is not None:
            params.num_frames = request.num_frames
        elif request.seconds is not None:
            if params.frame_rate is None:
                raise ValueError(
                    f"Cannot derive 'num_frames' from seconds={request.seconds}: "
                    "neither the request nor the loaded pipeline declares a "
                    "'frame_rate'. Pass 'fps' / 'frame_rate' alongside "
                    "'seconds', or pass 'num_frames' directly."
                )
            derived = int(request.seconds * params.frame_rate)
            if derived < 1:
                raise ValueError(
                    f"Derived frame count is {derived} (from seconds="
                    f"{request.seconds} * frame_rate={params.frame_rate}); "
                    "at least 1 frame is required. Pass a larger 'seconds' "
                    "value, a larger 'fps' / 'frame_rate', or 'num_frames' "
                    "directly."
                )
            params.num_frames = derived
        # Reference inputs: hand the pipeline a ``MediaRef`` carrying the
        # transport content (``bytes`` for an upload, the string otherwise).
        # The engine resolves it to bytes; the boundary never touches disk.
        image_refs = _build_reference_list(request.image_reference)
        if image_refs:
            params.image_reference = image_refs
        video_refs = _build_reference_list(request.video_reference)
        if video_refs:
            params.video_reference = video_refs
        audio_refs = _build_reference_list(request.audio_reference)
        if audio_refs:
            params.audio_reference = audio_refs
        _apply_deprecated_input_reference(
            request.input_reference, params, request.input_reference_format
        )

    _warn_if_set_with_no_semantic(request, getattr(generator, "model", None))
    _decode_inline_media(request.extra_params, generator.extra_param_specs)
    _merge_extra_params(params, request.extra_params, generator.extra_param_specs)

    return params


class AsyncDictStore:
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            self._items[key] = value

    async def update_fields(self, key: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            item.update(updates)
            return item

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.get(key)

    async def pop(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._items.pop(key, None)

    async def list_values(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._items.values())


# Global stores shared by OpenAI entrypoints
# [request_id, dict]
VIDEO_STORE = AsyncDictStore()
