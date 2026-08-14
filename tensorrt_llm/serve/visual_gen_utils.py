from __future__ import annotations

import asyncio
import base64
import binascii
import os
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from tensorrt_llm.inputs.media_io import (
    _normalize_file_uri,
    _safe_request_get,
    is_isobmff_image_bytes,
    sniff_media_kind,
)
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


def _read_reference_payload(reference: str) -> bytes:
    """Decode one base64 (optionally ``data:`` URI) reference string to bytes.

    Payload size is deliberately not checked here: encoded size is not part
    of the request-validity contract, and body limits belong to the
    proxy/ASGI deployment layer (HTTP 413). Base64 decodes strictly so
    malformed encodings — not sizes — are rejected.
    """
    data = reference
    if data.startswith("data:"):
        comma = data.find(",")
        if comma == -1:
            raise ValueError("reference data: URI is malformed (missing comma).")
        data = data[comma + 1 :]
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        # binascii.Error subclasses ValueError.
        raise ValueError("reference is not valid base64 data.") from exc


def _resolve_reference_string(reference: str) -> bytes:
    """Resolve one reference string to raw bytes, dispatching on URL scheme.

    Mirrors the LLM multimodal loader so serve references accept the same forms:
    ``http(s)`` fetches through the SSRF-guarded loader (private-address block,
    redirect re-validation, timeout, size cap); ``file://`` and bare local paths
    read from disk; ``data:`` and base64 strings decode inline. A bare string is
    decoded as base64 first and, failing that, read as a local file path.
    Fetch/read failures become ``ValueError`` so a bad URL or path is a client
    400, not a server 500.
    """
    scheme = urlparse(reference).scheme
    if scheme in ("http", "https"):
        try:
            return _safe_request_get(reference).content
        except Exception as exc:
            raise ValueError(f"reference URL could not be fetched: {exc}") from exc
    if scheme == "file":
        try:
            return Path(_normalize_file_uri(reference)).read_bytes()
        except OSError as exc:
            raise ValueError(f"reference file could not be read: {exc}") from exc
    if scheme == "data":
        return _read_reference_payload(reference)
    # Bare string: base64 first (the established default), else a local file path
    # so a plain path works without the file:// scheme.
    try:
        return _read_reference_payload(reference)
    except ValueError:
        try:
            return Path(reference).read_bytes()
        except OSError as exc:
            raise ValueError(
                f"reference is not valid base64 data, and not a readable local file: {exc}"
            ) from exc


def _reference_payload_and_role(ref) -> tuple[bytes, Optional[str]]:
    """Extract ``(payload_bytes, role)`` from one raw HTTP reference.

    ``ref`` is a string (base64/``data:`` URI, ``http(s)`` URL, or a local file
    path), a multipart ``UploadFile`` (has ``.file``), or a ``MediaReferenceItem``
    exposing ``content`` and an optional ``role``.
    """
    role = getattr(ref, "role", None)
    if isinstance(ref, str):
        return _resolve_reference_string(ref), role
    if hasattr(ref, "file"):  # multipart UploadFile
        return ref.file.read(), role
    data = getattr(ref, "content", None)
    if not isinstance(data, str):
        raise ValueError("reference item must carry a 'content' string.")
    return _resolve_reference_string(data), role


def _materialize_reference(
    payload: bytes, *, modality: str, ref_id: str, media_storage_path: Optional[str]
) -> str:
    """Content-validate a reference payload and persist it, returning its path.

    HEIF/AVIF images are rejected on signature alone (Pillow support depends
    on optional plugins the worker need not share). Video acceptance beyond the
    container signature happens in the worker's NVDEC demux.
    """
    if modality == "image":
        if sniff_media_kind(payload) != "image":
            raise ValueError(
                "image_reference is not a recognized image; supported inputs are PNG/JPEG."
            )
        if is_isobmff_image_bytes(payload):
            raise ValueError(
                "image_reference is a HEIF/AVIF image, which is not a supported "
                "reference format; convert it to PNG or JPEG."
            )
    elif modality == "video":
        if sniff_media_kind(payload) != "video":
            raise ValueError(
                "video_reference is not a recognized media container; supported "
                "inputs are MP4/AVI video."
            )
    # audio: no signature sniffing (sniff_media_kind detects only image/video);
    # the consuming pipeline validates the audio codec in its worker.
    if media_storage_path is None:
        raise ValueError(f"media_storage_path is required to store the {modality}_reference.")
    ref_path = os.path.join(media_storage_path, ref_id)
    with open(ref_path, "wb") as f:
        f.write(payload)
    return ref_path


def cleanup_reference_files(media_storage_path: Optional[str], request_id: str) -> None:
    """Remove the materialized reference inputs for one request.

    References are materialized as ``{request_id}_{modality}_ref_{i}`` (and the
    deprecated ``{request_id}_input_ref``) under ``media_storage_path``. They are
    input-only — unneeded once the pipeline has consumed them — so the request
    owner removes them by the ``request_id`` prefix, covering image/video/audio
    and the deprecated single reference regardless of count. Output files
    (``{request_id}_{i}.<ext>``) carry no ``ref`` and are left untouched.
    Best-effort: already-removed files are ignored.
    """
    if media_storage_path is None:
        return
    for path in Path(media_storage_path).glob(f"{request_id}_*ref*"):
        try:
            path.unlink()
        except OSError:
            pass


def _build_reference_list(value, *, modality: str, id: str, media_storage_path: Optional[str]):
    """Materialize an HTTP reference field into a list of ``MediaRef`` objects.

    ``value`` is None, a base64/data-URI string, a multipart ``UploadFile``, a
    ``MediaReferenceItem``, or a list of any of those. Each entry is decoded,
    content-validated for ``modality``, persisted to a per-index path, and
    wrapped as ``MediaRef`` (carrying ``role`` when present).
    """
    if value is None:
        return None
    # Local import: the visual_gen tree is already loaded in a VisualGen serving
    # process, and this keeps it out of every plain-LLM process (see TYPE_CHECKING).
    from tensorrt_llm.visual_gen.params import MediaRef

    raw_items = value if isinstance(value, list) else [value]
    refs = []
    created_paths: list[str] = []
    try:
        for i, item in enumerate(raw_items):
            payload, role = _reference_payload_and_role(item)
            ref_path = _materialize_reference(
                payload,
                modality=modality,
                ref_id=f"{id}_{modality}_ref_{i}",
                media_storage_path=media_storage_path,
            )
            created_paths.append(ref_path)
            refs.append(MediaRef(content=ref_path, role=role))
    except Exception:
        # A later item failed; remove the files earlier items already wrote so
        # a rejected multi-reference request leaves nothing on disk.
        for path in created_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        raise
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


def _write_bytes_with_limit(value: bytes, path: str) -> int:
    size = len(value)
    if size > IMAGE_EDIT_MAX_IMAGE_BYTES:
        raise ValueError(
            "Image edit input exceeds the per-image byte limit "
            f"({size} > {IMAGE_EDIT_MAX_IMAGE_BYTES})."
        )
    _validate_png_jpeg_image(value)
    with open(path, "wb") as f:
        f.write(value)
    return size


def _validate_png_jpeg_image(value: bytes) -> None:
    try:
        with Image.open(BytesIO(value)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(_INVALID_IMAGE_EDIT_INPUT_MESSAGE) from exc
    if image_format not in _IMAGE_EDIT_INPUT_FORMATS:
        raise ValueError(_INVALID_IMAGE_EDIT_INPUT_MESSAGE)


def _copy_upload_with_limit(value: Any, path: str) -> int:
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
    return _write_bytes_with_limit(b"".join(chunks), path)


def _materialize_conditioning_input(
    value: Any,
    path: str,
) -> tuple[str, int]:
    """Return a server-owned file path for upload or base64 inputs."""
    try:
        if isinstance(value, str):
            decoded = _decode_base64_media(value)
            if decoded is None:
                parsed = urlparse(value)
                if parsed.scheme in ("file", "http", "https"):
                    raise ValueError(
                        "Image edit inputs must be uploaded files or base64-encoded images; "
                        "local paths and URLs are not supported."
                    )
                raise ValueError("String image edit inputs must be base64-encoded image data.")
            return path, _write_bytes_with_limit(decoded, path)

        if isinstance(value, bytes):
            return path, _write_bytes_with_limit(value, path)

        if hasattr(value, "file"):
            return path, _copy_upload_with_limit(value, path)
    except Exception:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        raise

    raise ValueError(f"Unsupported conditioning input type: {type(value)}")


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


def _materialize_conditioning_inputs(
    value: Any,
    *,
    id: str,
    field_name: str,
    media_storage_path: str,
) -> str | List[str]:
    values = value if isinstance(value, list) else [value]
    paths = []
    total_bytes = 0
    try:
        for i, item in enumerate(values):
            path, size = _materialize_conditioning_input(
                item,
                os.path.join(media_storage_path, f"{id}_{field_name}_{i}.png"),
            )
            paths.append(path)
            total_bytes += size
            if total_bytes > IMAGE_EDIT_MAX_TOTAL_IMAGE_BYTES:
                raise ValueError(
                    "Image edit inputs exceed the total byte limit "
                    f"({total_bytes} > {IMAGE_EDIT_MAX_TOTAL_IMAGE_BYTES})."
                )
    except Exception:
        cleanup_materialized_conditioning_inputs(paths)
        raise
    return paths if isinstance(value, list) else paths[0]


def cleanup_materialized_conditioning_inputs(value: Any) -> None:
    paths = value if isinstance(value, list) else [value]
    for path in paths:
        if not isinstance(path, str):
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to remove temporary image edit input %r: %s", path, exc)

def _apply_deprecated_input_reference(
    input_reference: str | UploadFile | None,
    params: VisualGenParams,
    *,
    id: str,
    media_storage_path: str | None,
) -> None:
    """Back-compat for the deprecated single ``input_reference``.

    Sniff-routes the payload to ``image_reference`` (image) or ``video_reference``
    (video), preserving the pre-typed-fields behavior. Ignored when a typed
    image/video reference is already set — the typed fields take precedence.
    """
    if input_reference is None:
        return
    logger.warning("'input_reference' is deprecated; use 'image_reference' / 'video_reference'.")
    if params.image_reference or params.video_reference:
        return
    from tensorrt_llm.visual_gen.params import MediaRef

    payload, _ = _reference_payload_and_role(input_reference)
    kind = sniff_media_kind(payload)
    if kind == "image":
        path = _materialize_reference(
            payload,
            modality="image",
            ref_id=f"{id}_input_ref",
            media_storage_path=media_storage_path,
        )
        params.image_reference = [MediaRef(content=path)]
    elif kind == "video":
        path = _materialize_reference(
            payload,
            modality="video",
            ref_id=f"{id}_input_ref",
            media_storage_path=media_storage_path,
        )
        params.video_reference = [MediaRef(content=path)]
    else:
        raise ValueError(
            "input_reference is not a recognized media container; supported "
            "inputs are PNG/JPEG images and MP4/AVI video."
        )


def parse_visual_gen_params(
    request: ImageGenerationRequest | ImageEditRequest | VideoGenerationRequest,
    id: str,
    generator: VisualGen,
    media_storage_path: Optional[str] = None,
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
        if media_storage_path is None:
            raise ValueError("media_storage_path is required when image edit inputs are provided")
        _validate_image_edit_request_limits(request, generator)
        params.image = _materialize_conditioning_inputs(
            request.image,
            id=id,
            field_name="image",
            media_storage_path=media_storage_path,
        )

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
        # Reference inputs: materialize each transport (base64/data-URI/upload)
        # to a stored file and hand the pipeline a ``MediaRef`` carrying the
        # local path. Decode stays model-specific in the worker.
        image_refs = _build_reference_list(
            request.image_reference, modality="image", id=id, media_storage_path=media_storage_path
        )
        if image_refs:
            params.image_reference = image_refs
        video_refs = _build_reference_list(
            request.video_reference, modality="video", id=id, media_storage_path=media_storage_path
        )
        if video_refs:
            params.video_reference = video_refs
        audio_refs = _build_reference_list(
            request.audio_reference, modality="audio", id=id, media_storage_path=media_storage_path
        )
        if audio_refs:
            params.audio_reference = audio_refs
        _apply_deprecated_input_reference(
            request.input_reference, params, id=id, media_storage_path=media_storage_path
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
