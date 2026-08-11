from __future__ import annotations

import asyncio
import base64
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tensorrt_llm.inputs.media_io import is_isobmff_image_bytes, sniff_media_kind
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest, VideoGenerationRequest

if TYPE_CHECKING:
    # Type-only: importing tensorrt_llm.visual_gen at runtime would pull the
    # whole visual_gen tree into every LLM serving process.
    from tensorrt_llm.visual_gen import VisualGen, VisualGenParams

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


def _reference_payload_and_role(ref, data_field: str) -> tuple[bytes, Optional[str]]:
    """Extract ``(payload_bytes, role)`` from one raw HTTP reference.

    ``ref`` is a base64/data-URI string, a multipart ``UploadFile`` (has
    ``.file``), or a reference item exposing ``data_field`` ('image'/'video')
    and, for images, an optional ``role``.
    """
    role = getattr(ref, "role", None)
    if isinstance(ref, str):
        return _read_reference_payload(ref), role
    if hasattr(ref, "file"):  # multipart UploadFile
        return ref.file.read(), role
    data = getattr(ref, data_field, None)
    if not isinstance(data, str):
        raise ValueError(f"{data_field}_reference item must carry a base64 '{data_field}' string.")
    return _read_reference_payload(data), role


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


def _build_reference_list(
    value, *, modality: str, data_field: str, ref_cls, id: str, media_storage_path: Optional[str]
):
    """Materialize an HTTP reference field into a list of typed ``*Ref`` objects.

    ``value`` is None, a base64/data-URI string, a multipart ``UploadFile``, a
    reference item, or a list of any of those. Each entry is decoded,
    content-validated for ``modality``, persisted to a per-index path, and
    wrapped as ``ref_cls`` (carrying ``role`` for images).
    """
    if value is None:
        return None
    raw_items = value if isinstance(value, list) else [value]
    refs = []
    for i, item in enumerate(raw_items):
        payload, role = _reference_payload_and_role(item, data_field)
        ref_path = _materialize_reference(
            payload,
            modality=modality,
            ref_id=f"{id}_{data_field}_ref_{i}",
            media_storage_path=media_storage_path,
        )
        kwargs = {data_field: ref_path}
        if role is not None:
            kwargs["role"] = role
        refs.append(ref_cls(**kwargs))
    return refs


def parse_visual_gen_params(
    request: ImageGenerationRequest | VideoGenerationRequest,
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
        # to a stored file and hand the pipeline a typed ``*Ref`` carrying the
        # local path. Decode stays model-specific in the worker. Local import:
        # the visual_gen tree is already loaded in a VisualGen serving process,
        # and this keeps it out of every plain-LLM process (see TYPE_CHECKING).
        from tensorrt_llm.visual_gen.params import AudioRef, ImageRef, VideoRef

        image_refs = _build_reference_list(
            request.image_reference,
            modality="image",
            data_field="image",
            ref_cls=ImageRef,
            id=id,
            media_storage_path=media_storage_path,
        )
        if image_refs:
            params.image_reference = image_refs
        video_refs = _build_reference_list(
            request.video_reference,
            modality="video",
            data_field="video",
            ref_cls=VideoRef,
            id=id,
            media_storage_path=media_storage_path,
        )
        if video_refs:
            params.video_reference = video_refs
        audio_refs = _build_reference_list(
            request.audio_reference,
            modality="audio",
            data_field="audio",
            ref_cls=AudioRef,
            id=id,
            media_storage_path=media_storage_path,
        )
        if audio_refs:
            params.audio_reference = audio_refs

    _warn_if_set_with_no_semantic(request, getattr(generator, "model", None))
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
