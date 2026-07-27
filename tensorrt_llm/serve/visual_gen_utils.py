import asyncio
import base64
import os
import shutil
from typing import Any, Dict, List, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest, VideoGenerationRequest
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
        if request.input_reference is not None:
            if media_storage_path is None:
                raise ValueError("media_storage_path is required when input_reference is provided")
            ref_path = os.path.join(media_storage_path, f"{id}_reference.png")
            if isinstance(request.input_reference, str):
                with open(ref_path, "wb") as f:
                    f.write(base64.b64decode(request.input_reference))
            else:
                with open(ref_path, "wb") as f:
                    shutil.copyfileobj(request.input_reference.file, f)
            params.image = ref_path

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
