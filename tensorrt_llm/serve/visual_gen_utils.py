import asyncio
import base64
import os
import shutil
from typing import Any, Dict, List, Optional

from tensorrt_llm.llmapi.visual_gen import VisualGenParams
from tensorrt_llm.serve.openai_protocol import (
    ImageEditRequest,
    ImageGenerationRequest,
    VideoGenerationRequest,
)


def parse_visual_gen_params(
    request: ImageGenerationRequest | VideoGenerationRequest | ImageEditRequest,
    id: str,
    media_storage_path: Optional[str] = None,
) -> VisualGenParams:
    params = VisualGenParams()
    params.prompt = request.prompt
    if request.negative_prompt is not None:
        params.negative_prompt = request.negative_prompt
    if request.size is not None and request.size != "auto":
        params.width, params.height = map(int, request.size.split("x"))
    if request.guidance_scale is not None:
        params.guidance_scale = request.guidance_scale
    if request.guidance_rescale is not None:
        params.guidance_rescale = request.guidance_rescale

    if isinstance(request, ImageGenerationRequest) or isinstance(request, ImageEditRequest):
        if request.num_inference_steps is not None:
            params.num_inference_steps = request.num_inference_steps
        elif isinstance(request, ImageGenerationRequest) and request.quality == "hd":
            params.num_inference_steps = 30
        if request.n is not None:
            params.num_images_per_prompt = request.n
        if isinstance(request, ImageEditRequest):
            if request.image is not None:
                if isinstance(request.image, list):
                    params.image = [base64.b64decode(image) for image in request.image]
                else:
                    params.image = [base64.b64decode(request.image)]
            if request.mask is not None:
                if isinstance(request.mask, list):
                    params.mask = [base64.b64decode(mask) for mask in request.mask]
                else:
                    params.mask = base64.b64decode(request.mask)

    elif isinstance(request, VideoGenerationRequest):
        if request.num_inference_steps is not None:
            params.num_inference_steps = request.num_inference_steps
        if request.input_reference is not None:
            params.input_reference = os.path.join(media_storage_path, f"{id}_reference.png")
            with open(params.input_reference, "wb") as f:
                shutil.copyfileobj(request.input_reference.file, f)

        params.frame_rate = request.fps
        params.num_frames = int(request.seconds * request.fps)

        if request.seed is not None:
            params.seed = int(request.seed)

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
