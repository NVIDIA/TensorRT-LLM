import asyncio
import base64
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypedDict, Union
from urllib.parse import unquote, urlparse

import aiohttp
import numpy as np
import requests
import soundfile
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoProcessor, ProcessorMixin
from transformers.utils import logging

from tensorrt_llm.inputs.content_format import (ContentFormat,
                                                detect_content_format)
from tensorrt_llm.inputs.multimodal import (MultimodalServerConfig,
                                            default_hasher)
from tensorrt_llm.inputs.registry import (MULTIMODAL_PLACEHOLDER_REGISTRY,
                                          MultimodalPlaceholderPlacement)
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.tokenizer import TokenizerBase, TransformersTokenizer
from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer

logger = logging.get_logger(__name__)


@dataclass
class BaseModalityData:
    """Base class for modality-specific data.

    This class serves as the foundation for all modality data types (image, video, audio, etc.),
    providing a common interface for modality-specific data structures.

    Subclasses should define their own attributes based on the specific needs of each modality.
    """


@dataclass
class VideoData(BaseModalityData):
    """Data class for video loading results.

    Attributes:
        frames: List of video frames, either as PIL Images or PyTorch tensors.
        metadata: Dictionary containing video metadata including:
            - total_num_frames: Total number of frames in the video
            - fps: Original frames per second of the video
            - duration: Duration of the video in seconds
            - frames_indices: List of indices of the sampled frames
    """
    frames: Union[List[Image.Image], List[torch.Tensor]]
    """The loaded video frames, either as PIL Images or PyTorch tensors."""

    metadata: Dict[str, Any]
    """Metadata associated with the video (e.g., fps, duration, frame indices)."""

    def __post_init__(self):
        """Validate that frames list is not empty."""
        if not self.frames:
            raise ValueError("frames list cannot be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")


def rgba_to_rgb(
    image: Image.Image,
    background_color: Union[tuple[int, int, int], list[int]] = (255, 255, 255)
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color.

    Uses white (255, 255, 255) as the default background color because:
    1. It's the most neutral and commonly expected background for images
    2. Maintains backward compatibility with existing code
    """
    if image.mode != "RGBA":
        raise ValueError(
            f"Expected image mode to be 'RGBA', but got '{image.mode}'")
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def convert_image_mode(image: Image.Image, to_mode: str) -> Image.Image:
    """Convert image to specified mode with proper handling of RGBA to RGB conversion."""
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return rgba_to_rgb(image)
    else:
        return image.convert(to_mode)


def _load_and_convert_image(image):
    image = Image.open(image)
    image.load()
    return convert_image_mode(image, "RGB")


def load_base64_image(parsed_url: str) -> Image.Image:
    data_spec, data = parsed_url.path.split(",", 1)
    media_type, data_type = data_spec.split(";", 1)

    if data_type != "base64":
        msg = "Only base64 data URLs are supported for now."
        raise NotImplementedError(msg)

    content = base64.b64decode(data)
    image = _load_and_convert_image(BytesIO(content))
    return image


def load_base64_image_embeds(str_content: str) -> torch.Tensor:
    content_bytes = base64.b64decode(str_content)
    with BytesIO(content_bytes) as buf:
        image_data: torch.Tensor = torch.load(buf,
                                              weights_only=True,
                                              map_location="cpu")
    return image_data


def load_image(image: Union[str, Image.Image],
               format: str = "pt",
               device: str = "cpu") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    if isinstance(image, Image.Image):
        return image.convert('RGB')

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        image = requests.get(image, stream=True, timeout=10).raw
        image = _load_and_convert_image(image)
    elif parsed_url.scheme == "data":
        image = load_base64_image(parsed_url)
    else:
        image = _load_and_convert_image(image)

    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


async def async_load_image(
        image: Union[str, Image.Image],
        format: str = "pt",
        device: str = "cpu") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    if isinstance(image, Image.Image):
        return image.convert('RGB')

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            async with session.get(image) as response:
                content = await response.read()
                image = _load_and_convert_image(BytesIO(content))
    elif parsed_url.scheme == "data":
        image = load_base64_image(parsed_url)
    else:
        image = _load_and_convert_image(Path(parsed_url.path))

    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


def _load_video_by_cv2(video: str,
                       num_frames: int = 10,
                       fps: int = 30,
                       format: str = "pt",
                       device: str = "cpu") -> VideoData:
    # Keep this import local to avoid importing cv2 if not needed
    import cv2

    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video)

    if not vidcap.isOpened():
        raise ValueError(
            f"Video '{video}' could not be opened. Make sure opencv is installed with video support."
        )

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)

    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video}' has no frames.")

    duration = frame_count / original_fps if original_fps > 0 else 0
    num_frames_to_sample = frame_count
    if num_frames > 0:
        num_frames_to_sample = min(num_frames, frame_count)
    if fps > 0:
        num_frames_to_sample = min(num_frames_to_sample,
                                   math.floor(duration * fps))
    num_frames_to_sample = max(1, num_frames_to_sample)  # at least one sample

    if num_frames_to_sample == frame_count:
        indices = list(range(0, num_frames_to_sample))
    else:
        uniform_sampled_frames = np.linspace(0,
                                             frame_count - 1,
                                             num_frames_to_sample,
                                             dtype=int)
        indices = uniform_sampled_frames.tolist()

    frames = {}
    for index in indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = Image.fromarray(frame)

    assert len(
        frames
    ) == num_frames_to_sample, f"Expected {num_frames_to_sample} frames, got {len(frames)}"

    loaded_frames = [
        ToTensor()(frames[index]).to(
            device=device) if format == "pt" else frames[index]
        for index in indices if index in frames
    ]

    metadata = {
        "total_num_frames": frame_count,
        "fps": original_fps,
        "duration": duration,
        "frames_indices": list(indices),
    }

    return VideoData(frames=loaded_frames, metadata=metadata)


def load_base64_video(video: str) -> BytesIO:
    parsed_url = urlparse(video)
    data_spec, data = parsed_url.path.split(",", 1)
    media_type, data_type = data_spec.split(";", 1)

    if data_type != "base64":
        msg = "Only base64 data URLs are supported for now."
        raise NotImplementedError(msg)

    content = base64.b64decode(data)
    return content


def load_video(video: str,
               num_frames: int = 10,
               fps: int = 30,
               format: str = "pt",
               device: str = "cpu") -> VideoData:
    parsed_url = urlparse(video)
    results = None
    if parsed_url.scheme in ["http", "https", ""]:
        results = _load_video_by_cv2(video, num_frames, fps, format, device)
    elif parsed_url.scheme == "data":
        decoded_video = load_base64_video(video)
        # TODO: any ways to read videos from memory, instead of writing to a tempfile?
        with tempfile.NamedTemporaryFile(delete=True,
                                         suffix='.mp4') as tmp_file:
            tmp_file.write(decoded_video)
            tmp_file.flush()
            results = _load_video_by_cv2(tmp_file.name, num_frames, fps, format,
                                         device)
    else:
        raise ValueError(f"Unsupported video scheme: {parsed_url.scheme}")

    return results


async def async_load_video(video: str,
                           num_frames: int = 10,
                           fps: int = 30,
                           format: str = "pt",
                           device: str = "cpu") -> VideoData:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    parsed_url = urlparse(video)

    if parsed_url.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            async with session.get(video) as response:
                with tempfile.NamedTemporaryFile(delete=True,
                                                 suffix='.mp4') as tmp:
                    tmp.write(await response.content.read())
                    tmp.flush()
                    results = _load_video_by_cv2(tmp.name, num_frames, fps,
                                                 format, device)
    elif parsed_url.scheme == "data":
        decoded_video = load_base64_video(video)
        # TODO: any ways to read videos from memory, instead of writing to a tempfile?
        with tempfile.NamedTemporaryFile(delete=True,
                                         suffix='.mp4') as tmp_file:
            tmp_file.write(decoded_video)
            tmp_file.flush()
            results = _load_video_by_cv2(tmp_file.name, num_frames, fps, format,
                                         device)
    else:
        results = _load_video_by_cv2(video, num_frames, fps, format, device)
    return results


def _normalize_file_uri(uri: str) -> str:
    """Strip the file:// scheme and unquote percent-encoded characters."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return unquote(parsed.path)
    return uri


def load_audio(
    audio: str,
    format: str = "pt",
    device: str = "cuda",
) -> Tuple[np.ndarray, int]:
    parsed_url = urlparse(audio)
    if parsed_url.scheme in ["http", "https"]:
        audio = requests.get(audio, stream=True, timeout=10)
        audio = BytesIO(audio.content)
    elif parsed_url.scheme == "file":
        audio = _normalize_file_uri(audio)

    audio = soundfile.read(audio)
    return audio


async def async_load_audio(
    audio: str,
    format: str = "pt",
    device: str = "cuda",
) -> Tuple[np.ndarray, int]:
    parsed_url = urlparse(audio)
    if parsed_url.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            async with session.get(audio) as response:
                audio = BytesIO(await response.content.read())
    elif parsed_url.scheme == "file":
        audio = _normalize_file_uri(audio)

    audio = soundfile.read(audio)
    return audio


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url, timeout=10) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


def encode_base64_image(
    media: Image.Image,
    *,
    image_format: str = "JPEG",
) -> str:
    image = media

    with BytesIO() as buffer:
        image = convert_image_mode(image, "RGB")
        image.save(buffer, image_format)
        data = buffer.getvalue()

    return base64.b64encode(data).decode("utf-8")


# Helpers to always get the latest supported multimodal model types from the registry
def ALL_SUPPORTED_MULTIMODAL_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types()


def ALL_SUPPORTED_IMAGE_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_image_model_types()


def ALL_SUPPORTED_VIDEO_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_video_model_types()


def ALL_SUPPORTED_AUDIO_MODELS():
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_audio_model_types()


def retrieve_multimodal_placeholder(model_type: str, modality: str,
                                    current_count: int) -> Optional[str]:
    """
        Get the appropriate placeholder for a given modality and model type.

        Args:
            model_type: The type of the multimodal model.
            modality: The modality of the data.
            current_count: The number of multimodal data already added.

    """
    if MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(model_type, modality):
        """
        The placeholder is a string with a single placeholder for the current count.
            - For example, if the placeholder is "<|image_{0}|>", and the current count is 1,
              the placeholder will be "<|image_1|>".
            - However, if the placeholder is "<|image|>", the current count would be ignored.
              In this case, the placeholder would be "<|image|>".
        """
        return MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(
            model_type, modality).format(current_count)
    raise TypeError(f"Unknown modality: {modality}")


class MultimodalData(TypedDict):
    """Type definition for multimodal data structure."""
    modality: str
    data: Any
    is_embedding: bool


class ConversationMessage(TypedDict, total=False):
    """Type definition for conversation message structure.

    Attributes:
        role: The message role (e.g. "user", "assistant", "system").
        content: Flattened text content (all text parts joined by newlines).
        media: List of multimodal data items attached to this message.
        content_parts: Ordered list preserving the interleaved positions of text and media as the
            user originally sent them. Only present when the message contains media.

            Each element is either:
            - A `str` for a text segment, or
            - A `dict` of the form `{"type": "<modality>", "media_index": <int>}`
              marking where a media item (image/video/audio) appeared.
              `media_index` is a 0-based index into `media`.

            This is used by `interleave_mm_placeholders` to insert multimodal placeholders at the
            correct positions, and to reconstruct the OpenAI-style content list for templates that
            handle media natively.
    """
    role: str
    content: str
    media: List[MultimodalData]
    content_parts: List[Union[str, dict]]


class MultimodalDataTracker:
    """Tracks and manages multimodal data for both sync and async processing."""

    def __init__(
            self,
            model_type: str,
            multimodal_server_config: Optional[MultimodalServerConfig] = None):
        self._model_type = model_type
        self._data = defaultdict[str, list](list)
        self._embeddings = defaultdict[str, list](list)
        self._placeholder_counts = defaultdict[str, int](int)
        self._placeholder_to_modality: dict[str, str] = {}
        self._multimodal_server_config = multimodal_server_config if multimodal_server_config is not None else MultimodalServerConfig(
        )

    async def retrieve_all_async(
        self
    ) -> tuple[Optional[Dict[str, List[Any]]], Optional[Dict[str, List[Any]]]]:
        """Retrieve all collected multimodal data and embeddings."""

        async def _retrieve(
                data: Optional[dict[str,
                                    list]]) -> Optional[Dict[str, List[Any]]]:
            if not data:
                return None
            return {
                modality: await asyncio.gather(*items)
                for modality, items in data.items() if items
            }

        return await _retrieve(self._data), await _retrieve(self._embeddings)

    def retrieve_all_sync(
        self
    ) -> tuple[Optional[Dict[str, List[Any]]], Optional[Dict[str, List[Any]]]]:
        """Retrieve all collected multimodal data and embeddings."""

        def _retrieve(
                data: Optional[dict[str,
                                    list]]) -> Optional[Dict[str, List[Any]]]:
            if not data:
                return None
            return {
                modality: items
                for modality, items in data.items() if items
            }

        return _retrieve(self._data), _retrieve(self._embeddings)

    def add_data(self,
                 media_type: str,
                 data: Union[Coroutine, Any],
                 *,
                 is_embedding: bool = False) -> Optional[str]:
        current_count = len(self._data[media_type]) + len(
            self._embeddings[media_type]) + 1
        placeholder = retrieve_multimodal_placeholder(self._model_type,
                                                      media_type, current_count)
        (self._embeddings
         if is_embedding else self._data)[media_type].append(data)
        if placeholder:
            self._placeholder_counts[placeholder] += 1
            self._placeholder_to_modality[placeholder] = media_type
        return placeholder

    def placeholder_counts(self) -> Dict[str, int]:
        """Get the count of multimodal placeholders."""
        return dict(self._placeholder_counts)

    def placeholder_modalities(self) -> Dict[str, str]:
        """Get the mapping from placeholder string to modality name."""
        return dict(self._placeholder_to_modality)


def add_multimodal_placeholders(model_type: str, text_prompt: str,
                                mm_placeholder_counts: dict[str, int]) -> str:
    """Add multimodal placeholders to the text prompt."""
    placeholders = []
    for placeholder in mm_placeholder_counts:
        placeholders.extend([placeholder] * mm_placeholder_counts[placeholder])
    parts = []
    match MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_placement(model_type):
        case MultimodalPlaceholderPlacement.BEFORE_TEXT:
            parts.extend(placeholders)
            parts.append(text_prompt)
        case MultimodalPlaceholderPlacement.AFTER_TEXT:
            parts.append(text_prompt)
            parts.extend(placeholders)
    return MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholders_separator(
        model_type).join(parts)


def interleave_mm_placeholders(
    model_type: str,
    content_parts: list[Union[str, dict]],
    mm_placeholder_counts: dict[str, int],
    placeholder_modalities: Dict[str, str],
) -> str:
    """Build a prompt string with placeholders interleaved at media positions.

    When `content_parts` preserves the original ordering of text and media
    items from the user's request, this function inserts the correct
    placeholder at each media position instead of bulk-prepending/appending.

    Args:
        model_type: The model type string (used to look up placeholder info).
        content_parts: Ordered list of text strings and media position dicts.
        mm_placeholder_counts: Mapping of placeholder -> expected count.
        placeholder_modalities: Mapping of placeholder string to modality
            name (e.g. `{"<image>": "image"}`).

    Returns:
        A single string with placeholders inserted at the correct positions.
    """
    if not content_parts:
        return add_multimodal_placeholders(model_type, "",
                                           mm_placeholder_counts)

    # Build a per-modality queue of placeholder strings (expanded by count).
    # This handles both shared placeholders (e.g. "<image>" with count=3)
    # and unique per-item placeholders (e.g. "<|image_1|>", "<|image_2|>").
    modality_placeholders: dict[str, list[str]] = {}
    for placeholder, count in mm_placeholder_counts.items():
        if placeholder not in placeholder_modalities:
            raise KeyError(
                f"Placeholder '{placeholder}' not found in "
                f"placeholder_modalities mapping. Known placeholders: "
                f"{list(placeholder_modalities.keys())}")
        modality = placeholder_modalities[placeholder]
        modality_placeholders.setdefault(modality,
                                         []).extend([placeholder] * count)

    parts: list[str] = []
    separator = MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholders_separator(
        model_type)
    # Track how many placeholders have been consumed per modality
    modality_cursor: dict[str, int] = {}

    for part in content_parts:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            media_type = part.get("type", "image")
            queue = modality_placeholders.get(media_type)
            if not queue:
                continue
            cursor = modality_cursor.get(media_type, 0)
            if cursor < len(queue):
                parts.append(queue[cursor])
                modality_cursor[media_type] = cursor + 1

    return separator.join(parts)


def resolve_hf_chat_template(
    tokenizer: TokenizerBase,
    processor: ProcessorMixin,
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
) -> Optional[str]:
    """Resolve the appropriate chat template to use."""

    # 1. If chat_template is not None, return it
    if chat_template is not None:
        return chat_template

    # 2. If tool is not provided, use the processor's default chat template
    if not tools and processor and hasattr(processor, "tokenizer"):
        if hasattr(processor.tokenizer, "get_chat_template"):
            return processor.tokenizer.get_chat_template(chat_template,
                                                         tools=tools)

    # 3. If tool is provided, use the tool
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.warning(
            "Failed to load AutoTokenizer chat template for %s",
            getattr(tokenizer, "name_or_path",
                    type(tokenizer).__name__))
    return None


def _resolve_content_format(model_type: str,
                            chat_template: Optional[str]) -> ContentFormat:
    """Determine the content format for the given model and template.

    Resolution order:
    1. Registry override (explicit per-model annotation).
    2. Jinja AST auto-detection (if a template string is available).
    3. Default to STRING.
    """
    # 1. Check registry for an explicit override
    registry_format = MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
        model_type)
    if registry_format is not None:
        return registry_format

    # 2. Auto-detect from template AST
    if chat_template is not None:
        return detect_content_format(chat_template)

    # 3. Default
    return ContentFormat.STRING


def _build_openai_content(
        conv: ConversationMessage,
        mm_placeholder_count: dict[str, int]) -> list[dict[str, Any]]:
    """Reconstruct OpenAI-style content list from a ConversationMessage.

    Uses `content_parts` (preserving media position) when available, otherwise falls back to placing
    text first then media items.
    """
    content_list: list[dict[str, Any]] = []
    content_parts = conv.get("content_parts")

    if content_parts:
        for part in content_parts:
            if isinstance(part, str):
                content_list.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                media_type = part.get("type", "image")
                content_list.append({"type": media_type})
    else:
        # Fallback: text first, then media placeholders
        text = conv.get("content", "")
        if text:
            content_list.append({"type": "text", "text": text})
        for placeholder, count in mm_placeholder_count.items():
            # Infer modality from placeholder (e.g. "<image>" -> "image")
            modality = "image"
            if "video" in placeholder.lower():
                modality = "video"
            elif "audio" in placeholder.lower(
            ) or "so_embedding" in placeholder.lower():
                modality = "audio"
            for _ in range(count):
                content_list.append({"type": modality})

    return content_list


def apply_chat_template(
    *,
    model_type: str,
    tokenizer: Union[TransformersTokenizer, TokenizerBase],
    processor: ProcessorMixin,
    conversation: list[ConversationMessage],
    add_generation_prompt: bool,
    mm_placeholder_counts: list[dict[str, int]],
    tools: Optional[list[dict[str, Any]]] = None,
    documents: Optional[list[dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    enable_tokenize: bool = False,
) -> (str | List[str]):
    """Apply chat template to the conversation.

    Uses content-format-driven dispatch:
    - PASSTHROUGH: skip template rendering, just concatenate content strings
    - OPENAI: reconstructs content as list of dicts for the template to handle
    - STRING: keeps flattened text with pre-inserted placeholders
    """

    # Handle DeepSeek V32 tokenizer with custom chat template
    if isinstance(tokenizer, DeepseekV32Tokenizer):
        prompt = tokenizer.apply_chat_template(
            messages=conversation,
            tools=tools,
            **(chat_template_kwargs or {}),
        )
        if enable_tokenize:
            return tokenizer.encode(prompt)
        return prompt

    # Check for PASSTHROUGH early — before we need tokenizer/processor/template.
    # The registry may already know this model skips chat templates entirely.
    registry_format = MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
        model_type)
    if registry_format == ContentFormat.PASSTHROUGH:
        return "".join([conv["content"] for conv in conversation])

    if isinstance(tokenizer, TransformersTokenizer):
        tokenizer = tokenizer.tokenizer  # we need the TokenizerBase for apply_chat_template

    hf_chat_template = resolve_hf_chat_template(tokenizer, processor,
                                                chat_template, tools)
    if hf_chat_template is None:
        raise ValueError(
            "No chat template found for the given tokenizer and tools.")

    # Determine content format and prepare conversation accordingly
    content_format = _resolve_content_format(model_type, hf_chat_template)

    if content_format == ContentFormat.OPENAI:
        # Path OPENAI: reconstruct content as list of dicts for the template
        for conv, mm_placeholder_count in zip(conversation,
                                              mm_placeholder_counts):
            if mm_placeholder_count:
                conv["content"] = _build_openai_content(conv,
                                                        mm_placeholder_count)
    # STRING path: placeholders already inserted in content by caller

    result = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=enable_tokenize,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        documents=documents,
        chat_template=hf_chat_template,
        **(chat_template_kwargs or {}),
    )

    return result


def default_multimodal_input_loader(
        *,
        tokenizer: Optional[Union[TransformersTokenizer, TokenizerBase]],
        model_dir: str,
        model_type: str,
        modality: str,
        prompts: List[str],
        media: Optional[Union[List[str], List[List[str]]]] = None,
        image_data_format: str = "pt",
        num_frames: int = 8,
        mm_embeddings: Optional[Union[List[torch.Tensor],
                                      List[List[torch.Tensor]]]] = None,
        device: str = "cpu") -> List[dict[str, Union[str, torch.Tensor]]]:

    def convert_to_conversation_message(
        prompt: str,
        media: Union[Any, List[Any]],
        modality: str,
        is_embedding: bool = False,
    ) -> ConversationMessage:
        if isinstance(media, str):
            media = [media]
        if modality in ["image", "multiple_image"]:
            if is_embedding:
                _load = lambda mm: mm

                # each mm_embedding corresponds to each image placeholder
                if not isinstance(media, list):
                    media = [media]
            else:
                _load = lambda mm: load_image(
                    mm, format=image_data_format, device=device)

            mm_data = [
                MultimodalData(modality=modality,
                               data=_load(mm),
                               is_embedding=is_embedding) for mm in media
            ]
        elif modality == "video":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for video modality yet."
                )
            mm_data = [
                MultimodalData(
                    modality=modality,
                    data=load_video(i,
                                    num_frames,
                                    format=image_data_format,
                                    device=device),
                    is_embedding=False,
                ) for i in media
            ]
        elif modality == "audio":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for audio modality yet."
                )
            mm_data = [
                MultimodalData(
                    modality=modality,
                    data=load_audio(i, device=device),
                    is_embedding=False,
                ) for i in media
            ]
        elif modality == "image_audio":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for image_audio modality yet."
                )
            # Use different load_xxx functions to match the modality.
            mm_data = []
            for m in media:
                data = None
                _modal = None
                if _modal is None:
                    try:
                        data = load_image(m,
                                          format=image_data_format,
                                          device=device)
                        _modal = "image"
                    except Exception:
                        pass
                if _modal is None:
                    try:
                        data = load_audio(m, device=device)
                        _modal = "audio"
                    except Exception:
                        pass
                if _modal is None:
                    raise ValueError(f"Unknown matching modality: {modality}")
                mm_data.append(
                    MultimodalData(modality=_modal,
                                   data=data,
                                   is_embedding=False))
        elif modality == "mixture_text_image":
            mm_data = []
            for m in media:
                if m:
                    mm_data.append(
                        MultimodalData(
                            modality="image",
                            data=load_image(m,
                                            format=image_data_format,
                                            device=device),
                            is_embedding=False,
                        ))
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return ConversationMessage(role="user", content=prompt, media=mm_data)

    assert media is not None or mm_embeddings is not None, "Either media or mm_embeddings must be provided."
    assert media is None or mm_embeddings is None, "Either media or mm_embeddings must be provided, not both."
    media_or_embeddings = media if media is not None else mm_embeddings
    is_embedding = mm_embeddings is not None

    if len(media_or_embeddings) > len(prompts) and len(prompts) == 1:
        # 1 prompt + N media
        assert not isinstance(
            media_or_embeddings[0],
            list)  # media cannot be a list of lists in this case
        media_or_embeddings = [media_or_embeddings]
    assert len(media_or_embeddings) == len(prompts)

    is_passthrough = (MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
        model_type) == ContentFormat.PASSTHROUGH)

    if tokenizer is None and not is_passthrough:
        tokenizer = ModelLoader.load_hf_tokenizer(model_dir, use_fast=True)

    processor = None
    if not is_passthrough:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  use_fast=True,
                                                  trust_remote_code=True)

    inputs = []
    for prompt_idx, (prompt,
                     media) in enumerate(zip(prompts, media_or_embeddings)):
        conv = convert_to_conversation_message(prompt, media, modality,
                                               is_embedding)
        mm_data_tracker = MultimodalDataTracker(model_type)
        for mdata in conv["media"]:
            mdata_modality = mdata["modality"]
            if modality == "multiple_image":
                mdata_modality = "image"
            mm_data_tracker.add_data(mdata_modality,
                                     mdata["data"],
                                     is_embedding=is_embedding)
        mm_placeholder_counts = mm_data_tracker.placeholder_counts()
        prompt = conv["content"]
        if mm_placeholder_counts:
            # Resolve content format to decide whether to pre-insert
            # placeholders.  OPENAI templates handle media natively (e.g.
            # numbered <image N> tags), so we must NOT pre-insert or the
            # template's dedup guard will suppress its own output.
            hf_chat_template = resolve_hf_chat_template(tokenizer, processor,
                                                        None, None)
            content_format = _resolve_content_format(model_type,
                                                     hf_chat_template)
            if content_format != ContentFormat.OPENAI:
                conv["content"] = add_multimodal_placeholders(
                    model_type, conv["content"], mm_placeholder_counts)
        prompt = apply_chat_template(
            model_type=model_type,
            tokenizer=tokenizer,
            processor=processor,
            conversation=[conv],
            add_generation_prompt=True,
            mm_placeholder_counts=[mm_placeholder_counts])
        input = {"prompt": prompt}

        if mm_placeholder_counts:
            if mm_embeddings is not None:
                _, input[
                    "multi_modal_embeddings"] = mm_data_tracker.retrieve_all_sync(
                    )
            else:
                input[
                    "multi_modal_data"], _ = mm_data_tracker.retrieve_all_sync(
                    )
        inputs.append(input)

    return inputs


def get_cache_salt_id(cache_salt: str) -> int:
    b = cache_salt.encode("utf-8")
    h = default_hasher(b).digest(length=8)
    cache_salt_id = int.from_bytes(h, "little", signed=False)
    if cache_salt_id < 0 or cache_salt_id >= (1 << 64):
        raise ValueError(
            f"cache_salt_id must be in [0, 2**64 - 1], got {cache_salt_id}.")

    return cache_salt_id
