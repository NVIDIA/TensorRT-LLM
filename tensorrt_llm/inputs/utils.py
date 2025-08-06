import asyncio
import base64
import enum
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypedDict, Union
from urllib.parse import urlparse

import aiohttp
import numpy as np
import requests
import soundfile
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoProcessor, ProcessorMixin
from transformers.utils import logging

from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.llmapi.tokenizer import TokenizerBase, TransformersTokenizer

logger = logging.get_logger(__name__)


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


def load_image(image: str,
               format: str = "pt",
               device: str = "cpu") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

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
        image: str,
        format: str = "pt",
        device: str = "cpu") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

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


def load_video(
        video: str,
        num_frames: int = 10,
        format: str = "pt",
        device: str = "cpu") -> Union[List[Image.Image], List[torch.Tensor]]:

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
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video}' has no frames.")

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = Image.fromarray(frame)

    return [
        ToTensor()(frames[index]).to(
            device=device) if format == "pt" else frames[index]
        for index in indices if index in frames
    ]


async def async_load_video(
        video: str,
        num_frames: int = 10,
        format: str = "pt",
        device: str = "cpu") -> Union[List[Image.Image], List[torch.Tensor]]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    parsed_url = urlparse(video)

    if parsed_url.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            async with session.get(video) as response:
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix='.mp4') as tmp:
                    tmp.write(await response.content.read())
                    video_path = tmp.name
    # TODO: add case for video encoded in base64
    else:
        video_path = video

    return load_video(video_path, num_frames, format, device)


def load_audio(
    audio: str,
    format: str = "pt",
    device: str = "cuda",
) -> Tuple[np.ndarray, int]:
    parsed_url = urlparse(audio)
    if parsed_url.scheme in ["http", "https"]:
        audio = requests.get(audio, stream=True, timeout=10)
        audio = BytesIO(audio.content)

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

    audio = soundfile.read(audio)
    return audio


# Copied from https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py#L38
def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url, timeout=10) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


"""
VLM input preparation.

NOTE:
    When a new multimodal model is added, the following list(s) need
    to be updated with the new model type and the appropriate
    placeholder for the model needs to be added in retrieve_multimodal_placeholder().
"""

SUPPORTED_QWEN_MODEL_GROUP = ["qwen2_vl", "qwen2_5_vl"]
SUPPORTED_GEMMA_MODEL_GROUP = ["gemma3"]
SUPPORTED_LLAMA_MODEL_GROUP = ["mllama", "llama4"]
SUPPORTED_LLAVA_IMAGE_MODEL_GROUP = ["llava_llama", "llava_next"]
SUPPORTED_LLAVA_VIDEO_MODEL_GROUP = ["llava_llama"]
SUPPORTED_MISTRAL_IMAGE_MODEL_GROUP = ["mistral3"]
SUPPORTED_HYPERCLOVAX_MODEL_GROUP = ["hyperclovax_vlm"]
SUPPORTED_PHI_MODEL_GROUP = ["phi4mm"]

ALL_SUPPORTED_IMAGE_MODELS = SUPPORTED_QWEN_MODEL_GROUP \
    + SUPPORTED_LLAMA_MODEL_GROUP \
    + SUPPORTED_LLAVA_IMAGE_MODEL_GROUP \
    + SUPPORTED_HYPERCLOVAX_MODEL_GROUP \
    + SUPPORTED_GEMMA_MODEL_GROUP \
    + SUPPORTED_MISTRAL_IMAGE_MODEL_GROUP \
    + SUPPORTED_PHI_MODEL_GROUP

ALL_SUPPORTED_VIDEO_MODELS = SUPPORTED_QWEN_MODEL_GROUP \
    + SUPPORTED_LLAVA_VIDEO_MODEL_GROUP

ALL_SUPPORTED_AUDIO_MODELS = SUPPORTED_PHI_MODEL_GROUP

ALL_SUPPORTED_MULTIMODAL_MODELS = list(set(ALL_SUPPORTED_IMAGE_MODELS) \
    | set(ALL_SUPPORTED_VIDEO_MODELS) \
    | set(ALL_SUPPORTED_AUDIO_MODELS))

HF_CHAT_TEMPLATE_EXCEPTIONS = ["llava_llama"]
PLACEHOLDER_EXCEPTIONS = ["llava_next"]


class MultimodalPlaceholderPlacement(enum.Enum):
    INVALID = -1
    BEFORE_TEXT = 0
    AFTER_TEXT = 1


PLACEHOLDER_PLACEMENT_MAP = {
    "qwen2_vl": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    "qwen2_5_vl": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    "llava_llama": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    "llava_next": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    "llama4": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    "mllama": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    "hyperclovax_vlm": MultimodalPlaceholderPlacement.AFTER_TEXT,
    "gemma3": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    # NOTE: for mistral3 multimodal models, it does not strictly have to be before the text.
    # Ref: https://github.com/mistralai/mistral-common/blob/039465db2bdc0486df36365c9bdb428188482a18/
    #      src/mistral_common/tokens/tokenizers/base.py#L326
    # However, accuracy tests show that the model generates higher quality output when the image
    # precedes the text (the relative difference can be as much as ~30% for both vLLM and TRT-LLM).
    "mistral3": MultimodalPlaceholderPlacement.BEFORE_TEXT,
    "phi4mm": MultimodalPlaceholderPlacement.BEFORE_TEXT,
}
assert len(PLACEHOLDER_PLACEMENT_MAP) == len(ALL_SUPPORTED_MULTIMODAL_MODELS)


def retrieve_multimodal_placeholder(model_type: str, modality: str,
                                    current_count: int) -> Optional[str]:
    """
        Get the appropriate placeholder for a given modality and model type.

        Args:
            model_type: The type of the multimodal model.
            modality: The modality of the data.
            current_count: The number of multimodal data already added.

    """

    if modality == "image":
        if model_type in SUPPORTED_QWEN_MODEL_GROUP:
            return "<|vision_start|><|image_pad|><|vision_end|>"
        elif model_type in SUPPORTED_LLAMA_MODEL_GROUP:
            return "<|image|>"
        elif model_type in SUPPORTED_LLAVA_IMAGE_MODEL_GROUP:
            return "<image>"
        elif model_type in SUPPORTED_GEMMA_MODEL_GROUP:
            return "<start_of_image>"
        elif model_type in SUPPORTED_HYPERCLOVAX_MODEL_GROUP:
            return '<im_end>\n<|im_start|>user (mime) \n{"type": "image/jpeg", "filename": ""}<|im_end|>\n' + \
                    '<|im_start|>user (vector)\n<|dummy3|><|im_end|>\n' + \
                    '<|im_start|>image/aux\n다음 중 ocr은 사진에서 검출된 글자이고, lens_keyword는 사진에서 추출된 keyword와 bbox 위치입니다.' + \
                    'bbox는 0~1 사이로 정규화된 [x1, y1, x2, y2]의 형태입니다. 참고하여 답변하세요. {"ocr": "", "lens_keywords": "", "lens_local_keywords": ""}'
        elif model_type in SUPPORTED_MISTRAL_IMAGE_MODEL_GROUP:
            # Ref: https://github.com/mistralai/mistral-common/blob/26a6bb3a07ee0b78a3808f2797f23e1d28514b93/
            # src/mistral_common/tokens/tokenizers/base.py#L60
            return "[IMG]"
        elif model_type in SUPPORTED_PHI_MODEL_GROUP:
            return f"<|image_{current_count}|>"
        raise TypeError(
            f"For image modality, only {ALL_SUPPORTED_IMAGE_MODELS} are supported but got {model_type}"
        )
    elif modality == "video":
        if model_type in SUPPORTED_QWEN_MODEL_GROUP:
            return "<|vision_start|><|video_pad|><|vision_end|>"
        elif model_type in SUPPORTED_LLAVA_VIDEO_MODEL_GROUP:
            return "<vila/video>"
        raise TypeError(
            f"For video modality, only {ALL_SUPPORTED_VIDEO_MODELS} are supported but got {model_type}"
        )
    elif modality == "audio":
        if model_type in SUPPORTED_PHI_MODEL_GROUP:
            return f"<|audio_{current_count}|>"
    raise TypeError(f"Unknown modality: {modality}")


class MultimodalData(TypedDict):
    """Type definition for multimodal data structure."""
    modality: str
    data: Any


class ConversationMessage(TypedDict):
    """Type definition for conversation message structure."""
    role: str
    content: List[dict[str, Any]]
    media: List[MultimodalData] | List[torch.Tensor] | List[Dict[str, Any]]

    # @classmethod
    # def fromSample(cls, sample: dict[str, str]) -> "ConversationMessage":
    #     return cls(role="user", content=[{"type": "text", "text": prompt}])


class MultimodalDataTracker:
    """Tracks and manages multimodal data for both sync and async processing."""

    def __init__(self, model_type: str):
        self._model_type = model_type
        self._data = defaultdict[str](list)
        self._placeholder_counts = defaultdict[str](int)

    async def retrieve_all_async(self) -> Optional[Dict[str, List[Any]]]:
        """Retrieve all collected multimodal data."""
        if not self._data:
            return None

        return {
            modality: await asyncio.gather(*items)
            for modality, items in self._data.items()
        }

    def retrieve_all_sync(self) -> Optional[Dict[str, List[Any]]]:
        """Retrieve all collected multimodal data."""
        if not self._data:
            return None

        return {modality: items for modality, items in self._data.items()}

    def add_data(self, media_type: str, data: Union[Coroutine, Any]):
        current_count = len(self._data[media_type]) + 1
        placeholder = retrieve_multimodal_placeholder(self._model_type,
                                                      media_type, current_count)
        self._data[media_type].append(data)
        if placeholder:
            self._placeholder_counts[placeholder] += 1

    def placeholder_counts(self) -> Dict[str, int]:
        """Get the count of multimodal placeholders."""
        return dict(self._placeholder_counts)


def add_multimodal_placeholders(model_type: str, text_prompt: str,
                                mm_placeholder_counts: dict[str, int]) -> str:
    """Add multimodal placeholders to the text prompt."""
    if model_type in PLACEHOLDER_EXCEPTIONS:
        # no need to add placeholders, it is handled differently
        return text_prompt
    placeholders = []
    for placeholder in mm_placeholder_counts:
        placeholders.extend([placeholder] * mm_placeholder_counts[placeholder])
    parts = []
    match PLACEHOLDER_PLACEMENT_MAP[model_type]:
        case MultimodalPlaceholderPlacement.BEFORE_TEXT:
            parts.extend(placeholders)
            parts.append(text_prompt)
        case MultimodalPlaceholderPlacement.AFTER_TEXT:
            parts.append(text_prompt)
            parts.extend(placeholders)
    if model_type == "phi4mm":
        return "".join(parts)
    else:
        return "\n".join(parts)


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
    if not tools and processor and hasattr(processor, 'chat_template'):
        return processor.chat_template

    # 3. If tool is provided, use the tool
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except Exception:
        logger.warning("Failed to load AutoTokenizer chat template for %s",
                       tokenizer.name_or_path)
    return None


def handle_placeholder_exceptions(model_type: str,
                                  conversation: list[ConversationMessage],
                                  mm_placeholder_counts: dict[str, int]):
    if model_type == "llava_next":
        # we need to convert the flattened content back to conversation format
        for conv in conversation:
            conv["content"] = [{"type": "text", "text": conv["content"]}, \
                *[{"type": "image"} for _ in mm_placeholder_counts]]
    else:
        raise ValueError(f"This path should not be reached for: {model_type}")
    return conversation


def apply_chat_template(
    *,
    model_type: str,
    tokenizer: Union[TransformersTokenizer, TokenizerBase],
    processor: ProcessorMixin,
    conversation: list[ConversationMessage],
    add_generation_prompt: bool,
    mm_placeholder_counts: dict[str, int],
    tools: Optional[list[dict[str, Any]]] = None,
    documents: Optional[list[dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> (str | List[str]):
    """Apply chat template to the conversation."""

    if model_type in HF_CHAT_TEMPLATE_EXCEPTIONS:
        # special path for models like llava-llama
        return "".join([conv["content"] for conv in conversation])
    if isinstance(tokenizer, TransformersTokenizer):
        tokenizer = tokenizer.tokenizer  # we need the TokenizerBase for apply_chat_template

    hf_chat_template = resolve_hf_chat_template(tokenizer, processor,
                                                chat_template, tools)
    if hf_chat_template is None:
        raise ValueError(
            "No chat template found for the given tokenizer and tools.")
    if model_type in PLACEHOLDER_EXCEPTIONS:
        # flattened content do not work for these models, so go back to other formats as needed
        conversation = handle_placeholder_exceptions(model_type, conversation,
                                                     mm_placeholder_counts)

    return tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        documents=documents,
        chat_template=hf_chat_template,
        **(chat_template_kwargs or {}),
    )


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
                # each mm_embedding corresponds to each image placeholder
                if not isinstance(media, list):
                    media = [media]

                mm_data = [{
                    'modality': modality,
                    'mm_embedding_info': mm
                } for mm in media]
            else:
                mm_data = [
                    MultimodalData(modality=modality,
                                   data=load_image(i,
                                                   format=image_data_format,
                                                   device=device))
                    for i in media
                ]
        elif modality == "video":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for video modality yet."
                )
            mm_data = [
                MultimodalData(modality=modality,
                               data=load_video(i,
                                               num_frames,
                                               format=image_data_format,
                                               device=device)) for i in media
            ]
        elif modality == "audio":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for audio modality yet."
                )
            mm_data = [
                MultimodalData(modality=modality,
                               data=load_audio(i, device=device)) for i in media
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
                mm_data.append(MultimodalData(modality=_modal, data=data))
        elif modality == "mixture_text_image":
            mm_data = []
            for m in media:
                if m:
                    mm_data.append(
                        MultimodalData(modality="image",
                                       data=load_image(m,
                                                       format=image_data_format,
                                                       device=device)))
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

    if tokenizer is None and model_type not in HF_CHAT_TEMPLATE_EXCEPTIONS:
        tokenizer = ModelLoader.load_hf_tokenizer(model_dir, use_fast=True)

    processor = None
    if model_type not in HF_CHAT_TEMPLATE_EXCEPTIONS:
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
            # Check if mdata is a MultimodalData
            if isinstance(mdata,
                          dict) and "modality" in mdata and "data" in mdata:
                mm_data_tracker.add_data(mdata["modality"], mdata["data"])
            else:
                # Add embeddings to the tracker for placeholder handling
                mm_data_tracker.add_data(mdata["modality"],
                                         mdata["mm_embedding_info"])
        mm_placeholder_counts = mm_data_tracker.placeholder_counts()
        prompt = conv["content"]
        if mm_placeholder_counts:
            conv["content"] = add_multimodal_placeholders(
                model_type, conv["content"], mm_placeholder_counts)
        prompt = apply_chat_template(
            model_type=model_type,
            tokenizer=tokenizer,
            processor=processor,
            conversation=[conv],
            add_generation_prompt=True,
            mm_placeholder_counts=mm_placeholder_counts)
        input = {"prompt": prompt}
        if mm_placeholder_counts:
            if mm_embeddings is not None:
                input[
                    "multi_modal_embeddings"] = mm_data_tracker.retrieve_all_sync(
                    )
            else:
                input["multi_modal_data"] = mm_data_tracker.retrieve_all_sync()
        inputs.append(input)

    return inputs
