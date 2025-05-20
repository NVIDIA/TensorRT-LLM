import asyncio
from collections import defaultdict
from functools import partial
from typing import (Any, Callable, Coroutine, Dict, Iterable, List, Literal,
                    Optional, Tuple, TypeAlias, TypedDict, Union, cast)

from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import (ChatCompletionContentPartTextParam,
                               ChatCompletionMessageParam)
from transformers import AutoConfig, ProcessorMixin
from typing_extensions import Required

from tensorrt_llm.inputs import async_load_image, async_load_video
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
from tensorrt_llm.logger import logger


class VideoURL(TypedDict):
    """Type definition for video URL structure."""
    url: Required[str]


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    """Type definition for video content part parameters."""
    video_url: Required[VideoURL]
    type: Required[Literal["video_url"]]


class ConversationMessage(TypedDict):
    """Type definition for conversation message structure."""
    role: str
    content: str


# Type Aliases and Constants
ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, ChatCompletionContentPartVideoParam,
    str]

VALID_MESSAGE_CONTENT_MM_PART_TYPES = ["text", "image_url", "video_url"]

# Parser Functions
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageParser = partial(cast, ChatCompletionContentPartImageParam)
_VideoParser = partial(cast, ChatCompletionContentPartVideoParam)

MM_PARSER_MAP: dict[str, Callable[[ChatCompletionContentPartParam], Union[
    str, dict[str, str]]]] = {
        "text":
        lambda part: _TextParser(part).get("text", None),
        "image_url":
        lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
        "video_url":
        lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
    }


class AsyncMultimodalDataTracker:
    """Tracks and manages multimodal data for async processing."""

    def __init__(self, model_config: AutoConfig):
        self.model_config = model_config
        self.mm_data = defaultdict[str](list)
        self.mm_placeholder_counts = defaultdict[str](int)

    async def retrieve_all_mm_data(self) -> Optional[Dict[str, List[Any]]]:
        """Retrieve all collected multimodal data."""
        if not self.mm_data:
            return None

        return {
            modality: await asyncio.gather(*items)
            for modality, items in self.mm_data.items()
        }

    def retrieve_multimodal_placeholder(self, modality: str,
                                        current_count: int) -> Optional[str]:
        """Get the appropriate placeholder for a given modality and model type."""
        model_type = self.model_config.model_type

        if modality == "image":
            if model_type in ("qwen2_vl", "qwen2_5_vl"):
                return "<|vision_start|><|image_pad|><|vision_end|>"
            elif model_type in ("mllama", "llama4"):
                return "<|image|>"
            raise TypeError(f"Unknown {modality} model type: {model_type}")
        elif modality == "video":
            if model_type in ("qwen2_vl", "qwen2_5_vl"):
                return "<|vision_start|><|video_pad|><|vision_end|>"
            raise TypeError(f"Unknown {modality} model type: {model_type}")
        raise TypeError(f"Unknown modality: {modality}")

    def add_mm_data(self, media_type: str, data: Coroutine):
        current_count = len(self.mm_data[media_type]) + 1
        placeholder = self.retrieve_multimodal_placeholder(
            media_type, current_count)
        self.mm_data[media_type].append(data)
        if placeholder:
            self.mm_placeholder_counts[placeholder] += 1

    def mm_data_counts(self) -> Dict[str, int]:
        """Get the count of multimodal placeholders."""
        return dict(self.mm_placeholder_counts)


def add_multimodal_placeholders(text_prompt: str,
                                mm_placeholder_counts: dict[str, int]) -> str:
    """Add multimodal placeholders to the text prompt."""
    placeholders = []
    for placeholder in mm_placeholder_counts:
        placeholders.extend([placeholder] * mm_placeholder_counts[placeholder])
    return "\n".join(placeholders + [text_prompt])


def _parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam
) -> tuple[str, Union[str, dict[str, str]]]:
    """Parse a single multimodal part of a chat message."""
    assert isinstance(part, dict)
    part_type = part.get("type", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP:
        return part_type, MM_PARSER_MAP[part_type](part)

    if not isinstance(part_type, str):
        raise ValueError("Invalid 'type' field in multimodal part.")
    return part_type, "unknown part_type content"


def parse_chat_message_content_part(
    part: ChatCompletionMessageParam,
    mm_data_tracker: AsyncMultimodalDataTracker,
) -> Optional[str]:
    """Parse a single part of a chat message."""
    if isinstance(part, str):
        return part

    part_type, content = _parse_chat_message_content_mm_part(part)

    # if part_type is text/image_url/video_url but content is None, log a warning and skip
    if part_type in VALID_MESSAGE_CONTENT_MM_PART_TYPES and content is None:
        logger.warning(
            "Skipping multimodal part '%s' (type: '%s') with empty / unparsable content.",
            part, part_type)
        return None

    if part_type == "text":
        return cast(str, content)

    if part_type == "image_url":
        str_content = cast(str, content)

        async def load_image_async():
            try:
                return await async_load_image(str_content)
            except Exception as e:
                logger.error(f"Failed to load image: {str(e)}")
                return None

        mm_data_tracker.add_mm_data("image", load_image_async())
        return None

    if part_type == "video_url":
        str_content = cast(str, content)

        async def load_video_async():
            try:
                return await async_load_video(str_content, num_frames=8)
            except Exception as e:
                logger.error(f"Failed to load video: {str(e)}")
                return None

        mm_data_tracker.add_mm_data("video", load_video_async())
        return None

    raise NotImplementedError(f"Unknown part type: {part_type}")


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionMessageParam],
    mm_data_tracker: AsyncMultimodalDataTracker,
) -> List[ConversationMessage]:
    """Parse multiple parts of a chat message."""
    content_parts = []
    for part in parts:
        parse_res = parse_chat_message_content_part(part, mm_data_tracker)
        if parse_res:
            content_parts.append(parse_res)

    text_prompt = "\n".join(content_parts)
    mm_placeholder_counts = mm_data_tracker.mm_data_counts()

    if mm_placeholder_counts:
        text_prompt = add_multimodal_placeholders(text_prompt,
                                                  mm_placeholder_counts)

    return [ConversationMessage(role=role, content=text_prompt)]


def parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_data_tracker: AsyncMultimodalDataTracker,
) -> List[ConversationMessage]:
    """Parse the content of a chat message."""
    role = message["role"]
    content = message.get("content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [
            ChatCompletionContentPartTextParam(type="text", text=content)
        ]

    result = parse_chat_message_content_parts(
        role,
        content,
        mm_data_tracker,
    )
    return result


def parse_chat_messages_coroutines(
    messages: List[ChatCompletionMessageParam],
    model_config: AutoConfig,
) -> Tuple[List[ConversationMessage], Optional[Coroutine[
        Any, Any, Optional[Dict[str, List[Any]]]]]]:
    """Parse multiple chat messages and return conversation and coroutine."""
    conversation = []
    mm_data_tracker = AsyncMultimodalDataTracker(model_config)

    for msg in messages:
        sub_messages = parse_chat_message_content(msg, mm_data_tracker)
        conversation.extend(sub_messages)

    return conversation, mm_data_tracker.retrieve_all_mm_data()


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
        logger.debug("Failed to load AutoTokenizer chat template for %s",
                     tokenizer.name_or_path)
    return None


def apply_chat_template(
    *,
    tokenizer: TokenizerBase,
    processor: ProcessorMixin,
    conversation: list[ConversationMessage],
    add_generation_prompt: bool,
    tools: Optional[list[dict[str, Any]]] = None,
    documents: Optional[list[dict[str, str]]] = None,
    chat_template: Optional[str] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """Apply chat template to the conversation."""
    hf_chat_template = resolve_hf_chat_template(tokenizer, processor,
                                                chat_template, tools)

    if hf_chat_template is None:
        raise ValueError(
            "No chat template found for the given tokenizer and tools.")

    return tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        documents=documents,
        chat_template=hf_chat_template,
        **(chat_template_kwargs or {}),
    )
