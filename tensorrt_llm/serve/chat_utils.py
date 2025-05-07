import asyncio
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
    url: Required[str]


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    video_url: Required[VideoURL]
    type: Required[Literal["video_url"]]


class ConversationMessage(TypedDict):
    role: str
    content: str


_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageParser = partial(cast, ChatCompletionContentPartImageParam)
_VideoParser = partial(cast, ChatCompletionContentPartVideoParam)

ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, ChatCompletionContentPartVideoParam,
    str]

MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], Union[str, dict[str, str]]],
] = {
    "text":
    lambda part: _TextParser(part).get("text", None),
    "image_url":
    lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
    "video_url":
    lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
}

VALID_MESSAGE_CONTENT_MM_PART_TYPES = ["text", "image_url", "video_url"]


def retrieve_multimodal_placeholder(
    modality: str,
    model_config: AutoConfig,
    current_count: int,
) -> Optional[str]:
    """Retrieve the appropriate placeholder for a given modality and model type."""
    model_type = model_config.model_type

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


def add_multimodal_placeholders(
    text_prompt: str,
    mm_dicts: dict[str, Coroutine[Any, Any, Optional[Dict[str, List[Any]]]]],
    model_config: AutoConfig,
) -> str:
    placeholders = []
    counts = {}

    for media_type, media_list in mm_dicts.items():
        count = counts.get(media_type, 0)
        for _ in media_list:
            placeholder = retrieve_multimodal_placeholder(
                media_type, model_config, count)
            if placeholder is not None:
                placeholders.append(placeholder)
                count += 1
        counts[media_type] = count

    if not placeholders:
        return text_prompt

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
) -> Tuple[Optional[str], Optional[str], Optional[Coroutine[
        Any, Any, Optional[Dict[str, List[Any]]]]]]:
    """Parse a single part of a chat message."""
    if isinstance(part, str):
        return part, None, None

    # Handle structured dictionary parts
    part_type, content = _parse_chat_message_content_mm_part(part)

    # if part_type is text/image_url/video_url but content is None, log a warning and skip
    if part_type in VALID_MESSAGE_CONTENT_MM_PART_TYPES and content is None:
        logger.warning(
            "Skipping multimodal part '%s' (type: '%s') with empty / unparsable content.",
            part, part_type)
        return None, None, None

    if part_type == "text":
        return cast(str, content), None, None

    if part_type == "image_url":
        str_content = cast(str, content)

        async def load_image_async():
            try:
                return await async_load_image(str_content)
            except Exception as e:
                logger.error(f"Failed to load image: {str(e)}")
                return None

        return None, "image", load_image_async()

    elif part_type == "video_url":
        str_content = cast(str, content)

        async def load_video_async():
            try:
                return await async_load_video(str_content, num_frames=8)
            except Exception as e:
                logger.error(f"Failed to load video: {str(e)}")
                return None

        return None, "video", load_video_async()

    raise NotImplementedError(f"Unknown part type: {part_type}")


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionMessageParam],
    model_config: AutoConfig,
) -> Tuple[List[ConversationMessage], Optional[Coroutine[
        Any, Any, Optional[Dict[str, List[Any]]]]]]:
    """Parse multiple parts of a chat message."""
    content_parts = []
    mm_dicts = {}

    for part in parts:
        parse_res, media_type, media_loader_coroutine = parse_chat_message_content_part(
            part)
        if parse_res:
            content_parts.append(parse_res)
        if media_type is not None and media_loader_coroutine is not None:
            mm_dicts.setdefault(media_type, []).append(media_loader_coroutine)

    text_prompt = "\n".join(content_parts)

    text_prompt = add_multimodal_placeholders(text_prompt, mm_dicts,
                                              model_config)

    async def combined_media_loader():
        combined_data = {}
        for media_type, coroutines in mm_dicts.items():
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            valid_results = [
                r for r in results
                if r is not None and not isinstance(r, Exception)
            ]
            if valid_results:
                combined_data[media_type] = valid_results
        return combined_data if combined_data else None

    return [ConversationMessage(role=role,
                                content=text_prompt)], combined_media_loader()


def parse_chat_message_content(
    message: ChatCompletionMessageParam,
    model_config: AutoConfig,
) -> Tuple[List[ConversationMessage], Optional[Coroutine[
        Any, Any, Optional[Dict[str, List[Any]]]]]]:
    """Parse the content of a chat message."""
    role = message["role"]
    content = message.get("content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [
            ChatCompletionContentPartTextParam(type="text", text=content)
        ]

    result, mm_data_coroutine = parse_chat_message_content_parts(
        role,
        content,
        model_config,
    )
    return result, mm_data_coroutine


def parse_chat_messages_coroutines(
    messages: List[ChatCompletionMessageParam],
    model_config: AutoConfig,
) -> Tuple[List[ConversationMessage], Optional[Coroutine[
        Any, Any, Optional[Dict[str, List[Any]]]]]]:
    conversation = []
    mm_coroutines = []

    for msg in messages:
        sub_messages, mm_coroutine = parse_chat_message_content(
            msg, model_config)
        conversation.extend(sub_messages)
        if mm_coroutine is not None:
            mm_coroutines.append(mm_coroutine)

    async def combined_media_loader():
        results = await asyncio.gather(*mm_coroutines, return_exceptions=True)
        combined_data = {}

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue

            for media_type, data_list in result.items():
                if data_list:
                    combined_data.setdefault(media_type, []).extend(data_list)

        return combined_data if combined_data else None

    return conversation, combined_media_loader()


def resolve_hf_chat_template(
    tokenizer: TokenizerBase,
    processor: ProcessorMixin,
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
) -> str:

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
    **kwargs: Any,
) -> str:
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
