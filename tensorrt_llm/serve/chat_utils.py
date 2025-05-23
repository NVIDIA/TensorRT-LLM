from functools import partial
from typing import (Any, Callable, Coroutine, Dict, Iterable, List, Literal,
                    Optional, Tuple, TypeAlias, TypedDict, Union, cast)

from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import (ChatCompletionContentPartTextParam,
                               ChatCompletionMessageParam)
from transformers import AutoConfig
from typing_extensions import Required

from tensorrt_llm.inputs import (ConversationMessage, MultimodalData,
                                 MultimodalDataTracker,
                                 add_multimodal_placeholders, async_load_image,
                                 async_load_video)
from tensorrt_llm.logger import logger


class VideoURL(TypedDict):
    """Type definition for video URL structure."""
    url: Required[str]


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    """Type definition for video content part parameters."""
    video_url: Required[VideoURL]
    type: Required[Literal["video_url"]]


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
    part: ChatCompletionMessageParam, skip_loading: bool = False) -> Optional[Any]:
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
        async def noop_coroutine():
            return str_content

        return MultimodalData(modality="image", data=load_image_async() if not skip_loading else noop_coroutine())

    if part_type == "video_url":
        str_content = cast(str, content)

        async def load_video_async():
            try:
                return await async_load_video(str_content, num_frames=8)
            except Exception as e:
                logger.error(f"Failed to load video: {str(e)}")
                return None
        async def noop_coroutine():
            return str_content

        return MultimodalData(modality="video", data=load_video_async() if not skip_loading else noop_coroutine())

    raise NotImplementedError(f"Unknown part type: {part_type}")


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionMessageParam],
    skip_loading: bool = False,
) -> ConversationMessage:
    """Parse multiple parts of a chat message."""
    text_parts = []
    media_parts = []
    for part in parts:
        parse_res = parse_chat_message_content_part(part, skip_loading)
        if parse_res:
            if isinstance(parse_res, str):
                text_parts.append(parse_res)
            else:
                media_parts.append(parse_res)

    text_prompt = "\n".join(text_parts)

    return ConversationMessage(role=role,
                               content=text_prompt,
                               media=media_parts)


def parse_chat_message_content(
    message: ChatCompletionMessageParam, skip_loading: bool = False) -> ConversationMessage:
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
        skip_loading,
    )
    return result


def parse_chat_messages_coroutines(
    messages: List[ChatCompletionMessageParam],
    model_config: AutoConfig,
    skip_loading: bool = False,
) -> Tuple[List[ConversationMessage], Optional[Coroutine[
        Any, Any, Optional[Dict[str, List[Any]]]]]]:
    """Parse multiple chat messages and return conversation and coroutine."""
    conversation = []
    mm_data_tracker = MultimodalDataTracker(model_config.model_type)

    for msg in messages:
        parsed_msg = parse_chat_message_content(msg, skip_loading)
        conversation.append(parsed_msg)
        if parsed_msg["media"]:
            for mdata in parsed_msg["media"]:
                mm_data_tracker.add_data(mdata["modality"], mdata["data"])
    mm_placeholder_counts = mm_data_tracker.placeholder_counts()
    if mm_placeholder_counts:
        parsed_msg["content"] = add_multimodal_placeholders(
            model_config.model_type, parsed_msg["content"],
            mm_placeholder_counts)

    return conversation, mm_data_tracker.retrieve_all_async(
    ), mm_placeholder_counts


def check_multiple_response(n: int, backend: Optional[str]):
    if n > 1 and backend == "pytorch":
        raise ValueError(
            "Multiple response is not supported in PyTorch workflow")
