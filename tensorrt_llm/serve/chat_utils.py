import json
import uuid
from functools import lru_cache, partial
from pathlib import Path
from typing import (Any, Callable, Coroutine, Dict, Iterable, List, Literal,
                    Optional, Tuple, TypeAlias, TypedDict, Union, cast)

from openai.types.chat import (ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartInputAudioParam)
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import (ChatCompletionContentPartTextParam,
                               ChatCompletionMessageParam)
from transformers import AutoConfig
from typing_extensions import Required

from tensorrt_llm.inputs import (ConversationMessage, MultimodalData,
                                 MultimodalDataTracker,
                                 add_multimodal_placeholders, async_load_audio,
                                 async_load_image, async_load_video,
                                 load_base64_image_embeds)
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.logger import logger


class VideoURL(TypedDict):
    """Type definition for video URL structure."""
    url: Required[str]


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    """Type definition for video content part parameters."""
    video_url: Required[VideoURL]
    type: Required[Literal["video_url"]]


class ImageEmbedsData(TypedDict):
    """Type definition for serialized image embeddings structure."""
    data: Required[str]


class ChatCompletionContentPartImageEmbedsParam(TypedDict, total=False):
    """Type definition for image embeddings passed in base64-encoded PyTorch tensor format."""
    image_embeds: Required[
        # TODO: Besides "data", could support "url" and "ipc_handle" in the future.
        ImageEmbedsData]
    type: Required[Literal["image_embeds"]]


# Type Aliases and Constants
ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam,
    ChatCompletionContentPartVideoParam,
    ChatCompletionContentPartImageEmbedsParam,
    str,
]

# TODO: Add "input_audio" to support byte_encoded audio input.
VALID_MESSAGE_CONTENT_MM_PART_TYPES = [
    "text",
    "image_url",
    "video_url",
    "audio_url",
    "image_embeds",
]

# Parser Functions
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageParser = partial(cast, ChatCompletionContentPartImageParam)
_ImageEmbedsParser = partial(cast, ChatCompletionContentPartImageEmbedsParam)
_VideoParser = partial(cast, ChatCompletionContentPartVideoParam)
_AudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)

MM_PARSER_MAP: dict[str, Callable[[ChatCompletionContentPartParam], Union[
    str, dict[str, str], None]]] = {
        "text":
        lambda part: _TextParser(part).get("text", None),
        "image_url":
        lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
        "video_url":
        lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
        "audio_url":
        lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
        "image_embeds":
        lambda part: _ImageEmbedsParser(part).get("image_embeds", {}).get(
            "data", None),
    }


def _parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam
) -> tuple[str, Union[str, dict[str, str], None]]:
    """Parse a single multimodal part of a chat message."""
    assert isinstance(part, dict)
    part_type = part.get("type", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP:
        return part_type, MM_PARSER_MAP[part_type](part)

    if not isinstance(part_type, str):
        raise ValueError("Invalid 'type' field in multimodal part.")
    return part_type, "unknown part_type content"


def parse_chat_message_content_part(
    part: ChatCompletionContentPartParam,
    mm_data_tracker: MultimodalDataTracker,
) -> str | MultimodalData | None:
    """Parse a single part of a chat message."""
    if isinstance(part, str):
        return part

    part_type, content = _parse_chat_message_content_mm_part(part)

    # if part_type is text/image_url/video_url/audio_url but content is None, log a warning and skip
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
                image_kwargs = (
                    mm_data_tracker._multimodal_server_config.media_io_kwargs
                    or {}).get("image", {})
                return await async_load_image(str_content, **image_kwargs)
            except Exception as e:
                logger.error(f"Failed to load image: {str(e)}")
                return None

        return MultimodalData(modality="image",
                              data=load_image_async(),
                              is_embedding=False)

    if part_type == "image_embeds":
        str_content = cast(str, content)

        async def decode_image_embeds_async():
            try:
                return load_base64_image_embeds(str_content)
            except Exception as e:
                logger.error(f"Failed to decode image data: {str(e)}")
                return None

        return MultimodalData(modality="image",
                              data=decode_image_embeds_async(),
                              is_embedding=True)

    if part_type == "video_url":
        str_content = cast(str, content)

        async def load_video_async():
            try:
                video_kwargs = (
                    mm_data_tracker._multimodal_server_config.media_io_kwargs
                    or {}).get("video", {})
                return await async_load_video(str_content, **video_kwargs)
            except Exception as e:
                logger.error(f"Failed to load video: {str(e)}")
                return None

        return MultimodalData(modality="video",
                              data=load_video_async(),
                              is_embedding=False)

    if part_type == "audio_url":
        str_content = cast(str, content)

        async def load_audio_async():
            try:
                audio_kwargs = (
                    mm_data_tracker._multimodal_server_config.media_io_kwargs
                    or {}).get("audio", {})
                return await async_load_audio(str_content, **audio_kwargs)
            except Exception as e:
                logger.error(f"Failed to load audio: {str(e)}")
                return None

        return MultimodalData(modality="audio",
                              data=load_audio_async(),
                              is_embedding=False)

    raise NotImplementedError(f"Unknown part type: {part_type}")


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    mm_data_tracker: MultimodalDataTracker,
) -> ConversationMessage:
    """Parse multiple parts of a chat message."""
    text_parts = []
    media_parts = []
    for part in parts:
        parse_res = parse_chat_message_content_part(part, mm_data_tracker)
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
        message: ChatCompletionMessageParam,
        mm_data_tracker: MultimodalDataTracker) -> ConversationMessage:
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
    if role == "assistant":
        result.update(_parse_assistant_message_content(message))
    elif role == "tool":
        result.update(_parse_tool_message_content(message))
    return result


# Adapted from: https://github.com/vllm-project/vllm/blob/4574d48bab9c4e38b7c0a830eeefc8f0980e8c58/vllm/entrypoints/chat_utils.py#L1406
def _parse_assistant_message_content(message: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        result["tool_calls"] = []
        for item in tool_calls:
            if content := item["function"].get("arguments"):
                if isinstance(content, str):
                    item["function"]["arguments"] = json.loads(content)
                else:
                    item["function"]["arguments"] = content
            else:
                item["function"]["arguments"] = {}
            result["tool_calls"].append(item)

    return result


def _parse_tool_message_content(message: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    if "tool_call_id" in message:
        result["tool_call_id"] = message["tool_call_id"]
    return result


def parse_chat_messages_coroutines(
    messages: List[ChatCompletionMessageParam],
    model_config: AutoConfig,
    multimodal_server_config: Optional[MultimodalServerConfig] = None
) -> Tuple[List[ConversationMessage], Coroutine[Any, Any, tuple[Optional[Dict[
        str, List[Any]]], Optional[Dict[str, List[Any]]]]], list[dict[str,
                                                                      int]]]:
    """Parse multiple chat messages and return conversation and coroutine."""
    conversation = []
    mm_placeholder_counts = []
    mm_data_tracker = MultimodalDataTracker(model_config.model_type,
                                            multimodal_server_config)

    for msg in messages:
        parsed_msg = parse_chat_message_content(msg, mm_data_tracker)
        conversation.append(parsed_msg)
        if parsed_msg["media"]:
            for mdata in parsed_msg["media"]:
                mm_data_tracker.add_data(mdata["modality"],
                                         mdata["data"],
                                         is_embedding=mdata["is_embedding"])
        mm_placeholder_count = mm_data_tracker.placeholder_counts()
        if mm_placeholder_count:
            parsed_msg["content"] = add_multimodal_placeholders(
                model_config.model_type, parsed_msg["content"],
                mm_placeholder_count)
        mm_placeholder_counts.append(mm_placeholder_count)

    return conversation, mm_data_tracker.retrieve_all_async(
    ), mm_placeholder_counts


def make_tool_call_id(id_type: str = "random", func_name=None, idx=None):
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    else:
        # by default return random
        return f"chatcmpl-tool-{uuid.uuid4().hex}"


# Adapted from
# https://github.com/vllm-project/vllm/blob/44b5ce956d3cf28841615a58c1c0873af87bcfe2/vllm/entrypoints/chat_utils.py
@lru_cache
def load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    if chat_template is None:
        return None

    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError(
                "chat_template is expected to be read directly from its value")

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise

        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            msg = (f"The supplied chat template ({chat_template}) "
                   f"looks like a file path, but it failed to be "
                   f"opened. Reason: {e}")
            raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        return load_chat_template(chat_template, is_literal=True)
