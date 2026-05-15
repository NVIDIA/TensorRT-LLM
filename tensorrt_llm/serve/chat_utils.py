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

from tensorrt_llm.inputs import (ContentFormat, ConversationMessage,
                                 MultimodalData, MultimodalDataTracker,
                                 add_multimodal_placeholders, async_load_audio,
                                 async_load_image, async_load_video,
                                 load_base64_image_embeds)
from tensorrt_llm.inputs.media_io import MEDIA_IO_REGISTRY, BaseMediaIO
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
from tensorrt_llm.inputs.utils import interleave_mm_placeholders
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

VALID_MESSAGE_CONTENT_MM_PART_TYPES = frozenset([
    "text",
    "image_url",
    "video_url",
    "audio_url",
    "input_audio",
    "image_embeds",
])

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
        "input_audio":
        lambda part: cast(dict, part).get("input_audio", None),
        "image_embeds":
        lambda part: _ImageEmbedsParser(part).get("image_embeds", {}).get(
            "data", None),
    }


def resolve_media_io_kwargs(
    server_config: Optional[MultimodalServerConfig],
    request_kwargs: Optional[Dict[str, Dict[str, Any]]],
    modality: str,
) -> Dict[str, Any]:
    """Resolve the effective loader kwargs for one modality on one request.

    Delegates to `MediaIO.merge_kwargs` (see `inputs/media_io.py`); the
    default rule is a shallow merge with request keys winning. Unknown
    kwargs surface as a `TypeError` from the loader at request time.
    """
    server_kwargs = server_config.media_io_kwargs if server_config else None
    default_kwargs = (server_kwargs or {}).get(modality, {})
    runtime_kwargs = (request_kwargs or {}).get(modality, {})
    media_io_cls = MEDIA_IO_REGISTRY.get(modality, BaseMediaIO)
    return media_io_cls.merge_kwargs(default_kwargs, runtime_kwargs)


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
        image_kwargs = resolve_media_io_kwargs(
            mm_data_tracker._multimodal_server_config,
            mm_data_tracker.request_media_io_kwargs, "image")
        logger.debug("effective image_kwargs keys: %s", sorted(image_kwargs))
        return MultimodalData(modality="image",
                              data=async_load_image(str_content,
                                                    **image_kwargs),
                              is_embedding=False)

    if part_type == "image_embeds":
        str_content = cast(str, content)

        async def decode_image_embeds():
            return load_base64_image_embeds(str_content)

        return MultimodalData(modality="image",
                              data=decode_image_embeds(),
                              is_embedding=True)

    if part_type == "video_url":
        str_content = cast(str, content)
        video_kwargs = resolve_media_io_kwargs(
            mm_data_tracker._multimodal_server_config,
            mm_data_tracker.request_media_io_kwargs, "video")
        logger.debug("effective video_kwargs keys: %s", sorted(video_kwargs))
        return MultimodalData(modality="video",
                              data=async_load_video(str_content,
                                                    **video_kwargs),
                              is_embedding=False)

    if part_type == "audio_url":
        str_content = cast(str, content)
        audio_kwargs = resolve_media_io_kwargs(
            mm_data_tracker._multimodal_server_config,
            mm_data_tracker.request_media_io_kwargs, "audio")
        logger.debug("effective audio_kwargs keys: %s", sorted(audio_kwargs))
        return MultimodalData(modality="audio",
                              data=async_load_audio(str_content,
                                                    **audio_kwargs),
                              is_embedding=False)

    if part_type == "input_audio":
        dict_content = cast(dict, content)
        audio_data = dict_content.get("data")
        if not isinstance(audio_data, str) or not audio_data:
            raise ValueError(
                "input_audio part is missing a non-empty 'data' field with "
                "base64-encoded audio content.")
        return MultimodalData(modality="audio",
                              data=async_load_audio(audio_data, is_base64=True),
                              is_embedding=False)

    raise NotImplementedError(f"Unknown part type: {part_type}")


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    mm_data_tracker: MultimodalDataTracker,
) -> ConversationMessage:
    """Parse multiple parts of a chat message.

    Builds both the flattened text (`content`) for backward compatibility and an ordered
    `content_parts` list that preserves the interleaved positions of text and media items.
    """
    text_parts: list[str] = []
    media_parts: list[MultimodalData] = []
    content_parts: list[Union[str, dict]] = []

    media_index = 0
    for part in parts:
        parse_res = parse_chat_message_content_part(part, mm_data_tracker)
        if parse_res:
            if isinstance(parse_res, str):
                text_parts.append(parse_res)
                content_parts.append(parse_res)
            else:
                media_parts.append(parse_res)
                content_parts.append({
                    "type": parse_res["modality"],
                    "media_index": media_index,
                })
                media_index += 1

    text_prompt = "\n".join(text_parts)

    result = ConversationMessage(role=role,
                                 content=text_prompt,
                                 media=media_parts)
    # Only include content_parts when media is present (to preserve
    # interleaved ordering for multimodal dispatch).
    if media_parts:
        result["content_parts"] = content_parts
    return result


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
    # Include reasoning if present for interleaved thinking.
    reasoning_content = message.get("reasoning")
    if reasoning_content is None:
        reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None:
        result["reasoning_content"] = reasoning_content

    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        # Materialize Pydantic v2 ValidatorIterator (single-use) to a list.
        if not isinstance(tool_calls, list):
            tool_calls = list(tool_calls)

        result["tool_calls"] = []
        for item in tool_calls:
            # Bypass pydantic check to WAR `tau2-bench-telecom` ill-format tool_call.
            item = dict(item)
            if "function" in item:
                item["function"] = dict(item["function"])

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


def resolve_top_level_model_type(model_config: AutoConfig) -> str:
    """Return the top-level HF model_type for a loaded config.

    Newer composite configs (e.g. Qwen2_5_VLConfig) delegate the instance
    attribute to `text_config`, returning e.g. "qwen2_5_vl_text" instead of the
    top-level "qwen2_5_vl" used as the AutoConfig/registry key. The class-level
    attribute is unaffected by this delegation, so prefer it.
    """
    return getattr(type(model_config), "model_type", None) or getattr(
        model_config, "model_type", "")


def parse_chat_messages_coroutines(
    messages: List[ChatCompletionMessageParam],
    model_config: AutoConfig,
    multimodal_server_config: Optional[MultimodalServerConfig] = None,
    request_media_io_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[ConversationMessage], Coroutine[Any, Any, tuple[Optional[Dict[
        str, List[Any]]], Optional[Dict[str, List[Any]]]]], list[dict[str,
                                                                      int]]]:
    """Parse multiple chat messages and return conversation and coroutine.

    Multimodal items across all messages share one
    `MultimodalDataTracker` so they fetch and decode concurrently when
    the coroutine is awaited.

    Args:
        messages: Chat messages with text or multimodal parts.
        model_config: HF `AutoConfig`; selects the model's placeholder
            strategy via `MULTIMODAL_PLACEHOLDER_REGISTRY`.
        multimodal_server_config: Server-level multimodal config
            (e.g. `--media_io_kwargs`); defaults to empty.
        request_media_io_kwargs: Per-request override merged per
            modality with the server default via
            `resolve_media_io_kwargs`.

    Returns:
        `(conversation, mm_coroutine, mm_placeholder_counts)` where
        `mm_coroutine` yields `(mm_data, mm_embeddings)` when awaited
        and `mm_placeholder_counts` has one entry per message mapping
        placeholder string -> count.
    """
    conversation = []
    mm_placeholder_counts = []
    model_type = resolve_top_level_model_type(model_config)
    mm_data_tracker = MultimodalDataTracker(
        model_type,
        multimodal_server_config,
        request_media_io_kwargs=request_media_io_kwargs)

    # Determine content format to decide placeholder strategy.
    #
    # We intentionally check only the `MULTIMODAL_PLACEHOLDER_REGISTRY` here
    # (not `_resolve_content_format` / jinja AST detection) because:
    #
    # 1. The chat template string is not available at this stage - it is resolved later in
    #    `apply_chat_template` which has access to the tokenizer and processor.
    # 2. Defaulting to `STRING` when the registry has no entry is safe even if the template later
    #    turns out to be OPENAI-format. When multimodal data is present, `content_parts` is always
    #    populated (see `parse_chat_message_content_parts`), and `apply_chat_template`'s OPENAI
    #    path calls `_build_openai_content`, which reconstructs `conv["content"]` from
    #    `content_parts` - overwriting any STRING-style placeholders inserted here.
    # See also: `_resolve_content_format` (inputs/utils.py) for the full resolution used downstream.
    registry_format = MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format(
        model_type)
    if registry_format is not None:
        content_format = registry_format
    else:
        content_format = ContentFormat.STRING

    for msg in messages:
        parsed_msg = parse_chat_message_content(msg, mm_data_tracker)
        conversation.append(parsed_msg)

        # Track placeholders added for this message only.
        msg_placeholder_counts = {}
        if parsed_msg["media"]:
            for mdata in parsed_msg["media"]:
                placeholder = mm_data_tracker.add_data(
                    mdata["modality"],
                    mdata["data"],
                    is_embedding=mdata["is_embedding"])
                if placeholder:
                    msg_placeholder_counts[
                        placeholder] = msg_placeholder_counts.get(
                            placeholder, 0) + 1

        if msg_placeholder_counts and content_format == ContentFormat.STRING:
            # For STRING format, use interleaving when the model opts in
            # and content_parts is available, otherwise fall back to bulk
            # prepend/append according to placeholder_placement.
            content_parts = parsed_msg.get("content_parts")
            interleave = MULTIMODAL_PLACEHOLDER_REGISTRY.get_interleave_placeholders(
                model_type)
            if content_parts and interleave:
                parsed_msg["content"] = interleave_mm_placeholders(
                    model_type, content_parts, msg_placeholder_counts,
                    mm_data_tracker.placeholder_modalities())
            else:
                parsed_msg["content"] = add_multimodal_placeholders(
                    model_type, parsed_msg["content"], msg_placeholder_counts)
        mm_placeholder_counts.append(msg_placeholder_counts)

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
