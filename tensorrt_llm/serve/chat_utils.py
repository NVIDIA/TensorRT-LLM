from functools import partial
from typing import (Any, Callable, Dict, Iterable, List, Literal, Optional,
                    Tuple, TypeAlias, TypedDict, Union, cast)

from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import (ChatCompletionContentPartTextParam,
                               ChatCompletionMessageParam)
from transformers import AutoConfig, ProcessorMixin
from typing_extensions import Required

from tensorrt_llm.inputs import load_image, load_video
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
    model_type = model_config.model_type

    if modality == "image":
        if model_type in ("mllama", "llama4"):
            return "<|image|>"
        if model_type in ("qwen2_vl", "qwen2_5_vl"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        raise TypeError(f"Unknown {modality} model type: {model_type}")

    elif modality == "video":
        if model_type in ("qwen2_vl", "qwen2_5_vl"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise TypeError(f"Unknown {modality} model type: {model_type}")

    raise TypeError(f"Unknown modality: {modality}")


def add_multimodal_placeholders(
    text_prompt: str,
    mm_content_dict: dict[str, list[Any]],
    model_config: AutoConfig,
) -> str:
    placeholders = []
    counts = {}
    for media_type, _ in mm_content_dict.items():
        if media_type not in counts:
            counts[media_type] = 0
        counts[media_type] += 1
        placeholder = retrieve_multimodal_placeholder(media_type, model_config,
                                                      counts[media_type])
        if placeholder is not None:
            placeholders.append(placeholder)
    return "\n".join(placeholders + [text_prompt])


def _parse_chat_message_content_mm_part(
    part: ChatCompletionContentPartParam
) -> tuple[str, Union[str, dict[str, str]]]:
    assert isinstance(
        part, dict)  # This is needed to avoid mypy errors: part.get() from str
    part_type = part.get("type", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP:
        content = MM_PARSER_MAP[part_type](part)
        return part_type, content

    if not isinstance(part_type, str):
        raise ValueError("Invalid 'type' field in multimodal part.")
    return part_type, "unknown part_type content"


def parse_chat_message_content_part(part: ChatCompletionMessageParam, ):
    if isinstance(part, str):  # Handle plain text parts
        return part

    # Handle structured dictionary parts
    part_type, content = _parse_chat_message_content_mm_part(part)

    # if part_type is text/image_url/video_url but content is None, log a warning and skip
    if part_type in VALID_MESSAGE_CONTENT_MM_PART_TYPES and content is None:
        logger.warning(
            "Skipping multimodal part '%s' (type: '%s') "
            "with empty / unparsable content.", part, part_type)
        return None

    mm_content = None
    if part_type == "text":
        str_content = cast(str, content)
        return str_content, mm_content

    # TODO: make them async on multimodal data as loading video/image is time consuming

    # Handle all non-text multimodal types
    if part_type == "image_url":
        str_content = cast(str, content)
        mm_content = {"image": load_image(str_content)}
    elif part_type == "video_url":
        str_content = cast(str, content)
        mm_content = {"video": load_video(str_content, num_frames=8)}
    else:
        raise NotImplementedError(f"Unknown part type: {part_type}")

    return None, mm_content


def parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionMessageParam],
    model_config: AutoConfig,
) -> List[ConversationMessage]:
    content = []
    mm_content_dict = {}

    for part in parts:
        parse_res, mm_content = parse_chat_message_content_part(part, )
        if parse_res:
            content.append(parse_res)

        # Collect multimodal content
        if mm_content:
            for media_type, media_value in mm_content.items():
                if media_type not in mm_content_dict:
                    mm_content_dict[media_type] = []
                mm_content_dict[media_type].append(media_value)

    text_prompt = "\n".join(content)
    if mm_content_dict:
        text_prompt = add_multimodal_placeholders(text_prompt, mm_content_dict,
                                                  model_config)
    return [ConversationMessage(role=role,
                                content=text_prompt)], mm_content_dict


def parse_chat_message_content(
    message: ChatCompletionMessageParam,
    model_config: AutoConfig,
) -> Tuple[List[ConversationMessage], Optional[Dict[str, List[Any]]]]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [
            ChatCompletionContentPartTextParam(type="text", text=content)
        ]

    result, mm_data = parse_chat_message_content_parts(
        role,
        content,
        model_config,
    )
    return result, mm_data


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
