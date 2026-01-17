# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
For EdgeLLM API. Processes the chat template to create a JSON file with
chat template data for the following:

Roles:
- System
- User
- Assistant

Messages:
- Role
- Content
  - Type
    - text
    - image
    - video

The JSON file is saved to the exported ONNX model directory.

This implementation uses the HF tokenizer's apply_chat_template method with test cases
to extract the actual prefix/suffix patterns used by the model, rather than trying
to parse the Jinja template directly.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from ...utils.logger import ad_logger


def is_vlm(model_dir: str) -> bool:
    """Check if the model is a VLM."""
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg_dict = cfg.to_dict()
    has_vision = "vision_config" in cfg_dict
    has_phi4_vision = "image_embd_layer" in cfg_dict.get("embd_layer", {})
    if has_vision or has_phi4_vision:
        ad_logger.debug("Set use_prompt_tuning to True")
        return True
    else:
        ad_logger.debug("Set use_prompt_tuning to False")
        return False


@dataclass
class Message:
    role: str
    content: Union[str, List[Dict[str, str]]] = field(default_factory=list)


@dataclass
class SystemMessage(Message):
    role: str = "system"
    content: str = "<placeholder_system_prompt>"


@dataclass
class UserMessage(Message):
    role: str = "user"
    content: str = "<placeholder_user_text>"


@dataclass
class MultimodalUserMessage(Message):
    role: str = "user"
    content: List[Dict[str, str]] = field(
        default_factory=lambda: [{"type": "text", "text": "<placeholder_user_text>"}]
    )

    def add_text_content(self, text: str):
        self.content.append({"type": "text", "text": text})

    def add_image_content(self, image: str):
        self.content.append({"type": "image", "image": image})

    def add_video_content(self, video: str):
        self.content.append({"type": "video", "video": video})


@dataclass
class AssistantMessage(Message):
    role: str = "assistant"
    content: str = "<placeholder_assistant_text>"
    # TODO: Add tool calling


# TODO: Add ToolMessage


def _format_messages(
    tokenizer: Any, messages: List[Message], add_generation_prompt: bool = False
) -> str:
    """
    Format the messages using the tokenizer's chat template.

    Args:
        tokenizer: HuggingFace loaded tokenizer
        messages: List of messages
        add_generation_prompt: Whether to add generation prompt

    Returns:
        Formatted text

    Raises:
        ValueError: If unable to format messages
    """
    try:
        # Convert dataclass messages to dictionaries using asdict
        message_dicts = [asdict(msg) for msg in messages]

        return tokenizer.apply_chat_template(
            message_dicts, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception:
        # Try fallback: convert list content to string for tokenizers that don't support multimodal
        try:
            message_dicts = []
            for msg in messages:
                content = msg.content
                # If content is a list, extract the first text element
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content = item.get("text", "")
                            break
                message_dicts.append({"role": msg.role, "content": content})

            return tokenizer.apply_chat_template(
                message_dicts, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception as e2:
            raise ValueError(
                f"Unable to format messages using HuggingFace tokenizer's apply_chat_template method."
                f"Messages need to be in the format: role: <str>, content: <str|list of dicts>. "
                f"Check INPUT_FORMAT.md for more details."
                f"Error: {e2}"
            ) from e2


def _extract_prefix_suffix(text: str, placeholder: str) -> Tuple[str, str]:
    """
    Extract prefix and suffix from the differential text by finding the placeholder content.

    Args:
        text: The text to extract the prefix and suffix from
        placeholder   : The placeholder content to search for in the formatted output

    Returns:
        Tuple of (prefix, suffix) strings
    """
    content_start = text.find(placeholder)

    if content_start == -1:
        return "", ""

    prefix = text[:content_start]
    suffix = text[content_start + len(placeholder) :]

    return prefix, suffix


def _extract_content_pattern(
    tokenizer: Any,
    system_prompt: SystemMessage,
    content_type: str,
    placeholder: str,
    text_only_formatted: str,
    placeholder_text: str,
) -> Optional[str]:
    """
    Extract the pattern for a specific content type (image/video) by comparing
    with text-only message.

    Args:
        tokenizer: The loaded tokenizer
        system_prompt: System message to use
        content_type: Type of content ('image' or 'video')
        placeholder: Placeholder string for the content
        text_only_formatted: Formatted text-only message
        placeholder_text: The text placeholder used

    Returns:
        Extracted pattern string or None if failed or tokenizer does not support multimodal content
    """
    # Create user message with the content type
    user_with_content = MultimodalUserMessage()
    if content_type == "image":
        user_with_content.add_image_content(placeholder)
    elif content_type == "video":
        user_with_content.add_video_content(placeholder)
    else:
        return None

    with_content_formatted = _format_messages(tokenizer, [system_prompt, user_with_content])

    # Extract the differential - what was added for this content type
    if placeholder_text in text_only_formatted and placeholder_text in with_content_formatted:
        # Find position after the placeholder in both
        text_pos = text_only_formatted.find(placeholder_text) + len(placeholder_text)
        content_pos = with_content_formatted.find(placeholder_text) + len(placeholder_text)

        # Get what comes after the placeholder in both
        text_only_suffix = text_only_formatted[text_pos:]
        with_content_suffix = with_content_formatted[content_pos:]

        # The pattern is what was added (the difference)
        if text_only_suffix and with_content_suffix.endswith(text_only_suffix):
            pattern = with_content_suffix[: -len(text_only_suffix)]
        else:
            pattern = with_content_suffix

        # Strip dynamic prefixes like "Image 1:" or "Video 1:"
        pattern = re.sub(rf"^{content_type.capitalize()} \d+:\s*", "", pattern)

        return pattern if pattern else None

    return None


def process_chat_template(model_dir: str, output_dir: str) -> None:
    """
    Process the chat template from model's tokenizer and create a JSON file
    with parsed template information.

    This function uses the tokenizer's apply_chat_template method with various
    test cases to extract the actual prefix/suffix patterns.

    Args:
        model_dir: Path to the model directory containing tokenizer files
        output_dir: Path to save the chat_template.json file

    Returns:
        None
    """
    ad_logger.info(f"Processing chat template from {model_dir}")

    tokenizer = None
    loaders = (
        [AutoProcessor, AutoTokenizer] if is_vlm(model_dir) else [AutoTokenizer, AutoProcessor]
    )
    for ldr in loaders:
        try:
            tokenizer = ldr.from_pretrained(model_dir, trust_remote_code=True)
            if getattr(tokenizer, "chat_template", None):
                ad_logger.debug(f"Successfully loaded chat template from {ldr.__name__}")
                break
            else:
                ad_logger.debug(f"{ldr.__name__} loaded but no chat template found")
                tokenizer = None
        except Exception as e:
            ad_logger.error(f"Failed to load {ldr.__name__}: {e}")
            tokenizer = None

    if tokenizer is None:
        ad_logger.debug("Skipping chat template processing - no chat template available")
        return

    ad_logger.debug("Extracting patterns from chat template...")

    # Extract system role patterns (base case)
    system_prompt = SystemMessage()
    system_formatted = _format_messages(tokenizer, [system_prompt])
    system_prefix, system_suffix = _extract_prefix_suffix(system_formatted, system_prompt.content)

    # Extract user role patterns (compare with system base)
    user_prompt = UserMessage()
    user_formatted = _format_messages(tokenizer, [system_prompt, user_prompt])
    user_prefix, user_suffix = _extract_prefix_suffix(
        user_formatted[len(system_formatted) :], user_prompt.content
    )

    # Extract assistant role patterns (compare with user case)
    assistant_prompt = AssistantMessage()
    assistant_formatted = _format_messages(
        tokenizer, [system_prompt, user_prompt, assistant_prompt]
    )
    assistant_prefix, assistant_suffix = _extract_prefix_suffix(
        assistant_formatted[len(user_formatted) :], assistant_prompt.content
    )

    # Extract generation prompt
    generation_formatted = _format_messages(
        tokenizer, [system_prompt, user_prompt], add_generation_prompt=True
    )
    generation_prompt = generation_formatted[len(user_formatted) :]

    # Build content types
    content_types = {}

    # Only extract multimodal patterns if this is a VLM model
    if is_vlm(model_dir):
        ad_logger.debug("Detected VLM model, extracting multimodal content patterns...")
        # Get base text-only formatted message for comparison
        user_text_only = MultimodalUserMessage()
        text_only_formatted = _format_messages(tokenizer, [system_prompt, user_text_only])
        placeholder_text = user_text_only.content[0]["text"]

        # Extract image pattern
        image_pattern = _extract_content_pattern(
            tokenizer,
            system_prompt,
            "image",
            "<placeholder_image_path>",
            text_only_formatted,
            placeholder_text,
        )
        if image_pattern:
            content_types["image"] = {"format": image_pattern}

        # Extract video pattern
        video_pattern = _extract_content_pattern(
            tokenizer,
            system_prompt,
            "video",
            "<placeholder_video_path>",
            text_only_formatted,
            placeholder_text,
        )
        if video_pattern:
            content_types["video"] = {"format": video_pattern}
    else:
        ad_logger.debug("Text-only LLM detected, skipping multimodal content pattern extraction")

    # Extract default system prompt by testing without system message
    user_only_prompt = UserMessage()
    user_only_formatted = _format_messages(tokenizer, [user_only_prompt])

    # Extract default system prompt
    default_system_prompt = ""
    # Check if a default system prompt was added
    # The system message should appear in user_only_formatted if there's a default
    system_start = user_only_formatted.find(system_prefix)
    if system_start != -1:
        # Extract the system content between prefix and suffix
        content_start = system_start + len(system_prefix)
        content_end = user_only_formatted.find(system_suffix, content_start)
        if content_end != -1:
            default_system_prompt = user_only_formatted[content_start:content_end]
            # Remove the placeholder if it appears
            if default_system_prompt == system_prompt.content:
                default_system_prompt = ""

    # Build the final JSON structure
    chat_template_data = {
        "model_path": model_dir,
        "roles": {
            "system": {"prefix": system_prefix, "suffix": system_suffix},
            "user": {"prefix": user_prefix, "suffix": user_suffix},
            "assistant": {"prefix": assistant_prefix, "suffix": assistant_suffix},
        },
        "content_types": content_types,
        "generation_prompt": generation_prompt,
        "default_system_prompt": default_system_prompt,
    }

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_chat_template.json")

    with open(output_path, "w") as f:
        json.dump(chat_template_data, f, indent=2)

    ad_logger.info(f"Chat template saved to {output_path}")
