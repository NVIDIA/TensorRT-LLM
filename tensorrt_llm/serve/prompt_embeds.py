# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prompt_embeds`` on the OpenAI routes: request parts and payload decoding.

Completions take ``prompt_embeds`` as one or more base64 ``torch.save`` payloads,
each a whole prompt of ``[L, hidden_size]`` rows. Chat takes a content part
``{"type": "prompt_embeds", "prompt_embeds": {"data": <base64>}}``: the part
renders as a unique marker, and ``serve.render.prompt_embeds`` turns each marker
into a run of placeholder ids whose positions take the part's rows.

Tensor payloads from clients are untrusted, so the server accepts them only when
started with ``--enable_prompt_embeds``. Decoding is bounded by
``TRTLLM_PROMPT_EMBEDS_MAX_BYTES`` per request and runs off the event loop.
"""

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from tensorrt_llm.inputs.utils import load_base64_tensor

PROMPT_EMBEDS_PART_TYPE = "prompt_embeds"

# Key under which a parsed chat message carries its prompt_embeds parts. The
# chat route pops it before rendering.
PROMPT_EMBEDS_MESSAGE_KEY = "prompt_embeds_parts"

PROMPT_EMBEDS_MAX_BYTES_ENV = "TRTLLM_PROMPT_EMBEDS_MAX_BYTES"
DEFAULT_PROMPT_EMBEDS_MAX_BYTES = 256 * 1024 * 1024

DISABLED_MESSAGE = "prompt_embeds requires the server to be started with --enable_prompt_embeds."

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Token id at embedding positions; matches build_prompt_embeds' default.
PLACEHOLDER_TOKEN_ID = 0


@dataclass(frozen=True)
class ChatPromptEmbedsPart:
    """One ``prompt_embeds`` content part of a chat message.

    Attributes:
        marker: Unique text the part renders as in the chat template.
        data: The part's base64 ``torch.save`` payload, still encoded.
    """

    marker: str
    data: str


def prompt_embeds_max_bytes() -> int:
    """Return the per-request limit on decoded ``prompt_embeds`` bytes."""
    value = os.environ.get(PROMPT_EMBEDS_MAX_BYTES_ENV)
    if value is None:
        return DEFAULT_PROMPT_EMBEDS_MAX_BYTES
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"{PROMPT_EMBEDS_MAX_BYTES_ENV} must be positive, got {value!r}")
    return limit


def new_chat_part(part: Dict[str, Any]) -> ChatPromptEmbedsPart:
    """Validate a ``prompt_embeds`` content part without decoding it."""
    body = part.get(PROMPT_EMBEDS_PART_TYPE)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, str) or not data:
        raise ValueError(
            "A prompt_embeds content part needs prompt_embeds.data: a base64 "
            "torch.save of a [num_tokens, hidden_size] tensor."
        )
    return ChatPromptEmbedsPart(marker=f"<|prompt_embeds:{uuid.uuid4().hex}|>", data=data)


def decode_prompt_embeds(
    payloads: Sequence[str],
    *,
    max_bytes: int,
    hidden_size: Optional[int] = None,
) -> List[torch.Tensor]:
    """Decode a request's ``prompt_embeds`` payloads under one byte budget.

    Blocking; run it off the event loop.

    Args:
        payloads: Base64 ``torch.save`` payloads, each a ``[rows, hidden]`` tensor.
        max_bytes: Limit on the decoded bytes of all payloads together.
        hidden_size: Expected row width, when the server knows it.

    Returns:
        The tensors, in payload order.
    """
    encoded_limit = 4 * ((max_bytes + 2) // 3)
    # load_base64_tensor's allowance for line-wrapped (MIME) text.
    encoded_limit += encoded_limit // 38 + 2 * len(payloads)
    if sum(len(p) if isinstance(p, str) else 0 for p in payloads) > encoded_limit:
        raise ValueError(
            f"prompt_embeds payloads exceed the {max_bytes}-byte request limit "
            f"({PROMPT_EMBEDS_MAX_BYTES_ENV})."
        )
    tensors = []
    remaining = max_bytes
    for index, payload in enumerate(payloads):
        tensor = load_base64_tensor(payload, max_bytes=remaining)
        # The whole storage is held, even under a smaller view of it.
        remaining -= tensor.untyped_storage().nbytes()
        if tensor.dim() != 2:
            raise ValueError(
                f"prompt_embeds[{index}] must have shape [num_tokens, hidden_size], "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"prompt_embeds[{index}] dtype must be float16, bfloat16 or float32, "
                f"got {tensor.dtype}."
            )
        if tensor.shape[0] == 0:
            raise ValueError(f"prompt_embeds[{index}] has no rows.")
        if hidden_size is not None and tensor.shape[1] != hidden_size:
            raise ValueError(
                f"prompt_embeds[{index}] has hidden size {tensor.shape[1]}; the "
                f"model expects {hidden_size}."
            )
        tensors.append(tensor)
    return tensors


def model_hidden_size(model_config: Any) -> Optional[int]:
    """Return the text model's hidden size from a Hugging Face config, if known."""
    if model_config is None:
        return None
    get_text_config = getattr(model_config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else model_config
    hidden_size = getattr(text_config, "hidden_size", None)
    return hidden_size if isinstance(hidden_size, int) else None


def has_prompt_embeds_parts(messages: Any) -> bool:
    """Whether any raw chat message carries a ``prompt_embeds`` content part."""
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == PROMPT_EMBEDS_PART_TYPE
            for part in content
        ):
            return True
    return False


def pop_chat_parts(conversation: List[Dict[str, Any]]) -> List[ChatPromptEmbedsPart]:
    """Remove and return the ``prompt_embeds`` parts of parsed chat messages."""
    parts: List[ChatPromptEmbedsPart] = []
    for message in conversation:
        parts.extend(message.pop(PROMPT_EMBEDS_MESSAGE_KEY, None) or ())
    return parts
