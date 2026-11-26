# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Place chat ``prompt_embeds`` parts in a rendered prompt.

The chat template renders each part as its marker. Here the text between
markers is tokenized segment by segment and each marker becomes one placeholder
id per row of its part, with a mask selecting those positions. Text after a part
is tokenized as a continuation, so tokenizers that normalize the start of a
string (SentencePiece adds a leading space token) do not change it.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm.serve.prompt_embeds import PLACEHOLDER_TOKEN_ID, ChatPromptEmbedsPart

from .chat import tokenize_rendered
from .prompt_types import UnsupportedRenderError
from .resources import RenderResources

Tokenize = Callable[[str, bool], List[int]]
Continue = Callable[[str], Optional[List[int]]]

# Probe text for finding the special tokens a tokenizer wraps a prompt in.
_PROBE = "a"
# Text put before a continuation segment and then cut off by offsets; the
# first that does not merge with the segment's first token is used.
_CONTINUATION_PROBES = ("a", ".", ",")


def resource_tokenizer(res: RenderResources) -> Tokenize:
    """Return ``tokenize(text, add_special_tokens)`` with the engine's tokenizer."""

    def tokenize(text: str, add_special_tokens: bool) -> List[int]:
        """Tokenize one segment of a rendered prompt."""
        return tokenize_rendered(res, text, add_special_tokens=add_special_tokens)

    return tokenize


def resource_continuation(res: RenderResources) -> Continue:
    """Return ``continuation(text)``: ids of ``text`` as it reads mid-prompt.

    ``text`` is tokenized after a short probe that is then cut off: by offset
    mapping with a fast Hugging Face tokenizer, else by the probe's own ids.
    Returns None for ``text`` when every probe merges into its first token.
    """
    hf_tokenizer = getattr(res.tokenizer, "tokenizer", res.tokenizer)
    if not getattr(hf_tokenizer, "is_fast", False):

        def id_continuation(text: str) -> Optional[List[int]]:
            """Tokenize ``text`` after a probe and drop the probe's ids."""
            for probe in _CONTINUATION_PROBES:
                head = hf_tokenizer.encode(probe, add_special_tokens=False)
                ids = hf_tokenizer.encode(probe + text, add_special_tokens=False)
                if ids[: len(head)] == head:
                    return ids[len(head) :]
            return None

        return id_continuation

    def continuation(text: str) -> Optional[List[int]]:
        """Tokenize ``text`` after a probe and keep the tokens past the probe."""
        for probe in _CONTINUATION_PROBES:
            encoded = hf_tokenizer(
                probe + text, add_special_tokens=False, return_offsets_mapping=True
            )
            cut = len(probe)
            offsets = encoded["offset_mapping"]
            if any(start < cut < end for start, end in offsets):
                continue  # the probe merged into the text's first token
            return [i for i, (start, _) in zip(encoded["input_ids"], offsets) if start >= cut]
        return None

    return continuation


def _special_token_wrap(tokenize: Tokenize) -> Tuple[List[int], List[int]]:
    """Return the ids the tokenizer adds before and after a prompt."""
    plain = tokenize(_PROBE, False)
    wrapped = tokenize(_PROBE, True)
    starts = [
        i for i in range(len(wrapped) - len(plain) + 1) if wrapped[i : i + len(plain)] == plain
    ]
    if len(starts) != 1:
        raise UnsupportedRenderError(
            "This tokenizer's special tokens cannot be placed around prompt_embeds parts; "
            "set add_special_tokens to false."
        )
    start = starts[0]
    return wrapped[:start], wrapped[start + len(plain) :]


def splice_chat_prompt_embeds(
    text: str,
    parts: Sequence[ChatPromptEmbedsPart],
    rows: Sequence[torch.Tensor],
    tokenize: Tokenize,
    add_special_tokens: bool,
    continuation: Optional[Continue] = None,
) -> Dict[str, Any]:
    """Turn a rendered chat prompt with part markers into an embeddings prompt.

    Args:
        text: The rendered chat prompt.
        parts: The request's parts; each marker must occur exactly once.
        rows: Decoded rows of each part, in ``parts`` order.
        tokenize: ``tokenize(segment, add_special_tokens)`` with the engine's
            tokenizer.
        add_special_tokens: Whether the whole prompt gets the tokenizer's
            special tokens (e.g. BOS, EOS), placed once around it.
        continuation: Tokenizes text that follows a part as it reads
            mid-prompt (see ``resource_continuation``); without it such text is
            tokenized on its own.

    Returns:
        ``prompt_token_ids``, ``prompt_embeds`` (the parts' rows in prompt
        order) and ``prompt_embeds_mask``.
    """
    dtypes = {r.dtype for r in rows}
    if len(dtypes) > 1:
        raise ValueError(
            f"All prompt_embeds parts of a request must share one dtype, got "
            f"{sorted(str(d) for d in dtypes)}."
        )
    placed = []
    for part, part_rows in zip(parts, rows):
        offset = text.find(part.marker)
        if offset < 0:
            raise ValueError("The chat template dropped a message holding a prompt_embeds part.")
        if text.find(part.marker, offset + 1) >= 0:
            raise ValueError("The chat template repeated a message holding a prompt_embeds part.")
        placed.append((offset, part.marker, part_rows))
    placed.sort(key=lambda item: item[0])

    token_ids: List[int] = []
    mask: List[bool] = []

    def add_text(segment: str, after_part: bool) -> None:
        """Append the ids of one text segment."""
        if not segment:
            return
        if after_part and continuation is not None:
            ids = continuation(segment)
            if ids is None:
                raise UnsupportedRenderError(
                    "The text after a prompt_embeds part cannot be tokenized as "
                    "a continuation with this tokenizer."
                )
        else:
            ids = tokenize(segment, False)
        token_ids.extend(ids)
        mask.extend([False] * len(ids))

    cursor = 0
    for offset, marker, part_rows in placed:
        add_text(text[cursor:offset], after_part=cursor > 0)
        token_ids.extend([PLACEHOLDER_TOKEN_ID] * part_rows.shape[0])
        mask.extend([True] * part_rows.shape[0])
        cursor = offset + len(marker)
    add_text(text[cursor:], after_part=True)
    if add_special_tokens:
        prefix, suffix = _special_token_wrap(tokenize)
        token_ids = prefix + token_ids + suffix
        mask = [False] * len(prefix) + mask + [False] * len(suffix)
    return {
        "prompt_token_ids": token_ids,
        "prompt_embeds": placed[0][2]
        if len(placed) == 1
        else torch.cat([r for _, _, r in placed], dim=0),
        "prompt_embeds_mask": mask,
    }
